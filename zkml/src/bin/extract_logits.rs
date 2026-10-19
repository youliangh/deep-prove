#![allow(clippy::print_stderr)]
#![allow(clippy::print_stdout)]

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use itertools::Itertools;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    io::{BufWriter, Read, Write},
    path::{Path, PathBuf},
};
use tenstore::GenStore;
use zkml::{
    graph::Node,
    layers::{Layer, provable::Evaluate},
    model::{Trace, llm::Driver},
    parser::{
        ModelNameProvider,
        llm::{
            HFTokenizer, Token,
            models::{
                LLMModelLoader,
                gemma3::Gemma3,
                gpt2::{GPT2, is_gpt2_model},
                llama3::Llama3,
            },
            tokenizer::{LLMTokenizer, TokenizerLoader},
        },
        safe::RawSafeTensors,
    },
    quantization::{Dequantize, llm_quant::FPTransformModel},
};

/// Section separator for debug output
const SECTION_SEPARATOR: &str = "===";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ExecutionMode {
    Float,
    Int,
    Both,
}

#[derive(Parser, Debug)]
#[command(about = "Extract raw logits from ZKML inference engine")]
struct Args {
    /// Path to model directory containing model.safetensors, tokenizer.json, and config.json
    #[arg(short, long)]
    model: PathBuf,

    /// Input text for inference
    #[arg(short, long, default_value = "The quick brown fox")]
    text: String,

    /// Number of tokens to generate autoregressively (0 = no generation, just return input logits)
    #[arg(short, long, default_value = "0")]
    num_tokens: usize,

    /// Size of each sample to run, if provided with 0 will run a single sample over the whole input
    #[arg(short, long, default_value = "0")]
    sample_size: usize,

    /// The (flat) tokens to use as input instead of text (overrides --text)
    #[arg(long, value_delimiter = ',')]
    tokens: Option<Vec<usize>>,

    /// JSON array of token sequences; used for same-history numerical comparisons.
    #[arg(long, conflicts_with_all = ["tokens", "sample_size"]) ]
    tokens_file: Option<PathBuf>,

    /// Llama3 execution path. Float avoids quantization entirely.
    #[arg(long, value_enum, default_value = "both")]
    mode: ExecutionMode,

    /// Write little-endian f32 logits and a JSON manifest instead of a large JSON array.
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// Maximum model context allocated before loading RoPE constants.
    #[arg(long, default_value_t = 256)]
    max_context: usize,

    /// JSON array of independent calibration token sequences (training data).
    #[arg(long)]
    calibration_tokens: Option<PathBuf>,

    #[arg(long, default_value_t = 8)]
    calibration_samples: usize,

    /// Save dequantized inputs of every original layer for numerical diagnostics.
    #[arg(long, requires = "output_dir")]
    dump_layer_inputs: bool,
}

#[derive(Serialize)]
struct BinaryLogitsRecord {
    sample: usize,
    mode: String,
    file: String,
    shape: [usize; 2],
    tokens: Vec<usize>,
    generated_tokens: Vec<usize>,
    text: String,
}

fn load_token_samples(path: &Path) -> Result<Vec<Vec<Token>>> {
    let samples: Vec<Vec<usize>> = serde_json::from_reader(std::fs::File::open(path)?)?;
    anyhow::ensure!(!samples.is_empty(), "Token samples must not be empty");
    samples
        .into_iter()
        .map(|sample| {
            anyhow::ensure!(!sample.is_empty(), "Token sequence must not be empty");
            Ok(sample.into_iter().map(Token::from).collect())
        })
        .collect()
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let length = file.read(&mut buffer)?;
        if length == 0 {
            break;
        }
        hash.update(&buffer[..length]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn llama3_logits(format: RawSafeTensors, args: &Args) -> Result<()> {
    anyhow::ensure!(
        !args.dump_layer_inputs || args.num_tokens == 0,
        "Layer diagnostics require teacher-forcing (--num-tokens 0)"
    );
    anyhow::ensure!(
        args.sample_size == 0,
        "For Llama3 use --tokens-file for multiple samples"
    );
    let tokenizer = Llama3::new().load_tokenizer(&format)?;
    let samples = if let Some(path) = &args.tokens_file {
        load_token_samples(path)?
    } else if let Some(ids) = &args.tokens {
        vec![ids.iter().copied().map(Token::from).collect()]
    } else {
        vec![tokenizer.tokenize(&args.text)]
    };
    for tokens in &samples {
        anyhow::ensure!(!tokens.is_empty(), "Tokenized prompt is empty");
        let required_rows = if args.num_tokens == 0 {
            tokens.len()
        } else {
            zkml::model::llm::generation_context_length(tokens.len(), args.num_tokens)?
        };
        anyhow::ensure!(
            required_rows <= args.max_context,
            "Requested sequence exceeds --max-context"
        );
    }
    let mut float_driver = Driver::load_from_model(Llama3::new(), &format, Some(args.max_context))?;
    let mut records = Vec::new();
    let mut quantization_sha256 = None;
    let mut calibration_sha256 = None;
    let mut float_logits = Vec::new();
    let mut int_logits = Vec::new();
    if args.mode != ExecutionMode::Int {
        float_logits = collect_llama3_logits(
            &mut float_driver,
            &tokenizer,
            &samples,
            args,
            "float",
            |data, _, _| Ok(data.to_vec()),
            &mut records,
        )?;
    }
    let vocab_size = float_driver.vocab_size();
    if args.mode != ExecutionMode::Float {
        float_driver.model.reset();
        float_driver.with_max_context(args.max_context);
        let calibration = args
            .calibration_tokens
            .as_ref()
            .map(|path| {
                let mut samples = load_token_samples(path)?;
                anyhow::ensure!(
                    args.calibration_samples > 0 && args.calibration_samples <= samples.len(),
                    "Invalid calibration sample count"
                );
                samples.truncate(args.calibration_samples);
                Ok::<_, anyhow::Error>(samples)
            })
            .transpose()?;
        calibration_sha256 = Some(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&calibration)?)
        ));
        let (mut driver, metadata) =
            zkml::quantization::llm_quant::llama3::quantize(float_driver, &tokenizer, calibration)?;
        let snapshot = zkml::quantization::llm_quant::llama3::calibration_snapshot(&metadata)?;
        let snapshot_bytes = serde_json::to_vec(&snapshot)?;
        quantization_sha256 = Some(format!("{:x}", Sha256::digest(&snapshot_bytes)));
        if let Some(directory) = &args.output_dir {
            std::fs::create_dir_all(directory)?;
            std::fs::write(directory.join("quantization.json"), &snapshot_bytes)?;
        }
        int_logits = collect_llama3_logits(
            &mut driver,
            &tokenizer,
            &samples,
            args,
            "int",
            |data, node_id, port| {
                let factors = metadata.layer_input_scaling_factor(node_id);
                Ok(data.dequantize(factors.get(port).context("Missing input scale")?))
            },
            &mut records,
        )?;
    }
    if let Some(directory) = &args.output_dir {
        std::fs::create_dir_all(directory)?;
        let manifest = serde_json::json!({"records": records, "model": args.model,
        "max_context": args.max_context, "quantization_bits": *zkml::quantization::BIT_LEN,
        "calibration_tokens": args.calibration_tokens, "calibration_samples": args.calibration_samples,
        "calibration_sha256": calibration_sha256, "quantization_sha256": quantization_sha256,
        "hashes": {
            "model.safetensors": sha256_file(&args.model.join("model.safetensors"))?,
            "config.json": sha256_file(&args.model.join("config.json"))?,
            "tokenizer.json": sha256_file(&args.model.join("tokenizer.json"))?
        }});
        serde_json::to_writer_pretty(
            std::fs::File::create(directory.join("manifest.json"))?,
            &manifest,
        )?;
        Ok(())
    } else {
        anyhow::ensure!(
            args.mode == ExecutionMode::Both,
            "Single-mode output requires --output-dir"
        );
        write_logits_to_json(&float_logits, &int_logits, vocab_size)
    }
}

fn collect_llama3_logits<N, ExtractFn>(
    driver: &mut Driver<N>,
    tokenizer: &HFTokenizer,
    samples: &[Vec<Token>],
    args: &Args,
    mode: &str,
    extract: ExtractFn,
    records: &mut Vec<BinaryLogitsRecord>,
) -> Result<Vec<LogitsData>>
where
    N: zkml::tensor::TensorTypeParam,
    Layer<N>: Evaluate<N>,
    ExtractFn: Fn(&[N], zkml::graph::NodeId, usize) -> Result<Vec<f32>>,
{
    let mut all_logits = Vec::new();
    for (index, tokens) in samples.iter().enumerate() {
        let mut store = GenStore::default();
        let trace = if args.num_tokens == 0 {
            driver.with_max_context(tokens.len());
            driver.run(vec![driver.tokens_to_tensor(tokens)?], &mut store)?
        } else {
            driver.generate(tokens, args.num_tokens, &mut store)?
        };
        let generated = if args.num_tokens > 0 {
            driver.continuation_tokens(&trace, tokens.len())?
        } else {
            Vec::new()
        };
        let text = tokenizer.detokenize(&generated);
        eprintln!("{mode} sample {index}, generated: {text}");
        if args.dump_layer_inputs {
            dump_layer_inputs(
                driver,
                &trace,
                &extract,
                &args
                    .output_dir
                    .as_ref()
                    .unwrap()
                    .join(format!("{mode}-layers-{index:03}")),
            )?;
        }
        let mut logits =
            extract_logits_from_trace(driver, &trace, |data, node| extract(data, node, 0))?;
        let columns = logits.data.len() / logits.seq_len;
        let vocab = driver.vocab_size();
        let logical_rows = if args.num_tokens == 0 {
            tokens.len()
        } else {
            tokens.len() + generated.len() - 1
        };
        anyhow::ensure!(
            logits.seq_len >= logical_rows,
            "Logits are shorter than the input history"
        );
        logits.data.truncate(logical_rows * columns);
        logits.seq_len = logical_rows;
        anyhow::ensure!(
            columns >= vocab,
            "Logits have fewer columns than vocabulary"
        );
        anyhow::ensure!(
            logits.data.iter().all(|x| x.is_finite()),
            "Non-finite logits"
        );
        if columns != vocab {
            logits.data = logits
                .data
                .chunks_exact(columns)
                .flat_map(|row| row[..vocab].iter().copied())
                .collect();
        }
        if let Some(directory) = &args.output_dir {
            std::fs::create_dir_all(directory)?;
            let filename = format!("{mode}_{index:03}.f32");
            let mut output = BufWriter::new(std::fs::File::create(directory.join(&filename))?);
            for value in &logits.data {
                output.write_all(&value.to_le_bytes())?;
            }
            output.flush()?;
            records.push(BinaryLogitsRecord {
                sample: index,
                mode: mode.to_owned(),
                file: filename,
                shape: [logits.seq_len, vocab],
                tokens: tokens.iter().copied().map(usize::from).collect(),
                generated_tokens: generated.iter().copied().map(usize::from).collect(),
                text,
            });
        } else {
            all_logits.push(logits);
        }
        driver.model.reset();
    }
    Ok(all_logits)
}

fn dump_layer_inputs<N, ExtractFn>(
    driver: &Driver<N>,
    trace: &Trace<N>,
    extract: &ExtractFn,
    directory: &Path,
) -> Result<()>
where
    N: zkml::tensor::TensorTypeParam,
    ExtractFn: Fn(&[N], zkml::graph::NodeId, usize) -> Result<Vec<f32>>,
{
    std::fs::create_dir_all(directory)?;
    let mut records = Vec::new();
    for (node_id, node) in driver.model.graph().nodes() {
        let Node::Inner(layer) = node else { continue };
        // Compare inputs after any inserted requantization, using the original
        // graph IDs. Raw pre-requant outputs do not use the metadata output scale.
        if matches!(layer, Layer::Requant(_)) {
            continue;
        }
        let Some(step) = trace.get_step(node_id) else {
            continue;
        };
        for (port, handle) in step.inputs().iter().enumerate() {
            let tensor = handle.tensor()?;
            let values = if matches!(layer, Layer::Embeddings(_)) {
                // Token IDs are discrete indices: Embeddings ignores the input
                // quantization scale and receives the same IDs in both modes.
                tensor
                    .data()
                    .iter()
                    .map(zkml::Number::to_f32)
                    .collect::<Result<Vec<_>>>()?
            } else {
                extract(tensor.data(), *node_id, port)?
            };
            let filename = format!("{node_id}-{port}.f32");
            let mut output = BufWriter::new(std::fs::File::create(directory.join(&filename))?);
            for value in values {
                output.write_all(&value.to_le_bytes())?;
            }
            output.flush()?;
            records.push(serde_json::json!({
                "node": node_id, "port": port, "layer": layer.short_name(), "file": filename,
                "shape": tensor.shape().as_ref(), "unpadded_shape": tensor.unpadded_shape().as_ref(),
            }));
        }
    }
    serde_json::to_writer_pretty(
        std::fs::File::create(directory.join("manifest.json"))?,
        &serde_json::json!({"records": records, "model_metadata": driver.md}),
    )?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct LogitsOutput {
    /// Logits from float mode inference [seq_len * vocab_size], row-major
    logits_float: Vec<Vec<f32>>,
    /// Logits from integer/provable mode inference [seq_len_int * vocab_size], row-major
    logits_int: Vec<Vec<f32>>,
    /// Sequence length (float mode)
    seq_len: usize,
    /// Sequence length (int mode, potentially padded)
    seq_len_int: usize,
    /// Vocabulary size (actual, not padded)
    vocab_size: usize,
}

#[derive(Debug)]
struct LogitsData {
    data: Vec<f32>,
    seq_len: usize,
}

fn write_logits_to_json(
    logits_float: &[LogitsData],
    logits_int: &[LogitsData],
    vocab_size: usize,
) -> Result<()> {
    let seq_len = logits_float
        .first()
        .context("No float logits data found")?
        .seq_len;
    let seq_len_int = logits_int
        .first()
        .context("No int logits data found")?
        .seq_len;

    let logits_float = logits_float
        .iter()
        .map(|ld| ld.data.clone())
        .collect::<Vec<Vec<f32>>>();
    let logits_int = logits_int
        .iter()
        .map(|ld| ld.data.clone())
        .collect::<Vec<Vec<f32>>>();

    let output = LogitsOutput {
        logits_float,
        logits_int,
        seq_len,
        seq_len_int,
        vocab_size,
    };

    let stdout = std::io::stdout();
    let handle = stdout.lock();
    serde_json::to_writer(handle, &output)?;

    Ok(())
}

fn extract_dimensions(shape: &zkml::Shape) -> (usize, usize) {
    if shape.rank() >= 2 {
        let seq_len = shape.dim(0);
        let vocab_size = shape.dim(shape.rank() - 1);
        (seq_len, vocab_size)
    } else {
        (1, shape.dim(0))
    }
}

/// Validate that required model files exist in the given directory
fn validate_model_files(model_dir: &Path) -> Result<(PathBuf, PathBuf, PathBuf)> {
    if !model_dir.is_dir() {
        bail!("Model path must be a directory: {}", model_dir.display());
    }

    let required_files = [
        ("model.safetensors", "Model file"),
        ("tokenizer.json", "Tokenizer file"),
        ("config.json", "Config file"),
    ];

    let mut paths = Vec::new();
    for (filename, description) in &required_files {
        let path = model_dir.join(filename);
        if !path.exists() {
            bail!("{} not found: {}", description, path.display());
        }
        paths.push(path);
    }

    Ok((paths[0].clone(), paths[1].clone(), paths[2].clone()))
}

fn extract_logits_from_trace<N, F>(
    driver: &Driver<N>,
    trace: &Trace<N>,
    to_float: F,
) -> Result<LogitsData>
where
    N: zkml::tensor::TensorTypeParam,
    F: Fn(&[N], zkml::graph::NodeId) -> Result<Vec<f32>>,
{
    for (node_id, node) in driver.model.graph().nodes() {
        let Node::Inner(layer) = node else {
            continue;
        };

        let Some(step) = trace.get_step(node_id) else {
            continue;
        };

        if !matches!(layer, Layer::Logits(_)) {
            continue;
        }

        let step_inputs = step.inputs();
        if step_inputs.is_empty() {
            continue;
        }

        let logit_tensor = step_inputs[0]
            .tensor()
            .ok()
            .context("Failed to get logit tensor")?;
        let shape = logit_tensor.shape();
        let data = to_float(logit_tensor.data(), *node_id)?;
        let (seq_len, _vocab_size) = extract_dimensions(shape);

        return Ok(LogitsData { data, seq_len });
    }

    bail!(
        "Could not extract logits from trace. \
        Make sure the model has a Logits layer and was executed correctly."
    )
}

/// Run inference using run_elements for the given driver
fn run_generation<N, ExtractFn>(
    driver: &Driver<N>,
    tokenizer: &HFTokenizer,
    tokens: &[Token],
    mode_name: &str,
    extract_fn: ExtractFn,
) -> Result<LogitsData>
where
    N: zkml::tensor::TensorTypeParam,
    ExtractFn: Fn(&[N], zkml::graph::NodeId) -> Result<Vec<f32>>,
    Layer<N>: Evaluate<N>,
{
    let mut store = GenStore::default();
    let tensor_inputs = driver.tokens_to_tensor(tokens)?;
    let trace: Trace<N> = driver.run_elements(tensor_inputs, &mut store)?;
    let logits_step = extract_logits_from_trace(driver, &trace, &extract_fn)?;

    let input_text = tokenizer.detokenize(tokens);
    eprintln!("{} input: {}", mode_name, input_text);

    let output_tokens = trace
        .outputs()
        .last()
        .unwrap()
        .tensor()
        .unwrap()
        .data()
        .iter()
        .skip(tokens.len())
        .map(|t| Token::from(zkml::Number::to_usize(t)))
        .collect::<Vec<_>>();
    let output = tokenizer.detokenize(&output_tokens);
    eprintln!("{} output: {}", mode_name, output);

    Ok(LogitsData {
        data: logits_step.data,
        seq_len: logits_step.seq_len,
    })
}

fn run_multi_sample_generation<N, ExtractFn>(
    driver: &Driver<N>,
    tokenizer: &HFTokenizer,
    token_samples: &[&[Token]],
    mode_name: &str,
    extract_fn: ExtractFn,
) -> Result<Vec<LogitsData>>
where
    N: zkml::tensor::TensorTypeParam,
    ExtractFn: Fn(&[N], zkml::graph::NodeId) -> Result<Vec<f32>>,
    Layer<N>: Evaluate<N>,
{
    // First check all the tokens have the same length
    let all_equal_len = token_samples.iter().map(|t| t.len()).all_equal();
    anyhow::ensure!(
        all_equal_len,
        "All token sequences must have the same length to run multiple samples in mode {mode_name}"
    );
    let mut store = GenStore::default();

    let mut logits_data = Vec::<LogitsData>::with_capacity(token_samples.len());
    for tokens in token_samples {
        let tensor_inputs = driver.tokens_to_tensor(tokens)?;
        let trace: Trace<N> = driver.run_elements(tensor_inputs, &mut store)?;
        let logits_step = extract_logits_from_trace(driver, &trace, &extract_fn)?;

        let input_text = tokenizer.detokenize(tokens);
        eprintln!("{} input: {}", mode_name, input_text);

        let output_tokens = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .data()
            .iter()
            .skip(tokens.len())
            .map(|t| Token::from(zkml::Number::to_usize(t)))
            .collect::<Vec<_>>();
        let output = tokenizer.detokenize(&output_tokens);
        eprintln!("{} output: {}", mode_name, output);
        logits_data.push(logits_step);
    }

    Ok(logits_data)
}

fn internal_generate_logits<M>(format: RawSafeTensors, args: &Args) -> Result<()>
where
    M: LLMModelLoader<RawSafeTensors>
        + FPTransformModel
        + TokenizerLoader<RawSafeTensors>
        + Clone
        + Default,
{
    let model_type = M::default();
    let tokenizer = model_type.load_tokenizer(&format)?;
    let user_tokens = if let Some(tokens) = &args.tokens {
        eprintln!("Using provided tokens!");
        tokens
            .iter()
            .map(|&t| Token::from(t))
            .collect::<Vec<Token>>()
    } else {
        tokenizer.tokenize(&args.text)
    };
    eprintln!(
        "First 10 tokens: {:?}",
        &user_tokens[..user_tokens.len().min(10)]
    );
    let max_context = if args.sample_size != 0 {
        args.num_tokens + args.sample_size
    } else {
        args.num_tokens + user_tokens.len()
    };
    let mut driver = Driver::load_from_model(model_type.clone(), &format, Some(max_context))?;
    driver.with_max_context(max_context);
    if args.sample_size != 0 {
        let chunked_tokens = user_tokens
            .chunks_exact(args.sample_size + 1)
            .map(|chunk| &chunk[..args.sample_size])
            .collect::<Vec<&[Token]>>();
        // Run float mode
        eprintln!(
            "\n{} Running Float Mode {}",
            SECTION_SEPARATOR, SECTION_SEPARATOR
        );
        let logits_float = run_multi_sample_generation::<_, _>(
            &driver,
            &tokenizer,
            &chunked_tokens,
            "Float",
            |data, _node_id| Ok(data.to_vec()),
        )?;
        // Reset cache and convert to provable mode
        eprintln!(
            "\n{} Running Integer/Provable Mode {}",
            SECTION_SEPARATOR, SECTION_SEPARATOR
        );
        driver.model.reset();

        let (driver_int, metadata) = driver.into_provable_llm_with_transform::<M>(&tokenizer)?;

        // Run provable mode
        let logits_int = run_multi_sample_generation::<_, _>(
            &driver_int,
            &tokenizer,
            &chunked_tokens,
            "Integer",
            |data, node_id| {
                let scaling_factors = metadata.layer_input_scaling_factor(node_id);
                let scaling_factor = scaling_factors
                    .first()
                    .context("Failed to get scaling factor for logits layer")?;
                Ok(data.dequantize(scaling_factor))
            },
        )?;

        write_logits_to_json(&logits_float, &logits_int, driver_int.vocab_size())?;
    } else {
        let max_context = args.num_tokens + user_tokens.len();

        // Run float mode
        eprintln!(
            "\n{} Running Float Mode {}",
            SECTION_SEPARATOR, SECTION_SEPARATOR
        );
        let logits_float = run_generation::<_, _>(
            &driver,
            &tokenizer,
            &user_tokens,
            "Float",
            |data, _node_id| Ok(data.to_vec()),
        )?;
        // Reset cache and convert to provable mode
        eprintln!(
            "\n{} Running Integer/Provable Mode {}",
            SECTION_SEPARATOR, SECTION_SEPARATOR
        );
        driver.model.reset();

        let (mut driver_int, metadata) =
            driver.into_provable_llm_with_transform::<M>(&tokenizer)?;
        driver_int.with_max_context(max_context);

        // Run provable mode
        let logits_int = run_generation::<_, _>(
            &driver_int,
            &tokenizer,
            &user_tokens,
            "Integer",
            |data, node_id| {
                let scaling_factors = metadata.layer_input_scaling_factor(node_id);
                let scaling_factor = scaling_factors
                    .first()
                    .context("Failed to get scaling factor for logits layer")?;
                Ok(data.dequantize(scaling_factor))
            },
        )?;

        write_logits_to_json(&[logits_float], &[logits_int], driver_int.vocab_size())?;
    };
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();

    // Validate model files exist
    let (model_file, tokenizer_file, config_file) = validate_model_files(&args.model)?;

    let format = RawSafeTensors::new(model_file, tokenizer_file, config_file);

    // Detect model type from metadata
    let model_names = format.model_metadata()?;

    // Determine which model to use based on metadata
    let is_gpt2 = is_gpt2_model(&model_names);
    let is_gemma3 = model_names
        .iter()
        .any(|name| name.to_lowercase().contains("gemma"));

    eprintln!("is_gpt2 {is_gpt2}");

    if is_gpt2 {
        internal_generate_logits::<GPT2>(format, &args)?;
    } else if is_gemma3 {
        internal_generate_logits::<Gemma3>(format, &args)?;
    } else if model_names
        .iter()
        .any(|name| name.to_lowercase().contains("llama"))
    {
        llama3_logits(format, &args)?;
    } else {
        bail!("Unsupported model type. Supported: GPT-2, Gemma3, Llama3.");
    }

    Ok(())
}
