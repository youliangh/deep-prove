use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, bail, ensure};
use ark_bn254::Bn254;
use clap::{ArgGroup, Parser, ValueEnum, builder::ArgPredicate};
#[cfg(not(feature = "cuda"))]
use dp_crypto::arkyper::HyperKZG;
#[cfg(feature = "cuda")]
use dp_crypto::arkyper::hyperkzg_gpu::HyperKZGGpu;
use libc::{RUSAGE_SELF, getrusage, rusage};
use sha2::{Digest, Sha256};
use tenstore::GenStore;
use timed_core::Output;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use zkml::{
    ProverContext,
    measure::{self, Measure},
    model::{
        exec_graph::InferenceEngine,
        llm::{Driver, LLMVerifierContext, WithMaxContext, generation_context_length},
        llm_artifact::{ARTIFACT_VERSION, ArtifactKind, decode_verifier, encode_artifact},
    },
    parser::{
        file_cache,
        gguf::RawGGUF,
        llm::{
            HFTokenizer, LLMTokenizer, Token,
            models::{gemma3::Gemma3, gpt2::GPT2, llama2::Llama2, llama3::Llama3, qwen35::Qwen35},
            tokenizer::TokenizerLoader,
        },
        safe::RawSafeTensors,
    },
};

type F = ark_bn254::Fr;
// the hasher type is chosen depending on the feature flag inside the mpcs crate
#[cfg(not(feature = "cuda"))]
type Pcs = HyperKZG<Bn254>;
#[cfg(feature = "cuda")]
type Pcs = HyperKZGGpu<Bn254>;

#[derive(Clone, Debug, ValueEnum)]
#[clap(rename_all = "lower")]
enum Model {
    GPT2,
    Gemma3,
    Llama2,
    Llama3,
    Qwen35,
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Model::GPT2 => write!(f, "gpt2"),
            Model::Gemma3 => write!(f, "gemma3"),
            Model::Llama2 => write!(f, "llama2"),
            Model::Llama3 => write!(f, "llama3"),
            Model::Qwen35 => write!(f, "qwen35"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    group(ArgGroup::new("weights").args(["gguf", "hf", "hf_dir"]).required(true)),
    group(ArgGroup::new("prompt_input").args(["prompt", "prompt_tokens", "prompt_cases"])),
)]
struct LLMArgs {
    /// gguf file to load. It can be a local path or a URL to download.
    #[arg(short, long)]
    gguf: Option<String>,

    /// Hugging Face model ID. It MUST be a safetensors model for now. If the model isn't present
    /// in the cache, it will be downloaded.
    #[arg(long)]
    hf: Option<String>,

    /// Local HF directory containing model.safetensors, tokenizer.json and config.json.
    #[arg(long)]
    hf_dir: Option<PathBuf>,

    /// Text prompt to generate and prove, instead of random benchmark tokens.
    #[arg(long, conflicts_with = "prompt_tokens")]
    prompt: Option<String>,

    /// Apply the local Qwen3.5 non-thinking text chat format.
    #[arg(long, requires = "prompt")]
    chat: bool,

    /// JSON array of exact prompt token IDs, including any desired BOS token.
    #[arg(long)]
    prompt_tokens: Option<PathBuf>,

    /// JSON cases with name, prompt_tokens and max_new_tokens; calibrate and set up once.
    #[arg(long)]
    prompt_cases: Option<PathBuf>,

    /// Maximum number of generated tokens for --prompt or --prompt-tokens.
    #[arg(
        long,
        default_value_t = 64,
        requires = "prompt_input",
        conflicts_with = "prompt_cases"
    )]
    max_new_tokens: usize,

    /// JSON array of token-ID arrays used for Llama 3 or Qwen3.5 calibration.
    #[arg(long)]
    calibration_tokens: Option<PathBuf>,

    /// Number of leading calibration samples to use (increase 8 -> 16 -> 32).
    #[arg(long, default_value_t = 8, requires = "calibration_tokens")]
    calibration_samples: usize,

    /// Save generation-unverified diagnostics, then verified proof/IO/key/tokens on success.
    #[arg(long)]
    artifacts_dir: Option<PathBuf>,

    /// max context length (in tokens)
    #[arg(long)]
    max_context: Option<usize>,

    /// When specifying a sequence, the model will try each difference sequence length, just generating 2 tokens each time
    /// So for seqlen = n, it will start with a prompt of n-2 tokens, and generates two tokens.
    #[arg(long, value_delimiter = ',', conflicts_with_all = ["max_context", "prompt", "prompt_tokens", "prompt_cases"])]
    sequence: Vec<usize>,

    /// min user input length (in tokens)
    #[arg(
        long,
        requires = "max_context",
        conflicts_with = "sequence",
        default_value_t = 1
    )]
    min_user_len: usize,

    /// model to use
    #[arg(short, long, value_enum)]
    model: Model,

    /// DEPRECATED: output file name that records individual methods
    #[arg(short, long, default_value_t = {"bench-llm-deprecated.csv".to_string()})]
    output: String,

    /// Benchmark csv output file name
    #[arg(
        long,
        default_value_if("distributed", ArgPredicate::IsPresent, Some("bench_distributed.csv")),
        default_value_if("measure_layerwise", ArgPredicate::IsPresent, Some("bench_layerwise.csv")),
        default_value = "bench.csv" // Optional: what to use if NOT distributed
    )]
    bench: String,

    /// How many rayon threads to use
    /// If not provided, will use the number of logical cores
    /// If 0, will use the number of physical cores
    #[arg(long)]
    num_threads: Option<usize>,

    /// Profile distributed execution
    #[arg(long, default_value_t = false)]
    distributed: bool,

    /// Use improved accuracy model quantisation and transforms
    #[arg(long, default_value_t = false)]
    accuracy: bool,

    /// basename of the param files to use for the prover and verifier.
    /// e.g. "--params setup" will search for "setup.pk" and "setup.vk" in the current directory.
    #[arg(long)]
    load_params: Option<String>,

    /// Basename of the prover and verifier context files to save.
    /// e.g. "--save_params setup" will save the prover and verifier context to "setup.pk" and "setup.vk" in the current directory.
    #[arg(long)]
    save_params: Option<String>,

    #[arg(long)]
    memory: bool,

    #[arg(long, requires = "distributed")]
    num_chunks: Option<usize>,

    /// Measure layer-wise metrics
    #[arg(long, default_value_t = false)]
    measure_layerwise: bool,
}

const HEADER_MODEL: &str = "model_name";
const HEADER_MODEL_QUANT: &str = "quantization_time";
const HEADER_CONTEXT_GENERATION: &str = "context_generation_time";
const HEADER_MAX_CONTEXT: &str = "max_context";
const HEADER_NUM_THREADS: &str = "num_threads";
const HEADER_MIN_USER_LEN: &str = "min_user_len";
const HEADER_INFERENCE_TIME: &str = "inference_time";
const HEADER_PROOF_SIZE: &str = "proof_size";

#[derive(Debug, serde::Deserialize)]
struct GenerationCase {
    name: String,
    prompt_tokens: Vec<usize>,
    max_new_tokens: usize,
}

fn validate_cases(cases: &[GenerationCase]) -> anyhow::Result<()> {
    ensure!(!cases.is_empty(), "Prompt case list must not be empty");
    let mut names = std::collections::HashSet::new();
    for case in cases {
        ensure!(
            !case.name.is_empty()
                && case
                    .name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_'),
            "Case names must contain only ASCII letters, digits, '-' and '_'"
        );
        ensure!(
            names.insert(&case.name),
            "Duplicate case name: {}",
            case.name
        );
        generation_context_length(case.prompt_tokens.len(), case.max_new_tokens)?;
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let subscriber = tracing_subscriber::fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .finish();

    tracing::subscriber::set_global_default(subscriber).expect("Failed to set global subscriber");
    timed_core::set_output(Output::CSV("bench-llm.csv".to_string()));

    let args = LLMArgs::parse();
    ensure!(
        !matches!(args.model, Model::Llama3 | Model::Qwen35) || args.load_params.is_none(),
        "Calibrated model --load-params is disabled until cached parameters bind the model and calibration fingerprints"
    );

    // either its spceified and if 0 it's the physical cores otherwise what is specified but no more than the logical cores
    let num_threads = if let Some(nt) = args.num_threads {
        if nt == 0 {
            num_cpus::get_physical()
        } else {
            nt.min(num_cpus::get())
        }
    } else {
        num_cpus::get()
    };
    info!("Using {} threads", num_threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    ensure!(
        !args.chat || matches!(args.model, Model::Qwen35),
        "--chat requires --model qwen35"
    );
    let source = load_source(&args)?;
    let model_fingerprints = if args.artifacts_dir.is_some() {
        match &source {
            ModelSource::Safe(raw) => Some(serde_json::json!({
                "weights_sha256": sha256_hf_weights(raw)?,
                "weights_index_sha256": if raw.model_path().extension().is_some_and(|extension| extension == "json") {
                    Some(sha256_file(raw.model_path())?)
                } else { None },
                "config_sha256": sha256_file(raw.config_path())?,
                "tokenizer_sha256": sha256_file(raw.tokenizer_path())?
            })),
            ModelSource::Gguf(_) => None,
        }
    } else {
        None
    };
    let tokeniser = load_tokenizer(&args, &source)?;
    let prompt_tokens = if let Some(path) = &args.prompt_tokens {
        let ids: Vec<usize> = serde_json::from_reader(File::open(path)?)
            .with_context(|| format!("Reading prompt tokens from {}", path.display()))?;
        Some(ids.into_iter().map(Token::from).collect::<Vec<_>>())
    } else {
        args.prompt.as_ref().map(|prompt| {
            let text = if args.chat {
                zkml::parser::llm::models::qwen35::chat_prompt(prompt)
            } else {
                prompt.clone()
            };
            tokeniser.tokenize(&text)
        })
    };
    let prompt_cases = args
        .prompt_cases
        .as_ref()
        .map(|path| -> anyhow::Result<_> {
            let cases: Vec<GenerationCase> = serde_json::from_reader(File::open(path)?)
                .with_context(|| format!("Reading prompt cases from {}", path.display()))?;
            validate_cases(&cases)?;
            Ok(cases)
        })
        .transpose()?;
    let calibration = args
        .calibration_tokens
        .as_ref()
        .map(|path| -> anyhow::Result<_> {
            ensure!(
                matches!(args.model, Model::Llama3 | Model::Qwen35),
                "--calibration-tokens requires --model llama3 or qwen35"
            );
            let mut rows: Vec<Vec<usize>> = serde_json::from_reader(File::open(path)?)
                .with_context(|| format!("Reading calibration tokens from {}", path.display()))?;
            ensure!(
                !rows.is_empty() && rows.iter().all(|row| !row.is_empty()),
                "Calibration samples must not be empty"
            );
            ensure!(
                args.calibration_samples > 0 && args.calibration_samples <= rows.len(),
                "--calibration-samples must be in 1..={}",
                rows.len()
            );
            rows.truncate(args.calibration_samples);
            Ok(rows
                .into_iter()
                .map(|row| row.into_iter().map(Token::from).collect::<Vec<_>>())
                .collect::<Vec<_>>())
        })
        .transpose()?;
    let calibration_json = serde_json::to_vec(&calibration)?;
    let calibration_sha256 = format!("{:x}", Sha256::digest(&calibration_json));
    let calibration_context = calibration
        .as_ref()
        .and_then(|rows| rows.iter().map(Vec::len).max())
        .unwrap_or(0);
    let (max_context, sequence) = if prompt_tokens.is_some() || prompt_cases.is_some() {
        let sequence = if let Some(cases) = &prompt_cases {
            cases
                .iter()
                .map(|case| {
                    Ok((
                        case.prompt_tokens.len(),
                        generation_context_length(case.prompt_tokens.len(), case.max_new_tokens)?,
                    ))
                })
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            let tokens = prompt_tokens.as_ref().unwrap();
            vec![(
                tokens.len(),
                generation_context_length(tokens.len(), args.max_new_tokens)?,
            )]
        };
        let generation_context = sequence.iter().map(|(_, context)| *context).max().unwrap();
        let model_context = args
            .max_context
            .unwrap_or(generation_context.max(calibration_context));
        ensure!(
            generation_context <= model_context,
            "Prompt plus requested generation exceeds --max-context"
        );
        ensure!(
            calibration_context <= model_context,
            "Calibration samples exceed --max-context"
        );
        (model_context, sequence)
    } else if let Some(max_context) = args.max_context {
        ensure!(
            args.min_user_len > 0 && args.min_user_len <= max_context,
            "Random prompt length must be in 1..=max-context"
        );
        (max_context, vec![(args.min_user_len, max_context)])
    } else {
        let mut sequence = args.sequence.clone();
        sequence.sort();
        ensure!(
            !sequence.is_empty(),
            "Provide --prompt, --prompt-tokens, --prompt-cases, --sequence or --max-context"
        );
        ensure!(
            sequence.iter().all(|length| *length >= 3),
            "Benchmark sequence lengths must be at least 3"
        );
        (
            *sequence.last().unwrap(),
            sequence
                .into_iter()
                .map(|length| (length - 2, length))
                .collect(),
        )
    };
    info!(
        "Loading model with context capacity {max_context}; prompt and certification lengths: {sequence:?}"
    );
    let driver = parse_model(&args, &source, max_context)?;
    if let Some(tokens) = &prompt_tokens {
        driver.tokens_to_tensor(tokens)?;
    }
    if let Some(cases) = &prompt_cases {
        for case in cases {
            let tokens = case
                .prompt_tokens
                .iter()
                .copied()
                .map(Token::from)
                .collect::<Vec<_>>();
            driver.tokens_to_tensor(&tokens)?;
        }
    }
    if let Some(samples) = &calibration {
        for sample in samples {
            driver.tokens_to_tensor(sample)?;
        }
    }

    let mut premeasure = Measure::new()
        .with(HEADER_MODEL, &args.model.to_string())
        .with(HEADER_NUM_THREADS, &num_threads.to_string());
    if args.measure_layerwise {
        premeasure = premeasure.enable_layerwise_measures();
    }
    info!("Converting model into provable model...");
    let (mut driver, metadata) = if matches!(args.model, Model::Llama3) {
        premeasure.r(HEADER_MODEL_QUANT, || {
            zkml::quantization::llm_quant::llama3::quantize(driver, &tokeniser, calibration)
        })?
    } else if matches!(args.model, Model::Qwen35) {
        premeasure.r(HEADER_MODEL_QUANT, || {
            zkml::quantization::llm_quant::qwen35::quantize(driver, &tokeniser, calibration)
        })?
    } else if !args.accuracy {
        premeasure.r(HEADER_MODEL_QUANT, || driver.into_provable_llm(None))?
    } else {
        match args.model {
            Model::GPT2 => premeasure.r(HEADER_MODEL_QUANT, || {
                driver.into_provable_llm_with_transform::<GPT2>(&tokeniser)
            })?,
            Model::Gemma3 => premeasure.r(HEADER_MODEL_QUANT, || {
                driver.into_provable_llm_with_transform::<Gemma3>(&tokeniser)
            })?,
            Model::Llama2 => premeasure.r(HEADER_MODEL_QUANT, || driver.into_provable_llm(None))?,
            Model::Llama3 | Model::Qwen35 => {
                unreachable!("Model uses calibrated quantization above")
            }
        }
    };
    let quantization_sha256 = if matches!(args.model, Model::Llama3 | Model::Qwen35)
        && let Some(directory) = &args.artifacts_dir
    {
        let snapshot = if matches!(args.model, Model::Qwen35) {
            zkml::quantization::llm_quant::qwen35::compiled_quantization_snapshot(
                &driver, &metadata,
            )?
        } else {
            zkml::quantization::llm_quant::llama3::calibration_snapshot(&metadata)?
        };
        let bytes = serde_json::to_vec(&snapshot)?;
        let digest = format!("{:x}", Sha256::digest(&bytes));
        std::fs::create_dir_all(directory)?;
        std::fs::write(directory.join("quantization.json"), bytes)?;
        Some(digest)
    } else {
        None
    };

    let (prover_ctx, mut verifier_ctx) = if let Some(ref params) = args.load_params {
        info!("Loading proving contexts from {params}.pk and {params}.vk...");
        let prover_ctx = bincode::serde::decode_from_slice(
            &std::fs::read(format!("{params}.pk"))?,
            bincode::config::standard(),
        )?
        .0;
        let verifier_ctx = decode_verifier(&std::fs::read(format!("{params}.vk"))?)?;
        // set to 0 since we're not spending any time generating it
        premeasure.r(HEADER_CONTEXT_GENERATION, || 0);
        (prover_ctx, verifier_ctx)
    } else {
        info!("Generating proving contexts...");
        let (prover_ctx, verifier_ctx): (ProverContext<F, Pcs>, LLMVerifierContext<F, Pcs>) =
            premeasure.r(HEADER_CONTEXT_GENERATION, || driver.context())?;
        (prover_ctx, verifier_ctx)
    };

    if let Some(ref save_params) = args.save_params {
        if args.load_params.is_some() {
            bail!("Cannot save parameters if loading parameters is also specified");
        }
        info!("Saving proving contexts to {save_params}.pk and {save_params}.vk...");
        let mut io = File::create(format!("{save_params}.pk"))?;
        io.write_all(&bincode::serde::encode_to_vec(
            &prover_ctx,
            bincode::config::standard(),
        )?)?;
        let mut io = File::create(format!("{save_params}.vk"))?;
        io.write_all(&encode_artifact(ArtifactKind::Verifier, &verifier_ctx)?)?;
    }

    let multiple_sequences = sequence.len() > 1;
    let mut case_reports = Vec::new();
    for (case_index, (user_prompt, max_ctx)) in sequence.into_iter().enumerate() {
        let case = prompt_cases.as_ref().map(|cases| &cases[case_index]);
        let case_artifacts_dir = args.artifacts_dir.as_ref().map(|directory| {
            if let Some(case) = case {
                directory.join(&case.name)
            } else if multiple_sequences {
                directory.join(format!("sequence-{max_ctx}"))
            } else {
                directory.clone()
            }
        });
        // make a new measure for each trial, but always keep the initial measurements for each sample
        measure::set_global(premeasure.clone());

        measure::set(HEADER_MAX_CONTEXT, max_ctx.to_string());
        measure::set(HEADER_MIN_USER_LEN, user_prompt.to_string());

        driver.model.reset();
        driver.with_max_context(max_ctx);
        let user_tokens = if let Some(case) = case {
            case.prompt_tokens
                .iter()
                .copied()
                .map(Token::from)
                .collect()
        } else {
            prompt_tokens
                .clone()
                .unwrap_or_else(|| driver.random_sequence(user_prompt))
        };
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let (trace, chunks) = if args.distributed {
            let (chunks, split_node_info) = prover_ctx.split_in_chunks(
                args.num_chunks,
                driver.chunking_strategy(&input_tensor, &prover_ctx)?,
            )?;
            let engine = InferenceEngine::LLM(driver);
            let trace = measure::r(HEADER_INFERENCE_TIME, || {
                info!("Running inference...");
                engine.run(
                    vec![input_tensor],
                    &mut GenStore::default(),
                    &split_node_info,
                )
            })?;
            driver = match engine {
                InferenceEngine::LLM(driver) => driver,
                _ => bail!("Expected LLM engine"),
            };
            (trace, Some(chunks))
        } else {
            (
                measure::r(HEADER_INFERENCE_TIME, || {
                    info!("Running inference...");
                    driver.run_elements(input_tensor, &mut GenStore::default())
                })?,
                None,
            )
        };

        let generated_tokens = driver.continuation_tokens(&trace, user_tokens.len())?;
        let generated_text = tokeniser.detokenize(&generated_tokens);
        if let Some(directory) = &case_artifacts_dir {
            // Driver returns only after cached generation agrees with its full
            // causal replay. Keep that numerical evidence if proving later fails;
            // these separate files never claim cryptographic verification.
            save_unverified_generation(
                directory,
                serde_json::json!({
                    "case": case.map(|case| &case.name),
                    "model": args.model.to_string(),
                    "source": source.to_string(),
                    "model_fingerprints": model_fingerprints,
                    "model_context": max_context,
                    "certification_context": max_ctx,
                    "requested_max_new_tokens": max_ctx - user_prompt + 1,
                    "quantization_bits": *zkml::quantization::BIT_LEN,
                    "quantization_sha256": quantization_sha256,
                    "calibration_tokens_sha256": calibration_sha256,
                    "calibration_samples": args.calibration_tokens.as_ref().map(|_| args.calibration_samples),
                    "num_threads": num_threads,
                    "prompt_tokens": user_tokens.iter().map(u32::from).collect::<Vec<_>>(),
                    "generated_tokens": generated_tokens.iter().map(u32::from).collect::<Vec<_>>(),
                    "generated_text": generated_text,
                    "full_text": tokeniser.detokenize(&user_tokens.iter().chain(&generated_tokens).copied().collect::<Vec<_>>()),
                }),
            )?;
            info!(
                "Saved unverified generation diagnostics to {} before proving",
                directory.display()
            );
        }

        if args.memory {
            info!(
                "Running memory profiler - will save into {:?}",
                std::env::var("FLAMEGRAPH")
            );
            utils::track::flame_graph_enable();
        }
        let (proof, io) = if args.distributed {
            info!("Running distributed proving locally...");
            distributed::run_distributed(trace, &driver, &prover_ctx, chunks.unwrap())?
        } else {
            let peak_rss = peak_rss_bytes();
            info!("Running proving locally...");
            let (proof, io) = driver.prove(&prover_ctx, trace)?;
            let new_peak_rss = peak_rss_bytes();
            if new_peak_rss == peak_rss {
                warn!(
                    "Cannot reliably measure peak memory consumption during proving, setting upper bound"
                );
            }
            // new_peak_rss is the peak memory consumption during proving
            measure::set(
                "prove_full_memory_peak",
                (new_peak_rss / 1024 / 1024).to_string(),
            );

            (proof, io)
        };
        if args.memory {
            utils::track::flame_graph();
        }

        let proof_bytes = encode_artifact(ArtifactKind::Proof, &proof)?;
        measure::set(HEADER_PROOF_SIZE, proof_bytes.len());
        verifier_ctx = verifier_ctx.with_max_context(max_ctx);
        let verified_tokens = verifier_ctx.continuation_tokens(&user_tokens, &io)?;
        ensure!(
            verified_tokens == generated_tokens,
            "Proof IO differs from the generated continuation"
        );
        let serialized_artifacts = args
            .artifacts_dir
            .as_ref()
            .map(|_| -> anyhow::Result<_> {
                Ok((
                    encode_artifact(ArtifactKind::Io, &io)?,
                    encode_artifact(ArtifactKind::Verifier, &verifier_ctx)?,
                ))
            })
            .transpose()?;
        info!("Verifying proof...");
        verifier_ctx
            .verify(proof, user_tokens.clone(), io)
            .context("Generated proof failed verification")?;
        std::io::stdout()
            .write_all(format!("Verified continuation: {generated_text}\n").as_bytes())?;
        if let (Some(directory), Some((io_bytes, vk_bytes))) =
            (&case_artifacts_dir, serialized_artifacts)
        {
            std::fs::create_dir_all(&directory)?;
            let ids = |tokens: &[Token]| {
                tokens
                    .iter()
                    .map(|token| u32::from(token))
                    .collect::<Vec<_>>()
            };
            let all_tokens = user_tokens
                .iter()
                .chain(&generated_tokens)
                .copied()
                .collect::<Vec<_>>();
            let report = serde_json::json!({
                "verified": true,
                "case": case.map(|case| &case.name),
                "model": args.model.to_string(),
                "source": source.to_string(),
                "model_fingerprints": model_fingerprints,
                "verifier_sha256": format!("{:x}", Sha256::digest(&vk_bytes)),
                "model_context": max_context,
                "certification_context": max_ctx,
                "requested_max_new_tokens": max_ctx - user_prompt + 1,
                "quantization_bits": *zkml::quantization::BIT_LEN,
                "quantization_sha256": quantization_sha256,
                "calibration_source": args.calibration_tokens,
                "calibration_samples": args.calibration_tokens.as_ref().map(|_| args.calibration_samples),
                "calibration_tokens_sha256": calibration_sha256,
                "builtin_calibration_prompt": if matches!(args.model, Model::Llama3) && args.calibration_tokens.is_none() {
                    Some(zkml::quantization::get_default_prompt())
                } else { None },
                "num_threads": num_threads,
                "prompt_tokens": ids(&user_tokens),
                "generated_tokens": ids(&generated_tokens),
                "generated_text": generated_text,
                "full_text": tokeniser.detokenize(&all_tokens),
                "proof_bytes": proof_bytes.len(),
                "serialization": format!("deep-prove-llm-v{ARTIFACT_VERSION}/bincode-2-standard"),
                "artifact_version": ARTIFACT_VERSION,
                "pcs": "HyperKZG-BN254"
            });
            std::fs::write(directory.join("proof.bin"), &proof_bytes)?;
            std::fs::write(directory.join("io.bin"), io_bytes)?;
            std::fs::write(directory.join("verifier.bin"), vk_bytes)?;
            std::fs::write(directory.join("calibration_tokens.json"), &calibration_json)?;
            std::fs::write(
                directory.join("prompt_tokens.json"),
                serde_json::to_vec(&ids(&user_tokens))?,
            )?;
            std::fs::write(
                directory.join("generated_tokens.json"),
                serde_json::to_vec(&ids(&generated_tokens))?,
            )?;
            std::fs::write(directory.join("generated.txt"), generated_text)?;
            std::fs::write(
                directory.join("result.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
            case_reports.push(report);
            info!("Saved verified artifacts to {}", directory.display());
        }
        if !args.distributed {
            measure::post_process(|metrics| {
                let Ok(proof_time) = metrics.get("prove_full").unwrap().parse::<usize>() else {
                    return;
                };
                let Ok(ctx_length) = metrics.get(HEADER_MAX_CONTEXT).unwrap().parse::<usize>()
                else {
                    return;
                };
                let token_per_second = ctx_length as f64 / (proof_time as f64 / 1000.0);
                metrics.insert("token/sec".to_string(), token_per_second.to_string());
            })?;
        }
        measure::to_csv(&args.bench)?;
    }

    if prompt_cases.is_some()
        && let Some(directory) = &args.artifacts_dir
    {
        std::fs::write(
            directory.join("cases.json"),
            serde_json::to_vec_pretty(&case_reports)?,
        )?;
    }

    Ok(())
}

fn save_unverified_generation(
    directory: &Path,
    mut report: serde_json::Value,
) -> anyhow::Result<()> {
    let fields = report
        .as_object_mut()
        .context("Generation diagnostic must be a JSON object")?;
    fields.insert("verified".into(), false.into());
    fields.insert("proof_status".into(), "not_started".into());
    fields.insert("cached_generation_matches_full_pass".into(), true.into());
    fields.insert(
        "diagnostic_only".into(),
        "Native generation and causal replay passed; no cryptographic proof has been verified"
            .into(),
    );
    fields.insert(
        "recorded_at_unix_millis".into(),
        u64::try_from(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis(),
        )?
        .into(),
    );
    let text = fields
        .get("generated_text")
        .and_then(serde_json::Value::as_str)
        .context("Generation diagnostic must contain generated_text")?;
    std::fs::create_dir_all(directory)?;
    std::fs::write(directory.join("generation-unverified.txt"), text)?;
    std::fs::write(
        directory.join("generation-unverified.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(())
}

fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut r: rusage = std::mem::zeroed();
        getrusage(RUSAGE_SELF, &mut r);

        #[cfg(target_os = "linux")]
        {
            (r.ru_maxrss as u64) * 1024
        }

        #[cfg(any(target_os = "macos", target_os = "freebsd", target_os = "openbsd"))]
        {
            r.ru_maxrss as u64
        }
    }
}

fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let bytes = file.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        digest.update(&buffer[..bytes]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Bind indexed checkpoints to every actual shard, not just the JSON filenames.
fn sha256_hf_weights(raw: &RawSafeTensors) -> anyhow::Result<String> {
    if raw
        .model_path()
        .extension()
        .is_some_and(|ext| ext == "safetensors")
    {
        return sha256_file(raw.model_path());
    }
    let index: serde_json::Value = serde_json::from_reader(File::open(raw.model_path())?)?;
    let map = index["weight_map"]
        .as_object()
        .context("Missing SafeTensors weight_map")?;
    let directory = raw.model_path().parent().context("No model directory")?;
    let mut shards = std::collections::BTreeMap::new();
    for value in map.values() {
        let name = value.as_str().context("Invalid shard name")?;
        ensure!(
            Path::new(name)
                .components()
                .all(|p| matches!(p, std::path::Component::Normal(_))),
            "Unsafe shard path"
        );
        if !shards.contains_key(name) {
            shards.insert(name.to_owned(), sha256_file(&directory.join(name))?);
        }
    }
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&serde_json::json!({
            // The index maps tensor names to shards; identical shard files with
            // a different mapping must not share a model fingerprint.
            "index_sha256": sha256_file(raw.model_path())?,
            "shards": shards,
        }))?)
    ))
}

enum ModelSource {
    Gguf(RawGGUF),
    Safe(RawSafeTensors),
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf(raw) => write!(formatter, "{raw}"),
            Self::Safe(raw) => write!(formatter, "{raw}"),
        }
    }
}

fn load_source(args: &LLMArgs) -> anyhow::Result<ModelSource> {
    if let Some(path) = &args.gguf {
        return Ok(ModelSource::Gguf(RawGGUF::new(file_cache::from_cache(
            path,
        )?)));
    }
    if let Some(directory) = &args.hf_dir {
        return Ok(ModelSource::Safe(RawSafeTensors::from_directory(
            directory,
        )?));
    }
    let repository = args
        .hf
        .as_deref()
        .context("Provide --hf-dir, --hf or --gguf")?;
    Ok(ModelSource::Safe(RawSafeTensors::from_hugging_face_cached(
        repository,
    )?))
}

fn load_tokenizer(args: &LLMArgs, source: &ModelSource) -> anyhow::Result<HFTokenizer> {
    match source {
        ModelSource::Safe(raw) => HFTokenizer::from_tokenizer_json_path(raw.tokenizer_path()),
        ModelSource::Gguf(raw) => match args.model {
            Model::GPT2 => GPT2::new().load_tokenizer(raw),
            _ => bail!("Model {:?} not supported for gguf", args.model),
        },
    }
}

fn parse_model(
    args: &LLMArgs,
    source: &ModelSource,
    max_context: usize,
) -> anyhow::Result<Driver<f32>> {
    match source {
        ModelSource::Gguf(raw) => match args.model {
            Model::GPT2 => Driver::load_from_model(GPT2::new(), raw, Some(max_context)),
            _ => bail!("Model {:?} not supported for gguf", args.model),
        },
        ModelSource::Safe(raw) => match args.model {
            Model::GPT2 => Driver::load_from_model(GPT2::new(), raw, Some(max_context)),
            Model::Gemma3 => Driver::load_from_model(Gemma3::new(), raw, Some(max_context)),
            Model::Llama2 => Driver::load_from_model(Llama2::new(), raw, Some(max_context)),
            Model::Llama3 => Driver::load_from_model(Llama3::new(), raw, Some(max_context)),
            Model::Qwen35 => Driver::load_from_model(Qwen35::new(), raw, Some(max_context)),
        },
    }
}

#[cfg(test)]
mod cli_tests {
    use super::*;

    #[test]
    fn generation_diagnostic_is_unverified_and_preserves_verified_artifacts() -> anyhow::Result<()>
    {
        let directory = std::env::temp_dir().join(format!(
            "deep-prove-generation-diagnostic-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let result = (|| {
            save_unverified_generation(
                &directory,
                serde_json::json!({
                    "verified": true,
                    "prompt_tokens": [1, 2],
                    "generated_tokens": [3, 4],
                    "generated_text": "A native continuation",
                }),
            )?;
            assert_eq!(std::fs::read_dir(&directory)?.count(), 2);
            for name in ["result.json", "proof.bin", "io.bin", "generated.txt"] {
                assert!(!directory.join(name).exists());
            }
            let diagnostic: serde_json::Value = serde_json::from_slice(&std::fs::read(
                directory.join("generation-unverified.json"),
            )?)?;
            assert_eq!(diagnostic["verified"], false);
            assert_eq!(diagnostic["proof_status"], "not_started");
            assert_eq!(diagnostic["cached_generation_matches_full_pass"], true);
            assert_eq!(diagnostic["generated_tokens"], serde_json::json!([3, 4]));
            assert_eq!(
                std::fs::read_to_string(directory.join("generation-unverified.txt"))?,
                "A native continuation"
            );

            // A retry may share a directory with a previously verified run.
            // Writing fresh diagnostics must leave those existing artifacts intact.
            let original = b"previously verified result";
            std::fs::write(directory.join("result.json"), original)?;
            save_unverified_generation(
                &directory,
                serde_json::json!({"generated_text": "A later diagnostic"}),
            )?;
            assert_eq!(std::fs::read(directory.join("result.json"))?, original);
            anyhow::Ok(())
        })();
        let cleanup = std::fs::remove_dir_all(&directory);
        result?;
        cleanup?;
        Ok(())
    }

    #[test]
    fn accepts_local_prompt_and_existing_sequence_modes() {
        let args = LLMArgs::try_parse_from([
            "bench-llm",
            "--model",
            "llama3",
            "--hf-dir",
            "/models/llama3",
            "--prompt",
            "The capital of France is",
            "--max-new-tokens",
            "64",
            "--max-context",
            "256",
            "--calibration-tokens",
            "/tmp/calibration.json",
            "--calibration-samples",
            "8",
            "--artifacts-dir",
            "/tmp/proof",
        ])
        .unwrap();
        assert_eq!(args.max_context, Some(256));
        assert_eq!(args.max_new_tokens, 64);
        assert!(
            LLMArgs::try_parse_from([
                "bench-llm",
                "--model",
                "gpt2",
                "--hf",
                "openai-community/gpt2",
                "--sequence",
                "32,64",
            ])
            .is_ok()
        );
        assert!(
            LLMArgs::try_parse_from([
                "bench-llm",
                "--model",
                "llama3",
                "--hf-dir",
                "/models/llama3",
                "--prompt-tokens",
                "/tmp/prompt.json",
            ])
            .is_ok()
        );
    }

    #[test]
    fn rejects_ambiguous_prompt_and_weight_sources() {
        assert!(
            LLMArgs::try_parse_from([
                "bench-llm",
                "--model",
                "llama3",
                "--hf-dir",
                "/models/llama3",
                "--prompt",
                "Hello",
                "--prompt-tokens",
                "/tmp/prompt.json",
            ])
            .is_err()
        );
        assert!(
            LLMArgs::try_parse_from([
                "bench-llm",
                "--model",
                "llama3",
                "--hf-dir",
                "/models/llama3",
                "--hf",
                "meta-llama/Llama-3.2-1B",
                "--prompt",
                "Hello",
            ])
            .is_err()
        );
    }

    #[test]
    fn accepts_frozen_batch_cases_and_rejects_conflicting_generation_count() {
        let args = [
            "bench-llm",
            "--model",
            "llama3",
            "--hf-dir",
            "/models/llama3",
            "--prompt-cases",
            "/tmp/prompts.json",
            "--max-context",
            "256",
        ];
        assert!(LLMArgs::try_parse_from(args).is_ok());
        assert!(
            LLMArgs::try_parse_from(args.into_iter().chain(["--max-new-tokens", "32"])).is_err()
        );

        let mut cases = vec![GenerationCase {
            name: "factual".into(),
            prompt_tokens: vec![128000, 1, 2],
            max_new_tokens: 32,
        }];
        assert!(validate_cases(&cases).is_ok());
        cases.push(GenerationCase {
            name: "factual".into(),
            prompt_tokens: vec![128000, 3],
            max_new_tokens: 64,
        });
        assert!(validate_cases(&cases).is_err());
        cases.pop();
        cases[0].name = "../escape".into();
        assert!(validate_cases(&cases).is_err());
        cases[0].name = "context256".into();
        cases[0].max_new_tokens = 0;
        assert!(validate_cases(&cases).is_err());
    }
}

mod distributed {
    use std::collections::HashMap;

    use super::*;
    use anyhow::{anyhow, ensure};
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use tracing::debug;

    use zkml::{
        Element, IO, Proof, Prover,
        graph::{
            executor::{Executor, ThreadPoolExecutor},
            partition::PartitionScheduler,
            scheduler::ExecGraph,
        },
        iop::{
            chunking::ModelChunk,
            prover_graph::{LocalProverCtx, ProverGraphIO, ProverGraphNode},
        },
        model::Trace,
    };

    pub type T = Blake3Transcript;

    // Type of nodes of the graph to execute
    pub type Node<'a, 'b> = ProverGraphNode<'a, 'b, F, T, Pcs>;

    // Type of execution graph to be partitioned and executed in the workers
    pub type Graph<'a, 'b> = ExecGraph<Node<'a, 'b>, Color>;

    // Color is used to create the partitions, assign different nodes to different workers.
    // It can be usize or any other type such as IP address etc.
    pub type Color = usize;

    /// What a partition scheduler outputs
    pub type PartitionOutput<'a, 'b> = zkml::graph::partition::PartitionOutput<Node<'a, 'b>, Color>;

    const CHUNK_OUTPUT_SIZE: &str = "chunk_output_size";
    const CHUNK_INPUT_SIZE: &str = "chunk_input_size";

    fn run_next_partition<'a, 'b, E: Executor<Node<'a, 'b>, Color>>(
        schedulers: &mut HashMap<Color, PartitionScheduler<Node<'a, 'b>, Color, E>>,
    ) -> anyhow::Result<Option<PartitionOutput<'a, 'b>>> {
        let mut to_be_sent_outputs = Vec::new();
        let mut final_output = None;
        let mut done_schedulers = Vec::new();
        for (color, scheduler) in schedulers.iter_mut() {
            let outputs = scheduler.try_run_partition()?;
            for out in outputs {
                if let Some(to_node) = out.to {
                    let serialized_output = rmp_serde::to_vec(&out)?;
                    // ToDo: measure serialized_output
                    if to_node == 0 {
                        // this is data sent to the coordinator, so we add it to the set of data sent by workers
                        // to the coordinator
                        measure::accumulate_key(
                            CHUNK_OUTPUT_SIZE,
                            serialized_output.len(),
                            |a, b| a + b,
                        )?
                    } else if *color == 0 {
                        // this is data sent by the coordinator, so we add it to the set of data sent to the workers
                        measure::accumulate_key(
                            CHUNK_INPUT_SIZE,
                            serialized_output.len(),
                            |a, b| a + b,
                        )?
                    } else {
                        unreachable!(
                            "Data is either sent to coordinator or received by coordinator"
                        )
                    };
                    debug!("Node {} sending output to node {}", color, to_node);
                    to_be_sent_outputs.push((to_node, out));
                } else {
                    // we found the final output
                    final_output = Some(out);
                }
                if scheduler.is_done() {
                    done_schedulers.push(*color);
                }
            }
        }

        for (dest_color, out) in to_be_sent_outputs {
            schedulers
                .get_mut(&dest_color)
                .ok_or(anyhow!("Scheduler not found for color {dest_color}"))?
                .set_child_partition_output(out)?
        }

        for color in done_schedulers {
            schedulers.remove(&color);
        }

        Ok(final_output)
    }

    #[allow(clippy::type_complexity)]
    pub(super) fn run_distributed(
        full_trace: Trace<Element>,
        driver: &Driver<Element>,
        prover_ctx: &ProverContext<F, Pcs>,
        chunks: Vec<ModelChunk>,
    ) -> anyhow::Result<(Proof<F, Pcs>, IO<F>)> {
        let io = full_trace.to_verifier_io()?;

        let graph: Graph = Prover::build_execution_graph(chunks)?;

        let inputs = Prover::graph_inputs(full_trace, &graph)?;

        ensure!(
            inputs.len() == 1,
            "Expected exactly one input node (coordinator split)"
        );

        let flat_inputs =
            inputs
                .into_iter()
                .fold(Vec::new(), |mut ios, (node_input, chunk_prover_io)| {
                    ios.push((node_input.node_id(), chunk_prover_io));
                    ios
                });
        let partitions = graph.partition_by_color(flat_inputs)?;

        let mut schedulers = partitions
            .into_iter()
            .map(|(color, partitions)| {
                let ctx = LocalProverCtx::new(prover_ctx, &driver.model);
                Ok((
                    color,
                    PartitionScheduler::<_, _, ThreadPoolExecutor>::new(partitions, ctx, ())?,
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        let mut final_outputs = Vec::new();
        let peak_rss = peak_rss_bytes();
        while !schedulers.is_empty() {
            if let Some(final_output) = run_next_partition(&mut schedulers)? {
                final_outputs.push(final_output.output)
            };
        }
        let new_peak_rss = peak_rss_bytes();
        if new_peak_rss == peak_rss {
            warn!(
                "Cannot reliably measure peak memory consumption during proving, setting upper bound"
            );
        }
        measure::set(
            "prove_full_memory_peak",
            (new_peak_rss / 1024 / 1024).to_string(),
        );

        // Creates channels pairs to communicate with all other nodes
        ensure!(
            final_outputs.len() == 1,
            "Expected 1 outputs for the graph, {} outputs received",
            final_outputs.len()
        );
        let proof = match final_outputs.pop().unwrap() {
            ProverGraphIO::FinalProof(proof) => proof,
            _ => bail!("Invalid output type found after execution of ProverGraph"),
        };
        Ok((proof, io))
    }
}
