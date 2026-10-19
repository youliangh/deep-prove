//! This module implements a prover instance that generates proofs completely
//! locally, in a one-shot manner. After a successful proof generation, they are
//! written to a local file.
use std::{fmt::Display, io::BufWriter, path::PathBuf};

use anyhow::Context;
use ark_bn254::Bn254;
use deep_prove::store::MemStore;
use dp_crypto::arkyper::HyperKZG;
use memmap2::Mmap;
use tempfile::Builder;
use tenstore::GenStore;
use tracing::info;
use zkml::{
    Element,
    inputs::Input,
    model::llm::{Driver, LLMProof, WithMaxContext, generation_context_length},
    parser::{
        ModelLoader, ModelNameProvider,
        gguf::RawGGUF,
        llm::{
            LLMTokenizer, Token,
            metadata::LLMMetadata,
            models::{
                gemma3::Gemma3,
                gpt2::{GPT2, is_gpt2_model},
                llama3::Llama3,
            },
            tokenizer::{HFTokenizer, TokenizerLoader},
        },
        safe::RawSafeTensors,
    },
    quantization::ScalingStrategyKind,
};

use crate::{ModelFormat, RunMode};
use deep_prove::middleware::llm::LlmOneShotOutput;

type F = ark_bn254::Fr;
type Pcs = HyperKZG<Bn254>;

/// Run the prover once, directly feeding it the required inputs. The proofs are
/// written to a file.
pub async fn run(args: RunMode, tenstore: GenStore) -> anyhow::Result<()> {
    let RunMode::OneShot {
        model,
        model_format,
        inputs,
        prompt,
        tokenizer,
        config,
        max_new_tokens,
    } = args
    else {
        unreachable!()
    };

    let _telemetry_guard = telemetry::setup_logging("deep-prove-worker", false);

    match model_format {
        ModelFormat::Onnx => run_one_shot_onnx(model, inputs, tenstore).await,
        ModelFormat::Gguf => {
            run_one_shot_llm(
                LlmModel::Gguf { model_path: model },
                prompt,
                max_new_tokens,
                tenstore,
            )
            .await
        }
        ModelFormat::Safetensors => {
            run_one_shot_llm(
                LlmModel::Safetensors {
                    model_path: model,
                    tokenizer_path: tokenizer,
                    config_path: config,
                },
                prompt,
                max_new_tokens,
                tenstore,
            )
            .await
        }
    }
}

async fn run_one_shot_onnx(
    model: PathBuf,
    inputs: Option<PathBuf>,
    tenstore: GenStore,
) -> anyhow::Result<()> {
    let request_span = tracing::info_span!("dp_worker_prove_inference", proof_id = %"one-shot");
    let _entered = request_span.enter();
    let inputs =
        inputs.context("inputs are required for ONNX one-shot proving (--inputs <file>)")?;

    let input = Input::from_file(&inputs).context("loading input")?;
    let model_file = std::fs::File::open(&model).context("opening model file")?;
    let model = unsafe { Mmap::map(&model_file) }
        .context("mmap-ing model file")?
        .to_vec();

    let scaling_strategy = ScalingStrategyKind::AbsoluteMax;
    let scaling_input_hash = None;

    let request = crate::DeepProveRequestV1 {
        model,
        model_file_hash: None,
        input,
        scaling_strategy,
        scaling_input_hash,
    };
    let store = MemStore::default();
    let proofs = crate::run_model_v1(request, store, tenstore, "one-shot-onnx".to_string()).await?;

    // create a file to write the proofs to
    let file = tempfile::Builder::new()
        .prefix("proof-")
        .suffix(".json")
        .rand_bytes(10)
        .disable_cleanup(true)
        .tempfile_in(std::env::current_dir().unwrap_or("./".into()))?;

    serde_json::to_writer(BufWriter::new(&file), &proofs).context("writing proofs to file")?;
    info!(
        "Successfully generated {} proofs at {}",
        proofs.len(),
        file.path().display()
    );
    Ok(())
}

struct LlmArtifacts {
    driver: Driver<Element>,
    tokenizer: HFTokenizer,
    user_tokens: Vec<Token>,
    model_name: String,
    max_context: usize,
}

async fn run_one_shot_llm(
    model: LlmModel,
    prompt: Option<String>,
    max_new_tokens: usize,
    mut tenstore: GenStore,
) -> anyhow::Result<()> {
    let request_span = tracing::info_span!("dp_worker_prove_inference", proof_id = %"one-shot-llm");
    let _entered = request_span.enter();
    let prompt = prompt.expect("clap enforces --prompt for LLM formats");

    let prompt_tokens = prompt.clone();
    let LlmArtifacts {
        driver,
        tokenizer,
        user_tokens,
        model_name,
        max_context,
    } = load_llm_artifacts(model, prompt_tokens, max_new_tokens)?;

    let (prover_ctx, verifier_ctx) = driver
        .context::<F, Pcs>()
        .context("generating contexts for LLM")?
        .with_max_context(max_context);

    let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
    let trace = driver.run_elements(input_tensor, &mut tenstore)?;
    let final_tokens = driver.continuation_tokens(&trace, user_tokens.len())?;
    let (proof, io) = driver.prove(&prover_ctx, trace)?;
    verifier_ctx.verify(proof.clone(), user_tokens.clone(), io.clone())?;
    let llm_proof = LLMProof { proof, io };
    let llm_response = tokenizer.detokenize(&final_tokens);

    info!("LLM response: {}", llm_response);

    let output = LlmOneShotOutput {
        model_name,
        prompt,
        tokens: user_tokens,
        llm_response: Some(llm_response),
        proof: llm_proof,
        verifier: verifier_ctx,
    };

    let mut file = Builder::new()
        .prefix("llm-proof-")
        .suffix(".bin")
        .rand_bytes(10)
        .disable_cleanup(true)
        .tempfile_in(std::env::current_dir().unwrap_or("./".into()))?;
    bincode::serde::encode_into_std_write(&output, file.as_file_mut(), bincode::config::standard())
        .context("encoding and writing proof")?;

    info!(
        "Successfully generated LLM proof at {}",
        file.path().display()
    );
    Ok(())
}

#[derive(Clone)]
enum LlmModel {
    Gguf {
        model_path: PathBuf,
    },
    Safetensors {
        model_path: PathBuf,
        tokenizer_path: Option<PathBuf>,
        config_path: Option<PathBuf>,
    },
}

fn load_llm_artifacts(
    model: LlmModel,
    prompt: String,
    max_new_tokens: usize,
) -> anyhow::Result<LlmArtifacts> {
    let (names, tokens, tokenizer, driver) = match model {
        LlmModel::Gguf { model_path } => {
            let raw = RawGGUF::new(model_path.clone());
            let model_names = raw.model_metadata()?;
            let (driver, tokenizer, user_tokens) =
                build_driver_from_names(model_names.clone(), raw, prompt, max_new_tokens)?;
            (model_names, user_tokens, tokenizer, driver)
        }
        LlmModel::Safetensors {
            model_path,
            tokenizer_path,
            config_path,
        } => {
            let raw = RawSafeTensors::new(
                model_path.clone(),
                tokenizer_path.expect("tokenizer"),
                config_path.expect("config"),
            );
            let model_names = raw.model_metadata()?;
            let (driver, tokenizer, user_tokens) = if model_names
                .iter()
                .any(|name| name.to_lowercase().contains("llama"))
            {
                let tokenizer = Llama3::new().load_tokenizer(&raw)?;
                let tokens = tokenizer.tokenize(&prompt);
                let context = generation_context_length(tokens.len(), max_new_tokens)?;
                let driver = Driver::load_from_model(Llama3::new(), &raw, Some(context))?;
                let (driver, _) =
                    zkml::quantization::llm_quant::llama3::quantize(driver, &tokenizer, None)?;
                (driver, tokenizer, tokens)
            } else {
                build_driver_from_names(model_names.clone(), raw, prompt, max_new_tokens)?
            };
            (model_names, user_tokens, tokenizer, driver)
        }
    };

    let model_name = names
        .first()
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let max_context = generation_context_length(tokens.len(), max_new_tokens)?;

    Ok(LlmArtifacts {
        driver,
        tokenizer,
        user_tokens: tokens,
        model_name,
        max_context,
    })
}

fn build_driver_from_names<DataFormat: Display>(
    model_names: Vec<String>,
    raw: DataFormat,
    prompt: String,
    max_new_tokens: usize,
) -> anyhow::Result<(Driver<Element>, HFTokenizer, Vec<Token>)>
where
    GPT2: TokenizerLoader<DataFormat>,
    Gemma3: TokenizerLoader<DataFormat>,
    GPT2: ModelLoader<DataFormat, Metadata = LLMMetadata>,
    Gemma3: ModelLoader<DataFormat, Metadata = LLMMetadata>,
{
    let loader = detect_llm(&model_names)?;
    match loader {
        DetectedModel::Gpt2 => {
            let tokenizer = GPT2::new().load_tokenizer(&raw)?;
            let user_tokens = tokenizer.tokenize(&prompt);
            let max_context = generation_context_length(user_tokens.len(), max_new_tokens)?;
            let driver = Driver::load_from_model(GPT2::new(), &raw, Some(max_context))?
                .into_provable_llm(None)?
                .0;
            Ok((driver, tokenizer, user_tokens))
        }
        DetectedModel::Gemma3 => {
            let tokenizer = Gemma3::new().load_tokenizer(&raw)?;
            let user_tokens = tokenizer.tokenize(&prompt);
            let max_context = generation_context_length(user_tokens.len(), max_new_tokens)?;
            let driver = Driver::load_from_model(Gemma3::new(), &raw, Some(max_context))?
                .into_provable_llm(None)?
                .0;
            Ok((driver, tokenizer, user_tokens))
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum DetectedModel {
    Gpt2,
    Gemma3,
}

fn detect_llm(model_names: &[String]) -> anyhow::Result<DetectedModel> {
    if is_gpt2_model(model_names) {
        return Ok(DetectedModel::Gpt2);
    }

    if model_names
        .iter()
        .any(|name| name.to_lowercase().contains("gemma"))
    {
        return Ok(DetectedModel::Gemma3);
    }

    anyhow::bail!("Unsupported LLM model detected: {model_names:?} (supported: GPT2, Gemma3)")
}
