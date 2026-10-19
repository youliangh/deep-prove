//! Llama3 calibration without Gemma-specific architecture transformations.

use anyhow::Result;

use crate::{
    Element,
    model::llm::Driver,
    parser::{
        PipelineConfig,
        llm::{HFTokenizer, Token, models::llama3::Llama3},
    },
    quantization::{LLMInferenceObserver, ModelMetadata, llm_quant::FPTransformModel},
};

impl FPTransformModel for Llama3 {
    fn adapt_model(driver: Driver<f32>) -> Result<Driver<f32>> {
        // Gemma3's rotations rely on additional post-attention/post-FFN norms.
        // Llama3 must preserve its own graph and tied embedding semantics.
        Ok(driver)
    }
}

/// Freeze scales from representative natural-language token sequences.
pub fn quantize(
    driver: Driver<f32>,
    tokenizer: &HFTokenizer,
    samples: Option<Vec<Vec<Token>>>,
) -> Result<(Driver<Element>, ModelMetadata)> {
    let mut observer =
        LLMInferenceObserver::new_with_tokeniser(tokenizer.clone(), driver.md.clone(), None);
    if let Some(samples) = samples {
        observer = observer.with_representative_tokens(samples)?;
    }
    driver.into_provable_llm_with_strategy(PipelineConfig::new().with_strategy(observer))
}

/// Stable, weight-free record of the scales used by the numerical and proving runs.
/// The verifier key remains the cryptographic model identity; this record makes
/// independently repeated calibration runs directly comparable.
pub fn calibration_snapshot(metadata: &ModelMetadata) -> Result<serde_json::Value> {
    use std::collections::BTreeMap;
    let inputs: BTreeMap<_, _> = metadata.input_layers_scaling.iter().collect();
    let outputs: BTreeMap<_, _> = metadata.output_layers_scaling.iter().collect();
    Ok(serde_json::json!({
        "schema": 1,
        "quantization_bits": *crate::quantization::BIT_LEN,
        "rmsnorm": "epsilon-free-two-sided-norm-v1",
        "input_scales": inputs,
        "output_scales": outputs,
    }))
}
