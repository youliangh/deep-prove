//! Hermetic loader contract tests, kept separate from arithmetic source hashes.

use anyhow::Result;
use safetensors::{SafeTensors, tensor::TensorView};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

use crate::{
    model::llm::Driver,
    parser::{llm::models::qwen35::Qwen35, safe::RawSafeTensors},
};

const WEIGHTS: &[u8] = include_bytes!("llm/models/qwen35/fixtures/model.safetensors");
const CONFIG: &str = include_str!("llm/models/qwen35/fixtures/config.json");
const PREFIX: &str = "model.language_model.";

fn raw(directory: &Path, model: &str, config: &Value) -> Result<RawSafeTensors> {
    std::fs::write(directory.join("config.json"), serde_json::to_vec(config)?)?;
    Ok(RawSafeTensors::new(
        directory.join(model),
        directory.join("tokenizer.json"),
        directory.join("config.json"),
    ))
}

#[test]
fn qwen_loader_accepts_hf_prefixes_and_indexed_shards() -> Result<()> {
    let fixture = SafeTensors::deserialize(WEIGHTS)?;
    let config: Value = serde_json::from_str(CONFIG)?;
    for prefix in [PREFIX, "model.", "language_model."] {
        let directory = tempfile::tempdir()?;
        let tensors: Vec<_> = fixture
            .tensors()
            .into_iter()
            .map(|(name, view)| {
                (
                    format!("{prefix}{}", name.strip_prefix(PREFIX).unwrap()),
                    view,
                )
            })
            .collect();
        let map: BTreeMap<_, _> = tensors
            .iter()
            .map(|(name, _)| (name, "text-shard.safetensors"))
            .collect();
        std::fs::write(
            directory.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"weight_map":map}))?,
        )?;
        safetensors::tensor::serialize_to_file(
            tensors,
            None,
            &directory.path().join("text-shard.safetensors"),
        )?;
        let raw = raw(directory.path(), "model.safetensors.index.json", &config)?;
        let driver = Driver::load_from_model(Qwen35::new(), &raw, Some(16))?;
        assert_eq!(driver.md.transformers.len(), 2);
        assert!(driver.md.transformers[0].linear_attention.is_some());
        assert!(driver.md.transformers[1].transformer.is_some());
    }
    Ok(())
}

#[test]
fn qwen_loader_rejects_missing_and_misshaped_required_weights() -> Result<()> {
    let fixture = SafeTensors::deserialize(WEIGHTS)?;
    let config: Value = serde_json::from_str(CONFIG)?;
    for missing in [true, false] {
        let directory = tempfile::tempdir()?;
        let mut tensors = Vec::new();
        for (name, view) in fixture.tensors() {
            if missing && name.ends_with("layers.0.linear_attn.A_log") {
                continue;
            }
            let view = if !missing && name.ends_with("layers.1.self_attn.q_proj.weight") {
                TensorView::new(view.dtype(), vec![16, 16], view.data())?
            } else {
                view
            };
            tensors.push((name, view));
        }
        safetensors::tensor::serialize_to_file(
            tensors,
            None,
            &directory.path().join("model.safetensors"),
        )?;
        let raw = raw(directory.path(), "model.safetensors", &config)?;
        let error = Driver::load_from_model(Qwen35::new(), &raw, Some(16)).unwrap_err();
        let message = format!("{error:#}");
        assert!(
            message.contains(if missing { "A_log" } else { "Wrong shape" }),
            "{message}"
        );
    }
    Ok(())
}

#[test]
fn qwen_loader_rejects_unsupported_or_inconsistent_configuration() -> Result<()> {
    let config: Value = serde_json::from_str(CONFIG)?;
    let mutations = [
        ("model_type", json!("llama")),
        ("layer_types", json!(["linear_attention"])),
        ("layer_types", json!(["linear_attention", "unknown"])),
        ("num_attention_heads", json!(3)),
        ("hidden_act", json!("gelu")),
        ("attention_bias", json!(true)),
        ("attn_output_gate", json!(false)),
        ("rms_norm_eps", json!(0.0)),
        ("eos_token_id", json!(32)),
        ("linear_num_value_heads", json!(4)),
        ("linear_conv_kernel_dim", json!(0)),
        (
            "rope_parameters",
            json!({"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":0.3}),
        ),
        (
            "rope_parameters",
            json!({"rope_type":"dynamic","rope_theta":10000.,"partial_rotary_factor":0.5}),
        ),
    ];
    let directory = tempfile::tempdir()?;
    std::fs::write(directory.path().join("model.safetensors"), WEIGHTS)?;
    for (key, value) in mutations {
        let mut malformed = config.clone();
        malformed["text_config"][key] = value;
        let raw = raw(directory.path(), "model.safetensors", &malformed)?;
        assert!(
            Driver::load_from_model(Qwen35::new(), &raw, Some(16)).is_err(),
            "Accepted inconsistent {key}"
        );
    }
    let raw = raw(directory.path(), "model.safetensors", &config)?;
    for context in [0, usize::MAX] {
        assert!(Driver::load_from_model(Qwen35::new(), &raw, Some(context)).is_err());
    }
    Ok(())
}
