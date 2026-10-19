//! Shared Hugging Face loading and rotary embeddings for the Llama family.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::{
    Shape,
    layers::transformer::{
        attention_mask::AttentionSpan,
        positional::{Positional, RopeLayout},
    },
    model::Model,
    parser::{
        Load,
        llm::{
            LLMConfig, LLMIR, LLMMetadata,
            config::{AttentionConfig, AttentionHeadType, LLMStructure},
            transformer::NormType,
        },
        safe::{ConfigJSON, FileTensorLoader, RawSafeTensors},
    },
};

use super::llama2::decoder::Llama2Decoder;

#[derive(Deserialize)]
struct LlamaConfig {
    model_type: String,
    architectures: Option<Vec<String>>,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    head_dim: Option<usize>,
    num_hidden_layers: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    vocab_size: usize,
    eos_token_id: usize,
    hidden_act: String,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    partial_rotary_factor: Option<f32>,
}

impl LlamaConfig {
    fn structure(&self, model_name: &str, max_context: Option<usize>) -> Result<LLMStructure> {
        ensure!(self.model_type == "llama", "Expected model_type=llama");
        if let Some(architectures) = &self.architectures {
            ensure!(
                architectures.iter().any(|name| name == "LlamaForCausalLM"),
                "Expected LlamaForCausalLM architecture"
            );
        }
        ensure!(
            self.hidden_act == "silu",
            "Only Llama SiLU/SwiGLU is supported"
        );
        ensure!(
            !self.attention_bias && !self.mlp_bias,
            "Llama attention/MLP biases are not supported"
        );
        ensure!(
            self.partial_rotary_factor.unwrap_or(1.0) == 1.0,
            "Partial rotary embeddings are not supported"
        );
        ensure!(
            self.hidden_size > 0 && self.num_attention_heads > 0,
            "hidden_size and num_attention_heads must be positive"
        );
        let head_size = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);
        ensure!(
            head_size.is_power_of_two()
                && head_size >= 2
                && self.num_attention_heads.checked_mul(head_size) == Some(self.hidden_size),
            "head_dim must be a power of two >= 2 and hidden_size must equal heads * head_dim"
        );
        let kv_heads = self.num_key_value_heads.unwrap_or(self.num_attention_heads);
        ensure!(
            kv_heads > 0 && self.num_attention_heads % kv_heads == 0,
            "num_key_value_heads must be positive and divide num_attention_heads"
        );
        ensure!(
            self.intermediate_size > 0 && self.num_hidden_layers > 0 && self.vocab_size > 0,
            "Llama layer, intermediate and vocabulary sizes must be positive"
        );
        ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "rms_norm_eps must be positive and finite"
        );
        ensure!(
            self.eos_token_id < self.vocab_size,
            "EOS token is outside the vocabulary"
        );
        let context_length = max_context.unwrap_or(self.max_position_embeddings);
        ensure!(
            context_length > 0 && context_length <= self.max_position_embeddings,
            "Requested context must be in 1..=max_position_embeddings"
        );
        Ok(LLMStructure {
            generic: LLMConfig {
                model_name: model_name.to_owned(),
                embedding_size: self.hidden_size,
                hidden_size: self.hidden_size,
                intermediate_size: self.intermediate_size,
                num_heads: self.num_attention_heads,
                head_size,
                num_block: self.num_hidden_layers,
                context_length,
                norm_epsilon: self.rms_norm_eps,
                vocab_size: self.vocab_size,
                eos_token: self.eos_token_id.into(),
            },
            norm_type: NormType::RMSNorm,
            global_positional: None,
            attention_config: AttentionConfig {
                span: vec![AttentionSpan::Full; self.num_hidden_layers],
                head: AttentionHeadType::GQA(kv_heads),
            },
        })
    }
}

pub(super) fn parse(
    raw: &RawSafeTensors,
    max_context: Option<usize>,
    model_name: &str,
) -> Result<(Model<f32>, LLMMetadata)> {
    let mut config_value: serde_json::Value = raw.read_config_json()?.into();
    let config: LlamaConfig = serde_json::from_value(config_value.clone())
        .context("Invalid Hugging Face Llama configuration")?;
    let structure = config.structure(model_name, max_context)?;
    // HF Llama defaults to an independent output projection. Preserve the older
    // Llama2 adapter's fallback for checkpoints that omit this optional setting.
    let tied = config_value
        .get("tie_word_embeddings")
        .map(|value| {
            value
                .as_bool()
                .context("tie_word_embeddings must be a boolean")
        })
        .transpose()?
        .unwrap_or(model_name == "llama2");
    config_value["tie_word_embeddings"] = tied.into();
    let config_json = ConfigJSON::from(config_value);
    // Reject unsupported rotary variants before reading the large weight file.
    RopeParameters::from_config(&config_json)?;
    let loader = FileTensorLoader::from_path(raw.model_path())?;
    validate_weights(&loader, &structure, tied)?;
    let generic = structure.generic.clone();
    let ir = LLMIR::<Llama2Decoder>::from_loader(&loader, &(structure, config_json))?;
    ir.into_model(generic, Shape::from(vec![1]))
}

fn validate_weights(loader: &FileTensorLoader, structure: &LLMStructure, tied: bool) -> Result<()> {
    let c = &structure.generic;
    let AttentionHeadType::GQA(kv_heads) = structure.attention_config.head else {
        unreachable!("Llama config always represents attention as GQA");
    };
    let check = |name: &str, expected: &[usize]| -> Result<()> {
        let actual = loader.get_tensor_shape(name)?;
        ensure!(
            actual == Shape::from(expected.to_vec()),
            "Tensor {name} has shape {actual:?}, expected {expected:?}"
        );
        Ok(())
    };
    check("model.embed_tokens.weight", &[c.vocab_size, c.hidden_size])?;
    check("model.norm.weight", &[c.hidden_size])?;
    if !tied {
        check("lm_head.weight", &[c.vocab_size, c.hidden_size])?;
    }
    for layer in 0..c.num_block {
        let prefix = format!("model.layers.{layer}");
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            check(&format!("{prefix}.{norm}.weight"), &[c.hidden_size])?;
        }
        for projection in ["q_proj", "o_proj"] {
            check(
                &format!("{prefix}.self_attn.{projection}.weight"),
                &[c.hidden_size, c.hidden_size],
            )?;
        }
        for projection in ["k_proj", "v_proj"] {
            check(
                &format!("{prefix}.self_attn.{projection}.weight"),
                &[kv_heads * c.head_size, c.hidden_size],
            )?;
        }
        for projection in ["gate_proj", "up_proj"] {
            check(
                &format!("{prefix}.mlp.{projection}.weight"),
                &[c.intermediate_size, c.hidden_size],
            )?;
        }
        check(
            &format!("{prefix}.mlp.down_proj.weight"),
            &[c.hidden_size, c.intermediate_size],
        )?;
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
    rope_scaling: Option<RopeScaling>,
}

#[derive(Debug, Deserialize)]
struct RopeScaling {
    #[serde(alias = "type")]
    rope_type: String,
    factor: Option<f32>,
    low_freq_factor: Option<f32>,
    high_freq_factor: Option<f32>,
    original_max_position_embeddings: Option<usize>,
}

impl RopeParameters {
    fn from_config(config: &ConfigJSON) -> Result<Self> {
        let value: serde_json::Value = config.clone().into();
        let parameters: Self =
            serde_json::from_value(value).context("Invalid Llama RoPE configuration")?;
        ensure!(
            parameters.rope_theta.is_finite() && parameters.rope_theta > 0.0,
            "rope_theta must be positive and finite"
        );
        if let Some(scaling) = &parameters.rope_scaling {
            ensure!(
                matches!(scaling.rope_type.as_str(), "default" | "llama3"),
                "Unsupported Llama RoPE type: {}",
                scaling.rope_type
            );
            if scaling.rope_type == "llama3" {
                let factor = scaling.factor.context("llama3 RoPE requires factor")?;
                let low = scaling
                    .low_freq_factor
                    .context("llama3 RoPE requires low_freq_factor")?;
                let high = scaling
                    .high_freq_factor
                    .context("llama3 RoPE requires high_freq_factor")?;
                let original = scaling
                    .original_max_position_embeddings
                    .context("llama3 RoPE requires original_max_position_embeddings")?;
                ensure!(
                    factor.is_finite() && factor >= 1.0,
                    "RoPE factor must be finite and >= 1"
                );
                ensure!(
                    low.is_finite() && high.is_finite() && low > 0.0 && high > low,
                    "RoPE frequency factors must satisfy 0 < low < high"
                );
                ensure!(original > 0, "Original RoPE context must be positive");
            }
        }
        Ok(parameters)
    }

    fn inverse_frequencies(&self, head_size: usize) -> Vec<f32> {
        (0..head_size / 2)
            .map(|i| {
                let frequency = self
                    .rope_theta
                    .powf(2.0 * i as f32 / head_size as f32)
                    .recip();
                match &self.rope_scaling {
                    Some(scaling) if scaling.rope_type == "llama3" => {
                        scaling.scale_frequency(frequency)
                    }
                    _ => frequency,
                }
            })
            .collect()
    }

    fn storage_id(&self, head_size: usize, context: usize) -> String {
        let mut id = format!(
            "llama.rope.theta_{:08x}.head_{head_size}.context_{context}",
            self.rope_theta.to_bits()
        );
        if let Some(scaling) = &self.rope_scaling
            && scaling.rope_type == "llama3"
        {
            id.push_str(&format!(
                ".llama3.factor_{:08x}.low_{:08x}.high_{:08x}.original_{}",
                scaling.factor.unwrap().to_bits(),
                scaling.low_freq_factor.unwrap().to_bits(),
                scaling.high_freq_factor.unwrap().to_bits(),
                scaling.original_max_position_embeddings.unwrap()
            ));
        } else {
            id.push_str(".default");
        }
        id
    }
}

impl RopeScaling {
    fn scale_frequency(&self, frequency: f32) -> f32 {
        let factor = self.factor.unwrap();
        let low = self.low_freq_factor.unwrap();
        let high = self.high_freq_factor.unwrap();
        let original = self.original_max_position_embeddings.unwrap() as f32;
        let wavelength = std::f32::consts::TAU / frequency;
        if wavelength < original / high {
            frequency
        } else if wavelength > original / low {
            frequency / factor
        } else {
            let smooth = (original / wavelength - low) / (high - low);
            (1.0 - smooth) * frequency / factor + smooth * frequency
        }
    }
}

pub(super) fn load_rope(
    config: &ConfigJSON,
    head_size: usize,
    context: usize,
) -> Result<Positional<f32>> {
    let parameters = RopeParameters::from_config(config)?;
    Positional::new_rope(
        parameters.inverse_frequencies(head_size),
        parameters.storage_id(head_size, context).into(),
        context,
        RopeLayout::RotateHalf,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn scaled_config() -> ConfigJSON {
        json!({"rope_theta": 500000.0, "rope_scaling": {
            "rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192
        }})
        .into()
    }

    fn model_config() -> serde_json::Value {
        json!({"model_type":"llama", "architectures":["LlamaForCausalLM"],
            "hidden_size":2048, "intermediate_size":8192, "num_attention_heads":32,
            "num_key_value_heads":8, "head_dim":64, "num_hidden_layers":16,
            "max_position_embeddings":131072, "rms_norm_eps":0.00001,
            "vocab_size":128256, "eos_token_id":128001, "hidden_act":"silu"})
    }

    #[test]
    fn llama3_configuration_validates_gqa_and_context() -> Result<()> {
        let config: LlamaConfig = serde_json::from_value(model_config())?;
        let structure = config.structure("llama3", Some(256))?;
        assert_eq!(structure.generic.context_length, 256);
        assert_eq!(structure.attention_config.head, AttentionHeadType::GQA(8));
        assert!(config.structure("llama3", Some(0)).is_err());
        assert!(config.structure("llama3", Some(131073)).is_err());
        for (key, value) in [
            ("num_attention_heads", json!(0)),
            ("num_key_value_heads", json!(3)),
            ("head_dim", json!(32)),
            ("attention_bias", json!(true)),
            ("hidden_act", json!("gelu")),
            ("eos_token_id", json!(128256)),
        ] {
            let mut value_config = model_config();
            value_config[key] = value;
            let config: LlamaConfig = serde_json::from_value(value_config)?;
            assert!(
                config.structure("llama3", Some(256)).is_err(),
                "accepted {key}"
            );
        }
        Ok(())
    }

    #[test]
    fn llama3_rope_all_frequency_bands_and_boundaries() -> Result<()> {
        let parameters = RopeParameters::from_config(&scaled_config())?;
        let scaling = parameters.rope_scaling.unwrap();
        // Independent wavelengths exercise the high, transition and low bands.
        for (wavelength, ratio) in [
            (1024.0, 1.0),
            (2048.0, 1.0),
            (4096.0, 17.0 / 48.0),
            (8192.0, 1.0 / 32.0),
            (16384.0, 1.0 / 32.0),
        ] {
            let frequency = std::f32::consts::TAU / wavelength;
            assert!((scaling.scale_frequency(frequency) / frequency - ratio).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn llama_rope_unscaled_and_commitment_identity() -> Result<()> {
        let unscaled = RopeParameters::from_config(&json!({"rope_theta": 10000.0}).into())?;
        let expected = [1.0, 0.1, 0.01, 0.001];
        for (actual, expected) in unscaled.inverse_frequencies(8).iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-7);
        }
        let scaled = RopeParameters::from_config(&scaled_config())?;
        let mut changed: serde_json::Value = scaled_config().into();
        changed["rope_scaling"]["factor"] = json!(8.0);
        let changed = RopeParameters::from_config(&changed.into())?;
        assert_ne!(scaled.storage_id(64, 256), changed.storage_id(64, 256));
        assert_ne!(scaled.storage_id(64, 256), scaled.storage_id(64, 128));
        assert_ne!(scaled.storage_id(64, 256), scaled.storage_id(128, 256));
        Ok(())
    }

    #[test]
    fn llama_rope_rejects_unsupported_or_invalid_scaling() {
        for replacement in [
            json!({"rope_type":"linear","factor":32.0}),
            json!({"rope_type":"llama3","factor":32.0}),
            json!({"rope_type":"llama3","factor":32.0,"low_freq_factor":4.0,
                "high_freq_factor":1.0,"original_max_position_embeddings":8192}),
        ] {
            let mut config: serde_json::Value = scaled_config().into();
            config["rope_scaling"] = replacement;
            assert!(RopeParameters::from_config(&config.into()).is_err());
        }
    }
}
