//! Calibration for the unmodified Qwen3.5 hybrid decoder graph.

use crate::{
    Element,
    model::llm::Driver,
    parser::{
        PipelineConfig,
        llm::{HFTokenizer, Token},
    },
    quantization::{LLMInferenceObserver, ModelMetadata},
};
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

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
    let (mut driver, mut metadata) =
        driver.into_provable_llm_with_strategy(PipelineConfig::new().with_strategy(observer))?;
    insert_activation_output_clamps(&mut driver, &mut metadata)?;
    Ok((driver, metadata))
}

/// A table's full input interval can produce outputs beyond the observed output
/// interval (e.g. SiLU calibrated on mostly negative convolution activations).
/// Bound the complete table and prove an explicit clamp to the downstream scale.
/// This Qwen-specific pass leaves the legacy transformer quantizer unchanged.
fn insert_activation_output_clamps(
    driver: &mut Driver<Element>,
    metadata: &mut ModelMetadata,
) -> Result<()> {
    use crate::{
        ScalingFactor,
        layers::{
            Layer,
            activation::{Activation, ActivationLayer},
            requant::{FIXED_POINT_SCALE, Requant},
        },
    };
    let mut nodes = Vec::new();
    for (id, layer) in driver.model.graph().inner_nodes() {
        let Layer::Activation(Activation::Plain(activation)) = layer else {
            continue;
        };
        let data = match activation {
            ActivationLayer::Relu(Some(data), _)
            | ActivationLayer::Gelu(Some(data), _)
            | ActivationLayer::Silu(Some(data), _)
            | ActivationLayer::Sigmoid(Some(data), _)
            | ActivationLayer::Softplus(Some(data), _)
            | ActivationLayer::ExpNegative(Some(data), _) => data,
            _ => bail!("Qwen activation {id} has not been quantized"),
        };
        anyhow::ensure!(
            data.table.table_bit_size() <= 20,
            "Qwen activation table exceeds supported exact range audit size"
        );
        let mut bound = 0u64;
        for input in data.table.min_input_value()..=data.table.max_input_value() {
            bound = bound.max(data.table.lookup(input)?.unsigned_abs());
        }
        let output_scale = *metadata.output_layers_scaling[&id]
            .get(&0usize.into())
            .context("Missing activation output scale")?;
        let bits = (u64::BITS - bound.leading_zeros()) as usize + 1;
        let bits = bits.max(output_scale.bit_size() + 1);
        anyhow::ensure!(
            bits + FIXED_POINT_SCALE <= 63,
            "Qwen activation {id} output exceeds fixed-point range"
        );
        let raw_scale = ScalingFactor::from_scale(
            output_scale.scale(),
            Some((-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)),
        );
        nodes.push((id, raw_scale, output_scale, bits));
    }
    for (id, raw_scale, output_scale, bits) in nodes {
        let requants = driver
            .model
            .add_requant_layer(vec![Requant::from_multiplier(1.0, bits, output_scale)], id)?;
        anyhow::ensure!(
            requants.len() == 1,
            "Plain Qwen activation must have one output"
        );
        let requant_id = requants[0];
        metadata
            .output_layers_scaling
            .insert(id, [(0usize.into(), raw_scale)].into());
        metadata
            .input_layers_scaling
            .insert(requant_id, [(0usize.into(), raw_scale)].into());
        metadata
            .output_layers_scaling
            .insert(requant_id, [(0usize.into(), output_scale)].into());
    }
    Ok(())
}

pub fn calibration_snapshot(metadata: &ModelMetadata) -> Result<serde_json::Value> {
    let inputs: BTreeMap<_, _> = metadata.input_layers_scaling.iter().collect();
    let outputs: BTreeMap<_, _> = metadata.output_layers_scaling.iter().collect();
    Ok(serde_json::json!({"schema":1,"architecture":"qwen35",
        "quantization_bits":*crate::quantization::BIT_LEN,
        "normalization":"epsilon-preserving-rms-and-l2-v1",
        "recurrence":"gated-delta-fixed-point-v1",
        "input_scales":inputs,"output_scales":outputs}))
}

/// Freeze the compiled integer graph, including internal recurrence scales,
/// reciprocal precision, lookup descriptions, rounding and requantization.
/// Weights are bound separately by source fingerprints and the trusted verifier;
/// transient state and cache contents deliberately do not affect this digest.
pub fn compiled_quantization_snapshot(
    driver: &Driver<Element>,
    metadata: &ModelMetadata,
) -> Result<Value> {
    use crate::graph::Node;
    let mut nodes = BTreeMap::new();
    let mut feeds = Vec::new();
    for (id, node) in driver.model.graph().nodes() {
        let parameters = match node {
            Node::Input(index) => json!({"kind":"input", "index":index}),
            Node::Output(index) => json!({"kind":"output", "index":index}),
            Node::Inner(layer) => json!({"kind":layer.as_kind_str(),
                "parameters":layer_quantization_summary(layer)
                    .with_context(|| format!("Freezing Qwen quantization node {id}"))?}),
        };
        nodes.insert(*id, parameters);
        feeds.extend(
            driver
                .model
                .graph()
                .incoming_feeds(*id)
                .into_iter()
                .map(|feed| {
                    (
                        feed.source().node_id(),
                        feed.source().port(),
                        feed.target().node_id(),
                        feed.target().port(),
                    )
                }),
        );
    }
    feeds.sort();
    let compiled = json!({
        "formula_version":"qwen35-integer-graph-v1",
        "source_sha256":arithmetic_source_fingerprints(),
        "model_metadata":driver.md,
        "input_shapes":driver.model.input_shapes(),
        "nodes":nodes,
        "feeds":feeds,
        "calibration":calibration_snapshot(metadata)?,
    });
    let digest = format!("{:x}", Sha256::digest(serde_json::to_vec(&compiled)?));
    Ok(json!({"schema":2,"architecture":"qwen35",
        "compiled_parameters_sha256":digest,"compiled":compiled}))
}

fn layer_quantization_summary(layer: &crate::layers::Layer<Element>) -> Result<Value> {
    use crate::layers::Layer;
    Ok(match layer {
        Layer::Requant(op) => serde_json::to_value(op)?,
        Layer::Activation(op) => serde_json::to_value(op)?,
        Layer::Add(op) => serde_json::to_value(op)?,
        Layer::Reshape(op) => serde_json::to_value(op)?,
        Layer::AttentionMask(op) => serde_json::to_value(op)?,
        Layer::Logits(op) => serde_json::to_value(op)?,
        Layer::Softmax(op) => json!({"scalar":op.scalar,"max_size":op.max_size,
            "quant":op.quant_info().context("Softmax has not been quantized")?}),
        Layer::QwenNorm(op) => {
            let quant = op
                .quant
                .as_ref()
                .context("Qwen norm has not been quantized")?;
            json!({"epsilon_f32_bits":op.eps.to_bits(),"dimension":op.dimension,
                "kind":op.kind,"weighted":op.alpha.is_some(),
                "quant":{"dimension":quant.dimension,
                    // Decimal strings preserve the full u128 values in portable JSON.
                    "epsilon_scaled":quant.epsilon_scaled.to_string(),
                    "numerator":quant.numerator.to_string(),
                    "multiplier_fraction_bits":quant.multiplier_fraction_bits,
                    "input_abs_bound":quant.input_abs_bound,
                    "denominator_fraction_bits":crate::layers::transformer::normalisation::qwen_norm::integer::DENOMINATOR_FRACTION_BITS}})
        }
        Layer::GatedDeltaScan(op) => op.quantization_summary()?,
        Layer::CausalConv1d(op) => op.quantization_summary(),
        Layer::Positional(op) => op.quantization_summary(),
        Layer::EinSum(op) => json!({"equation":op.equation,
            "constant_unpadded_shapes":op.constant_unpadded_shapes,
            "bias_unpadded_shapes":op.bias_unpadded_shapes,
            "requantise":op.requantise}),
        Layer::Embeddings(op) => json!({"embedding_size":op.emb_size,"vocab_size":op.vocab_size}),
        other => bail!("Unsupported Qwen integer layer {}", other.as_kind_str()),
    })
}

fn arithmetic_source_fingerprints() -> BTreeMap<&'static str, String> {
    // Compile-time sources bind the arithmetic recipe as well as configured data.
    // Updates to any operator therefore cannot silently reuse an older digest.
    let sources: &[(&str, &[u8])] = &[
        (
            "qwen35-loader",
            include_bytes!("../../parser/llm/models/qwen35/mod.rs"),
        ),
        ("qwen35-snapshot", include_bytes!("qwen35.rs")),
        ("llm-scaling-strategy", include_bytes!("../strategy/llm.rs")),
        (
            "llm-calibration-tracker",
            include_bytes!("../strategy/llm_tracker.rs"),
        ),
        (
            "qwen-norm",
            include_bytes!("../../layers/transformer/normalisation/qwen_norm/mod.rs"),
        ),
        (
            "qwen-norm-integer",
            include_bytes!("../../layers/transformer/normalisation/qwen_norm/integer.rs"),
        ),
        (
            "gated-delta",
            include_bytes!("../../layers/transformer/gated_delta.rs"),
        ),
        (
            "causal-conv",
            include_bytes!("../../layers/transformer/causal_conv1d.rs"),
        ),
        ("activation", include_bytes!("../../layers/activation.rs")),
        (
            "activation-lookup",
            include_bytes!("../../layers/activation/lookup_data.rs"),
        ),
        ("lookup-tables", include_bytes!("../../lookup/table/mod.rs")),
        (
            "lookup-table-operations",
            include_bytes!("../../lookup/table/tabletype.rs"),
        ),
        ("requant", include_bytes!("../../layers/requant/mod.rs")),
        (
            "requant-evaluate",
            include_bytes!("../../layers/requant/evaluate.rs"),
        ),
        (
            "softmax",
            include_bytes!("../../layers/transformer/softmax/mod.rs"),
        ),
        (
            "softmax-evaluate",
            include_bytes!("../../layers/transformer/softmax/evaluate.rs"),
        ),
        (
            "softmax-quantise",
            include_bytes!("../../layers/transformer/softmax/quantise.rs"),
        ),
        ("add", include_bytes!("../../layers/add.rs")),
        (
            "einsum-evaluate",
            include_bytes!("../../layers/einsum/evaluate.rs"),
        ),
        (
            "rope",
            include_bytes!("../../layers/transformer/positional/rope.rs"),
        ),
    ];
    sources
        .iter()
        .map(|(name, bytes)| (*name, format!("{:x}", Sha256::digest(bytes))))
        .collect()
}

#[cfg(test)]
mod tests;
