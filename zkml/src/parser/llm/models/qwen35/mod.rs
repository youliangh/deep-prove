//! Native Qwen3.5 text decoder, including its gated delta and full-attention blocks.
//!
//! Vision and auxiliary multi-token-prediction weights in a conditional-generation
//! checkpoint are not part of ordinary text decoding and are not loaded into the graph.

use std::marker::PhantomData;

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

use crate::{
    Shape, Tensor,
    graph::NodeId,
    layers::{
        Layer,
        activation::{Activation, ActivationLayer},
        add::Add,
        einsum::EinSum,
        reshape::Reshape,
        transformer::{
            attention_mask::AttentionMask,
            causal_conv1d::CausalConv1d,
            embeddings::Embeddings,
            gated_delta::GatedDeltaScan,
            logits::Logits,
            normalisation::qwen_norm::{NormKind, QwenNorm},
            positional::Positional,
            softmax::Softmax,
        },
    },
    model::{Model, llm::Driver},
    parser::{
        LayerInsertion, ModelLoader,
        llm::{
            HFTokenizer, LLMConfig, LLMMetadata,
            metadata::{AttentionMetadata, LinearAttentionMetadata, TransformerMetadata},
            tokenizer::TokenizerLoader,
            transformer::feed_forward::FeedForwardNetwork,
        },
        safe::{FileTensorLoader, RawSafeTensors},
    },
    quantization::llm_quant::FPTransformModel,
    tensor::{KeyedTensor, TensorHandle},
};

pub const QWEN35_NAME: &str = "qwen35";

/// The checkpoint's single-user, text-only chat template with thinking disabled.
pub fn chat_prompt(text: &str) -> String {
    format!(
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        text.trim()
    )
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Qwen35 {
    max_context: Option<usize>,
}

impl Qwen35 {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_max_ctx_length(mut self, length: usize) -> Self {
        self.max_context = Some(length);
        self
    }
}

impl TokenizerLoader<RawSafeTensors> for Qwen35 {
    fn load_tokenizer(&self, raw: &RawSafeTensors) -> Result<HFTokenizer> {
        HFTokenizer::from_tokenizer_json_path(raw.tokenizer_path())
            .map(HFTokenizer::with_exact_decoding)
    }
}

impl super::LLMModelLoader<RawSafeTensors> for Qwen35 {
    fn with_max_context_length(self, length: usize) -> Self {
        self.with_max_ctx_length(length)
    }
}

impl FPTransformModel for Qwen35 {
    fn adapt_model(driver: Driver<f32>) -> Result<Driver<f32>> {
        Ok(driver)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Config {
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    eos_token_id: usize,
    hidden_act: String,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default = "default_true")]
    attn_output_gate: bool,
    #[serde(default)]
    tie_word_embeddings: bool,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_value_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    rope_parameters: RopeParameters,
}

#[derive(Debug, Clone, Deserialize)]
struct RopeParameters {
    rope_type: String,
    rope_theta: f32,
    partial_rotary_factor: f32,
}

fn default_true() -> bool {
    true
}

impl Config {
    fn validate(&self, context: usize) -> Result<()> {
        ensure!(
            self.model_type == "qwen3_5_text",
            "Expected Qwen3.5 text configuration"
        );
        ensure!(
            self.hidden_size > 0 && self.intermediate_size > 0 && self.vocab_size > 0,
            "Qwen dimensions must be positive"
        );
        ensure!(
            self.num_hidden_layers > 0 && self.layer_types.len() == self.num_hidden_layers,
            "layer_types must describe every decoder block"
        );
        ensure!(
            self.layer_types
                .iter()
                .all(|kind| matches!(kind.as_str(), "linear_attention" | "full_attention")),
            "Unsupported Qwen3.5 block type"
        );
        ensure!(
            self.num_key_value_heads > 0
                && self.num_attention_heads > 0
                && self.num_attention_heads % self.num_key_value_heads == 0,
            "KV heads must divide query heads"
        );
        ensure!(
            self.head_dim.is_power_of_two() && self.head_dim >= 2,
            "Qwen head_dim must be a power of two"
        );
        ensure!(
            self.hidden_act == "silu" && !self.attention_bias && self.attn_output_gate,
            "This Qwen3.5 adapter requires SiLU, bias-free attention and output gates"
        );
        ensure!(
            self.linear_num_key_heads > 0
                && self.linear_num_value_heads == self.linear_num_key_heads,
            "This Qwen3.5 adapter requires equal linear key/value head counts"
        );
        ensure!(
            self.linear_key_head_dim.is_power_of_two()
                && self.linear_value_head_dim.is_power_of_two()
                && self.linear_conv_kernel_dim > 0,
            "Invalid linear attention dimensions"
        );
        ensure!(
            self.rms_norm_eps.is_finite() && self.rms_norm_eps > 0.0,
            "Normalization epsilon must be positive and finite"
        );
        ensure!(
            self.eos_token_id < self.vocab_size,
            "EOS token exceeds vocabulary"
        );
        ensure!(
            context > 0 && context <= self.max_position_embeddings,
            "Requested context exceeds configured model context"
        );
        ensure!(
            self.rope_parameters.rope_type == "default"
                && self.rope_parameters.rope_theta.is_finite()
                && self.rope_parameters.rope_theta > 0.0,
            "Only default Qwen3.5 RoPE is supported"
        );
        let rotary = self.head_dim as f32 * self.rope_parameters.partial_rotary_factor;
        ensure!(
            rotary.is_finite()
                && rotary >= 2.0
                && rotary <= self.head_dim as f32
                && rotary.fract() == 0.0
                && (rotary as usize).is_power_of_two(),
            "Partial RoPE dimension must be a power of two within head_dim"
        );
        Ok(())
    }

    fn metadata(&self, context: usize) -> LLMConfig {
        LLMConfig {
            model_name: QWEN35_NAME.to_owned(),
            embedding_size: self.hidden_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_attention_heads,
            head_size: self.head_dim,
            num_block: self.num_hidden_layers,
            context_length: context,
            norm_epsilon: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            eos_token: self.eos_token_id.into(),
        }
    }
}

impl ModelLoader<RawSafeTensors> for Qwen35 {
    type Metadata = LLMMetadata;
    fn model_name(&self) -> String {
        QWEN35_NAME.to_owned()
    }

    fn parse(&self, raw: &RawSafeTensors) -> Result<(Model<f32>, LLMMetadata)> {
        let root: serde_json::Value = raw.read_config_json()?.into();
        let mut text = root.get("text_config").unwrap_or(&root).clone();
        if text.get("tie_word_embeddings").is_none() {
            text["tie_word_embeddings"] = root
                .get("tie_word_embeddings")
                .cloned()
                .unwrap_or(false.into());
        }
        let config: Config =
            serde_json::from_value(text).context("Invalid Qwen3.5 configuration")?;
        let context = self.max_context.unwrap_or(config.max_position_embeddings);
        config.validate(context)?;
        let loader = raw.loader()?;
        let prefix = ["model.language_model.", "model.", "language_model."]
            .into_iter()
            .find(|prefix| {
                loader
                    .get_tensor_shape(&format!("{prefix}embed_tokens.weight"))
                    .is_ok()
            })
            .context("Qwen text embedding tensor was not found")?;
        let language = loader.pp(prefix);
        let embedding = weight(
            &language,
            "embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
        )?;
        let mut output_weight = if config.tie_word_embeddings {
            embedding.clone()
        } else {
            weight(
                &loader,
                "lm_head.weight",
                &[config.vocab_size, config.hidden_size],
            )?
        };
        output_weight.key = format!("{}_lm_head", output_weight.key).into();
        let mut model = Model::new_from_input_shapes(vec![Shape::from(vec![1])]);
        let embeddings = model
            .add_consecutive_layer(Layer::Embeddings(Embeddings::new(embedding.into())?), None)?;
        let mut previous = embeddings;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for (index, kind) in config.layer_types.iter().enumerate() {
            let block = language.pp(&format!("layers.{index}."));
            let norm_id = add_norm(
                &mut model,
                previous,
                &block,
                "input_layernorm.weight",
                config.hidden_size,
                config.rms_norm_eps,
                true,
            )?;
            let (mixed, transformer, linear_attention) = if kind == "full_attention" {
                let (output, metadata) =
                    full_attention(&mut model, norm_id, &block, &config, context, index)?;
                (output, Some(metadata), None)
            } else {
                let (output, metadata) =
                    linear_attention_block(&mut model, norm_id, &block, &config)?;
                (output, None, Some(metadata))
            };
            let residual = add_residual(&mut model, previous, mixed)?;
            let post_norm = add_norm(
                &mut model,
                residual,
                &block,
                "post_attention_layernorm.weight",
                config.hidden_size,
                config.rms_norm_eps,
                true,
            )?;
            let ffn = FeedForwardNetwork {
                gate: Some(
                    weight(
                        &block,
                        "mlp.gate_proj.weight",
                        &[config.intermediate_size, config.hidden_size],
                    )?
                    .try_map_tensor(|tensor| tensor.transpose())?,
                ),
                up: weight(
                    &block,
                    "mlp.up_proj.weight",
                    &[config.intermediate_size, config.hidden_size],
                )?
                .try_map_tensor(|tensor| tensor.transpose())?,
                down: weight(
                    &block,
                    "mlp.down_proj.weight",
                    &[config.hidden_size, config.intermediate_size],
                )?
                .try_map_tensor(|tensor| tensor.transpose())?,
                up_bias: None,
                down_bias: None,
                activation: ActivationLayer::Silu(None, PhantomData),
            };
            let (ffn_output, ffn_metadata) = ffn.add_to_model(&mut model, Some(post_norm))?;
            previous = add_residual(&mut model, residual, ffn_output)?;
            blocks.push(TransformerMetadata {
                norm_id,
                transformer,
                linear_attention,
                ffn: ffn_metadata,
            });
        }
        previous = add_norm(
            &mut model,
            previous,
            &language,
            "norm.weight",
            config.hidden_size,
            config.rms_norm_eps,
            true,
        )?;
        let final_proj = model.add_consecutive_layer(
            Layer::EinSum(
                EinSum::new(
                    "X(se)@W(ve)->O(sv)".into(),
                    vec![Some(output_weight.into())],
                    vec![None],
                )?
                .disable_requantisation(),
            ),
            Some(previous),
        )?;
        let argmax =
            model.add_consecutive_layer(Layer::Logits(Logits::new_argmax()), Some(final_proj))?;
        model.automatic_output_labelling()?;
        Ok((
            model,
            LLMMetadata {
                config: config.metadata(context),
                additional_eos_tokens: additional_eos_tokens(raw, &config)?,
                transformers: blocks,
                embeddings,
                positional: None,
                final_proj,
                logits: final_proj,
                argmax,
            },
        ))
    }
}

fn additional_eos_tokens(
    raw: &RawSafeTensors,
    config: &Config,
) -> Result<Vec<crate::parser::llm::Token>> {
    // Text-only synthetic fixtures may omit a tokenizer; production loading also
    // invokes TokenizerLoader and requires the actual tokenizer.json.
    if !raw.tokenizer_path().is_file() {
        return Ok(Vec::new());
    }
    let tokenizer: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(raw.tokenizer_path())?)?;
    let mut tokens = Vec::new();
    for token in tokenizer["added_tokens"]
        .as_array()
        .context("Qwen tokenizer has no added tokens")?
    {
        if matches!(
            token["content"].as_str(),
            Some("<|im_end|>" | "<|endoftext|>")
        ) {
            let id = usize::try_from(token["id"].as_u64().context("Invalid Qwen stop token ID")?)?;
            ensure!(id < config.vocab_size, "Qwen stop token exceeds vocabulary");
            if id != config.eos_token_id && !tokens.contains(&id.into()) {
                tokens.push(id.into());
            }
        }
    }
    Ok(tokens)
}

fn weight(loader: &FileTensorLoader, name: &str, expected: &[usize]) -> Result<KeyedTensor<f32>> {
    ensure!(
        loader.get_tensor_shape(name)? == Shape::from(expected.to_vec()),
        "Wrong shape for {}{name}: expected {expected:?}",
        loader.current_prefix()
    );
    loader.get_tensor(name)
}

fn add_norm(
    model: &mut Model<f32>,
    previous: NodeId,
    loader: &FileTensorLoader,
    name: &str,
    dim: usize,
    eps: f32,
    add_one: bool,
) -> Result<NodeId> {
    let mut alpha = weight(loader, name, &[dim])?;
    if add_one {
        alpha = alpha.try_map_tensor(|tensor| tensor.try_map(|value| Ok(*value + 1.0)))?;
        alpha.key = format!("{}_effective", alpha.key).into();
    }
    model.add_consecutive_layer(
        Layer::QwenNorm(QwenNorm::new(Some(alpha.into()), eps, dim, NormKind::Rms)?),
        Some(previous),
    )
}

fn add_residual(model: &mut Model<f32>, left: NodeId, right: NodeId) -> Result<NodeId> {
    let output = model.graph_mut().add_inner(Layer::Add(Add::new()))?;
    model.add_edge(left, output, (0, 0))?;
    model.add_edge(right, output, (0, 1))?;
    Ok(output)
}

fn unary(
    model: &mut Model<f32>,
    previous: NodeId,
    port: usize,
    layer: Layer<f32>,
) -> Result<NodeId> {
    let output = model.graph_mut().add_inner(layer)?;
    model.add_edge(previous, output, (port, 0))?;
    Ok(output)
}

fn rows(
    source: &KeyedTensor<f32>,
    start: usize,
    count: usize,
    suffix: &str,
) -> Result<KeyedTensor<f32>> {
    let width = source.shape()[1];
    Ok(KeyedTensor::new(
        format!("{}_{suffix}", source.key),
        Tensor::new(
            vec![count, width].into(),
            source.data()[start * width..(start + count) * width].to_vec(),
        )?,
    ))
}

fn full_attention(
    model: &mut Model<f32>,
    input: NodeId,
    block: &FileTensorLoader,
    config: &Config,
    context: usize,
    index: usize,
) -> Result<(NodeId, AttentionMetadata)> {
    let c = config;
    let loader = block.pp("self_attn.");
    let h = c.num_key_value_heads;
    let g = c.num_attention_heads / h;
    let d = c.head_dim;
    let e = c.hidden_size;
    // HF's q projection interleaves Q and gate within each head, not globally.
    let q_gate = weight(
        &loader,
        "q_proj.weight",
        &[c.num_attention_heads * 2 * d, e],
    )?;
    let split = |part: usize, name: &str| -> Result<TensorHandle<f32>> {
        let mut data = Vec::with_capacity(c.num_attention_heads * d * e);
        for head in 0..c.num_attention_heads {
            let start = (head * 2 * d + part * d) * e;
            data.extend_from_slice(&q_gate.data()[start..start + d * e]);
        }
        Ok(KeyedTensor::new(
            format!("{}_{name}", q_gate.key),
            Tensor::new(vec![c.num_attention_heads * d, e].into(), data)?
                .transpose()?
                .reshaped(vec![e, h, g, d].into())?,
        )
        .into())
    };
    let k = weight(&loader, "k_proj.weight", &[h * d, e])?
        .try_map_tensor(|tensor| tensor.transpose()?.reshaped(vec![e, h, d].into()))?;
    let v = weight(&loader, "v_proj.weight", &[h * d, e])?
        .try_map_tensor(|tensor| tensor.transpose()?.reshaped(vec![e, h, d].into()))?;
    let mut projection = EinSum::new(
        "X(se)@WQ(ehgd):WK(ehd):WV(ehd)->Q(ghsd):K(hsd):V(hsd)".into(),
        vec![Some(split(0, "query")?), Some(k.into()), Some(v.into())],
        vec![None; 3],
    )?;
    projection.with_caches(vec![None, None, Some(1)])?;
    let qkv_id = model.add_consecutive_layer(Layer::EinSum(projection), Some(input))?;
    // EinSum supports at most four total inputs (activation plus three weights).
    let gate_projection = model.add_consecutive_layer(
        Layer::EinSum(EinSum::new(
            "X(se)@WG(ehgd)->G(ghsd)".into(),
            vec![Some(split(1, "gate")?)],
            vec![None],
        )?),
        Some(input),
    )?;
    let normalized = |model: &mut Model<f32>, port, name| -> Result<NodeId> {
        let mut alpha = weight(&loader, name, &[d])?
            .try_map_tensor(|tensor| tensor.try_map(|value| Ok(*value + 1.0)))?;
        alpha.key = format!("{}_effective", alpha.key).into();
        unary(
            model,
            qkv_id,
            port,
            Layer::QwenNorm(QwenNorm::new(
                Some(alpha.into()),
                c.rms_norm_eps,
                d,
                NormKind::Rms,
            )?),
        )
    };
    let query = normalized(model, 0, "q_norm.weight")?;
    let key = normalized(model, 1, "k_norm.weight")?;
    let rope = Positional::new_partial_rope_from_frequency(
        c.rope_parameters.rope_theta,
        format!("qwen35_rope_{index}").into(),
        d,
        (d as f32 * c.rope_parameters.partial_rotary_factor) as usize,
        context,
    )?;
    let query = model.add_consecutive_layer(Layer::Positional(rope.clone()), Some(query))?;
    let key = model.add_consecutive_layer(
        Layer::Positional(rope.with_cache().with_rope_cache(3, 1)?),
        Some(key),
    )?;
    let qkt_id = model.graph_mut().add_inner(Layer::EinSum(
        EinSum::new("Q(ghqd)@K(hsd)->A(ghqs)".into(), vec![None], vec![None])?.no_requant(),
    ))?;
    model.add_edge(query, qkt_id, (0, 0))?;
    model.add_edge(key, qkt_id, (0, 1))?;
    let mask_id = model.add_consecutive_layer(
        Layer::AttentionMask(AttentionMask::new(f32::NEG_INFINITY)),
        Some(qkt_id),
    )?;
    let softmax_id = model.add_consecutive_layer(
        Layer::Softmax(Softmax::new(context).with_scale(1.0 / (d as f32).sqrt())),
        Some(mask_id),
    )?;
    let values = model.graph_mut().add_inner(Layer::EinSum(EinSum::new(
        "A(ghqs)@V(hsd)->O(ghqd)".into(),
        vec![None],
        vec![None],
    )?))?;
    model.add_edge(softmax_id, values, (0, 0))?;
    model.add_edge(qkv_id, values, (2, 1))?;
    let gate = unary(
        model,
        gate_projection,
        0,
        Layer::Activation(Activation::new_sigmoid()),
    )?;
    let gated = model.graph_mut().add_inner(Layer::EinSum(EinSum::new(
        "X(ghsd)@G(ghsd)->O(ghsd)".into(),
        vec![None],
        vec![None],
    )?))?;
    model.add_edge(values, gated, (0, 0))?;
    model.add_edge(gate, gated, (0, 1))?;
    let output_weight = weight(&loader, "o_proj.weight", &[e, c.num_attention_heads * d])?
        .try_map_tensor(|tensor| {
            let tensor = tensor.transpose()?;
            let de = d * e;
            let mut data = vec![0.0; tensor.data().len()];
            for head in 0..h {
                for group in 0..g {
                    let src = (head * g + group) * de;
                    let dst = (group * h + head) * de;
                    data[dst..dst + de].copy_from_slice(&tensor.data()[src..src + de]);
                }
            }
            Tensor::new(vec![g, h, d, e].into(), data)
        })?;
    let final_proj_id = model.add_consecutive_layer(
        Layer::EinSum(
            EinSum::new(
                "X(ghsd)@W(ghde)->O(se)".into(),
                vec![Some(output_weight.into())],
                vec![None],
            )?
            .no_requant(),
        ),
        Some(gated),
    )?;
    Ok((
        final_proj_id,
        AttentionMetadata {
            qkv_id,
            qkt_id,
            mask_id,
            softmax_id,
            residual_id: values,
            final_proj_id,
        },
    ))
}

fn linear_attention_block(
    model: &mut Model<f32>,
    input: NodeId,
    block: &FileTensorLoader,
    config: &Config,
) -> Result<(NodeId, LinearAttentionMetadata)> {
    let c = config;
    let loader = block.pp("linear_attn.");
    let h = c.linear_num_key_heads;
    let dk = c.linear_key_head_dim;
    let dv = c.linear_value_head_dim;
    let e = c.hidden_size;
    let widths = [h * dk, h * dk, h * dv];
    let total: usize = widths.iter().sum();
    let qkv = weight(&loader, "in_proj_qkv.weight", &[total, e])?;
    let conv = weight(
        &loader,
        "conv1d.weight",
        &[total, 1, c.linear_conv_kernel_dim],
    )?
    .try_map_tensor(|tensor| tensor.reshaped(vec![total, c.linear_conv_kernel_dim].into()))?;
    let mut projections = Vec::new();
    let mut convolutions = Vec::new();
    let mut mixed = Vec::new();
    let mut offset = 0;
    for (index, width) in widths.into_iter().enumerate() {
        let name = ["query", "key", "value"][index];
        let projection =
            rows(&qkv, offset, width, name)?.try_map_tensor(|tensor| tensor.transpose())?;
        let projection = model.add_consecutive_layer(
            Layer::EinSum(EinSum::new(
                "X(se)@W(ec)->O(sc)".into(),
                vec![Some(projection.into())],
                vec![None],
            )?),
            Some(input),
        )?;
        let kernel = rows(&conv, offset, width, name)?;
        let convolution = model.add_consecutive_layer(
            Layer::CausalConv1d(CausalConv1d::new(kernel)?),
            Some(projection),
        )?;
        let activation = model
            .add_consecutive_layer(Layer::Activation(Activation::new_silu()), Some(convolution))?;
        let dim = if index == 2 { dv } else { dk };
        let shaped = model.add_consecutive_layer(
            Layer::Reshape(Reshape::new_subspace(1..2, vec![h, dim])),
            Some(activation),
        )?;
        let normalized = if index < 2 {
            let alpha = if index == 0 {
                Some(
                    KeyedTensor::new(
                        format!("{}query_scale", loader.current_prefix()),
                        Tensor::new(vec![dk].into(), vec![1.0 / (dk as f32).sqrt(); dk])?,
                    )
                    .into(),
                )
            } else {
                None
            };
            model.add_consecutive_layer(
                Layer::QwenNorm(QwenNorm::new(alpha, 1e-6, dk, NormKind::L2)?),
                Some(shaped),
            )?
        } else {
            shaped
        };
        projections.push(projection);
        convolutions.push(convolution);
        mixed.push(normalized);
        offset += width;
    }
    let a_weight = weight(&loader, "in_proj_a.weight", &[h, e])?
        .try_map_tensor(|tensor| tensor.transpose())?;
    let dt_bias = weight(&loader, "dt_bias", &[h])?;
    let a = model.add_consecutive_layer(
        Layer::EinSum(EinSum::new(
            "X(se)@W(eh)->O(sh)+BIAS(h)".into(),
            vec![Some(a_weight.into())],
            vec![Some(dt_bias.into())],
        )?),
        Some(input),
    )?;
    let a = model.add_consecutive_layer(Layer::Activation(Activation::new_softplus()), Some(a))?;
    let mut decay = weight(&loader, "A_log", &[h])?
        .try_map_tensor(|tensor| tensor.try_map(|value| Ok(-value.exp())))?;
    decay.key = format!("{}_negative_exp", decay.key).into();
    let a = model.add_consecutive_layer(
        Layer::EinSum(EinSum::new(
            "X(sh)@W(h)->O(sh)".into(),
            vec![Some(decay.into())],
            vec![None],
        )?),
        Some(a),
    )?;
    let alpha =
        model.add_consecutive_layer(Layer::Activation(Activation::new_exp_negative()), Some(a))?;
    let b_weight = weight(&loader, "in_proj_b.weight", &[h, e])?
        .try_map_tensor(|tensor| tensor.transpose())?;
    let b = model.add_consecutive_layer(
        Layer::EinSum(EinSum::new(
            "X(se)@W(eh)->O(sh)".into(),
            vec![Some(b_weight.into())],
            vec![None],
        )?),
        Some(input),
    )?;
    let beta =
        model.add_consecutive_layer(Layer::Activation(Activation::new_sigmoid()), Some(b))?;
    let recurrence_id = model
        .graph_mut()
        .add_inner(Layer::GatedDeltaScan(GatedDeltaScan::new(h, dk, dv)))?;
    for (port, id) in mixed.into_iter().chain([alpha, beta]).enumerate() {
        model.add_edge(id, recurrence_id, (0, port))?;
    }
    let normalized = add_norm(
        model,
        recurrence_id,
        &loader,
        "norm.weight",
        dv,
        c.rms_norm_eps,
        false,
    )?;
    let z_weight = weight(&loader, "in_proj_z.weight", &[h * dv, e])?
        .try_map_tensor(|tensor| tensor.transpose()?.reshaped(vec![e, h, dv].into()))?;
    let z = model.add_consecutive_layer(
        Layer::EinSum(EinSum::new(
            "X(se)@W(ehd)->O(shd)".into(),
            vec![Some(z_weight.into())],
            vec![None],
        )?),
        Some(input),
    )?;
    let gated = model
        .graph_mut()
        .add_inner(Layer::Activation(Activation::new_swiglu()))?;
    model.add_edge(z, gated, (0, Activation::<f32>::GATE_INPUT_INDEX))?;
    model.add_edge(normalized, gated, (0, Activation::<f32>::UP_INPUT_INDEX))?;
    let output_weight = weight(&loader, "out_proj.weight", &[e, h * dv])?
        .try_map_tensor(|tensor| tensor.transpose()?.reshaped(vec![h, dv, e].into()))?;
    let final_proj_id = model.add_consecutive_layer(
        Layer::EinSum(
            EinSum::new(
                "X(shd)@W(hde)->O(se)".into(),
                vec![Some(output_weight.into())],
                vec![None],
            )?
            .no_requant(),
        ),
        Some(gated),
    )?;
    Ok((
        final_proj_id,
        LinearAttentionMetadata {
            qkv_ids: projections.try_into().expect("three projections"),
            convolution_ids: convolutions.try_into().expect("three convolutions"),
            recurrence_id,
            final_proj_id,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixture is generated independently by transformers; the complete
    /// graph exercises both mixers, effective 1+w norms, gates and partial RoPE.
    #[test]
    fn qwen35_hybrid_graph_matches_hugging_face_oracle() -> Result<()> {
        let directory = tempfile::tempdir()?;
        std::fs::write(
            directory.path().join("model.safetensors"),
            include_bytes!("fixtures/model.safetensors"),
        )?;
        std::fs::write(
            directory.path().join("config.json"),
            include_str!("fixtures/config.json"),
        )?;
        let raw = RawSafeTensors::new(
            directory.path().join("model.safetensors"),
            directory.path().join("tokenizer.json"),
            directory.path().join("config.json"),
        );
        let mut driver = Driver::load_from_model(Qwen35::new(), &raw, Some(16))?;
        let tokens = [1usize, 2, 3].map(crate::parser::llm::Token::from);
        let expected: Vec<Vec<f32>> = serde_json::from_str(include_str!("fixtures/logits.json"))?;
        let trace = driver.run(
            vec![driver.tokens_to_tensor(&tokens)?],
            &mut tenstore::GenStore::default(),
        )?;
        let block_references: std::collections::BTreeMap<String, Vec<Vec<Vec<f32>>>> =
            serde_json::from_str(include_str!("fixtures/traces.json"))?;
        for (index, block) in driver.md.transformers.iter().enumerate() {
            let output = driver.model.graph().successors(block.ffn.down_id)[0];
            let actual = trace
                .get_step(&output)
                .context("Missing block output")?
                .output_tensors()?[0]
                .clone();
            let reference = &block_references[&format!("layer_{index}")];
            let max_error = actual
                .data()
                .iter()
                .zip(reference.iter().flatten().flatten())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_error < 1e-6,
                "HF block {index} max hidden-state error {max_error}"
            );
        }
        let logits = trace
            .get_step(&driver.md.final_proj)
            .context("Missing logits")?
            .output_tensors()?[0]
            .clone();
        assert_eq!(logits.shape(), &Shape::from(vec![3, 32]));
        let mut max_error = 0.0f32;
        for (actual, expected) in logits.data().iter().zip(expected.iter().flatten()) {
            max_error = max_error.max((actual - expected).abs());
        }
        assert!(
            max_error < 2e-6,
            "HF full-graph max logit error {max_error}"
        );
        assert!(driver.md.transformers[0].transformer.is_none());
        assert!(driver.md.transformers[0].linear_attention.is_some());
        assert!(driver.md.transformers[1].transformer.is_some());
        // Driver asserts cached token predictions equal full causal replay.
        let first = driver.generate(&tokens, 3, &mut tenstore::GenStore::default())?;
        let first_tokens = driver.continuation_tokens(&first, tokens.len())?;
        let second = driver.generate(&tokens, 3, &mut tenstore::GenStore::default())?;
        assert_eq!(
            first_tokens,
            driver.continuation_tokens(&second, tokens.len())?
        );
        Ok(())
    }

    #[test]
    fn qwen35_config_accepts_independent_attention_width_and_rejects_invalid_layers() -> Result<()>
    {
        let mut value = serde_json::json!({
            "model_type":"qwen3_5_text", "hidden_size":8,"intermediate_size":16,
            "num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":4,
            "vocab_size":32,"max_position_embeddings":128,"rms_norm_eps":1e-6,
            "eos_token_id":31,"hidden_act":"silu","attn_output_gate":true,
            "layer_types":["linear_attention","full_attention"],"linear_conv_kernel_dim":4,
            "linear_key_head_dim":4,"linear_value_head_dim":4,"linear_num_key_heads":2,
            "linear_num_value_heads":2,"rope_parameters":{"rope_type":"default","rope_theta":10000.,"partial_rotary_factor":0.5}
        });
        let config: Config = serde_json::from_value(value.clone())?;
        config.validate(32)?;
        ensure!(config.hidden_size != config.num_attention_heads * config.head_dim);
        value["layer_types"][1] = "unsupported".into();
        ensure!(
            serde_json::from_value::<Config>(value)?
                .validate(32)
                .is_err()
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires QWEN35_MODEL_DIR and local full Qwen3.5 weights"]
    fn qwen35_local_full_model_float_logits() -> Result<()> {
        use crate::parser::llm::LLMTokenizer;
        let directory = std::env::var("QWEN35_MODEL_DIR").context("Set QWEN35_MODEL_DIR")?;
        let raw = RawSafeTensors::from_directory(directory)?;
        let tokenizer = Qwen35::new().load_tokenizer(&raw)?;
        let tokens = tokenizer.tokenize("The capital of France is");
        let driver = Driver::load_from_model(Qwen35::new(), &raw, Some(64))?;
        ensure!(driver.md.transformers.len() == 24);
        ensure!(
            driver
                .md
                .transformers
                .iter()
                .filter(|block| block.linear_attention.is_some())
                .count()
                == 18
        );
        let trace = driver.run(
            vec![driver.tokens_to_tensor(&tokens)?],
            &mut tenstore::GenStore::default(),
        )?;
        let logits = trace
            .get_step(&driver.md.final_proj)
            .context("Missing logits")?
            .output_tensors()?[0]
            .clone();
        ensure!(logits.shape() == &Shape::from(vec![tokens.len(), 248320]));
        ensure!(logits.data().iter().all(|value| value.is_finite()));
        ensure!(logits.data().iter().any(|value| value.abs() > 1.0));
        Ok(())
    }
}
