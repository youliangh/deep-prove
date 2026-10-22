//! Defines the decoder only transformer used in the GPT-2 model.

use std::marker::PhantomData;

use crate::{
    graph::NodeId,
    layers::{
        Layer, activation::ActivationLayer, add::Add,
        transformer::normalisation::layernorm::LayerNorm,
    },
    model::Model,
    parser::{
        LayerInsertion, Load,
        gguf::FileTensorLoader as GGUFLoader,
        json::FileTensorLoader as JSONLoader,
        llm::{
            ConfigJSON, LLMConfig, SafeLoader, config::LLMStructure, metadata::TransformerMetadata,
            models::gpt2::decoder::attention::GPT2Attention,
            transformer::feed_forward::FeedForwardNetwork,
        },
    },
};

use anyhow::{Context, Result, ensure};

pub mod attention;

pub struct GPT2Decoder {
    pre_attention_layernorm: LayerNorm<f32>,
    attention_mechanism: GPT2Attention,
    pre_ffn_layernorm: LayerNorm<f32>,
    feed_forward: FeedForwardNetwork,
}

impl LayerInsertion for GPT2Decoder {
    type Metadata = TransformerMetadata;
    fn add_to_model(
        self,
        model: &mut Model<f32>,
        previous_node_id: Option<NodeId>,
    ) -> Result<(NodeId, Self::Metadata)> {
        // First we check that there is a previous node to connect to, a transformer block should never be the first node
        ensure!(
            previous_node_id.is_some(),
            "Transformer block cannot be the first node in the model"
        );

        let GPT2Decoder {
            pre_attention_layernorm,
            attention_mechanism,
            pre_ffn_layernorm,
            feed_forward,
        } = self;

        let initial_norm_id = model
            .add_consecutive_layer(Layer::LayerNorm(pre_attention_layernorm), previous_node_id)?;
        // Unwrap is safe because we have checked for previous_node_id above
        let residual_id = previous_node_id.unwrap();
        let (post_attention_id, attention_md) =
            attention_mechanism.add_to_model(model, Some(initial_norm_id))?;

        let add_id = model.graph_mut().add_inner(Layer::Add(Add::<f32>::new()))?;
        model.add_edge(residual_id, add_id, (0, 0))?;
        model.add_edge(post_attention_id, add_id, (0, 1))?;

        let pre_ffn_norm_id =
            model.add_consecutive_layer(Layer::LayerNorm(pre_ffn_layernorm), Some(add_id))?;

        let (post_ffn_id, ffn_md) = feed_forward.add_to_model(model, Some(pre_ffn_norm_id))?;
        let final_add_id = model.graph_mut().add_inner(Layer::Add(Add::<f32>::new()))?;
        model.add_edge(add_id, final_add_id, (0, 0))?;
        model.add_edge(post_ffn_id, final_add_id, (0, 1))?;

        let md = TransformerMetadata {
            norm_id: initial_norm_id,
            transformer: Some(attention_md),
            linear_attention: None,
            ffn: ffn_md,
        };
        Ok((final_add_id, md))
    }
}

impl Load<SafeLoader> for GPT2Decoder {
    type Config = (LLMStructure, ConfigJSON);

    fn from_loader(loader: &SafeLoader, config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let (structure, config_json) = config;

        // Load pre-attention LayerNorm (ln_1)
        let ln1_weight = loader.get_tensor("ln_1.weight")?;
        let ln1_bias = loader.get_tensor("ln_1.bias")?;
        ensure!(
            ln1_weight.shape().as_ref() == &[structure.generic.embedding_size],
            "ln_1.weight must have shape [{}] vs given {:?}",
            structure.generic.embedding_size,
            ln1_weight.shape()
        );
        ensure!(
            ln1_bias.shape().as_ref() == &[structure.generic.embedding_size],
            "ln_1.bias must have shape [{}] vs given {:?}",
            structure.generic.embedding_size,
            ln1_bias.shape()
        );
        let eps = config_json
            .get::<f32, _>("layer_norm_epsilon")
            .context("layer_norm_epsilon not found")?;
        let pre_attention_layernorm = LayerNorm::new(ln1_weight.into(), ln1_bias.into(), eps)?;

        // Load attention mechanism
        // Don't add "attn." prefix because GPT2Attention::from_loader adds it internally
        let attention_mechanism = GPT2Attention::from_loader(loader, config)?;

        // Load pre-FFN LayerNorm (ln_2)
        let ln2_weight = loader.get_tensor("ln_2.weight")?;
        let ln2_bias = loader.get_tensor("ln_2.bias")?;
        ensure!(
            ln2_weight.shape().as_ref() == &[structure.generic.embedding_size],
            "ln_2.weight must have shape [{}] vs given {:?}",
            structure.generic.embedding_size,
            ln2_weight.shape()
        );
        ensure!(
            ln2_bias.shape().as_ref() == &[structure.generic.embedding_size],
            "ln_2.bias must have shape [{}] vs given {:?}",
            structure.generic.embedding_size,
            ln2_bias.shape()
        );
        let pre_ffn_layernorm = LayerNorm::new(ln2_weight.into(), ln2_bias.into(), eps)?;

        // Load feed forward network (MLP)
        // Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py#L301
        //
        // HuggingFace GPT-2 MLP uses Conv1D layers:
        // ```python
        // self.c_fc = Conv1D(intermediate_size, embed_dim)  # Up projection
        // self.c_proj = Conv1D(embed_dim, intermediate_size)  # Down projection
        // ```
        //
        // Conv1D stores weights as (in_features, out_features), so:
        // - c_fc.weight: [hidden_size, intermediate_size] (typically [768, 3072] for GPT-2 base)
        // - c_proj.weight: [intermediate_size, hidden_size]
        let feed_forward = {
            let up = loader.get_tensor("mlp.c_fc.weight")?;
            let up_bias = Some(loader.get_tensor("mlp.c_fc.bias")?);
            let down = loader.get_tensor("mlp.c_proj.weight")?;
            let down_bias = Some(loader.get_tensor("mlp.c_proj.bias")?);

            // Validate shapes match expected Conv1D layout
            ensure!(
                up.shape()[0] == structure.generic.hidden_size,
                "mlp.c_fc.weight should have shape [hidden_size, intermediate_size], got {:?}",
                up.shape()
            );
            ensure!(
                down.shape()[1] == structure.generic.embedding_size,
                "mlp.c_proj.weight should have shape [intermediate_size, hidden_size], got {:?}",
                down.shape()
            );

            FeedForwardNetwork {
                gate: None,
                up,
                up_bias,
                down,
                down_bias,
                activation: ActivationLayer::Gelu(None, PhantomData),
            }
        };

        Ok(Self {
            pre_attention_layernorm,
            attention_mechanism,
            pre_ffn_layernorm,
            feed_forward,
        })
    }
}

impl Load<GGUFLoader> for GPT2Decoder {
    type Config = LLMStructure;

    fn from_loader(loader: &GGUFLoader, structure: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embedding_size = structure.generic.embedding_size;
        let hidden_size = structure.generic.hidden_size;
        ensure!(
            embedding_size == hidden_size,
            "embedding_size must be equal to hidden_size"
        );

        let attention_mechanism = GPT2Attention::from_loader(loader, structure)?;

        let attn_norm_loader = loader.pp("attn_");
        // Use new LayerNorm::from_loader
        let pre_norm = LayerNorm::from_gguf(&attn_norm_loader, &structure.generic)?;

        let ffn_norm_loader = loader.pp("ffn_");

        let pre_ffn_norm = LayerNorm::from_gguf(&ffn_norm_loader, &structure.generic)?;

        // Use new FeedForward::from_loader
        let feed_forward = {
            let up = loader
                .get_tensor("ffn_up.weight")?
                .try_map_tensor(|t| t.transpose())?;
            let up_bias = Some(loader.get_tensor("ffn_up.bias")?);
            let down = loader
                .get_tensor("ffn_down.weight")?
                .try_map_tensor(|t| t.transpose())?;
            let down_bias = Some(loader.get_tensor("ffn_down.bias")?);
            ensure!(
                up.shape()[0] == structure.generic.hidden_size,
                "up have shape {:?} but in features should be equal to hidden_size: {}",
                up.shape(),
                structure.generic.hidden_size
            );
            ensure!(
                down.shape()[1] == structure.generic.embedding_size,
                "down have shape {:?} but out features should be equal to embedding_size: {}",
                down.shape(),
                structure.generic.embedding_size
            );
            FeedForwardNetwork {
                gate: None,
                up,
                up_bias,
                down,
                down_bias,
                activation: ActivationLayer::Gelu(None, PhantomData),
            }
        };

        Ok(Self {
            pre_attention_layernorm: pre_norm,
            attention_mechanism,
            pre_ffn_layernorm: pre_ffn_norm,
            feed_forward,
        })
    }
}

impl Load<JSONLoader> for GPT2Decoder {
    type Config = LLMConfig;

    fn from_loader(l: &JSONLoader, config: &Self::Config) -> anyhow::Result<Self> {
        let pre_attention_layernorm = LayerNorm::from_json(&l.pp("attn_"), config)
            .context("Failed to load LayerNorm for attention in from_json")?;

        let attention_mechanism = GPT2Attention::from_loader(l, config)
            .context("Failed to load GPT2Attention in from_json")?;

        let pre_ffn_norm = LayerNorm::from_json(&l.pp("ffn_"), config)?;
        let feed_forward = {
            let up = l.get_tensor("ffn_up.weight")?;
            let up_bias = l.get_tensor("ffn_up.bias")?;
            let down = l.get_tensor("ffn_down.weight")?;
            let down_bias = l.get_tensor("ffn_down.bias")?;
            ensure!(
                up.shape()[0] == config.hidden_size,
                "up have shape {:?} but in features should be equal to hidden_size: {}",
                up.shape(),
                config.hidden_size
            );
            ensure!(
                down.shape()[1] == config.embedding_size,
                "down have shape {:?} but out features should be equal to embedding_size: {}",
                down.shape(),
                config.embedding_size
            );
            FeedForwardNetwork {
                gate: None,
                up,
                up_bias: Some(up_bias),
                down,
                down_bias: Some(down_bias),
                activation: ActivationLayer::Gelu(None, PhantomData),
            }
        };

        Ok(Self {
            pre_attention_layernorm,
            attention_mechanism,
            pre_ffn_layernorm: pre_ffn_norm,
            feed_forward,
        })
    }
}
