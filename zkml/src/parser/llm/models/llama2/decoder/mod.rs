//! Llama2 model decoder implementation.

use std::marker::PhantomData;

use crate::{
    graph::NodeId,
    layers::{
        Layer, activation::ActivationLayer, add::Add, transformer::normalisation::rmsnorm::RMSNorm,
    },
    model::Model,
    parser::{
        LayerInsertion, Load,
        llm::{
            config::LLMStructure, metadata::TransformerMetadata,
            transformer::feed_forward::FeedForwardNetwork,
        },
        safe::{ConfigJSON, FileTensorLoader as SafeLoader},
    },
};

use anyhow::{Result, ensure};

pub mod attention;
use attention::Llama2Attention;

#[derive(Debug, Clone)]
pub struct Llama2Decoder {
    input_layernorm: RMSNorm<f32>,
    attention: Llama2Attention,
    post_attention_norm: RMSNorm<f32>,
    ffn: FeedForwardNetwork,
}

impl LayerInsertion for Llama2Decoder {
    type Metadata = TransformerMetadata;
    fn add_to_model(
        self,
        model: &mut Model<f32>,
        previous_node_id: Option<NodeId>,
    ) -> Result<(NodeId, Self::Metadata)> {
        let Llama2Decoder {
            input_layernorm,
            attention: attention_mechanism,
            post_attention_norm: post_attention_layernorm,
            ffn: feed_forward,
        } = self;

        let post_input_norm_id =
            model.add_consecutive_layer(Layer::RMSNorm(input_layernorm), previous_node_id)?;

        // Attention mechanism
        let (post_attention_id, attention_md) =
            attention_mechanism.add_to_model(model, Some(post_input_norm_id))?;

        // Residual add after attention
        let residual_id = previous_node_id.ok_or(anyhow::anyhow!(
            "Llama2Decoder block should never be the first layer of the model"
        ))?;
        let add_id = model.graph_mut().add_inner(Layer::Add(Add::new()))?;
        model.add_edge(residual_id, add_id, (0, 0))?;
        model.add_edge(post_attention_id, add_id, (0, 1))?;

        // Post-attention LayerNorm
        let post_attention_norm_id =
            model.add_consecutive_layer(Layer::RMSNorm(post_attention_layernorm), Some(add_id))?;

        // Feed forward network
        let (post_ffn_id, ffn_md) =
            feed_forward.add_to_model(model, Some(post_attention_norm_id))?;

        // Final residual add
        let final_add_id = model.graph_mut().add_inner(Layer::Add(Add::new()))?;
        model.add_edge(add_id, final_add_id, (0, 0))?;
        model.add_edge(post_ffn_id, final_add_id, (0, 1))?;
        let md = TransformerMetadata {
            norm_id: post_input_norm_id,
            transformer: Some(attention_md),
            linear_attention: None,
            ffn: ffn_md,
        };
        Ok((final_add_id, md))
    }
}

impl Load<SafeLoader> for Llama2Decoder {
    type Config = (LLMStructure, ConfigJSON);

    fn from_loader(loader: &SafeLoader, config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let (structure, _cfg) = config;
        let input_layernorm =
            RMSNorm::from_safe(&loader.pp("input_layernorm."), &structure.generic)?;

        let attention_mechanism = Llama2Attention::from_loader(loader, config)?;

        let post_attention_layernorm =
            RMSNorm::from_safe(&loader.pp("post_attention_layernorm."), &structure.generic)?;

        let feed_forward = {
            let gate = loader
                .get_tensor("mlp.gate_proj.weight")?
                .try_map_tensor(|t| t.transpose())?;
            ensure!(
                gate.shape()[0] == structure.generic.hidden_size,
                "gate has shape {:?} but in features should be equal to hidden_size: {}",
                gate.shape(),
                structure.generic.hidden_size
            );

            let up = loader
                .get_tensor("mlp.up_proj.weight")?
                .try_map_tensor(|t| t.transpose())?;

            let down = loader
                .get_tensor("mlp.down_proj.weight")?
                .try_map_tensor(|t| t.transpose())?;

            ensure!(
                up.shape()[0] == structure.generic.hidden_size,
                "up has shape {:?} but in features should be equal to hidden_size: {}",
                up.shape(),
                structure.generic.hidden_size
            );
            ensure!(
                down.shape()[1] == structure.generic.embedding_size,
                "down has shape {:?} but out features should be equal to embedding_size: {}",
                down.shape(),
                structure.generic.embedding_size
            );
            FeedForwardNetwork {
                gate: Some(gate),
                up,
                up_bias: None,
                down,
                down_bias: None,
                activation: ActivationLayer::Silu(None, PhantomData),
            }
        };

        Ok(Self {
            input_layernorm,
            attention: attention_mechanism,
            post_attention_norm: post_attention_layernorm,
            ffn: feed_forward,
        })
    }
}
