//! Gemma3 model decoder implementation.

use std::marker::PhantomData;

use crate::{
    graph::NodeId,
    layers::{
        Layer, activation::ActivationLayer, add::Add, transformer::normalisation::rmsnorm::RMSNorm,
    },
    model::Model,
    parser::{
        LayerInsertion, Load,
        gguf::FileTensorLoader as GGUFLoader,
        llm::{
            config::LLMStructure, metadata::TransformerMetadata,
            transformer::feed_forward::FeedForwardNetwork,
        },
        safe::{ConfigJSON, FileTensorLoader as SafeLoader},
    },
};

use anyhow::{Result, anyhow, ensure};

pub mod attention;
use attention::Gemma3Attention;

#[derive(Debug, Clone)]
pub struct Gemma3Decoder {
    input_rmsnorm: RMSNorm<f32>,
    attention_mechanism: Gemma3Attention,
    post_attention_rmsnorm: RMSNorm<f32>,
    pre_ffn_rmsnorm: RMSNorm<f32>,
    feed_forward: FeedForwardNetwork,
    post_ffn_rmsnorm: RMSNorm<f32>,
}

impl LayerInsertion for Gemma3Decoder {
    type Metadata = TransformerMetadata;
    fn add_to_model(
        self,
        model: &mut Model<f32>,
        previous_node_id: Option<NodeId>,
    ) -> Result<(NodeId, Self::Metadata)> {
        let Gemma3Decoder {
            input_rmsnorm,
            attention_mechanism,
            post_attention_rmsnorm,
            pre_ffn_rmsnorm,
            feed_forward,
            post_ffn_rmsnorm,
        } = self;

        let post_input_norm_id =
            model.add_consecutive_layer(Layer::RMSNorm(input_rmsnorm), previous_node_id)?;

        let (post_attention_id, attention_md) =
            attention_mechanism.add_to_model(model, Some(post_input_norm_id))?;

        let post_attention_norm_id = model.add_consecutive_layer(
            Layer::RMSNorm(post_attention_rmsnorm),
            Some(post_attention_id),
        )?;

        let residul_id = previous_node_id.ok_or(anyhow!(
            "Gemma3Decoder block should never be the first layer of the model"
        ))?;

        let add_id = model.graph_mut().add_inner(Layer::Add(Add::new()))?;
        model.add_edge(residul_id, add_id, (0, 0))?;
        model.add_edge(post_attention_norm_id, add_id, (0, 1))?;

        let pre_ffn_norm_id = model
            .graph_mut()
            .add_inner(Layer::RMSNorm(pre_ffn_rmsnorm))?;
        model.add_edge(add_id, pre_ffn_norm_id, (0, 0))?;

        let (post_ffn_id, ffn_md) = feed_forward.add_to_model(model, Some(pre_ffn_norm_id))?;

        let post_ffn_norm_id =
            model.add_consecutive_layer(Layer::RMSNorm(post_ffn_rmsnorm), Some(post_ffn_id))?;

        let final_add_id = model.graph_mut().add_inner(Layer::Add(Add::new()))?;
        model.add_edge(add_id, final_add_id, (0, 0))?;
        model.add_edge(post_ffn_norm_id, final_add_id, (0, 1))?;
        let md = TransformerMetadata {
            norm_id: post_input_norm_id,
            transformer: Some(attention_md),
            linear_attention: None,
            ffn: ffn_md,
        };
        Ok((final_add_id, md))
    }
}

/// Scale the RMSNorm alpha by adding 1.0 to each element.
fn scale_norm(rmsnorm: &mut RMSNorm<f32>) {
    if let Some(mut alpha) = rmsnorm.alpha.take() {
        let tensor = alpha.take_wrapped_tensor().unwrap();
        let scaled = tensor.add_scalar(1.0);
        alpha.set_wrapped_tensor(scaled).unwrap();
        rmsnorm.alpha = Some(alpha);
    }
}

impl Load<GGUFLoader> for Gemma3Decoder {
    type Config = LLMStructure;

    fn from_loader(loader: &GGUFLoader, structure: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let mut input_rmsnorm = RMSNorm::from_gguf(&loader.pp("attn_"), &structure.generic)?;

        let attention_mechanism = Gemma3Attention::from_loader(loader, structure)?;

        let mut post_attention_rmsnorm =
            RMSNorm::from_gguf(&loader.pp("post_attention_"), &structure.generic)?;
        let mut pre_ffn_rmsnorm = RMSNorm::from_gguf(&loader.pp("ffn_"), &structure.generic)?;

        let feed_forward = {
            let gate = loader
                .get_tensor("ffn_gate.weight")?
                .try_map_tensor(|t| t.transpose())?;
            ensure!(
                gate.shape()[0] == structure.generic.hidden_size,
                "gate have shape {:?} but in features should be equal to hidden_size: {}",
                gate.shape(),
                structure.generic.hidden_size
            );

            let up = loader
                .get_tensor("ffn_up.weight")?
                .try_map_tensor(|t| t.transpose())?;

            let down = loader
                .get_tensor("ffn_down.weight")?
                .try_map_tensor(|t| t.transpose())?;

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
                gate: Some(gate),
                up,
                up_bias: None,
                down,
                down_bias: None,
                activation: ActivationLayer::Gelu(None, PhantomData),
            }
        };

        let mut post_ffn_rmsnorm = RMSNorm::from_gguf(&loader.pp("post_ffw_"), &structure.generic)?;

        // Scale the RMSNorm alphas because Gemma3 expects that we add 1.0 to each element.
        scale_norm(&mut input_rmsnorm);
        scale_norm(&mut post_attention_rmsnorm);
        scale_norm(&mut pre_ffn_rmsnorm);
        scale_norm(&mut post_ffn_rmsnorm);

        Ok(Self {
            input_rmsnorm,
            attention_mechanism,
            post_attention_rmsnorm,
            pre_ffn_rmsnorm,
            feed_forward,
            post_ffn_rmsnorm,
        })
    }
}

impl Load<SafeLoader> for Gemma3Decoder {
    type Config = (LLMStructure, ConfigJSON); // (config, structure, block_number)

    fn from_loader(loader: &SafeLoader, config: &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let (structure, _) = config;
        let mut input_rmsnorm =
            RMSNorm::from_safe(&loader.pp("input_layernorm."), &structure.generic)?;

        let attention_mechanism = Gemma3Attention::from_loader(loader, config)?;

        let mut post_attention_rmsnorm =
            RMSNorm::from_safe(&loader.pp("post_attention_layernorm."), &structure.generic)?;

        let mut pre_ffn_rmsnorm =
            RMSNorm::from_safe(&loader.pp("pre_feedforward_layernorm."), &structure.generic)?;

        let feed_forward = {
            let gate = loader
                .get_tensor("mlp.gate_proj.weight")?
                .try_map_tensor(|t| t.transpose())?;
            ensure!(
                gate.shape()[0] == structure.generic.hidden_size,
                "gate have shape {:?} but in features should be equal to hidden_size: {}",
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
                gate: Some(gate),
                up,
                up_bias: None,
                down,
                down_bias: None,
                activation: ActivationLayer::Gelu(None, PhantomData),
            }
        };
        let mut post_ffn_rmsnorm = RMSNorm::from_safe(
            &loader.pp("post_feedforward_layernorm."),
            &structure.generic,
        )?;

        // Scale the RMSNorm alphas because Gemma3 expects that we add 1.0 to each element.
        scale_norm(&mut input_rmsnorm);
        scale_norm(&mut post_attention_rmsnorm);
        scale_norm(&mut pre_ffn_rmsnorm);
        scale_norm(&mut post_ffn_rmsnorm);
        Ok(Self {
            input_rmsnorm,
            attention_mechanism,
            post_attention_rmsnorm,
            pre_ffn_rmsnorm,
            feed_forward,
            post_ffn_rmsnorm,
        })
    }
}
