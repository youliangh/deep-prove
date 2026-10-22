use serde::{Deserialize, Serialize};

use crate::{graph::NodeId, parser::llm::LLMConfig};

/// Structure containing the fixed config of the LLM model but as well some helper information to quickly
/// locate "llm-structures" inside the generic graph, such as transformers, attentions, etc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LLMMetadata {
    pub config: LLMConfig,
    /// Additional stop tokens bound into the trusted verifier context.
    #[serde(default)]
    pub additional_eos_tokens: Vec<super::Token>,
    pub transformers: Vec<TransformerMetadata>,
    pub embeddings: NodeId,
    pub positional: Option<NodeId>,
    pub final_proj: NodeId,
    // the output of this layer contains the logits
    pub logits: NodeId,
    // the output of this layer contains the argmax of the logits
    pub argmax: NodeId,
}

impl LLMMetadata {
    pub fn eos_tokens(&self) -> Vec<super::Token> {
        let mut tokens = vec![self.config.eos_token];
        for &token in &self.additional_eos_tokens {
            if !tokens.contains(&token) {
                tokens.push(token);
            }
        }
        tokens
    }
    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    pub fn context_length(&self) -> usize {
        self.config.context_length
    }
}

/// Structure containing relevant node IDs of an attention layer when a LLM is compiled into our graph system.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TransformerMetadata {
    pub norm_id: NodeId,
    /// Present only for softmax attention blocks.
    pub transformer: Option<AttentionMetadata>,
    /// Present only for causal recurrent attention blocks.
    #[serde(default)]
    pub linear_attention: Option<LinearAttentionMetadata>,
    pub ffn: FFNGraphID,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LinearAttentionMetadata {
    pub qkv_ids: [NodeId; 3],
    pub convolution_ids: [NodeId; 3],
    pub recurrence_id: NodeId,
    pub final_proj_id: NodeId,
}

/// Structure containing relevant node IDs of a transformer, e.g. qkv, softmax, residual, etc.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AttentionMetadata {
    pub qkv_id: NodeId,
    pub qkt_id: NodeId,
    pub mask_id: NodeId,
    pub softmax_id: NodeId,
    pub residual_id: NodeId,
    pub final_proj_id: NodeId,
}

/// Similar to `[TransformerGraphID]`, but for the feed-forward network.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FFNGraphID {
    pub up_id: NodeId,
    pub down_id: NodeId,
    pub activation_id: NodeId,
    pub uses_glu: bool,
}

#[cfg(test)]
mod tests {
    use crate::{
        layers::{Layer, provable::OpInfo},
        parser::{
            ModelLoader, file_cache,
            gguf::RawGGUF,
            llm::models::gemma3::{Gemma3, tests::GEMMA3_Q8},
        },
    };

    #[test]
    fn test_transformer_metadata() -> anyhow::Result<()> {
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let mygguf = RawGGUF::new(model_path);
        let (model, md) = Gemma3::new().with_max_context(2048).parse(&mygguf)?;
        let embeddings_id = model.graph().inner_nodes().next().unwrap().0;
        let computed_embeddings_id = md.embeddings;
        assert_eq!(embeddings_id, computed_embeddings_id);
        let norm_id = model.graph().successors(embeddings_id)[0];
        let qkv_id = model.graph().successors(norm_id)[0];
        assert_eq!(
            qkv_id,
            md.transformers[0]
                .transformer
                .as_ref()
                .expect("softmax attention metadata")
                .qkv_id
        );
        // So attention is
        // * qkv, norm, rope, qkt, mask, softmax, qv, final_proj,
        // * then norm , and residual and norm again and then ffn
        // so 9 more layers
        let mut last_layer_id = qkv_id;
        for _ in 0..11 {
            last_layer_id = model.graph().successors(last_layer_id)[0];
            let node_ref = model
                .graph()
                .node(last_layer_id)
                .unwrap()
                .as_inner()
                .unwrap();
            println!(
                "last_layer_id: {:?} -> node: {:?}",
                last_layer_id,
                node_ref.describe()
            );
            if let Layer::Requant(_) = node_ref {
                // we dont count requant nodes
                last_layer_id = model.graph().successors(last_layer_id)[0];
            }
        }
        assert_eq!(last_layer_id, md.transformers[0].ffn.up_id);
        Ok(())
    }
}
