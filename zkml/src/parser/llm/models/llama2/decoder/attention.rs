//! Llama2 attention mechanism implementation.

use crate::{
    graph::NodeId,
    layers::{
        Layer,
        transformer::{attention_mask::AttentionSpan, positional::Positional},
    },
    model::Model,
    parser::{
        Load,
        llm::{
            config::{AttentionHeadType, LLMStructure},
            models::llama::load_rope,
            transformer::attention_layer::{AttentionCacheConfig, AttentionMechanism},
        },
        safe::{ConfigJSON, FileTensorLoader as SafeLoader},
    },
    tensor::{Tensor, TensorHandle},
};

use anyhow::{Result, bail};

#[derive(Clone, Debug)]
pub struct Llama2Attention {
    max_context_length: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    wq: TensorHandle<f32>,
    wk: TensorHandle<f32>,
    wv: TensorHandle<f32>,
    wo: TensorHandle<f32>,
    span: AttentionSpan,
    rope: Positional<f32>,
}

impl AttentionMechanism for Llama2Attention {
    fn attention_span(&self) -> AttentionSpan {
        self.span
    }

    fn total_heads(&self) -> usize {
        self.num_heads
    }

    fn kv_total_heads(&self) -> usize {
        self.num_kv_heads
    }

    fn heads_per_kv_head(&self) -> usize {
        self.num_heads / self.num_kv_heads
    }

    fn max_context_length(&self) -> usize {
        self.max_context_length
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    fn uses_qkv_bias(&self) -> bool {
        false
    }

    fn uses_out_bias(&self) -> bool {
        false
    }

    fn qkv_tensors(&self) -> (Vec<TensorHandle<f32>>, Vec<Option<TensorHandle<f32>>>) {
        (
            vec![self.wq.clone(), self.wk.clone(), self.wv.clone()],
            vec![None, None, None],
        )
    }

    fn out_tensors(&self) -> (TensorHandle<f32>, Option<TensorHandle<f32>>) {
        (self.wo.clone(), None)
    }

    fn insert_custom_logic(
        self,
        model: &mut Model<f32>,
        qkv_einsum_id: NodeId,
        query_key_id: NodeId,
    ) -> Result<()> {
        // First we add the caching to the QKV einsum
        let AttentionCacheConfig {
            kv_rank,
            caching_dim,
        } = self.caching_dim();

        let qkv_einsum = model.graph_mut().node_mut(qkv_einsum_id).ok_or(anyhow::anyhow!("Could not insert Llama2Attention, QKV EinSum with ID({qkv_einsum_id}) has not been inserted"))?;

        if let Some(Layer::EinSum(einsum)) = qkv_einsum.as_inner_mut() {
            // Only V (output 2) is cached in the einsum; K is cached via the RoPE layer
            einsum.with_caches(vec![None, None, Some(caching_dim)])?;
        } else {
            bail!("QKV Einsum Node is not an EinSum layer");
        }

        // RoPE is applied directly after QKV (no Q/K norms like Gemma3)
        let q_rope_id = model
            .graph_mut()
            .add_inner(Layer::Positional(self.rope.clone()))?;
        let k_rope_id = model.graph_mut().add_inner(Layer::Positional(
            self.rope
                .with_cache() // ensure it's a fresh cache
                .with_rope_cache(kv_rank, caching_dim)?,
        ))?;

        // Wire QKV outputs to RoPE: output 0 (Q) → Q RoPE, output 1 (K) → K RoPE
        model.add_edge(qkv_einsum_id, q_rope_id, vec![(0, 0)])?;
        model.add_edge(qkv_einsum_id, k_rope_id, vec![(1, 0)])?;

        // Wire RoPE outputs to query-key attention
        model.add_edge(q_rope_id, query_key_id, vec![(0, 0)])?;
        model.add_edge(k_rope_id, query_key_id, vec![(0, 1)])?;
        Ok(())
    }
}

impl Load<SafeLoader> for Llama2Attention {
    type Config = (LLMStructure, ConfigJSON);

    fn from_loader(loader: &SafeLoader, (structure, config): &Self::Config) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let embedding_size = structure.generic.embedding_size;
        let max_ctx_length = structure.generic.context_length;
        let head_size = structure.generic.head_size;
        let num_heads = structure.generic.num_heads;
        let AttentionHeadType::GQA(num_groups) = structure.attention_config.head else {
            bail!("GQA is expected for Llama2 models");
        };

        let heads_per_group = num_heads / num_groups;

        // ===================================================================================
        // Weight Reshape Logic for GQA (Grouped Query Attention)
        // ===================================================================================
        //
        // For TinyLlama: num_heads=32, num_groups=4, heads_per_group=8, head_size=64
        //
        // Our einsum uses indices:
        //   g = heads_per_group (8)   - which query head within the KV group
        //   h = num_groups (4)        - which KV group
        //   d = head_size (64)        - dimension within a head
        //   e = embedding_size (2048) - output embedding dimension
        //
        // The QKV einsum produces Q with shape [g, h, s, d], where the head at position
        // [g, h, :, :] corresponds to HuggingFace head number: h * heads_per_group + g
        //
        // For example with TinyLlama:
        //   Q[0, 0, :, :] = head 0*8 + 0 = head 0
        //   Q[1, 0, :, :] = head 0*8 + 1 = head 1
        //   Q[0, 1, :, :] = head 1*8 + 0 = head 8
        //   Q[7, 3, :, :] = head 3*8 + 7 = head 31
        // ===================================================================================
        // Llama2 always uses GQA, so we use grouped shapes directly
        let wq_reshape = vec![embedding_size, num_groups, heads_per_group, head_size];
        let wk_wv_reshape = vec![embedding_size, num_groups, head_size];
        // WO final shape is [g, h, d, e] to match attention output O's layout
        let wo_reshape = vec![heads_per_group, num_groups, head_size, embedding_size];

        let wq = loader
            .get_tensor("self_attn.q_proj.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wq_reshape.clone().into()))?;

        let wk = loader
            .get_tensor("self_attn.k_proj.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wk_wv_reshape.clone().into()))?;

        let wv = loader
            .get_tensor("self_attn.v_proj.weight")?
            .try_map_tensor(|t| t.transpose()?.reshaped(wk_wv_reshape.clone().into()))?;

        // ===================================================================================
        // Output Projection Weight Permutation (GQA only)
        // ===================================================================================
        //
        // PROBLEM:
        // HuggingFace stores o_proj.weight with shape [hidden_size, hidden_size] = [2048, 2048]
        // where heads are laid out linearly: head i occupies columns [i*64, (i+1)*64).
        //
        // Our output einsum is: O(ghqd) @ WO(ghde) -> Y(qe)
        // The attention output O has shape [g, h, q, d] where O[g, h, :, :] contains
        // head h*heads_per_group + g (e.g., O[1, 2, :, :] = head 2*8 + 1 = head 17).
        //
        // For the einsum to work correctly, WO[g, h, d, e] must contain the weight
        // for the SAME head: h*heads_per_group + g.
        //
        // NAIVE RESHAPE FAILS:
        // If we naively reshape [2048] to [g=8, h=4, d=64]:
        //   - Linear position x maps to [x/256, (x/64)%4, x%64]
        //   - Position [g, h, d] = linear g*256 + h*64 + d
        //   - This gives head (g*256 + h*64)/64 = g*4 + h
        //
        // But we NEED head h*8 + g at position [g, h, :]. These don't match!
        //   - [1, 0, :] would have head 1*4 + 0 = 4, but we need head 0*8 + 1 = 1
        //
        // SOLUTION:
        // 1. First reshape to [h, g, d*e] - this preserves HuggingFace's head ordering
        //    because linear position x maps to [x/512, (x/64)%8, ...] giving head h*8 + g
        // 2. Then permute axes [h, g, d*e] -> [g, h, d*e] to match O's layout
        // 3. Finally reshape to [g, h, d, e]
        //
        // After this, WO[g, h, d, e] correctly contains weight for head h*8 + g.
        // ===================================================================================
        let wo = loader
            .get_tensor("self_attn.o_proj.weight")?
            .try_map_tensor(|t| {
                let t = t.transpose()?;
                // Reshape to [h, g, d*e] where h=num_groups, g=heads_per_group
                let de = head_size * embedding_size;
                let data = t.data();
                // Permute [h, g, de] -> [g, h, de]: element at [h, g, k] goes to [g, h, k]
                let mut permuted = vec![0.0f32; data.len()];
                for h in 0..num_groups {
                    for g in 0..heads_per_group {
                        let src_offset = (h * heads_per_group + g) * de;
                        let dst_offset = (g * num_groups + h) * de;
                        permuted[dst_offset..dst_offset + de]
                            .copy_from_slice(&data[src_offset..src_offset + de]);
                    }
                }
                Tensor::new(wo_reshape.clone().into(), permuted)
            })?;

        Ok(Self {
            max_context_length: structure.generic.context_length,
            num_heads,
            num_kv_heads: num_groups,
            head_dim: head_size,
            wq: wq.into(),
            wk: wk.into(),
            wv: wv.into(),
            wo: wo.into(),
            span: AttentionSpan::Full, // Llama2 always uses full attention
            rope: load_rope(config, head_size, max_ctx_length)?,
        })
    }
}
