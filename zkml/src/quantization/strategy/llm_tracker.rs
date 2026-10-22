//! LLM Inference tracker for more advanced quantisation strategies.

use dp_crypto::util::ceil_log2;

use super::*;
use crate::{
    layers::{
        Layer, provable::Evaluate, requant::FIXED_POINT_SCALE,
        transformer::positional::PositionalVariant,
    },
    lookup::table::SHIFT_CHECK_TABLE_BIT_SIZE,
    model::{LayerRunner, ModelGraph, RunInput, RunResult},
    parser::llm::metadata::LLMMetadata,
};
use std::collections::HashSet;

/// The Qwen FFN retains the unrounded up projection until the GLU product.
/// Its magnitude has `2 * (bits - 1) + log2(embedding)` bits. The following
/// Requant consumes that product before the down projection, so the down
/// contraction must not consume this lookup's separate precision budget.
fn qwen_ffn_output_table_bits(embedding: usize, bits: usize) -> Option<usize> {
    if embedding == 0 || bits < 2 {
        return None;
    }
    let up_magnitude_bits = (bits - 1)
        .checked_mul(2)?
        .checked_add(ceil_log2(embedding))?;
    let output_bits = 63usize
        .checked_sub(FIXED_POINT_SCALE)?
        .checked_sub(up_magnitude_bits)?
        .min(SHIFT_CHECK_TABLE_BIT_SIZE);
    (output_bits >= 2).then_some(output_bits)
}

pub struct LLMInferenceTracker {
    /// Streaming estimator of the selected statistics for each output of each
    /// node.
    accumulators: HashMap<NodeOutput, InferenceTrackingAccumulator>,
    add_layers: HashSet<NodeOutput>,
    /// Streaming estimator of the selected statistics for given intermediate data of
    /// each node, if any
    intermediate_data_trackers: HashMap<(NodeId, TrackedDataId), InferenceTrackingAccumulator>,
    /// Streaming estimator of selected statistics that are global across a model. For instance
    /// the inputs and outputs of activation functions for working out suitable tables.
    global_data_trackers: HashMap<TrackedDataId, InferenceTrackingAccumulator>,
    /// The LLM metadata
    llm_metadata: LLMMetadata,
    /// Embeddings matrix
    embeddings: Option<WrappedTensor<f32>>,
    /// Positional embeddings matrix
    positional_embeddings: Option<WrappedTensor<f32>>,
}

impl LLMInferenceTracker {
    pub fn new(llm_metadata: LLMMetadata) -> Self {
        Self {
            accumulators: HashMap::new(),
            add_layers: HashSet::new(),
            intermediate_data_trackers: HashMap::new(),
            global_data_trackers: HashMap::new(),
            llm_metadata,
            embeddings: None,
            positional_embeddings: None,
        }
    }

    pub(crate) fn track<N>(
        &mut self,
        output: NodeOutput,
        layer: &Layer<N>,
        handle: &TensorHandle<N>,
    ) -> anyhow::Result<()>
    where
        N: TensorTypeParam,
    {
        if self.embeddings.is_none() && matches!(layer, Layer::Embeddings(..)) {
            let Layer::Embeddings(em) = layer else {
                unreachable!();
            };
            let wrapped_embeddings = N::tensor_to_float(em.mat.wrapped_tensor()?.clone());
            self.embeddings = Some(wrapped_embeddings);
        } else if self.positional_embeddings.is_none() && matches!(layer, Layer::Positional(..)) {
            let Layer::Positional(pos_layer) = layer else {
                unreachable!();
            };
            if matches!(pos_layer.variant, PositionalVariant::Absolute(..)) {
                let abs_pos = match &pos_layer.variant {
                    PositionalVariant::Absolute(abs) => abs,
                    _ => unreachable!(),
                };
                let wrapped_tensor =
                    N::tensor_to_float(abs_pos.positional.wrapped_tensor()?.clone());
                self.positional_embeddings = Some(wrapped_tensor);
            }
        }

        if let Layer::Add(_) = layer {
            self.add_layers.insert(output);
        }

        let accumulator = self
            .accumulators
            .entry(output)
            .or_insert_with(|| InferenceTrackingMode::MinMax.new_accumulator());
        let guard = handle.wrapped_tensor().unwrap();
        let tensor = guard.deref();
        for el in N::tensor_to_float(tensor.clone()).get_data() {
            accumulator.add(el as f64);
        }
        Ok(())
    }

    pub(crate) fn calculate_embeddings_max(&self) -> Result<f32> {
        ensure!(self.embeddings.is_some(), "Embeddings not tracked");
        let embeddings = self.embeddings.as_ref().unwrap();
        match self.positional_embeddings {
            None => Ok(embeddings.clone().max_abs().get_data()[0]),
            Some(ref pos_embeddings) => {
                let WrappedTensor::Rank2 {
                    tensor: embeddings, ..
                } = embeddings.clone()
                else {
                    anyhow::bail!("Embeddings tensor is not rank 2");
                };
                let WrappedTensor::Rank2 {
                    tensor: pos_embeddings,
                    ..
                } = pos_embeddings.clone()
                else {
                    anyhow::bail!("Positional embeddings tensor is not rank 2");
                };

                let embeddings_maxes = embeddings.max_abs_dim(0);
                let positional_maxes = pos_embeddings.max_abs_dim(0);

                let full_maxes = embeddings_maxes.add(positional_maxes).max_abs();
                let shape = full_maxes.shape();
                Ok(WrappedTensor::Rank1 {
                    tensor: full_maxes,
                    unpadded_shape: shape,
                }
                .max_abs()
                .get_data()[0])
            }
        }
    }

    pub(crate) fn is_add_layer(&self, output: NodeOutput) -> bool {
        self.add_layers.contains(&output)
    }

    pub(crate) fn llm_metadata(&self) -> &LLMMetadata {
        &self.llm_metadata
    }

    pub(crate) fn track_inputs<N>(
        &mut self,
        output: NodeOutput,
        handle: &TensorHandle<N>,
    ) -> anyhow::Result<()>
    where
        N: TensorTypeParam,
    {
        let mode = InferenceTrackingMode::NSigmas(2);
        let accumulator = self
            .accumulators
            .entry(output)
            .or_insert_with(|| mode.new_accumulator());
        let guard = handle.wrapped_tensor().unwrap();
        let tensor = guard.deref();
        for el in N::tensor_to_float(tensor.clone()).get_data() {
            accumulator.add(el as f64);
        }
        Ok(())
    }

    pub(crate) fn track_intermediate_data<N>(
        &mut self,
        node_id: NodeId,
        data_id: TrackedDataId,
        layer: &Layer<N>,
        wrapped_tensor: WrappedTensor<N>,
    ) where
        N: TensorTypeParam,
    {
        let mode = InferenceTrackingMode::MinMax;
        let is_ffn_activation = matches!(layer, Layer::Activation(..))
            && self
                .llm_metadata
                .transformers
                .iter()
                .any(|block| block.ffn.activation_id == node_id);
        let accumulator = if is_ffn_activation {
            self.global_data_trackers
                .entry(data_id)
                .or_insert_with(|| mode.new_accumulator())
        } else {
            self.intermediate_data_trackers
                .entry((node_id, data_id))
                .or_insert_with(|| mode.new_accumulator())
        };

        for el in N::tensor_to_float(wrapped_tensor.clone()).get_data() {
            accumulator.add(el as f64);
        }
    }

    pub fn scaling_range(&self, output: NodeOutput) -> Option<(f32, f32)> {
        self.accumulators.get(&output).map(|a| a.range())
    }

    pub(crate) fn scaling_factor_for_intermediate_data(
        &self,
        node_id: NodeId,
        data_id: TrackedDataId,
    ) -> ScalingFactor {
        let is_ffn_activation = self
            .llm_metadata
            .transformers
            .iter()
            .any(|block| block.ffn.activation_id == node_id);
        let maybe_global = is_ffn_activation
            .then(|| self.global_data_trackers.get(&data_id))
            .flatten();
        let tracking_accumulator = if let Some(global_accumulator) = maybe_global {
            global_accumulator
        } else {
            self.intermediate_data_trackers
                .get(&(node_id, data_id.clone()))
                .unwrap()
        };

        let (min, max) = tracking_accumulator.range();
        let name: String = data_id.clone().into();
        let is_qwen_hybrid = self
            .llm_metadata
            .transformers
            .iter()
            .any(|block| block.linear_attention.is_some());
        if is_ffn_activation && is_qwen_hybrid && matches!(name.as_str(), "SILU_IN" | "SILU_OUT") {
            // A shared 16-bit input table preserves gate resolution without
            // changing its arithmetic output range. At 12 global bits the
            // fused up projection allows 16 output bits; at 14 it allows 12.
            // The output interval covers the *entire* symmetric input table,
            // since |SiLU(x)| <= |x|, including an unobserved positive tail.
            let input_range = self.global_data_trackers[&"SILU_IN".to_string().into()].range();
            let maximum = min
                .abs()
                .max(max.abs())
                .max(input_range.0.abs())
                .max(input_range.1.abs())
                .max(1e-12);
            let bits = if name == "SILU_IN" {
                SHIFT_CHECK_TABLE_BIT_SIZE
            } else {
                qwen_ffn_output_table_bits(
                    self.llm_metadata.config.embedding_size,
                    *quantization::BIT_LEN,
                )
                .expect("Qwen FFN lookup has no valid signed integer precision budget")
            };
            return ScalingFactor::from_absolute_max(
                maximum,
                Some((-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)),
            );
        }
        if matches!(
            name.as_str(),
            "DELTA_STATE" | "DELTA_MEMORY" | "DELTA_DELTA"
        ) {
            return ScalingFactor::from_absolute_max(
                min.abs().max(max.abs()).max(1e-8),
                Some((-(1 << 23), (1 << 23) - 1)),
            );
        }
        // This determines how many bits we have to allow for for additions in the following linear layer
        if maybe_global.is_some() {
            let block = self
                .llm_metadata
                .transformers
                .iter()
                .find(|b| b.ffn.activation_id == node_id)
                .unwrap();
            let uses_glu = block.ffn.uses_glu;

            let contraction_size = ceil_log2(self.llm_metadata.config.intermediate_size);
            let max_bits = 63usize;
            let available_bits = if !uses_glu {
                max_bits - contraction_size - *quantization::BIT_LEN - 1 - FIXED_POINT_SCALE
            } else {
                max_bits - contraction_size - 2 * *quantization::BIT_LEN - FIXED_POINT_SCALE
            };
            let bit_size = (available_bits - 1).min(SHIFT_CHECK_TABLE_BIT_SIZE - 1);
            ScalingFactor::from_absolute_max(
                min.abs().max(max.abs()),
                Some((-1 << bit_size, (1 << bit_size) - 1)),
            )
        } else {
            ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None)
        }
    }
}

/// A runner used apply an inference tracker.
pub struct LLMTrackerRunner<'a, I> {
    pub inner: I,
    pub tracker: &'a mut LLMInferenceTracker,
}

impl<'a, I, N> LayerRunner<N, RunInput<N>> for LLMTrackerRunner<'a, I>
where
    I: LayerRunner<N, RunInput<N>>,
    N: TensorTypeParam,
{
    fn model_inputs(
        &mut self,
        graph: &ModelGraph<N>,
        inputs: &[TensorHandle<N>],
    ) -> anyhow::Result<()> {
        for (i, handle) in inputs.iter().enumerate() {
            let input_node_id = graph.input_node_id(i)?;
            let port = 0;
            self.tracker
                .track_inputs(input_node_id.output_at(port), handle)?;
        }
        self.inner.model_inputs(graph, inputs)?;
        Ok(())
    }

    fn run_layer(
        &mut self,
        node_id: NodeId,
        graph: &ModelGraph<N>,
        layer: &Layer<N>,
        inputs: &RunInput<N>,
    ) -> RunResult<N>
    where
        Layer<N>: Evaluate<N>,
    {
        let mut layer_out = self.inner.run_layer(node_id, graph, layer, inputs)?;

        let out_ports = graph.outgoing_ports(node_id);
        for (port, handle) in out_ports.iter().zip(&layer_out.outputs) {
            self.tracker
                .track(node_id.output_at(port.port), layer, handle)?;
        }

        for (data_id, handle) in layer_out.tracked_data.drain() {
            self.tracker
                .track_intermediate_data(node_id, data_id, layer, handle);
        }

        Ok(layer_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        layers::activation::Activation,
        lookup::table::Table,
        parser::llm::{
            LLMConfig,
            metadata::{FFNGraphID, LinearAttentionMetadata, TransformerMetadata},
        },
    };

    fn metadata() -> LLMMetadata {
        let unused = NodeId::from(0);
        LLMMetadata {
            additional_eos_tokens: Vec::new(),
            config: LLMConfig {
                model_name: "qwen35".into(),
                embedding_size: 8,
                hidden_size: 8,
                intermediate_size: 16,
                num_heads: 2,
                head_size: 4,
                num_block: 1,
                context_length: 8,
                norm_epsilon: 1e-6,
                vocab_size: 32,
                eos_token: 31usize.into(),
            },
            transformers: Vec::new(),
            embeddings: unused,
            positional: None,
            final_proj: unused,
            logits: unused,
            argmax: unused,
        }
    }

    #[test]
    fn qwen_ffn_lookup_precision_preserves_complete_table_integer_bounds() -> Result<()> {
        for (bits, expected) in [(12, 16), (14, 12)] {
            let output_bits = qwen_ffn_output_table_bits(1024, bits).unwrap();
            assert_eq!(output_bits, expected);
            let up_magnitude_bits = 2 * (bits - 1) + 10;
            let product_bits = (output_bits - 1) + up_magnitude_bits + 1;
            assert_eq!(product_bits + FIXED_POINT_SCALE, 63);
            for maximum in [1.0, 14.0, 80.0] {
                let input = ScalingFactor::from_absolute_max(maximum, Some((-32768, 32767)));
                let output = ScalingFactor::from_absolute_max(
                    maximum,
                    Some((-(1 << (output_bits - 1)), (1 << (output_bits - 1)) - 1)),
                );
                let table = Table::new_silu(input.scale().recip(), output.scale().recip(), 16);
                let mut maximum_integer = 0u64;
                for x in table.min_input_value()..=table.max_input_value() {
                    let value = table.lookup(x)?;
                    assert!((output.domain().0..=output.domain().1).contains(&value));
                    maximum_integer = maximum_integer.max(value.unsigned_abs());
                }
                let raw_bound = i128::from(maximum_integer) * (1i128 << up_magnitude_bits);
                // Even the largest normalized multiplier (nearly 2^16) fits;
                // the selected signed-bit budget retains the sign-bit margin.
                assert!(raw_bound * ((1 << 16) - 1) < i128::from(i64::MAX));
            }
        }
        assert_eq!(qwen_ffn_output_table_bits(0, 12), None);
        assert_eq!(qwen_ffn_output_table_bits(1024, 32), None);
        Ok(())
    }

    #[test]
    fn qwen_ffn_lookup_covers_unobserved_positive_silu_tail() {
        let mut metadata = metadata();
        metadata.config.embedding_size = 1024;
        let unused = NodeId::from(0);
        let activation_id = NodeId::from(1);
        metadata.transformers.push(TransformerMetadata {
            norm_id: unused,
            transformer: None,
            linear_attention: Some(LinearAttentionMetadata {
                qkv_ids: [unused; 3],
                convolution_ids: [unused; 3],
                recurrence_id: unused,
                final_proj_id: unused,
            }),
            ffn: FFNGraphID {
                up_id: unused,
                down_id: unused,
                activation_id,
                uses_glu: true,
            },
        });
        let mut tracker = LLMInferenceTracker::new(metadata);
        let activation = Layer::Activation(Activation::<f32>::new_silu());
        for (id, values) in [
            ("SILU_IN", vec![-80.0, 1.0]),
            ("SILU_OUT", vec![-0.2, 0.73]),
        ] {
            tracker.track_intermediate_data(
                activation_id,
                id.to_string().into(),
                &activation,
                WrappedTensor::try_from(values).unwrap(),
            );
        }
        let input = tracker
            .scaling_factor_for_intermediate_data(activation_id, "SILU_IN".to_string().into());
        let output = tracker
            .scaling_factor_for_intermediate_data(activation_id, "SILU_OUT".to_string().into());
        assert_eq!(input.bit_size() + 1, 16);
        assert_eq!(
            output.bit_size() + 1,
            qwen_ffn_output_table_bits(1024, *quantization::BIT_LEN).unwrap()
        );
        assert_eq!(input.max(), 80.0);
        assert_eq!(output.max(), 80.0);
    }

    #[test]
    fn mixer_activations_use_local_statistics_and_delta_states_keep_precision() {
        let mut tracker = LLMInferenceTracker::new(metadata());
        let activation = Layer::Activation(Activation::<f32>::new_silu());
        let first = NodeId::from(1);
        let second = NodeId::from(2);
        for (node, magnitude) in [(first, 1.0f32), (second, 100.0)] {
            tracker.track_intermediate_data(
                node,
                "SILU_IN".to_owned().into(),
                &activation,
                WrappedTensor::try_from(vec![-magnitude, magnitude]).unwrap(),
            );
        }
        let small =
            tracker.scaling_factor_for_intermediate_data(first, "SILU_IN".to_owned().into());
        let large =
            tracker.scaling_factor_for_intermediate_data(second, "SILU_IN".to_owned().into());
        assert!((large.scale() / small.scale() - 100.0).abs() < 1e-4);
        tracker.track_intermediate_data(
            first,
            "DELTA_STATE".to_owned().into(),
            &activation,
            WrappedTensor::try_from(vec![0.0f32]).unwrap(),
        );
        let state =
            tracker.scaling_factor_for_intermediate_data(first, "DELTA_STATE".to_owned().into());
        assert_eq!(state.bit_size(), 23);
        assert!(state.scale().is_finite());
    }
}
