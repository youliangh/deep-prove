//! LLM quantisation strategy for Deep Prove.

use super::*;
use crate::{
    layers::requant::FIXED_POINT_SCALE,
    parser::llm::{
        HFTokenizer, LLMTokenizer, Token, metadata::LLMMetadata, tokenizer::TokenizerLoader,
    },
    quantization::strategy::llm_tracker::{LLMInferenceTracker, LLMTrackerRunner},
};

use dp_crypto::util::ceil_log2;
use serde::{Deserialize, Serialize};

pub const fn get_default_prompt() -> &'static str {
    "Artificial intelligence and machine learning have revolutionized numerous industries by enabling computers to perform 
    tasks that traditionally required human intelligence, such as natural language processing, computer vision, predictive analytics, 
    and autonomous decision-making, while simultaneously creating new opportunities for innovation in healthcare through diagnostic imaging 
    and drug discovery, in finance through algorithmic trading and fraud detection, in transportation through self-driving vehicles and route 
    optimization, in entertainment through recommendation systems and content generation, in education through personalized learning platforms 
    and automated grading systems, in manufacturing through predictive maintenance and quality control, in agriculture through precision farming 
    and crop monitoring, in cybersecurity through threat detection and response automation, in retail through inventory management and customer 
    behavior analysis, and in scientific research through data analysis and hypothesis generation, thereby demonstrating the transformative potential 
    of these technologies to augment human capabilities, improve efficiency, reduce costs, enhance accuracy, and solve complex problems across 
    diverse domains while also raising important questions about ethical considerations, privacy protection, job displacement, algorithmic bias, 
    transparency, accountability, and the need for responsible development and deployment of AI systems that benefit society as a whole rather 
    than exacerbating existing inequalities or creating new risks."
}
/// Quantization strategy that observes the inference of the model with different inputs and uses the
/// min/max values of the output to determine the output scaling factor of each layer that needs
/// requantization afterwards.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMInferenceObserver {
    quantising_prompt: Option<String>,
    /// Fixed token sequences, kept separate from quality-evaluation inputs.
    #[serde(default)]
    representative_tokens: Option<Vec<Vec<Token>>>,
    tokeniser: HFTokenizer,
    llm_metadata: LLMMetadata,
}

fn activation_output_bits(intermediate_size: usize, bit_len: usize) -> Option<usize> {
    if intermediate_size == 0 || bit_len == 0 {
        return None;
    }
    let magnitude_bits = 63usize
        .checked_sub(ceil_log2(intermediate_size))?
        .checked_sub(bit_len)?
        .checked_sub(FIXED_POINT_SCALE)?;
    let output_bits = (magnitude_bits + 1) / bit_len * bit_len;
    (output_bits > 0).then_some(output_bits)
}

impl LLMInferenceObserver {
    pub fn new<Model, R>(
        model: Model,
        raw: &R,
        llm_metadata: LLMMetadata,
        prompt: Option<&str>,
    ) -> Self
    where
        Model: TokenizerLoader<R>,
    {
        Self {
            quantising_prompt: prompt.map(|s| s.to_owned()),
            representative_tokens: None,
            tokeniser: model.load_tokenizer(raw).expect("Failed to load tokenizer"),
            llm_metadata,
        }
    }

    pub fn new_with_tokeniser(
        tokeniser: HFTokenizer,
        llm_metadata: LLMMetadata,
        prompt: Option<&str>,
    ) -> Self {
        Self {
            quantising_prompt: prompt.map(|s| s.to_owned()),
            representative_tokens: None,
            tokeniser,
            llm_metadata,
        }
    }

    /// Accumulate calibration statistics across explicitly selected sequences.
    pub fn with_representative_tokens(mut self, samples: Vec<Vec<Token>>) -> Result<Self> {
        anyhow::ensure!(!samples.is_empty(), "Calibration samples must not be empty");
        for sample in &samples {
            anyhow::ensure!(!sample.is_empty(), "Calibration sequence must not be empty");
            anyhow::ensure!(
                sample
                    .iter()
                    .all(|token| usize::from(*token) < self.llm_metadata.config.vocab_size),
                "Calibration token is outside the vocabulary"
            );
        }
        self.representative_tokens = Some(samples);
        Ok(self)
    }

    pub fn make_representative_input(&self, max_context_length: usize) -> Vec<Token> {
        let prompt = self
            .quantising_prompt
            .as_deref()
            .unwrap_or(get_default_prompt());
        let mut input_ids = self.tokeniser.tokenize(prompt);
        if input_ids.len() >= max_context_length {
            input_ids.truncate(max_context_length);
        } else {
            input_ids = input_ids
                .into_iter()
                .cycle()
                .take(max_context_length)
                .collect::<Vec<Token>>();
        }
        input_ids
    }

    pub fn get_representative_input_length(&self) -> usize {
        let prompt = self
            .quantising_prompt
            .as_deref()
            .unwrap_or(get_default_prompt());
        let token_ids = self.tokeniser.tokenize(prompt);
        token_ids.len()
    }
}

impl ScalingStrategy for LLMInferenceObserver {
    type AuxData = LLMInferenceTracker;

    fn quantize(
        &self,
        model: Model<f32>,
        store: &mut GenStore,
    ) -> Result<(Model<Element>, ModelMetadata)> {
        let mut tracker = LLMInferenceTracker::new(self.llm_metadata.clone());

        let unpadded_input_shapes = model.input_shapes();
        let context_length = unpadded_input_shapes[0].numel();
        let input_node_id = model.graph().input_node_id(0)?;
        let samples = self
            .representative_tokens
            .clone()
            .unwrap_or_else(|| vec![self.make_representative_input(context_length)]);
        let sample_count = samples.len();
        for (index, tokens) in samples.into_iter().enumerate() {
            tracing::info!(
                sample = index + 1,
                sample_count,
                tokens = tokens.len(),
                "Calibrating LLM"
            );
            anyhow::ensure!(
                tokens.len() <= context_length,
                "Calibration sequence exceeds model context"
            );
            model.reset();
            let input_tensor = Tensor::<f32>::new(
                vec![tokens.len()].into(),
                tokens
                    .iter()
                    .map(|t| t.as_tensor_type_param::<f32>())
                    .collect(),
            )?;
            let handle = TensorHandle::from_tensor(
                input_node_id.output_at(0).to_storage_key(),
                store.clone(),
                input_tensor,
            );
            let runner = BaseRunner::from(store.clone());
            let runner = LLMTrackerRunner {
                inner: runner,
                tracker: &mut tracker,
            };
            #[cfg(test)]
            let runner = SanityCheckRunner { inner: runner };
            let mut runner = HandleLifetimeRunner::new(runner, model.graph());
            model.run_with_runner(&mut runner, vec![handle])?;
            tracing::info!(
                sample = index + 1,
                sample_count,
                "Calibration sample complete"
            );
        }
        model.reset();

        // Get the scaling factor of the input
        let num_model_inputs = unpadded_input_shapes.len();
        let input_scaling = (0..num_model_inputs)
            .map(|i| {
                let input_node_id = model.graph().input_node_id(i)?;
                let (input_min, input_max) =
                    tracker.scaling_range(input_node_id.output_at(0)).unwrap();
                Ok(ScalingFactor::from_absolute_max(
                    input_min.abs().max(input_max.abs()),
                    None,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        quantize_model::<LLMInferenceObserver>(model, tracker, input_scaling)
    }

    fn scaling_factors_for_node(
        tracker: &LLMInferenceTracker,
        node_id: NodeId,
        num_outputs: usize,
    ) -> Vec<ScalingFactor> {
        let llm_metadata = tracker.llm_metadata();
        if let Some(positional_id) = llm_metadata.positional
            && positional_id == node_id
        {
            // In this case we want to add every row of the positional embedding matrix to the embedding matrix and find the max value
            let embeddings_max = tracker.calculate_embeddings_max().unwrap();
            let shift = 23;
            let domain_min = -1 << shift;
            let domain_max = (1 << shift) - 1;
            let sf =
                ScalingFactor::from_absolute_max(embeddings_max, Some((domain_min, domain_max)));
            return vec![sf];
        }

        let is_activation = llm_metadata
            .transformers
            .iter()
            .any(|block| block.ffn.activation_id == node_id);
        (0..num_outputs)
            .map(|i| {
                let node_output = node_id.output_at(i);
                let (min, max) = tracker.scaling_range(node_output).unwrap();
                if is_activation {
                    // Activation nodes are not Add nodes. Select this branch first so
                    // GLU outliers do not force all ordinary activations onto a coarse
                    // BIT_LEN grid. Requant decomposes its signed output into BIT_LEN
                    // chunks; a 14/17-bit output would fail its domain assertion.
                    // For Llama3-1B: 24 output bits give 23 + 12 + 13 + 15 = 63
                    // bits through the following down projection and fixed multiplier.
                    let output_bits = activation_output_bits(
                        llm_metadata.config.intermediate_size,
                        *quantization::BIT_LEN,
                    ).expect("FFN contraction and quantization width leave no valid activation output chunk within i64");
                    let bit_size = output_bits - 1;
                    ScalingFactor::from_absolute_max(
                        min.abs().max(max.abs()),
                        Some((-1 << bit_size, (1 << bit_size) - 1)),
                    )
                } else if !tracker.is_add_layer(node_output) {
                    ScalingFactor::from_absolute_max(min.abs().max(max.abs()), None)
                } else {
                    // In this case we are in an Add layer, so we know its the residual that we wish to keep in higher precision.
                    // We pick the domain size to be the largest multiple of quantization::BIT_LEN that is less than or equal to 24.
                    let mut multiple = 1usize;
                    while (multiple + 1) * *quantization::BIT_LEN <= 24 {
                        multiple += 1;
                    }
                    let multiple_bit_size = multiple * *quantization::BIT_LEN;
                    let domain_min = -1 << (multiple_bit_size - 1);
                    let domain_max = (1 << (multiple_bit_size - 1)) - 1;
                    ScalingFactor::from_absolute_max(
                        min.abs().max(max.abs()),
                        Some((domain_min, domain_max)),
                    )
                }
            })
            .collect()
    }

    fn scaling_factor_for_intermediate_data(
        tracker: &LLMInferenceTracker,
        node_id: NodeId,
        data_id: TrackedDataId,
    ) -> ScalingFactor {
        tracker.scaling_factor_for_intermediate_data(node_id, data_id)
    }

    fn name(&self) -> String {
        "LLMInferenceObserver".to_string()
    }
}

#[cfg(test)]
mod activation_precision_tests {
    use super::*;
    use crate::{
        layers::{
            Layer,
            activation::{Activation, ActivationLayer},
            add::Add,
            provable::Evaluate,
            requant::Requant,
        },
        parser::llm::{
            LLMConfig,
            metadata::{AttentionMetadata, FFNGraphID, TransformerMetadata},
        },
    };

    #[test]
    fn activation_budget_handles_widths_and_impossible_contractions() {
        for (intermediate, bits, expected) in [
            (1, 12, 36),
            (8192, 8, 24),
            (8192, 10, 20),
            (8192, 12, 24),
            (8192, 14, 14),
            (8192, 16, 16),
            (1 << 25, 12, 12),
        ] {
            let output = activation_output_bits(intermediate, bits).unwrap();
            assert_eq!(output, expected);
            assert_eq!(output % bits, 0);
            assert!(output - 1 + bits + ceil_log2(intermediate) + FIXED_POINT_SCALE <= 63);
        }
        for (intermediate, bits) in [
            (0, 12),
            (8192, 0),
            (8192, 64),
            (1 << 26, 12),
            (usize::MAX, 12),
        ] {
            assert_eq!(activation_output_bits(intermediate, bits), None);
        }
    }

    #[test]
    fn glu_outliers_preserve_small_values_with_valid_requant_chunks() -> Result<()> {
        assert_eq!(
            *quantization::BIT_LEN,
            12,
            "This regression uses the default Llama3 quantization"
        );
        let activation_id = NodeId::from(14);
        let unused = NodeId::from(0);
        let metadata = LLMMetadata {
            additional_eos_tokens: Vec::new(),
            config: LLMConfig {
                model_name: "llama3".into(),
                embedding_size: 2048,
                hidden_size: 2048,
                intermediate_size: 8192,
                num_heads: 32,
                head_size: 64,
                num_block: 1,
                context_length: 256,
                norm_epsilon: 1e-5,
                vocab_size: 128256,
                eos_token: Token::from(128001usize),
            },
            transformers: vec![TransformerMetadata {
                norm_id: unused,
                transformer: Some(AttentionMetadata {
                    qkv_id: unused,
                    qkt_id: unused,
                    mask_id: unused,
                    softmax_id: unused,
                    residual_id: unused,
                    final_proj_id: unused,
                }),
                linear_attention: None,
                ffn: FFNGraphID {
                    up_id: unused,
                    down_id: unused,
                    activation_id,
                    uses_glu: true,
                },
            }],
            embeddings: unused,
            positional: None,
            final_proj: unused,
            logits: unused,
            argmax: unused,
        };
        let mut tracker = LLMInferenceTracker::new(metadata);
        // The observed block-1 GLU maximum is 623.644. At 12 bits, a typical
        // 0.125 activation rounds to zero even though it affects the down projection.
        let values = vec![0.125f32, -0.125, 623.64435, -623.64435];
        let handle = TensorHandle::from_wrapped_tensor(
            "glu_precision_calibration".into(),
            GenStore::default(),
            WrappedTensor::try_from(values.clone())?,
        );
        let layer = Layer::Activation(Activation::new_for_glu(ActivationLayer::Silu(
            None,
            std::marker::PhantomData,
        )));
        tracker.track(activation_id.output_at(0), &layer, &handle)?;
        let scaling = LLMInferenceObserver::scaling_factors_for_node(&tracker, activation_id, 1)[0];
        assert_eq!(scaling.domain(), (-(1 << 23), (1 << 23) - 1));
        assert!((scaling.bit_size() + 1).is_multiple_of(*quantization::BIT_LEN));
        let old_scaling = ScalingFactor::from_absolute_max(623.64435, None);
        assert_eq!(old_scaling.quantize(&values[0]), 0);
        assert!((scaling.dequantize(&scaling.quantize(&values[0])) - values[0]).abs() < 0.0001);

        // Exercise the actual integer Requant evaluator with two output chunks.
        let target = values
            .iter()
            .map(|value| scaling.quantize(value))
            .collect::<Vec<_>>();
        let input =
            WrappedTensor::try_from(target.iter().map(|value| value * 2).collect::<Vec<_>>())?;
        let requant = Requant::from_multiplier(0.5, 44, scaling);
        let result = requant.evaluate(&[&input])?.outputs.remove(0).get_data();
        assert_eq!(result, target);

        // Unrelated residual Add precision stays at its established 24 bits.
        let residual_id = NodeId::from(16);
        tracker.track(residual_id.output_at(0), &Layer::Add(Add::new()), &handle)?;
        let residual = LLMInferenceObserver::scaling_factors_for_node(&tracker, residual_id, 1)[0];
        assert_eq!(residual.domain(), scaling.domain());

        let down_intermediate_bits = scaling.bit_size() + 12 + ceil_log2(8192);
        assert_eq!(down_intermediate_bits + FIXED_POINT_SCALE, 63);
        // All 16 measured down-to-residual multipliers are <1 (max <0.001).
        // Their normalized fixed multipliers are <=2^15, including rounding.
        let down_requant = Requant::from_multiplier(0.0009947564, down_intermediate_bits, residual);
        assert!(
            down_requant.activation_lookup_data.fixed_point_multiplier <= 1 << FIXED_POINT_SCALE
        );
        let worst_sum = 1i128 << (scaling.bit_size() + 11 + ceil_log2(8192));
        assert!(
            worst_sum * i128::from(down_requant.activation_lookup_data.fixed_point_multiplier)
                <= i128::from(i64::MAX)
        );
        Ok(())
    }
}
