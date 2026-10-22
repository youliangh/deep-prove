//! A LLM driver runs the model on a given input and can inspect the output of each layer
//! and the output of the model. It can decide to re-run the model on a different input,
//! to modify the inference trace, to modify the model, etc.
//! The main usage of a driver for now is to run the LLM forward loop until a specific token or
//! the maximum context length is reached. It will also be used to prepend a system model correctly.

use crate::{
    iop::chunking::LLMChunkingStrategy,
    parser::llm::{HFTokenizer, metadata::LLMMetadata},
    quantization::{LLMInferenceObserver, llm_quant::FPTransformModel},
};
use std::{fmt::Display, ops::Deref};

use crate::{
    Element, IO, Proof, Prover, ProverContext, ScalingStrategy, Shape, Tensor,
    graph::executor::{Executor, SequentialExecutor},
    iop::{
        chunking::{ChunkingStrategy, DefaultChunkingStrategy},
        context::VerifierContext,
        prover_graph::ProverGraphNode,
    },
    layers::{Layer, provable::Evaluate},
    model::{
        BaseRunner, HandleLifetimeRunner, LayerRunner, Model, RunInput, Trace, TrackerRunner,
        tensor_to_handles, trace::SplittedNodesInfo, wrapped_tensor_to_handles,
    },
    number::Number,
    padding::PaddingMode,
    parser::{
        PipelineConfig, default_pipeline_config,
        llm::{Token, models::LLMModelLoader},
        to_quantized,
    },
    quantization::{InferenceObserver, InferenceTracker, ModelMetadata},
    tensor::{TensorHandle, TensorTypeParam, WrappedTensor},
    verify,
};
use anyhow::{Context, ensure};
use ark_ff::PrimeField;
use ark_std::rand::Rng;
use dp_crypto::arkyper::{CommitmentScheme, transcript::blake3::Blake3Transcript};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use tenstore::GenStore;
use tracing::{debug, info, info_span};

#[cfg(test)]
use crate::model::SanityCheckRunner;

pub trait WithMaxContext {
    fn with_max_context(self, max_context: usize) -> Self;
}

/// Number of input rows needed to certify exactly `max_new_tokens` predictions.
/// The final prediction is an output, so it need not be appended to the input.
pub fn generation_context_length(
    prompt_len: usize,
    max_new_tokens: usize,
) -> anyhow::Result<usize> {
    ensure!(prompt_len > 0, "The tokenized prompt must not be empty");
    ensure!(max_new_tokens > 0, "max_new_tokens must be positive");
    prompt_len
        .checked_add(max_new_tokens - 1)
        .context("Requested generation length overflows")
}

/// Extract continuation predictions from unpadded output rows, including EOS.
/// An EOS before the final continuation row is rejected rather than hiding data.
pub fn continuation_tokens(
    output_rows: &[Token],
    prompt_len: usize,
    eos: Token,
) -> anyhow::Result<Vec<Token>> {
    continuation_tokens_with_eos(output_rows, prompt_len, &[eos])
}

pub fn continuation_tokens_with_eos(
    output_rows: &[Token],
    prompt_len: usize,
    eos: &[Token],
) -> anyhow::Result<Vec<Token>> {
    ensure!(!eos.is_empty(), "At least one EOS token is required");
    ensure!(prompt_len > 0, "The tokenized prompt must not be empty");
    ensure!(
        output_rows.len() >= prompt_len,
        "Output is shorter than the prompt"
    );
    let continuation = &output_rows[prompt_len - 1..];
    ensure!(
        !continuation[..continuation.len() - 1]
            .iter()
            .any(|token| eos.contains(token)),
        "Continuation contains tokens after EOS"
    );
    Ok(continuation.to_vec())
}

fn numeric_tokens<N: Number>(values: &[N], vocab_size: usize) -> anyhow::Result<Vec<Token>> {
    values
        .iter()
        .map(|value| {
            let token = value.to_usize();
            ensure!(
                token < vocab_size && *value == N::from_usize(token),
                "Token must be an integer in the model vocabulary"
            );
            Ok(Token::from(token))
        })
        .collect()
}

#[cfg(test)]
fn validate_sequence(
    prompt: &[Token],
    input: &[Token],
    output: &[Token],
    max_context: usize,
    vocab_size: usize,
    eos: Token,
) -> anyhow::Result<Vec<Token>> {
    validate_sequence_with_eos(prompt, input, output, max_context, vocab_size, &[eos])
}

fn validate_sequence_with_eos(
    prompt: &[Token],
    input: &[Token],
    output: &[Token],
    max_context: usize,
    vocab_size: usize,
    eos: &[Token],
) -> anyhow::Result<Vec<Token>> {
    ensure!(
        !eos.is_empty() && eos.iter().all(|token| token.0 < vocab_size),
        "EOS tokens must be nonempty and within the vocabulary"
    );
    ensure!(!prompt.is_empty(), "The tokenized prompt must not be empty");
    ensure!(
        max_context > 0 && input.len() <= max_context,
        "Sequence exceeds the configured context length"
    );
    ensure!(
        input.len() == output.len(),
        "Input/output sequence lengths differ"
    );
    ensure!(
        input.len() >= prompt.len(),
        "Input is shorter than the prompt"
    );
    ensure!(
        prompt
            .iter()
            .chain(input)
            .chain(output)
            .all(|t| t.0 < vocab_size),
        "Token is outside the model vocabulary"
    );
    ensure!(
        input[..prompt.len()] == *prompt,
        "Public prompt does not match model input"
    );
    for i in prompt.len()..input.len() {
        ensure!(
            input[i] == output[i - 1],
            "Input token at position {i} is not the preceding output prediction"
        );
    }
    let continuation = continuation_tokens_with_eos(output, prompt.len(), eos)?;
    ensure!(
        input.len() == max_context || continuation.last().is_some_and(|token| eos.contains(token)),
        "Generation ended before its token limit without EOS"
    );
    Ok(continuation)
}

#[cfg(test)]
fn validate_llm_io<F: PrimeField>(
    io: &IO<F>,
    prompt: &[Token],
    max_context: usize,
    vocab_size: usize,
    eos: Token,
) -> anyhow::Result<Vec<Token>> {
    validate_llm_io_with_eos(io, prompt, max_context, vocab_size, &[eos])
}

fn validate_llm_io_with_eos<F: PrimeField>(
    io: &IO<F>,
    prompt: &[Token],
    max_context: usize,
    vocab_size: usize,
    eos: &[Token],
) -> anyhow::Result<Vec<Token>> {
    ensure!(
        io.input.len() == 1 && io.output.len() == 1,
        "Expected one input and one output tensor"
    );
    let vocab_bound = F::BigInt::from(u64::try_from(vocab_size)?);
    let mut tokens = Vec::with_capacity(2);
    for (is_input, tensor) in [(true, &io.input[0]), (false, &io.output[0])] {
        tensor.validate_shape()?;
        let shape = tensor.shape();
        ensure!(
            shape.rank() == 1 || (!is_input && shape.rank() == 2 && shape.dim(1) == 1),
            "Expected a token vector or a single-column output tensor"
        );
        let rows = tensor.unpadded_shape().dim(0);
        ensure!(
            rows <= max_context,
            "Unpadded sequence exceeds the context length"
        );
        ensure!(
            rows.checked_next_power_of_two() == Some(shape.dim(0)),
            "Token tensor does not have canonical power-of-two padding"
        );
        let values = tensor.data()[..rows]
            .iter()
            .map(|value| {
                // Do not use ToElement on untrusted fields: it panics for large
                // representatives and interprets modular negatives as integers.
                let integer = (**value).into_bigint();
                ensure!(
                    integer < vocab_bound,
                    "Public token is outside the model vocabulary"
                );
                Ok(Token::from(integer.as_ref()[0] as usize))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        ensure!(
            tensor.data()[rows..].iter().all(|value| **value == F::ZERO),
            "Token padding must be zero"
        );
        tokens.push(values);
    }
    io.validate_split_binding()?;
    validate_sequence_with_eos(prompt, &tokens[0], &tokens[1], max_context, vocab_size, eos)
}

/// The main struct responsible to verify an LLM proof.
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct LLMVerifierContext<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    pub verifier_ctx: VerifierContext<F, PCS>,
    pub md: LLMMetadata,
    pub max_context: Option<usize>,
}

impl<F, PCS> WithMaxContext for LLMVerifierContext<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// Sets the `max_context` window for this context.
    fn with_max_context(mut self, max_context: usize) -> Self {
        self.max_context = Some(max_context);
        self
    }
}

impl<'a, F, PCS> WithMaxContext for (ProverContext<'a, F, PCS>, LLMVerifierContext<F, PCS>)
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// Sets the `max_context` window for this context.
    fn with_max_context(self, max_context: usize) -> Self {
        (self.0, self.1.with_max_context(max_context))
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct LLMProof<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    /// The inference proof.
    pub proof: Proof<F, PCS>,

    /// Note the IO contains the _full_ input, e.g. the user input + the generated tokens
    pub io: IO<F>,
}

/// The main struct responsible for generating the trace and the proof related
/// to LLM proving. This requires a wrapper on top of the model to drive the
/// auto regressive loop correctly.
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
pub struct Driver<N>
where
    N: TensorTypeParam,
{
    pub model: Model<N>,
    pub md: LLMMetadata,
    pub(crate) max_context: Option<usize>,
    pub(crate) padding_mode: PaddingMode,
}

impl<N: TensorTypeParam> Driver<N> {
    /// Returns the vocabulary size of the model
    pub fn vocab_size(&self) -> usize {
        self.md.vocab_size()
    }
}

impl Driver<f32> {
    /// Load and returns the raw model in float precision.
    ///
    /// Loads a model from a gguf, safetensors, or json external file.
    ///
    /// NOTE: the max_context is only there to hack around the creation of Rope
    /// to avoid loading the full matrix if we don't need it. That should be
    /// removed if when we remove this hack.
    pub fn load_from_model<DataFormat: Display, M: LLMModelLoader<DataFormat>>(
        mut model_type: M,
        data_format: &DataFormat,
        max_context: Option<usize>,
    ) -> anyhow::Result<Self> {
        info!(
            "LLM Driver: Loading model {} from {}",
            model_type.model_name(),
            data_format
        );
        if let Some(max_context) = max_context {
            model_type = model_type.with_max_context_length(max_context);
        }
        let (model, md) = model_type.parse(data_format)?;
        Ok(Self {
            model,
            md,
            max_context,
            padding_mode: PaddingMode::NoPadding,
        })
    }

    /// Quantizes and pads this model.
    ///
    /// The result can be serialized and deserialized at will to serve
    /// inference+proving for this model.
    pub fn into_provable_llm<'a>(
        self,
        mut pipeline_config: Option<PipelineConfig<'a, InferenceObserver>>,
    ) -> anyhow::Result<(Driver<Element>, ModelMetadata)> {
        let numel = self.max_context.unwrap_or(self.md.context_length());
        let n_inputs = 1;
        let representative_inputs = (0..n_inputs)
            .map(|_| {
                self.random_sequence(numel)
                    .into_iter()
                    .map(|t| t.as_number::<f32>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let quantization_strategy =
            InferenceObserver::new_with_representative_input(vec![representative_inputs]);
        let mut conf = pipeline_config.take().unwrap_or_else(|| {
            // override the shapes to adhere to the expected input shape of the representative inputs
            default_pipeline_config()
        });

        if conf.input_shapes.is_none() {
            conf = conf.with_input_shapes(vec![Shape::from(vec![numel])]);
        }
        if conf.quant_strategy.is_none() {
            conf = conf.with_strategy(quantization_strategy);
        }
        let (mut quantized_model, metadata) = to_quantized(self.model, conf)?;
        // just set to one because we run one token after another to derive the full trace.
        quantized_model.input_shapes = vec![Shape::from(vec![1])];
        Ok((
            Driver {
                model: quantized_model,
                md: self.md,
                max_context: self.max_context,
                padding_mode: PaddingMode::Padding,
            },
            metadata,
        ))
    }

    /// Quantizes and pads this model using a specific pipeline configuration.
    ///
    /// The result can be serialized and deserialized at will to serve
    /// inference+proving for this model.
    pub fn into_provable_llm_with_strategy<'a, S: ScalingStrategy>(
        self,
        mut pipeline_config: PipelineConfig<'a, S>,
    ) -> anyhow::Result<(Driver<Element>, ModelMetadata)> {
        let numel = self.max_context.unwrap_or(self.md.context_length());

        if pipeline_config.input_shapes.is_none() {
            pipeline_config = pipeline_config.with_input_shapes(vec![Shape::from(vec![numel])]);
        }

        let (mut quantized_model, metadata) = to_quantized(self.model, pipeline_config)?;
        // just set to one because we run one token after another to derive the full trace.
        quantized_model.input_shapes = vec![Shape::from(vec![1])];
        Ok((
            Driver {
                model: quantized_model,
                md: self.md,
                max_context: self.max_context,
                padding_mode: PaddingMode::Padding,
            },
            metadata,
        ))
    }

    pub fn into_provable_llm_with_transform<M: FPTransformModel>(
        self,
        tokeniser: &HFTokenizer,
    ) -> anyhow::Result<(Driver<Element>, ModelMetadata)> {
        let max_context = self.max_context;
        let md = self.md.clone();

        let llm_inference_observer =
            LLMInferenceObserver::new_with_tokeniser(tokeniser.clone(), self.md.clone(), None);

        let representative_input_length = if let Some(length) = max_context {
            length
        } else {
            llm_inference_observer.get_representative_input_length()
        };

        let pipeline_config = PipelineConfig::<LLMInferenceObserver>::new()
            .with_input_shapes(vec![Shape::from(vec![representative_input_length])])
            .with_strategy(llm_inference_observer);

        let adapted_driver = M::adapt_model(self)?;
        let (mut quantized_model, metadata) = to_quantized(adapted_driver.model, pipeline_config)?;
        // just set to one because we run one token after another to derive the full trace.
        quantized_model.input_shapes = vec![Shape::from(vec![1])];
        Ok((
            Driver {
                model: quantized_model,
                md,
                max_context,
                padding_mode: PaddingMode::Padding,
            },
            metadata,
        ))
    }
}

impl<N> Driver<N>
where
    N: TensorTypeParam,
    Layer<N>: Evaluate<N>,
{
    /// Returns the current `max_context` window, if set.
    pub fn max_context(&self) -> Option<usize> {
        self.max_context
    }

    /// Sets the `max_context` window for this driver.
    pub fn with_max_context(&mut self, max_context: usize) {
        self.max_context = Some(max_context);
    }

    /// Returns a sequence of `len` random tokens.
    pub fn random_sequence(&self, len: usize) -> Vec<Token> {
        let mut rng = crate::rng_from_env_or_random();
        (0..len)
            .map(|_| Token::from(rng.gen_range(0..self.md.vocab_size())))
            .collect()
    }

    /// Converts a slice of [Token]s into a [Tensor] to be used with this driver.
    ///
    /// # Errors
    ///
    /// If the number of tokens exceeds the current configured `context_length`.
    pub fn tokens_to_tensor(&self, input: &[Token]) -> anyhow::Result<Tensor<N>> {
        ensure!(
            !input.is_empty() && input.len() <= self.md.context_length(),
            "Input sequence must be nonempty and fit the model context length",
        );
        ensure!(
            input.iter().all(|token| token.0 < self.md.vocab_size()),
            "Input token is outside the model vocabulary"
        );
        let input_tokens = input
            .iter()
            .map(|t| t.as_tensor_type_param::<N>())
            .collect::<Vec<_>>();
        Tensor::new(Shape::from([input_tokens.len()]), input_tokens)
    }

    /// Returns the strategy employed to chunk the LLM in case of distributed proving.
    pub fn chunking_strategy<F: PrimeField, PCS: CommitmentScheme<Field = F>>(
        &self,
        input_tensor: &Tensor<N>,
        ctx: &ProverContext<F, PCS>,
    ) -> anyhow::Result<LLMChunkingStrategy> {
        // check that the input tensor size is smaller than max context
        let max_window = self.max_context.unwrap_or(self.md.context_length());
        let num_input_tokens = input_tensor.data().len();
        ensure!(
            num_input_tokens <= max_window,
            "Input sequence length greater than the context length",
        );
        Ok(LLMChunkingStrategy::new(
            max_window, /* we always use the max context length to be safe, as before running inference we don't know yet the actual input length for the prover */
            &ctx.model_ctx,
        ))
    }

    /// Performs a single run of the model, returning the produced trace.
    pub fn run(&self, inputs: Vec<Tensor<N>>, store: &mut GenStore) -> anyhow::Result<Trace<N>> {
        let result = self.model.run(inputs, store);
        self.model.reset();
        result
    }

    /// Run a single iteration of the model using the provided `runner`.
    pub fn run_with_runner<R>(
        &self,
        runner: &mut R,
        inputs: Vec<TensorHandle<N>>,
    ) -> anyhow::Result<()>
    where
        R: LayerRunner<N, ()>,
    {
        let result = self.model.run_with_runner(runner, inputs);
        self.model.reset();
        result
    }

    pub(crate) fn run_elements_with_split_info(
        &self,
        input_tensor: Tensor<N>,
        store: &mut GenStore,
        split_node_info: Option<&SplittedNodesInfo>,
    ) -> anyhow::Result<Trace<N>> {
        let runner = BaseRunner::from(store.clone());
        #[cfg(test)]
        let runner = SanityCheckRunner { inner: runner };

        self.run_elements_with_runner(input_tensor, store, runner, split_node_info)
    }

    /// Generate a sentence.
    ///
    /// Notes:
    /// - The sentence is limited by the EOS token or the context length.
    /// - The returned trace contains the whole sequence.
    /// - The layer runner will be used for each token infernece, but not for
    ///   the trace generation.
    pub fn run_elements(
        &self,
        input_tensor: Tensor<N>,
        store: &mut GenStore,
    ) -> anyhow::Result<Trace<N>> {
        self.run_elements_with_split_info(input_tensor, store, None)
    }

    /// Generate at most `max_new_tokens` tokens, stopping at EOS.
    ///
    /// Updates the driver's certification context length. Use that same length
    /// in the verifier context; obtain it with [`Driver::max_context`].
    pub fn generate(
        &mut self,
        prompt: &[Token],
        max_new_tokens: usize,
        store: &mut GenStore,
    ) -> anyhow::Result<Trace<N>> {
        let context = generation_context_length(prompt.len(), max_new_tokens)?;
        ensure!(
            context <= self.md.context_length(),
            "Requested generation exceeds the loaded model context length"
        );
        let input = self.tokens_to_tensor(prompt)?;
        self.max_context = Some(context);
        self.run_elements(input, store)
    }

    /// Read the exact unpadded continuation represented by a generation trace.
    pub fn continuation_tokens(
        &self,
        trace: &Trace<N>,
        prompt_len: usize,
    ) -> anyhow::Result<Vec<Token>> {
        ensure!(trace.outputs().len() == 1, "Expected one model output");
        let output = trace.outputs()[0].tensor()?;
        output.validate_shape()?;
        ensure!(
            output.shape().rank() == 1
                || (output.shape().rank() == 2 && output.shape().dim(1) == 1),
            "Expected one output token per sequence row"
        );
        let length = output.unpadded_shape().dim(0);
        let tokens = numeric_tokens(&output.data()[..length], self.md.vocab_size())?;
        continuation_tokens_with_eos(&tokens, prompt_len, &self.md.eos_tokens())
    }

    /// Generate a sentence.
    ///
    /// Notes:
    /// - The sentence is limited by the EOS token or the context length.
    /// - The returned trace contains the whole sequence.
    pub fn run_elements_with_tracker(
        &self,
        input_tensor: Tensor<N>,
        store: &mut GenStore,
        tracker: &mut InferenceTracker,
    ) -> anyhow::Result<Trace<N>> {
        let runner = BaseRunner::from(store.clone());
        let runner = TrackerRunner {
            inner: runner,
            tracker,
        };
        #[cfg(test)]
        let runner = SanityCheckRunner { inner: runner };

        self.run_elements_with_runner(input_tensor, store, runner, None)
    }

    /// Generate a sentence.
    ///
    /// Notes:
    /// - The sentence is limited by the EOS token or the context length.
    /// - The returned trace contains the whole sequence.
    /// - The layer runner will be used for each token inferenece, but not for
    ///   the trace generation.
    fn run_elements_with_runner<I>(
        &self,
        input_tensor: Tensor<N>,
        store: &mut GenStore,
        runner: I,
        split_node_info: Option<&SplittedNodesInfo>,
    ) -> anyhow::Result<Trace<N>>
    where
        HandleLifetimeRunner<I, N>: LayerRunner<N, ()>,
        I: LayerRunner<N, RunInput<N>>,
    {
        // Reset the model
        self.model.reset();

        input_tensor.validate_shape()?;
        ensure!(
            input_tensor.shape().rank() == 1
                && input_tensor.shape() == input_tensor.unpadded_shape(),
            "Generation requires an unpadded one-dimensional token tensor"
        );
        let input_data = input_tensor.data().to_vec();
        let num_input_tokens = input_data.len();

        let mut runner = HandleLifetimeRunner::new(runner, &self.model.graph);

        let max_window = self.max_context.unwrap_or(self.md.context_length());
        ensure!(
            max_window > 0 && max_window <= self.md.context_length(),
            "Generation context is outside the loaded model context length"
        );
        ensure!(
            num_input_tokens <= max_window,
            "max window {} is smaller than prompt token count {}",
            max_window,
            num_input_tokens,
        );

        let prompt_tokens = numeric_tokens(&input_data, self.md.vocab_size())?;

        // Initialise the full tokens with the input data, generated tokens will be appended.
        let mut full_sentence = input_data;

        // Inference always runs on unpadded tensors. Padding is only needed at the proving layer.
        info!(
            "LLM: running model with initial shape: {:?}",
            input_tensor.shape(),
        );
        let mut input_handles = tensor_to_handles(&[input_tensor], &self.model.graph, store)?;

        let mut generated_tokens = Vec::with_capacity(max_window - num_input_tokens + 1);
        let eos_tokens: Vec<N> = self
            .md
            .eos_tokens()
            .iter()
            .map(Token::as_tensor_type_param)
            .collect();

        // Resolve each prediction before scheduling the next token: EOS must not
        // leave speculative tokens in the softmax/normalisation caches used by
        // the final full-sequence certification pass.
        for unpadded_seq_len in num_input_tokens..=max_window {
            info!(
                "Running iteration {} with input tensor {:?}",
                unpadded_seq_len,
                input_handles[0].shape(),
            );

            self.model
                .run_with_runner(&mut runner, input_handles)
                .with_context(|| {
                    format!(
                        "running the {} iteration loop",
                        unpadded_seq_len - num_input_tokens
                    )
                })?;

            let model_outputs = runner.model_outputs(&self.model.graph)?;
            ensure!(
                model_outputs.len() == 1,
                "expected 1 output, got {}",
                model_outputs.len(),
            );

            let output = &model_outputs[0]
                .wrapped_tensor()
                .context("Output handler should have associated tensor data")?;

            // Take the last token before the padding
            let index = if output.shape().num_elements() == 1 {
                WrappedTensor::try_from(vec![0])?
            } else {
                WrappedTensor::try_from(vec![(unpadded_seq_len - 1) as Element])?
            };

            let last_token_tensor = output.deref().clone().flatten_1d().select(0, index)?;

            let token = last_token_tensor.get_data()[0];
            generated_tokens.push(token);
            if eos_tokens.contains(&token) {
                break;
            }

            input_handles =
                wrapped_tensor_to_handles(&[last_token_tensor], &self.model.graph, store)?;
        }

        ensure!(
            !generated_tokens.is_empty(),
            "Model did not generate a token"
        );
        // The last prediction (including EOS) is certified as an output only.
        full_sentence.extend_from_slice(&generated_tokens[..generated_tokens.len() - 1]);
        // Inference always runs on unpadded tensors. Padding is handled at the proving layer.
        let tensor = Tensor::new(Shape::new(vec![full_sentence.len()]), full_sentence.clone())?;

        // Reset the model and its caches, otherwise a single token is
        // generated.
        self.model
            .reset_selective(&[crate::layers::transformer::softmax::SOFTMAX_LAYER]);

        info!(
            "Running last iteration (heavy) with {} tokens",
            full_sentence.len(),
        );

        // Use the passed store (worker's remote store) so TensorHandle data
        // is accessible during proving in distributed execution
        let trace = self
            .model
            .run_with_split_nodes_info(vec![tensor], store, split_node_info)?;
        let generated_result = (|| {
            ensure!(trace.inputs().len() == 1, "Expected one model input");
            let input = trace.inputs()[0].tensor()?;
            let output_tokens = self.continuation_tokens(&trace, num_input_tokens)?;
            let expected = numeric_tokens(&generated_tokens, self.md.vocab_size())?;
            ensure!(
                output_tokens == expected,
                "Cached generation differs from the full causal certification pass"
            );
            let input_tokens = numeric_tokens(input.data(), self.md.vocab_size())?;
            let output = trace.outputs()[0].tensor()?;
            let all_predictions = numeric_tokens(output.data(), self.md.vocab_size())?;
            validate_sequence_with_eos(
                &prompt_tokens,
                &input_tokens,
                &all_predictions,
                max_window,
                self.md.vocab_size(),
                &self.md.eos_tokens(),
            )?;
            anyhow::Ok(())
        })();

        // Reset the model so it can be reused.
        self.model.reset();
        generated_result?;

        Ok(trace)
    }
}

impl Driver<Element> {
    /// Compute the set of contexts necessary for all the possible input shapes of the LLM.
    /// It returns a `HashMap` which associates a given context to the maximum polynomial size supported
    /// by that context. The proper context to be used for a given input size `input_len` is found in the
    /// entry `HashMap` returned by this method identified by the key `self.get_max_poly_size_for_input(input_len)`
    pub fn compute_all_contexts<F, PCS>(
        &self,
    ) -> anyhow::Result<(ProverContext<'_, F, PCS>, LLMVerifierContext<F, PCS>)>
    where
        F: PrimeField,
        PCS: CommitmentScheme<Field = F>,
    {
        // compute shapes for all possible input sequence lengths
        let max_input_shapes = vec![Shape::new(vec![
            self.md.context_length().next_power_of_two(),
        ])];
        ensure!(
            max_input_shapes.len() == 1,
            "Expected 1 input shape in LLM model, found {}",
            max_input_shapes.len()
        );
        let max_input_length = max_input_shapes[0].numel();
        ensure!(max_input_length.is_power_of_two());

        let (prover_ctx, verifier_ctx) = self
            .model
            .generate_contexts_for_input_shapes(max_input_shapes)?;
        Ok((
            prover_ctx,
            LLMVerifierContext {
                verifier_ctx,
                md: self.md.clone(),
                max_context: self.max_context,
            },
        ))
    }

    /// Create the prover & verifier for a given model
    pub fn context<F, PCS>(
        &self,
    ) -> anyhow::Result<(ProverContext<'static, F, PCS>, LLMVerifierContext<F, PCS>)>
    where
        F: PrimeField,
        PCS: CommitmentScheme<Field = F>,
    {
        debug!(
            "Generating context for model with {} layers",
            self.model.graph.node_count()
        );
        let (prover_ctx, verifier_ctx) = self.model.generate_contexts()?;
        Ok((
            prover_ctx,
            LLMVerifierContext {
                verifier_ctx,
                md: self.md.clone(),
                // The verifier should put itself the max context here
                max_context: None,
            },
        ))
    }

    fn chunked_prove_local<'a, 'b, S, F, PCS, Ex>(
        &'a self,
        ctx: &'b ProverContext<F, PCS>,
        trace: Trace<Element>,
        num_chunks: Option<usize>,
        chunking_strategy: S,
        executor_config: Ex::Config,
    ) -> anyhow::Result<(Proof<F, PCS>, IO<F>)>
    where
        S: ChunkingStrategy,
        F: PrimeField,
        PCS: CommitmentScheme<Field = F> + 'static,
        Ex: Executor<ProverGraphNode<'b, 'a, F, Blake3Transcript, PCS>, usize>,
    {
        info!("Proving the trace");
        let proof = Prover::<F, Blake3Transcript, PCS>::chunked_prove_local::<_, Ex>(
            ctx,
            trace,
            num_chunks,
            chunking_strategy,
            &self.model,
            executor_config,
        )?;
        info!("Proof generated");
        Ok(proof)
    }

    /// Utility method to run `chunked_prove_local` with a predefined executor
    #[cfg(test)]
    pub(crate) fn chunked_prove_default_executor<S, F, PCS>(
        &self,
        ctx: &ProverContext<F, PCS>,
        trace: Trace<Element>,
        num_chunks: Option<usize>,
        chunking_strategy: S,
    ) -> anyhow::Result<(Proof<F, PCS>, IO<F>)>
    where
        S: ChunkingStrategy,
        F: PrimeField,
        PCS: CommitmentScheme<Field = F> + 'static,
    {
        // NOTE: until https://github.com/Plonky3/Plonky3/pull/999 is fixed, we have
        // to use the sequential executor and not the threadpool executor.
        self.chunked_prove_local::<_, _, _, SequentialExecutor>(
            ctx,
            trace,
            num_chunks,
            chunking_strategy,
            (),
        )
    }

    pub fn prove<F, PCS>(
        &self,
        ctx: &ProverContext<F, PCS>,
        trace: Trace<Element>,
    ) -> anyhow::Result<(Proof<F, PCS>, IO<F>)>
    where
        F: PrimeField,
        PCS: CommitmentScheme<Field = F> + 'static,
    {
        let span = info_span!(
            "zkml_llm_prove",
            max_context = self.max_context.unwrap_or_default()
        );
        let _entered = span.enter();
        let chunking_strategy = DefaultChunkingStrategy::from(&trace);
        // Recurrent-state certificates scale as sequence × heads × key × value.
        // Commit/prove bounded groups sequentially; boundary commitments bind
        // every group into the same complete-model statement.
        let chunks = if self.md.transformers.iter().any(|block| block.linear_attention.is_some()) {
            None
        } else {
            Some(1)
        };
        self.chunked_prove_local::<_, _, _, SequentialExecutor>(
            ctx,
            trace,
            chunks,
            chunking_strategy,
            (),
        )
    }
}

impl<F, PCS> LLMVerifierContext<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Validate the public sequence statement and extract its continuation.
    /// This does not verify the cryptographic proof; call [`Self::verify`] before
    /// accepting these tokens as certified model output.
    pub fn continuation_tokens(&self, prompt: &[Token], io: &IO<F>) -> anyhow::Result<Vec<Token>> {
        let max_context = self.max_context.unwrap_or(self.md.context_length());
        ensure!(
            max_context > 0 && max_context <= self.md.context_length(),
            "Verifier context length is outside the loaded model context"
        );
        validate_llm_io_with_eos(
            io,
            prompt,
            max_context,
            self.md.vocab_size(),
            &self.md.eos_tokens(),
        )
    }

    pub fn verify(
        &self,
        proof: Proof<F, PCS>,
        user_input: Vec<Token>,
        inference_output: IO<F>,
    ) -> anyhow::Result<()>
    where
        PCS::Commitment: PartialEq,
    {
        let span = info_span!(
            "zkml_llm_verify",
            max_context = self.max_context.unwrap_or(self.md.config.context_length),
            input_tokens = inference_output.input.len(),
            output_tensors = inference_output.output.len()
        );
        let _entered = span.enter();
        self.continuation_tokens(&user_input, &inference_output)?;
        verify::<_, Blake3Transcript, _>(&self.verifier_ctx, proof, inference_output)?;
        Ok(())
    }
}

#[cfg(test)]
mod generation_tests {
    use super::*;
    use crate::{
        graph::NodeId,
        layers::transformer::{embeddings::Embeddings, logits::Logits},
        parser::llm::LLMConfig,
        tensor::KeyedTensor,
        testing::Pcs,
    };
    use ark_ff::{AdditiveGroup, Field};

    type F = ark_bn254::Fr;

    fn tokens(values: &[usize]) -> Vec<Token> {
        values.iter().copied().map(Token::from).collect()
    }

    #[test]
    fn multiple_eos_tokens_stop_generation_and_bind_public_sequence() -> anyhow::Result<()> {
        let mut driver = tiny_driver()?;
        driver.md.additional_eos_tokens = tokens(&[3]);
        let prompt = tokens(&[0, 1]);
        let trace = driver.generate(&prompt, 6, &mut GenStore::default())?;
        assert_eq!(
            driver.continuation_tokens(&trace, prompt.len())?,
            tokens(&[2, 3])
        );
        let io = public_io(&[0, 1, 2], &[1, 2, 3])?;
        assert_eq!(
            validate_llm_io_with_eos(&io, &prompt, 7, 8, &driver.md.eos_tokens())?,
            tokens(&[2, 3])
        );
        assert!(
            validate_sequence(
                &prompt,
                &tokens(&[0, 1, 2]),
                &tokens(&[1, 2, 3]),
                7,
                8,
                7usize.into()
            )
            .is_err()
        );
        assert!(continuation_tokens_with_eos(&tokens(&[1, 2, 3, 4]), 2, &tokens(&[3, 7])).is_err());
        assert!(
            validate_sequence_with_eos(
                &prompt,
                &tokens(&[0, 1, 2, 3]),
                &tokens(&[1, 2, 3, 4]),
                7,
                8,
                &tokens(&[3, 7])
            )
            .is_err()
        );
        Ok(())
    }

    fn public_io(input: &[usize], output: &[usize]) -> anyhow::Result<IO<F>> {
        let make = |values: &[usize], output: bool| {
            let mut padded = vec![F::ZERO; values.len().max(1).next_power_of_two()];
            for (dst, value) in padded.iter_mut().zip(values) {
                *dst = F::from(*value as u64);
            }
            let shape = if output {
                vec![padded.len(), 1]
            } else {
                vec![padded.len()]
            };
            let actual = if output {
                vec![values.len(), 1]
            } else {
                vec![values.len()]
            };
            Tensor::new_with_unpadded_shape(shape.into(), actual.into(), padded).map(Into::into)
        };
        Ok(IO::new(
            vec![make(input, false)?],
            vec![make(output, true)?],
        ))
    }

    fn check(io: &IO<F>, prompt: &[usize], context: usize) -> anyhow::Result<Vec<Token>> {
        validate_llm_io(io, &tokens(prompt), context, 8, Token::from(7usize))
    }

    fn tiny_driver() -> anyhow::Result<Driver<f32>> {
        let mut model = Model::new_from_input_shapes(vec![vec![1].into()]);
        let mut weights = vec![0.0f32; 64];
        for row in 0..8 {
            weights[row * 8 + (row + 1).min(7)] = 1.0;
        }
        let embedding = model.add_consecutive_layer(
            Layer::Embeddings(Embeddings::new(
                KeyedTensor::new(
                    "generation_test_embeddings",
                    Tensor::new(vec![8, 8].into(), weights)?,
                )
                .into(),
            )?),
            None,
        )?;
        let argmax =
            model.add_consecutive_layer(Layer::Logits(Logits::new_argmax()), Some(embedding))?;
        model.automatic_output_labelling()?;
        Ok(Driver {
            model,
            md: LLMMetadata {
                additional_eos_tokens: Vec::new(),
                config: LLMConfig {
                    model_name: "generation_test".into(),
                    embedding_size: 8,
                    hidden_size: 8,
                    intermediate_size: 8,
                    num_heads: 1,
                    head_size: 8,
                    num_block: 0,
                    context_length: 8,
                    norm_epsilon: 1e-5,
                    vocab_size: 8,
                    eos_token: Token::from(7usize),
                },
                transformers: vec![],
                embeddings: embedding,
                positional: None,
                final_proj: embedding,
                logits: embedding,
                argmax,
            },
            max_context: Some(8),
            padding_mode: PaddingMode::NoPadding,
        })
    }

    #[test]
    fn exact_generation_lengths_and_eos() -> anyhow::Result<()> {
        assert_eq!(generation_context_length(3, 1)?, 3);
        assert_eq!(generation_context_length(3, 4)?, 6);
        assert!(generation_context_length(0, 1).is_err());
        assert!(generation_context_length(1, 0).is_err());
        assert!(generation_context_length(usize::MAX, 2).is_err());
        let mut driver = tiny_driver()?;
        for (prompt, count, expected) in [
            (vec![0, 1], 1, vec![2]),
            (vec![0, 1], 4, vec![2, 3, 4, 5]),
            (vec![0, 6], 4, vec![7]),
            (vec![0, 4], 5, vec![5, 6, 7]),
        ] {
            let trace = driver.generate(&tokens(&prompt), count, &mut GenStore::default())?;
            assert_eq!(
                driver.continuation_tokens(&trace, prompt.len())?,
                tokens(&expected)
            );
            assert_eq!(
                trace.inputs()[0].unpadded_shape().numel(),
                prompt.len() + expected.len() - 1
            );
        }
        assert!(driver.generate(&[], 1, &mut GenStore::default()).is_err());
        assert!(
            driver
                .generate(&tokens(&[0]), 0, &mut GenStore::default())
                .is_err()
        );
        assert!(
            driver
                .generate(&tokens(&[0]), 9, &mut GenStore::default())
                .is_err()
        );
        assert!(driver.tokens_to_tensor(&tokens(&[8])).is_err());
        assert!(numeric_tokens(&[-1i64], 8).is_err());
        assert!(numeric_tokens(&[1.5f32], 8).is_err());
        Ok(())
    }

    #[test]
    fn public_sequence_checks_all_continuations_and_eos_offsets() -> anyhow::Result<()> {
        let valid = public_io(&[0, 1, 2, 3, 4], &[1, 2, 3, 4, 5])?;
        assert_eq!(check(&valid, &[0, 1], 5)?, tokens(&[2, 3, 4, 5]));
        assert!(check(&valid, &[0, 2], 5).is_err());
        assert!(check(&valid, &[], 5).is_err());
        assert!(check(&valid, &[0, 1], 4).is_err());
        assert!(check(&valid, &[0, 1], 6).is_err());
        assert!(check(&public_io(&[0, 1, 2, 6, 4], &[1, 2, 3, 4, 5])?, &[0, 1], 5).is_err());
        assert!(check(&public_io(&[0, 1, 2, 3, 4], &[1, 2, 7, 4, 5])?, &[0, 1], 5).is_err());
        assert!(check(&public_io(&[0, 1], &[1])?, &[0, 1], 5).is_err());
        assert_eq!(
            check(&public_io(&[0, 6], &[1, 7])?, &[0, 6], 5)?,
            tokens(&[7])
        );
        // EOS belongs to absolute output row 3, not continuation-relative row 2.
        assert_eq!(
            check(&public_io(&[0, 4, 5, 6], &[1, 5, 6, 7])?, &[0, 4], 6)?,
            tokens(&[5, 6, 7])
        );
        assert!(check(&public_io(&[0, 4, 2, 6], &[1, 5, 6, 7])?, &[0, 4], 6).is_err());
        assert!(check(&public_io(&[0, 6, 7], &[1, 7, 7])?, &[0, 6], 3).is_err());
        Ok(())
    }

    #[test]
    fn malformed_public_tensors_are_rejected_without_panics() -> anyhow::Result<()> {
        let valid = public_io(&[0, 1, 2], &[1, 2, 3])?;
        let mut changed = valid.clone();
        changed.input.clear();
        assert!(check(&changed, &[0, 1], 3).is_err());
        let mut changed = valid.clone();
        changed.output[0][2] = F::from(8u64).into();
        assert!(check(&changed, &[0, 1], 3).is_err());
        changed.output[0][2] = (-F::ONE).into();
        assert!(check(&changed, &[0, 1], 3).is_err());
        changed.output[0][2] = F::from(1u64 << 63).into();
        assert!(check(&changed, &[0, 1], 3).is_err());
        let mut changed = valid.clone();
        changed.output[0][3] = F::ONE.into();
        assert!(check(&changed, &[0, 1], 3).is_err());
        let mut changed = valid.clone();
        changed.output[0] = Tensor::<F>::new_with_unpadded_shape(
            vec![4, 1].into(),
            vec![5, 1].into(),
            vec![F::ZERO; 4],
        )?
        .into();
        assert!(check(&changed, &[0, 1], 3).is_err());
        let mut changed = valid.clone();
        changed.output[0] =
            Tensor::<F>::new_with_unpadded_shape(vec![].into(), vec![].into(), vec![F::ZERO])?
                .into();
        assert!(check(&changed, &[0, 1], 3).is_err());
        let mut changed = valid;
        changed.output[0] =
            Tensor::<F>::new_with_unpadded_shape(vec![4].into(), vec![0].into(), vec![F::ZERO; 4])?
                .into();
        assert!(check(&changed, &[0, 1], 3).is_err());
        Ok(())
    }

    #[test]
    fn split_public_outputs_are_bound_to_full_sequence() -> anyhow::Result<()> {
        let split = [(0, vec![NodeId::from(0), NodeId::from(1)])]
            .into_iter()
            .collect();
        let mut io = public_io(&[0, 1, 2, 3, 4], &[1, 2, 3, 4, 5])?
            .with_splitted_inputs(Some(&split))?
            .with_splitted_outputs(Some(&split))?;
        assert!(check(&io, &[0, 1], 5).is_ok());
        let mut malformed = io.clone();
        let chunk = malformed.splitted_outputs.as_ref().unwrap()[&0][0].clone();
        malformed
            .splitted_outputs
            .as_mut()
            .unwrap()
            .insert(0, vec![chunk; 4]);
        assert!(check(&malformed, &[0, 1], 5).is_err());
        io.splitted_outputs.as_mut().unwrap().get_mut(&0).unwrap()[0][0] = F::from(6u64).into();
        assert!(check(&io, &[0, 1], 5).is_err());
        Ok(())
    }

    #[test]
    fn generation_proof_binds_prompt_continuation_last_token_and_length() -> anyhow::Result<()> {
        let (mut driver, _) = tiny_driver()?.into_provable_llm(None)?;
        for (prompt, count) in [(vec![0, 1], 4), (vec![0, 6], 4), (vec![0, 4], 5)] {
            let prompt = tokens(&prompt);
            let trace = driver.generate(&prompt, count, &mut GenStore::default())?;
            let expected = driver.continuation_tokens(&trace, prompt.len())?;
            let (pk, vk) = driver
                .context::<F, Pcs>()?
                .with_max_context(driver.max_context().unwrap());
            let (proof, io) = driver.prove(&pk, trace)?;
            assert_eq!(vk.continuation_tokens(&prompt, &io)?, expected);
            vk.verify(proof.clone(), prompt.clone(), io.clone())?;
            let mut wrong_prompt = prompt.clone();
            wrong_prompt[0] = Token::from(3usize);
            assert!(
                vk.verify(proof.clone(), wrong_prompt.clone(), io.clone())
                    .is_err()
            );
            let mut wrong_prompt_io = io.clone();
            wrong_prompt_io.input[0][0] = F::from(3u64).into();
            assert!(
                vk.continuation_tokens(&wrong_prompt, &wrong_prompt_io)
                    .is_ok()
            );
            assert!(
                vk.verify(proof.clone(), wrong_prompt, wrong_prompt_io)
                    .is_err()
            );
            let last = io.output[0].unpadded_shape().dim(0) - 1;
            let mut wrong_last = io.clone();
            wrong_last.output[0][last] = F::ZERO.into();
            assert!(
                vk.verify(proof.clone(), prompt.clone(), wrong_last)
                    .is_err()
            );
            let mut wrong_middle = io.clone();
            wrong_middle.output[0][prompt.len() - 1] = F::from(6u64).into();
            assert!(
                vk.verify(proof.clone(), prompt.clone(), wrong_middle)
                    .is_err()
            );
            if last + 1 > prompt.len() {
                let mut wrong_input = io.clone();
                wrong_input.input[0][prompt.len()] = F::ZERO.into();
                assert!(
                    vk.verify(proof.clone(), prompt.clone(), wrong_input)
                        .is_err()
                );
                // Keep the causal link internally consistent: only the model
                // proof can reject this jointly changed prediction and input.
                let mut wrong_pair = io.clone();
                wrong_pair.input[0][prompt.len()] = F::ZERO.into();
                wrong_pair.output[0][prompt.len() - 1] = F::ZERO.into();
                assert!(vk.continuation_tokens(&prompt, &wrong_pair).is_ok());
                assert!(
                    vk.verify(proof.clone(), prompt.clone(), wrong_pair)
                        .is_err()
                );
            }
            let mut wrong_eos = io.clone();
            wrong_eos.output[0][prompt.len() - 1] = F::from(7u64).into();
            if expected != tokens(&[7]) {
                assert!(vk.verify(proof.clone(), prompt.clone(), wrong_eos).is_err());
            }
            let mut wrong_length = io.clone();
            let shorter = (last + 1).saturating_sub(1);
            wrong_length.output[0] = Tensor::new_with_unpadded_shape(
                io.output[0].shape().clone(),
                vec![shorter, 1].into(),
                io.output[0].data().clone(),
            )?;
            assert!(vk.verify(proof, prompt, wrong_length).is_err());
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        Number, init_test_logging,
        iop::{ProverContext, chunking::LLMChunkingStrategy},
        model::{
            llm::{Driver, LLMVerifierContext, WithMaxContext},
            test::F,
        },
        parser::{
            file_cache,
            gguf::{RawGGUF, TensorLoader},
            llm::{
                HFTokenizer, LLMTokenizer, Token,
                models::{
                    gemma3::{Gemma3, tests::GEMMA3_Q8},
                    gpt2::{GPT2, GPT2_Q8_0},
                },
                tokenizer::TokenizerLoader,
            },
            safe::RawSafeTensors,
        },
        rng_from_env_or_random,
        testing::Pcs,
    };
    use anyhow::Context;
    use ark_std::rand::Rng;
    use tenstore::GenStore;
    use tracing::info;

    fn test_llm_driver_generic_prove_gpt2<const DISTRIBUTE_PROVE: bool>() -> anyhow::Result<()> {
        const MAX_CONTEXT: usize = 10;
        init_test_logging("debug");

        let model_path = file_cache::from_cache(GPT2_Q8_0)?;
        let (driver, _metadata) = Driver::load_from_model(
            GPT2::new(),
            &RawGGUF::new(model_path.clone()),
            Some(MAX_CONTEXT),
        )?
        .into_provable_llm(None)?;

        let (prover_ctx, verifier_ctx) = driver.context::<F, Pcs>()?.with_max_context(MAX_CONTEXT);

        driver.model.describe();

        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = GPT2::new().load_tokenizer(&RawGGUF::new(model_path))?;
        let user_tokens = tokenizer.tokenize(sentence);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;

        // Prove the trace
        let num_provers = if DISTRIBUTE_PROVE {
            Some(rng_from_env_or_random().gen_range(1..6))
        } else {
            Some(1) // equivalent to sequential proving
        };
        let chunking_strategy = LLMChunkingStrategy::from((&trace, &prover_ctx.model_ctx));
        let (proof, io) = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            num_provers,
            chunking_strategy,
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens, io)?;
        Ok(())
    }

    #[test]
    fn test_llm_driver_distributed_prove_gpt2() -> anyhow::Result<()> {
        test_llm_driver_generic_prove_gpt2::<true>()
    }

    #[test]
    #[ignore = "Sequential case covered already by distributed prove test, use this test only when checking performance to ensure sequential proving is used"]
    fn test_llm_driver_prove_gpt2() -> anyhow::Result<()> {
        test_llm_driver_generic_prove_gpt2::<false>()
    }

    #[test]
    #[ignore = "failing test - need to check why"]
    fn test_llm_ctx_serialization() -> anyhow::Result<()> {
        let prover_ctx_path = "prover_ctx.bin";
        let verifier_ctx_path = "verifier_ctx.bin";
        let _ = std::fs::remove_file(prover_ctx_path);
        let _ = std::fs::remove_file(verifier_ctx_path);
        let model_path = file_cache::from_cache("toy_gpt2.gguf")?;
        let (driver, _metadata) =
            Driver::load_from_model(GPT2::new(), &RawGGUF::new(model_path.clone()), Some(10))?
                .into_provable_llm(None)?;
        let (prover_ctx, _) = driver.context::<F, Pcs>()?.with_max_context(10);

        let pctx = bincode::serde::encode_to_vec(&prover_ctx, bincode::config::standard())?;
        let (deser_prover_ctx, _): (ProverContext<F, Pcs>, _) =
            bincode::serde::decode_from_slice(&pctx, bincode::config::standard())?;
        let pctx2 = bincode::serde::encode_to_vec(&deser_prover_ctx, bincode::config::standard())?;
        assert_eq!(pctx, pctx2);
        Ok(())
    }

    #[test]
    fn test_llm_driver_inference_gpt2_gguf() -> anyhow::Result<()> {
        init_test_logging("debug");
        const PRUNED_GPT2: &str = "gpt2.Q2_K.gguf";
        let model_path = file_cache::from_cache(PRUNED_GPT2)?;
        println!("model path: {:?}", model_path);

        let raw = RawGGUF::new(model_path.clone());
        let (driver, _metadata) =
            Driver::load_from_model(GPT2::new(), &raw, Some(10))?.into_provable_llm(None)?;
        test_llm_driver_inference_inner(raw, driver)
    }

    #[test]
    fn test_llm_driver_inference_gpt2_safe() -> anyhow::Result<()> {
        const GPT2_SAFE_MODEL: &str = "openai-community/gpt2";
        init_test_logging("debug");
        let raw = RawSafeTensors::from_hugging_face_cached(GPT2_SAFE_MODEL)?;

        let (driver, _metadata) =
            Driver::load_from_model(GPT2::new(), &raw, Some(10))?.into_provable_llm(None)?;
        test_llm_driver_inference_inner(raw, driver)
    }

    fn test_llm_driver_inference_inner<R>(raw: R, driver: Driver<i64>) -> anyhow::Result<()>
    where
        GPT2: TokenizerLoader<R>,
    {
        let sentence = "The sky is";

        // Best to load the tokenizer from the gguf file if it's available.
        let tokenizer = GPT2::new().load_tokenizer(&raw)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let detokenized = tokenizer.detokenize(&user_tokens);
        let mut store = GenStore::default();
        assert_eq!(detokenized, sentence);
        println!("user input in tokens: {user_tokens:?}");
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let output = tokenizer.detokenize(&output);
        println!("{}", output);
        Ok(())
    }

    #[test]
    fn test_llm_driver_inference_gemma3() -> anyhow::Result<()> {
        const CONTEXT_SIZE: usize = 10;
        init_test_logging("debug");
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let gguf = RawGGUF::new(model_path.clone());
        let driver = Driver::load_from_model(Gemma3::new(), &gguf, Some(CONTEXT_SIZE))?;

        println!("LLM DRIVER: config: {:?}", driver.md.config);

        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = Gemma3::new().load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let tensor_inputs = vec![driver.tokens_to_tensor(&user_tokens)?];
        let trace = driver.run(tensor_inputs, &mut store)?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let _output = tokenizer.detokenize(&output);
        Ok(())
    }

    #[test]
    fn test_llm_driver_int_inference_gemma3() -> anyhow::Result<()> {
        const CONTEXT_SIZE: usize = 10;
        init_test_logging("debug");
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let gguf = RawGGUF::new(model_path.clone());
        let (driver, _metadata) =
            Driver::load_from_model(Gemma3::new(), &gguf, Some(CONTEXT_SIZE))?
                .into_provable_llm(None)?;

        println!("LLM DRIVER: config: {:?}", driver.md.config);

        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = Gemma3::new().load_tokenizer(&gguf)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;
        let output = trace
            .outputs()
            .last()
            .unwrap()
            .tensor()
            .unwrap()
            .data()
            .iter()
            .map(|t| Token::from(t.to_usize()))
            .collect::<Vec<_>>();
        let _output = tokenizer.detokenize(&output);
        Ok(())
    }

    fn test_generic_prove_llm_gemma3<const DISTRIBUTE_PROVE: bool>() -> anyhow::Result<()> {
        init_test_logging("debug");
        const MAX_CONTEXT: usize = 8;
        let model_path = file_cache::from_cache(GEMMA3_Q8)?;
        let cache_filename = {
            let mut hasher = blake3::Hasher::new();
            hasher
                .update_mmap(&model_path)
                .context("hashing model file")?;
            let hash = hasher.finalize();
            format!("cache-{GEMMA3_Q8}-{hash}.bin")
        };

        // Generate or load the prover & verifier contexts
        #[allow(clippy::type_complexity)]
        let (driver, prover_ctx, verifier_ctx): (
            _,
            ProverContext<F, Pcs>,
            LLMVerifierContext<F, Pcs>,
        ) = file_cache::deserialize_or_create_with(&cache_filename, || {
            let gguf = RawGGUF::new(model_path.clone());
            let (driver, _metadata) =
                Driver::load_from_model(Gemma3::new(), &gguf, Some(MAX_CONTEXT))?
                    .into_provable_llm(None)?;

            let (prover_ctx, verifier_ctx) =
                driver.context::<F, Pcs>()?.with_max_context(MAX_CONTEXT);

            Ok((driver, prover_ctx, verifier_ctx))
        })?;

        println!("LLM DRIVER: config: {:?}", driver.md.config);
        // Generate the trace
        let mut store = GenStore::default();
        let sentence = "The sky is";
        let loader = TensorLoader::from_path(model_path)?;
        let tokenizer = HFTokenizer::sentencepiece_from_gguf(&loader)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;

        // Prove the trace
        let num_provers = if DISTRIBUTE_PROVE {
            Some(rng_from_env_or_random().gen_range(1..6))
        } else {
            Some(1) // equivalent to sequential proving
        };
        let chunking_strategy = LLMChunkingStrategy::from((&trace, &prover_ctx.model_ctx));
        let (proof, io) = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            num_provers,
            chunking_strategy,
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens, io)?;
        Ok(())
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_distribute_prove_llm_gemma3() -> anyhow::Result<()> {
        test_generic_prove_llm_gemma3::<true>()
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_prove_llm_gemma3() -> anyhow::Result<()> {
        test_generic_prove_llm_gemma3::<false>()
    }

    // ---- Llama2 (SafeTensors) ----

    fn test_generic_prove_llm_llama2<const DISTRIBUTE_PROVE: bool>() -> anyhow::Result<()> {
        use crate::parser::llm::models::llama2::{Llama2, tests::LLAMA2_SAFE_MODEL};

        init_test_logging("debug");
        const MAX_CONTEXT: usize = 10;

        let raw = RawSafeTensors::from_hugging_face_cached(LLAMA2_SAFE_MODEL)?;
        let (driver, _metadata) = Driver::load_from_model(Llama2::new(), &raw, Some(MAX_CONTEXT))?
            .into_provable_llm(None)?;

        let (prover_ctx, verifier_ctx) = driver.context::<F, Pcs>()?.with_max_context(MAX_CONTEXT);

        driver.model.describe();

        // Generate the trace
        let mut store = GenStore::default();
        let sentence = "The sky is";
        let tokenizer = Llama2::new().load_tokenizer(&raw)?;
        let user_tokens = tokenizer.tokenize(sentence);
        let input_tensor = driver.tokens_to_tensor(&user_tokens)?;
        let trace = driver.run_elements(input_tensor, &mut store)?;

        // Prove the trace
        let num_provers = if DISTRIBUTE_PROVE {
            Some(rng_from_env_or_random().gen_range(1..6))
        } else {
            Some(1)
        };
        let chunking_strategy = LLMChunkingStrategy::from((&trace, &prover_ctx.model_ctx));
        let (proof, io) = driver.chunked_prove_default_executor(
            &prover_ctx,
            trace,
            num_provers,
            chunking_strategy,
        )?;

        // Serialize the proof
        let proof_bytes =
            bincode::serde::encode_to_vec(&proof, bincode::config::standard()).unwrap();
        info!(
            "Proof size: {}",
            humansize::format_size(proof_bytes.len(), humansize::BINARY)
        );

        // Verify the proof
        verifier_ctx.verify(proof, user_tokens, io)?;
        Ok(())
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_distribute_prove_llm_llama2() -> anyhow::Result<()> {
        test_generic_prove_llm_llama2::<true>()
    }

    #[test]
    #[ignore = "Test requires large machine to run"]
    fn test_prove_llm_llama2() -> anyhow::Result<()> {
        test_generic_prove_llm_llama2::<false>()
    }
}
