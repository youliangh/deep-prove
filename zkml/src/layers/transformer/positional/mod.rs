use crate::{
    Claim, Element, NextPowerOfTwo, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    VectorTranscript,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx,
        add::Add,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            Splittable, VerifiableCtx,
        },
        transformer::positional::{
            absolute::{Absolute, AbsoluteCtx, AbsoluteProof},
            rope::{Rope, RopeCtx, RopeProof},
        },
    },
    model::Step,
    padding::{PaddingMode, ShapeInfo},
    parser::{
        gguf, json,
        llm::{
            LLMConfig,
            config::{LLMStructure, PositionalConfig, RopeConfig},
        },
    },
    quantization::ToField,
    tensor::{CommitmentId, TensorHandle, TensorSlice, TensorTypeParam, WrappedTensor},
};
use anyhow::{Ok, Result, bail, ensure};
use ark_ff::PrimeField;
use dp_crypto::arkyper::{CommitmentScheme, transcript::Transcript};
use itertools::Itertools;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

pub(crate) mod absolute;
pub(crate) mod rope;
pub use rope::RopeLayout;

/// The short name used to identify the positional layer.
pub const POSITIONAL_LAYER: &str = "POSI";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PositionalCtx {
    Absolute(AbsoluteCtx),
    Rope(RopeCtx),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub enum PositionalProof<F: PrimeField> {
    // Proofs for all the inputs of the positional encoding layer
    Absolute(AbsoluteProof<F>),
    Rope(RopeProof<F>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PositionalCache {
    seq_len: usize,
    initialized: bool,
}

impl PositionalCache {
    fn new() -> Self {
        Self {
            seq_len: 0,
            initialized: false,
        }
    }
    fn set_seq_len(&mut self, seq_len: usize) -> anyhow::Result<()> {
        if !self.initialized {
            self.seq_len = seq_len;
            self.initialized = true;
        } else {
            ensure!(
                self.seq_len + 1 == seq_len,
                "with caching, new seq_len should be old one + 1 = {} + 1 vs given {}",
                self.seq_len,
                seq_len
            );
            self.seq_len = seq_len;
        }
        Ok(())
    }
    fn reset(&mut self) {
        self.seq_len = 0;
        self.initialized = false;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub struct Positional<T>
where
    T: TensorTypeParam,
{
    /// The `cache` is used to store the current sequence length during inference with caching.
    /// It is an [`Option`] because some `PositionalVariant`s do not require caching (RoPE).
    cache: Option<Arc<Mutex<PositionalCache>>>,
    pub(crate) variant: PositionalVariant<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
#[allow(clippy::large_enum_variant)]
pub(crate) enum PositionalVariant<T>
where
    T: TensorTypeParam,
{
    Absolute(Absolute<T>),
    Rope(Rope<T>),
}

impl<N: TensorTypeParam> PositionalVariant<N> {
    #[allow(dead_code)]
    /// Create a new absolute positional variant
    pub fn new_absolute(absolute: Absolute<N>) -> Self {
        PositionalVariant::Absolute(absolute)
    }
}

impl<N: TensorTypeParam> Positional<N> {
    /// Static arithmetic settings, excluding learned/generated matrices and caches.
    pub(crate) fn quantization_summary(&self) -> serde_json::Value {
        match &self.variant {
            PositionalVariant::Absolute(_) => serde_json::json!({
                "variant": "absolute", "shape": self.shape(),
            }),
            PositionalVariant::Rope(rope) => serde_json::json!({
                "variant": "rope", "shape": self.shape(),
                "unpadded_shape": rope.unpadded_shape, "layout": rope.layout,
            }),
        }
    }

    pub fn shape(&self) -> Shape {
        match &self.variant {
            PositionalVariant::Absolute(abs) => abs.positional.shape().clone(),
            PositionalVariant::Rope(rope) => rope.cosine_matrix.shape().clone(),
        }
    }

    pub fn reset_cache(&self) {
        if let Some(cache) = &self.cache {
            cache.lock().unwrap().reset();
        }
        if let PositionalVariant::Rope(rope) = &self.variant {
            rope.reset_cache();
        }
    }

    pub(crate) fn new_from_variant(variant: PositionalVariant<N>) -> Self {
        let cache = Arc::new(Mutex::new(PositionalCache::new()));
        Self {
            cache: Some(cache),
            variant,
        }
    }

    pub(crate) fn with_cache(self) -> Self {
        let cache = Arc::new(Mutex::new(PositionalCache::new()));
        Self {
            cache: Some(cache),
            ..self
        }
    }

    /// Adds a concatenation cache if the [`PositionalVariant`] is RoPE, otherwise returns self.
    pub(crate) fn with_rope_cache(self, rank: usize, dim: usize) -> Result<Self> {
        match self.variant {
            PositionalVariant::Absolute(abs) => Ok(Self {
                cache: self.cache,
                variant: PositionalVariant::Absolute(abs),
            }),
            PositionalVariant::Rope(rope) => {
                let cached_rope = rope.with_concatenation_cache(rank, dim)?;
                Ok(Self {
                    cache: self.cache,
                    variant: PositionalVariant::Rope(cached_rope),
                })
            }
        }
    }

    pub fn new_absolute(matrix: TensorHandle<N>) -> anyhow::Result<Self> {
        let absolute = Absolute::new(matrix)?;
        Ok(Self::new_from_variant(PositionalVariant::Absolute(
            absolute,
        )))
    }

    pub fn new_rope(
        angles: Vec<f32>,
        base_frequency_id: CommitmentId,
        max_content_length: usize,
        layout: RopeLayout,
    ) -> anyhow::Result<Self> {
        Ok(Self::new_from_variant(PositionalVariant::Rope(
            Rope::build_from_angles(angles, base_frequency_id, max_content_length, layout)?,
        )))
    }

    pub fn new_rope_from_frequency(
        base_frequency: f32,
        base_frequency_id: CommitmentId,
        head_size: usize,
        max_content_length: usize,
        layout: RopeLayout,
    ) -> anyhow::Result<Self> {
        Ok(Self::new_from_variant(PositionalVariant::Rope(
            Rope::build_from_frequency(
                base_frequency,
                base_frequency_id,
                head_size,
                max_content_length,
                layout,
            )?,
        )))
    }

    /// Text-only Qwen partial RoPE: rotate the leading `rotary_dim` coordinates.
    pub fn new_partial_rope_from_frequency(
        base_frequency: f32,
        base_frequency_id: CommitmentId,
        head_size: usize,
        rotary_dim: usize,
        max_context_length: usize,
    ) -> Result<Self> {
        ensure!(rotary_dim >= 2 && rotary_dim.is_power_of_two()
            && head_size.is_power_of_two() && rotary_dim <= head_size,
            "Partial RoPE requires power-of-two head/rotary dimensions");
        let mut angles = vec![0.0; head_size / 2];
        for (i, angle) in angles[..rotary_dim / 2].iter_mut().enumerate() {
            *angle = base_frequency.powf(-2.0 * i as f32 / rotary_dim as f32);
        }
        Self::new_rope(angles, base_frequency_id, max_context_length,
            RopeLayout::RotateBlock(rotary_dim))
    }

    pub fn new_rope_from_matrices(
        cosine_matrix: TensorHandle<N>,
        sine_matrix: TensorHandle<N>,
        layout: RopeLayout,
    ) -> anyhow::Result<Self> {
        Ok(Self::new_from_variant(PositionalVariant::Rope(Rope::new(
            cosine_matrix,
            sine_matrix,
            layout,
        )?)))
    }

    /// Disable caching for `self` positional layer
    pub fn with_no_cache(self) -> Self {
        let variant = match self.variant {
            PositionalVariant::Absolute(abs) => PositionalVariant::Absolute(abs),
            PositionalVariant::Rope(rope) => PositionalVariant::Rope(rope.with_no_cache()),
        };
        Self {
            cache: None,
            variant,
        }
    }

    // Sample from the transcript `t` `num_coordinates` random coordinates to be employed
    // for proving. It requires as input also the claim related to the output polynomial and
    // the claim about the sub-matrix of positional matrix being actually added to the
    // input polynomial
    fn sample_random_coordinates<F: PrimeField, T: Transcript>(
        num_coordinates: usize,
        t: &mut T,
        output_claim: &Claim<F>,
        sub_matrix_claim: &Claim<F>,
    ) -> Vec<F> {
        // first, we add `output_claim` and `sub_pos_claim` to the transcript
        t.append_scalars(&output_claim.point);
        t.append_scalars(&[output_claim.eval]);
        t.append_scalars(&sub_matrix_claim.point);
        t.append_scalars(&[sub_matrix_claim.eval]);

        // then, we get `num_coordinates` challenges
        t.read_challenges(num_coordinates)
    }

    // Compute a claim about the positional encoding matrix at point `evaluation_point`.
    // The claim is computed from the following evaluations:
    // - `sub_pos_eval`: Evaluation of the sub-matrix that is added to the input tensor
    // - `sub_matrix_evals`: Evaluations of all the sub-matrices employed to derive the claim
    // about the positional encoding matrix
    // Note that the sub-matrices the evaluations are referred to corresponds to a partition
    // of the positional encoding matrix, that is their concatenation yields the positional
    // encoding matrix itself
    fn compute_positional_matrix_claim<F: PrimeField>(
        evaluation_point: Vec<F>,
        sub_pos_eval: F,
        sub_matrix_evals: &[F],
    ) -> Claim<F> {
        // compute the slice of the evaluation point to be used to compute the claim
        let eval_point = {
            let diff_vars = sub_matrix_evals.len();
            let start_extra_coordinates = evaluation_point.len() - diff_vars;
            &evaluation_point[start_extra_coordinates..]
        };
        let positional_matrix_eval = sub_matrix_evals
            .iter()
            .zip(eval_point)
            .fold(sub_pos_eval, |eval, (&sub_eval, &coordinate)| {
                eval * (F::ONE - coordinate) + sub_eval * coordinate
            });
        Claim::new(evaluation_point, positional_matrix_eval)
    }

    /// Build a claim for the positional matrix provided as input, starting from a claim
    /// `sub_matrix_claim` on the sub-matrix given by the first `input_num_rows` of positional matrix;
    /// it returns the claim for the positional matrix and the list of evaluations for the sub-matrices
    /// employed to build the claim, which have to be included in the proof
    fn bind_sub_claim_to_positional_matrix<'a, F: PrimeField, T: Transcript>(
        sub_matrix_claim: Claim<F>,
        output_claim: &Claim<F>,
        matrix_slice: &TensorSlice<'a, N>,
        positional_matrix: &Tensor<N>,
        input_num_rows: usize,
        transcript: &mut T,
    ) -> anyhow::Result<(Vec<F>, Claim<F>)>
    where
        N: ToField<F>,
    {
        // first, we compute the number of variables that we need to fill to get to the `positional_matrix`
        // polynomial
        let num_vars = {
            let num_vars = positional_matrix.shape().num_vars_2d();
            num_vars.0 + num_vars.1
        };

        let sub_pos_vars = sub_matrix_claim.point.len();
        let diff_vars = num_vars - sub_pos_vars;

        ensure!(diff_vars >= 0);

        // now, we need to squeeze `diff_vars` coordinates from the transcript
        let extra_coordinates = Positional::<Element>::sample_random_coordinates(
            diff_vars,
            transcript,
            output_claim,
            &sub_matrix_claim,
        );

        let sub_pos_eval = sub_matrix_claim.eval;

        let evaluation_point = sub_matrix_claim
            .point
            .into_iter()
            .chain(extra_coordinates)
            .collect_vec();

        let mut slice_start = input_num_rows;
        let sub_matrices = (0..diff_vars)
            .map(|_| {
                let sub_matrix = matrix_slice.slice_over_first_dim(slice_start, slice_start * 2);
                slice_start *= 2;
                sub_matrix
            })
            .collect_vec();

        // check that all the slices of `positional_matrix` have been computed
        ensure!(slice_start == positional_matrix.shape()[0]);

        // now, evaluate the MLE of each sub-matrix
        let sub_matrix_evals = (0..diff_vars)
            .into_par_iter()
            .map(|i| -> anyhow::Result<_> {
                Ok((
                    i,
                    sub_matrices[i]
                        .to_field()
                        .to_mle_2d()?
                        .evaluate(&evaluation_point[..sub_pos_vars + i])?,
                ))
            })
            .collect::<anyhow::Result<BTreeMap<_, _>>>()?
            .into_values()
            .collect::<Vec<_>>();

        let positional_matrix_claim = Positional::<Element>::compute_positional_matrix_claim(
            evaluation_point,
            sub_pos_eval,
            &sub_matrix_evals,
        );

        Ok((sub_matrix_evals, positional_matrix_claim))
    }
}

impl<N> Evaluate<N> for Positional<N>
where
    N: TensorTypeParam,
    Add<N>: Evaluate<N>,
{
    fn evaluate(&self, inputs: &[&WrappedTensor<N>]) -> anyhow::Result<LayerOut<N>> {
        ensure!(
            inputs.len() == 1,
            "Positional layer expects 1 input, got {}",
            inputs.len()
        );

        // default cache to be provided in case the layer was not initialized with a cache
        let new_cache = Arc::new(Mutex::new(PositionalCache::new()));

        match &self.variant {
            PositionalVariant::Absolute(absolute) => {
                ensure!(
                    inputs.iter().all(|x| x.rank() == 2),
                    "Absolute positional embeddings only support 2d tensors"
                );
                absolute.evaluate(inputs[0], self.cache.as_ref().unwrap_or(&new_cache))
            }
            PositionalVariant::Rope(rope) => {
                rope.evaluate(inputs[0], self.cache.as_ref().unwrap_or(&new_cache))
            }
        }
    }
}

fn output_shapes(
    input_shapes: &[Shape],
    pos_matrix_unpadded_shape: &Shape,
    padding_mode: PaddingMode,
) -> Vec<Shape> {
    let pos_shape = match padding_mode {
        PaddingMode::NoPadding => pos_matrix_unpadded_shape.clone(),
        PaddingMode::Padding => pos_matrix_unpadded_shape.next_power_of_two(),
    };

    input_shapes.iter().for_each(|s| {
        let s_rank = s.rank();
        assert!(s[s_rank - 2] <= pos_shape[0]);
        assert_eq!(s[s_rank - 1], pos_shape[1])
    });
    input_shapes.to_vec()
}

impl<N: TensorTypeParam> OpInfo for Positional<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match &self.variant {
            PositionalVariant::Absolute(absolute) => Ok(output_shapes(
                input_shapes,
                &absolute.unpadded_shape,
                padding_mode,
            )),
            PositionalVariant::Rope(rope) => rope.output_shapes(input_shapes, padding_mode),
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        format!("Positional({:?}x{:?})", self.shape()[0], self.shape()[1])
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl ProveInfo for Positional<Element> {
    fn step_info<F: PrimeField>(
        &self,
        aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<F>, ContextAux)> {
        let (ctx, aux) = match &self.variant {
            PositionalVariant::Absolute(absolute) => {
                let (ctx, aux) = absolute.step_info::<F>(aux)?;
                (PositionalCtx::Absolute(ctx), aux)
            }
            PositionalVariant::Rope(rope) => {
                let (ctx, aux) = rope.step_info(aux)?;
                (PositionalCtx::Rope(ctx), aux)
            }
        };

        Ok((LayerCtx::Positional(ctx), aux))
    }
}

impl QuantizeOp for Positional<f32> {
    type QuantizedOp = Positional<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input_scaling.len() == 1,
            "Expected 1 input scaling factor for positional layer, found {}",
            input_scaling.len()
        );
        ensure!(
            output_scalings.len() == 1,
            "Expected 1 output scaling factor for positional layer, found {}",
            output_scalings.len()
        );
        ensure!(
            unpadded_output_shapes.len() == 1,
            "Expected 1 output shape for positional layer, found {}",
            unpadded_output_shapes.len()
        );
        // re-initialize the cache for quantized node, if the original node has a cache enabled
        let new_cache = self
            .cache
            .map(|_| Arc::new(Mutex::new(PositionalCache::new())));

        let quantized_op = match self.variant {
            PositionalVariant::Absolute(abs) => {
                let quantized_abs = abs.quantize::<S>(
                    data,
                    node_id,
                    input_scaling[0],
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput {
                    quantized_op: Positional {
                        cache: new_cache,
                        variant: PositionalVariant::Absolute(quantized_abs.quantized_op),
                    },
                    output_scalings: quantized_abs.output_scalings,
                    requant_layer: quantized_abs.requant_layer,
                    post_quant_rule: None,
                }
            }
            PositionalVariant::Rope(rope) => {
                let quantized_rope = rope.quantize::<S>(data, node_id, input_scaling[0])?;
                QuantizeOutput {
                    quantized_op: Positional {
                        cache: new_cache,
                        variant: PositionalVariant::Rope(quantized_rope.quantized_op),
                    },
                    output_scalings: quantized_rope.output_scalings,
                    requant_layer: quantized_rope.requant_layer,
                    post_quant_rule: None,
                }
            }
        };

        Ok(quantized_op)
    }
}

impl PadOp for Positional<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let cache = self
            .cache
            .map(|_| Arc::new(Mutex::new(PositionalCache::new())));

        let padded_variant = match self.variant {
            PositionalVariant::Absolute(pos) => PositionalVariant::Absolute(pos.pad_node(si)?),
            PositionalVariant::Rope(rope) => PositionalVariant::Rope(rope.pad_node(si)?),
        };

        // no need to change `si` since the layer doesn't change the input shapes
        Ok(Self {
            cache,
            variant: padded_variant,
        })
    }
}

impl Positional<f32> {
    pub fn from_gguf(
        loader: &gguf::FileTensorLoader,
        c: &LLMStructure,
        positional_config: &PositionalConfig,
    ) -> anyhow::Result<Self> {
        match positional_config {
            PositionalConfig::Rope(rope_config) => {
                let RopeConfig {
                    base_frequency,
                    max_seq_length,
                    layout,
                } = *rope_config;
                let base_freq_id =
                    format!("{}.rope.freq_base.{base_frequency}", c.generic.model_name);
                let head_size = c.generic.head_size;
                Self::new_rope_from_frequency(
                    base_frequency,
                    base_freq_id.to_string().into(),
                    head_size,
                    max_seq_length,
                    layout,
                )
            }
            PositionalConfig::FixedPositional => {
                let mut position_embd = loader.get_tensor("position_embd.weight")?;
                position_embd.truncate_on_first_dim(c.generic.context_length)?;
                let shape = position_embd.shape();
                ensure!(
                    shape[0] == c.generic.context_length,
                    "position_embd must have shape [{}] vs given {:?}",
                    c.generic.context_length,
                    position_embd.shape()
                );
                ensure!(
                    shape[1] == c.generic.embedding_size,
                    "position_embd must have shape [{}] vs given {:?}",
                    c.generic.embedding_size,
                    position_embd.shape()
                );
                Ok(Self::new_absolute(position_embd.into())?)
            }
        }
    }
    pub fn from_json(l: &json::FileTensorLoader, c: &LLMConfig) -> anyhow::Result<Self> {
        let mut position_embd = l.get_tensor("position_embd.weight")?;
        position_embd.truncate_on_first_dim(c.context_length)?;
        ensure!(
            position_embd.shape().rank() == 2,
            "position_embd must be 2d"
        );
        ensure!(
            position_embd.shape()[0] == c.context_length,
            "position_embd must have shape [0] [{}] vs given {:?}",
            c.context_length,
            position_embd.shape()
        );
        ensure!(
            position_embd.shape()[1] == c.embedding_size,
            "position_embd must have shape [1] [{}] vs given {:?}",
            c.embedding_size,
            position_embd.shape()
        );
        Ok(Self::new_absolute(position_embd.into())?)
    }
    pub fn from_safetensors_loader(
        loader: &crate::parser::safe::FileTensorLoader,
        structure: &LLMStructure,
        positional_config: &PositionalConfig,
    ) -> anyhow::Result<Self> {
        match positional_config {
            PositionalConfig::Rope(rope_config) => {
                // Build angles per head and repeat across heads (no theta in safetensors header here)
                let head_size = structure.generic.head_size;
                let RopeConfig {
                    base_frequency,
                    max_seq_length,
                    layout,
                } = *rope_config;

                Self::new_rope_from_frequency(
                    base_frequency,
                    format!("rope.freq.{base_frequency}").into(),
                    head_size,
                    max_seq_length,
                    layout,
                )
            }
            PositionalConfig::FixedPositional => {
                let mut position_embd = loader
                    .get_tensor("position_embd.weight")
                    .or_else(|_| loader.get_tensor("transformer.wpe.weight"))
                    .or_else(|_| loader.get_tensor("wpe.weight"))?;
                position_embd.truncate_on_first_dim(structure.generic.context_length)?;
                let shape = position_embd.shape();
                ensure!(
                    shape[0] == structure.generic.context_length,
                    "position_embd must have shape [{}] vs given {:?}",
                    structure.generic.context_length,
                    position_embd.shape()
                );
                ensure!(
                    shape[1] == structure.generic.embedding_size,
                    "position_embd must have shape [{}] vs given {:?}",
                    structure.generic.embedding_size,
                    position_embd.shape()
                );
                Ok(Self::new_absolute(position_embd.into())?)
            }
        }
    }
}

impl<F: PrimeField, PCS> ProvableOp<F, PCS> for Positional<Element>
where
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = PositionalCtx;

    fn prove<T: Transcript>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<F, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        ensure!(
            last_claims.len() == 1,
            "Expected 1 output claim for positional layer, found {}",
            last_claims.len(),
        );
        ensure!(
            step_data.node_inputs.len() == 1,
            "Expected 1 input in trace for positional layer, found {}",
            step_data.node_inputs.len(),
        );

        let output_claim = last_claims[0];
        match &self.variant {
            PositionalVariant::Absolute(absolute) => {
                absolute.prove_step(node_id, output_claim, step_data, prover)
            }
            PositionalVariant::Rope(rope) => {
                rope.prove_step(node_id, output_claim, step_data, prover)
            }
        }
    }
}

impl OpInfo for PositionalCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        Ok(output_shapes(
            input_shapes,
            self.unpadded_shape(),
            padding_mode,
        ))
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        Ok(num_inputs)
    }

    fn describe(&self) -> String {
        let unpadded_shape = self.unpadded_shape();
        format!(
            "PositionalCtx:({:?}x{:?})",
            unpadded_shape[0], unpadded_shape[1]
        )
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Splittable for PositionalCtx {}

impl PositionalCtx {
    fn unpadded_shape(&self) -> &Shape {
        match self {
            PositionalCtx::Absolute(absolute_ctx) => &absolute_ctx.unpadded_shape,
            PositionalCtx::Rope(rope_ctx) => &rope_ctx.unpadded_shape,
        }
    }
    /// Compute the claim for a full positional matrix from a claim `sub_claim` about a sub matrix
    /// and the evaluations for other sub-matrices provided as input.
    fn build_positional_matrix_claim<F: PrimeField, T: Transcript>(
        sub_claim: Claim<F>,
        output_claim: &Claim<F>,
        num_vars_full_matrix: usize,
        transcript: &mut T,
        sub_matrix_evals: &[F],
    ) -> anyhow::Result<Claim<F>> {
        // first, we compute the number of variables that we need to fill to get to the `positional_matrix`
        // polynomial
        let sub_pos_vars = sub_claim.point.len();
        let diff_vars = num_vars_full_matrix - sub_pos_vars;

        ensure!(diff_vars >= 0);

        // then, we sample the extra coordinates corresponding to these variables
        let extra_coordinates = Positional::<Element>::sample_random_coordinates(
            diff_vars,
            transcript,
            output_claim,
            &sub_claim,
        );

        let sub_pos_eval = sub_claim.eval;

        let evaluation_point = sub_claim
            .point
            .into_iter()
            .chain(extra_coordinates)
            .collect_vec();

        let positional_matrix_claim = Positional::<Element>::compute_positional_matrix_claim(
            evaluation_point,
            sub_pos_eval,
            sub_matrix_evals,
        );

        Ok(positional_matrix_claim)
    }
}

impl<F: PrimeField, PCS: CommitmentScheme<Field = F>> VerifiableCtx<F, PCS> for PositionalCtx {
    type Proof = PositionalProof<F>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        let num_outputs = last_claims.len();

        ensure!(
            shape_step.unpadded_output_shape.len() == num_outputs,
            "Expected {num_outputs} unpadded output shapes for Positional layer verifier, found {}",
            shape_step.unpadded_output_shape.len()
        );

        ensure!(
            shape_step.padded_output_shape.len() == num_outputs,
            "Expected {num_outputs} padded output shapes for Positional layer verifier, found {}",
            shape_step.padded_output_shape.len()
        );

        ensure!(
            shape_step.unpadded_input_shape.len() == num_outputs,
            "Expected {num_outputs} unpadded input shapes for Positional layer verifier, found {}",
            shape_step.unpadded_input_shape.len()
        );

        ensure!(
            shape_step.padded_input_shape.len() == num_outputs,
            "Expected {num_outputs} padded input shapes for Positional layer verifier, found {}",
            shape_step.padded_input_shape.len()
        );

        ensure!(
            last_claims.len() == 1,
            "Expected 1 output claim when verifying positional layer, found {}",
            last_claims.len(),
        );

        let output_claim = last_claims[0];

        match (self, proof) {
            (PositionalCtx::Absolute(absolute_ctx), PositionalProof::Absolute(proof)) => {
                absolute_ctx.verify(proof, verifier, output_claim, shape_step, node_id)
            }
            (PositionalCtx::Rope(rope_ctx), PositionalProof::Rope(proof)) => {
                rope_ctx.verify(proof, output_claim, verifier, shape_step, node_id)
            }
            _ => bail!("Incompatible ctx and proof types found for positional layer"),
        }
    }

    fn write_proof_to_transcript<T: Transcript>(
        &self,
        _proof: &Self::Proof,
        _transcript: &mut T,
    ) -> anyhow::Result<()> {
        // No commitment so just return Ok(())
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use tenstore::GenStore;

    use crate::{
        Element, Tensor,
        layers::{
            provable::PadOp,
            transformer::positional::{Positional, PositionalVariant},
        },
        padding::{ShapeData, ShapeInfo},
        tensor::TensorHandle,
    };

    #[test]
    fn test_positional_padding() {
        let seq_len = 14;
        let context_length = 17;
        let embedding_size = 45;
        let input_shape = vec![seq_len, embedding_size];
        let matrix_shape = vec![context_length, embedding_size];
        let positional_matrix = TensorHandle::from_tensor(
            "absolute_positional_mat".into(),
            GenStore::new_empty(),
            Tensor::<Element>::random(&matrix_shape.into()),
        );
        let pos = Positional::new_absolute(positional_matrix.clone()).unwrap();

        let mut si = ShapeInfo::from(vec![ShapeData::new(input_shape.into())].as_slice());

        let padded_layer = pos.pad_node(&mut si).unwrap();

        let PositionalVariant::Absolute(padded_pos) = padded_layer.variant else {
            unreachable!()
        };

        // After pad_node, positional matrix is stored unpadded
        let stored_shape = padded_pos.positional.shape();
        assert_eq!(padded_pos.unpadded_shape, *positional_matrix.shape());
        assert_eq!(*stored_shape, *positional_matrix.shape());

        // check that stored positional matrix has the same data as original matrix
        let stored_tensor = Tensor::try_from(padded_pos.positional.clone()).unwrap();
        let tensor = Tensor::try_from(positional_matrix).unwrap();
        for i in 0..stored_shape[0] {
            for j in 0..stored_shape[1] {
                assert_eq!(stored_tensor.get_2d(i, j), tensor.get_2d(i, j));
            }
        }
    }
}
