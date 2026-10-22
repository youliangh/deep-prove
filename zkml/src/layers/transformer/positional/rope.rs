use crate::{
    Claim, Element, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor, eval_zeroifier_mle,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerProof,
        add::Add,
        provable::{Evaluate, LayerOut, PadOp, QuantizeOutput},
        requant::Requant,
        transformer::{
            ConcatenationCache,
            positional::{Positional, PositionalCache, PositionalCtx, PositionalProof},
        },
    },
    model::Step,
    padding::PaddingMode,
    poly_commit::identity_eval,
    quantization::{self, Quantize, ToField},
    tensor::{
        CommitmentId, TensorHandle, TensorSlice, TensorTypeParam, WrappedTensor,
        is_close_with_tolerance,
    },
    to_bit_sequence_le,
    util::from_mle_list_dimensions,
};
use anyhow::{Ok, Result, ensure};
use ark_ff::PrimeField;
use burn::tensor::{Shape as BShape, Tensor as BTensor};
use dp_crypto::{
    IntoMLE,
    arkyper::{CommitmentScheme, transcript::Transcript},
    poly::eq::evals,
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::{ceil_log2, optimal_sumcheck_threads},
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use itertools::Itertools;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::sync::{Arc, Mutex};
use tenstore::{GenStore, StorageKey};
use tracing::warn;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct RopeProof<F: PrimeField> {
    // Evaluations of the sub-matrices required to compute the claim
    // about the cosine matrix.
    #[serde(with = "dp_crypto::serialization")]
    sub_cosine_evals: Vec<F>,
    // Evaluations of the sub-matrices required to compute the claim
    // about the sine matrix.
    #[serde(with = "dp_crypto::serialization")]
    sub_sine_evals: Vec<F>,
    // Proof to link the output with the input
    sumcheck_proof: IOPProof<F>,
    // Evaluations of the polynomials involved in `sumcheck_proof`
    #[serde(with = "dp_crypto::serialization")]
    sumcheck_evals: Vec<F>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RopeCtx {
    pub(super) unpadded_shape: Shape,
    num_vars_positional_matrix: usize,
    cosine_key: CommitmentId,
    sine_key: CommitmentId,
    layout: RopeLayout,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
/// Enum used for specifying different rope layout strategies.
pub enum RopeLayout {
    /// In adjacent layout given an input tensor `x=[x_1,x_2,x_3,x_4,...,x_n-1,x_n]`,
    /// we calculate the permuted input tensor `x_perm = [-x_2,x_1,-x_4,x_3,...,-x_n,x_n-1]`
    /// and then the output is computed as `output = x_perm * sine_matrix + x * cosine_matrix`
    Adjacent,
    /// In RotateHalf layout given an input tensor `x=[x_1,x_2,x_3,x_4,...,x_n-1,x_n]`,
    /// we calculate the permuted input tensor `x_perm = [-x_n/2+1,...,-x_n,x_1,...,x_n/2]`
    /// and then the output is computed as `output = x_perm * sine_matrix + x * cosine_matrix`
    RotateHalf,
    /// Rotate halves within fixed-size blocks. Blocks with zero angles are identities.
    RotateBlock(usize),
}

impl RopeLayout {
    /// Splits the input tensor according to the layout strategy.
    /// Returns a wrapped tensor with the appropriate shape.
    pub(crate) fn split_input<T: TensorTypeParam>(
        &self,
        input: WrappedTensor<T>,
    ) -> WrappedTensor<T> {
        let input_shape = input.shape();
        match self {
            RopeLayout::RotateBlock(block) => {
                let shape = Shape::from(input.shape());
                assert!(block.is_power_of_two() && *block >= 2 && shape.dim(-1) % block == 0);
                let mut values = input.get_data();
                for row in values.chunks_exact_mut(*block) {
                    let (left, right) = row.split_at_mut(block / 2);
                    for (a, b) in left.iter_mut().zip(right) {
                        let previous = *a;
                        *a = T::zero() - *b;
                        *b = previous;
                    }
                }
                let tensor =
                    Tensor::new_with_unpadded_shape(shape, input.unpadded_shape().into(), values)
                        .expect("RoPE shape preserved");
                WrappedTensor::try_from(&tensor).expect("RoPE rank preserved")
            }
            RopeLayout::Adjacent => {
                // In this case we need to split the input tensor in half along the last dimension, but first we reshape so the
                // the tensor has shape [*, seq_len, head_size / 2, 2]
                let reshape_shape = BShape::from(
                    input_shape
                        .iter()
                        .take(input_shape.rank() - 1)
                        .cloned()
                        .chain([input_shape.dims[input_shape.rank() - 1] / 2, 2])
                        .collect::<Vec<usize>>(),
                );
                let chunking_dim = reshape_shape.len() - 1;
                match input {
                    WrappedTensor::Rank1 {
                        tensor: input,
                        unpadded_shape,
                    } => {
                        let mut reshaped =
                            input.reshape::<2, _>(reshape_shape).chunk(2, chunking_dim);
                        let x1 = reshaped.remove(0);
                        let x2 = reshaped.remove(0);
                        let x2 = x2.neg();
                        let output =
                            BTensor::cat(vec![x2, x1], chunking_dim).reshape::<1, _>(input_shape);
                        WrappedTensor::Rank1 {
                            tensor: output,
                            unpadded_shape,
                        }
                    }
                    WrappedTensor::Rank2 {
                        tensor: input,
                        unpadded_shape,
                    } => {
                        let mut reshaped =
                            input.reshape::<3, _>(reshape_shape).chunk(2, chunking_dim);
                        let x1 = reshaped.remove(0);
                        let x2 = reshaped.remove(0);
                        let x2 = x2.neg();
                        let output =
                            BTensor::cat(vec![x2, x1], chunking_dim).reshape::<2, _>(input_shape);
                        WrappedTensor::Rank2 {
                            tensor: output,
                            unpadded_shape,
                        }
                    }
                    WrappedTensor::Rank3 {
                        tensor: input,
                        unpadded_shape,
                    } => {
                        let mut reshaped =
                            input.reshape::<4, _>(reshape_shape).chunk(2, chunking_dim);
                        let x1 = reshaped.remove(0);
                        let x2 = reshaped.remove(0);
                        let x2 = x2.neg();
                        let output =
                            BTensor::cat(vec![x2, x1], chunking_dim).reshape::<3, _>(input_shape);
                        WrappedTensor::Rank3 {
                            tensor: output,
                            unpadded_shape,
                        }
                    }
                    WrappedTensor::Rank4 {
                        tensor: input,
                        unpadded_shape,
                    } => {
                        let mut reshaped =
                            input.reshape::<5, _>(reshape_shape).chunk(2, chunking_dim);
                        let x1 = reshaped.remove(0);
                        let x2 = reshaped.remove(0);
                        let x2 = x2.neg();
                        let output =
                            BTensor::cat(vec![x2, x1], chunking_dim).reshape::<4, _>(input_shape);
                        WrappedTensor::Rank4 {
                            tensor: output,
                            unpadded_shape,
                        }
                    }
                }
            }
            RopeLayout::RotateHalf => {
                let chunking_dim = input_shape.rank() - 1;
                match input {
                    WrappedTensor::Rank1 {
                        tensor: input,
                        unpadded_shape,
                    } => {
                        let mut chunked = input.chunk(2, chunking_dim);
                        let x1 = chunked.remove(0);
                        let x2 = chunked.remove(0).neg();
                        let output = BTensor::cat(vec![x2, x1], chunking_dim);
                        WrappedTensor::Rank1 {
                            tensor: output,
                            unpadded_shape,
                        }
                    }
                    WrappedTensor::Rank2 {
                        tensor: input,
                        unpadded_shape,
                    } => {
                        let mut chunked = input.chunk(2, chunking_dim);
                        let x1 = chunked.remove(0);
                        let x2 = chunked.remove(0).neg();
                        let output = BTensor::cat(vec![x2, x1], chunking_dim);
                        WrappedTensor::Rank2 {
                            tensor: output,
                            unpadded_shape,
                        }
                    }
                    WrappedTensor::Rank3 {
                        tensor: input,
                        unpadded_shape,
                    } => {
                        let mut chunked = input.chunk(2, chunking_dim);
                        let x1 = chunked.remove(0);
                        let x2 = chunked.remove(0).neg();
                        let output = BTensor::cat(vec![x2, x1], chunking_dim);
                        WrappedTensor::Rank3 {
                            tensor: output,
                            unpadded_shape,
                        }
                    }
                    WrappedTensor::Rank4 {
                        tensor: input,
                        unpadded_shape,
                    } => {
                        let mut chunked = input.chunk(2, chunking_dim);
                        let x1 = chunked.remove(0);
                        let x2 = chunked.remove(0).neg();
                        let output = BTensor::cat(vec![x2, x1], chunking_dim);
                        WrappedTensor::Rank4 {
                            tensor: output,
                            unpadded_shape,
                        }
                    }
                }
            }
        }
    }

    fn permuted_eq_poly<F: PrimeField>(&self, eq_evals: &[F], row_size: usize) -> Vec<F> {
        match self {
            RopeLayout::RotateBlock(block) => eq_evals
                .chunks(*block)
                .flat_map(|chunk| {
                    let (left, right) = chunk.split_at(block / 2);
                    right.iter().copied().chain(left.iter().map(|x| -*x))
                })
                .collect(),
            RopeLayout::Adjacent => eq_evals
                .chunks(2)
                .flat_map(|chunk| vec![chunk[1], F::ZERO - chunk[0]])
                .collect_vec(),
            RopeLayout::RotateHalf => {
                let half_len = row_size / 2;
                eq_evals
                    .chunks(row_size)
                    .flat_map(|chunk| {
                        let (first_half, second_half) = chunk.split_at(half_len);
                        second_half
                            .iter()
                            .copied()
                            .chain(first_half.iter().map(|e| F::ZERO - *e))
                    })
                    .collect_vec()
            }
        }
    }

    fn evaluate_permuted_eq_poly<F: PrimeField>(
        &self,
        eq_point: &[F],
        sumcheck_point: &[F],
        row_num_vars: usize,
    ) -> F {
        match self {
            RopeLayout::RotateBlock(block) => {
                let bit = block.ilog2() as usize - 1;
                identity_eval(&eq_point[..bit], &sumcheck_point[..bit])
                    * (eq_point[bit] - sumcheck_point[bit])
                    * identity_eval(&eq_point[bit + 1..], &sumcheck_point[bit + 1..])
            }
            RopeLayout::Adjacent => evaluate_permutation_eq_poly(eq_point, sumcheck_point),
            RopeLayout::RotateHalf => {
                let no_final_var_eval = identity_eval(
                    &eq_point[..row_num_vars - 1],
                    &sumcheck_point[..row_num_vars - 1],
                );
                // Normally we would do (eq_last * sumcheck_last + (E::ONE - eq_last) * (E::ONE - sumcheck_last)) but because we want to flip the order (and multiply the first half by -1)
                // we do -(E::ONE - eq_last) * sumcheck_last + eq_last * (E::ONE - sumcheck_last)
                let eq_last = eq_point[row_num_vars - 1];
                let sumcheck_last = sumcheck_point[row_num_vars - 1];
                let remaining_eval =
                    identity_eval(&eq_point[row_num_vars..], &sumcheck_point[row_num_vars..]);
                no_final_var_eval * (eq_last - sumcheck_last) * remaining_eval
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub struct Rope<T>
where
    T: TensorTypeParam,
{
    pub(super) cosine_matrix: TensorHandle<T>,
    pub(super) sine_matrix: TensorHandle<T>,
    pub(super) unpadded_shape: Shape,
    pub(super) layout: RopeLayout,
    concatenation_cache: Option<Arc<Mutex<ConcatenationCache<T>>>>,
}

impl<N: TensorTypeParam> Rope<N> {
    pub(crate) fn build_from_angles(
        angles: Vec<f32>,
        base_id: CommitmentId,
        max_content_length: usize,
        layout: RopeLayout,
    ) -> Result<Self> {
        // build the rotational vectors
        let matrix_shape = Shape::new(vec![max_content_length, angles.len() * 2]);
        let mut cosine_data = Vec::with_capacity(matrix_shape.numel());
        let mut sine_data = Vec::with_capacity(matrix_shape.numel());
        for i in 0..max_content_length {
            let mut tmp_cosine = Vec::with_capacity(angles.len());
            let mut tmp_sine = Vec::with_capacity(angles.len());
            angles.iter().try_for_each(|angle| {
                let cosine = N::from_f32((angle * i as f32).cos())?;
                let sine = N::from_f32((angle * i as f32).sin())?;
                match layout {
                    RopeLayout::Adjacent => {
                        cosine_data.append(&mut vec![cosine; 2]);
                        sine_data.append(&mut vec![sine; 2]);
                    }
                    RopeLayout::RotateHalf | RopeLayout::RotateBlock(_) => {
                        // In this layout the trig values are repeated only once per half a row
                        tmp_cosine.push(cosine);
                        tmp_sine.push(sine);
                    }
                }

                Ok(())
            })?;
            if let RopeLayout::RotateHalf = layout {
                let mut cosines = [tmp_cosine.clone(), tmp_cosine].concat();
                cosine_data.append(&mut cosines);
                let mut sines = [tmp_sine.clone(), tmp_sine].concat();
                sine_data.append(&mut sines);
            } else if let RopeLayout::RotateBlock(block) = layout {
                ensure!(
                    block.is_power_of_two() && block >= 2 && angles.len() * 2 % block == 0,
                    "Invalid rotary block width"
                );
                for chunk in tmp_cosine.chunks(block / 2) {
                    cosine_data.extend_from_slice(chunk);
                    cosine_data.extend_from_slice(chunk);
                }
                for chunk in tmp_sine.chunks(block / 2) {
                    sine_data.extend_from_slice(chunk);
                    sine_data.extend_from_slice(chunk);
                }
            }
        }

        let cosine_matrix = TensorHandle::from_tensor(
            StorageKey::from(format!("{}_cosine", base_id)),
            GenStore::new_empty(),
            Tensor::new(matrix_shape.clone(), cosine_data)?,
        );
        let sine_matrix = TensorHandle::from_tensor(
            StorageKey::from(format!("{}_sine", base_id)),
            GenStore::new_empty(),
            Tensor::new(matrix_shape.clone(), sine_data)?,
        );
        Ok(Self {
            cosine_matrix: cosine_matrix.wrapped_tensor_variant()?,
            sine_matrix: sine_matrix.wrapped_tensor_variant()?,
            unpadded_shape: matrix_shape,
            layout,
            concatenation_cache: None,
        })
    }

    pub(crate) fn build_from_frequency(
        base_frequency: f32,
        base_frequency_id: CommitmentId,
        head_size: usize,
        max_content_length: usize,
        layout: RopeLayout,
    ) -> Result<Self> {
        // if the layout is RotateHalf then head_size must be a power of 2
        if let RopeLayout::RotateHalf = layout {
            ensure!(
                head_size.is_power_of_two(),
                "For RotateHalf layout head_size must be a power of 2, found head_size = {head_size}",
            );
        }
        let angles = (0..head_size / 2)
            .map(|i| base_frequency.powf((-2.0 * i as f32) / head_size as f32))
            .collect_vec();
        Self::build_from_angles(angles, base_frequency_id, max_content_length, layout)
    }

    pub(crate) fn new(
        cosine_matrix: TensorHandle<N>,
        sine_matrix: TensorHandle<N>,
        layout: RopeLayout,
    ) -> Result<Self> {
        ensure!(
            cosine_matrix.shape() == sine_matrix.shape(),
            "Shapes of provided cosine and sine matrices are different: cosine_shape {:?} vs sine shape {:?}",
            cosine_matrix.shape(),
            sine_matrix.shape(),
        );
        let matrix_shape = cosine_matrix.shape().clone();

        if let RopeLayout::RotateHalf = layout {
            ensure!(
                matrix_shape.dim(-1).is_power_of_two(),
                "For RotateHalf layout row size must be a power of 2, found row size = {}",
                matrix_shape.dim(-1)
            );
        }
        Ok(Self {
            cosine_matrix: cosine_matrix.wrapped_tensor_variant()?,
            sine_matrix: sine_matrix.wrapped_tensor_variant()?,
            unpadded_shape: matrix_shape,
            layout,
            concatenation_cache: None,
        })
    }

    /// Reset the cache if present
    pub(crate) fn reset_cache(&self) {
        if let Some(cache) = &self.concatenation_cache {
            cache.lock().unwrap().reset();
        }
    }

    /// Attach a concatenation cache to the rope layer
    pub(crate) fn with_concatenation_cache(
        self,
        input_rank: usize,
        concatenation_dim: usize,
    ) -> anyhow::Result<Self> {
        ensure!(
            concatenation_dim < input_rank,
            "Concatenation dimension {concatenation_dim} out of bounds for input rank {input_rank}"
        );

        let concat_cache = Arc::new(Mutex::new(ConcatenationCache::new(
            input_rank,
            concatenation_dim,
        )));
        Ok(Self {
            concatenation_cache: Some(concat_cache),
            ..self
        })
    }

    /// Disable concatenation caching for RoPE layer
    pub fn with_no_cache(self) -> Self {
        Self {
            concatenation_cache: None,
            ..self
        }
    }

    /// Gets the output shape after applying RoPE to an input of given shape.
    /// If using a [`ConcatenationCache`], the output shape will account for the concatenation but should be called
    /// BEFORE evaluating the layer.
    pub fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        let input_shapes = if let Some(concatenation_cache) = self.concatenation_cache.as_ref() {
            let cache = concatenation_cache.lock().unwrap();

            input_shapes
                .iter()
                .map(|shape| cache.next_shape(shape.clone(), padding_mode))
                .collect::<Result<Vec<Shape>>>()?
        } else {
            input_shapes.to_vec()
        };
        Ok(input_shapes)
    }

    /// Since cosine and since matrix are committed with one variable less, we need to drop a point coordinate in the
    /// claims for such matrices; this variable changes depending on the layout strategy. If Adjacent is used we drop the
    /// first variable, which corresponds to the lowest bit of the row index; if RotateHalf is used we drop the variable
    /// corresponding to the highest bit of the row index.
    fn claim_for_committed_matrix<F: PrimeField>(&self, claim: &mut Claim<F>) {
        match self.layout {
            RopeLayout::Adjacent => claim.point.remove(0),
            RopeLayout::RotateBlock(block) => claim.point.remove(block.ilog2() as usize - 1),
            RopeLayout::RotateHalf => {
                let row_vars = ceil_log2(self.cosine_matrix.shape().dim(-1));
                claim.point.remove(row_vars - 1)
            }
        };
    }

    pub(super) fn prove_step<F, T, PCS>(
        &self,
        node_id: NodeId,
        output_claim: &Claim<F>,
        step_data: &Step<Element>,
        prover: &mut Prover<F, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<F>>>
    where
        N: ToField<F>,
        F: PrimeField,
        T: Transcript,
        PCS: CommitmentScheme<Field = F>,
    {
        let input = step_data.padded_input_tensor_at(0)?;
        let input_shape = input.shape().clone();
        let unpadded_input_shape = input.unpadded_shape().clone();

        let cosine_matrix = Tensor::try_from(&self.cosine_matrix)?.pad_next_power_of_two();
        let sine_matrix = Tensor::try_from(&self.sine_matrix)?.pad_next_power_of_two();

        let cosine_matrix_slice = TensorSlice::from(&cosine_matrix);
        let sine_matrix_slice = TensorSlice::from(&sine_matrix);

        let sub_cos_matrix = cosine_matrix_slice
            .slice_over_first_dim(0, input_shape.dim(-2))
            .to_field();
        let sub_sin_matrix = sine_matrix_slice
            .slice_over_first_dim(0, input_shape.dim(-2))
            .to_field();

        // build sum-check to compute the output as `input*sub_cos_matrix + permuted_input*sub_sin_matrix`
        // The permuted input is the input tensor permuted as follows:
        // given an input tensor `x=[x_1,x_2,x_3,x_4,...,x_n-1,x_n]`,
        // the permuted input tensor is `x' = [-x_2,x_1,-x_4,x_3,...,-x_n,x_n-1]`
        // The input MLE is structured, for variables y_1, y_2, ..., y_m, as
        // x_1*eq(0,y_2..)(1-y_1) + x_2*eq(1,y_2..)y_1 + x_3*eq(2, y_2..)(1-y_1) + x_4*eq(3, y_2..)y_1  + ...
        // The permuted input MLE is structured, for variables y_1, y_2, ..., y_m, as
        // -x_2*eq(0,y_2..)(1-y_1) + x_1*eq(1,y_2..)y_1 - x_4*eq(2, y_2..)(1-y_1) + x_3*eq(3, y_2..)y_1 + ...
        // Therefore, given a claim `input_claim` for the input MLE, it holds that:
        // `input_claim.eval = \sum_{i < n} eq_poly(i) * input_mle(i)`, where `eq_poly(i) = \beta(input_claim.point, i)`
        // Similarly, for a claim `permuted_input_claim` for the permuted input MLE, it holds that:
        // `permuted_input_claim.eval = \sum_{i < n} permuted_eq_poly(i) * input_mle(i)`,
        // where `permuted_eq_poly(i) = \beta(permuted_input_claim.point, i + 1) if i is even, else -\beta(permuted_input_claim.point, i - 1)`.
        // Indeed, if `permuted_input_claim.point = [r_1, r_2, ..., r_m]`, then:
        // - `permuted_eq_poly(0) = eq(1,r_2..)r_1`, which is exactly the term related to `x_1` in the permuted input MLE
        // - `permuted_eq_poly(1) = -eq(0,r_2..)(1-r_1)`, which is exactly the term related to `x_2` in the permuted input MLE
        // and so on for all other n-2 elements of input polynomial.
        // In conclusion, we can prove the correctness of the output computation by running a sumcheck over
        // the relationship `output = eq_poly * input * sub_cos_matrix + permuted_eq_poly * input * sub_sin_matrix`

        // compute MLEs of the tensors involved in the sum-check
        let input_mle = input.to_field_mle();
        let sub_cos_mle = sub_cos_matrix.into_mle();
        let sub_sin_mle = sub_sin_matrix.into_mle();
        ensure!(
            output_claim.point.len() == input_mle.num_vars(),
            "Number of variables of input tensor ({}) different from number of variables of output claim point ({})",
            input_mle.num_vars(),
            output_claim.point.len(),
        );
        // We split the output claim point into its relative dim parts
        let split_point = input_shape.split_point(output_claim.point())?;
        // We need to fix the input poly everywhere apart from the last 2 dimensions
        let fix_vars = split_point
            .iter()
            .take(input_shape.rank() - 2)
            .rev()
            .flat_map(|v| *v)
            .cloned()
            .collect::<Vec<F>>();
        let beta_vars = split_point
            .iter()
            .skip(input_shape.rank() - 2)
            .rev()
            .flat_map(|v| *v)
            .cloned()
            .collect::<Vec<F>>();

        // compute evals of EQ poly for the output claim
        let beta_vec = evals(&beta_vars);
        let input_mle = input_mle.fix_high_variables_parallel(&fix_vars);
        // compute `permuted_eq_poly`: by definition, given the evals of `eq_poly` (i.e., `beta_vec`),
        // it is constructed as `permuted_eq_vec[i] = beta_vec[i+1] if i is even, -beta_vec[i-1] if i is odd`
        let permuted_eq_evals = self.layout.permuted_eq_poly(&beta_vec, input_shape.dim(-1));
        let permuted_eq_mle = permuted_eq_evals.into_mle();
        let eq_mle = beta_vec.into_mle();
        let lt_poly = (0..input_shape.dim(-2))
            .flat_map(|i| {
                if i < unpadded_input_shape.dim(-2) {
                    vec![F::ONE; input_shape.dim(-1)]
                } else {
                    vec![F::ZERO; input_shape.dim(-1)]
                }
            })
            .collect::<Vec<F>>()
            .into_mle();

        let num_vars = input_mle.num_vars();
        let num_threads = optimal_sumcheck_threads(num_vars);
        let mut expr_builder = VirtualPolynomialsBuilder::<F>::new(num_threads, num_vars);
        let input_expr = expr_builder.lift(Either::Left(&input_mle));
        let sub_cos_expr = expr_builder.lift(Either::Left(&sub_cos_mle));
        let sub_sin_expr = expr_builder.lift(Either::Left(&sub_sin_mle));
        let eq_expr = expr_builder.lift(Either::Left(&eq_mle));
        let permuted_eq_expr = expr_builder.lift(Either::Left(&permuted_eq_mle));
        let lt_expr = expr_builder.lift(Either::Left(&lt_poly));

        let virtual_poly = expr_builder.to_virtual_polys(
            &[input_expr * lt_expr * (eq_expr * sub_cos_expr + permuted_eq_expr * sub_sin_expr)],
            &[],
        );
        let (sumcheck_proof, state) = IOPProverState::<F>::prove(virtual_poly, prover.transcript);

        let sumcheck_point = state.collect_raw_challenges();
        let sumcheck_evals = state.get_mle_flatten_final_evaluations()[..3].to_vec();
        let input_eval = sumcheck_evals[0];
        let input_claim = Claim::new(
            sumcheck_point
                .iter()
                .copied()
                .chain(fix_vars)
                .collect::<Vec<F>>(),
            input_eval,
        );

        let sub_cos_claim = Claim::new(sumcheck_point.clone(), sumcheck_evals[1]);
        let sub_sin_claim = Claim::new(sumcheck_point.clone(), sumcheck_evals[2]);

        let (sub_cos_evals, mut cosine_claim) =
            Positional::<N>::bind_sub_claim_to_positional_matrix(
                sub_cos_claim,
                output_claim,
                &cosine_matrix_slice,
                &cosine_matrix,
                input_shape.dim(-2),
                prover.transcript,
            )?;

        self.claim_for_committed_matrix(&mut cosine_claim);

        let (sub_sin_evals, mut sine_claim) = Positional::<N>::bind_sub_claim_to_positional_matrix(
            sub_sin_claim,
            output_claim,
            &sine_matrix_slice,
            &sine_matrix,
            input_shape.dim(-2),
            prover.transcript,
        )?;

        self.claim_for_committed_matrix(&mut sine_claim);

        let proof = RopeProof {
            sub_cosine_evals: sub_cos_evals,
            sub_sine_evals: sub_sin_evals,
            sumcheck_proof,
            sumcheck_evals,
        };

        let commons_claims = [
            (self.cosine_matrix.storage_key().into(), cosine_claim),
            (self.sine_matrix.storage_key().into(), sine_claim),
        ]
        .into_iter()
        .collect();

        prover.add_common_claims(node_id, commons_claims);

        prover.push_proof(
            node_id,
            LayerProof::Positional(PositionalProof::Rope(proof)),
        );

        Ok(vec![input_claim])
    }
}

impl<N> Rope<N>
where
    N: TensorTypeParam,
{
    pub(super) fn evaluate(
        &self,
        input: &WrappedTensor<N>,
        positional_cache: &Arc<Mutex<PositionalCache>>,
    ) -> Result<LayerOut<N>>
    where
        N: TensorTypeParam,
        Add<N>: Evaluate<N>,
    {
        // We allow inputs of rank 2 or greater, the second to last dim is always considered as the sequence length dim
        // and the last dim is the feature dim.
        ensure!(
            input.shape().rank() >= 2,
            "Rope evaluation requires input tensor of rank 2 or greater, got rank {}",
            input.shape().rank(),
        );

        let input = input.clone();

        let input_shape = input.shape();
        let feature_size = input_shape.dims[input_shape.rank() - 1];
        let current_seq_len = input_shape.dims[input_shape.rank() - 2];

        let cosine_matrix = self.cosine_matrix.wrapped_tensor()?;
        let sine_matrix = self.sine_matrix.wrapped_tensor()?;

        let (cosine_matrix, sine_matrix) = (cosine_matrix.clone(), sine_matrix.clone());

        let past_length = positional_cache.lock().unwrap().seq_len;
        let cosine_slice_bt = cosine_matrix.slice([
            past_length..(past_length + current_seq_len),
            0..feature_size,
        ]);
        let sine_slice_bt = sine_matrix.slice([
            past_length..(past_length + current_seq_len),
            0..feature_size,
        ]);

        positional_cache
            .lock()
            .unwrap()
            .set_seq_len(past_length + current_seq_len)?;

        let second_input = self.layout.split_input(input.clone());

        let output = match input.rank() {
            2 => input
                .mul(cosine_slice_bt)?
                .add(second_input.mul(sine_slice_bt)?)?,
            3 => input
                .mul(cosine_slice_bt.unsqueeze_dim(0)?)?
                .add(second_input.mul(sine_slice_bt.unsqueeze_dim(0)?)?)?,
            4 => input
                .mul(cosine_slice_bt.unsqueeze_dim(0)?.unsqueeze_dim(0)?)?
                .add(second_input.mul(sine_slice_bt.unsqueeze_dim(0)?.unsqueeze_dim(0)?)?)?,
            _ => anyhow::bail!(
                "Rope evaluation not implemented for input tensors of rank greater than 4, got rank {}",
                input.rank()
            ),
        };
        if let Some(concat_cache) = &self.concatenation_cache {
            let mut cache = concat_cache.lock().unwrap();

            let cached = cache.concatenate(output)?;
            Ok(LayerOut::from_tensor(cached))
        } else {
            Ok(LayerOut::from_tensor(output))
        }

        // Old vs new implementation
        // Let input be: x = [x_0, x_1, x_2, x_3, ..., x_{2k}, x_{2k+1}, ..., x_{d-2}, x_{d-1}] with even d.
        // Let cosine and sine generated by build_from_angles() be:
        //   c = [c_0, c_0, c_1, c_1, ..., c_k, c_k, ..., c_{d/2-1}, c_{d/2-1}]
        //   s = [s_0, s_0, s_1, s_1, ..., s_k, s_k, ..., s_{d/2-1}, s_{d/2-1}]
        //
        // OLD:
        //   n(x) = [-x_1, x_0, -x_3, x_2, ..., -x_{2k+1}, x_{2k}, ..., -x_{d-1}, x_{d-2}].
        // Element-wise output:
        //   y = x * c + n(x) * s
        // Expanding a single pair with indices (2k, 2k+1):
        //   y_{2k}   =  x_{2k} * c_k + (-x_{2k+1}) * s_k =  x_{2k} c_k - x_{2k+1} s_k
        //   y_{2k+1} =  x_{2k+1} * c_k + ( x_{2k})   * s_k =  x_{2k+1} c_k + x_{2k} s_k
        //
        // NEW:
        // Instead of constructing n(x), reshape x, c, s to [rows, d/2, 2] so that each last-dimension pair
        // corresponds to one rotational block. From the reshaped tensors we define:
        //   x1 = x[:,:,0]  (even positions  2k)
        //   x2 = x[:,:,1]  (odd  positions  2k+1)
        //   c1 = c[:,:,0]; c2 = c[:,:,1]
        //   s1 = s[:,:,0]; s2 = s[:,:,1]
        // Note: c1 = c2 and s1 = s2 because of the way the cosine and sine matrices are built
        // Compute the two rotation components per pair:
        //   out_even = x1 * c1 - x2 * s1        (gives y_{2k})
        //   out_odd  = x2 * c2 + x1 * s2        (gives y_{2k+1})
        // Because of duplication (c1==c2, s1==s2) these match the expanded formulas above.
    }
}

impl Rope<f32> {
    pub(super) fn quantize<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Rope<Element>>> {
        let cosine_matrix = self.cosine_matrix.wrapped_tensor()?;
        let sine_matrix = self.sine_matrix.wrapped_tensor()?;

        // compute scaling factor for cosine and sine matrices
        let max_cos = cosine_matrix.clone().max().get_data()[0];
        let max_sin = sine_matrix.clone().max().get_data()[0];

        // check that the maximum values are close
        if !is_close_with_tolerance(&[max_cos], &[max_sin], 1e-3, 0.0) {
            warn!(
                "Maximum values are too distant for cosine/sine matrices in positional rope with id {node_id}"
            );
        }

        let min_cos = cosine_matrix.clone().min().get_data()[0];
        let min_sin = sine_matrix.clone().min().get_data()[0];

        // check that the minimum values are close
        if !is_close_with_tolerance(&[min_cos], &[min_sin], 1e-3, 0.0) {
            warn!(
                "Minimum values are too distant for cosine/sine matrices in positional rope with id {node_id}"
            );
        }

        // compute the scaling factor for both matrices
        let matrix_scale =
            ScalingFactor::from_span(min_cos.min(min_sin), max_cos.max(max_sin), None);

        let output_scalings = S::scaling_factors_for_node(data, node_id, 1);
        ensure!(
            output_scalings.len() == 1,
            "Expected 1 output scaling factor for Positional layer {node_id}, found {}",
            output_scalings.len(),
        );
        let output_scaling = &output_scalings[0];
        // in the layer, given a row of input tensor `x``, a row of cosine matrix `c` and a row of sine matrix `s``,
        // we compute the corresponding row in the output tensor `out` as:
        // out = x*c + pi(x)*s, where `pi(x)` is a permutation of input vector x.
        // We are using the same scaling factor `matrix_scale` for `c` and `s`, and `pi(x)` has the same
        // scaling factor `input_scaling` of `x`, as it is simply a permutation of `x`.
        // So, both the element-wise multiplications in the above equation are between an item with
        // scaling factor `input_scaling` and an item with scaling factor `matrix_scale`.
        // Therefore, we can use the multiplier `input_scaling*matrix_scale/output_scaling` to requantize
        let multiplier = input_scaling.m(&matrix_scale, output_scaling);
        let output_bit_size = 2 * *quantization::BIT_LEN + 1; // +1 because we are adding 2 products of items with `quantization::BIT_LEN` bits
        let requant = Requant::from_multiplier(multiplier, output_bit_size, *output_scaling);

        let concatenation_cache = self.concatenation_cache.map(|cache| {
            let mut cache = cache.lock().unwrap();
            cache.reset();
            let (rank, concatenation_dim) = cache.cache_info();
            Arc::new(Mutex::new(ConcatenationCache::<Element>::new(
                rank,
                concatenation_dim,
            )))
        });

        let quantized_rope = Rope {
            cosine_matrix: self.cosine_matrix.quantize(&matrix_scale),
            sine_matrix: self.sine_matrix.quantize(&matrix_scale),
            unpadded_shape: self.unpadded_shape,
            layout: self.layout,
            concatenation_cache,
        };

        Ok(QuantizeOutput::new(quantized_rope, output_scalings).with_requant(requant)?)
    }
}

impl PadOp for Rope<Element> {
    fn pad_node(self, _si: &mut crate::padding::ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

impl Rope<Element> {
    pub(super) fn step_info(&self, mut aux: ContextAux) -> anyhow::Result<(RopeCtx, ContextAux)> {
        // this closure retains only values in odd columns, relying on the fact that `self.cosine_matrix`
        // and `self.sine_matrix` have pairwise identical elements in each column. In this way, the size
        // of the polynomial being committed is halved

        let matrix_to_evals = |matrix: &Tensor<Element>| match self.layout {
            RopeLayout::Adjacent => matrix.data().chunks(2).map(|chunk| chunk[0]).collect_vec(),
            RopeLayout::RotateHalf => {
                let rows = matrix.shape().dim(-1);
                let half_rows = rows / 2;
                matrix
                    .data()
                    .chunks(rows)
                    .flat_map(|chunk| chunk[..half_rows].to_vec())
                    .collect::<Vec<Element>>()
            }
            RopeLayout::RotateBlock(block) => matrix
                .data()
                .chunks(block)
                .flat_map(|chunk| chunk[..block / 2].iter().copied())
                .collect_vec(),
        };
        let wrapped_cosine_matrix = self.cosine_matrix.wrapped_tensor()?.clone();
        let cosine_matrix = Tensor::try_from(&wrapped_cosine_matrix)?.pad_next_power_of_two();
        let wrapped_sine_matrix = self.sine_matrix.wrapped_tensor()?.clone();
        let sine_matrix = Tensor::try_from(&wrapped_sine_matrix)?.pad_next_power_of_two();
        aux.model_polys = Some(
            [
                (
                    CommitmentId::from(self.cosine_matrix.storage_key()),
                    matrix_to_evals(cosine_matrix.as_ref()),
                ),
                (
                    CommitmentId::from(self.sine_matrix.storage_key()),
                    matrix_to_evals(sine_matrix.as_ref()),
                ),
            ]
            .into_iter()
            .collect(),
        );

        let num_vars = cosine_matrix.shape().num_vars().into_iter().sum();
        let ctx = RopeCtx {
            unpadded_shape: self.unpadded_shape.clone(),
            num_vars_positional_matrix: num_vars,
            cosine_key: CommitmentId::from(self.cosine_matrix.storage_key()),
            sine_key: CommitmentId::from(self.sine_matrix.storage_key()),
            layout: self.layout,
        };

        Ok((ctx, aux))
    }
}

impl RopeCtx {
    pub(super) fn verify<F: PrimeField, T: Transcript, PCS: CommitmentScheme<Field = F>>(
        &self,
        proof: &RopeProof<F>,
        last_claim: &Claim<F>,
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        let input_rank = shape_step.padded_input_shape[0].rank();

        let dim_vars = shape_step.padded_input_shape[0].num_vars();

        let input_num_vars = shape_step.padded_input_shape[0]
            .num_vars()
            .into_iter()
            .skip(input_rank - 2)
            .sum();
        let aux_info = from_mle_list_dimensions(&[
            vec![
                input_num_vars,
                input_num_vars,
                input_num_vars,
                input_num_vars,
            ],
            vec![
                input_num_vars,
                input_num_vars,
                input_num_vars,
                input_num_vars,
            ],
        ]);
        let subclaim = IOPVerifierState::verify(
            last_claim.eval,
            &proof.sumcheck_proof,
            &aux_info,
            verifier.transcript,
        );
        let sumcheck_point = &subclaim.point;

        let beta_eval = identity_eval(&last_claim.point()[..input_num_vars], sumcheck_point);
        let row_num_vars = dim_vars[input_rank - 1];
        let permuted_eq_eval = self.layout.evaluate_permuted_eq_poly(
            &last_claim.point()[..input_num_vars],
            sumcheck_point,
            row_num_vars,
        );

        let unpadded_seq_len = shape_step.unpadded_input_shape[0].dim(input_rank - 2);

        let lt_point_len = shape_step.padded_input_shape[0].num_vars()[input_rank - 2];
        let lt_bits = to_bit_sequence_le(unpadded_seq_len - 1, lt_point_len)
            .map(|b| F::from(b as u64))
            .collect::<Vec<F>>();
        let lt_eval = eval_zeroifier_mle(
            &sumcheck_point[sumcheck_point.len() - lt_point_len..],
            &lt_bits,
        );

        // verify sum-check evaluation
        ensure!(
            proof.sumcheck_evals.len() == 3,
            "expected 3 evaluations for sumcheck in Positional::Rope proof, found {}",
            proof.sumcheck_evals.len(),
        );
        let input_eval = proof.sumcheck_evals[0];
        let cosine_eval = proof.sumcheck_evals[1];
        let sine_eval = proof.sumcheck_evals[2];
        ensure!(
            subclaim.expected_evaluation
                == input_eval * lt_eval * (beta_eval * cosine_eval + permuted_eq_eval * sine_eval),
            "Sumcheck verification failed for Positional::Rope verifier"
        );

        let input_claim = Claim::new(
            sumcheck_point
                .iter()
                .chain(last_claim.point()[input_num_vars..].iter())
                .copied()
                .collect::<Vec<F>>(),
            input_eval,
        );

        let sub_cos_claim = Claim::new(sumcheck_point.clone(), cosine_eval);
        let sub_sine_claim = Claim::new(sumcheck_point.clone(), sine_eval);

        let mut cosine_matrix_claim = PositionalCtx::build_positional_matrix_claim(
            sub_cos_claim,
            last_claim,
            self.num_vars_positional_matrix,
            verifier.transcript,
            &proof.sub_cosine_evals,
        )?;

        self.claim_for_committed_matrix(&mut cosine_matrix_claim);

        let mut sine_matrix_claim = PositionalCtx::build_positional_matrix_claim(
            sub_sine_claim,
            last_claim,
            self.num_vars_positional_matrix,
            verifier.transcript,
            &proof.sub_sine_evals,
        )?;

        self.claim_for_committed_matrix(&mut sine_matrix_claim);

        verifier.add_common_claims(
            node_id,
            [
                (self.cosine_key.clone(), cosine_matrix_claim),
                (self.sine_key.clone(), sine_matrix_claim),
            ]
            .into_iter()
            .collect(),
        );

        Ok(vec![input_claim])
    }

    fn claim_for_committed_matrix<F: PrimeField>(&self, claim: &mut Claim<F>) {
        match self.layout {
            RopeLayout::Adjacent => claim.point.remove(0),
            RopeLayout::RotateBlock(block) => claim.point.remove(block.ilog2() as usize - 1),
            RopeLayout::RotateHalf => {
                let row_vars = ceil_log2(self.unpadded_shape.dim(-1));
                claim.point.remove(row_vars - 1)
            }
        };
    }
}

// compute evaluation of the `permute_eq_poly` employed in proving.
// For a claim point `r_c`, the `permute_eq_poly` is constructed over
// `n`` variables as:
// - `permute_eq_poly[i] = \beta(r_c, i+1) if i is even`
// - `permute_eq_poly[i] = -\beta(r_c, i-1) if i is odd`
// This method evalautes `permuted_eq_poly(input)` given an input point
// with `n` coordinates and the claim point `r_c` employed to construct
// `permuted_eq_poly`
fn evaluate_permutation_eq_poly<F: PrimeField>(claim_point: &[F], input: &[F]) -> F {
    // Given the coordinates `r_1, ..., r_n` of `claim_point` and the `n`
    // input variables `y_1,...,y_n`, we have:
    // - `permute_eq_poly[i]` = eq(i,r_2..)*r_1*(1-y_1) when i is even
    // - `permute_eq_poly[i]` = -eq(i,r_2..)*(1-r_1)*y_1 when i is odd
    // Since all these terms are summed in the MLE, we obtain:
    // eq(i,r_2..)[r_1*(1-y_1) - y_1*(1-r_1)] = eq(i,r_2..)[r_1 - y_1]
    // so, we split the evaluation in 2 terms:
    // - the first term `eq(i,r_2..)`, which involves all the variables
    //  besides the first one, and it is the same as computing the EQ
    //  polynomial for these variables
    // - the term related to the first variable, which is what differentiates
    //  the `permuted_eq_poly` from a generic EQ polynomial
    let eq_eval = identity_eval(&claim_point[1..], &input[1..]);
    eq_eval * (claim_point[0] - input[0])
}

#[cfg(test)]
mod tests {
    use std::f32::consts::PI;

    use ark_std::rand::Rng;
    use itertools::Itertools;

    use rstest::rstest;

    use tenstore::GenStore;

    use crate::{
        ScalingFactor, Tensor,
        layers::{
            Layer,
            transformer::positional::{
                Positional, PositionalCache,
                rope::{Rope, RopeLayout},
            },
        },
        model::{Model, test::prove_model},
        quantization::{AbsoluteMax, Quantize},
        rng_from_env_or_random,
        tensor::{TensorTypeParam, is_close_with_tolerance},
    };

    use proptest::prelude::*;
    use std::sync::{Arc, Mutex};

    type F = ark_bn254::Fr;

    fn random_angles(num_angles: usize) -> Vec<f32> {
        let mut rng = rng_from_env_or_random();
        (0..num_angles)
            .map(|_| rng.gen_range(0.0..2.0 * PI))
            .collect_vec()
    }

    #[test]
    fn qwen_partial_rope_identity_tail_and_cached_parity() -> anyhow::Result<()> {
        // First four of sixteen features rotate; zero-frequency blocks stay exact.
        let angles = vec![1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let rope = Rope::<f32>::build_from_angles(
            angles,
            "partial_rope".to_string().into(),
            8,
            RopeLayout::RotateBlock(4),
        )?;
        let input = Tensor::new(
            vec![5, 16].into(),
            (0..80).map(|x| x as f32 / 8.0 - 3.0).collect(),
        )?;
        let whole_cache = Arc::new(Mutex::new(PositionalCache::new()));
        let full = rope.evaluate(&input.as_wrapped(), &whole_cache)?.outputs[0].get_data();
        let cache = Arc::new(Mutex::new(PositionalCache::new()));
        let prefix = Tensor::new(vec![2, 16].into(), input.data()[..32].to_vec())?;
        let mut incremental = rope.evaluate(&prefix.as_wrapped(), &cache)?.outputs[0].get_data();
        for t in 2..5 {
            let row = Tensor::new(
                vec![1, 16].into(),
                input.data()[t * 16..(t + 1) * 16].to_vec(),
            )?;
            incremental.extend(rope.evaluate(&row.as_wrapped(), &cache)?.outputs[0].get_data());
        }
        assert_eq!(full, incremental);
        for t in 0..5 {
            assert_eq!(
                &full[t * 16 + 4..(t + 1) * 16],
                &input.data()[t * 16 + 4..(t + 1) * 16]
            );
            for pair in 0..2 {
                let phase = t as f32 * [1.0, 0.1][pair];
                let a = input.data()[t * 16 + pair];
                let b = input.data()[t * 16 + pair + 2];
                assert!((full[t * 16 + pair] - (a * phase.cos() - b * phase.sin())).abs() < 1e-6);
                assert!(
                    (full[t * 16 + pair + 2] - (b * phase.cos() + a * phase.sin())).abs() < 1e-6
                );
            }
        }
        Ok(())
    }

    #[test]
    fn qwen_partial_rope_proves_with_padded_sequence_and_heads() -> anyhow::Result<()> {
        for block in [2, 4] {
            let mut angles = vec![0.0; 8];
            for (i, angle) in angles.iter_mut().take(block / 2).enumerate() {
                *angle = 10000.0f32.powf(-2.0 * i as f32 / block as f32);
            }
            let mut model = Model::new_from_input_shapes(vec![vec![3, 5, 16].into()]);
            model.add_consecutive_layer(
                Layer::Positional(Positional::new_rope(
                    angles,
                    format!("partial_{block}").into(),
                    8,
                    RopeLayout::RotateBlock(block),
                )?),
                None,
            )?;
            model.automatic_output_labelling()?;
            prove_model(model, &mut GenStore::default())?;
        }
        Ok(())
    }

    #[test]
    fn test_cosine_and_sine_bounds() {
        let mut rng = rng_from_env_or_random();
        for _ in 0..20 {
            let num_angles = rng.gen_range(16..768);
            let angles = random_angles(num_angles);
            let max_context_length = rng.gen_range(2..1024);
            let _rope = Rope::<f32>::build_from_angles(
                angles,
                "rope_angles".to_string().into(),
                max_context_length,
                RopeLayout::Adjacent,
            )
            .unwrap();
        }
    }

    #[rstest]
    #[case::less_input_than_context_length_adj(14, 16, 31, RopeLayout::Adjacent)]
    #[case::less_input_than_context_length_rotate_half(14, 16, 31, RopeLayout::RotateHalf)]
    #[case::same_input_as_context_length_adj(31, 16, 31, RopeLayout::Adjacent)]
    #[case::same_input_as_context_length_rotate_half(31, 16, 31, RopeLayout::RotateHalf)]
    fn test_proven_rope_positional_layer(
        #[case] seq_len: usize,
        #[case] embedding_size: usize,
        #[case] context_length: usize,
        #[case] layout: RopeLayout,
    ) {
        let extra_dim = 6;
        let input_shape = vec![extra_dim, seq_len, embedding_size];

        let mut model = Model::new_from_input_shapes(vec![input_shape.into()]);

        // build angles for rotational matrix
        assert!(embedding_size.is_power_of_two());
        let angles = random_angles(embedding_size / 2);

        let _ = model
            .add_consecutive_layer(
                Layer::Positional(
                    Positional::new_rope(
                        angles,
                        "rope_angles".to_string().into(),
                        context_length,
                        layout,
                    )
                    .unwrap(),
                ),
                None,
            )
            .unwrap();

        model.automatic_output_labelling().unwrap();

        let _ = prove_model(model, &mut GenStore::default()).unwrap();
    }

    #[rstest]
    #[case::adjacent(RopeLayout::Adjacent, 18)]
    #[case::split_half_power_of_2(RopeLayout::RotateHalf, 32)]
    fn test_padded_evaluation_consistency(
        #[case] variant: RopeLayout,
        #[case] hidden_size: usize,
    ) -> anyhow::Result<()> {
        use crate::{
            layers::transformer::positional::PositionalVariant, model::test::quantize_model,
            padding::pad_model,
        };

        let seq_len = 12;
        let num_heads = 14;
        let context_length = 27;

        let input_shape = vec![num_heads, seq_len, hidden_size].into();
        let mut model = Model::new_from_input_shapes(vec![input_shape]);

        // build angles for rotational matrix
        assert!(hidden_size.is_multiple_of(2));
        let angles = random_angles(hidden_size / 2);

        let _ = model.add_consecutive_layer(
            Layer::Positional(Positional::new_from_variant(PositionalVariant::Rope(
                Rope::build_from_angles(
                    angles,
                    "rope_angles".to_string().into(),
                    context_length,
                    variant,
                )?,
            ))),
            None,
        )?;

        model.automatic_output_labelling()?;

        let mut store = Default::default();
        let (quantized_model, _) = quantize_model(model, vec![], None, &mut store)?;

        let inputs = quantized_model
            .input_shapes()
            .into_iter()
            .map(|shape| Tensor::random(&shape))
            .collect_vec();

        // run quantized model and get the output
        let trace = quantized_model.run(inputs.clone(), &mut store)?;
        let io = trace.to_verifier_io::<F>()?;

        // pad the model
        let padded_model = pad_model(quantized_model)?;
        padded_model.reset();

        let padded_inputs = padded_model.prepare_inputs(inputs)?;

        // run the padded model
        let padded_trace = padded_model.run(padded_inputs, &mut Default::default())?;

        let padded_io = padded_trace.to_verifier_io::<F>()?;

        let output = &io.output[0];
        let padded_output = &padded_io.output[0];

        let padded_out_shape = padded_output.shape();
        let padded_shape_strides = padded_out_shape.strides();
        let unpadded_out_shape = output.shape();
        let unpadded_shape_strides = unpadded_out_shape.strides();

        for i in 0..padded_out_shape[0] {
            for j in 0..padded_out_shape[1] {
                for k in 0..padded_out_shape[2] {
                    if i < unpadded_out_shape[0]
                        && j < unpadded_out_shape[1]
                        && k < unpadded_out_shape[2]
                    {
                        // check that corresponding entries in `output` and `padded_output` are the same
                        let out_data = &output.data()
                            [i * unpadded_shape_strides[0] + j * unpadded_shape_strides[1] + k];
                        let padded_out_data = &padded_output.data()
                            [i * padded_shape_strides[0] + j * padded_shape_strides[1] + k];
                        assert_eq!(out_data, padded_out_data);
                    }
                }
            }
        }

        Ok(())
    }

    #[derive(Clone)]
    struct Input<T> {
        seq_len: usize,
        embedding_size: usize,
        context_length: usize,
        input: Tensor<T>,
        angles: Vec<f32>,
        layout: RopeLayout,
    }

    impl<T: core::fmt::Debug> core::fmt::Debug for Input<T> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Input")
                .field("seq_len", &self.seq_len)
                .field("embedding_size", &self.embedding_size)
                .field("context_length", &self.context_length)
                .finish_non_exhaustive()
        }
    }

    fn rope_input<T: TensorTypeParam>() -> impl Strategy<Value = Input<T>> {
        (1..32usize, 4..8usize).prop_flat_map(|(seq_len, log_embed)| {
            let embedding_size = 1 << (log_embed + 1);
            let layout_strategy =
                prop_oneof![Just(RopeLayout::Adjacent), Just(RopeLayout::RotateHalf),];
            (
                seq_len..=64usize,
                Tensor::<T>::any(vec![seq_len, embedding_size].into()),
                proptest::collection::vec(0.0..2.0 * PI, embedding_size / 2),
                layout_strategy,
            )
                .prop_map(move |(context_length, input, angles, layout)| Input {
                    seq_len,
                    embedding_size,
                    context_length,
                    input,
                    angles,
                    layout,
                })
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]

        #[test]
        fn test_rope_f32(inp in rope_input::<f32>()) {
            let Input { seq_len, embedding_size, context_length, input, angles, layout } = inp.clone();
            prop_assume!(embedding_size.is_power_of_two() && embedding_size >= 2);
            let layer = Rope::<f32>::build_from_angles(angles.clone(), "rope_angles".to_string().into(), context_length, layout).expect("build rope");
            let cache = Arc::new(Mutex::new(PositionalCache::new()));

            let out = layer
                .evaluate(&input.as_wrapped(), &cache)
                .expect("rope evaluate").outputs.into_iter().next().unwrap();

            let mut expected = Vec::with_capacity(seq_len * embedding_size);
            let in_data = input.data();
            for row in 0..seq_len {
                let mut tmp_row = vec![0.0f32; embedding_size];
                for (k, angle) in angles.iter().enumerate() {
                    let angle = *angle * row as f32;
                    let cosv = angle.cos();
                    let sinv = angle.sin();
                    let (p1, p2) = match layout {
                        RopeLayout::Adjacent => { (2 * k, 2 * k + 1) },
                        RopeLayout::RotateHalf => { (k, embedding_size / 2 + k) },
                        RopeLayout::RotateBlock(block) => {
                            let start = (k / (block / 2)) * block;
                            (start + k % (block / 2), start + block / 2 + k % (block / 2))
                        },
                    };
                    let x1 = in_data[row * embedding_size + p1];
                    let x2 = in_data[row * embedding_size + p2];
                    tmp_row[p1] = x1 * cosv - x2 * sinv;
                    tmp_row[p2] = x2 * cosv + x1 * sinv;
                }
                expected.extend(tmp_row);
            }
            let expected_t = Tensor::new(vec![seq_len, embedding_size].into(), expected).unwrap();
            let out = is_close_with_tolerance(&out.get_data(), expected_t.data(), 1e-5, 1e-4);
            prop_assert!(out, "RoPE with layout {:?} produced incorrect output", layout);
        }

        #[test]
        fn test_rope_quantized(inp in rope_input::<f32>()) {
            let Input { seq_len, embedding_size, context_length, input, angles, layout } = inp.clone();
            prop_assume!(embedding_size.is_power_of_two()  && embedding_size >= 2);

            let layer = Rope::<f32>::build_from_angles(angles.clone(), "rope_angles".to_string().into(), context_length, layout).expect("build rope");
            let input_sf = ScalingFactor::from_tensor(&input, None);
            let q = layer.quantize::<AbsoluteMax>(&(), 0.into(), input_sf).expect("quantize rope");
            let layer_q = q.quantized_op;
            let input_q = input.quantize(&input_sf);
            let cache = Arc::new(Mutex::new(PositionalCache::new()));

            let out_q = layer_q
                .evaluate(&input_q.as_wrapped(), &cache)
                .expect("rope evaluate quant").outputs.into_iter().next().unwrap();

            prop_assert_eq!(out_q.shape(), vec![seq_len, embedding_size].into());
        }

        #[test]
        fn test_rope_proving_prop(inp in rope_input::<f32>()) {
            let Input { seq_len, embedding_size, context_length, input: _, angles, layout} = inp.clone();
            prop_assume!(embedding_size.is_power_of_two() && embedding_size >= 2);
            prop_assume!(seq_len >= 2);

            let input_shape = vec![seq_len, embedding_size];
            let mut model = Model::new_from_input_shapes(vec![input_shape.into()]);

            model.add_consecutive_layer(Layer::Positional(Positional::new_rope(angles, "rope_angles".to_string().into(), context_length, layout).expect("rope")), None).expect("rope layer");
            model.automatic_output_labelling().expect("route output");
            let _ = prove_model(model, &mut GenStore::default()).expect("prove model");
        }
    }
}
