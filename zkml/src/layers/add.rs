use crate::{
    Claim, Element, NextPowerOfTwo, Number, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        activation::lookup_data::ActivationLookupData,
        provable::{
            Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
            Splittable, VerifiableCtx,
        },
        requant::{FIXED_POINT_SCALE, Requant},
    },
    lookup::table::Table,
    model::Step,
    padding::{PaddingMode, ShapeInfo},
    quantization::{self, ToField},
    tensor::{CommitmentId, TensorTypeParam, WrappedTensor},
};
use anyhow::{Context, Result, bail, ensure};
use ark_ff::PrimeField;
use dp_crypto::{
    arkyper::{CommitmentScheme, transcript::Transcript},
    util::ceil_log2,
};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap, marker::PhantomData};

/// The short name used to identify the Add layer.
pub const ADD_LAYER: &str = "_ADD";

/// Add layer that adds two tensors together.
/// If there is two inputs, no static weight, then the output shape is the same as the first input.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Add<N> {
    quant_info: Option<AddQuantInfo>,
    phantom: PhantomData<N>,
}

impl<N: TensorTypeParam> Default for Add<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Context info for the add layer.
/// NOTE: In LLM, we assume the same scaling info regardless of the sequence length.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AddCtx {
    quant_info: AddQuantInfo,
    operand_key: Option<CommitmentId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct AddProof<F> {
    #[serde(with = "dp_crypto::serialization")]
    left_eval: F,
    #[serde(with = "dp_crypto::serialization")]
    right_eval: F,
}

impl<N: TensorTypeParam> Add<N> {
    pub fn new() -> Self {
        Self {
            quant_info: None,
            phantom: PhantomData {},
        }
    }
}

impl Add<Element> {
    pub(crate) fn prove_step<A, F, T, PCS>(
        &self,
        _node_id: NodeId,
        last_claims: Vec<&Claim<F>>,
        inputs: &[A],
        _prover: &mut Prover<F, T, PCS>,
    ) -> anyhow::Result<(Vec<Claim<F>>, AddProof<F>)>
    where
        PCS: CommitmentScheme<Field = F>,
        A: AsRef<Tensor<Element>>,
        F: PrimeField,
        T: Transcript,
    {
        ensure!(last_claims.len() == 1, "Add layer expects 1 claim");
        let last_claim = last_claims[0];
        ensure!(self.quant_info.is_some(), "Add layer is not quantized");
        // assuming last_claim is f(r) = y
        // we want to prove that x1(r) + x2(r) = y
        // in the case there is no operand, we output two claims, x1(r) and x2(r)
        // in the case there is an operand, we output one claim, x1(r) and we
        // add the claim OPERAND(r) to the list of claims to verify via the committed weights PCS.
        // Regarding the scaling operation, we actually want to prove
        // that x1(r) * M1 / 2^shift1 + x2(r) * M2 / 2^shift2 = y, so the prover outputs only x1(r) and x2(r)
        // and the verifier will "scale" the claims accordingly to check the equation.
        let left_input = inputs[0].as_ref();
        let left_eval = left_input.to_field_mle().evaluate(&last_claim.point)?;
        let mut output_claims = vec![Claim::new(last_claim.point.clone(), left_eval)];
        let right_eval = inputs[1]
            .as_ref()
            .to_field_mle()
            .evaluate(&last_claim.point)?;
        // this claims gets passed to the previous layer alongside the left one.
        output_claims.push(Claim::new(last_claim.point.clone(), right_eval));

        let proof = AddProof {
            left_eval,
            right_eval,
        };

        Ok((output_claims, proof))
    }
}

impl Evaluate<f32> for Add<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> anyhow::Result<LayerOut<f32>> {
        ensure!(
            inputs.len() == 2,
            "Add layer with an operand expects two inputs. got: {}",
            inputs.len()
        );
        let left = inputs[0].clone();
        let right = inputs[1].clone();

        let left_shape = left.shape();
        let right_shape = right.shape();

        ensure!(
            left_shape.num_elements() == right_shape.num_elements(),
            "Add layer expects inputs to have the same shape: {left_shape:?} vs {right_shape:?}",
        );

        let result = left.add(right)?;
        Ok(LayerOut::from_tensor(result))
    }
}

impl Evaluate<Element> for Add<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> anyhow::Result<LayerOut<Element>> {
        ensure!(
            inputs.len() == 2,
            "Add layer expects 2 inputs if there is no operand"
        );
        let left = inputs[0].clone();
        let right = inputs[1].clone();

        let quant_info = self
            .quant_info
            .as_ref()
            .context("Add layer is not quantized")?;

        let left_scaled = left.mul_scalar(quant_info.left_scale());
        let right_scaled = right.mul_scalar(quant_info.right_scale());
        let result = left_scaled.add(right_scaled)?;
        Ok(LayerOut::from_tensor(result))
    }
}

impl<N> OpInfo for Add<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        assert!(
            input_shapes.len() == 2,
            "Add layer expects 2 inputs if there is no operand"
        );
        assert!(
            input_shapes[0] == input_shapes[1],
            "Add layer input shapes mismatch: {:?} vs {:?}",
            input_shapes[0],
            input_shapes[1],
        );
        match padding_mode {
            PaddingMode::NoPadding => Ok(vec![input_shapes[0].clone()]),
            PaddingMode::Padding => Ok(vec![input_shapes[0].next_power_of_two()]),
        }
    }

    fn num_outputs(&self, _num_inputs: usize) -> Result<usize> {
        Ok(1)
    }

    fn describe(&self) -> String {
        format!("Add {:?}", self.quant_info)
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl OpInfo for AddCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match padding_mode {
            PaddingMode::NoPadding => Ok(input_shapes.to_vec()),
            PaddingMode::Padding => Ok(input_shapes
                .iter()
                .map(|shape| shape.next_power_of_two())
                .collect()),
        }
    }

    fn num_outputs(&self, _num_inputs: usize) -> Result<usize> {
        Ok(1)
    }

    fn describe(&self) -> String {
        "Add".to_string()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
/// Quantization info for the add layer.
/// When we perform quantised addition between two tensors A and B we need both tensors to be quantised with the same
/// [`ScalingFactor`]. Often this is not the case and so we use [`AddQuantInfo`] to calculate a suitable common [`ScalingFactor`].
pub struct AddQuantInfo {
    left_multiplier: Element,
    right_multiplier: Element,
    right_shift: usize,
    intermediate_bit_size: usize,
    pub(crate) output_scaling: ScalingFactor,
}

impl AddQuantInfo {
    fn scaled_sum_bit_size(
        left_multiplier: Element,
        left_input_bits: usize,
        right_multiplier: Element,
        right_input_bits: usize,
    ) -> usize {
        let product_bits = |multiplier: Element, input_bits| {
            if multiplier == 0 {
                0
            } else {
                ceil_log2(multiplier.unsigned_abs() as usize) + input_bits
            }
        };
        // Pair each multiplier with its own input. Combining the largest
        // multiplier with the widest *other* input can erase that branch when
        // the common fixed-point multipliers are shifted down to fit i64.
        let max_product_bits = product_bits(left_multiplier, left_input_bits)
            .max(product_bits(right_multiplier, right_input_bits));
        // Preserve the addition/sign margin and the bookkeeping floor used by
        // requant_from_add when it removes FIXED_POINT_SCALE from this bound.
        (max_product_bits + 2).max(FIXED_POINT_SCALE)
    }

    pub fn new(
        left_scaling: &ScalingFactor,
        right_scaling: &ScalingFactor,
        output_scaling: &ScalingFactor,
    ) -> Self {
        let left_rescale = left_scaling.scale() / output_scaling.scale();
        let right_rescale = right_scaling.scale() / output_scaling.scale();

        let left_log = left_rescale.log2();
        let right_log = right_rescale.log2();

        let left_int = left_log.trunc();
        let right_int = right_log.trunc();

        let left_fract = left_log.fract();
        let right_fract = right_log.fract();

        // TO work out the overall shift we would need to apply to both inputs we subtract FIXED_POINT_SCALE from both
        let left_input_shift = left_int - FIXED_POINT_SCALE as f32;
        let right_input_shift = right_int - FIXED_POINT_SCALE as f32;

        let (mut final_right_shift, mut left_multiplier, mut right_multiplier) =
            match Number::compare(&left_input_shift, &right_input_shift) {
                Ordering::Less => {
                    // The left input has a "smaller" shift than the right input, so we work out what we need to subtract from the right input to make it equal to the left input
                    let shift_diff = right_input_shift - left_input_shift;
                    // shift_diff will always be positive here so the right multiplier is 2^shift_diff * 2^right_fract
                    let right_multiplier = (2.0f32.powf(right_fract)
                        * ((1u64 << (FIXED_POINT_SCALE + shift_diff.trunc().abs() as usize))
                            as f32))
                        .round_ties_even() as Element;
                    let left_multiplier = (2.0f32.powf(left_fract)
                        * (1u64 << FIXED_POINT_SCALE) as f32)
                        .round_ties_even() as Element;
                    (
                        left_input_shift.trunc().abs() as usize,
                        left_multiplier,
                        right_multiplier,
                    )
                }
                Ordering::Equal => {
                    // No shift diff here so both multipliers are just 2^fract * 2^FIXED_POINT_SCALE
                    let right_multiplier = (2.0f32.powf(right_fract)
                        * (1u64 << FIXED_POINT_SCALE) as f32)
                        .round_ties_even() as Element;
                    let left_multiplier = (2.0f32.powf(left_fract)
                        * (1u64 << FIXED_POINT_SCALE) as f32)
                        .round_ties_even() as Element;
                    (
                        left_input_shift.trunc().abs() as usize,
                        left_multiplier,
                        right_multiplier,
                    )
                }
                Ordering::Greater => {
                    // The right input has a "smaller" shift than the left input, so we work out what we need to subtract from the left input to make it equal to the right input
                    let shift_diff = left_input_shift - right_input_shift;
                    // shift_diff will always be positive here so the left multiplier is 2^shift_diff * 2^left_fract
                    let left_multiplier = (2.0f32.powf(left_fract)
                        * ((1u64 << (FIXED_POINT_SCALE + shift_diff.trunc().abs() as usize))
                            as f32))
                        .round_ties_even() as Element;
                    let right_multiplier = (2.0f32.powf(right_fract)
                        * (1u64 << FIXED_POINT_SCALE) as f32)
                        .round_ties_even() as Element;
                    (
                        right_input_shift.trunc().abs() as usize,
                        left_multiplier,
                        right_multiplier,
                    )
                }
            };

        // Now we need to make sure we can actually perform the requantisation, if not we sacrifice some accuracy to make it possible
        let lhs_input_bit_size = left_scaling.bit_size();
        let rhs_input_bit_size = right_scaling.bit_size();
        let mut total_bit_size = Self::scaled_sum_bit_size(
            left_multiplier,
            lhs_input_bit_size,
            right_multiplier,
            rhs_input_bit_size,
        );

        while total_bit_size >= 63 {
            final_right_shift = final_right_shift
                .checked_sub(1)
                .expect("Add scaling cannot fit i64 without a negative right shift");
            left_multiplier >>= 1;
            right_multiplier >>= 1;
            total_bit_size = Self::scaled_sum_bit_size(
                left_multiplier,
                lhs_input_bit_size,
                right_multiplier,
                rhs_input_bit_size,
            );
        }

        Self {
            left_multiplier,
            right_multiplier,
            right_shift: final_right_shift,
            intermediate_bit_size: total_bit_size,
            output_scaling: *output_scaling,
        }
    }
    /// The value to scalar multiply the left input by
    pub fn left_scale(&self) -> Element {
        self.left_multiplier
    }
    /// The value to scalar multiply the right input by
    pub fn right_scale(&self) -> Element {
        self.right_multiplier
    }

    pub fn intermediate_bit_size(&self) -> usize {
        self.intermediate_bit_size
    }

    pub fn right_shift(&self) -> usize {
        self.right_shift
    }
}

/// In the Add layer quantisation we need to make sure both inputs have the same scale factor in order to add them.
/// To achieve this we calculate the fixed point multiplier for the left and right inputs so that they have the same scaling factor and can be added together.
/// Then we add a requantisation step after the addition that performs only the right shift part of the fixed point multiplication.
impl Add<f32> {
    pub fn quantize(
        self,
        input_scaling: &[ScalingFactor],
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<Add<Element>>> {
        let left = input_scaling[0];
        let right = input_scaling[1];
        let add_quant_info = AddQuantInfo::new(&left, &right, &output_scaling);

        let quantized_model = Add::<Element> {
            quant_info: Some(add_quant_info),
            phantom: PhantomData {},
        };

        let requant = requant_from_add(add_quant_info);
        QuantizeOutput::new(quantized_model, vec![output_scaling]).with_requant(requant)
    }
}

/// Function used to instantiate a new [`Requant`] from the [`AddQuantInfo`] calculated during quantization of the Add layer.
/// This [`Requant`] will perform just the right shift part of the fixed point multiplication.
pub fn requant_from_add(add_quant_info: AddQuantInfo) -> Requant {
    let output_bit_size = add_quant_info.output_scaling.bit_size() + 1;
    // We work out how many value chunks for this requantisation operation here, it must be a multiple of the requantisation BIT_LEN
    let value_chunks = output_bit_size / *quantization::BIT_LEN;

    assert!(
        output_bit_size.is_multiple_of(*quantization::BIT_LEN),
        "Output bit size after requantisation must be a multiple of {}, got {}",
        *quantization::BIT_LEN,
        output_bit_size
    );

    let activation_lookup_data = ActivationLookupData::new(
        add_quant_info.right_shift(),
        1,
        // We subtract FIXED_POINT_SCALE here because the ActivationLookupData will add this back on when calculating the max bit size
        add_quant_info.intermediate_bit_size() - FIXED_POINT_SCALE,
        0,
        Table::new_requantise(),
        false,
        value_chunks,
    );
    Requant {
        output_scaling: add_quant_info.output_scaling,
        table: Table::new_requantise(),
        activation_lookup_data,
    }
}

impl QuantizeOp for Add<f32> {
    type QuantizedOp = Add<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        input_scaling: &[ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            output_scalings.len() == 1,
            "Output scaling for convolution layer different from 1"
        );
        self.quantize(input_scaling, output_scalings[0])
    }
}

impl ProveInfo for Add<Element> {
    fn step_info<F: PrimeField>(
        &self,
        aux: ContextAux,
    ) -> anyhow::Result<(LayerCtx<F>, ContextAux)> {
        let Some(ref quant_info) = self.quant_info else {
            bail!("Add layer is not quantized");
        };
        let ctx = AddCtx {
            quant_info: *quant_info,
            operand_key: None,
        };
        Ok((LayerCtx::Add(ctx), aux))
    }
}

impl PadOp for Add<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        ensure!(si.shapes.len() == 2, "Add layer expects 2 input shapes");
        Ok(self)
    }
}

impl<F, PCS> ProvableOp<F, PCS> for Add<Element>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = AddCtx;

    fn prove<T>(
        &self,
        node_id: NodeId,
        _ctx: &Self::Ctx,
        last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<F, T, PCS>,
    ) -> anyhow::Result<Vec<Claim<F>>>
    where
        T: Transcript,
    {
        let (output_claims, proof) = self.prove_step(
            node_id,
            last_claims,
            &step_data.padded_input_tensors()?,
            prover,
        )?;

        prover.push_proof(node_id, LayerProof::Add(proof));
        Ok(output_claims)
    }
}

impl Splittable for AddCtx {}

impl<F, PCS> VerifiableCtx<F, PCS> for AddCtx
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = AddProof<F>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        _shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        ensure!(last_claims.len() == 1, "Add layer expects 1 claim");
        let last_claim = last_claims[0];
        // just making sure downsizing due to API of E is ok
        ensure!((self.quant_info.left_scale() as u64) as Element == self.quant_info.left_scale());
        ensure!((self.quant_info.right_scale() as u64) as Element == self.quant_info.right_scale());
        // we have the output claim f(r) = y = x1(r) * x1_scale + x2(r) * x2_scale
        // and the proof gives us x1(r) and x2(r) so we just need to "scale" these and
        // verify the equation.
        let left_scale: F = self.quant_info.left_scale().to_field();
        let scaled_left = proof.left_eval * left_scale;
        let right_scale: F = self.quant_info.right_scale().to_field();
        let left_claim = Claim::new(last_claim.point.clone(), proof.left_eval);
        let scaled_right = proof.right_eval * right_scale;
        let right_claim = Claim::new(last_claim.point.clone(), proof.right_eval);
        ensure!(
            scaled_left + scaled_right == last_claim.eval,
            "Add layer verification failed"
        );
        if let Some(key) = &self.operand_key {
            // in this case we need to verify the opening for the operand via PCS
            let mut claims = HashMap::new();
            claims.insert(
                key.clone(),
                Claim::new(last_claim.point.clone(), proof.right_eval),
            );
            verifier.add_common_claims(node_id, claims);
            // in this case we return only the left claim since the right one is verified by PCS
            Ok(vec![left_claim])
        } else {
            // in this case we return both claims
            Ok(vec![left_claim, right_claim])
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
mod test {
    use std::{fmt::Debug, ops::Range};

    use proptest::prelude::*;
    use tenstore::GenStore;

    use super::*;
    use crate::{
        Element,
        layers::Layer,
        model::{Model, test::prove_model},
        quantization::{Dequantize, Quantize},
        tensor::is_close_with_tolerance,
    };

    #[test]
    fn test_add_quantization() {
        let add = Add::<f32>::new();
        let t1 = Tensor::<f32>::random(&vec![2, 2].into());
        let t2 = Tensor::<f32>::random(&vec![2, 2].into());
        let domain: (Element, Element) = (*quantization::MIN, *quantization::MAX);
        let s1 = ScalingFactor::from_tensor(&t1, Some(domain));
        let s2 = ScalingFactor::from_tensor(&t2, None);
        let qt1 = t1.quantize(&s1); // x1_q = round(x1 / s1)
        let qt2 = t2.quantize(&s2);
        let dequant_t1 = qt1.dequantize(&s1);
        let dequant_t2 = qt2.dequantize(&s2);
        let t3 = dequant_t1.add(&dequant_t2);
        let s3 = ScalingFactor::from_tensor(&t3, Some(domain));

        let qadd = add.quantize(&[s1, s2], s3).unwrap().quantized_op;
        let qadd_result = qadd
            .evaluate(&[&qt1.as_wrapped(), &qt2.as_wrapped()])
            .unwrap();

        let quant_info = qadd.quant_info.as_ref().unwrap();

        let domain = quant_info.output_scaling.domain();
        let computed_result = Tensor::<f32>::new(
            qadd_result.outputs()[0].shape().to_vec().into(),
            qadd_result.outputs()[0]
                .get_data()
                .iter()
                .map(|x| {
                    let unclamped = *x >> quant_info.right_shift();

                    if unclamped >= domain.1 {
                        domain.1 as f32 * s3.scale()
                    } else if unclamped <= domain.0 {
                        domain.0 as f32 * s3.scale()
                    } else {
                        unclamped as f32 * s3.scale()
                    }
                })
                .collect::<Vec<_>>(),
        )
        .unwrap();

        let close_to_float =
            is_close_with_tolerance(computed_result.data(), t3.data(), 1e-2_f32, 1e-1_f32);

        assert!(
            close_to_float,
            "output is not close to float: float {:?} vs computed {:?}",
            t3.data(),
            computed_result.data()
        );
    }

    #[test]
    fn test_add_proving_no_operand() {
        let input_shape = Shape::from(vec![2, 2]);
        for _ in 0..25 {
            let mut model =
                Model::new_from_input_shapes(vec![input_shape.clone(), input_shape.clone()]);

            let add = Add::new();
            let _ = model.add_consecutive_layer(Layer::Add(add), None).unwrap();
            model.automatic_output_labelling().unwrap();
            model.describe();
            prove_model(model, &mut GenStore::default()).unwrap();
        }
    }

    #[test]
    fn mixed_width_residual_keeps_both_branches_and_proves() -> anyhow::Result<()> {
        assert_eq!(*quantization::BIT_LEN, 12);
        // Frozen Llama3 block-2 scales after retaining 24-bit GLU output.
        // The right input is a 48-bit unrequantized down projection, while the
        // left residual is 24-bit. The old max(multiplier)+max(input) bound
        // shifted the right multiplier to zero for these concrete scales.
        let residual_domain = (-(1i64 << 23), (1i64 << 23) - 1);
        let down_domain = (-(1i64 << 47), (1i64 << 47) - 1);
        let left_sf = ScalingFactor::from_scale(2.5950814e-5, Some(residual_domain));
        let right_sf =
            ScalingFactor::from_scale(3.159912e-7 * (4095.0 / 16777215.0), Some(down_domain));
        let output_sf = ScalingFactor::from_scale(2.5980135e-5, Some(residual_domain));
        let info = AddQuantInfo::new(&left_sf, &right_sf, &output_sf);
        assert!(info.left_scale() > 0 && info.right_scale() > 0);
        assert!(info.intermediate_bit_size() < 63);
        let rounding = 1i128 << (info.right_shift() - 1);
        for left in [residual_domain.0, residual_domain.1] {
            for right in [down_domain.0, down_domain.1] {
                let result = i128::from(left) * i128::from(info.left_scale())
                    + i128::from(right) * i128::from(info.right_scale())
                    + rounding;
                assert!((i128::from(i64::MIN)..=i128::from(i64::MAX)).contains(&result));
            }
        }

        let shape: Shape = vec![2, 4].into();
        let left = Tensor::new(
            shape.clone(),
            vec![1.0f32, -1.0, 0.0, 0.0, 0.25, -0.25, 0.0, 0.0],
        )?
        .quantize(&left_sf);
        let right = Tensor::new(
            shape.clone(),
            vec![0.5f32, -0.5, 1.0, -1.0, 0.0, 0.0, 1e6, -1e6],
        )?
        .quantize(&right_sf);
        let expected = left.dequantize(&left_sf).add(&right.dequantize(&right_sf));
        let quantized = Add::<f32>::new().quantize(&[left_sf, right_sf], output_sf)?;
        let mut model = Model::<Element>::new_from_input_shapes(vec![shape.clone(), shape]);
        let add_id = model.add_consecutive_layer(Layer::Add(quantized.quantized_op), None)?;
        model.add_consecutive_layer(
            Layer::Requant(quantized.requant_layer.unwrap()[0]),
            Some(add_id),
        )?;
        model.automatic_output_labelling()?;
        // Runs Add + Requant, generates its full proof, and verifies it.
        let output = crate::model::test::prove_quantized_model(
            model,
            vec![left, right],
            &mut GenStore::default(),
        )?;
        let output = output[0].tensor()?;
        for (actual, expected) in output.data()[..6].iter().zip(expected.data()) {
            assert!((output_sf.dequantize(actual) - expected).abs() < 0.001);
        }
        assert_eq!(output.data()[6], residual_domain.1);
        assert_eq!(output.data()[7], residual_domain.0);
        Ok(())
    }

    #[test]
    fn zero_multiplier_does_not_inherit_the_unused_input_width() {
        assert_eq!(AddQuantInfo::scaled_sum_bit_size(0, 60, 1 << 15, 11), 28);
        assert_eq!(AddQuantInfo::scaled_sum_bit_size(1 << 15, 11, 0, 60), 28);
        assert_eq!(
            AddQuantInfo::scaled_sum_bit_size(0, 60, 0, 60),
            FIXED_POINT_SCALE
        );
    }

    proptest! {
        #[test]
        fn test_add_with_f32(input in any_input::<f32>(1..256, 1..256)) {
            let Input { operand, input } = input;

            let expected = input.add(&operand);
            let add = Add::<f32>::new();
            let computed = add.evaluate(&[&input.as_wrapped(), &operand.as_wrapped()]).unwrap();

            prop_assert_eq!(expected, computed.outputs[0].to_native());
        }

        #[test]
        fn test_add_with_element(
            input in any_input::<Element>(1..256, 1..256),
            left_multiplier in 1..=10_i64,
            right_multiplier in 1..=10_i64,
        ) {
            let Input { operand, input } = input;

            let expected = input.scalar_mul(&left_multiplier).add(&operand.scalar_mul(&right_multiplier));

            let quant_info = AddQuantInfo {
                left_multiplier,
                right_multiplier,
                right_shift: 1,
                intermediate_bit_size: 13,
                output_scaling: ScalingFactor::default(),
            };

            let mut add = Add::<Element>::new();
            add.quant_info = Some(quant_info);

            let computed = add.evaluate(&[&input.as_wrapped(), &operand.as_wrapped()]).unwrap();

            prop_assert_eq!(expected, computed.outputs[0].to_native());
        }
    }

    struct Input<T> {
        operand: Tensor<T>,
        input: Tensor<T>,
    }

    impl<T> Debug for Input<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Input").finish_non_exhaustive()
        }
    }

    fn any_input<T: TensorTypeParam>(
        dim_x: Range<usize>,
        dim_y: Range<usize>,
    ) -> impl Strategy<Value = Input<T>> {
        (dim_x, dim_y).prop_flat_map(|(dim_x, dim_y)| {
            let shape = Shape::new(vec![dim_x, dim_y]);
            let operand = Tensor::<T>::any(shape.clone());
            let input = Tensor::<T>::any(shape.clone());
            (operand, input).prop_map(|(operand, input)| Input { operand, input })
        })
    }
}
