use super::provable::{Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, VerifiableCtx};
use crate::{
    Claim, Element, NextPowerOfTwo, Prover, ProverContext, ScalingFactor, ScalingStrategy, Shape,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        Layer, LayerCtx, LayerProof,
        provable::{QuantizeOp, QuantizeOutput, Splittable},
        requant::Requant,
    },
    lookup::{
        context::LookupWitnessGen,
        logup_gkr::structs::LogUpBatchProof,
        operation::{
            LookupOp,
            generic_prove::{GenericLookupProof, LookupProverResult, prove_lookup_op},
            generic_verify::{LookupVerifyResult, verify_lookup_op},
        },
        table::Table,
    },
    model::Step,
    padding::PaddingMode,
    poly_commit::verifier::VerifierCommitment,
    tensor::{Tensor, TensorTypeParam, WrappedModuleFn, WrappedTensor},
};

use ark_ff::PrimeField;
use dp_crypto::{
    IntoMLE,
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    structs::IOPProof,
};

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, marker::PhantomData};

pub mod lookup_data;
use lookup_data::ActivationLookupData;

/// The short name used to identify an activation layer.
pub const ACTIVATION_LAYER: &str = "ACTI";

use anyhow::{Result, ensure};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Activation<N> {
    /// Plain activation layer where we apply the activation function to each input independently
    Plain(ActivationLayer<N>),
    /// Variant of Activation employed in a Gated Linear Unit (GLU); in this case, the activation layer
    /// has 2 inputs: one is passed through the activation function; the output of the layer is
    /// computed as the element-wise product between the result of the activation function and the
    /// other input. More specifically, for inputs x and y, the output is activation(x) * y
    GLU(ActivationLayer<N>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ActivationLayer<N> {
    Relu(Option<ActivationLookupData>, PhantomData<N>),
    Gelu(Option<ActivationLookupData>, PhantomData<N>),
    Silu(Option<ActivationLookupData>, PhantomData<N>),
    Sigmoid(Option<ActivationLookupData>, PhantomData<N>),
    Softplus(Option<ActivationLookupData>, PhantomData<N>),
    ExpNegative(Option<ActivationLookupData>, PhantomData<N>),
}

impl<N> ActivationLayer<N> {
    pub fn tracked_input_data_id(&self) -> String {
        match self {
            ActivationLayer::Relu(_, _) => "RELU_IN".to_string(),
            ActivationLayer::Gelu(_, _) => "GELU_IN".to_string(),
            ActivationLayer::Silu(_, _) => "SILU_IN".to_string(),
            ActivationLayer::Sigmoid(_, _) => "SIGMOID_IN".to_string(),
            ActivationLayer::Softplus(_, _) => "SOFTPLUS_IN".to_string(),
            ActivationLayer::ExpNegative(_, _) => "EXP_NEG_IN".to_string(),
        }
    }

    pub fn tracked_output_data_id(&self) -> String {
        match self {
            ActivationLayer::Relu(_, _) => "RELU_OUT".to_string(),
            ActivationLayer::Gelu(_, _) => "GELU_OUT".to_string(),
            ActivationLayer::Silu(_, _) => "SILU_OUT".to_string(),
            ActivationLayer::Sigmoid(_, _) => "SIGMOID_OUT".to_string(),
            ActivationLayer::Softplus(_, _) => "SOFTPLUS_OUT".to_string(),
            ActivationLayer::ExpNegative(_, _) => "EXP_NEG_OUT".to_string(),
        }
    }
}

/// Currently holds the poly info for the output polynomial of the RELU
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActivationCtx {
    pub op: Activation<Element>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct ActivationProof<F: PrimeField, PCS: CommitmentScheme> {
    /// proof for the accumulation of the claim from m2v + claim from lookup for the same poly
    /// e.g. the "link" between a m2v and relu layer
    pub(crate) io_accumulation: IOPProof<F>,
    /// The evaluations output by the linking sumcheck
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) evaluations: Vec<F>,
    /// the lookup proof for the relu
    pub(crate) lookup: LogUpBatchProof<F>,
    /// The witness commitments from this function
    pub(crate) commitments: Vec<VerifierCommitment<PCS>>,
}

impl<F, PCS> ActivationProof<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    pub(crate) fn write_commitment<T: Transcript>(&self, transcript: &mut T) -> Result<()> {
        self.commitments
            .iter()
            .for_each(|commit| commit.append_to_transcript(transcript));
        Ok(())
    }
}

/// Type wrapping an activation layer employed in GeGlu (i.e., a GLU with Gelu as activation function)
pub struct GeGlu<N>(Activation<N>);

impl<N> Activation<N> {
    /// The port index for the "up" projection input when used in a GLU.
    pub const UP_INPUT_INDEX: usize = 1;

    /// The port index for the "gate" input when used in a GLU.
    pub const GATE_INPUT_INDEX: usize = 0;

    /// Returns a new rectified linear activation.
    pub fn new_relu() -> Self {
        Self::Plain(ActivationLayer::<N>::Relu(None, PhantomData))
    }

    /// Returns a new gaussian error activation.
    pub fn new_gelu() -> Self {
        Self::Plain(ActivationLayer::<N>::Gelu(None, PhantomData))
    }

    /// Returns a new sigmoid linear unit activation.
    pub fn new_silu() -> Self {
        Self::Plain(ActivationLayer::<N>::Silu(None, PhantomData))
    }

    pub fn new_sigmoid() -> Self {
        Self::Plain(ActivationLayer::<N>::Sigmoid(None, PhantomData))
    }

    pub fn new_softplus() -> Self {
        Self::Plain(ActivationLayer::<N>::Softplus(None, PhantomData))
    }

    /// Exponential on nonpositive inputs, used for recurrent decay gates.
    pub fn new_exp_negative() -> Self {
        Self::Plain(ActivationLayer::<N>::ExpNegative(None, PhantomData))
    }

    /// Returns a Gated SiLU (SwiGLU).
    pub fn new_swiglu() -> Self {
        Self::GLU(ActivationLayer::<N>::Silu(None, PhantomData))
    }

    /// Instantiate a new Activation layer configured to be used in a GLU.
    pub fn new_for_glu(activation_type: ActivationLayer<N>) -> Self {
        Self::GLU(activation_type)
    }

    /// Instantiate a new Activation layer configured to be used as a plain activation function.
    pub fn new_plain(activation_type: ActivationLayer<N>) -> Self {
        Self::Plain(activation_type)
    }

    /// Returns a Gated GELU.
    pub fn new_geglu() -> GeGlu<N> {
        GeGlu::<N>(Self::GLU(ActivationLayer::<N>::Gelu(None, PhantomData)))
    }

    pub(crate) fn activation_type(&self) -> &ActivationLayer<N> {
        match self {
            Self::Plain(layer) | Self::GLU(layer) => layer,
        }
    }
}

impl<N> From<GeGlu<N>> for Layer<N>
where
    N: TensorTypeParam,
{
    fn from(geglu: GeGlu<N>) -> Self {
        Layer::Activation(geglu.0)
    }
}

impl<N> GeGlu<N> {
    /// Position expected in the set of layer inputs for the input value that is passed through Gelu
    pub const GELU_INPUT_INDEX: usize = 0;

    /// Position expected in the set of layers inputs for the input value that is multiplied to the output of Gelu
    pub const LINEAR_INPUT_INDEX: usize = 1;
}

impl ActivationLayer<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<Vec<WrappedTensor<f32>>> {
        match self {
            ActivationLayer::Relu(..) => Ok(inputs
                .iter()
                .map(|tensor| WrappedTensor::relu((*tensor).clone()))
                .collect::<Vec<_>>()),
            ActivationLayer::Gelu(..) => Ok(inputs
                .iter()
                .map(|input| WrappedTensor::gelu((*input).clone()))
                .collect::<Vec<_>>()),
            ActivationLayer::Silu(..) => Ok(inputs
                .iter()
                .map(|input| WrappedTensor::silu((*input).clone()))
                .collect::<Vec<_>>()),
            ActivationLayer::Sigmoid(..)
            | ActivationLayer::Softplus(..)
            | ActivationLayer::ExpNegative(..) => inputs
                .iter()
                .map(|input| {
                    let shape = Shape::from(input.shape());
                    let unpadded_shape = Shape::from(input.unpadded_shape());
                    let input = (*input).clone().reduce_to_unpadded_shape()?;
                    let values = input
                        .get_data()
                        .into_iter()
                        .map(|x| match self {
                            ActivationLayer::Sigmoid(..) => crate::lookup::table::sigmoid_float(x),
                            ActivationLayer::Softplus(..) => {
                                crate::lookup::table::softplus_float(x)
                            }
                            ActivationLayer::ExpNegative(..) => x.min(0.0).exp(),
                            _ => unreachable!(),
                        })
                        .collect();
                    let mut tensor = Tensor::new(unpadded_shape, values)?;
                    tensor.pad_to_shape(shape)?;
                    WrappedTensor::try_from(&tensor)
                })
                .collect(),
        }
    }

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        num_outputs: usize,
    ) -> anyhow::Result<QuantizeOutput<ActivationLayer<Element>>> {
        let output_scalings = S::scaling_factors_for_node(data, node_id, num_outputs);
        let table_input_scaling = S::scaling_factor_for_intermediate_data(
            data,
            node_id,
            self.tracked_input_data_id().into(),
        );
        let table_output_scaling = S::scaling_factor_for_intermediate_data(
            data,
            node_id,
            self.tracked_output_data_id().into(),
        );
        let activation_lookup_data = ActivationLookupData::new_from_scalings(
            input_scaling[0],
            output_scalings[0],
            table_input_scaling,
            table_output_scaling,
            &self,
        )?;

        match self {
            ActivationLayer::Relu(..) => Ok(QuantizeOutput::new(
                ActivationLayer::<Element>::Relu(Some(activation_lookup_data), PhantomData),
                output_scalings,
            )),
            ActivationLayer::Gelu(..) => Ok(QuantizeOutput::new(
                ActivationLayer::<Element>::Gelu(Some(activation_lookup_data), PhantomData),
                vec![table_output_scaling],
            )),
            ActivationLayer::Silu(..) => Ok(QuantizeOutput::new(
                ActivationLayer::<Element>::Silu(Some(activation_lookup_data), PhantomData),
                vec![table_output_scaling],
            )),
            ActivationLayer::Sigmoid(..) => Ok(QuantizeOutput::new(
                ActivationLayer::<Element>::Sigmoid(Some(activation_lookup_data), PhantomData),
                vec![table_output_scaling],
            )),
            ActivationLayer::Softplus(..) => Ok(QuantizeOutput::new(
                ActivationLayer::<Element>::Softplus(Some(activation_lookup_data), PhantomData),
                vec![table_output_scaling],
            )),
            ActivationLayer::ExpNegative(..) => Ok(QuantizeOutput::new(
                ActivationLayer::<Element>::ExpNegative(Some(activation_lookup_data), PhantomData),
                vec![table_output_scaling],
            )),
        }
    }
}

impl ActivationLayer<Element> {
    fn get_lookup_data(&self) -> &ActivationLookupData {
        match self {
            ActivationLayer::Relu(Some(data), ..)
            | ActivationLayer::Gelu(Some(data), ..)
            | ActivationLayer::Silu(Some(data), ..)
            | ActivationLayer::Sigmoid(Some(data), ..)
            | ActivationLayer::Softplus(Some(data), ..)
            | ActivationLayer::ExpNegative(Some(data), ..) => data,
            _ => panic!("Activation layer lookup data not initialized"),
        }
    }

    fn set_glu_flag(&mut self, is_glu: bool) {
        match self {
            ActivationLayer::Relu(Some(data), ..)
            | ActivationLayer::Gelu(Some(data), ..)
            | ActivationLayer::Silu(Some(data), ..)
            | ActivationLayer::Sigmoid(Some(data), ..)
            | ActivationLayer::Softplus(Some(data), ..)
            | ActivationLayer::ExpNegative(Some(data), ..) => {
                data.set_glu(is_glu);
            }
            _ => unreachable!("ActivationLayer Element lookup data not initialized"),
        }
    }

    pub(crate) fn lookup_tables(&self) -> Vec<Table> {
        let lookup_data = self.get_lookup_data();
        let value_table = lookup_data.table;
        match value_table.is_signed() {
            true => {
                let chunking_info = lookup_data.chunking_info(&value_table).unwrap();
                let number_zero_chunks = chunking_info.number_of_zeroing_chunks();
                match number_zero_chunks {
                    0 => vec![Table::new_shift_check(), value_table],
                    1 => vec![
                        Table::new_shift_check(),
                        value_table,
                        Table::new_signed_zero_check(),
                    ],
                    _ => vec![
                        Table::new_shift_check(),
                        value_table,
                        Table::new_zero_check(),
                        Table::new_signed_zero_check(),
                    ],
                }
            }
            false => {
                let mut tables = vec![Table::new_shift_check(), value_table];
                if lookup_data
                    .chunking_info(&value_table)
                    .unwrap()
                    .number_of_zeroing_chunks()
                    > 0
                {
                    tables.push(Table::new_zero_check());
                }
                tables
            }
        }
    }

    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<Vec<WrappedTensor<Element>>> {
        let lookup_data = self.get_lookup_data();
        inputs
            .iter()
            .map(|tensor| lookup_data.evaluate((*tensor).clone()))
            .collect()
    }

    fn lookup_witness<F: PrimeField, PCS: CommitmentScheme<Field = F>>(
        &self,
        id: NodeId,
        _ctx: &ProverContext<F, PCS>,
        activation_input: &Tensor<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        let lookup_data = self.get_lookup_data();
        let lookup_witness = lookup_data.get_lookup_witness(activation_input.clone())?;

        let element_counts = lookup_witness.get_counts(&lookup_data.table);

        let input_evals = lookup_witness.input_mle_evals::<F>(lookup_data.table.num_columns());
        let output_evals = lookup_witness.output_mle_evals::<F>();

        let mles = input_evals
            .into_iter()
            .chain(output_evals)
            .map(|evals| evals.into_mle())
            .collect();

        let mut gen_w = LookupWitnessGen::<F>::default();
        let tables = vec![
            Table::new_shift_check(),
            lookup_data.table,
            Table::new_zero_check(),
            Table::new_signed_zero_check(),
        ];

        gen_w.insert_layer_witness_data(id, mles, tables, element_counts);

        Ok(gen_w)
    }
}

impl<N> OpInfo for Activation<N> {
    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        match self {
            Self::Plain(_) => Ok(num_inputs),
            Self::GLU(_) => Ok(1),
        }
    }

    fn describe(&self) -> String {
        match self.activation_type() {
            ActivationLayer::Relu(..) => "ReLU".to_string(),
            ActivationLayer::Gelu(..) => "GeLU".to_string(),
            ActivationLayer::Silu(..) => "SiLU".to_string(),
            ActivationLayer::Sigmoid(..) => "Sigmoid".to_string(),
            ActivationLayer::Softplus(..) => "Softplus".to_string(),
            ActivationLayer::ExpNegative(..) => "ExpNegative".to_string(),
        }
    }

    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        _padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match self {
            Self::Plain(_) => Ok(input_shapes.to_vec()), // same as input shapes,
            Self::GLU(_) => Ok(vec![input_shapes[0].clone()]), /* in GLU, there is only one output, which has the same shape as the first input */
        }
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl QuantizeOp for Activation<f32> {
    type QuantizedOp = Activation<Element>;

    fn quantize_op<S: crate::ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[crate::ScalingFactor],
        _unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        _unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        let num_outputs = self.num_outputs(input_scaling.len())?;
        match self {
            Self::Plain(layer) => layer
                .quantize_op::<S>(data, node_id, input_scaling, num_outputs)
                .map(|q_op| {
                    QuantizeOutput::new(Activation::Plain(q_op.quantized_op), q_op.output_scalings)
                }),
            Self::GLU(layer) => {
                ensure!(
                    input_scaling.len() == 2,
                    "Expected 2 input scaling factors for activation layer used in GLU, found {}",
                    input_scaling.len(),
                );

                let QuantizeOutput {
                    mut quantized_op,
                    output_scalings: activation_out_scalings,
                    ..
                } = layer.quantize_op::<S>(data, node_id, input_scaling, num_outputs)?;
                ensure!(
                    activation_out_scalings.len() == 1,
                    "Expected 1 output scaling factor for activation layer used in GLU, found {}",
                    activation_out_scalings.len(),
                );
                // Set the GLU flag in the lookup data
                quantized_op.set_glu_flag(true);

                let multiplier =
                    activation_out_scalings[0].m(&input_scaling[1], &output_scalings[0]);
                let intermediate_bit_size =
                    activation_out_scalings[0].bit_size() + input_scaling[1].bit_size() + 1;

                let requant =
                    Requant::from_multiplier(multiplier, intermediate_bit_size, output_scalings[0]);
                Ok(
                    QuantizeOutput::new(Activation::GLU(quantized_op), output_scalings.to_vec())
                        .with_requant(requant)?,
                )
            }
        }
    }
}

impl Evaluate<f32> for Activation<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        match self {
            Activation::Plain(layer) => {
                let mut activation_outputs = layer.evaluate(inputs)?;
                let activation_out = activation_outputs.pop().unwrap();
                Ok(
                    LayerOut::from_vec(vec![activation_out.clone()]).with_data_to_be_tracked(
                        HashMap::from([
                            (layer.tracked_output_data_id().into(), activation_out),
                            (layer.tracked_input_data_id().into(), inputs[0].clone()),
                        ]),
                    ),
                )
            }
            Activation::GLU(layer) => {
                ensure!(
                    inputs.len() == 2,
                    "Expected 2 inputs for activation layer used in GLU, found {} inputs instead",
                    inputs.len(),
                );
                let mut activation_outputs = layer.evaluate(&[inputs[0]])?;
                // double-check that there is only one output
                assert_eq!(activation_outputs.len(), 1);
                let activation_out = activation_outputs.pop().unwrap();
                Ok(LayerOut::from_vec(
                    // multiply `activation_out` with `inputs[1]`
                    vec![activation_out.clone().mul(inputs[1].clone())?],
                )
                .with_data_to_be_tracked(HashMap::from([
                    (layer.tracked_output_data_id().into(), activation_out),
                    (layer.tracked_input_data_id().into(), inputs[0].clone()),
                ])))
            }
        }
    }
}

impl Evaluate<Element> for Activation<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        match self {
            Activation::Plain(activation_layer) => {
                activation_layer.evaluate(inputs).map(LayerOut::from_vec)
            }
            Activation::GLU(activation_layer) => {
                ensure!(
                    inputs.len() == 2,
                    "Expected 2 inputs for activation layer used in GLU, found {} inputs instead",
                    inputs.len(),
                );
                let mut activation_outputs = activation_layer.evaluate(&[inputs[0]])?;
                // double-check that there is only one output
                assert_eq!(activation_outputs.len(), 1);
                let activation_output = activation_outputs.pop().unwrap();
                let layer_out = LayerOut::from_vec(vec![activation_output.mul(inputs[1].clone())?]);
                Ok(layer_out)
            }
        }
    }
}

impl ProveInfo for Activation<Element> {
    fn step_info<F: PrimeField>(&self, mut aux: ContextAux) -> Result<(LayerCtx<F>, ContextAux)> {
        // Set the model polys to be empty
        aux.model_polys = None;
        aux.max_poly_len = aux
            .last_output_shape
            .iter()
            .fold(aux.max_poly_len, |acc, shapes| {
                acc.max(shapes.next_power_of_two().product())
            });
        let act = self.clone();

        Ok((LayerCtx::Activation(ActivationCtx { op: act }), aux))
    }
}

impl<N> PadOp for Activation<N> {}

impl<F, PCS> ProvableOp<F, PCS> for Activation<Element>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = ActivationCtx;

    fn prove<'a, 'b, 'c, 'd, T: Transcript>(
        &'a self,
        id: NodeId,
        _ctx: &'b Self::Ctx,
        last_claims: Vec<&Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut Prover<'c, 'd, F, T, PCS>,
    ) -> Result<Vec<Claim<F>>> {
        let inputs = &step_data.node_inputs;
        ensure!(
            !inputs.is_empty(),
            "Expected at least 1 input in inferece data for activation layer",
        );
        self.prove_step(prover, last_claims[0], step_data, id)
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<F, PCS>,
        step_data: &Step<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        ensure!(
            step_data.outputs().len() == 1,
            "Found more than 1 output tensor in inference step of activation layer"
        );

        let input = step_data.input_tensor_at(0)?;

        match self {
            Activation::Plain(activation_layer) => {
                ensure!(
                    step_data.node_inputs.len() == 1,
                    "Found more than 1 input tensor in inference step of activation layer"
                );
                activation_layer.lookup_witness(id, ctx, &input)
            }
            Activation::GLU(activation_layer) => {
                ensure!(
                    step_data.node_inputs.len() == 2,
                    "Found more than 2 input tensor in inference step of activation layer"
                );
                activation_layer.lookup_witness(id, ctx, &input)
            }
        }
    }
}

impl OpInfo for ActivationCtx {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        self.op.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        self.op.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        self.op.describe()
    }

    fn is_provable(&self) -> bool {
        true
    }
}

impl Splittable for ActivationCtx {}

impl<F, PCS> VerifiableCtx<F, PCS> for ActivationCtx
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Proof = ActivationProof<F, PCS>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> Result<Vec<Claim<F>>> {
        self.verify_activation(verifier, last_claims[0], proof, node_id, shape_step)
    }

    fn write_proof_to_transcript<T: Transcript>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        proof.write_commitment(transcript)
    }
}

impl Activation<Element> {
    #[timed::timed_instrument(name = "Prover::prove_activation_step")]
    pub(crate) fn prove_step<'a, 'b, F, T: Transcript, PCS>(
        &self,
        prover: &mut Prover<'a, 'b, F, T, PCS>,
        last_claim: &Claim<F>,
        step: &Step<Element>,
        node_id: NodeId,
    ) -> anyhow::Result<Vec<Claim<F>>>
    where
        F: PrimeField,
        PCS: CommitmentScheme<Field = F>,
    {
        let lookup_op = self.activation_type().get_lookup_data();

        let lookup_claim =
            lookup_op.lookup_output_claim(last_claim, step.inputs()[0].unpadded_shape())?;

        let LookupProverResult {
            generic_proof,
            input_claims,
            ..
        } = prove_lookup_op(
            lookup_op,
            &lookup_claim,
            step,
            &lookup_op.table,
            None,
            prover,
            node_id,
        )?;

        let GenericLookupProof {
            logup_proof,
            sumcheck_proof,
            evaluations,
            commitments,
            ..
        } = generic_proof;

        let proof = ActivationProof {
            io_accumulation: sumcheck_proof,
            evaluations,
            lookup: logup_proof,
            commitments,
        };

        // Add the proof to the prover
        prover.push_proof(node_id, LayerProof::Activation(proof));
        // Return the input claim for the next layer
        Ok(input_claims)
    }
}

impl ActivationCtx {
    pub(crate) fn verify_activation<F: PrimeField, T: Transcript, PCS: CommitmentScheme>(
        &self,
        verifier: &mut Verifier<F, T, PCS>,
        last_claim: &Claim<F>,
        proof: &ActivationProof<F, PCS>,
        node_id: NodeId,
        shape_step: &ShapeStep,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        let ActivationProof {
            io_accumulation,
            evaluations,
            lookup,
            commitments,
        } = proof;

        let generic_lookup_proof = GenericLookupProof::<F, PCS> {
            logup_proof: lookup.clone(),
            sumcheck_proof: io_accumulation.clone(),
            evaluations: evaluations.clone(),
            commitments: commitments.clone(),
            weight_evaluation: None,
            shift_evaluations: None,
        };
        let lookup_op = self.op.activation_type().get_lookup_data();
        let lookup_claim =
            lookup_op.lookup_output_claim(last_claim, &shape_step.unpadded_output_shape[0])?;
        let LookupVerifyResult { input_claims, .. } = verify_lookup_op(
            lookup_op,
            &lookup_claim,
            shape_step,
            &lookup_op.table,
            &generic_lookup_proof,
            verifier,
            node_id,
        )?;

        Ok(input_claims)
    }
}

#[cfg(test)]
mod test {
    use ark_std::rand::Rng;
    use burn::tensor::activation::{gelu, silu};
    use proptest::prelude::*;

    use crate::{
        layers::{EinSum, Layer},
        lookup::table::{gelu_float, silu_float},
        model::{Model, test::prove_model},
        rng_from_env_or_random,
        tensor::{IntoBTensor, KeyedTensor},
    };

    use super::*;

    #[derive(Clone, Debug)]
    struct Input {
        weight: KeyedTensor<f32>,
        bias: KeyedTensor<f32>,
        input: Tensor<f32>,
    }

    impl Input {
        fn random(rows_max: usize, columns_max: usize) -> Input {
            let mut rng = rng_from_env_or_random();
            let rows = rng.gen_range(8..rows_max);
            let columns = rng.gen_range(8..columns_max);
            let matrix_size = rows * columns;
            let weight_data: Vec<f32> = (0..matrix_size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();
            let bias_data: Vec<f32> = (0..rows).map(|_| rng.gen_range(-10.0..10.0)).collect();

            let input_rank = rng.gen_range(1usize..=4);

            let mut all_dims: Vec<usize> =
                (0..(input_rank - 1)).map(|_| rng.gen_range(3..8)).collect();
            all_dims.push(columns);

            let total_data_size = all_dims.iter().product::<usize>();
            let input_shape = Shape::from(all_dims);
            let input_data: Vec<f32> = (0..total_data_size)
                .map(|_| rng.gen_range(-10.0..10.0))
                .collect();

            Input {
                weight: KeyedTensor::new(
                    "W".to_string(),
                    Tensor::new(vec![rows, columns].into(), weight_data).unwrap(),
                ),
                bias: KeyedTensor::new(
                    "BIAS".to_string(),
                    Tensor::new(vec![rows].into(), bias_data).unwrap(),
                ),
                input: Tensor::new(input_shape, input_data).unwrap(),
            }
        }
    }

    #[test]
    fn test_activation_proving() -> anyhow::Result<()> {
        test_activation_proving_helper(Activation::<f32>::new_relu)?;
        test_activation_proving_helper(Activation::<f32>::new_gelu)
    }

    fn test_activation_proving_helper<F>(f: F) -> anyhow::Result<()>
    where
        F: Fn() -> Activation<f32>,
    {
        for _ in 0..25 {
            let Input {
                weight,
                bias,
                input: random_input,
            } = Input::random(25, 25);

            let input_rank = random_input.shape().rank();
            let equation = match input_rank {
                1 => "I(j)@W(ij)->O(i)+BIAS(i)",
                2 => "I(aj)@W(ij)->O(ai)+BIAS(i)",
                3 => "I(abj)@W(ij)->O(abi)+BIAS(i)",
                4 => "I(abcj)@W(ij)->O(abci)+BIAS(i)",
                _ => panic!("Input rank too high for test"),
            }
            .to_string();
            let dense = EinSum::<f32>::new(
                equation.to_owned(),
                vec![Some(weight.into())],
                vec![Some(bias.into())],
            )
            .unwrap()
            .no_requant();

            let mut model = Model::new_from_input_shapes(vec![random_input.shape().clone()]);

            let dense_id = model
                .add_consecutive_layer(Layer::EinSum(dense), None)
                .unwrap();

            let _ = model
                .add_consecutive_layer(Layer::Activation(f()), Some(dense_id))
                .unwrap();

            model.automatic_output_labelling().unwrap();
            model.describe();
            prove_model(model, &mut Default::default()).unwrap();
        }

        Ok(())
    }

    #[test]
    fn test_glu_activation_proving() -> anyhow::Result<()> {
        for _ in 0..25 {
            let input_shape = vec![7, 94].into();
            let mut model = Model::new_from_input_shapes(
                vec![input_shape; 2], // 2 inputs in case of GLU variant
            );
            model.add_consecutive_layer(Activation::new_geglu().into(), None)?;
            model.automatic_output_labelling()?;
            prove_model(model, &mut Default::default()).unwrap();
        }
        Ok(())
    }

    #[test]
    fn test_activation_silu_proving() -> anyhow::Result<()> {
        let input_shape = vec![3, 100].into();
        let mut model = Model::new_from_input_shapes(vec![input_shape]);
        model.add_consecutive_layer(Layer::Activation(Activation::new_silu()), None)?;
        model.automatic_output_labelling()?;
        prove_model(model, &mut Default::default()).unwrap();
        Ok(())
    }

    #[test]
    fn test_glu_silu_activation_proving() -> anyhow::Result<()> {
        let input_shape = vec![7, 94].into();
        let mut model = Model::new_from_input_shapes(vec![input_shape; 2]);
        model.add_consecutive_layer(Layer::Activation(Activation::new_swiglu()), None)?;
        model.automatic_output_labelling()?;
        prove_model(model, &mut Default::default()).unwrap();
        Ok(())
    }

    #[test]
    fn qwen_gate_activations_match_stable_scalar_reference() -> anyhow::Result<()> {
        let values = vec![-100.0, -20.0, -1.0, 0.0, 1.0, 20.0, 100.0];
        let input = Tensor::new(vec![7].into(), values.clone())?;
        for (kind, activation) in [Activation::new_sigmoid(), Activation::new_softplus()]
            .into_iter()
            .enumerate()
        {
            let result = activation.evaluate(&[&input.as_wrapped()])?.outputs[0].get_data();
            for (&x, &actual) in values.iter().zip(&result) {
                let x = x as f64;
                let expected = if kind == 0 {
                    1.0 / (1.0 + (-x).exp())
                } else {
                    x.max(0.0) + (-x.abs()).exp().ln_1p()
                } as f32;
                assert!(actual.is_finite());
                assert!((actual - expected).abs() <= 2e-6 * expected.abs().max(1.0));
            }
        }
        let input = Tensor::new(vec![5].into(), vec![-100.0, -20.0, -1.0, -0.001, 0.0])?;
        let result = Activation::new_exp_negative()
            .evaluate(&[&input.as_wrapped()])?
            .outputs[0]
            .get_data();
        for (&x, &actual) in input.data().iter().zip(&result) {
            assert!(actual.is_finite() && (0.0..=1.0).contains(&actual));
            assert!((actual - (x as f64).exp() as f32).abs() <= 1e-7);
        }
        Ok(())
    }

    #[test]
    fn qwen_gate_activations_prove_real_lookup_paths() -> anyhow::Result<()> {
        use crate::model::test::prove_model_with;
        for (i, activation) in [
            Activation::new_sigmoid(),
            Activation::new_softplus(),
            Activation::new_exp_negative(),
        ]
        .into_iter()
        .enumerate()
        {
            let values = if i == 2 {
                vec![-8.0, -2.0, -0.25, 0.0, -1.0, -0.5]
            } else {
                vec![-8.0, -2.0, -0.25, 0.0, 1.0, 8.0]
            };
            for shape in [vec![2, 3], vec![3, 3, 3]] {
                let data = values
                    .iter()
                    .copied()
                    .cycle()
                    .take(shape.iter().product())
                    .collect();
                let input = Tensor::new(shape.into(), data)?;
                let mut model = Model::new_from_input_shapes(vec![input.shape().clone()]);
                model.add_consecutive_layer(Layer::Activation(activation.clone()), None)?;
                model.automatic_output_labelling()?;
                prove_model_with(model, vec![input], &mut Default::default())?;
            }
        }
        Ok(())
    }

    #[test]
    fn qwen_gate_padding_is_zero_in_float_and_quantized_outputs() -> anyhow::Result<()> {
        use crate::model::test::quantize_model;
        let input = Tensor::new(
            vec![3, 3].into(),
            vec![-2.0, -1.0, 0.0, -3.0, 0.0, -0.5, -1.0, -4.0, 0.0],
        )?;
        for activation in [
            Activation::new_sigmoid(),
            Activation::new_softplus(),
            Activation::new_exp_negative(),
        ] {
            let padded = input.pad_next_power_of_two();
            let data = activation.evaluate(&[&padded.as_wrapped()])?.outputs[0].get_data();
            for i in 0..16 {
                if i / 4 == 3 || i % 4 == 3 {
                    assert_eq!(data[i], 0.0);
                }
            }
            let mut model = Model::new_from_input_shapes(vec![input.shape().clone()]);
            model.add_consecutive_layer(Layer::Activation(activation), None)?;
            model.automatic_output_labelling()?;
            let (model, inputs) = quantize_model(
                model,
                vec![input.clone()],
                Some(vec![input.clone()]),
                &mut Default::default(),
            )?;
            let (_, layer) = model
                .graph()
                .forward_inners()
                .next()
                .expect("activation node");
            let padded = inputs[0].pad_next_power_of_two();
            let data = layer.evaluate(&[&padded.as_wrapped()])?.outputs[0].get_data();
            for i in 0..16 {
                if i / 4 == 3 || i % 4 == 3 {
                    assert_eq!(data[i], 0);
                }
            }
        }
        Ok(())
    }

    proptest! {
        #[test]
        fn gelu_kernel_test(size in 1usize..1024) {
            let shape = Shape::new(vec![size]);
            let tensor = Tensor::<f32>::random(&shape);

            let btensor = tensor.to_btensor::<1>();
            let data = gelu(btensor).to_data().into_vec().expect("Failed to compute GELU");
            let resultb = Tensor::<f32>::new(shape.clone(), data).unwrap();

            let data = tensor.data();
            let data = data.iter().map(gelu_float).collect::<Vec<_>>();
            let result = Tensor::new(shape, data).unwrap();

            resultb.data().iter().zip(result.data().iter()).try_for_each(|(left, right)| {
                prop_assert!(
                    (left - right).abs() < 1e-3,
                    "Actual: {left}, Expected: {right}",
                );
                Ok(())
            })?;
        }
    }

    proptest! {
        #[test]
        fn silu_kernel_test(size in 1usize..1024) {
            let shape = Shape::new(vec![size]);
            let tensor = Tensor::<f32>::random(&shape);

            let btensor = tensor.clone().to_btensor::<1>();
            let data = silu(btensor).to_data().into_vec().expect("Failed to compute SiLU");
            let resultb = Tensor::<f32>::new(shape.clone(), data).unwrap();

            let data = tensor.data();
            let data = data.iter().map(silu_float).collect::<Vec<_>>();
            let result = Tensor::new(shape, data).unwrap();

            resultb.data().iter().zip(result.data().iter()).try_for_each(|(left, right)| {
                prop_assert!(
                    (left - right).abs() < 1e-3,
                    "Actual: {left}, Expected: {right}",
                );
                Ok(())
            })?;
        }
    }
}
