use super::{
    LayerCtx, LayerProof, flatten::Flatten, requant::Requant, transformer::softmax::SoftmaxData,
};
use crate::{
    Claim, Element, InitTranscript, Prover, ProverContext, ScalingFactor, ScalingStrategy, Shape,
    Tensor,
    graph::NodeId,
    iop::{
        context::{ContextAux, ShapeStep},
        verifier::Verifier,
    },
    layers::{
        convolution::{ConvFFTData, ConvFFTHandle},
        transformer::{
            logits::{ArgmaxData, ArgmaxHandle},
            normalisation::{
                layernorm::{LayerNormProvingData, evaluate::LayerNormHandle},
                rmsnorm::{RMSNormProvingData, evaluate::RMSNormHandle},
            },
            softmax::SoftmaxHandle,
        },
    },
    lookup::context::LookupWitnessGen,
    model::{trace::Step, transform::ModelTransform},
    padding::{PaddingMode, ShapeInfo},
    quantization::ToField,
    tensor::{TensorHandle, TensorTypeParam, WrappedTensor},
};
use anyhow::{Result, bail, ensure};
use ark_ff::PrimeField;
use derive_more::{From, Into};
use dp_crypto::arkyper::{CommitmentScheme, transcript::Transcript};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug, ops::Deref};
use tenstore::{GenStore, StorageKey};

/// Extra data returned by layers needed for proving.
///
/// This should only be implemented for quantised layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProvingData {
    /// Data from evaluation of convolution.
    Convolution(ConvFFTData),
    /// Data from evaluation of softmax.
    Softmax(SoftmaxData),
    /// Variant used for extra data used to prove [LayerNorm][`crate::layers::transformer::normalisation::layernorm::LayerNorm`]
    LayerNorm(LayerNormProvingData),
    /// Variant used for extra data used to prove [RMSNorm][`crate::layers::transformer::normalisation::rmsnorm::RMSNorm`]
    RMSNorm(RMSNormProvingData),
    /// Variant used for extra data used to prove [ArgMax][`crate::layers::transformer::logits::Logits`]
    ArgMax(ArgmaxData),
    /// No extra data
    None,
}

/// Extra data returned by layers needed for proving.
///
/// This should only be implemented for quantised layers.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProvingHandle {
    Convolution(ConvFFTHandle),
    Softmax(SoftmaxHandle),
    LayerNorm(LayerNormHandle),
    RMSNorm(RMSNormHandle),
    ArgMax(ArgmaxHandle),
    None,
}

impl ProvingHandle {
    pub(crate) fn new(
        storage_key: StorageKey<Vec<Element>>,
        proving_data: ProvingData,
        store: GenStore,
    ) -> Self {
        match proving_data {
            ProvingData::Convolution(conv_fftdata) => {
                ProvingHandle::Convolution(ConvFFTHandle::new(storage_key, conv_fftdata, store))
            }
            ProvingData::Softmax(softmax_data) => {
                ProvingHandle::Softmax(SoftmaxHandle::new(storage_key, softmax_data, store))
            }
            ProvingData::LayerNorm(layer_norm_data) => {
                ProvingHandle::LayerNorm(LayerNormHandle::new(storage_key, layer_norm_data, store))
            }
            ProvingData::RMSNorm(rms_norm_data) => {
                ProvingHandle::RMSNorm(RMSNormHandle::new(storage_key, rms_norm_data, store))
            }
            ProvingData::ArgMax(argmax_data) => {
                ProvingHandle::ArgMax(ArgmaxHandle::new(storage_key, argmax_data, store))
            }
            ProvingData::None => ProvingHandle::None,
        }
    }

    pub(crate) fn split_proving_data(&self, num_chunks: usize) -> anyhow::Result<Vec<Self>> {
        match self {
            ProvingHandle::None => Ok(vec![ProvingHandle::None; num_chunks]),
            ProvingHandle::ArgMax(handle) => handle
                .split_tensors(num_chunks)
                .map(|new_handles| new_handles.into_iter().map(ProvingHandle::ArgMax).collect()),
            _ => todo!(),
        }
    }

    pub(crate) fn handles(&self) -> impl Iterator<Item = &TensorHandle<Element>> {
        match self {
            ProvingHandle::Convolution(conv_ffthandle) => vec![&conv_ffthandle.handle],
            ProvingHandle::Softmax(softmax_handle) => vec![&softmax_handle.shift_handle],
            ProvingHandle::LayerNorm(layer_norm_handle) => {
                vec![&layer_norm_handle.mean, &layer_norm_handle.std_dev]
            }
            ProvingHandle::RMSNorm(rmsnorm_handle) => vec![&rmsnorm_handle.normalisation],
            ProvingHandle::ArgMax(argmax_handle) => argmax_handle.max_values.iter().collect_vec(),
            ProvingHandle::None => vec![],
        }
        .into_iter()
    }

    pub(crate) fn try_into_map<
        F: FnMut(TensorHandle<Element>) -> anyhow::Result<TensorHandle<Element>>,
    >(
        self,
        mut f: F,
    ) -> anyhow::Result<Self> {
        Ok(match self {
            ProvingHandle::Convolution(mut conv_ffthandle) => {
                conv_ffthandle.handle = f(conv_ffthandle.handle)?;
                ProvingHandle::Convolution(conv_ffthandle)
            }
            ProvingHandle::Softmax(mut softmax_handle) => {
                softmax_handle.shift_handle = f(softmax_handle.shift_handle)?;
                ProvingHandle::Softmax(softmax_handle)
            }
            ProvingHandle::LayerNorm(mut layer_norm_handle) => {
                layer_norm_handle.mean = f(layer_norm_handle.mean)?;
                layer_norm_handle.std_dev = f(layer_norm_handle.std_dev)?;
                ProvingHandle::LayerNorm(layer_norm_handle)
            }
            ProvingHandle::ArgMax(argmax_handle) => {
                let mut new_handles = Vec::with_capacity(argmax_handle.max_values.len());
                for handle in argmax_handle.max_values {
                    new_handles.push(f(handle)?);
                }
                ProvingHandle::ArgMax(ArgmaxHandle {
                    max_values: new_handles,
                })
            }
            ProvingHandle::None => ProvingHandle::None,
            ProvingHandle::RMSNorm(mut rmsnorm_handle) => {
                rmsnorm_handle.normalisation = f(rmsnorm_handle.normalisation)?;
                ProvingHandle::RMSNorm(rmsnorm_handle)
            }
        })
    }

    pub(crate) fn attach_store(&mut self, store: GenStore) {
        match self {
            ProvingHandle::Convolution(handle) => handle.attach_store(store),
            ProvingHandle::Softmax(handle) => handle.attach_store(store),
            ProvingHandle::LayerNorm(handle) => handle.attach_store(store),
            ProvingHandle::RMSNorm(handle) => handle.attach_store(store),
            ProvingHandle::ArgMax(handle) => handle.attach_store(store),
            ProvingHandle::None => {}
        }
    }

    pub(crate) fn isolate(&self) -> ProvingHandle {
        match self {
            ProvingHandle::Convolution(conv_ffthandle) => {
                ProvingHandle::Convolution(conv_ffthandle.isolate())
            }
            ProvingHandle::Softmax(softmax_handle) => {
                ProvingHandle::Softmax(softmax_handle.isolate())
            }
            ProvingHandle::LayerNorm(layer_norm_handle) => {
                ProvingHandle::LayerNorm(layer_norm_handle.isolate())
            }
            ProvingHandle::RMSNorm(rms_norm_handle) => {
                ProvingHandle::RMSNorm(rms_norm_handle.isolate())
            }
            ProvingHandle::ArgMax(argmax_handle) => ProvingHandle::ArgMax(argmax_handle.isolate()),
            ProvingHandle::None => ProvingHandle::None,
        }
    }
}

/// Identifier for an intermediate tensor of a layer, i.e., a tensor which is neither an
/// input nor an output tensor of the layer. The ID is employed to keep track of the tensor
/// for quantization purposes
#[derive(
    Clone,
    From,
    Into,
    Hash,
    PartialEq,
    Eq,
    Serialize,
    Deserialize,
    PartialOrd,
    Ord,
    derive_more::Debug,
)]
pub struct TrackedDataId(String);

#[derive(Clone, Debug)]
pub struct LayerOut<T>
where
    T: TensorTypeParam,
{
    pub(crate) outputs: Vec<WrappedTensor<T>>,
    pub(crate) proving_data: ProvingData,
    pub(crate) tracked_layer_data: HashMap<TrackedDataId, WrappedTensor<T>>,
}

impl<T: TensorTypeParam> LayerOut<T> {
    pub(crate) fn from_vec(out: Vec<WrappedTensor<T>>) -> Self {
        Self {
            outputs: out,
            proving_data: ProvingData::None,
            tracked_layer_data: Default::default(),
        }
    }

    pub(crate) fn with_proving_data(self, data: ProvingData) -> Self {
        Self {
            outputs: self.outputs,
            proving_data: data,
            tracked_layer_data: self.tracked_layer_data,
        }
    }

    /// Add a set of intermediate data tensors to be tracked for quantization purposes;
    /// Each intermediate tensor is identified by a corresponding `TrackedDataId`
    pub(crate) fn with_data_to_be_tracked(
        self,
        data: HashMap<TrackedDataId, WrappedTensor<T>>,
    ) -> Self {
        Self {
            outputs: self.outputs,
            proving_data: self.proving_data,
            tracked_layer_data: data,
        }
    }

    pub fn outputs(&self) -> Vec<&WrappedTensor<T>> {
        self.outputs.iter().collect()
    }

    pub fn from_tensor(out: WrappedTensor<T>) -> Self {
        Self::from_vec(vec![out])
    }

    pub fn try_convdata(&self) -> Option<&ConvFFTData> {
        match self.proving_data {
            ProvingData::Convolution(ref conv_data) => Some(conv_data),
            _ => None,
        }
    }

    pub fn try_softmax_data(&self) -> Option<&SoftmaxData> {
        match self.proving_data {
            ProvingData::Softmax(ref softmax_data) => Some(softmax_data),
            _ => None,
        }
    }

    pub fn try_argmax_data(&self) -> Option<&ArgmaxData> {
        match self.proving_data {
            ProvingData::ArgMax(ref argmax_data) => Some(argmax_data),
            _ => None,
        }
    }

    pub fn try_layernorm_data(&self) -> Option<&LayerNormProvingData> {
        match self.proving_data {
            ProvingData::LayerNorm(ref layernorm_data) => Some(layernorm_data),
            _ => None,
        }
    }
    pub fn try_rmsnorm_data(&self) -> Option<&RMSNormProvingData> {
        match self.proving_data {
            ProvingData::RMSNorm(ref rmsnorm_data) => Some(rmsnorm_data),
            _ => None,
        }
    }
}

pub trait OpInfo {
    /// Returns the shapes of the outputs (in the same order)
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>>;

    /// Compute the number of output tensors, given the number of input tensors
    /// `num_inputs`
    fn num_outputs(&self, num_inputs: usize) -> Result<usize>;

    /// Textual description of the operation
    fn describe(&self) -> String;

    /// Specify whether the operation needs to be proven or not
    fn is_provable(&self) -> bool;
}

pub trait Evaluate<T: TensorTypeParam> {
    /// Evaluates the operation given any inputs tensors and constant inputs.
    fn evaluate(&self, inputs: &[&WrappedTensor<T>]) -> Result<LayerOut<T>>;
}

/// Helper method employed to call `Evaluate::evaluate` when there are no `unpadded_input_shapes`
/// or when the `E` type cannot be inferred automatically by the compiler
pub fn evaluate_layer<T: TensorTypeParam, O: Evaluate<T>>(
    layer: &O,
    inputs: &[&WrappedTensor<T>],
) -> Result<LayerOut<T>> {
    layer.evaluate(inputs)
}

pub trait ProveInfo {
    /// Compute the proving context for the operation
    fn step_info<F: PrimeField>(&self, aux: ContextAux) -> Result<(LayerCtx<F>, ContextAux)>;
}

/// Output of `QuantizeOp` method over a layer
pub struct QuantizeOutput<Op> {
    /// The actual layer after quantization
    pub quantized_op: Op,
    /// The scaling factor of the output wires of the operation
    pub(crate) output_scalings: Vec<ScalingFactor>,
    /// The requant layer to be added to the model, if any
    pub(crate) requant_layer: Option<Vec<Requant>>,
    /// Optional rule to apply after quantisation
    pub(crate) post_quant_rule: Option<Box<dyn ModelTransform<Element>>>,
}

impl<Op> QuantizeOutput<Op> {
    pub fn new(quantized_op: Op, output_scalings: Vec<ScalingFactor>) -> Self {
        Self {
            quantized_op,
            output_scalings,
            requant_layer: None,
            post_quant_rule: None,
        }
    }
    pub fn with_requant(self, requant: Requant) -> Result<Self> {
        ensure!(
            self.output_scalings.len() == 1,
            "Number of output scalings must be 1"
        );
        Self::with_requants(self, vec![requant])
    }
    pub fn with_requants(self, requants: Vec<Requant>) -> Result<Self> {
        ensure!(self.requant_layer.is_none(), "Requant layer already exists");
        ensure!(
            self.output_scalings.len() == requants.len(),
            "Number of output scalings and requants must be the same"
        );
        Ok(Self {
            quantized_op: self.quantized_op,
            output_scalings: self.output_scalings,
            requant_layer: Some(requants),
            post_quant_rule: self.post_quant_rule,
        })
    }
    pub fn with_transform(self, transform: Box<dyn ModelTransform<Element>>) -> Result<Self> {
        ensure!(
            self.post_quant_rule.is_none(),
            "Post quantization rule already exists"
        );
        Ok(Self {
            quantized_op: self.quantized_op,
            output_scalings: self.output_scalings,
            requant_layer: self.requant_layer,
            post_quant_rule: Some(transform),
        })
    }
    pub fn maybe_requants(self, requant: Option<Vec<Requant>>) -> Result<Self> {
        match requant {
            Some(requant) => self.with_requants(requant),
            None => Ok(self),
        }
    }
    pub fn maybe_transform(
        self,
        transform: Option<Box<dyn ModelTransform<Element>>>,
    ) -> Result<Self> {
        match transform {
            Some(transform) => self.with_transform(transform),
            None => Ok(self),
        }
    }
}

pub trait QuantizeOp {
    type QuantizedOp: Sized;

    /// Convert an operation into its quantized version
    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        unpadded_input_shapes: &[Shape],
        output_scaling: &[ScalingFactor],
        unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>>;
}

pub trait PadOp {
    // Pad the dimensions of the tensors in node `self`, updating the `ShapeInfo` with the output shapes
    // of the node
    fn pad_node(self, _si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(self)
    }
}

pub trait ProvableOp<F, PCS>: OpInfo + PadOp + ProveInfo
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    type Ctx: VerifiableCtx<F, PCS>;

    /// Produces a proof of correct execution for this operation.
    fn prove<'a, 'b, 'c, 'd, T: Transcript + InitTranscript>(
        &'a self,
        _node_id: NodeId,
        _ctx: &'b Self::Ctx,
        _last_claims: Vec<&Claim<F>>,
        _step_data: &Step<Element>,
        _prover: &mut Prover<'c, 'd, F, T, PCS>,
    ) -> Result<Vec<Claim<F>>> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        ensure!(
            !self.is_provable(),
            "Running default prove implementation for a provable operation! Implement prove method"
        );
        Ok(vec![Claim::default()])
    }

    /// Generate witness for a node where a lookup table is employed in proving
    fn gen_lookup_witness(
        &self,
        _id: NodeId,
        _ctx: &ProverContext<F, PCS>,
        _step_data: &Step<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        Ok(Default::default())
    }
}

pub trait Splittable {
    /// This method returns the ideal number of chunks to be used to split the
    /// proving logic, if `self.splittable` returns true.
    /// If the layer is not splittable, this method should return `None`,
    /// since chunking is not applicable    
    fn ideal_num_chunks(&self, _input_shapes: &[Shape]) -> Option<usize> {
        None // by default, NO SPLIT
    }

    // whether the layer can be horizontally split
    fn is_splittable(&self) -> bool {
        false
    }
}
pub trait VerifiableCtx<F, PCS>: Debug + OpInfo
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    type Proof: Sized;

    /// Verify proof for the given operation
    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> Result<Vec<Claim<F>>>;

    /// Variant of `verify` employed to verify a chunk when the layer inputs are split into multiple chunks
    fn verify_chunk<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
        _chunk_number: usize,
    ) -> Result<Vec<Claim<F>>> {
        // as a default implementation, this is the same as `verify`
        self.verify(proof, last_claims, verifier, shape_step, node_id)
    }

    /// Verify the claim about the input of the model. Sometimes
    /// the input needs to be processed in a certain way before being evaluated.
    /// For example, Embeddings use one hot encoding of the input before
    /// running the matmul protocol.
    /// By default, it simply evaluates the input against the input claim.
    fn verify_input_claim<D: Deref<Target = F> + ToField<F>, A: AsRef<Tensor<D>>>(
        &self,
        inputs: &[A],
        claims: &[&Claim<F>],
    ) -> anyhow::Result<()> {
        ensure!(
            inputs.len() == claims.len(),
            "number of input tensors and claims must be the same"
        );
        for (i, (input, claim)) in inputs.iter().zip(claims).enumerate() {
            let computed = input
                .as_ref()
                .to_field()
                .into_mle()
                .evaluate(&claim.point)?;
            ensure!(
                computed == claim.eval,
                "input claim {:?} is incorrect, computed: {:?}, given: {:?}",
                i,
                computed,
                claim.eval,
            );
        }
        Ok(())
    }

    /// Writes the associated type [`Self::Proof`] to the transcript if it contains any [`PCS::Commitment`].
    fn write_proof_to_transcript<T: Transcript>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()>;
}

pub(crate) fn verify_input_claim<F, PCS, V, D, A>(
    ctx: &V,
    inputs: &[A],
    claims: &[&Claim<F>],
) -> anyhow::Result<()>
where
    V: VerifiableCtx<F, PCS>,
    F: PrimeField,
    PCS: CommitmentScheme,
    D: Deref<Target = F> + ToField<F>,
    A: AsRef<Tensor<D>>,
{
    <V as VerifiableCtx<F, PCS>>::verify_input_claim(ctx, inputs, claims)
}

pub(crate) fn write_proof_to_transcript<
    F: PrimeField,
    PCS: CommitmentScheme,
    T: Transcript,
    V: VerifiableCtx<F, PCS>,
>(
    ctx: &V,
    proof: &<V as VerifiableCtx<F, PCS>>::Proof,
    transcript: &mut T,
) -> anyhow::Result<()> {
    <V as VerifiableCtx<F, PCS>>::write_proof_to_transcript(ctx, proof, transcript)
}

#[derive(Clone, Debug)]
pub(crate) struct NonProvableVerifierCtx<'a, O>(&'a O);

impl<'a, O: OpInfo> OpInfo for NonProvableVerifierCtx<'a, O> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        self.0.output_shapes(input_shapes, padding_mode)
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        self.0.num_outputs(num_inputs)
    }

    fn describe(&self) -> String {
        self.0.describe()
    }

    fn is_provable(&self) -> bool {
        false
    }
}

impl<'a, O: OpInfo + Debug, F: PrimeField, PCS: CommitmentScheme> VerifiableCtx<F, PCS>
    for NonProvableVerifierCtx<'a, O>
{
    type Proof = ();

    fn verify<T: Transcript>(
        &self,
        _proof: &Self::Proof,
        _last_claims: &[&Claim<F>],
        _verifier: &mut Verifier<F, T, PCS>,
        _shape_step: &ShapeStep,
        _node_id: NodeId,
    ) -> Result<Vec<Claim<F>>> {
        // Default implementation, to avoid having to implement this method in case `is_provable` is false
        ensure!(
            !self.is_provable(),
            "Running default prove implementation for a provable operation! Implement prove method"
        );
        Ok(vec![Claim::default()])
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

impl<F: PrimeField> OpInfo for LayerCtx<F> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match self {
            LayerCtx::Convolution(conv_ctx) => conv_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Positional(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Add(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::LayerNorm(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::RMSNorm(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::QwenNorm(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::GatedDeltaScan(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::CausalConv1d(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Logits(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Embeddings(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Reshape(ctx) => ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Activation(activation_ctx) => {
                activation_ctx.output_shapes(input_shapes, padding_mode)
            }
            LayerCtx::Requant(requant_ctx) => requant_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Flatten => {
                <Flatten as OpInfo>::output_shapes(&Flatten(true), input_shapes, padding_mode)
            }
            LayerCtx::AttentionMask(attention_mask_ctx) => {
                attention_mask_ctx.output_shapes(input_shapes, padding_mode)
            }
            LayerCtx::EinSum(einsum_ctx) | LayerCtx::EinSumV2(einsum_ctx) => {
                einsum_ctx.output_shapes(input_shapes, padding_mode)
            }
            LayerCtx::Split(split_ctx) => split_ctx.output_shapes(input_shapes, padding_mode),
            LayerCtx::Recombination(rec_ctx) => rec_ctx.output_shapes(input_shapes, padding_mode),
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        match self {
            LayerCtx::Convolution(conv_ctx) => conv_ctx.num_outputs(num_inputs),
            LayerCtx::Positional(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Add(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::LayerNorm(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::RMSNorm(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::QwenNorm(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::GatedDeltaScan(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::CausalConv1d(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.num_outputs(num_inputs),
            LayerCtx::Logits(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Embeddings(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Reshape(ctx) => ctx.num_outputs(num_inputs),
            LayerCtx::Activation(activation_ctx) => activation_ctx.num_outputs(num_inputs),
            LayerCtx::Requant(requant_ctx) => requant_ctx.num_outputs(num_inputs),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.num_outputs(num_inputs),
            LayerCtx::Flatten => <Flatten as OpInfo>::num_outputs(&Flatten(true), num_inputs),
            LayerCtx::AttentionMask(attention_mask_ctx) => {
                attention_mask_ctx.num_outputs(num_inputs)
            }
            LayerCtx::EinSum(einsum_ctx) | LayerCtx::EinSumV2(einsum_ctx) => {
                einsum_ctx.num_outputs(num_inputs)
            }
            LayerCtx::Split(split_ctx) => split_ctx.num_outputs(num_inputs),
            LayerCtx::Recombination(rec_ctx) => rec_ctx.num_outputs(num_inputs),
        }
    }

    fn describe(&self) -> String {
        match self {
            LayerCtx::Convolution(conv_ctx) => conv_ctx.describe(),
            LayerCtx::Add(ctx) => ctx.describe(),
            LayerCtx::Positional(ctx) => ctx.describe(),
            LayerCtx::LayerNorm(ctx) => ctx.describe(),
            LayerCtx::RMSNorm(ctx) => ctx.describe(),
            LayerCtx::QwenNorm(ctx) => ctx.describe(),
            LayerCtx::GatedDeltaScan(ctx) => ctx.describe(),
            LayerCtx::CausalConv1d(ctx) => ctx.describe(),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.describe(),
            LayerCtx::Logits(ctx) => ctx.describe(),
            LayerCtx::Embeddings(ctx) => ctx.describe(),
            LayerCtx::Reshape(ctx) => ctx.describe(),
            LayerCtx::Activation(activation_ctx) => activation_ctx.describe(),
            LayerCtx::Requant(requant_ctx) => requant_ctx.describe(),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.describe(),
            LayerCtx::Flatten => Flatten(true).describe(),
            LayerCtx::AttentionMask(attention_mask_ctx) => attention_mask_ctx.describe(),
            LayerCtx::EinSum(einsum_ctx) | LayerCtx::EinSumV2(einsum_ctx) => einsum_ctx.describe(),
            LayerCtx::Split(split_ctx) => split_ctx.describe(),
            LayerCtx::Recombination(rec_ctx) => rec_ctx.describe(),
        }
    }

    fn is_provable(&self) -> bool {
        match self {
            LayerCtx::Convolution(conv_ctx) => conv_ctx.is_provable(),
            LayerCtx::Activation(activation_ctx) => activation_ctx.is_provable(),
            LayerCtx::Positional(ctx) => ctx.is_provable(),
            LayerCtx::Add(ctx) => ctx.is_provable(),
            LayerCtx::LayerNorm(ctx) => ctx.is_provable(),
            LayerCtx::RMSNorm(ctx) => ctx.is_provable(),
            LayerCtx::QwenNorm(ctx) => ctx.is_provable(),
            LayerCtx::GatedDeltaScan(ctx) => ctx.is_provable(),
            LayerCtx::CausalConv1d(ctx) => ctx.is_provable(),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.is_provable(),
            LayerCtx::Logits(ctx) => ctx.is_provable(),
            LayerCtx::Embeddings(ctx) => ctx.is_provable(),
            LayerCtx::Reshape(ctx) => ctx.is_provable(),
            LayerCtx::Requant(requant_ctx) => requant_ctx.is_provable(),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.is_provable(),
            LayerCtx::Flatten => Flatten(true).is_provable(),
            LayerCtx::AttentionMask(attention_mask_ctx) => attention_mask_ctx.is_provable(),
            LayerCtx::EinSum(einsum_ctx) | LayerCtx::EinSumV2(einsum_ctx) => {
                einsum_ctx.is_provable()
            }
            LayerCtx::Split(split_ctx) => split_ctx.is_provable(),
            LayerCtx::Recombination(rec_ctx) => rec_ctx.is_provable(),
        }
    }
}

impl<F: PrimeField, PCS: CommitmentScheme<Field = F>> VerifiableCtx<F, PCS> for LayerCtx<F> {
    type Proof = LayerProof<F, PCS>;

    fn verify<T: Transcript>(
        &self,
        proof: &LayerProof<F, PCS>,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
    ) -> Result<Vec<Claim<F>>> {
        match (self, proof) {
            (LayerCtx::Convolution(conv_ctx), LayerProof::Convolution(proof)) => {
                conv_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Embeddings(ctx), LayerProof::Embeddings(proof)) => {
                ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Positional(pos_ctx), LayerProof::Positional(proof)) => {
                pos_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Add(ctx), LayerProof::Add(proof)) => {
                ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Logits(ctx), LayerProof::Logits(proof)) => {
                ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Activation(activation_ctx), LayerProof::Activation(proof)) => {
                activation_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::LayerNorm(layernorm_ctx), LayerProof::LayerNorm(proof)) => {
                layernorm_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::RMSNorm(rmsnorm_ctx), LayerProof::RMSNorm(proof)) => {
                rmsnorm_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::QwenNorm(rmsnorm_ctx), LayerProof::QwenNorm(proof)) => {
                rmsnorm_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::GatedDeltaScan(rmsnorm_ctx), LayerProof::GatedDeltaScan(proof)) => {
                rmsnorm_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::CausalConv1d(rmsnorm_ctx), LayerProof::CausalConv1d(proof)) => {
                rmsnorm_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Requant(requant_ctx), LayerProof::Requant(proof)) => {
                requant_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Pooling(pooling_ctx), LayerProof::Pooling(proof)) => {
                pooling_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Softmax(softmax_ctx), LayerProof::Softmax(proof)) => {
                softmax_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Flatten, _) | (LayerCtx::Reshape(_), _) => {
                unreachable!("Trying to verify a non-provable layer")
            }
            (LayerCtx::AttentionMask(attention_mask_ctx), LayerProof::AttentionMask(proof)) => {
                attention_mask_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::EinSum(einsum_ctx), LayerProof::EinSum(proof)) => {
                einsum_ctx.verify_legacy(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::EinSumV2(einsum_ctx), LayerProof::EinSum(proof)) => {
                einsum_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Split(split_ctx), LayerProof::SplitLayer(proof)) => {
                split_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::Recombination(rec_ctx), LayerProof::Recombination(proof)) => {
                rec_ctx.verify(proof, last_claims, verifier, shape_step, node_id)
            }
            _ => bail!(
                "Incompatible layer {} and proof {} found",
                self.describe(),
                proof.variant_name()
            ),
        }
    }

    fn verify_chunk<T: Transcript>(
        &self,
        proof: &LayerProof<F, PCS>,
        last_claims: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape_step: &ShapeStep,
        node_id: NodeId,
        chunk_number: usize,
    ) -> Result<Vec<Claim<F>>> {
        match (self, proof) {
            (LayerCtx::Convolution(conv_ctx), LayerProof::Convolution(proof)) => conv_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::Embeddings(ctx), LayerProof::Embeddings(proof)) => ctx.verify_chunk(
                proof,
                last_claims,
                verifier,
                shape_step,
                node_id,
                chunk_number,
            ),
            (LayerCtx::Positional(pos_ctx), LayerProof::Positional(proof)) => pos_ctx.verify_chunk(
                proof,
                last_claims,
                verifier,
                shape_step,
                node_id,
                chunk_number,
            ),
            (LayerCtx::Add(ctx), LayerProof::Add(proof)) => ctx.verify_chunk(
                proof,
                last_claims,
                verifier,
                shape_step,
                node_id,
                chunk_number,
            ),
            (LayerCtx::Logits(ctx), LayerProof::Logits(proof)) => ctx.verify_chunk(
                proof,
                last_claims,
                verifier,
                shape_step,
                node_id,
                chunk_number,
            ),
            (LayerCtx::Activation(activation_ctx), LayerProof::Activation(proof)) => activation_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::LayerNorm(layernorm_ctx), LayerProof::LayerNorm(proof)) => layernorm_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::RMSNorm(rmsnorm_ctx), LayerProof::RMSNorm(proof)) => rmsnorm_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::QwenNorm(rmsnorm_ctx), LayerProof::QwenNorm(proof)) => rmsnorm_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::GatedDeltaScan(rmsnorm_ctx), LayerProof::GatedDeltaScan(proof)) => {
                rmsnorm_ctx.verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                )
            }
            (LayerCtx::CausalConv1d(rmsnorm_ctx), LayerProof::CausalConv1d(proof)) => rmsnorm_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::Requant(requant_ctx), LayerProof::Requant(proof)) => requant_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::Pooling(pooling_ctx), LayerProof::Pooling(proof)) => pooling_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::Softmax(softmax_ctx), LayerProof::Softmax(proof)) => softmax_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            (LayerCtx::Flatten, _) | (LayerCtx::Reshape(_), _) => {
                unreachable!("Trying to verify a non-provable layer")
            }
            (LayerCtx::AttentionMask(attention_mask_ctx), LayerProof::AttentionMask(proof)) => {
                attention_mask_ctx.verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                )
            }
            (LayerCtx::EinSum(einsum_ctx), LayerProof::EinSum(proof)) => {
                einsum_ctx.verify_legacy(proof, last_claims, verifier, shape_step, node_id)
            }
            (LayerCtx::EinSumV2(einsum_ctx), LayerProof::EinSum(proof)) => einsum_ctx.verify_chunk(
                proof,
                last_claims,
                verifier,
                shape_step,
                node_id,
                chunk_number,
            ),
            (LayerCtx::Split(split_ctx), LayerProof::SplitLayer(proof)) => split_ctx.verify_chunk(
                proof,
                last_claims,
                verifier,
                shape_step,
                node_id,
                chunk_number,
            ),
            (LayerCtx::Recombination(rec_ctx), LayerProof::Recombination(proof)) => rec_ctx
                .verify_chunk(
                    proof,
                    last_claims,
                    verifier,
                    shape_step,
                    node_id,
                    chunk_number,
                ),
            _ => bail!(
                "Incompatible layer {} and proof {} found",
                self.describe(),
                proof.variant_name()
            ),
        }
    }

    fn verify_input_claim<D: Deref<Target = F> + ToField<F>, A: AsRef<Tensor<D>>>(
        &self,
        inputs: &[A],
        claims: &[&Claim<F>],
    ) -> anyhow::Result<()> {
        match self {
            LayerCtx::Convolution(conv_ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(conv_ctx, inputs, claims)
            }

            LayerCtx::Activation(activation_ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(activation_ctx, inputs, claims)
            }
            LayerCtx::LayerNorm(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::RMSNorm(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::QwenNorm(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::GatedDeltaScan(ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims)
            }
            LayerCtx::CausalConv1d(ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims)
            }
            LayerCtx::Softmax(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::Logits(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::Embeddings(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::Add(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::Positional(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::Reshape(ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(&NonProvableVerifierCtx(ctx), inputs, claims)
            }
            LayerCtx::Requant(requant_ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(requant_ctx, inputs, claims)
            }
            LayerCtx::Pooling(pooling_ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(pooling_ctx, inputs, claims)
            }
            LayerCtx::Flatten => verify_input_claim::<F, PCS, _, _, _>(
                &NonProvableVerifierCtx(&Flatten(true)),
                inputs,
                claims,
            ),
            LayerCtx::AttentionMask(ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims)
            }
            LayerCtx::EinSum(ctx) | LayerCtx::EinSumV2(ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims)
            }
            LayerCtx::Split(ctx) => verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims),
            LayerCtx::Recombination(ctx) => {
                verify_input_claim::<F, PCS, _, _, _>(ctx, inputs, claims)
            }
        }
    }

    fn write_proof_to_transcript<T: Transcript>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> anyhow::Result<()> {
        match (self, proof) {
            (LayerCtx::Convolution(ctx), LayerProof::Convolution(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }

            (LayerCtx::Activation(ctx), LayerProof::Activation(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::LayerNorm(ctx), LayerProof::LayerNorm(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::RMSNorm(ctx), LayerProof::RMSNorm(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::QwenNorm(ctx), LayerProof::QwenNorm(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::GatedDeltaScan(ctx), LayerProof::GatedDeltaScan(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::CausalConv1d(ctx), LayerProof::CausalConv1d(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Softmax(ctx), LayerProof::Softmax(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Logits(ctx), LayerProof::Logits(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Embeddings(ctx), LayerProof::Embeddings(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Add(ctx), LayerProof::Add(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Positional(ctx), LayerProof::Positional(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Requant(ctx), LayerProof::Requant(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Pooling(ctx), LayerProof::Pooling(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::AttentionMask(ctx), LayerProof::AttentionMask(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::EinSum(ctx), LayerProof::EinSum(p))
            | (LayerCtx::EinSumV2(ctx), LayerProof::EinSum(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Split(ctx), LayerProof::SplitLayer(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Recombination(ctx), LayerProof::Recombination(p)) => {
                write_proof_to_transcript::<F, PCS, _, _>(ctx, p, transcript)
            }
            (LayerCtx::Flatten, _) => Ok(()),
            (LayerCtx::Reshape(_), _) => Ok(()),
            _ => bail!(
                "Could not append LayerProof to transcript, Incompatible layer {} and proof {} found",
                self.describe(),
                proof.variant_name()
            ),
        }
    }
}
