use self::transformer::{
    causal_conv1d::{CausalConv1d, CausalConv1dCtx, CausalConv1dProof},
    gated_delta::{GatedDeltaScan, GatedDeltaScanCtx, GatedDeltaScanProof},
    normalisation::qwen_norm::{QwenNorm, QwenNormCtx, QwenNormProof},
};
use crate::{
    Element, InitTranscript, ProverContext, ScalingStrategy, Shape,
    graph::NodeId,
    iop::context::{ContextAux, ShapeStep},
    layers::{
        activation::{ACTIVATION_LAYER, Activation, ActivationProof},
        add::{ADD_LAYER, Add, AddCtx, AddProof},
        convolution::{CONVOLUTION_LAYER, ConvFFTHandle, Convolution},
        einsum::{EINSUM_LAYER, EinSum, EinSumContext, EinSumProof},
        flatten::FLATTEN_LAYER,
        pooling::{POOLING_LAYER, Pooling},
        provable::{ProvingHandle, Splittable},
        recombination::{RECOMBINATION_LAYER, RecombinationLayer, RecombinationProof},
        requant::{REQUANT_LAYER, Requant, RequantProof},
        reshape::{RESHAPE_LAYER, Reshape, ReshapeCtx},
        split::{SPLIT_LAYER, SplitLayer, SplitLayerProof},
        transformer::{
            attention_mask::{
                ATTENTION_MASK_LAYER, AttentionMask, AttentionMaskCtx, AttentionMaskProof,
            },
            embeddings::{EMBEDDINGS_LAYER, Embeddings, EmbeddingsCtx, EmbeddingsProof},
            logits::{ArgmaxHandle, LOGITS_LAYER, Logits, LogitsCtx, LogitsProof},
            normalisation::{
                layernorm::{
                    LAYERNORM_LAYER, LayerNorm, LayerNormCtx, LayerNormProof,
                    evaluate::LayerNormHandle,
                },
                rmsnorm::{
                    RMSNORM_LAYER, RMSNorm, RMSNormCtx, RMSNormProof, evaluate::RMSNormHandle,
                },
            },
            positional::{POSITIONAL_LAYER, Positional, PositionalCtx, PositionalProof},
            softmax::{SOFTMAX_LAYER, Softmax, SoftmaxCtx, SoftmaxHandle, SoftmaxProof},
        },
    },
    lookup::{context::LookupWitnessGen, table::Table},
    model::Step,
    padding::{PaddingMode, ShapeInfo},
    quantization::{ModelMetadata, ScalingFactor},
    tensor::{TensorHandle, TensorTypeParam, WrappedTensor},
};
use activation::ActivationCtx;
use anyhow::{Result, bail, ensure};
use ark_ff::PrimeField;
use convolution::{ConvCtx, ConvProof};
use dp_crypto::arkyper::{CommitmentScheme, transcript::Transcript};
use flatten::Flatten;
use pooling::{PoolingCtx, PoolingProof};
use provable::{
    Evaluate, LayerOut, OpInfo, PadOp, ProvableOp, ProveInfo, QuantizeOp, QuantizeOutput,
};
use requant::RequantCtx;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::fmt::Debug;
use tenstore::StoreError;

pub mod activation;
pub mod add;
pub mod convolution;
pub mod einsum;
pub mod flatten;
pub mod hadamard;
pub mod pooling;
pub mod provable;
pub mod recombination;
pub mod requant;
pub mod reshape;
pub mod split;
pub mod transformer;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[allow(clippy::large_enum_variant)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub enum Layer<T>
where
    T: TensorTypeParam,
{
    Convolution(Convolution<T>),
    Activation(Activation<T>),
    // this is the output quant info. Since we always do a requant layer after each dense,
    // then we assume the inputs requant info are default()
    Requant(Requant),
    Pooling(Pooling),
    // TODO: so far it's only flattening the input tensor, e.g. new_shape = vec![shape.iter().product()]
    Flatten(Flatten),
    EinSum(EinSum<T>),
    LayerNorm(LayerNorm<T>),
    RMSNorm(RMSNorm<T>),
    Softmax(Softmax<T>),
    Add(Add<T>),
    Reshape(Reshape),
    Embeddings(Embeddings<T>),
    Positional(Positional<T>),
    AttentionMask(AttentionMask<T>),
    Logits(Logits),
    Split(SplitLayer),
    Recombination(RecombinationLayer),
    // Append new variants to preserve serialized legacy discriminants.
    QwenNorm(QwenNorm<T>),
    CausalConv1d(CausalConv1d<T>),
    GatedDeltaScan(GatedDeltaScan<T>),
}

impl<T> Layer<T>
where
    T: TensorTypeParam,
{
    pub fn short_name(&self) -> &str {
        let r = match self {
            Layer::Convolution(_) => CONVOLUTION_LAYER,
            Layer::Activation(_) => ACTIVATION_LAYER,
            Layer::Requant(_) => REQUANT_LAYER,
            Layer::Pooling(_) => POOLING_LAYER,
            Layer::Flatten(_) => FLATTEN_LAYER,
            Layer::LayerNorm(_) => LAYERNORM_LAYER,
            Layer::RMSNorm(_) => RMSNORM_LAYER,
            Layer::QwenNorm(_) => "QNRM",
            Layer::CausalConv1d(_) => "CC1D",
            Layer::GatedDeltaScan(_) => "GDSC",
            Layer::Softmax(_) => SOFTMAX_LAYER,
            Layer::Add(_) => ADD_LAYER,
            Layer::Reshape(_) => RESHAPE_LAYER,
            Layer::Embeddings(_) => EMBEDDINGS_LAYER,
            Layer::Positional(_) => POSITIONAL_LAYER,
            Layer::Logits(_) => LOGITS_LAYER,
            Layer::AttentionMask(_) => ATTENTION_MASK_LAYER,
            Layer::EinSum(_) => EINSUM_LAYER,
            Layer::Split(_) => SPLIT_LAYER,
            Layer::Recombination(_) => RECOMBINATION_LAYER,
        };
        assert_eq!(r.len(), 4, "layer short name must be 4 chars long: {r}");
        r
    }

    /// Convert a layer to a string only containing its kind
    pub fn as_kind_str(&self) -> &'static str {
        match self {
            Layer::Convolution(_) => "convolution",
            Layer::Activation(_) => "activation",
            Layer::Requant(_) => "requant",
            Layer::Pooling(_) => "pooling",
            Layer::Flatten(_) => "flatten",
            Layer::EinSum(_) => "einsum",
            Layer::LayerNorm(_) => "layer-norm",
            Layer::RMSNorm(_) => "rms-norm",
            Layer::QwenNorm(_) => "qwen-norm",
            Layer::CausalConv1d(_) => "causal-conv1d",
            Layer::GatedDeltaScan(_) => "gated-delta-scan",
            Layer::Softmax(_) => "softmax",
            Layer::Add(_) => "add",
            Layer::Reshape(_) => "reshape",
            Layer::Embeddings(_) => "embeddings",
            Layer::Positional(_) => "positional",
            Layer::Logits(_) => "logits",
            Layer::AttentionMask(_) => "attention-mask",
            Layer::Split(_) => "split",
            Layer::Recombination(_) => "recombination",
        }
    }

    /// Resets the internal state of the layer if any
    pub fn reset(&self) {
        match self {
            Layer::Positional(pos) => pos.reset_cache(),
            Layer::EinSum(einsum) => einsum.reset_caches(),
            Layer::Softmax(softmax) => softmax.reset_cache(),
            Layer::LayerNorm(layer_norm) => layer_norm.reset_cache(),
            Layer::RMSNorm(rms_norm) => rms_norm.reset_cache(),
            Layer::CausalConv1d(x) => x.reset_cache(),
            Layer::GatedDeltaScan(x) => x.reset_cache(),
            _ => (),
        }
    }
}

/// Describes a steps wrt the polynomial to be proven/looked at. Verifier needs to know
/// the sequence of steps and the type of each step from the setup phase so it can make sure the prover is not
/// cheating on this.
/// NOTE: The context automatically appends a requant step after each dense layer.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub enum LayerCtx<F: PrimeField> {
    Convolution(ConvCtx),
    Activation(ActivationCtx),
    Requant(RequantCtx),
    Pooling(PoolingCtx),
    EinSum(EinSumContext<F>),
    LayerNorm(LayerNormCtx),
    RMSNorm(RMSNormCtx),
    Flatten,
    Add(AddCtx),
    Softmax(SoftmaxCtx),
    Reshape(ReshapeCtx),
    Embeddings(EmbeddingsCtx),
    Positional(PositionalCtx),
    AttentionMask(AttentionMaskCtx),
    Logits(LogitsCtx),
    Split(SplitLayer),
    Recombination(RecombinationLayer),
    QwenNorm(QwenNormCtx),
    CausalConv1d(CausalConv1dCtx),
    GatedDeltaScan(GatedDeltaScanCtx),
    /// Binds stacked and contracted axes in one sumcheck. The old EinSum
    /// discriminant is retained exclusively for existing verifier artifacts.
    EinSumV2(EinSumContext<F>),
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub enum LayerProof<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    Convolution(Box<ConvProof<F>>),
    Activation(ActivationProof<F, PCS>),
    Requant(RequantProof<F, PCS>),
    Pooling(PoolingProof<F, PCS>),
    EinSum(EinSumProof<F>),
    Add(AddProof<F>),
    LayerNorm(LayerNormProof<F, PCS>),
    RMSNorm(RMSNormProof<F, PCS>),
    Softmax(SoftmaxProof<F, PCS>),
    Embeddings(EmbeddingsProof<F>),
    Logits(LogitsProof<F, PCS>),
    Positional(PositionalProof<F>),
    AttentionMask(AttentionMaskProof<F>),
    SplitLayer(SplitLayerProof<F>),
    Recombination(RecombinationProof<F>),
    Dummy, // To be used for non-provable layers
    QwenNorm(QwenNormProof<F, PCS>),
    CausalConv1d(CausalConv1dProof<F>),
    GatedDeltaScan(GatedDeltaScanProof<F, PCS>),
}

impl<F: PrimeField> LayerCtx<F> {
    pub fn variant_name(&self) -> String {
        match self {
            // Chunking uses the operation name; the serialized variant binds
            // the proof protocol independently of that scheduling label.
            Self::EinSum(_) | Self::EinSumV2(_) => "EinSum".to_string(),
            Self::LayerNorm(_) => "LayerNorm".to_string(),
            Self::RMSNorm(_) => "RMSNorm".to_string(),
            Self::QwenNorm(_) => "QwenNorm".to_string(),
            Self::GatedDeltaScan(_) => "GatedDeltaScan".to_string(),
            Self::CausalConv1d(_) => "CausalConv1d".to_string(),
            Self::Softmax(_) => "Softmax".to_string(),
            Self::Add(_) => "Add".to_string(),
            Self::Logits(_) => "Logits".to_string(),
            Self::Reshape(_) => "Reshape".to_string(),
            Self::Positional(_) => "Positional".to_string(),
            Self::Embeddings(_) => "Embeddings".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Flatten => "Reshape".to_string(),
            Self::AttentionMask(_) => "AttentionMask".to_string(),
            Self::Split(_) => "Split".to_string(),
            Self::Recombination(_) => "Recombination".to_string(),
        }
    }

    pub fn has_proof(&self) -> bool {
        !matches!(self, Self::Flatten | Self::Reshape(_))
    }

    pub fn next_shape_step(&self, last_step: &ShapeStep) -> Result<ShapeStep> {
        let unpadded_output =
            self.output_shapes(&last_step.unpadded_output_shape, PaddingMode::NoPadding)?;
        let padded_output =
            self.output_shapes(&last_step.padded_output_shape, PaddingMode::Padding)?;
        Ok(ShapeStep::next_step(
            last_step,
            unpadded_output,
            padded_output,
        ))
    }

    pub fn shape_step(
        &self,
        unpadded_input: &[Shape],
        padded_input: &[Shape],
    ) -> Result<ShapeStep> {
        if let LayerCtx::Split(split_layer) = self {
            // check that the unpadded input shapes are equal to the split layer unpadded input shapes, which is needed
            // to properly compute the padded output shapes of the layer
            ensure!(
                unpadded_input == split_layer.unpadded_input_shapes,
                "Unpadded input shapes are not equal to split layer unpadded input shapes"
            );
        }
        let unpadded_output = self.output_shapes(unpadded_input, PaddingMode::NoPadding)?;
        let padded_output = self.output_shapes(padded_input, PaddingMode::Padding)?;
        Ok(ShapeStep::new(
            unpadded_input.to_vec(),
            padded_input.to_vec(),
            unpadded_output,
            padded_output,
        ))
    }

    pub(crate) fn lookup_tables(&self) -> Option<Vec<Table>> {
        match self {
            LayerCtx::Convolution(_) => None,
            LayerCtx::Activation(activation_ctx) => {
                let inner_op = activation_ctx.op.activation_type();
                Some(inner_op.lookup_tables())
            }
            LayerCtx::Requant(requant_ctx) => Some(requant_ctx.lookup_tables()),
            LayerCtx::Pooling(pooling_ctx) => Some(pooling_ctx.lookup_tables()),
            LayerCtx::EinSum(_) | LayerCtx::EinSumV2(_) => None,
            LayerCtx::LayerNorm(layernorm_ctx) => Some(layernorm_ctx.lookup_tables()),
            LayerCtx::RMSNorm(rmsnorm_ctx) => Some(rmsnorm_ctx.lookup_tables()),
            LayerCtx::QwenNorm(rmsnorm_ctx) => Some(rmsnorm_ctx.lookup_tables()),
            LayerCtx::GatedDeltaScan(rmsnorm_ctx) => Some(rmsnorm_ctx.lookup_tables()),
            LayerCtx::CausalConv1d(_) => None,
            LayerCtx::Flatten => None,
            LayerCtx::Add(_) => None,
            LayerCtx::Softmax(softmax_ctx) => Some(softmax_ctx.lookup_tables()),
            LayerCtx::Reshape(_) => None,
            LayerCtx::Embeddings(_) => None,
            LayerCtx::Positional(_) => None,
            LayerCtx::AttentionMask(_) => None,
            LayerCtx::Logits(logits_ctx) => Some(logits_ctx.lookup_tables()),
            LayerCtx::Split(_) => None,
            LayerCtx::Recombination(_) => None,
        }
    }
}

impl<F: PrimeField> Splittable for LayerCtx<F> {
    fn ideal_num_chunks(&self, input_shapes: &[Shape]) -> Option<usize> {
        match self {
            LayerCtx::Convolution(conv_ctx) => conv_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Activation(activation_ctx) => activation_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Requant(requant_ctx) => requant_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::EinSum(ein_sum_context) | LayerCtx::EinSumV2(ein_sum_context) => {
                ein_sum_context.ideal_num_chunks(input_shapes)
            }
            LayerCtx::LayerNorm(layer_norm_ctx) => layer_norm_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::RMSNorm(rmsnorm_ctx) => rmsnorm_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::QwenNorm(rmsnorm_ctx) => rmsnorm_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::GatedDeltaScan(rmsnorm_ctx) => rmsnorm_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::CausalConv1d(rmsnorm_ctx) => rmsnorm_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Flatten => None,
            LayerCtx::Add(add_ctx) => add_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Reshape(reshape_ctx) => reshape_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Embeddings(embeddings_ctx) => embeddings_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Positional(positional_ctx) => positional_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::AttentionMask(attention_mask_ctx) => {
                attention_mask_ctx.ideal_num_chunks(input_shapes)
            }
            LayerCtx::Logits(logits_ctx) => logits_ctx.ideal_num_chunks(input_shapes),
            LayerCtx::Split(_) => None,
            LayerCtx::Recombination(_) => None,
        }
    }

    fn is_splittable(&self) -> bool {
        match self {
            LayerCtx::Convolution(conv_ctx) => conv_ctx.is_splittable(),
            LayerCtx::Activation(activation_ctx) => activation_ctx.is_splittable(),
            LayerCtx::Requant(requant_ctx) => requant_ctx.is_splittable(),
            LayerCtx::Pooling(pooling_ctx) => pooling_ctx.is_splittable(),
            LayerCtx::EinSum(ein_sum_context) | LayerCtx::EinSumV2(ein_sum_context) => {
                ein_sum_context.is_splittable()
            }
            LayerCtx::LayerNorm(layer_norm_ctx) => layer_norm_ctx.is_splittable(),
            LayerCtx::RMSNorm(rmsnorm_ctx) => rmsnorm_ctx.is_splittable(),
            LayerCtx::QwenNorm(rmsnorm_ctx) => rmsnorm_ctx.is_splittable(),
            LayerCtx::GatedDeltaScan(rmsnorm_ctx) => rmsnorm_ctx.is_splittable(),
            LayerCtx::CausalConv1d(rmsnorm_ctx) => rmsnorm_ctx.is_splittable(),
            LayerCtx::Flatten => false,
            LayerCtx::Add(add_ctx) => add_ctx.is_splittable(),
            LayerCtx::Softmax(softmax_ctx) => softmax_ctx.is_splittable(),
            LayerCtx::Reshape(reshape_ctx) => reshape_ctx.is_splittable(),
            LayerCtx::Embeddings(embeddings_ctx) => embeddings_ctx.is_splittable(),
            LayerCtx::Positional(positional_ctx) => positional_ctx.is_splittable(),
            LayerCtx::AttentionMask(attention_mask_ctx) => attention_mask_ctx.is_splittable(),
            LayerCtx::Logits(logits_ctx) => logits_ctx.is_splittable(),
            LayerCtx::Split(_) => false,
            LayerCtx::Recombination(_) => false,
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub(crate) struct NodeOut<T>
where
    T: TensorTypeParam,
{
    pub(crate) outputs: Vec<TensorHandle<T>>,
    pub(crate) proving_data: ProvingHandle,
}

impl<T> NodeOut<T>
where
    T: TensorTypeParam,
{
    pub(crate) fn new(outputs: Vec<TensorHandle<T>>, proving_data: ProvingHandle) -> Self {
        Self {
            outputs,
            proving_data,
        }
    }

    pub fn try_convdata(&self) -> Option<&ConvFFTHandle> {
        match self.proving_data {
            ProvingHandle::Convolution(ref conv_data) => Some(conv_data),
            _ => None,
        }
    }

    pub fn try_softmax_data(&self) -> Option<&SoftmaxHandle> {
        match self.proving_data {
            ProvingHandle::Softmax(ref softmax_data) => Some(softmax_data),
            _ => None,
        }
    }

    pub fn try_argmax_data(&self) -> Option<&ArgmaxHandle> {
        match self.proving_data {
            ProvingHandle::ArgMax(ref argmax_data) => Some(argmax_data),
            _ => None,
        }
    }

    pub fn try_layernorm_data(&self) -> Option<&LayerNormHandle> {
        match self.proving_data {
            ProvingHandle::LayerNorm(ref layernorm_data) => Some(layernorm_data),
            _ => None,
        }
    }

    pub fn try_rmsnorm_data(&self) -> Option<&RMSNormHandle> {
        match self.proving_data {
            ProvingHandle::RMSNorm(ref rmsnorm_data) => Some(rmsnorm_data),
            _ => None,
        }
    }
}

impl NodeOut<Element> {
    pub(crate) fn to_dequantize(
        &self,
        model_metadata: &ModelMetadata,
        node_id: NodeId,
    ) -> Result<NodeOut<f32>, StoreError> {
        let outputs = self
            .outputs
            .iter()
            .zip(model_metadata.layer_output_scaling_factor(node_id))
            .map(|(handle, scale_factor)| handle.cast(|x| scale_factor.dequantize(x)))
            .collect::<Result<Vec<_>, StoreError>>()?;

        Ok(NodeOut {
            outputs,
            proving_data: self.proving_data.clone(),
        })
    }
}

impl<N: TensorTypeParam> OpInfo for Layer<N> {
    fn output_shapes(
        &self,
        input_shapes: &[Shape],
        padding_mode: PaddingMode,
    ) -> Result<Vec<Shape>> {
        match self {
            Layer::Convolution(convolution) => {
                convolution.output_shapes(input_shapes, padding_mode)
            }
            Layer::Add(add) => add.output_shapes(input_shapes, padding_mode),
            Layer::Logits(logits) => logits.output_shapes(input_shapes, padding_mode),
            Layer::Positional(positional) => positional.output_shapes(input_shapes, padding_mode),
            Layer::LayerNorm(layernorm) => layernorm.output_shapes(input_shapes, padding_mode),
            Layer::RMSNorm(rmsnorm) => rmsnorm.output_shapes(input_shapes, padding_mode),
            Layer::QwenNorm(rmsnorm) => rmsnorm.output_shapes(input_shapes, padding_mode),
            Layer::CausalConv1d(rmsnorm) => rmsnorm.output_shapes(input_shapes, padding_mode),
            Layer::GatedDeltaScan(rmsnorm) => rmsnorm.output_shapes(input_shapes, padding_mode),
            Layer::Softmax(softmax) => softmax.output_shapes(input_shapes, padding_mode),
            Layer::Embeddings(embeddings) => embeddings.output_shapes(input_shapes, padding_mode),
            Layer::Reshape(reshape) => reshape.output_shapes(input_shapes, padding_mode),
            Layer::Activation(activation) => activation.output_shapes(input_shapes, padding_mode),
            Layer::Requant(requant) => requant.output_shapes(input_shapes, padding_mode),
            Layer::Pooling(pooling) => pooling.output_shapes(input_shapes, padding_mode),
            Layer::Flatten(reshape) => reshape.output_shapes(input_shapes, padding_mode),
            Layer::AttentionMask(attention_mask) => {
                attention_mask.output_shapes(input_shapes, padding_mode)
            }
            Layer::EinSum(einsum) => einsum.output_shapes(input_shapes, padding_mode),
            Layer::Split(split) => split.output_shapes(input_shapes, padding_mode),
            Layer::Recombination(rec) => rec.output_shapes(input_shapes, padding_mode),
        }
    }

    fn num_outputs(&self, num_inputs: usize) -> Result<usize> {
        match self {
            Layer::Convolution(convolution) => convolution.num_outputs(num_inputs),
            Layer::LayerNorm(layernorm) => layernorm.num_outputs(num_inputs),
            Layer::RMSNorm(rmsnorm) => rmsnorm.num_outputs(num_inputs),
            Layer::QwenNorm(rmsnorm) => rmsnorm.num_outputs(num_inputs),
            Layer::CausalConv1d(rmsnorm) => rmsnorm.num_outputs(num_inputs),
            Layer::GatedDeltaScan(rmsnorm) => rmsnorm.num_outputs(num_inputs),
            Layer::Softmax(softmax) => softmax.num_outputs(num_inputs),
            Layer::Add(add) => add.num_outputs(num_inputs),
            Layer::Logits(logits) => logits.num_outputs(num_inputs),
            Layer::Reshape(reshape) => reshape.num_outputs(num_inputs),
            Layer::Positional(positional) => positional.num_outputs(num_inputs),
            Layer::Embeddings(embeddings) => embeddings.num_outputs(num_inputs),
            Layer::Activation(activation) => activation.num_outputs(num_inputs),
            Layer::Requant(requant) => requant.num_outputs(num_inputs),
            Layer::Pooling(pooling) => pooling.num_outputs(num_inputs),
            Layer::Flatten(reshape) => reshape.num_outputs(num_inputs),
            Layer::AttentionMask(attention_mask) => attention_mask.num_outputs(num_inputs),
            Layer::EinSum(einsum) => einsum.num_outputs(num_inputs),
            Layer::Split(split) => split.num_outputs(num_inputs),
            Layer::Recombination(rec) => rec.num_outputs(num_inputs),
        }
    }

    fn describe(&self) -> String {
        match self {
            Layer::Convolution(convolution) => convolution.describe(),
            Layer::LayerNorm(layernorm) => layernorm.describe(),
            Layer::RMSNorm(rmsnorm) => rmsnorm.describe(),
            Layer::QwenNorm(rmsnorm) => rmsnorm.describe(),
            Layer::CausalConv1d(rmsnorm) => rmsnorm.describe(),
            Layer::GatedDeltaScan(rmsnorm) => rmsnorm.describe(),
            Layer::Softmax(softmax) => softmax.describe(),
            Layer::Add(add) => add.describe(),
            Layer::Logits(logits) => logits.describe(),
            Layer::Positional(positional) => positional.describe(),
            Layer::Reshape(reshape) => reshape.describe(),
            Layer::Embeddings(embeddings) => embeddings.describe(),
            Layer::Activation(activation) => activation.describe(),
            Layer::Requant(requant) => requant.describe(),
            Layer::Pooling(pooling) => pooling.describe(),
            Layer::Flatten(reshape) => reshape.describe(),
            Layer::AttentionMask(attention_mask) => attention_mask.describe(),
            Layer::EinSum(einsum) => einsum.describe(),
            Layer::Split(split) => split.describe(),
            Layer::Recombination(rec) => rec.describe(),
        }
    }

    fn is_provable(&self) -> bool {
        match self {
            Layer::Convolution(convolution) => convolution.is_provable(),
            Layer::LayerNorm(layernorm) => layernorm.is_provable(),
            Layer::RMSNorm(rmsnorm) => rmsnorm.is_provable(),
            Layer::QwenNorm(rmsnorm) => rmsnorm.is_provable(),
            Layer::CausalConv1d(rmsnorm) => rmsnorm.is_provable(),
            Layer::GatedDeltaScan(rmsnorm) => rmsnorm.is_provable(),
            Layer::Softmax(softmax) => softmax.is_provable(),
            Layer::Positional(positional) => positional.is_provable(),
            Layer::Add(add) => add.is_provable(),
            Layer::Logits(logits) => logits.is_provable(),
            Layer::Reshape(reshape) => reshape.is_provable(),
            Layer::Embeddings(embeddings) => embeddings.is_provable(),
            Layer::Activation(activation) => activation.is_provable(),
            Layer::Requant(requant) => requant.is_provable(),
            Layer::Pooling(pooling) => pooling.is_provable(),
            Layer::Flatten(reshape) => reshape.is_provable(),
            Layer::AttentionMask(attention_mask) => attention_mask.is_provable(),
            Layer::EinSum(einsum) => einsum.is_provable(),
            Layer::Split(split) => split.is_provable(),
            Layer::Recombination(rec) => rec.is_provable(),
        }
    }
}

impl Evaluate<f32> for Layer<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        match self {
            Layer::Convolution(convolution) => convolution.evaluate(inputs),
            Layer::LayerNorm(layernorm) => layernorm.evaluate(inputs),
            Layer::RMSNorm(rmsnorm) => rmsnorm.evaluate(inputs),
            Layer::QwenNorm(rmsnorm) => rmsnorm.evaluate(inputs),
            Layer::CausalConv1d(rmsnorm) => rmsnorm.evaluate(inputs),
            Layer::GatedDeltaScan(rmsnorm) => rmsnorm.evaluate(inputs),
            Layer::Softmax(softmax) => softmax.evaluate(inputs),
            Layer::Add(add) => add.evaluate(inputs),
            Layer::Logits(logits) => logits.evaluate(inputs),
            Layer::Positional(positional) => positional.evaluate(inputs),
            Layer::Reshape(reshape) => reshape.evaluate(inputs),
            Layer::Embeddings(embeddings) => embeddings.evaluate(inputs),
            Layer::Activation(activation) => activation.evaluate(inputs),
            Layer::Requant(_) => unreachable!("Requant layer found when evaluating over float"),
            Layer::Pooling(pooling) => pooling.evaluate(inputs),
            Layer::Flatten(reshape) => reshape.evaluate(inputs),
            Layer::AttentionMask(attention_mask) => attention_mask.evaluate(inputs),
            Layer::EinSum(einsum) => einsum.evaluate(inputs),
            Layer::Split(split) => split.evaluate(inputs),
            Layer::Recombination(rec) => rec.evaluate(inputs),
        }
    }
}

impl Evaluate<Element> for Layer<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        let output = match self {
            Layer::Convolution(convolution) => convolution.evaluate(inputs),
            Layer::LayerNorm(layernorm) => layernorm.evaluate(inputs),
            Layer::RMSNorm(rmsnorm) => rmsnorm.evaluate(inputs),
            Layer::QwenNorm(rmsnorm) => rmsnorm.evaluate(inputs),
            Layer::CausalConv1d(rmsnorm) => rmsnorm.evaluate(inputs),
            Layer::GatedDeltaScan(rmsnorm) => rmsnorm.evaluate(inputs),
            Layer::Softmax(softmax) => softmax.evaluate(inputs),
            Layer::Add(add) => add.evaluate(inputs),
            Layer::Logits(logits) => logits.evaluate(inputs),
            Layer::Positional(positional) => positional.evaluate(inputs),
            Layer::Embeddings(embeddings) => embeddings.evaluate(inputs),
            Layer::Reshape(reshape) => reshape.evaluate(inputs),
            Layer::Activation(activation) => activation.evaluate(inputs),
            Layer::Requant(requant) => requant.evaluate(inputs),
            Layer::Pooling(pooling) => pooling.evaluate(inputs),
            Layer::Flatten(reshape) => reshape.evaluate(inputs),
            Layer::AttentionMask(attention_mask) => attention_mask.evaluate(inputs),
            Layer::EinSum(einsum) => einsum.evaluate(inputs),
            Layer::Split(split) => split.evaluate(inputs),
            Layer::Recombination(rec) => rec.evaluate(inputs),
        };

        #[cfg(feature = "capture-layers-quant")]
        {
            if let Ok(output) = output.as_ref() {
                let layer_kind = self.as_kind_str();
                let out_dir = std::path::PathBuf::from("layers-quant").join(layer_kind);
                crate::capture::store(&out_dir, &(self, inputs), &output.outputs);
            }
        }

        output
    }
}

impl ProveInfo for Layer<Element> {
    fn step_info<F: PrimeField>(&self, aux: ContextAux) -> Result<(LayerCtx<F>, ContextAux)> {
        match self {
            Layer::Add(add) => add.step_info(aux),
            Layer::LayerNorm(layernorm) => layernorm.step_info(aux),
            Layer::RMSNorm(rmsnorm) => rmsnorm.step_info(aux),
            Layer::QwenNorm(rmsnorm) => rmsnorm.step_info(aux),
            Layer::CausalConv1d(rmsnorm) => rmsnorm.step_info(aux),
            Layer::GatedDeltaScan(rmsnorm) => rmsnorm.step_info(aux),
            Layer::Softmax(softmax) => softmax.step_info(aux),
            Layer::Logits(logits) => logits.step_info(aux),
            Layer::Positional(positional) => positional.step_info(aux),
            Layer::Embeddings(embeddings) => embeddings.step_info(aux),
            Layer::Reshape(reshape) => reshape.step_info(aux),
            Layer::Convolution(conv) => conv.step_info(aux),
            Layer::Activation(activation) => activation.step_info(aux),
            Layer::Requant(requant) => requant.step_info(aux),
            Layer::Pooling(pooling) => pooling.step_info(aux),
            Layer::Flatten(reshape) => reshape.step_info(aux),
            Layer::AttentionMask(attention_mask) => attention_mask.step_info(aux),
            Layer::EinSum(einsum) => einsum.step_info(aux),
            Layer::Split(split) => split.step_info(aux),
            Layer::Recombination(rec) => rec.step_info(aux),
        }
    }
}

impl PadOp for Layer<Element> {
    fn pad_node(self, si: &mut ShapeInfo) -> Result<Self>
    where
        Self: Sized,
    {
        Ok(match self {
            Layer::Convolution(convolution) => Layer::Convolution(convolution.pad_node(si)?),
            Layer::Add(add) => Layer::Add(add.pad_node(si)?),
            Layer::LayerNorm(layernorm) => Layer::LayerNorm(layernorm.pad_node(si)?),
            Layer::RMSNorm(rmsnorm) => Layer::RMSNorm(rmsnorm.pad_node(si)?),
            Layer::QwenNorm(rmsnorm) => Layer::QwenNorm(rmsnorm.pad_node(si)?),
            Layer::CausalConv1d(rmsnorm) => Layer::CausalConv1d(rmsnorm.pad_node(si)?),
            Layer::GatedDeltaScan(rmsnorm) => Layer::GatedDeltaScan(rmsnorm.pad_node(si)?),
            Layer::Softmax(softmax) => Layer::Softmax(softmax.pad_node(si)?),
            Layer::Logits(logits) => Layer::Logits(logits.pad_node(si)?),
            Layer::Positional(positional) => Layer::Positional(positional.pad_node(si)?),
            Layer::Embeddings(embeddings) => Layer::Embeddings(embeddings.pad_node(si)?),
            Layer::Activation(activation) => Layer::Activation(activation.pad_node(si)?),
            Layer::Requant(requant) => Layer::Requant(requant.pad_node(si)?),
            Layer::Pooling(pooling) => Layer::Pooling(pooling.pad_node(si)?),
            Layer::Flatten(flatten) => Layer::Flatten(flatten.pad_node(si)?),
            Layer::Reshape(reshape) => Layer::Reshape(reshape.pad_node(si)?),
            Layer::AttentionMask(attention_mask) => {
                Layer::AttentionMask(attention_mask.pad_node(si)?)
            }
            Layer::EinSum(einsum) => Layer::EinSum(einsum.pad_node(si)?),
            Layer::Split(split) => Layer::Split(split.pad_node(si)?),
            Layer::Recombination(rec) => Layer::Recombination(rec.pad_node(si)?),
        })
    }
}

impl<F, PCS> ProvableOp<F, PCS> for Layer<Element>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    type Ctx = LayerCtx<F>;

    fn prove<'a, 'b, 'c, 'd, T: Transcript + InitTranscript>(
        &'a self,
        node_id: NodeId,
        ctx: &'b Self::Ctx,
        last_claims: Vec<&crate::Claim<F>>,
        step_data: &Step<Element>,
        prover: &mut crate::Prover<'c, 'd, F, T, PCS>,
    ) -> Result<Vec<crate::Claim<F>>> {
        match (self, ctx) {
            (Layer::Convolution(convolution), LayerCtx::Convolution(info)) => {
                convolution.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Embeddings(embeddings), LayerCtx::Embeddings(ctx)) => {
                embeddings.prove(node_id, ctx, last_claims, step_data, prover)
            }
            (Layer::Positional(positional), LayerCtx::Positional(info)) => {
                positional.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Add(add), LayerCtx::Add(info)) => {
                add.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Logits(logits), LayerCtx::Logits(info)) => {
                logits.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Activation(activation), LayerCtx::Activation(info)) => {
                activation.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Requant(requant), LayerCtx::Requant(info)) => {
                requant.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Pooling(pooling), LayerCtx::Pooling(info)) => {
                pooling.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Flatten(_), LayerCtx::Flatten) => {
                unreachable!("prove cannot be called for reshape")
            }
            (Layer::Softmax(softmax), LayerCtx::Softmax(info)) => {
                softmax.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::LayerNorm(layernorm), LayerCtx::LayerNorm(info)) => {
                layernorm.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::RMSNorm(rmsnorm), LayerCtx::RMSNorm(info)) => {
                rmsnorm.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::QwenNorm(rmsnorm), LayerCtx::QwenNorm(info)) => {
                rmsnorm.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::GatedDeltaScan(rmsnorm), LayerCtx::GatedDeltaScan(info)) => {
                rmsnorm.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::CausalConv1d(rmsnorm), LayerCtx::CausalConv1d(info)) => {
                rmsnorm.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::AttentionMask(attention_mask), LayerCtx::AttentionMask(info)) => {
                attention_mask.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::EinSum(einsum), LayerCtx::EinSumV2(info)) => {
                einsum.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Split(split), LayerCtx::Split(info)) => {
                split.prove(node_id, info, last_claims, step_data, prover)
            }
            (Layer::Recombination(rec), LayerCtx::Recombination(info)) => {
                rec.prove(node_id, info, last_claims, step_data, prover)
            }
            _ => bail!(
                "Incompatible layer {} and ctx {} found for node id {}",
                self.describe(),
                ctx.variant_name(),
                node_id
            ),
        }
    }

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        ctx: &ProverContext<F, PCS>,
        step_data: &Step<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        match self {
            Layer::Convolution(convolution) => convolution.gen_lookup_witness(id, ctx, step_data),
            Layer::Add(add) => add.gen_lookup_witness(id, ctx, step_data),
            Layer::Softmax(softmax) => softmax.gen_lookup_witness(id, ctx, step_data),
            Layer::Logits(logits) => logits.gen_lookup_witness(id, ctx, step_data),
            Layer::LayerNorm(layernorm) => layernorm.gen_lookup_witness(id, ctx, step_data),
            Layer::RMSNorm(rmsnorm) => rmsnorm.gen_lookup_witness(id, ctx, step_data),
            Layer::QwenNorm(rmsnorm) => rmsnorm.gen_lookup_witness(id, ctx, step_data),
            Layer::CausalConv1d(rmsnorm) => rmsnorm.gen_lookup_witness(id, ctx, step_data),
            Layer::GatedDeltaScan(scan) => scan.gen_lookup_witness(id, ctx, step_data),
            Layer::Positional(positional) => positional.gen_lookup_witness(id, ctx, step_data),
            Layer::Embeddings(embeddings) => embeddings.gen_lookup_witness(id, ctx, step_data),
            Layer::Activation(activation) => activation.gen_lookup_witness(id, ctx, step_data),
            Layer::Requant(requant) => requant.gen_lookup_witness(id, ctx, step_data),
            Layer::Pooling(pooling) => pooling.gen_lookup_witness(id, ctx, step_data),
            Layer::Reshape(r) => {
                ensure!(!r.is_provable(), "reshape {r:?} is provable");
                Ok(Default::default())
            }
            Layer::Flatten(r) => {
                ensure!(!r.is_provable(), "flatten {r:?} is provable");
                Ok(Default::default())
            }
            Layer::AttentionMask(attention_mask) => {
                attention_mask.gen_lookup_witness(id, ctx, step_data)
            }
            Layer::EinSum(einsum) => einsum.gen_lookup_witness(id, ctx, step_data),
            Layer::Split(split) => split.gen_lookup_witness(id, ctx, step_data),
            Layer::Recombination(rec) => rec.gen_lookup_witness(id, ctx, step_data),
        }
    }
}

impl QuantizeOp for Layer<f32> {
    type QuantizedOp = Layer<Element>;

    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        node_id: NodeId,
        input_scaling: &[ScalingFactor],
        unpadded_input_shapes: &[Shape],
        output_scalings: &[ScalingFactor],
        unpadded_output_shapes: &[Shape],
    ) -> anyhow::Result<QuantizeOutput<Self::QuantizedOp>> {
        Ok(match self {
            Layer::Convolution(convolution) => {
                let output = convolution.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(
                    Layer::Convolution(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)?
                .maybe_transform(output.post_quant_rule)?
            }
            Layer::LayerNorm(layernorm) => {
                let output = layernorm.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(
                    Layer::LayerNorm(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)?
                .maybe_transform(output.post_quant_rule)?
            }
            Layer::RMSNorm(rmsnorm) => {
                let output = rmsnorm.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(Layer::RMSNorm(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)?
                    .maybe_transform(output.post_quant_rule)?
            }
            Layer::QwenNorm(rmsnorm) => {
                let output = rmsnorm.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(Layer::QwenNorm(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)?
                    .maybe_transform(output.post_quant_rule)?
            }
            Layer::CausalConv1d(rmsnorm) => {
                let output = rmsnorm.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(
                    Layer::CausalConv1d(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)?
                .maybe_transform(output.post_quant_rule)?
            }
            Layer::GatedDeltaScan(rmsnorm) => {
                let output = rmsnorm.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(
                    Layer::GatedDeltaScan(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)?
                .maybe_transform(output.post_quant_rule)?
            }
            Layer::Softmax(softmax) => {
                let output = softmax.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(Layer::Softmax(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)?
                    .maybe_transform(output.post_quant_rule)?
            }
            Layer::Add(add) => {
                let output = add.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(Layer::Add(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)?
                    .maybe_transform(output.post_quant_rule)?
            }
            Layer::Logits(logits) => {
                let output = logits.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(Layer::Logits(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)?
                    .maybe_transform(output.post_quant_rule)?
            }
            Layer::Positional(positional) => {
                let output = positional.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(
                    Layer::Positional(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)?
                .maybe_transform(output.post_quant_rule)?
            }
            Layer::Embeddings(embeddings) => {
                let output = embeddings.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(
                    Layer::Embeddings(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)?
                .maybe_transform(output.post_quant_rule)?
            }
            Layer::Activation(activation) => {
                let output = activation.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(
                    Layer::Activation(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)?
            }
            Layer::Requant(requant) => {
                QuantizeOutput::new(Layer::Requant(requant), input_scaling.to_vec())
            }
            Layer::Pooling(pooling) => {
                QuantizeOutput::new(Layer::Pooling(pooling), input_scaling.to_vec())
            }
            Layer::Flatten(flatten) => {
                QuantizeOutput::new(Layer::Flatten(flatten), input_scaling.to_vec())
            }
            Layer::Reshape(reshape) => {
                QuantizeOutput::new(Layer::Reshape(reshape), input_scaling.to_vec())
            }
            Layer::AttentionMask(attention_mask) => {
                let output = attention_mask.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(
                    Layer::AttentionMask(output.quantized_op),
                    output.output_scalings,
                )
                .maybe_requants(output.requant_layer)?
                .maybe_transform(output.post_quant_rule)?
            }
            Layer::EinSum(einsum) => {
                let output = einsum.quantize_op::<S>(
                    data,
                    node_id,
                    input_scaling,
                    unpadded_input_shapes,
                    output_scalings,
                    unpadded_output_shapes,
                )?;
                QuantizeOutput::new(Layer::EinSum(output.quantized_op), output.output_scalings)
                    .maybe_requants(output.requant_layer)?
                    .maybe_transform(output.post_quant_rule)?
            }
            Layer::Split(split) => QuantizeOutput::new(Layer::Split(split), input_scaling.to_vec()),
            Layer::Recombination(rec) => {
                QuantizeOutput::new(Layer::Recombination(rec), input_scaling.to_vec())
            }
        })
    }
}

impl<F, PCS> LayerProof<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    pub fn variant_name(&self) -> String {
        match self {
            Self::LayerNorm(_) => "LayerNorm".to_string(),
            Self::RMSNorm(_) => "RMSNorm".to_string(),
            Self::QwenNorm(_) => "QwenNorm".to_string(),
            Self::GatedDeltaScan(_) => "GatedDeltaScan".to_string(),
            Self::CausalConv1d(_) => "CausalConv1d".to_string(),
            Self::Softmax(_) => "Softmax".to_string(),
            Self::Logits(_) => "Logits".to_string(),
            Self::Positional(_) => "Positional".to_string(),
            Self::Add(_) => "Add".to_string(),
            Self::Embeddings(_) => "Embeddings".to_string(),
            Self::Convolution(_) => "Convolution".to_string(),
            Self::Activation(_) => "Activation".to_string(),
            Self::Requant(_) => "Requant".to_string(),
            Self::Pooling(_) => "Pooling".to_string(),
            Self::Dummy => "Dummy".to_string(),
            Self::AttentionMask(_) => "AttentionMask".to_string(),
            Self::EinSum(_) => "EinSum".to_string(),
            Self::SplitLayer(_) => "Split".to_string(),
            Self::Recombination(_) => "Recombination".to_string(),
        }
    }
}

impl<T> std::fmt::Display for Layer<T>
where
    T: TensorTypeParam,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.describe())
    }
}
