//! Depthwise causal convolution. The kernel follows PyTorch's cross-correlation order.
use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use ark_ff::PrimeField;
use dp_crypto::{IntoMLE, arkyper::{CommitmentScheme, transcript::Transcript}, poly::eq::evals,
    structs::{IOPProof, IOPProverState, IOPVerifierState}, util::optimal_sumcheck_threads,
    virtual_polys::VirtualPolynomialsBuilder};
use either::Either;
use serde::{Deserialize, Serialize};

use crate::{Claim, Element, NextPowerOfTwo, Prover, ScalingFactor, ScalingStrategy, Shape, Tensor,
    graph::NodeId, iop::{context::{ContextAux, ShapeStep}, verifier::Verifier},
    layers::{LayerCtx, LayerProof, provable::{Evaluate, LayerOut, OpInfo, PadOp, ProvableOp,
        ProveInfo, QuantizeOp, QuantizeOutput, Splittable, VerifiableCtx}, requant::Requant},
    model::Step, padding::{PaddingMode, ShapeInfo}, poly_commit::identity_eval,
    quantization::{Quantize, ToField}, tensor::{CommitmentId, KeyedTensor, TensorTypeParam, WrappedTensor}};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalConv1d<N> {
    weight: KeyedTensor<N>,
    #[serde(skip)]
    history: Arc<Mutex<Vec<N>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CausalConv1dCtx {
    channels: usize,
    weight_key: CommitmentId,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "F: ark_serialize::CanonicalSerialize", deserialize = "F: ark_serialize::CanonicalDeserialize"))]
pub struct CausalConv1dProof<F: PrimeField> {
    convolution: IOPProof<F>,
    shift: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    evaluations: Vec<F>,
}

impl<N: TensorTypeParam> CausalConv1d<N> {
    pub fn new(weight: KeyedTensor<N>) -> Result<Self> {
        ensure!(weight.shape().rank() == 2 && weight.shape().dim(1) == 4,
            "CausalConv1d expects [channels,4] weights");
        Ok(Self { weight, history: Default::default() })
    }
    pub fn reset_cache(&self) { self.history.lock().unwrap().clear(); }
    /// Static convolution arithmetic; model weights are fingerprinted separately.
    pub(crate) fn quantization_summary(&self) -> serde_json::Value {
        serde_json::json!({
            "weight_shape": self.weight.shape(),
            "weight_unpadded_shape": self.weight.unpadded_shape(),
            "kernel_order": "pytorch-cross-correlation",
            "accumulation": "checked-i128-to-i64",
            "initial_history": "zero",
        })
    }
    fn evaluate_impl(&self, inputs: &[&WrappedTensor<N>], mac: impl Fn(N,N,N)->Result<N>) -> Result<LayerOut<N>> {
        ensure!(inputs.len() == 1, "CausalConv1d expects one input");
        let input = inputs[0];
        let shape = Shape::from(input.shape());
        let unpadded = Shape::from(input.unpadded_shape());
        ensure!(shape.rank() == 2 && unpadded.dim(1) == self.weight.unpadded_shape().dim(0), "Convolution shape mismatch");
        let channels = shape.dim(1);
        let valid_channels = unpadded.dim(1);
        let values = input.get_data();
        let weights = self.weight.data();
        let mut history = self.history.lock().unwrap();
        if history.is_empty() { history.resize(3 * channels, N::zero()); }
        ensure!(history.len() == 3 * channels, "Convolution cache shape changed without reset");
        let mut output = vec![N::zero(); shape.product()];
        for t in 0..unpadded.dim(0) {
            for c in 0..valid_channels {
                let mut sum = N::zero();
                for j in 0..4 {
                    let x = if j == 3 { values[t*channels+c] } else { history[j*channels+c] };
                    sum = mac(sum, x, weights[c*4+j])?;
                }
                output[t*channels+c] = sum;
            }
            history.copy_within(channels..3*channels, 0);
            history[2*channels..].copy_from_slice(&values[t*channels..(t+1)*channels]);
        }
        let tensor = Tensor::new_with_unpadded_shape(shape, unpadded, output)?;
        Ok(LayerOut::from_tensor(WrappedTensor::try_from(&tensor)?))
    }
}

impl Evaluate<f32> for CausalConv1d<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        self.evaluate_impl(inputs, |s,x,w| Ok(s + x*w))
    }
}
impl Evaluate<Element> for CausalConv1d<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        self.evaluate_impl(inputs, |s,x,w| Ok(i64::try_from(s as i128 + x as i128 * w as i128)?))
    }
}

fn output_shapes(input: &[Shape], channels: usize, padding: PaddingMode) -> Result<Vec<Shape>> {
    ensure!(input.len()==1 && input[0].rank()==2, "Convolution expects [sequence,channels]");
    let expected = if matches!(padding, PaddingMode::Padding) { channels.next_power_of_two() } else { channels };
    ensure!(input[0].dim(1)==expected, "Convolution channel count mismatch");
    Ok(input.to_vec())
}
impl<N: TensorTypeParam> OpInfo for CausalConv1d<N> {
    fn num_outputs(&self,n:usize)->Result<usize> { ensure!(n==1,"Expected one convolution input"); Ok(1) }
    fn output_shapes(&self,s:&[Shape],p:PaddingMode)->Result<Vec<Shape>> { output_shapes(s,self.weight.unpadded_shape().dim(0),p) }
    fn describe(&self)->String { "CausalConv1d".into() }
    fn is_provable(&self)->bool { true }
}
impl OpInfo for CausalConv1dCtx {
    fn num_outputs(&self,n:usize)->Result<usize> { ensure!(n==1,"Expected one convolution input"); Ok(1) }
    fn output_shapes(&self,s:&[Shape],p:PaddingMode)->Result<Vec<Shape>> { output_shapes(s,self.channels,p) }
    fn describe(&self)->String { "CausalConv1d".into() }
    fn is_provable(&self)->bool { true }
}
impl Splittable for CausalConv1dCtx {}
impl PadOp for CausalConv1d<Element> {
    fn pad_node(mut self,_:&mut ShapeInfo)->Result<Self> {
        self.weight = self.weight.try_map_tensor(|t| Ok(t.pad_next_power_of_two()))?;
        self.reset_cache(); Ok(self)
    }
}
impl QuantizeOp for CausalConv1d<f32> {
    type QuantizedOp = CausalConv1d<Element>;
    fn quantize_op<S:ScalingStrategy>(self,_:&S::AuxData,_:NodeId,input:&[ScalingFactor],_:&[Shape],output:&[ScalingFactor],_:&[Shape])->Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(input.len()==1 && output.len()==1,"Convolution scales mismatch");
        let sf = ScalingFactor::from_absolute_max(self.weight.max_abs(), None);
        let bits=input[0].bit_size()+sf.bit_size()+3;
        ensure!(bits + crate::layers::requant::FIXED_POINT_SCALE <= 63,"Convolution exceeds integer range");
        let rq=Requant::from_scaling_factors(input[0],sf,output[0],bits);
        QuantizeOutput::new(CausalConv1d::new(self.weight.quantize(&sf))?,output.to_vec()).with_requant(rq)
    }
}
impl ProveInfo for CausalConv1d<Element> {
    fn step_info<F:PrimeField>(&self,mut aux:ContextAux)->Result<(LayerCtx<F>,ContextAux)> {
        let weight=self.weight.tensor().clone().pad_next_power_of_two();
        aux.max_poly_len=aux.max_poly_len.max(aux.last_output_shape[0].next_power_of_two().product()*4);
        aux.model_polys=Some([(self.weight.commitment_id(),weight.data().to_vec())].into());
        Ok((LayerCtx::CausalConv1d(CausalConv1dCtx{channels:self.weight.unpadded_shape().dim(0),weight_key:self.weight.commitment_id()}),aux))
    }
}

/// Adjoint of the public causal gather: reduces a claim on x[t-3+j,c] to x.
fn shift_coefficients<F:PrimeField>(point:&[F], channels:usize, rows:usize, valid_rows:usize, valid_channels:usize)->Vec<F> {
    let cbits=channels.ilog2() as usize;
    let kernel=evals(&point[..2]);
    let channel=evals(&point[2..2+cbits]);
    let time=evals(&point[2+cbits..]);
    let mut out=vec![F::ZERO;rows*channels];
    for t in 0..valid_rows {
        let mut value=F::ZERO;
        for j in 0..4 { if t+3-j < valid_rows { value += kernel[j]*time[t+3-j]; } }
        for c in 0..valid_channels { out[t*channels+c]=value*channel[c]; }
    }
    out
}

impl<F:PrimeField,PCS:CommitmentScheme<Field=F>> ProvableOp<F,PCS> for CausalConv1d<Element> {
    type Ctx=CausalConv1dCtx;
    fn prove<'a,'b,'c,'d,T:Transcript>(&'a self,id:NodeId,_:&'b Self::Ctx,last:Vec<&Claim<F>>,step:&Step<Element>,prover:&mut Prover<'c,'d,F,T,PCS>)->Result<Vec<Claim<F>>> {
        ensure!(last.len()==1,"Convolution output claim count");
        let input=step.padded_input_tensor_at(0)?;
        let rows=input.shape().dim(0); let channels=input.shape().dim(1);
        let valid_rows=input.unpadded_shape().dim(0); let valid_channels=input.unpadded_shape().dim(1);
        let weight=self.weight.tensor().clone().pad_next_power_of_two();
        let beta=evals(&last[0].point);
        let mut shifted=vec![F::ZERO;rows*channels*4]; let mut w=shifted.clone(); let mut b=shifted.clone();
        for t in 0..rows { for c in 0..channels { for j in 0..4 {
            let i=(t*channels+c)*4+j;
            w[i]=weight.data()[c*4+j].to_field(); b[i]=beta[t*channels+c];
            if t<valid_rows && c<valid_channels && t+j>=3 { shifted[i]=input.data()[(t+j-3)*channels+c].to_field(); }
        }}}
        let polys=[shifted.into_mle(),w.into_mle(),b.into_mle()];
        let n=polys[0].num_vars(); let mut builder=VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(n),n);
        let a=builder.lift(Either::Left(&polys[0])); let w=builder.lift(Either::Left(&polys[1])); let b=builder.lift(Either::Left(&polys[2]));
        let (convolution,state)=IOPProverState::prove(builder.to_virtual_polys(&[a*w*b],&[]),prover.transcript);
        ensure!(convolution.extract_sum()==last[0].eval,"Invalid convolution output witness");
        let point=state.collect_raw_challenges(); let values=state.get_mle_flatten_final_evaluations();
        let coeff=shift_coefficients(&point,channels,rows,valid_rows,valid_channels).into_mle();
        let x=input.to_field_mle(); let n=x.num_vars();
        let mut builder=VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(n),n);
        let a=builder.lift(Either::Left(&x)); let b=builder.lift(Either::Left(&coeff));
        let (shift,state2)=IOPProverState::prove(builder.to_virtual_polys(&[a*b],&[]),prover.transcript);
        ensure!(shift.extract_sum()==values[0],"Invalid convolution history witness");
        let input_eval=state2.get_mle_flatten_final_evaluations()[0];
        prover.add_common_claims(id,[(self.weight.commitment_id(),Claim::new(point[..2+channels.ilog2() as usize].to_vec(),values[1]))].into());
        prover.push_proof(id,LayerProof::CausalConv1d(CausalConv1dProof{convolution,shift,evaluations:vec![values[0],values[1],input_eval]}));
        Ok(vec![Claim::new(state2.collect_raw_challenges(),input_eval)])
    }
}
impl<F:PrimeField,PCS:CommitmentScheme<Field=F>> VerifiableCtx<F,PCS> for CausalConv1dCtx {
    type Proof=CausalConv1dProof<F>;
    fn write_proof_to_transcript<T:Transcript>(&self,_:&Self::Proof,_:&mut T)->Result<()> { Ok(()) }
    fn verify<T:Transcript>(&self,proof:&Self::Proof,last:&[&Claim<F>],verifier:&mut Verifier<F,T,PCS>,shapes:&ShapeStep,id:NodeId)->Result<Vec<Claim<F>>> {
        ensure!(last.len()==1 && proof.evaluations.len()==3,"Malformed convolution proof");
        let shape=&shapes.padded_input_shape[0]; let original=&shapes.unpadded_input_shape[0];
        let rows=shape.dim(0); let channels=shape.dim(1); let n=(rows*channels).ilog2() as usize;
        ensure!(last[0].point.len()==n,"Convolution claim dimension mismatch");
        let aux=crate::util::from_mle_list_dimensions(&[vec![n+2;3]]);
        let sub=IOPVerifierState::verify(last[0].eval,&proof.convolution,&aux,verifier.transcript);
        let beta=identity_eval(&last[0].point,&sub.point[2..]);
        ensure!(sub.expected_evaluation==proof.evaluations[0]*proof.evaluations[1]*beta,"Convolution product failed");
        let aux=crate::util::from_mle_list_dimensions(&[vec![n,n]]);
        let shift=IOPVerifierState::verify(proof.evaluations[0],&proof.shift,&aux,verifier.transcript);
        let coeff=shift_coefficients(&sub.point,channels,rows,original.dim(0),original.dim(1)).into_mle();
        ensure!(shift.expected_evaluation==proof.evaluations[2]*coeff.evaluate(&shift.point)?,"Convolution history binding failed");
        verifier.add_common_claims(id,[(self.weight_key.clone(),Claim::new(sub.point[..2+channels.ilog2() as usize].to_vec(),proof.evaluations[1]))].into());
        Ok(vec![Claim::new(shift.point,proof.evaluations[2])])
    }
}

#[cfg(test)]
mod tests;
