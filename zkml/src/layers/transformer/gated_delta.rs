//! Gated DeltaNet recurrence, using one layer per sequence rather than a token-expanded graph.
use crate::{
    Element, ScalingFactor, ScalingStrategy, Shape, Tensor,
    graph::NodeId,
    iop::context::ContextAux,
    layers::{
        LayerCtx,
        provable::{Evaluate, LayerOut, OpInfo, PadOp, ProveInfo, QuantizeOp, QuantizeOutput},
    },
    padding::PaddingMode,
    tensor::{TensorTypeParam, WrappedTensor},
};
use anyhow::{Result, ensure};
use ark_ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    marker::PhantomData,
    sync::{Arc, Mutex},
};

pub mod proof;
mod witness;
pub use proof::{GatedDeltaScanCtx, GatedDeltaScanProof};
#[cfg(test)]
mod tests;

/// Multiplication by a frozen positive scale, with round-to-nearest ties towards +infinity.
/// Integer intermediates are checked in i128; no floating point is used at inference time.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct FixedScale {
    pub multiplier: i64,
    pub shift: u32,
}
impl FixedScale {
    pub fn new(scale: f64) -> Result<Self> {
        ensure!(
            scale.is_finite() && scale > 0.0,
            "Invalid fixed-point scale"
        );
        let shift = (50.0 - scale.log2().ceil()).clamp(0.0, 100.0) as u32;
        let multiplier = (scale * 2.0f64.powi(shift as i32)).round_ties_even();
        ensure!(
            multiplier >= 1.0 && multiplier < (1u64 << 52) as f64,
            "Unrepresentable fixed-point scale"
        );
        Ok(Self {
            multiplier: multiplier as i64,
            shift,
        })
    }
    pub fn apply(self, x: i128) -> Result<i64> {
        Ok(self.apply_with_remainder(x)?.0)
    }
    pub(crate) fn apply_with_remainder(self, x: i128) -> Result<(i64, u128)> {
        ensure!(
            self.shift <= 100 && self.multiplier > 0 && self.multiplier < (1i64 << 52),
            "Invalid recurrent fixed-point parameters"
        );
        let y = x
            .checked_mul(self.multiplier as i128)
            .ok_or_else(|| anyhow::anyhow!("Recurrent fixed-point product overflow"))?;
        let numerator = y
            .checked_add(if self.shift == 0 {
                0
            } else {
                1i128 << (self.shift - 1)
            })
            .ok_or_else(|| anyhow::anyhow!("Recurrent rounding overflow"))?;
        let rounded = numerator >> self.shift;
        Ok((
            i64::try_from(rounded)?,
            numerator.rem_euclid(1i128 << self.shift) as u128,
        ))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DeltaQuant {
    decay: FixedScale,
    memory: FixedScale,
    value: FixedScale,
    delta: FixedScale,
    update: FixedScale,
    output: FixedScale,
    input_domains: Vec<(Element, Element)>,
}
fn bounded(value: Element) -> Result<Element> {
    ensure!(
        i32::try_from(value).is_ok(),
        "DeltaNet intermediate {value} exceeds the proved signed 32-bit domain"
    );
    Ok(value)
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatedDeltaScan<N> {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    quant: Option<DeltaQuant>,
    #[serde(skip)]
    state: Arc<Mutex<Vec<N>>>,
    marker: PhantomData<N>,
}
impl<N: TensorTypeParam> GatedDeltaScan<N> {
    pub fn new(heads: usize, key_dim: usize, value_dim: usize) -> Self {
        Self {
            heads,
            key_dim,
            value_dim,
            quant: None,
            state: Default::default(),
            marker: PhantomData,
        }
    }
    pub fn reset_cache(&self) {
        self.state.lock().unwrap().clear();
    }
    pub fn quantization_summary(&self) -> Result<serde_json::Value> {
        ensure!(self.quant.is_some(), "DeltaNet has not been quantized");
        Ok(serde_json::json!({
            "heads": self.heads, "key_dim": self.key_dim,
            "value_dim": self.value_dim, "fixed_scales": self.quant,
            "rounding": "nearest-ties-towards-positive-infinity",
            "intermediate_signed_bits": 32,
            "initial_state": "zero-at-full-replay",
        }))
    }
    fn validate(&self, inputs: &[&WrappedTensor<N>]) -> Result<(usize, usize)> {
        ensure!(inputs.len() == 5, "DeltaNet expects q,k,v,alpha,beta");
        let shapes: Vec<Shape> = inputs.iter().map(|x| x.unpadded_shape().into()).collect();
        self.output_shapes(&shapes, PaddingMode::NoPadding)?;
        let shape = Shape::from(inputs[0].shape());
        ensure!(
            shape.dim(1) == self.heads && shape.dim(2) == self.key_dim,
            "DeltaNet requires power-of-two head dimensions"
        );
        Ok((shape.dim(0), shapes[0].dim(0)))
    }
}
impl<N: TensorTypeParam> OpInfo for GatedDeltaScan<N> {
    fn num_outputs(&self, n: usize) -> Result<usize> {
        ensure!(n == 5, "Expected five DeltaNet inputs");
        Ok(1)
    }
    fn output_shapes(&self, s: &[Shape], _: PaddingMode) -> Result<Vec<Shape>> {
        ensure!(s.len() == 5, "DeltaNet input count mismatch");
        ensure!(
            self.heads.is_power_of_two()
                && self.key_dim.is_power_of_two()
                && self.value_dim.is_power_of_two(),
            "DeltaNet dimensions must be powers of two"
        );
        ensure!(s[0].rank() == 3, "DeltaNet q must have shape [s,h,k]");
        let n = s[0].dim(0);
        ensure!(
            s[0] == Shape::from(vec![n, self.heads, self.key_dim])
                && s[1] == s[0]
                && s[2] == Shape::from(vec![n, self.heads, self.value_dim])
                && s[3] == Shape::from(vec![n, self.heads])
                && s[4] == s[3],
            "DeltaNet input shape mismatch: {s:?}"
        );
        Ok(vec![s[2].clone()])
    }
    fn describe(&self) -> String {
        "GatedDeltaScan".into()
    }
    fn is_provable(&self) -> bool {
        true
    }
}
impl Evaluate<f32> for GatedDeltaScan<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        let (rows, valid) = self.validate(inputs)?;
        let data: Vec<Vec<f32>> = inputs.iter().map(|x| x.get_data()).collect();
        let mut state = self.state.lock().unwrap();
        let width = self.key_dim * self.value_dim;
        if state.is_empty() {
            state.resize(self.heads * width, 0.0);
        }
        let results: Vec<_> = state
            .par_chunks_mut(width)
            .enumerate()
            .map(|(h, s)| {
                let mut output = vec![0.0; rows * self.value_dim];
                let mut max_state = 0.0f32;
                let mut max_memory = 0.0f32;
                let mut max_delta = 0.0f32;
                let mut delta = vec![0.0; self.value_dim];
                for t in 0..valid {
                    let q = &data[0][(t * self.heads + h) * self.key_dim..][..self.key_dim];
                    let k = &data[1][(t * self.heads + h) * self.key_dim..][..self.key_dim];
                    let v = &data[2][(t * self.heads + h) * self.value_dim..][..self.value_dim];
                    let alpha = data[3][t * self.heads + h];
                    let beta = data[4][t * self.heads + h];
                    for x in s.iter_mut() {
                        *x *= alpha;
                    }
                    delta.fill(0.0);
                    for i in 0..self.key_dim {
                        for j in 0..self.value_dim {
                            delta[j] += k[i] * s[i * self.value_dim + j];
                        }
                    }
                    for j in 0..self.value_dim {
                        // v and k^T S share this integer scale before subtraction.
                        max_memory = max_memory.max(delta[j].abs()).max(v[j].abs());
                        delta[j] = (v[j] - delta[j]) * beta;
                        max_delta = max_delta.max(delta[j].abs());
                    }
                    for i in 0..self.key_dim {
                        for j in 0..self.value_dim {
                            let x = &mut s[i * self.value_dim + j];
                            *x += k[i] * delta[j];
                            max_state = max_state.max(x.abs());
                            output[t * self.value_dim + j] += q[i] * *x;
                        }
                    }
                }
                (output, [max_state, max_memory, max_delta])
            })
            .collect();
        let mut output = vec![0.0; rows * self.heads * self.value_dim];
        let mut maxima = [0.0f32; 3];
        for (h, (values, max)) in results.into_iter().enumerate() {
            for t in 0..valid {
                output[(t * self.heads + h) * self.value_dim..][..self.value_dim]
                    .copy_from_slice(&values[t * self.value_dim..][..self.value_dim]);
            }
            for i in 0..3 {
                maxima[i] = maxima[i].max(max[i]);
            }
        }
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "Nonfinite DeltaNet output"
        );
        let tensor = Tensor::new_with_unpadded_shape(
            vec![rows, self.heads, self.value_dim].into(),
            vec![valid, self.heads, self.value_dim].into(),
            output,
        )?;
        let tracked = ["DELTA_STATE", "DELTA_MEMORY", "DELTA_DELTA"]
            .into_iter()
            .zip(maxima)
            .map(|(id, value)| {
                Ok((
                    id.to_string().into(),
                    WrappedTensor::try_from(&Tensor::new(vec![1].into(), vec![value.max(1e-9)])?)?,
                ))
            })
            .collect::<Result<HashMap<_, _>>>()?;
        Ok(LayerOut::from_tensor(WrappedTensor::try_from(&tensor)?)
            .with_data_to_be_tracked(tracked))
    }
}
impl Evaluate<Element> for GatedDeltaScan<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        let (rows, valid) = self.validate(inputs)?;
        let quant = self
            .quant
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DeltaNet has not been quantized"))?;
        let data: Vec<Vec<Element>> = inputs.iter().map(|x| x.get_data()).collect();
        ensure!(
            quant.input_domains.len() == 5,
            "DeltaNet input domain metadata missing"
        );
        for (values, &(minimum, maximum)) in data.iter().zip(&quant.input_domains) {
            ensure!(
                values.iter().all(|x| *x >= minimum && *x <= maximum),
                "DeltaNet input exceeds its proved integer domain"
            );
        }
        let mut state = self.state.lock().unwrap();
        let width = self.key_dim * self.value_dim;
        if state.is_empty() {
            state.resize(self.heads * width, 0);
        }
        let results: Vec<Vec<Element>> = state
            .par_chunks_mut(width)
            .enumerate()
            .map(|(h, s)| -> Result<_> {
                let mut output = vec![0; rows * self.value_dim];
                let mut delta = vec![0; self.value_dim];
                let mut memory = vec![0i128; self.value_dim];
                for t in 0..valid {
                    let q = &data[0][(t * self.heads + h) * self.key_dim..][..self.key_dim];
                    let k = &data[1][(t * self.heads + h) * self.key_dim..][..self.key_dim];
                    let v = &data[2][(t * self.heads + h) * self.value_dim..][..self.value_dim];
                    let alpha = data[3][t * self.heads + h];
                    let beta = data[4][t * self.heads + h];
                    for x in s.iter_mut() {
                        *x = bounded(quant.decay.apply(*x as i128 * alpha as i128)?)?;
                    }
                    memory.fill(0);
                    for i in 0..self.key_dim {
                        for j in 0..self.value_dim {
                            memory[j] += k[i] as i128 * s[i * self.value_dim + j] as i128;
                        }
                    }
                    for j in 0..self.value_dim {
                        let difference = bounded(quant.value.apply(v[j] as i128)?)? as i128
                            - bounded(quant.memory.apply(memory[j])?)? as i128;
                        delta[j] = bounded(quant.delta.apply(difference * beta as i128)?)?;
                    }
                    for i in 0..self.key_dim {
                        for j in 0..self.value_dim {
                            let x = &mut s[i * self.value_dim + j];
                            *x = bounded(
                                x.checked_add(quant.update.apply(k[i] as i128 * delta[j] as i128)?)
                                    .ok_or_else(|| anyhow::anyhow!("DeltaNet state overflow"))?,
                            )?;
                        }
                    }
                    memory.fill(0);
                    for i in 0..self.key_dim {
                        for j in 0..self.value_dim {
                            memory[j] += q[i] as i128 * s[i * self.value_dim + j] as i128;
                        }
                    }
                    for j in 0..self.value_dim {
                        output[t * self.value_dim + j] = bounded(quant.output.apply(memory[j])?)?;
                    }
                }
                Ok(output)
            })
            .collect::<Result<_>>()?;
        let mut output = vec![0; rows * self.heads * self.value_dim];
        for (h, values) in results.into_iter().enumerate() {
            for t in 0..valid {
                output[(t * self.heads + h) * self.value_dim..][..self.value_dim]
                    .copy_from_slice(&values[t * self.value_dim..][..self.value_dim]);
            }
        }
        let tensor = Tensor::new_with_unpadded_shape(
            vec![rows, self.heads, self.value_dim].into(),
            vec![valid, self.heads, self.value_dim].into(),
            output,
        )?;
        Ok(LayerOut::from_tensor(WrappedTensor::try_from(&tensor)?))
    }
}
impl QuantizeOp for GatedDeltaScan<f32> {
    type QuantizedOp = GatedDeltaScan<Element>;
    fn quantize_op<S: ScalingStrategy>(
        self,
        data: &S::AuxData,
        id: NodeId,
        input: &[ScalingFactor],
        _: &[Shape],
        output: &[ScalingFactor],
        _: &[Shape],
    ) -> Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            input.len() == 5 && output.len() == 1,
            "DeltaNet scaling arity mismatch"
        );
        // All factors are real units per integer unit, fixed for the complete sequence.
        let state =
            S::scaling_factor_for_intermediate_data(data, id, "DELTA_STATE".to_string().into())
                .scale() as f64;
        let memory =
            S::scaling_factor_for_intermediate_data(data, id, "DELTA_MEMORY".to_string().into())
                .scale() as f64;
        let delta =
            S::scaling_factor_for_intermediate_data(data, id, "DELTA_DELTA".to_string().into())
                .scale() as f64;
        let x: Vec<f64> = input.iter().map(|s| s.scale() as f64).collect();
        let quant = DeltaQuant {
            decay: FixedScale::new(x[3])?,
            memory: FixedScale::new(x[1] * state / memory)?,
            value: FixedScale::new(x[2] / memory)?,
            delta: FixedScale::new(memory * x[4] / delta)?,
            update: FixedScale::new(x[1] * delta / state)?,
            output: FixedScale::new(x[0] * state / output[0].scale() as f64)?,
            input_domains: input.iter().map(ScalingFactor::domain).collect(),
        };
        QuantizeOutput::new(
            GatedDeltaScan {
                heads: self.heads,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                quant: Some(quant),
                state: Default::default(),
                marker: PhantomData,
            },
            output.to_vec(),
        )
        .with_requant(crate::layers::requant::Requant::from_multiplier(
            1.0, 32, output[0],
        ))
    }
}
impl PadOp for GatedDeltaScan<Element> {}
