//! Epsilon-preserving Qwen RMS and L2 normalisation.
//!
//! The integer operator returns the unrounded product `x * multiplier * gamma`.
//! Its automatically inserted Requant layer performs the final signed rounding
//! and saturation. All scales, epsilon and reciprocal precision are static.

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    Element, NextPowerOfTwo, ScalingFactor, ScalingStrategy, Shape, Tensor,
    graph::NodeId,
    layers::{
        provable::{Evaluate, LayerOut, OpInfo, PadOp, QuantizeOp, QuantizeOutput, Splittable},
        requant::{FIXED_POINT_SCALE, Requant},
    },
    padding::PaddingMode,
    quantization::Quantize,
    tensor::{TensorHandle, TensorTypeParam, WrappedTensor},
};

pub mod integer;
pub mod proof;

pub use proof::{QwenNormCtx, QwenNormProof};

pub const QWEN_NORM_LAYER: &str = "QNRM";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum NormKind {
    Rms,
    L2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "N: Serialize", deserialize = "N: DeserializeOwned"))]
pub struct QwenNorm<N: TensorTypeParam> {
    pub alpha: Option<TensorHandle<N>>,
    pub eps: f32,
    pub dimension: usize,
    pub kind: NormKind,
    pub quant: Option<integer::NormQuant>,
}

impl<N: TensorTypeParam> QwenNorm<N> {
    pub fn new(
        alpha: Option<TensorHandle<N>>,
        eps: f32,
        dimension: usize,
        kind: NormKind,
    ) -> Result<Self> {
        ensure!(
            dimension > 0 && dimension.is_power_of_two(),
            "Qwen normalisation dimension must be a positive power of two"
        );
        ensure!(
            eps.is_finite() && eps > 0.0,
            "Qwen normalisation requires positive finite epsilon"
        );
        if let Some(weight) = &alpha {
            ensure!(
                weight.shape().as_ref().as_slice() == [dimension],
                "Qwen normalisation weight must have shape [{dimension}]"
            );
        }
        let alpha = alpha
            .map(|weight| weight.wrapped_tensor_variant())
            .transpose()?;
        Ok(Self {
            alpha,
            eps,
            dimension,
            kind,
            quant: None,
        })
    }

    fn validate_inputs(&self, inputs: &[&WrappedTensor<N>]) -> Result<()> {
        ensure!(inputs.len() == 1, "Qwen normalisation expects one input");
        ensure!(
            inputs[0].dim(-1)? == self.dimension,
            "Qwen normalisation last dimension mismatch"
        );
        Ok(())
    }
}

impl Evaluate<f32> for QwenNorm<f32> {
    fn evaluate(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        self.validate_inputs(inputs)?;
        let input = inputs[0];
        let weights = self
            .alpha
            .as_ref()
            .map(|a| -> Result<_> { Ok(a.wrapped_tensor()?.get_data()) })
            .transpose()?;
        let divisor = if self.kind == NormKind::Rms {
            self.dimension as f32
        } else {
            1.0
        };
        let values = input.get_data();
        let mut output = Vec::with_capacity(values.len());
        for row in values.chunks(self.dimension) {
            let squared: f32 = row.iter().map(|x| x * x).sum();
            let inv = (squared / divisor + self.eps).sqrt().recip();
            output.extend(
                row.iter()
                    .enumerate()
                    .map(|(j, x)| x * inv * weights.as_ref().map_or(1.0, |w| w[j])),
            );
        }
        let tensor = Tensor::new_with_unpadded_shape(
            input.shape().into(),
            input.unpadded_shape().clone().into(),
            output,
        )?;
        Ok(LayerOut::from_tensor(WrappedTensor::try_from(&tensor)?))
    }
}

impl Evaluate<Element> for QwenNorm<Element> {
    fn evaluate(&self, inputs: &[&WrappedTensor<Element>]) -> Result<LayerOut<Element>> {
        self.validate_inputs(inputs)?;
        let input = inputs[0];
        let quant = self
            .quant
            .as_ref()
            .context("Qwen normalisation is not quantised")?;
        let weights = self
            .alpha
            .as_ref()
            .map(|a| -> Result<_> { Ok(a.wrapped_tensor()?.get_data()) })
            .transpose()?;
        let values = input.get_data();
        let mut output = Vec::with_capacity(values.len());
        for row in values.chunks(self.dimension) {
            let certificate = quant.certify_row(row)?;
            for (j, &x) in row.iter().enumerate() {
                let raw = i128::from(x)
                    .checked_mul(i128::from(certificate.multiplier))
                    .and_then(|v| v.checked_mul(i128::from(weights.as_ref().map_or(1, |w| w[j]))))
                    .context("Qwen normalisation product exceeds i128")?;
                output.push(
                    Element::try_from(raw).context("Qwen normalisation product exceeds i64")?,
                );
            }
        }
        let tensor = Tensor::new_with_unpadded_shape(
            input.shape().into(),
            input.unpadded_shape().clone().into(),
            output,
        )?;
        Ok(LayerOut::from_tensor(WrappedTensor::try_from(&tensor)?))
    }
}

impl<N: TensorTypeParam> OpInfo for QwenNorm<N> {
    fn output_shapes(&self, inputs: &[Shape], padding: PaddingMode) -> Result<Vec<Shape>> {
        ensure!(
            inputs.len() == 1 && inputs[0].dim(-1) == self.dimension,
            "Qwen norm shape mismatch"
        );
        Ok(vec![match padding {
            PaddingMode::NoPadding => inputs[0].clone(),
            PaddingMode::Padding => inputs[0].next_power_of_two(),
        }])
    }
    fn num_outputs(&self, inputs: usize) -> Result<usize> {
        ensure!(inputs == 1, "Qwen norm expects one input");
        Ok(1)
    }
    fn describe(&self) -> String {
        format!(
            "QwenNorm({:?}, {}, eps={})",
            self.kind, self.dimension, self.eps
        )
    }
    fn is_provable(&self) -> bool {
        true
    }
}

impl PadOp for QwenNorm<Element> {}
impl<N: TensorTypeParam> Splittable for QwenNorm<N> {}

impl QuantizeOp for QwenNorm<f32> {
    type QuantizedOp = QwenNorm<Element>;
    fn quantize_op<S: ScalingStrategy>(
        self,
        _data: &S::AuxData,
        _node_id: NodeId,
        inputs: &[ScalingFactor],
        _input_shapes: &[Shape],
        outputs: &[ScalingFactor],
        _output_shapes: &[Shape],
    ) -> Result<QuantizeOutput<Self::QuantizedOp>> {
        ensure!(
            inputs.len() == 1 && outputs.len() == 1,
            "Qwen norm requires one input/output scale"
        );
        let factor = if self.kind == NormKind::Rms {
            self.dimension as f64
        } else {
            1.0
        };
        let normal_scale = ScalingFactor::from_absolute_max(factor.sqrt() as f32, None);
        let (alpha, alpha_scale, alpha_bound) = match self.alpha {
            Some(alpha) => {
                let maximum = alpha.max_abs()?;
                ensure!(maximum.is_finite(), "Qwen norm weight must be finite");
                let sf = ScalingFactor::from_absolute_max(maximum.max(1e-12), None);
                let bound = sf
                    .domain()
                    .0
                    .unsigned_abs()
                    .max(sf.domain().1.unsigned_abs());
                (Some(alpha.quantize(&sf)), sf.scale(), bound)
            }
            None => (None, 1.0, 1),
        };
        // Leave room for Requant's fixed-point multiplier. Precision is fixed
        // for the layer, never chosen from the current token/prefix.
        let mut precision = 24u32;
        let input_bound = inputs[0]
            .domain()
            .0
            .unsigned_abs()
            .max(inputs[0].domain().1.unsigned_abs());
        let intermediate_bits = loop {
            let normal_bound = (factor.sqrt() / f64::from(normal_scale.scale())
                * 2f64.powi(precision as i32)
                / (1.0 - 2f64.powi(-(integer::DENOMINATOR_FRACTION_BITS as i32 / 2)))
                + input_bound as f64 / 2.0
                + 2.0)
                .ceil();
            let product_bound = normal_bound * alpha_bound as f64;
            let bits = product_bound.log2().ceil() as usize + 1;
            if bits + FIXED_POINT_SCALE <= 63 {
                break bits;
            }
            ensure!(
                precision > 8,
                "Qwen normalisation precision exceeds supported integer range"
            );
            precision -= 1;
        };
        let quant = integer::NormQuant::new(
            self.dimension,
            self.kind,
            self.eps,
            inputs[0],
            normal_scale,
            precision,
        )?;
        let requant_multiplier =
            normal_scale.scale() * alpha_scale / (2f32.powi(precision as i32) * outputs[0].scale());
        ensure!(
            requant_multiplier.is_finite() && requant_multiplier > 0.0,
            "Invalid Qwen norm requant multiplier"
        );
        let requant = Requant::from_multiplier(requant_multiplier, intermediate_bits, outputs[0]);
        QuantizeOutput::new(
            QwenNorm {
                alpha,
                eps: self.eps,
                dimension: self.dimension,
                kind: self.kind,
                quant: Some(quant),
            },
            outputs.to_vec(),
        )
        .with_requant(requant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        layers::Layer,
        model::{
            Model,
            test::{prove_model_with, quantize_model},
        },
        padding::pad_model,
        tensor::KeyedTensor,
        testing::Pcs,
    };
    use ark_bn254::Fr;
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use tenstore::GenStore;

    fn model_and_input(kind: NormKind, rows: usize, weighted: bool) -> (Model<f32>, Tensor<f32>) {
        let shape = Shape::new(vec![rows, 4]);
        let mut values = Vec::new();
        for row in 0..rows {
            values.extend(if row == 1 {
                vec![0.0; 4]
            } else {
                vec![0.1, -0.2, 0.3, -0.4]
            });
        }
        let input = Tensor::new(shape.clone(), values).unwrap();
        let weight = weighted.then(|| {
            KeyedTensor::new(
                "qwen-norm-gamma",
                Tensor::new(vec![4].into(), vec![1.0, 0.5, -0.25, 1.5]).unwrap(),
            )
            .into()
        });
        let layer = QwenNorm::new(weight, 1e-6, 4, kind).unwrap();
        let mut model = Model::new_from_input_shapes(vec![shape]);
        model
            .add_consecutive_layer(Layer::QwenNorm(layer), None)
            .unwrap();
        model.automatic_output_labelling().unwrap();
        (model, input)
    }

    #[test]
    fn epsilon_norm_float_zero_and_effective_gamma() {
        let weights = KeyedTensor::new(
            "gamma",
            Tensor::new(vec![4].into(), vec![1.0, 2.0, -1.0, 0.5]).unwrap(),
        );
        let layer = QwenNorm::new(Some(weights.into()), 1e-6, 4, NormKind::Rms).unwrap();
        let input = Tensor::new(
            vec![2, 4].into(),
            vec![0.0, 0.0, 0.0, 0.0, 1e-4, -1e-4, 1e-4, -1e-4],
        )
        .unwrap();
        let output = layer.evaluate(&[&input.as_wrapped()]).unwrap().outputs()[0].get_data();
        assert_eq!(&output[..4], &[0.0; 4]);
        let expected = 1e-4f32 / (1e-8f32 + 1e-6f32).sqrt();
        for (actual, expected) in
            output[4..]
                .iter()
                .zip([expected, -2.0 * expected, -expected, -0.5 * expected])
        {
            assert!((actual - expected).abs() < 1e-7);
        }
    }

    #[test]
    fn epsilon_norm_complete_proof_for_small_and_padded_rows() {
        for (kind, rows, weighted) in [
            (NormKind::Rms, 1, false),
            (NormKind::L2, 3, false),
            (NormKind::Rms, 4, true),
        ] {
            let (model, input) = model_and_input(kind, rows, weighted);
            prove_model_with(model, vec![input], &mut GenStore::default()).unwrap();
        }
    }

    #[test]
    fn epsilon_norm_proof_rejects_modified_claim_and_witness_evaluation() {
        type P<'a, 'b> = crate::Prover<'a, 'b, Fr, Blake3Transcript, Pcs>;
        let (model, input) = model_and_input(NormKind::Rms, 4, true);
        let mut store = GenStore::default();
        let (model, inputs) =
            quantize_model(model, vec![input.clone()], Some(vec![input]), &mut store).unwrap();
        let model = pad_model(model).unwrap();
        let inputs = model.prepare_inputs(inputs).unwrap();
        let trace = model.run(inputs, &mut store).unwrap();
        let (prover, verifier) = model.generate_contexts::<Fr, Pcs>().unwrap();
        let (proof, io) = P::prove(&prover, trace, &model).unwrap();
        crate::verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), io.clone()).unwrap();

        let mut bad_io = io.clone();
        let output = &bad_io.output[0];
        let mut data = output.data().to_vec();
        data[0] = crate::SerializableField(*data[0] + Fr::from(1u64));
        bad_io.output[0] = Tensor::new_with_unpadded_shape(
            output.shape().clone(),
            output.unpadded_shape().clone(),
            data,
        )
        .unwrap();
        // dp-crypto currently asserts on inconsistent sumcheck rounds. Either
        // an error or that assertion is a rejection; neither may accept.
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), bad_io)
        }));
        assert!(rejected.is_err() || rejected.unwrap().is_err());

        let mut encoded = serde_json::to_value(&proof).unwrap();
        let steps = encoded["chunk_proofs"][0]["steps"].as_object_mut().unwrap();
        let norm = steps
            .values_mut()
            .find_map(|step| step.get_mut("QwenNorm"))
            .unwrap();
        // Replace the claimed denominator evaluation by the input evaluation.
        // Both use identical field serialization; this retains a well-formed
        // proof encoding while violating the committed square-sum relation.
        norm["squared_sum_eval"] = norm["squared_input_eval"].clone();
        let corrupted = serde_json::from_value(encoded).unwrap();
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::verify::<_, Blake3Transcript, _>(&verifier, corrupted, io)
        }));
        assert!(rejected.is_err() || rejected.unwrap().is_err());
    }
}
