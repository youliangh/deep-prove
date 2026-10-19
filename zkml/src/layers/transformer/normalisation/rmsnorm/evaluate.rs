//! Module containing code for evaluating [`RMSNorm`] layers using both floating point and quantised inference.
use super::*;

use crate::layers::{provable::ProvingData, requant::FIXED_POINT_SCALE};
use burn::tensor::TensorData;
use tenstore::GenStore;

impl RMSNorm<f32> {
    /// Evaluates the [`RMSNorm`] layer over [`f32`] tensors.
    pub fn evaluate_float_internal(&self, inputs: &[&WrappedTensor<f32>]) -> Result<LayerOut<f32>> {
        let input = inputs[0];
        // Check that the final dimension of the input matches the expected dim_size
        let final_dim_size = input.dim(-1)?;
        ensure!(
            final_dim_size == self.normalisation_dim_size(),
            "RMSNorm input must have final dimension of {}: found {:?}",
            self.normalisation_dim_size(),
            input.shape(),
        );

        let alpha_opt = self
            .alpha
            .as_ref()
            .map(|keyed_alpha| keyed_alpha.wrapped_tensor())
            .transpose()?
            .map(|t| t.clone());

        let output = WrappedTensor::rms_norm_forward(
            input.clone(),
            self.normalisation_dim_size(),
            self.eps as f64,
            alpha_opt,
        )?;

        Ok(LayerOut::from_tensor(output))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormHandle {
    pub(crate) normalisation: TensorHandle<Element>,
    pub(crate) lookup_verifier: RMSNormLookupVerifier,
}

impl RMSNormHandle {
    pub(crate) fn new(
        storage_key: StorageKey<Vec<Element>>,
        rmsnorm_data: RMSNormProvingData,
        store: tenstore::GenStore,
    ) -> Self {
        let key = StorageKey::new(format!("{}-normalisation", storage_key.id()));
        let normalisation =
            TensorHandle::from_wrapped_tensor(key, store, rmsnorm_data.normalisation);
        Self {
            normalisation,
            lookup_verifier: rmsnorm_data.lookup_verifier,
        }
    }

    pub(crate) fn to_proving_data(&self) -> Result<RMSNormProvingData> {
        let normalisation_native = self.normalisation.tensor()?;
        let normalisation = WrappedTensor::try_from(normalisation_native.as_ref())?;
        Ok(RMSNormProvingData {
            normalisation,
            lookup_verifier: self.lookup_verifier,
        })
    }

    pub(crate) fn isolate(&self) -> RMSNormHandle {
        Self {
            normalisation: self.normalisation.isolate(),
            lookup_verifier: self.lookup_verifier,
        }
    }

    pub(crate) fn attach_store(&mut self, store: GenStore) {
        self.normalisation.attach_store(store);
    }
}

impl RMSNorm<Element> {
    /// Evaluates the [`RMSNorm`] layer over quantized tensors.
    pub fn evaluate_quantised_internal(
        &self,
        inputs: &[&WrappedTensor<Element>],
    ) -> Result<LayerOut<Element>> {
        ensure!(
            inputs.len() == 1,
            "Exactly one input must be provided to RMSnorm. got {}",
            inputs.len(),
        );
        let input = inputs[0].clone();
        let shape = input.shape();
        // Ensure we have the quantisation scaling factors
        let (input_scaling_factor, normalisation_scaling_factor) = self
            .get_quantisation_scaling_factors()
            .context("Quantisation scaling factors not found for RMSNorm evaluation")?;

        // Appendix C normalizes the nonzero row to a fixed squared norm; it
        // does not include the Hugging Face epsilon. Float evaluation above
        // retains that epsilon, allowing this numerical difference to be measured.
        let (normalised_sum_sq, error_bound) = bounds::norm_parameters(
            self.normalisation_dim_size(),
            normalisation_scaling_factor.scale(),
        )?;
        ensure!(
            input.unpadded_shape()[shape.rank() - 1] == self.normalisation_dim_size(),
            "RMSNorm dimension mismatch"
        );
        let row_width = input.dim(-1)?;
        let input_data = input.get_data();
        let multiplier_shape = shape
            .iter()
            .copied()
            .take(shape.rank() - 1)
            .chain([1])
            .collect::<Vec<_>>();
        let mut cache = self.cache.lock().unwrap();
        let mut fract_data = Vec::new();
        let mut shift_data = Vec::new();
        // Accumulate exactly in u128, then round once to f64 for the witness
        // proposal. This avoids both i64 overflow and a row-length-dependent
        // summation error in the fixed-point rounding envelope.
        for row in input_data.chunks(row_width) {
            let sumsq = row.iter().try_fold(0u128, |acc, &x| {
                let x = i128::from(x);
                acc.checked_add((x * x) as u128)
                    .ok_or(anyhow!("RMSNorm squared input norm exceeds u128"))
            })?;
            if sumsq == 0 {
                // Numerical zero behavior remains defined. Proving a genuine
                // zero row is rejected explicitly by the paper-mode norm check.
                fract_data.push(0);
                shift_data.push(FIXED_POINT_SCALE as Element);
                cache.update(FIXED_POINT_SCALE as Element);
                continue;
            }
            let multiplier = (normalised_sum_sq as f64 / sumsq as f64).sqrt();
            let exponent = multiplier.log2().floor() as i32;
            let mantissa = multiplier / 2f64.powi(exponent);
            let shift = FIXED_POINT_SCALE as i32 - exponent;
            ensure!(
                (1..63).contains(&shift),
                "RMSNorm normalising shift is outside the supported range: {shift}"
            );
            let fixed =
                (mantissa * (1u64 << FIXED_POINT_SCALE) as f64).round_ties_even() as Element;
            cache.update(Element::from(shift));
            fract_data.push(fixed);
            shift_data.push(Element::from(shift));
        }
        let max_shift = cache.get_shift();
        for (row, (multiplier, shift)) in input_data
            .chunks(row_width)
            .zip(fract_data.iter_mut().zip(&shift_data))
        {
            let widened = i128::from(*multiplier) << (max_shift - *shift);
            ensure!(
                widened < (1i128 << crate::lookup::operation::variant::rms::RMS_MULTIPLIER_BITS),
                "RMSNorm multiplier exceeds its 48-bit proof range"
            );
            *multiplier = Element::try_from(widened)?;
            let rounding = 1i128 << (max_shift - 1);
            ensure!(
                row.iter()
                    .all(|&x| i64::try_from(i128::from(x) * widened + rounding).is_ok()),
                "RMSNorm fixed-point product exceeds i64; recalibrate the input scale"
            );
        }
        drop(cache);

        let mult_tensor_data = TensorData::new(fract_data, multiplier_shape);
        let mult = WrappedTensor::try_from(mult_tensor_data)?;

        let proving_data = RMSNormProvingData::new(
            mult,
            max_shift.unsigned_abs() as usize,
            normalised_sum_sq,
            error_bound,
            input_scaling_factor.bit_size() + 1,
            self.alpha.is_some(),
        );

        let table = Table::new_normalisation(normalisation_scaling_factor.bit_size() + 2);
        let tmp_output = proving_data.apply(input, &table)?;

        // Get the alpha
        let output = match self.alpha.as_ref() {
            Some(keyed_alpha) => {
                let alpha_tensor = keyed_alpha.wrapped_tensor()?.clone();
                let expanded_alpha = match shape.rank() {
                    1 => alpha_tensor,
                    2 => alpha_tensor.unsqueeze_dim_2(),
                    3 => alpha_tensor.unsqueeze_dim_3(),
                    4 => alpha_tensor.unsqueeze_dim_4(),
                    _ => {
                        anyhow::bail!(
                            "Unsupported input rank for RMSNorm quantised evaluation: {}",
                            shape.rank()
                        );
                    }
                };
                tmp_output.mul(expanded_alpha)?
            }
            None => tmp_output,
        };

        Ok(LayerOut::from_tensor(output).with_proving_data(ProvingData::RMSNorm(proving_data)))
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use proptest::prelude::*;
    use std::{
        fmt::{Debug, Display},
        ops::Range,
    };

    #[test]
    fn llama_width_and_zero_rows_are_numerically_defined() {
        let dim = 2048;
        let mut layer = RMSNorm::<Element>::new(None, 1e-5, Some(dim)).unwrap();
        let scale = bounds::normalisation_scale(dim, 12);
        layer.input_scaling_factor = Some(scale);
        layer.normalisation_scaling_factor = Some(scale);
        let mut data = vec![1; dim];
        data.extend((0..dim).map(|i| i as i64 - 1024));
        data.extend(std::iter::repeat_n(0, dim));
        let input = Tensor::new(Shape::new(vec![3, dim]), data).unwrap();
        let wrapped = WrappedTensor::try_from(&input).unwrap();
        let output = layer.evaluate_quantised_internal(&[&wrapped]).unwrap();
        let values = output.outputs()[0].get_data();
        let (target, bound) = bounds::norm_parameters(dim, scale.scale()).unwrap();
        for row in values[..2 * dim].chunks(dim) {
            let squared: Element = row.iter().map(|x| x * x).sum();
            assert!((squared - target).abs() <= bound);
        }
        assert!(values[2 * dim..].iter().all(|&v| v == 0));
    }

    #[test]
    fn test_quantised() {
        let input = Tensor::<f32>::random(&Shape::new(vec![60, 640]));
        let input_shape = input.shape();
        let layer = RMSNorm::new(None, 1e-5, Some(input_shape.dim(-1))).unwrap();

        let wrapped_input = WrappedTensor::try_from(&input).unwrap();
        let LayerOut { outputs, .. } = layer.evaluate_float_internal(&[&wrapped_input]).unwrap();

        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let output_native = outputs[0].to_native();
        let output_scaling = ScalingFactor::from_tensor(&output_native, None);

        let normalisation_scaling_factor =
            bounds::normalisation_scale(layer.normalisation_dim_size(), *quantization::BIT_LEN);

        let QuantizeOutput { quantized_op, .. } = layer
            .quantise(input_scaling, normalisation_scaling_factor, output_scaling)
            .unwrap();

        let quantised_input = input.quantize(&input_scaling);
        let wrapped_quantised_input = WrappedTensor::try_from(&quantised_input).unwrap();
        let quantised_layer_out = quantized_op
            .evaluate_quantised_internal(&[&wrapped_quantised_input])
            .unwrap();

        let quantised_output = quantised_layer_out.outputs();
        let RMSNormProvingData {
            normalisation: _,
            lookup_verifier,
        } = quantised_layer_out.try_rmsnorm_data().unwrap();

        let dim_size = quantized_op.normalisation_dim_size();
        let quantized_dim_size = lookup_verifier.normalisation_sum_sq;
        let bound_dim_size = lookup_verifier.error_bound;

        for output_chunk in quantised_output[0].get_data().chunks(dim_size) {
            let square_sum = output_chunk.iter().map(|x| x * x).sum::<Element>();
            let diff = (quantized_dim_size - square_sum).abs();
            assert!(
                diff <= bound_dim_size || square_sum == 0,
                "Norm not within range bound, got square sum {square_sum} expected to be within {bound_dim_size} of {quantized_dim_size} or equal 0, was actually {diff}, out by {}, chunk: {:?}",
                diff - bound_dim_size,
                output_chunk
            );
        }
    }

    #[derive(Clone)]
    struct Input {
        input: Tensor<f32>,
    }

    impl Debug for Input {
        fn fmt(
            &self,
            fmt: &mut std::fmt::Formatter<'_>,
        ) -> std::result::Result<(), std::fmt::Error> {
            let non_zero = self.input.data().iter().filter(|&&x| x != 0.0).count();
            fmt.debug_struct("Input")
                .field(
                    "input",
                    &format_args!(
                        "Input{{shape: {:?}, non_zero: {}}}",
                        self.input.shape(),
                        non_zero
                    ),
                )
                .finish()
        }
    }

    impl Display for Input {
        fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(fmt, "Input{{input: {:?}}}", self.input.shape(),)
        }
    }

    fn rmsnorm_input(
        normalisation_dim: Range<usize>,
        other_dims: Range<usize>,
    ) -> impl Strategy<Value = Input> {
        (
            normalisation_dim,
            prop::collection::vec(other_dims, 1usize..=3),
        )
            .prop_flat_map(move |(normalisation_dim, mut shape)| {
                shape.push(normalisation_dim);
                let input = Tensor::any(Shape::new(shape));
                input.prop_map(|input| Input { input })
            })
    }

    /// Regression test for the epsilon-bias bug (see comment in evaluate_quantised_internal).
    /// Creates input where Row 0 has ~100x smaller values than other rows, which makes
    /// epsilon significant for Row 0's mean(input²). Without the norm_with_scale clamp,
    /// Row 0's output sumsq falls below the normalisation check range and overflows
    /// ShiftCheckTable, causing "Final numerator was non-zero" during proving.
    #[test]
    fn test_quantised_heterogeneous_magnitudes() {
        let dim = 640;
        let num_rows = 5;

        // Create input where Row 0 has ~10x smaller values than other rows
        let mut data = Vec::with_capacity(num_rows * dim);
        for row in 0..num_rows {
            let scale = if row == 0 { 0.01 } else { 1.0 };
            for j in 0..dim {
                // Deterministic pseudo-random-ish values
                let val = ((row * dim + j) as f32 * 0.618034 % 2.0 - 1.0) * scale;
                data.push(val);
            }
        }

        let input = Tensor::new(Shape::new(vec![num_rows, dim]), data).unwrap();
        let input_shape = input.shape();
        let layer = RMSNorm::new(None, 1e-5, Some(input_shape.dim(-1))).unwrap();

        let wrapped_input = WrappedTensor::try_from(&input).unwrap();
        let LayerOut { outputs, .. } = layer.evaluate_float_internal(&[&wrapped_input]).unwrap();

        let input_scaling = ScalingFactor::from_tensor(&input, None);
        let output_native = outputs[0].to_native();
        let output_scaling = ScalingFactor::from_tensor(&output_native, None);

        let normalisation_scaling_factor =
            bounds::normalisation_scale(layer.normalisation_dim_size(), *quantization::BIT_LEN);

        let QuantizeOutput { quantized_op, .. } = layer
            .quantise(input_scaling, normalisation_scaling_factor, output_scaling)
            .unwrap();

        let quantised_input = input.quantize(&input_scaling);
        let wrapped_quantised_input = WrappedTensor::try_from(&quantised_input).unwrap();

        let quantised_layer_out = quantized_op
            .evaluate_quantised_internal(&[&wrapped_quantised_input])
            .unwrap();

        let quantised_output = quantised_layer_out.outputs();
        let RMSNormProvingData {
            normalisation: _,
            lookup_verifier,
        } = quantised_layer_out.try_rmsnorm_data().unwrap();

        let normalised_sumsq = lookup_verifier.normalisation_sum_sq;
        let error_bound = lookup_verifier.error_bound;

        let dim_size = quantized_op.normalisation_dim_size();
        for (row, output_chunk) in quantised_output[0].get_data().chunks(dim_size).enumerate() {
            let square_sum = output_chunk.iter().map(|x| x * x).sum::<Element>();
            let diff = (normalised_sumsq - square_sum).abs();
            assert!(
                diff <= error_bound || square_sum == 0,
                "Row {}: Norm not within range bound, got square sum {square_sum} expected to be within {error_bound} of {normalised_sumsq} or equal 0, was actually {diff}, out by {}",
                row,
                diff - error_bound,
            );
        }
    }

    proptest! {
        #[test]
        fn proptest_rmsnorm_evaluate_quantised(input in rmsnorm_input(10usize..768, 2usize..64)) {
            let Input {
                input,
            } = input;

            let input_shape = input.shape();
            let layer = RMSNorm::new(None, 1e-5, Some(input_shape.dim(-1))).unwrap();

            let wrapped_input = WrappedTensor::try_from(&input).unwrap();
            let LayerOut { outputs, .. } = layer.evaluate_float_internal(&[&wrapped_input]).unwrap();


            let input_scaling = ScalingFactor::from_tensor(&input, None);
            let output_native = outputs[0].to_native();
            let output_scaling = ScalingFactor::from_tensor(&output_native, None);

            let normalisation_scaling_factor = bounds::normalisation_scale(
            layer.normalisation_dim_size(), *quantization::BIT_LEN,
        );

            let QuantizeOutput { quantized_op, .. } = layer.quantise(input_scaling, normalisation_scaling_factor, output_scaling).unwrap();

            let quantised_input = input.quantize(&input_scaling);
            let wrapped_quantised_input = WrappedTensor::try_from(&quantised_input).unwrap();

            let quantised_layer_out = quantized_op.evaluate_quantised_internal(&[&wrapped_quantised_input]).unwrap();


            let quantised_output = quantised_layer_out.outputs();
            let RMSNormProvingData {
                normalisation: _,
                lookup_verifier,
            } = quantised_layer_out.try_rmsnorm_data().unwrap();

            let normalised_sumsq = lookup_verifier.normalisation_sum_sq;
            let error_bound = lookup_verifier.error_bound;

            let dim_size = quantized_op.normalisation_dim_size();
            let scale_squared = normalisation_scaling_factor.scale().powi(2);
            for output_chunk in quantised_output[0].get_data().chunks(dim_size) {
                let square_sum = output_chunk.iter().map(|x| x * x).sum::<Element>();
                let diff = (normalised_sumsq - square_sum).abs();
                prop_assert!(diff <= error_bound || square_sum == 0,
                    "Norm not within range bound, got square sum {square_sum} expected to be within {error_bound} (float: {}) of {normalised_sumsq} or equal 0, was actually {diff}, out by {}, (float: {}, scale factor: {}), chunk: {:?} (float: {:?}, norm max: {}, norm min: {})",
                 error_bound as f32 * scale_squared, diff - error_bound, (diff - error_bound) as f32 * scale_squared, scale_squared, output_chunk, output_chunk.iter().map(|x| *x as f32 * normalisation_scaling_factor.scale()).collect::<Vec<f32>>(), normalisation_scaling_factor.max(), normalisation_scaling_factor.min());
            }
        }
    }
}
