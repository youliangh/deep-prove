//! Module containing code for quantising [`RMSNorm`] layers.
use crate::layers::requant::Requant;

use super::*;

impl RMSNorm<f32> {
    /// Quantises the RMSNorm layer.
    pub fn quantise(
        self,
        input_scaling_factor: ScalingFactor,
        normalisation_scaling_factor: ScalingFactor,
        output_scaling: ScalingFactor,
    ) -> anyhow::Result<QuantizeOutput<RMSNorm<Element>>> {
        // Now we construct the `model_scaling` from `self.gamma`
        let (alpha_scaling, alpha_bit_size, quantised_alpha) = match self.alpha {
            Some(alpha) => {
                let alpha_max = alpha.max_abs()?;
                let scaling = ScalingFactor::from_absolute_max(alpha_max, None);
                let bit_size = scaling.bit_size();
                (scaling.scale(), bit_size, Some(alpha.quantize(&scaling)))
            }
            None => (1.0f32, 0, None),
        };

        if quantised_alpha.is_some() {
            let intermediate_bit_size =
                normalisation_scaling_factor.bit_size() + alpha_bit_size + 2;
            let multiplier =
                alpha_scaling * normalisation_scaling_factor.scale() / output_scaling.scale();
            let requant =
                Requant::from_multiplier(multiplier, intermediate_bit_size, output_scaling);

            let quantised_rmsnorm = RMSNorm {
                alpha: quantised_alpha,
                eps: self.eps,
                normalisation_dim_size: self.normalisation_dim_size,
                input_scaling_factor: Some(input_scaling_factor),
                normalisation_scaling_factor: Some(normalisation_scaling_factor),
                cache: Arc::new(Mutex::new(NormalisationCache::new())),
            };

            QuantizeOutput::new(quantised_rmsnorm, vec![output_scaling]).with_requant(requant)
        } else {
            let quantised_rmsnorm = RMSNorm {
                alpha: quantised_alpha,
                eps: self.eps,
                normalisation_dim_size: self.normalisation_dim_size,
                input_scaling_factor: Some(input_scaling_factor),
                normalisation_scaling_factor: Some(normalisation_scaling_factor),
                cache: Arc::new(Mutex::new(NormalisationCache::new())),
            };
            // If there is no alpha then the result of the normalisation (with the normalisation_scaling_factor) is passed directly to the output.
            Ok(QuantizeOutput::new(
                quantised_rmsnorm,
                vec![normalisation_scaling_factor],
            ))
        }
    }
}
