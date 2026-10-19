//! Internal code for generating the [`RMSNorm`] looup witness.

use dp_crypto::IntoMLE;

use crate::{lookup::context::LookupWitnessGen, model::Step, to_field};

use super::*;

impl RMSNorm<Element> {
    pub(crate) fn lookup_witness<F: PrimeField>(
        &self,
        id: NodeId,
        step: &Step<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        let rmsnorm_data = step
            .node_outputs
            .try_rmsnorm_data()
            .ok_or(anyhow!(
                "Could not get RMSNorm proving data for lookup witness generation"
            ))?
            .to_proving_data()?;

        let input = step.input_tensor_at(0)?;
        let input_shape = input.shape().clone();
        let (_, normalisation_scaling_factor) = self.get_quantisation_scaling_factors().ok_or(
            anyhow!("Quantisation scaling factors not found for RMSNorm lookup witness generation"),
        )?;
        let table = Table::new_normalisation(normalisation_scaling_factor.bit_size() + 2);

        let wrapped_input = WrappedTensor::try_from(input.as_ref())?;
        let lookup_output = rmsnorm_data.apply(wrapped_input.clone(), &table)?;
        let scaled = wrapped_input.mul(rmsnorm_data.normalisation.clone())?;
        let scaled_input = Tensor::try_from(&scaled)?;
        let output = Tensor::try_from(&lookup_output)?;

        let rank = input_shape.rank();
        let number_of_chunks = input_shape[..rank.saturating_sub(2)]
            .iter()
            .product::<usize>();

        let lookup_witness = rmsnorm_data.generate_witness(scaled_input, &table)?;

        let mut element_counts = lookup_witness.get_counts(&table);
        let variant = rmsnorm_data.variant();

        let mut norm_counts =
            variant.compute_normalisation_witness_counts(number_of_chunks, output.as_ref())?;

        // Merge in the normalisation counts
        let norm_map = norm_counts.remove(0);
        for (key, val) in norm_map {
            let entry = element_counts[0].entry(key).or_insert(0);
            *entry += val;
        }

        // Range-check the very same multiplier witness committed below. Zero
        // padding is included in both these counts and the LogUp columns.
        use crate::lookup::operation::variant::rms::{RMS_MULTIPLIER_LIMBS, decompose};
        let rows = if rank >= 2 { input_shape[rank - 2] } else { 1 };
        for chunk in rmsnorm_data.normalisation.get_data().chunks(rows) {
            for value in chunk.iter().copied().chain(std::iter::repeat_n(
                0,
                rows.next_power_of_two() - chunk.len(),
            )) {
                for limb in decompose(value, RMS_MULTIPLIER_LIMBS)? {
                    *element_counts[0].entry(limb).or_insert(0) +=
                        (4 / rows.next_power_of_two().min(4)) as u64;
                }
            }
        }

        let input_evals = lookup_witness.input_mle_evals::<F>(table.num_columns());
        let output_evals = lookup_witness.output_mle_evals::<F>();

        // Make the commitments to the multipliers
        let dim_size = if rank >= 2 { input_shape[rank - 2] } else { 1 };

        let normalisation_evals = rmsnorm_data
            .normalisation
            .get_data()
            .chunks(dim_size)
            .map(|chunk| {
                let evals = chunk.iter().copied().chain(std::iter::repeat_n(
                    0,
                    dim_size.next_power_of_two() - chunk.len(),
                ));
                to_field::<_, F, _>(evals)
            })
            .collect::<Vec<Vec<F>>>();

        let mles = input_evals
            .into_iter()
            .chain(output_evals)
            .chain(normalisation_evals)
            .map(|evals| evals.into_mle())
            .collect();

        let mut gen_w = LookupWitnessGen::<F>::default();
        let tables = vec![
            Table::new_shift_check(),
            table,
            Table::new_zero_check(),
            Table::new_signed_zero_check(),
        ];
        gen_w.insert_layer_witness_data(id, mles, tables, element_counts);

        Ok(gen_w)
    }
}
