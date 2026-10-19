//! Methods and functionality called by the prover during lookup variant operations.

use dp_crypto::{
    IntoMLE,
    poly::{dense::DensePolynomial, eq::evals},
};

use crate::to_field;

use super::*;

impl LookupVariant {
    /// Provides the number of times items are looked up in the normalisation range checks lookup for this variant.
    pub fn compute_normalisation_witness_counts(
        &self,
        number_of_chunks: usize,
        output: &Tensor<Element>,
    ) -> Result<Vec<HashMap<Element, u64>>> {
        if self.is_rmsnorm() {
            return Ok(vec![count_elements(
                self.rms_residual_columns(number_of_chunks, output)?
                    .into_iter()
                    .flatten(),
            )]);
        }
        let unpadded_shape = output.unpadded_shape();
        let chunk_size = unpadded_shape.numel() / number_of_chunks;
        let dim_size = unpadded_shape.dim(-1);
        let unpadded_output = output.reduce_to_shape(unpadded_shape)?;

        match self {
            LookupVariant::Softmax {
                normalised_sum_value,
                error_bound,
            } => {
                // This offset is used to ensure the range check performed constrains that the values are within the specified error bound.
                // The largest diff possible is twice the error bound, so we offset al the values so the lie in the final (2 * error_bound) values of the range check table.
                let offset: Element = (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * error_bound);
                let norm_sum_const = normalised_sum_value + error_bound;

                let all_lookups = unpadded_output
                    .data()
                    .chunks(chunk_size)
                    .flat_map(|chunk| {
                        let len = chunk.len() / dim_size;
                        let diff = len.next_power_of_two() - len;

                        chunk
                            .chunks(dim_size)
                            .map(|row| offset + norm_sum_const - row.iter().sum::<Element>())
                            .chain(std::iter::repeat_n(offset, diff))
                            .collect::<Vec<Element>>()
                    })
                    .collect::<Vec<Element>>();

                Ok(vec![count_elements(all_lookups)])
            }

            LookupVariant::Normalisation {
                normalised_magnitude_value,
                magnitude_error_bound,
                normalised_sum_value,
                ..
            } => {
                let (norm_sum, sum_error_bound) = match normalised_sum_value {
                    Some((v, e)) => (*v, *e),
                    None => (0, 0),
                };
                // This offset is used to ensure the range check performed constrains that the values are within the specified error bound.
                // The largest diff possible is twice the error bound, so we offset al the values so the lie in the final (2 * error_bound) values of the range check table.
                let sum_offset: Element =
                    (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * sum_error_bound);
                let sumsq_offset: Element =
                    (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * magnitude_error_bound);

                // Since the value we are checking can be either side of the specified value we add the error bound to it as an offset,
                // This was if we are on the lower limit the value becomes 0, and the upper limit becomes 2 * error_bound.
                let norm_sum_const = norm_sum + sum_error_bound;
                let normalised_magnitude_const = normalised_magnitude_value + magnitude_error_bound;

                let (all_sum_lookups, all_sumsq_lookups) = unpadded_output
                    .data()
                    .chunks(chunk_size)
                    .map(|chunk| {
                        let len = chunk.len() / dim_size;
                        let diff = len.next_power_of_two() - len;

                        chunk
                            .chunks(dim_size)
                            .map(|row| {
                                let (sum, sumsq) =
                                    row.iter().fold((0, 0), |(sum_acc, sumsq_acc), x| {
                                        (sum_acc + x, sumsq_acc + x * x)
                                    });
                                (
                                    sum_offset + norm_sum_const - sum,
                                    sumsq_offset + normalised_magnitude_const - sumsq,
                                )
                            })
                            .chain(std::iter::repeat_n((sum_offset, sumsq_offset), diff))
                            .unzip::<Element, Element, Vec<Element>, Vec<Element>>()
                    })
                    .unzip::<Vec<Element>, Vec<Element>, Vec<Vec<Element>>, Vec<Vec<Element>>>();

                if normalised_sum_value.is_none() {
                    Ok(vec![count_elements(all_sumsq_lookups.concat())])
                } else {
                    Ok(vec![
                        count_elements(all_sum_lookups.concat()),
                        count_elements(all_sumsq_lookups.concat()),
                    ])
                }
            }
            LookupVariant::Standard | LookupVariant::GLU => Ok(vec![]),
        }
    }

    /// Method that computes the extra lookup inputs needed for normalisation variants.
    pub fn compute_extra_lookup_inputs<F: PrimeField>(
        &self,
        number_of_chunks: usize,
        output: &Tensor<Element>,
        constant_challenge: F,
        column_sep_challenge: F,
    ) -> Result<Vec<LogUpInput<F>>> {
        if self.is_rmsnorm() {
            let columns = self.rms_residual_columns(number_of_chunks, output)?;
            return columns
                .chunks(number_of_chunks)
                .map(|limb_columns| {
                    LogUpInput::<F>::new_lookup(
                        limb_columns
                            .iter()
                            .map(|column| column.to_field())
                            .collect(),
                        constant_challenge,
                        column_sep_challenge,
                        1,
                    )
                })
                .collect();
        }
        // Check that we have been supplied with output tensor in this case
        let unpadded_shape = output.unpadded_shape();
        let chunk_size = unpadded_shape.numel() / number_of_chunks;
        let dim_size = unpadded_shape.dim(-1);
        let unpadded_output = output.reduce_to_shape(unpadded_shape)?;

        let output_chunks = unpadded_output.data().chunks(chunk_size);
        let shift_check_table = Table::new_shift_check();

        match self {
            LookupVariant::Softmax {
                normalised_sum_value,
                error_bound,
            } => {
                // This offset is used to ensure the range check performed constrains that the values are within the specified error bound.
                // The largest diff possible is twice the error bound, so we offset all the values so they lie in the final (2 * error_bound) values of the range check table.
                let offset: Element = (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * error_bound);
                let norm_sum_const = normalised_sum_value + error_bound;
                let column_evals = output_chunks
                    .map(|chunk| {
                        let len = chunk.len() / dim_size;
                        let diff = len.next_power_of_two() - len;

                        let chunk_evals = chunk
                            .chunks(dim_size)
                            .map(|row| offset + norm_sum_const - row.iter().sum::<Element>())
                            .chain(std::iter::repeat_n(offset, diff));
                        to_field::<_, F, _>(chunk_evals)
                    })
                    .collect::<Vec<Vec<F>>>();
                let normalisation_input = LogUpInput::<F>::new_lookup(
                    column_evals,
                    constant_challenge,
                    column_sep_challenge,
                    shift_check_table.num_columns(),
                )?;
                Ok(vec![normalisation_input])
            }

            LookupVariant::Normalisation {
                normalised_magnitude_value,
                magnitude_error_bound,
                normalised_sum_value,
                ..
            } => {
                let (norm_sum, sum_error_bound) = match normalised_sum_value {
                    Some((v, e)) => (*v, *e),
                    None => (0, 0),
                };

                // This offset is used to ensure the range check performed constrains that the values are within the specified error bound.
                // The largest diff possible is twice the error bound, so we offset all the values so they lie in the final (2 * error_bound) values of the range check table.
                let sum_offset: Element =
                    (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * sum_error_bound);
                let sumsq_offset: Element =
                    (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * magnitude_error_bound);

                let norm_sum_const = norm_sum + sum_error_bound;
                let normalised_magnitude_const = normalised_magnitude_value + magnitude_error_bound;

                let (sum_column_evals, sumsq_column_evals) = output_chunks
                    .map(|chunk| {
                        let len = chunk.len() / dim_size;
                        let diff = len.next_power_of_two() - len;
                        let (sum_chunk_evals, sumsq_chunk_evals): (Vec<Element>, Vec<Element>) =
                            chunk
                                .chunks(dim_size)
                                .map(|row| {
                                    let (sum, sumsq) =
                                        row.iter().fold((0, 0), |(sum_acc, sumsq_acc), x| {
                                            (sum_acc + x, sumsq_acc + x * x)
                                        });
                                    (
                                        sum_offset + norm_sum_const - sum,
                                        sumsq_offset + normalised_magnitude_const - sumsq,
                                    )
                                })
                                .chain(std::iter::repeat_n((sum_offset, sumsq_offset), diff))
                                .unzip();
                        (sum_chunk_evals.to_field(), sumsq_chunk_evals.to_field())
                    })
                    .unzip::<Vec<F>, Vec<F>, Vec<Vec<F>>, Vec<Vec<F>>>();

                let sumsq_normalisation_input = LogUpInput::<F>::new_lookup(
                    sumsq_column_evals,
                    constant_challenge,
                    column_sep_challenge,
                    shift_check_table.num_columns(),
                )?;

                if normalised_sum_value.is_none() {
                    Ok(vec![sumsq_normalisation_input])
                } else {
                    let sum_normalisation_input = LogUpInput::<F>::new_lookup(
                        sum_column_evals,
                        constant_challenge,
                        column_sep_challenge,
                        shift_check_table.num_columns(),
                    )?;
                    Ok(vec![sumsq_normalisation_input, sum_normalisation_input])
                }
            }
            LookupVariant::Standard | LookupVariant::GLU => Ok(Vec::new()),
        }
    }

    pub(crate) fn build_eq_polys<F: PrimeField>(
        &self,
        last_claim_point: &[F],
        logup_point: &[F],
        final_dim_vars: usize,
    ) -> Vec<DensePolynomial<'_, F>> {
        let output_eq_evals = evals(last_claim_point);
        let logup_eq_evals = evals(logup_point);

        match self {
            LookupVariant::Standard | LookupVariant::GLU => {
                vec![output_eq_evals.into_mle(), logup_eq_evals.into_mle()]
            }
            LookupVariant::Softmax { .. } | LookupVariant::Normalisation { .. } => {
                let normalisation_point = std::iter::repeat_n(
                    F::from(2).inverse().expect("Cannot fail when inverting 2"),
                    final_dim_vars,
                )
                .chain(logup_point[final_dim_vars..].iter().cloned())
                .collect::<Vec<F>>();
                let normalisation_eq_evals = evals(&normalisation_point);
                vec![
                    output_eq_evals.into_mle(),
                    logup_eq_evals.into_mle(),
                    normalisation_eq_evals.into_mle(),
                ]
            }
        }
    }

    /// Flag signalling whether the output tensor is required for computing any of the LogUp witness.
    pub fn requires_output(&self) -> bool {
        matches!(
            self,
            LookupVariant::Softmax { .. } | LookupVariant::Normalisation { .. }
        )
    }
}
