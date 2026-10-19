//! Methods and functionality called by the verifier during lookup variant operations.

use super::*;

/// Evaluates the row less than polynomial at the given row point for the given unpadded sequence length.
pub(crate) fn evaluate_dim_lt_poly<F: PrimeField>(
    row_point: &[F],
    unpadded_seq_len: usize,
) -> Result<F> {
    let bit_len = ceil_log2(unpadded_seq_len);
    ensure!(
        row_point.len() == bit_len,
        "Row point length {} does not match unpadded seq len log2 {bit_len}",
        row_point.len(),
    );

    let seq_len_bits = to_bit_sequence_le(unpadded_seq_len - 1, bit_len)
        .map(|bit| F::from(bit as u64))
        .collect::<Vec<F>>();
    let row_eval = eval_zeroifier_mle(row_point, &seq_len_bits);
    Ok(row_eval)
}

impl LookupVariant {
    /// Method that given the normalisation lookup verifier evals for a chunk transforms them into evaluations that can be linked to the lookup output.
    pub fn compute_norm_eval<F: PrimeField>(
        &self,
        unpadded_dim_size: usize,
        point: &[F],
        evals: &[F],
    ) -> Result<Vec<F>> {
        // We need the less than evaluation for the dimension so that we can zero out the contribution from the padded portion of the input
        let lt_eval = evaluate_dim_lt_poly(point, unpadded_dim_size)?;

        match self {
            LookupVariant::Softmax {
                normalised_sum_value,
                error_bound,
            } => {
                ensure!(
                    evals.len() == 1,
                    "Expected single eval for softmax normalisation"
                );
                let offset: Element = (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * error_bound);
                let offset_field: F = offset.to_field();
                let norm_sum_const: F = (*normalised_sum_value + error_bound).to_field();

                Ok(vec![offset_field + lt_eval * norm_sum_const - evals[0]])
            }

            LookupVariant::Normalisation {
                normalised_magnitude_value,
                magnitude_error_bound,
                normalised_sum_value,
                ..
            } => {
                if normalised_sum_value.is_none() {
                    // Both residuals and the positive multiplier are unsigned
                    // limb integers. Their sums must not wrap modulo the field.
                    ensure!(
                        F::MODULUS_BIT_SIZE > 49,
                        "RMSNorm requires a field larger than 49 bits"
                    );
                    let limbs = rms::residual_limbs(*magnitude_error_bound)?;
                    ensure!(
                        evals.len() == 2 * limbs + rms::RMS_MULTIPLIER_LIMBS,
                        "Incorrect RMSNorm limb evaluation count"
                    );
                    let upper = rms::recompose(&evals[..limbs]);
                    let lower = rms::recompose(&evals[limbs..2 * limbs]);
                    let twice_bound: F = (2 * *magnitude_error_bound).to_field();
                    ensure!(
                        upper + lower == lt_eval * twice_bound,
                        "RMSNorm complementary residuals do not sum to twice the error bound"
                    );
                    let upper_target: F =
                        (*normalised_magnitude_value + *magnitude_error_bound).to_field();
                    return Ok(vec![
                        lt_eval * upper_target - upper,
                        rms::recompose(&evals[2 * limbs..]),
                    ]);
                }
                // If there is no norm sum we just set these to zero (we won't use the values calculated for these if there is no normalised sum value)
                let (norm_sum, sum_error_bound) = match normalised_sum_value {
                    Some((v, e)) => (*v, *e),
                    None => (0, 0),
                };

                let sum_offset: Element =
                    (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * sum_error_bound);
                let sum_offset_field: F = sum_offset.to_field();

                let sumsq_offset: Element =
                    (1 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1 - (2 * magnitude_error_bound);
                let sumsq_offset_field: F = sumsq_offset.to_field();
                let norm_sum_const: F = (norm_sum + sum_error_bound).to_field();
                let norm_magnitude_const: F =
                    (*normalised_magnitude_value + magnitude_error_bound).to_field();

                let out = evals
                    .iter()
                    .zip([
                        (sumsq_offset_field, norm_magnitude_const),
                        (sum_offset_field, norm_sum_const),
                    ])
                    .map(|(&inner_eval, (offset, norm_sum))| {
                        offset + lt_eval * norm_sum - inner_eval
                    })
                    .collect::<Vec<F>>();
                Ok(out)
            }
            LookupVariant::Standard | LookupVariant::GLU => Ok(vec![]),
        }
    }

    /// Method to compute the initial claimed sum for the sumcheck in a lookup operation of this variant.
    pub fn prepare_sumcheck_verification<F: PrimeField>(
        &self,
        logup_claim: &LogUpBatchVerifierClaim<F>,
        last_claim_eval: F,
        shift_evals: &Option<Vec<F>>,
        challenges: &[F],
        chunking_info: &ChunkingInfo,
        unpadded_input_shape: &Shape,
    ) -> Result<F> {
        // First we need to know what size to chunk the logup evals into
        let table = chunking_info.table();
        let rank = unpadded_input_shape.rank();
        let number_of_chunks = unpadded_input_shape[..rank.saturating_sub(2)]
            .iter()
            .product::<usize>();
        let (second_last_dim, last_dim) = if rank >= 2 {
            (
                unpadded_input_shape.dim(rank - 2),
                unpadded_input_shape.dim(rank - 1),
            )
        } else {
            (1, unpadded_input_shape.dim(0))
        };
        let last_dim_vars = ceil_log2(last_dim);
        // This tells us how many non-normalisation lookups are performed per chunk
        let io_lookups_per_chunk =
            chunking_info.total_inputs_per_chunk() + chunking_info.total_outputs_per_chunk();
        // Convert the logup claims into just the evaluations
        let logup_evals = logup_claim
            .output_claims()
            .iter()
            .map(|c| c.evaluation())
            .collect::<Vec<F>>();
        // Split the evaluations into the IO evals and the normalisation evals, the normalisation evals are always at the end
        let (io_evals, norm_evals) = logup_evals.split_at(io_lookups_per_chunk * number_of_chunks);
        let logup_point = logup_claim.point();

        // Check that if we have some shift evals they are the correct length
        if let Some(evals) = shift_evals {
            ensure!(
                evals.len() == number_of_chunks,
                "Shift evals length {} does not match number of chunks {}",
                evals.len(),
                number_of_chunks
            );
        }

        // The sum challenge is always the first challenge
        let sum_challenge = challenges[0];

        izip!(
            0..number_of_chunks,
            io_evals.chunks(io_lookups_per_chunk),
            challenges[1..1 + number_of_chunks].iter()
        )
        .try_fold(
            last_claim_eval,
            |acc, (chunk_idx, chunk_evals, &chunk_challenge)| {
                // The output linking equation is always the first eval in the chunk
                let (shift_check_evals, rest) =
                    chunk_evals.split_at(chunking_info.number_of_shifted_chunks());
                // Split out the value evals
                let (all_value_evals, rest) = rest.split_at(table.num_columns() * chunking_info.number_of_value_chunks());
                let (value_in_evals, value_out_evals) = match table.num_columns() {
                    1 =>  (all_value_evals.to_vec(), all_value_evals.to_vec()), // If we only have one column then all the value evals are input evals and output evals
                    2 => all_value_evals.chunks(2).map(|pair| (pair[0], pair[1])).unzip::<F, F, Vec<F>, Vec<F>>(), // If we have two columns then we split them into input evals and output evals
                    _ => bail!("Unsupported number of table columns: {}", table.num_columns()),
                };

                let ValueExpression { value: _, initial_sum, witness_offset: _, sum_challenge_offset } = ValueExpression::<F>::evaluate(&value_out_evals, sum_challenge, chunking_info);
                let (zeroing_evals, _) = rest
                    .split_at(2 * chunking_info.number_of_zeroing_chunks());
                let (zero_in_evals, zero_out_evals) = zeroing_evals
                    .chunks(2)
                    .map(|pair| (pair[0], pair[1]))
                    .unzip::<F, F, Vec<F>, Vec<F>>();

                // Calculate the sum contribution from this chunk
                let sum_contribution =  zero_out_evals
                        .iter()
                        .fold(
                            (initial_sum, sum_challenge.pow([sum_challenge_offset as u64])),
                            |(acc, challenge), &zero_out_eval| {
                                (acc + zero_out_eval * challenge, challenge * sum_challenge)
                            },
                        )
                        .0;

                // Now we build the initial_claim contribution based on a variety of options
                let mut initial_claim_contribution = sum_contribution;
                // First add the norm evals if needed
                if !norm_evals.is_empty() {
                    let norm_challenge = *challenges.get(1 + number_of_chunks).ok_or(anyhow!(
                        "Not enough challenges supplied for normalisation evals"
                    ))?;

                    let norm_evals_per_chunk = self.norm_lookups_per_chunk();
                    // If there are multiple normalisation checks then each type of check is grouped together,
                    // so we have to pull out the ones required for this chunk.
                    let chunk_norm_evals = norm_evals
                        .iter()
                        .skip(chunk_idx)
                        .step_by(number_of_chunks)
                        .take(norm_evals_per_chunk)
                        .copied()
                        .collect::<Vec<F>>();

                    initial_claim_contribution += self
                        .compute_norm_eval(
                            second_last_dim,
                            &logup_point[last_dim_vars..],
                            &chunk_norm_evals,
                        )?
                        .into_iter()
                        .fold((F::ZERO, norm_challenge), |(acc, challenge), eval| {
                            (acc + eval * challenge, challenge * norm_challenge)
                        })
                        .0;
                }
                // If this is the normalisation variant we also need to add the input contribution
                // because in normalisation layers the lookup inputs are calculated via sumcheck.
                if matches!(self, LookupVariant::Normalisation { .. }) {
                    let input_challenge = challenges
                        .get(2 + number_of_chunks)
                        .ok_or(anyhow!("Not enough challenges supplied for shift evals"))?;

                    // Use the chunking info to compute the initial_input eval
                    let mut chunk_input_eval = chunking_info.combine_input_claims(
                        shift_check_evals,
                        &value_in_evals,
                        &zero_in_evals,
                    );
                    // We need to subtract the rounding constant used
                    let rounding_field: F = chunking_info.rounding_constant().to_field();
                    chunk_input_eval -= rounding_field;

                    if let Some(shift_evals) = shift_evals {
                        // Since there is a shift eval we need to subtract the shift from the unpadded portion
                        chunk_input_eval += shift_evals[chunk_idx];
                    }
                    // In this case we don't need to multiply by the fixed point multiplier because that is handled by the sumcheck.
                    initial_claim_contribution += chunk_input_eval * *input_challenge;
                }

                Ok(acc + chunk_challenge * initial_claim_contribution)
            },
        )
    }

    /// Internal method to get the number of normalisation lookups per chunk for this variant.
    pub(crate) fn norm_lookups_per_chunk(&self) -> usize {
        match self {
            LookupVariant::Standard | LookupVariant::GLU => 0,
            LookupVariant::Softmax { .. } => 1,
            LookupVariant::Normalisation {
                normalised_sum_value,
                magnitude_error_bound,
                ..
            } => {
                if normalised_sum_value.is_none() {
                    2 * rms::residual_limbs(*magnitude_error_bound)
                        .expect("Validated RMSNorm error bound")
                        + rms::RMS_MULTIPLIER_LIMBS
                } else {
                    // have to check both the sum and the sum of squares so we have two checks per chunk
                    2
                }
            }
        }
    }
}
