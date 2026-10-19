//! Module containing code for transforming the chunked input MLEs into lookup operation inputs.

use crate::{lookup::operation::variant::LookupVariant, to_field};

use super::*;

use dp_crypto::{Expression, IntoMLE, poly::dense::DensePolynomial, util::ceil_log2};
use itertools::izip;

pub mod proving;
pub mod verifying;

#[derive(Debug, Clone)]
/// Struct holding the configuration for generating [`LogUpInput`] from the chunked input MLEs.
pub struct LookupInputConfig {
    /// The information about how many chunks there are.
    chunking_info: ChunkingInfo,
    /// The type of normalisation check to be performed. Defaults to [`NormalisationType::None`] unless specified with [`Self::with_normalisation_type`].
    variant: LookupVariant,
    /// Holds the unpadded input [`Shape`].
    unpadded_input_shape: Shape,
}

impl LookupInputConfig {
    /// Creates a new [`LookupInputConfig`] with the provided [`ChunkingInfo`].
    pub fn new(
        chunking_info: ChunkingInfo,
        variant: LookupVariant,
        unpadded_input_shape: Shape,
    ) -> Self {
        Self {
            chunking_info,
            variant,
            unpadded_input_shape,
        }
    }

    pub fn chunking_info(&self) -> &ChunkingInfo {
        &self.chunking_info
    }

    /// Getter for the number of chunks for this lookup operation.
    pub fn number_of_chunks(&self) -> usize {
        self.unpadded_input_shape[..self.unpadded_input_shape.rank().saturating_sub(2)]
            .iter()
            .product::<usize>()
    }

    /// Getter for the unpadded input shape.
    pub fn unpadded_input_shape(&self) -> &Shape {
        &self.unpadded_input_shape
    }

    pub fn create_logup_inputs<F: PrimeField>(
        &self,
        mles: &[Arc<DensePolynomial<F>>],
        output: Option<&Tensor<Element>>,
        normalisers: Option<&[Arc<DensePolynomial<F>>]>,
        challenge_storage: &ChallengeStorage<F>,
    ) -> Result<Vec<LogUpInput<F>>> {
        let chunking_info = &self.chunking_info;
        let number_of_chunks = self.number_of_chunks();
        // Calculate how many input mles we expect per chunk
        let total_inputs_per_chunk = chunking_info.total_inputs_per_chunk();

        let (all_input_mles, all_output_mles) =
            mles.split_at(number_of_chunks * total_inputs_per_chunk);

        // Each output chunk will have size equal to total inputs per chunk minus the number of shifted chunks
        let output_chunk_size = chunking_info.total_outputs_per_chunk();

        let shift_check_table = Table::new_shift_check();
        let zero_check_table = Table::new_zero_check();
        let signed_zero_check_table = Table::new_signed_zero_check();

        let (constant_challenge, shift_column_sep) = challenge_storage
            .get_challenges_by_name(&shift_check_table.name())
            .ok_or(anyhow!("Could not find challenges for ShiftCheckTable"))?;
        let (value_constant_challenge, value_column_sep) = challenge_storage
            .get_challenges_by_name(&chunking_info.table().name())
            .ok_or(anyhow!("Could not find challenges for ValueTable"))?;
        let zero_opt = challenge_storage.get_challenges_by_name(&zero_check_table.name());
        // For the signed zero check table we return an option since not all lookup operations will use it
        let signed_zero_opt =
            challenge_storage.get_challenges_by_name(&signed_zero_check_table.name());

        let number_zero_chunks = chunking_info.number_of_zeroing_chunks();
        let number_of_value_chunks = chunking_info.number_of_value_chunks();

        let mut logup_inputs = all_input_mles
            .chunks(total_inputs_per_chunk)
            .zip(all_output_mles.chunks(output_chunk_size))
            .try_fold(
                Vec::<LogUpInput<F>>::new(),
                |mut acc, (input_chunk_mles, output_chunk_mles)| {
                    // Split the input mles into shifted, value, and zeroing mles
                    let (shifted_input_mles, rest) =
                        input_chunk_mles.split_at(chunking_info.number_of_shifted_chunks());
                    let shifted_column_evals = shifted_input_mles
                        .iter()
                        .map(|mle| mle.evals())
                        .collect::<Vec<Vec<F>>>();

                    // The structure of the remaining MLEs depends on the number of columns in the value table
                    let (value_column_evals, zeroing_column_evals) = match chunking_info
                        .num_value_columns()
                    {
                        1 => {
                            // In this case the input chunk doesn't have commitment to the for the value table so `rest` only contains zeroing input MLEs
                            let zeroing_column_evals = rest
                                .iter()
                                .take(number_zero_chunks)
                                .zip(output_chunk_mles.iter().skip(number_of_value_chunks))
                                .flat_map(|(input_mle, output_mle)| {
                                    [input_mle.evals(), output_mle.evals()]
                                })
                                .collect::<Vec<Vec<F>>>();
                            // The value evals are then just given by the output part
                            let value_columns = output_chunk_mles
                                .iter()
                                .take(number_of_value_chunks)
                                .map(|mle| mle.evals())
                                .collect::<Vec<Vec<F>>>();
                            (value_columns, zeroing_column_evals)
                        }
                        2 => {
                            // In this case the first MLE in `rest` is the value input MLE, and the rest are zeroing input MLEs
                            let (value_input_mles, zeroing_input_mles) =
                                rest.split_at(number_of_value_chunks);
                            let zeroing_column_evals = zeroing_input_mles
                                .iter()
                                .take(number_zero_chunks)
                                .zip(output_chunk_mles.iter().skip(number_of_value_chunks))
                                .flat_map(|(input_mle, output_mle)| {
                                    [input_mle.evals(), output_mle.evals()]
                                })
                                .collect::<Vec<Vec<F>>>();

                            let value_columns = value_input_mles
                                .iter()
                                .zip(output_chunk_mles.iter().take(number_of_value_chunks))
                                .flat_map(|(input_mle, output_mle)| {
                                    [input_mle.evals(), output_mle.evals()]
                                })
                                .collect::<Vec<Vec<F>>>();

                            (value_columns, zeroing_column_evals)
                        }
                        _ => bail!("Value table has unsupported number of columns for LogUp proof"),
                    };

                    // Create the LogUpInput structures for the shifted inputs and value inputs
                    let shift_logup_input = LogUpInput::<F>::new_lookup(
                        shifted_column_evals,
                        constant_challenge,
                        F::ONE,
                        shift_check_table.num_columns(),
                    )
                    .map_err(|e| {
                        anyhow!("Failed to create LogUpInput for Shifted Inputs: {e:?}")
                    })?;
                    let value_logup_input = LogUpInput::<F>::new_lookup(
                        value_column_evals,
                        value_constant_challenge,
                        value_column_sep,
                        chunking_info.num_value_columns(),
                    )
                    .map_err(|e| anyhow!("Failed to create LogUpInput for Value Inputs: {e:?}"))?;

                    acc.push(shift_logup_input);
                    acc.push(value_logup_input);

                    // Now we handle the zeroing chunks, which may include a signed zero check chunk
                    match (number_zero_chunks, chunking_info.is_signed()) {
                        (1, true) => {
                            // In this case we have a single zeroing chunk which is for the signed zero check table
                            let (signed_zero_constant_challenge, signed_zero_column_sep) =
                                signed_zero_opt.ok_or(anyhow!(
                                    "Could not find challenges for SignedZeroCheckTable"
                                ))?;
                            let signed_zero_logup_input = LogUpInput::<F>::new_lookup(
                                zeroing_column_evals,
                                signed_zero_constant_challenge,
                                signed_zero_column_sep,
                                signed_zero_check_table.num_columns(),
                            )
                            .map_err(|e| {
                                anyhow!("Failed to create LogUpInput for Signed Zero Inputs: {e:?}")
                            })?;
                            acc.push(signed_zero_logup_input);
                        }
                        (n, true) if n > 1 => {
                            // In this case we split off the last two columns for the signed zero check table
                            let (zero_check_columns, signed_columns) =
                                zeroing_column_evals.split_at(2 * (n - 1));
                            let (zero_constant_challenge, zero_column_sep) = zero_opt
                                .ok_or(anyhow!("Could not find challenges for ZeroCheckTable"))?;
                            let zero_check_logup_input = LogUpInput::<F>::new_lookup(
                                zero_check_columns.to_vec(),
                                zero_constant_challenge,
                                zero_column_sep,
                                zero_check_table.num_columns(),
                            )
                            .map_err(|e| {
                                anyhow!("Failed to create LogUpInput for Zero Check Inputs: {e:?}")
                            })?;
                            acc.push(zero_check_logup_input);

                            let (signed_zero_constant_challenge, signed_zero_column_sep) =
                                signed_zero_opt.ok_or(anyhow!(
                                    "Could not find challenges for SignedZeroCheckTable"
                                ))?;

                            let signed_zero_logup_input = LogUpInput::<F>::new_lookup(
                                signed_columns.to_vec(),
                                signed_zero_constant_challenge,
                                signed_zero_column_sep,
                                signed_zero_check_table.num_columns(),
                            )
                            .map_err(|e| {
                                anyhow!("Failed to create LogUpInput for Signed Zero Inputs: {e:?}")
                            })?;
                            acc.push(signed_zero_logup_input);
                        }
                        (n, false) if n > 0 => {
                            // In this case we have only zero check tables to handle
                            let (zero_constant_challenge, zero_column_sep) = zero_opt
                                .ok_or(anyhow!("Could not find challenges for ZeroCheckTable"))?;
                            let zero_check_logup_input = LogUpInput::<F>::new_lookup(
                                zeroing_column_evals,
                                zero_constant_challenge,
                                zero_column_sep,
                                zero_check_table.num_columns(),
                            )
                            .map_err(|e| {
                                anyhow!("Failed to create LogUpInput for Zero Check Inputs: {e:?}")
                            })?;
                            acc.push(zero_check_logup_input);
                        }
                        _ => {}
                    }

                    Ok(acc)
                },
            )?;

        // Add the normalisation inputs after the other inputs
        if let Some(output) = output {
            let normalisation_inputs = self.variant.compute_extra_lookup_inputs(
                number_of_chunks,
                output,
                constant_challenge,
                shift_column_sep,
            )?;
            logup_inputs.extend(normalisation_inputs);
        }
        if self.variant.is_rmsnorm() {
            use super::variant::rms::{RMS_MULTIPLIER_LIMBS, decompose};
            let normalisers = normalisers.ok_or(anyhow!("Missing RMSNorm multiplier witnesses"))?;
            ensure!(
                normalisers.len() == number_of_chunks,
                "Wrong RMSNorm multiplier chunk count"
            );
            let mut columns = vec![Vec::new(); RMS_MULTIPLIER_LIMBS * number_of_chunks];
            for (chunk, poly) in normalisers.iter().enumerate() {
                for value in poly.evals_ref() {
                    let integer = value.into_bigint();
                    let words = integer.as_ref();
                    ensure!(
                        words.iter().skip(1).all(|&word| word == 0),
                        "Negative or oversized RMSNorm multiplier"
                    );
                    let value = Element::try_from(words[0])?;
                    for (limb, value) in decompose(value, RMS_MULTIPLIER_LIMBS)?
                        .into_iter()
                        .enumerate()
                    {
                        columns[limb * number_of_chunks + chunk].push(value.to_field());
                    }
                }
            }
            for column in &mut columns {
                let copies = 4 / column.len().min(4);
                if copies > 1 {
                    *column = column
                        .iter()
                        .flat_map(|value| std::iter::repeat_n(*value, copies))
                        .collect();
                }
            }
            for columns in columns.chunks(number_of_chunks) {
                logup_inputs.push(LogUpInput::<F>::new_lookup(
                    columns.to_vec(),
                    constant_challenge,
                    shift_column_sep,
                    1,
                )?);
            }
        }

        Ok(logup_inputs)
    }

    pub fn create_logup_verifier_instances<F: PrimeField>(
        &self,
        challenge_storage: &ChallengeStorage<F>,
    ) -> Result<Vec<LogUpVerifierInstance<F>>> {
        // Build the verifier instances for the LogUp proof
        let table = self.chunking_info().table();
        let unpadded_input_shape = self.unpadded_input_shape();
        let rank = unpadded_input_shape.rank();
        let num_chunks = self.number_of_chunks();
        let (total_padded_elems, norm_dim_size) = if rank >= 2 {
            (
                unpadded_input_shape[rank - 2].next_power_of_two()
                    * unpadded_input_shape[rank - 1].next_power_of_two(),
                unpadded_input_shape[rank - 2],
            )
        } else {
            (unpadded_input_shape[0].next_power_of_two(), 1)
        };

        let full_vars = ceil_log2(total_padded_elems);
        let norm_vars = ceil_log2(norm_dim_size);
        let chunking_info = self.chunking_info();

        let num_shifted_chunks = chunking_info.number_of_shifted_chunks();
        let num_value_chunks = chunking_info.number_of_value_chunks();
        let num_value_columns = table.num_columns();
        let num_zeroing_chunks = chunking_info.number_of_zeroing_chunks();

        let shift_check_table = Table::new_shift_check();
        let zero_check_table = Table::new_zero_check();
        let signed_zero_check_table = Table::new_signed_zero_check();

        // Retrieve all the necessary challenges
        let (shift_constant_challenge, shift_column_sep) = challenge_storage
            .get_challenges_by_name(&shift_check_table.name())
            .ok_or(anyhow!("Could not find challenges for ShiftCheckTable"))?;
        let (value_constant_challenge, value_column_sep) = challenge_storage
            .get_challenges_by_name(&table.name())
            .ok_or(anyhow!("Could not find challenges for ValueTable"))?;
        let zero_opt = challenge_storage.get_challenges_by_name(&zero_check_table.name());
        let signed_zero_opt =
            challenge_storage.get_challenges_by_name(&signed_zero_check_table.name());

        let mut instances = vec![];
        let mut norm_instances = vec![];
        for _ in 0..num_chunks {
            // Add the shifted instances
            (0..num_shifted_chunks).for_each(|_| {
                instances.push(LogUpVerifierInstance::<F>::new(
                    shift_constant_challenge,
                    shift_column_sep,
                    shift_check_table.num_columns(),
                    ProofType::Lookup,
                    full_vars - 1,
                ));
            });
            // Add the value instances
            (0..num_value_chunks).for_each(|_| {
                instances.push(LogUpVerifierInstance::<F>::new(
                    value_constant_challenge,
                    value_column_sep,
                    num_value_columns,
                    ProofType::Lookup,
                    full_vars - 1,
                ));
            });
            // Add the zeroing instances
            match (num_zeroing_chunks, table.is_signed()) {
                (1, true) => {
                    let (signed_zero_constant_challenge, signed_zero_column_sep) = signed_zero_opt
                        .ok_or(anyhow!(
                            "Could not find challenges for SignedZeroCheckTable"
                        ))?;
                    instances.push(LogUpVerifierInstance::<F>::new(
                        signed_zero_constant_challenge,
                        signed_zero_column_sep,
                        signed_zero_check_table.num_columns(),
                        ProofType::Lookup,
                        full_vars - 1,
                    ));
                }
                (n, true) if n > 1 => {
                    // First add the zero check instance
                    let (zero_constant_challenge, zero_column_sep) =
                        zero_opt.ok_or(anyhow!("Could not find challenges for ZeroCheckTable"))?;
                    (0..(n - 1)).for_each(|_| {
                        instances.push(LogUpVerifierInstance::<F>::new(
                            zero_constant_challenge,
                            zero_column_sep,
                            zero_check_table.num_columns(),
                            ProofType::Lookup,
                            full_vars - 1,
                        ));
                    });

                    // Now add the signed zero instance
                    let (signed_zero_constant_challenge, signed_zero_column_sep) = signed_zero_opt
                        .ok_or(anyhow!(
                            "Could not find challenges for SignedZeroCheckTable"
                        ))?;
                    instances.push(LogUpVerifierInstance::<F>::new(
                        signed_zero_constant_challenge,
                        signed_zero_column_sep,
                        signed_zero_check_table.num_columns(),
                        ProofType::Lookup,
                        full_vars - 1,
                    ));
                }
                (n, false) if n > 0 => {
                    // In this case we have only zero check tables to handle
                    let (zero_constant_challenge, zero_column_sep) =
                        zero_opt.ok_or(anyhow!("Could not find challenges for ZeroCheckTable"))?;
                    (0..n).for_each(|_| {
                        instances.push(LogUpVerifierInstance::<F>::new(
                            zero_constant_challenge,
                            zero_column_sep,
                            zero_check_table.num_columns(),
                            ProofType::Lookup,
                            full_vars - 1,
                        ));
                    });
                }
                _ => {}
            }

            let chunk_normalisation_instances = self
                .compute_normalisation_lookup_verifier_instances(
                    shift_constant_challenge,
                    F::ONE,
                    norm_vars,
                );
            norm_instances.extend(chunk_normalisation_instances);
        }

        // Extend with the normalisation instances
        instances.extend(norm_instances);

        Ok(instances)
    }

    /// Creates any extra MLEs required for the lookup operation.
    /// For [`LookupVariant::GLU`] this will create the MLEs corresponding to the second input to the GLU.
    /// For [`LookupVariant::Normalisation`] this will create the MLEs corresponding to the normalisation values.
    pub fn create_extra_sumcheck_mles<F: PrimeField>(
        &self,
        input_tensor: &Tensor<Element>,
    ) -> Result<Vec<DensePolynomial<'_, F>>> {
        match self.variant {
            LookupVariant::Standard | LookupVariant::Softmax { .. } => Ok(vec![]),
            LookupVariant::GLU | LookupVariant::Normalisation { .. } => {
                let unpadded_shape = input_tensor.unpadded_shape();
                let unpadded_input = input_tensor.reduce_to_shape(unpadded_shape)?;

                let rank = unpadded_shape.rank();
                let (second_last_dim, last_dim) = if rank >= 2 {
                    (unpadded_shape[rank - 2], unpadded_shape[rank - 1])
                } else {
                    (1, unpadded_shape[0])
                };

                let row_diff = last_dim.next_power_of_two() - last_dim;
                let column_diff = (second_last_dim.next_power_of_two() - second_last_dim)
                    * last_dim.next_power_of_two();
                let mles = unpadded_input
                    .data()
                    .chunks(second_last_dim * last_dim)
                    .map(|chunk| {
                        let evaluations = chunk
                            .chunks(last_dim)
                            .flat_map(|row| {
                                to_field::<Element, F, _>(row)
                                    .into_iter()
                                    .chain(std::iter::repeat_n(F::ZERO, row_diff))
                            })
                            .chain(std::iter::repeat_n(F::ZERO, column_diff))
                            .collect::<Vec<F>>();
                        evaluations.into_mle()
                    })
                    .collect::<Vec<DensePolynomial<F>>>();
                Ok(mles)
            }
        }
    }

    pub fn build_full_sumcheck_expression<F: PrimeField>(&self) -> Expression<F> {
        let chunking_info = self.chunking_info();
        let final_dim_size = self.unpadded_input_shape().dim(-1);
        // First we work out how many witness polynomials there will be total
        self.variant.build_full_sumcheck_expression(
            self.number_of_chunks(),
            final_dim_size,
            chunking_info,
        )
    }

    pub(crate) fn construct_lookup_evaluations<F: PrimeField>(
        &self,
        logup_evals: &[F],
        sumcheck_evals: &[F],
        batching_challenges: &[F],
        shift_evals: &Option<Vec<F>>,
    ) -> Result<LookupEvaluations<F>> {
        let number_of_chunks = self.number_of_chunks();
        let chunking_info = self.chunking_info();

        let lookup_inputs_per_chunk = chunking_info.total_inputs_per_chunk();
        let lookup_outputs_per_chunk = chunking_info.total_outputs_per_chunk();
        let total_claims_per_chunk = lookup_inputs_per_chunk + lookup_outputs_per_chunk;

        // In some cases additional evals will be empty (like LookupVariant::Standard or LookupVariant::Softmax)
        let (output_evals, additional_evals) =
            sumcheck_evals.split_at(lookup_outputs_per_chunk * number_of_chunks);

        // If the Lookup is part of a GLU operation then there are two input claims for the layer.
        let input_claim_evals = match self.variant {
            LookupVariant::GLU => vec![F::ZERO, F::ZERO],
            _ => vec![F::ZERO],
        };

        // Initialise the lookup evaluations struct
        // The output commeitment evals are already in order from the sumcheck evaluations
        let lookup_evaluations = LookupEvaluations {
            input_commitment_evals: vec![],
            output_commitment_evals: output_evals.to_vec(),
            normalisation_commitment_evals: vec![],
            input_claim_evals,
        };

        izip!(
            0..number_of_chunks,
            logup_evals.chunks(total_claims_per_chunk),
            batching_challenges.iter()
        )
        .try_fold(
            lookup_evaluations,
            |mut acc, (chunk_idx, logup_chunk, &batch_chal)| {
                acc.update_for_chunk(
                    self.variant,
                    chunk_idx,
                    chunking_info,
                    logup_chunk,
                    batch_chal,
                    shift_evals.as_ref(),
                    additional_evals,
                    number_of_chunks,
                )?;
                Ok(acc)
            },
        )
    }
}

pub(crate) struct LookupEvaluations<F: PrimeField> {
    pub input_commitment_evals: Vec<F>,
    pub output_commitment_evals: Vec<F>,
    pub normalisation_commitment_evals: Vec<F>,
    pub input_claim_evals: Vec<F>,
}

impl<F: PrimeField> LookupEvaluations<F> {
    #[allow(clippy::too_many_arguments)]
    fn update_for_chunk(
        &mut self,
        variant: LookupVariant,
        chunk_idx: usize,
        chunking_info: &ChunkingInfo,
        logup_chunk: &[F],
        batch_chal: F,
        shift_evals: Option<&Vec<F>>,
        additional_evals: &[F],
        number_of_chunks: usize,
    ) -> Result<()> {
        // Process input evaluations
        let (shift_check_evals, other_logup) =
            logup_chunk.split_at(chunking_info.number_of_shifted_chunks());
        let table = chunking_info.table();

        let (all_value_evals, zero_evals) =
            other_logup.split_at(table.num_columns() * chunking_info.number_of_value_chunks());
        let (value_in_evals, _) = match table.num_columns() {
            1 => (all_value_evals.to_vec(), vec![]), /* If we only have one column then all the value evals are input evals and output evals */
            2 => all_value_evals
                .chunks(2)
                .map(|pair| (pair[0], pair[1]))
                .unzip::<F, F, Vec<F>, Vec<F>>(), /* If we have two columns then we split them into input evals and output evals */
            _ => bail!(
                "Unsupported number of table columns: {}",
                table.num_columns()
            ),
        };

        let zero_in_evals = zero_evals.iter().step_by(2).copied().collect::<Vec<F>>();

        // Append the input commitment evaluations
        self.input_commitment_evals
            .extend_from_slice(shift_check_evals);
        if chunking_info.num_value_columns() != 1 {
            self.input_commitment_evals
                .extend_from_slice(&value_in_evals);
        }
        self.input_commitment_evals
            .extend_from_slice(&zero_in_evals);

        match variant {
            LookupVariant::GLU => {
                // Here we have an additional input witness per chunk, accessed via the only additional chunk evaluation
                let glu_chunk_witness = additional_evals[chunk_idx];

                let combined_lookup_chunk_input = chunking_info.combine_input_claims(
                    shift_check_evals,
                    &value_in_evals,
                    &zero_in_evals,
                );

                self.input_claim_evals[0] += combined_lookup_chunk_input * batch_chal;
                self.input_claim_evals[1] += glu_chunk_witness * batch_chal;
            }
            LookupVariant::Standard => {
                let combined_lookup_chunk_input = chunking_info.combine_input_claims(
                    shift_check_evals,
                    &value_in_evals,
                    &zero_in_evals,
                );
                self.input_claim_evals[0] += combined_lookup_chunk_input * batch_chal;
            }
            LookupVariant::Softmax { .. } => {
                let combined_lookup_chunk_input = chunking_info.combine_input_claims(
                    shift_check_evals,
                    &value_in_evals,
                    &zero_in_evals,
                );
                let shift_eval = shift_evals
                    .as_ref()
                    .map(|evals| evals[chunk_idx])
                    .ok_or(anyhow!("Shift evaluations missing for Softmax variant"))?;
                self.input_claim_evals[0] +=
                    (combined_lookup_chunk_input + shift_eval) * batch_chal;
            }
            LookupVariant::Normalisation { .. } => {
                // Here the input comes from the sumcheck evaluations in the additional chunk evaluations
                let input_chunk_eval = additional_evals[chunk_idx];

                let normalisation_eval = additional_evals[number_of_chunks + chunk_idx];
                self.input_claim_evals[0] += input_chunk_eval * batch_chal;
                self.normalisation_commitment_evals.push(normalisation_eval);
            }
        }
        Ok(())
    }
}
