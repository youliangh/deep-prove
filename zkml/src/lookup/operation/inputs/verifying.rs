//! Lookup verifying methods.
use dp_crypto::{
    arkyper::{CommitmentScheme, transcript::Transcript},
    structs::IOPVerifierState,
    utils::eval_by_expr_with_instance,
    virtual_poly::VPAuxInfo,
};

use crate::{
    iop::verifier::Verifier, lookup::operation::inputs::proving::LookupSumcheckProof,
    poly_commit::identity_eval,
};

use super::*;

impl LookupInputConfig {
    /// Method that computes the verifier instances needed for normalisation variants.
    pub fn compute_normalisation_lookup_verifier_instances<F: PrimeField>(
        &self,
        shift_check_constant_challenge: F,
        shift_check_column_sep: F,
        norm_vars: usize,
    ) -> Vec<LogUpVerifierInstance<F>> {
        // Short circuit if no normalisation is needed
        if matches!(self.variant, LookupVariant::Standard | LookupVariant::GLU) {
            return Vec::new();
        }

        let shift_table = Table::new_shift_check();
        match self.variant {
            LookupVariant::Softmax { .. } => {
                vec![LogUpVerifierInstance::<F>::new(
                    shift_check_constant_challenge,
                    shift_check_column_sep,
                    shift_table.num_columns(),
                    ProofType::Lookup,
                    norm_vars - 1,
                )]
            }
            LookupVariant::Normalisation {
                normalised_sum_value,
                ..
            } => {
                if normalised_sum_value.is_none() {
                    vec![
                        LogUpVerifierInstance::<F>::new(
                            shift_check_constant_challenge,
                            shift_check_column_sep,
                            shift_table.num_columns(),
                            ProofType::Lookup,
                            norm_vars.saturating_sub(1).max(1),
                        );
                        self.variant.norm_lookups_per_chunk()
                    ]
                } else {
                    vec![
                        LogUpVerifierInstance::<F>::new(
                            shift_check_constant_challenge,
                            shift_check_column_sep,
                            shift_table.num_columns(),
                            ProofType::Lookup,
                            norm_vars - 1,
                        );
                        2
                    ]
                }
            }
            _ => unreachable!("Already checked for Standard and GLU cases"),
        }
    }

    /// Method that computes the verifier instances needed for normalisation variants.
    pub fn sort_fractional_outputs<F, T, PCS>(
        &self,
        verifier: &mut Verifier<F, T, PCS>,
        logup_proof: &LogUpBatchProof<F>,
    ) -> Result<()>
    where
        F: PrimeField,
        T: Transcript,
        PCS: CommitmentScheme,
    {
        let chunking_info = self.chunking_info();
        let number_of_chunks = self.number_of_chunks();

        let regular_lookups_per_chunk = chunking_info.number_of_shifted_chunks()
            + chunking_info.number_of_value_chunks()
            + chunking_info.number_of_zeroing_chunks();

        let (numerators, denominators) = logup_proof.fractional_outputs();

        let shift_check_table = Table::new_shift_check();
        let table = chunking_info.table();
        let zero_check_table = Table::new_zero_check();
        let signed_zero_check = Table::new_signed_zero_check();

        for (numerator_chunk, denominator_chunk) in numerators
            .chunks(regular_lookups_per_chunk)
            .zip(denominators.chunks(regular_lookups_per_chunk))
            .take(number_of_chunks)
        {
            // First we split out the shift check numerators and denominators
            let (shifted_nums, other_nums) =
                numerator_chunk.split_at(chunking_info.number_of_shifted_chunks());
            let (shifted_dens, others_dens) =
                denominator_chunk.split_at(chunking_info.number_of_shifted_chunks());
            // Update the stored fractional outputs, checking no denominators are zero
            let (shift_num, shift_denom) = verifier
                .numerators_and_denominators
                .entry(shift_check_table.name())
                .or_insert((F::ZERO, F::ONE));
            shifted_nums
                .iter()
                .zip(shifted_dens.iter())
                .try_for_each(|(&num, &den)| {
                    ensure!(
                        den != F::ZERO,
                        "Denominator was zero in shift check lookup!"
                    );
                    *shift_num = num * *shift_denom + *shift_num * den;
                    *shift_denom *= den;
                    Ok(())
                })?;
            // Update the fractional outputs for the value lookup
            let (value_nums, zero_nums) =
                other_nums.split_at(chunking_info.number_of_value_chunks());
            let (value_denoms, zero_denoms) =
                others_dens.split_at(chunking_info.number_of_value_chunks());

            let (value_n, value_d) = verifier
                .numerators_and_denominators
                .entry(table.name())
                .or_insert((F::ZERO, F::ONE));

            value_nums
                .iter()
                .zip(value_denoms.iter())
                .try_for_each(|(&num, &den)| {
                    ensure!(den != F::ZERO, "Denominator was zero in value lookup!");
                    *value_n = num * *value_d + *value_n * den;
                    *value_d *= den;
                    Ok(())
                })?;

            // Now we split into cases depending on the number of zeroing chunks and whether the table is signed
            match (table.is_signed(), chunking_info.number_of_zeroing_chunks()) {
                // No zero chunks, so no fractional outputs to update in this case
                (_, 0) => {}
                // Signed and one zero chunk, so we only need to deal with the signed zero check table
                (true, 1) => {
                    let (signed_zero_num, signed_zero_denom) = verifier
                        .numerators_and_denominators
                        .entry(signed_zero_check.name())
                        .or_insert((F::ZERO, F::ONE));
                    zero_nums
                        .iter()
                        .zip(zero_denoms.iter())
                        .try_for_each(|(&num, &den)| {
                            ensure!(
                                den != F::ZERO,
                                "Denominator was zero in signed zero check lookup!"
                            );
                            *signed_zero_num = num * *signed_zero_denom + *signed_zero_num * den;
                            *signed_zero_denom *= den;
                            Ok(())
                        })?;
                }
                // Signed and more than one zero chunk so we need to update both the zero check accumulator and the signed zero check accumulator
                (true, n) if n > 1 => {
                    let (zero_nums, signed_num) = zero_nums.split_at(n - 1);
                    let (zero_denoms, signed_denom) = zero_denoms.split_at(n - 1);
                    let (zero_check_num, zero_check_denom) = verifier
                        .numerators_and_denominators
                        .entry(zero_check_table.name())
                        .or_insert((F::ZERO, F::ONE));
                    zero_nums
                        .iter()
                        .zip(zero_denoms.iter())
                        .try_for_each(|(&num, &den)| {
                            ensure!(den != F::ZERO, "Denominator was zero in zero check lookup!");
                            *zero_check_num = num * *zero_check_denom + *zero_check_num * den;
                            *zero_check_denom *= den;
                            Ok(())
                        })?;

                    let (signed_zero_num, signed_zero_denom) = verifier
                        .numerators_and_denominators
                        .entry(signed_zero_check.name())
                        .or_insert((F::ZERO, F::ONE));

                    ensure!(
                        signed_denom[0] != F::ZERO,
                        "Denominator was zero in signed zero check lookup!"
                    );
                    *signed_zero_num =
                        signed_num[0] * *signed_zero_denom + *signed_zero_num * signed_denom[0];
                    *signed_zero_denom *= signed_denom[0];
                }
                // Unsigned, so we need only update the zero check accumulator
                (false, n) if n > 0 => {
                    let (zero_check_num, zero_check_denom) = verifier
                        .numerators_and_denominators
                        .entry(zero_check_table.name())
                        .or_insert((F::ZERO, F::ONE));
                    zero_nums
                        .iter()
                        .zip(zero_denoms.iter())
                        .try_for_each(|(&num, &den)| {
                            ensure!(den != F::ZERO, "Denominator was zero in zero check lookup!");
                            *zero_check_num = num * *zero_check_denom + *zero_check_num * den;
                            *zero_check_denom *= den;
                            Ok(())
                        })?;
                }
                _ => unreachable!("All cases covered above"),
            }
        }
        // If there are any remaining numerators and denominators they correspond to normalisation checks, which use the shift check table so we update the accumulators here.
        let (shift_num, shift_denom) = verifier
            .numerators_and_denominators
            .entry(shift_check_table.name())
            .or_insert((F::ZERO, F::ONE));

        numerators
            .iter()
            .zip(denominators.iter())
            .skip(number_of_chunks * regular_lookups_per_chunk)
            .try_for_each(|(&num, &den)| {
                ensure!(
                    den != F::ZERO,
                    "Denominator was zero in normalisation shift check lookup!"
                );
                *shift_num = num * *shift_denom + *shift_num * den;
                *shift_denom *= den;
                Ok(())
            })?;
        Ok(())
    }
    /// Method that verifies the sumcheck proof linking the lookup argument to the inputs/outputs of the layer. It takes in the proof, the current claim, the claim for the logup proof and optionally the shift evaluations (if this is a normalisation/softmax variant) and returns the challenges and point used in the sumcheck.
    pub fn verify_linking_sumcheck<F, T>(
        &self,
        lookup_sumcheck_proof: &LookupSumcheckProof<F>,
        transcript: &mut T,
        last_claim: &Claim<F>,
        logup_claim: &LogUpBatchVerifierClaim<F>,
        shift_evals: &Option<Vec<F>>,
    ) -> Result<(Vec<F>, Vec<F>)>
    where
        F: PrimeField,
        T: Transcript,
    {
        let chunking_info = self.chunking_info();
        let unpadded_input_shape = self.unpadded_input_shape();

        let (output_eq_point, mut batching_challenges) = unpadded_input_shape
            .compute_eq_point_and_batching_challenges::<F>(last_claim.point())?;

        // Now we squeeze the required challenges from the transcript, we always need the sum challenge so we squeeze that explicitly first.
        let sum_challenge = transcript.append_and_sample(b"sum");
        let other_challenges: Vec<F> = self.variant.squeeze_sumcheck_challenges(transcript);

        batching_challenges.insert(0, sum_challenge);
        batching_challenges.extend(other_challenges);

        // Construct the data needed for the sumcheck verification
        let initial_claim = self.variant.prepare_sumcheck_verification(
            logup_claim,
            last_claim.evaluation(),
            shift_evals,
            &batching_challenges,
            chunking_info,
            unpadded_input_shape,
        )?;

        let expression = self.build_full_sumcheck_expression::<F>();
        let degree = expression.degree();
        let max_num_variables = output_eq_point.len();
        let aux_info = VPAuxInfo::<F> {
            max_degree: degree,
            max_num_variables,
            ..Default::default()
        };
        let subclaim = IOPVerifierState::<F>::verify(
            initial_claim,
            &lookup_sumcheck_proof.sumcheck_proof,
            &aux_info,
            transcript,
        );
        let sumcheck_point = &subclaim.point;

        let output_eq_eval = identity_eval(&output_eq_point, sumcheck_point);
        let logup_eq_eval = identity_eval(logup_claim.point(), sumcheck_point);

        let mut all_evals = lookup_sumcheck_proof.evaluations.clone();
        all_evals.push(output_eq_eval);
        all_evals.push(logup_eq_eval);

        if matches!(
            self.variant,
            LookupVariant::Normalisation { .. } | LookupVariant::Softmax { .. }
        ) {
            let norm_dim_vars = ceil_log2(unpadded_input_shape.dim(-1));
            let norm_point = std::iter::repeat_n(
                F::from(2).inverse().expect("Cannot fail when inverting 2"),
                norm_dim_vars,
            )
            .chain(logup_claim.point()[norm_dim_vars..].iter().cloned())
            .collect::<Vec<F>>();
            let norm_eq_eval = identity_eval(&norm_point, sumcheck_point);
            all_evals.push(norm_eq_eval);
        }

        if let Some(weight_eval) = lookup_sumcheck_proof.weight_eval {
            all_evals.push(weight_eval);
        }

        let calculated_claim = eval_by_expr_with_instance(
            &[],
            &all_evals,
            &[],
            &[],
            &batching_challenges,
            &expression,
        );

        ensure!(
            calculated_claim == subclaim.expected_evaluation,
            "Lookup sumcheck verification failed: calculated claim {calculated_claim} does not match expected claim {}",
            subclaim.expected_evaluation
        );

        Ok((batching_challenges, sumcheck_point.clone()))
    }
}
