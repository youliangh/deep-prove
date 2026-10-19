//! Internal code for verifying RMSNorm layers.

use crate::lookup::operation::{
    generic_prove::GenericLookupProof,
    generic_verify::{LookupVerifyResult, verify_lookup_op},
};

use super::*;
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub(crate) struct RMSNormLookupVerifier {
    pub(crate) right_shift: Element,
    pub(crate) normalisation_sum_sq: Element,
    pub(crate) error_bound: Element,
    pub(crate) intermediate_bit_size: usize,
    pub(crate) has_weight: bool,
}

impl RMSNormLookupVerifier {
    fn new<F, PCS>(ctx: &RMSNormCtx, proof: &RMSNormProof<F, PCS>) -> Result<Self>
    where
        F: PrimeField,
        PCS: CommitmentScheme,
    {
        let intermediate_bit_size = ctx.input_scaling.bit_size() + 1;
        let (normalisation_sum_sq, error_bound) = bounds::norm_parameters(
            ctx.normalisation_dim_size,
            ctx.normalisation_scaling.scale(),
        )?;
        ensure!(
            (1..63).contains(&proof.right_shift),
            "Invalid RMSNorm proof right shift"
        );
        Ok(RMSNormLookupVerifier {
            right_shift: proof.right_shift,
            normalisation_sum_sq,
            error_bound,
            intermediate_bit_size,
            has_weight: ctx.alpha_key.is_some(),
        })
    }

    pub(crate) fn new_from_parts(
        right_shift: Element,
        normalisation_sum_sq: Element,
        error_bound: Element,
        intermediate_bit_size: usize,
        has_weight: bool,
    ) -> Self {
        RMSNormLookupVerifier {
            right_shift,
            normalisation_sum_sq,
            error_bound,
            intermediate_bit_size,
            has_weight,
        }
    }
}

impl LookupOp for RMSNormLookupVerifier {
    fn intermediate_bit_size(&self) -> usize {
        self.intermediate_bit_size
    }

    fn right_shift(&self) -> usize {
        self.right_shift.unsigned_abs() as usize
    }

    fn variant(&self) -> LookupVariant {
        LookupVariant::Normalisation {
            normalised_magnitude_value: self.normalisation_sum_sq,
            magnitude_error_bound: self.error_bound,
            normalised_sum_value: None,
            has_weight: self.has_weight,
        }
    }

    fn chunking_info(&self, table: &Table) -> Result<ChunkingInfo> {
        ensure!(
            (1..63).contains(&self.right_shift),
            "Invalid RMSNorm right shift"
        );
        // Appendix C relates a rounded positive multiple to its squared norm.
        // Saturation would invalidate that inference. The guard-bit table puts
        // either clamping endpoint outside the accepted sphere, so a clamped
        // coordinate cannot occur in an accepted norm proof.
        let endpoint = i128::from(table.max_output_value())
            .abs()
            .min(i128::from(table.min_output_value()).abs());
        ensure!(
            endpoint * endpoint
                > i128::from(self.normalisation_sum_sq) + i128::from(self.error_bound),
            "RMSNorm table needs a guard bit to exclude saturation"
        );
        // The normalized row bounds every coordinate, independent of the
        // input's calibration scale. One guard/zeroing chunk suffices.
        let max_bit_size = self.right_shift() + table.table_bit_size() + 1;
        ensure!(
            max_bit_size <= 63,
            "RMSNorm fixed-point decomposition exceeds i64"
        );
        ChunkingInfo::new(self.right_shift(), table, max_bit_size, 1)
    }

    fn fixed_point_multiplier(&self) -> Element {
        1
    }

    fn is_signed(&self) -> bool {
        true
    }

    fn padding_value(&self) -> Element {
        0
    }

    fn apply(
        &self,
        _input: WrappedTensor<Element>,
        _table: &Table,
    ) -> Result<WrappedTensor<Element>> {
        Err(anyhow!(
            "Cannot Apply the operation using RMSNormLookupVerifier"
        ))
    }

    fn generate_witness(
        &self,
        _input: Tensor<Element>,
        _value_table: &Table,
    ) -> Result<crate::lookup::operation::LookupOpWitness> {
        Err(anyhow!(
            "Cannot generate a witness for the operation using RMSNormLookupVerifier"
        ))
    }
}

impl RMSNormCtx {
    pub(crate) fn verify_internal<F, PCS, T>(
        &self,
        proof: &RMSNormProof<F, PCS>,
        id: NodeId,
        verifier: &mut Verifier<F, T, PCS>,
        last_claim: &Claim<F>,
        shape_step: &ShapeStep,
    ) -> Result<Vec<Claim<F>>>
    where
        F: PrimeField,
        PCS: CommitmentScheme<Field = F>,
        T: Transcript,
    {
        // First we make the lookup verifier context
        let lookup_op = RMSNormLookupVerifier::new(self, proof)?;
        verifier
            .transcript
            .append_scalars(&[F::from(lookup_op.right_shift() as u64)]);
        let RMSNormProof {
            logup_proof,
            commitments,
            io_proof,
            io_evaluations,
            alpha_evaluation,
            ..
        } = proof;

        let generic_lookup_proof = GenericLookupProof::<F, PCS> {
            logup_proof: logup_proof.clone(),
            sumcheck_proof: io_proof.clone(),
            evaluations: io_evaluations.clone(),
            commitments: commitments.clone(),
            weight_evaluation: *alpha_evaluation,
            shift_evaluations: None,
        };

        let normalisation_scaling_factor = self.normalisation_scaling();
        let table = Table::new_normalisation(normalisation_scaling_factor.bit_size() + 2);

        let LookupVerifyResult {
            input_claims,
            weight_evaluation_point,
        } = verify_lookup_op(
            &lookup_op,
            last_claim,
            shape_step,
            &table,
            &generic_lookup_proof,
            verifier,
            id,
        )?;

        match (*alpha_evaluation, weight_evaluation_point, self.alpha_key()) {
            (Some(alpha_eval), Some(weight_eval_point), Some(alpha_key)) => {
                // Add the weight commitment claim to the verifier
                let claims_map = HashMap::from([(
                    alpha_key.clone(),
                    Claim::<F>::new(weight_eval_point, alpha_eval),
                )]);
                verifier.add_common_claims(id, claims_map);
            }
            (None, None, None) => {
                // No alpha key, nothing to check
            }
            _ => {
                bail!("Inconsistent alpha key/evaluation state in RMSNorm verification");
            }
        }

        Ok(input_claims)
    }
}
