//! Internal methods for proving the [`RMSNorm`] layer.

use crate::{
    layers::LayerProof,
    lookup::operation::generic_prove::{GenericLookupProof, LookupProverResult, prove_lookup_op},
};

use std::collections::HashMap;

use super::*;

impl RMSNorm<Element> {
    pub(crate) fn prove_internal<F, T, PCS>(
        &self,
        id: NodeId,
        last_claim: &Claim<F>,
        step_data: &Step<Element>,
        prover: &mut Prover<F, T, PCS>,
    ) -> Result<Vec<Claim<F>>>
    where
        F: PrimeField,
        T: Transcript,
        PCS: CommitmentScheme<Field = F>,
    {
        let lookup_op = step_data
            .node_outputs
            .try_rmsnorm_data()
            .ok_or(anyhow!(
                "Could not get RMSNorm proving data for lookup witness generation"
            ))?
            .to_proving_data()?;

        let (_, normalisation_scaling_factor) = self.get_quantisation_scaling_factors().ok_or(
            anyhow!("Quantisation scaling factors not found for RMSNorm lookup witness generation"),
        )?;
        let table = Table::new_normalisation(normalisation_scaling_factor.bit_size() + 2);

        prover
            .transcript
            .append_scalars(&[F::from(lookup_op.right_shift() as u64)]);
        let LookupProverResult {
            generic_proof,
            input_claims,
            weight_evaluation_point,
        } = if let Some(alpha_handle) = self.alpha.as_ref() {
            let wrapped_alpha = alpha_handle.wrapped_tensor()?.clone();
            let alpha = Tensor::try_from(&wrapped_alpha)?;

            prove_lookup_op(
                &lookup_op,
                last_claim,
                step_data,
                &table,
                Some(&alpha),
                prover,
                id,
            )?
        } else {
            prove_lookup_op(&lookup_op, last_claim, step_data, &table, None, prover, id)?
        };

        let GenericLookupProof {
            logup_proof,
            sumcheck_proof,
            evaluations,
            commitments,
            weight_evaluation,
            ..
        } = generic_proof;

        let proof = RMSNormProof {
            logup_proof,
            right_shift: lookup_op.right_shift() as Element,
            commitments,
            io_proof: sumcheck_proof,
            io_evaluations: evaluations,
            alpha_evaluation: weight_evaluation,
        };

        // Add the proof to the prover
        prover.push_proof(id, LayerProof::RMSNorm(proof));

        match (
            weight_evaluation,
            weight_evaluation_point,
            self.alpha.as_ref(),
        ) {
            (Some(alpha_eval), Some(weight_eval_point), Some(alpha_tensor)) => {
                // Add the weight commitment claim to the prover
                let alpha_key = CommitmentId::from(alpha_tensor.storage_key());
                let claims_map =
                    HashMap::from([(alpha_key, Claim::<F>::new(weight_eval_point, alpha_eval))]);
                prover.add_common_claims(id, claims_map);
            }
            (None, None, None) => {
                // No alpha key, nothing to check
            }
            _ => {
                bail!("Inconsistent alpha key/evaluation state in RMSNorm proving");
            }
        }
        // Return the input claim for the next layer
        Ok(input_claims)
    }
}
