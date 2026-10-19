//! Generic proving of lookup layer for implementors of [`LookupOp`]

use crate::{
    Prover,
    graph::NodeId,
    layers::activation::Activation,
    lookup::operation::{
        inputs::{LookupEvaluations, proving::LookupSumcheckProof},
        variant::verifying::evaluate_dim_lt_poly,
    },
    model::Step,
    poly_commit::verifier::VerifierCommitment,
};

use dp_crypto::{
    IntoMLE,
    arkyper::{CommitmentScheme, transcript::Transcript},
    poly::dense::DensePolynomial,
    structs::IOPProof,
    util::ceil_log2,
};
use itertools::Itertools;

use super::*;

#[derive(Debug, Clone)]
pub struct GenericLookupProof<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    pub logup_proof: LogUpBatchProof<F>,
    pub sumcheck_proof: IOPProof<F>,
    pub evaluations: Vec<F>,
    pub weight_evaluation: Option<F>,
    pub shift_evaluations: Option<Vec<F>>,
    pub commitments: Vec<VerifierCommitment<PCS>>,
}

pub struct LookupProverResult<F, PCS>
where
    F: PrimeField,
    PCS: CommitmentScheme,
{
    pub generic_proof: GenericLookupProof<F, PCS>,
    pub input_claims: Vec<Claim<F>>,
    pub weight_evaluation_point: Option<Vec<F>>,
}

impl<F: PrimeField, PCS: CommitmentScheme> LookupProverResult<F, PCS> {
    pub fn new(
        generic_proof: GenericLookupProof<F, PCS>,
        input_claims: Vec<Claim<F>>,
        weight_evaluation_point: Option<Vec<F>>,
    ) -> Self {
        Self {
            generic_proof,
            input_claims,
            weight_evaluation_point,
        }
    }
}
/// Generic function to prove a lookup operation for any implementor of [`LookupOp`]. This function handles the generation of the LogUp proof and the linking sumcheck proof, as well as the necessary evaluations and claims.
/// It takes in the lookup operation, the last claim, the current step, the table, an optional weight tensor (if this is a normalisation variant), the prover, and the node id for which we are proving.
/// It returns a [`LookupProverResult`] containing the generated proof and any relevant evaluation points.
pub fn prove_lookup_op<L, F, T, PCS>(
    lookup_op: &L,
    last_claim: &Claim<F>,
    step: &Step<Element>,
    table: &Table,
    weight_tensor: Option<&Tensor<Element>>,
    prover: &mut Prover<F, T, PCS>,
    node_id: NodeId,
) -> Result<LookupProverResult<F, PCS>>
where
    L: LookupOp,
    F: PrimeField,
    T: Transcript,
    PCS: CommitmentScheme<Field = F>,
{
    // First we get the layer commitment and extract it into parts
    let layer_commitment = prover.lookup_witness(node_id)?;

    let (layer_polys, commitments): (Vec<_>, Vec<_>) = layer_commitment
        .iter()
        .map(|committed_poly| {
            (
                committed_poly.polynomial.clone(),
                VerifierCommitment::from(committed_poly),
            )
        })
        .unzip();

    let variant = lookup_op.variant();
    let unpadded_input_shape = step.inputs()[0].unpadded_shape();

    let chunking_info = lookup_op.chunking_info(table)?;
    let input_config = lookup_op.input_config(chunking_info, unpadded_input_shape);

    // Split the mles into the lookup input/output, normalisation and shift mles
    let rank = unpadded_input_shape.rank();
    let number_of_chunks = input_config.number_of_chunks();

    let lookup_inputs_per_chunk = chunking_info.total_inputs_per_chunk();
    let lookup_outputs_per_chunk = chunking_info.total_outputs_per_chunk();
    let lookup_polys_per_chunk = lookup_inputs_per_chunk + lookup_outputs_per_chunk;

    let (lookup_polys, others) = layer_polys.split_at(number_of_chunks * lookup_polys_per_chunk);

    let normalisers = variant.is_rmsnorm().then(|| &others[..number_of_chunks]);

    // Generate the LogUp proof
    // First we have to sort and pair off the MLEs for the LogUp proof.
    let logup_inputs = if !variant.requires_output() {
        input_config.create_logup_inputs::<F>(
            lookup_polys,
            None,
            normalisers,
            &prover.challenge_storage,
        )?
    } else if weight_tensor.is_none() {
        let output = step.output_tensor_at(0)?;
        input_config.create_logup_inputs::<F>(
            lookup_polys,
            Some(output.as_ref()),
            normalisers,
            &prover.challenge_storage,
        )?
    } else {
        let input = step.input_tensor_at(0)?;
        let wrapped_input = WrappedTensor::try_from(input.as_ref())?.reduce_to_unpadded_shape()?;
        let lookup_output = lookup_op.apply(wrapped_input, table)?;
        let output = Tensor::try_from(&lookup_output)?;
        input_config.create_logup_inputs::<F>(
            lookup_polys,
            Some(&output),
            normalisers,
            &prover.challenge_storage,
        )?
    };

    let logup_proof = new_batch_multiple_sizes_prove(&logup_inputs, prover.transcript)?;

    // Here we extract all the MLEs required in the linking sumcheck proof, the exact MLEs required depend on the variant of the lookup operation and we handle this in the match statement below.
    // We also handle the generation of any extra MLEs required for the sumcheck proof in this match statement, for example in the normalisation variant we have extra MLEs corresponding to the normalisation checks.
    let (mles, shift_mles) = match variant {
        LookupVariant::Standard => (
            // We just need the lookup output MLEs in this case
            lookup_polys
                .iter()
                .skip(number_of_chunks * lookup_inputs_per_chunk)
                .map(|p| p.as_view())
                .collect::<Vec<DensePolynomial<F>>>(),
            None,
        ),
        LookupVariant::GLU => {
            // In this case we need the lookup output MLEs and the MLEs for the other half of the GLU input (the part that is not used in the lookup and is just passed through).
            let input_tensor = step.input_tensor_at(Activation::<Element>::UP_INPUT_INDEX)?;
            let extra_sumcheck_mles =
                input_config.create_extra_sumcheck_mles(input_tensor.as_ref())?;
            (
                lookup_polys
                    .iter()
                    .skip(number_of_chunks * lookup_inputs_per_chunk)
                    .map(|p| p.as_view())
                    .chain(extra_sumcheck_mles)
                    .collect::<Vec<DensePolynomial<F>>>(),
                None,
            )
        }
        LookupVariant::Softmax { .. } => (
            // Here we just need the lookup output MLEs for the linking proof, however we also need to split off the shift MLEs for reconstructing the input evaluation.
            lookup_polys
                .iter()
                .skip(number_of_chunks * lookup_inputs_per_chunk)
                .map(|p| p.as_view())
                .collect::<Vec<DensePolynomial<F>>>(),
            Some(others),
        ),
        LookupVariant::Normalisation {
            normalised_sum_value,
            ..
        } => {
            // In this case we have everything pretty much. Extra nput MLEs to check we have multiplied by the correct normalising factor,
            // and shift MLEs if this is the LayerNorm variant.
            let input_tensor = step.input_tensor_at(0)?;
            let extra_sumcheck_mles =
                input_config.create_extra_sumcheck_mles(input_tensor.as_ref())?;

            let (normaliser_mles, shift_mles) = others.split_at(number_of_chunks);
            let padded_final_dim = unpadded_input_shape.dim(-1).next_power_of_two();
            let expanded_normaliser_mles = normaliser_mles
                .iter()
                .map(|m| {
                    m.evals_ref()
                        .iter()
                        .flat_map(|v| std::iter::repeat_n(*v, padded_final_dim))
                        .collect::<Vec<F>>()
                        .into_mle()
                })
                .collect::<Vec<DensePolynomial<F>>>();
            let mles = lookup_polys
                .iter()
                .skip(number_of_chunks * lookup_inputs_per_chunk)
                .map(|p| p.as_view())
                .chain(extra_sumcheck_mles)
                .chain(expanded_normaliser_mles)
                .collect::<Vec<DensePolynomial<F>>>();
            if normalised_sum_value.is_some() {
                (mles, Some(shift_mles))
            } else {
                (mles, None)
            }
        }
    };

    // Make the expanded weight tensor MLE if it is provided (this is multiplied element wise with the lookup output).
    let weight_mle = weight_tensor.map(|weight| {
        let padded_weight = weight.pad_next_power_of_two().into_data();
        let weight_field = padded_weight.to_field();
        let second_last_dim = if rank >= 2 {
            unpadded_input_shape.dim(-2).next_power_of_two()
        } else {
            1
        };

        vec![weight_field; second_last_dim].concat().into_mle()
    });

    // Generate the linking sumcheck proof
    let (sumcheck_proof, challenges) = input_config.prove_linking_sumcheck(
        &mles,
        prover.transcript,
        last_claim,
        logup_proof.point(),
        weight_mle,
    )?;

    let LookupSumcheckProof {
        sumcheck_proof,
        evaluations,
        sumcheck_point,
        weight_eval,
    } = sumcheck_proof;

    let all_logup_evals = logup_proof
        .output_claims()
        .iter()
        .map(|c| c.evaluation())
        .collect::<Vec<F>>();
    let final_dim_vars = ceil_log2(unpadded_input_shape.dim(-1));

    let shift_point = logup_proof.point()[final_dim_vars..].to_vec();
    let shift_evals = shift_mles
        .map(|s_mles| {
            let lt_eval = evaluate_dim_lt_poly(
                &logup_proof.point()[..final_dim_vars],
                unpadded_input_shape.dim(-1),
            )
            .unwrap();
            s_mles
                .iter()
                .map(|m| m.evaluate(&shift_point).map(|eval| eval * lt_eval))
                .collect::<anyhow::Result<Vec<F>>>()
        })
        .transpose()?;

    // Construct the witness commitment claims
    let LookupEvaluations {
        input_commitment_evals,
        output_commitment_evals,
        normalisation_commitment_evals,
        input_claim_evals,
    } = input_config.construct_lookup_evaluations(
        &all_logup_evals,
        &evaluations,
        &challenges[1..],
        &shift_evals,
    )?;

    let mut commitment_claims = input_commitment_evals
        .into_iter()
        .map(|eval| Claim::new(logup_proof.point().to_vec(), eval))
        .chain(
            output_commitment_evals
                .into_iter()
                .map(|eval| Claim::new(sumcheck_point.clone(), eval)),
        )
        .collect_vec();

    // Construct the input claims to be passed to the next layer.
    let input_claims = variant.produce_input_claims(
        unpadded_input_shape,
        lookup_op,
        last_claim.point(),
        logup_proof.point(),
        &sumcheck_point,
        input_claim_evals,
    )?;

    if matches!(variant, LookupVariant::Normalisation { .. }) {
        // In this case we don't need to deal with rounding constants and the fixed point multiplier
        // because the input evaluation comes directly from the Sumcheck (since we had to prove that the values in the lookup
        // are the elementwise product of the input tensor and a witness specific tensor of normalising values).
        for eval in normalisation_commitment_evals {
            commitment_claims.push(Claim::new(sumcheck_point[final_dim_vars..].to_vec(), eval))
        }
    }

    // Process the shift evaluations if they are present, we need to divide them by the evaluation of the dim_lt polynomial at the logup point to get the correct shift evaluations to be used in the next layer, and we also add them as claims for the witness commitment.
    let shift_evals = shift_evals
        .as_ref()
        .map(|shift_evals| {
            let inverse_lt_eval = evaluate_dim_lt_poly(
                &logup_proof.point()[..final_dim_vars],
                unpadded_input_shape.dim(-1),
            )?
            .inverse()
            .ok_or(anyhow!("Tried to invert dim lt poly eval"))?;
            anyhow::Ok(
                shift_evals
                    .iter()
                    .map(|e| {
                        let eval = *e * inverse_lt_eval;
                        commitment_claims.push(Claim::new(shift_point.clone(), eval));
                        eval
                    })
                    .collect(),
            )
        })
        .transpose()?;

    // Add the witness claims to the prover.
    prover.add_witness_claim_per_poly(node_id, commitment_claims);

    // Make the generic lookup proof to be returned to the caller.
    let generic_proof = GenericLookupProof {
        logup_proof,
        sumcheck_proof,
        evaluations,
        weight_evaluation: weight_eval,
        shift_evaluations: shift_evals,
        commitments,
    };
    // make the weight tensor evaluation point if a weight tensor was provided.
    let weight_point = weight_eval.map(|_| sumcheck_point[..final_dim_vars].to_vec());
    Ok(LookupProverResult::new(
        generic_proof,
        input_claims,
        weight_point,
    ))
}
