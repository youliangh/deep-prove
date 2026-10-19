use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    slice,
};

use anyhow::{anyhow, ensure};
use ark_ff::PrimeField;
use dp_crypto::{
    IntoMLE,
    arkyper::{CommitmentScheme, transcript::Transcript},
    poly::{
        dense::{DensePolynomial, Point},
        eq::evals,
    },
    structs::{IOPProof, IOPProverState},
    util::optimal_sumcheck_threads,
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use itertools::Itertools;
use tracing::debug;

use crate::{
    Claim,
    graph::NodeId,
    lookup::table::Table,
    poly_commit::{
        ChunkedCommitment, OpeningProof, build_sumcheck_expression,
        context::{CommitmentProverCtx, CommittedPolynomial},
        table_poly_id,
    },
    tensor::CommitmentId,
};

// Type alias used to represent the set of claims of the model polys
pub(crate) type ModelClaims<F> = HashMap<CommitmentId, BTreeMap<NodeId, Claim<F>>>;

/// Struct used in this file for the return type of the [`CommitmentProver::claims_batching_sumcheck`] method.
struct ClaimsBatchingProof<'a, F: PrimeField> {
    sumcheck_proof: IOPProof<F>,
    sumcheck_evals: Vec<F>,
    sumcheck_point: Point<F>,
    committed_polys: Vec<DensePolynomial<'a, F>>,
}

#[derive(Debug)]
pub struct CommitmentProver<F: PrimeField, PCS: CommitmentScheme> {
    /// A map storing all the claims for tensors fixed by the model.
    /// The `NodeId` is only employed to sort the claims related to the same
    /// static tensor, assuming that only one claim for a static tensor is
    /// produced in each node
    pub(crate) model_claims: ModelClaims<F>,
    /// The list of claims about witness polynomial. Each entry in the map
    /// refers to a layer, and it contains one or more claims for each witness
    /// polynomial committed in that layer
    pub(crate) witness_claims: BTreeMap<NodeId, Vec<Vec<Claim<F>>>>,
    _phantom: PhantomData<PCS>,
}

impl<F: PrimeField, PCS: CommitmentScheme> Default for CommitmentProver<F, PCS> {
    fn default() -> Self {
        Self {
            model_claims: Default::default(),
            witness_claims: Default::default(),
            _phantom: Default::default(),
        }
    }
}

impl<F: PrimeField, PCS: CommitmentScheme> CommitmentProver<F, PCS> {
    pub fn add_witness_claim(&mut self, node_id: NodeId, claims: Vec<Vec<Claim<F>>>) {
        self.witness_claims.insert(node_id, claims);
    }

    pub fn add_common_claims(&mut self, claims: HashMap<CommitmentId, Vec<(NodeId, Claim<F>)>>) {
        claims.into_iter().for_each(|(poly_id, claims)| {
            let poly_claims = self.model_claims.entry(poly_id.clone()).or_default();
            claims.into_iter().for_each(|(node_id, claim)| {
                assert!(
                    poly_claims.insert(node_id, claim).is_none(),
                    "Failed for poly {poly_id} in node {node_id}"
                )
            });
        });
    }

    pub fn add_table_claim(&mut self, table_id: NodeId, table: &Table, claim: Claim<F>) {
        assert!(
            self.model_claims
                .entry(table_poly_id(table.name()))
                .or_default()
                .insert(table_id, claim)
                .is_none()
        );
    }

    pub fn prove<T: Transcript>(
        mut self,
        ctx: &CommitmentProverCtx<F, PCS>,
        witness_commitments: &HashMap<NodeId, Vec<CommittedPolynomial<F, PCS>>>,
        transcript: &mut T,
    ) -> anyhow::Result<OpeningProof<F, PCS>>
    where
        PCS: CommitmentScheme<Field = F>,
    {
        debug!("Opening: starting claims batching");
        let ClaimsBatchingProof {
            sumcheck_proof,
            sumcheck_evals,
            sumcheck_point,
            committed_polys,
        } = self.claims_batching_sumcheck(witness_commitments, ctx, transcript)?;
        debug!("Opening: finished claims batching");

        // Sample the challenges to batch the committed_polys into a single polynomial;
        // Before sampling the challenges, we add all claims to the transcript
        transcript.append_scalars(&sumcheck_point);
        transcript.append_scalars(&sumcheck_evals);
        let (committed_polys, challenges): (Vec<_>, Vec<_>) = committed_polys
            .iter()
            .map(|poly| (poly, transcript.challenge_scalar::<F>()))
            .unzip();

        debug!(
            committed_chunks = committed_polys.len(),
            "Opening: starting polynomial linear combination"
        );
        let rlc_poly = DensePolynomial::linear_combination(&committed_polys, &challenges);
        debug!("Opening: finished polynomial linear combination");

        // Make the PCS opening proof
        debug!("Opening: starting PCS proof");
        let pcs_proof = PCS::prove(
            &ctx.prover_params,
            &rlc_poly,
            &sumcheck_point,
            None,
            transcript,
        )?;
        debug!("Opening: finished PCS proof");

        Ok(OpeningProof {
            sumcheck_proof,
            sumcheck_evals,
            pcs_proof,
        })
    }

    fn claims_batching_sumcheck<'a, 'b, 'c, T: Transcript>(
        &mut self,
        witness_commitments: &'a HashMap<NodeId, Vec<CommittedPolynomial<'c, F, PCS>>>,
        commit_ctx: &'b CommitmentProverCtx<'c, F, PCS>,
        transcript: &mut T,
    ) -> anyhow::Result<ClaimsBatchingProof<'c, F>>
    where
        'b: 'c,
        'a: 'c,
    {
        let model_claims = &mut self.model_claims;
        let witness_claims = &mut self.witness_claims;

        // Here we iterate over the model commitments and witness commitments so that we can construct the EQ polys and claimed evalautions
        // for each of the committed polys.
        // In addition we also compute the `num_claims_per_poly` vector, whose i-th entry specifies how many claims
        // are found in `model_claims` for the i-th model polynomial employed in the sumcheck.
        let mut eq_polys = vec![];
        let mut committed_polys = vec![];
        let mut max_num_vars = 0;

        let mut batching_coeffs_per_poly = commit_ctx
            .model_commitments
            .iter()
            .filter_map(|(id, poly)| {
                if let Some(claims) = model_claims.remove(id) {
                    let batching_coefficients = claims
                        .into_values()
                        .map(|claim| {
                            let Claim { point, eval } = claim;
                            // Append the evaluations to the transcript
                            transcript.append_scalars(&[eval]);
                            eq_polys.push(evals(poly.eq_point_per_chunk(&point)).into_mle());
                            poly.batching_coefficients_for_chunks(&point)
                        })
                        .collect_vec();
                    committed_polys.extend(poly.chunked_polys());
                    max_num_vars = max_num_vars.max(poly.num_vars_for_chunk());
                    Some(poly.to_chunked_info(batching_coefficients))
                } else {
                    None
                }
            })
            .collect_vec();

        for (node_id, claims) in witness_claims {
            let comms = witness_commitments
                .get(node_id)
                .ok_or(anyhow!("No commitments found for node {node_id}"))?;
            ensure!(
                comms.len() == claims.len(),
                "Number of claims for node {node_id} different from number of committed polys: {} vs  {}",
                claims.len(),
                comms.len(),
            );
            claims.iter().zip(comms).for_each(|(poly_claims, poly)| {
                let batching_coeffs = poly_claims
                    .iter()
                    .map(|claim| {
                        // Append the evaluations to the transcript
                        transcript.append_scalars::<F>(&[&claim.eval]);
                        let eq = evals(poly.eq_point_per_chunk(&claim.point)).into_mle();
                        eq_polys.push(eq);
                        poly.batching_coefficients_for_chunks(&claim.point)
                    })
                    .collect_vec();
                batching_coeffs_per_poly.push(poly.to_chunked_info(batching_coeffs));
                max_num_vars = max_num_vars.max(poly.num_vars_for_chunk());
                committed_polys.extend(poly.chunked_polys())
            });
        }

        debug!(
            committed_polynomials = batching_coeffs_per_poly.len(),
            claims = eq_polys.len(),
            committed_chunks = committed_polys.len(),
            eq_polynomials = eq_polys.len(),
            builder_witnesses = committed_polys.len() + eq_polys.len(),
            max_num_vars,
            "Opening: constructing batching expression"
        );
        let sumcheck_expression = build_sumcheck_expression(batching_coeffs_per_poly)?;
        debug!("Opening: finished batching expression");

        let poly_views = committed_polys
            .iter()
            .chain(&eq_polys)
            .map(|poly| {
                // we pad each polynomial to `max_num_vars` used the optimized 0-padding strategy
                // provided by `DensePolynomial`. This padding allows to get claims already for the
                // 0-padded polynomials, which are the ones needed to later generate the PCS opening proof;
                // besides that, it also make the sumcheck slightly more efficient
                let mut poly_view = poly.as_view();
                poly_view.zero_pad_num_vars(max_num_vars)?;
                Ok(poly_view)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let either_polys = poly_views
            .iter()
            .map(Either::Left)
            .collect::<Vec<Either<_, _>>>();

        let challenge = transcript.append_and_sample(b"model_batching");
        // Make the VirtualPolynomials and run Sumcheck proving
        let num_threads = optimal_sumcheck_threads(max_num_vars);
        debug!(
            builder_witnesses = either_polys.len(),
            num_threads, "Opening: constructing virtual polynomial builder"
        );
        let expr_builder =
            VirtualPolynomialsBuilder::<F>::new_with_mles(num_threads, max_num_vars, either_polys);
        debug!("Opening: expanding expression into virtual polynomials");
        let virtual_poly =
            expr_builder.to_virtual_polys(slice::from_ref(&sumcheck_expression), &[challenge]);
        debug!("Opening: finished virtual polynomial expansion");
        debug!("Opening: starting batching sumcheck");
        let (sumcheck_proof, state) = IOPProverState::<F>::prove(virtual_poly, transcript);
        debug!("Opening: finished batching sumcheck");
        let all_evals = state.get_mle_flatten_final_evaluations();
        let sumcheck_point = state.collect_raw_challenges();

        // all_evals contains the model poly evals and the eq poly evals; we want to include in the proof
        // only the model poly evals, which are the first `total_polys` evaluations, according to how the
        // sumcheck expression is built
        let sumcheck_evals = all_evals[..committed_polys.len()].to_vec();

        Ok(ClaimsBatchingProof {
            sumcheck_proof,
            sumcheck_evals,
            sumcheck_point,
            committed_polys,
        })
    }
}
