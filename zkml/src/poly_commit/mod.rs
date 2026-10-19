use anyhow::ensure;
use ark_ff::PrimeField;
use dp_crypto::{Expression, arkyper::CommitmentScheme, structs::IOPProof, util::log2_strict};
use itertools::Itertools;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{tensor::CommitmentId, to_bit_sequence_le};

pub mod context;
pub mod prover;
pub mod verifier;

/// Compute the `CommitmentId` employed to identify the constant polynomials
/// associated to the lookup tables
fn table_poly_id(table_name: String) -> CommitmentId {
    format!("table_{table_name}").into()
}

/// Data structure containing all the information about polynomials committed as chunks
/// necessary to build the expression for the opening proof sumcheck
struct ChunkedCommittedPolyInfo<F> {
    // number of chunks the committed polynomial is split into
    num_chunks: usize,
    // number of claims to be aggregated for the committed polynomial
    num_claims: usize,
    // A set of `num_chunks` batching challenges for each of the `num_claims` claims for the committed polynomial
    batching_coefficients: Vec<Vec<F>>,
}

/// A utility trait to be implemented by data structures that operate over committed
/// polynomials that have been split in chunks in order to be committed
trait ChunkedCommitment {
    fn num_chunks(&self) -> usize;

    fn num_vars(&self) -> usize;

    /// Compute the coefficients employed in claims batching sumheck in the opening proof
    /// to reconstruct the claim of a committed polynomial from claims over the chunks of
    /// the committed polynomials. In other words, given a MLE P(x_1,...,x_n) with n variables,
    /// which is committed as M chunks MLEs P_1(x_1,..,x_m),...,P_M(x_1,...,x_m) with m variables each,
    /// this method computes the M coefficients for a claim over a point r=(r_1,...,r_n) such that
    /// P(r_1,...,r_n) = C_1*P_1(r_1,...,r_m)+C_2*P_2(r_1,...r_m)+...+C_M*P_M(r_1,...,r_m).
    /// More specifically, these coefficiens corresponds to C_1 = eq(0,r_{m+1}..r_n), C_2 = eq(1, r_{m+1}..r_n),
    /// and so on until C_M=eq(M-1,r_{m+1}..r_n)
    fn batching_coefficients_for_chunks<F: PrimeField>(&self, point: &[F]) -> Vec<F> {
        let num_chunks = self.num_chunks();
        let num_batching_vars = log2_strict(num_chunks);
        let (_, batching_coords) = point.split_at(point.len() - num_batching_vars);
        (0..num_chunks)
            .map(|i| {
                let bits = to_bit_sequence_le(i, num_batching_vars)
                    .map(|b| F::from(b as u64))
                    .collect_vec();
                identity_eval(batching_coords, &bits)
            })
            .collect()
    }

    /// Given a claim over a point r=(r_1,...,r_n) for a polynomial P with `n` variables
    /// committed as `M` chunks MLEs `P_1,...,P_M` of `m` variables each, this method
    /// computes the portion of the point employed to build the `eq` polynomials to be
    /// employed in the opening proof sumcheck for each chunked MLE P_1,...,P_M
    fn eq_point_per_chunk<'a, F: PrimeField>(&self, point: &'a [F]) -> &'a [F] {
        let num_chunks = self.num_chunks();
        let num_batching_vars = log2_strict(num_chunks);
        point.split_at(point.len() - num_batching_vars).0
    }

    fn to_chunked_info<F>(&self, batching_challenges: Vec<Vec<F>>) -> ChunkedCommittedPolyInfo<F> {
        ChunkedCommittedPolyInfo {
            num_chunks: self.num_chunks(),
            num_claims: batching_challenges.len(),
            batching_coefficients: batching_challenges,
        }
    }

    /// Compute the humber of variables each chunk of a committed polynomial
    fn num_vars_for_chunk(&self) -> usize {
        num_vars_for_chunk(self.num_vars(), self.num_chunks())
    }
}

/// Compute the humber of variables each chunk of a committed polynomial with `poly_num_vars`
/// variables
fn num_vars_for_chunk(poly_num_vars: usize, num_chunks: usize) -> usize {
    poly_num_vars - log2_strict(num_chunks)
}

/// Preserve summand order while bounding the expression's recursive traversal/drop depth.
/// A left fold over all model and lookup claims can overflow the main thread's stack
/// when dp-crypto expands the expression into monomials, before sumcheck even starts.
fn balanced_expression_sum<F: PrimeField>(
    summands: impl IntoIterator<Item = Expression<F>>,
) -> Expression<F> {
    let mut level = summands.into_iter().collect_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut pairs = level.into_iter();
        while let Some(left) = pairs.next() {
            next.push(match pairs.next() {
                Some(right) => left + right,
                None => left,
            });
        }
        level = next;
    }
    level.pop().unwrap_or(Expression::ZERO)
}

/// Build the sumcheck expression for the sumcheck employed in the opening proof to batch claims
/// over the committed polynomials. It requires as input the information about the committed polynomials
/// and how they are chunked in order to be committed
fn build_sumcheck_expression<F: PrimeField>(
    chunked_poly_info: Vec<ChunkedCommittedPolyInfo<F>>,
) -> anyhow::Result<Expression<F>> {
    // dp-crypto assigns u16 witness IDs and increments a u16 witness counter.
    // Reject unsupported counts before either index cast below can truncate.
    let mut witness_count = 0usize;
    for info in &chunked_poly_info {
        ensure!(
            info.num_claims == info.batching_coefficients.len(),
            "Expected {} claim coefficient rows, found {}",
            info.num_claims,
            info.batching_coefficients.len()
        );
        for count in [info.num_chunks, info.num_claims] {
            ensure!(
                count <= usize::from(u16::MAX) - witness_count,
                "Opening expression exceeds the supported {} witness polynomials",
                u16::MAX
            );
            witness_count += count;
        }
    }
    let eq_offset: usize = chunked_poly_info
        .iter()
        .map(|chunked_poly_info| chunked_poly_info.num_chunks)
        .sum();
    tracing::info!(
        committed_polynomials = chunked_poly_info.len(),
        committed_chunks = eq_offset,
        claims = witness_count - eq_offset,
        witness_slots = witness_count,
        "Building opening claims expression"
    );
    // basically, for each pair of (poly, claim), we need to add a term to the sumcheck of the
    // type `challenge*poly*eq_poly(claim.point)`. Given that there might be more than one
    // input claim for each model polynomial, we need to make sure to use the same `poly`
    // for all the terms referring to the same polynomial; instead, the `eq_poly` will be different
    // in each term (even for terms related to the same model polynomial), since it depends on
    // `claim.point`. If a polynomial is split into multiple chunks when committed, we replace the
    // polynomial `poly` in each term with its chunk decomposition for the corresponding claim point:
    // in other words, if a polynomial `P` is split into multiple chunks `P_1, ..., P_M`, the polynomial
    // `poly` is replaced in each term of the expression by `C_1*P_1 + C_2*P_2 + ... + C_M*P_M`,
    // where `C_1,...,C_M` are the coefficients found in `chunked_poly_info` for each claim of
    // the polynomial `P` to be batched
    let mut terms = vec![Expression::ZERO];
    let (mut total_num_terms, mut total_polys) = (0, 0);
    for chunked_poly_info in chunked_poly_info {
        for (i, chunk_coeffs) in chunked_poly_info
            .batching_coefficients
            .into_iter()
            .enumerate()
        {
            ensure!(
                chunk_coeffs.len() == chunked_poly_info.num_chunks,
                "Expected {} chunk challenges, found {}",
                chunked_poly_info.num_chunks,
                chunk_coeffs.len()
            );
            let chunk_poly_expr = balanced_expression_sum(
                std::iter::once(Expression::ZERO).chain(chunk_coeffs.into_iter().enumerate().map(
                    |(chunk_index, coeff)| {
                        Expression::Constant(coeff)
                            * Expression::WitIn((total_polys + chunk_index) as u16)
                    },
                )),
            );
            terms.push(
                Expression::Challenge(0, total_num_terms + i, F::ONE, F::ZERO)
                    * chunk_poly_expr
                    * Expression::WitIn((total_num_terms + i + eq_offset) as u16),
            );
        }
        total_num_terms += chunked_poly_info.num_claims;
        total_polys += chunked_poly_info.num_chunks;
    }
    Ok(balanced_expression_sum(terms))
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct OpeningProof<F: PrimeField, PCS: CommitmentScheme> {
    /// This is the sumcheck proof that is used so that all model polynomials are evaluated at the same point.
    sumcheck_proof: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    /// This is the list of evals for all the model commitments after the sumcheck.
    sumcheck_evals: Vec<F>,
    /// The opening proof for the commitments FOR the witness polynomials
    pcs_proof: PCS::Proof,
}

pub fn aggregated_rlc<F: PrimeField>(claims: &[F], challenges: &[F]) -> anyhow::Result<F> {
    ensure!(
        claims.len() == challenges.len(),
        "claims length not equal to challenges length {} != {}",
        claims.len(),
        challenges.len()
    );
    Ok(claims
        .par_iter()
        .zip(challenges)
        .fold(|| F::ZERO, |acc, (claim, r)| acc + *claim * *r)
        .reduce(|| F::ZERO, |res, acc| res + acc))
}

/// Compute multilinear identity test between two points: returns 1 if points are equal, 0 if different.
/// Used as equality checker in polynomial commitment verification.
/// Compute Beta(r1,r2) = prod_{i \in [n]}((1-r1[i])(1-r2[i]) + r1[i]r2[i])
/// NOTE: the two vectors don't need to be of equal size. It compute the identity eval on the
/// minimum size between the two vector
pub(crate) fn identity_eval<F: PrimeField>(r1: &[F], r2: &[F]) -> F {
    let max_elem = std::cmp::min(r1.len(), r2.len());
    let v1 = &r1[..max_elem];
    let v2 = &r2[..max_elem];
    v1.iter().zip(v2).fold(F::ONE, |eval, (r1_i, r2_i)| {
        let one = F::ONE;
        eval * (*r1_i * *r2_i + (one - *r1_i) * (one - *r2_i))
    })
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use anyhow::anyhow;
    use ark_ff::{AdditiveGroup, Field, UniformRand};
    use ark_std::rand::{Rng, rngs::StdRng};
    use dp_crypto::{
        Expression,
        arkyper::{HyperKZG, transcript::blake3::Blake3Transcript},
        poly::dense::DensePolynomial,
        utils::eval_by_expr,
    };
    use itertools::Itertools;
    use parking_lot::RwLockUpgradableReadGuard;
    use rstest::rstest;

    use crate::{
        Claim,
        lookup::table::Table,
        poly_commit::{
            context::{CHUNK_SIZE_THRESHOLD, GlobalCommitmentContext},
            prover::CommitmentProver,
            verifier::{CommitmentVerifier, VerifierClaim},
        },
        rng_from_env_or_random,
        tensor::CommitmentId,
    };
    type F = ark_bn254::Fr;

    type Pcs = HyperKZG<ark_bn254::Bn254>;

    #[test]
    fn test_opening_expression_preserves_claim_and_chunk_order() {
        use super::{ChunkedCommittedPolyInfo, build_sumcheck_expression};

        let expression = build_sumcheck_expression(vec![
            ChunkedCommittedPolyInfo {
                num_chunks: 2,
                num_claims: 2,
                batching_coefficients: vec![
                    vec![F::from(2), -F::from(3)],
                    vec![F::ZERO, F::from(5)],
                ],
            },
            ChunkedCommittedPolyInfo {
                num_chunks: 1,
                num_claims: 3,
                batching_coefficients: vec![
                    vec![F::from(7)],
                    vec![F::from(11)],
                    vec![-F::from(13)],
                ],
            },
        ])
        .unwrap();
        let witnesses = [3, 5, 7, 11, 13, 17, 19, 23].map(F::from);
        let challenge = F::from(29);
        // Two chunked-polynomial claims followed by three claims for another
        // polynomial; EQ witnesses start after all three committed chunks.
        let expected = (F::from(2 * 3) - F::from(3 * 5)) * F::from(11)
            + challenge * F::from(5 * 5 * 13)
            + challenge.pow([2]) * F::from(7 * 7 * 17)
            + challenge.pow([3]) * F::from(11 * 7 * 19)
            - challenge.pow([4]) * F::from(13 * 7 * 23);
        assert_eq!(
            eval_by_expr(&witnesses, &[], &[challenge], &expression),
            expected
        );

        let expanded = expression.get_monomial_terms();
        let expanded_value: F = expanded
            .into_iter()
            .map(|term| {
                eval_by_expr(&witnesses, &[], &[challenge], &term.scalar)
                    * term
                        .product
                        .iter()
                        .map(|factor| eval_by_expr(&witnesses, &[], &[challenge], factor))
                        .product::<F>()
            })
            .sum();
        assert_eq!(expanded_value, expected);
    }

    #[test]
    fn test_opening_expression_many_claims_on_small_stack() {
        std::thread::Builder::new()
            .name("opening-expression-small-stack".into())
            .stack_size(256 * 1024)
            .spawn(|| {
                use super::{ChunkedCommittedPolyInfo, build_sumcheck_expression};

                const POLYNOMIALS: usize = 3_000;
                const CHUNKS: usize = 2;
                const CLAIMS_PER_POLYNOMIAL: usize = 4;
                const CLAIMS: usize = POLYNOMIALS * CLAIMS_PER_POLYNOMIAL;
                const EQ_OFFSET: usize = POLYNOMIALS * CHUNKS;
                // Small, one-variable MLEs isolate expression size from model
                // size, SRS generation and the cost of a full-model proof.
                let values = (0..EQ_OFFSET + CLAIMS)
                    .map(|i| {
                        DensePolynomial::new(vec![F::from((i + 1) as u64), F::from((i + 9) as u64)])
                            .evaluate(&[F::from(7)])
                            .unwrap()
                    })
                    .collect_vec();
                let challenge = F::from(17);
                let mut challenge_power = F::ONE;
                let mut expected = F::ZERO;
                let infos = (0..POLYNOMIALS)
                    .map(|poly| {
                        let batching_coefficients = (0..CLAIMS_PER_POLYNOMIAL)
                            .map(|claim| {
                                let coefficients =
                                    vec![F::from((poly + 1) as u64), -F::from((claim + 2) as u64)];
                                let chunk_value = coefficients[0] * values[poly * CHUNKS]
                                    + coefficients[1] * values[poly * CHUNKS + 1];
                                expected += challenge_power
                                    * chunk_value
                                    * values[EQ_OFFSET + poly * CLAIMS_PER_POLYNOMIAL + claim];
                                challenge_power *= challenge;
                                coefficients
                            })
                            .collect();
                        ChunkedCommittedPolyInfo {
                            num_chunks: CHUNKS,
                            num_claims: CLAIMS_PER_POLYNOMIAL,
                            batching_coefficients,
                        }
                    })
                    .collect();
                let expression = build_sumcheck_expression(infos).unwrap();
                assert_eq!(
                    eval_by_expr(&values, &[], &[challenge], &expression),
                    expected
                );
                let expanded = expression.get_monomial_terms();
                let expanded_value: F = expanded
                    .iter()
                    .map(|term| {
                        eval_by_expr(&values, &[], &[challenge], &term.scalar)
                            * term
                                .product
                                .iter()
                                .map(|factor| eval_by_expr(&values, &[], &[challenge], factor))
                                .product::<F>()
                    })
                    .sum();
                assert_eq!(expanded_value, expected);
                // Both the tree and its expansion are also dropped on this
                // restricted stack, covering recursive expression destruction.
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn test_opening_expression_rejects_invalid_counts() {
        use super::{ChunkedCommittedPolyInfo, build_sumcheck_expression};

        let empty = build_sumcheck_expression::<F>(vec![]).unwrap();
        assert_eq!(empty, Expression::ZERO);
        for info in [
            ChunkedCommittedPolyInfo {
                num_chunks: 1,
                num_claims: 2,
                batching_coefficients: vec![vec![F::ONE]],
            },
            ChunkedCommittedPolyInfo {
                num_chunks: 2,
                num_claims: 1,
                batching_coefficients: vec![vec![F::ONE]],
            },
            ChunkedCommittedPolyInfo {
                num_chunks: usize::from(u16::MAX),
                num_claims: 1,
                batching_coefficients: vec![vec![F::ONE]],
            },
            ChunkedCommittedPolyInfo {
                num_chunks: usize::MAX,
                num_claims: 0,
                batching_coefficients: vec![],
            },
        ] {
            assert!(build_sumcheck_expression(vec![info]).is_err());
        }
    }

    #[rstest]
    #[case::no_chunked_committed_polys(false)]
    #[case::chunked_committed_polys(true)]
    fn test_poly_commit(#[case] chunked_committed_polys: bool) -> anyhow::Result<()> {
        use parking_lot::RwLockWriteGuard;

        let lock = if chunked_committed_polys {
            // Set CHUNK_SIZE_THRESHOLD to 10 to test chunked committed polynomials
            let mut lock = CHUNK_SIZE_THRESHOLD.write();
            lock.replace(10);
            lock
        } else {
            // Set `CHUNK_SIZE_THRESHOLD` to `None` to use the default value
            let mut lock = CHUNK_SIZE_THRESHOLD.write();
            lock.take();
            lock
        };

        // downgrade to read lock on `CHUNK_SIZE_THRESHOLD`
        let read_lock = RwLockWriteGuard::downgrade_to_upgradable(lock);
        let max_witness_num_vars = 14;
        // sample model polynomials
        let mut rng = rng_from_env_or_random();
        let model_polys: HashMap<_, _> = (0..5)
            .map(|i| {
                let num_vars = rng.gen_range(10..16);
                (
                    CommitmentId::from(format!("model_poly_{i}")),
                    DensePolynomial::<F>::random(num_vars, &mut rng),
                )
            })
            .collect();

        let max_node_id = 10;

        let lookup_ctx = [Table::new_gelu(1.0, 1.0, 10)];

        let (prover_ctx, verifier_ctx) = GlobalCommitmentContext::<F, Pcs>::new(
            max_witness_num_vars,
            model_polys.clone(),
            lookup_ctx.iter().collect_vec().as_slice(),
            max_node_id.into(),
            &mut rng,
        )?
        .generate_contexts()?;

        let gen_point = |num_vars, rng: &mut StdRng| {
            let mut point = vec![];
            for _ in 0..num_vars {
                point.push(F::rand(rng))
            }
            point
        };

        let mut commit_prover = CommitmentProver::<F, Pcs>::default();

        // compute claims for model polynomials
        let model_claims = model_polys
            .iter()
            .enumerate()
            .map(|(i, (poly_id, poly))| {
                let point = gen_point(poly.num_vars(), &mut rng);
                let eval = poly.evaluate(&point)?;
                let mut claims = vec![(0.into(), Claim::new(point, eval))];
                if i % 2 == 1 {
                    // add another claim
                    let point = gen_point(poly.num_vars(), &mut rng);
                    let eval = poly.evaluate(&point)?;
                    claims.push((1.into(), Claim::new(point, eval)));
                }
                commit_prover.add_common_claims([(poly_id.clone(), claims.clone())].into());
                anyhow::Ok((
                    poly_id.clone(),
                    claims.into_iter().collect::<HashMap<_, _>>(),
                ))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        // compute claim for tables
        let table_claims = lookup_ctx
            .iter()
            .filter_map(|table| table.committed_columns::<F>().map(|poly| (table, poly)))
            .map(|(table, poly)| {
                let point = gen_point(poly.num_vars(), &mut rng);
                let eval = poly.evaluate(&point)?;
                let claim = Claim::new(point, eval);
                commit_prover.add_table_claim(prover_ctx.table_node_id(), table, claim.clone());
                Ok((table, claim))
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        // compute witness polynomials and their commitments
        let num_witness_polys = 8;
        let witness_polys = (0..num_witness_polys)
            .map(|_| {
                let num_vars = rng.gen_range(10..max_witness_num_vars);
                DensePolynomial::<F>::random(num_vars, &mut rng)
            })
            .collect_vec();

        // compute commitemnts
        let committed_polys = prover_ctx.batch_commit(witness_polys)?;

        // split committed polynomials in different batches, each identified by an incremental node id
        let batch_sizes: [usize; 2] = [2, 3];

        let witness_commitments = committed_polys
            .into_iter()
            .fold(
                (HashMap::new(), 0, 0),
                |(mut witness_commitments, mut node_id, mut current_batch_size), poly| {
                    witness_commitments
                        .entry(node_id.into())
                        .or_insert(vec![])
                        .push(poly);
                    current_batch_size += 1;
                    if current_batch_size == batch_sizes[node_id % batch_sizes.len()] {
                        // the current batch has enough polynomials, so we go to the next batch
                        node_id += 1;
                        current_batch_size = 0;
                    }
                    (witness_commitments, node_id, current_batch_size)
                },
            )
            .0;

        let witness_claims = witness_commitments
            .iter()
            .map(|(node_id, committed_polys)| {
                let claims = committed_polys
                    .iter()
                    .map(|poly| {
                        let point = gen_point(poly.polynomial.num_vars(), &mut rng);
                        let eval = poly.polynomial.evaluate(&point)?;
                        let claim = Claim::new(point, eval);
                        Ok(vec![claim])
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                commit_prover.add_witness_claim(*node_id, claims.clone());
                Ok((*node_id, claims))
            })
            .collect::<anyhow::Result<BTreeMap<_, _>>>()?;

        let mut transcript = Blake3Transcript::new(b"commit test");
        let proof = commit_prover.prove(&prover_ctx, &witness_commitments, &mut transcript)?;

        let mut commit_verifier = CommitmentVerifier::<F, Pcs>::default();

        commit_verifier.add_common_claims(model_claims);

        table_claims.into_iter().for_each(|(table_type, claim)| {
            commit_verifier.add_table_claim(verifier_ctx.table_node_id(), table_type, claim)
        });

        for (node_id, claims) in witness_claims {
            let committed_polys = witness_commitments
                .get(&node_id)
                .ok_or(anyhow!("Commitments not found for node {node_id}"))?;
            let verifier_claims = claims
                .into_iter()
                .zip(committed_polys)
                .map(|(claims, poly)| VerifierClaim {
                    commitment: poly.into(),
                    claims,
                })
                .collect();
            commit_verifier.add_witness_claim(node_id, verifier_claims)
        }

        let mut transcript = Blake3Transcript::new(b"commit test");
        commit_verifier.verify(&verifier_ctx, proof, &mut transcript)?;
        // Write `None` to `CHUNK_SIZE_THRESHOLD`
        RwLockUpgradableReadGuard::upgrade(read_lock).take();
        Ok(())
    }
}
