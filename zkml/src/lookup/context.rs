//! File containing code for lookup witness generation.
use crate::{
    Element,
    graph::NodeId,
    iop::{
        ChallengeStorage,
        chunking::{BoundaryEdgeType, ChunkIOCommitments, ChunkedNode, ModelChunk},
        context::ProverContext,
        prover::ModelLayersRef,
    },
    layers::provable::ProvableOp,
    lookup::{
        logup_gkr::structs::{LogUpInput, LogUpVerifierInstance, ProofType},
        table::Table,
    },
    measure,
    measure::LAYER_WISE_MEASURE_PREFIX,
    model::Trace,
    poly_commit::{context::CommittedPolynomial, verifier::VerifierCommitment},
};
use anyhow::{Context as CC, anyhow};
use ark_ff::PrimeField;
use dp_crypto::{
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    poly::dense::DensePolynomial,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, btree_map};
use tracing::{debug, warn};
use utils::Metrics;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookupContext {
    /// Store the tables found in the model, with a list of the nodes
    /// using the given table
    pub(crate) tables: BTreeMap<Table, Vec<NodeId>>,
}

impl LookupContext {
    pub fn new(set: &BTreeMap<Table, Vec<NodeId>>) -> LookupContext {
        LookupContext {
            tables: set.clone(),
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &Table> {
        self.tables.keys()
    }

    pub fn is_empty(&self) -> bool {
        self.tables.is_empty()
    }

    pub fn create_logup_inputs<PCS, F>(
        &self,
        multiplicities_commitments: &[CommittedPolynomial<F, PCS>],
        challenge_storage: &ChallengeStorage<F>,
    ) -> anyhow::Result<Vec<LogUpInput<F>>>
    where
        F: PrimeField,
        PCS: CommitmentScheme,
    {
        // First we retrieve the multiplicity polynomials
        let multiplicity_polys = multiplicities_commitments
            .iter()
            .map(|committed_poly| &committed_poly.polynomial);
        self.iter()
            .zip(multiplicity_polys)
            .map(|(table, m_poly)| {
                let multiplicities = m_poly.evals();
                let column_evals = table.get_table_columns::<F>();
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&table.name())
                    .ok_or(anyhow!(
                        "No challenges found for Table {}, cannot generate LogUp input",
                        table.name()
                    ))?;
                LogUpInput::<F>::new_table(
                    column_evals,
                    multiplicities,
                    constant_challenge,
                    column_separation_challenge,
                )
                .map_err(|e| anyhow!("Table: {}, {e:?}", table.name()))
            })
            .collect::<anyhow::Result<Vec<LogUpInput<F>>>>()
    }

    pub fn create_logup_verifier_instances<F>(
        &self,
        challenge_storage: &ChallengeStorage<F>,
    ) -> anyhow::Result<Vec<LogUpVerifierInstance<F>>>
    where
        F: PrimeField,
    {
        self.iter()
            .map(|table| {
                let (constant_challenge, column_separation_challenge) = challenge_storage
                    .get_challenges_by_name(&table.name())
                    .ok_or(anyhow!(
                        "No challenges found for Table {}, cannot generate LogUp input",
                        table.name()
                    ))?;
                Ok(LogUpVerifierInstance::<F>::new(
                    constant_challenge,
                    column_separation_challenge,
                    table.num_columns(),
                    ProofType::Table,
                    table.table_bit_size() - 1,
                ))
            })
            .collect::<anyhow::Result<Vec<LogUpVerifierInstance<F>>>>()
    }
}

pub(crate) fn count_elements<I: IntoIterator<Item = Element>>(i: I) -> HashMap<Element, u64> {
    let mut count = HashMap::<Element, u64>::new();
    for v in i.into_iter() {
        *count.entry(v).or_default() += 1;
    }
    count
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct LookupWitnessGen<'a, F: PrimeField> {
    /// Contains the count of elements per table type.
    ///
    /// These values are later used to compute the GKR's multiplicities.
    element_count: BTreeMap<Table, HashMap<Element, u64>>,
    /// Contains the witness polynomials that must be committed. All polynomials are committed at once
    /// using `batch_commit` method.
    poly_to_commit: HashMap<NodeId, Vec<DensePolynomial<'a, F>>>,
}

impl<'a, F: PrimeField> Clone for LookupWitnessGen<'a, F> {
    fn clone(&self) -> Self {
        LookupWitnessGen {
            element_count: self.element_count.clone(),
            poly_to_commit: self.poly_to_commit.clone(),
        }
    }
}

impl<'a, F: PrimeField> Default for LookupWitnessGen<'a, F> {
    fn default() -> Self {
        LookupWitnessGen {
            element_count: BTreeMap::default(),
            poly_to_commit: HashMap::default(),
        }
    }
}

impl<'a, F: PrimeField> LookupWitnessGen<'a, F> {
    pub fn insert_logup_witness(&mut self, node_id: NodeId, witness: Vec<DensePolynomial<'a, F>>) {
        self.poly_to_commit.insert(node_id, witness);
    }

    pub fn insert_element_count(&mut self, table: Table, elements: HashMap<Element, u64>) {
        self.element_count.insert(table, elements);
    }
    pub fn insert_layer_witness_data(
        &mut self,
        node_id: NodeId,
        witness: Vec<DensePolynomial<'a, F>>,
        tables: Vec<Table>,
        element_counts: Vec<HashMap<Element, u64>>,
    ) {
        // First insert the witness
        self.insert_logup_witness(node_id, witness);
        // Now insert each of the lookup counts for their respective tables
        for (table, count) in tables.into_iter().zip(element_counts.into_iter()) {
            if !count.is_empty() {
                self.insert_element_count(table, count);
            }
        }
    }

    /// Consume the lookups and witness of `other` into this instance.
    fn consume(&mut self, other: Self) {
        for (table, elements) in other.element_count.into_iter() {
            match self.element_count.entry(table) {
                btree_map::Entry::Vacant(vacant_entry) => {
                    vacant_entry.insert(elements);
                }
                btree_map::Entry::Occupied(mut occupied_entry) => {
                    let agg_count = occupied_entry.get_mut();
                    for (element, count) in elements.into_iter() {
                        *agg_count.entry(element).or_default() += count;
                    }
                }
            }
        }
        self.poly_to_commit.extend(other.poly_to_commit);
    }
}

pub(crate) const COLUMN_SEPARATOR: Element = 1 << 32;

#[derive(Debug)]
pub struct LookupWitness<'a, F: PrimeField, PCS: CommitmentScheme> {
    pub(crate) logup_witnesses: HashMap<NodeId, Vec<CommittedPolynomial<'a, F, PCS>>>,
    pub(crate) table_witnesses: Option<Vec<CommittedPolynomial<'a, F, PCS>>>,
    pub(crate) chunk_commitments: ChunkIOCommitments<VerifierCommitment<PCS>>,
}

impl<'a, F: PrimeField, PCS: CommitmentScheme> Default for LookupWitness<'a, F, PCS> {
    fn default() -> Self {
        LookupWitness {
            logup_witnesses: HashMap::default(),
            table_witnesses: None,
            chunk_commitments: ChunkIOCommitments::default(),
        }
    }
}

pub(crate) fn generate_lookup_witness_for_chunk<'c, F, T, PCS>(
    chunk: &ModelChunk,
    chunk_lookup_ctx: &LookupContext,
    chunk_trace: &Trace<Element>,
    ctx: &ProverContext<F, PCS>,
    transcript: &mut T,
    layers: &'c ModelLayersRef<'c>,
) -> anyhow::Result<LookupWitness<'c, F, PCS>>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
    T: Transcript,
{
    // If there are no tables in the lookup context and there are no input/output boundary edges for this chunk,
    // then there are no lookup witnesses to generate so we return default values
    if chunk_lookup_ctx.is_empty()
        && chunk.boundary_edges(BoundaryEdgeType::Incoming).is_empty()
        && chunk.boundary_edges(BoundaryEdgeType::Outgoing).is_empty()
    {
        warn!("Lookup witness generation: no tables found, returning empty context TEST?");
        return Ok(LookupWitness::default());
    }

    // Make the witness gen struct that stores relevant table lookup data
    debug!("== Witness poly commitments generation ==");
    let metrics = Metrics::new();

    let node_ids = chunk
        .subgraph
        .forward_inners()
        .map(|(node_id, _)| node_id)
        .collect::<Vec<_>>();
    let witness_gens = node_ids
        .par_iter()
        .map(|&node_id| {
            let step = chunk_trace
                .get_step(&node_id)
                .with_context(|| format!("fetching trace for {node_id}"))?;
            let node = chunk
                .subgraph
                .node(node_id)
                .expect("Node must be found")
                .as_inner()
                .expect("Must be an inner node");
            // fetch the layer op depending on the node type
            match node {
                ChunkedNode::OriginalNode(_) => {
                    let node = layers
                        .get(&node_id)
                        .ok_or(anyhow!("Node {node_id} not found in model"))?;
                    measure::r_and_accumulate(
                        format!(
                            "{LAYER_WISE_MEASURE_PREFIX}commitment_{}",
                            node.as_kind_str()
                        )
                        .as_str(),
                        || node.gen_lookup_witness(node_id, ctx, step),
                        Some(|a, b| a + b),
                    )?
                }
                ChunkedNode::ChunkedLayer(chunked_layer) => {
                    let node = layers.get(&chunked_layer.original_node_id).ok_or(anyhow!(
                        "Node {} not found in model",
                        chunked_layer.original_node_id
                    ))?;
                    measure::r_and_accumulate(
                        format!(
                            "{LAYER_WISE_MEASURE_PREFIX}commitment_{}",
                            node.as_kind_str()
                        )
                        .as_str(),
                        || node.gen_lookup_witness(node_id, ctx, step),
                        Some(|a, b| a + b),
                    )?
                }
                ChunkedNode::SplitLayer(_) => Ok(Default::default()),
                ChunkedNode::RecombinationLayer(_) => Ok(Default::default()),
            }
            .with_context(|| format!("generating lookup witness for {node_id}: {node:?}"))
        })
        .collect::<anyhow::Result<Vec<LookupWitnessGen<'_, F>>>>()?;
    let mut witness_gen =
        witness_gens
            .into_iter()
            .fold(LookupWitnessGen::<F>::default(), |mut acc, gen_w| {
                acc.consume(gen_w);
                acc
            });

    // calculate the table multiplicities
    let multiplicities = witness_gen
        .element_count
        .par_iter()
        .map(|(table, table_lookup_data)| {
            let table_column = table.get_merged_table_columns();

            // Check to see that all the lookup values are present in the table
            #[cfg(test)]
            {
                let mut total_not_in_table = 0;
                let mut bad_values = Vec::new();
                for (key, count) in table_lookup_data.iter() {
                    let check = table_column.contains(key);
                    if !check {
                        total_not_in_table += 1;
                        bad_values.push((*key, *count));
                    }
                }
                if total_not_in_table > 0 {
                    println!(
                        "For table {}, total lookup values: {}, not in table: {}, bad values (value, count): {:?}",
                        table.name(),
                        table_lookup_data.len(),
                        total_not_in_table,
                        bad_values
                    );
                }
            }
            // We have to account for repeated entries in the lookup table. This is usually the case if the table we want to lookup from is not a power of two size, in that case we pick a row from the table
            // and repeat it until the table has the desired size.
            let table_column_map =
                table_column
                    .iter()
                    .fold(BTreeMap::<Element, u64>::new(), |mut map, elem| {
                        *map.entry(*elem).or_insert(0) += 1;
                        map
                    });
            let multiplicities = table_column
                .iter()
                .map(|table_val| {
                    if let Some(lookup_count) = table_lookup_data.get(table_val) {
                        let table_count = *table_column_map.get(table_val).unwrap();
                        let inv = if table_count != 1 {
                            F::from(table_count)
                                .inverse()
                                .expect("Tried to invert zero when computing table multiplicities")
                        } else {
                            F::ONE
                        };
                        F::from(*lookup_count) * inv
                    } else {
                        F::ZERO
                    }
                })
                .collect::<Vec<F>>();

            Ok(DensePolynomial::new(multiplicities))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    // let's put all polynomials to commit into a single vector and batch commit them
    // given flattening all polys loose information we must keep that info for later to unflatten them
    // how many polynomials per node
    let witness_len = chunk
        .subgraph
        .forward_inners()
        .filter_map(|(node_id, _)| {
            witness_gen
                .poly_to_commit
                .get(&node_id)
                .map(|v| (node_id, v.len()))
        })
        .collect::<BTreeMap<NodeId, usize>>();
    let all_witness_len = witness_len.values().sum();
    let multiplicities_len = multiplicities.len();
    let multiplicities_poly_len = multiplicities
        .iter()
        .map(|p| p.len())
        .collect::<Vec<usize>>();
    let mut all_polys = chunk
        .subgraph
        .forward_inners()
        .flat_map(|(node_id, _)| {
            if let Some(polys) = witness_gen.poly_to_commit.get_mut(&node_id) {
                std::mem::take(polys).into_iter()
            } else {
                Vec::new().into_iter()
            }
        })
        .chain(multiplicities)
        .collect::<Vec<_>>();
    assert_eq!(all_polys.len(), all_witness_len + multiplicities_len);

    // compute polys for input and outputs of the chunk
    let (input_poly_ids, input_mles): (Vec<_>, Vec<_>) = chunk
        .to_be_committed_polys(chunk_trace, BoundaryEdgeType::Incoming)?
        .into_iter()
        .unzip();

    let (output_poly_ids, output_mles): (Vec<_>, Vec<_>) = chunk
        .to_be_committed_polys(chunk_trace, BoundaryEdgeType::Outgoing)?
        .into_iter()
        .unzip();

    all_polys.extend(input_mles);
    all_polys.extend(output_mles);

    let mut all_commitments = measure::r("witness_commitment", || {
        ctx.commitment_ctx.batch_commit(all_polys)
    })?;
    let mut chunk_boundary_commitments =
        all_commitments.split_off(all_witness_len + multiplicities_len);
    assert_eq!(
        chunk_boundary_commitments.len(),
        input_poly_ids.len() + output_poly_ids.len()
    );
    let multiplicities_commitments = all_commitments.split_off(all_witness_len);
    assert_eq!(multiplicities_commitments.len(), multiplicities_len);
    assert_eq!(
        multiplicities_commitments
            .iter()
            .map(|p| p.polynomial.len())
            .collect::<Vec<usize>>(),
        multiplicities_poly_len
    );
    let mut witness_commitments = all_commitments.into_iter();
    let mut logup_witness = HashMap::new();
    debug!(
        "== Witness poly commitments generation metrics {} ==",
        metrics.to_span()
    );

    // Write the witness commitments to the transcript
    for (node_id, _) in chunk.subgraph.forward_inners() {
        let Some(len_commit) = witness_len.get(&node_id) else {
            continue;
        };
        let mut node_commits = Vec::with_capacity(*len_commit);
        for _ in 0..*len_commit {
            let committed_poly = witness_commitments.next().unwrap();
            committed_poly.append_to_transcript(transcript);
            node_commits.push(committed_poly);
        }
        logup_witness.insert(node_id, node_commits);
    }

    multiplicities_commitments
        .iter()
        .for_each(|committed_poly| committed_poly.append_to_transcript(transcript));

    // build chunk io commitments
    let output_boundary_comms = chunk_boundary_commitments
        .split_off(input_poly_ids.len())
        .into_iter()
        .zip(output_poly_ids)
        .map(|(committed_poly, poly_id)| {
            let comm = VerifierCommitment::from(&committed_poly);
            logup_witness.insert(poly_id, vec![committed_poly]);
            (poly_id, comm)
        })
        .collect();

    let input_boundary_comms = chunk_boundary_commitments
        .into_iter()
        .zip(input_poly_ids)
        .map(|(committed_poly, poly_id)| {
            let comm = VerifierCommitment::from(&committed_poly);
            logup_witness.insert(poly_id, vec![committed_poly]);
            (poly_id, comm)
        })
        .collect();

    let chunk_commitments = ChunkIOCommitments {
        inputs: input_boundary_comms,
        outputs: output_boundary_comms,
    };

    // append chunk io commitments to the transcript
    chunk_commitments.add_to_transcript(chunk.chunk_id, transcript);

    Ok(LookupWitness {
        logup_witnesses: logup_witness,
        table_witnesses: Some(multiplicities_commitments),
        chunk_commitments,
    })
}
