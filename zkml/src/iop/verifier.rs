use crate::{
    InitTranscript, Proof, SerializableField,
    iop::{
        ChunkProof, TableProof,
        chunking::{
            ChunkID, ChunkedInNode, ChunkedInput, ChunkedNode, ChunkedOutNode, ChunkedOutput,
            SplittedIOInfo, initialize_transcript_for_chunk,
        },
        compute_claim,
        context::ShapeStep,
    },
    layers::{provable::Evaluate, split::SplitLayer},
    lookup::logup_gkr::verifier::new_verify_logup_proof_multiple_sizes,
    measure,
    poly_commit::verifier::{CommitmentVerifier, VerifierClaim},
    quantization::ToField,
    tensor::WrappedTensor,
};

use crate::{
    Claim, VectorTranscript,
    graph::{Node, NodeId, NodeInput, PortId},
    iop::{ChallengeStorage, context::VerifierContext, prover::MergeClaimsProof},
    layers::{
        LayerCtx, LayerProof,
        provable::{OpInfo, VerifiableCtx},
    },
    lookup::context::LookupContext,
    tensor::{CommitmentId, Tensor},
};
use anyhow::{Context as _, anyhow, bail, ensure};
use ark_ff::PrimeField;
use dp_crypto::arkyper::{CommitmentScheme, transcript::Transcript};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tracing::{info_span, trace};

pub(crate) type SplittedIO<F> = HashMap<usize, Vec<Tensor<SerializableField<F>>>>;

fn split_io_tensors<F: PrimeField>(
    split_io_info: &SplittedIOInfo,
    io: &[Tensor<SerializableField<F>>],
) -> anyhow::Result<SplittedIO<F>> {
    split_io_info.iter()
        .map(|(io_id, new_nodes)| {
            let num_chunks = new_nodes.len();
            let split_layer = SplitLayer {
                unpadded_input_shapes: vec![io[*io_id].unpadded_shape().clone(); 1],
                num_chunks: vec![num_chunks; 1], // this is employed to split only one input/output tensor
            };
            let input = WrappedTensor::try_from(io[*io_id].to_element())?;
            let layer_out = split_layer.evaluate(&[&input])?;
            let outputs = layer_out.outputs.into_iter()
                .map(|out|
                    Tensor::try_from(out).map(|t| t.pad_next_power_of_two().to_field())
                ).collect::<anyhow::Result<Vec<_>>>()?;
            Ok((*io_id, outputs))
        }).collect::<anyhow::Result<HashMap<_,_>>>()
}

/// What the verifier must have besides the proof
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct IO<F: PrimeField> {
    /// Input of the inference given to the model
    pub input: Vec<Tensor<SerializableField<F>>>,
    /// Output of the inference
    pub output: Vec<Tensor<SerializableField<F>>>,
    splitted_inputs: Option<SplittedIO<F>>,
    pub(crate) splitted_outputs: Option<SplittedIO<F>>,
}

impl<F: PrimeField> IO<F> {
    pub fn new(
        input: Vec<Tensor<SerializableField<F>>>,
        output: Vec<Tensor<SerializableField<F>>>,
    ) -> Self {
        Self {
            input,
            output,
            ..Default::default()
        }
    }

    pub(crate) fn with_splitted_inputs(
        mut self,
        splitted_inputs: Option<&SplittedIOInfo>,
    ) -> anyhow::Result<Self> {
        self.splitted_inputs = splitted_inputs
            .map(|split_io_info| split_io_tensors(split_io_info, &self.input))
            .transpose()?;
        Ok(self)
    }

    pub(crate) fn with_splitted_outputs(
        mut self,
        splitted_outputs: Option<&SplittedIOInfo>,
    ) -> anyhow::Result<Self> {
        self.splitted_outputs = splitted_outputs
            .map(|split_io_info| split_io_tensors(split_io_info, &self.output))
            .transpose()?;
        Ok(self)
    }

    pub fn inputs(&self) -> &[Tensor<SerializableField<F>>] {
        &self.input
    }

    /// Bind cached chunk IO to the full public tensors used by the LLM statement.
    pub(crate) fn validate_split_binding(&self) -> anyhow::Result<()> {
        for (full, split) in [
            (&self.input, self.splitted_inputs.as_ref()),
            (&self.output, self.splitted_outputs.as_ref()),
        ] {
            let Some(split) = split else { continue };
            let mut split_info = SplittedIOInfo::new();
            for (&index, chunks) in split {
                let tensor = full
                    .get(index)
                    .ok_or_else(|| anyhow!("Invalid split IO index"))?;
                tensor.validate_shape()?;
                ensure!(
                    !chunks.is_empty() && chunks.len() <= tensor.unpadded_shape().dim(0),
                    "Invalid split IO chunk count"
                );
                let rows = tensor.unpadded_shape().dim(0);
                let padded_rows = rows
                    .checked_next_power_of_two()
                    .context("Split IO row count overflows")?;
                let padded_chunks = chunks
                    .len()
                    .checked_next_power_of_two()
                    .context("Split IO chunk count overflows")?;
                let chunk_size = padded_rows / padded_chunks;
                ensure!(
                    chunk_size > 0 && rows.div_ceil(chunk_size) == chunks.len(),
                    "Split IO chunk count would create missing or empty chunks"
                );
                for chunk in chunks {
                    chunk.validate_shape()?;
                }
                // Only the number of nodes is used to reconstruct the split.
                split_info.insert(index, vec![NodeId::from(0); chunks.len()]);
            }
            let expected = split_io_tensors(&split_info, full)?;
            for (&index, chunks) in split {
                for (actual, expected) in chunks.iter().zip(&expected[&index]) {
                    ensure!(
                        actual.shape() == expected.shape()
                            && actual.unpadded_shape() == expected.unpadded_shape()
                            && actual.data() == expected.data(),
                        "Split IO does not match the full public tensor"
                    );
                }
            }
        }
        Ok(())
    }
}

pub struct Verifier<'a, F: PrimeField, T: Transcript, PCS>
where
    PCS: CommitmentScheme,
{
    pub(crate) io: &'a IO<F>,
    pub(crate) commit_verifier: CommitmentVerifier<F, PCS>,
    pub(crate) transcript: &'a mut T,
    pub(crate) challenge_storage: ChallengeStorage<F>,
    pub(crate) numerators_and_denominators: HashMap<String, (F, F)>,
}

impl<'a, F: PrimeField, T: Transcript, PCS: CommitmentScheme<Field = F>> Verifier<'a, F, T, PCS> {
    pub(crate) fn new(transcript: &'a mut T, io: &'a IO<F>) -> Self {
        let commit_verifier = CommitmentVerifier::<F, PCS>::default();
        Self {
            io,
            commit_verifier,
            transcript,
            challenge_storage: ChallengeStorage::<F>::default(),
            numerators_and_denominators: HashMap::new(),
        }
    }

    fn initialise_transcript(ctx: &VerifierContext<F, PCS>) -> anyhow::Result<T>
    where
        T: InitTranscript,
    {
        let mut transcript = T::new(T::InitData::from(b"model_proving"));
        ctx.write_to_transcript(&mut transcript)?;
        Ok(transcript)
    }

    pub(crate) fn verify_chunk(
        mut self,
        ctx: &VerifierContext<F, PCS>,
        proof: ChunkProof<F, PCS>,
        shape_steps: &HashMap<NodeId, ShapeStep>,
    ) -> anyhow::Result<()> {
        let chunk = &proof.chunk_data.model_chunk;
        let chunk_id = chunk.chunk_id;
        // Add chunk splitting info to the transcript
        chunk.add_chunk_data_to_transcript(self.transcript)?;

        let lookup_ctx = chunk.chunk_lookup_ctx(&ctx.lookup);

        // Instantiate everything and append relevant info to the transcript
        // iterate over the step proofs in inference order
        for (node_id, node) in chunk.subgraph.forward_inners() {
            let split_layer = if let ChunkedNode::SplitLayer(split_layer) = node {
                Some(LayerCtx::Split(split_layer.clone()))
            } else {
                None
            };
            let recombination_layer = if let ChunkedNode::RecombinationLayer(rec_layer) = node {
                Some(LayerCtx::Recombination(rec_layer.clone()))
            } else {
                None
            };
            let layer_ctx = match node {
                ChunkedNode::OriginalNode(_) => ctx
                    .model
                    .nodes
                    .node(node_id)
                    .ok_or(anyhow!("Node {node_id} not found verifier context"))?
                    .as_inner()
                    .expect("Node {node_id} must be an inner node"),
                ChunkedNode::ChunkedLayer(chunked_layer) => ctx
                    .model
                    .nodes
                    .node(chunked_layer.original_node_id)
                    .ok_or(anyhow!(
                        "Node {} not found verifier context",
                        chunked_layer.original_node_id
                    ))?
                    .as_inner()
                    .unwrap_or_else(|| {
                        panic!(
                            "Node {} must be an inner node",
                            chunked_layer.original_node_id
                        )
                    }),
                ChunkedNode::SplitLayer(_) => split_layer.as_ref().unwrap(),
                ChunkedNode::RecombinationLayer(_) => recombination_layer.as_ref().unwrap(),
            };

            if !layer_ctx.has_proof() {
                // if the current node is not provable, there is no proof, so we can skip it
                continue;
            }
            let node_proof = proof
                .steps
                .get(&node_id)
                .ok_or(anyhow!("Proof for node {node_id} not found"))?;
            layer_ctx.write_proof_to_transcript(node_proof, self.transcript)?;
        }

        if let Some(table_proof) = &proof.table_proof {
            table_proof.write_commitments(self.transcript);
        }

        // Add chunk commitments to the transcript
        let chunk_commitments = &proof.chunk_data.commitments;
        chunk_commitments.add_to_transcript::<PCS, T>(chunk_id, self.transcript);

        // Here we generate and store all lookup related challenges
        // TODO: make this part of verifier struct
        self.challenge_storage = if lookup_ctx.is_empty() {
            ChallengeStorage::<F>::default()
        } else {
            ChallengeStorage::<F>::initialise(&lookup_ctx, self.transcript)
        };

        // compute the claims for the model outputs produced in this chunk, each identified by the
        // model output port ID
        let output_claims_by_port = chunk.model_outputs_in_chunk()?.into_iter()
            .try_fold(
                BTreeMap::new(), // we first collect all the output tensors, sorted by the output port ID
                |mut outputs, edge_id| {
                let output_edge = chunk.edge(&edge_id)?;
                let target_node = chunk.subgraph.target_node(&edge_id)?;
                ensure!(
                    output_edge.ports().len() == 1,
                    "Expected 1 port link for model output edge {edge_id} in chunk {chunk_id}, found {}",
                    output_edge.ports().len()
                );
                let out_node = target_node.as_output().ok_or(
                    anyhow!("Edge {edge_id} is not an output edge of the model")
                )?;
                let output_tensor = match out_node {
                    ChunkedOutNode::OriginalNode(out_id) => {
                        ensure!(
                            *out_id < self.io.output.len(),
                            "No model output found for {out_id}, there are only {} outputs",
                            self.io.output.len(),
                        );
                        self.io.output[*out_id].to_field()
                    },
                    ChunkedOutNode::Chunked(ChunkedOutput {
                        io_id: out_id,
                        chunk_id,
                    }) => {
                        let Some(splitted_outs) = self.io.splitted_outputs.as_ref().ok_or(
                            anyhow!("No splitted model outputs in verifier IO")
                        )?.get(out_id) else {
                            bail!("No splitted tensors found for output {out_id}")
                        };
                        ensure!(
                            *chunk_id < splitted_outs.len(),
                            "No tensor found for chunk {chunk_id} of model output {out_id}"
                        );
                        splitted_outs[*chunk_id].to_field()
                    }
                };
                let output_id = ChunkedOutput::from(out_node);
                ensure!(
                    outputs.insert(output_id.clone(), output_tensor).is_none(),
                    "Found output tensor twice for chunk {} of output id {} in chunk {chunk_id}",
                    output_id.chunk_id,
                    output_id.io_id,
                );
                Ok(outputs)
            })? // then, we compute the claims for each output
            .into_iter()
            .map(|(port_id, tensor)| {
                // For the output, we manually evaluate the MLE and check if it's the same as what prover
                // gave. Note prover could ellude that but it's simpler to avoid that special check right
                // now.
                Ok((port_id, compute_claim(self.transcript, tensor)?))
            }).collect::<anyhow::Result<HashMap<_,_>>>()?;

        let chunk_output_claims = proof
            .chunk_data
            .output_evals
            .iter()
            .fold(
                (HashMap::new(), vec![]),
                |(mut output_claims, mut common_point), (port, poly_eval)| {
                    if poly_eval.num_vars > common_point.len() {
                        // we need to add `poly_eval.num_vars - common_point.len()` coordinates to `common_point`
                        let mut new_coordinates = self
                            .transcript
                            .read_challenges(poly_eval.num_vars - common_point.len());
                        common_point.append(&mut new_coordinates);
                    }
                    output_claims.insert(
                        *port,
                        Claim::new(common_point[..poly_eval.num_vars].to_vec(), poly_eval.eval),
                    );
                    (output_claims, common_point)
                },
            )
            .0;

        // ===== Verify each proof sequentially =====
        //
        // always make sure the proof corresponds to the expected type of proof
        // in the context.
        let claims = measure::r("verify_claims", || {
            chunk.subgraph.backward_iter().try_fold(
            HashMap::<NodeInput, Claim<F>>::new(),
            |mut claims, (node_id, node)| -> anyhow::Result<HashMap<NodeInput, Claim<F>>> {
                match node {
                    Node::Inner(inner_node) => {
                        let split_layer = if let ChunkedNode::SplitLayer(split_layer) = inner_node {
                            Some(LayerCtx::Split(split_layer.clone()))
                        } else {
                            None
                        };
                        let recombination_layer = if let ChunkedNode::RecombinationLayer(rec_layer) = inner_node {
                            Some(LayerCtx::Recombination(rec_layer.clone()))
                        } else {
                            None
                        };
                        let layer = match inner_node {
                            ChunkedNode::OriginalNode(_) => ctx.model.nodes.node(node_id)
                                .ok_or(anyhow!("Node {node_id} not found verifier context"))?
                                .as_inner()
                                .expect("Node {node_id} must be an inner node"),
                            ChunkedNode::ChunkedLayer(chunked_layer) => ctx.model.nodes.node(chunked_layer.original_node_id)
                                .ok_or(anyhow!("Node {} not found verifier context", chunked_layer.original_node_id))?
                                .as_inner()
                                .unwrap_or_else(|| panic!("Node {} must be an inner node", chunked_layer.original_node_id)),
                            ChunkedNode::SplitLayer(_) => split_layer.as_ref().unwrap(),
                            ChunkedNode::RecombinationLayer(_) => recombination_layer.as_ref().unwrap(),
                        };
                        let node_proof = proof
                                .steps
                                .get(&node_id)
                                .unwrap_or(&LayerProof::Dummy);

                        // In a bug-less situation, there is no reason for that
                        // to ever happen.
                        let shape_step = shape_steps
                            .get(&node_id)
                            .ok_or(anyhow!("Shape for node {node_id} not found"))?;
                        trace!(
                            "Verifying proof {} for node {node_id}",
                            node_proof.variant_name(),
                        );

                        // ===== Compute the claims generated by this node =====
                        //
                        // There is one claim per input port for this node. This
                        // claim is computed from the claims fed into this node
                        // output ports. These are the claims associated to the
                        // input port by which this node successors are
                        // connected to this node.
                        //
                        // --
                        //  |
                        //  -> A -> B
                        //  ->
                        //  |
                        // --
                        //
                        // A will generate two claims, for its ports 1 and 2,
                        // computed from its incoming claim on output port 0,
                        // which is the claim generated by B on its input port 0.
                        let claims_for_node = chunk.claims_for_node(
                            node_id,
                            &claims,
                            &chunk_output_claims,
                        )?;

                        // Merge claims that are redundant. If a node out port
                        // is fed to two distinct nodes, then this will result
                        // in this port receiving two claims. These must be
                        // aggregated into a single one.
                        let claims_for_verify = self.verify_merge_claims_proof(
                            claims_for_node,
                            proof.merge_claim_proofs.get(&node_id),
                        )?;

                        let my_claims = {
                            if layer.is_provable() {
                                if let ChunkedNode::ChunkedLayer(chunked_layer) = inner_node {
                                    layer.verify_chunk(
                                        node_proof,
                                        &claims_for_verify.iter().collect_vec(),
                                        &mut self,
                                        shape_step,
                                        node_id,
                                        chunked_layer.chunk_number,
                                    )
                                } else {
                                    layer
                                        .verify(
                                            node_proof,
                                            &claims_for_verify.iter().collect_vec(),
                                            &mut self,
                                            shape_step,
                                            node_id,
                                        )
                                }.with_context(|| format!(
                                    "Verification failed for node with ID {node_id}: {}",layer.describe()
                                ))?
                            } else {
                                // we only propagate the claims, without
                                // changing them, as a non-provable layer
                                // shouldn't change the input values
                                claims_for_verify
                            }
                        };

                        // Insert the claims generated by this node in the
                        // global claim register.
                        claims.extend(
                            my_claims
                                .into_iter()
                                .enumerate()
                                .map(|(i, claim)| (NodeInput::new(node_id, i), claim)),
                        );

                        // ===== Input verification =====
                        //
                        // evaluating the input at the random evaluation point
                        // from the sumcheck.
                        let (inputs, input_claims) = chunk.model_inputs_in_chunk()?
                            .into_iter()
                            .map(|edge_id| {
                                let edge = chunk.edge(&edge_id)?;
                                Ok((edge_id, edge))
                            })
                            .filter_map_ok(|(edge_id, edge)| {
                                (edge.target() == node_id).then_some((edge_id, edge))
                            })
                            .try_fold(
                                (Vec::new(), Vec::new()),
                                |(mut inputs, mut input_claims), res: anyhow::Result<_>| {
                                    let (edge_id, edge) = res?;
                                    let input_node = chunk
                                        .subgraph
                                        .source_node(&edge_id)?
                                        .as_input()
                                        .ok_or(anyhow!(
                                            "Edge {edge_id} is not a model input edge in chunk {chunk_id}"
                                        ))?;
                                    let input_tensor = match input_node {
                                        ChunkedInNode::OriginalNode(input_id) => {
                                            ensure!(
                                                *input_id < self.io.input.len(),
                                                "No model input found for {input_id}, there are only {} inputs",
                                                self.io.input.len(),
                                            );
                                            &self.io.input[*input_id]
                                        },
                                        ChunkedInNode::Chunked(ChunkedInput {
                                            io_id: input_id,
                                            chunk_id,
                                        }) => {
                                            let Some(splitted_ins) = self.io.splitted_inputs.as_ref().ok_or(
                                                anyhow!("No splitted model inputs in verifier IO")
                                            )?.get(input_id) else {
                                                bail!("No splitted tensors found for input {input_id}")
                                            };
                                            ensure!(
                                                *chunk_id < splitted_ins.len(),
                                                "No tensor found for chunk {chunk_id} of model input {input_id}"
                                            );
                                            &splitted_ins[*chunk_id]
                                        }
                                    };
                                    edge.ports().iter().for_each(|port| {
                                        inputs.push(
                                            input_tensor,
                                        );
                                        input_claims
                                            .push(&claims[&NodeInput::new(node_id, port.target_port)])
                                    });
                                    anyhow::Ok((inputs, input_claims))
                                }
                            )?;
                        if !inputs.is_empty() {
                            <LayerCtx<F> as VerifiableCtx<F, PCS>>::verify_input_claim(
                                layer,
                                &inputs,
                                &input_claims,
                            )?;
                        }
                    }
                    Node::Input(_) => {}
                    Node::Output(o) => {
                        claims.insert(NodeInput::new(node_id, 0), output_claims_by_port[&o.into()].clone());
                    }
                };
                Ok(claims)
            },
        )
        })?;

        // Now we need add the claims about the input and output of the chunk
        chunk
            .compute_output_boundary_edges_claims(&chunk_output_claims)?
            .into_iter()
            .try_for_each(|(commitment_id, claims)| {
                let claim = VerifierClaim {
                    commitment: chunk_commitments
                        .outputs
                        .get(&commitment_id)
                        .ok_or(anyhow!(
                            "No commitment found for polynomial {commitment_id} in chunk {chunk_id}"
                        ))?
                        .clone(),
                    claims,
                };
                self.commit_verifier
                    .add_witness_claim(commitment_id, vec![claim]);
                anyhow::Ok(())
            })?;

        chunk.compute_input_boundary_edges_claims(&claims)?
            .into_iter()
            .try_for_each(|(commitment_id, claims)| {
                let claim = VerifierClaim {
                        commitment: chunk_commitments
                            .inputs
                            .get(&commitment_id)
                            .ok_or(anyhow!(
                                "No commitment found for input group {commitment_id} in chunk {chunk_id}"
                            ))?.clone(),
                        claims,
                    };
                self.commit_verifier
                    .add_witness_claim(commitment_id, vec![claim]);
                anyhow::Ok(())
        })?;

        // ===== Verify the lookup table proofs =====

        if let Some(proof) = &proof.table_proof {
            let TableProof { lookup, .. } = proof;
            let (proof_nums, proof_dens) = lookup.fractional_outputs();
            itertools::izip!(lookup_ctx.iter(), proof_nums, proof_dens).try_for_each(
                |(table, num, denom)| {
                    ensure!(
                        denom != F::ZERO,
                        "Denominator was zero for lookup table {}",
                        table.name()
                    );
                    let (table_num, table_denom) = self
                        .numerators_and_denominators
                        .entry(table.name())
                        .or_insert((F::ZERO, F::ONE));
                    *table_num = num * *table_denom + *table_num * denom;
                    *table_denom *= denom;
                    Ok(())
                },
            )?;
            verify_table::<_, _, _>(
                proof,
                &lookup_ctx,
                chunk_id,
                ctx.commitment_ctx.table_node_id(),
                &mut self.commit_verifier,
                self.transcript,
                &self.challenge_storage,
            )?;
        }

        // ===== Verify the opening of the accumulation of claims =====
        self.commit_verifier
            .verify(&ctx.commitment_ctx, proof.commit, self.transcript)?;

        // ===== Verify that the accumulated numerator is zero and accumulated denominator is non-zero =====
        for (table_name, (num, _)) in self.numerators_and_denominators.iter() {
            // We don't have to check the denominator here because they are checked as they are added to the HashMap
            ensure!(
                *num == F::ZERO,
                "Final numerator was non-zero for lookup table {table_name}, got: {num:?}",
            );
        }

        Ok(())
    }

    pub(crate) fn verify(
        ctx: &VerifierContext<F, PCS>,
        io: &IO<F>,
        proof: Proof<F, PCS>,
    ) -> anyhow::Result<()>
    where
        T: InitTranscript,
    {
        let mut transcript = Self::initialise_transcript(ctx)?;
        let verifier = Verifier::<'_, F, T, PCS>::new(&mut transcript, io);

        // compute padded and unpadded input shapes, using the splitted inputs/output tensors, if any
        let splitted_inputs = if let Some(splitted_inputs) = io.splitted_inputs.as_ref() {
            splitted_inputs
        } else {
            &HashMap::new()
        };

        let (unpadded_input_shapes, padded_input_shapes): (HashMap<_, _>, HashMap<_, _>) = io
            .input
            .iter()
            .enumerate()
            .flat_map(|(input_id, t)| {
                splitted_inputs
                    .get(&input_id)
                    .map(|splitted_ins| {
                        splitted_ins
                            .iter()
                            .enumerate()
                            .map(|(chunk_id, t)| {
                                let input_id = ChunkedInput {
                                    io_id: input_id,
                                    chunk_id,
                                };
                                (input_id.clone(), t)
                            })
                            .collect_vec()
                    })
                    .unwrap_or(vec![(ChunkedInput::from(&input_id), t)])
                    .into_iter()
                    .map(|(input_id, t)| {
                        (
                            (input_id.clone(), t.unpadded_shape().clone()),
                            (input_id, t.shape().clone()),
                        )
                    })
            })
            .unzip();

        let shape_steps = if proof.chunk_proofs.len() == 1 {
            // it's a single chunk, so for simplicity we compute the shape steps directly for the whole model
            ctx.model
                .shape_steps(&unpadded_input_shapes, &padded_input_shapes)?
        } else {
            ctx.model.shape_steps_for_chunks(
                proof
                    .chunk_proofs
                    .iter()
                    .map(|chunk_proof| &chunk_proof.chunk_data.model_chunk),
                &unpadded_input_shapes,
                &padded_input_shapes,
            )?
        };

        // verify chunks are well defined
        ctx.model.check_model_chunking(
            proof
                .chunk_proofs
                .iter()
                .map(|chunk_proof| &chunk_proof.chunk_data.model_chunk),
        )?;

        // verify consistency of commitments among chunks
        let chunk_commitments_by_id = proof
            .chunk_proofs
            .iter()
            .map(|chunk_proof| {
                let chunk_data = &chunk_proof.chunk_data;
                (chunk_data.model_chunk.chunk_id, &chunk_data.commitments)
            })
            .collect();
        proof.chunk_proofs.iter().try_for_each(|chunk_proof| {
            chunk_proof
                .chunk_data
                .model_chunk
                .check_chunk_commitment_consistency::<PCS>(&chunk_commitments_by_id)
        })?;

        // verify chunks
        // there is a distinct proof for model claims, so we need to verify each chunk
        // and then verify the model opening proof
        // first, squeeze the common challenge to initialize the transcript for each cbunk
        let challenge: F = verifier.transcript.challenge_scalar();
        proof.chunk_proofs.into_iter().try_for_each(|proof| {
            // initialise a verifier for the given chunk
            let mut transcript: T = initialize_transcript_for_chunk(challenge);
            let verifier = Verifier::new(&mut transcript, verifier.io);
            verifier.verify_chunk(ctx, proof, &shape_steps)
        })?;

        Ok(())
    }

    fn verify_merge_claims_proof(
        &mut self,
        claims: BTreeMap<PortId, Vec<&Claim<F>>>,
        proof: Option<&MergeClaimsProof<F>>,
    ) -> anyhow::Result<Vec<Claim<F>>> {
        if proof.is_none() {
            ensure!(claims.iter().all(|(_, claims)| claims.len() == 1));
            return Ok(claims
                .into_values()
                .map(|claims| claims[0].clone())
                .collect());
        }
        let proof = proof.unwrap();
        claims
            .into_iter()
            .map(|(port, claims)| {
                if claims.len() == 1 {
                    // there is only one claim, no need to merge anything
                    Ok(claims[0].clone())
                } else {
                    let merge_claim_proof = proof.get_proof(*port).ok_or(anyhow!(
                        "Merge claim proof for output index {} not found",
                        port
                    ))?;
                    merge_claim_proof.verify_proof(self.transcript, &claims)
                }
            })
            .collect()
    }

    pub(crate) fn add_common_claims(
        &mut self,
        node_id: NodeId,
        claims: HashMap<CommitmentId, Claim<F>>,
    ) {
        self.commit_verifier.add_common_claims(
            claims
                .into_iter()
                .map(|(poly_id, claim)| (poly_id, HashMap::from([(node_id, claim)])))
                .collect(),
        )
    }
}

/// Verifies an inference proof given a context, a proof and the input / output of the model.
pub fn verify<F, T: Transcript + InitTranscript, PCS: CommitmentScheme<Field = F>>(
    ctx: &VerifierContext<F, PCS>,
    proof: Proof<F, PCS>,
    io: IO<F>,
) -> anyhow::Result<()>
where
    F: PrimeField,
{
    let span = info_span!(
        "zkml_verify_proof",
        inputs = io.input.len(),
        outputs = io.output.len()
    );
    let _guard = span.enter();
    measure::r("verify_full", || {
        Verifier::<F, T, PCS>::verify(ctx, &io, proof)
    })
}

fn verify_table<F: PrimeField, T: Transcript, PCS: CommitmentScheme>(
    proof: &TableProof<F, PCS>,
    lookup_ctx: &LookupContext,
    chunk_id: ChunkID,
    table_node_id: NodeId,
    witness_verifier: &mut CommitmentVerifier<F, PCS>,
    t: &mut T,
    challenge_storage: &ChallengeStorage<F>,
) -> anyhow::Result<()> {
    // 1. Verify the lookup proof
    let TableProof {
        multiplicity_commit,
        lookup,
    } = proof;
    let instances = lookup_ctx.create_logup_verifier_instances(challenge_storage)?;
    let batch_claim = new_verify_logup_proof_multiple_sizes(lookup, &instances, t)?;

    let poly_evals = batch_claim.poly_evals();
    let point = batch_claim.point();
    let point_len = point.len();

    let (mult_claims, _) = lookup_ctx
        .iter()
        .try_fold((vec![], 0), |(mut acc, skip), tt| {
            let take = tt.num_columns() + 1;
            let nv = tt.table_bit_size();
            let evals = poly_evals
                .iter()
                .skip(skip)
                .take(take)
                .copied()
                .collect::<Vec<F>>();
            let mult_eval = evals[0];

            acc.push(Claim::new(point[point_len - nv..].to_vec(), mult_eval));
            if tt.commit_output_column() {
                witness_verifier.add_table_claim(
                    chunk_id.0.into(),
                    tt,
                    Claim::<F>::new(point[point_len - nv..].to_vec(), evals[take - 1]),
                );
            }

            Result::<(_, _), anyhow::Error>::Ok((acc, skip + take))
        })?;

    let verifier_claims = mult_claims
        .into_iter()
        .zip(multiplicity_commit)
        .map(|(claim, commitment)| VerifierClaim::from((commitment.clone(), claim)))
        .collect::<Vec<VerifierClaim<F, PCS>>>();

    witness_verifier.add_witness_claim(table_node_id, verifier_claims);

    Ok(())
}
