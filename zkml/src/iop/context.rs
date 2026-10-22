use crate::{
    Element, NextPowerOfTwo, Shape,
    graph::{Node, NodeId, NodeInput, NodeOutput, order_by_in_port},
    iop::chunking::{ChunkingStrategy, ModelChunk},
    layers::{
        Layer, LayerCtx,
        provable::{OpInfo, ProveInfo},
    },
    lookup::{context::LookupContext, table::Table},
    model::{Model, ModelCtx, trace::SplittedNodesInfo},
    poly_commit::context::{CommitmentProverCtx, CommitmentVerifierCtx, GlobalCommitmentContext},
    rng_from_env_or_random,
    tensor::CommitmentId,
    to_field,
};
use anyhow::ensure;
use ark_ff::PrimeField;
use dp_crypto::{
    arkyper::{CommitmentScheme, transcript::Transcript},
    poly::dense::DensePolynomial,
};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap},
    iter::successors,
};
use tracing::{debug, info_span, trace};
use utils::Metrics;

/// Fiat–Shamir statement policy fixed by the trusted setup context.
/// Legacy is accepted only by explicit old-artifact decoders, never selected by
/// a proof. Every newly generated setup binds the complete public IO.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofProtocol {
    LegacyV1,
    #[default]
    PublicIoV2,
}

impl ProofProtocol {
    pub(crate) fn write_domain<T: Transcript>(self, transcript: &mut T) {
        if self == Self::PublicIoV2 {
            transcript.append_bytes(b"deep-prove/public-io/v2");
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct ProverContext<'a, F: PrimeField, PCS: CommitmentScheme> {
    pub(crate) protocol: ProofProtocol,
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub model_ctx: ModelCtx<F>,
    /// The commitment context used to generate both model commitments and witness commitments
    pub commitment_ctx: CommitmentProverCtx<'a, F, PCS>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
}

impl<'a, F: PrimeField, PCS: CommitmentScheme<Field = F>> ProverContext<'a, F, PCS> {
    pub fn write_to_transcript<T: Transcript>(&self, t: &mut T) -> anyhow::Result<()> {
        ensure!(
            self.protocol == ProofProtocol::PublicIoV2,
            "Legacy proving is disabled"
        );
        self.protocol.write_domain(t);
        self.commitment_ctx.write_to_transcript(t);
        Ok(())
    }

    pub fn split_in_chunks<S: ChunkingStrategy>(
        &self,
        num_chunks: Option<usize>,
        strategy: S,
    ) -> anyhow::Result<(Vec<ModelChunk>, SplittedNodesInfo)> {
        self.model_ctx
            .split_in_chunks(num_chunks, &strategy, self.next_node_iter())
    }

    pub(crate) fn next_node_iter(&self) -> impl Iterator<Item = NodeId> {
        successors(
            self.commitment_ctx
                .table_node_id()
                .checked_add(1)
                .map(|n| n.into()),
            |n: &NodeId| n.checked_add(1).map(|n| n.into()),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierContext<F: PrimeField, PCS: CommitmentScheme> {
    pub(crate) protocol: ProofProtocol,
    /// Information about each steps of the model. That's the information that the verifier
    /// needs to know from the setup to avoid the prover being able to cheat.
    /// in REVERSED order already since proving goes from last layer to first layer.
    pub model: ModelCtx<F>,
    /// The commitment context used to generate both model commitments and witness commitments
    pub commitment_ctx: CommitmentVerifierCtx<PCS>,
    /// Context holding all the different table types we use in lookups
    pub lookup: LookupContext,
}

impl<F: PrimeField, PCS: CommitmentScheme> VerifierContext<F, PCS> {
    pub fn write_to_transcript<T: Transcript>(&self, t: &mut T) -> anyhow::Result<()> {
        self.protocol.write_domain(t);
        self.commitment_ctx.write_to_transcript(t);
        Ok(())
    }
}

impl Model<Element> {
    /// Helper method employed to build the context data which is independent from the input shape.
    fn build_global_context_data<F: PrimeField, PCS: CommitmentScheme<Field = F>>(
        &self,
        input_shapes: &[Shape],
    ) -> anyhow::Result<(
        ModelCtx<F>,
        GlobalCommitmentContext<'static, F, PCS>,
        LookupContext,
    )> {
        let mut max_poly_len = 1;

        // An accumulator used to carry information over while converting the graph
        let mut ctx_aux = ContextAux {
            last_output_shape: input_shapes.to_vec(),
            model_polys: None,
            max_poly_len,
        };

        // TODO: refactor that management of polys
        let mut model_polys = HashMap::<CommitmentId, DensePolynomial<F>>::new();
        let mut tables = BTreeMap::<Table, Vec<NodeId>>::new();
        // The shape register is filled along the traversal of the graph.
        let mut shapes: HashMap<NodeOutput, Shape> = HashMap::new();
        debug!("Context : layer info generation ...");
        let graph_ctx = self.graph().try_map_forward(|id, node| {
            Ok(match node {
                Node::Inner(layer) => {
                    let inner_metrics = Metrics::new();
                    // Collect the shapes of the tensor fed to this layer
                    ctx_aux.last_output_shape =
                        order_by_in_port(self.graph().incoming_feeds(id).into_iter().map(|feed| {
                            (
                                NodeInput::new(id, feed.target.port),
                                shapes[&feed.source].clone(),
                            )
                        }))
                        .collect();
                    let layer_ctx =
                        compute_layer_ctx::<F>(&mut ctx_aux, &mut model_polys, id, layer)?;
                    if let Some(layer_tables) = layer_ctx.lookup_tables() {
                        layer_tables
                            .into_iter()
                            .for_each(|table_type| tables.entry(table_type).or_default().push(id));
                    }
                    // NOTE: `ctx.last_output_shape` **will** have been modified
                    // by `compute_layer_ctx`, so these shapes are not the one
                    // that have been computed above.
                    shapes.extend(
                        ctx_aux
                            .last_output_shape
                            .iter()
                            .enumerate()
                            .map(|(i, shape)| (NodeOutput::new(id, i), shape.clone())),
                    );

                    max_poly_len = max_poly_len.max(ctx_aux.max_poly_len);
                    debug!(
                        "{} node: {id} ({}), max_poly_len: {max_poly_len}",
                        inner_metrics.to_span(),
                        layer.describe(),
                    );
                    Node::Inner(layer_ctx)
                }
                Node::Input(i) => {
                    // Seed the shape register
                    shapes.insert(
                        NodeOutput::new(self.graph().input_node_id(*i)?, 0),
                        input_shapes[*i].clone(),
                    );
                    Node::Input(*i)
                }
                Node::Output(o) => Node::Output(*o),
            })
        })?;
        // Check to see if we use a lookup table larger than any of the individual polynomials
        tables.keys().for_each(|table| {
            let inner_metrics = Metrics::new();
            let multiplicity_vars = table.table_bit_size();
            max_poly_len = max_poly_len.max(1 << multiplicity_vars);
            debug!(
                "{} table type: {table:?}, max_poly_len: {max_poly_len}",
                inner_metrics.to_span()
            );
        });

        let metrics = Metrics::new();
        debug!("Context: lookup generation ...");
        let lookup_ctx = LookupContext::new(&tables);
        debug!("{} lookup generated.", metrics.to_span());

        let metrics = Metrics::new();
        debug!("Context: commitment generating ...");
        ensure!(
            max_poly_len.is_power_of_two(),
            "Polynomial length is now a power of 2"
        );
        let mut rng = rng_from_env_or_random();
        let commitment_ctx = GlobalCommitmentContext::<F, PCS>::new(
            max_poly_len.trailing_zeros() as usize, // we need to provide the number of variables
            model_polys,
            &lookup_ctx.tables.keys().collect_vec(),
            graph_ctx.next_node_id(),
            &mut rng,
        )?;
        debug!("{} commitment generated.", metrics.to_span());
        Ok((ModelCtx::new(graph_ctx), commitment_ctx, lookup_ctx))
    }

    /// Generate the prover and verifier contexts for the input shape embedded in the model
    pub fn generate_contexts<F: PrimeField, PCS: CommitmentScheme<Field = F>>(
        &self,
    ) -> anyhow::Result<(ProverContext<'static, F, PCS>, VerifierContext<F, PCS>)> {
        // NOTE: here we pad the tensor because trace is unpadded - both the layer and this logic pad the tensor
        // - could be later optimized to only pad once if necessary
        let input_shapes: Vec<Shape> = self
            .input_shapes()
            .into_iter()
            .map(|s| s.next_power_of_two())
            .collect();
        let span = info_span!("zkml_generate_contexts", inputs = input_shapes.len());
        let _guard = span.enter();
        let (step_info, commitment_ctx, lookup) = self.build_global_context_data(&input_shapes)?;
        let (prover_ctx, verifier_ctx) = commitment_ctx.generate_contexts()?;

        let prover_ctx = ProverContext {
            protocol: ProofProtocol::default(),
            model_ctx: step_info.clone(),
            commitment_ctx: prover_ctx,
            lookup: lookup.clone(),
        };

        let verifier_ctx = VerifierContext {
            protocol: ProofProtocol::default(),
            model: step_info,
            commitment_ctx: verifier_ctx,
            lookup,
        };

        Ok((prover_ctx, verifier_ctx))
    }

    /// Generate the prover and verifier contexts for shapes provided as input.
    /// For models with variable input shapes (like LLMs) the maximum possible input shape should be passed here.
    /// Given the maximum possible input shape this method constructs prover and verifier contexts suitable for all
    /// acceptable inputs.
    pub fn generate_contexts_for_input_shapes<F: PrimeField, PCS: CommitmentScheme<Field = F>>(
        &self,
        input_shapes: Vec<Shape>,
    ) -> anyhow::Result<(ProverContext<'_, F, PCS>, VerifierContext<F, PCS>)> {
        let span = info_span!(
            "zkml_generate_contexts_for_input_shapes",
            inputs = input_shapes.len()
        );
        let _guard = span.enter();
        debug!("Building global context");
        let (steps_info, commitment_ctx, lookup) = self.build_global_context_data(&input_shapes)?;
        debug!("Built global context");
        debug!("Building all contexts");
        let (commit_prover_ctx, commit_verifier_ctx) = commitment_ctx.generate_contexts()?;
        let prover_context = ProverContext {
            protocol: ProofProtocol::default(),
            model_ctx: steps_info.clone(),
            commitment_ctx: commit_prover_ctx,
            lookup: lookup.clone(),
        };

        let verifier_context = VerifierContext {
            protocol: ProofProtocol::default(),
            model: steps_info,
            commitment_ctx: commit_verifier_ctx,
            lookup,
        };
        Ok((prover_context, verifier_context))
    }
}
/// Similar to the InferenceStep but only records the input and output shapes
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShapeStep {
    pub unpadded_input_shape: Vec<Shape>,
    pub unpadded_output_shape: Vec<Shape>,
    pub padded_input_shape: Vec<Shape>,
    pub padded_output_shape: Vec<Shape>,
}

impl ShapeStep {
    pub fn new(
        unpadded_input: Vec<Shape>,
        padded_input: Vec<Shape>,
        unpadded_output: Vec<Shape>,
        padded_output: Vec<Shape>,
    ) -> ShapeStep {
        Self {
            unpadded_input_shape: unpadded_input,
            padded_input_shape: padded_input,
            unpadded_output_shape: unpadded_output,
            padded_output_shape: padded_output,
        }
    }
    pub fn next_step(
        last_step: &ShapeStep,
        unpadded_output: Vec<Shape>,
        padded_output: Vec<Shape>,
    ) -> ShapeStep {
        ShapeStep {
            unpadded_input_shape: last_step.unpadded_output_shape.clone(),
            unpadded_output_shape: unpadded_output,
            padded_input_shape: last_step.padded_output_shape.clone(),
            padded_output_shape: padded_output,
        }
    }
}

/// Auxiliary information for the context creation
#[derive(Clone, Debug)]
pub struct ContextAux {
    pub last_output_shape: Vec<Shape>,
    pub model_polys: Option<HashMap<CommitmentId, Vec<Element>>>,
    /// This field is only used in macro layers like MHA
    pub max_poly_len: usize,
}

fn compute_layer_ctx<F: PrimeField>(
    ctx_aux: &mut ContextAux,
    model_polys: &mut HashMap<CommitmentId, DensePolynomial<F>>,
    id: NodeId,
    layer: &Layer<Element>,
) -> anyhow::Result<LayerCtx<F>> {
    trace!(
        "Context : {}-th layer {}info generation ...",
        id,
        layer.describe()
    );
    let (info, mut new_aux) = layer.step_info(ctx_aux.clone())?;
    // Retrieve any model polynomials that need to be committed
    if let Some(aux_model_polys) = &mut new_aux.model_polys {
        for (poly_id, evals) in aux_model_polys.drain() {
            let mle = DensePolynomial::new(to_field::<_, F, _>(evals));
            if let Some(expected_mle) = model_polys.get(&poly_id) {
                // check that the same poly was stored for `poly_id`
                debug_assert!(
                    expected_mle == &mle,
                    "Found different MLE for polynomial {poly_id}"
                );
            } else {
                model_polys.insert(poly_id, mle);
            }
        }
    }
    *ctx_aux = new_aux;
    Ok(info)
}
