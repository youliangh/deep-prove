use crate::{
    Shape,
    graph::{Direction, Edge, Graph, Node, NodeId, PortLink, order_by_in_port},
    iop::{
        chunking::{
            ChunkID, ChunkedIONode, ChunkedInput, ChunkedLayer, ChunkedNode, ChunkedOutput,
            ChunkingStrategy, ModelChunk,
        },
        context::ShapeStep,
    },
    layers::{LayerCtx, provable::Splittable},
    model::trace::SplittedNodesInfo,
};
use anyhow::{Context, anyhow, bail, ensure};
use ark_ff::PrimeField;
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[cfg(test)]
#[path = "context_chunk_guard_tests.rs"]
mod chunk_guard_tests;

pub type ContextGraph<N> = Graph<LayerCtx<N>, usize, usize, ()>;

/// Collection of the proving contexts of all the nodes in the model
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct ModelCtx<F: PrimeField> {
    pub(crate) nodes: ContextGraph<F>,
}

impl<F: PrimeField> ModelCtx<F> {
    pub fn new(nodes: ContextGraph<F>) -> Self {
        Self { nodes }
    }

    pub(crate) fn split_in_chunks<S: ChunkingStrategy>(
        &self,
        num_chunks: Option<usize>,
        strategy: &S,
        next_node_id: impl Iterator<Item = NodeId>,
    ) -> anyhow::Result<(Vec<ModelChunk>, SplittedNodesInfo)> {
        ModelChunk::build_chunks(self, num_chunks, strategy, next_node_id)
    }

    /// Layer witnesses, boundary tensors and lookup-table witnesses share the
    /// opening accumulator. Their IDs must not overwrite another claim set.
    pub(crate) fn check_chunk_commitment_ids<'a>(
        &self,
        chunks: impl Iterator<Item = &'a ModelChunk>,
        table_node_id: NodeId,
    ) -> anyhow::Result<()> {
        let chunks = chunks.collect::<Vec<_>>();
        let node_ids = chunks
            .iter()
            .flat_map(|chunk| chunk.subgraph.nodes().map(|(id, _)| *id))
            .collect::<HashSet<_>>();
        ensure!(
            !node_ids.contains(&table_node_id),
            "Chunk node ID collides with the reserved lookup-table witness ID"
        );
        let mut boundary_ids = HashMap::new();
        for chunk in chunks {
            for edge_id in chunk
                .incoming_edges
                .keys()
                .chain(chunk.outgoing_edges.keys())
            {
                let edge = chunk.edge(edge_id)?;
                for link in edge.ports().iter() {
                    let port = crate::graph::NodeOutput::new(edge.source(), link.source_port);
                    let id = NodeId::from(ModelChunk::compute_commitment_id(port));
                    ensure!(
                        id != table_node_id && !node_ids.contains(&id),
                        "Boundary commitment ID collides with a layer or lookup-table witness ID"
                    );
                    if let Some(previous) = boundary_ids.insert(id, port) {
                        ensure!(
                            previous == port,
                            "Distinct boundary tensors have colliding commitment IDs"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// Chunk metadata is supplied by the prover. In particular, a synthetic
    /// split/merge node must never replace the trusted operation at an existing
    /// node ID, and an unsplittable operation must retain its full input.
    fn check_chunk_node_identities(
        &self,
        chunks: &BTreeMap<crate::iop::chunking::ChunkID, &ModelChunk>,
    ) -> anyhow::Result<()> {
        let mut replacement_origins = HashMap::new();
        let mut replacement_numbers = BTreeMap::<NodeId, BTreeSet<usize>>::new();
        for chunk in chunks.values() {
            for (&original_id, replacement_ids) in &chunk.replaced_nodes {
                ensure!(
                    !replacement_ids.is_empty(),
                    "Empty replacement list for original node {original_id}"
                );
                let original = self.nodes.node(original_id).ok_or(anyhow!(
                    "Replacement refers to unknown original node {original_id}"
                ))?;
                if let Node::Inner(layer) = original {
                    ensure!(
                        layer.is_splittable(),
                        "Original node {original_id} does not support horizontal splitting"
                    );
                }
                for &replacement_id in replacement_ids {
                    ensure!(
                        self.nodes.node(replacement_id).is_none(),
                        "Replacement node {replacement_id} collides with a trusted original node"
                    );
                    ensure!(
                        replacement_origins
                            .insert(replacement_id, original_id)
                            .is_none(),
                        "Replacement node {replacement_id} is listed more than once"
                    );
                    let replacement = chunk.subgraph.node(replacement_id).ok_or(anyhow!(
                        "Replacement node {replacement_id} is absent from its chunk"
                    ))?;
                    let number = match (original, replacement) {
                        (Node::Inner(_), Node::Inner(ChunkedNode::ChunkedLayer(layer))) => {
                            ensure!(
                                layer.original_node_id == original_id,
                                "Replacement node {replacement_id} refers to a different original operation"
                            );
                            layer.chunk_number
                        }
                        (Node::Input(expected), Node::Input(ChunkedIONode::Chunked(io))) => {
                            ensure!(
                                io.io_id == *expected,
                                "Replacement node {replacement_id} refers to a different public tensor"
                            );
                            io.chunk_id
                        }
                        (Node::Output(expected), Node::Output(ChunkedIONode::Chunked(io))) => {
                            ensure!(
                                io.io_id == *expected,
                                "Replacement node {replacement_id} refers to a different public tensor"
                            );
                            io.chunk_id
                        }
                        _ => bail!(
                            "Replacement node {replacement_id} changes the original node kind"
                        ),
                    };
                    ensure!(
                        replacement_numbers
                            .entry(original_id)
                            .or_default()
                            .insert(number),
                        "Duplicate horizontal chunk {number} for original node {original_id}"
                    );
                }
            }
        }
        for (original_id, numbers) in replacement_numbers {
            ensure!(
                numbers.iter().copied().eq(0..numbers.len()),
                "Horizontal chunks for original node {original_id} are not consecutive from zero"
            );
        }
        for chunk in chunks.values() {
            for (&node_id, node) in chunk.subgraph.nodes() {
                if let Some(original) = self.nodes.node(node_id) {
                    let same_kind = match (original, node) {
                        (Node::Inner(_), Node::Inner(ChunkedNode::OriginalNode(()))) => true,
                        (
                            Node::Input(expected),
                            Node::Input(ChunkedIONode::OriginalNode(actual)),
                        )
                        | (
                            Node::Output(expected),
                            Node::Output(ChunkedIONode::OriginalNode(actual)),
                        ) => expected == actual,
                        _ => false,
                    };
                    ensure!(
                        same_kind,
                        "Chunk node {node_id} changes the trusted original node kind"
                    );
                } else {
                    match node {
                        Node::Inner(
                            ChunkedNode::SplitLayer(_) | ChunkedNode::RecombinationLayer(_),
                        ) => {}
                        Node::Inner(ChunkedNode::ChunkedLayer(_))
                        | Node::Input(ChunkedIONode::Chunked(_))
                        | Node::Output(ChunkedIONode::Chunked(_)) => ensure!(
                            replacement_origins.contains_key(&node_id),
                            "Synthetic node {node_id} is not a declared replacement"
                        ),
                        _ => bail!("Unknown original node {node_id} in chunk graph"),
                    }
                }
            }
        }
        Ok(())
    }

    fn check_model_graph_edges_from_chunks(
        &self,
        chunks: &BTreeMap<ChunkID, &ModelChunk>,
    ) -> anyhow::Result<()> {
        let mut chunked_nodes_map = HashMap::new();
        let mut split_layers = HashSet::new();
        let mut recombination_layers = HashSet::new();
        let chunk_for_node = ModelChunk::node_to_chunk_map(chunks.values().copied());
        for chunk in chunks.values() {
            for (node_id, node) in chunk.subgraph.nodes() {
                if let Some(inner_node) = node.as_inner() {
                    if matches!(inner_node, ChunkedNode::SplitLayer(_)) {
                        split_layers.insert(node_id);
                    } else if matches!(inner_node, ChunkedNode::RecombinationLayer(_)) {
                        recombination_layers.insert(node_id);
                    }
                }
            }
            chunk.replaced_nodes.iter().try_for_each(|(original_node, new_nodes)|
                new_nodes.iter().try_for_each(|&new_node| {
                    // check that `new_node` is indeed a `ChunkedLayer` mapped to `original_node`
                    let node = chunk.subgraph.node(new_node)
                        .ok_or(
                            anyhow!("Chunked node {new_node} not found in chunk {}", chunk.chunk_id)
                        )?;
                    match node {
                        Node::Input(input_node) => {
                            let ChunkedIONode::Chunked(ChunkedInput {
                                io_id: input_id,
                                ..
                            }) = input_node else {
                                bail!("Input node {new_node} in chunk {} is not chunked", chunk.chunk_id)
                            };
                            let original_input_node = self.nodes.input_node_id(*input_id)?;
                            ensure!(original_input_node == *original_node,
                                "Chunked input node {new_node} in chunk {} does not correspond to original node {original_node}",
                                chunk.chunk_id
                            );
                        },
                        Node::Inner(inner_node) => {
                            ensure!(
                                matches!(inner_node, ChunkedNode::ChunkedLayer(
                                    ChunkedLayer {
                                        original_node_id: on,
                                        ..
                                    }
                                ) if on == original_node),
                                "Chunked node {new_node} in chunk {} does not correspond to original node {original_node}",
                                chunk.chunk_id
                            );
                        },
                        Node::Output(output_node) => {
                            let ChunkedIONode::Chunked(ChunkedOutput {
                                io_id: output_id,
                                ..
                            }) = output_node else {
                                bail!("Output node {new_node} in chunk {} is not chunked", chunk.chunk_id)
                            };
                            let original_input_node = self.nodes.output_node_ids()[*output_id];
                            ensure!(original_input_node == *original_node,
                                "Chunked output node {new_node} in chunk {} does not correspond to original node {original_node}",
                                chunk.chunk_id
                            );
                        },
                    };
                    chunked_nodes_map.insert(new_node, *original_node);
                    Ok(())
                })
            )?;
        }

        let mut graph_edges = BTreeMap::new();
        for chunk in chunks.values() {
            for (&node_id, node) in chunk.subgraph.nodes() {
                let mut graph_edges_for_node = BTreeMap::new();
                for (edge_id, edge) in chunk.subgraph.neighbors(node_id, Direction::Any) {
                    let source = edge.source();
                    let target = edge.target();
                    let is_source_in_original_graph = self.nodes.node(source).is_some();
                    let is_target_in_original_graph = self.nodes.node(target).is_some();
                    if is_source_in_original_graph && is_target_in_original_graph {
                        // this edge is between nodes found in the original graph, so it must be present
                        // also in the original graph
                        graph_edges_for_node.insert(*edge_id, edge.clone());
                        continue;
                    }
                    if is_source_in_original_graph {
                        // this edge is an original node of the graph linked to a chunked node, so we need to
                        // check that it is linked to a SplitLayer, and find to what chunked node the SplitLayer
                        // is linked to
                        ensure!(split_layers.contains(&target));
                        let split_layer_chunk = chunk_for_node
                            .get(&target)
                            .ok_or(anyhow!("Split layer {target} not found in any chunk"))?;
                        let ChunkedNode::SplitLayer(split_layer) = chunks[split_layer_chunk]
                            .subgraph
                            .node(target)
                            .unwrap_or_else(|| {
                                panic!("Split layer {target} not found in its chunk")
                            })
                            .as_inner()
                            .unwrap_or_else(|| panic!("Split layer {target} is not an inner node"))
                        else {
                            anyhow::bail!(
                                "Node {target} in chunk {} is not a SplitLayer",
                                chunk.chunk_id
                            );
                        };
                        let num_chunks = {
                            ensure!(split_layer.num_chunks.iter().all_equal());
                            split_layer.num_chunks[0]
                        };
                        let split_layer_out_feeds =
                            chunks[split_layer_chunk].subgraph.outgoing_feeds(target);
                        let out_ports = edge
                            .ports()
                            .iter()
                            .flat_map(|port| {
                                // input port i of `SplitLayer` is mapped to outputs `num_chunks*...num_chunks*(i+1)` of the `SplitLayer`,
                                // so we need to find the chunked node linked to one of these output ports
                                let input_port: usize = port.target_port.into();
                                num_chunks * input_port..num_chunks * (input_port + 1)
                            })
                            .collect::<HashSet<_>>();
                        let mut original_node = None;
                        for feed in split_layer_out_feeds {
                            if out_ports.contains(&feed.source.port.into()) {
                                let chunked_node = feed.target.node_id;
                                let original_node_id =
                                    chunked_nodes_map.get(&chunked_node).ok_or(anyhow!(
                                        "Chunked node {chunked_node} not found in chunk {}",
                                        split_layer_chunk
                                    ))?;
                                ensure!(
                                    original_node.is_none()
                                        || original_node == Some(*original_node_id),
                                    "Split layer {target} in chunk {} has output feeds linked to different original nodes",
                                    split_layer_chunk,
                                );
                                original_node = Some(*original_node_id);
                            }
                        }
                        let original_node = original_node.ok_or(anyhow!(
                            "Original node linked to Split layer {target} not found"
                        ))?;
                        // build the corresponding edge in the original graph
                        let new_edge = Edge::new(
                            edge.source(),
                            original_node,
                            edge.ports().clone(),
                            edge.weight,
                        );
                        // find the edge in the original model
                        let (edge_id, _) = self.nodes.edges().find(|(_, edge)|
                            *edge == &new_edge
                        ).ok_or(
                            anyhow!("Edge from {source} to {original_node} with ports {:?} not found in original graph", edge.ports())
                        )?;
                        graph_edges_for_node.insert(*edge_id, new_edge);
                        continue;
                    }
                    if is_target_in_original_graph {
                        // this edge is an original node of the graph with inputs coming from a chunked node, so we need to
                        // check that it is linked to a RecombinationLayer, and find to what chunked node the RecombinationLayer
                        // is linked to
                        ensure!(recombination_layers.contains(&source));
                        let rec_layer_chunk = chunk_for_node.get(&source).ok_or(anyhow!(
                            "Recombination layer {source} not found in any chunk"
                        ))?;
                        let ChunkedNode::RecombinationLayer(rec_layer) = chunks[rec_layer_chunk]
                            .subgraph
                            .node(source)
                            .unwrap_or_else(|| {
                                panic!("Recombination layer {source} not found in its chunk")
                            })
                            .as_inner()
                            .unwrap_or_else(|| {
                                panic!("Recombination layer {source} is not an inner node")
                            })
                        else {
                            anyhow::bail!(
                                "Node {source} in chunk {} is not a RecombinationLayer",
                                rec_layer_chunk
                            );
                        };
                        let num_chunks = {
                            ensure!(rec_layer.num_chunks.iter().all_equal());
                            rec_layer.num_chunks[0]
                        };
                        let rec_layer_in_feeds =
                            chunks[rec_layer_chunk].subgraph.incoming_feeds(source);
                        let in_ports = edge
                            .ports()
                            .iter()
                            .flat_map(|port| {
                                // output port i of `RecombinationLayer` is mapped to inputs `num_chunks*...num_chunks*(i+1)`
                                // of the `RecombinationLayer`, so we need to find the chunked nodes linked to these input ports
                                let out_port: usize = port.source_port.into();
                                num_chunks * out_port..num_chunks * (out_port + 1)
                            })
                            .collect::<HashSet<_>>();
                        let mut original_node = None;
                        for feed in rec_layer_in_feeds {
                            if in_ports.contains(&feed.target.port.into()) {
                                let chunked_node = feed.source.node_id;
                                let original_node_id =
                                    chunked_nodes_map.get(&chunked_node).ok_or(anyhow!(
                                        "Chunked node {chunked_node} not found in chunk {}",
                                        rec_layer_chunk
                                    ))?;
                                ensure!(
                                    original_node.is_none()
                                        || original_node == Some(*original_node_id),
                                    "Recombination layer {source} in chunk {} has input feeds linked to different original nodes",
                                    rec_layer_chunk
                                );
                                original_node = Some(*original_node_id);
                            }
                        }
                        let original_node = original_node.ok_or(anyhow!(
                            "Original node linked to Recombination layer {source} not found"
                        ))?;
                        // build the corresponding edge in the original graph
                        let new_edge = Edge::new(
                            original_node,
                            edge.target(),
                            edge.ports().clone(),
                            edge.weight,
                        );
                        // find the edge in the original model
                        let (edge_id, _) = self.nodes.edges().find(|(_, edge)|
                            *edge == &new_edge
                        ).ok_or(
                            anyhow!("Edge from {original_node} to {target} with ports {:?} not found in original graph", edge.ports())
                        )?;
                        graph_edges_for_node.insert(*edge_id, new_edge);
                        continue;
                    }
                    let source_node = if source == node_id {
                        node
                    } else {
                        let source_node_chunk = chunk_for_node
                            .get(&source)
                            .ok_or(anyhow!("Node {source} not found in any chunk"))?;
                        chunks[source_node_chunk]
                            .subgraph
                            .node(source)
                            .ok_or(anyhow!(
                                "Node {source} not found in chunk {}",
                                source_node_chunk
                            ))?
                    };
                    let source_as_chunked_layer = source_node.as_inner().and_then(|inner_node| {
                        if let ChunkedNode::ChunkedLayer(cl) = inner_node {
                            Some(cl)
                        } else {
                            None
                        }
                    });
                    let target_node = if target == node_id {
                        node
                    } else {
                        let target_node_chunk = chunk_for_node
                            .get(&target)
                            .ok_or(anyhow!("Node {target} not found in any chunk"))?;
                        chunks[target_node_chunk]
                            .subgraph
                            .node(target)
                            .ok_or(anyhow!(
                                "Node {target} not found in chunk {}",
                                target_node_chunk
                            ))?
                    };
                    let target_as_chunked_layer = target_node.as_inner().and_then(|inner_node| {
                        if let ChunkedNode::ChunkedLayer(cl) = inner_node {
                            Some(cl)
                        } else {
                            None
                        }
                    });
                    if let Some(source_chunked_layer) = source_as_chunked_layer
                        && let Some(target_chunked_layer) = target_as_chunked_layer
                    {
                        // this edge is between two chunked nodes, so we must find the corresponding original nodes
                        // in the original graph
                        let source_original_node = source_chunked_layer.original_node_id;
                        let target_original_node = target_chunked_layer.original_node_id;
                        let new_edge = Edge::new(
                            source_original_node,
                            target_original_node,
                            edge.ports().clone(),
                            edge.weight,
                        );
                        // find the edge in the original model
                        let (edge_id, _) = self.nodes.edges().find(|(_, edge)|
                            *edge == &new_edge
                        ).ok_or(
                            anyhow!("Edge from {source_original_node} to {target_original_node} with ports {:?} not found in original graph", edge.ports())
                        )?;
                        graph_edges_for_node.insert(*edge_id, new_edge);
                        continue;
                    }
                    if let Some(source_chunked_layer) = source_as_chunked_layer {
                        let source_original_node = source_chunked_layer.original_node_id;
                        // this edge is either between a chunked node and a `RecombinationLayer`, or between a chunked node and a chunked output node
                        if !recombination_layers.contains(&target) {
                            // The target of this edge must be chunked output node
                            let ChunkedIONode::Chunked(chunked_output_node) = target_node
                                .as_output()
                                .ok_or(anyhow!("Node {target} is not a model output node"))?
                            else {
                                bail!("Node {target} is not a chunked output node")
                            };
                            let target_original_node =
                                self.nodes.output_node_ids()[chunked_output_node.io_id];
                            let new_edge = Edge::new(
                                source_original_node,
                                target_original_node,
                                edge.ports().clone(),
                                edge.weight,
                            );
                            // find the edge in the original model
                            let (edge_id, _) = self.nodes.edges().find(|(_, edge)|
                                *edge == &new_edge
                            ).ok_or(
                                anyhow!("Edge from {source_original_node} to {target_original_node} with ports {:?} not found in original graph", edge.ports())
                            )?;
                            graph_edges_for_node.insert(*edge_id, new_edge);
                            continue;
                        }
                        // The target of this edge must be a `RecombinationLayer`, so we need to find the node linked
                        // to the outputs of the `RecombinationLayer` to buld the corresponding edge
                        let rec_layer_chunk = chunk_for_node.get(&target).ok_or(anyhow!(
                            "Recombination layer {target} not found in any chunk"
                        ))?;
                        let ChunkedNode::RecombinationLayer(rec_layer) = chunks[rec_layer_chunk]
                            .subgraph
                            .node(target)
                            .unwrap_or_else(|| {
                                panic!("Recombination layer {target} not found in its chunk")
                            })
                            .as_inner()
                            .unwrap_or_else(|| {
                                panic!("Recombination layer {target} is not an inner node")
                            })
                        else {
                            anyhow::bail!(
                                "Node {target} in chunk {} is not a RecombinationLayer",
                                rec_layer_chunk
                            );
                        };
                        let num_chunks = {
                            ensure!(rec_layer.num_chunks.iter().all_equal());
                            rec_layer.num_chunks[0]
                        };
                        let rec_layer_out_feeds =
                            chunks[rec_layer_chunk].subgraph.outgoing_feeds(target);
                        let out_ports = edge
                            .ports()
                            .iter()
                            .map(|port| {
                                // input port i of `RecombinationLayer` is mapped to output `i/num_chunks`
                                // of the `RecombinationLayer`, so we need to find the chunked nodes linked to these output ports
                                let in_port: usize = port.target_port.into();
                                (in_port / num_chunks, port.source_port)
                            })
                            .collect::<HashMap<_, _>>();
                        let mut new_edges_by_target_node = HashMap::new();
                        for feed in rec_layer_out_feeds {
                            if let Some(source_port) = out_ports.get(&feed.source.port.into()) {
                                let target_node = feed.target.node_id;
                                // check that target node is found in the original graph
                                ensure!(
                                    self.nodes.node(target_node).is_some(),
                                    "Target node {target_node} of RecombinationLayer {target} not found in original graph"
                                );
                                new_edges_by_target_node
                                    .entry(target_node)
                                    .or_insert(vec![])
                                    .push(PortLink::new(*source_port, feed.target.port));
                            }
                        }
                        for (target_node, ports) in new_edges_by_target_node {
                            // build the corresponding edge in the original graph
                            let new_edge =
                                Edge::new(source_original_node, target_node, ports, edge.weight);
                            // find the edge in the original model
                            let (edge_id, _) = self.nodes.edges().find(|(_, edge)|
                                *edge == &new_edge
                            ).ok_or(
                                anyhow!("Edge from {source_original_node} to {target_node} not found in original graph")
                            )?;
                            graph_edges_for_node.insert(*edge_id, new_edge);
                        }
                        continue;
                    }
                    if let Some(target_chunked_layer) = target_as_chunked_layer {
                        let target_original_node = target_chunked_layer.original_node_id;
                        // this edge is either between a `SplitLayer` and a chunked node, or between a chunked input node
                        // and a chunked node
                        if !split_layers.contains(&source) {
                            // the source of this edge must be a chunked input node
                            let ChunkedIONode::Chunked(chunked_input_node) = source_node
                                .as_input()
                                .ok_or(anyhow!("Node {source} is not a model input node"))?
                            else {
                                bail!("Node {source} is not a chunked input node")
                            };
                            let source_original_node =
                                self.nodes.input_node_id(chunked_input_node.io_id)?;
                            let new_edge = Edge::new(
                                source_original_node,
                                target_original_node,
                                edge.ports().clone(),
                                edge.weight,
                            );
                            // find the edge in the original model
                            let (edge_id, _) = self.nodes.edges().find(|(_, edge)|
                                *edge == &new_edge
                            ).ok_or(
                                anyhow!("Edge from {source_original_node} to {target_original_node} with ports {:?} not found in original graph", edge.ports())
                            )?;
                            graph_edges_for_node.insert(*edge_id, new_edge);
                            continue;
                        }
                        // the source of this edge is a `SplitLayer, so we need to find the node linked
                        // to the inputs of the `SplitLayer` to build the corresponding edge
                        let split_layer_chunk = chunk_for_node
                            .get(&source)
                            .ok_or(anyhow!("Split layer {source} not found in any chunk"))?;
                        let ChunkedNode::SplitLayer(split_layer) = chunks[split_layer_chunk]
                            .subgraph
                            .node(source)
                            .unwrap_or_else(|| {
                                panic!("Split layer {source} not found in its chunk")
                            })
                            .as_inner()
                            .unwrap_or_else(|| panic!("Split layer {source} is not an inner node"))
                        else {
                            anyhow::bail!(
                                "Node {source} in chunk {} is not a SplitLayer",
                                split_layer_chunk
                            );
                        };
                        let num_chunks = {
                            ensure!(split_layer.num_chunks.iter().all_equal());
                            split_layer.num_chunks[0]
                        };
                        let split_layer_in_feeds =
                            chunks[split_layer_chunk].subgraph.incoming_feeds(source);
                        let in_ports = edge
                            .ports()
                            .iter()
                            .map(|port| {
                                // output port i of `SplitLayer` is mapped to input `i/num_chunks`
                                // of the `SplitLayer`, so we need to find the chunked nodes linked to these input ports
                                let out_port: usize = port.source_port.into();
                                (out_port / num_chunks, port.target_port)
                            })
                            .collect::<HashMap<_, _>>();
                        let mut new_edges_by_source = HashMap::new();
                        for feed in split_layer_in_feeds {
                            if let Some(target_port) = in_ports.get(&feed.target.port.into()) {
                                let source_node = feed.source.node_id;
                                // check that source node is found in the original graph
                                ensure!(
                                    self.nodes.node(source_node).is_some(),
                                    "Source node {source_node} of SplitLayer {source} not found in original graph"
                                );
                                new_edges_by_source
                                    .entry(source_node)
                                    .or_insert(vec![])
                                    .push(PortLink::new(feed.source.port, *target_port));
                            }
                        }
                        for (source_node, ports) in new_edges_by_source {
                            // build the corresponding edge in the original graph
                            let new_edge =
                                Edge::new(source_node, target_original_node, ports, edge.weight);
                            // find the edge in the original model
                            let (edge_id, _) = self.nodes.edges().find(|(_, edge)|
                                *edge == &new_edge
                            ).ok_or(
                                anyhow!("Edge from {source_node} to {target_original_node} with ports not found in original graph")
                            )?;
                            graph_edges_for_node.insert(*edge_id, new_edge);
                        }
                        continue;
                    }
                    // the edges is either between a SplitLayer and a chunked node, or between a RecombinationLayer and a chunked node,
                    // so we do nothing since it doesn't correspond to any edge in the original graph that we didn't cover in the previous cases
                    // ToDo: maybe we need anyhow to check that it corresponds to a valid node in the graph?
                }
                // get edges for this node from the original graph, if it exists, of from the corresponding original node
                let original_node = if self.nodes.node(node_id).is_some() {
                    Some(node_id)
                } else {
                    chunked_nodes_map.get(&node_id).copied()
                };
                // if `original_node` is `None`, then this node is either a `SplitLayer` or a `RecombinationLayer`,
                // so we don't need to check its edges since they don't correspond to any edge in the original graph;
                // otherwise, we need to check that the edges for `original_node` in the original graph are the same as `graph_edges_for_node`
                if let Some(original_node) = original_node {
                    let edges_in_model = self
                        .nodes
                        .neighbors(original_node, Direction::Any)
                        .map(|(edge_id, edge)| (*edge_id, edge.clone()))
                        .collect::<BTreeMap<_, _>>();
                    ensure!(
                        graph_edges_for_node == edges_in_model,
                        "The edges for node {node_id} in the chunk do not match the edges for original node {original_node} in the model graph: {:?} vs {:?}",
                        graph_edges_for_node,
                        edges_in_model,
                    );
                }
                graph_edges.extend(graph_edges_for_node);
            }
        }

        graph_edges
            .into_iter()
            .zip_eq(self.nodes.edges())
            .try_for_each(|(chunked_graph_edge, original_edge)| {
                ensure!(
                    chunked_graph_edge.0 == *original_edge.0,
                    "Edge id in chunked graph does not match original edge id: {:?} vs {:?}",
                    chunked_graph_edge.0,
                    original_edge.0
                );
                ensure!(
                    chunked_graph_edge.1 == *original_edge.1,
                    "Edge data in chunked graph does not match original edge data: {:?} vs {:?}",
                    chunked_graph_edge.1,
                    original_edge.1
                );
                Ok(())
            })?;

        Ok(())
    }

    /// This method allows to check that the given chunks are consistent with the model graph in `self`.
    /// In a nutshell, consistency means that:
    /// - the chunks form a partition of the model graph
    /// - the nodes in all the chunks have the same edges as in the original model graph
    /// - the incoming and outgoing edges of the chunk are coherent with the graph structure
    /// - the incoming and outgoing edges are correctly linked to the corresponding chunks
    ///
    /// This method is useful for the verifier to check that the prover is not providing chunks
    /// for a different model than `self`, and that chunks are built coherently with the model.
    /// Note that this method does not guarantee to the verifier that these were the actual chunks
    /// used by the prover; this will be prevented by forcing the prover to add some data about
    /// the chunks in the transcript.
    pub(crate) fn check_model_chunking<'a>(
        &self,
        chunks: impl Iterator<Item = &'a ModelChunk>,
    ) -> anyhow::Result<()> {
        // collect each chunk in a map, altogether with its id
        let mut indexed_chunks = BTreeMap::new();
        for chunk in chunks {
            ensure!(
                indexed_chunks.insert(chunk.chunk_id, chunk).is_none(),
                "Duplicate model chunk ID {}",
                chunk.chunk_id
            );
        }
        let chunks = indexed_chunks;
        self.check_chunk_node_identities(&chunks)?;
        // check that the chunks are a partition of the model graph
        // first, compute the set of nodes found in all the chunks
        let nodes_in_chunks =
            chunks
                .values()
                .try_fold(BTreeSet::new(), |mut node_ids, chunk| {
                    chunk.subgraph.nodes().try_for_each(|(node_id, node)| {
                        // filter out Split and Recombination layers
                        if let Some(inner_node) = node.as_inner()
                            && (matches!(inner_node, ChunkedNode::SplitLayer(_))
                                || matches!(inner_node, ChunkedNode::RecombinationLayer(_)))
                        {
                            return Ok(());
                        }
                        let not_yet_inserted = node_ids.insert(*node_id);
                        if node.is_inner() {
                            // we need to ensure that each inner node is found only once across all the chunks
                            ensure!(
                                not_yet_inserted,
                                "Node {node_id} found multiple times in model chunks"
                            );
                        }
                        Ok(())
                    })?;
                    anyhow::Ok(node_ids)
                })?;
        // build a map of nodes replaced with chunked layers
        let replaced_nodes = chunks
            .values()
            .fold(HashMap::new(), |mut replaced_nodes, chunk| {
                for (original_node_id, chunked_node_ids) in &chunk.replaced_nodes {
                    replaced_nodes
                        .entry(*original_node_id)
                        .or_insert(vec![])
                        .extend(chunked_node_ids.iter().copied());
                }
                replaced_nodes
            });

        // check that the nodes in the model are the same as `node_in_chunks`
        let model_node_ids = self
            .nodes
            .nodes()
            .flat_map(|(node_id, _)| {
                if let Some(chunked_node_ids) = replaced_nodes.get(node_id) {
                    chunked_node_ids.to_vec()
                } else {
                    vec![*node_id]
                }
            })
            .collect::<BTreeSet<_>>();
        ensure!(
            nodes_in_chunks == model_node_ids,
            "The chunks do not form a partition of the model graph"
        );
        // check that all edges found in the chunks are the same as in the model graph
        // let edges_in_chunks = chunks
        // .values()
        // .flat_map(|chunk| chunk.subgraph.edges())
        // .collect::<BTreeMap<_, _>>();
        // let edges_in_model = self.nodes.edges().collect::<BTreeMap<_, _>>();
        // ensure!(
        // edges_in_chunks == edges_in_model,
        // "The edges in the chunks do not match the model graph"
        // );
        // for each node in each chunk, check that all the edges that don't involve a node in `replaced_nodes`
        // are the same as in the model graph
        // chunks.values().try_for_each(|chunk| {
        // chunk.subgraph.nodes().try_for_each(|(node_id, _)| {
        // let edges_in_chunk = chunk
        // .subgraph
        // .neighbors(*node_id, Direction::Any)
        // .collect::<BTreeMap<_, _>>();
        // let edges_in_model = self
        // .nodes
        // .neighbors(*node_id, Direction::Any)
        // .filter(|(_, edge)|
        // we filter out the edges that involve a node in `replaced_nodes`,
        // as these edges will not be found in the chunk, but we'll check consistency
        // later on
        // !replaced_nodes.contains_key(&edge.source()) &&
        // !replaced_nodes.contains_key(&edge.target())
        // )
        // .collect::<BTreeMap<_, _>>();
        // ensure!(
        // edges_in_chunk == edges_in_model,
        // "The edges for node {node_id} in the chunk do not match the model graph"
        // );
        // Ok(())
        // })
        // })?;
        self.check_model_graph_edges_from_chunks(&chunks)?;

        // ToDo: check edges involving replaced nodes

        // check that the incoming and outgoing edges of each chunk are correct and grouped accordingly
        let chunk_for_node = ModelChunk::node_to_chunk_map(chunks.values().copied());

        chunks.values().try_for_each(|chunk| {
            let built_incoming_edges = chunk.build_incoming_boundary_edges(&chunk_for_node)?;
            ensure!(
                built_incoming_edges == chunk.incoming_edges,
                "The incoming edges for chunk {} are not correctly grouped",
                chunk.chunk_id
            );
            let built_outgoing_edges = chunk.build_outgoing_boundary_edges(&chunk_for_node)?;
            ensure!(
                built_outgoing_edges == chunk.outgoing_edges,
                "The outgoing edges for chunk {} are not correctly grouped",
                chunk.chunk_id
            );
            Ok(())
        })?;

        // check that incoming and outgoing edges among linked chunks are consistent
        ModelChunk::check_boundary_edges_consistency(&chunks)
    }

    /// Computes the shape step for each node in the model, so each layer knows
    /// the expected input and output shape to correctly verify the proof.
    pub fn shape_steps(
        &self,
        unpadded_input_shapes: &HashMap<ChunkedInput, Shape>,
        padded_input_shapes: &HashMap<ChunkedInput, Shape>,
    ) -> anyhow::Result<HashMap<NodeId, ShapeStep>> {
        let mut shapes = HashMap::<NodeId, ShapeStep>::new();
        shape_steps_for_graph(
            &self.nodes,
            unpadded_input_shapes,
            padded_input_shapes,
            &mut shapes,
            |_, layer, unpad, pad| layer.shape_step(unpad, pad),
        )?;
        Ok(shapes)
    }

    pub(crate) fn shape_steps_for_chunks<'a>(
        &self,
        chunks: impl Iterator<Item = &'a ModelChunk>,
        unpadded_input_shapes: &HashMap<ChunkedInput, Shape>,
        padded_input_shapes: &HashMap<ChunkedInput, Shape>,
    ) -> anyhow::Result<HashMap<NodeId, ShapeStep>> {
        let chunks = ModelChunk::visit_chunks_forward(chunks)?;
        let mut shapes = HashMap::new();
        for chunk in chunks {
            chunk.shape_steps_for_chunk(
                unpadded_input_shapes,
                padded_input_shapes,
                &mut shapes,
                self,
            )?;
        }
        Ok(shapes)
    }
}

pub(crate) fn shape_steps_for_graph<
    'a,
    T,
    I: std::fmt::Debug,
    O,
    F: FnMut(NodeId, &T, &[Shape], &[Shape]) -> anyhow::Result<ShapeStep>,
>(
    graph: &'a Graph<T, I, O, ()>,
    unpadded_input_shapes: &HashMap<ChunkedInput, Shape>,
    padded_input_shapes: &HashMap<ChunkedInput, Shape>,
    shapes: &mut HashMap<NodeId, ShapeStep>,
    mut update_shape_step: F,
) -> anyhow::Result<()>
where
    ChunkedInput: From<&'a I>,
{
    for (node_id, node) in graph.forward_iter() {
        match node {
            Node::Inner(layer) => {
                let (un, pad): (Vec<Shape>, Vec<Shape>) =
                    order_by_in_port(graph.incomings(node_id).flat_map(|(_, e)| e.feeds()).map(
                        |feed| {
                            // fetch the input shapes for this node in
                            // the register, that will have been
                            // recursively filled with all the preceding
                            // nodes output shapes as the graph is
                            // traversed.
                            let ShapeStep {
                                unpadded_output_shape,
                                padded_output_shape,
                                ..
                            } = shapes
                                .get(&feed.source.node_id)
                                .with_context(|| {
                                    format!("fetching shape step for {:?}", feed.source)
                                })
                                .unwrap();
                            (
                                feed.target,
                                (
                                    unpadded_output_shape[feed.source.port].clone(),
                                    padded_output_shape[feed.source.port].clone(),
                                ),
                            )
                        },
                    ))
                    .unzip();
                shapes.insert(node_id, update_shape_step(node_id, layer, &un, &pad)?);
            }
            Node::Input(i) => {
                shapes.insert(
                    node_id,
                    ShapeStep {
                        unpadded_input_shape: vec![],
                        unpadded_output_shape: vec![unpadded_input_shapes[&i.into()].clone()],
                        padded_input_shape: vec![],
                        padded_output_shape: vec![padded_input_shapes[&i.into()].clone()],
                    },
                );
            }
            Node::Output(_) => {}
        }
    }
    Ok(())
}
