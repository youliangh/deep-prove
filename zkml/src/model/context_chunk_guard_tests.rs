use super::*;
use crate::{
    ScalingFactor,
    iop::chunking::DefaultChunkingStrategy,
    layers::{
        requant::{Requant, RequantCtx},
        split::SplitLayer,
        transformer::normalisation::qwen_norm::{NormKind, QwenNormCtx, integer::NormQuant},
    },
};

type F = ark_bn254::Fr;

fn norm() -> anyhow::Result<LayerCtx<F>> {
    let scale = ScalingFactor::from_scale(0.001, None);
    Ok(LayerCtx::QwenNorm(QwenNormCtx {
        dimension: 4,
        quant: NormQuant::new(4, NormKind::Rms, 1e-6, scale, scale, 21)?,
        alpha_key: None,
    }))
}

fn graph(layers: Vec<LayerCtx<F>>) -> anyhow::Result<(ModelCtx<F>, Vec<NodeId>)> {
    let mut graph = ContextGraph::new();
    let mut previous = graph.add_input(0)?;
    let mut ids = Vec::new();
    for layer in layers {
        let next = graph.add_inner(layer)?;
        graph.add_consecutive_edge(previous, next, None::<()>)?;
        ids.push(next);
        previous = next;
    }
    let output = graph.add_output(0)?;
    graph.add_consecutive_edge(previous, output, None::<()>)?;
    Ok((ModelCtx::new(graph), ids))
}

fn chunks(model: &ModelCtx<F>, count: usize) -> anyhow::Result<Vec<ModelChunk>> {
    let strategy = DefaultChunkingStrategy::new(vec![vec![512, 4].into()]);
    Ok(model
        .split_in_chunks(Some(count), &strategy, (100usize..).map(NodeId::from))?
        .0)
}

fn horizontal_fixture() -> anyhow::Result<(ModelCtx<F>, NodeId, Vec<ModelChunk>)> {
    let logits = serde_json::from_value(serde_json::json!({"range_check_chunks": 1}))?;
    // DefaultChunkingStrategy reserves vertical groups before and after a
    // horizontal group, so request four chunks around a splittable middle op.
    let (model, ids) = graph(vec![norm()?, LayerCtx::Logits(logits), LayerCtx::Flatten])?;
    let partition = chunks(&model, 4)?;
    assert!(
        partition
            .iter()
            .any(|chunk| chunk.replaced_nodes.contains_key(&ids[1]))
    );
    Ok((model, ids[1], partition))
}

#[test]
fn chunk_guard_rejects_identity_split_impersonating_norm_or_requant() -> anyhow::Result<()> {
    let requant = Requant::from_multiplier(1.0, 24, ScalingFactor::from_scale(1.0, None));
    for layer in [
        norm()?,
        LayerCtx::Requant(RequantCtx {
            requant,
            num_vars: 11,
        }),
    ] {
        let (model, ids) = graph(vec![layer])?;
        let mut forged = chunks(&model, 1)?;
        model.check_model_chunking(forged.iter())?;
        let id = ids[0];
        *forged[0].subgraph.node_mut(id).unwrap() =
            Node::Inner(ChunkedNode::SplitLayer(SplitLayer {
                unpadded_input_shapes: vec![vec![512, 4].into()],
                // Deserialization does not go through the constructor that
                // rejects one chunk. This operation is an identity.
                num_chunks: vec![1],
            }));
        assert!(
            model
                .check_model_chunking(forged.iter())
                .unwrap_err()
                .to_string()
                .contains("trusted original node kind")
        );

        // The former partition check erased this original ID when given an
        // empty replacement list, while its unchanged edges still matched.
        forged[0].replaced_nodes.insert(id, vec![]);
        assert!(
            model
                .check_model_chunking(forged.iter())
                .unwrap_err()
                .to_string()
                .contains("Empty replacement list")
        );
    }
    Ok(())
}

#[test]
fn chunk_guard_accepts_vertical_and_horizontal_partitions() -> anyhow::Result<()> {
    let (vertical, _) = graph(vec![norm()?, norm()?, norm()?, norm()?])?;
    let partition = chunks(&vertical, 2)?;
    assert_eq!(partition.len(), 2);
    assert!(
        partition
            .iter()
            .all(|chunk| chunk.replaced_nodes.is_empty())
    );
    vertical.check_model_chunking(partition.iter())?;

    // Logits supports horizontal splitting; the actual chunking strategy
    // produces its split/recombination and public-IO replacement metadata.
    let (horizontal, _, partition) = horizontal_fixture()?;
    horizontal.check_model_chunking(partition.iter())?;
    Ok(())
}

#[test]
fn chunk_guard_rejects_horizontal_reset_of_unsplittable_operator() -> anyhow::Result<()> {
    let (mut model, id, partition) = horizontal_fixture()?;
    model.check_model_chunking(partition.iter())?;

    *model.nodes.node_mut(id).unwrap() = Node::Inner(norm()?);
    assert!(
        model
            .check_model_chunking(partition.iter())
            .unwrap_err()
            .to_string()
            .contains("does not support horizontal splitting")
    );

    let scale = serde_json::json!({"multiplier": 1, "shift": 0});
    let scan = serde_json::from_value(serde_json::json!({
        "heads": 1, "key_dim": 4, "value_dim": 4,
        "quant": {
            "decay": scale, "memory": scale, "value": scale,
            "delta": scale, "update": scale, "output": scale,
            "input_domains": [[-8, 7], [-8, 7], [-8, 7], [-8, 7], [-8, 7]]
        }
    }))?;
    *model.nodes.node_mut(id).unwrap() = Node::Inner(LayerCtx::GatedDeltaScan(scan));
    assert!(
        model
            .check_model_chunking(partition.iter())
            .unwrap_err()
            .to_string()
            .contains("does not support horizontal splitting")
    );
    Ok(())
}

#[test]
fn chunk_guard_rejects_duplicate_chunk_ids_and_horizontal_indices() -> anyhow::Result<()> {
    let (model, _, mut partition) = horizontal_fixture()?;
    model.check_model_chunking(partition.iter())?;
    let duplicate = partition[0].clone();
    partition.push(duplicate);
    assert!(
        model
            .check_model_chunking(partition.iter())
            .unwrap_err()
            .to_string()
            .contains("Duplicate model chunk ID")
    );
    partition.pop();

    for chunk in &mut partition {
        let ids = chunk
            .subgraph
            .inner_nodes()
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        for id in ids {
            if let Some(Node::Inner(ChunkedNode::ChunkedLayer(layer))) = chunk.subgraph.node_mut(id)
            {
                layer.chunk_number = 0;
            }
        }
    }
    assert!(
        model
            .check_model_chunking(partition.iter())
            .unwrap_err()
            .to_string()
            .contains("Duplicate horizontal chunk")
    );
    Ok(())
}

#[test]
fn chunk_guard_rejects_witness_opening_id_collisions() -> anyhow::Result<()> {
    let (model, _) = graph(vec![norm()?, norm()?, norm()?, norm()?])?;
    let mut partition = chunks(&model, 2)?;
    let table_id = NodeId::from(90usize);
    model.check_chunk_commitment_ids(partition.iter(), table_id)?;
    let (owner, edge_id) = partition
        .iter()
        .enumerate()
        .find_map(|(i, chunk)| chunk.outgoing_edges.keys().next().map(|id| (i, *id)))
        .unwrap();
    let edge = partition[owner].edge(&edge_id)?;
    let port = crate::graph::NodeOutput::new(edge.source(), edge.ports()[0].source_port);
    let boundary_id = NodeId::from(ModelChunk::compute_commitment_id(port));
    let split = Node::Inner(ChunkedNode::SplitLayer(SplitLayer {
        unpadded_input_shapes: vec![vec![512, 4].into()],
        num_chunks: vec![1],
    }));
    partition[0]
        .subgraph
        .add_node_with_id(boundary_id, split.clone())?;
    assert!(
        model
            .check_chunk_commitment_ids(partition.iter(), table_id)
            .unwrap_err()
            .to_string()
            .contains("Boundary commitment ID collides")
    );
    partition[0].subgraph.add_node_with_id(table_id, split)?;
    assert!(
        model
            .check_chunk_commitment_ids(partition.iter(), table_id)
            .unwrap_err()
            .to_string()
            .contains("reserved lookup-table witness ID")
    );
    Ok(())
}
