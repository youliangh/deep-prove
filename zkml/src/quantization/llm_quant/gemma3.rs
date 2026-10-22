//! Implementation of the DuQuant quantisation method as described in https://arxiv.org/pdf/2406.01721

use crate::{
    graph::{Direction, NodeId},
    layers::{Layer, einsum::EinSum, transformer::embeddings::Embeddings},
    model::{
        llm::Driver,
        transform::{ModelTransform, impls::gemma3transform::Gemma3Transform},
    },
    parser::llm::models::gemma3::Gemma3,
    quantization::llm_quant::rotation::{apply_hadamard, apply_hadamard_with_perm},
    tensor::{TensorHandle, WrappedTensor},
};

use super::*;

use anyhow::{Result, anyhow, bail, ensure};

impl FPTransformModel for Gemma3 {
    fn adapt_model(driver: Driver<f32>) -> Result<Driver<f32>> {
        let Driver {
            model,
            md,
            max_context,
            padding_mode,
        } = driver;
        let mut norm_adapted_model = Gemma3Transform.apply(model)?;

        let Layer::Embeddings(embeddings) = norm_adapted_model
            .graph_mut()
            .node_mut(md.embeddings)
            .ok_or(anyhow!("No Embeddings node for model"))?
            .as_inner_mut()
            .ok_or(anyhow!("COuldn't get inner embedding node"))?
        else {
            unreachable!("Expected embeddings node to be of Embeddings type");
        };

        modify_embeddings_layer(embeddings)?;

        for transformer_metadata in md.transformers.iter() {
            let Layer::EinSum(qkv_einsum) = norm_adapted_model
                .graph_mut()
                .node_mut(
                    transformer_metadata
                        .transformer
                        .as_ref()
                        .expect("softmax attention metadata")
                        .qkv_id,
                )
                .ok_or(anyhow!("No QKV Einsum node for model"))?
                .as_inner_mut()
                .ok_or(anyhow!("Couldn't get inner QKV Einsum node"))?
            else {
                unreachable!("Expected QKV node to be of EinSum type");
            };
            pre_hadamard_einsum_layer(qkv_einsum)?;
            post_hadamard_qkv_einsum_layer(qkv_einsum)?;

            let Layer::EinSum(final_proj_einsum) = norm_adapted_model
                .graph_mut()
                .node_mut(
                    transformer_metadata
                        .transformer
                        .as_ref()
                        .expect("softmax attention metadata")
                        .final_proj_id,
                )
                .ok_or(anyhow!("No Final Projection Einsum node for model"))?
                .as_inner_mut()
                .ok_or(anyhow!("Couldn't get inner Final Projection Einsum node"))?
            else {
                unreachable!("Expected Final Projection node to be of EinSum type");
            };
            pre_hadamard_output_projection_layer(final_proj_einsum)?;
            post_hadamard_einsum_layer(final_proj_einsum)?;

            // After the output projection is a RMSNorm layer, then we have moved the scaling part of this layer into
            // an additional EinSum layer, we transform the additional Einsum layer now
            let output_norm_node = norm_adapted_model
                .graph()
                .neighbors(
                    transformer_metadata
                        .transformer
                        .as_ref()
                        .expect("softmax attention metadata")
                        .final_proj_id,
                    Direction::Outgoing,
                )
                .map(|(_, edge)| edge.target())
                .collect::<Vec<NodeId>>();
            ensure!(
                output_norm_node.len() == 1,
                "Expected exactly one output from final projection to RMSNorm"
            );
            let rmsnorm_node_id = output_norm_node[0];
            let scaling_einsum_id = norm_adapted_model
                .graph()
                .neighbors(rmsnorm_node_id, Direction::Outgoing)
                .map(|(_, edge)| edge.target())
                .collect::<Vec<NodeId>>();
            ensure!(
                scaling_einsum_id.len() == 1,
                "Expected exactly one output from RMSNorm to scaling Einsum"
            );

            let Layer::EinSum(final_proj_scaling_einsum) = norm_adapted_model
                .graph_mut()
                .node_mut(scaling_einsum_id[0])
                .ok_or(anyhow!("No Final Projection Scaling Einsum node for model"))?
                .as_inner_mut()
                .ok_or(anyhow!(
                    "Couldn't get inner Final Projection Scaling Einsum node"
                ))?
            else {
                unreachable!("Expected Final Projection node to be of EinSum type");
            };

            pre_hadamard_einsum_layer(final_proj_scaling_einsum)?;
            post_hadamard_einsum_layer(final_proj_scaling_einsum)?;

            let Layer::EinSum(ffn_up_einsum) = norm_adapted_model
                .graph_mut()
                .node_mut(transformer_metadata.ffn.up_id)
                .ok_or(anyhow!("No FFN Up Einsum node for model"))?
                .as_inner_mut()
                .ok_or(anyhow!("Couldn't get inner FFN Up Einsum node"))?
            else {
                unreachable!("Expected FFN Up node to be of EinSum type");
            };
            pre_hadamard_einsum_layer(ffn_up_einsum)?;
            let Layer::EinSum(ffn_down_einsum) = norm_adapted_model
                .graph_mut()
                .node_mut(transformer_metadata.ffn.down_id)
                .ok_or(anyhow!("No FFN Down Einsum node for model"))?
                .as_inner_mut()
                .ok_or(anyhow!("Couldn't get inner FFN Down Einsum node"))?
            else {
                unreachable!("Expected FFN Down node to be of EinSum type");
            };
            post_hadamard_einsum_layer(ffn_down_einsum)?;
            // After the down projection is a RMSNorm layer, then we have moved the scaling part of this layer into
            // an additional EinSum layer, we transform the additional Einsum layer now
            let down_output_norm_node = norm_adapted_model
                .graph()
                .neighbors(transformer_metadata.ffn.down_id, Direction::Outgoing)
                .map(|(_, edge)| edge.target())
                .collect::<Vec<NodeId>>();
            ensure!(
                down_output_norm_node.len() == 1,
                "Expected exactly one output from down projection to RMSNorm"
            );
            let rmsnorm_node_id = down_output_norm_node[0];
            let scaling_einsum_id = norm_adapted_model
                .graph()
                .neighbors(rmsnorm_node_id, Direction::Outgoing)
                .map(|(_, edge)| edge.target())
                .collect::<Vec<NodeId>>();
            ensure!(
                scaling_einsum_id.len() == 1,
                "Expected exactly one output from RMSNorm to scaling Einsum"
            );

            let Layer::EinSum(down_scaling_einsum) = norm_adapted_model
                .graph_mut()
                .node_mut(scaling_einsum_id[0])
                .ok_or(anyhow!("No Final Projection Scaling Einsum node for model"))?
                .as_inner_mut()
                .ok_or(anyhow!(
                    "Couldn't get inner Final Projection Scaling Einsum node"
                ))?
            else {
                unreachable!("Expected Final Projection node to be of EinSum type");
            };

            pre_hadamard_einsum_layer(down_scaling_einsum)?;
            post_hadamard_einsum_layer(down_scaling_einsum)?;
        }

        let Layer::EinSum(final_proj) = norm_adapted_model
            .graph_mut()
            .node_mut(md.final_proj)
            .ok_or(anyhow!("No Final Projection Einsum node for model"))?
            .as_inner_mut()
            .ok_or(anyhow!("Couldn't get inner Final Projection Einsum node"))?
        else {
            unreachable!("Expected Final Projection node to be of EinSum type");
        };
        pre_hadamard_final_proj_layer(final_proj)?;

        Ok(Driver {
            model: norm_adapted_model,
            md,
            max_context,
            padding_mode,
        })
    }
}

fn modify_embeddings_layer(embeddings: &mut Embeddings<f32>) -> Result<()> {
    let tensor = &mut embeddings.mat;
    let dim_size = tensor.shape().dim(-1);
    let log_dim_size = dim_size.ilog2() as usize;
    let block_size = dim_size - (1 << log_dim_size);
    let log_block_size = block_size.ilog2() as usize;
    let inner_tensor = Tensor::try_from(tensor.wrapped_tensor()?.as_ref())?;
    let rotated = apply_hadamard_with_perm::<true, false>(&inner_tensor, log_block_size)?;
    drop(inner_tensor);

    *tensor = TensorHandle::from_tensor(
        tensor.storage_key().to_owned(),
        tensor.store().clone(),
        rotated,
    )
    .wrapped_tensor_variant()?;
    Ok(())
}

fn pre_hadamard_einsum_layer(einsum: &mut EinSum<f32>) -> Result<()> {
    einsum
        .constant_tensors
        .iter_mut()
        .try_for_each(|opt_tensor| {
            if let Some(tensor) = opt_tensor {
                let dim_size = tensor.shape().dim(0);
                let log_dim_size = dim_size.ilog2() as usize;
                let block_size = dim_size - (1 << log_dim_size);
                let log_block_size = block_size.ilog2() as usize;
                let inner_tensor = Tensor::try_from(tensor.wrapped_tensor()?.as_ref())?;
                let rotated =
                    apply_hadamard_with_perm::<false, true>(&inner_tensor, log_block_size)?;

                drop(inner_tensor);
                *tensor = TensorHandle::from_tensor(
                    tensor.storage_key().to_owned(),
                    tensor.store().clone(),
                    rotated,
                )
                .wrapped_tensor_variant()?;
                Result::<()>::Ok(())
            } else {
                bail!("Expected constant tensor in Einsum to modify")
            }
        })?;

    Ok(())
}

fn pre_hadamard_final_proj_layer(einsum: &mut EinSum<f32>) -> Result<()> {
    einsum
        .constant_tensors
        .iter_mut()
        .try_for_each(|opt_tensor| {
            if let Some(old_tensor) = opt_tensor {
                let tensor = Tensor::try_from(old_tensor.wrapped_tensor()?.clone().transpose())?;
                let dim_size = tensor.dim(0);
                let log_dim_size = dim_size.ilog2() as usize;
                let block_size = dim_size - (1 << log_dim_size);
                let log_block_size = block_size.ilog2() as usize;

                let rotated = apply_hadamard_with_perm::<false, true>(&tensor, log_block_size)?;
                let new_tensor = rotated.transpose()?;
                *old_tensor = TensorHandle::from_tensor(
                    old_tensor.storage_key().to_owned(),
                    old_tensor.store().clone(),
                    new_tensor,
                )
                .wrapped_tensor_variant()?;
                Result::<()>::Ok(())
            } else {
                bail!("Expected constant tensor in Einsum to modify")
            }
        })?;

    Ok(())
}

fn pre_hadamard_output_projection_layer(einsum: &mut EinSum<f32>) -> Result<()> {
    einsum
        .constant_tensors
        .iter_mut()
        .try_for_each(|opt_tensor| {
            if let Some(old_tensor) = opt_tensor {
                let rank = old_tensor.shape().rank();
                let wrapped_tensor = old_tensor.wrapped_tensor()?.clone();
                let mut axes: Vec<isize> = (0..rank as isize).collect();
                axes[0] = (rank - 2) as isize;
                axes[rank - 2] = 0;
                let permuted = wrapped_tensor.permute(&axes)?;
                let tensor = Tensor::try_from(&permuted)?;

                let dim_size = tensor.dim(0);

                let block_size = dim_size;
                let log_block_size = block_size.ilog2() as usize;

                let rotated = apply_hadamard::<false, true>(&tensor, log_block_size)?;
                let wrapped_rotated = WrappedTensor::try_from(&rotated)?;
                let permuted_wrapped = wrapped_rotated.permute(&axes)?;

                *old_tensor = TensorHandle::from_wrapped_tensor(
                    old_tensor.storage_key().to_owned(),
                    old_tensor.store().clone(),
                    permuted_wrapped,
                );
                Result::<()>::Ok(())
            } else {
                bail!("Expected constant tensor in Einsum to modify")
            }
        })?;

    Ok(())
}

fn post_hadamard_einsum_layer(einsum: &mut EinSum<f32>) -> Result<()> {
    einsum
        .constant_tensors
        .iter_mut()
        .try_for_each(|opt_tensor| {
            if let Some(tensor) = opt_tensor {
                let dim_size = tensor.shape().dim(-1);
                let log_dim_size = dim_size.ilog2() as usize;
                let block_size = dim_size - (1 << log_dim_size);
                let log_block_size = block_size.ilog2() as usize;

                let inner_tensor = Tensor::try_from(tensor.wrapped_tensor()?.as_ref())?;
                let rotated =
                    apply_hadamard_with_perm::<true, false>(&inner_tensor, log_block_size)?;

                *tensor = TensorHandle::from_tensor(
                    tensor.storage_key().to_owned(),
                    tensor.store().clone(),
                    rotated,
                )
                .wrapped_tensor_variant()?;
                Result::<()>::Ok(())
            } else {
                bail!("Expected constant tensor in Einsum to modify")
            }
        })?;

    Ok(())
}

fn post_hadamard_qkv_einsum_layer(einsum: &mut EinSum<f32>) -> Result<()> {
    einsum
        .constant_tensors
        .iter_mut()
        .skip(2)
        .try_for_each(|opt_tensor| {
            if let Some(tensor) = opt_tensor {
                let dim_size = tensor.shape().dim(-1);
                let log_dim_size = dim_size.ilog2() as usize;
                let block_size = 1usize << log_dim_size;
                let log_block_size = block_size.ilog2() as usize;

                let inner_tensor = Tensor::try_from(tensor.wrapped_tensor()?.as_ref())?;
                let rotated = apply_hadamard::<true, false>(&inner_tensor, log_block_size)?;

                *tensor = TensorHandle::from_tensor(
                    tensor.storage_key().to_owned(),
                    tensor.store().clone(),
                    rotated,
                )
                .wrapped_tensor_variant()?;
                Result::<()>::Ok(())
            } else {
                bail!("Expected constant tensor in Einsum to modify")
            }
        })?;

    Ok(())
}
