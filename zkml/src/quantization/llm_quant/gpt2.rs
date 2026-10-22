//! Implementation of the DuQuant quantisation method as described in https://arxiv.org/pdf/2406.01721

use crate::{
    layers::{
        Layer,
        einsum::EinSum,
        transformer::{
            embeddings::Embeddings,
            positional::{Positional, PositionalVariant, absolute::Absolute},
        },
    },
    model::{
        llm::Driver,
        transform::{ModelTransform, impls::gpt2rmsnorm::GPT2RMSNorm},
    },
    parser::llm::models::gpt2::GPT2,
    quantization::llm_quant::rotation::{apply_hadamard, apply_hadamard_with_perm},
    tensor::{TensorHandle, WrappedTensor},
};

use super::*;

use anyhow::{Result, anyhow, bail};

#[derive(Debug, Clone, Default, Copy)]
pub struct DuQuantQuantiser;

impl FPTransformModel for GPT2 {
    fn adapt_model(driver: Driver<f32>) -> Result<Driver<f32>> {
        let Driver {
            model,
            md,
            max_context,
            padding_mode,
        } = driver;
        let mut norm_adapted_model = GPT2RMSNorm.apply(model)?;

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

        if let Some(positional_id) = md.positional {
            let Layer::Positional(positional) = norm_adapted_model
                .graph_mut()
                .node_mut(positional_id)
                .ok_or(anyhow!("No Positional node for model"))?
                .as_inner_mut()
                .ok_or(anyhow!("Couldn't get inner positional node"))?
            else {
                unreachable!("Expected positional node to be of Positional type");
            };

            modify_positional_layer(positional)?;
        }

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

    let storage_key = tensor.storage_key();
    let store = tensor.store().clone();
    *tensor = TensorHandle::from_tensor(storage_key.to_owned(), store, rotated)
        .wrapped_tensor_variant()?;
    Ok(())
}

fn modify_positional_layer(positional: &mut Positional<f32>) -> Result<()> {
    // Match on the type of positional encoding, we expect `Learned` here
    *positional = match &positional.variant {
        PositionalVariant::Absolute(absolute) => {
            let Absolute::<f32> {
                positional: tensor, ..
            } = absolute;

            let dim_size = tensor.shape().dim(-1);
            let log_dim_size = dim_size.ilog2() as usize;
            let block_size = dim_size - (1 << log_dim_size);
            let log_block_size = block_size.ilog2() as usize;
            let inner_tensor = Tensor::try_from(tensor.wrapped_tensor()?.as_ref())?;
            let rotated = apply_hadamard_with_perm::<true, false>(&inner_tensor, log_block_size)?;

            let storage_key = tensor.storage_key();
            let store = tensor.store().clone();

            Positional::new_absolute(
                TensorHandle::from_tensor(storage_key.to_owned(), store, rotated)
                    .wrapped_tensor_variant()?,
            )?
        }
        PositionalVariant::Rope(_) => {
            bail!(
                "Transformation not implemented for Rope, expected to be applicable only with Absolute positional encoding"
            );
        }
    };
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

                let storage_key = tensor.storage_key();
                let store = tensor.store().clone();
                *tensor = TensorHandle::from_tensor(storage_key.to_owned(), store, rotated)
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
                let inner_tensor = old_tensor.wrapped_tensor()?.clone().transpose();
                let tensor = Tensor::try_from(inner_tensor)?;

                let dim_size = tensor.dim(0);
                let log_dim_size = dim_size.ilog2() as usize;
                let block_size = dim_size - (1 << log_dim_size);
                let log_block_size = block_size.ilog2() as usize;

                let rotated = apply_hadamard_with_perm::<false, true>(&tensor, log_block_size)?;
                let new_tensor = rotated.transpose()?;
                let storage_key = old_tensor.storage_key();
                let store = old_tensor.store().clone();
                *old_tensor = TensorHandle::from_tensor(storage_key.to_owned(), store, new_tensor)
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

    einsum.biases.iter_mut().try_for_each(|opt_tensor| {
        if let Some(tensor) = opt_tensor {
            let dim_size = tensor.shape().dim(-1);
            let log_dim_size = dim_size.ilog2() as usize;
            let block_size = dim_size - (1 << log_dim_size);
            let log_block_size = block_size.ilog2() as usize;
            let inner_tensor = Tensor::try_from(tensor.wrapped_tensor()?.as_ref())?;
            let rotated = apply_hadamard_with_perm::<true, false>(&inner_tensor, log_block_size)?;
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

    einsum.biases.iter_mut().try_for_each(|opt_tensor| {
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
