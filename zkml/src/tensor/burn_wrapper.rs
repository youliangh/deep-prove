//! Wrapper type for burn's tensor.

pub use burn::tensor::{Shape as BShape, TensorKind as BTensorKind};
use serde::{Deserialize, Serialize};

use crate::{
    NextPowerOfTwo, Number,
    backend::{Backend, Conv2dConfig, Maxpool2dConfig, zkml_conv2d_i, zkml_max_pool2d_i},
    quantization::{Dequantize, Quantize, ScalingFactor},
};
use anyhow::{Context, Result, bail, ensure};
use burn::{
    module::Param,
    nn::{LayerNormConfig, RmsNormConfig},
    tensor::{
        AsIndex, BasicOps, DimIter as BDimIter, Numeric, Slice, SliceArg, Tensor as BTensor,
        TensorData, activation, ops::ConvOptions, s,
    },
};

use super::{Element, KeyedTensor, Shape, Tensor};

/// Burn tensor wrapper type with a dynamic rank
#[derive(Debug, Clone)]
pub enum WrappedTensor<T>
where
    T: TensorTypeParam,
{
    Rank1 {
        tensor: BTensor<Backend, 1, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank2 {
        tensor: BTensor<Backend, 2, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank3 {
        tensor: BTensor<Backend, 3, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank4 {
        tensor: BTensor<Backend, 4, T::Kind>,
        unpadded_shape: BShape,
    },
}

pub enum DimIter<T>
where
    T: TensorTypeParam,
{
    Rank1 {
        tensor: BDimIter<Backend, 1, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank2 {
        tensor: BDimIter<Backend, 2, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank3 {
        tensor: BDimIter<Backend, 3, T::Kind>,
        unpadded_shape: BShape,
    },
    Rank4 {
        tensor: BDimIter<Backend, 4, T::Kind>,
        unpadded_shape: BShape,
    },
}

/// Calls `$method` in the underlyig `burn::Tensor` and return its result
/// without re-wrapping.
///
/// Use this for methods that either don't return tensors or return tensors
/// with a different rank or different shape.
macro_rules! delegate_plain {
    // Method with generic type param(s) given in parentheses before any fn args
    ($tensor: expr, $method: ident, ( $($type_arg: tt),* ), $($arg: expr),*) => {
        match $tensor {
            WrappedTensor::Rank1{tensor, ..} => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank2{tensor, ..} => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank3{tensor, ..} => tensor.$method::<$($type_arg),*>($($arg),*),
            WrappedTensor::Rank4{tensor, ..} => tensor.$method::<$($type_arg),*>($($arg),*),
        }
    };

    ($tensor: expr, $method: ident $(, $($arg: expr),* )?) => {
        match $tensor {
            WrappedTensor::Rank1{tensor, ..} => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank2{tensor, ..} => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank3{tensor, ..} => tensor.$method($($($arg),*)?),
            WrappedTensor::Rank4{tensor, ..} => tensor.$method($($($arg),*)?),
        }
    };
}

/// Calls `$method` in the underlyig `burn::Tensor` and wrap the result.
///
/// Use this for methods that return tensors.
///
/// NOTE: that the rank output of the delegated method must match the rank of the
/// input.
macro_rules! delegate {
    ($tensor: expr, $method: ident $(, $($arg: expr),* )?) => {
        match $tensor {
            WrappedTensor::Rank1{tensor, unpadded_shape} => {
                let tensor = tensor.$method($($($arg),*)?);
                WrappedTensor::Rank1{
                    tensor,
                    unpadded_shape,
                }
            },
            WrappedTensor::Rank2{tensor, unpadded_shape} => {
                let tensor = tensor.$method($($($arg),*)?);
                WrappedTensor::Rank2{
                    tensor,
                    unpadded_shape,
                }
            },
            WrappedTensor::Rank3{tensor, unpadded_shape} => {
                let tensor = tensor.$method($($($arg),*)?);
                WrappedTensor::Rank3{
                    tensor,
                    unpadded_shape,
                }
            },
            WrappedTensor::Rank4{tensor, unpadded_shape} => {
                let tensor = tensor.$method($($($arg),*)?);
                WrappedTensor::Rank4{
                    tensor,
                    unpadded_shape,
                }
            },
        }
    };
}

/// Calls the binary op `$method` in the underlyig `burn::Tensor` and wrap the
/// result.
///
/// NOTE: The rank of both tensors and the output must match.
macro_rules! delegate_binop {
    ($tensor: expr, $binop: ident, $other: expr) => {{
        let left_rank = $tensor.rank();
        let right_rank = $other.rank();
        let out = match ($tensor, $other) {
            (
                WrappedTensor::Rank1 {
                    tensor,
                    unpadded_shape,
                },
                WrappedTensor::Rank1 { tensor: other, .. },
            ) => WrappedTensor::Rank1 {
                tensor: tensor.$binop(other),
                unpadded_shape,
            },
            (
                WrappedTensor::Rank2 {
                    tensor,
                    unpadded_shape,
                },
                WrappedTensor::Rank2 { tensor: other, .. },
            ) => WrappedTensor::Rank2 {
                tensor: tensor.$binop(other),
                unpadded_shape,
            },
            (
                WrappedTensor::Rank3 {
                    tensor,
                    unpadded_shape,
                },
                WrappedTensor::Rank3 { tensor: other, .. },
            ) => WrappedTensor::Rank3 {
                tensor: tensor.$binop(other),
                unpadded_shape,
            },
            (
                WrappedTensor::Rank4 {
                    tensor,
                    unpadded_shape,
                },
                WrappedTensor::Rank4 { tensor: other, .. },
            ) => WrappedTensor::Rank4 {
                tensor: tensor.$binop(other),
                unpadded_shape,
            },
            _ => bail!("Unmatched input ranks. Left: {left_rank}, right: {right_rank}."),
        };
        Ok(out)
    }};
}

impl<T> WrappedTensor<T>
where
    T: TensorTypeParam,
{
    /// Returns a `Vec` with the elements from this tensor.
    ///
    /// NOTE: This is a blocking call that will cause synchronization with the
    /// acceleration hardware. For external GPUs this will wait for the tensor
    /// to be computed and then download the resulting data from the card.
    pub fn get_data(&self) -> Vec<T> {
        self.clone().to_data().into_vec().unwrap()
    }

    /// Converts the tensor into a primitive tensor.
    pub fn into_primitive(self) -> <T::Kind as BTensorKind<Backend>>::Primitive {
        delegate_plain!(self, into_primitive)
    }

    /// Returns a [WrappedTensor] filled with random data of `shape`.
    pub fn random(shape: &Shape) -> Self {
        Self::try_from(Tensor::random(shape)).unwrap()
    }

    /// Returns a copy of this tensor's [BShape].
    pub fn shape(&self) -> BShape {
        delegate_plain!(self, shape)
    }
}

impl<T> WrappedTensor<T>
where
    T: TensorTypeParam,
{
    /// Returns the wrapped tensor's rank.
    pub(crate) const fn rank(&self) -> usize {
        match self {
            Self::Rank1 { .. } => 1,
            Self::Rank2 { .. } => 2,
            Self::Rank3 { .. } => 3,
            Self::Rank4 { .. } => 4,
        }
    }

    /// Reshape the tensor to have the given shape.
    ///
    /// NOTE: This will change the `unpadded_shape` accordingly, as if a similar
    /// operation was applied to a tensor of that shape.
    pub(crate) fn reshape(self, new_shape: BShape) -> Result<WrappedTensor<T>> {
        let new_unpadded = if !self.is_padded() {
            new_shape.clone()
        } else {
            // XXX: Figure out how to split unpadded shape
            ensure!(
                new_shape.len() <= self.unpadded_shape().len(),
                "Increasing the number of dimensions is not supported. shape: {:?} unppaded_shape: {:?} new_shape: {new_shape:?}",
                self.shape(),
                self.unpadded_shape(),
            );

            let curr_shape = self.shape();
            let mut curr_dims = curr_shape.iter();
            let mut curr_unpadded = self.unpadded_shape().iter();

            let mut new_unpadded = Vec::with_capacity(new_shape.len());
            for new in new_shape.iter() {
                let mut curr = *curr_dims
                    .next()
                    .expect("Current shape rank is at least equal to new");
                let mut unpadded = *curr_unpadded
                    .next()
                    .expect("Current unpadded shape rank is at least equal to new");

                while *new != curr {
                    curr *= curr_dims
                        .next()
                        .expect("Current shape rank is at least equal to new");
                    unpadded *= curr_unpadded
                        .next()
                        .expect("Current unpadded shape rank is at least equal to new");
                }

                new_unpadded.push(unpadded);
            }
            BShape::from(new_unpadded)
        };

        let out = match new_shape.num_dims() {
            1 => WrappedTensor::Rank1 {
                tensor: delegate_plain!(self, reshape, (1, _), new_shape),
                unpadded_shape: new_unpadded,
            },
            2 => WrappedTensor::Rank2 {
                tensor: delegate_plain!(self, reshape, (2, _), new_shape),
                unpadded_shape: new_unpadded,
            },
            3 => WrappedTensor::Rank3 {
                tensor: delegate_plain!(self, reshape, (3, _), new_shape),
                unpadded_shape: new_unpadded,
            },
            4 => WrappedTensor::Rank4 {
                tensor: delegate_plain!(self, reshape, (4, _), new_shape),
                unpadded_shape: new_unpadded,
            },
            _ => bail!("Unexpected tensor rank: {}.", new_shape.num_dims()),
        };

        Ok(out)
    }

    /// Returns a reference to the `unpadded_shape`.
    pub(crate) fn unpadded_shape(&self) -> &BShape {
        match self {
            WrappedTensor::Rank1 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank2 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank3 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank4 { unpadded_shape, .. } => unpadded_shape,
        }
    }

    /// Returns a mutable reference to the `unpadded_shape`.
    pub(crate) fn unpadded_shape_mut(&mut self) -> &mut BShape {
        match self {
            WrappedTensor::Rank1 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank2 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank3 { unpadded_shape, .. } => unpadded_shape,
            WrappedTensor::Rank4 { unpadded_shape, .. } => unpadded_shape,
        }
    }

    /// Set the `unpadded_shape`.
    pub(crate) fn set_unpadded_shape(&mut self, new_shape: BShape) {
        match self {
            WrappedTensor::Rank1 { unpadded_shape, .. } => *unpadded_shape = new_shape,
            WrappedTensor::Rank2 { unpadded_shape, .. } => *unpadded_shape = new_shape,
            WrappedTensor::Rank3 { unpadded_shape, .. } => *unpadded_shape = new_shape,
            WrappedTensor::Rank4 { unpadded_shape, .. } => *unpadded_shape = new_shape,
        }
    }

    /// `True` if the `unpadded_shape` and `shape` differ.
    pub(crate) fn is_padded(&self) -> bool {
        self.unpadded_shape() != &self.shape()
    }

    /// Creates a tensor from [TensorData].
    pub(crate) fn from_data(data: TensorData) -> Result<Self> {
        let rank = data.shape.len();
        let unpadded_shape = BShape::from(&data.shape);
        let out = match rank {
            1 => WrappedTensor::Rank1 {
                tensor: BTensor::<Backend, 1, T::Kind>::from_data(data, &Default::default()),
                unpadded_shape,
            },
            2 => WrappedTensor::Rank2 {
                tensor: BTensor::<Backend, 2, T::Kind>::from_data(data, &Default::default()),
                unpadded_shape,
            },
            3 => WrappedTensor::Rank3 {
                tensor: BTensor::<Backend, 3, T::Kind>::from_data(data, &Default::default()),
                unpadded_shape,
            },
            4 => WrappedTensor::Rank4 {
                tensor: BTensor::<Backend, 4, T::Kind>::from_data(data, &Default::default()),
                unpadded_shape,
            },
            _ => bail!("Unexpected tensor rank: {rank}."),
        };
        Ok(out)
    }

    /// Returns a copy of this tensor's [TensorData].
    ///
    /// NOTE: This is a blocking call that will cause synchronization with the
    /// acceleration hardware. For external GPUs this will wait for the tensor
    /// to be computed and then download the resulting data from the card.
    pub(crate) fn to_data(&self) -> TensorData {
        delegate_plain!(self, to_data)
    }

    /// Concatenates all tensors into a new one along the given dimension.
    pub(crate) fn cat(tensors: Vec<Self>, dim: usize) -> Result<Self> {
        ensure!(tensors.len() > 1, "There must be at least one tensor");

        let mut unpadded_shape = tensors[0].unpadded_shape().clone();
        let summed_concat_dim = tensors
            .iter()
            .map(|tensor| tensor.shape().dims[dim])
            .sum::<usize>();
        unpadded_shape.dims[dim] = summed_concat_dim;

        match tensors[0].rank() {
            1 => {
                let to_concat = tensors
                    .into_iter()
                    .map(|tensor| match tensor {
                        WrappedTensor::Rank1 { tensor, .. } => Ok(tensor),
                        _ => bail!("Cat does not support mixed ranks"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(WrappedTensor::Rank1 {
                    tensor: BTensor::cat(to_concat, dim),
                    unpadded_shape,
                })
            }
            2 => {
                let to_concat = tensors
                    .into_iter()
                    .map(|tensor| match tensor {
                        WrappedTensor::Rank2 { tensor, .. } => Ok(tensor),
                        _ => bail!("Cat does not support mixed ranks"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(WrappedTensor::Rank2 {
                    tensor: BTensor::cat(to_concat, dim),
                    unpadded_shape,
                })
            }
            3 => {
                let to_concat = tensors
                    .into_iter()
                    .map(|tensor| match tensor {
                        WrappedTensor::Rank3 { tensor, .. } => Ok(tensor),
                        _ => bail!("Cat does not support mixed ranks"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(WrappedTensor::Rank3 {
                    tensor: BTensor::cat(to_concat, dim),
                    unpadded_shape,
                })
            }
            4 => {
                let to_concat = tensors
                    .into_iter()
                    .map(|tensor| match tensor {
                        WrappedTensor::Rank4 { tensor, .. } => Ok(tensor),
                        _ => bail!("Cat does not support mixed ranks"),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(WrappedTensor::Rank4 {
                    tensor: BTensor::cat(to_concat, dim),
                    unpadded_shape,
                })
            }
            _ => unreachable!(),
        }
    }

    /// Find the maximum absolute value.
    pub(crate) fn max_abs(self) -> Self {
        let tensor = delegate_plain!(self, max_abs);

        // fix the unpadded shape, the final tensor shape is `[1]`
        let unpadded_shape = tensor.shape();

        WrappedTensor::Rank1 {
            tensor,
            unpadded_shape,
        }
    }

    /// Attempts to split the tensor into a specified number of chunks along a
    /// given dimension.
    ///
    /// May return less chunks than requested if the tensor size is not
    /// divisible by the number of chunks.
    pub fn chunk(self, chunks: usize, dim: usize) -> Result<Vec<Self>> {
        ensure!(
            dim < self.rank(),
            "Chunk dimension {dim} out of bounds for tensor of rank {}",
            self.rank(),
        );

        let original_shape = self.shape();
        let original_unpadded = self.unpadded_shape().clone();

        // Every dimension should have at least one non padding element.
        // XXX: Unclear what exactly should be the rule here, this is a relaxed approach.
        // let padding = original_shape.dims[dim] - original_unpadded.dims[dim];
        // ensure!(
        // padding <= chunks,
        // "Chunk must not be larger than the padding to include at least one non-padding element",
        // );

        let mut out: Vec<Self> = match self {
            WrappedTensor::Rank1 { tensor, .. } => tensor
                .chunk(chunks, dim)
                .into_iter()
                .map(WrappedTensor::from)
                .collect(),
            WrappedTensor::Rank2 { tensor, .. } => tensor
                .chunk(chunks, dim)
                .into_iter()
                .map(WrappedTensor::from)
                .collect(),
            WrappedTensor::Rank3 { tensor, .. } => tensor
                .chunk(chunks, dim)
                .into_iter()
                .map(WrappedTensor::from)
                .collect(),
            WrappedTensor::Rank4 { tensor, .. } => tensor
                .chunk(chunks, dim)
                .into_iter()
                .map(WrappedTensor::from)
                .collect(),
        };

        // Fix unpadded shapes
        if original_shape != original_unpadded {
            out.iter_mut().fold(0, |num_items_on_dim, chunk| {
                let dim_size_in_chunk = chunk.shape()[dim];
                let unpadded_dim_size =
                    if num_items_on_dim + dim_size_in_chunk > original_unpadded[dim] {
                        original_unpadded[dim].saturating_sub(num_items_on_dim)
                    } else {
                        dim_size_in_chunk
                    };
                let mut chunk_shape = original_unpadded.clone();
                chunk_shape.dims[dim] = unpadded_dim_size;
                chunk.set_unpadded_shape(chunk_shape);
                num_items_on_dim + dim_size_in_chunk
            });
        }

        Ok(out)
    }

    /// Find the maximum value.
    pub(crate) fn max(self) -> Self {
        let tensor = delegate_plain!(self, max);

        // fix the unpadded shape, the final tensor shape is `[1]`
        let unpadded_shape = tensor.shape();

        WrappedTensor::Rank1 {
            tensor,
            unpadded_shape,
        }
    }

    /// Find the minimum value.
    pub(crate) fn min(self) -> Self {
        let tensor = delegate_plain!(self, min);

        // fix the unpadded shape, the final tensor shape is `[1]`
        let unpadded_shape = tensor.shape();

        WrappedTensor::Rank1 {
            tensor,
            unpadded_shape,
        }
    }

    /// Applies element wise absolute value operation.
    pub fn abs(self) -> Self {
        delegate!(self, abs)
    }

    /// Transpose the tensor.
    ///
    /// For a 2D tensor, this is the standard matrix transpose. For `D > 2`, the transpose is
    /// applied on the last two dimensions. For example, the transpose of a tensor with shape
    pub(crate) fn transpose(self) -> Self {
        let mut result = delegate!(self, transpose);

        // fix the unpadded shape, apply the transform operation for it too
        let unpadded_shape = result.unpadded_shape().clone();
        let ndims = unpadded_shape.num_dims();
        let unpadded_shape = unpadded_shape.swap(ndims - 2, ndims - 1).unwrap();
        result.set_unpadded_shape(unpadded_shape);

        result
    }

    /// Squeeze the tensor along the given dimension, removing the specified dimension
    /// of size one, and effectively reducing the rank of the tensor by one.
    pub(crate) fn squeeze(self, dim: usize) -> Result<Self> {
        let mut unpadded_shape = self.unpadded_shape().clone();
        ensure!(
            unpadded_shape.dims.remove(dim) == 1,
            "Dimension must be equal to 1 to be squeezed. dims: {:?} squeezing: {}",
            unpadded_shape.dims,
            dim,
        );
        let out = match self {
            WrappedTensor::Rank1 { .. } => bail!("Cannot squeeze 1D tensor"),
            WrappedTensor::Rank2 { tensor, .. } => WrappedTensor::Rank1 {
                tensor: tensor.squeeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank3 { tensor, .. } => WrappedTensor::Rank2 {
                tensor: tensor.squeeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank4 { tensor, .. } => WrappedTensor::Rank3 {
                tensor: tensor.squeeze_dim(dim),
                unpadded_shape,
            },
        };
        Ok(out)
    }

    /// Select tensor elements along the given dimension corresponding to the given indices.
    pub(crate) fn select(
        self,
        dim: impl AsIndex,
        indices: WrappedTensor<Element>,
    ) -> anyhow::Result<Self> {
        let mut unpadded_shape = self.unpadded_shape().clone();

        let WrappedTensor::Rank1 {
            tensor: indices, ..
        } = indices
        else {
            bail!("Only 1D indices is supported");
        };
        // XXX: how should indices outside of the unpadded shape be handled?
        unpadded_shape.dims[dim.index() as usize] = indices.shape().num_elements();

        let res = match self {
            WrappedTensor::Rank1 { tensor, .. } => WrappedTensor::Rank1 {
                tensor: tensor.select(dim, indices),
                unpadded_shape,
            },
            WrappedTensor::Rank2 { tensor, .. } => WrappedTensor::Rank2 {
                tensor: tensor.select(dim, indices),
                unpadded_shape,
            },
            WrappedTensor::Rank3 { tensor, .. } => WrappedTensor::Rank3 {
                tensor: tensor.select(dim, indices),
                unpadded_shape,
            },
            WrappedTensor::Rank4 { tensor, .. } => WrappedTensor::Rank4 {
                tensor: tensor.select(dim, indices),
                unpadded_shape,
            },
        };

        Ok(res)
    }

    /// Applies element wise addition operation.
    #[allow(clippy::should_implement_trait)]
    pub(crate) fn add(self, other: Self) -> Result<Self> {
        delegate_binop!(self, add, other)
    }

    /// Applies element wise addition operation with a scalar.
    pub(crate) fn add_scalar(self, other: T) -> Self {
        delegate!(self, add_scalar, other)
    }

    /// Applies element wise subtraction operation.
    #[allow(clippy::should_implement_trait)]
    pub(crate) fn sub(self, other: Self) -> Result<Self> {
        delegate_binop!(self, sub, other)
    }

    /// Applies element wise negation operation.
    #[allow(clippy::should_implement_trait)]
    pub(crate) fn neg(self) -> Self {
        delegate!(self, neg)
    }

    /// Applies element wise multiplication operation.
    #[allow(clippy::should_implement_trait)]
    pub(crate) fn mul(self, other: Self) -> Result<Self> {
        delegate_binop!(self, mul, other)
    }

    /// Applies element wise multiplication operation with a scalar.
    pub(crate) fn mul_scalar(self, other: T) -> Self {
        delegate!(self, mul_scalar, other)
    }

    /// Applies element wise division operation with a scalar.
    pub(crate) fn div_scalar(self, other: T) -> Self {
        delegate!(self, div_scalar, other)
    }

    /// Applies the matrix multiplication operation.
    pub(crate) fn matmul(self, other: Self) -> Result<Self> {
        fn batch_dim(left_dim: usize, right_dim: usize) -> usize {
            if left_dim == right_dim {
                left_dim
            } else if left_dim == 1 {
                // broadcast left
                right_dim
            } else if right_dim == 1 {
                // broadcast right.
                left_dim
            } else {
                panic!("Shouldnt happen, matmul should validate the dimensions");
            }
        }

        match (self, other) {
            (
                WrappedTensor::Rank1 {
                    tensor: left,
                    unpadded_shape: left_unpadded_shape,
                },
                WrappedTensor::Rank1 {
                    tensor: right,
                    unpadded_shape: right_unpadded_shape,
                },
            ) => {
                let tensor = left.matmul(right);
                let _left_unpadded_shape = left_unpadded_shape;
                let _right_unpadded_shape = right_unpadded_shape;

                Ok(WrappedTensor::Rank1 {
                    tensor,
                    unpadded_shape: BShape::new([1]),
                })
            }
            (
                WrappedTensor::Rank2 {
                    tensor: left,
                    unpadded_shape: left_unpadded_shape,
                },
                WrappedTensor::Rank2 {
                    tensor: right,
                    unpadded_shape: right_unpadded_shape,
                },
            ) => {
                let tensor = T::matmul(left, right)?;
                let unpadded_shape = BShape::new([left_unpadded_shape[0], right_unpadded_shape[1]]);

                Ok(WrappedTensor::Rank2 {
                    tensor,
                    unpadded_shape,
                })
            }
            (
                WrappedTensor::Rank3 {
                    tensor: left,
                    unpadded_shape: left_unpadded_shape,
                },
                WrappedTensor::Rank3 {
                    tensor: right,
                    unpadded_shape: right_unpadded_shape,
                },
            ) => {
                let tensor = T::matmul(left, right)?;
                let unpadded_shape = BShape::new([
                    batch_dim(left_unpadded_shape[0], right_unpadded_shape[0]),
                    left_unpadded_shape[1],
                    right_unpadded_shape[2],
                ]);

                Ok(WrappedTensor::Rank3 {
                    tensor,
                    unpadded_shape,
                })
            }
            (
                WrappedTensor::Rank4 {
                    tensor: left,
                    unpadded_shape: left_unpadded_shape,
                },
                WrappedTensor::Rank4 {
                    tensor: right,
                    unpadded_shape: right_unpadded_shape,
                },
            ) => {
                let tensor = T::matmul(left, right)?;
                let unpadded_shape = BShape::new([
                    batch_dim(left_unpadded_shape[0], right_unpadded_shape[0]),
                    batch_dim(left_unpadded_shape[1], right_unpadded_shape[1]),
                    left_unpadded_shape[2],
                    right_unpadded_shape[3],
                ]);

                Ok(WrappedTensor::Rank4 {
                    tensor,
                    unpadded_shape,
                })
            }
            (left, right) => bail!(
                "Unmatched input ranks. left: {}, right: {}",
                left.shape(),
                right.shape(),
            ),
        }
    }

    /// Clamp element wise between the given min and max values.
    pub(crate) fn clamp(self, min: T, max: T) -> Self {
        delegate!(self, clamp, min, max)
    }

    /// Aggregate all elements along the given dimension or axis in the tensor with the sum operation.
    pub(crate) fn sum_dim(self, dim: isize) -> Self {
        delegate!(self, sum_dim, dim)
    }

    /// Find the maximum value along the given dimension.
    pub(crate) fn max_dim(self, dim: isize) -> Self {
        let mut result = delegate!(self, max_dim, dim);

        // fix the unpadded shape, apply the transform operation for it too
        let dim = if dim < 0 {
            (result.rank() as isize) + dim
        } else {
            dim
        };
        result.unpadded_shape_mut()[dim as usize] = 1;

        result
    }

    /// Flatten the tensor into 1D shape.
    pub(crate) fn flatten_1d(self) -> Self {
        let end_dim = self.rank() - 1;
        let unpadded_shape = self.unpadded_shape().clone().flatten();
        let tensor = delegate_plain!(self, flatten, 0, end_dim);
        WrappedTensor::Rank1 {
            tensor,
            unpadded_shape,
        }
    }

    /// Flatten the tensor along a given range of dimensions into 2 dimensions.
    pub(crate) fn flatten_to_dim_2(self, start_dim: usize, end_dim: usize) -> Self {
        let mut unpadded_shape = self.unpadded_shape().clone();
        let new_size = unpadded_shape.dims.drain(start_dim..=end_dim).product();
        unpadded_shape.dims.insert(start_dim, new_size);
        let tensor = delegate_plain!(self, flatten, start_dim, end_dim);
        Self::Rank2 {
            tensor,
            unpadded_shape,
        }
    }

    ///  Find the maximum value along the given dimension.
    pub fn max_dim_with_indices(self, dim: isize) -> (Self, WrappedTensor<Element>) {
        // XXX: fix unpadded shapes
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    WrappedTensor::Rank1 {
                        tensor: maxes,
                        unpadded_shape: unpadded_shape.clone(),
                    },
                    WrappedTensor::Rank1 {
                        tensor: indices,
                        unpadded_shape,
                    },
                )
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    WrappedTensor::Rank2 {
                        tensor: maxes,
                        unpadded_shape: unpadded_shape.clone(),
                    },
                    WrappedTensor::Rank2 {
                        tensor: indices,
                        unpadded_shape,
                    },
                )
            }
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    WrappedTensor::Rank3 {
                        tensor: maxes,
                        unpadded_shape: unpadded_shape.clone(),
                    },
                    WrappedTensor::Rank3 {
                        tensor: indices,
                        unpadded_shape,
                    },
                )
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                let (maxes, indices) = tensor.max_dim_with_indices(dim);
                (
                    WrappedTensor::Rank4 {
                        tensor: maxes,
                        unpadded_shape: unpadded_shape.clone(),
                    },
                    WrappedTensor::Rank4 {
                        tensor: indices,
                        unpadded_shape,
                    },
                )
            }
        }
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 2 dimensions.
    pub(crate) fn unsqueeze_dim_2(self) -> Self {
        let mut unpadded_shape = self.unpadded_shape().clone();
        while unpadded_shape.len() != 2 {
            unpadded_shape.dims.insert(0, 1);
        }

        let tensor = delegate_plain!(self, unsqueeze, (2),);
        WrappedTensor::Rank2 {
            tensor,
            unpadded_shape,
        }
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 2 dimensions.
    pub(crate) fn unsqueeze_dim_3(self) -> Self {
        let mut unpadded_shape = self.unpadded_shape().clone();
        while unpadded_shape.len() != 3 {
            unpadded_shape.dims.insert(0, 1);
        }

        let tensor = delegate_plain!(self, unsqueeze, (3),);
        WrappedTensor::Rank3 {
            tensor,
            unpadded_shape,
        }
    }

    /// Unsqueeze the current tensor. Create new leading dimensions to fit 4 dimensions.
    pub(crate) fn unsqueeze_dim_4(self) -> Self {
        let mut unpadded_shape = self.unpadded_shape().clone();
        while unpadded_shape.len() != 4 {
            unpadded_shape.dims.insert(0, 1);
        }

        let tensor = delegate_plain!(self, unsqueeze, (4),);
        WrappedTensor::Rank4 {
            tensor,
            unpadded_shape,
        }
    }

    /// Creates a new tensor with a dimension of size one inserted at the specified position.
    pub(crate) fn unsqueeze_dim(self, dim: usize) -> Result<Self> {
        let mut unpadded_shape = self.unpadded_shape().clone();
        unpadded_shape.dims.insert(dim, 1);

        let out = match self {
            WrappedTensor::Rank1 { tensor, .. } => WrappedTensor::Rank2 {
                tensor: tensor.unsqueeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank2 { tensor, .. } => WrappedTensor::Rank3 {
                tensor: tensor.unsqueeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank3 { tensor, .. } => WrappedTensor::Rank4 {
                tensor: tensor.unsqueeze_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank4 { .. } => bail!("Cannot unsqueeze 4D tensor"),
        };
        Ok(out)
    }

    /// Iterate over slices of tensors alongside a given dimension.
    pub(crate) fn iter_dim(self, dim: usize) -> DimIter<T> {
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => DimIter::Rank1 {
                tensor: tensor.iter_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => DimIter::Rank2 {
                tensor: tensor.iter_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => DimIter::Rank3 {
                tensor: tensor.iter_dim(dim),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => DimIter::Rank4 {
                tensor: tensor.iter_dim(dim),
                unpadded_shape,
            },
        }
    }

    /// Permute the dimensions of the tensor.
    pub(crate) fn permute(self, axes: &[isize]) -> Result<Self> {
        let out = match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => {
                let axes: [isize; 1] = TryFrom::try_from(axes).with_context(|| {
                    format!(
                        "Unexpected permutation axes length. Expected 1, got {}",
                        axes.len(),
                    )
                })?;
                let shape_axes = axes.map(|d| d as usize);
                let unpadded_shape = unpadded_shape
                    .permute(shape_axes.as_slice())
                    .map_err(|e| anyhow::anyhow!("Could not permute unpadded shape: permutation: {shape_axes:?}, inner error:{e:?}"))?;
                WrappedTensor::Rank1 {
                    tensor: tensor.permute(axes),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => {
                let axes: [isize; 2] = TryFrom::try_from(axes).with_context(|| {
                    format!(
                        "Unexpected permutation axes length. Expected 2, got {}",
                        axes.len(),
                    )
                })?;
                let shape_axes = axes.map(|d| d as usize);
                let unpadded_shape = unpadded_shape
                    .permute(shape_axes.as_slice())
                    .map_err(|e| anyhow::anyhow!("Could not permute unpadded shape: permutation: {shape_axes:?}, inner error:{e:?}"))?;
                WrappedTensor::Rank2 {
                    tensor: tensor.permute(axes),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                let axes: [isize; 3] = TryFrom::try_from(axes).with_context(|| {
                    format!(
                        "Unexpected permutation axes length. Expected 3, got {}",
                        axes.len(),
                    )
                })?;
                let shape_axes = axes.map(|d| d as usize);
                let unpadded_shape = unpadded_shape
                    .permute(shape_axes.as_slice())
                    .map_err(|e| anyhow::anyhow!("Could not permute unpadded shape: permutation: {shape_axes:?}, inner error:{e:?}"))?;
                WrappedTensor::Rank3 {
                    tensor: tensor.permute(axes),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                let axes: [isize; 4] = TryFrom::try_from(axes).with_context(|| {
                    format!(
                        "Unexpected permutation axes length. Expected 4, got {}",
                        axes.len(),
                    )
                })?;
                let shape_axes = axes.map(|d| d as usize);
                let unpadded_shape = unpadded_shape
                    .permute(shape_axes.as_slice())
                    .map_err(|e| anyhow::anyhow!("Could not permute unpadded shape: permutation: {shape_axes:?}, inner error:{e:?}"))?;
                WrappedTensor::Rank4 {
                    tensor: tensor.permute(axes),
                    unpadded_shape,
                }
            }
        };
        Ok(out)
    }

    /// Returns a tensor containing the elements selected from the given ranges.
    pub(crate) fn slice<R>(self, ranges: R) -> Self
    where
        R: Clone + SliceArg,
    {
        fn to_shape<R: SliceArg>(shape: &BShape, ranges: R) -> BShape {
            let slices = ranges.into_slices(shape);
            let dims: Vec<usize> = slices
                .iter()
                .enumerate()
                .map(|(dim, slice)| slice.output_size(shape.dims[dim]))
                .collect();
            BShape::from(dims)
        }

        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: tensor.slice(ranges.clone()),
                unpadded_shape: to_shape(&unpadded_shape, ranges),
            },
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: tensor.slice(ranges.clone()),
                unpadded_shape: to_shape(&unpadded_shape, ranges),
            },
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: tensor.slice(ranges.clone()),
                unpadded_shape: to_shape(&unpadded_shape, ranges),
            },
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: tensor.slice(ranges.clone()),
                unpadded_shape: to_shape(&unpadded_shape, ranges),
            },
        }
    }

    /// Returns a new tensor with the specified dimension sliced.
    pub(crate) fn slice_dim<S>(self, dim: usize, slice: S) -> Self
    where
        S: Into<Slice>,
    {
        let slice = slice.into();
        let mut result = delegate!(self, slice_dim, dim, slice);

        // fix the unpadded shape, apply the transform operation for it too
        let mut unpadded_shape = result.unpadded_shape().clone();
        let dim_size = slice.output_size(unpadded_shape[dim]);
        unpadded_shape[dim] = dim_size;
        result.set_unpadded_shape(unpadded_shape);

        result
    }

    /// Slice the tensor on the first dimension.
    ///
    /// # Arguments
    ///
    /// - start: New start for the first dimension, inclusive.
    /// - end: New end for the first dimension, exclusive.
    pub(crate) fn slice_2d(self, start: usize, end: usize) -> Self {
        self.slice_dim(0, start..end)
    }

    /// Returns the size of the given dimension.
    ///
    /// When given a negative `dim` indexes from the back.
    pub(crate) fn dim(&self, dim: isize) -> Result<usize> {
        match self {
            WrappedTensor::Rank1 { tensor, .. } => {
                let dim = if dim < 0 {
                    (tensor.dims().len() as isize + dim) as usize
                } else {
                    dim as usize
                };
                tensor
                    .dims()
                    .get(dim)
                    .context("Dimension out of bounds")
                    .copied()
            }
            WrappedTensor::Rank2 { tensor, .. } => {
                let dim = if dim < 0 {
                    (tensor.dims().len() as isize + dim) as usize
                } else {
                    dim as usize
                };
                tensor
                    .dims()
                    .get(dim)
                    .context("Dimension out of bounds")
                    .copied()
            }
            WrappedTensor::Rank3 { tensor, .. } => {
                let dim = if dim < 0 {
                    (tensor.dims().len() as isize + dim) as usize
                } else {
                    dim as usize
                };
                tensor
                    .dims()
                    .get(dim)
                    .context("Dimension out of bounds")
                    .copied()
            }
            WrappedTensor::Rank4 { tensor, .. } => {
                let dim = if dim < 0 {
                    (tensor.dims().len() as isize + dim) as usize
                } else {
                    dim as usize
                };
                tensor
                    .dims()
                    .get(dim)
                    .context("Dimension out of bounds")
                    .copied()
            }
        }
    }

    /// Copies the sub-slice `shape` from this tensor.
    ///
    /// Returns a new [Tensor] with shape `shape` initialised from `self`.
    pub(crate) fn reduce_to_shape(self, shape: &BShape) -> Result<Self> {
        let out = match shape.num_dims() {
            1 => self.slice(shape.dims::<1>().map(|i| 0..i)),
            2 => self.slice(shape.dims::<2>().map(|i| 0..i)),
            3 => self.slice(shape.dims::<3>().map(|i| 0..i)),
            4 => self.slice(shape.dims::<4>().map(|i| 0..i)),
            rank => bail!("Unexpected shape rank: {rank}."),
        };
        Ok(out)
    }

    pub(crate) fn reduce_to_unpadded_shape(self) -> Result<Self> {
        let unpadded_shape = self.unpadded_shape().clone();
        self.reduce_to_shape(&unpadded_shape)
    }

    /// Update the given tensor with the value where the mask is true.
    pub fn mask_fill_4d(
        self,
        mask: BTensor<Backend, 4, burn::tensor::Bool>,
        value: T,
    ) -> Result<Self> {
        let input_rank = self.rank();
        let Self::Rank4 {
            tensor,
            unpadded_shape,
        } = self
        else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        Ok(WrappedTensor::Rank4 {
            tensor: tensor.mask_fill(mask, value),
            unpadded_shape,
        })
    }

    /// Update the given tensor with the value where the mask is true.
    pub fn mask_fill_3d(
        self,
        mask: BTensor<Backend, 3, burn::tensor::Bool>,
        value: T,
    ) -> Result<Self> {
        let input_rank = self.rank();
        let Self::Rank3 {
            tensor,
            unpadded_shape,
        } = self
        else {
            bail!("Unexpected input rank: {input_rank}, expected 3.")
        };
        Ok(WrappedTensor::Rank3 {
            tensor: tensor.mask_fill(mask, value),
            unpadded_shape,
        })
    }

    pub fn mask_fill(
        self,
        mask: BTensor<Backend, 2, burn::tensor::Bool>,
        value: T,
    ) -> Result<Self> {
        match self {
            WrappedTensor::Rank1 { .. } => {
                bail!("mask_fill only works for rank 2 or higher tensors")
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank2 {
                tensor: tensor.mask_fill(mask, value),
                unpadded_shape,
            }),
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                // We expand the mask to rank 3 by unsqueezing a new leading dimension.
                let mask = mask.unsqueeze::<3>().expand(tensor.shape());
                Ok(WrappedTensor::Rank3 {
                    tensor: tensor.mask_fill(mask, value),
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                // We expand the mask to rank 4 by unsqueezing a new leading dimension.
                let mask = mask.unsqueeze::<4>().expand(tensor.shape());
                Ok(WrappedTensor::Rank4 {
                    tensor: tensor.mask_fill(mask, value),
                    unpadded_shape,
                })
            }
        }
    }

    /// Pads the tensor to the new shape.
    pub(crate) fn pad(self, new_shape: BShape, pad_value: T) -> anyhow::Result<Self> {
        let curr_shape = self.shape();

        ensure!(
            curr_shape.rank() == new_shape.rank(),
            "The new shape must have the same rank",
        );
        ensure!(
            curr_shape
                .iter()
                .zip(new_shape.iter())
                .all(|(curr, new)| curr <= new),
            "Padding must maintain or increase existing dimensions",
        );
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => {
                #[allow(clippy::single_range_in_vec_init)]
                let copy_slices = [0..curr_shape[0]];
                let out = BTensor::full(new_shape, pad_value, &tensor.device());
                let tensor = out.slice_assign(copy_slices, tensor);
                Ok(WrappedTensor::Rank1 {
                    tensor,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => {
                let copy_slices = [0..curr_shape[0], 0..curr_shape[1]];
                let out = BTensor::full(new_shape, pad_value, &tensor.device());
                let tensor = out.slice_assign(copy_slices, tensor);
                Ok(WrappedTensor::Rank2 {
                    tensor,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                let copy_slices = [0..curr_shape[0], 0..curr_shape[1], 0..curr_shape[2]];
                let out = BTensor::full(new_shape, pad_value, &tensor.device());
                let tensor = out.slice_assign(copy_slices, tensor);
                Ok(WrappedTensor::Rank3 {
                    tensor,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                let copy_slices = [
                    0..curr_shape[0],
                    0..curr_shape[1],
                    0..curr_shape[2],
                    0..curr_shape[3],
                ];
                let out = BTensor::full(new_shape, pad_value, &tensor.device());
                let tensor = out.slice_assign(copy_slices, tensor);
                Ok(WrappedTensor::Rank4 {
                    tensor,
                    unpadded_shape,
                })
            }
        }
    }

    /// Pads the tensor to the next power-of-two.
    pub(crate) fn pad_next_power_of_two(self) -> Self {
        let BShape { dims } = self.shape();
        let shape = BShape {
            dims: dims.next_power_of_two(),
        };
        self.pad(shape, T::default())
            .expect("next_power_of_two makes all dimensions equal-or-greater")
    }

    /// Pads a matrix so it can be used with the output of a FFT-based
    /// convolution.
    ///
    /// The FFT-based convolution has all dimensions of the original convolution
    /// padded to the next power of 2. This method performs an equivalent
    /// padding to the a matrix, so it can be multiplied against the FFT based
    /// convolution. This is used to transform a dense layer that comes after a
    /// FFT-convolution, and ensure the vec-matrix multiplication is performed
    /// correctly.
    ///
    /// Given a convolution result `X` and a matrix `M`, the FFT-based
    /// convolution is such that`X' = fft(X)`, here `M` is padded to `M'` such
    /// that `M * X == M' * X'`, ensuring the result remains consistent despite
    /// the padding in `X'`.
    pub(crate) fn pad_matrix_to_ignore_garbage(
        self,
        conv_shape_og: &Shape,
        conv_shape_pad: &Shape,
        matrix_shape_pad: &Shape,
    ) -> Result<Self> {
        let WrappedTensor::Rank2 {
            tensor,
            unpadded_shape,
        } = self
        else {
            bail!("The matrix must be of rank 2");
        };
        ensure!(
            matrix_shape_pad.rank() == 2,
            "The new padded matrix shape must have rank 2",
        );

        ensure!(
            conv_shape_og.rank() == 3 && conv_shape_pad.rank() == 3,
            "The conv2d output shape should be 3d: conv_shape_og: {:?}, conv_shape_pad: {:?}",
            conv_shape_og.rank(),
            conv_shape_pad.rank(),
        );

        // Compute the size of the result and allocate the zero initialized tensor
        ensure!(
            tensor.shape()[1] == conv_shape_og.numel(),
            "The size last dimension of the matrix must match the number of entries in the original conv2d output",
        );
        let mut shape_pad = Vec::from_iter(matrix_shape_pad.iter().cloned());
        shape_pad.remove(1);
        shape_pad.extend(conv_shape_pad.iter().cloned());
        let result = BTensor::full(shape_pad, 0, &tensor.device());

        // Compute the unflattened shape, the copy slices, and unflatten the matrix
        ensure!(
            matrix_shape_pad[1] == conv_shape_pad.numel(),
            "The size last dimension of the padded matrix must match the number of entries in the padded conv2d output",
        );
        let mut shape = tensor.shape();
        shape.remove(1);
        shape.extend(conv_shape_og.iter().cloned());
        let copy_slices: Vec<_> = shape.iter().map(|&v| s![0..v]).collect();
        let tensor = tensor.reshape::<4, _>(shape);

        // Copy the data from the unflatenned tensor to the padded one
        let tensor = result.slice_assign(&copy_slices, tensor);

        // Flatten the result and update the unpadded shape
        let tensor = tensor.reshape::<2, _>(BShape::from(matrix_shape_pad));
        let unpadded_shape = BShape::new([unpadded_shape[0], matrix_shape_pad[1]]);

        Ok(WrappedTensor::Rank2 {
            tensor,
            unpadded_shape,
        })
    }

    pub(crate) fn equal_elem<const D: usize>(
        self,
        elem: T,
    ) -> Result<BTensor<Backend, D, burn::tensor::Bool>> {
        ensure!(
            self.rank() == D,
            "Unexpected tensor rank: {}, expected {}",
            self.rank(),
            D
        );
        let shape = self.shape();
        match D {
            1 => {
                let Self::Rank1 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            2 => {
                let Self::Rank2 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            3 => {
                let Self::Rank3 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            4 => {
                let Self::Rank4 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            _ => unreachable!(),
        }
    }

    /// Transforms this tensor by centering its rows around the mean.
    ///
    /// This has the effect of making the mean of a row equal to zero.
    pub(crate) fn mean_center_rows(self) -> Self {
        let rank = self.rank();
        self.mean_center_dim(rank - 1)
    }

    /// Transforms this tensor by centering its rows around the mean.
    ///
    /// This has the effect of making the mean of a row equal to zero.
    pub(crate) fn mean_center_dim(self, dim: usize) -> Self {
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => {
                let mean = tensor.clone().mean_dim(dim);
                let row_mean = mean.repeat_dim(dim, tensor.shape()[dim]);
                let centered = tensor - row_mean;

                WrappedTensor::Rank1 {
                    tensor: centered,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => {
                let mean = tensor.clone().mean_dim(dim);
                let row_mean = mean.repeat_dim(dim, tensor.shape()[dim]);
                let centered = tensor - row_mean;

                WrappedTensor::Rank2 {
                    tensor: centered,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                let mean = tensor.clone().mean_dim(dim);
                let row_mean = mean.repeat_dim(dim, tensor.shape()[dim]);
                let centered = tensor - row_mean;

                WrappedTensor::Rank3 {
                    tensor: centered,
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                let mean = tensor.clone().mean_dim(dim);
                let row_mean = mean.repeat_dim(dim, tensor.shape()[dim]);
                let centered = tensor - row_mean;

                WrappedTensor::Rank4 {
                    tensor: centered,
                    unpadded_shape,
                }
            }
        }
    }

    pub fn lower_equal_elem<const D: usize>(
        self,
        elem: T,
    ) -> Result<BTensor<Backend, D, burn::tensor::Bool>> {
        anyhow::ensure!(
            self.rank() == D,
            "Unexpected tensor rank: {}, expected {}",
            self.rank(),
            D
        );
        let shape = self.shape();
        match D {
            1 => {
                let Self::Rank1 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.lower_equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            2 => {
                let Self::Rank2 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.lower_equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            3 => {
                let Self::Rank3 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.lower_equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            4 => {
                let Self::Rank4 { tensor, .. } = self else {
                    unreachable!()
                };
                let result = tensor.lower_equal_elem(elem);
                Ok(result.reshape::<D, _>(shape))
            }
            _ => unreachable!(),
        }
    }

    /// Utility to make tests more readable
    #[cfg(test)]
    pub(crate) fn to_native(&self) -> Tensor<T> {
        Tensor::try_from(self.clone()).unwrap()
    }
}

impl WrappedTensor<f32> {
    /// Applies element wise root square operation.
    pub(crate) fn sqrt(self) -> Self {
        delegate!(self, sqrt)
    }

    /// Applies reciprocal operation (or multiplicative inverse) element wise.
    pub(crate) fn recip(self) -> Self {
        delegate!(self, recip)
    }

    /// Applies element wise round operation.
    pub(crate) fn round(self) -> Self {
        delegate!(self, round)
    }

    /// Returns a new tensor with the same shape and device as the current tensor and the data cast to Integer.
    pub(crate) fn int(self) -> WrappedTensor<Element> {
        delegate!(self, int)
    }

    /// Performs exp element-wise on the tensor.
    pub(crate) fn exp(self) -> Self {
        delegate!(self, exp)
    }

    /// Performs ln (natural logarithm) element-wise on the tensor.
    pub(crate) fn log(self) -> Self {
        delegate!(self, log)
    }

    /// Applies the Gaussian Error Linear Units function as described in the paper
    /// [Gaussian Error Linear Units (GELUs)](https://arxiv.org/pdf/1606.08415v3.pdf).
    pub(crate) fn gelu(self) -> Self {
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: burn::tensor::activation::gelu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: burn::tensor::activation::gelu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: burn::tensor::activation::gelu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: burn::tensor::activation::gelu(tensor),
                unpadded_shape,
            },
        }
    }

    /// Applies the Sigmoid Linear Unit (SiLU) function, also known as swish.
    /// SiLU is defined as `x * sigmoid(x)`.
    pub(crate) fn silu(self) -> Self {
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: burn::tensor::activation::silu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: burn::tensor::activation::silu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: burn::tensor::activation::silu(tensor),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: burn::tensor::activation::silu(tensor),
                unpadded_shape,
            },
        }
    }

    pub(crate) fn conv2d(
        self,
        weight: Self,
        bias: Option<Self>,
        options: ConvOptions<2>,
    ) -> Result<Self> {
        let input_rank = self.rank();
        let Self::Rank4 {
            tensor: input,
            unpadded_shape,
        } = self
        else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        let weight_rank = weight.rank();
        let Self::Rank4 { tensor: weight, .. } = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 4.")
        };
        let bias = match bias {
            Some(bias) => {
                let bias_rank = bias.rank();
                let Self::Rank1 { tensor: bias, .. } = bias else {
                    bail!("Unexpected bias rank: {bias_rank}, expected 1.")
                };
                Some(bias)
            }
            None => None,
        };
        let out = burn::tensor::module::conv2d(input, weight, bias, options);
        Ok(WrappedTensor::Rank4 {
            tensor: out,
            unpadded_shape,
        })
    }

    pub(crate) fn max_pool2d(
        self,
        kernel_size: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<Self> {
        let input_rank = self.rank();
        let Self::Rank4 {
            tensor,
            unpadded_shape,
        } = self
        else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        let result =
            burn::tensor::module::max_pool2d(tensor, kernel_size, stride, padding, dilation, false);
        Ok(WrappedTensor::Rank4 {
            tensor: result,
            unpadded_shape,
        })
    }

    pub(crate) fn layer_norm(
        self,
        embedding_size: usize,
        epsilon: f64,
        gamma: Self,
        beta: Self,
    ) -> Result<Self> {
        // NOTE: simply use the burn tensor API for now as we want to move towards using more burn features
        // instead of re-implementing everything ourselves.
        // copy implementation https://docs.rs/burn-core/0.17.0/src/burn_core/nn/norm/layer.rs.html#67
        let gamma_rank = gamma.rank();
        let Self::Rank1 { tensor: gamma, .. } = gamma else {
            bail!("Unexpected gamma rank: {gamma_rank}, expected 1.")
        };
        let beta_rank = beta.rank();
        let Self::Rank1 { tensor: beta, .. } = beta else {
            bail!("Unexpected beta rank: {beta_rank}, expected 1.")
        };
        let config = LayerNormConfig::new(embedding_size).with_epsilon(epsilon);
        let device = Default::default();
        let mut norm = config.init(&device);
        norm.gamma = Param::from_tensor(gamma);
        norm.beta = Some(Param::from_tensor(beta));

        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => {
                let output = norm.forward(tensor);
                Ok(WrappedTensor::Rank1 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => {
                let output = norm.forward(tensor);
                Ok(WrappedTensor::Rank2 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => {
                let output = norm.forward(tensor);
                Ok(WrappedTensor::Rank3 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => {
                let output = norm.forward(tensor);
                Ok(WrappedTensor::Rank4 {
                    tensor: output,
                    unpadded_shape,
                })
            }
        }
    }

    pub fn normalising_factors(input: Self, epsilon: f32) -> Result<Self> {
        match input {
            Self::Rank1 {
                tensor: input,
                unpadded_shape: shape,
            } => {
                let (var, mean) = input.clone().var_mean_bias(0);
                let normalised = input.sub(mean).div(var.add_scalar(epsilon).sqrt());
                Ok(Self::Rank1 {
                    tensor: normalised,
                    unpadded_shape: shape,
                })
            }
            Self::Rank2 {
                tensor: input,
                unpadded_shape: shape,
            } => {
                let (var, mean) = input.clone().var_mean_bias(1);
                let normalised = input.sub(mean).div(var.add_scalar(epsilon).sqrt());
                Ok(Self::Rank2 {
                    tensor: normalised,
                    unpadded_shape: shape,
                })
            }
            Self::Rank3 {
                tensor: input,
                unpadded_shape: shape,
            } => {
                let (var, mean) = input.clone().var_mean_bias(2);
                let normalised = input.sub(mean).div(var.add_scalar(epsilon).sqrt());
                Ok(Self::Rank3 {
                    tensor: normalised,
                    unpadded_shape: shape,
                })
            }
            Self::Rank4 {
                tensor: input,
                unpadded_shape: shape,
            } => {
                let (var, mean) = input.clone().var_mean_bias(3);
                let normalised = input.sub(mean).div(var.add_scalar(epsilon).sqrt());
                Ok(Self::Rank4 {
                    tensor: normalised,
                    unpadded_shape: shape,
                })
            }
        }
    }

    pub(crate) fn softmax(self, dim: usize) -> Result<Self> {
        ensure!(
            dim < self.rank(),
            "Softmax dimension {dim} out of bounds, (tensor rank: {}).",
            self.rank(),
        );
        match self {
            WrappedTensor::Rank1 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank1 {
                tensor: activation::softmax(tensor, dim),
                unpadded_shape,
            }),
            WrappedTensor::Rank2 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank2 {
                tensor: activation::softmax(tensor, dim),
                unpadded_shape,
            }),
            WrappedTensor::Rank3 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank3 {
                tensor: activation::softmax(tensor, dim),
                unpadded_shape,
            }),
            WrappedTensor::Rank4 {
                tensor,
                unpadded_shape,
            } => Ok(WrappedTensor::Rank4 {
                tensor: activation::softmax(tensor, dim),
                unpadded_shape,
            }),
        }
    }

    pub(crate) fn rms_norm_forward(
        self,
        embedding_size: usize,
        epsilon: f64,
        gamma: Option<Self>,
    ) -> Result<Self> {
        let config = RmsNormConfig::new(embedding_size).with_epsilon(epsilon);
        let device = Default::default();
        let norm = if let Some(gamma) = gamma {
            let mut norm = config.init(&device);
            let gamma_rank = gamma.rank();
            let Self::Rank1 { tensor: gamma, .. } = gamma else {
                bail!("Unexpected gamma rank: {gamma_rank}, expected 1.")
            };
            norm.gamma = Param::from_tensor(gamma);
            norm
        } else {
            config.init(&device)
        };

        match self {
            WrappedTensor::Rank1 {
                tensor: input,
                unpadded_shape,
            } => {
                let output = norm.forward(input);
                Ok(WrappedTensor::Rank1 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank2 {
                tensor: input,
                unpadded_shape,
            } => {
                let output = norm.forward(input);
                Ok(WrappedTensor::Rank2 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank3 {
                tensor: input,
                unpadded_shape,
            } => {
                let output = norm.forward(input);
                Ok(WrappedTensor::Rank3 {
                    tensor: output,
                    unpadded_shape,
                })
            }
            WrappedTensor::Rank4 {
                tensor: input,
                unpadded_shape,
            } => {
                let output = norm.forward(input);
                Ok(WrappedTensor::Rank4 {
                    tensor: output,
                    unpadded_shape,
                })
            }
        }
    }

    /// Squares the tensor element-wise.
    pub fn square(self) -> Self {
        delegate!(self, square)
    }
}

impl WrappedTensor<Element> {
    /// Applies the bitwise right shift operation with the scalar.
    pub(crate) fn bitwise_right_shift_scalar(self, other: Element) -> Self {
        delegate!(self, bitwise_right_shift_scalar, other)
    }

    /// Convert the element tensor into a float tensor.
    pub(crate) fn float(self) -> WrappedTensor<f32> {
        delegate!(self, float)
    }

    pub(crate) fn conv2d(self, weight: Self, bias: Self, options: Conv2dConfig) -> Result<Self> {
        let x_rank = self.rank();
        let Self::Rank4 {
            tensor: input,
            unpadded_shape,
        } = self
        else {
            bail!("Unexpected x rank: {x_rank}, expected 4.")
        };
        let weight_rank = weight.rank();
        let Self::Rank4 { tensor: weight, .. } = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 4.")
        };
        let bias_rank = bias.rank();
        let Self::Rank1 { tensor: bias, .. } = bias else {
            bail!("Unexpected bias rank: {bias_rank}, expected 1.")
        };
        let out = zkml_conv2d_i(input, weight, bias, options)?;
        Ok(WrappedTensor::Rank4 {
            tensor: out,
            unpadded_shape,
        })
    }

    pub(crate) fn max_pool2d(self, config: Maxpool2dConfig) -> Result<Self> {
        let input_rank = self.rank();
        let Self::Rank4 {
            tensor: input,
            unpadded_shape,
        } = self
        else {
            bail!("Unexpected input rank: {input_rank}, expected 4.")
        };
        Ok(WrappedTensor::Rank4 {
            tensor: zkml_max_pool2d_i(input, config)?,
            unpadded_shape,
        })
    }
}

impl Quantize for WrappedTensor<f32> {
    type Output = WrappedTensor<Element>;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        let (min, max) = scaling.domain();
        self.clone()
            .div_scalar(scaling.scale())
            .round()
            .int()
            .clamp(min, max)
    }
}

impl Dequantize for WrappedTensor<Element> {
    type Output = WrappedTensor<f32>;
    fn dequantize(&self, scaling: &ScalingFactor) -> Self::Output {
        self.clone().float().mul_scalar(scaling.scale())
    }
}

impl Quantize for WrappedTensor<Element> {
    type Output = WrappedTensor<Element>;

    fn quantize(&self, _scaling: &ScalingFactor) -> Self::Output {
        self.clone()
    }
}

pub trait WrappedModuleFn {
    fn linear(self, weight: Self, bias: Option<Self>) -> Result<Self>
    where
        Self: Sized;

    /// Applies the rectified linear unit function element-wise
    /// as described in the paper [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375).
    fn relu(self) -> Self;
}

impl WrappedModuleFn for WrappedTensor<f32> {
    fn linear(self, weight: Self, bias: Option<Self>) -> Result<Self> {
        let weight_rank = weight.rank();
        let Self::Rank2 { tensor: weight, .. } = weight else {
            bail!("Unexpected weight rank: {weight_rank}, expected 2.")
        };
        let bias = match bias {
            Some(bias) => {
                let bias_rank = bias.rank();
                let Self::Rank1 { tensor: bias, .. } = bias else {
                    bail!("Unexpected bias rank: {bias_rank}, expected 1.")
                };
                Some(bias)
            }
            None => None,
        };
        let out = match self {
            WrappedTensor::Rank1 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: burn::tensor::module::linear(input, weight, bias),
                unpadded_shape,
            },
        };
        Ok(out)
    }

    /// Applies the rectified linear unit function element-wise
    /// as described in the paper [Deep Learning using Rectified Linear Units (ReLU)](https://arxiv.org/pdf/1803.08375).
    fn relu(self) -> Self {
        match self {
            WrappedTensor::Rank1 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank1 {
                tensor: burn::tensor::activation::relu(input),
                unpadded_shape,
            },
            WrappedTensor::Rank2 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank2 {
                tensor: burn::tensor::activation::relu(input),
                unpadded_shape,
            },
            WrappedTensor::Rank3 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank3 {
                tensor: burn::tensor::activation::relu(input),
                unpadded_shape,
            },
            WrappedTensor::Rank4 {
                tensor: input,
                unpadded_shape,
            } => WrappedTensor::Rank4 {
                tensor: burn::tensor::activation::relu(input),
                unpadded_shape,
            },
        }
    }
}

impl WrappedModuleFn for WrappedTensor<Element> {
    fn linear(self, weight: Self, bias: Option<Self>) -> Result<Self> {
        let input = self.unsqueeze_dim(1)?;
        let matmul = weight.transpose().matmul(input)?;
        let matmul = matmul.squeeze(1)?;
        let out = if let Some(bias) = bias {
            matmul.add(bias)?
        } else {
            matmul
        };
        Ok(out)
    }

    fn relu(self) -> Self {
        match self {
            WrappedTensor::Rank1 {
                tensor: input,
                unpadded_shape,
            } => {
                let mask = input.clone().lower_equal_elem(0);

                WrappedTensor::Rank1 {
                    tensor: input.mask_fill(mask, 0),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank2 {
                tensor: input,
                unpadded_shape,
            } => {
                let mask = input.clone().lower_equal_elem(0);
                WrappedTensor::Rank2 {
                    tensor: input.mask_fill(mask, 0),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank3 {
                tensor: input,
                unpadded_shape,
            } => {
                let mask = input.clone().lower_equal_elem(0);
                WrappedTensor::Rank3 {
                    tensor: input.mask_fill(mask, 0),
                    unpadded_shape,
                }
            }
            WrappedTensor::Rank4 {
                tensor: input,
                unpadded_shape,
            } => {
                let mask = input.clone().lower_equal_elem(0);
                WrappedTensor::Rank4 {
                    tensor: input.mask_fill(mask, 0),
                    unpadded_shape,
                }
            }
        }
    }
}

pub trait Conversion {
    fn float(self) -> WrappedTensor<f32>;
}

impl<T: TensorTypeParam> Conversion for WrappedTensor<T> {
    fn float(self) -> WrappedTensor<f32> {
        T::tensor_to_float(self)
    }
}

pub trait IntoBTensor {
    type Kind: BTensorKind<Backend> + BasicOps<Backend> + Numeric<Backend>;

    fn to_btensor<const D: usize>(&self) -> BTensor<Backend, D, Self::Kind>;
    fn into_btensor<const D: usize>(self) -> BTensor<Backend, D, Self::Kind>;
}

impl<T> IntoBTensor for Tensor<T>
where
    T: TensorTypeParam,
{
    type Kind = <T as TensorTypeParam>::Kind;

    fn to_btensor<const D: usize>(&self) -> BTensor<Backend, D, Self::Kind> {
        let shape = self.shape().clone();
        BTensor::from_data(
            TensorData::new(self.data.clone(), shape),
            &Default::default(),
        )
    }

    fn into_btensor<const D: usize>(self) -> BTensor<Backend, D, Self::Kind> {
        let Tensor { data, shape, .. } = self;
        BTensor::from_data(TensorData::new(data, shape), &Default::default())
    }
}

/// Tensor parameter type ([`f32`] or [`Element`]/[`i64`])
pub trait TensorTypeParam:
    burn::tensor::Element + Number + PartialEq + Serialize + for<'de> Deserialize<'de>
{
    /// Burn TensorKind
    type Kind: BTensorKind<Backend> + BasicOps<Backend> + Numeric<Backend>;

    fn tensor_to_float(tensor: WrappedTensor<Self>) -> WrappedTensor<f32>;

    /// Keep dtype dispatch outside the dynamic-rank wrapper. Float and GPU
    /// tensors retain Burn's existing high-level validation and kernels.
    fn matmul<const D: usize>(
        lhs: BTensor<Backend, D, Self::Kind>,
        rhs: BTensor<Backend, D, Self::Kind>,
    ) -> Result<BTensor<Backend, D, Self::Kind>> {
        Ok(lhs.matmul(rhs))
    }
}

impl TensorTypeParam for f32 {
    type Kind = burn::tensor::Float;

    fn tensor_to_float(tensor: WrappedTensor<Self>) -> WrappedTensor<f32> {
        tensor
    }
}

impl TensorTypeParam for Element {
    type Kind = burn::tensor::Int;

    #[cfg(all(feature = "cpu", not(any(feature = "wgpu", feature = "cuda"))))]
    fn matmul<const D: usize>(
        lhs: BTensor<Backend, D, Self::Kind>,
        rhs: BTensor<Backend, D, Self::Kind>,
    ) -> Result<BTensor<Backend, D, Self::Kind>> {
        crate::backend::zkml_matmul_i::<Backend, D>(lhs, rhs)
    }

    fn tensor_to_float(tensor: WrappedTensor<Self>) -> WrappedTensor<f32> {
        tensor.float()
    }
}

impl<T> AsRef<WrappedTensor<T>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn as_ref(&self) -> &WrappedTensor<T> {
        self
    }
}

impl<T> TryFrom<&Tensor<T>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(value: &Tensor<T>) -> Result<Self> {
        value.clone().try_into()
    }
}

impl<T> TryFrom<Tensor<T>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(tensor: Tensor<T>) -> Result<Self> {
        let unpadded_shape = tensor.unpadded_shape().clone().into();

        match tensor.shape().rank() {
            1 => Ok(WrappedTensor::Rank1 {
                tensor: tensor.into_btensor::<1>(),
                unpadded_shape,
            }),
            2 => Ok(WrappedTensor::Rank2 {
                tensor: tensor.into_btensor::<2>(),
                unpadded_shape,
            }),
            3 => Ok(WrappedTensor::Rank3 {
                tensor: tensor.into_btensor::<3>(),
                unpadded_shape,
            }),
            4 => Ok(WrappedTensor::Rank4 {
                tensor: tensor.into_btensor::<4>(),
                unpadded_shape,
            }),
            _ => {
                bail!("Unexpected tensor rank: {}", tensor.shape().rank())
            }
        }
    }
}

impl<T> TryFrom<&WrappedTensor<T>> for Tensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(tensor: &WrappedTensor<T>) -> Result<Self, Self::Error> {
        let shape = tensor.shape().into();
        let data = tensor.get_data();
        let unpadded_shape = tensor.unpadded_shape().into();
        Tensor::<T>::new_with_unpadded_shape(shape, unpadded_shape, data)
    }
}

impl<T> TryFrom<WrappedTensor<T>> for Tensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(tensor: WrappedTensor<T>) -> Result<Self, Self::Error> {
        let shape = tensor.shape().into();
        let unpadded_shape = tensor.unpadded_shape().into();
        let data = tensor.get_data();
        Tensor::<T>::new_with_unpadded_shape(shape, unpadded_shape, data)
    }
}

impl<T> TryFrom<&[T]> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(data: &[T]) -> Result<Self, Self::Error> {
        let data = TensorData::new(data.to_vec(), [data.len()]);
        WrappedTensor::from_data(data)
    }
}

impl<T> TryFrom<Vec<T>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(data: Vec<T>) -> Result<Self, Self::Error> {
        let shape = [data.len()];
        let data = TensorData::new(data, shape);
        WrappedTensor::from_data(data)
    }
}

impl<T> From<WrappedTensor<T>> for Vec<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: WrappedTensor<T>) -> Self {
        tensor.get_data()
    }
}

impl<T> TryFrom<&KeyedTensor<T>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(value: &KeyedTensor<T>) -> Result<Self> {
        value.tensor().try_into()
    }
}

impl<T> From<BTensor<Backend, 1, T::Kind>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: BTensor<Backend, 1, T::Kind>) -> Self {
        let unpadded_shape = tensor.shape().clone();
        WrappedTensor::Rank1 {
            tensor,
            unpadded_shape,
        }
    }
}

impl<T> From<BTensor<Backend, 2, T::Kind>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: BTensor<Backend, 2, T::Kind>) -> Self {
        let unpadded_shape = tensor.shape().clone();
        WrappedTensor::Rank2 {
            tensor,
            unpadded_shape,
        }
    }
}

impl<T> From<BTensor<Backend, 3, T::Kind>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: BTensor<Backend, 3, T::Kind>) -> Self {
        let unpadded_shape = tensor.shape().clone();
        WrappedTensor::Rank3 {
            tensor,
            unpadded_shape,
        }
    }
}

impl<T> From<BTensor<Backend, 4, T::Kind>> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn from(tensor: BTensor<Backend, 4, T::Kind>) -> Self {
        let unpadded_shape = tensor.shape().clone();
        WrappedTensor::Rank4 {
            tensor,
            unpadded_shape,
        }
    }
}

impl<T> TryFrom<TensorData> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    type Error = anyhow::Error;

    fn try_from(data: TensorData) -> Result<Self, Self::Error> {
        match data.shape.len() {
            1 => {
                let tensor = BTensor::from_data(data, &Default::default());
                let unpadded_shape = tensor.shape().clone();
                Ok(WrappedTensor::Rank1 {
                    tensor,
                    unpadded_shape,
                })
            }
            2 => {
                let tensor = BTensor::from_data(data, &Default::default());
                let unpadded_shape = tensor.shape().clone();
                Ok(WrappedTensor::Rank2 {
                    tensor,
                    unpadded_shape,
                })
            }
            3 => {
                let tensor = BTensor::from_data(data, &Default::default());
                let unpadded_shape = tensor.shape().clone();
                Ok(WrappedTensor::Rank3 {
                    tensor,
                    unpadded_shape,
                })
            }
            4 => {
                let tensor = BTensor::from_data(data, &Default::default());
                let unpadded_shape = tensor.shape().clone();
                Ok(WrappedTensor::Rank4 {
                    tensor,
                    unpadded_shape,
                })
            }
            _ => bail!("Unexpected tensor rank: {}", data.shape.len()),
        }
    }
}

impl<T> Iterator for DimIter<T>
where
    T: TensorTypeParam,
{
    type Item = WrappedTensor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DimIter::Rank1 {
                tensor,
                unpadded_shape,
            } => tensor.next().map(|tensor| WrappedTensor::Rank1 {
                tensor,
                unpadded_shape: unpadded_shape.clone(),
            }),
            DimIter::Rank2 {
                tensor,
                unpadded_shape,
            } => tensor.next().map(|tensor| WrappedTensor::Rank2 {
                tensor,
                unpadded_shape: unpadded_shape.clone(),
            }),
            DimIter::Rank3 {
                tensor,
                unpadded_shape,
            } => tensor.next().map(|tensor| WrappedTensor::Rank3 {
                tensor,
                unpadded_shape: unpadded_shape.clone(),
            }),
            DimIter::Rank4 {
                tensor,
                unpadded_shape,
            } => tensor.next().map(|tensor| WrappedTensor::Rank4 {
                tensor,
                unpadded_shape: unpadded_shape.clone(),
            }),
        }
    }
}

impl<T> Serialize for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        Tensor::try_from(self.clone())
            .map_err(serde::ser::Error::custom)?
            .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for WrappedTensor<T>
where
    T: TensorTypeParam,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let tensor: Tensor<T> = Tensor::deserialize(deserializer)?;
        WrappedTensor::try_from(tensor).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    impl<T> WrappedTensor<T>
    where
        T: TensorTypeParam,
    {
        /// Aggregate all elements along the given *dimension* or *axis* in the
        /// tensor with the mean operation.
        pub(crate) fn mean_dim(self, dim: isize) -> Self {
            let mut result = delegate!(self, mean_dim, dim);

            // fix the unpadded shape, apply the transform operation for it too
            let dim = if dim < 0 {
                (result.rank() as isize) + dim
            } else {
                dim
            };
            result.unpadded_shape_mut()[dim as usize] = 1;

            result
        }
    }

    #[test]
    fn test_tensor_next_pow_of_two() {
        let shape = Shape::new(vec![1, 1, 1, 1]);
        let tensor = Tensor::new(shape.clone(), vec![1]).unwrap().into_wrapped();
        assert_eq!(
            tensor.clone().pad_next_power_of_two().get_data(),
            tensor.get_data(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![2, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2])
            .unwrap()
            .into_wrapped();
        assert_eq!(
            tensor.clone().pad_next_power_of_two().get_data(),
            tensor.get_data(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![4, 4]);
        let tensor = WrappedTensor::<Element>::random(&shape.clone());
        assert_eq!(
            tensor.clone().pad_next_power_of_two().get_data(),
            tensor.get_data(),
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![3, 3]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3])
            .unwrap()
            .into_wrapped();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            Shape::from(new_tensor.shape()),
            Shape::new(vec![4, 4]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            &new_tensor.get_data(),
            &[1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0],
            "Tensor padding to next power of two failed."
        );

        let shape = Shape::new(vec![3, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2, 1, 2])
            .unwrap()
            .into_wrapped();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            Shape::from(new_tensor.shape()),
            Shape::new(vec![4, 2]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            &new_tensor.get_data(),
            &[1, 2, 1, 2, 1, 2, 0, 0],
            "Tensor padding to next power of two failed."
        );

        let shape = Shape::new(vec![2, 3, 3]);
        let tensor = Tensor::new(
            shape.clone(),
            vec![
                1, 1, 1, 2, 2, 2, 3, 3, 3, 11, 11, 11, 12, 12, 12, 13, 13, 13,
            ],
        )
        .unwrap()
        .into_wrapped();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            Shape::from(new_tensor.shape()),
            Shape::new(vec![2, 4, 4]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            &new_tensor.get_data(),
            &[
                1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0, 0, 0, 0, 0, 11, 11, 11, 0, 12, 12, 12, 0, 13,
                13, 13, 0, 0, 0, 0, 0,
            ],
            "Tensor padding to next power of two failed."
        );
    }

    #[derive(Debug)]
    struct Slice2d {
        x: usize,
        y: usize,
        start: usize,
        end: usize,
    }

    fn slice_2d_args() -> impl Strategy<Value = Slice2d> {
        (1usize..1024, 1usize..1024)
            .prop_flat_map(|(x, y)| (Just(x), Just(y), 0..x))
            // +1 because end is exclusive
            .prop_flat_map(|(x, y, start)| (Just(x), Just(y), Just(start), start + 1..x + 1))
            .prop_map(|(x, y, start, end)| Slice2d { x, y, start, end })
    }

    proptest! {
        #[test]
        fn proptest_mean_center_rows(col_size in 4usize..20, row_size in 4usize..20, num_cols in 4usize..20) {
            let shape = Shape::new(vec![col_size, row_size]);
            let matrix = Tensor::<f32>::random(&shape);
            let centered = WrappedTensor::try_from(&matrix).unwrap().mean_center_rows();
            let centered_matrix = Tensor::<f32>::try_from(centered).unwrap();

            let input_shape = Shape::new(vec![num_cols, col_size]);
            let input_matrix = Tensor::<f32>::random(&input_shape);

            let centered_result = input_matrix.matmul(&centered_matrix).unwrap();
            let result = input_matrix.matmul(&matrix).unwrap();

            let row_pairs = centered_result.slice_last_dim().zip(result.slice_last_dim());
            for (centered_row, row) in row_pairs {
                let sum = row.iter().sum::<f32>();
                let mean = sum / row.len() as f32;

                for (value_centered, value) in centered_row.iter().zip(row.iter()) {
                    let diff = value_centered - (value - mean);
                    prop_assert!(diff.abs() < 2e-6, "Difference is too large: {diff}");
                }
            }
        }

        #[test]
        fn proptest_mean_center_1d(x in 1usize..=1024) {
            let shape = Shape::new(vec![x]);
            let tensor = Tensor::<f32>::random(&shape);
            let wrapped = WrappedTensor::try_from(tensor).unwrap();

            let centered = wrapped.mean_center_rows();
            let mean = centered.mean_dim(0);
            let max = mean.max_abs().get_data()[0];

            prop_assert!(max < 1e-6);
        }

        #[test]
        fn proptest_mean_center_2d(x in 1usize..=1024, y in 1usize..=1024) {
            let shape = Shape::new(vec![x, y]);
            let tensor = Tensor::<f32>::random(&shape);
            let wrapped = WrappedTensor::try_from(tensor).unwrap();
            let centered = wrapped.mean_center_rows();
            let mean = centered.mean_dim(1);
            let max = mean.max_abs().get_data()[0];

            prop_assert!(max < 1e-6);
        }

        #[test]
        fn proptest_mean_center_3d(x in 1usize..=256, y in 1usize..=256, z in 1usize..=256) {
            let shape = Shape::new(vec![x, y, z]);
            let tensor = Tensor::<f32>::random(&shape);
            let wrapped = WrappedTensor::try_from(tensor).unwrap();
            let centered = wrapped.mean_center_rows();
            let mean = centered.mean_dim(2);
            let max = mean.max_abs().get_data()[0];

            prop_assert!(max < 1e-6);
        }

        #[test]
        fn proptest_quantize(x in 1usize..=1024) {
            let shape = Shape::new(vec![x]);
            let tensor = Tensor::<f32>::random(&shape);
            let wrapped = WrappedTensor::try_from(&tensor).unwrap();

            let scaling = ScalingFactor::from_tensor(&tensor, None);

            let scaled_tensor = tensor.quantize(&scaling);
            let scaled_wrapped = wrapped.quantize(&scaling);

            let to_compare = WrappedTensor::try_from(scaled_tensor).unwrap();
            let diff = scaled_wrapped.sub(to_compare).unwrap().max_abs().get_data()[0];

            prop_assert!(diff == 0, "Wrapped tensor scaling doesnt agree with native Tensor");
        }

        #[test]
        fn proptest_slice_2d(args in slice_2d_args()) {
            let shape = Shape::new(vec![args.x, args.y]);
            let tensor = Tensor::<f32>::random(&shape);
            let wrapped = WrappedTensor::try_from(&tensor).unwrap();

            let tensor = tensor.slice_2d(args.start, args.end).unwrap();
            let wrapped = wrapped.slice_2d(args.start, args.end);

            let to_compare = Tensor::try_from(wrapped).unwrap();

            prop_assert!(tensor == to_compare, "{tensor:?} {to_compare:?}");
        }

        #[test]
        fn proptest_mean_subtracted_tensor(x in 4usize..10, y in 4usize..10, z in 4usize..10, dim in 0usize..3) {
            let shape = Shape::new(vec![x, y, z]);
            let tensor = Tensor::<f32>::random(&shape);

            let wrapped = WrappedTensor::try_from(&tensor).unwrap();
            let centered = wrapped.mean_center_dim(dim);
            let modified_tensor = Tensor::try_from(centered).unwrap();

            let mut indices = vec![0; shape.rank()];
            let total_iters = shape.numel() / shape.dim(dim);
            for _ in 0..total_iters {
                let mut sum = 0.0;
                for i in 0..shape.dim(dim) {
                    indices[dim] = i;
                    sum += tensor.get(&indices).unwrap();
                }
                let mean = sum / shape.dim(dim) as f32;
                for i in 0..shape.dim(dim) {
                    indices[dim] = i;
                    let modified_value = modified_tensor.get(&indices).unwrap();
                    let original_value = tensor.get(&indices).unwrap();
                    let expected_value = original_value - mean;
                    let diff = modified_value - expected_value;
                    assert!(
                        diff.abs() < 1e-6,
                        "Difference is too large at index {:?}: {diff}",
                        indices
                    );
                }
                // Increment indices for next iteration
                for i in (0..shape.rank()).rev() {
                    if i == dim {
                        continue;
                    }
                    indices[i] += 1;
                    if indices[i] < shape.dim(i) {
                        break;
                    } else {
                        indices[i] = 0;
                    }
                }
            }
        }
    }
}
