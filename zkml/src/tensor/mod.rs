#![allow(clippy::needless_range_loop)]

pub use burn_wrapper::{
    BTensorKind, Conversion, IntoBTensor, TensorTypeParam, WrappedModuleFn, WrappedTensor,
};

use crate::{
    NextPowerOfTwo, ScalingFactor, SerializableField,
    layers::convolution,
    number::Number,
    quantization::{Dequantize, Quantize, ToElement},
    shape::Shape,
};
use anyhow::{Result, bail, ensure};
use ark_ff::PrimeField;
use dp_crypto::{IntoMLE, poly::dense::DensePolynomial};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};
use std::{
    cmp::{PartialEq, min},
    fmt::{self, Debug},
};

pub use commitment_id::CommitmentId;
pub use handle::TensorHandle;
pub use keyed::KeyedTensor;
pub use slice::TensorSlice;

use crate::{Element, layers::pooling::MAXPOOL2D_KERNEL_SIZE, quantization::ToField};

mod burn_wrapper;
mod commitment_id;
mod handle;
mod keyed;
mod slice;

/// Chunk size tuned to allow for the compiler's auto vectorisation.
///
/// This value should be a multiple of the target's cache line (taking into
/// consideration the number of bytes, including padding, of the data being
/// iterated over). To support multiple architectures this should be the least
/// common multiple of all target architectures.
///
/// This value is also used to increase utilisation when utilising rayon,
/// reducing the overhead cost of the rayon framework for super cheap operation.
/// (like a simple addition or subtraction).
const AUTO_VECTORISATION_CHUNK: usize = 512;

/// Macro to generate code that can be auto-vectorised over two tensors.
///
/// The auto vectorisation works by:
///
/// - Using `as_chunks`, which guarantees the data will be split in a continuous
///   non-overlapping chunks.
/// - The above chunks are then converted to an array with sizes known at compile
///   time
/// - The data is then iterated over the above arrays using a simple incrementing
///   index
macro_rules! auto_vec_binop {
    ($self: ident, $other: ident, $op: tt) => {{
        assert!(
            $self.shape.product() == $other.shape.product(),
            "Shape mismatch for addition {:?} != {:?}",
            $self.shape,
            $other.shape,
        );

        let mut data = Vec::with_capacity($self.data.len());
        let spare_data = data.spare_capacity_mut();
        assert!(
            spare_data.len() >= $self.data.len(),
            "Preallocated vector must have enough capacity"
        );

        let (left_chunks, left_remainder) = $self.data.as_chunks::<AUTO_VECTORISATION_CHUNK>();
        let (right_chunks, right_remainder) = $other.data.as_chunks::<AUTO_VECTORISATION_CHUNK>();
        let (result_chunks, result_remainder) = spare_data.as_chunks_mut::<AUTO_VECTORISATION_CHUNK>();

        (left_chunks, right_chunks, result_chunks)
            .into_par_iter()
            .for_each(|(left, right, result)| {
                for i in 0..AUTO_VECTORISATION_CHUNK {
                    result[i].write(left[i] $op right[i]);
                }
            });

        // Handle remainder data
        for pos in 0..left_remainder.len()  {
            result_remainder[pos].write(left_remainder[pos] $op right_remainder[pos]);
        }

        // Safety: the memory was initialised above
        unsafe {
            data.set_len($self.data.len());
        }

        Tensor {
            shape: $self.shape.clone(),
            unpadded_shape: $self.unpadded_shape.clone(),
            data,
        }
    }}
}

#[derive(Debug, Clone, Serialize, Deserialize, derive_more::Index, derive_more::IndexMut)]
pub struct Tensor<T> {
    // Indexing the `Tensor` indexes the underlying storage.
    #[index]
    #[index_mut]
    data: Vec<T>,
    shape: Shape,
    unpadded_shape: Shape,
}

impl<T> Tensor<T> {
    /// Create a new tensor with given shape and data
    pub fn new(shape: Shape, data: Vec<T>) -> Result<Self> {
        Self::new_with_unpadded_shape(shape.clone(), shape, data)
    }

    /// Return an immutable reference to this tensor data.
    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    /// Validate dimensions received through serialization before indexing them.
    pub(crate) fn validate_shape(&self) -> Result<()> {
        ensure!(!self.shape.is_empty(), "Tensor shape must not be empty");
        ensure!(
            self.shape.rank() == self.unpadded_shape.rank(),
            "Padded and unpadded tensor ranks differ"
        );
        let size = self.shape.iter().try_fold(1usize, |size, &dim| {
            ensure!(dim > 0, "Tensor dimensions must be positive");
            size.checked_mul(dim)
                .ok_or_else(|| anyhow::anyhow!("Tensor dimensions overflow"))
        })?;
        ensure!(
            size == self.data.len(),
            "Tensor shape does not match data length"
        );
        ensure!(
            self.unpadded_shape
                .iter()
                .zip(self.shape.iter())
                .all(|(&actual, &padded)| actual > 0 && actual <= padded),
            "Invalid unpadded tensor dimensions"
        );
        Ok(())
    }

    /// Returns a reference to the tensor's [Shape].
    pub fn shape(&self) -> &Shape {
        assert!(!self.shape.is_empty(), "Empty tensor");
        &self.shape
    }
}

impl<T> Tensor<T>
where
    T: std::ops::Add<Output = T> + Copy + Send + Sync,
{
    /// Element-wise addition
    pub fn add(&self, other: &Tensor<T>) -> Tensor<T> {
        auto_vec_binop!(self, other, +)
    }
}

impl<T> Tensor<T>
where
    T: std::ops::Mul<Output = T> + Copy + Send + Sync,
{
    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor<T>) -> Tensor<T> {
        auto_vec_binop!(self, other, *)
    }
}

impl<T> Tensor<T> {
    /// Create a new tensor with given shape, unpadded shape and data
    pub(crate) fn new_with_unpadded_shape(
        shape: Shape,
        unpadded_shape: Shape,
        data: Vec<T>,
    ) -> Result<Self> {
        ensure!(
            shape.product() == data.len(),
            "Shape does not match data length: shape {:?}->{} vs data.len() {}",
            shape,
            shape.product(),
            data.len(),
        );
        Ok(Self {
            data,
            shape,
            unpadded_shape,
        })
    }

    /// Create a new tensor with the given shapes & data, not ensuring that they
    /// actually match.
    pub(crate) fn new_unchecked(shape: Shape, data: Vec<T>) -> Self {
        Self {
            data,
            shape: shape.clone(),
            unpadded_shape: shape,
        }
    }

    /// Consume this tensore, returning its backing.
    pub(crate) fn into_data(self) -> Vec<T> {
        self.data
    }

    /// Returns an iterator that yields slices of the last dimension.
    ///
    /// For a tensor of shape `[2, 3, 3]`, it will yield 6 slices `2 * 3` of 3 elements each.
    pub(crate) fn slice_last_dim(&self) -> impl Iterator<Item = &[T]> {
        let (it, _) = self.slice_on_dim(self.shape.len() - 2);
        it
    }

    /// Returns an iterator of slices whose length corresponds to the subspace
    /// the dimension represents. Note dim is the dimension _index_ (0-based indexing).
    /// Example: if dimension is [2,3,4], and we call `slice_on_dim(1)`,
    /// it will yield 2x3 slices of 4 elements each. If we call `slice_on_dim(0)`,
    /// it will yield 2 slices of 3x4=12 element each.
    /// If dim is the last dimension, it will simply yield a slice of the whole tensor.
    /// The shape returned is the shape of each slice. The shape is the same as the shape of the tensor
    /// if the dim is the last dimension or more
    pub(crate) fn slice_on_dim(&self, dim: usize) -> (impl Iterator<Item = &[T]>, Shape) {
        assert!(
            dim < self.shape.len(),
            "can't slice on dim {:?} if shape is {:?}",
            dim,
            self.shape
        );
        let (stride, shape) = if dim < self.shape.len() - 1 {
            let slice = self.shape.slice(dim + 1..);
            (slice.product(), slice)
        } else {
            (self.shape.product(), self.shape.clone())
        };
        (self.data.chunks(stride), shape)
    }

    /// Get the number of rows from the matrix
    pub(crate) fn nrows_2d(&self) -> Result<usize> {
        Ok(self.shape().nrows_2d())
    }

    /// Get the number of cols from the matrix
    pub(crate) fn ncols_2d(&self) -> Result<usize> {
        Ok(self.shape().ncols_2d())
    }

    /// Returns a reference to the tensor's original unpadded [Shape].
    pub fn unpadded_shape(&self) -> &Shape {
        &self.unpadded_shape
    }

    /// Returns a mutable reference to the tensor's [Shape].
    pub(crate) fn shape_mut(&mut self) -> &mut Shape {
        &mut self.shape
    }

    /// Returns the size of the given dimension.
    ///
    /// # Panics
    ///
    /// If the shape rank is lower-or-equal to `dim`.
    pub(crate) fn dim(&self, dim: usize) -> usize {
        self.shape[dim]
    }

    pub(crate) fn truncate_on_first_dim(&mut self, new_size: usize) -> anyhow::Result<()> {
        let shape = self.shape();
        let first_dim = shape[0];
        ensure!(
            new_size <= first_dim,
            "Truncating to {new_size} while first dimension is {first_dim}"
        );
        let stride = shape.strides()[0];
        // retain `new_size` strides of the tensor
        self.data.truncate(new_size * stride);
        self.shape[0] = new_size;
        self.unpadded_shape[0] = new_size;
        Ok(())
    }

    /// Reshape the tensor, ensuring that the new shape is compatible with the
    /// tensor cardinality.
    pub(crate) fn reshape(&mut self, new_shape: Shape) -> Result<()> {
        ensure!(
            self.shape.product() == new_shape.product(),
            "Shape mismatch for reshape: current {:?} ({}), required {:?} ({})",
            self.shape,
            self.shape.product(),
            new_shape,
            new_shape.product()
        );
        // If the tensor has not yet been padded also update the unpadded shape
        if self.shape == self.unpadded_shape {
            self.unpadded_shape = new_shape.clone();
        }
        self.shape = new_shape;

        Ok(())
    }

    /// Chainable version of [`reshape`]
    pub(crate) fn reshaped(mut self, new_shape: Shape) -> Result<Tensor<T>> {
        self.reshape(new_shape)?;
        Ok(self)
    }
}

impl<T> AsRef<Tensor<T>> for Tensor<T> {
    fn as_ref(&self) -> &Tensor<T> {
        self
    }
}

impl Tensor<Element> {
    /// Converts this [Tensor<Element>] into [DensePolynomial].
    ///
    /// This will convert the element into an extension field and convert that
    /// into a multilinear extension.
    ///
    /// see [Tensor::into_mle_2d].
    ///
    /// # Panics
    ///
    /// If the input is not a 2D tensor or if either dimension is not a power of two.
    pub(crate) fn to_2d_mle<F: PrimeField>(&self) -> Result<DensePolynomial<'static, F>> {
        Tensor::<F>::from(self).into_mle_2d()
    }

    /// Converts this [Tensor<Element>] into a [DensePolynomial].
    ///
    /// This will convert the element into an extension field and convert that
    /// into a multilinear extension.
    ///
    /// This method does not enforce a dimensionality.
    ///
    /// see [Tensor::into_mle].
    ///
    /// # Panics
    ///
    /// If the number of elements in the Tensor is not a power of two.
    pub(crate) fn to_field_mle<F: PrimeField>(&self) -> DensePolynomial<'static, F> {
        Tensor::<F>::from(self).into_mle()
    }
}

impl<F> Tensor<F>
where
    F: PrimeField,
{
    /// Clone this [Tensor] and convert into a [DensePolynomial].
    ///
    /// see [Tensor::into_mle_2d].
    pub(crate) fn to_mle_2d(&self) -> Result<DensePolynomial<'static, F>> {
        self.clone().into_mle_2d()
    }

    /// Consumes this [Tensor] into a [DensePolynomial].
    ///
    /// The [Tensor] must be in evaluation form.
    ///
    /// # Panics
    ///
    /// - If the tensor is not 2D.
    /// - If either dimension is not a power-of-two.
    pub(crate) fn into_mle_2d(self) -> Result<DensePolynomial<'static, F>> {
        ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
        ensure!(
            self.nrows_2d()?.is_power_of_two(),
            "number of rows {} is not a power of two",
            self.nrows_2d()?
        );
        ensure!(
            self.ncols_2d()?.is_power_of_two(),
            "number of columns {} is not a power of two",
            self.ncols_2d()?
        );
        Ok(DensePolynomial::new(self.data))
    }
}

impl<F: PrimeField> From<Tensor<F>> for Tensor<SerializableField<F>> {
    fn from(value: Tensor<F>) -> Self {
        Tensor {
            data: value
                .data
                .into_iter()
                .map(|x| SerializableField(x))
                .collect(),
            shape: value.shape,
            unpadded_shape: value.unpadded_shape,
        }
    }
}

impl<F> Tensor<F>
where
    F: PrimeField,
{
    /// Clone this [Tensor] and convert into a [DensePolynomial].
    ///
    /// see [Tensor::into_mle].
    pub(crate) fn to_mle(&self) -> DensePolynomial<'static, F> {
        self.data.clone().into_mle()
    }

    /// Consumes this [Tensor] into a [MultilinearExtension].
    ///
    /// The [Tensor] must be in evaluation form.
    ///
    /// # Panics
    ///
    /// - If the number of elements in the tensor is not a power of two.
    pub(crate) fn into_mle(self) -> DensePolynomial<'static, F> {
        self.data.into_mle()
    }
}

impl<F: ToElement> Tensor<F> {
    pub(crate) fn to_element(&self) -> Tensor<Element> {
        let data = self.data.iter().map(ToElement::to_element).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }
}

impl<F> From<&Tensor<Element>> for Tensor<F>
where
    F: PrimeField,
{
    fn from(value: &Tensor<Element>) -> Self {
        value.to_field()
    }
}

impl<T> Tensor<T>
where
    T: Clone,
{
    /// Create a copy of the data and flatten the tensor's shape.
    pub(crate) fn to_flatten(&self) -> Self {
        let new_data = self.data.to_vec();
        let new_shape = vec![new_data.len()];
        Self {
            data: new_data,
            shape: new_shape.into(),
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }
}

impl<T: Number> Tensor<T> {
    /// Instantiate a new tensor with `shape` initialised to `default`.
    pub(crate) fn zeros(shape: Shape) -> Self {
        Self::initialised(shape, T::zero())
    }

    /// Creates a new [Tensor] with `shape` initialised to `T::unit`.
    ///
    /// ```rust
    /// # use zkml::{Tensor, Shape, Element};
    /// let shape = Shape::new(vec![2, 2]);
    /// let tensor = Tensor::<Element>::one(shape);
    /// assert_eq!(tensor.data().as_slice(), [1, 1, 1, 1]);
    /// ```
    pub fn one(shape: Shape) -> Self {
        Tensor {
            data: vec![T::unit(); shape.numel()],
            shape: shape.clone(),
            unpadded_shape: shape,
        }
    }

    /// Returns the largest value in the [Tensor].
    ///
    /// ```rust
    /// # use zkml::{Tensor, Shape, Element};
    /// let tensor = Tensor::<Element>::new(Shape::new(vec![4, 2]), vec![3, 1, 0, 11, 7, 11, 9, 2]).unwrap();
    /// assert_eq!(tensor.max(), 11);
    /// ```
    pub fn max(&self) -> T {
        self.data.par_iter().cloned().reduce(
            || T::MIN,
            |acc, value| match acc.compare(&value) {
                std::cmp::Ordering::Less => value,
                _ => acc,
            },
        )
    }

    /// Returns the the largest absolute element in the [Tensor].
    ///
    /// ```rust
    /// # use zkml::{Tensor, Shape, Element};
    /// let tensor = Tensor::<Element>::new(Shape::new(vec![7]), vec![3, 1, 0, -11, 7, 9,
    /// 2]).unwrap();
    /// assert_eq!(tensor.max_abs(), 11);
    /// ```
    pub fn max_abs(&self) -> T {
        self.data
            .par_iter()
            .cloned()
            .reduce(|| T::zero(), |max, x| max.cmp_max(&x.absolute_value()))
    }

    /// Transpose the matrix (2D tensor)
    pub(crate) fn transpose(&self) -> Result<Tensor<T>> {
        ensure!(self.shape.is_matrix(), "Tensor is not a matrix.");
        let (m, n) = (self.shape[0], self.shape[1]);

        let mut result = Tensor::zeros(vec![n, m].into());
        result
            .data
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, val)| {
                let i = idx % m; // Row in the result matrix
                let j = idx / m; // Column in the result matrix
                *val = self.data[i * n + j];
            });

        Ok(result)
    }

    pub(crate) fn maxpool2d(&self, kernel_size: usize, stride: usize) -> Result<Tensor<T>> {
        ensure!(
            kernel_size == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for kernel size {MAXPOOL2D_KERNEL_SIZE}"
        );
        ensure!(
            stride == MAXPOOL2D_KERNEL_SIZE,
            "Maxpool2D works only for stride size {MAXPOOL2D_KERNEL_SIZE}"
        );

        let dims = self.shape().rank();
        ensure!(dims >= 2, "Input tensor must have at least 2 dimensions.");

        let (h, w) = (self.shape[dims - 2], self.shape[dims - 1]);

        // https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        // Assumes dilation = 1
        ensure!(
            h >= kernel_size,
            "Kernel size ({kernel_size}) is larger than input dimensions ({h}, {w})"
        );
        let out_h = (h - kernel_size) / stride + 1;
        let out_w = (w - kernel_size) / stride + 1;

        let outer_dims: usize = self.shape[..dims - 2].iter().product();
        let output: Vec<T> = (0..outer_dims * out_h * out_w)
            .into_par_iter()
            .map(|flat_idx| {
                let n = flat_idx / (out_h * out_w);
                let i = (flat_idx / out_w) % out_h;
                let j = flat_idx % out_w;

                let matrix_idx = n * (h * w);
                let src_idx = matrix_idx + (i * stride) * w + (j * stride);
                let mut max_val = self.data[src_idx];

                for ki in 0..kernel_size {
                    for kj in 0..kernel_size {
                        let src_idx = matrix_idx + (i * stride + ki) * w + (j * stride + kj);
                        let value = self.data[src_idx];
                        max_val = max_val.cmp_max(&value);
                    }
                }

                max_val
            })
            .collect();

        let mut new_shape = self.shape.clone();
        new_shape[dims - 2] = out_h;
        new_shape[dims - 1] = out_w;

        Ok(Tensor {
            data: output,
            shape: new_shape,
            unpadded_shape: self.unpadded_shape.clone(),
        })
    }

    /// Applies a 2-dimensional convolution.
    pub(crate) fn conv2d(
        &self,
        kernels: &Tensor<T>,
        bias: &Tensor<T>,
        stride: usize,
    ) -> Result<Tensor<T>> {
        convolution::conv2d(self, kernels, bias, stride)
    }

    /// Converts this [Tensor] to f32.
    pub fn to_f32(&self) -> anyhow::Result<Tensor<f32>> {
        Ok(Tensor {
            data: self
                .data
                .iter()
                .map(Number::to_f32)
                .collect::<anyhow::Result<Vec<_>>>()?,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        })
    }

    /// Creates a diagonal matrix of given size with the given value on the diagonal.
    pub fn diagonal(size: usize, value: T) -> Tensor<T> {
        let mut data = vec![T::default(); size * size];
        for i in 0..size {
            data[i * size + i] = value;
        }
        Tensor {
            data,
            shape: vec![size, size].into(),
            unpadded_shape: vec![size, size].into(),
        }
    }

    pub fn min_value(&self) -> T {
        self.data.iter().fold(T::MAX, |min, x| min.cmp_min(x))
    }

    pub fn try_map<F: Fn(&T) -> anyhow::Result<T>>(&self, f: F) -> anyhow::Result<Self> {
        Ok(Self {
            data: self
                .data
                .iter()
                .map(f)
                .collect::<anyhow::Result<Vec<_>>>()?,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        })
    }

    pub fn random(shape: &Shape) -> Self {
        Self::random_seed(shape, Some(crate::seed_from_env_or_rng()))
    }

    /// Creates a random matrix with a given number of rows and cols.
    /// NOTE: doesn't take a rng as argument because to generate it in parallel it needs be sync +
    /// sync which is not true for basic rng core.
    pub(crate) fn random_seed(shape: &Shape, seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(crate::seed_from_env_or_rng); // Use provided seed or default
        let mut rng = <crate::StdRng as ark_std::rand::SeedableRng>::seed_from_u64(seed);
        let size = shape.product();
        let data = (0..size).map(|_| T::random(&mut rng)).collect();

        Self {
            data,
            shape: shape.clone(),
            unpadded_shape: shape.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy,
{
    /// Instantiate a new tensor with `shape` initialised to `default`.
    pub(crate) fn initialised(shape: Shape, default: T) -> Self {
        Tensor {
            data: vec![default; shape.numel()],
            shape: shape.clone(),
            unpadded_shape: shape,
        }
    }

    /// Retrieves an element using (N, C, H, W) indexing
    pub(crate) fn get_at_4d(&self, n: usize, c: usize, h: usize, w: usize) -> Result<T> {
        ensure!(self.shape.len() <= 4);

        let shape = self.shape.to_4d();
        let idx: usize = [n, c, h, w]
            .iter()
            .zip(shape.strides())
            .map(|(pos, stride)| pos * stride)
            .sum();

        Ok(self.data[idx])
    }
}

impl<T> fmt::Display for Tensor<T>
where
    T: std::fmt::Debug + std::fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut shape = self.shape.clone();

        while shape.len() < 4 {
            shape.reverse();
            shape.push(1);
            shape.reverse();
        }

        if shape.len() == 4 {
            let (batches, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
            let channel_size = height * width;
            let batch_size = channels * channel_size;

            for b in 0..batches {
                writeln!(f, "Batch {b} [{channels} channels, {height}x{width}]:")?;
                for c in 0..channels {
                    writeln!(f, "  Channel {c}:")?;
                    let offset = b * batch_size + c * channel_size;
                    for i in 0..height {
                        let row_start = offset + i * width;
                        let row_data: Vec<String> = (0..width)
                            .map(|j| format!("{:>4.2}", self.data[row_start + j]))
                            .collect();
                        writeln!(f, "    {:>3}: [{}]", i, row_data.join(", "))?;
                    }
                }
            }
            write!(f, "Shape: {:?}", self.shape)
        } else {
            write!(f, "Tensor(shape={:?}, data={:?})", self.shape, self.data) // Fallback
        }
    }
}

impl<T: PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<T> Quantize for Tensor<T>
where
    T: Quantize,
{
    type Output = Tensor<<T as Quantize>::Output>;

    fn quantize(&self, scaling: &ScalingFactor) -> Self::Output {
        Tensor {
            data: self.data.quantize(scaling),
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }
}

impl<T> Dequantize for Tensor<T>
where
    T: Dequantize,
{
    type Output = Tensor<<T as Dequantize>::Output>;

    fn dequantize(&self, scaling: &ScalingFactor) -> Self::Output {
        let data = self.data.dequantize(scaling);
        Tensor {
            shape: self.shape.clone(),
            data,
            unpadded_shape: self.unpadded_shape.clone(),
        }
    }
}

impl<T> Tensor<T>
where
    T: Copy + Default,
{
    /// Copies the sub-slice `new_shape` from this tensor.
    ///
    /// Returns a new [Tensor] with shape `new_shape` initialised from `self`.
    pub(crate) fn reduce_to_shape(&self, new_shape: &Shape) -> Result<Self> {
        ensure!(
            self.shape().rank() >= new_shape.rank(),
            "The target shape must be smaller than the current",
        );
        ensure!(
            self.shape()
                .iter()
                .rev()
                .zip(new_shape.iter().rev())
                .all(|(from, to)| from >= to),
            "The target dimensions can not be larger than the current",
        );
        // current position being copied
        let mut coord: Vec<_> = vec![0; new_shape.len()];
        // number of copies is equal to the number of elements in the target shape
        let mut copies = new_shape.numel();
        // auxiliary variable to convert from target to source coordinates
        let source_strides = self.shape.strides();
        let target_strides = new_shape.strides();

        let mut result = Tensor::initialised(new_shape.clone(), T::default());

        while copies > 0 {
            let source: usize = coord
                .iter()
                .zip(source_strides.iter())
                .map(|(pos, source)| pos * source)
                .sum();
            let dest: usize = coord
                .iter()
                .zip(target_strides.iter())
                .map(|(pos, target)| pos * target)
                .sum();

            result.data[dest] = self.data[source];
            copies -= 1;

            let mut rank = coord.len();
            while rank > 0 {
                rank -= 1;
                coord[rank] += 1;

                if coord[rank] == new_shape[rank] {
                    coord[rank] = 0;
                } else {
                    break;
                }
            }
        }

        Ok(result)
    }

    /// Pads the tensor to the next power-of-two.
    pub(crate) fn pad_next_power_of_two(&self) -> Self {
        self.pad_next_power_of_two_with_value(T::default())
    }

    /// Pads the tensor to the next power-of-two using the specified value for padding.
    pub(crate) fn pad_next_power_of_two_with_value(&self, value: T) -> Self {
        let new_shape = self.shape().next_power_of_two();

        // Pre allocate the necessary capacity
        let mut new_data = Vec::with_capacity(new_shape.numel());
        new_data.extend(&self.data);

        // Create the new tensor and use the in-place implementation to change the shape
        let mut new_tensor = Tensor {
            data: new_data,
            shape: self.shape.clone(),
            unpadded_shape: self.unpadded_shape.clone(),
        };
        new_tensor.pad_to_shape_with_value(new_shape, value).expect(
            "padding within pad_to_next_power_of_two always \
            creates a new shape of correct rank; qed",
        );

        new_tensor
    }

    /// Changes the shape of the current [Tensor] to `target_shape`.
    ///
    /// This method will modify the current tensor in place, extending it
    /// to comply with the new shape.
    ///
    /// # Panics
    ///
    /// If the `target_shape` differs in rank or has a dimension smaller than
    /// the current tensor.
    pub(crate) fn pad_to_shape(&mut self, target_shape: Shape) -> Result<()> {
        self.pad_to_shape_with_value(target_shape, T::default())
    }

    /// Changes the shape of the current [Tensor] to `target_shape` by padding with the specified value.
    ///
    /// This method will modify the current tensor in place, extending it
    /// to comply with the new shape.
    ///
    /// # Panics
    ///
    /// If the `target_shape` differs in rank or has a dimension smaller than
    /// the current tensor.
    pub(crate) fn pad_to_shape_with_value(&mut self, target_shape: Shape, value: T) -> Result<()> {
        ensure!(
            target_shape.rank() == self.shape.rank(),
            "Target shape must have the same rank as the current tensor. current {:?} target {:?}",
            self.shape(),
            target_shape,
        );

        if self.shape == target_shape {
            return Ok(());
        }

        let distance = self
            .shape
            .iter()
            .zip(target_shape.iter())
            .map(|(original, new)| new.checked_sub(*original))
            .collect::<Option<Vec<usize>>>();

        let Some(distance) = distance else {
            bail!(
                "All dimensions of target shape must be greater-than-or-equal to the current tensor",
            );
        };

        debug_assert!(
            distance.iter().any(|v| *v != 0),
            "At least one dimensions must grow",
        );

        // At this point, the target_shape is known to be strictly bigger than the current shape.
        // And at least one of the dimensions has a non-zero distance

        // First expand the underlying storage vector to the new size
        self.data.resize(target_shape.product(), value);

        // Walks the shapes in reverse and count number of equal dimensions, if any.
        //
        // This determines the copy chunk size, when the dimensions don't
        // change no padding is necessary, so the data can be chunked together.
        let equal = distance.iter().rev().take_while(|v| **v == 0).count();
        let different = distance.len() - equal;
        debug_assert!(
            equal != target_shape.len(),
            "At least one dimensions must grown",
        );

        let strides = target_shape.strides();
        debug_assert_eq!(
            &strides[different..],
            &self.shape.strides()[different..],
            "The lower strides must be equal",
        );

        // Difference in size for a given dimension, i.e. how many empty spaces
        // are in between the dimensions after re-shaping.
        //
        // NOTE: this is non cumulative, each dimension padding is applied
        // separately.
        let mut padding = distance
            .iter()
            .zip(strides.iter())
            .map(|(distance, new)| distance * new)
            .take(different)
            .collect::<Vec<_>>();

        // the top-most dimension padding is implicit. This pop aligns the dimensions to remove off-by-one
        padding.remove(0);

        // Compute the chunk size for each copy
        let chunk = self
            .shape
            .iter()
            .rev()
            .take(equal + 1) // at least the last dimension can be copied in batches
            .product::<usize>();

        // The old data is being copied, so the number of copies is equal to the old shape
        let mut counters = self.shape.clone();
        debug_assert!(counters.len() > different - 1, "rank must not increase");
        counters.resize(different - 1, 0);

        // -1 because the very first entries are in the correct spot and don't need to be copied
        let mut copies = counters.iter().product::<usize>() - 1;

        let mut dest = target_shape.product();
        let mut end_pos = self.shape.product();

        // account for padding at the end
        dest -= distance
            .iter()
            .zip(strides)
            .map(|(dist, stride)| dist * stride)
            .sum::<usize>();

        // Copy original data into the new position, back to front to avoid overwrites
        while copies > 0 {
            copies -= 1;

            let start_pos = end_pos - chunk;
            dest -= chunk;

            self.data.copy_within(start_pos..end_pos, dest); // copy data to new position
            self.data[start_pos..min(end_pos, dest)].fill(value); // write the padding

            end_pos = start_pos;

            // Compute the new destination taking padding into consideration
            let mut dim = counters.len() - 1;
            loop {
                dest -= padding[dim];
                counters[dim] -= 1;

                if counters[dim] == 0 {
                    counters[dim] = self.shape[dim];
                    dim -= 1;
                } else {
                    break;
                }
            }
        }

        self.shape = target_shape;

        Ok(())
    }
}

/// Determines whether two slices of `f32` values are element-wise close within
/// the specified absolute (`atol`) and relative (`rtol`) tolerances.
///
/// The condition checked is the same as PyTorch's `torch.isclose`:
/// `|a - b| <= atol + rtol * |b|` for every corresponding element.
pub(crate) fn is_close_with_tolerance(a: &[f32], b: &[f32], atol: f32, rtol: f32) -> bool {
    // taken from https://docs.pytorch.org/docs/stable/generated/torch.isclose.html
    if a.len() != b.len() {
        return false;
    }

    a.iter().zip(b.iter()).all(|(x, y)| {
        let diff = (*x - *y).abs();
        diff <= atol + rtol * y.abs()
    })
}

/// Backwards-compatible wrapper that uses the historical default tolerances
/// (`atol = 1e-8`, `rtol = 1e-5`).
pub(crate) fn is_close(a: &[f32], b: &[f32]) -> bool {
    is_close_with_tolerance(a, b, 1e-8_f32, 1e-5_f32)
}

#[cfg(test)]
mod test {
    use std::cmp::Ordering;

    use ark_std::rand::Rng;
    use itertools::Itertools;
    use ndarray::{Array, Ix2, Order};
    use rayon::slice::{ParallelSlice, ParallelSliceMut};

    type F = ark_bn254::Fr;

    use crate::{
        rng_from_env_or_random, testing::random_field_vector, to_bit_sequence_le, to_field,
    };

    use super::*;
    use proptest::prelude::*;

    /// Macro to apply an operation to each element of an array.
    ///
    /// The expanded code is intended to be easily vectorisable, by using the
    /// `as_chunks` api.
    macro_rules! auto_vec_op {
        ($data: expr, $op: expr) => {{
            let mut result = Vec::with_capacity($data.len());
            let spare_result = result.spare_capacity_mut();
            assert!(
                spare_result.len() >= $data.len(),
                "Preallocated vector must have enough capacity"
            );

            let (left_chunks, left_remainder) = $data.as_chunks::<AUTO_VECTORISATION_CHUNK>();
            let (result_chunks, result_remainder) =
                spare_result.as_chunks_mut::<AUTO_VECTORISATION_CHUNK>();

            (left_chunks, result_chunks)
                .into_par_iter()
                .for_each(|(left, result)| {
                    for i in 0..AUTO_VECTORISATION_CHUNK {
                        result[i].write($op(left[i]));
                    }
                });

            // Handle remainder data
            for pos in 0..left_remainder.len() {
                result_remainder[pos].write($op(left_remainder[pos]));
            }

            // Safety: the memory was initialised above
            unsafe {
                result.set_len($data.len());
            }

            result
        }};
    }

    impl<T: Number> Tensor<T> {
        /// Returns the first position of the largest element in the [Tensor].
        ///
        /// ```rust
        /// # use zkml::{Tensor, Shape, Element};
        /// let tensor = Tensor::<Element>::new(Shape::new(vec![4, 2]), vec![3, 1, 0, 11, 7, 11, 9, 2]).unwrap();
        /// assert_eq!(tensor.argmax(), 3);
        /// ```
        pub(crate) fn argmax(&self) -> usize {
            let (position, _value) = self.data.par_iter().cloned().enumerate().reduce(
                || (usize::MAX, T::MIN),
                |acc, (position, value)| match acc.1.compare(&value) {
                    Ordering::Less => (position, value),
                    _ => acc,
                },
            );

            position
        }

        /// Perform matrix-matrix multiplication
        pub(crate) fn matmul(&self, other: &Tensor<T>) -> Result<Tensor<T>> {
            ensure!(
                self.shape.is_matrix() && other.shape.is_matrix(),
                "Both tensors must be 2D for matrix multiplication."
            );
            let (m, n) = (self.shape[0], self.shape[1]);
            let (n2, p) = (other.shape[0], other.shape[1]);
            ensure!(
                n == n2,
                "Matrix multiplication shape mismatch: {:?} cannot be multiplied with {:?}",
                self.shape,
                other.shape
            );

            let mut result = Tensor::zeros(vec![m, p].into());

            result
                .data
                .par_iter_mut()
                .enumerate()
                .for_each(|(index, res)| {
                    let i = index / p;
                    let j = index % p;

                    *res = (0..n)
                        .map(|k| self.data[i * n + k] * other.data[k * p + j])
                        .sum::<T>();
                });

            Ok(result)
        }

        /// Perform matrix-vector multiplication
        pub(crate) fn matvec(&self, vector: &Tensor<T>) -> Result<Tensor<T>> {
            ensure!(self.shape.is_matrix(), "First argument must be a matrix.");
            ensure!(
                vector.shape.is_vector(),
                "Second argument must be a vector."
            );

            let (m, n) = (self.shape[0], self.shape[1]);
            let vec_len = vector.shape[0];

            ensure!(n == vec_len, "Matrix columns must match vector size.");

            let mut result = Tensor::zeros(vec![m].into());

            result.data.par_iter_mut().enumerate().for_each(|(i, res)| {
                *res = (0..n)
                    .map(|j| self.data[i * n + j] * vector.data[j])
                    .sum::<T>();
            });

            Ok(result)
        }

        /// Concatenate a matrix (2D tensor) with a vector (1D tensor) as columns
        pub(crate) fn concat_matvec_col(&self, vector: &Tensor<T>) -> Result<Tensor<T>> {
            ensure!(self.shape.is_matrix(), "First tensor is not a matrix.");
            ensure!(vector.shape.is_vector(), "Second tensor is not a vector.");

            let (rows, cols) = (self.shape[0], self.shape[1]);
            let vector_len = vector.shape[0];

            ensure!(
                rows == vector_len,
                "Matrix row count must match vector length."
            );

            let new_cols = cols + 1;
            let mut result = Tensor::zeros(vec![rows, new_cols].into());

            result
                .data
                .par_chunks_mut(new_cols)
                .enumerate()
                .for_each(|(i, row)| {
                    row[..cols].copy_from_slice(&self.data[i * cols..(i + 1) * cols]); // Copy matrix row
                    row[cols] = vector.data[i]; // Append vector element as the last column
                });

            Ok(result)
        }

        /// Replaces every value of a tensor with the maxpool of its kernel
        pub(crate) fn padded_maxpool2d(&self) -> Result<(Tensor<T>, Tensor<T>)> {
            let kernel_size = MAXPOOL2D_KERNEL_SIZE;
            let stride = MAXPOOL2D_KERNEL_SIZE;

            let maxpool_result = self.maxpool2d(kernel_size, stride)?;

            let dims: usize = self.shape().rank();
            ensure!(dims >= 2, "Input tensor must have at least 2 dimensions.");

            let (h, w) = (self.shape[dims - 2], self.shape[dims - 1]);

            ensure!(
                h % MAXPOOL2D_KERNEL_SIZE == 0,
                "Currently works only with kernel size {MAXPOOL2D_KERNEL_SIZE}"
            );
            ensure!(
                w % MAXPOOL2D_KERNEL_SIZE == 0,
                "Currently works only with stride size {MAXPOOL2D_KERNEL_SIZE}"
            );

            let outer_dims: usize = self.shape[..dims - 2].iter().product();
            let maxpool_h = (h - kernel_size) / stride + 1;
            let maxpool_w = (w - kernel_size) / stride + 1;

            let padded_maxpool_data: Vec<T> = (0..outer_dims * h * w)
                .into_par_iter()
                .map(|out_idx| {
                    let n = out_idx / (h * w);
                    let i_full = (out_idx / w) % h;
                    let j_full = out_idx % w;

                    let i = i_full / stride;
                    let j = j_full / stride;

                    let maxpool_idx = n * maxpool_h * maxpool_w + i * maxpool_w + j;
                    maxpool_result.data[maxpool_idx]
                })
                .collect();

            let padded_maxpool_tensor = Tensor {
                data: padded_maxpool_data,
                shape: self.shape().clone(),
                unpadded_shape: self.unpadded_shape.clone(),
            };

            Ok((self.maxpool2d(kernel_size, stride)?, padded_maxpool_tensor))
        }

        /// Makes a [`Tensor`] that is a batch of lower triangular matrices.
        /// - `matrix_dim` the number specifying the dimensions of each individual matrix (lower triangular matrix must be square)
        /// - `num_matrices` specifies how many matrices to make
        /// - `diag` specifies the "offset" for the diagonal, an offset of `1` means we keep two `1`s on the first row instead of 1, and offset of `-1` means all `zeroes`
        pub(crate) fn tril(matrix_dim: usize, num_matrices: usize, diag: i32) -> Result<Tensor<T>> {
            Self::tri(matrix_dim, num_matrices, diag, T::unit(), T::default())
        }

        /// Makes a [`Tensor`] that is a batch of lower triangular matrices.
        /// - `matrix_dim` the number specifying the dimensions of each individual matrix (lower triangular matrix must be square)
        /// - `num_matrices` specifies how many matrices to make
        /// - `diag` specifies the "offset" for the diagonal, an offset of `1` means we keep two `lower_val`s on the first row instead of 1, and offset of `-1` means all `upper_val`
        /// - `lower_val` specifies the value to fill the lower triangular part with
        /// - `upper_val` specifies the value to fill the upper triangular part with
        pub(crate) fn tri(
            matrix_dim: usize,
            num_matrices: usize,
            diag: i32,
            lower_val: T,
            upper_val: T,
        ) -> Result<Tensor<T>> {
            // We make one matrix and then just clone it
            let data = (0i32..matrix_dim as i32)
                .flat_map(|i| {
                    if (i + diag).is_negative() {
                        vec![upper_val; matrix_dim]
                    } else {
                        std::iter::repeat_n(lower_val, (i + diag + 1) as usize)
                            .chain(std::iter::repeat(upper_val))
                            .take(matrix_dim)
                            .collect::<Vec<T>>()
                    }
                })
                .cycle()
                .take(num_matrices * matrix_dim * matrix_dim)
                .collect::<Vec<T>>();

            Tensor::<T>::new(vec![num_matrices, matrix_dim, matrix_dim].into(), data)
        }

        /// slice on the third dimension.
        /// start inclusive, end exclusive
        pub(crate) fn slice_3d(&self, start: usize, end: usize) -> Result<Self> {
            ensure!(self.shape.len() == 3);
            ensure!(start < self.shape[0]);
            ensure!(end <= self.shape[0]);
            let blocks = self.shape[1] * self.shape[2];
            let sliced = self.data[blocks * start..blocks * end].to_vec();
            let shape: Shape = vec![end - start, self.shape[1], self.shape[2]].into();

            Ok(Self {
                data: sliced,
                shape: shape.clone(),
                unpadded_shape: shape,
            })
        }

        /// Slice the tensor on the first dimension.
        ///
        /// # Arguments
        ///
        /// - start: New start for the first dimension, inclusive.
        /// - end: New end for the first dimension, exclusive.
        pub(crate) fn slice_2d(&self, start: usize, end: usize) -> Result<Self> {
            ensure!(self.shape.len() == 2);
            ensure!(start < end);
            ensure!(end <= self.shape[0] + 1);

            let range_start = start * self.shape[1];
            let range_end = end * self.shape[1];
            let data = self.data[range_start..range_end].to_vec();

            let new_shape = Shape::new(vec![end - start, self.shape[1]]);
            debug_assert!(
                new_shape.numel() == data.len(),
                "{new_shape:?} data.len(): {} start: {start} end: {end}",
                data.len(),
            );

            Ok(Self {
                data,
                shape: new_shape.clone(),
                unpadded_shape: new_shape,
            })
        }

        pub(crate) fn any(shape: Shape) -> impl proptest::prelude::Strategy<Value = Self> {
            use proptest::prelude::*;
            let size = shape.product();
            let data = proptest::collection::vec(T::any(), size);
            data.prop_map(move |data| Self {
                data,
                shape: shape.clone(),
                unpadded_shape: shape.clone(),
            })
        }
    }

    impl<T: TensorTypeParam> Tensor<T> {
        pub(crate) fn into_wrapped(self) -> WrappedTensor<T> {
            WrappedTensor::try_from(&self).unwrap()
        }

        pub(crate) fn as_wrapped(&self) -> WrappedTensor<T> {
            WrappedTensor::try_from(self).unwrap()
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
            &self,
            conv_shape_og: &Shape,
            conv_shape_pad: &Shape,
            matrix_shape_pad: &Shape,
        ) -> Result<Self> {
            let wrapped = WrappedTensor::try_from(self)?;
            let result = wrapped.pad_matrix_to_ignore_garbage(
                conv_shape_og,
                conv_shape_pad,
                matrix_shape_pad,
            )?;
            Tensor::try_from(result)
        }
    }

    impl<T> Tensor<T>
    where
        T: Default,
    {
        /// The new shape of self will be [S1_1+S2_1,S1,...Sn]
        /// In other words, we only concatenate another vector if it's exactly size of the highest dimension
        pub(crate) fn concat_from_unpadded(
            &mut self,
            self_unpadded_first_dim: usize,
            other: Self,
            other_unpadded_first_dim: usize,
        ) -> anyhow::Result<()> {
            ensure!(
                self.shape().rank() == other.shape().rank(),
                "self and other shapes must have the same length"
            );
            ensure!(
                self.shape()
                    .iter()
                    .zip(other.shape().iter())
                    .skip(1)
                    .all(|(a, b)| a == b),
                "self and other shapes must have the same dimensions"
            );
            ensure!(
                other.shape().numel() == other.data().len(),
                "The other tensor data length is not equal to the other shape product"
            );

            // 0-based indexing
            let max_stride: usize = self.shape().iter().skip(1).product();
            let init_pos = self_unpadded_first_dim * max_stride;
            let mut pos = init_pos;
            let end_of_slice = self.data().len();
            // only take the non-padded part of the other tensor
            for new_v in other
                .data
                .into_iter()
                .take(other_unpadded_first_dim * max_stride)
            {
                if pos >= end_of_slice {
                    self.data.push(new_v);
                } else {
                    self.data[pos] = new_v;
                }
                pos += 1;
            }
            ensure!(
                pos.is_multiple_of(max_stride),
                "The part going beyond must be a multiple of the stride"
            );
            // how many times have we added a "big" chunk to the recipient, and therefore by how much to increase the new shape
            let new_dim = self_unpadded_first_dim + (pos - init_pos) / max_stride;
            // we only update the shape for the part that goes beyond the existing padding
            if new_dim > self.shape.dim(0) {
                self.shape.set_dim(0, new_dim);
                self.data.resize_with(self.shape.numel(), Default::default);
            }
            ensure!(
                self.data().len() == self.shape().product(),
                "The new data length {} is not equal to the new shape product {}",
                self.data().len(),
                self.shape().product()
            );
            Ok(())
        }
    }

    impl<T> Tensor<T>
    where
        T: Copy + Default,
    {
        /// Permute a tensor.
        ///
        /// The tensor's dimensions will be moved according to `order`. The `i`-th
        /// entry in the `order` vector specifies which dimension of the original
        /// tensor should become the `i`-th dimension of the output tensor.
        ///
        /// ```
        /// # use zkml::{Tensor, Shape};
        /// let tensor = Tensor::<i64>::random(&Shape::new(vec![2,3,5]));
        /// let permuted = tensor.permute3d(&[2,1,0]).unwrap();
        /// assert_eq!(tensor.dim(0), permuted.dim(2));
        /// assert_eq!(tensor.dim(1), permuted.dim(1));
        /// assert_eq!(tensor.dim(2), permuted.dim(0));
        /// ```
        pub(crate) fn permute3d(&self, order: &[usize]) -> Result<Self> {
            ensure!(
                self.shape().rank() == 3,
                "Current tensor must be 3D. got {}",
                self.shape().rank(),
            );
            ensure!(
                order.len() == 3,
                "New order must be 3D. got {}",
                order.len(),
            );
            let count = order.iter().filter(|x| **x < 3).sorted().dedup().count();
            ensure!(
                count == 3,
                "Order must have unique elements 0, 1, 2. got {order:?}",
            );

            // Special case, do nothing
            if order == [0, 1, 2] {
                return Ok(self.clone());
            }

            let new_shape = Shape::new(vec![
                self.dim(order[0]),
                self.dim(order[1]),
                self.dim(order[2]),
            ]);

            // reverse map from old position of a dimension to its new position
            let old_to_new = [
                order.iter().position(|v| *v == 0).unwrap(),
                order.iter().position(|v| *v == 1).unwrap(),
                order.iter().position(|v| *v == 2).unwrap(),
            ];
            let new_strides = new_shape.strides();
            let reverse_strides = [
                new_strides[old_to_new[0]],
                new_strides[old_to_new[1]],
                new_strides[old_to_new[2]],
            ];

            let (a, b, c) = (self.dim(0), self.dim(1), self.dim(2));
            let mut pos = 0;
            let mut data = vec![T::default(); self.shape.numel()];

            for i in 0..a {
                for j in 0..b {
                    for k in 0..c {
                        let new_loc = i * reverse_strides[0]
                            + j * reverse_strides[1]
                            + k * reverse_strides[2];

                        data[new_loc] = self.data[pos];
                        pos += 1;
                    }
                }
            }

            Ok(Self {
                data,
                shape: new_shape,
                unpadded_shape: self.unpadded_shape.clone(),
            })
        }
    }

    impl<T> Tensor<T>
    where
        T: Copy,
    {
        /// Returns the value at the specified coordinate.
        ///
        /// # Error
        ///
        /// - If the accessor dimensionality doesn't match the shape.
        pub(crate) fn get(&self, accessors: impl AsRef<[usize]>) -> Result<T> {
            let flat_index = self.shape().get_idx(accessors)?;
            Ok(self.data[flat_index])
        }
    }

    impl<T> Tensor<T>
    where
        T: std::ops::Add<Output = T> + Copy + Send + Sync,
    {
        /// Add a vector to each sub-tensor of the second dimension of the tensor
        /// If self is 2d, then add a vector to each row of self.
        pub(crate) fn add_dim2(&self, other: &Tensor<T>) -> Tensor<T> {
            assert!(self.shape.len() == 2, "Tensor is not a matrix");
            assert!(other.shape.len() == 1, "Tensor is not a vector");
            assert!(
                self.shape[1] == other.shape[0],
                "Shape mismatch for addition2: {:?} != {:?}",
                self.shape,
                other.shape
            );
            let data = self
                .data
                .par_chunks(self.shape[1])
                .flat_map_iter(|chunk| chunk.iter().zip(other.data.iter()).map(|(a, b)| *a + *b))
                .collect::<Vec<_>>();
            Tensor {
                shape: self.shape.clone(),
                data,
                unpadded_shape: self.unpadded_shape.clone(),
            }
        }
    }

    impl<T> Tensor<T>
    where
        T: std::ops::Sub<Output = T> + Copy + Send + Sync,
    {
        /// Element-wise subtraction
        pub(crate) fn sub(&self, other: &Tensor<T>) -> Tensor<T> {
            auto_vec_binop!(self, other, -)
        }
    }

    impl<T> Tensor<T>
    where
        T: std::ops::Mul<Output = T> + Copy + Send + Sync,
    {
        /// Scalar multiplication
        pub(crate) fn scalar_mul(&self, scalar: &T) -> Tensor<T> {
            let scalar = *scalar;

            let data = auto_vec_op!(self.data, |el| el * scalar);

            Tensor {
                data,
                shape: self.shape.clone(),
                unpadded_shape: self.unpadded_shape.clone(),
            }
        }
    }

    impl<T> Tensor<T> {
        /// Concatenate the other tensor to the first one.
        /// RESTRICTIOn: self shape is [a1,a2...,an] we
        /// expect other shape to be [a2...,an] OR [1, a2...,an]
        /// The new shape of self will be [a1+1,...an]
        /// In other words, we only concatenate another vector if it's exactly size of the highest dimension
        /// If it's 2d, then we expect other to be a vector
        pub(crate) fn concat(&mut self, other: Self) -> Result<()> {
            // make sure that the all dimension but the highest one are the same
            let common_shape = self.shape.len().min(other.shape.len());
            let added_higher = if common_shape < self.shape.len() {
                ensure!(
                    self.shape
                        .iter()
                        .rev()
                        .zip(other.shape.iter().rev())
                        .take(common_shape)
                        .all(|(a, b)| a == b)
                );
                ensure!(common_shape + 1 == self.shape.len());
                1
            } else {
                ensure!(common_shape == self.shape.len());
                *other.shape.first().unwrap()
            };
            // then the new shape has this higher dimension + 1 simply
            // common_shape since 0-based indexing
            *self.shape.get_mut(0).unwrap() += added_higher;
            self.data.extend(other.data);

            Ok(())
        }

        /// Returns the boolean iterator indicating the given row in the right endianness to be
        /// evaluated by an MLE
        pub(crate) fn row_to_boolean_2d<F: PrimeField>(
            &self,
            row: usize,
        ) -> Result<impl Iterator<Item = F>> {
            ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
            let (nvars_rows, _) = self.shape().num_vars_2d();
            Ok(to_bit_sequence_le(row, nvars_rows).map(|b| F::from(b as u64)))
        }

        /// Returns the boolean iterator indicating the given row in the right endianness to be
        /// evaluated by an MLE
        pub(crate) fn col_to_boolean_2d<F: PrimeField>(
            &self,
            col: usize,
        ) -> Result<impl Iterator<Item = F>> {
            ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
            let (_, nvars_col) = self.shape().num_vars_2d();
            Ok(to_bit_sequence_le(col, nvars_col).map(|b| F::from(b as u64)))
        }

        /// From a given row and a given column, return the vector of field elements in the right
        /// format to evaluate the MLE.
        /// little endian so we need to read cols before rows
        pub(crate) fn position_to_boolean_2d<F: PrimeField>(
            &self,
            row: usize,
            col: usize,
        ) -> Result<Vec<F>> {
            ensure!(self.shape.is_matrix(), "Tensor is not a matrix");
            Ok(self
                .col_to_boolean_2d(col)?
                .chain(self.row_to_boolean_2d(row)?)
                .collect_vec())
        }

        /// Removes a dimension from the tensor.
        ///
        /// # Panics
        ///
        /// Panics if `index` is larger than the shape size, or if the
        /// removed dimension size is not 1.
        pub(crate) fn squeeze(mut self, index: usize) -> Result<Self> {
            ensure!(
                self.shape[index] == 1,
                "The dimension to be squeezed must have a value of 1"
            );
            self.shape.remove(index);
            Ok(self)
        }
    }

    #[test]
    fn test_tensor_basic_ops() {
        let tensor1 = Tensor::new(vec![2, 2].into(), vec![1, 2, 3, 4]).unwrap();
        let tensor2 = Tensor::new(vec![2, 2].into(), vec![5, 6, 7, 8]).unwrap();

        let result_add = tensor1.add(&tensor2);
        assert_eq!(
            result_add,
            Tensor::new(vec![2, 2].into(), vec![6, 8, 10, 12]).unwrap(),
            "Element-wise addition failed."
        );

        let result_sub = tensor2.sub(&tensor2);
        assert_eq!(
            result_sub,
            Tensor::zeros(vec![2, 2].into()),
            "Element-wise subtraction failed."
        );

        let result_mul = tensor1.mul(&tensor2);
        assert_eq!(
            result_mul,
            Tensor::new(vec![2, 2].into(), vec![5, 12, 21, 32]).unwrap(),
            "Element-wise multiplication failed."
        );

        let result_scalar = tensor1.scalar_mul(&2);
        assert_eq!(
            result_scalar,
            Tensor::new(vec![2, 2].into(), vec![2, 4, 6, 8]).unwrap(),
            "Element-wise scalar multiplication failed."
        );
    }

    #[test]
    fn test_tensor_matvec() {
        let shape_m = vec![3, 3];
        let tensor_m =
            Tensor::new(shape_m.clone().into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let matrix = Array::from_vec(tensor_m.data().to_vec())
            .into_shape_with_order((shape_m, Order::RowMajor))
            .unwrap()
            .into_dimensionality::<Ix2>()
            .unwrap();
        let tensor_v = Tensor::new(vec![3].into(), vec![10, 20, 30]).unwrap();
        let vector = Array::from_vec(tensor_v.data().to_vec());

        let result = tensor_m.matvec(&tensor_v).unwrap();
        let expected_result = matrix.dot(&vector);

        assert_eq!(
            Array::from_vec(result.data().to_vec()),
            expected_result,
            "Matrix-vector multiplication failed."
        );
    }

    #[test]
    fn test_tensor_matmul() {
        let shape_a = vec![4, 3];
        let tensor_a = Tensor::new(
            shape_a.clone().into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        .unwrap();
        let matrix_a = Array::from_vec(tensor_a.data().to_vec())
            .into_shape_with_order((shape_a, Order::RowMajor))
            .unwrap();
        let shape_b = vec![3, 3];
        let tensor_b = Tensor::new(
            shape_b.clone().into(),
            vec![10, 20, 30, 40, 50, 60, 70, 80, 90],
        )
        .unwrap();
        let matrix_b = Array::from_vec(tensor_b.data().to_vec())
            .into_shape_with_order((shape_b, Order::RowMajor))
            .unwrap();

        let result = tensor_a.matmul(&tensor_b).unwrap();

        let expected_result = matrix_a
            .into_dimensionality::<Ix2>()
            .unwrap()
            .dot(&matrix_b.into_dimensionality::<Ix2>().unwrap());

        assert_eq!(
            Array::from_vec(result.data().to_vec())
                .into_shape_with_order((expected_result.shape(), Order::RowMajor))
                .unwrap()
                .into_dimensionality::<Ix2>()
                .unwrap(),
            expected_result,
            "Matrix-matrix multiplication failed."
        );
    }

    #[test]
    fn test_tensor_transpose() {
        let matrix_a = Tensor::new(
            vec![3, 4].into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        .unwrap();
        let matrix_b = Tensor::new(
            vec![4, 3].into(),
            vec![1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12],
        )
        .unwrap();

        let result = matrix_a.transpose().unwrap();

        assert_eq!(result, matrix_b, "Matrix transpose failed.");
    }

    #[test]
    fn test_tensor_next_pow_of_two() {
        let shape = Shape::new(vec![1, 1, 1, 1]);
        let tensor = Tensor::new(shape.clone(), vec![1]).unwrap();
        assert_eq!(
            tensor.pad_next_power_of_two(),
            tensor,
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![2, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2]).unwrap();
        assert_eq!(
            tensor.pad_next_power_of_two(),
            tensor,
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![4, 4]);
        let tensor = Tensor::<Element>::random(&shape.clone());
        assert_eq!(
            tensor.pad_next_power_of_two(),
            tensor,
            "Tensor should not change if the shape is already power of two",
        );

        let shape = Shape::new(vec![3, 3]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 3, 1, 2, 3, 1, 2, 3]).unwrap();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            *new_tensor.shape(),
            Shape::new(vec![4, 4]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            new_tensor.data().as_slice(),
            [1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 0, 0, 0],
            "Tensor padding to next power of two failed."
        );

        let shape = Shape::new(vec![3, 2]);
        let tensor = Tensor::new(shape.clone(), vec![1, 2, 1, 2, 1, 2]).unwrap();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            *new_tensor.shape(),
            Shape::new(vec![4, 2]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            new_tensor.data().as_slice(),
            [1, 2, 1, 2, 1, 2, 0, 0],
            "Tensor padding to next power of two failed."
        );

        let shape = Shape::new(vec![2, 3, 3]);
        let tensor = Tensor::new(
            shape.clone(),
            vec![
                1, 1, 1, 2, 2, 2, 3, 3, 3, 11, 11, 11, 12, 12, 12, 13, 13, 13,
            ],
        )
        .unwrap();
        let new_tensor = tensor.pad_next_power_of_two();
        assert_eq!(
            *new_tensor.shape(),
            Shape::new(vec![2, 4, 4]),
            "Tensor padding to next power of two failed."
        );
        assert_eq!(
            new_tensor.data().as_slice(),
            [
                1, 1, 1, 0, 2, 2, 2, 0, 3, 3, 3, 0, 0, 0, 0, 0, 11, 11, 11, 0, 12, 12, 12, 0, 13,
                13, 13, 0, 0, 0, 0, 0,
            ],
            "Tensor padding to next power of two failed."
        );
    }

    impl Tensor<Element> {
        pub fn get_2d(&self, i: usize, j: usize) -> Element {
            assert!(self.shape.is_matrix());
            self.data[i * self.shape()[1] + j]
        }

        pub fn random_eval_point(&self) -> Vec<F> {
            let mut rng = rng_from_env_or_random();
            let r = rng.gen_range(0..self.nrows_2d().unwrap());
            let c = rng.gen_range(0..self.ncols_2d().unwrap());
            self.position_to_boolean_2d(r, c).unwrap()
        }
    }

    #[test]
    fn test_tensor_mle() {
        let mat = Tensor::random(&vec![3, 5].into());
        let shape = mat.shape();
        let mat = mat.pad_next_power_of_two();
        let mut mle = mat.to_2d_mle::<F>().unwrap();
        let mut rng = rng_from_env_or_random();
        let (chosen_row, chosen_col) = (rng.gen_range(0..shape[0]), rng.gen_range(0..shape[1]));
        let elem = mat.get_2d(chosen_row, chosen_col);
        let elem_field: F = elem.to_field();
        println!("(x,y) = ({chosen_row},{chosen_col}) ==> {elem:?}");
        let inputs = mat.position_to_boolean_2d(chosen_row, chosen_col).unwrap();
        let output = mle.evaluate(&inputs).unwrap();
        assert_eq!(elem_field, output);

        // now try to address one at a time, and starting by the row, which is the opposite order
        // of the boolean variables expected by the MLE API, given it's expecting in LE format.
        let row_input = mat.row_to_boolean_2d(chosen_row).unwrap();
        mle.fix_high_variables_in_place(&row_input.collect_vec());
        let col_input = mat.col_to_boolean_2d(chosen_col).unwrap();
        let output = mle.evaluate(&col_input.collect_vec()).unwrap();
        assert_eq!(elem_field, output);
    }

    #[test]
    fn test_tensor_matvec_concatenate() {
        let matrix = Tensor::new(vec![3, 3].into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let vector = Tensor::new(vec![3].into(), vec![10, 20, 30]).unwrap();

        let result = matrix.concat_matvec_col(&vector).unwrap();

        assert_eq!(
            result,
            Tensor::new(
                vec![3, 4].into(),
                vec![1, 2, 3, 10, 4, 5, 6, 20, 7, 8, 9, 30]
            )
            .unwrap(),
            "Concatenate matrix vector as columns failed."
        );
    }

    #[test]
    fn test_tensor_ext_ops() {
        let matrix_a_data = [1 as Element, 2, 3, 4, 5, 6, 7, 8, 9];
        let matrix_b_data = [10 as Element, 20, 30, 40, 50, 60, 70, 80, 90];
        let matrix_c_data = [300 as Element, 360, 420, 660, 810, 960, 1020, 1260, 1500];
        let vector_a_data = [10 as Element, 20, 30];
        let vector_b_data = [140 as Element, 320, 500];

        let matrix_a_data: Vec<F> = to_field(matrix_a_data);
        let matrix_b_data: Vec<F> = to_field(matrix_b_data);
        let matrix_c_data: Vec<F> = to_field(matrix_c_data);
        let vector_a_data: Vec<F> = to_field(vector_a_data);
        let vector_b_data: Vec<F> = to_field(vector_b_data);
        let matrix = Tensor::new(vec![3usize, 3].into(), matrix_a_data.clone()).unwrap();
        let vector = Tensor::new(vec![3usize].into(), vector_a_data).unwrap();
        let vector_expected = Tensor::new(vec![3usize].into(), vector_b_data).unwrap();

        let result = matrix.matvec(&vector).unwrap();

        assert_eq!(
            result, vector_expected,
            "Matrix-vector multiplication failed."
        );

        let matrix_a = Tensor::new(vec![3, 3].into(), matrix_a_data).unwrap();
        let matrix_b = Tensor::new(vec![3, 3].into(), matrix_b_data).unwrap();
        let matrix_c = Tensor::new(vec![3, 3].into(), matrix_c_data).unwrap();

        let result = matrix_a.matmul(&matrix_b).unwrap();

        assert_eq!(result, matrix_c, "Matrix-matrix multiplication failed.");
    }

    #[test]
    fn test_tensor_maxpool2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 3, 4].into(),
            vec![
                99, -35, 18, 104, -26, -48, -80, 106, 10, 8, 79, -7, -128, -45, 24, -91, -7, 88,
                -119, -37, -38, -113, -84, 86, 116, 72, -83, 100, 83, 81, 87, 58, -109, -13, -123,
                102,
            ],
        )
        .unwrap();
        let expected =
            Tensor::<Element>::new(vec![1, 3, 1, 2].into(), vec![99, 106, 88, 24, 116, 100])
                .unwrap();

        let result = input.maxpool2d(2, 2).unwrap();
        assert_eq!(result, expected, "Maxpool (Element) failed.");
    }

    #[test]
    fn test_tensor_pad_maxpool2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                93, 56, -3, -1, 104, -68, -71, -96, 5, -16, 3, -8, 74, -34, -16, -31, -42, -59,
                -64, 70, -77, 19, -17, -114, 79, 55, 4, -26, -7, -17, -94, 21, 59, -116, -113, 47,
                8, 112, 65, -99, 35, 3, -126, -52, 28, 69, 105, 33,
            ],
        )
        .unwrap();
        let expected = Tensor::<Element>::new(
            vec![1, 3, 2, 2].into(),
            vec![104, -1, 74, 3, 19, 70, 79, 21, 112, 65, 69, 105],
        )
        .unwrap();

        let padded_expected = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                104, 104, -1, -1, 104, 104, -1, -1, 74, 74, 3, 3, 74, 74, 3, 3, 19, 19, 70, 70, 19,
                19, 70, 70, 79, 79, 21, 21, 79, 79, 21, 21, 112, 112, 65, 65, 112, 112, 65, 65, 69,
                69, 105, 105, 69, 69, 105, 105,
            ],
        )
        .unwrap();

        let (result, padded_result) = input.padded_maxpool2d().unwrap();
        assert_eq!(result, expected, "Maxpool (Element) failed.");
        assert_eq!(
            padded_result, padded_expected,
            "Padded Maxpool (Element) failed."
        );
    }

    #[test]
    fn test_pad_tensor_for_mle() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 4, 4].into(),
            vec![
                93, 56, -3, -1, 104, -68, -71, -96, 5, -16, 3, -8, 74, -34, -16, -31, -42, -59,
                -64, 70, -77, 19, -17, -114, 79, 55, 4, -26, -7, -17, -94, 21, 59, -116, -113, 47,
                8, 112, 65, -99, 35, 3, -126, -52, 28, 69, 105, 33,
            ],
        )
        .unwrap();

        let padded = input.pad_next_power_of_two();

        padded
            .shape()
            .iter()
            .zip(input.shape().iter())
            .for_each(|(padded_dim, input_dim)| {
                assert_eq!(*padded_dim, input_dim.next_power_of_two())
            });

        let input_data = input.data();
        let padded_data = padded.data();
        for i in 0..1 {
            for j in 0..3 {
                for k in 0..4 {
                    for l in 0..4 {
                        let index = 3 * 4 * 4 * i + 4 * 4 * j + 4 * k + l;
                        assert_eq!(input_data[index], padded_data[index]);
                    }
                }
            }
        }
    }

    #[test]
    fn test_tensor_pad() {
        let shape_a = Shape::from_it([3, 1, 1]);
        let tensor_a = Tensor::<Element>::new(shape_a.clone(), vec![1; shape_a.product()]).unwrap();

        let shape_b = vec![4, 1, 1];
        let tensor_b = Tensor::<Element>::new(shape_b.clone().into(), vec![1, 1, 1, 0]).unwrap();

        let tensor_c = tensor_a.pad_next_power_of_two();
        assert_eq!(tensor_b, tensor_c);
    }

    #[test]
    fn test_tensor_pad_to_shape() {
        let shape = Shape::from_it([1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]).unwrap();
        let target = Shape::from_it([2]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([2]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1, 2]).unwrap();
        let target = Shape::from_it([3]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 2, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]).unwrap();
        let target = Shape::from_it([2, 1]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 1]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1]).unwrap();
        let target = Shape::from_it([1, 2]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([2, 2]);
        let mut tensor = Tensor::<Element>::new(shape, vec![1, 2, 3, 4]).unwrap();
        let target = Shape::from_it([3, 3]);
        let res = Tensor::<Element>::new(target.clone(), vec![1, 2, 0, 3, 4, 0, 0, 0, 0]).unwrap();
        tensor.pad_to_shape(target.clone()).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 1, 1]);
        let mut tensor = Tensor::<Element>::new(shape.clone(), vec![1, 1, 1]).unwrap();
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),

            vec![
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ).unwrap();
        tensor.pad_to_shape(target).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 1, 3]);
        let mut tensor =
            Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2, 3, 3, 3]).unwrap();
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),

            vec![
                1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
        ).unwrap();
        tensor.pad_to_shape(target).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([3, 3, 1]);
        let mut tensor =
            Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2, 3, 3, 3]).unwrap();
        let target = Shape::from_it([3, 4, 4]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),
            vec![
                1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0,
                3, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0,
            ],
        ).unwrap();
        tensor.pad_to_shape(target).unwrap();
        assert_eq!(tensor, res);

        let shape = Shape::from_it([1, 2, 1, 3]);
        let mut tensor = Tensor::<Element>::new(shape.clone(), vec![1, 1, 1, 2, 2, 2]).unwrap();
        let target = Shape::from_it([2, 3, 5, 7]);
        #[rustfmt::skip]
        let res = Tensor::<Element>::new(
            target.clone(),
            vec![
                // x=0 y=0
                1, 1, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=0 y=1
                2, 2, 2, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=0 y=2
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=0
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=1
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,

                // x=1 y=2
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0,
            ],
        ).unwrap();
        tensor.pad_to_shape(target).unwrap();
        assert_eq!(tensor, res);
    }

    #[test]
    fn test_tensor_conv2d() {
        let input = Tensor::<Element>::new(
            vec![1, 3, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3,
            ],
        )
        .unwrap();

        let weights = Tensor::<Element>::new(
            vec![2, 3, 2, 2].into(),
            vec![
                1, 0, -1, 2, 0, 1, -1, 1, 1, -1, 0, 2, -1, 1, 2, 0, 1, 0, 2, -1, 0, -1, 1, 1,
            ],
        )
        .unwrap();

        let bias = Tensor::<Element>::new(vec![2].into(), vec![3, -3]).unwrap();

        let expected = Tensor::<Element>::new(
            vec![1, 2, 2, 2].into(),
            vec![21, 22, 26, 27, 25, 25, 26, 26],
        )
        .unwrap();

        let result = input.conv2d(&weights, &bias, 1).unwrap();
        assert_eq!(result, expected, "Conv2D (Element) failed.");
    }

    #[test]
    fn test_tensor_minimal_conv2d() {
        // k_n,k_c,k_h,k_w
        let conv_shape = vec![2, 3, 3, 3].into();
        let conv = Tensor::<Element>::random(&conv_shape);
        // minimal input shape is 1,k_c,k_h,k_w
        let input_shape = vec![1, 3, 3, 3].into();
        let input = Tensor::<Element>::random(&input_shape);
        // minimal bias shape is k_n
        let bias = Tensor::<Element>::random(&vec![2].into());
        let output = input.conv2d(&conv, &bias, 1).unwrap();
        assert_eq!(*output.shape(), vec![1, 2, 1, 1].into());
    }

    #[test]
    fn test_tensor_pad_matrix_to_ignore_garbage() {
        let conv_shape_og = Shape::new(vec![2usize, 3, 3]);
        let mat_shape_og = Shape::new(vec![10usize, conv_shape_og.product()]);

        let conv_shape_pad = Shape::new(vec![3usize, 4, 4]);
        let mat_shape_pad = Shape::new(vec![12usize, conv_shape_pad.product()]);

        // The flattened output of a conv2d
        let conv_out_og = Tensor::<Element>::random(&conv_shape_og);
        let conv_out_og_flat = conv_out_og.to_flatten();

        // The layer after conv2d, in this case a dense layer
        let dense_og = Tensor::random(&mat_shape_og);
        let result_og = dense_og.matvec(&conv_out_og_flat).unwrap();

        // The equivalent output for a FFT-conv2d
        let mut conv_out_pad = conv_out_og.clone();
        conv_out_pad.pad_to_shape(conv_shape_pad.clone()).unwrap();
        let conv_out_pad_flat = conv_out_pad.to_flatten();

        // The layer after FFT-conv2d
        let dense_pad = dense_og
            .pad_matrix_to_ignore_garbage(&conv_shape_og, &conv_shape_pad, &mat_shape_pad)
            .unwrap();
        let result_pad = dense_pad.matvec(&conv_out_pad_flat).unwrap();

        assert_eq!(
            result_og.data()[..mat_shape_og.dim(0)],
            result_pad.data()[..mat_shape_og.dim(0)],
            "Unable to get rid of garbage values from conv fft."
        );
    }

    #[test]
    fn test_tensor_slice_2d() {
        let tensor =
            Tensor::<Element>::new(vec![3, 3].into(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9]).unwrap();
        let sliced = tensor.slice_2d(0, 2).unwrap();
        assert_eq!(*sliced.shape(), vec![2, 3].into());
        assert_eq!(sliced.data(), &[1, 2, 3, 4, 5, 6]);
        let sliced = tensor.slice_2d(2, 3).unwrap();
        assert_eq!(*sliced.shape(), vec![1, 3].into());
        assert_eq!(sliced.data(), &[7, 8, 9]);
    }

    #[test]
    fn test_tensor_add_dim2() {
        let tensor = Tensor::<Element>::new(vec![2, 3].into(), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let vector = Tensor::<Element>::new(vec![3].into(), vec![10, 20, 30]).unwrap();
        let result = tensor.add_dim2(&vector);
        assert_eq!(*result.shape(), vec![2, 3].into());
        assert_eq!(result.data(), &[11, 22, 33, 14, 25, 36]);
    }

    #[test]
    fn test_tensor_concat() {
        let mut tensor = Tensor::<Element>::new(vec![2, 3].into(), vec![1, 2, 3, 4, 5, 6]).unwrap();
        let vector = Tensor::<Element>::new(vec![3].into(), vec![10, 20, 30]).unwrap();
        tensor.concat(vector).unwrap();

        assert_eq!(*tensor.shape(), vec![3, 3].into());
        assert_eq!(tensor.data(), &[1, 2, 3, 4, 5, 6, 10, 20, 30]);

        let vector = Tensor::<Element>::new(vec![1, 3].into(), vec![66, 77, 88]).unwrap();
        tensor.concat(vector).unwrap();
        assert_eq!(*tensor.shape(), vec![4, 3].into());
        assert_eq!(tensor.data(), &[1, 2, 3, 4, 5, 6, 10, 20, 30, 66, 77, 88]);
    }

    #[test]
    fn test_tensor_get() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        )
        .unwrap();
        // 2 + 2 * 3 + 1 * 3 * 3 = 17
        assert_eq!(tensor.get(vec![1, 2, 2]).unwrap(), tensor.data[17]);
    }

    #[test]
    fn test_tensor_permute3d() {
        #[rustfmt::skip]
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,

                10, 11, 12,
                13, 14, 15,
                16, 17, 18,
            ],
        ).unwrap();

        let permuted = tensor.permute3d(&[1, 0, 2]).unwrap();
        assert_eq!(*permuted.shape(), vec![3, 2, 3].into());
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..3 {
                    let [new_i, new_j, new_k] = [j, i, k];
                    let expected = tensor.get([i, j, k]).unwrap();
                    let given = permuted.get([new_i, new_j, new_k]).unwrap();
                    assert_eq!(expected, given);
                }
            }
        }

        let tensor = Tensor::<Element>::random(&vec![18, 5, 27].into());
        let permuted = tensor.permute3d(&[1, 2, 0]).unwrap();
        assert_eq!(*permuted.shape(), Shape::new(vec![5, 27, 18]));
    }

    #[test]
    fn test_tensor_slice_3d() {
        let tensor = Tensor::<Element>::new(
            vec![3, 2, 2].into(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )
        .unwrap();
        let sliced = tensor.slice_3d(1, 3).unwrap();
        assert_eq!(sliced.data(), &[5, 6, 7, 8, 9, 10, 11, 12]);
        assert_eq!(*sliced.shape(), vec![2, 2, 2].into());
    }

    #[test]
    fn test_tensor_slices_last_dim() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        )
        .unwrap();

        let mut slices = tensor.slice_last_dim();

        // First slice
        assert_eq!(slices.next().unwrap(), &[1, 2, 3]);
        // Second slice
        assert_eq!(slices.next().unwrap(), &[4, 5, 6]);
        // Third slice
        assert_eq!(slices.next().unwrap(), &[7, 8, 9]);
        // Fourth slice
        assert_eq!(slices.next().unwrap(), &[10, 11, 12]);
        // Fifth slice
        assert_eq!(slices.next().unwrap(), &[13, 14, 15]);
        // Sixth slice
        assert_eq!(slices.next().unwrap(), &[16, 17, 18]);
        // No more slices
        assert_eq!(slices.next(), None);
    }

    #[test]
    fn test_tensor_slice_on_dim() {
        let tensor = Tensor::<Element>::new(
            vec![2, 3, 3].into(),
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            ],
        )
        .unwrap();
        let (mut slices, shape) = tensor.slice_on_dim(1);
        assert_eq!(shape, Shape::new(vec![3]));
        assert_eq!(slices.next().unwrap(), &[1, 2, 3]);
        assert_eq!(slices.next().unwrap(), &[4, 5, 6]);
        assert_eq!(slices.next().unwrap(), &[7, 8, 9]);
        assert_eq!(slices.next().unwrap(), &[10, 11, 12]);
        assert_eq!(slices.next().unwrap(), &[13, 14, 15]);
        assert_eq!(slices.next().unwrap(), &[16, 17, 18]);
        assert_eq!(slices.next(), None);

        let (mut slices, shape) = tensor.slice_on_dim(0);
        assert_eq!(shape, Shape::new(vec![3, 3]));
        assert_eq!(slices.next().unwrap(), &[1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            slices.next().unwrap(),
            &[10, 11, 12, 13, 14, 15, 16, 17, 18]
        );
        assert_eq!(slices.next(), None);

        let (slices, shape) = tensor.slice_on_dim(2);
        assert_eq!(shape, *tensor.shape());
        let data = slices.flatten().cloned().collect::<Vec<_>>();
        assert_eq!(
            data,
            vec![
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
            ]
        );
    }

    #[test]
    fn test_tensor_argmax() {
        let tensor = Tensor::<Element>::new(vec![3].into(), vec![1, 2, 3]).unwrap();
        let argmax = tensor.argmax();
        assert_eq!(argmax, 2);
    }

    fn eval_lteq_poly(x_i: &[Element], y_i: &[Element]) -> Element {
        assert_eq!(x_i.len(), y_i.len());
        x_i.iter()
            .rev()
            .zip(y_i.iter().rev())
            .fold(Element::from(1), |acc, (x, y)| {
                acc * (1 - x - y + 2 * x * y) + (1 - x) * y
            })
    }

    fn eval_mle<F: PrimeField>(point: &[F]) -> F {
        let x_i = &point[..point.len() / 2];
        let y_i = &point[point.len() / 2..];
        x_i.iter().zip(y_i).fold(F::from(1), |acc, (&x, &y)| {
            acc * (F::from(1) - x - y + F::from(2) * x * y) + (F::from(1) - x) * y
        })
    }

    fn to_be_bits<const NUM_BITS: usize>(x: Element) -> [Element; NUM_BITS] {
        (0..NUM_BITS)
            .rev()
            .map(|i| {
                let mask = 1 << i;

                (x & Element::from(mask)) >> i
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    }

    #[test]
    fn test_zeroifier_evaluation() {
        // create zeroifier matrix
        const NUM_BITS: usize = 4;
        let num_columns = 1 << NUM_BITS;
        let zeroifier_data = (0..num_columns * num_columns)
            .map(|i| {
                let r = i / num_columns;
                let c = i % num_columns;
                if r >= c {
                    Element::from(1)
                } else {
                    Element::from(0)
                }
            })
            .collect_vec();
        println!("Data: {zeroifier_data:?}");
        let zeroifier = Tensor::new(vec![num_columns, num_columns].into(), zeroifier_data).unwrap();
        assert_eq!(zeroifier.get_2d(0, 0), Element::from(1));
        assert_eq!(
            zeroifier.get_2d(num_columns - 1, num_columns - 1),
            Element::from(1)
        );
        assert_eq!(zeroifier.get_2d(0, 1), Element::from(0));
        assert_eq!(zeroifier.get_2d(1, 1), Element::from(1));
        assert_eq!(zeroifier.get_2d(1, 2), Element::from(0));

        let mle = zeroifier.to_2d_mle::<F>().unwrap();

        for i in 0..num_columns {
            for j in 0..num_columns {
                let x_i = to_be_bits::<NUM_BITS>(Element::from(i as u32));
                let y_i = to_be_bits::<NUM_BITS>(Element::from(j as u32));
                let cmp = eval_lteq_poly(&y_i, &x_i);
                assert_eq!(
                    zeroifier.get_2d(i, j),
                    cmp,
                    "Zeroifier evaluation failed for ({i}, {j})"
                );
                // build point for MLE: first column bits in little-endiian order, then rows bits in little-endian order
                let point = y_i
                    .into_iter()
                    .rev()
                    .chain(x_i.into_iter().rev())
                    .map(|bit| F::from(bit as u64))
                    .collect_vec();
                let eval = mle.evaluate(&point).unwrap();
                assert_eq!(eval, F::from(cmp as u64));
                let quick_eval = eval_mle(&point);
                assert_eq!(eval, quick_eval);
            }
        }

        // test over random points
        for _ in 0..10 {
            let point = random_field_vector::<F>(NUM_BITS * 2);
            assert_eq!(mle.evaluate(&point).unwrap(), eval_mle(&point),);
        }
    }

    #[test]
    fn test_tril() {
        // Test diag = 0
        let tensor = Tensor::<Element>::tril(4, 1, 0).unwrap();
        let real_value: Vec<Element> = vec![1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1];
        assert_eq!(tensor.data(), &real_value);
        // Test diag = 1
        let tensor = Tensor::<Element>::tril(4, 1, 1).unwrap();
        let real_value: Vec<Element> = vec![1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1];
        assert_eq!(tensor.data(), &real_value);
        // Test diag = -1
        let tensor = Tensor::<Element>::tril(4, 1, -1).unwrap();
        assert_eq!(
            tensor.data(),
            &[0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]
        );
    }

    #[test]
    fn test_concat_from() {
        let t1_shape: Shape = vec![5, 3, 3].into();
        let mut t1 = Tensor::<Element>::random(&t1_shape).pad_next_power_of_two();
        let t2_shape: Shape = vec![1, 3, 3].into();
        let t2 = Tensor::<Element>::random(&t2_shape).pad_next_power_of_two();
        t1.concat_from_unpadded(t1_shape.dim(0), t2, t2_shape.dim(0))
            .unwrap();
        // 8 since 5 padded next power of two is 8, and 5+1=6 so we don't go over the dimension
        let expected_shape = Shape::new(vec![8, 4, 4]);
        assert_eq!(*t1.shape(), expected_shape);
        // 6 + 3 = 9  so resulting
        let t3_shape: Shape = vec![3, 3, 3].into();
        let t3 = Tensor::<Element>::random(&t3_shape).pad_next_power_of_two();
        t1.concat_from_unpadded(6, t3, t3_shape.dim(0)).unwrap();
        let expected_shape = Shape::new(vec![9, 4, 4]);
        assert_eq!(*t1.shape(), expected_shape);
    }

    proptest! {
        #[test]
        fn test_pad_next_power_of_two_prop(x in 1usize..5, y in 2usize..=32, z in 2usize..=32) {
            pub fn pad_next_power_of_two(t: &Tensor<Element>) -> Tensor<Element> {
                let shape = t.shape();

                if shape.iter().all(|dim| dim.is_power_of_two()) {
                    return t.clone();
                }

                let padded_data = recursive_pad(t.data(), shape, Element::default());
                let padded_shape = shape.next_power_of_two();
                Tensor::new(padded_shape, padded_data).unwrap()
            }

            fn recursive_pad(data: &[Element], remaining_dims: &[usize], padding_value: Element) -> Vec<Element> {
                match remaining_dims.len() {
                    1 => data
                        .iter()
                        .cloned()
                        .chain(std::iter::repeat(padding_value))
                        .take(remaining_dims[0].next_power_of_two())
                        .collect::<Vec<_>>(),
                    _ => {
                        let chunk_size = remaining_dims[1..].iter().product::<usize>();
                        let mut unpadded_data = data
                            .par_chunks(chunk_size)
                            .map(|data_chunk| {
                                recursive_pad(data_chunk, &remaining_dims[1..], padding_value)
                            })
                            .collect::<Vec<Vec<_>>>();
                        let elem_size = unpadded_data[0].len();
                        unpadded_data.resize(
                            remaining_dims[0].next_power_of_two(),
                            vec![padding_value; elem_size],
                        );
                        unpadded_data.concat()
                    }
                }
            }

            let shape = Shape::new(vec![x, y, z]);
            let t = Tensor::<Element>::random(&shape);

            assert_eq!(t.pad_next_power_of_two(), pad_next_power_of_two(&t), "original: {t:?}");
        }

        #[test]
        fn proptest_tensor_permute3d(a in 2usize..=32, b in 2usize..=32, c in 2usize..=32) {
            fn permute3d<T: Default + Copy>(tensor: &Tensor<T>, order: &[usize]) -> Tensor<T> {
                let (a, b, c) = (tensor.shape[0], tensor.shape[1], tensor.shape[2]);
                let new_a = tensor.shape[order[0]];
                let new_b = tensor.shape[order[1]];
                let new_c = tensor.shape[order[2]];

                let mut data = vec![T::default(); tensor.shape.numel()];
                for i in 0..a {
                    for j in 0..b {
                        for k in 0..c {
                            let old_loc = i * b * c + j * c + k;
                            let pos = [i, j, k];
                            let new_i = pos[order[0]];
                            let new_j = pos[order[1]];
                            let new_k = pos[order[2]];
                            let new_loc = new_i * new_b * new_c + new_j * new_c + new_k;
                            data[new_loc] = tensor.data[old_loc];
                        }
                    }
                }
                Tensor {
                    data,
                    shape: Shape::new(vec![new_a, new_b, new_c]),
                    unpadded_shape: Shape::new(vec![new_a, new_b, new_c]),
                }
            }

            let permutations = [
                [0, 1, 2],
                [1, 0, 2],
                [1, 2, 0],
                [0, 2, 1],
                [2, 0, 1],
                [2, 1, 0],
            ];

            let data = Tensor::<Element>::random(&Shape::new(vec![a, b, c]));
            for order in &permutations {
                let expected = permute3d(&data, order);
                let result = data.permute3d(order).unwrap();
                prop_assert_eq!(&expected, &result, "order {:?} original {:?}", order, data);
            }
        }

        #[test]
        fn proptest_tensor_pad_matrix_to_ignore_garbage(x in 1usize..32, y in 1usize..256, z in 1usize..256, w in 1usize..8) {
            let conv_shape_og = Shape::new(vec![x, y, z]);
            let mat_shape_og = Shape::new(vec![w, conv_shape_og.product()]);

            let conv_shape_pad = conv_shape_og.next_power_of_two();
            let matrix_shape_pad = Shape::new(vec![w, conv_shape_pad.product()]);

            // The flattened output of a conv2d
            let conv_out_og = Tensor::<Element>::random(&conv_shape_og);
            let conv_out_og_flat = conv_out_og.to_flatten();

            // The layer after conv2d, in this case a dense layer
            let dense_og = Tensor::random(&mat_shape_og);
            let result_og = dense_og.matvec(&conv_out_og_flat).unwrap();

            // The equivalent output for a FFT-conv2d
            let mut conv_out_pad = conv_out_og.clone();
            conv_out_pad.pad_to_shape(conv_shape_pad.clone()).unwrap();
            let conv_out_pad_flat = conv_out_pad.to_flatten();

            // The layer after FFT-conv2d
            let dense_pad = dense_og
                .pad_matrix_to_ignore_garbage(&conv_shape_og, &conv_shape_pad, &matrix_shape_pad)
                .unwrap();
            let result_pad = dense_pad.matvec(&conv_out_pad_flat).unwrap();

            prop_assert_eq!(
                &result_og.data()[..mat_shape_og.dim(0)],
                &result_pad.data()[..mat_shape_og.dim(0)],
                "Unable to get rid of garbage values from conv fft.",
            );
        }
    }
}
