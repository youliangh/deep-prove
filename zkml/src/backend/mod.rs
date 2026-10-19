use burn::tensor::{Int, Tensor as BTensor, TensorData, ops::IntTensor, read_sync};

use crate::{Element, Shape, Tensor};

#[cfg(feature = "gpu")]
mod cubecl;

#[cfg(feature = "gpu")]
mod kernels;

#[cfg(feature = "cpu")]
mod ndarray;

#[cfg(all(not(feature = "wgpu"), not(feature = "cuda")))]
pub type Backend = burn::backend::NdArray<f32, Element>;

#[cfg(feature = "cuda")]
pub type Backend = burn::backend::Cuda<f32, Element>;

#[cfg(all(feature = "wgpu", not(feature = "cuda")))]
pub type Backend = burn::backend::Wgpu<f32, Element>;

#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    pub stride: usize,
}

#[derive(Debug, Clone)]
pub struct Maxpool2dConfig {
    pub kernel_size: usize,
    pub stride: usize,
}

fn to_tensor<B: ZKMLBackend>(data: IntTensor<B>) -> anyhow::Result<Tensor<Element>> {
    let data = read_sync(B::int_into_data(data)).expect("Failed to read input data");
    Tensor::new(
        Shape::new(data.shape.clone()),
        data.into_vec().expect("Couldnt convert input data"),
    )
}

pub(crate) trait ZKMLBackend: burn::tensor::backend::Backend {
    /// Exact integer GEMM. CPU backends may override the generic Burn kernel.
    fn zkml_matmul_i(
        lhs: IntTensor<Self>,
        rhs: IntTensor<Self>,
    ) -> anyhow::Result<IntTensor<Self>> {
        Ok(Self::int_matmul(lhs, rhs))
    }

    /// Conv2D implementation over integers (only floats are supported by burn)
    fn zkml_conv2d_i(
        input: IntTensor<Self>,
        kernels: IntTensor<Self>,
        bias: IntTensor<Self>,
        config: Conv2dConfig,
    ) -> anyhow::Result<IntTensor<Self>> {
        let device = Self::int_device(&input);

        // Convert the burn's to a local tensor and use our CPU implementation
        let input = to_tensor::<Self>(input)?;
        let kernels = to_tensor::<Self>(kernels)?;
        let bias = to_tensor::<Self>(bias)?;

        let res: Tensor<Element> = input.conv2d(&kernels, &bias, config.stride)?;

        let shape = res.shape().clone();
        Ok(Self::int_from_data(
            TensorData::new(res.into_data(), shape),
            &device,
        ))
    }

    /// Max Pool2d implementation over integers (only floats are supported by burn)
    fn zkml_max_pool2d_i(
        input: IntTensor<Self>,
        config: Maxpool2dConfig,
    ) -> anyhow::Result<IntTensor<Self>> {
        let device = Self::int_device(&input);

        // Convert the burn's to a local tensor and use our CPU implementation
        let input = to_tensor::<Self>(input)?;

        let res: Tensor<Element> = input.maxpool2d(config.kernel_size, config.stride)?;

        let shape = res.shape().clone();
        Ok(Self::int_from_data(
            TensorData::new(res.into_data(), shape),
            &device,
        ))
    }
}

pub(crate) fn zkml_matmul_i<B: ZKMLBackend, const D: usize>(
    lhs: BTensor<B, D, Int>,
    rhs: BTensor<B, D, Int>,
) -> anyhow::Result<BTensor<B, D, Int>> {
    Ok(BTensor::from_primitive(B::zkml_matmul_i(
        lhs.into_primitive(),
        rhs.into_primitive(),
    )?))
}

pub(crate) fn zkml_conv2d_i<B: ZKMLBackend>(
    input: BTensor<B, 4, Int>,
    kernels: BTensor<B, 4, Int>,
    bias: BTensor<B, 1, Int>,
    config: Conv2dConfig,
) -> anyhow::Result<BTensor<B, 4, Int>> {
    let output = B::zkml_conv2d_i(
        input.into_primitive(),
        kernels.into_primitive(),
        bias.into_primitive(),
        config,
    )?;
    Ok(BTensor::from_primitive(output))
}

pub(crate) fn zkml_max_pool2d_i<B: ZKMLBackend>(
    input: BTensor<B, 4, Int>,
    config: Maxpool2dConfig,
) -> anyhow::Result<BTensor<B, 4, Int>> {
    let output = B::zkml_max_pool2d_i(input.into_primitive(), config)?;
    Ok(BTensor::from_primitive(output))
}
