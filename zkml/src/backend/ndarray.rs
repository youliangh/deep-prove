use burn::backend::NdArray;

use crate::Element;

use super::ZKMLBackend;

mod matmul;

impl ZKMLBackend for NdArray<f32, Element> {
    fn zkml_matmul_i(
        lhs: burn::tensor::ops::IntTensor<Self>,
        rhs: burn::tensor::ops::IntTensor<Self>,
    ) -> anyhow::Result<burn::tensor::ops::IntTensor<Self>> {
        matmul::matmul(lhs, rhs)
    }
}
