//! Exact, cache-blocked CPU i64 matrix multiplication.
//!
//! Burn's ndarray integer GEMM uses a serial generic dot loop. Here disjoint
//! output blocks run on the existing Rayon pool; small RHS panels are packed
//! once per row block and reused. Float/GPU dispatch is unchanged. Arithmetic
//! explicitly has i64 wrapping semantics, never a floating-point shortcut.

use std::borrow::Cow;

use anyhow::{Result, anyhow, ensure};
use burn::tensor::{TensorData, ops::IntTensorOps};
use burn_ndarray::{NdArray, NdArrayTensor};
use rayon::prelude::*;

const ROW_BLOCK: usize = 8;
const COL_BLOCK: usize = 64;
const INNER_BLOCK: usize = 64;

struct Input<'a> {
    data: Cow<'a, [i64]>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    origin: isize,
}

impl<'a> Input<'a> {
    fn new(
        shape: &[usize],
        strides: &[isize],
        memory_order: Option<&'a [i64]>,
        logical_values: impl Iterator<Item = i64>,
    ) -> Self {
        if let Some(data) = memory_order {
            let origin = shape
                .iter()
                .zip(strides)
                .filter(|(_, s)| **s < 0)
                .map(|(&dim, &stride)| dim.saturating_sub(1) as isize * -stride)
                .sum();
            Self {
                data: Cow::Borrowed(data),
                shape: shape.to_vec(),
                strides: strides.to_vec(),
                origin,
            }
        } else {
            // Noncontiguous slices are packed in logical order. Permuted dense
            // weights use the branch above and never copy the full matrix.
            let mut strides = vec![1; shape.len()];
            for i in (0..shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * shape[i + 1] as isize;
            }
            Self {
                data: Cow::Owned(logical_values.collect()),
                shape: shape.to_vec(),
                strides,
                origin: 0,
            }
        }
    }

    fn batch_offset(&self, mut batch: usize, output_shape: &[usize]) -> isize {
        let mut offset = self.origin;
        for axis in (0..output_shape.len() - 2).rev() {
            let coord = batch % output_shape[axis];
            batch /= output_shape[axis];
            if self.shape[axis] != 1 {
                offset += coord as isize * self.strides[axis];
            }
        }
        offset
    }

    #[inline]
    fn at(&self, base: isize, row: usize, col: usize) -> i64 {
        let rank = self.shape.len();
        let offset =
            base + row as isize * self.strides[rank - 2] + col as isize * self.strides[rank - 1];
        self.data[offset as usize]
    }
}

fn output_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>> {
    ensure!(
        lhs.len() == rhs.len() && (2..=4).contains(&lhs.len()),
        "Integer GEMM requires equal ranks between 2 and 4"
    );
    let rank = lhs.len();
    ensure!(
        lhs[rank - 1] == rhs[rank - 2],
        "Integer GEMM inner dimensions differ"
    );
    let mut output = lhs.to_vec();
    for axis in 0..rank - 2 {
        output[axis] = if lhs[axis] == rhs[axis] {
            lhs[axis]
        } else if lhs[axis] == 1 {
            rhs[axis]
        } else if rhs[axis] == 1 {
            lhs[axis]
        } else {
            return Err(anyhow!(
                "Integer GEMM batch dimension {axis} cannot broadcast"
            ));
        };
    }
    output[rank - 1] = rhs[rank - 1];
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn multiply_block(
    lhs: &Input<'_>,
    rhs: &Input<'_>,
    lhs_base: isize,
    rhs_base: isize,
    row_start: usize,
    rows: usize,
    col_start: usize,
    columns: usize,
    output: &mut [i64],
    output_stride: usize,
) {
    let inner = lhs.shape[lhs.shape.len() - 1];
    let mut panel = vec![0i64; INNER_BLOCK * COL_BLOCK];
    for j0 in (0..columns).step_by(COL_BLOCK) {
        let width = COL_BLOCK.min(columns - j0);
        for k0 in (0..inner).step_by(INNER_BLOCK) {
            let depth = INNER_BLOCK.min(inner - k0);
            for k in 0..depth {
                for j in 0..width {
                    panel[k * width + j] = rhs.at(rhs_base, k0 + k, col_start + j0 + j);
                }
            }
            for row in 0..rows {
                let start = row * output_stride + j0;
                let output_row = &mut output[start..start + width];
                for k in 0..depth {
                    let a = lhs.at(lhs_base, row_start + row, k0 + k);
                    let rhs_row = &panel[k * width..(k + 1) * width];
                    for (c, &b) in output_row.iter_mut().zip(rhs_row) {
                        *c = c.wrapping_add(a.wrapping_mul(b));
                    }
                }
            }
        }
    }
}

pub(super) fn matmul(lhs: NdArrayTensor, rhs: NdArrayTensor) -> Result<NdArrayTensor> {
    let (NdArrayTensor::I64(lhs), NdArrayTensor::I64(rhs)) = (&lhs, &rhs) else {
        return Ok(NdArray::<f32, i64>::int_matmul(lhs, rhs));
    };
    let lhs_view = lhs.view();
    let rhs_view = rhs.view();
    let shape = output_shape(lhs_view.shape(), rhs_view.shape())?;
    let size = shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(anyhow!("Integer GEMM output shape overflows usize"))?;
    ensure!(
        size <= isize::MAX as usize / std::mem::size_of::<i64>(),
        "Integer GEMM output is too large"
    );
    let mut output = vec![0i64; size];
    let rank = shape.len();
    let rows = shape[rank - 2];
    let columns = shape[rank - 1];
    if size > 0 {
        let lhs = Input::new(
            lhs_view.shape(),
            lhs_view.strides(),
            lhs_view.as_slice_memory_order(),
            lhs_view.iter().copied(),
        );
        let rhs = Input::new(
            rhs_view.shape(),
            rhs_view.strides(),
            rhs_view.as_slice_memory_order(),
            rhs_view.iter().copied(),
        );
        output
            .par_chunks_mut(rows * columns)
            .enumerate()
            .for_each(|(batch, matrix)| {
                let lhs_base = lhs.batch_offset(batch, &shape);
                let rhs_base = rhs.batch_offset(batch, &shape);
                if rows <= ROW_BLOCK {
                    // Decode and tiny prompts still distribute the large vocabulary
                    // projection across cores, even when there is only one row.
                    matrix
                        .par_chunks_mut(columns)
                        .enumerate()
                        .for_each(|(row, output)| {
                            output.par_chunks_mut(COL_BLOCK).enumerate().for_each(
                                |(column_block, block)| {
                                    multiply_block(
                                        &lhs,
                                        &rhs,
                                        lhs_base,
                                        rhs_base,
                                        row,
                                        1,
                                        column_block * COL_BLOCK,
                                        block.len(),
                                        block,
                                        block.len(),
                                    );
                                },
                            );
                        });
                } else {
                    matrix
                        .par_chunks_mut(ROW_BLOCK * columns)
                        .enumerate()
                        .for_each(|(row_block, block)| {
                            multiply_block(
                                &lhs,
                                &rhs,
                                lhs_base,
                                rhs_base,
                                row_block * ROW_BLOCK,
                                block.len() / columns,
                                0,
                                columns,
                                block,
                                columns,
                            );
                        });
                }
            });
    }
    Ok(NdArray::<f32, i64>::int_from_data(
        TensorData::new(output, shape),
        &Default::default(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Slice;

    type Cpu = NdArray<f32, i64>;

    fn tensor(shape: &[usize], values: Vec<i64>) -> NdArrayTensor {
        Cpu::int_from_data(TensorData::new(values, shape.to_vec()), &Default::default())
    }

    fn sample(shape: &[usize], seed: usize) -> NdArrayTensor {
        tensor(
            shape,
            (0..shape.iter().product())
                .map(|i| ((i * 17 + seed) % 101) as i64 - 50)
                .collect(),
        )
    }

    fn data(tensor: NdArrayTensor) -> (Vec<usize>, Vec<i64>) {
        let data = burn::tensor::read_sync(Cpu::int_into_data(tensor)).unwrap();
        (data.shape.clone(), data.into_vec().unwrap())
    }

    // Deliberately scalar, logical-order reference, independent of storage
    // strides, blocking, parallelism, and the production batch-offset helper.
    fn reference(lhs: NdArrayTensor, rhs: NdArrayTensor) -> (Vec<usize>, Vec<i64>) {
        let (ls, lv) = data(lhs);
        let (rs, rv) = data(rhs);
        let rank = ls.len();
        let mut shape = ls.clone();
        for i in 0..rank - 2 {
            shape[i] = ls[i].max(rs[i]);
        }
        shape[rank - 1] = rs[rank - 1];
        let mut output = vec![0i64; shape.iter().product()];
        for (index, out) in output.iter_mut().enumerate() {
            let mut rem = index;
            let mut coord = vec![0; rank];
            for i in (0..rank).rev() {
                coord[i] = rem % shape[i];
                rem /= shape[i];
            }
            for k in 0..ls[rank - 1] {
                let mut li = 0;
                let mut ri = 0;
                for i in 0..rank - 2 {
                    li = li * ls[i] + coord[i] % ls[i];
                    ri = ri * rs[i] + coord[i] % rs[i];
                }
                li = (li * ls[rank - 2] + coord[rank - 2]) * ls[rank - 1] + k;
                ri = (ri * rs[rank - 2] + k) * rs[rank - 1] + coord[rank - 1];
                *out = out.wrapping_add(lv[li].wrapping_mul(rv[ri]));
            }
        }
        (shape, output)
    }

    fn compare(lhs: NdArrayTensor, rhs: NdArrayTensor) {
        let expected = reference(lhs.clone(), rhs.clone());
        let burn = data(Cpu::int_matmul(lhs.clone(), rhs.clone()));
        assert_eq!(burn, expected, "Burn/reference mismatch");
        assert_eq!(data(matmul(lhs, rhs).unwrap()), expected);
    }

    #[test]
    fn integer_gemm_ranks_broadcast_and_tile_edges() {
        for (m, k, n) in [
            (1, 7, 67),
            (3, 65, 63),
            (8, 11, 65),
            (9, 67, 3),
            (17, 65, 67),
        ] {
            compare(sample(&[m, k], 4), sample(&[k, n], 7));
        }
        for (lhs, rhs) in [
            (vec![1, 3, 11], vec![4, 11, 7]),
            (vec![4, 3, 11], vec![1, 11, 7]),
            // Burn's high-level batched-vector optimization has the same result.
            (vec![5, 1, 11], vec![1, 11, 7]),
            (vec![2, 1, 9, 11], vec![1, 3, 11, 7]),
            (vec![1, 3, 1, 11], vec![2, 1, 11, 7]),
        ] {
            compare(sample(&lhs, 9), sample(&rhs, 13));
        }
    }

    #[test]
    fn integer_gemm_transposed_flipped_and_sliced_storage() {
        let lhs = Cpu::int_swap_dims(sample(&[11, 7], 1), 0, 1);
        let rhs = Cpu::int_swap_dims(sample(&[13, 11], 2), 0, 1);
        compare(lhs.clone(), rhs.clone());
        compare(Cpu::int_flip(lhs, &[0, 1]), Cpu::int_flip(rhs, &[0, 1]));
        let lhs = Cpu::int_slice(
            sample(&[9, 23], 3),
            &[
                Slice::with_step(1, Some(8), 1),
                Slice::with_step(1, Some(23), 2),
            ],
        );
        let rhs = Cpu::int_slice(
            sample(&[23, 17], 5),
            &[
                Slice::with_step(1, Some(23), 2),
                Slice::with_step(2, Some(15), 1),
            ],
        );
        compare(lhs, rhs);
        compare(
            Cpu::int_permute(sample(&[9, 2, 11, 1], 7), &[1, 3, 0, 2]),
            Cpu::int_permute(sample(&[7, 1, 3, 11], 8), &[1, 2, 3, 0]),
        );
    }

    #[test]
    fn integer_gemm_exact_large_values_and_wrapping() {
        let large = (1i64 << 54) + 1;
        compare(
            tensor(&[2, 2], vec![large, -large + 2, 7, -9]),
            tensor(&[2, 2], vec![2, -3, -1, 2]),
        );
        let lhs = tensor(
            &[2, 3],
            vec![i64::MAX, i64::MIN, -3, i64::MIN + 1, 9, i64::MAX - 1],
        );
        let rhs = tensor(&[3, 2], vec![2, -1, -1, 3, i64::MAX, i64::MIN]);
        let expected = reference(lhs.clone(), rhs.clone());
        assert_eq!(data(matmul(lhs, rhs).unwrap()), expected);
    }

    #[test]
    fn integer_gemm_storage_origin_and_logical_fallback() {
        let values = [0, 1, 2, 3, 4, 5];
        let reversed = Input::new(&[2, 3], &[-3, -1], Some(&values), std::iter::empty());
        assert!(matches!(reversed.data, Cow::Borrowed(_)));
        assert_eq!(reversed.at(reversed.origin, 0, 0), 5);
        assert_eq!(reversed.at(reversed.origin, 1, 2), 0);
        let packed = Input::new(&[2, 3], &[8, 2], None, values.into_iter());
        assert!(matches!(packed.data, Cow::Owned(_)));
        assert_eq!(packed.at(packed.origin, 1, 2), 5);
    }

    #[test]
    fn integer_gemm_empty_and_invalid_shapes() {
        assert_eq!(
            data(matmul(sample(&[0, 3], 0), sample(&[3, 4], 0)).unwrap()),
            (vec![0, 4], vec![])
        );
        assert_eq!(
            data(matmul(sample(&[2, 0], 0), sample(&[0, 4], 0)).unwrap()),
            (vec![2, 4], vec![0; 8])
        );
        assert!(matmul(sample(&[2, 3], 0), sample(&[4, 5], 0)).is_err());
        assert!(matmul(sample(&[2, 2, 3], 0), sample(&[3, 3, 4], 0)).is_err());
        assert!(matmul(sample(&[2, 3], 0), sample(&[1, 3, 4], 0)).is_err());
    }

    #[test]
    #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
    fn integer_gemm_wrapper_preserves_padding_and_float_dispatch() {
        use crate::tensor::{TensorTypeParam, WrappedTensor};
        use burn::tensor::{Int, Shape, Tensor};
        let lhs = Tensor::<Cpu, 2, Int>::from_primitive(sample(&[4, 8], 3));
        let rhs = Tensor::<Cpu, 2, Int>::from_primitive(sample(&[8, 16], 4));
        let expected = data(Cpu::int_matmul(
            lhs.clone().into_primitive(),
            rhs.clone().into_primitive(),
        ));
        let output = WrappedTensor::<i64>::Rank2 {
            tensor: lhs,
            unpadded_shape: Shape::new([3, 7]),
        }
        .matmul(WrappedTensor::Rank2 {
            tensor: rhs,
            unpadded_shape: Shape::new([7, 13]),
        })
        .unwrap();
        let WrappedTensor::Rank2 {
            tensor,
            unpadded_shape,
        } = output
        else {
            panic!("rank changed")
        };
        assert_eq!(unpadded_shape.dims, vec![3, 13]);
        assert_eq!(data(tensor.into_primitive()), expected);
        let lhs = Tensor::<Cpu, 2>::from_data([[1.25f32, -2.5], [3.0, 4.0]], &Default::default());
        let rhs = Tensor::<Cpu, 2>::from_data([[5.0f32, 6.0], [-7.0, 8.0]], &Default::default());
        assert_eq!(
            <f32 as TensorTypeParam>::matmul(lhs.clone(), rhs.clone())
                .unwrap()
                .into_data(),
            lhs.matmul(rhs).into_data()
        );
    }

    /// One representative opt-in timing; not part of routine test runs.
    #[test]
    #[ignore = "manual release timing: 64x2048 @ 2048x2048"]
    fn integer_gemm_release_timing() {
        use std::time::Instant;
        // Invoke with --release: the old kernel's overflow parity is checked
        // here because the regular test profile enables overflow checks.
        let overflow_lhs = tensor(&[1, 2], vec![i64::MAX, i64::MIN]);
        let overflow_rhs = tensor(&[2, 1], vec![2, -1]);
        let expected = reference(overflow_lhs.clone(), overflow_rhs.clone());
        assert_eq!(
            data(Cpu::int_matmul(overflow_lhs.clone(), overflow_rhs.clone())),
            expected
        );
        assert_eq!(data(matmul(overflow_lhs, overflow_rhs).unwrap()), expected);
        let lhs = sample(&[64, 2048], 5);
        // Linear weights arrive as a transposed contiguous [N,K] allocation.
        let rhs = Cpu::int_swap_dims(sample(&[2048, 2048], 7), 0, 1);
        let start = Instant::now();
        let expected = data(Cpu::int_matmul(lhs.clone(), rhs.clone()));
        let burn_elapsed = start.elapsed();
        let start = Instant::now();
        let actual = data(matmul(lhs, rhs).unwrap());
        let blocked_elapsed = start.elapsed();
        assert_eq!(actual, expected);
        eprintln!(
            "i64 GEMM 64x2048x2048, threads={}: Burn={burn_elapsed:?}, blocked={blocked_elapsed:?}, speedup={:.2}x",
            rayon::current_num_threads(),
            burn_elapsed.as_secs_f64() / blocked_elapsed.as_secs_f64()
        );
    }
}
