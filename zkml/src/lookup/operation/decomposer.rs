//! This module contains the [`ChunkingInfo`] struct and its associated methods.
//! This is used for general chunking of input values for lookup operations.

use crate::lookup::table::{
    SHIFT_CHECK_TABLE_BIT_SIZE, TableSign, TableType, ZERO_CHECK_TABLE_BIT_SIZE,
};

use super::*;
use anyhow::{Result, ensure};

#[derive(Debug, Clone, Copy)]
/// This struct holds all of the information needed for decomposing input values into shifted chunks, value chunk and zeroing chunks.
pub struct ChunkingInfo {
    /// This defines the how many bits are used for the value chunk
    value_chunk_size: usize,
    /// This is the number of value chunks that are looked up. If this is greater than 1 then
    /// each chunk is looked up in the lookup table separately and the outputs are combined in some way.
    number_of_value_chunks: usize,
    /// This is the size of the right shift being applied to the input value. It determines how many shifted_chunks there are
    /// by ceiling dividing `right_shift` by [`SHIFT_CHECK_TABLE_BIT_SIZE`].
    right_shift: usize,
    /// This is the offset to apply to the value before decomposition. It will be zero if the input values always have the same sign,
    /// if not it will be set to the minimum possible input value plus (1 << (value_chunk_size - 1)) << right_shift.
    offset: Element,
    /// This is the offset to subtract from the value chunk after decomposition. If the input is signed this will be (1 << (value_chunk_size - 1)), otherwise it will be zero.
    value_chunk_offset: Element,
    /// This value is the total number of shifted chunks.
    number_of_shifted_chunks: usize,
    /// This value is the total number of zeroing chunks.
    number_of_zeroing_chunks: usize,
    /// Store whether the input is signed or not.
    is_signed: bool,
    /// The most significant `shifted_chunk` values are multiplied by this value to ensure they are range checked correctly.
    /// If the `right_shift` is not a multiple of [`SHIFT_CHECK_TABLE_BIT_SIZE`] then this will be 1 << ([`SHIFT_CHECK_TABLE_BIT_SIZE`] - right_shift % [`SHIFT_CHECK_TABLE_BIT_SIZE`]), otherwise it will be 1.
    shifted_chunk_multiplier: Element,
    /// This is the offset applied to the most significant zeroing chunk which allows us to correctly handle the clamping of signed values.
    /// It is calculated as offset >> (right_shift + value_chunk_size + (number_of_zeroing_chunks - 1) * [`ZERO_CHECK_TABLE_BIT_SIZE`]).
    top_zeroing_chunk_offset: Element,
    /// The value table associated with this chunking info.
    table: Table,
}

#[derive(Debug, Clone)]
/// This struct holds the decomposed chunks of an input value.
pub struct ChunkedInput {
    /// Shifted chunks obtained from right shifting the input value.
    shifted_chunks: Vec<Vec<Element>>,
    /// The main value chunks obtained after right shifting the input value.
    value_chunks: Vec<Vec<Element>>,
    /// Zeroing chunks obtained from the higher bits of the input value.
    /// It has had the top_zeroing_chunk_offset subtracted from the most significant zeroing chunk if the input is signed.
    zeroing_chunks: Vec<Vec<Element>>,
}

impl ChunkedInput {
    /// Initialises an empty [`ChunkedInput`] given the number of shifted chunks and zeroing chunks.
    pub fn new(
        number_of_shifted_chunks: usize,
        number_of_value_chunks: usize,
        number_of_zeroing_chunks: usize,
    ) -> Self {
        Self {
            shifted_chunks: vec![vec![]; number_of_shifted_chunks],
            value_chunks: vec![vec![]; number_of_value_chunks],
            zeroing_chunks: vec![vec![]; number_of_zeroing_chunks],
        }
    }

    pub fn to_evals<F: PrimeField>(&self, num_value_columns: usize) -> Vec<Vec<F>> {
        let mut evals = Vec::with_capacity(
            self.shifted_chunks.len() + (num_value_columns - 1) + self.zeroing_chunks.len(),
        );
        for chunk in &self.shifted_chunks {
            evals.push(chunk.to_field());
        }
        if num_value_columns != 1 {
            for chunk in &self.value_chunks {
                evals.push(chunk.to_field());
            }
        }
        for chunk in &self.zeroing_chunks {
            evals.push(chunk.to_field());
        }
        evals
    }

    pub fn shifted_iter(&self) -> impl Iterator<Item = &Element> + '_ {
        self.shifted_chunks.iter().flatten()
    }

    pub fn value_iter(&self) -> impl Iterator<Item = &Element> + '_ {
        self.value_chunks.iter().flatten()
    }

    pub fn zeroing_iter_no_sign(&self) -> impl Iterator<Item = &Element> + '_ {
        self.zeroing_chunks.iter().flatten()
    }

    pub fn signed_chunk_iter(&self) -> impl Iterator<Item = &Element> + '_ {
        if let Some(last_chunk) = self.zeroing_chunks.last() {
            last_chunk.iter()
        } else {
            [].iter()
        }
    }

    pub fn unsigned_chunk_iter(&self) -> impl Iterator<Item = &Element> + '_ {
        self.zeroing_chunks
            .iter()
            .take(self.zeroing_chunks.len().saturating_sub(1))
            .flatten()
    }
}

#[derive(Debug, Clone)]
/// This struct holds the decomposed chunks of a lookup output value.
pub struct ChunkedOutput {
    /// The output corresponding to the value chunks.
    pub value_chunk_outputs: Vec<Vec<Element>>,
    /// The outputs corresponding to the zeroing chunks.
    pub zeroing_chunk_outputs: Vec<Vec<Element>>,
}

impl ChunkedOutput {
    /// Initialises an empty [`ChunkedOutput`] given the number of zeroing chunks.
    pub fn new(
        value_chunk_outputs: Vec<Vec<Element>>,
        zeroing_chunk_outputs: Vec<Vec<Element>>,
    ) -> Self {
        Self {
            value_chunk_outputs,
            zeroing_chunk_outputs,
        }
    }

    pub fn to_evals<F: PrimeField>(&self) -> Vec<Vec<F>> {
        let mut evals = Vec::new();
        for chunk in &self.value_chunk_outputs {
            evals.push(chunk.to_field());
        }
        for chunk in &self.zeroing_chunk_outputs {
            evals.push(chunk.to_field());
        }
        evals
    }

    pub fn value_iter(&self) -> impl Iterator<Item = &Element> + '_ {
        self.value_chunk_outputs.iter().flatten()
    }

    pub fn zeroing_iter_no_sign(&self) -> impl Iterator<Item = &Element> + '_ {
        self.zeroing_chunk_outputs.iter().flatten()
    }

    pub fn signed_chunk_iter(&self) -> impl Iterator<Item = &Element> + '_ {
        if let Some(last_chunk) = self.zeroing_chunk_outputs.last() {
            last_chunk.iter()
        } else {
            [].iter()
        }
    }

    pub fn unsigned_chunk_iter(&self) -> impl Iterator<Item = &Element> + '_ {
        self.zeroing_chunk_outputs
            .iter()
            .take(self.zeroing_chunk_outputs.len().saturating_sub(1))
            .flatten()
    }
}

impl ChunkingInfo {
    /// Create a new [`ChunkingInfo`] given the right shift, value chunk size, whether the input is signed and the maximum bit size of the input.
    pub fn new(
        right_shift: usize,
        table: &Table,
        max_bit_size: usize,
        number_of_value_chunks: usize,
    ) -> Result<Self> {
        let value_chunk_size = table.table_bit_size();
        let table_sign = table.operation().input_sign();
        // We check that the right shift is non-zero
        ensure!(
            right_shift > 0,
            "Right shift must be greater than zero to decompose value, got {right_shift}"
        );

        // Currently we only support multiple chunks in requantisation operations, so if value chunks > 1 the TableType has to be Requantisation
        ensure!(
            number_of_value_chunks == 1 || table.operation() == TableType::Requantise,
            "Multiple value chunks are currently only supported for Requantisation tables, got {number_of_value_chunks} value chunks and table type {:?}",
            table.operation(),
        );

        // Calculate the number of shifted chunks and zeroing chunks
        let number_of_shifted_chunks = right_shift.div_ceil(SHIFT_CHECK_TABLE_BIT_SIZE);
        let number_of_zeroing_chunks = (max_bit_size
            .saturating_sub(right_shift + value_chunk_size * number_of_value_chunks))
        .div_ceil(ZERO_CHECK_TABLE_BIT_SIZE);

        // Now we calculate the shifted chunk multiplier
        let shifted_chunk_multiplier = if !right_shift.is_multiple_of(SHIFT_CHECK_TABLE_BIT_SIZE) {
            1i64 << (SHIFT_CHECK_TABLE_BIT_SIZE - (right_shift % SHIFT_CHECK_TABLE_BIT_SIZE))
        } else {
            1
        };

        // Calculate the offset and zeroing chunk offset
        // If the TableSign is `Mixed` then we need to ensure all the values
        // are positive before decomposition. To do this we work out the largest possible negative value and add an offset to make this value zero.
        // This means the offset is at least the minimum possible input value plus (1 << (value_chunk_size - 1)) << right_shift to ensure that after applying the offset the value chunk is always non-negative.
        // If the input is signed then we also need to subtract (1 << (value_chunk_size - 1)) from the value chunk after decomposition to ensure it is in the correct range for lookup in the table.
        let (offset, value_chunk_offset) = match table_sign {
            TableSign::Mixed => {
                let value_chunk_offset = 1i64 << (value_chunk_size - 1);
                let combined_value_offset = 1i64 << (value_chunk_size * number_of_value_chunks - 1);
                let no_zero_chunks = if number_of_zeroing_chunks > 0 { 1 } else { 0 };
                let first_offset_part =
                    (1i64 << (max_bit_size - 1)).max(combined_value_offset << right_shift);
                let offset =
                    first_offset_part + no_zero_chunks * (combined_value_offset << right_shift);
                (offset, value_chunk_offset)
            }
            TableSign::Positive | TableSign::Negative => (0, 0),
        };

        let top_zeroing_chunk_offset =
            if number_of_zeroing_chunks > 0 && matches!(table_sign, TableSign::Mixed) {
                offset
                    >> (right_shift
                        + value_chunk_size * number_of_value_chunks
                        + (number_of_zeroing_chunks - 1) * ZERO_CHECK_TABLE_BIT_SIZE)
            } else {
                0
            };

        Ok(Self {
            value_chunk_size,
            number_of_value_chunks,
            right_shift,
            offset,
            value_chunk_offset,
            number_of_shifted_chunks,
            number_of_zeroing_chunks,
            is_signed: matches!(table_sign, TableSign::Mixed),
            shifted_chunk_multiplier,
            top_zeroing_chunk_offset,
            table: *table,
        })
    }

    /// Decomposes a vector of input values into shifted chunks, value chunk and zeroing chunks.
    pub fn decompose_input(&self, input: Vec<Element>) -> ChunkedInput {
        input.into_iter().fold(
            ChunkedInput::new(
                self.number_of_shifted_chunks,
                self.number_of_value_chunks,
                self.number_of_zeroing_chunks,
            ),
            |mut acc, value| {
                // The offset is only an encoding aid, not a proved bound on
                // the input. In particular, a valid signed i64 input plus the
                // offset can exceed i64::MAX. Keep the exact integer here;
                // only the range-checked limbs are converted back to Element.
                let mut remaining_value = match self.table.table_sign() {
                    TableSign::Mixed => value as i128 + self.offset as i128,
                    TableSign::Positive => value as i128,
                    TableSign::Negative => -(value as i128),
                };

                // Store the shifted chunks
                let shift_chunk_mask = (1i128 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1;
                for (i, chunk_storage) in acc.shifted_chunks.iter_mut().enumerate() {
                    if i != self.number_of_shifted_chunks - 1
                        || self.right_shift.is_multiple_of(SHIFT_CHECK_TABLE_BIT_SIZE)
                    {
                        let chunk = remaining_value & shift_chunk_mask;
                        chunk_storage.push(chunk as Element);
                        remaining_value >>= SHIFT_CHECK_TABLE_BIT_SIZE;
                    } else {
                        let chunk_size = self.right_shift % SHIFT_CHECK_TABLE_BIT_SIZE;
                        let chunk_mask = (1i128 << chunk_size) - 1;
                        let chunk = remaining_value & chunk_mask;
                        chunk_storage
                            .push((chunk * self.shifted_chunk_multiplier as i128) as Element);
                        remaining_value >>= chunk_size;
                    };
                }

                // Store the value chunks
                let value_chunk_mask = (1i128 << self.value_chunk_size) - 1;
                for chunk_storage in acc.value_chunks.iter_mut() {
                    let value_chunk = remaining_value & value_chunk_mask;
                    let offset_value = match self.table.table_sign() {
                        TableSign::Mixed => value_chunk - self.value_chunk_offset as i128,
                        TableSign::Positive => value_chunk,
                        TableSign::Negative => -value_chunk,
                    };
                    chunk_storage.push(offset_value as Element);
                    remaining_value >>= self.value_chunk_size;
                }

                // Store the zeroing chunks
                let zero_chunk_mask = (1i128 << ZERO_CHECK_TABLE_BIT_SIZE) - 1;
                for (i, chunk_storage) in acc.zeroing_chunks.iter_mut().enumerate() {
                    if i != self.number_of_zeroing_chunks - 1 || !self.is_signed {
                        chunk_storage.push((remaining_value & zero_chunk_mask) as Element);
                    } else {
                        // This is the SIGNED top limb. Masking it first maps a
                        // negative quotient such as -1 to 65535 and can also
                        // discard nonzero high bits. The signed lookup checks
                        // the exact quotient against its unchanged 16-bit
                        // domain; reconstruction binds it to the original input.
                        chunk_storage.push(
                            (remaining_value - self.top_zeroing_chunk_offset as i128) as Element,
                        );
                    }
                    remaining_value >>= ZERO_CHECK_TABLE_BIT_SIZE;
                }
                acc
            },
        )
    }

    /// Given a lookup table `value_table`, this function performs lookups for all decomposed values and returns the outputs as a [`ChunkedOutput`].
    pub fn table_output(
        &self,
        chunked_input: &ChunkedInput,
        value_table: &Table,
    ) -> Result<ChunkedOutput> {
        // Lookup the value chunk using the provided table
        let value_chunk_outputs = chunked_input
            .value_chunks
            .iter()
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|&val| value_table.lookup(val))
                    .collect::<Result<Vec<Element>>>()
            })
            .collect::<Result<Vec<Vec<Element>>>>()?;

        // Now we match on the number of zeroing chunks and whether the input is signed to perform the correct lookups
        // for each zeroing chunk
        let zero_check_table = Table::new_zero_check();
        let zeroing_chunk_outputs = match (self.number_of_zeroing_chunks, self.is_signed) {
            (0, _) => {
                vec![]
            } // No zero chunks, no zero table lookups needed
            (_, false) => {
                // In this case we can use the zero check table for all zeroing chunks because if clamping is required
                // we must be clamping to the maximum output value of the value table
                chunked_input
                    .zeroing_chunks
                    .iter()
                    .map(|chunk| {
                        chunk
                            .iter()
                            .map(|&val| zero_check_table.lookup(val))
                            .collect::<Result<Vec<Element>>>()
                    })
                    .collect::<Result<Vec<Vec<Element>>>>()?
            }
            (_, true) => {
                // In this case we need to use the zero check table for all but the most significant zeroing chunk
                // which needs to use the signed zero check table
                let signed_zero_check_table = Table::new_signed_zero_check();

                chunked_input
                    .zeroing_chunks
                    .iter()
                    .enumerate()
                    .map(|(i, chunk)| {
                        if i != self.number_of_zeroing_chunks - 1 {
                            chunk
                                .iter()
                                .map(|&val| zero_check_table.lookup(val))
                                .collect::<Result<Vec<Element>>>()
                        } else {
                            chunk
                                .iter()
                                .map(|&val| signed_zero_check_table.lookup(val))
                                .collect::<Result<Vec<Element>>>()
                        }
                    })
                    .collect::<Result<Vec<Vec<Element>>>>()?
            }
        };
        Ok(ChunkedOutput::new(
            value_chunk_outputs,
            zeroing_chunk_outputs,
        ))
    }

    /// Getter for the number of shifted chunks.
    pub fn number_of_shifted_chunks(&self) -> usize {
        self.number_of_shifted_chunks
    }
    /// Getter for the number of zeroing chunks.
    pub fn number_of_zeroing_chunks(&self) -> usize {
        self.number_of_zeroing_chunks
    }

    /// Getter for the number of value chunks.
    pub fn number_of_value_chunks(&self) -> usize {
        self.number_of_value_chunks
    }

    /// Getter for the number of value table columns.
    pub fn num_value_columns(&self) -> usize {
        self.table.num_columns()
    }

    /// Getter for whether the input is signed.
    pub fn is_signed(&self) -> bool {
        self.table.is_signed()
    }

    /// Getter for the value [`Table`]
    pub fn table(&self) -> &Table {
        &self.table
    }

    /// Gets the total number of input mles per chunk.
    pub fn total_inputs_per_chunk(&self) -> usize {
        self.number_of_shifted_chunks
            + self.number_of_value_chunks * (self.table.num_columns() - 1)
            + self.number_of_zeroing_chunks
    }

    /// Gets the total number of output mles per chunk.
    pub fn total_outputs_per_chunk(&self) -> usize {
        self.number_of_value_chunks + self.number_of_zeroing_chunks
    }

    /// Get the rounding constant used.
    pub fn rounding_constant(&self) -> Element {
        1 << (self.right_shift - 1)
    }

    /// Function to recombine input claims
    pub fn combine_input_claims<F: PrimeField>(
        &self,
        shifted_claims: &[F],
        value_claims: &[F],
        zeroing_claims: &[F],
    ) -> F {
        let offset_val =
            shifted_claims
                .iter()
                .enumerate()
                .fold(F::ZERO, |acc, (j, &chunk_claim)| {
                    let shift_amount = j * SHIFT_CHECK_TABLE_BIT_SIZE;
                    let shift_value_field = F::from(1u64 << shift_amount);
                    if j != self.number_of_shifted_chunks - 1
                        || self.right_shift.is_multiple_of(SHIFT_CHECK_TABLE_BIT_SIZE)
                    {
                        acc + (chunk_claim * shift_value_field)
                    } else {
                        let shifted_chunk_multiplier_field =
                            F::from(self.shifted_chunk_multiplier as u64);
                        let inv = shifted_chunk_multiplier_field
                            .inverse()
                            .expect("Tried to invert 0 when inverting shifted chunk multiplier");
                        acc + ((chunk_claim * inv) * shift_value_field)
                    }
                });
        let right_shift_field = F::from(1u64 << self.right_shift);
        let value_chunk_offset_field: F = self.value_chunk_offset.to_field();

        let combined_value =
            value_claims
                .iter()
                .enumerate()
                .fold(F::ZERO, |acc, (j, &value_claim)| {
                    let shift_amount = j * self.value_chunk_size;
                    let shift_value_field = F::from(1u64 << shift_amount);
                    let reconstructed_value = match self.table.table_sign() {
                        TableSign::Mixed => value_claim + value_chunk_offset_field,
                        TableSign::Positive => value_claim,
                        TableSign::Negative => -value_claim,
                    };
                    acc + (reconstructed_value * shift_value_field)
                });

        let initial_val = offset_val + (combined_value * right_shift_field);

        let full_offset_val =
            zeroing_claims
                .iter()
                .enumerate()
                .fold(initial_val, |acc, (j, &chunk_claim)| {
                    let shift_amount = self.value_chunk_size * self.number_of_value_chunks
                        + self.right_shift
                        + j * ZERO_CHECK_TABLE_BIT_SIZE;
                    let shift_value_field = F::from(1u64 << shift_amount);
                    if j != self.number_of_zeroing_chunks - 1 || !self.is_signed {
                        acc + (chunk_claim * shift_value_field)
                    } else {
                        let top_chunk_offset_field: F = self.top_zeroing_chunk_offset.to_field();
                        acc + ((chunk_claim + top_chunk_offset_field) * shift_value_field)
                    }
                });
        let offset_field: F = self.offset.to_field();
        match self.table.table_sign() {
            TableSign::Mixed => full_offset_val - offset_field,
            TableSign::Positive => full_offset_val,
            TableSign::Negative => -full_offset_val,
        }
    }

    #[cfg(test)]
    /// Used for testing/sanity checking to see if the values obtained in sumchecks/lookups are the same as the ones obtained during inference
    pub fn combine_outputs(&self, chunked_output: &ChunkedOutput) -> Vec<Element> {
        let number_value_chunks = self.number_of_value_chunks();
        let number_zero_chunks = self.number_of_zeroing_chunks();

        let len = chunked_output.value_chunk_outputs[0].len();
        let mut full_output = Vec::<Element>::with_capacity(len);
        let table = self.table();

        for i in 0..len {
            let value = if number_value_chunks == 1 {
                chunked_output.value_chunk_outputs[0][i]
            } else {
                let full_value_offset: Element =
                    1 << (number_value_chunks * table.table_bit_size() - 1);

                chunked_output.value_chunk_outputs.iter().enumerate().fold(
                    -full_value_offset,
                    |value_acc, (idx, value_chunk)| {
                        let shift_amount: Element = 1 << (table.table_bit_size() * idx);
                        let value_offset_expr: Element = 1 << (table.table_bit_size() - 1);
                        let value_part = shift_amount * (value_chunk[i] + value_offset_expr);

                        value_acc + value_part
                    },
                )
            };

            if table.is_signed() {
                let mut prod = chunked_output
                    .zeroing_chunk_outputs
                    .iter()
                    .take(number_zero_chunks.saturating_sub(1))
                    .fold(1, |acc, zero_chunk| acc * zero_chunk[i]);

                let (clamping_min, clamping_max) = if self.number_of_value_chunks() == 1 {
                    (table.min_output_value(), table.max_output_value())
                } else {
                    let full_bit_size = self.number_of_value_chunks() * table.table_bit_size();
                    let min: Element = -1 << (full_bit_size - 1);
                    let max: Element = (1 << (full_bit_size - 1)) - 1;
                    (min, max)
                };

                let clamping_expression = if number_zero_chunks != 0 {
                    let last_chunk_expr =
                        chunked_output.zeroing_chunk_outputs[number_zero_chunks - 1][i];

                    let last_chunk_bit = 1 - last_chunk_expr * last_chunk_expr;
                    let lower_chunks = prod;
                    prod *= 1 - last_chunk_expr * last_chunk_expr;

                    let max_coeff = clamping_max;
                    let min_coeff = clamping_min;

                    let clamping_first_part = last_chunk_expr * (last_chunk_expr + 1) / 2
                        + (last_chunk_bit - lower_chunks * last_chunk_bit);
                    let clamping_second_part = last_chunk_expr * (last_chunk_expr - 1) / 2;

                    clamping_first_part * max_coeff + clamping_second_part * min_coeff
                } else {
                    // No zero chunks means no clamping needed
                    0
                };

                full_output.push(clamping_expression + prod * value);
            } else {
                let prod = chunked_output
                    .zeroing_chunk_outputs
                    .iter()
                    .fold(1, |prod_acc, zero_chunk| prod_acc * zero_chunk[i]);

                let clamping_expression = if number_zero_chunks != 0 {
                    // The clamping value is based on whether inputs are positive or negative.
                    match table.operation().input_sign() {
                        TableSign::Positive => table.max_output_value(),
                        TableSign::Negative => table.min_output_value(),
                        TableSign::Mixed => {
                            unreachable!("Already checked that the table doesn't have mixed signs")
                        }
                    }
                } else {
                    1
                };
                full_output.push(clamping_expression + prod * (value - clamping_expression));
            }
        }

        full_output
    }
}

#[cfg(test)]
mod tests {
    use crate::quantization::ToElement;

    use super::*;
    use ark_bn254::Fr as F;
    use proptest::prelude::*;

    #[test]
    fn qwen_lookup_signed_top_reproduces_65533_without_relaxing_range() -> Result<()> {
        let table = Table::new(
            TableType::Requantise,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            14,
        );
        let info = ChunkingInfo::new(15, &table, 31, 1)?;
        let rounded = -50_000i64 * 32768 + (1 << 14);
        // The old unsigned mask turned an exact top quotient of -1 into
        // 65535, then subtracted 2. This is the full-model failure value.
        let old_top = (((rounded + info.offset) >> 29) & 65535) - info.top_zeroing_chunk_offset;
        assert_eq!(old_top, 65533);
        assert!(Table::new_signed_zero_check().lookup(old_top).is_err());
        let inputs = [-50_000, -40_960, -32_769, -32_768, -1, 0, 32_767, 50_000]
            .map(|x| x * 32768 + (1 << 14));
        let chunks = info.decompose_input(inputs.to_vec());
        assert_eq!(chunks.zeroing_chunks[0][0], -3);
        let output = info.table_output(&chunks, &table)?;
        assert_eq!(
            info.combine_outputs(&output),
            inputs
                .iter()
                .map(|x| table.lookup(x >> 15))
                .collect::<Result<Vec<_>>>()?
        );
        for (i, &input) in inputs.iter().enumerate() {
            let claim = info.combine_input_claims::<F>(
                &chunks
                    .shifted_chunks
                    .iter()
                    .map(|c| c[i].to_field())
                    .collect::<Vec<_>>(),
                &chunks
                    .value_chunks
                    .iter()
                    .map(|c| c[i].to_field())
                    .collect::<Vec<_>>(),
                &chunks
                    .zeroing_chunks
                    .iter()
                    .map(|c| c[i].to_field())
                    .collect::<Vec<_>>(),
            );
            assert_eq!(claim, input.to_field());
        }
        // The same signed 16-bit table still rejects either out-of-range side.
        assert!(Table::new_signed_zero_check().lookup(-32769).is_err());
        assert!(Table::new_signed_zero_check().lookup(32768).is_err());
        Ok(())
    }

    #[test]
    fn qwen_lookup_signed_top_handles_offset_overflow_exactly() -> Result<()> {
        let table = Table::new(
            TableType::Requantise,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            14,
        );
        let info = ChunkingInfo::new(47, &table, 63, 1)?;
        let input = (1i64 << 62) - 1;
        assert!(input.checked_add(info.offset).is_none());
        let inputs = [-(1i64 << 62), -1, 0, input];
        let chunks = info.decompose_input(inputs.to_vec());
        let output = info.table_output(&chunks, &table)?;
        assert_eq!(
            info.combine_outputs(&output),
            inputs
                .iter()
                .map(|x| table.lookup(x >> 47))
                .collect::<Result<Vec<_>>>()?
        );
        for (i, &input) in inputs.iter().enumerate() {
            assert_eq!(
                info.combine_input_claims::<F>(
                    &chunks
                        .shifted_chunks
                        .iter()
                        .map(|c| c[i].to_field())
                        .collect::<Vec<_>>(),
                    &chunks
                        .value_chunks
                        .iter()
                        .map(|c| c[i].to_field())
                        .collect::<Vec<_>>(),
                    &chunks
                        .zeroing_chunks
                        .iter()
                        .map(|c| c[i].to_field())
                        .collect::<Vec<_>>(),
                ),
                input.to_field()
            );
        }
        Ok(())
    }

    #[test]
    fn qwen_lookup_silu16_output12_and_signed_clamp_complete_proofs() -> Result<()> {
        use crate::{
            ScalingFactor,
            layers::{
                Layer,
                activation::{Activation, ActivationLayer, lookup_data::ActivationLookupData},
                requant::Requant,
            },
            model::Model,
            padding::pad_model,
            testing::Pcs,
        };
        use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
        use std::marker::PhantomData;
        type P<'a, 'b> = crate::Prover<'a, 'b, F, Blake3Transcript, Pcs>;
        let prove = |model: Model<Element>, inputs: Vec<Tensor<Element>>| -> Result<()> {
            let model = pad_model(model)?;
            let trace = model.run(model.prepare_inputs(inputs)?, &mut Default::default())?;
            let (prover, verifier) = model.generate_contexts::<F, Pcs>()?;
            let (proof, io) = P::prove(&prover, trace, &model)?;
            crate::verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), io.clone())?;
            for port in 0..=io.input.len() {
                let mut bad = io.clone();
                let tensor = if port == bad.input.len() {
                    &mut bad.output[0]
                } else {
                    &mut bad.input[port]
                };
                let mut data = tensor.data().to_vec();
                data[0] = crate::SerializableField(*data[0] + F::from(1u64));
                *tensor = Tensor::new_with_unpadded_shape(
                    tensor.shape().clone(),
                    tensor.unpadded_shape().clone(),
                    data,
                )?;
                let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    crate::verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), bad)
                }));
                assert!(
                    !matches!(rejected, Ok(Ok(()))),
                    "accepted modified public port {port}"
                );
            }
            let mut encoded = serde_json::to_value(&proof)?;
            let requant = encoded["chunk_proofs"][0]["steps"]
                .as_object_mut()
                .unwrap()
                .values_mut()
                .find_map(|step| step.get_mut("Requant"))
                .expect("the regression graph includes Requant");
            let mut changed: crate::layers::requant::RequantProof<F, Pcs> =
                serde_json::from_value(requant.clone())?;
            changed.io_eval[0] += F::from(1u64);
            *requant = serde_json::to_value(changed)?;
            let corrupted = serde_json::from_value(encoded)?;
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::verify::<_, Blake3Transcript, _>(&verifier, corrupted, io)
            }));
            assert!(
                !matches!(rejected, Ok(Ok(()))),
                "accepted modified committed lookup evaluation"
            );
            Ok(())
        };
        // Generic Requant contexts obtain this table from the process-wide
        // BIT_LEN. Match it so the same regression also runs in default CI;
        // ZKML_BIT_LEN=14 exercises the exact frozen full-model table.
        let table = Table::new_requantise();
        let scaling = ScalingFactor::from_absolute_max(1.0, None);
        let requant = Requant {
            output_scaling: scaling,
            table,
            activation_lookup_data: ActivationLookupData::new(15, 32768, 16, 0, table, false, 1),
        };
        let shape: Shape = vec![2, 4].into();
        let mut model = Model::new_from_input_shapes(vec![shape.clone()]);
        model.add_consecutive_layer(Layer::Requant(requant), None)?;
        model.automatic_output_labelling()?;
        prove(
            model,
            vec![Tensor::new(
                shape.clone(),
                vec![-50000, -40960, -32769, -1, 0, 32767, 40960, 50000],
            )?],
        )?;

        // Exact frozen real-model N31 table and fixed-point parameters: a
        // 16-bit SiLU input table whose complete output lies in signed 12 bits.
        let silu_table =
            Table::new_silu(f32::from_bits(1158775787), f32::from_bits(1125219172), 16);
        for x in -32768..=32767 {
            assert!((-2048..=2047).contains(&silu_table.lookup(x)?));
        }
        let lookup = ActivationLookupData::new(28, 28165, 37, 0, silu_table, true, 1);
        let activation = Activation::GLU(ActivationLayer::Silu(Some(lookup), PhantomData));
        let mut model = Model::new_from_input_shapes(vec![shape.clone(); 2]);
        let id = model.add_consecutive_layer(Layer::Activation(activation), None)?;
        // Exact frozen first FFN GLU requantization parameters (N720).
        model.add_consecutive_layer(
            Layer::Requant(Requant {
                output_scaling: scaling,
                table,
                activation_lookup_data: ActivationLookupData::new(
                    36, 22116, 48, 0, table, false, 1,
                ),
            }),
            Some(id),
        )?;
        model.automatic_output_labelling()?;
        let gate = Tensor::new(
            shape.clone(),
            vec![
                -5 * (1i64 << 36),
                -(1i64 << 36),
                -1,
                0,
                1,
                1 << 25,
                1 << 34,
                (1 << 36) - 1,
            ],
        )?;
        let up = Tensor::new(
            shape,
            vec![
                1 << 35,
                -(1 << 35),
                -7,
                8,
                -9,
                -(1 << 35),
                1 << 35,
                -(1 << 35),
            ],
        )?;
        let witness = lookup.get_lookup_witness(gate.clone())?;
        let witness_output =
            crate::lookup::operation::lookup_test_utils::compute_unpadded_witness_output(
                &lookup,
                gate.shape(),
                &witness,
                &silu_table,
            )?;
        let numerical_output = lookup.evaluate(gate.as_wrapped())?.get_data();
        assert_eq!(witness_output.data(), &numerical_output);
        prove(model, vec![gate, up])
    }

    impl ChunkingInfo {
        /// This function recombines the decomposed values back into their shifted form for testing purposes.
        /// So if the original value was `v`, this SHOULD return `v >> right_shift` for each input.
        fn test_recombination_after_shift(&self, chunked_input: &ChunkedInput) -> Vec<Element> {
            let len = chunked_input.value_chunks[0].len();
            let combined_values = (0..len)
                .map(|i| {
                    chunked_input
                        .value_chunks
                        .iter()
                        .enumerate()
                        .fold(0, |acc, (j, chunk_vec)| {
                            let initial_val = match self.table.table_sign() {
                                TableSign::Mixed => chunk_vec[i] + self.value_chunk_offset,
                                TableSign::Positive => chunk_vec[i],
                                TableSign::Negative => -chunk_vec[i],
                            };
                            acc + (initial_val << (j * self.value_chunk_size))
                        })
                })
                .collect::<Vec<Element>>();

            combined_values
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let offset_val = chunked_input.zeroing_chunks.iter().enumerate().fold(
                        val,
                        |acc, (j, chunk_vec)| {
                            let shift_amount =
                                self.value_chunk_size * self.number_of_value_chunks + j * 16;
                            if j != self.number_of_zeroing_chunks - 1 || !self.is_signed {
                                acc + (chunk_vec[i] << shift_amount)
                            } else {
                                acc + ((chunk_vec[i] + self.top_zeroing_chunk_offset)
                                    << shift_amount)
                            }
                        },
                    );
                    match self.table.table_sign() {
                        TableSign::Mixed => offset_val - (self.offset >> self.right_shift),
                        TableSign::Positive => offset_val,
                        TableSign::Negative => -offset_val,
                    }
                })
                .collect()
        }

        /// This function recombines the decomposed values back into their original form for testing purposes.
        fn test_full_recombination(&self, chunked_input: &ChunkedInput) -> Vec<Element> {
            let len = chunked_input.value_chunks[0].len();
            let combined_values = (0..len)
                .map(|i| {
                    chunked_input
                        .value_chunks
                        .iter()
                        .enumerate()
                        .fold(0, |acc, (j, chunk_vec)| {
                            let initial_val = match self.table.table_sign() {
                                TableSign::Mixed => chunk_vec[i] + self.value_chunk_offset,
                                TableSign::Positive => chunk_vec[i],
                                TableSign::Negative => -chunk_vec[i],
                            };
                            acc + (initial_val << (j * self.value_chunk_size))
                        })
                })
                .collect::<Vec<Element>>();

            combined_values
                .iter()
                .enumerate()
                .map(|(i, &val)| {
                    let offset_val = chunked_input.shifted_chunks.iter().enumerate().fold(
                        0,
                        |acc, (j, chunk_vec)| {
                            let shift_amount = j * 16;
                            if j != self.number_of_shifted_chunks - 1
                                || self.right_shift.is_multiple_of(16)
                            {
                                acc + (chunk_vec[i] << shift_amount)
                            } else {
                                acc + ((chunk_vec[i] / self.shifted_chunk_multiplier)
                                    << shift_amount)
                            }
                        },
                    );
                    let initial_val = offset_val + (val << self.right_shift);
                    let full_offset_val = chunked_input.zeroing_chunks.iter().enumerate().fold(
                        initial_val,
                        |acc, (j, chunk_vec)| {
                            let shift_amount = self.value_chunk_size * self.number_of_value_chunks
                                + self.right_shift
                                + j * 16;
                            if j != self.number_of_zeroing_chunks - 1 || !self.is_signed {
                                acc + (chunk_vec[i] << shift_amount)
                            } else {
                                acc + ((chunk_vec[i] + self.top_zeroing_chunk_offset)
                                    << shift_amount)
                            }
                        },
                    );
                    match self.table.table_sign() {
                        TableSign::Mixed => full_offset_val - self.offset,
                        TableSign::Positive => full_offset_val,
                        TableSign::Negative => -full_offset_val,
                    }
                })
                .collect()
        }
    }

    #[derive(Clone, Debug)]
    struct Input {
        right_shift: usize,
        value_chunk_size: usize,
        is_signed: TableSign,
        max_bit_size: usize,
        inputs: Vec<Element>,
    }

    fn decomposer_input() -> impl Strategy<Value = Input> {
        let is_signed_strategy = prop_oneof![
            Just(TableSign::Positive),
            Just(TableSign::Mixed),
            Just(TableSign::Negative)
        ];
        (10..28usize, 8..16usize, is_signed_strategy).prop_flat_map(
            |(right_shift, value_chunk_size, is_signed)| {
                let max_bit_size_range = right_shift + value_chunk_size + 16..=62;

                max_bit_size_range.prop_flat_map(move |max_bit_size| {
                    let min_val = match is_signed {
                        TableSign::Mixed => -(1i64 << (max_bit_size - 1)),
                        TableSign::Positive => 0,
                        TableSign::Negative => -(1i64 << max_bit_size) + 1,
                    };
                    let max_val = match is_signed {
                        TableSign::Mixed => (1i64 << (max_bit_size - 1)) - 1,
                        TableSign::Positive => (1i64 << max_bit_size) - 1,
                        TableSign::Negative => 0,
                    };

                    prop::collection::vec(min_val..=max_val, 1..20).prop_map(move |inputs| Input {
                        right_shift,
                        value_chunk_size,
                        is_signed,
                        max_bit_size,
                        inputs,
                    })
                })
            },
        )
    }

    type Decomposed<F> = (Vec<Vec<F>>, Vec<Vec<F>>, Vec<Vec<F>>);
    impl ChunkedInput {
        fn to_field<F: PrimeField>(&self) -> Decomposed<F> {
            let shifted = self
                .shifted_chunks
                .iter()
                .map(|chunk| chunk.iter().map(|&v| v.to_field()).collect::<Vec<F>>())
                .collect::<Vec<Vec<F>>>();
            let value = self
                .value_chunks
                .iter()
                .map(|chunk| chunk.iter().map(|&v| v.to_field()).collect::<Vec<F>>())
                .collect::<Vec<Vec<F>>>();
            let zero = self
                .zeroing_chunks
                .iter()
                .map(|chunk| chunk.iter().map(|&v| v.to_field()).collect::<Vec<F>>())
                .collect::<Vec<Vec<F>>>();
            (shifted, value, zero)
        }
    }

    proptest! {
        #[test]
        fn proptest_value_decomposer(inp in decomposer_input()) {
            let Input { right_shift, value_chunk_size, is_signed, max_bit_size, inputs } = inp.clone();
            let table = Table::new_test_table(value_chunk_size, is_signed);

            let chunking_info = ChunkingInfo::new(right_shift, &table, max_bit_size, 1).unwrap();

            let chunked_input = chunking_info.decompose_input(inputs.clone());

            let recombined_values = chunking_info.test_full_recombination(&chunked_input);
            let recombined_shifted_values = chunking_info.test_recombination_after_shift(&chunked_input);

            let shifted_inputs = inputs.iter().map(|&v| {if !matches!(is_signed, TableSign::Negative) { v >> right_shift } else { let neg = -v; let shifted = neg >> right_shift; -shifted}}).collect::<Vec<Element>>();
            prop_assert_eq!(recombined_values, inputs, "Value decomposition and recombination failed for input");
            prop_assert_eq!(recombined_shifted_values, shifted_inputs, "Shifted value decomposition and recombination failed for input");
        }

        #[test]
        fn proptest_input_recombiner(inp in decomposer_input()) {
            let Input { right_shift, value_chunk_size, is_signed, max_bit_size, inputs } = inp.clone();

            let table = Table::new_test_table(value_chunk_size, is_signed);
            let chunking_info = ChunkingInfo::new(right_shift, &table, max_bit_size, 1).unwrap();

            let chunked_input = chunking_info.decompose_input(inputs.clone());
            let (shifted_field, value_field, zero_field) = chunked_input.to_field::<F>();
            let total_length = value_field[0].len();
            let recombined_field_values = (0..total_length).map(|i| {
                let val = value_field.iter().map(|chunk| chunk[i]).collect::<Vec<F>>();
                let shifted_claims = shifted_field.iter().map(|chunk| chunk[i]).collect::<Vec<F>>();
                let zeroing_claims = zero_field.iter().map(|chunk| chunk[i]).collect::<Vec<F>>();
                chunking_info.combine_input_claims::<F>(&shifted_claims, &val, &zeroing_claims)
            }).collect::<Vec<F>>();
            let recombined_values = recombined_field_values.iter().map(|v| v.to_element()).collect::<Vec<Element>>();



            prop_assert_eq!(recombined_values, inputs, "Value decomposition and recombination failed for input");

        }
    }
}
