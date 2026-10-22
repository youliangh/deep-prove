//! Module providing definitions and implementations for lookup tables used in Deep Prove.

use crate::{
    Element, Shape, Tensor, lookup::context::COLUMN_SEPARATOR, quantization::ToField,
    tensor::WrappedTensor, to_field,
};
use ark_ff::PrimeField;
use dp_crypto::{arkyper::transcript::Transcript, poly::dense::DensePolynomial};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fmt::Display;

use anyhow::{Result, anyhow};

mod tabletype;
pub use tabletype::*;

/// The bit size for the zero check table.
pub const ZERO_CHECK_TABLE_BIT_SIZE: usize = 16;

/// The bit size for the shift check table.
pub const SHIFT_CHECK_TABLE_BIT_SIZE: usize = 16;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash, Copy)]
/// Struct representing a lookup table with its properties.
pub struct Table {
    /// The operation being performed by the table.
    operation: TableType,
    /// The input scale factor stored as the bit representation of a [`f32`].
    input_scale_factor: u32,
    /// The output scale factor stored as the bit representation of a [`f32`].
    output_scale_factor: u32,
    /// The bit size of the table.
    table_bit_size: usize,
}

impl PartialOrd for Table {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Table {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match Ord::cmp(&other.table_bit_size(), &self.table_bit_size()) {
            std::cmp::Ordering::Equal => Ord::cmp(
                &(
                    self.operation,
                    self.input_scale_factor,
                    self.output_scale_factor,
                ),
                &(
                    other.operation,
                    other.input_scale_factor,
                    other.output_scale_factor,
                ),
            ),
            order => order,
        }
    }
}

impl Table {
    /// Creates a new [`Table`] instance.
    pub fn new(
        operation: TableType,
        input_scale_factor: u32,
        output_scale_factor: u32,
        table_bit_size: usize,
    ) -> Self {
        Self {
            operation,
            input_scale_factor,
            output_scale_factor,
            table_bit_size,
        }
    }

    /// Returns the name of the table.
    pub fn name(&self) -> String {
        format!(
            "{}_{}_bits_{}_input_sf_{}_output_sf",
            self.operation,
            self.table_bit_size,
            self.input_scale_factor(),
            self.output_scale_factor()
        )
    }

    /// Returns the operation type of the table.
    pub fn operation(&self) -> TableType {
        self.operation
    }

    /// Returns the input scale factor as a [`f32`].
    pub fn input_scale_factor(&self) -> f32 {
        f32::from_bits(self.input_scale_factor)
    }

    /// Returns the output scale factor as a [`f32`].
    pub fn output_scale_factor(&self) -> f32 {
        f32::from_bits(self.output_scale_factor)
    }

    /// Returns the number of columns in the table.
    pub fn num_columns(&self) -> usize {
        self.operation.num_columns()
    }

    /// Returns the bit size of the table.
    pub fn table_bit_size(&self) -> usize {
        self.table_bit_size
    }

    /// Returns whether the table input is signed.
    pub fn is_signed(&self) -> bool {
        self.operation.is_signed()
    }

    /// Returns the [`TableSign`].
    pub fn table_sign(&self) -> TableSign {
        self.operation.input_sign()
    }

    /// Returns the maximum input value the table can handle.
    pub fn max_input_value(&self) -> Element {
        match self.operation.input_sign() {
            TableSign::Positive => (1 << self.table_bit_size()) - 1,
            TableSign::Mixed => (1 << (self.table_bit_size() - 1)) - 1,
            TableSign::Negative => 0,
        }
    }

    /// Returns the minimum input value the table can handle.
    pub fn min_input_value(&self) -> Element {
        match self.operation.input_sign() {
            TableSign::Positive => 0,
            TableSign::Mixed => -1i64 << (self.table_bit_size() - 1),
            TableSign::Negative => (-1i64 << self.table_bit_size()) + 1,
        }
    }

    /// Returns the maximum output value the table can produce.
    pub fn max_output_value(&self) -> Element {
        match self.operation {
            // Unwrap is safe here because all table operations should be able to handle their max input value
            TableType::Exp
            | TableType::ExpNegative
            | TableType::ReLU
            | TableType::GeLU
            | TableType::SiLU
            | TableType::Sigmoid
            | TableType::Softplus
            | TableType::Requantise
            | TableType::ShiftCheck
            | TableType::SignedZeroCheck => self
                .operation
                .apply_op(
                    self.max_input_value(),
                    self.input_scale_factor(),
                    self.output_scale_factor(),
                    self.min_input_value(),
                    self.max_input_value(),
                )
                .unwrap(),
            // The zero check table obtains its maximum when the input is 0
            TableType::ZeroCheck => 1,
        }
    }

    /// Returns the minimum output value the table can produce.
    pub fn min_output_value(&self) -> Element {
        match self.operation {
            // Unwrap is safe here because all table operations should be able to handle their minimum input value
            TableType::Exp
            | TableType::ExpNegative
            | TableType::ReLU
            | TableType::GeLU
            | TableType::SiLU
            | TableType::Sigmoid
            | TableType::Softplus
            | TableType::Requantise
            | TableType::ShiftCheck
            | TableType::SignedZeroCheck => self
                .operation
                .apply_op(
                    self.min_input_value(),
                    self.input_scale_factor(),
                    self.output_scale_factor(),
                    self.min_input_value(),
                    self.max_input_value(),
                )
                .unwrap(),
            // The zero check table obtains its minimum when the input is non-zero
            TableType::ZeroCheck => 0,
        }
    }

    /// Looks up a value in the table given an input.
    pub fn lookup(&self, input: Element) -> Result<Element> {
        self.operation.apply_op(
            input,
            self.input_scale_factor(),
            self.output_scale_factor(),
            self.min_input_value(),
            self.max_input_value(),
        )
    }

    /// Performs the Lookup on a [`WrappedTensor`] input.
    pub fn lookup_tensor(&self, input: WrappedTensor<Element>) -> Result<WrappedTensor<Element>> {
        let shape: Shape = input.shape().into();
        let unpadded_shape: Shape = input.unpadded_shape().into();
        let data = input
            .get_data()
            .into_par_iter()
            .with_min_len(8192)
            .map(|x| self.lookup(x))
            .collect::<Result<Vec<Element>>>()?;

        let tensor = Tensor::<Element>::new_with_unpadded_shape(shape, unpadded_shape, data)?;
        WrappedTensor::try_from(tensor)
    }

    /// Returns the table columns as elements of the specified extension field's basefield.
    pub fn get_table_columns<F: PrimeField>(&self) -> Vec<Vec<F>> {
        match self.num_columns() {
            1 => {
                vec![to_field::<_, F, _>(
                    self.min_input_value()..=self.max_input_value(),
                )]
            }
            2 => {
                let inputs = self.min_input_value()..=self.max_input_value();
                let outputs = inputs.clone().map(|inp| self.lookup(inp).unwrap());
                vec![to_field::<_, F, _>(inputs), to_field::<_, F, _>(outputs)]
            }
            _ => unreachable!("No table has more than 2 columns"),
        }
    }

    /// Returns the committed table columns
    pub fn committed_columns<'a, F: PrimeField>(&self) -> Option<DensePolynomial<'a, F>> {
        if self.operation.commit_output_column() {
            let inputs = self.min_input_value()..=self.max_input_value();
            Some(DensePolynomial::new(
                inputs
                    .clone()
                    .map(|inp| F::from(self.lookup(inp).unwrap()))
                    .collect(),
            ))
        } else {
            None
        }
    }

    /// Returns the merged table columns as [`Element`]s for witness generation purposes.
    pub fn get_merged_table_columns(&self) -> Vec<Element> {
        match self.num_columns() {
            1 => (self.min_input_value()..=self.max_input_value()).collect(),
            2 => (self.min_input_value()..=self.max_input_value())
                .map(|inp| {
                    let out = self.lookup(inp).unwrap();
                    inp + COLUMN_SEPARATOR * out
                })
                .collect(),
            _ => unreachable!("No table has more than 2 columns"),
        }
    }

    /// Flag indicating whether the table has committed columns.
    pub fn commit_output_column(&self) -> bool {
        self.operation.commit_output_column()
    }

    /// Called by the verifier to evaluate _some_ columns itself. If the verifier can't verify the table
    /// efficiently, then it is done by regular PCS.
    pub fn evaluate_table_columns<F: PrimeField>(&self, point: &[F]) -> Result<Vec<F>> {
        if point.len() != self.table_bit_size() {
            return Err(anyhow!(
                "Point was not the correct size to produce a table evaluation, point size: {}, expected: {}",
                point.len(),
                self.table_bit_size()
            ));
        }
        match self.operation {
            TableType::ReLU => {
                let min_input_field: F = self.min_input_value().to_field();

                let first_column = point
                    .iter()
                    .enumerate()
                    .fold(F::ZERO, |acc, (index, p)| acc + *p * F::from(1u64 << index))
                    + min_input_field;

                let second_column = point
                    .iter()
                    .enumerate()
                    .take(point.len() - 1)
                    .fold(F::ZERO, |acc, (index, p)| {
                        acc + *p * F::from(1u64 << index) * point[point.len() - 1]
                    });
                Ok(vec![first_column, second_column])
            }
            _ => {
                let min_input_field: F = self.min_input_value().to_field();

                Ok(vec![
                    point
                        .iter()
                        .enumerate()
                        .fold(F::ZERO, |acc, (index, p)| acc + *p * F::from(1u64 << index))
                        + min_input_field,
                ])
            }
        }
    }

    pub fn generate_challenge<F: PrimeField, T: Transcript>(&self, transcript: &mut T) -> F {
        self.operation.generate_challenge(transcript)
    }
}

// Constructors for various table types
impl Table {
    /// Creates a new [`Table`] instance for a ReLU operation.
    pub fn new_relu() -> Self {
        Self::new(
            TableType::ReLU,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            *crate::quantization::BIT_LEN,
        )
    }

    /// Creates a new [`Table`] instance for a Exp operation.
    pub fn new_exp(input_sf: f32, output_sf: f32, table_bit_size: usize) -> Self {
        Self::new(
            TableType::Exp,
            input_sf.to_bits(),
            output_sf.to_bits(),
            table_bit_size,
        )
    }

    /// Creates a new [`Table`] instance for a GeLU operation.
    pub fn new_gelu(input_sf: f32, output_sf: f32, table_bit_size: usize) -> Self {
        Self::new(
            TableType::GeLU,
            input_sf.to_bits(),
            output_sf.to_bits(),
            table_bit_size,
        )
    }

    /// Creates a new [`Table`] instance for a SiLU operation.
    pub fn new_silu(input_sf: f32, output_sf: f32, table_bit_size: usize) -> Self {
        Self::new(
            TableType::SiLU,
            input_sf.to_bits(),
            output_sf.to_bits(),
            table_bit_size,
        )
    }

    /// Creates a new [`Table`] instance for a Requantisation operation.
    pub fn new_requantise() -> Self {
        Self::new(
            TableType::Requantise,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            *crate::quantization::BIT_LEN,
        )
    }

    /// Creates a new [`Table`] instance for normalisation checks.
    pub fn new_normalisation(bit_size: usize) -> Self {
        Self::new(
            TableType::Requantise,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            bit_size,
        )
    }

    /// Creates a new [`Table`] instance for a ShiftCheck operation.
    pub fn new_shift_check() -> Self {
        Self::new(
            TableType::ShiftCheck,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            SHIFT_CHECK_TABLE_BIT_SIZE,
        )
    }

    /// Creates a new [`Table`] instance for a SignedZeroCheck operation.
    pub fn new_signed_zero_check() -> Self {
        Self::new(
            TableType::SignedZeroCheck,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            ZERO_CHECK_TABLE_BIT_SIZE,
        )
    }

    /// Creates a new [`Table`] instance for a ZeroCheck operation.
    pub fn new_zero_check() -> Self {
        Self::new(
            TableType::ZeroCheck,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            ZERO_CHECK_TABLE_BIT_SIZE,
        )
    }
}

#[cfg(test)]
impl Table {
    pub(crate) fn new_test_table(value_chunk_size: usize, table_sign: TableSign) -> Self {
        let operation = match table_sign {
            TableSign::Positive => TableType::ZeroCheck,
            TableSign::Mixed => TableType::ReLU,
            TableSign::Negative => TableType::Exp,
        };
        Self::new(
            operation,
            1.0f32.to_bits(),
            1.0f32.to_bits(),
            value_chunk_size,
        )
    }
}
