//! Generic lookup data required for activation layers.
use dp_crypto::util::ceil_log2;

use super::*;

use crate::{
    layers::requant::FIXED_POINT_SCALE,
    lookup::operation::{
        LookupOp, LookupOpWitness, decomposer::ChunkingInfo, variant::LookupVariant,
    },
};

#[derive(Clone, Debug, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd)]
pub struct ActivationLookupData {
    /// After multiplying by `self.fixed_point_multiplier` the value need to be shifted by this plus 25.
    pub right_shift: usize,
    /// The normalised scaling factor represented as a fixed point multiplier (it should have 24 fractional bits)
    pub fixed_point_multiplier: Element,
    /// This field represents how many bits the max absolute value can be
    pub(crate) intermediate_bit_size: usize,
    pub(crate) padding_value: Element,
    pub(crate) table: Table,
    pub(crate) is_glu: bool,
    pub(crate) value_chunks: usize,
}

impl ActivationLookupData {
    pub fn new(
        right_shift: usize,
        fixed_point_multiplier: Element,
        intermediate_bit_size: usize,
        padding_value: Element,
        table: Table,
        is_glu: bool,
        value_chunks: usize,
    ) -> Self {
        Self {
            right_shift,
            fixed_point_multiplier,
            intermediate_bit_size,
            padding_value,
            table,
            is_glu,
            value_chunks,
        }
    }

    pub fn new_from_scalings(
        input_scaling: ScalingFactor,
        output_scaling: ScalingFactor,
        table_input_scaling: ScalingFactor,
        table_output_scaling: ScalingFactor,
        operation: &ActivationLayer<f32>,
    ) -> Result<Self> {
        let (multiplier, table, padding_value): (f32, Table, Element) = match operation {
            ActivationLayer::Relu(..) => (
                input_scaling.scale() / output_scaling.scale(),
                Table::new_relu(),
                0,
            ),
            ActivationLayer::Gelu(..) => {
                let table = Table::new_gelu(
                    table_input_scaling.scale().recip(),
                    table_output_scaling.scale().recip(),
                    table_input_scaling.bit_size() + 1,
                );
                (
                    input_scaling.scale() / table_input_scaling.scale(),
                    table,
                    0,
                )
            }
            ActivationLayer::Silu(..) => {
                let table = Table::new_silu(
                    table_input_scaling.scale().recip(),
                    table_output_scaling.scale().recip(),
                    table_input_scaling.bit_size() + 1,
                );
                (
                    input_scaling.scale() / table_input_scaling.scale(),
                    table,
                    0,
                )
            }
            ActivationLayer::Sigmoid(..)
            | ActivationLayer::Softplus(..)
            | ActivationLayer::ExpNegative(..) => {
                use crate::lookup::table::TableType;
                let operation = match operation {
                    ActivationLayer::Sigmoid(..) => TableType::Sigmoid,
                    ActivationLayer::Softplus(..) => TableType::Softplus,
                    ActivationLayer::ExpNegative(..) => TableType::ExpNegative,
                    _ => unreachable!(),
                };
                let table = Table::new(
                    operation,
                    table_input_scaling.scale().recip().to_bits(),
                    table_output_scaling.scale().recip().to_bits(),
                    table_input_scaling.bit_size() + 1,
                );
                // Lookup padding is an INPUT value, before fixed-point rescaling.
                // Nonzero f(0) is accounted for by the public output-claim mask.
                (
                    input_scaling.scale() / table_input_scaling.scale(),
                    table,
                    0,
                )
            }
        };

        let intermediate_bit_size = input_scaling.bit_size() + 1;

        let log_m = multiplier.log2();
        // This is the right shift
        let int_part = log_m.trunc() as Element;
        // This is used to calculate the fixed point multiplier
        let float_part = log_m.fract();

        let epsilon = 2.0f32.powf(float_part);

        let fp_scale = FIXED_POINT_SCALE;
        let fixed_point_multiplier =
            (epsilon * (1u64 << FIXED_POINT_SCALE) as f32).round_ties_even() as Element;

        // Assertion to check that we can perform requantisation, we need intermediate_bit_size + fp_scale <= 63
        ensure!(
            intermediate_bit_size + fp_scale <= 63,
            "intermediate bit size: {intermediate_bit_size}, fp scale: {fp_scale}, int part: {int_part}",
        );

        let right_shift = (int_part - fp_scale as Element).unsigned_abs() as usize;

        Ok(ActivationLookupData::new(
            right_shift,
            fixed_point_multiplier,
            intermediate_bit_size,
            padding_value,
            table,
            false,
            1,
        ))
    }

    pub fn set_glu(&mut self, is_glu: bool) {
        self.is_glu = is_glu;
    }

    /// Link the zero-padded layer output to lookup witnesses whose padded slots
    /// contain f(0). The correction is determined by the public shape and table,
    /// so it introduces no unchecked witness. GLU already multiplies these slots
    /// by its zero-padded second input and therefore needs no correction.
    pub(crate) fn lookup_output_claim<F: PrimeField>(
        &self,
        claim: &Claim<F>,
        unpadded_shape: &Shape,
    ) -> Result<Claim<F>> {
        use crate::{lookup::operation::compute_lookup_padding_evaluation, quantization::ToField};
        if self.is_glu {
            return Ok(claim.clone());
        }
        let zero_output = self.table.lookup(0)?;
        if zero_output == 0 {
            return Ok(claim.clone());
        }
        let padded_shape = unpadded_shape.next_power_of_two();
        let dim_points = padded_shape.split_point(claim.point())?;
        let padding =
            compute_lookup_padding_evaluation(zero_output.to_field(), unpadded_shape, &dim_points)?;
        Ok(Claim::new(
            claim.point().to_vec(),
            claim.evaluation() + padding,
        ))
    }

    pub fn evaluate(&self, input: WrappedTensor<Element>) -> Result<WrappedTensor<Element>> {
        let shape = Shape::from(input.shape());
        let zero_padding = self.table.lookup(0)? != 0 && input.shape() != *input.unpadded_shape();
        let input = if zero_padding {
            input.reduce_to_unpadded_shape()?
        } else {
            input
        };
        let result = if self.value_chunks == 1 {
            let table = &self.table;
            self.apply(input, table)
        } else {
            let table = Table::new(
                self.table.operation(),
                self.table.input_scale_factor().to_bits(),
                self.table.output_scale_factor().to_bits(),
                self.table.table_bit_size() * self.value_chunks,
            );
            self.apply(input, &table)
        }?;
        if zero_padding {
            let mut tensor = Tensor::try_from(result)?;
            tensor.pad_to_shape(shape)?;
            WrappedTensor::try_from(tensor)
        } else {
            Ok(result)
        }
    }

    pub fn get_lookup_witness(&self, input: Tensor<Element>) -> Result<LookupOpWitness> {
        let table = &self.table;
        self.generate_witness(input, table)
    }
}

impl LookupOp for ActivationLookupData {
    fn intermediate_bit_size(&self) -> usize {
        self.intermediate_bit_size
    }

    fn right_shift(&self) -> usize {
        self.right_shift
    }

    fn variant(&self) -> LookupVariant {
        if self.is_glu {
            LookupVariant::GLU
        } else {
            LookupVariant::Standard
        }
    }

    fn chunking_info(&self, table: &Table) -> Result<ChunkingInfo> {
        let fpm_bits = ceil_log2(self.fixed_point_multiplier.unsigned_abs() as usize);
        let max_bits = if fpm_bits > FIXED_POINT_SCALE {
            self.intermediate_bit_size + fpm_bits
        } else {
            self.intermediate_bit_size + FIXED_POINT_SCALE
        };
        ChunkingInfo::new(self.right_shift(), table, max_bits, self.value_chunks)
    }

    fn rounding_constant(&self) -> Element {
        1 << (self.right_shift - 1)
    }

    fn fixed_point_multiplier(&self) -> Element {
        self.fixed_point_multiplier
    }

    fn is_signed(&self) -> bool {
        self.table.is_signed()
    }

    fn padding_value(&self) -> Element {
        self.padding_value
    }
}
