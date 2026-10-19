//! Module defining different variants of the lookup operation, specifically the how to construct the sumcheck equation for each.

use crate::{
    NextPowerOfTwo,
    lookup::table::{SHIFT_CHECK_TABLE_BIT_SIZE, TableSign},
};

use super::*;

use dp_crypto::{Expression, arkyper::transcript::Transcript, util::ceil_log2};
use itertools::izip;
use serde::{Deserialize, Serialize};

pub mod proving;
pub(crate) mod rms;
pub mod verifying;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
/// Enum representing the different variants of the lookup operation. This is used to determine how the sumcheck linking the lookup to the layer output is constructed and what checks are performed within it.
pub enum LookupVariant {
    /// Standard Lookup Operation, no extra checks have to be enforced.
    Standard,
    /// Lookup Operation as part of a GLU layer, involves an extra term in linking the layer output with the lookup.
    GLU,
    /// Lookup Operation as part of a Softmax layer, involves an extra term to enforce the normalisation is correct.
    Softmax {
        /// The normalisation value that the sum along the specified dimension of the lookup outputs should equal to.
        normalised_sum_value: Element,
        /// The error bound allowed on the normalisation.
        error_bound: Element,
    },
    /// Lookup Operation as part of a normalisation layer. These layers always have a normalisation term on the magntiude of one of the dimensions
    /// and can also optionally enforce normalisation of the sum along the same dimension (aka enforce the mean is zero).
    Normalisation {
        /// The quantised value representing the square of the normalised magnitude.
        normalised_magnitude_value: Element,
        /// The error bound allowed on the normalised magnitude.
        magnitude_error_bound: Element,
        /// Optional normalised sum and sum error bound if the mean is also to be enforced as zero.
        normalised_sum_value: Option<(Element, Element)>,
        /// Flag used to indicate whether the normalisation layer also has a weight to multiply element-wise post normalisation.
        has_weight: bool,
    },
}

impl LookupVariant {
    /// Returns true if this variant requires normalisation checks.
    pub fn requires_normalisation(&self) -> bool {
        matches!(
            self,
            LookupVariant::Softmax { .. } | LookupVariant::Normalisation { .. }
        )
    }

    /// Builds the full sumcheck expression for the lookup operation by summing the expressions for each chunk.
    pub fn build_full_sumcheck_expression<F: PrimeField>(
        &self,
        total_chunks: usize,
        final_dim_size: usize,
        chunking_info: &ChunkingInfo,
    ) -> Expression<F> {
        // First we work out how many witness polynomials there will be total
        (0..total_chunks).fold(Expression::ZERO, |acc, chunk_number| {
            acc + self.sumcheck_expression_for_chunk(
                chunk_number,
                total_chunks,
                final_dim_size,
                chunking_info,
            )
        })
    }

    /// Internal method that builds the expressions needed for the output part of the lookup operation.
    /// This is the same for all variants.
    pub(crate) fn build_lookup_output_expressions<F: PrimeField>(
        &self,
        current_chunk: usize,
        chunking_info: &ChunkingInfo,
    ) -> LookupExpressions<F> {
        let table = chunking_info.table();
        let number_zero_chunks = chunking_info.number_of_zeroing_chunks();

        let total_witnesses_per_chunk = chunking_info.number_of_value_chunks() + number_zero_chunks;
        let offset = (current_chunk * total_witnesses_per_chunk) as u16;

        let ValueExpression {
            value: value_expr,
            initial_sum,
            witness_offset,
            sum_challenge_offset,
        } = ValueExpression::<Expression<F>>::new(chunking_info, offset);
        // How we handle zeroing chunks changes depending on the whether the operation is signed or not.
        // If its signed then the most significant zeroing chunk has value +/- 1 if the most significant input chunk was non-zero,
        // if its not a signed operation then all zeroing chunks are 1 if and only if the input chunk was zero.
        if table.is_signed() {
            let (mut prod, mut sum) = (0..number_zero_chunks.saturating_sub(1)).fold(
                (Expression::ONE, initial_sum),
                |(prod_acc, sum_acc), idx| {
                    let chunk_expr = Expression::<F>::WitIn(witness_offset + idx as u16);
                    // The sum challenge is always the first challenge
                    let sum_challenge =
                        Expression::<F>::Challenge(0, sum_challenge_offset + idx, F::ONE, F::ZERO);
                    (
                        prod_acc * chunk_expr.clone(),
                        sum_acc + chunk_expr * sum_challenge,
                    )
                },
            );

            // If prod == 1 then the output is the value expression, else its the corresponding clamping value
            // The clamping part is defined as clamping_max * (x * (x + 1)/2 + (1 - x^2)*(1-prod)) + clamping_min * (x * (x - 1)/2)
            let two_field = F::from(2);
            let (clamping_min, clamping_max): (F, F) = if chunking_info.number_of_value_chunks()
                == 1
            {
                (
                    table.min_output_value().to_field(),
                    table.max_output_value().to_field(),
                )
            } else {
                let full_bit_size = chunking_info.number_of_value_chunks() * table.table_bit_size();
                let min: Element = -1 << (full_bit_size - 1);
                let max: Element = (1 << (full_bit_size - 1)) - 1;
                (min.to_field(), max.to_field())
            };

            let (clamping_expression, squared_clamping_expression) = if number_zero_chunks != 0 {
                let last_chunk_expr =
                    Expression::<F>::WitIn(witness_offset - 1 + number_zero_chunks as u16);
                let lower_chunks_expr = prod.clone();
                let one_minus_tc_squared =
                    Expression::<F>::ONE - last_chunk_expr.clone() * last_chunk_expr.clone();
                prod *= one_minus_tc_squared.clone();
                sum += last_chunk_expr.clone()
                    * Expression::<F>::Challenge(
                        0,
                        sum_challenge_offset - 1 + number_zero_chunks,
                        F::ONE,
                        F::ZERO,
                    );

                let max_coeff = Expression::<F>::Constant(clamping_max);
                let min_coeff = Expression::<F>::Constant(clamping_min);
                let two_inv_expr = Expression::<F>::Constant(
                    two_field.inverse().expect("Cannot fail when inverting 2"),
                );

                let clamping_first_part = last_chunk_expr.clone()
                    * (last_chunk_expr.clone() + Expression::ONE)
                    * two_inv_expr.clone()
                    + (one_minus_tc_squared.clone() - lower_chunks_expr * one_minus_tc_squared);
                let clamping_second_part = last_chunk_expr.clone()
                    * (last_chunk_expr - Expression::ONE)
                    * two_inv_expr.clone();

                let clamping_expression = clamping_first_part.clone() * max_coeff.clone()
                    + clamping_second_part.clone() * min_coeff.clone();

                let squared_clamping_expression =
                    if matches!(self, LookupVariant::Normalisation { .. }) {
                        clamping_first_part.clone() * max_coeff.clone() * max_coeff
                            + clamping_second_part.clone() * min_coeff.clone() * min_coeff
                    } else {
                        Expression::ZERO
                    };
                (clamping_expression, squared_clamping_expression)
            } else {
                // No zero chunks means no clamping needed
                (Expression::ZERO, Expression::ZERO)
            };

            LookupExpressions {
                value: value_expr,
                prod_selector: prod,
                clamping_expression,
                squared_clamping_expression,
                sum,
            }
        } else {
            let (prod, sum) = (0..number_zero_chunks).fold(
                (Expression::<F>::ONE, initial_sum),
                |(prod_acc, sum_acc), idx| {
                    let chunk_expr = Expression::<F>::WitIn(witness_offset + idx as u16);
                    // The sum challenge is always the first challenge
                    let sum_challenge =
                        Expression::<F>::Challenge(0, sum_challenge_offset + idx, F::ONE, F::ZERO);
                    (
                        prod_acc * chunk_expr.clone(),
                        sum_acc + chunk_expr * sum_challenge,
                    )
                },
            );

            let clamping_expression = if number_zero_chunks != 0 {
                // The clamping value is based on whether inputs are positive or negative.
                let clamping_value: F = match table.operation().input_sign() {
                    TableSign::Positive => table.max_output_value().to_field(),
                    TableSign::Negative => table.min_output_value().to_field(),
                    TableSign::Mixed => {
                        unreachable!("Already checked that the table doesn't have mixed signs")
                    }
                };
                Expression::<F>::Constant(clamping_value)
            } else {
                // No zero chunks means no clamping needed
                Expression::ONE
            };

            // squared clamping expression is only required in signed case, so set to ONE here
            let squared_clamping_expression = Expression::ZERO;

            LookupExpressions {
                value: value_expr,
                prod_selector: prod,
                clamping_expression,
                squared_clamping_expression,
                sum,
            }
        }
    }

    /// Builds the chunk Sumcheck expression for this variant.
    pub fn sumcheck_expression_for_chunk<F: PrimeField>(
        &self,
        chunk_number: usize,
        total_chunks: usize,
        final_dim_size: usize,
        chunking_info: &ChunkingInfo,
    ) -> Expression<F> {
        // The number of output related witnesses per chunk can be calculated from the chunking info
        let number_zero_chunks = chunking_info.number_of_zeroing_chunks();
        let output_witnesses_per_chunk =
            chunking_info.number_of_value_chunks() + number_zero_chunks;
        let total_output_witnesses = total_chunks * output_witnesses_per_chunk;

        let total_witnesses =
            total_chunks * self.additional_witnesses_per_chunk() + total_output_witnesses;

        // We need to go case by case depending on the variant
        let output_linking_eq = Expression::<F>::WitIn(total_witnesses as u16);
        let lookup_linking_eq = Expression::<F>::WitIn((total_witnesses + 1) as u16);
        let chunk_challenge =
            Expression::<F>::Challenge((1 + chunk_number) as u16, 1, F::ONE, F::ZERO);

        let LookupExpressions {
            value,
            prod_selector,
            clamping_expression,
            squared_clamping_expression,
            sum,
        } = self.build_lookup_output_expressions(chunk_number, chunking_info);

        match self {
            LookupVariant::Standard => {
                // The prod expression evaluates to the lookup output value, or the clamped output value if clamping was required.
                // let prod_expression =
                //     clamping_expression.clone() + prod_selector * (value - clamping_expression);
                let prod_expression = clamping_expression + prod_selector * value;
                chunk_challenge * (output_linking_eq * prod_expression + lookup_linking_eq * sum)
            }
            LookupVariant::GLU => {
                // The prod expression evaluates to the lookup output value, or the clamped output value if clamping was required.
                // let prod_expression =
                //     clamping_expression.clone() + prod_selector * (value - clamping_expression);
                let prod_expression = clamping_expression + prod_selector * value;
                // The GLU witness is multiplied element wise with the prod_expression.
                let glu_chunk_witness =
                    Expression::<F>::WitIn((total_output_witnesses + chunk_number) as u16);

                chunk_challenge
                    * (output_linking_eq * prod_expression * glu_chunk_witness
                        + lookup_linking_eq * sum)
            }
            LookupVariant::Softmax { .. } => {
                // The prod expression evaluates to the lookup output value, or the clamped output value if clamping was required.
                // let prod_expression =
                //     clamping_expression.clone() + prod_selector * (value - clamping_expression);
                let prod_expression = clamping_expression + prod_selector * value;

                let normalisation_eq = Expression::<F>::WitIn((total_witnesses + 2) as u16);

                let final_dim_log = ceil_log2(final_dim_size);
                let pow_two: F = (1i64 << final_dim_log).to_field();
                let sum_norm_challenge =
                    Expression::<F>::Challenge((1 + total_chunks) as u16, 1, pow_two, F::ZERO);
                // In this case we want to link the normalisation check to the lookup output and that is done via the normalisation_eq poly.
                chunk_challenge
                    * (output_linking_eq * prod_expression.clone()
                        + sum_norm_challenge * (normalisation_eq * prod_expression)
                        + lookup_linking_eq * sum)
            }
            LookupVariant::Normalisation {
                normalised_magnitude_value: _,
                magnitude_error_bound: _,
                normalised_sum_value,
                has_weight,
            } => {
                let normalisation_eq = Expression::<F>::WitIn((total_witnesses + 2) as u16);

                let final_dim_log = ceil_log2(final_dim_size);
                let pow_two: F = (1i64 << final_dim_log).to_field();

                // let prod_expression =
                //     clamping_expression.clone() + prod_selector * (value - clamping_expression);
                let prod_expression = clamping_expression + prod_selector.clone() * value.clone();

                // In order to keep te total degree of the sumcheck polynomial low we compute the squared product expression here
                // as the prod_selector is boolean this is fine
                // let prod_squared_expression = squared_clamping_expression.clone()
                //     + prod_selector * (value.clone() * value - squared_clamping_expression);
                let prod_squared_expression =
                    squared_clamping_expression + prod_selector * value.clone() * value;

                let sumsq_norm_challenge =
                    Expression::<F>::Challenge((1 + total_chunks) as u16, 1, pow_two, F::ZERO);

                let output_linking_eq = if *has_weight {
                    output_linking_eq * Expression::<F>::WitIn((total_witnesses + 3) as u16)
                } else {
                    output_linking_eq
                };

                let input_chunk_expression =
                    Expression::<F>::WitIn((total_output_witnesses + chunk_number) as u16);
                let scaling_witness_expression = Expression::<F>::WitIn(
                    (total_output_witnesses + total_chunks + chunk_number) as u16,
                );
                let input_challenge =
                    Expression::<F>::Challenge((2 + total_chunks) as u16, 1, F::ONE, F::ZERO);

                // Now we adat the output part again depending on whether we have sum normalisation or not
                let mut output_part = output_linking_eq * prod_expression.clone();
                if normalised_sum_value.is_some() {
                    let sum_norm_challenge =
                        Expression::<F>::Challenge((1 + total_chunks) as u16, 2, pow_two, F::ZERO);
                    output_part +=
                        sum_norm_challenge * (normalisation_eq.clone() * prod_expression);
                } else {
                    // The multiplier is constant along the last dimension, so
                    // averaging it with normalisation_eq needs no dimension factor.
                    let multiplier_challenge =
                        Expression::<F>::Challenge((1 + total_chunks) as u16, 2, F::ONE, F::ZERO);
                    output_part += multiplier_challenge
                        * normalisation_eq.clone()
                        * scaling_witness_expression.clone();
                };

                chunk_challenge
                    * (output_part
                        + sumsq_norm_challenge * (normalisation_eq * prod_squared_expression)
                        + lookup_linking_eq
                            * (sum
                                + input_challenge
                                    * (input_chunk_expression * scaling_witness_expression)))
            }
        }
    }

    /// Returns the number of additional witnesses per chunk required by this variant.
    pub(crate) fn additional_witnesses_per_chunk(&self) -> usize {
        match self {
            // Standard and Softmax variants have no additional witnesses per chunk
            LookupVariant::Standard | LookupVariant::Softmax { .. } => 0,
            // GLU variant has one additional witness per chunk for the output linking equation
            LookupVariant::GLU => 1,
            // Normalisation variant has two additional witnesses per chunk for the enforcing the input is scaled by the correct value along each dimension
            LookupVariant::Normalisation { .. } => 2,
        }
    }

    /// Method that squeezes all the challenges required during sumcheck proving/verification for this variant.
    pub fn squeeze_sumcheck_challenges<F: PrimeField, T: Transcript>(
        &self,
        transcript: &mut T,
    ) -> Vec<F> {
        match self {
            LookupVariant::Standard | LookupVariant::GLU => vec![],
            LookupVariant::Softmax { .. } => {
                vec![transcript.append_and_sample(b"normalisation")]
            }
            LookupVariant::Normalisation { .. } => {
                vec![
                    transcript.append_and_sample(b"normalisation"),
                    transcript.append_and_sample(b"input"),
                ]
            }
        }
    }
    /// Produce the input claims that should be output by proving/verifying this operation.
    pub fn produce_input_claims<F: PrimeField, L: LookupOp>(
        &self,
        unpadded_input_shape: &Shape,
        lookup_op: &L,
        last_claim_point: &[F],
        logup_point: &[F],
        sumcheck_point: &[F],
        mut input_claim_evals: Vec<F>,
    ) -> Result<Vec<Claim<F>>> {
        match self {
            LookupVariant::Standard | LookupVariant::GLU | LookupVariant::Softmax { .. } => {
                // We only want to subtract the rounding constant from the unpadded portion of the input
                // when the rnak is greater than 2. This is because the rounding constant is added to each padded 2D sub tensor
                // but the total evaluation is for the entire padded tensor, so we subtract the rounding constant from every padded
                // 2D sub tensor that occurs in th eunpadded input but not the padded input.
                let rank = unpadded_input_shape.rank();
                let dims_to_skip = rank.saturating_sub(2);
                let padded_shape = unpadded_input_shape.next_power_of_two();
                let dim_points = padded_shape.split_point(last_claim_point)?;

                let unbroadcast_shape = std::iter::repeat_n(1usize, dims_to_skip)
                    .chain(unpadded_input_shape[dims_to_skip..].iter().copied())
                    .collect::<Vec<usize>>();
                let rounding_constant: F = lookup_op.rounding_constant().to_field();
                let lt_eval = unpadded_input_shape
                    .broadcasting_evaluation(&dim_points, &unbroadcast_shape)?;
                let rounding_to_sub = lt_eval * rounding_constant;
                input_claim_evals[0] -= rounding_to_sub;

                let fpm: F = lookup_op.fixed_point_multiplier().to_field();
                let fpm_inv = fpm
                    .inverse()
                    .expect("Tried to invert 0 as fixed point multiplier");
                input_claim_evals[0] *= fpm_inv;

                let first_point = [logup_point, &last_claim_point[logup_point.len()..]].concat();

                // If the padding value is non-zero we also have to subtract that here
                if lookup_op.padding_value() != 0 {
                    let dim_points = padded_shape.split_point(&first_point)?;
                    let padding_to_sub = compute_lookup_padding_evaluation(
                        lookup_op.padding_value().to_field(),
                        unpadded_input_shape,
                        &dim_points,
                    )?;
                    input_claim_evals[0] -= padding_to_sub;
                }

                let second_point =
                    [sumcheck_point, &last_claim_point[sumcheck_point.len()..]].concat();

                Ok(input_claim_evals
                    .iter()
                    .zip([first_point, second_point])
                    .map(|(&eval, point)| Claim::<F>::new(point, eval))
                    .collect::<Vec<Claim<F>>>())
            }
            LookupVariant::Normalisation { .. } => {
                let input_point =
                    [sumcheck_point, &last_claim_point[sumcheck_point.len()..]].concat();
                Ok(vec![Claim::<F>::new(input_point, input_claim_evals[0])])
            }
        }
    }
}

#[derive(Debug, Clone)]
/// Struct holding the expressions needed for the output part of the lookup operation.
pub(crate) struct LookupExpressions<F: PrimeField> {
    /// The expression representing the output value of the lookup table.
    pub value: Expression<F>,
    /// The selector used to pick the clamping value if clamping is required.
    pub prod_selector: Expression<F>,
    /// The expression that returns the correct clamping value (min or max) based on the sign of the input.
    pub clamping_expression: Expression<F>,
    /// The expression that returns the square of the clamping value, this is only used in the normalisation variant with signed tables.
    pub squared_clamping_expression: Expression<F>,
    /// The expression that is just a random linear combination of the lookup output related witnesses, this is used to link the lookup output to the sumcheck and also to enforce the zeroing of the zero chunks.
    pub sum: Expression<F>,
}

#[derive(Debug, Clone)]
/// Struct holding all the info about the Value part of a lookup operation
pub(crate) struct ValueExpression<T> {
    /// The Expression representing the value used for linking with the output of the layer.
    pub value: T,
    /// The initial [`Expression`] used in the sum part of the sumcheck
    pub initial_sum: T,
    /// The offset for remaining witnesses in this chunk to use in the sumcheck expression construction.
    pub witness_offset: u16,
    /// The offset for the sum challenge powers in this chunk to use in the sumcheck expression construction.
    pub sum_challenge_offset: usize,
}

impl<F: PrimeField> ValueExpression<Expression<F>> {
    fn new(
        chunking_info: &ChunkingInfo,
        initial_witness_offset: u16,
    ) -> ValueExpression<Expression<F>> {
        let number_value_chunks = chunking_info.number_of_value_chunks();

        // With one value chunk we can just return the result now
        if number_value_chunks == 1 {
            ValueExpression {
                value: Expression::<F>::WitIn(initial_witness_offset),
                initial_sum: Expression::<F>::WitIn(initial_witness_offset)
                    * Expression::<F>::Challenge(0, 1, F::ONE, F::ZERO),
                witness_offset: initial_witness_offset + 1,
                sum_challenge_offset: 2,
            }
        } else {
            let table = chunking_info.table();

            let full_value_offset: Element =
                1 << (number_value_chunks * table.table_bit_size() - 1);
            let full_value_offset_field: F = full_value_offset.to_field();
            let (value, sum) = (0..number_value_chunks).fold(
                (
                    Expression::<F>::Constant(-full_value_offset_field),
                    Expression::ZERO,
                ),
                |(value_acc, sum_acc), idx| {
                    let chunk_expr = Expression::<F>::WitIn(initial_witness_offset + idx as u16);
                    let shift_amount =
                        Expression::<F>::from(1u64 << (table.table_bit_size() * idx));
                    let value_offset_expr =
                        Expression::<F>::from(1u64 << (table.table_bit_size() - 1));
                    let value_part = shift_amount * (chunk_expr.clone() + value_offset_expr);
                    let sum_challenge = Expression::<F>::Challenge(0, idx + 1, F::ONE, F::ZERO);
                    (value_acc + value_part, sum_acc + chunk_expr * sum_challenge)
                },
            );

            ValueExpression {
                value,
                initial_sum: sum,
                witness_offset: initial_witness_offset + number_value_chunks as u16,
                sum_challenge_offset: 1 + number_value_chunks,
            }
        }
    }
}

impl<F: PrimeField> ValueExpression<F> {
    /// Method to evaluate the value expression on a given witness assignment and set of challenges, this is used in the prover when we need to compute the value expression for the next chunk based on the witness values for the current chunk.
    pub(crate) fn evaluate(
        values: &[F],
        sum_challenge: F,
        chunking_info: &ChunkingInfo,
    ) -> ValueExpression<F> {
        let number_value_chunks = chunking_info.number_of_value_chunks();

        // With one value chunk we can just return the result now
        if number_value_chunks == 1 {
            ValueExpression {
                value: values[0],
                initial_sum: values[0] * sum_challenge,
                witness_offset: 0u16,
                sum_challenge_offset: 2,
            }
        } else {
            let table = chunking_info.table();

            let full_value_offset: Element =
                1 << (number_value_chunks * table.table_bit_size() - 1);
            let full_value_offset_field: F = full_value_offset.to_field();
            let mut challenge_acc = sum_challenge;
            let (value, sum) = values.iter().enumerate().fold(
                (-full_value_offset_field, F::ZERO),
                |(value_acc, sum_acc), (idx, &val)| {
                    let shift_amount = F::from(1u64 << (table.table_bit_size() * idx));
                    let value_offset_expr = F::from(1u64 << (table.table_bit_size() - 1));
                    let value_part = shift_amount * (val + value_offset_expr);
                    let sum_part = sum_acc + val * challenge_acc;
                    challenge_acc *= sum_challenge;
                    (value_acc + value_part, sum_part)
                },
            );

            ValueExpression {
                value,
                initial_sum: sum,
                witness_offset: 0u16,
                sum_challenge_offset: 1 + number_value_chunks,
            }
        }
    }
}
