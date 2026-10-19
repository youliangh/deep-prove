//! Multi-limb, two-sided RMSNorm checks (DeepProve Appendix C).
//!
//! A single offset lookup only supplies one useful inequality. Instead range
//! check both `T + B - ||y||²` and `||y||² - (T - B)`, and bind their sum to
//! `2 B`. Each has at most 48 bits, so their sum cannot wrap in our field.
//! The positive multiplier is range checked separately and linked by sumcheck.

use super::*;

pub(crate) const RMS_MULTIPLIER_LIMBS: usize = 3;
pub(crate) const RMS_MULTIPLIER_BITS: usize = RMS_MULTIPLIER_LIMBS * SHIFT_CHECK_TABLE_BIT_SIZE;

pub(crate) fn residual_limbs(bound: Element) -> Result<usize> {
    ensure!(bound >= 0, "RMSNorm error bound must be nonnegative");
    let twice = (bound as u64)
        .checked_mul(2)
        .ok_or(anyhow!("RMSNorm bound overflow"))?;
    let bits = (u64::BITS - twice.leading_zeros()).max(1) as usize;
    let limbs = bits.div_ceil(SHIFT_CHECK_TABLE_BIT_SIZE);
    ensure!(
        limbs <= 3,
        "RMSNorm residual exceeds the supported 48-bit range"
    );
    Ok(limbs)
}

pub(crate) fn decompose(value: Element, limbs: usize) -> Result<Vec<Element>> {
    ensure!(
        value >= 0 && (value as u64) < (1u64 << (limbs * SHIFT_CHECK_TABLE_BIT_SIZE)),
        "RMSNorm range witness {value} is outside its unsigned limb range"
    );
    let mask = (1i64 << SHIFT_CHECK_TABLE_BIT_SIZE) - 1;
    Ok((0..limbs)
        .map(|i| (value >> (i * SHIFT_CHECK_TABLE_BIT_SIZE)) & mask)
        .collect())
}

pub(crate) fn recompose<F: PrimeField>(limbs: &[F]) -> F {
    limbs.iter().enumerate().fold(F::ZERO, |acc, (i, limb)| {
        acc + *limb * F::from(1u64 << (i * SHIFT_CHECK_TABLE_BIT_SIZE))
    })
}

impl LookupVariant {
    pub(crate) fn is_rmsnorm(&self) -> bool {
        matches!(
            self,
            Self::Normalisation {
                normalised_sum_value: None,
                ..
            }
        )
    }

    /// Columns are ordered by residual side, limb, chunk. Padding rows have
    /// zero residuals; the verifier multiplies the constants by the row mask.
    pub(crate) fn rms_residual_columns(
        &self,
        number_of_chunks: usize,
        output: &Tensor<Element>,
    ) -> Result<Vec<Vec<Element>>> {
        let Self::Normalisation {
            normalised_magnitude_value: target,
            magnitude_error_bound: bound,
            normalised_sum_value: None,
            ..
        } = self
        else {
            bail!("Expected RMSNorm variant")
        };
        let limbs = residual_limbs(*bound)?;
        let shape = output.unpadded_shape();
        let output = output.reduce_to_shape(shape)?;
        let dim = shape.dim(-1);
        let chunk_size = shape.numel() / number_of_chunks;
        let rows = chunk_size / dim;
        let mut columns = vec![vec![0; rows.next_power_of_two()]; 2 * limbs * number_of_chunks];
        for (chunk_idx, chunk) in output.data().chunks(chunk_size).enumerate() {
            for (row_idx, row) in chunk.chunks(dim).enumerate() {
                let sq: i128 = row.iter().map(|&v| i128::from(v) * i128::from(v)).sum();
                ensure!(
                    sq != 0,
                    "Paper-mode RMSNorm cannot prove a genuine zero input row"
                );
                let upper = i128::from(*target) + i128::from(*bound) - sq;
                let lower = sq - (i128::from(*target) - i128::from(*bound));
                ensure!(
                    upper >= 0 && lower >= 0,
                    "RMSNorm squared norm {sq} is outside {target} ± {bound}"
                );
                for (side, residual) in [upper, lower].into_iter().enumerate() {
                    let value = Element::try_from(residual)?;
                    for (limb, value) in decompose(value, limbs)?.into_iter().enumerate() {
                        columns[(side * limbs + limb) * number_of_chunks + chunk_idx][row_idx] =
                            value;
                    }
                }
            }
        }
        // The multi-size LogUp engine requires at least one sumcheck layer,
        // hence four entries. Repeat each row (the low variables), preserving
        // dependence on the tail variables used for the actual row point.
        let copies = 4 / rows.next_power_of_two().min(4);
        if copies > 1 {
            for column in &mut columns {
                *column = column
                    .iter()
                    .flat_map(|value| std::iter::repeat_n(*value, copies))
                    .collect();
            }
        }
        Ok(columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bn254::Fr;
    use ark_ff::{AdditiveGroup, Field};

    fn variant(target: Element, bound: Element) -> LookupVariant {
        LookupVariant::Normalisation {
            normalised_magnitude_value: target,
            magnitude_error_bound: bound,
            normalised_sum_value: None,
            has_weight: false,
        }
    }

    #[test]
    fn rms_two_sided_bound_and_zero_rejection() {
        let tensor = |v| Tensor::new(Shape::new(vec![1, 1]), vec![v]).unwrap();
        assert!(variant(100, 19).rms_residual_columns(1, &tensor(9)).is_ok());
        assert!(
            variant(100, 18)
                .rms_residual_columns(1, &tensor(9))
                .is_err()
        );
        assert!(
            variant(100, 21)
                .rms_residual_columns(1, &tensor(11))
                .is_ok()
        );
        assert!(
            variant(100, 20)
                .rms_residual_columns(1, &tensor(11))
                .is_err()
        );
        assert!(
            variant(100, 100)
                .rms_residual_columns(1, &tensor(0))
                .is_err()
        );
        assert!(decompose(-1, RMS_MULTIPLIER_LIMBS).is_err());
    }

    #[test]
    fn rms_multilimb_verifier_rejects_changed_lower_bound() {
        let variant = variant(4_194_304, 100_000);
        let output = Tensor::new(Shape::new(vec![3, 2]), vec![2048, 0, 2048, 0, 2048, 0]).unwrap();
        let columns = variant.rms_residual_columns(1, &output).unwrap();
        assert_eq!(columns.len(), 4);
        assert!(columns.iter().all(|c| c.len() == 4 && c[3] == 0));
        // Evaluate at a genuine row. Extra entries are the multiplier limbs.
        let mut evals: Vec<Fr> = columns.iter().map(|c| Fr::from(c[0] as u64)).collect();
        evals.extend([Fr::from(17u64), Fr::ZERO, Fr::ZERO]);
        let point = [Fr::ZERO, Fr::ZERO];
        let result = variant.compute_norm_eval(3, &point, &evals).unwrap();
        assert_eq!(result, vec![Fr::from(4_194_304u64), Fr::from(17u64)]);
        // Changing only the lower residual used to go unchecked by the
        // single-offset implementation. Complementary binding detects it.
        evals[2] += Fr::ONE;
        assert!(variant.compute_norm_eval(3, &point, &evals).is_err());
        // Padded rows have neither norm nor multiplier contribution.
        let padding = vec![Fr::ZERO; 7];
        assert_eq!(
            variant
                .compute_norm_eval(3, &[Fr::ONE, Fr::ONE], &padding)
                .unwrap(),
            vec![Fr::ZERO, Fr::ZERO]
        );
    }
}
