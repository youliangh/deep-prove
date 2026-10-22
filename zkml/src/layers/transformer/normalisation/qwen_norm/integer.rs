//! Deterministic integer witness generation; no floating arithmetic at runtime.
use super::*;

pub const DENOMINATOR_FRACTION_BITS: u32 = 48;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NormQuant {
    pub dimension: usize,
    pub epsilon_scaled: u128,
    pub numerator: u128,
    pub multiplier_fraction_bits: u32,
    pub input_abs_bound: u64,
}

#[derive(Clone, Debug)]
pub struct NormCertificate {
    pub sum_squares: u128,
    pub root: u64,
    pub multiplier: u64,
    pub root_lower_gap: u128,
    pub root_upper_gap: u128,
    pub division_remainder: u128,
    pub division_upper_gap: u128,
}

impl NormQuant {
    pub fn new(
        dimension: usize,
        kind: NormKind,
        eps: f32,
        input: ScalingFactor,
        normal: ScalingFactor,
        precision: u32,
    ) -> Result<Self> {
        ensure!(
            dimension > 0 && dimension.is_power_of_two(),
            "Invalid norm dimension"
        );
        ensure!(
            input.scale().is_finite()
                && input.scale() > 0.0
                && normal.scale().is_finite()
                && normal.scale() > 0.0,
            "Invalid norm scales"
        );
        ensure!(
            eps.is_finite() && eps > 0.0 && (8..=24).contains(&precision),
            "Invalid norm epsilon/precision"
        );
        let factor = if kind == NormKind::Rms {
            dimension as f64
        } else {
            1.0
        };
        let epsilon = (f64::from(eps) * factor / f64::from(input.scale()).powi(2)
            * 2f64.powi(DENOMINATOR_FRACTION_BITS as i32))
        .round_ties_even();
        let numerator = (factor.sqrt() / f64::from(normal.scale())
            * 2f64.powi((precision + DENOMINATOR_FRACTION_BITS / 2) as i32))
        .round_ties_even();
        ensure!(
            epsilon >= 1.0 && epsilon < 2f64.powi(120),
            "Epsilon cannot be represented at supported precision"
        );
        ensure!(
            numerator >= 1.0 && numerator < 2f64.powi(120),
            "Reciprocal numerator exceeds supported precision"
        );
        let input_abs_bound = input
            .domain()
            .0
            .unsigned_abs()
            .max(input.domain().1.unsigned_abs());
        let maximum_sum = u128::from(input_abs_bound)
            .checked_mul(u128::from(input_abs_bound))
            .and_then(|v| v.checked_mul(dimension as u128))
            .context("Norm sum bound overflow")?;
        ensure!(
            maximum_sum < (1u128 << (128 - DENOMINATOR_FRACTION_BITS)),
            "Norm input range exceeds denominator budget"
        );
        maximum_sum
            .checked_shl(DENOMINATOR_FRACTION_BITS)
            .and_then(|v| v.checked_add(epsilon as u128))
            .context("Norm denominator bound exceeds u128")?;
        Ok(Self {
            dimension,
            epsilon_scaled: epsilon as u128,
            numerator: numerator as u128,
            multiplier_fraction_bits: precision,
            input_abs_bound,
        })
    }

    pub fn denominator(&self, sum_squares: u128) -> Result<u128> {
        ensure!(
            sum_squares < (1u128 << (128 - DENOMINATOR_FRACTION_BITS)),
            "Norm squared sum exceeds supported range"
        );
        (sum_squares << DENOMINATOR_FRACTION_BITS)
            .checked_add(self.epsilon_scaled)
            .context("Norm denominator overflow")
    }

    pub fn certify_row(&self, row: &[Element]) -> Result<NormCertificate> {
        ensure!(row.len() == self.dimension, "Norm row length mismatch");
        let sum_squares = row.iter().try_fold(0u128, |sum, &x| {
            ensure!(
                x.unsigned_abs() <= self.input_abs_bound,
                "Norm input {x} exceeds calibrated integer magnitude {}",
                self.input_abs_bound
            );
            let magnitude = u128::from(x.unsigned_abs());
            sum.checked_add(magnitude * magnitude)
                .context("Norm squared sum overflow")
        })?;
        let denominator = self.denominator(sum_squares)?;
        let root = denominator.isqrt();
        ensure!(
            root > 0 && root <= u64::MAX as u128,
            "Norm sqrt is outside u64"
        );
        let doubled_root = root * 2;
        let dividend = self
            .numerator
            .checked_mul(2)
            .and_then(|n| n.checked_add(root))
            .context("Norm quotient numerator overflow")?;
        let multiplier = dividend / doubled_root;
        ensure!(
            multiplier <= u64::MAX as u128,
            "Norm multiplier exceeds u64"
        );
        let remainder = dividend % doubled_root;
        let lower = denominator - root * root;
        // (r+1)^2-1-D = 2r - (D-r²), avoiding overflow at u128::MAX.
        let upper = doubled_root
            .checked_sub(lower)
            .context("Invalid integer sqrt")?;
        let certificate = NormCertificate {
            sum_squares,
            root: root as u64,
            multiplier: multiplier as u64,
            root_lower_gap: lower,
            root_upper_gap: upper,
            division_remainder: remainder,
            division_upper_gap: doubled_root - 1 - remainder,
        };
        self.verify_certificate(&certificate)?;
        Ok(certificate)
    }

    pub fn verify_certificate(&self, c: &NormCertificate) -> Result<()> {
        ensure!(c.root > 0, "Zero reciprocal denominator");
        let d = self.denominator(c.sum_squares)?;
        let r = u128::from(c.root);
        ensure!(
            r * r <= d && d - r * r == c.root_lower_gap,
            "Incorrect square root lower bound"
        );
        ensure!(
            c.root_lower_gap.checked_add(c.root_upper_gap) == Some(2 * r),
            "Incorrect square root upper bound"
        );
        let target = self
            .numerator
            .checked_mul(2)
            .and_then(|v| v.checked_add(r))
            .context("Norm numerator overflow")?;
        let quotient = (2 * r)
            .checked_mul(u128::from(c.multiplier))
            .and_then(|v| v.checked_add(c.division_remainder))
            .context("Norm quotient overflow")?;
        ensure!(quotient == target, "Incorrect reciprocal quotient");
        ensure!(
            c.division_remainder.checked_add(c.division_upper_gap) == Some(2 * r - 1),
            "Incorrect reciprocal remainder range"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn quant(kind: NormKind) -> NormQuant {
        NormQuant::new(
            4,
            kind,
            1e-6,
            ScalingFactor::from_scale(0.001, Some((-2048, 2047))),
            ScalingFactor::from_scale(0.001, Some((-2048, 2047))),
            24,
        )
        .unwrap()
    }
    #[test]
    fn zero_and_small_rows_preserve_epsilon() {
        for kind in [NormKind::Rms, NormKind::L2] {
            let q = quant(kind);
            for row in [[0, 0, 0, 0], [1, -1, 0, 0], [100, -200, 300, -400]] {
                let certificate = q.certify_row(&row).unwrap();
                q.verify_certificate(&certificate).unwrap();
                let divisor = if kind == NormKind::Rms { 4.0 } else { 1.0 };
                let norm = ((row.iter().map(|x| (*x as f64 * 0.001).powi(2)).sum::<f64>()
                    / divisor)
                    + 1e-6)
                    .sqrt();
                for x in row {
                    let actual = x as f64 * certificate.multiplier as f64 / 2f64.powi(24) * 0.001;
                    let expected = x as f64 * 0.001 / norm;
                    assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
                }
            }
        }
    }
    #[test]
    fn modified_integer_certificates_are_rejected() {
        let q = quant(NormKind::Rms);
        let honest = q.certify_row(&[1, 2, 3, 4]).unwrap();
        let mut bad = honest.clone();
        bad.root += 1;
        assert!(q.verify_certificate(&bad).is_err());
        let mut bad = honest.clone();
        bad.multiplier += 1;
        assert!(q.verify_certificate(&bad).is_err());
        let mut bad = honest;
        bad.division_remainder += 1;
        assert!(q.verify_certificate(&bad).is_err());
    }
}
