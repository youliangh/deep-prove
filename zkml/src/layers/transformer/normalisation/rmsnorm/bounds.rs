//! Integer target and rounding envelope shared by witness generation/verifying.

use super::*;

/// Appendix C's Cauchy--Schwarz rounding bound, including the error introduced
/// by rounding the fixed-point mantissa. We normalize toward the integer target
/// T, so `||z|| = sqrt(T)`. With a mantissa in [1, 2), its relative error is at
/// most eta = 2^(-FIXED_POINT_SCALE-1). Final rounding contributes at most
/// sqrt(d)/2 in Euclidean norm. Thus Delta = eta*sqrt(T)+sqrt(d)/2 and
/// | ||y||² - T | <= 2*sqrt(T)*Delta + Delta².
pub(super) fn norm_parameters(dim: usize, scale: f32) -> Result<(Element, Element)> {
    ensure!(
        dim > 0 && scale.is_finite() && scale > 0.0,
        "Invalid RMSNorm dimension/scale"
    );
    let target = (dim as f64 / f64::from(scale).powi(2)).round_ties_even();
    ensure!(
        target >= 1.0 && target < (1u64 << 48) as f64,
        "RMSNorm target outside supported range"
    );
    // The tiny extra allowance covers f64 square-root/mantissa arithmetic; the
    // final integer ceiling plus one is conservative at floating boundaries.
    let eta = 2f64.powi(-(FIXED_POINT_SCALE as i32) - 1) + 2f64.powi(-45);
    let delta = eta * target.sqrt() + 0.5 * (dim as f64).sqrt();
    let bound = (2.0 * target.sqrt() * delta + delta * delta).ceil() + 1.0;
    ensure!(
        bound < target,
        "RMSNorm precision is too low to constrain a nonzero norm"
    );
    ensure!(
        bound < (1u64 << 47) as f64,
        "RMSNorm rounding bound outside supported range"
    );
    Ok((target as Element, bound as Element))
}

/// Static precision no longer depends on fitting the norm error in one table.
pub(super) fn normalisation_scale(dim: usize, bits: usize) -> ScalingFactor {
    let max = (dim as f32).sqrt();
    ScalingFactor::from_parts(
        max,
        -max,
        2.0 * max / (1u64 << bits) as f32,
        (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama_width_retains_twelve_bits() {
        let scale = normalisation_scale(2048, 12);
        assert_eq!(scale.bit_size() + 1, 12);
        let (target, bound) = norm_parameters(2048, scale.scale()).unwrap();
        assert!((target - 4_194_304).abs() <= 1);
        assert!(bound > 65_535 && bound < 100_000);
        assert_eq!(
            crate::lookup::operation::variant::rms::residual_limbs(bound).unwrap(),
            2
        );
    }
}
