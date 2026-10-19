# RMSNorm for the Llama 3 integration

This document describes the RMSNorm numerical and proof changes. It does **not**
claim that a complete Llama 3.2 1B inference or proof has passed validation.

## Paper relation and Hugging Face epsilon

[DeepProve](https://eprint.iacr.org/2026/1112.pdf) §4.3.3, Appendix B.6.1
(Protocol 10, p. 24), and Appendix C (pp. 32–34) use the nonzero-row
relation

\[
\operatorname{RMSNorm}(x)=\frac{x}{\sqrt{\sum_i x_i^2/d}}.
\]

The formula omits the additive Hugging Face `rms_norm_eps`. Appendix C's
\(\epsilon_j\) is a fixed-point **mantissa**, not that additive stabilizer. This
implementation follows the paper relation in quantized inference. Float
inference retains the model's additive epsilon, so the difference can be
measured against the Hugging Face reference. There is no epsilon-aware proof
extension here.

The learned weight is applied after normalization, followed by the existing
requantization operation. Norm constraints refer to the unweighted result.

## Twelve-bit precision and the scale

For requested signed precision \(b\), the nominal integer domain is
\([-2^{b-1},2^{b-1}-1]\), and the normalization scale is

\[
s=\frac{2\sqrt d}{2^b}.
\]

`ScalingFactor::bit_size()` reports the magnitude exponent, excluding the sign:
this domain returns \(b-1\), not \(b\). At \(d=2048,b=12\), the scale is about
0.0220971 and the integer squared-norm target is about 4,194,304. The scale is
stored as `f32`; both witness generation and verification use the same shared
routine to derive the exact integer target and bound from that stored scale.

The previous scale-selection loop reduced precision until its norm check fit a
single 16-bit table. The new implementation retains the requested precision
and decomposes the norm residual into as many 16-bit limbs as needed. A guard
bit in the normalization lookup table is separate from the requested precision:
it increases proof-table capacity without changing \(s\).

This is a specific strategy for preserving the 12-bit normalization
intermediate: its range is the static spherical bound \(\sqrt d\), not an
observed calibration maximum. Learned weights, residuals, and layer outputs
retain their independent scales. If complete-model quality measurements later
require more normalization precision, that intermediate can be tuned separately;
the operator checks alone do not justify changing the global bit width.

## Integer target and rounding bound

Let \(x\) be a nonzero integer row, and define

\[
T=\operatorname{RoundEven}(d/s^2),\qquad
a=\sqrt{T/\sum_i x_i^2},\qquad z=ax.
\]

Thus \(\|z\|_2=\sqrt T\). Rounding the paper's real target to \(T\) changes its
squared norm by at most one half; the proof statement explicitly uses \(T\).

The witness builder chooses an exponent \(e=\lfloor\log_2 a\rfloor\), writes
\(a=2^e m\) with \(m\in[1,2)\), and rounds \(m2^{15}\) to an integer. The
resulting positive fixed-point multiplier has relative rounding error at most
\(2^{-16}\). This differs from the former `fract()/trunc()` decomposition,
whose mantissa could fall below one for negative exponents. A common right
shift is obtained by left-shifting mantissas; that operation preserves their
represented ratios exactly.

For final integer rounding, each coordinate has error at most \(1/2\).
Consequently, with \(\eta=2^{-16}\),

\[
\|y-z\|_2\leq\Delta=\eta\sqrt T+\frac{\sqrt d}{2},
\]

and Cauchy--Schwarz gives

\[
\left|\sum_i y_i^2-T\right|
\leq 2\sqrt T\Delta+\Delta^2.
\]

This is Appendix C's rounding argument with the mantissa approximation added.
The implemented envelope uses \(\eta=2^{-16}+2^{-45}\), rounds the expression
upward to an integer, and adds one integer unit. The small additional allowance
and integer margin cover the floating-point witness arithmetic. Input squares
are accumulated exactly with checked `u128` arithmetic and converted to `f64`
only once before division and square root; there is no row-length-dependent
floating-point accumulation error. These calculations only propose a witness:
the integer norm constraint must still pass.

The implementation rejects invalid scales, unsupported targets/bounds, and
precision so low that \(B\geq T\). It also checks the fixed-point products and
rounding additions fit `i64`. For the 2048-wide, 12-bit configuration, the bound
is below 100,000, exceeds 65,535, and uses two limbs per residual.

## Two-sided range proof and multiplier binding

Let \(S=\sum_i y_i^2\). The proof range-checks both

\[
U=T+B-S,\qquad L=S-(T-B).
\]

Each residual is decomposed into unsigned base-\(2^{16}\) limbs, looked up in
the existing 16-bit shift-check table. Residuals can use up to three limbs.
Their recompositions are bound at the LogUp row point by

\[
U+L=2B.
\]

This makes both inequalities effective: \(U,L\geq0\) together with their sum
gives \(T-B\leq S\leq T+B\). Merely shifting one residual into a large lookup
table does not provide this pair of bounds. Two 48-bit residuals sum to less
than \(2^{49}\); verification checks that the field is large enough for this
recomposition. The application and tests use BN254, where these integer
relations and the `i64`-input/48-bit-multiplier products do not wrap modulo the
field.

The row multiplier itself is decomposed into three unsigned 16-bit limbs,
constraining it to \([0,2^{48})\). An additional normalisation-challenge term in
the existing linking sumcheck binds that reconstruction to the **same multiplier
witness** used in the input-times-multiplier relation and opened against its
commitment. This is necessary because a squared-norm check alone accepts a
sign-flipped multiplier. The proof's right shift is validated and appended to
the transcript before the layer's LogUp and linking challenges.

These additional lookups and transcript binding change the proof relation.
Existing RMSNorm proofs must be regenerated; retaining the Rust proof-struct
fields does not make old proof transcripts compatible.

## Guard bit and saturation

Appendix C infers the normalization multiplier from a rounded multiple of the
input. Saturating that multiple would invalidate the inference: clipping a
dominant coordinate can hide a multiplier that is too large.

The RMS lookup table therefore uses `normalisation_scaling.bit_size() + 2`
bits, one guard bit beyond the nominal signed domain. The implementation
requires the square of either clamping endpoint to exceed \(T+B\). Since the
sum of squares is constrained to at most \(T+B\), an accepted row cannot
contain a clamped coordinate. The weighted-output intermediate range also
reserves this additional bit.

## Padding and zero rows

For three or more rows, residual and multiplier lookup columns use ordinary
power-of-two padding with zeros. The verifier multiplies the constant
\(2B\) and target terms by the row-selector polynomial, so padded rows have no
normalization contribution.

The multi-size LogUp implementation needs at least four evaluations to include
an instance in a sumcheck round. For a single row, its constant lookup value is
repeated four times. For two rows, **each** row is repeated twice, producing
`[row0, row0, row1, row1]`. This order makes the added low variable irrelevant
and preserves dependence on the original tail row variable. Multiplicities
include the same repetitions; the committed multiplier remains at its original
row shape. These are repeated lookup values, not extra model-output rows.

Numerical quantized evaluation maps a genuine zero row to zero, without a
division-by-zero or logarithm panic. Paper-mode proving rejects that row with a
clear error. Its \(B<T\) constraint also prevents a zero normalized output from
being accepted. Ordinary padding zeros are excluded by the row selector and
are supported. An additive-epsilon relation or a separately constrained zero
branch would be needed to expand this proof domain.

## Validation record

The final linked release test binary passed all eight RMSNorm module tests
(72.84 seconds) and both RMS range-helper tests. The module run used
`PROPTEST_CASES=8`. Coverage includes:

- 2048-wide, three-row proving and verification, including a padded fourth row.
- Actual one-row and two-row proving and verification with the minimum-four
  lookup repetition scheme.
- The existing test containing 25 random small norm models, with and without
  learned weights and with multiple tensor ranks.
- Twelve-bit scale/target and two-limb bound assertions.
- Numerical 2048-wide mixed-magnitude and genuine-zero rows.
- Existing numerical and heterogeneous-magnitude checks, plus eight numerical
  property-test cases.
- Both norm-bound endpoints, out-of-range residuals, negative multiplier
  decomposition, a changed complementary residual, and row padding.

Commands used after the final release library test build:

```sh
PROPTEST_CASES=8 target/release/deps/zkml-cc020cfd0f94709f \
  layers::transformer::normalisation::rmsnorm --nocapture
target/release/deps/zkml-cc020cfd0f94709f \
  lookup::operation::variant::rms::tests --nocapture
```

The binary suffix is build-specific. To rebuild and rerun through Cargo, use
`CARGO_HOME=/tmp/deep-prove-cargo PROPTEST_CASES=8 cargo test --release --offline
--locked -p zkml layers::transformer::normalisation::rmsnorm --lib`, and run the
second helper filter separately. The same library build also passed the
separate generation/proof-sequence tests, but those are not evidence of
complete-model numerical quality.

An earlier run exposed the single-row LogUp zero-layer failure; the validated run
above includes its fix, the final guard checks, and the exact `u128` input-norm
accumulation. No complete-model accuracy, text-generation, or 1B-model proof
claim follows from these operator tests.

### Controlled normalization-resolution ablation

A separate HF FP32 experiment ran all 16 decoder blocks on the first frozen
development sequence (255 prediction tokens). It removed epsilon and rounded
only the unweighted normalized vectors using the static scale above; input,
learned-weight, and output quantization were disabled, and the normalizer was
computed directly in floating point. This isolates normalization resolution;
it is not a substitute for native integer inference or proof verification.

| Normalized-vector precision | Mean cosine vs HF | Top-1 agreement | PPL ratio vs HF |
| --- | ---: | ---: | ---: |
| No rounding, epsilon omitted | 0.99997472 | 0.99216 | 0.999946 |
| 12 bits | 0.99983783 | 0.97255 | 1.002113 |
| 14 bits | 0.99996543 | 0.98431 | 0.999895 |
| 16 bits | 0.99997474 | 0.98824 | 1.000023 |

At 12 bits, the largest local relative L2 error after applying the learned norm
weight was 0.00779. These results did not reproduce the much larger error in
the initial integer run. A separate GLU-output-only ablation did reproduce that
error, motivating the GLU precision correction documented in
[llama3.md](llama3.md). The RMS intermediate therefore remains at 12 bits.
The detailed record is `target/llama3/rms-resolution-ablation-dev0.json`.
