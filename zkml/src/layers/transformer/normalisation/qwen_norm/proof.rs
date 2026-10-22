//! Committed integer square-root and reciprocal certificates.
//!
//! Only random evaluations are disclosed. Twenty-eight 16-bit limb columns
//! certify positivity and the root/quotient inequalities. A squared-input
//! sumcheck binds the denominator; a product sumcheck binds the output and
//! effective gamma. The two input claims are combined before leaving the layer.

use std::collections::HashMap;

use anyhow::{Context, Result, ensure};
use ark_ff::PrimeField;
use dp_crypto::{
    Expression, IntoMLE,
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    poly::eq::evals,
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use serde::{Deserialize, Serialize};

use super::*;
use crate::{
    Claim, InitTranscript, Prover, ProverContext,
    iop::{
        context::{ContextAux, ShapeStep},
        same_poly,
        verifier::Verifier,
    },
    layers::{
        LayerCtx, LayerProof,
        provable::{ProvableOp, ProveInfo, VerifiableCtx},
    },
    lookup::{
        context::LookupWitnessGen,
        logup_gkr::{
            prover::new_batch_multiple_sizes_prove,
            structs::{LogUpBatchProof, LogUpInput, LogUpVerifierInstance, ProofType},
            verifier::new_verify_logup_proof_multiple_sizes,
        },
        table::Table,
    },
    model::Step,
    poly_commit::{
        identity_eval,
        verifier::{VerifierClaim, VerifierCommitment},
    },
    quantization::ToField,
    tensor::CommitmentId,
    util::from_mle_list_dimensions,
};

const ROOT: std::ops::Range<usize> = 1..5;
const MULTIPLIER: std::ops::Range<usize> = 5..9;
const LOWER: std::ops::Range<usize> = 9..14;
const UPPER: std::ops::Range<usize> = 14..19;
const REMAINDER: std::ops::Range<usize> = 19..24;
const REMAINDER_UPPER: std::ops::Range<usize> = 24..29;
const COLUMNS: usize = 29;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QwenNormCtx {
    pub dimension: usize,
    pub quant: integer::NormQuant,
    pub alpha_key: Option<CommitmentId>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct QwenNormProof<F: PrimeField, PCS: CommitmentScheme> {
    pub(crate) commitments: Vec<VerifierCommitment<PCS>>,
    pub(crate) ranges: LogUpBatchProof<F>,
    pub(crate) constraints: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) constraint_evals: Vec<F>,
    pub(crate) squared: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) squared_sum_eval: F,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) squared_input_eval: F,
    pub(crate) product: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) product_input_eval: F,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) product_multiplier_evals: Vec<F>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) alpha_eval: F,
    pub(crate) merge: same_poly::Proof<F>,
}

fn field_u128<F: PrimeField>(x: u128) -> F {
    F::from(x as u64) + F::from((x >> 64) as u64) * F::from(2u64).pow([64])
}

fn join<F: PrimeField>(limbs: &[F]) -> F {
    limbs
        .iter()
        .rev()
        .fold(F::ZERO, |acc, x| acc * F::from(65536u64) + x)
}

fn join_expr<F: PrimeField>(limbs: &[Expression<F>]) -> Expression<F> {
    limbs.iter().rev().fold(Expression::ZERO, |acc, x| {
        acc * Expression::Constant(F::from(65536u64)) + x.clone()
    })
}

fn constraints<F: PrimeField>(
    columns: &[Expression<F>],
    quant: &integer::NormQuant,
    alpha: F,
) -> Expression<F> {
    let one = Expression::Constant(F::ONE);
    let two = Expression::Constant(F::from(2u64));
    let root = join_expr(&columns[ROOT]) + one.clone();
    let multiplier = join_expr(&columns[MULTIPLIER]);
    let lower = join_expr(&columns[LOWER]);
    let upper = join_expr(&columns[UPPER]);
    let remainder = join_expr(&columns[REMAINDER]);
    let remainder_upper = join_expr(&columns[REMAINDER_UPPER]);
    let denominator = columns[0].clone()
        * Expression::Constant(F::from(1u64 << integer::DENOMINATOR_FRACTION_BITS))
        + Expression::Constant(field_u128::<F>(quant.epsilon_scaled));
    let identities = [
        root.clone() * root.clone() + lower.clone() - denominator,
        lower + upper - two.clone() * root.clone(),
        two.clone() * root.clone() * multiplier + remainder.clone()
            - Expression::Constant(field_u128::<F>(quant.numerator) * F::from(2u64))
            - root.clone(),
        remainder + remainder_upper - two * root + one,
    ];
    let mut coefficient = F::ONE;
    identities
        .into_iter()
        .fold(Expression::ZERO, |acc, identity| {
            let result = acc + Expression::Constant(coefficient) * identity;
            coefficient *= alpha;
            result
        })
}

fn constraints_eval<F: PrimeField>(c: &[F], quant: &integer::NormQuant, alpha: F) -> F {
    let r = join(&c[ROOT]) + F::ONE;
    let m = join(&c[MULTIPLIER]);
    let a = join(&c[LOWER]);
    let b = join(&c[UPPER]);
    let rem = join(&c[REMAINDER]);
    let upper = join(&c[REMAINDER_UPPER]);
    let d = c[0] * F::from(1u64 << integer::DENOMINATOR_FRACTION_BITS)
        + field_u128::<F>(quant.epsilon_scaled);
    let two = F::from(2u64);
    (r * r + a - d)
        + alpha * (a + b - two * r)
        + alpha.square() * (two * r * m + rem - two * field_u128::<F>(quant.numerator) - r)
        + alpha.square() * alpha * (rem + upper - two * r + F::ONE)
}

fn witness_columns<F: PrimeField>(
    input: &Tensor<Element>,
    quant: &integer::NormQuant,
) -> Result<Vec<Vec<F>>> {
    let rows = (input.data().len() / quant.dimension).max(4);
    ensure!(
        rows.is_power_of_two(),
        "Qwen norm proof expects padded rows"
    );
    let mut columns = vec![Vec::with_capacity(rows); COLUMNS];
    let zero = vec![0; quant.dimension];
    for i in 0..rows {
        let row = input
            .data()
            .get(i * quant.dimension..(i + 1) * quant.dimension)
            .unwrap_or(&zero);
        let c = quant.certify_row(row)?;
        columns[0].push(field_u128(c.sum_squares));
        for (range, value) in [
            (ROOT, u128::from(c.root - 1)),
            (MULTIPLIER, u128::from(c.multiplier)),
            (LOWER, c.root_lower_gap),
            (UPPER, c.root_upper_gap),
            (REMAINDER, c.division_remainder),
            (REMAINDER_UPPER, c.division_upper_gap),
        ] {
            ensure!(
                value < (1u128 << (16 * range.len())),
                "Qwen norm certificate exceeds limb budget"
            );
            for (limb, column) in range.enumerate() {
                columns[column].push(F::from(((value >> (16 * limb)) & 65535) as u64));
            }
        }
    }
    Ok(columns)
}

impl QwenNormCtx {
    pub fn lookup_tables(&self) -> Vec<Table> {
        vec![Table::new_shift_check()]
    }

    fn dimensions(&self, shape: &Shape) -> Result<(usize, usize, usize)> {
        ensure!(
            shape.is_power_of_two() && shape.dim(-1) == self.dimension,
            "Invalid padded Qwen norm shape"
        );
        let variables = shape.product().ilog2() as usize;
        let columns = self.dimension.ilog2() as usize;
        let rows = (variables - columns).max(2);
        Ok((variables, columns, rows))
    }
}

impl OpInfo for QwenNormCtx {
    fn output_shapes(&self, inputs: &[Shape], mode: PaddingMode) -> Result<Vec<Shape>> {
        ensure!(
            inputs.len() == 1 && inputs[0].dim(-1) == self.dimension,
            "Qwen norm context shape mismatch"
        );
        Ok(vec![if matches!(mode, PaddingMode::Padding) {
            inputs[0].next_power_of_two()
        } else {
            inputs[0].clone()
        }])
    }
    fn num_outputs(&self, inputs: usize) -> Result<usize> {
        ensure!(inputs == 1, "Qwen norm expects one input");
        Ok(1)
    }
    fn describe(&self) -> String {
        format!("QwenNormCtx({})", self.dimension)
    }
    fn is_provable(&self) -> bool {
        true
    }
}
impl Splittable for QwenNormCtx {}

impl ProveInfo for QwenNorm<Element> {
    fn step_info<F: PrimeField>(&self, mut aux: ContextAux) -> Result<(LayerCtx<F>, ContextAux)> {
        ensure!(
            F::MODULUS_BIT_SIZE >= 192,
            "Qwen norm integer constraints require at least a 192-bit prime field"
        );
        let quant = self.quant.clone().context("Qwen norm is not quantized")?;
        aux.model_polys = None;
        let alpha_key = if let Some(alpha) = &self.alpha {
            let key = CommitmentId::from(alpha.storage_key());
            aux.model_polys = Some(HashMap::from([(
                key.clone(),
                alpha.pad_next_power_of_two().wrapped_tensor()?.get_data(),
            )]));
            Some(key)
        } else {
            None
        };
        aux.max_poly_len = aux
            .max_poly_len
            .max(aux.last_output_shape[0].next_power_of_two().product())
            .max(4);
        Ok((
            LayerCtx::QwenNorm(QwenNormCtx {
                dimension: self.dimension,
                quant,
                alpha_key,
            }),
            aux,
        ))
    }
}

impl<F: PrimeField, PCS: CommitmentScheme<Field = F>> ProvableOp<F, PCS> for QwenNorm<Element> {
    type Ctx = QwenNormCtx;

    fn gen_lookup_witness(
        &self,
        id: NodeId,
        _ctx: &ProverContext<F, PCS>,
        step: &Step<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        let input = step
            .input_tensor_at(0)?
            .as_ref()
            .clone()
            .pad_next_power_of_two();
        let columns = witness_columns::<F>(
            &input,
            self.quant.as_ref().context("Qwen norm not quantized")?,
        )?;
        let mut counts = HashMap::new();
        for limb in columns.iter().skip(1).flatten() {
            let integer = limb.into_bigint().as_ref()[0] as Element;
            *counts.entry(integer).or_insert(0) += 1;
        }
        let mut witness = LookupWitnessGen::default();
        witness.insert_layer_witness_data(
            id,
            columns.into_iter().map(IntoMLE::into_mle).collect(),
            vec![Table::new_shift_check()],
            vec![counts],
        );
        Ok(witness)
    }

    fn prove<'a, 'b, 'c, 'd, T: Transcript + InitTranscript>(
        &'a self,
        id: NodeId,
        ctx: &'b Self::Ctx,
        last: Vec<&Claim<F>>,
        step: &Step<Element>,
        prover: &mut Prover<'c, 'd, F, T, PCS>,
    ) -> Result<Vec<Claim<F>>> {
        ensure!(last.len() == 1, "Qwen norm needs one output claim");
        let input = step
            .input_tensor_at(0)?
            .as_ref()
            .clone()
            .pad_next_power_of_two()
            .to_field();
        let (original_vars, col_vars, row_vars) = ctx.dimensions(input.shape())?;
        ensure!(
            last[0].point.len() == original_vars,
            "Invalid Qwen norm output claim size"
        );
        let full_vars = col_vars + row_vars;
        let original = input.clone().into_mle();
        let mut extended = input.data().to_vec();
        extended.resize(1 << full_vars, F::ZERO);
        let extended = extended.into_mle();
        let committed = prover.lookup_witness(id)?;
        ensure!(
            committed.len() == COLUMNS,
            "Incorrect Qwen norm witness count"
        );
        let columns = committed
            .iter()
            .map(|p| p.polynomial.clone())
            .collect::<Vec<_>>();
        let commitments = committed
            .iter()
            .map(VerifierCommitment::from)
            .collect::<Vec<_>>();
        let table = Table::new_shift_check();
        let (constant, separation) = prover
            .challenge_storage
            .get_challenges_by_name(&table.name())
            .context("Missing Qwen norm range table challenges")?;
        let range_input = LogUpInput::new_lookup(
            columns.iter().skip(1).map(|p| p.evals()).collect(),
            constant,
            separation,
            1,
        )?;
        let ranges = new_batch_multiple_sizes_prove(&[range_input], prover.transcript)?;
        ensure!(
            ranges.output_claims().len() == COLUMNS - 1,
            "Qwen norm range claim count mismatch"
        );
        let mut openings = vec![Vec::new(); COLUMNS];
        for (target, claim) in openings.iter_mut().skip(1).zip(ranges.output_claims()) {
            target.push(claim.clone());
        }

        let alpha = prover.transcript.challenge_scalar::<F>();
        let row_point = (0..row_vars)
            .map(|_| prover.transcript.challenge_scalar())
            .collect::<Vec<F>>();
        let row_eq = evals(&row_point).into_mle();
        let mut builder =
            VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(row_vars), row_vars);
        let expressions = columns
            .iter()
            .map(|p| builder.lift(Either::Left(p)))
            .collect::<Vec<_>>();
        let eq = builder.lift(Either::Left(&row_eq));
        let polynomial =
            builder.to_virtual_polys(&[eq * constraints(&expressions, &ctx.quant, alpha)], &[]);
        let (constraint_proof, state) = IOPProverState::prove(polynomial, prover.transcript);
        let point = state.collect_raw_challenges();
        let constraint_evals = columns
            .iter()
            .map(|p| p.evaluate(&point))
            .collect::<Result<Vec<_>>>()?;
        for (target, &value) in openings.iter_mut().zip(&constraint_evals) {
            target.push(Claim::new(point.clone(), value));
        }
        prover.transcript.append_scalars(&constraint_evals);

        let squared_point = (0..row_vars)
            .map(|_| prover.transcript.challenge_scalar())
            .collect::<Vec<F>>();
        let squared_sum_eval = columns[0].evaluate(&squared_point)?;
        prover.transcript.append_scalars(&[squared_sum_eval]);
        openings[0].push(Claim::new(squared_point.clone(), squared_sum_eval));
        let squared_eq = evals(&squared_point)
            .into_iter()
            .flat_map(|x| std::iter::repeat_n(x, self.dimension))
            .collect::<Vec<_>>()
            .into_mle();
        let mut builder =
            VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(full_vars), full_vars);
        let x = builder.lift(Either::Left(&extended));
        let eq = builder.lift(Either::Left(&squared_eq));
        let poly = builder.to_virtual_polys(&[eq * x.clone() * x], &[]);
        let (squared, state) = IOPProverState::prove(poly, prover.transcript);
        let point = state.collect_raw_challenges();
        let squared_input_eval = original.evaluate(&point[..original_vars])?;
        let squared_claim = Claim::new(point[..original_vars].to_vec(), squared_input_eval);
        prover.transcript.append_scalars(&[squared_input_eval]);

        let multipliers = (0..(1 << row_vars))
            .map(|i| {
                join(
                    &columns[MULTIPLIER]
                        .iter()
                        .map(|p| p.evals_ref()[i])
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<F>>();
        let multiplier_poly = multipliers
            .into_iter()
            .flat_map(|m| std::iter::repeat_n(m, self.dimension))
            .collect::<Vec<_>>()
            .into_mle();
        let weights = self
            .alpha
            .as_ref()
            .map(|w| -> Result<Vec<F>> { Ok(w.wrapped_tensor()?.get_data().to_field()) })
            .transpose()?
            .unwrap_or_else(|| vec![F::ONE; self.dimension]);
        let weight_poly = (0..(1 << row_vars))
            .flat_map(|_| weights.iter().copied())
            .collect::<Vec<_>>()
            .into_mle();
        let mut output_point = last[0].point.clone();
        output_point.resize(full_vars, F::ZERO);
        let output_eq = evals(&output_point).into_mle();
        let mut builder =
            VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(full_vars), full_vars);
        let x = builder.lift(Either::Left(&extended));
        let m = builder.lift(Either::Left(&multiplier_poly));
        let w = builder.lift(Either::Left(&weight_poly));
        let eq = builder.lift(Either::Left(&output_eq));
        let poly = builder.to_virtual_polys(&[eq * x * m * w], &[]);
        let (product, state) = IOPProverState::prove(poly, prover.transcript);
        let point = state.collect_raw_challenges();
        let product_input_eval = original.evaluate(&point[..original_vars])?;
        let product_multiplier_evals = columns[MULTIPLIER]
            .iter()
            .map(|p| p.evaluate(&point[col_vars..]))
            .collect::<Result<Vec<_>>>()?;
        for (target, &value) in openings[MULTIPLIER]
            .iter_mut()
            .zip(&product_multiplier_evals)
        {
            target.push(Claim::new(point[col_vars..].to_vec(), value));
        }
        let alpha_eval = weights.into_mle().evaluate(&point[..col_vars])?;
        if let Some(key) = &ctx.alpha_key {
            prover.add_common_claims(
                id,
                HashMap::from([(
                    key.clone(),
                    Claim::new(point[..col_vars].to_vec(), alpha_eval),
                )]),
            );
        }
        prover
            .transcript
            .append_scalars(&[product_input_eval, alpha_eval]);
        prover.transcript.append_scalars(&product_multiplier_evals);
        let mut merge = same_poly::Prover::new(original);
        merge.add_claim(squared_claim)?;
        merge.add_claim(Claim::new(
            point[..original_vars].to_vec(),
            product_input_eval,
        ))?;
        let (merge, claim) = merge.prove(prover.transcript)?;
        prover.add_witness_claim(id, openings);
        prover.push_proof(
            id,
            LayerProof::QwenNorm(QwenNormProof {
                commitments,
                ranges,
                constraints: constraint_proof,
                constraint_evals,
                squared,
                squared_sum_eval,
                squared_input_eval,
                product,
                product_input_eval,
                product_multiplier_evals,
                alpha_eval,
                merge,
            }),
        );
        Ok(vec![claim])
    }
}

impl<F: PrimeField, PCS: CommitmentScheme<Field = F>> VerifiableCtx<F, PCS> for QwenNormCtx {
    type Proof = QwenNormProof<F, PCS>;

    fn verify<T: Transcript>(
        &self,
        proof: &Self::Proof,
        last: &[&Claim<F>],
        verifier: &mut Verifier<F, T, PCS>,
        shape: &ShapeStep,
        id: NodeId,
    ) -> Result<Vec<Claim<F>>> {
        ensure!(
            F::MODULUS_BIT_SIZE >= 192,
            "Qwen norm needs a large prime field"
        );
        ensure!(
            last.len() == 1 && shape.padded_input_shape.len() == 1,
            "Qwen norm expects one input/output"
        );
        let (original_vars, col_vars, row_vars) = self.dimensions(&shape.padded_input_shape[0])?;
        let full_vars = col_vars + row_vars;
        ensure!(
            last[0].point.len() == original_vars,
            "Invalid Qwen norm output point"
        );
        ensure!(
            proof.commitments.len() == COLUMNS
                && proof.constraint_evals.len() == COLUMNS
                && proof.product_multiplier_evals.len() == 4,
            "Malformed Qwen norm proof"
        );
        use crate::poly_commit::ChunkedCommitment;
        ensure!(
            proof.commitments.iter().all(|c| c.num_vars() == row_vars),
            "Qwen norm witness size mismatch"
        );
        let table = Table::new_shift_check();
        let (constant, separation) = verifier
            .challenge_storage
            .get_challenges_by_name(&table.name())
            .context("Missing Qwen norm range table challenges")?;
        let instances = vec![
            LogUpVerifierInstance::new(
                constant,
                separation,
                1,
                ProofType::Lookup,
                row_vars - 1
            );
            COLUMNS - 1
        ];
        ensure!(
            proof.ranges.num_instances() == COLUMNS - 1
                && proof.ranges.circuit_outputs().iter().all(|v| v.len() == 4),
            "Malformed Qwen norm range proof"
        );
        let batch =
            new_verify_logup_proof_multiple_sizes(&proof.ranges, &instances, verifier.transcript)?;
        ensure!(
            batch.output_claims().len() == COLUMNS - 1,
            "Qwen norm range claims mismatch"
        );
        let (numerators, denominators) = proof.ranges.fractional_outputs();
        let (num, denom) = verifier
            .numerators_and_denominators
            .entry(table.name())
            .or_insert((F::ZERO, F::ONE));
        for (n, d) in numerators.iter().zip(&denominators) {
            ensure!(*d != F::ZERO, "Zero Qwen norm lookup denominator");
            *num = *n * *denom + *num * *d;
            *denom *= d;
        }
        let mut openings = vec![Vec::new(); COLUMNS];
        for (target, claim) in openings.iter_mut().skip(1).zip(batch.output_claims()) {
            target.push(claim.clone());
        }
        let alpha = verifier.transcript.challenge_scalar::<F>();
        let row_point = (0..row_vars)
            .map(|_| verifier.transcript.challenge_scalar())
            .collect::<Vec<F>>();
        let sub = IOPVerifierState::verify(
            F::ZERO,
            &proof.constraints,
            &from_mle_list_dimensions(&[vec![row_vars; 3]]),
            verifier.transcript,
        );
        ensure!(
            sub.expected_evaluation
                == identity_eval(&row_point, &sub.point)
                    * constraints_eval(&proof.constraint_evals, &self.quant, alpha),
            "Qwen norm integer certificate constraints failed"
        );
        for (target, &value) in openings.iter_mut().zip(&proof.constraint_evals) {
            target.push(Claim::new(sub.point.clone(), value));
        }
        verifier.transcript.append_scalars(&proof.constraint_evals);

        let squared_point = (0..row_vars)
            .map(|_| verifier.transcript.challenge_scalar())
            .collect::<Vec<F>>();
        verifier
            .transcript
            .append_scalars(&[proof.squared_sum_eval]);
        openings[0].push(Claim::new(squared_point.clone(), proof.squared_sum_eval));
        let sub = IOPVerifierState::verify(
            proof.squared_sum_eval,
            &proof.squared,
            &from_mle_list_dimensions(&[vec![full_vars; 3]]),
            verifier.transcript,
        );
        let zero_pad: F = sub.point[original_vars..]
            .iter()
            .map(|x| F::ONE - x)
            .product();
        let x = proof.squared_input_eval * zero_pad;
        ensure!(
            sub.expected_evaluation
                == identity_eval(&squared_point, &sub.point[col_vars..]) * x.square(),
            "Qwen norm squared-input relation failed"
        );
        let squared_claim = Claim::new(
            sub.point[..original_vars].to_vec(),
            proof.squared_input_eval,
        );
        verifier
            .transcript
            .append_scalars(&[proof.squared_input_eval]);

        let mut output_point = last[0].point.clone();
        output_point.resize(full_vars, F::ZERO);
        let sub = IOPVerifierState::verify(
            last[0].eval,
            &proof.product,
            &from_mle_list_dimensions(&[vec![full_vars; 4]]),
            verifier.transcript,
        );
        let zero_pad: F = sub.point[original_vars..]
            .iter()
            .map(|x| F::ONE - x)
            .product();
        ensure!(
            sub.expected_evaluation
                == identity_eval(&output_point, &sub.point)
                    * zero_pad
                    * proof.product_input_eval
                    * join(&proof.product_multiplier_evals)
                    * proof.alpha_eval,
            "Qwen norm output product relation failed"
        );
        for (target, &value) in openings[MULTIPLIER]
            .iter_mut()
            .zip(&proof.product_multiplier_evals)
        {
            target.push(Claim::new(sub.point[col_vars..].to_vec(), value));
        }
        if let Some(key) = &self.alpha_key {
            verifier.add_common_claims(
                id,
                HashMap::from([(
                    key.clone(),
                    Claim::new(sub.point[..col_vars].to_vec(), proof.alpha_eval),
                )]),
            );
        } else {
            ensure!(proof.alpha_eval == F::ONE, "Unexpected Qwen norm weight");
        }
        verifier
            .transcript
            .append_scalars(&[proof.product_input_eval, proof.alpha_eval]);
        verifier
            .transcript
            .append_scalars(&proof.product_multiplier_evals);
        let context = same_poly::Context::new(original_vars);
        let mut merge = same_poly::Verifier::new(&context);
        merge.add_claim(squared_claim)?;
        merge.add_claim(Claim::new(
            sub.point[..original_vars].to_vec(),
            proof.product_input_eval,
        ))?;
        let claim = merge.verify(&proof.merge, verifier.transcript)?;
        verifier.commit_verifier.add_witness_claim(
            id,
            proof
                .commitments
                .iter()
                .cloned()
                .zip(openings)
                .map(|(commitment, claims)| VerifierClaim { commitment, claims })
                .collect(),
        );
        Ok(vec![claim])
    }

    fn write_proof_to_transcript<T: Transcript>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<()> {
        ensure!(
            proof.commitments.len() == COLUMNS,
            "Malformed Qwen norm commitments"
        );
        for commitment in &proof.commitments {
            commitment.append_to_transcript(transcript);
        }
        Ok(())
    }
}
