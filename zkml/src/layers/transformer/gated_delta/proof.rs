//! Sequence-wide committed proof of the exact integer DeltaNet recurrence.
//!
//! One cubic sumcheck binds all arithmetic and the two contractions. A second
//! quadratic sumcheck binds the previous-state evaluation to the same committed
//! state tensor, shifted causally with a fixed zero initial state. No state rows
//! or other intermediate values are revealed.
use super::{witness::*, *};
use crate::{
    Claim, InitTranscript, Prover, ProverContext,
    iop::{context::ShapeStep, verifier::Verifier},
    layers::{
        LayerProof,
        provable::{ProvableOp, Splittable, VerifiableCtx},
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
        ChunkedCommitment, identity_eval,
        verifier::{VerifierClaim, VerifierCommitment},
    },
    quantization::ToField,
    util::from_mle_list_dimensions,
};
use anyhow::Context;
use dp_crypto::{
    Expression, IntoMLE,
    arkyper::{
        CommitmentScheme,
        transcript::{AppendToTranscript, Transcript},
    },
    poly::{dense::DensePolynomial, eq::evals},
    structs::{IOPProof, IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use std::{
    collections::BTreeMap,
    ops::{Add, Mul, Sub},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GatedDeltaScanCtx {
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    quant: DeltaQuant,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
pub struct GatedDeltaScanProof<F: PrimeField, PCS: CommitmentScheme> {
    pub(crate) commitments: Vec<VerifierCommitment<PCS>>,
    // Large range columns are proved one at a time, bounding GKR peak memory.
    pub(crate) ranges: Vec<LogUpBatchProof<F>>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) output_evals: Vec<F>,
    pub(crate) arithmetic: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) arithmetic_evals: Vec<F>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) input_evals: Vec<F>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) previous_eval: F,
    pub(crate) shift: IOPProof<F>,
    #[serde(with = "dp_crypto::serialization")]
    pub(crate) shift_evals: Vec<F>,
}

#[derive(Clone, Copy)]
struct Dimensions {
    rows: usize,
    valid: usize,
    h: usize,
    k: usize,
    v: usize,
}
impl Dimensions {
    fn state_len(self) -> usize {
        self.rows * self.h * self.k * self.v
    }
    fn vector_len(self) -> usize {
        self.rows * self.h * self.v
    }
    fn vars(self) -> usize {
        self.state_len().ilog2() as usize
    }
    fn vector_vars(self) -> usize {
        self.vector_len().ilog2() as usize
    }
    fn kv_vars(self) -> usize {
        (self.k * self.v).ilog2() as usize
    }
    fn time_start(self) -> usize {
        (self.h * self.k * self.v).ilog2() as usize
    }
    fn vector_index(self, i: usize) -> usize {
        i / (self.k * self.v) * self.v + i % self.v
    }
    fn vector_point<F: Clone>(self, p: &[F]) -> Vec<F> {
        let v = self.v.ilog2() as usize;
        p[..v]
            .iter()
            .chain(&p[v + self.k.ilog2() as usize..])
            .cloned()
            .collect()
    }
    fn input_point<F: Clone>(self, p: &[F], input: usize) -> Vec<F> {
        match input {
            0 | 1 => p[self.v.ilog2() as usize..].to_vec(),
            2 => self.vector_point(p),
            3 | 4 => p[self.kv_vars()..].to_vec(),
            _ => unreachable!(),
        }
    }
    fn column_point<F: Clone>(self, p: &[F], state: bool) -> Vec<F> {
        if state {
            p.to_vec()
        } else {
            self.vector_point(p)
        }
    }
    fn valid_eval<F: PrimeField>(self, p: &[F]) -> F {
        evals(&p[self.time_start()..])
            .into_iter()
            .take(self.valid)
            .sum()
    }
}

impl DeltaQuant {
    fn validate(&self) -> Result<()> {
        for scale in [
            self.decay,
            self.memory,
            self.value,
            self.delta,
            self.update,
            self.output,
        ] {
            ensure!(
                scale.multiplier > 0
                    && scale.multiplier < (1i64 << 52)
                    && scale.shift <= 100
                    && scale.shift != 1,
                "DeltaNet fixed scale outside the proved arithmetic domain"
            );
        }
        ensure!(
            self.input_domains.len() == 5
                && self.input_domains.iter().all(|&(lo, hi)| lo <= 0
                    && hi >= 0
                    && lo >= i32::MIN as i64
                    && hi <= i32::MAX as i64),
            "DeltaNet proof requires bounded signed32 inputs"
        );
        Ok(())
    }
}
impl GatedDeltaScanCtx {
    pub fn lookup_tables(&self) -> Vec<Table> {
        Layout::new(&self.quant).tables()
    }
    fn dimensions(&self, padded: &[Shape], unpadded: &[Shape]) -> Result<Dimensions> {
        let layer = GatedDeltaScan::<Element>::new(self.heads, self.key_dim, self.value_dim);
        layer.output_shapes(padded, PaddingMode::NoPadding)?;
        layer.output_shapes(unpadded, PaddingMode::NoPadding)?;
        self.quant.validate()?;
        ensure!(
            self.key_dim <= 256,
            "DeltaNet proof contraction bound exceeded"
        );
        let d = Dimensions {
            rows: padded[0].dim(0),
            valid: unpadded[0].dim(0),
            h: self.heads,
            k: self.key_dim,
            v: self.value_dim,
        };
        ensure!(
            d.rows.is_power_of_two() && d.valid > 0 && d.valid <= d.rows && d.vector_len() >= 4,
            "Invalid DeltaNet proof dimensions"
        );
        Ok(d)
    }
}
impl OpInfo for GatedDeltaScanCtx {
    fn output_shapes(&self, shapes: &[Shape], mode: PaddingMode) -> Result<Vec<Shape>> {
        GatedDeltaScan::<Element>::new(self.heads, self.key_dim, self.value_dim)
            .output_shapes(shapes, mode)
    }
    fn num_outputs(&self, n: usize) -> Result<usize> {
        ensure!(n == 5, "DeltaNet expects five inputs");
        Ok(1)
    }
    fn describe(&self) -> String {
        "GatedDeltaScanCtx".into()
    }
    fn is_provable(&self) -> bool {
        true
    }
}
impl Splittable for GatedDeltaScanCtx {}
impl ProveInfo for GatedDeltaScan<Element> {
    fn step_info<F: PrimeField>(&self, mut aux: ContextAux) -> Result<(LayerCtx<F>, ContextAux)> {
        ensure!(
            F::MODULUS_BIT_SIZE >= 192,
            "DeltaNet integer proof needs at least a 192-bit prime"
        );
        let quant = self.quant.clone().context("DeltaNet is not quantized")?;
        quant.validate()?;
        ensure!(
            self.key_dim <= 256,
            "DeltaNet contraction exceeds proved bound"
        );
        aux.model_polys = None;
        let rows = aux.last_output_shape[0].dim(0).next_power_of_two();
        aux.max_poly_len = aux
            .max_poly_len
            .max(rows * self.heads * self.key_dim * self.value_dim);
        Ok((
            LayerCtx::GatedDeltaScan(GatedDeltaScanCtx {
                heads: self.heads,
                key_dim: self.key_dim,
                value_dim: self.value_dim,
                quant,
            }),
            aux,
        ))
    }
}

fn pow2<F: PrimeField>(bits: u32) -> F {
    F::from(2u64).pow([bits as u64])
}
fn half<F: PrimeField>(scale: FixedScale) -> F {
    if scale.shift == 0 {
        F::ZERO
    } else {
        pow2(scale.shift - 1)
    }
}

/// Identical symbolic equations are used by prover and verifier.
fn relation<F: PrimeField, A>(
    g: &[A],
    x: &[A],
    previous: A,
    valid: A,
    eq: A,
    vector_eq: A,
    quant: &DeltaQuant,
    key_dim: usize,
    alpha: F,
    c: impl Fn(F) -> A,
) -> A
where
    A: Clone + Add<Output = A> + Sub<Output = A> + Mul<Output = A>,
{
    let identity = |scale: FixedScale, raw: A, result: A, remainder: A| {
        c(F::from(scale.multiplier as u64)) * raw + c(half(scale)) * valid.clone()
            - c(pow2(scale.shift)) * result
            - remainder
    };
    let mut identities = vec![
        identity(
            quant.decay,
            x[3].clone() * previous,
            g[U].clone(),
            g[RDEC].clone(),
        ),
        identity(
            quant.update,
            x[1].clone() * g[D].clone(),
            g[S].clone() - g[U].clone(),
            g[RUPD].clone(),
        ),
        identity(
            quant.value,
            x[2].clone() * valid.clone(),
            g[B].clone(),
            g[RVAL].clone(),
        ),
        identity(
            quant.delta,
            x[4].clone() * (g[B].clone() - g[M].clone()),
            g[D].clone(),
            g[RDELTA].clone(),
        ),
    ];
    for value in g {
        identities.push((c(F::ONE) - valid.clone()) * value.clone());
    }
    let mut coefficient = F::ONE;
    let mut local = c(F::ZERO);
    for identity in identities {
        local = local + c(coefficient) * identity;
        coefficient *= alpha;
    }
    let inverse_k = F::from(key_dim as u64).inverse().unwrap();
    // The vector RHS is broadcast over K, hence its 1/K coefficient.
    let memory = c(F::from(quant.memory.multiplier as u64)) * x[1].clone() * g[U].clone()
        - c(inverse_k)
            * (c(pow2(quant.memory.shift)) * g[M].clone() + g[RMEM].clone()
                - c(half(quant.memory)) * valid.clone());
    let output = c(F::from(quant.output.multiplier as u64)) * x[0].clone() * g[S].clone()
        - c(inverse_k)
            * (c(pow2(quant.output.shift)) * g[O].clone() + g[ROUT].clone()
                - c(half(quant.output)) * valid);
    eq * local + vector_eq * (c(coefficient) * memory + c(coefficient * alpha) * output)
}

fn group_polynomial<F: PrimeField>(
    layout: &Layout,
    group: usize,
    columns: &[std::sync::Arc<DensePolynomial<'_, F>>],
    d: Dimensions,
) -> DensePolynomial<'static, F> {
    let group_spec = &layout.groups[group];
    let limbs = group_spec
        .columns
        .clone()
        .rev()
        .map(|column| {
            (
                F::from(1u64 << layout.columns[column].bits),
                columns[column].evals_ref(),
            )
        })
        .collect::<Vec<_>>();
    let offset = if group_spec.signed {
        F::from(1u64 << 31)
    } else {
        F::ZERO
    };
    let length = if group_spec.state {
        d.state_len()
    } else {
        d.vector_len()
    };
    let values = (0..length)
        .into_par_iter()
        .map(|index| {
            limbs.iter().fold(F::ZERO, |value, (base, column)| {
                value * base + column[index]
            }) - offset
        })
        .collect::<Vec<_>>();
    if group_spec.state {
        values.into_mle()
    } else {
        // Vector witnesses are independent of K. Reconstruct each value once
        // before broadcasting, preserving exactly the same multilinear table.
        (0..d.state_len())
            .into_par_iter()
            .map(|index| values[d.vector_index(index)])
            .collect::<Vec<_>>()
            .into_mle()
    }
}

fn shift_coefficients<F: PrimeField>(d: Dimensions, point: &[F]) -> DensePolynomial<'static, F> {
    let rest = evals(&point[..d.time_start()]);
    let time = evals(&point[d.time_start()..]);
    (0..d.state_len())
        .into_par_iter()
        .map(|i| {
            let t = i / rest.len();
            if t + 1 < d.valid {
                rest[i % rest.len()] * time[t + 1]
            } else {
                F::ZERO
            }
        })
        .collect::<Vec<_>>()
        .into_mle()
}
fn shift_coefficient_eval<F: PrimeField>(d: Dimensions, original: &[F], target: &[F]) -> F {
    let a = evals(&original[d.time_start()..]);
    let b = evals(&target[d.time_start()..]);
    identity_eval(&original[..d.time_start()], &target[..d.time_start()])
        * (0..d.valid.saturating_sub(1))
            .map(|t| a[t + 1] * b[t])
            .sum::<F>()
}

#[cfg(test)]
mod reconstruction_tests {
    use super::*;
    use ark_ff::AdditiveGroup;

    #[test]
    fn qwen_delta_group_reconstruction_matches_scalar_reference() -> Result<()> {
        type F = ark_bn254::Fr;
        let quarter = FixedScale::new(0.25)?;
        let quant = DeltaQuant {
            decay: quarter,
            memory: quarter,
            value: FixedScale {
                multiplier: 1,
                shift: 0,
            },
            delta: quarter,
            update: FixedScale {
                multiplier: 1,
                shift: 17,
            },
            output: quarter,
            input_domains: vec![(-128, 127); 5],
        };
        let layout = Layout::new(&quant);
        for (rows, h, k, v) in [(8, 2, 4, 2), (2, 1, 128, 4)] {
            let d = Dimensions {
                rows,
                valid: rows - 1,
                h,
                k,
                v,
            };
            let columns = layout
                .columns
                .iter()
                .enumerate()
                .map(|(column, spec)| {
                    let length = if spec.state {
                        d.state_len()
                    } else {
                        d.vector_len()
                    };
                    std::sync::Arc::new(
                        (0..length)
                            .map(|index| {
                                F::from(
                                    ((index * 73 + column * 139) as u64)
                                        & ((1u64 << spec.bits) - 1),
                                )
                            })
                            .collect::<Vec<_>>()
                            .into_mle(),
                    )
                })
                .collect::<Vec<_>>();
            for group in 0..GROUPS {
                let spec = &layout.groups[group];
                let expected = (0..d.state_len())
                    .map(|index| {
                        let index = if spec.state {
                            index
                        } else {
                            d.vector_index(index)
                        };
                        let mut value = F::ZERO;
                        for column in spec.columns.clone().rev() {
                            value = value * F::from(1u64 << layout.columns[column].bits)
                                + columns[column].evals_ref()[index];
                        }
                        value
                            - if spec.signed {
                                F::from(1u64 << 31)
                            } else {
                                F::ZERO
                            }
                    })
                    .collect::<Vec<_>>();
                let actual = group_polynomial(&layout, group, &columns, d);
                assert_eq!(
                    actual.evals_ref(),
                    expected.as_slice(),
                    "group {group}, K={k}"
                );
            }
        }
        Ok(())
    }
}

impl<F: PrimeField, PCS: CommitmentScheme<Field = F>> ProvableOp<F, PCS>
    for GatedDeltaScan<Element>
{
    type Ctx = GatedDeltaScanCtx;
    fn gen_lookup_witness(
        &self,
        id: NodeId,
        _: &ProverContext<F, PCS>,
        step: &Step<Element>,
    ) -> Result<LookupWitnessGen<'_, F>> {
        let inputs = step.padded_input_tensors()?;
        let data = inputs.iter().map(|x| x.data().to_vec()).collect::<Vec<_>>();
        let quant = self.quant.as_ref().context("DeltaNet is not quantized")?;
        quant.validate()?;
        let layout = Layout::new(quant);
        let rows = inputs[0].shape().dim(0);
        let valid = inputs[0].unpadded_shape().dim(0);
        let columns = build::<F>(self, &data, rows, valid)?;
        // Replaying this certificate must match the actual model trace exactly.
        let actual = step
            .output_tensor_at(0)?
            .as_ref()
            .clone()
            .pad_next_power_of_two();
        for (index, &expected) in actual.data().iter().enumerate() {
            let mut values = vec![F::ZERO; columns.len()];
            for column in layout.groups[O].columns.clone() {
                values[column] = columns[column][index];
            }
            ensure!(
                layout.reconstruct(O, &values) == expected.to_field(),
                "DeltaNet trace differs from zero-initialized certified replay"
            );
        }
        let mut counts = BTreeMap::<Table, HashMap<Element, u64>>::new();
        for (spec, column) in layout.columns.iter().zip(&columns) {
            let counts = counts.entry(spec.table()).or_default();
            for value in column {
                *counts
                    .entry(value.into_bigint().as_ref()[0] as i64)
                    .or_default() += 1;
            }
        }
        let (tables, counts) = counts.into_iter().unzip();
        let mut witness = LookupWitnessGen::default();
        witness.insert_layer_witness_data(
            id,
            columns.into_iter().map(IntoMLE::into_mle).collect(),
            tables,
            counts,
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
        ensure!(last.len() == 1, "DeltaNet expects one output claim");
        let inputs = step.padded_input_tensors()?;
        let d = ctx.dimensions(
            &inputs.iter().map(|x| x.shape().clone()).collect::<Vec<_>>(),
            &inputs
                .iter()
                .map(|x| x.unpadded_shape().clone())
                .collect::<Vec<_>>(),
        )?;
        ensure!(
            last[0].point.len() == d.vector_vars(),
            "DeltaNet output claim dimension mismatch"
        );
        let layout = Layout::new(&ctx.quant);
        let committed = prover.lookup_witness(id)?.clone();
        let columns = committed
            .iter()
            .map(|p| p.polynomial.clone())
            .collect::<Vec<_>>();
        ensure!(
            columns.len() == layout.columns.len(),
            "DeltaNet witness column count mismatch"
        );
        let commitments = committed.iter().map(VerifierCommitment::from).collect();
        let mut openings = vec![Vec::new(); columns.len()];
        let mut ranges = Vec::with_capacity(columns.len());
        for (i, (column, spec)) in columns.iter().zip(&layout.columns).enumerate() {
            let (constant, separation) = prover
                .challenge_storage
                .get_challenges_by_name(&spec.table().name())
                .context("Missing DeltaNet range challenges")?;
            let range = new_batch_multiple_sizes_prove(
                &[LogUpInput::new_lookup(
                    vec![column.evals()],
                    constant,
                    separation,
                    1,
                )?],
                prover.transcript,
            )?;
            ensure!(
                range.output_claims().len() == 1,
                "Malformed DeltaNet range proof"
            );
            openings[i].push(range.output_claims()[0].clone());
            ranges.push(range);
        }
        let output_evals = layout.groups[O]
            .columns
            .clone()
            .map(|i| columns[i].evaluate(&last[0].point))
            .collect::<Result<Vec<_>>>()?;
        let mut output_values = vec![F::ZERO; columns.len()];
        for (i, &value) in layout.groups[O].columns.clone().zip(&output_evals) {
            output_values[i] = value;
            openings[i].push(Claim::new(last[0].point.clone(), value));
        }
        ensure!(
            layout.reconstruct(O, &output_values) == last[0].eval,
            "DeltaNet output binding failed"
        );
        prover.transcript.append_scalars(&output_evals);
        let alpha = prover.transcript.challenge_scalar::<F>();
        let point = (0..d.vars())
            .map(|_| prover.transcript.challenge_scalar::<F>())
            .collect::<Vec<_>>();
        let vector_point = (0..d.vector_vars())
            .map(|_| prover.transcript.challenge_scalar::<F>())
            .collect::<Vec<_>>();
        let original_inputs = inputs
            .into_iter()
            .map(|x| x.to_field().into_mle())
            .collect::<Vec<_>>();
        let mut polys = (0..GROUPS)
            .map(|g| group_polynomial(&layout, g, &columns, d))
            .collect::<Vec<_>>();
        for (input, p) in original_inputs.iter().enumerate() {
            polys.push(
                (0..d.state_len())
                    .into_par_iter()
                    .map(|i| {
                        p.evals_ref()[match input {
                            0 | 1 => i / d.v,
                            2 => d.vector_index(i),
                            _ => i / (d.k * d.v),
                        }]
                    })
                    .collect::<Vec<_>>()
                    .into_mle(),
            );
        }
        let stride = d.h * d.k * d.v;
        let previous = (0..d.state_len())
            .into_par_iter()
            .map(|i| {
                if i >= stride && i < d.valid * stride {
                    polys[S].evals_ref()[i - stride]
                } else {
                    F::ZERO
                }
            })
            .collect::<Vec<_>>()
            .into_mle();
        let valid = (0..d.state_len())
            .map(|i| {
                if i < d.valid * stride {
                    F::ONE
                } else {
                    F::ZERO
                }
            })
            .collect::<Vec<_>>()
            .into_mle();
        let eq = evals(&point).into_mle();
        let ve = evals(&vector_point);
        let vector_eq = (0..d.state_len())
            .map(|i| ve[d.vector_index(i)])
            .collect::<Vec<_>>()
            .into_mle();
        let mut builder =
            VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(d.vars()), d.vars());
        let expressions = polys
            .iter()
            .map(|p| builder.lift(Either::Left(p)))
            .collect::<Vec<_>>();
        let previous_expr = builder.lift(Either::Left(&previous));
        let valid_expr = builder.lift(Either::Left(&valid));
        let eq_expr = builder.lift(Either::Left(&eq));
        let vector_eq_expr = builder.lift(Either::Left(&vector_eq));
        let expression = relation(
            &expressions[..GROUPS],
            &expressions[GROUPS..],
            previous_expr,
            valid_expr,
            eq_expr,
            vector_eq_expr,
            &ctx.quant,
            d.k,
            alpha,
            Expression::Constant,
        );
        let (arithmetic, state) = IOPProverState::prove(
            builder.to_virtual_polys(&[expression], &[]),
            prover.transcript,
        );
        ensure!(
            arithmetic.extract_sum() == F::ZERO,
            "DeltaNet witness violates integer recurrence equations"
        );
        let arithmetic_point = state.collect_raw_challenges();
        drop(state);
        let arithmetic_evals = columns
            .iter()
            .zip(&layout.columns)
            .map(|(p, s)| p.evaluate(&d.column_point(&arithmetic_point, s.state)))
            .collect::<Result<Vec<_>>>()?;
        let input_evals = original_inputs
            .iter()
            .enumerate()
            .map(|(i, p)| p.evaluate(&d.input_point(&arithmetic_point, i)))
            .collect::<Result<Vec<_>>>()?;
        let previous_eval = previous.evaluate(&arithmetic_point)?;
        for (i, &value) in arithmetic_evals.iter().enumerate() {
            openings[i].push(Claim::new(
                d.column_point(&arithmetic_point, layout.columns[i].state),
                value,
            ));
        }
        prover.transcript.append_scalars(&arithmetic_evals);
        prover.transcript.append_scalars(&input_evals);
        prover.transcript.append_scalars(&[previous_eval]);
        let state_poly = polys.remove(S);
        drop(polys);
        drop(previous);
        drop(valid);
        drop(eq);
        drop(vector_eq);
        let coefficients = shift_coefficients(d, &arithmetic_point);
        let mut builder =
            VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(d.vars()), d.vars());
        let s = builder.lift(Either::Left(&state_poly));
        let coefficients = builder.lift(Either::Left(&coefficients));
        let (shift, state) = IOPProverState::prove(
            builder.to_virtual_polys(&[s * coefficients], &[]),
            prover.transcript,
        );
        ensure!(
            shift.extract_sum() == previous_eval,
            "DeltaNet witness violates causal state shift"
        );
        let shift_point = state.collect_raw_challenges();
        let shift_evals = layout.groups[S]
            .columns
            .clone()
            .map(|i| columns[i].evaluate(&shift_point))
            .collect::<Result<Vec<_>>>()?;
        for (i, &value) in layout.groups[S].columns.clone().zip(&shift_evals) {
            openings[i].push(Claim::new(shift_point.clone(), value));
        }
        prover.transcript.append_scalars(&shift_evals);
        let claims = input_evals
            .iter()
            .enumerate()
            .map(|(i, &value)| Claim::new(d.input_point(&arithmetic_point, i), value))
            .collect();
        prover.add_witness_claim(id, openings);
        prover.push_proof(
            id,
            LayerProof::GatedDeltaScan(GatedDeltaScanProof {
                commitments,
                ranges,
                output_evals,
                arithmetic,
                arithmetic_evals,
                input_evals,
                previous_eval,
                shift,
                shift_evals,
            }),
        );
        Ok(claims)
    }
}

impl<F: PrimeField, PCS: CommitmentScheme<Field = F>> VerifiableCtx<F, PCS> for GatedDeltaScanCtx {
    type Proof = GatedDeltaScanProof<F, PCS>;
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
            "DeltaNet proof requires a large prime field"
        );
        let d = self.dimensions(&shape.padded_input_shape, &shape.unpadded_input_shape)?;
        let layout = Layout::new(&self.quant);
        let n = layout.columns.len();
        ensure!(
            last.len() == 1 && last[0].point.len() == d.vector_vars(),
            "DeltaNet output claim mismatch"
        );
        ensure!(
            proof.commitments.len() == n
                && proof.ranges.len() == n
                && proof.arithmetic_evals.len() == n
                && proof.input_evals.len() == 5
                && proof.output_evals.len() == layout.groups[O].columns.len()
                && proof.shift_evals.len() == layout.groups[S].columns.len(),
            "Malformed DeltaNet proof"
        );
        let mut openings = vec![Vec::new(); n];
        for (i, spec) in layout.columns.iter().enumerate() {
            let vars = if spec.state {
                d.vars()
            } else {
                d.vector_vars()
            };
            ensure!(
                proof.commitments[i].num_vars() == vars,
                "DeltaNet witness commitment dimension mismatch"
            );
            let table = spec.table();
            let (constant, separation) = verifier
                .challenge_storage
                .get_challenges_by_name(&table.name())
                .context("Missing DeltaNet range table challenges")?;
            let range = &proof.ranges[i];
            ensure!(
                range.num_instances() == 1 && range.circuit_outputs().iter().all(|x| x.len() == 4),
                "Malformed DeltaNet range proof"
            );
            let instance =
                LogUpVerifierInstance::new(constant, separation, 1, ProofType::Lookup, vars - 1);
            let batch =
                new_verify_logup_proof_multiple_sizes(range, &[instance], verifier.transcript)?;
            ensure!(
                batch.output_claims().len() == 1,
                "DeltaNet range opening count mismatch"
            );
            openings[i].push(batch.output_claims()[0].clone());
            let (numerators, denominators) = range.fractional_outputs();
            let (num, den) = verifier
                .numerators_and_denominators
                .entry(table.name())
                .or_insert((F::ZERO, F::ONE));
            for (n, d) in numerators.iter().zip(&denominators) {
                ensure!(*d != F::ZERO, "Zero DeltaNet lookup denominator");
                *num = *n * *den + *num * *d;
                *den *= d;
            }
        }
        let mut values = vec![F::ZERO; n];
        for (i, &value) in layout.groups[O].columns.clone().zip(&proof.output_evals) {
            values[i] = value;
            openings[i].push(Claim::new(last[0].point.clone(), value));
        }
        ensure!(
            layout.reconstruct(O, &values) == last[0].eval,
            "DeltaNet output binding failed"
        );
        verifier.transcript.append_scalars(&proof.output_evals);
        let alpha = verifier.transcript.challenge_scalar::<F>();
        let point = (0..d.vars())
            .map(|_| verifier.transcript.challenge_scalar::<F>())
            .collect::<Vec<_>>();
        let vector_point = (0..d.vector_vars())
            .map(|_| verifier.transcript.challenge_scalar::<F>())
            .collect::<Vec<_>>();
        let sub = IOPVerifierState::verify(
            F::ZERO,
            &proof.arithmetic,
            &from_mle_list_dimensions(&[vec![d.vars(); 3]]),
            verifier.transcript,
        );
        let groups = (0..GROUPS)
            .map(|g| layout.reconstruct(g, &proof.arithmetic_evals))
            .collect::<Vec<_>>();
        let expected = relation(
            &groups,
            &proof.input_evals,
            proof.previous_eval,
            d.valid_eval(&sub.point),
            identity_eval(&point, &sub.point),
            identity_eval(&vector_point, &d.vector_point(&sub.point)),
            &self.quant,
            d.k,
            alpha,
            |x| x,
        );
        ensure!(
            sub.expected_evaluation == expected,
            "DeltaNet integer recurrence constraints failed"
        );
        for (i, &value) in proof.arithmetic_evals.iter().enumerate() {
            openings[i].push(Claim::new(
                d.column_point(&sub.point, layout.columns[i].state),
                value,
            ));
        }
        verifier.transcript.append_scalars(&proof.arithmetic_evals);
        verifier.transcript.append_scalars(&proof.input_evals);
        verifier.transcript.append_scalars(&[proof.previous_eval]);
        let claims = proof
            .input_evals
            .iter()
            .enumerate()
            .map(|(i, &value)| Claim::new(d.input_point(&sub.point, i), value))
            .collect();
        let shifted = IOPVerifierState::verify(
            proof.previous_eval,
            &proof.shift,
            &from_mle_list_dimensions(&[vec![d.vars(); 2]]),
            verifier.transcript,
        );
        values.fill(F::ZERO);
        for (i, &value) in layout.groups[S].columns.clone().zip(&proof.shift_evals) {
            values[i] = value;
            openings[i].push(Claim::new(shifted.point.clone(), value));
        }
        ensure!(
            shifted.expected_evaluation
                == layout.reconstruct(S, &values)
                    * shift_coefficient_eval(d, &sub.point, &shifted.point),
            "DeltaNet causal shift/initial state binding failed"
        );
        verifier.transcript.append_scalars(&proof.shift_evals);
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
        Ok(claims)
    }
    fn write_proof_to_transcript<T: Transcript>(
        &self,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<()> {
        ensure!(
            proof.commitments.len() == Layout::new(&self.quant).columns.len(),
            "Malformed DeltaNet commitments"
        );
        for commitment in &proof.commitments {
            commitment.append_to_transcript(transcript);
        }
        Ok(())
    }
}
