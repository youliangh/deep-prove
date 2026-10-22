//! Version two: bind stacked axes inside the sumcheck, not by an unbound list
//! of per-stack evaluations. Outer axes remain fixed before the sumcheck.
use super::{axis::Axis, prove::EinSumProofInfo, verify::EinSumVerifierInfo, *};
use crate::{
    Tensor, poly_commit::identity_eval, quantization::ToField, util::from_mle_list_dimensions,
};
use anyhow::{Context, bail};
use dp_crypto::{
    IntoMLE,
    poly::{dense::DensePolynomial, eq::evals},
    structs::{IOPProverState, IOPVerifierState},
    util::optimal_sumcheck_threads,
    virtual_polys::VirtualPolynomialsBuilder,
};
use either::Either;
use rayon::prelude::*;
use std::{
    borrow::Cow,
    collections::{HashMap, HashSet},
    ops::Range,
};

struct Plan {
    axes: Vec<Axis>,
    ranges: Vec<Option<Range<usize>>>,
    strides: Vec<usize>,
    sizes: Vec<usize>,
    variables: usize,
}
impl Plan {
    fn new(mapping: &AxesMapping, shapes: &[Shape]) -> Result<Self> {
        ensure!(
            shapes.len() == mapping.input_count() && shapes.iter().all(Shape::is_power_of_two),
            "Invalid padded EinSumV2 input shapes"
        );
        let axes = mapping.axes().cloned().collect::<Vec<_>>();
        let mut keep = axes
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                if let Dimension::Present(position) = a.lhs_input {
                    // A shared LHS axis, or one reduced by any RHS operation, must
                    // remain a sumcheck variable for every operation in the batch.
                    (a.rhs_inputs
                        .iter()
                        .any(|d| matches!(d, Dimension::Present(_)))
                        || a.outputs.iter().any(|d| matches!(d, Dimension::Absent)))
                    .then_some((position, i))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        keep.sort_unstable();
        let mut ranges = vec![None; axes.len()];
        let mut strides = vec![0; axes.len()];
        let mut sizes = vec![0; axes.len()];
        let mut variables = 0usize;
        for &(position, i) in keep.iter().rev() {
            let size = shapes[0].dim(position);
            let bits = size.ilog2() as usize;
            ranges[i] = Some(variables..variables + bits);
            strides[i] = 1 << variables;
            sizes[i] = size;
            variables += bits;
        }
        for (i, axis) in axes.iter().enumerate() {
            if ranges[i].is_some() {
                for (input, dimension) in axis.inputs().enumerate() {
                    if let Dimension::Present(position) = dimension {
                        ensure!(
                            shapes[input].dim(*position) == sizes[i],
                            "EinSumV2 shared axis size mismatch"
                        );
                    }
                }
            }
        }
        Ok(Self {
            axes,
            ranges,
            strides,
            sizes,
            variables,
        })
    }
    fn proof_variables(&self) -> usize {
        self.variables.max(1)
    }
    fn input_position(axis: &Axis, input: usize) -> Dimension {
        if input == 0 {
            axis.lhs_input
        } else {
            axis.rhs_inputs[input - 1]
        }
    }
    fn input_axes(&self, input: usize, rank: usize) -> Result<Vec<usize>> {
        let mut axes = vec![usize::MAX; rank];
        for (index, axis) in self.axes.iter().enumerate() {
            if let Dimension::Present(position) = Self::input_position(axis, input) {
                ensure!(
                    position < rank && axes[position] == usize::MAX,
                    "Malformed EinSumV2 tensor axes"
                );
                axes[position] = index;
            }
        }
        ensure!(
            axes.iter().all(|i| *i != usize::MAX),
            "Missing EinSumV2 tensor axis"
        );
        Ok(axes)
    }
    fn output_axis_point<'a, F>(
        &self,
        axis: usize,
        output: usize,
        point: &[&'a [F]],
    ) -> Result<&'a [F]> {
        match self.axes[axis].outputs[output] {
            Dimension::Present(position) => point
                .get(position)
                .copied()
                .context("Missing EinSumV2 output axis point"),
            Dimension::Absent => bail!("EinSumV2 cannot fix a reduced outer axis"),
        }
    }
    fn input_point<F: Clone>(
        &self,
        input: usize,
        output: usize,
        shape: &Shape,
        output_point: &[&[F]],
        sample: &[F],
    ) -> Result<Vec<F>> {
        let axes = self.input_axes(input, shape.rank())?;
        axes.iter().rev().try_fold(Vec::new(), |mut point, &axis| {
            let part = if let Some(range) = &self.ranges[axis] {
                &sample[range.clone()]
            } else {
                self.output_axis_point(axis, output, output_point)?
            };
            point.extend_from_slice(part);
            Ok(point)
        })
    }
    fn coefficient<F: PrimeField>(&self, output: usize, point: &[&[F]], sample: &[F]) -> F {
        if self.variables == 0 {
            return F::ONE - sample[0];
        }
        self.axes
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                if let (Some(range), Dimension::Present(position)) =
                    (&self.ranges[i], a.outputs[output])
                {
                    Some(identity_eval(point[position], &sample[range.clone()]))
                } else {
                    None
                }
            })
            .product()
    }
    fn coefficient_poly<F: PrimeField>(
        &self,
        output: usize,
        point: &[&[F]],
    ) -> DensePolynomial<'static, F> {
        if self.variables == 0 {
            return vec![F::ONE, F::ZERO].into_mle();
        }
        let factors = self
            .axes
            .iter()
            .enumerate()
            .filter_map(|(i, a)| {
                if self.ranges[i].is_some() {
                    if let Dimension::Present(position) = a.outputs[output] {
                        return Some((i, evals(point[position])));
                    }
                }
                None
            })
            .collect::<Vec<_>>();
        (0..1usize << self.variables)
            .into_par_iter()
            .map(|index| {
                factors
                    .iter()
                    .map(|(axis, beta)| beta[index / self.strides[*axis] % self.sizes[*axis]])
                    .product()
            })
            .collect::<Vec<F>>()
            .into_mle()
    }
    fn fix_outer<F: PrimeField>(
        &self,
        input: usize,
        output: usize,
        tensor: &Tensor<F>,
        point: &[&[F]],
    ) -> Result<DensePolynomial<'static, F>> {
        let axes = self.input_axes(input, tensor.shape().rank())?;
        let mut data: Cow<'_, [F]> = Cow::Borrowed(tensor.data());
        let mut stride = 1;
        for (position, &axis) in axes.iter().enumerate().rev() {
            let size = tensor.shape().dim(position);
            if self.ranges[axis].is_some() {
                stride *= size;
                continue;
            }
            let beta = evals(self.output_axis_point(axis, output, point)?);
            ensure!(beta.len() == size, "EinSumV2 fixed axis size mismatch");
            data = Cow::Owned(
                data.par_chunks(size * stride)
                    .flat_map_iter(|chunk| {
                        (0..stride)
                            .map(|j| (0..size).map(|i| chunk[i * stride + j] * beta[i]).sum())
                    })
                    .collect(),
            );
        }
        // Rearrange retained axes to one canonical LHS order, broadcasting any
        // retained axis that is absent from this particular RHS tensor.
        let mut input_strides = vec![0; self.axes.len()];
        let mut size = 1;
        for &axis in axes.iter().rev().filter(|&&a| self.ranges[a].is_some()) {
            input_strides[axis] = size;
            size *= self.sizes[axis];
        }
        ensure!(
            data.len() == size,
            "EinSumV2 partial evaluation shape mismatch"
        );
        let length = 1usize << self.variables;
        let values = (0..length)
            .into_par_iter()
            .map(|i| {
                let source = input_strides
                    .iter()
                    .enumerate()
                    .filter(|(_, stride)| **stride > 0)
                    .map(|(a, stride)| (i / self.strides[a] % self.sizes[a]) * stride)
                    .sum::<usize>();
                data[source]
            })
            .collect::<Vec<_>>();
        if self.variables == 0 {
            Ok(vec![values[0]; 2].into_mle())
        } else {
            Ok(values.into_mle())
        }
    }
}

fn unique_constants<F: PrimeField>(ctx: &EinSumContext<F>) -> Result<()> {
    let mut keys = HashSet::new();
    ensure!(
        ctx.constant_keys
            .iter()
            .chain(&ctx.bias_keys)
            .flatten()
            .all(|key| keys.insert(key)),
        "EinSumV2 requires distinct constant/bias keys within one node"
    );
    Ok(())
}

impl EinSum<Element> {
    pub(crate) fn prove_bound<F: PrimeField, T: Transcript>(
        &self,
        ctx: &EinSumContext<F>,
        last: Vec<&Claim<F>>,
        inputs: &[Tensor<Element>],
        transcript: &mut T,
    ) -> Result<EinSumProofInfo<F>> {
        unique_constants(ctx)?;
        let outputs = self.mapping.output_count();
        ensure!(
            last.len() == outputs
                && inputs.len() == 1 + self.constant_tensors.iter().filter(|x| x.is_none()).count(),
            "EinSumV2 input/output arity mismatch"
        );
        let mut runtime = inputs.iter();
        let mut full = vec![runtime.next().context("Missing EinSumV2 input")?.to_field()];
        for tensor in &self.constant_tensors {
            full.push(if let Some(tensor) = tensor {
                Tensor::try_from(tensor)?.pad_next_power_of_two().to_field()
            } else {
                runtime
                    .next()
                    .context("Missing EinSumV2 runtime RHS")?
                    .to_field()
            });
        }
        let shapes = full.iter().map(|x| x.shape().clone()).collect::<Vec<_>>();
        let unpadded = full
            .iter()
            .map(|x| x.unpadded_shape().clone())
            .collect::<Vec<_>>();
        let output_shapes = self.mapping.output_shapes(&shapes)?;
        let unpadded_outputs = self.mapping.output_shapes(&unpadded)?;
        let split = output_shapes
            .iter()
            .zip(&last)
            .map(|(s, c)| s.split_point(&c.point))
            .collect::<Result<Vec<_>>>()?;
        let plan = Plan::new(&self.mapping, &shapes)?;
        let mut constants = HashMap::new();
        let mut bias_evals = Vec::new();
        let mut claimed = last.iter().map(|c| c.eval).collect::<Vec<_>>();
        let mut bias_id = 0;
        for (output, bias) in self.biases.iter().enumerate() {
            if let Some(bias) = bias {
                let point = self
                    .mapping
                    .bias_evaluation_point(output, bias_id, &split[output])?;
                let value = Tensor::try_from(bias)?
                    .pad_next_power_of_two()
                    .to_field_mle()
                    .evaluate(&point)?;
                let (broadcast, claim) = self.mapping.compute_bias_evaluation(
                    output,
                    bias_id,
                    &split[output],
                    value,
                    &unpadded_outputs[output],
                )?;
                claimed[output] -= broadcast;
                bias_evals.push(value);
                constants.insert(CommitmentId::from(bias.storage_key()), claim);
                bias_id += 1;
            }
        }
        transcript.append_scalars(&bias_evals);
        let alpha = transcript.append_and_sample::<F>(b"bound_einsum_batch_v2");
        let lhs = (0..outputs)
            .map(|i| plan.fix_outer(0, i, &full[0], &split[i]))
            .collect::<Result<Vec<_>>>()?;
        let rhs = (0..outputs)
            .map(|i| plan.fix_outer(i + 1, i, &full[i + 1], &split[i]))
            .collect::<Result<Vec<_>>>()?;
        let coefficients = (0..outputs)
            .map(|i| plan.coefficient_poly(i, &split[i]))
            .collect::<Vec<_>>();
        let vars = plan.proof_variables();
        let mut builder = VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(vars), vars);
        let mut expression = Expression::ZERO;
        let mut power = F::ONE;
        for i in 0..outputs {
            let a = builder.lift(Either::Left(&lhs[i]));
            let b = builder.lift(Either::Left(&rhs[i]));
            let eq = builder.lift(Either::Left(&coefficients[i]));
            expression = expression + Expression::Constant(power) * eq * a * b;
            power *= alpha;
        }
        let (einsum_sumcheck, state) =
            IOPProverState::prove(builder.to_virtual_polys(&[expression], &[]), transcript);
        let expected = claimed.iter().rev().fold(F::ZERO, |sum, x| sum * alpha + x);
        ensure!(
            einsum_sumcheck.extract_sum() == expected,
            "EinSumV2 integer trace relation failed"
        );
        let point = state.collect_raw_challenges();
        let einsum_evaluations = lhs
            .iter()
            .chain(&rhs)
            .map(|p| p.evaluate(&point))
            .collect::<Result<Vec<_>>>()?;
        transcript.append_scalars(&einsum_evaluations);
        let lhs_points = (0..outputs)
            .map(|i| plan.input_point(0, i, &shapes[0], &split[i], &point))
            .collect::<Result<Vec<_>>>()?;
        let mut claims = Vec::new();
        for i in 0..outputs {
            let point = plan.input_point(i + 1, i, &shapes[i + 1], &split[i], &point)?;
            let claim = Claim::new(point, einsum_evaluations[outputs + i]);
            if let Some(key) = &ctx.constant_keys[i] {
                constants.insert(key.clone(), claim);
            } else {
                claims.push(claim);
            }
        }
        let (input_aggregation_sumcheck, lhs_claim) = if outputs > 1 && !lhs_points[0].is_empty() {
            let input = full[0].clone().into_mle();
            let vars = input.num_vars();
            let beta = transcript.append_and_sample::<F>(b"bound_einsum_lhs_v2");
            let eqs = lhs_points
                .iter()
                .map(|p| evals(p).into_mle())
                .collect::<Vec<_>>();
            let mut builder = VirtualPolynomialsBuilder::new(optimal_sumcheck_threads(vars), vars);
            let x = builder.lift(Either::Left(&input));
            let mut eq = Expression::ZERO;
            let mut power = F::ONE;
            for p in &eqs {
                eq = eq + Expression::Constant(power) * builder.lift(Either::Left(p));
                power *= beta;
            }
            let (proof, state) =
                IOPProverState::prove(builder.to_virtual_polys(&[x * eq], &[]), transcript);
            let expected = einsum_evaluations[..outputs]
                .iter()
                .rev()
                .fold(F::ZERO, |sum, x| sum * beta + x);
            ensure!(
                proof.extract_sum() == expected,
                "EinSumV2 LHS aggregation failed"
            );
            let point = state.collect_raw_challenges();
            let claim = Claim::new(point.clone(), input.evaluate(&point)?);
            (Some(proof), claim)
        } else {
            ensure!(
                einsum_evaluations[..outputs]
                    .iter()
                    .all(|x| outputs == 1 || *x == einsum_evaluations[0]),
                "Inconsistent EinSumV2 scalar LHS"
            );
            (
                None,
                Claim::new(lhs_points[0].clone(), einsum_evaluations[0]),
            )
        };
        claims.insert(0, lhs_claim);
        Ok(EinSumProofInfo {
            claims,
            proof: EinSumProof {
                bias_evals,
                einsum_sumcheck,
                einsum_evaluations,
                input_aggregation_sumcheck,
            },
            commitment_map: constants,
        })
    }
}

impl<F: PrimeField> EinSumContext<F> {
    pub(crate) fn verify_bound<T: Transcript>(
        &self,
        proof: &EinSumProof<F>,
        last: &[&Claim<F>],
        shape: &ShapeStep,
        transcript: &mut T,
    ) -> Result<EinSumVerifierInfo<F>> {
        unique_constants(self)?;
        let outputs = self.mapping.output_count();
        ensure!(
            last.len() == outputs
                && proof.einsum_evaluations.len() == 2 * outputs
                && proof.bias_evals.len() == self.bias_keys.iter().flatten().count(),
            "Malformed EinSumV2 proof"
        );
        let mut padded = shape.padded_input_shape.iter();
        let mut unpadded = shape.unpadded_input_shape.iter();
        let mut shapes = vec![
            padded
                .next()
                .context("Missing EinSumV2 input shape")?
                .clone(),
        ];
        let mut unpadded_shapes = vec![
            unpadded
                .next()
                .context("Missing EinSumV2 input shape")?
                .clone(),
        ];
        for constant in &self.constant_unpadded_shapes {
            if let Some(shape) = constant {
                shapes.push(shape.next_power_of_two());
                unpadded_shapes.push(shape.clone());
            } else {
                shapes.push(padded.next().context("Missing EinSumV2 RHS shape")?.clone());
                unpadded_shapes.push(
                    unpadded
                        .next()
                        .context("Missing EinSumV2 RHS shape")?
                        .clone(),
                );
            }
        }
        ensure!(
            padded.next().is_none() && unpadded.next().is_none(),
            "Extra EinSumV2 input shapes"
        );
        let output_shapes = self.mapping.output_shapes(&shapes)?;
        let unpadded_outputs = self.mapping.output_shapes(&unpadded_shapes)?;
        let split = output_shapes
            .iter()
            .zip(last)
            .map(|(s, c)| s.split_point(&c.point))
            .collect::<Result<Vec<_>>>()?;
        let plan = Plan::new(&self.mapping, &shapes)?;
        let mut constants = HashMap::new();
        let mut claimed = last.iter().map(|c| c.eval).collect::<Vec<_>>();
        let mut bias_id = 0;
        for (i, key) in self.bias_keys.iter().enumerate() {
            if let Some(key) = key {
                let (broadcast, claim) = self.mapping.compute_bias_evaluation(
                    i,
                    bias_id,
                    &split[i],
                    proof.bias_evals[bias_id],
                    &unpadded_outputs[i],
                )?;
                claimed[i] -= broadcast;
                constants.insert(key.clone(), claim);
                bias_id += 1;
            }
        }
        transcript.append_scalars(&proof.bias_evals);
        let alpha = transcript.append_and_sample::<F>(b"bound_einsum_batch_v2");
        let expected = claimed.iter().rev().fold(F::ZERO, |sum, x| sum * alpha + x);
        let sub = IOPVerifierState::verify(
            expected,
            &proof.einsum_sumcheck,
            &from_mle_list_dimensions(&[vec![plan.proof_variables(); 3]]),
            transcript,
        );
        let calculated = (0..outputs).rev().fold(F::ZERO, |sum, i| {
            sum * alpha
                + plan.coefficient(i, &split[i], &sub.point)
                    * proof.einsum_evaluations[i]
                    * proof.einsum_evaluations[outputs + i]
        });
        ensure!(
            sub.expected_evaluation == calculated,
            "EinSumV2 stacked product binding failed"
        );
        transcript.append_scalars(&proof.einsum_evaluations);
        let lhs_points = (0..outputs)
            .map(|i| plan.input_point(0, i, &shapes[0], &split[i], &sub.point))
            .collect::<Result<Vec<_>>>()?;
        let mut claims = Vec::new();
        for i in 0..outputs {
            let point = plan.input_point(i + 1, i, &shapes[i + 1], &split[i], &sub.point)?;
            let claim = Claim::new(point, proof.einsum_evaluations[outputs + i]);
            if let Some(key) = &self.constant_keys[i] {
                constants.insert(key.clone(), claim);
            } else {
                claims.push(claim);
            }
        }
        let lhs = if outputs > 1 && !lhs_points[0].is_empty() {
            let aggregation = proof
                .input_aggregation_sumcheck
                .as_ref()
                .context("Missing EinSumV2 LHS binding")?;
            let beta = transcript.append_and_sample::<F>(b"bound_einsum_lhs_v2");
            let expected = proof.einsum_evaluations[..outputs]
                .iter()
                .rev()
                .fold(F::ZERO, |sum, x| sum * beta + x);
            let sub = IOPVerifierState::verify(
                expected,
                aggregation,
                &from_mle_list_dimensions(&[vec![lhs_points[0].len(); 2]]),
                transcript,
            );
            let eq = lhs_points
                .iter()
                .rev()
                .fold(F::ZERO, |sum, p| sum * beta + identity_eval(p, &sub.point));
            let inverse = eq
                .inverse()
                .context("Zero EinSumV2 LHS aggregation coefficient")?;
            Claim::new(sub.point, sub.expected_evaluation * inverse)
        } else {
            ensure!(
                proof.input_aggregation_sumcheck.is_none(),
                "Unexpected EinSumV2 LHS binding proof"
            );
            ensure!(
                proof.einsum_evaluations[..outputs]
                    .iter()
                    .all(|x| outputs == 1 || *x == proof.einsum_evaluations[0]),
                "Inconsistent EinSumV2 scalar LHS"
            );
            Claim::new(lhs_points[0].clone(), proof.einsum_evaluations[0])
        };
        claims.insert(0, lhs);
        Ok(EinSumVerifierInfo {
            claims,
            constants_map: constants,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{layers::Layer, model::Model, padding::pad_model, testing::Pcs};
    use ark_ff::Field;
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    use tenstore::{GenStore, StorageKey};
    type F = ark_bn254::Fr;

    fn check(equation: &str, shapes: Vec<Vec<usize>>, tamper: bool) -> Result<()> {
        type P<'a, 'b> = crate::Prover<'a, 'b, F, Blake3Transcript, Pcs>;
        let shapes = shapes.into_iter().map(Shape::from).collect::<Vec<_>>();
        let inputs = shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                Tensor::new(
                    shape.clone(),
                    (0..shape.product())
                        .map(|j| ((j * 3 + i * 5) % 11) as i64 - 4)
                        .collect(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        // Include genuine committed biases. An all-runtime graph has no PCS
        // claims and currently hits the generic empty-opening-batch edge case.
        // Biases also exercise zero padding and broadcast masks independently
        // of the product-preserving input-evaluation attack below.
        let mapping = AxesMapping::from_string(equation)?;
        let output_shapes = mapping.output_shapes(&shapes)?;
        let (left, right) = equation.split_once("->").unwrap();
        let mut biases = Vec::new();
        let outputs = right
            .split(':')
            .enumerate()
            .map(|(i, output)| {
                let dimension = output_shapes[i].rank() - 1;
                let axis = mapping
                    .axes()
                    .find(|axis| matches!(axis.outputs[i], Dimension::Present(d) if d == dimension))
                    .unwrap();
                let size = output_shapes[i].dim(dimension);
                let bias = Tensor::new(
                    vec![size].into(),
                    (0..size).map(|j| j as i64 % 7 - 2).collect(),
                )?;
                biases.push(Some(TensorHandle::from_tensor(
                    StorageKey::from(format!("bound_einsum_bias_{i}")),
                    GenStore::new_empty(),
                    bias,
                )));
                Ok(format!("{output}+BIAS({})", axis.repr))
            })
            .collect::<Result<Vec<_>>>()?;
        let layer = EinSum::new(
            format!("{left}->{}", outputs.join(":")),
            vec![None; shapes.len() - 1],
            biases,
        )?
        .no_requant();
        let mut model = Model::new_from_input_shapes(shapes);
        model.add_consecutive_layer(Layer::EinSum(layer), None)?;
        model.automatic_output_labelling()?;
        let model = pad_model(model)?;
        let inputs = model.prepare_inputs(inputs)?;
        let trace = model.run(inputs, &mut Default::default())?;
        let (prover, verifier) = model.generate_contexts::<F, Pcs>()?;
        let (proof, io) = P::prove(&prover, trace, &model)?;
        crate::verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), io.clone())?;
        if tamper {
            let mut encoded = serde_json::to_value(proof)?;
            let step = encoded["chunk_proofs"][0]["steps"]
                .as_object_mut()
                .unwrap()
                .values_mut()
                .find_map(|x| x.get_mut("EinSum"))
                .unwrap();
            let mut local: EinSumProof<F> = serde_json::from_value(step.clone())?;
            let outputs = local.einsum_evaluations.len() / 2;
            // Preserve the terminal product exactly. A verifier that only
            // checked the product, without binding both inputs, would accept.
            local.einsum_evaluations[0] *= F::from(2u64);
            local.einsum_evaluations[outputs] *= F::from(2u64).inverse().unwrap();
            *step = serde_json::to_value(local)?;
            let corrupted = serde_json::from_value(encoded)?;
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::verify::<_, Blake3Transcript, _>(&verifier, corrupted, io)
            }));
            ensure!(
                !matches!(rejected, Ok(Ok(()))),
                "Accepted product-preserving fake EinSumV2 input evaluations"
            );
        }
        Ok(())
    }
    #[test]
    fn qwen_einsum_v2_stacked_permuted_and_multi_output_binding() -> Result<()> {
        check(
            "X(abc)@Y(cbd)->O(abd)",
            vec![vec![2, 3, 5], vec![5, 3, 4]],
            true,
        )?;
        check(
            "X(abc)@Y(cbd):Z(cbe)->O(abd):P(abe)",
            vec![vec![2, 3, 5], vec![5, 3, 4], vec![5, 3, 2]],
            true,
        )?;
        Ok(())
    }
    #[test]
    fn qwen_einsum_v2_outer_product_and_unit_contraction() -> Result<()> {
        check("X(a)@Y(b)->O(ab)", vec![vec![3], vec![5]], true)?;
        check("X(ai)@Y(ib)->O(ab)", vec![vec![3, 1], vec![1, 5]], true)?;
        Ok(())
    }
    #[test]
    fn qwen_einsum_v2_pointwise_exceeds_legacy_witness_index_limit() -> Result<()> {
        // The legacy enumeration required 65536 separate u16 witness ids.
        // V2 has just lhs, rhs and the equality polynomial regardless of size.
        check("X(i)@Y(i)->O(i)", vec![vec![32768], vec![32768]], true)
    }
}
