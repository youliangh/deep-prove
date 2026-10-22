use super::*;
use crate::{layers::Layer, model::Model, padding::pad_model, testing::Pcs};
use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;

fn integer_layer() -> GatedDeltaScan<Element> {
    let mut layer = GatedDeltaScan::new(2, 2, 2);
    let quarter = FixedScale::new(0.25).unwrap();
    layer.quant = Some(DeltaQuant {
        decay: quarter,
        memory: quarter,
        value: FixedScale {
            multiplier: 1,
            shift: 0,
        },
        delta: quarter,
        update: FixedScale {
            multiplier: 1 << 15,
            shift: 17,
        },
        output: quarter,
        input_domains: vec![(-128, 127); 5],
    });
    layer
}
fn inputs(rows: usize) -> Vec<Tensor<Element>> {
    [4, 4, 4, 2, 2]
        .into_iter()
        .enumerate()
        .map(|(input, width)| {
            let shape = if input < 3 {
                vec![rows, 2, 2]
            } else {
                vec![rows, 2]
            };
            let values = (0..rows * width)
                .map(|i| match input {
                    0 => [2, -3, 1, 4][i % 4],
                    1 => [1, 2, -2, 1][i % 4],
                    2 => [8, -12, 20, 4, 16, 8, -4, 12][i % 8],
                    3 => 3,
                    _ => 2,
                })
                .collect();
            Tensor::new(shape.into(), values).unwrap()
        })
        .collect()
}
fn oracle(layer: &GatedDeltaScan<Element>, inputs: &[Tensor<Element>]) -> Vec<Element> {
    let scale = |x: i128, s: FixedScale| -> i64 {
        ((x * s.multiplier as i128
            + if s.shift == 0 {
                0
            } else {
                1i128 << (s.shift - 1)
            })
        .div_euclid(1i128 << s.shift)) as i64
    };
    let quant = layer.quant.as_ref().unwrap();
    let mut result = vec![0; inputs[2].shape().numel()];
    let mut s = vec![0i64; 8];
    for t in 0..inputs[0].shape().dim(0) {
        for h in 0..2 {
            let qi = &inputs[0].data()[(t * 2 + h) * 2..];
            let ki = &inputs[1].data()[(t * 2 + h) * 2..];
            let vi = &inputs[2].data()[(t * 2 + h) * 2..];
            let alpha = inputs[3].data()[t * 2 + h];
            let beta = inputs[4].data()[t * 2 + h];
            let state = &mut s[h * 4..][..4];
            for value in state.iter_mut() {
                *value = scale(*value as i128 * alpha as i128, quant.decay);
            }
            let mut delta = [0i64; 2];
            for j in 0..2 {
                let memory = (0..2)
                    .map(|i| ki[i] as i128 * state[i * 2 + j] as i128)
                    .sum();
                delta[j] = scale(
                    (scale(vi[j] as i128, quant.value) as i128
                        - scale(memory, quant.memory) as i128)
                        * beta as i128,
                    quant.delta,
                );
            }
            for i in 0..2 {
                for j in 0..2 {
                    state[i * 2 + j] += scale(ki[i] as i128 * delta[j] as i128, quant.update);
                }
            }
            for j in 0..2 {
                result[(t * 2 + h) * 2 + j] = scale(
                    (0..2)
                        .map(|i| qi[i] as i128 * state[i * 2 + j] as i128)
                        .sum(),
                    quant.output,
                );
            }
        }
    }
    result
}

#[test]
fn qwen_delta_signed_rounding_ties_and_checked_bounds() -> Result<()> {
    let scale = FixedScale {
        multiplier: 1,
        shift: 2,
    };
    for x in -40..40 {
        let (rounded, remainder) = scale.apply_with_remainder(x)?;
        assert_eq!(x + 2, rounded as i128 * 4 + remainder as i128);
        assert!(remainder < 4);
        assert_eq!(rounded, ((x + 2).div_euclid(4)) as i64);
    }
    assert_eq!(scale.apply(-2)?, 0);
    assert_eq!(scale.apply(-6)?, -1);
    assert_eq!(scale.apply(2)?, 1);
    assert!(scale.apply(i128::MAX).is_err());
    assert!(bounded(i32::MAX as i64 + 1).is_err());
    assert!(bounded(i32::MIN as i64 - 1).is_err());
    Ok(())
}

#[test]
fn qwen_delta_integer_oracle_cache_causality_and_padding() -> Result<()> {
    let layer = integer_layer();
    let data = inputs(5);
    let expected = oracle(&layer, &data);
    let wrapped = data.iter().map(Tensor::as_wrapped).collect::<Vec<_>>();
    assert_eq!(
        layer.evaluate(&wrapped.iter().collect::<Vec<_>>())?.outputs[0].get_data(),
        expected
    );
    layer.reset_cache();
    let mut cached = Vec::new();
    for t in 0..5 {
        let row = data
            .iter()
            .enumerate()
            .map(|(i, x)| {
                let width = if i < 3 { 4 } else { 2 };
                Tensor::new(
                    if i < 3 {
                        vec![1, 2, 2].into()
                    } else {
                        vec![1, 2].into()
                    },
                    x.data()[t * width..(t + 1) * width].to_vec(),
                )
                .unwrap()
                .as_wrapped()
            })
            .collect::<Vec<_>>();
        cached.extend(layer.evaluate(&row.iter().collect::<Vec<_>>())?.outputs[0].get_data());
    }
    assert_eq!(cached, expected);
    layer.reset_cache();
    let padded = data
        .into_iter()
        .map(|x| x.pad_next_power_of_two().as_wrapped())
        .collect::<Vec<_>>();
    let actual = layer.evaluate(&padded.iter().collect::<Vec<_>>())?.outputs[0].get_data();
    assert_eq!(&actual[..20], expected.as_slice());
    assert_eq!(&actual[20..], &[0; 12]);
    Ok(())
}

#[test]
fn qwen_delta_complete_proof_rejects_input_output_and_state_tampering() -> Result<()> {
    type F = ark_bn254::Fr;
    type P<'a, 'b> = crate::Prover<'a, 'b, F, Blake3Transcript, Pcs>;
    let inputs = inputs(3);
    let mut model =
        Model::new_from_input_shapes(inputs.iter().map(|x| x.shape().clone()).collect());
    model.add_consecutive_layer(Layer::GatedDeltaScan(integer_layer()), None)?;
    model.automatic_output_labelling()?;
    let model = pad_model(model)?;
    let inputs = model.prepare_inputs(inputs)?;
    let trace = model.run(inputs, &mut Default::default())?;
    let (prover, verifier) = model.generate_contexts::<F, Pcs>()?;
    let (proof, io) = P::prove(&prover, trace, &model)?;
    crate::verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), io.clone())?;
    for i in 0..6 {
        let mut bad = io.clone();
        let tensor = if i < 5 {
            &mut bad.input[i]
        } else {
            &mut bad.output[0]
        };
        let mut data = tensor.data().to_vec();
        data[0] = crate::SerializableField(*data[0] + F::from(1u64));
        *tensor = Tensor::new_with_unpadded_shape(
            tensor.shape().clone(),
            tensor.unpadded_shape().clone(),
            data,
        )?;
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), bad)
        }));
        assert!(
            !matches!(rejected, Ok(Ok(()))),
            "Accepted modified public input/output {i}"
        );
    }
    let original = serde_json::to_value(&proof)?;
    let layout = witness::Layout::new(integer_layer().quant.as_ref().unwrap());
    // Every state, intermediate, and rounding-remainder group must be opened.
    // Tampering their sampled evaluations must never survive PCS verification.
    for group in (0..witness::GROUPS).chain([witness::GROUPS, witness::GROUPS + 1]) {
        let mut encoded = original.clone();
        let step = encoded["chunk_proofs"][0]["steps"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .find_map(|x| x.get_mut("GatedDeltaScan"))
            .unwrap();
        let mut scan: GatedDeltaScanProof<F, Pcs> = serde_json::from_value(step.clone())?;
        if group == witness::GROUPS {
            scan.previous_eval += F::from(1u64);
        } else if group == witness::GROUPS + 1 {
            scan.shift_evals[0] += F::from(1u64);
        } else if let Some(index) = layout.groups[group].columns.clone().next() {
            scan.arithmetic_evals[index] += F::from(1u64);
        } else {
            continue;
        }
        *step = serde_json::to_value(scan)?;
        let corrupted = serde_json::from_value(encoded)?;
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::verify::<_, Blake3Transcript, _>(&verifier, corrupted, io.clone())
        }));
        assert!(
            !matches!(rejected, Ok(Ok(()))),
            "Accepted modified DeltaNet witness group {group}"
        );
    }
    Ok(())
}

/// Opt-in resource measurement with the real model's frozen integer parameters.
/// Inputs are deterministic dense synthetic vectors within those proved domains;
/// this benchmarks a complete scan proof, not full-model quality.
#[test]
#[ignore = "large committed scan benchmark; set QWEN_SCAN_BENCH_QUANTIZATION"]
fn qwen_delta_resources_real_shape() -> Result<()> {
    use std::time::Instant;
    type F = ark_bn254::Fr;
    type P<'a, 'b> = crate::Prover<'a, 'b, F, Blake3Transcript, Pcs>;
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
    let report = |stage: &str, elapsed: std::time::Duration| {
        let memory = std::fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .filter(|line| line.starts_with("VmRSS:") || line.starts_with("VmHWM:"))
            .collect::<Vec<_>>()
            .join("; ");
        eprintln!(
            "SCAN_RESOURCE {stage}: {:.3}s {memory}",
            elapsed.as_secs_f64()
        );
    };
    let quantization = std::env::var("QWEN_SCAN_BENCH_QUANTIZATION")?;
    let snapshot: serde_json::Value = serde_json::from_slice(&std::fs::read(quantization)?)?;
    let node = std::env::var("QWEN_SCAN_BENCH_NODE").unwrap_or_else(|_| "23".into());
    let parameters = &snapshot["compiled"]["nodes"][&node]["parameters"];
    let quant: DeltaQuant = serde_json::from_value(parameters["fixed_scales"].clone())?;
    let rows: usize = std::env::var("QWEN_SCAN_BENCH_ROWS")
        .unwrap_or_else(|_| "32".into())
        .parse()?;
    ensure!(
        rows.is_power_of_two() && rows <= 256,
        "Unsupported benchmark rows"
    );
    let mut layer = GatedDeltaScan::new(16, 128, 128);
    layer.quant = Some(quant.clone());
    let layout = witness::Layout::new(&quant);
    let state_columns = layout.columns.iter().filter(|column| column.state).count();
    let vector_columns = layout.columns.len() - state_columns;
    let witness_bytes = (state_columns * rows * 16 * 128 * 128 + vector_columns * rows * 16 * 128)
        * std::mem::size_of::<F>();
    eprintln!(
        "SCAN_RESOURCE dimensions T={rows},H=16,K=128,V=128, state_columns={state_columns}, vector_columns={vector_columns}, committed_bytes={witness_bytes}, rayon_threads={}",
        rayon::current_num_threads()
    );
    let started = Instant::now();
    let mut random = 0x3bd39e10cb0ef621u64;
    let inputs = (0..5)
        .map(|input| {
            let shape = if input < 3 {
                vec![rows, 16, 128]
            } else {
                vec![rows, 16]
            };
            let domain = quant.input_domains[input];
            let magnitude = domain.0.unsigned_abs().min(domain.1.unsigned_abs()) as i64;
            let radius = (magnitude / if input == 2 { 4 } else { 8 }).max(1);
            let values = (0..shape.iter().product())
                .map(|_| {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    match input {
                        3 => magnitude * 9 / 10,
                        4 => magnitude / 2,
                        _ => (random % (2 * radius + 1) as u64) as i64 - radius,
                    }
                })
                .collect();
            Tensor::new(shape.into(), values)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut model =
        Model::new_from_input_shapes(inputs.iter().map(|input| input.shape().clone()).collect());
    model.add_consecutive_layer(Layer::GatedDeltaScan(layer), None)?;
    model.automatic_output_labelling()?;
    let model = pad_model(model)?;
    let inputs = model.prepare_inputs(inputs)?;
    let trace = model.run(inputs, &mut Default::default())?;
    report("integer-trace", started.elapsed());
    let stage = Instant::now();
    let (prover, verifier) = model.generate_contexts::<F, Pcs>()?;
    report("setup", stage.elapsed());
    let stage = Instant::now();
    let (proof, io) = P::prove(&prover, trace, &model)?;
    report("prove", stage.elapsed());
    let proof_bytes = bincode::serde::encode_to_vec(&proof, bincode::config::standard())?.len();
    let stage = Instant::now();
    crate::verify::<_, Blake3Transcript, _>(&verifier, proof, io)?;
    report("verify", stage.elapsed());
    eprintln!("SCAN_RESOURCE verified_proof_bytes={proof_bytes}");
    report("total", started.elapsed());
    Ok(())
}
