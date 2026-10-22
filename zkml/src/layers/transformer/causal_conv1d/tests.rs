use super::*;
use crate::{layers::Layer, model::Model, padding::pad_model, testing::Pcs, verify};
use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;

fn integer_layer() -> CausalConv1d<Element> {
    CausalConv1d::new(KeyedTensor::new(
        "causal_kernel",
        Tensor::new(
            vec![3, 4].into(),
            vec![1, 2, 3, 4, -1, 2, -3, 4, 4, 3, 2, 1],
        )
        .unwrap(),
    ))
    .unwrap()
}

fn input() -> Tensor<Element> {
    Tensor::new(
        vec![5, 3].into(),
        vec![1, 2, 3, 4, 5, 6, -1, -2, -3, 7, 8, 9, 2, 3, 4],
    )
    .unwrap()
}

#[test]
fn qwen_causal_conv_kernel_order_cache_and_padding() -> Result<()> {
    let layer = integer_layer();
    let x = input();
    let actual = layer.evaluate(&[&x.as_wrapped()])?.outputs[0].get_data();
    // Independent cross-correlation oracle: last kernel entry multiplies x[t].
    let mut expected = vec![0; 15];
    for t in 0..5 {
        for c in 0..3 {
            for lag in 0..4 {
                if t >= lag {
                    expected[t * 3 + c] +=
                        x.data()[(t - lag) * 3 + c] * layer.weight.data()[c * 4 + 3 - lag];
                }
            }
        }
    }
    assert_eq!(actual, expected);
    assert_eq!(&actual[..3], &[4, 8, 3]);
    layer.reset_cache();
    let prefix = Tensor::new(vec![2, 3].into(), x.data()[..6].to_vec())?;
    let mut cached = layer.evaluate(&[&prefix.as_wrapped()])?.outputs[0].get_data();
    for row in 2..5 {
        let one = Tensor::new(vec![1, 3].into(), x.data()[row * 3..(row + 1) * 3].to_vec())?;
        cached.extend(layer.evaluate(&[&one.as_wrapped()])?.outputs[0].get_data());
    }
    assert_eq!(cached, expected);
    layer.reset_cache();
    assert_eq!(
        layer.evaluate(&[&x.as_wrapped()])?.outputs[0].get_data(),
        expected
    );

    let padded_layer = CausalConv1d::new(KeyedTensor::new(
        "causal_padded_kernel",
        layer.weight.tensor().clone().pad_next_power_of_two(),
    ))?;
    let padded = x.clone().pad_next_power_of_two();
    let out = padded_layer.evaluate(&[&padded.as_wrapped()])?.outputs[0].get_data();
    for t in 0..8 {
        for c in 0..4 {
            assert_eq!(
                out[t * 4 + c],
                if t < 5 && c < 3 {
                    expected[t * 3 + c]
                } else {
                    0
                }
            );
        }
    }
    Ok(())
}

#[test]
fn qwen_causal_conv_float_matches_integer_small_exact_values() -> Result<()> {
    let x = input();
    let integer = integer_layer();
    let float = CausalConv1d::new(KeyedTensor::new(
        "causal_float_kernel",
        Tensor::new(
            vec![3, 4].into(),
            integer.weight.data().iter().map(|x| *x as f32).collect(),
        )?,
    ))?;
    let xf = Tensor::new(
        x.shape().clone(),
        x.data().iter().map(|x| *x as f32).collect(),
    )?;
    let expected: Vec<f32> = integer.evaluate(&[&x.as_wrapped()])?.outputs[0]
        .get_data()
        .iter()
        .map(|x| *x as f32)
        .collect();
    assert_eq!(
        float.evaluate(&[&xf.as_wrapped()])?.outputs[0].get_data(),
        expected
    );
    Ok(())
}

#[test]
fn qwen_causal_conv_rejects_bad_kernel_and_overflow() -> Result<()> {
    assert!(
        CausalConv1d::new(KeyedTensor::new(
            "bad",
            Tensor::<Element>::new(vec![2, 3].into(), vec![1; 6])?
        ))
        .is_err()
    );
    let layer = CausalConv1d::new(KeyedTensor::new(
        "overflow",
        Tensor::<Element>::new(vec![1, 4].into(), vec![2; 4])?,
    ))?;
    let x = Tensor::new(vec![1, 1].into(), vec![i64::MAX])?;
    assert!(layer.evaluate(&[&x.as_wrapped()]).is_err());
    Ok(())
}

#[test]
fn qwen_causal_conv_full_proof_rejects_tampered_history_and_output() -> Result<()> {
    type F = ark_bn254::Fr;
    type P<'a, 'b> = Prover<'a, 'b, F, Blake3Transcript, Pcs>;
    let mut model = Model::new_from_input_shapes(vec![vec![5, 3].into()]);
    model.add_consecutive_layer(Layer::CausalConv1d(integer_layer()), None)?;
    model.automatic_output_labelling()?;
    let model = pad_model(model)?;
    let inputs = model.prepare_inputs(vec![input()])?;
    let trace = model.run(inputs, &mut Default::default())?;
    let (prover, verifier) = model.generate_contexts::<F, Pcs>()?;
    let (proof, io) = P::prove(&prover, trace, &model)?;
    verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), io.clone())?;
    let mut wrong_input = io.clone();
    let original = &wrong_input.input[0];
    let mut values = original.data().to_vec();
    values[0] = crate::SerializableField(*values[0] + F::from(1));
    wrong_input.input[0] = Tensor::new_with_unpadded_shape(
        original.shape().clone(),
        original.unpadded_shape().clone(),
        values,
    )?;
    // dp-crypto currently uses assertions for malformed sumcheck claims, so
    // rejection can be an error or a panic; successful verification is forbidden.
    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify::<_, Blake3Transcript, _>(&verifier, proof.clone(), wrong_input)
    }));
    assert!(!matches!(rejected, Ok(Ok(()))));
    let mut wrong_output = io;
    let original = &wrong_output.output[0];
    let mut values = original.data().to_vec();
    values[0] = crate::SerializableField(*values[0] + F::from(1));
    wrong_output.output[0] = Tensor::new_with_unpadded_shape(
        original.shape().clone(),
        original.unpadded_shape().clone(),
        values,
    )?;
    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify::<_, Blake3Transcript, _>(&verifier, proof, wrong_output)
    }));
    assert!(!matches!(rejected, Ok(Ok(()))));
    Ok(())
}
