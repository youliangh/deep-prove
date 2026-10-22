//! Canonical public statement binding before any chunk or lookup challenges.

use crate::{IO, iop::context::ProofProtocol};
use ark_ff::PrimeField;
use dp_crypto::arkyper::transcript::Transcript;

/// Bind full tensors; split copies are first checked against these tensors.
/// Encoding uses fixed-width lengths and field encodings, never HashMap order.
pub(crate) fn bind_public_io<F: PrimeField, T: Transcript>(
    protocol: ProofProtocol,
    io: &IO<F>,
    transcript: &mut T,
) -> anyhow::Result<()> {
    if protocol == ProofProtocol::LegacyV1 {
        return Ok(());
    }
    io.validate_split_binding()?;
    transcript.append_bytes(b"public-tensors/v2");
    for (tag, tensors) in [
        (b"inputs".as_slice(), &io.input),
        (b"outputs".as_slice(), &io.output),
    ] {
        transcript.append_bytes(tag);
        transcript.append_bytes(&u64::try_from(tensors.len())?.to_le_bytes());
        for tensor in tensors {
            tensor.validate_shape()?;
            for shape in [tensor.shape(), tensor.unpadded_shape()] {
                transcript.append_bytes(&u64::try_from(shape.rank())?.to_le_bytes());
                for &dimension in shape.iter() {
                    transcript.append_bytes(&u64::try_from(dimension)?.to_le_bytes());
                }
            }
            transcript.append_bytes(&u64::try_from(tensor.data().len())?.to_le_bytes());
            for element in tensor.data() {
                transcript.append_scalars(&[**element]);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SerializableField, Tensor};
    use dp_crypto::arkyper::transcript::blake3::Blake3Transcript;
    type F = ark_bn254::Fr;

    fn tensor(
        shape: Vec<usize>,
        actual: Vec<usize>,
        values: &[u64],
    ) -> Tensor<SerializableField<F>> {
        Tensor::new_with_unpadded_shape(
            shape.into(),
            actual.into(),
            values
                .iter()
                .map(|&v| SerializableField(F::from(v)))
                .collect(),
        )
        .unwrap()
    }

    fn challenge(io: &IO<F>) -> F {
        let mut transcript = Blake3Transcript::new(b"public-statement-test");
        ProofProtocol::PublicIoV2.write_domain(&mut transcript);
        bind_public_io(ProofProtocol::PublicIoV2, io, &mut transcript).unwrap();
        transcript.challenge_scalar()
    }

    #[test]
    fn public_io_v2_binds_values_shapes_order_and_tensor_roles() {
        let input = tensor(vec![2], vec![2], &[1, 0]);
        let output = tensor(vec![2], vec![2], &[2, 0]);
        let baseline = IO::new(vec![input.clone()], vec![output.clone()]);
        let expected = challenge(&baseline);
        assert_eq!(expected, challenge(&baseline));
        for changed in [
            IO::new(
                vec![tensor(vec![2], vec![2], &[3, 0])],
                vec![output.clone()],
            ),
            IO::new(vec![input.clone()], vec![tensor(vec![2], vec![2], &[4, 0])]),
            IO::new(
                vec![tensor(vec![1, 2], vec![1, 2], &[1, 0])],
                vec![output.clone()],
            ),
            IO::new(
                vec![tensor(vec![2], vec![1], &[1, 0])],
                vec![output.clone()],
            ),
            IO::new(vec![output.clone()], vec![input.clone()]),
            IO::new(vec![input.clone(), output.clone()], vec![]),
        ] {
            assert_ne!(expected, challenge(&changed));
        }
    }
}
