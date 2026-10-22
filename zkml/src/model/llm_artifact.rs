//! Versioned on-disk LLM artifacts and decoding of pre-Qwen verifier metadata.
//!
//! New files carry a type and version before the standard bincode payload. Old
//! proof/IO payloads retain their schema. Old verifier metadata is decoded with
//! its original layout, then converted explicitly rather than guessed.

use anyhow::{Context, Result, ensure};
use ark_ff::PrimeField;
use dp_crypto::arkyper::CommitmentScheme;
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::{
    graph::NodeId,
    iop::context::{ProofProtocol, VerifierContext},
    lookup::context::LookupContext,
    model::{ModelCtx, llm::LLMVerifierContext},
    parser::llm::{
        LLMConfig,
        metadata::{AttentionMetadata, FFNGraphID, LLMMetadata, TransformerMetadata},
    },
    poly_commit::context::CommitmentVerifierCtx,
};

const MAGIC: &[u8; 8] = b"DPZKLLM\0";
const HEADER_LEN: usize = 11;
pub const ARTIFACT_VERSION: u16 = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ArtifactKind {
    Verifier = 1,
    Proof = 2,
    Io = 3,
}

pub fn encode_artifact<T: Serialize>(kind: ArtifactKind, value: &T) -> Result<Vec<u8>> {
    let mut bytes = Vec::from(MAGIC.as_slice());
    bytes.extend_from_slice(&ARTIFACT_VERSION.to_le_bytes());
    bytes.push(kind as u8);
    bytes.extend(bincode::serde::encode_to_vec(
        value,
        bincode::config::standard(),
    )?);
    Ok(bytes)
}

fn payload(bytes: &[u8], expected: ArtifactKind) -> Result<Option<(u16, &[u8])>> {
    if !bytes.starts_with(MAGIC) {
        return Ok(None);
    }
    ensure!(bytes.len() >= HEADER_LEN, "Truncated LLM artifact header");
    let version = u16::from_le_bytes(bytes[8..10].try_into()?);
    ensure!(
        version == 1 || version == ARTIFACT_VERSION,
        "Unsupported LLM artifact version {version}"
    );
    ensure!(
        bytes[10] == expected as u8,
        "LLM artifact type differs from expected {expected:?}"
    );
    Ok(Some((version, &bytes[HEADER_LEN..])))
}

fn decode_exact<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let (value, used) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
    ensure!(used == bytes.len(), "Trailing bytes in LLM artifact");
    Ok(value)
}

/// Decode a versioned proof/IO, or an unchanged bare legacy proof/IO payload.
/// Verifier contexts must use [`decode_verifier`] to translate legacy metadata.
pub fn decode_artifact<T: DeserializeOwned>(kind: ArtifactKind, bytes: &[u8]) -> Result<T> {
    ensure!(
        kind != ArtifactKind::Verifier,
        "Use decode_verifier for verifier contexts"
    );
    // A proof or IO envelope never chooses the Fiat–Shamir protocol. The
    // separately trusted verifier context supplies that policy.
    decode_exact(payload(bytes, kind)?.map(|(_, body)| body).unwrap_or(bytes))
}

pub fn decode_verifier<F, PCS>(bytes: &[u8]) -> Result<LLMVerifierContext<F, PCS>>
where
    F: PrimeField,
    PCS: CommitmentScheme<Field = F>,
{
    if let Some((version, body)) = payload(bytes, ArtifactKind::Verifier)? {
        if version == ARTIFACT_VERSION {
            let current: LLMVerifierContext<F, PCS> =
                decode_exact(body).context("Decoding versioned verifier context")?;
            ensure!(
                current.verifier_ctx.protocol == ProofProtocol::PublicIoV2,
                "Version2 verifier contexts must bind public IO; protocol downgrade rejected"
            );
            return Ok(current);
        }
        let legacy: Version1VerifierContext<F, PCS> =
            decode_exact(body).context("Decoding version1 verifier context")?;
        return accept_legacy(LLMVerifierContext {
            verifier_ctx: legacy.verifier_ctx.into_current(),
            md: legacy.md,
            max_context: legacy.max_context,
        });
    }
    let legacy: LegacyVerifierContext<F, PCS> =
        decode_exact(bytes).context("Decoding legacy verifier context")?;
    accept_legacy(LLMVerifierContext {
        verifier_ctx: legacy.verifier_ctx.into_current(),
        md: legacy.md.into(),
        max_context: legacy.max_context,
    })
}

fn accept_legacy<F: PrimeField, PCS: CommitmentScheme>(
    context: LLMVerifierContext<F, PCS>,
) -> Result<LLMVerifierContext<F, PCS>> {
    ensure!(
        context.md.config.model_name != "qwen35"
            && context
                .md
                .transformers
                .iter()
                .all(|block| block.linear_attention.is_none()),
        "Legacy Qwen verifier contexts do not bind public IO; regenerate the proof with protocol V2"
    );
    Ok(context)
}

/// Exact old nested IOP context layout. Bincode defaults cannot decode an added
/// field safely; old formats are parsed explicitly and marked legacy here only.
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
struct LegacyIopVerifierContext<F: PrimeField, PCS: CommitmentScheme> {
    model: ModelCtx<F>,
    commitment_ctx: CommitmentVerifierCtx<PCS>,
    lookup: LookupContext,
}

impl<F: PrimeField, PCS: CommitmentScheme> LegacyIopVerifierContext<F, PCS> {
    fn into_current(self) -> VerifierContext<F, PCS> {
        VerifierContext {
            protocol: ProofProtocol::LegacyV1,
            model: self.model,
            commitment_ctx: self.commitment_ctx,
            lookup: self.lookup,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
struct Version1VerifierContext<F: PrimeField, PCS: CommitmentScheme> {
    verifier_ctx: LegacyIopVerifierContext<F, PCS>,
    md: LLMMetadata,
    max_context: Option<usize>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: ark_serialize::CanonicalSerialize",
    deserialize = "F: ark_serialize::CanonicalDeserialize"
))]
struct LegacyVerifierContext<F: PrimeField, PCS: CommitmentScheme> {
    verifier_ctx: LegacyIopVerifierContext<F, PCS>,
    md: LegacyMetadata,
    max_context: Option<usize>,
}

#[derive(Serialize, Deserialize)]
struct LegacyMetadata {
    config: LLMConfig,
    transformers: Vec<LegacyTransformerMetadata>,
    embeddings: NodeId,
    positional: Option<NodeId>,
    final_proj: NodeId,
    logits: NodeId,
    argmax: NodeId,
}

#[derive(Serialize, Deserialize)]
struct LegacyTransformerMetadata {
    norm_id: NodeId,
    transformer: AttentionMetadata,
    ffn: FFNGraphID,
}

impl From<LegacyMetadata> for LLMMetadata {
    fn from(value: LegacyMetadata) -> Self {
        Self {
            config: value.config,
            additional_eos_tokens: Vec::new(),
            transformers: value
                .transformers
                .into_iter()
                .map(|block| TransformerMetadata {
                    norm_id: block.norm_id,
                    transformer: Some(block.transformer),
                    linear_attention: None,
                    ffn: block.ffn,
                })
                .collect(),
            embeddings: value.embeddings,
            positional: value.positional,
            final_proj: value.final_proj,
            logits: value.logits,
            argmax: value.argmax,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_artifact_envelope_and_legacy_payloads_are_strict() -> Result<()> {
        let value = vec![1u32, 2, 3];
        let bytes = encode_artifact(ArtifactKind::Proof, &value)?;
        assert_eq!(
            decode_artifact::<Vec<u32>>(ArtifactKind::Proof, &bytes)?,
            value
        );
        assert!(decode_artifact::<Vec<u32>>(ArtifactKind::Io, &bytes).is_err());
        let mut unknown = bytes.clone();
        unknown[8] = 3;
        assert!(decode_artifact::<Vec<u32>>(ArtifactKind::Proof, &unknown).is_err());
        assert!(decode_artifact::<Vec<u32>>(ArtifactKind::Proof, &bytes[..10]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(decode_artifact::<Vec<u32>>(ArtifactKind::Proof, &trailing).is_err());
        let legacy = bincode::serde::encode_to_vec(&value, bincode::config::standard())?;
        assert_eq!(
            decode_artifact::<Vec<u32>>(ArtifactKind::Proof, &legacy)?,
            value
        );
        Ok(())
    }

    #[test]
    fn llm_artifact_legacy_metadata_preserves_attention_and_eos() -> Result<()> {
        let node = |value| NodeId::from(value);
        let legacy = LegacyMetadata {
            config: LLMConfig {
                model_name: "llama3".into(),
                embedding_size: 8,
                hidden_size: 8,
                intermediate_size: 16,
                num_heads: 2,
                head_size: 4,
                num_block: 1,
                context_length: 32,
                norm_epsilon: 1e-6,
                vocab_size: 64,
                eos_token: 63usize.into(),
            },
            transformers: vec![LegacyTransformerMetadata {
                norm_id: node(1),
                transformer: AttentionMetadata {
                    qkv_id: node(2),
                    qkt_id: node(3),
                    mask_id: node(4),
                    softmax_id: node(5),
                    residual_id: node(6),
                    final_proj_id: node(7),
                },
                ffn: FFNGraphID {
                    up_id: node(8),
                    down_id: node(9),
                    activation_id: node(10),
                    uses_glu: true,
                },
            }],
            embeddings: node(0),
            positional: None,
            final_proj: node(11),
            logits: node(11),
            argmax: node(12),
        };
        let old_bytes = bincode::serde::encode_to_vec(&legacy, bincode::config::standard())?;
        let converted: LLMMetadata = decode_exact::<LegacyMetadata>(&old_bytes)?.into();
        assert_eq!(converted.eos_tokens(), vec![63usize.into()]);
        assert_eq!(
            converted.transformers[0]
                .transformer
                .as_ref()
                .unwrap()
                .softmax_id,
            node(5)
        );
        assert!(converted.transformers[0].linear_attention.is_none());
        assert_eq!(converted.argmax, node(12));
        Ok(())
    }

    #[test]
    #[ignore = "requires LEGACY_LLM_VERIFIER pointing to an existing bare verifier.bin"]
    fn llm_artifact_reads_existing_legacy_verifier() -> Result<()> {
        let path = std::env::var("LEGACY_LLM_VERIFIER").context("Set LEGACY_LLM_VERIFIER")?;
        let old_bytes = std::fs::read(path)?;
        type F = ark_bn254::Fr;
        type Pcs = crate::testing::Pcs;
        let context = decode_verifier::<F, Pcs>(&old_bytes)?;
        ensure!(
            !context.md.transformers.is_empty(),
            "Legacy transformer graph is empty"
        );
        ensure!(
            context
                .md
                .transformers
                .iter()
                .all(|block| block.transformer.is_some() && block.linear_attention.is_none()),
            "Legacy model was not converted faithfully"
        );
        assert_eq!(context.verifier_ctx.protocol, ProofProtocol::LegacyV1);
        // A legacy context cannot be relabelled as a current trusted setup.
        let wrapped = encode_artifact(ArtifactKind::Verifier, &context)?;
        assert!(decode_verifier::<F, Pcs>(&wrapped).is_err());
        let v1 = Version1VerifierContext {
            verifier_ctx: LegacyIopVerifierContext {
                model: context.verifier_ctx.model.clone(),
                commitment_ctx: context.verifier_ctx.commitment_ctx.clone(),
                lookup: context.verifier_ctx.lookup.clone(),
            },
            md: context.md.clone(),
            max_context: context.max_context,
        };
        let mut v1_bytes = MAGIC.to_vec();
        v1_bytes.extend_from_slice(&1u16.to_le_bytes());
        v1_bytes.push(ArtifactKind::Verifier as u8);
        v1_bytes.extend(bincode::serde::encode_to_vec(
            &v1,
            bincode::config::standard(),
        )?);
        let restored = decode_verifier::<F, Pcs>(&v1_bytes)?;
        assert_eq!(restored.md, context.md);
        assert_eq!(restored.max_context, context.max_context);
        assert_eq!(restored.verifier_ctx.protocol, ProofProtocol::LegacyV1);
        Ok(())
    }
}
