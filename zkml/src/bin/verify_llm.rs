//! Offline verification of bench-llm artifacts against an explicitly trusted verifier context.
use std::{fs, path::PathBuf};

use anyhow::{Context, Result, ensure};
use ark_bn254::Bn254;
use clap::Parser;
use dp_crypto::arkyper::HyperKZG;
use serde::{Deserialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use zkml::{IO, Proof, model::llm::LLMVerifierContext, parser::llm::Token};

type F = ark_bn254::Fr;
type Pcs = HyperKZG<Bn254>;

#[derive(Debug, Parser)]
#[command(about = "Verify saved LLM proofs offline using an explicitly trusted verifier context")]
struct Args {
    /// Directory containing proof.bin, io.bin, prompt/generated tokens, and result.json.
    #[arg(long)]
    artifacts_dir: PathBuf,

    /// Trusted verifier context. Its association with the intended model must be established externally.
    #[arg(long)]
    verifier: PathBuf,

    /// Optional expected SHA256 from a trusted setup record, independent of the artifact directory.
    #[arg(long, value_parser = parse_sha256)]
    verifier_sha256: Option<String>,

    /// Expected prompt token IDs; defaults to the statement in artifacts-dir/prompt_tokens.json.
    #[arg(long)]
    prompt_tokens: Option<PathBuf>,

    /// Trusted local tokenizer.json; when supplied, also verifies generated.txt and result text.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Optional expected SHA256 of the trusted tokenizer.json.
    #[arg(long, requires = "tokenizer", value_parser = parse_sha256)]
    tokenizer_sha256: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ArtifactReport {
    prompt_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    generated_text: String,
    full_text: String,
    verifier_sha256: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    reject_verifier_unwind(|| verify_artifacts(args))
}

/// Some cryptographic verifier routines reject malformed proofs by panicking.
/// Keep that behavior inside the offline input boundary: an unwind must produce
/// an explicit failure, never a successful statement or an unhandled exit 101.
fn reject_verifier_unwind(verify: impl FnOnce() -> Result<()>) -> Result<()> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(verify)).map_err(|_| {
        anyhow::anyhow!("Malformed or invalid proof caused a verifier panic; verification rejected")
    })?
}

fn verify_artifacts(args: Args) -> Result<()> {
    let verifier_bytes = fs::read(&args.verifier)
        .with_context(|| format!("Reading trusted verifier {}", args.verifier.display()))?;
    let verifier_sha256 = check_digest(&verifier_bytes, args.verifier_sha256.as_deref())?;
    let verifier: LLMVerifierContext<F, Pcs> = decode_exact(&verifier_bytes)?;

    let artifact_prompt: Vec<u32> = read_json(&args.artifacts_dir.join("prompt_tokens.json"))?;
    let prompt = if let Some(path) = &args.prompt_tokens {
        let expected: Vec<u32> = read_json(path)?;
        ensure!(
            expected == artifact_prompt,
            "Artifact prompt differs from the expected prompt"
        );
        expected
    } else {
        artifact_prompt
    };
    let generated: Vec<u32> = read_json(&args.artifacts_dir.join("generated_tokens.json"))?;
    let report: ArtifactReport = read_json(&args.artifacts_dir.join("result.json"))?;
    validate_report(&report, &prompt, &generated, &verifier_sha256)?;

    let proof: Proof<F, Pcs> = decode_exact(&fs::read(args.artifacts_dir.join("proof.bin"))?)?;
    let io: IO<F> = decode_exact(&fs::read(args.artifacts_dir.join("io.bin"))?)?;
    let prompt_tokens = prompt.iter().copied().map(Token::from).collect::<Vec<_>>();
    let certified = verifier.continuation_tokens(&prompt_tokens, &io)?;
    let certified_ids = certified.iter().map(u32::from).collect::<Vec<_>>();
    ensure!(
        certified_ids == generated,
        "Frozen generated tokens differ from the public proof IO"
    );
    verifier
        .verify(proof, prompt_tokens, io)
        .context("LLM proof verification failed")?;

    let (generated_text, tokenizer_sha256) = if let Some(path) = &args.tokenizer {
        let bytes = fs::read(path)?;
        let digest = check_digest(&bytes, args.tokenizer_sha256.as_deref())?;
        let tokenizer = tokenizers::Tokenizer::from_bytes(bytes)
            .map_err(|error| anyhow::anyhow!("Loading trusted tokenizer: {error}"))?;
        let text = decode_text(&tokenizer, &certified_ids)?;
        let all_tokens = prompt
            .iter()
            .chain(&certified_ids)
            .copied()
            .collect::<Vec<_>>();
        let full_text = decode_text(&tokenizer, &all_tokens)?;
        ensure!(
            text == report.generated_text,
            "result.json generated text differs from decoded certified tokens"
        );
        ensure!(
            full_text == report.full_text,
            "result.json full text differs from decoded certified tokens"
        );
        ensure!(
            fs::read_to_string(args.artifacts_dir.join("generated.txt"))? == text,
            "generated.txt differs from decoded certified tokens"
        );
        (Some(text), Some(digest))
    } else {
        (None, None)
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "verified": true,
            "verifier_sha256": verifier_sha256,
            "prompt_tokens": prompt,
            "generated_tokens": certified_ids,
            "text_verified": generated_text.is_some(),
            "generated_text": generated_text,
            "tokenizer_sha256": tokenizer_sha256,
        }))?
    );
    Ok(())
}

fn read_json<T: DeserializeOwned>(path: &std::path::Path) -> Result<T> {
    serde_json::from_slice(&fs::read(path).with_context(|| format!("Reading {}", path.display()))?)
        .with_context(|| format!("Parsing {}", path.display()))
}

fn parse_sha256(value: &str) -> Result<String, String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(value.to_ascii_lowercase())
    } else {
        Err("SHA256 must contain exactly 64 hexadecimal characters".to_owned())
    }
}

fn check_digest(bytes: &[u8], expected: Option<&str>) -> Result<String> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if let Some(expected) = expected {
        ensure!(
            actual == expected,
            "SHA256 mismatch: expected {expected}, got {actual}"
        );
    }
    Ok(actual)
}

fn decode_exact<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let (value, consumed) = bincode::serde::decode_from_slice(bytes, bincode::config::standard())?;
    ensure!(
        consumed == bytes.len(),
        "Trailing bytes after serialized artifact"
    );
    Ok(value)
}

fn validate_report(
    report: &ArtifactReport,
    prompt: &[u32],
    generated: &[u32],
    verifier_sha256: &str,
) -> Result<()> {
    ensure!(
        report.prompt_tokens == prompt,
        "result.json prompt tokens differ from prompt_tokens.json"
    );
    ensure!(
        report.generated_tokens == generated,
        "result.json generated tokens differ from generated_tokens.json"
    );
    if let Some(recorded) = &report.verifier_sha256 {
        ensure!(
            recorded == verifier_sha256,
            "result.json verifier fingerprint differs from the trusted verifier"
        );
    }
    Ok(())
}

fn decode_text(tokenizer: &tokenizers::Tokenizer, ids: &[u32]) -> Result<String> {
    let text = tokenizer
        .decode(ids, true)
        .map_err(|error| anyhow::anyhow!("Decoding certified tokens: {error}"))?;
    // Match HFTokenizer::detokenize, used by bench-llm to write these artifacts.
    Ok(text.strip_prefix(' ').unwrap_or(&text).to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verification_requires_explicit_verifier_but_no_weights() {
        assert!(Args::try_parse_from(["verify-llm", "--artifacts-dir", "/tmp/artifacts"]).is_err());
        let args = Args::try_parse_from([
            "verify-llm",
            "--artifacts-dir",
            "/tmp/artifacts",
            "--verifier",
            "/trusted/setup.vk",
        ])
        .unwrap();
        assert!(args.prompt_tokens.is_none());
        assert!(args.tokenizer.is_none());
    }

    #[test]
    fn digest_pin_and_exact_decoding_reject_modified_artifacts() {
        let bytes =
            bincode::serde::encode_to_vec(vec![1u32, 2, 3], bincode::config::standard()).unwrap();
        assert_eq!(decode_exact::<Vec<u32>>(&bytes).unwrap(), vec![1, 2, 3]);
        let digest = check_digest(&bytes, None).unwrap();
        assert!(check_digest(&bytes, Some(&digest)).is_ok());
        let mut changed = bytes.clone();
        changed[0] ^= 1;
        assert!(check_digest(&changed, Some(&digest)).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(decode_exact::<Vec<u32>>(&trailing).is_err());
        assert!(parse_sha256("1234").is_err());
        assert!(parse_sha256(&"z".repeat(64)).is_err());
    }

    #[test]
    fn metadata_cannot_change_verified_statement() {
        let report = ArtifactReport {
            prompt_tokens: vec![1, 2],
            generated_tokens: vec![3, 4],
            generated_text: "output".into(),
            full_text: "input output".into(),
            verifier_sha256: Some("trusted".into()),
        };
        assert!(validate_report(&report, &[1, 2], &[3, 4], "trusted").is_ok());
        assert!(validate_report(&report, &[1, 5], &[3, 4], "trusted").is_err());
        assert!(validate_report(&report, &[1, 2], &[3, 5], "trusted").is_err());
        assert!(validate_report(&report, &[1, 2], &[3, 4], "different").is_err());
    }

    #[test]
    fn encoded_sumcheck_proofs_reject_invalid_rounds_without_unwinding() {
        use dp_crypto::{
            arkyper::transcript::blake3::Blake3Transcript,
            structs::{IOPProof, IOPProverMessage, IOPVerifierState},
            virtual_poly::VPAuxInfo,
        };

        let verify_encoded = |proof: &IOPProof<F>| {
            let bytes = bincode::serde::encode_to_vec(proof, bincode::config::standard()).unwrap();
            reject_verifier_unwind(|| {
                let decoded: IOPProof<F> = decode_exact(&bytes)?;
                let aux = VPAuxInfo {
                    max_degree: 1,
                    max_num_variables: 1,
                    ..Default::default()
                };
                let mut transcript = Blake3Transcript::new(b"offline-verifier-boundary-test");
                // f(x) = 1 + x has f(0) + f(1) = 3.
                let result =
                    IOPVerifierState::verify(F::from(3u64), &decoded, &aux, &mut transcript);
                ensure!(result.point.len() == 1, "Unexpected sumcheck point");
                Ok(())
            })
        };
        let valid = IOPProof {
            proofs: vec![IOPProverMessage {
                evaluations: vec![F::from(1u64), F::from(2u64)],
            }],
        };
        assert!(verify_encoded(&valid).is_ok());

        let mut inconsistent = valid.clone();
        inconsistent.proofs[0].evaluations[0] = F::from(4u64);
        let mut wrong_degree = valid.clone();
        wrong_degree.proofs[0].evaluations.pop();
        let incomplete = IOPProof { proofs: vec![] };
        for invalid in [inconsistent, wrong_degree, incomplete] {
            let error = verify_encoded(&invalid).unwrap_err();
            assert!(error.to_string().contains("verification rejected"));
        }
    }
}
