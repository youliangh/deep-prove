use super::*;
use crate::{
    Tensor,
    layers::{
        Layer,
        transformer::normalisation::qwen_norm::{NormKind, QwenNorm, integer::NormQuant},
    },
};

struct CachedGeneration {
    tokens: Vec<Token>,
    history: Vec<Element>,
    logits: Vec<Element>,
    step_logit_rows: Vec<usize>,
}

// Call Model::run directly: unlike Driver::run, it preserves the original
// prefill/decode caches. Each trace is discarded after saving its actual logits.
fn collect_cached_generation(
    driver: &mut Driver<Element>,
    prompt: &[Token],
    max_new_tokens: usize,
) -> Result<CachedGeneration> {
    anyhow::ensure!(!prompt.is_empty() && max_new_tokens > 0, "Empty generation");
    anyhow::ensure!(
        prompt.len() + max_new_tokens - 1 <= driver.md.context_length(),
        "Requested history exceeds the loaded context"
    );
    driver.model.reset();
    let vocab = driver.md.vocab_size();
    let mut history: Vec<Element> = prompt.iter().map(|x| usize::from(*x) as Element).collect();
    let mut input = driver.tokens_to_tensor(prompt)?;
    let mut tokens = Vec::new();
    let mut logits = Vec::new();
    let mut step_logit_rows = Vec::new();
    for step in 0..max_new_tokens {
        let trace = driver
            .model
            .run(vec![input], &mut tenstore::GenStore::default())?;
        let step_logits = trace
            .get_step(&driver.md.argmax)
            .context("Missing cached-step argmax inputs")?
            .inputs()[0]
            .tensor()?;
        anyhow::ensure!(step_logits.shape().rank() == 2, "Expected matrix logits");
        let rows = step_logits.unpadded_shape().dim(0);
        let columns = step_logits.shape().dim(1);
        anyhow::ensure!(rows > 0 && columns >= vocab, "Invalid cached logits shape");
        step_logit_rows.push(rows);
        let first = if step == 0 {
            anyhow::ensure!(
                rows == prompt.len(),
                "Prefill did not expose every prompt row"
            );
            0
        } else {
            rows - 1
        };
        for row in first..rows {
            logits.extend_from_slice(&step_logits.data()[row * columns..row * columns + vocab]);
        }
        let prediction = trace.outputs()[0].tensor()?;
        let last = prediction.unpadded_shape().dim(0) - 1;
        let token_id = usize::try_from(prediction.data()[last])?;
        anyhow::ensure!(token_id < vocab, "Invalid cached token ID");
        let token = Token::from(token_id);
        tokens.push(token);
        if driver.md.eos_tokens().contains(&token) || step + 1 == max_new_tokens {
            break;
        }
        history.push(token_id as Element);
        input = driver.tokens_to_tensor(&[token])?;
    }
    anyhow::ensure!(
        logits.len() == history.len() * vocab,
        "Missing cached logit rows"
    );
    Ok(CachedGeneration {
        tokens,
        history,
        logits,
        step_logit_rows,
    })
}

fn tiny_integer_driver() -> Result<(Driver<Element>, ModelMetadata, Vec<Token>)> {
    use crate::parser::{llm::models::qwen35::Qwen35, safe::RawSafeTensors};
    let directory = tempfile::tempdir()?;
    std::fs::write(
        directory.path().join("model.safetensors"),
        include_bytes!("../../../parser/llm/models/qwen35/fixtures/model.safetensors"),
    )?;
    std::fs::write(
        directory.path().join("config.json"),
        include_str!("../../../parser/llm/models/qwen35/fixtures/config.json"),
    )?;
    let raw = RawSafeTensors::new(
        directory.path().join("model.safetensors"),
        directory.path().join("tokenizer.json"),
        directory.path().join("config.json"),
    );
    let float_driver = Driver::load_from_model(Qwen35::new(), &raw, Some(16))?;
    // Calibration supplies IDs directly and never invokes this tokenizer.
    let tokenizer: HFTokenizer = serde_json::from_value(json!({"tokenizer":
        tokenizers::Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default())}))?;
    let tokens: Vec<Token> = [1usize, 2, 3].into_iter().map(Token::from).collect();
    let (driver, metadata) = quantize(float_driver, &tokenizer, Some(vec![tokens.clone()]))?;
    Ok((driver, metadata, tokens))
}

#[test]
fn qwen35_tiny_integer_graph_replays_and_freezes_all_arithmetic() -> Result<()> {
    let (mut driver, metadata, tokens) = tiny_integer_driver()?;
    let before = compiled_quantization_snapshot(&driver, &metadata)?;
    let result = driver.generate(&tokens, 3, &mut tenstore::GenStore::default())?;
    let generated = driver.continuation_tokens(&result, tokens.len())?;
    assert!(!generated.is_empty());
    assert!(generated.iter().all(|token| usize::from(*token) < 32));
    // Driver::generate checks integer cached predictions against full causal replay.
    let repeated = driver.generate(&tokens, 3, &mut tenstore::GenStore::default())?;
    assert_eq!(
        generated,
        driver.continuation_tokens(&repeated, tokens.len())?
    );
    assert_eq!(before, compiled_quantization_snapshot(&driver, &metadata)?);
    let id = driver
        .model
        .graph()
        .inner_nodes()
        .find_map(|(id, layer)| matches!(layer, Layer::QwenNorm(_)).then_some(id))
        .context("Tiny graph has no norm")?;
    let Layer::QwenNorm(norm) = driver
        .model
        .graph_mut()
        .node_mut(id)
        .and_then(|node| node.as_inner_mut())
        .context("Missing norm node")?
    else {
        unreachable!()
    };
    norm.quant.as_mut().context("Unquantized norm")?.numerator += 1;
    assert_ne!(
        before["compiled_parameters_sha256"],
        compiled_quantization_snapshot(&driver, &metadata)?["compiled_parameters_sha256"]
    );
    Ok(())
}

#[test]
fn qwen35_tiny_hybrid_proof_artifact_roundtrip_and_wrong_prompt_rejection() -> Result<()> {
    tiny_hybrid_proof(2)
}

#[test]
fn qwen35_tiny_integer_generation_matches_replay_after_clearing_every_cache() -> Result<()> {
    let (mut driver, _, tokens) = tiny_integer_driver()?;
    for requested in [1, 2, 3, 5, 9] {
        let cached = collect_cached_generation(&mut driver, &tokens, requested)?;
        let history = Tensor::new(vec![cached.history.len()].into(), cached.history.clone())?;

        // Unlike the certification pass inside generate(), this deliberately
        // discards Softmax shift witnesses as well as KV/conv/Delta state.
        driver.model.reset();
        let fresh = driver.run(vec![history], &mut tenstore::GenStore::default())?;
        assert_eq!(
            cached.tokens,
            driver.continuation_tokens(&fresh, tokens.len())?,
            "Fresh-cache token mismatch for {requested} requested tokens"
        );
        let actual_logits = fresh
            .get_step(&driver.md.argmax)
            .context("Missing fresh argmax inputs")?
            .inputs()[0]
            .tensor()?;
        let width = actual_logits.shape().dim(1);
        let actual: Vec<_> = (0..cached.history.len())
            .flat_map(|row| {
                actual_logits.data()[row * width..row * width + driver.md.vocab_size()]
                    .iter()
                    .copied()
            })
            .collect();
        assert_eq!(
            cached.logits, actual,
            "Fresh-cache integer logits differ for {requested} requested tokens"
        );
    }
    Ok(())
}

/// Run only after the independent development evaluator has accepted the frozen
/// quantization digest. This intentionally does not reuse Softmax shift witnesses.
#[test]
#[ignore = "requires local full weights, accepted frozen manifest/digests and replay cases"]
fn qwen35_full_integer_independent_cache_replay() -> Result<()> {
    full_integer_replay_or_lookup_preflight(false)
}

/// Exercise the real trace's lookup witnesses without polynomial commitments or
/// setup, so lookup domain failures identify their exact node before full proving.
#[test]
#[ignore = "requires full local weights and the accepted frozen32-row replay case"]
fn qwen35_full_integer_lookup_witness_preflight() -> Result<()> {
    full_integer_replay_or_lookup_preflight(true)
}

fn full_integer_replay_or_lookup_preflight(lookup_preflight: bool) -> Result<()> {
    use crate::parser::{
        llm::{LLMTokenizer, models::qwen35::Qwen35, tokenizer::TokenizerLoader},
        safe::RawSafeTensors,
    };
    use std::{io::Read, path::Path, time::Instant};

    #[derive(serde::Deserialize)]
    struct Case {
        name: String,
        prompt_tokens: Vec<usize>,
        max_new_tokens: usize,
        #[serde(default)]
        require_full_budget: bool,
    }
    fn file_sha256(path: &Path) -> Result<String> {
        let mut reader = std::fs::File::open(path)?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0u8; 1024 * 1024];
        loop {
            let used = reader.read(&mut buffer)?;
            if used == 0 {
                break;
            }
            digest.update(&buffer[..used]);
        }
        Ok(format!("{:x}", digest.finalize()))
    }
    fn run_case(
        driver: &mut Driver<Element>,
        tokenizer: &HFTokenizer,
        case: &Case,
        lookup_preflight: bool,
        snapshot: &Value,
    ) -> Result<Value> {
        let started = Instant::now();
        let tokens: Vec<Token> = case
            .prompt_tokens
            .iter()
            .copied()
            .map(Token::from)
            .collect();
        eprintln!("Independent replay {}: cached generation", case.name);
        let cached = collect_cached_generation(driver, &tokens, case.max_new_tokens)?;
        let expected_tokens = &cached.tokens;
        let history_tokens = &cached.history;
        let history = Tensor::new(vec![history_tokens.len()].into(), history_tokens.clone())?;

        // Clear every cache, including the shifts retained by the production
        // certification pass, before evaluating the actual generated history.
        driver.model.reset();
        eprintln!(
            "Independent replay {}: fresh full history ({} rows)",
            case.name,
            history_tokens.len()
        );
        let fresh = driver.run(vec![history], &mut tenstore::GenStore::default())?;
        let actual_tokens = driver.continuation_tokens(&fresh, tokens.len())?;
        if lookup_preflight {
            use crate::{
                layers::{
                    activation::{Activation, ActivationLayer},
                    provable::OpInfo,
                },
                lookup::operation::LookupOp,
            };
            anyhow::ensure!(
                history_tokens.len() == 32,
                "Preflight requires exactly32 computed rows"
            );
            let mut checks = Vec::new();
            for (id, layer) in driver.model.graph().forward_inners() {
                let data = match layer {
                    Layer::Activation(
                        Activation::Plain(activation) | Activation::GLU(activation),
                    ) => match activation {
                        ActivationLayer::Relu(Some(data), _)
                        | ActivationLayer::Gelu(Some(data), _)
                        | ActivationLayer::Silu(Some(data), _)
                        | ActivationLayer::Sigmoid(Some(data), _)
                        | ActivationLayer::Softplus(Some(data), _)
                        | ActivationLayer::ExpNegative(Some(data), _) => Some(data),
                        _ => anyhow::bail!("Activation {id} is not quantized"),
                    },
                    Layer::Requant(requant) => Some(&requant.activation_lookup_data),
                    Layer::Softmax(_) => None,
                    _ => continue,
                };
                let step = fresh
                    .get_step(&id)
                    .with_context(|| format!("Missing lookup trace node {id}"))?;
                let input = step.input_tensor_at(0)?;
                let minimum = input.data().iter().copied().min();
                let maximum = input.data().iter().copied().max();
                let parameters =
                    &snapshot["compiled"]["nodes"][id.to_string().trim_start_matches('N')];
                eprintln!(
                    "Lookup preflight {id} {} input={minimum:?}..={maximum:?} parameters={parameters}",
                    layer.describe()
                );
                let (result, arithmetic) = if let Some(data) = data {
                    let rounding = i128::from(data.rounding_constant());
                    let multiplier = i128::from(data.fixed_point_multiplier());
                    let low = minimum.map(|x| i128::from(x) * multiplier + rounding);
                    let high = maximum.map(|x| i128::from(x) * multiplier + rounding);
                    let chunking = data.chunking_info(&data.table)?;
                    let mut legacy_invalid = 0usize;
                    let mut encoding_i64_overflows = 0usize;
                    let mut negative_offset_remainders = 0usize;
                    let mut first_legacy_failure = Value::Null;
                    if chunking.is_signed() && chunking.number_of_zeroing_chunks() > 0 {
                        let value_bits =
                            data.table.table_bit_size() * chunking.number_of_value_chunks();
                        let value_offset = (1i128 << (value_bits - 1)) << data.right_shift();
                        let offset = (1i128 << (data.intermediate_bit_size() + 15 - 1))
                            .max(value_offset)
                            + value_offset;
                        let top_shift = data.right_shift()
                            + value_bits
                            + (chunking.number_of_zeroing_chunks() - 1) * 16;
                        let top_offset = offset >> top_shift;
                        for &x in input.data() {
                            let scaled = i128::from(x) * multiplier + rounding;
                            let encoded = scaled + offset;
                            encoding_i64_overflows += usize::from(
                                encoded > i128::from(i64::MAX) || encoded < i128::from(i64::MIN),
                            );
                            negative_offset_remainders += usize::from(encoded < 0);
                            let legacy_top =
                                (((encoded as i64) >> top_shift) & 65535) as i128 - top_offset;
                            if !(-32768..=32767).contains(&legacy_top) {
                                legacy_invalid += 1;
                                if first_legacy_failure.is_null() {
                                    first_legacy_failure = json!({"input":x,"rescaled":scaled.to_string(),"encoded":encoded.to_string(),
                                        "legacy_top":legacy_top.to_string(),"correct_signed_top":((encoded>>top_shift)-top_offset).to_string()});
                                }
                            }
                        }
                    }
                    let diagnostics = json!({
                        "rescaled_minimum_i128":low.map(|x|x.to_string()),
                        "rescaled_maximum_i128":high.map(|x|x.to_string()),
                        "rescaled_values_fit_i64":low.is_none_or(|x|x>=i128::from(i64::MIN))
                            && high.is_none_or(|x|x<=i128::from(i64::MAX)),
                        "chunking":format!("{chunking:?}"),
                        "legacy_signed_top_out_of_domain_values":legacy_invalid,
                        "encoding_additions_outside_i64":encoding_i64_overflows,
                        "negative_offset_remainders":negative_offset_remainders,
                        "first_legacy_failure":first_legacy_failure,
                    });
                    (
                        data.get_lookup_witness((*input).clone()).map(|_| ()),
                        diagnostics,
                    )
                } else if let Layer::Softmax(softmax) = layer {
                    (
                        softmax
                            .lookup_witness::<ark_bn254::Fr>(id, step)
                            .map(|_| ()),
                        Value::Null,
                    )
                } else {
                    unreachable!()
                };
                let record = json!({
                    "node_id":id.to_string(),"operator":layer.describe(),
                    "input_shape":input.shape().to_vec(),"input_minimum":minimum,"input_maximum":maximum,
                    "parameters":parameters,"arithmetic":arithmetic,
                    "passed":result.is_ok(),"error":result.err().map(|error|format!("{error:#}")),
                });
                eprintln!("Lookup preflight {id}: passed={}", record["passed"]);
                checks.push(record);
            }
            let passed = checks.iter().all(|record| record["passed"] == true)
                && actual_tokens == *expected_tokens;
            return Ok(json!({
                "name":case.name,"passed":passed,"history_tokens":history_tokens,
                "computed_rows":history_tokens.len(),"generated_tokens":actual_tokens.iter().copied().map(usize::from).collect::<Vec<_>>(),
                "generated_text":tokenizer.detokenize(&actual_tokens),
                "lookup_nodes_checked":checks.len(),"lookup_checks":checks,
                "elapsed_seconds":started.elapsed().as_secs_f64(),
            }));
        }
        let actual_logits = fresh
            .get_step(&driver.md.argmax)
            .context("Missing fresh argmax inputs")?
            .inputs()[0]
            .tensor()?;
        anyhow::ensure!(actual_logits.shape().rank() == 2, "Expected matrix logits");
        let rows = history_tokens.len();
        let columns = actual_logits.shape().dim(1);
        let vocab = driver.md.vocab_size();
        anyhow::ensure!(
            columns >= vocab && actual_logits.unpadded_shape().dim(0) == rows,
            "Logits do not match the complete unpadded history"
        );
        let mut mismatches = 0usize;
        let mut row_mismatches = vec![0usize; rows];
        let mut first_mismatch = None;
        let mut max_absolute_difference = 0i128;
        let mut cached_digest = Sha256::new();
        let mut fresh_digest = Sha256::new();
        for row in 0..rows {
            let cached = &cached.logits[row * vocab..(row + 1) * vocab];
            let fresh = &actual_logits.data()[row * columns..row * columns + vocab];
            cached_digest.update(
                cached
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            fresh_digest.update(
                fresh
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            for (column, (&a, &b)) in cached.iter().zip(fresh).enumerate() {
                if a != b {
                    mismatches += 1;
                    row_mismatches[row] += 1;
                    max_absolute_difference =
                        max_absolute_difference.max((i128::from(a) - i128::from(b)).abs());
                    if first_mismatch.is_none() {
                        first_mismatch =
                            Some(json!({"row":row,"vocabulary_index":column,"cached":a,"fresh":b}));
                    }
                }
            }
        }
        let token_match = *expected_tokens == actual_tokens;
        let budget_met = !case.require_full_budget || expected_tokens.len() == case.max_new_tokens;
        let ids = |tokens: &[Token]| tokens.iter().copied().map(usize::from).collect::<Vec<_>>();
        let passed = mismatches == 0 && token_match && budget_met;
        eprintln!(
            "Independent replay {}: passed={passed}, logits mismatches={mismatches}, generated={}",
            case.name,
            expected_tokens.len()
        );
        Ok(json!({
            "name":case.name,"passed":passed,"prompt_tokens":case.prompt_tokens,
            "cached_collection":"original-prefill-and-incremental-decode-logits",
            "cached_step_logit_rows":cached.step_logit_rows,
            "requested_max_new_tokens":case.max_new_tokens,"require_full_budget":case.require_full_budget,
            "full_budget_met":budget_met,"history_tokens":history_tokens,
            "logit_shape":[rows,vocab],"compared_integer_logits":rows*vocab,
            "logit_mismatches":mismatches,"row_mismatches":row_mismatches,
            "first_mismatch":first_mismatch,"max_absolute_difference":max_absolute_difference.to_string(),
            "cached_logits_i64le_sha256":format!("{:x}",cached_digest.finalize()),
            "fresh_logits_i64le_sha256":format!("{:x}",fresh_digest.finalize()),
            "continuation_tokens_match":token_match,"cached_generated_tokens":ids(&expected_tokens),
            "fresh_generated_tokens":ids(&actual_tokens),"generated_text":tokenizer.detokenize(&actual_tokens),
            "elapsed_seconds":started.elapsed().as_secs_f64()
        }))
    }

    let required = |key: &str| std::env::var(key).with_context(|| format!("Set {key}"));
    let model_directory = required("QWEN35_MODEL_DIR")?;
    let calibration_path = required("QWEN35_CALIBRATION_TOKENS")?;
    let cases_path = required("QWEN35_REPLAY_CASES")?;
    let output_directory = required("QWEN35_REPLAY_OUTPUT")?;
    let reference_path = required("QWEN35_REFERENCE_MANIFEST")?;
    let expected_digest = required("QWEN35_EXPECTED_QUANTIZATION_SHA256")?;
    let expected_compiled_digest = required("QWEN35_EXPECTED_COMPILED_SHA256")?;
    anyhow::ensure!(
        *crate::quantization::BIT_LEN == 14,
        "Set ZKML_BIT_LEN=14 before process startup"
    );
    anyhow::ensure!(rayon::current_num_threads() == 4, "Set RAYON_NUM_THREADS=4");
    let reference: Value = serde_json::from_slice(&std::fs::read(&reference_path)?)?;
    anyhow::ensure!(
        reference["quantization_sha256"].as_str() == Some(expected_digest.as_str()),
        "Reference manifest differs from independently accepted quantization SHA256"
    );
    anyhow::ensure!(
        reference["max_context"] == 256 && reference["quantization_bits"] == 14,
        "Reference must use context256 and14-bit quantization"
    );
    let hashes = reference["hashes"]
        .as_object()
        .context("Missing reference model fingerprints")?;
    anyhow::ensure!(!hashes.is_empty(), "Reference model fingerprints are empty");
    for (name, expected) in hashes {
        anyhow::ensure!(
            Path::new(name)
                .components()
                .all(|p| matches!(p, std::path::Component::Normal(_))),
            "Invalid reference model filename"
        );
        anyhow::ensure!(
            expected.as_str()
                == Some(file_sha256(&Path::new(&model_directory).join(name))?.as_str()),
            "Model file differs from accepted reference: {name}"
        );
    }
    let mut cases: Vec<Case> = serde_json::from_slice(&std::fs::read(&cases_path)?)?;
    let computed_rows = |case: &Case| {
        case.prompt_tokens
            .len()
            .checked_add(case.max_new_tokens)
            .and_then(|value| value.checked_sub(1))
    };
    anyhow::ensure!(
        cases.iter().any(|c| c.name.starts_with("en_"))
            && cases.iter().any(|c| c.name.starts_with("zh_"))
            && cases.iter().any(|c| computed_rows(c) == Some(32)
                && c.max_new_tokens > 1
                && c.require_full_budget)
            && cases.iter().all(|c| !c.prompt_tokens.is_empty()
                && c.max_new_tokens > 0
                && computed_rows(c).is_some_and(|rows| rows <= 32)),
        "Replay cases require English, Chinese, and a full32-row cache boundary; all histories must fit32 rows"
    );
    if lookup_preflight {
        cases.retain(|case| case.name == "en_factual_32");
        anyhow::ensure!(
            cases.len() == 1,
            "Lookup preflight requires the frozen en_factual_32 case"
        );
    }
    let mut calibration: Vec<Vec<Token>> =
        serde_json::from_slice::<Vec<Vec<usize>>>(&std::fs::read(&calibration_path)?)?
            .into_iter()
            .map(|row| row.into_iter().map(Token::from).collect())
            .collect();
    anyhow::ensure!(calibration.len() >= 8, "Need at least8 calibration samples");
    calibration.truncate(8);
    let calibration_digest = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&Some(&calibration))?)
    );
    anyhow::ensure!(
        reference["calibration_sha256"].as_str() == Some(calibration_digest.as_str()),
        "Calibration prefix differs from frozen reference"
    );
    let raw = RawSafeTensors::from_directory(&model_directory)?;
    let tokenizer = Qwen35::new().load_tokenizer(&raw)?;
    let float_driver = Driver::load_from_model(Qwen35::new(), &raw, Some(256))?;
    anyhow::ensure!(
        float_driver.md.transformers.len() == 24,
        "Expected complete24-layer model"
    );
    eprintln!("Independent replay: calibrating final14-bit model with8 frozen samples");
    let (mut driver, metadata) = quantize(float_driver, &tokenizer, Some(calibration))?;
    let snapshot = compiled_quantization_snapshot(&driver, &metadata)?;
    let snapshot_bytes = serde_json::to_vec(&snapshot)?;
    let digest = format!("{:x}", Sha256::digest(&snapshot_bytes));
    anyhow::ensure!(
        digest == expected_digest,
        "Compiled quantization SHA256 mismatch: {digest}"
    );
    anyhow::ensure!(
        snapshot["compiled_parameters_sha256"].as_str() == Some(expected_compiled_digest.as_str()),
        "Compiled arithmetic parameters differ from frozen reference"
    );
    let output = Path::new(&output_directory);
    std::fs::create_dir_all(output)?;
    std::fs::write(output.join("quantization.json"), &snapshot_bytes)?;
    let mut summary = json!({
        "schema":"qwen35-independent-cache-replay-v1","status":"running","passed":false,
        "independent_softmax_shift_recomputation":true,"all_caches_cleared_before_replay":true,
        "model_directory":model_directory,"model_hashes":hashes,"model_layers":24,
        "quantization_bits":14,"calibration_samples":8,"rayon_threads":4,
        "maximum_computed_history_rows":32,"loaded_max_context":256,
        "quantization_sha256":digest,"compiled_parameters_sha256":expected_compiled_digest,
        "calibration_sha256":calibration_digest,"reference_manifest_sha256":file_sha256(Path::new(&reference_path))?,
        "cases_sha256":file_sha256(Path::new(&cases_path))?,
        "test_binary_sha256":file_sha256(&std::env::current_exe()?)?,
        "cargo_lock_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../../../../Cargo.lock"))),
        "cases":[]
    });
    if lookup_preflight {
        summary["schema"] = "qwen35-lookup-witness-preflight-v1".into();
        summary["polynomial_commitments_or_setup"] = false.into();
    }
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    for case in &cases {
        let record = match run_case(&mut driver, &tokenizer, case, lookup_preflight, &snapshot) {
            Ok(record) => record,
            Err(error) => json!({"name":case.name,"passed":false,"error":format!("{error:#}")}),
        };
        summary["cases"].as_array_mut().unwrap().push(record);
        std::fs::write(
            output.join("summary.json"),
            serde_json::to_vec_pretty(&summary)?,
        )?;
    }
    let passed = summary["cases"]
        .as_array()
        .unwrap()
        .iter()
        .all(|case| case["passed"] == true);
    summary["status"] = "complete".into();
    summary["passed"] = passed.into();
    std::fs::write(
        output.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    anyhow::ensure!(
        snapshot == compiled_quantization_snapshot(&driver, &metadata)?,
        "Inference mutated frozen arithmetic parameters"
    );
    anyhow::ensure!(
        passed,
        "Independent replay failed; inspect {}",
        output.join("summary.json").display()
    );
    Ok(())
}

#[test]
fn qwen35_tiny_padded_hybrid_proof_artifact_roundtrip() -> Result<()> {
    tiny_hybrid_proof(1)
}

fn tiny_hybrid_proof(generated: usize) -> Result<()> {
    use crate::{
        IO, Proof,
        model::{
            llm::WithMaxContext,
            llm_artifact::{ArtifactKind, decode_artifact, decode_verifier, encode_artifact},
        },
        testing::Pcs,
    };
    type F = ark_bn254::Fr;
    let (mut driver, _, tokens) = tiny_integer_driver()?;
    let trace = driver.generate(&tokens, generated, &mut tenstore::GenStore::default())?;
    let token_limit = tokens.len() + generated - 1;
    let (mut prover, verifier) = driver.context::<F, Pcs>()?.with_max_context(token_limit);
    prover.protocol = crate::iop::context::ProofProtocol::LegacyV1;
    assert!(
        prover
            .write_to_transcript(
                &mut dp_crypto::arkyper::transcript::blake3::Blake3Transcript::new(
                    b"reject-legacy-prover"
                )
            )
            .is_err()
    );
    prover.protocol = crate::iop::context::ProofProtocol::PublicIoV2;
    let (proof, io) = driver.prove(&prover, trace)?;
    let verifier_bytes = encode_artifact(ArtifactKind::Verifier, &verifier)?;
    let proof_bytes = encode_artifact(ArtifactKind::Proof, &proof)?;
    let io_bytes = encode_artifact(ArtifactKind::Io, &io)?;
    let verifier = decode_verifier::<F, Pcs>(&verifier_bytes)?;
    assert_eq!(
        verifier.verifier_ctx.protocol,
        crate::iop::context::ProofProtocol::PublicIoV2
    );
    verifier.verify(
        decode_artifact::<Proof<F, Pcs>>(ArtifactKind::Proof, &proof_bytes)?,
        tokens.clone(),
        decode_artifact::<IO<F>>(ArtifactKind::Io, &io_bytes)?,
    )?;
    // Exercise the cryptographic verifier directly, bypassing the LLM's prompt
    // and token-chain checks: both public tensor roles must be bound by FS.
    for change_input in [true, false] {
        let mut altered = io.clone();
        let tensor = if change_input {
            &mut altered.input[0]
        } else {
            &mut altered.output[0]
        };
        let mut values = tensor.data().to_vec();
        values[0] = crate::SerializableField(*values[0] + F::from(1u64));
        *tensor = Tensor::new_with_unpadded_shape(
            tensor.shape().clone(),
            tensor.unpadded_shape().clone(),
            values,
        )?;
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::verify::<F, dp_crypto::arkyper::transcript::blake3::Blake3Transcript, Pcs>(
                &verifier.verifier_ctx,
                proof.clone(),
                altered,
            )
        }));
        assert!(
            !matches!(rejected, Ok(Ok(()))),
            "Changed public IO was accepted"
        );
    }
    let mut downgraded = decode_verifier::<F, Pcs>(&verifier_bytes)?;
    downgraded.verifier_ctx.protocol = crate::iop::context::ProofProtocol::LegacyV1;
    assert!(
        decode_verifier::<F, Pcs>(&encode_artifact(ArtifactKind::Verifier, &downgraded)?).is_err()
    );
    let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::verify::<F, dp_crypto::arkyper::transcript::blake3::Blake3Transcript, Pcs>(
            &downgraded.verifier_ctx,
            proof.clone(),
            io.clone(),
        )
    }));
    assert!(
        !matches!(rejected, Ok(Ok(()))),
        "Protocol downgrade was accepted"
    );
    let mut wrong_prompt = tokens;
    wrong_prompt[0] = 4usize.into();
    assert!(verifier.verify(proof, wrong_prompt, io).is_err());
    Ok(())
}

#[test]
fn qwen35_snapshot_preserves_large_norm_parameters_and_changes_with_arithmetic() -> Result<()> {
    let mut norm = QwenNorm::<Element>::new(None, 1e-6, 4, NormKind::L2)?;
    norm.quant = Some(NormQuant {
        dimension: 4,
        epsilon_scaled: 1u128 << 100,
        numerator: (1u128 << 110) + 17,
        multiplier_fraction_bits: 20,
        input_abs_bound: 2048,
    });
    let before = layer_quantization_summary(&Layer::QwenNorm(norm.clone()))?;
    assert_eq!(
        before["quant"]["numerator"],
        ((1u128 << 110) + 17).to_string()
    );
    norm.quant.as_mut().unwrap().numerator += 1;
    let after = layer_quantization_summary(&Layer::QwenNorm(norm))?;
    assert_ne!(
        Sha256::digest(serde_json::to_vec(&before)?),
        Sha256::digest(serde_json::to_vec(&after)?)
    );
    Ok(())
}
