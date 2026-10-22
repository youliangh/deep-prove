#!/usr/bin/env python3
"""Offline Qwen3.5 text reference, frozen fixtures, traces and streaming comparison.

Install requirements-qwen35.txt in a separate environment. All model/data inputs
are local. `prepare` reads raw WikiText parquet files and retokenizes them; it
never reuses token IDs from another model. Large outputs belong outside the repo.

The .f32 format is little-endian, contiguous row-major, with shapes/token histories
in manifest.json. Example:
  python qwen35_validation.py prepare --dataset-dir /path/wikitext-2-raw-v1 --output /tmp/qwen/data
  python qwen35_validation.py reference --samples /tmp/qwen/data/smoke.json --output /tmp/qwen/hf
  python qwen35_validation.py dump --samples /tmp/qwen/data/smoke.json --output /tmp/qwen/trace
  python qwen35_validation.py generate --prompts /tmp/qwen/data/prompts.json --output /tmp/qwen/generation.json
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
import time

import numpy as np


MODEL_DEFAULT = "/home/yliangh/models/Qwen/Qwen3.5-0.8B"
CONTEXT_LIMIT = 256
EOS_IDS = [248044, 248046]
PAD_ID = 248044
PINNED = {"torch": "2.6.0+cpu", "transformers": "5.3.0", "tokenizers": "0.22.2",
          "numpy": "2.2.6", "safetensors": "0.6.2", "jinja2": "3.1.6"}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer(model):
    # The checkpoint says Qwen2Tokenizer although its JSON regex includes \p{M}.
    # Bypass class reconstruction and preserve the serialized tokenizer exactly.
    from transformers import TokenizersBackend
    return TokenizersBackend.from_pretrained(model, local_files_only=True)


def chat_tokens(tok, text):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False,
    )
    return tok.encode(rendered, add_special_tokens=False)


def checked_samples(path):
    samples = read_json(path)
    if not isinstance(samples, list) or not samples:
        raise ValueError("Samples must be a nonempty list of token lists")
    for ids in samples:
        if not isinstance(ids, list) or not 1 <= len(ids) <= CONTEXT_LIMIT:
            raise ValueError("Each sample must contain 1..256 tokens")
        if any(type(i) is not int or not 0 <= i < 248320 for i in ids):
            raise ValueError("Token IDs must be integers in the Qwen3.5 vocabulary")
    return samples



def prepare_boundary(args):
    """Freeze a 192+64 numerical/cache case, without counting it as semantic evidence."""
    output = Path(args.output)
    if (output / "boundary-manifest.json").exists():
        raise ValueError("Boundary case already frozen; choose a new output directory")
    cases = [case for case in read_json(args.generation_reference) if case["name"] == "context256"]
    if len(cases) != 1:
        raise ValueError("Expected exactly one context256 numerical reference")
    case = cases[0]
    prompt, generated = case["prompt_tokens"], case["generated_tokens"]
    if len(prompt) != 192 or len(generated) != 64 or any(x in EOS_IDS for x in generated):
        raise ValueError("Boundary reference must produce exactly 64 non-EOS tokens after 192 inputs")
    write_json(output / "prompt-tokens.json", [prompt])
    write_json(output / "full-history.json", [prompt + generated[:-1]])
    write_json(output / "boundary-manifest.json", {
        "purpose": "numerical cache boundary only; not semantic acceptance",
        "prompt_tokens": 192, "generated_tokens": 64, "computed_logit_rows": 255,
        "reference_sha256": sha256(args.generation_reference),
        "generated_ids": generated,
        "hashes": {name: sha256(output / name) for name in ["prompt-tokens.json", "full-history.json"]},
    })


def load_model(args):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise RuntimeError(f"Reference version mismatch: expected {PINNED}, got {versions}")
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    config = Qwen3_5TextConfig.from_pretrained(args.model, local_files_only=True)
    model, info = Qwen3_5ForCausalLM.from_pretrained(
        args.model, config=config, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True,
    )
    problems = {k: v for k, v in info.items() if v}
    if problems:
        raise RuntimeError(f"Reference weights did not load exactly: {problems}")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise RuntimeError("The output head must share the token embedding")
    return model.eval()


def source_metadata(args):
    root = Path(args.model)
    files = ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
             "model.safetensors.index.json"]
    index = read_json(root / "model.safetensors.index.json")
    files += sorted(set(index["weight_map"].values()))
    return {
        "model": str(root.resolve()), "dtype": "float32", "attention": "eager",
        "threads": args.threads, "seed": 42, "context_limit": CONTEXT_LIMIT,
        "eos_token_ids": EOS_IDS, "pad_token_id": PAD_ID,
        "versions": {name: importlib.metadata.version(name) for name in PINNED},
        "hashes": {name: sha256(root / name) for name in files},
        "reference_script_sha256": sha256(__file__),
    }


def prepare(args):
    import pyarrow.parquet as parquet
    from tokenizers import Tokenizer
    output = Path(args.output)
    if (output / "data-manifest.json").exists():
        raise ValueError("Frozen fixtures already exist; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(args.model)
    raw = Tokenizer.from_file(str(Path(args.model) / "tokenizer.json"))
    # Check both multilingual and combining-character paths against the JSON.
    for text in ["Hello 你好", "e\u0301 déjà vu", "0123456789\n\t", "नमस्ते दुनिया"]:
        if tok.encode(text, add_special_tokens=False) != raw.encode(text, add_special_tokens=False).ids:
            raise AssertionError("Tokenizer no longer matches the checkpoint JSON")
    manifest = {
        "schema": "qwen35-fixtures-v1", "dataset": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1", "dataset_revision": args.dataset_revision,
        "length": CONTEXT_LIMIT, "tokenizer_sha256": sha256(Path(args.model) / "tokenizer.json"),
        "selection": "first nonempty rows with >=256 tokens; no BOS; truncate to first 256",
        "split_policy": "calibration=train; development=validation; test=test; never tune on test",
    }
    for split, count, label in [("train", 32, "calibration"), ("validation", 4, "development"),
                                 ("test", 25, "test")]:
        paths = sorted(Path(args.dataset_dir).glob(f"{split}-*.parquet"))
        if not paths:
            raise ValueError(f"Missing local {split} parquet in {args.dataset_dir}")
        samples, indices = [], []
        row_index = 0
        for path in paths:
            for batch in parquet.ParquetFile(path).iter_batches(columns=["text"], batch_size=256):
                for text in batch.column(0).to_pylist():
                    if text and text.strip():
                        ids = tok.encode(text, add_special_tokens=False)
                        if len(ids) >= CONTEXT_LIMIT:
                            samples.append(ids[:CONTEXT_LIMIT])
                            indices.append(row_index)
                    row_index += 1
                    if len(samples) == count:
                        break
                if len(samples) == count:
                    break
            if len(samples) == count:
                break
        if len(samples) != count:
            raise ValueError(f"Insufficient {split} samples: {len(samples)}")
        write_json(output / f"{label}.json", samples)
        manifest[label] = {"split": split, "row_indices": indices,
                           "sources": {p.name: sha256(p) for p in paths},
                           "samples_sha256": sha256(output / f"{label}.json")}
    prompts = []
    for name, text, answer, budget in [
        ("en_factual", "What is the capital of France? Answer briefly.", "Paris", 32),
        ("en_arithmetic", "What is 2 + 3? Reply with just the number.", "5", 32),
        ("zh_factual", "中国的首都是哪里？请简短回答。", "北京", 32),
        ("zh_arithmetic", "只回答一个数字：2加3等于几？", "5", 32),
        ("zh_explanation", "用一句话解释为什么白天能看见太阳。", None, 64),
        ("en_instruction", "Write one short sentence about a cat sleeping in the sun.", None, 64),
    ]:
        prompts.append({"name": name, "prompt": text, "prompt_tokens": chat_tokens(tok, text),
                        "max_new_tokens": budget, "expected_substring": answer})
    prompts.append({"name": "context256", "prompt_tokens": read_json(output / "test.json")[0][:192],
                    "max_new_tokens": 64, "expected_substring": None})
    write_json(output / "prompts.json", prompts)
    write_json(output / "smoke.json", [p["prompt_tokens"] for p in prompts[:4]])
    write_json(output / "primitive.json", [chat_tokens(tok, "Hello.")])
    manifest["fixtures"] = {name: sha256(output / name)
                            for name in ["prompts.json", "smoke.json", "primitive.json"]}
    write_json(output / "data-manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def prepare_prompts(args):
    """Freeze semantic cases separately from numerical WikiText continuation."""
    tok = tokenizer(args.model)
    output = Path(args.output)
    if (output / "generation-manifest.json").exists():
        raise ValueError("Generation fixtures already frozen; choose a new output directory")
    prompts = []
    definitions = [
        ("en_factual", "What is the capital of France? Answer briefly.",
         {"exact": ["Paris", "The capital of France is Paris"]}, 32),
        ("zh_factual", "中国的首都是哪里？请简短回答。",
         {"exact": ["北京", "中国的首都是北京"]}, 32),
        ("en_arithmetic", "What is 2 + 3? Reply with just the number.", {"exact": ["5"]}, 32),
        ("zh_arithmetic", "只回答一个数字：2加3等于几？", {"exact": ["5"]}, 32),
        ("en_translation", "Translate 'Good morning' into Chinese. Give only the translation.",
         {"exact": ["早上好", "早安", "上午好"]}, 32),
        ("zh_translation", "把“谢谢”翻译成英语，只给出翻译。",
         {"exact": ["Thank you", "Thanks"]}, 32),
        ("en_creation", "Write one short sentence about a cat sleeping in the sun.",
         {"keyword_groups": [["cat"], ["sun"]], "one_sentence": True}, 64),
        ("zh_creation", "用一句话解释为什么白天能看见太阳。",
         {"keyword_groups": [["太阳"], ["光", "发亮"]], "one_sentence": True}, 64),
    ]
    for name, prompt, rubric, budget in definitions:
        prompts.append({"name": name, "prompt": prompt, "prompt_tokens": chat_tokens(tok, prompt),
                        "max_new_tokens": budget, "rubric": rubric, "purpose": "semantic"})
    # Find a naturally tokenized chat prompt of exactly 192 tokens. The final
    # question and generation markers are intact; 64 generated tokens reach 256.
    for count in range(300):
        prompt = ("Background notes: " + "The sky is blue and grass is green. " * 10 +
                  "blue " * count + "\nIgnore these notes. What is the capital of France? Answer briefly.")
        ids = chat_tokens(tok, prompt)
        if len(ids) == 192:
            break
    else:
        raise RuntimeError("Could not construct a 192-token chat boundary fixture")
    prompts.append({"name": "context256_chat", "prompt": prompt, "prompt_tokens": ids,
                    "max_new_tokens": 64, "purpose": "semantic",
                    "rubric": {"exact": ["Paris", "The capital of France is Paris"]}})
    rubric = {"schema": "qwen35-generation-rubric-v1", "required_cases": 8,
              "boundary_case": "192 prompt tokens + 64 token budget",
              "automatic": "nonempty; no replacement character; no 4-word or 16-character ngram repeated three times; per-case exact/keyword/one-sentence checks",
              "manual_review": "Every semantic answer must be coherent, responsive, and match the requested language; creation must satisfy the requested topic. Automatic checks alone do not certify meaning.",
              "raw_wikitext": "Numerical/context stress only; repetition by the official greedy reference is not a semantic pass"}
    write_json(output / "validation-prompts.json", prompts)
    write_json(output / "generation-rubric.json", rubric)
    write_json(output / "generation-manifest.json", {
        "schema": "qwen35-generation-fixtures-v1", "cases": len(prompts),
        "tokenizer_sha256": sha256(Path(args.model) / "tokenizer.json"),
        "hashes": {name: sha256(output / name) for name in ["validation-prompts.json", "generation-rubric.json"]},
    })
    print(f"Frozen {len(prompts)} chat fixtures in {output}")


def content_check(text, rubric):
    normalized = text.strip().strip(".。!！?？\"'“”‘’ ").casefold()
    if not normalized or "\ufffd" in text:
        return False
    words = re.findall(r"\w+", normalized)
    if len(words) >= 12 and max(Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3)).values()) >= 3:
        return False
    compact = re.sub(r"\s+", "", normalized)
    if len(compact) >= 48 and max(Counter(compact[i:i + 16] for i in range(len(compact) - 15)).values()) >= 3:
        return False
    if "exact" in rubric and normalized not in [x.casefold() for x in rubric["exact"]]:
        return False
    if any(not any(keyword.casefold() in normalized for keyword in group)
           for group in rubric.get("keyword_groups", [])):
        return False
    if rubric.get("one_sentence") and len([x for x in re.split(r"[.!?。！？]+", text) if x.strip()]) != 1:
        return False
    return True


def prepare_calibration(args):
    """Interleave independent Chinese training passages with WikiText train rows."""
    output = Path(args.output)
    if (output / "calibration-manifest.json").exists():
        raise ValueError("Bilingual calibration already frozen; choose a new output directory")
    data_dir = Path(args.data_dir)
    original = read_json(data_dir / "data-manifest.json")
    source = data_dir / "calibration.json"
    if original["calibration"]["split"] != "train" or sha256(source) != original["calibration"]["samples_sha256"]:
        raise ValueError("Calibration must derive from the frozen WikiText TRAIN fixture")
    wiki = checked_samples(source)
    passages_path = Path(__file__).with_name("qwen35_calibration_zh.json")
    passages = read_json(passages_path)
    if len(passages) != 16 or len(wiki) < 16:
        raise ValueError("Expected sixteen authored passages and at least sixteen WikiText rows")
    tok = tokenizer(args.model)
    samples, provenance = [], []
    for index, passage in enumerate(passages):
        samples.append(wiki[index])
        provenance.append({"sample": len(samples) - 1, "source": "wikitext/train",
                           "row": original["calibration"]["row_indices"][index]})
        ids = chat_tokens(tok, "请阅读下面的学习资料，并概括其中的要点。\n" + passage["text"])
        if not 1 <= len(ids) <= CONTEXT_LIMIT:
            raise ValueError(f"Chinese training passage exceeds 256 tokens: {passage['name']} ({len(ids)})")
        samples.append(ids)
        provenance.append({"sample": len(samples) - 1, "source": "authored-training-only",
                           "name": passage["name"], "tokens": len(ids)})
    write_json(output / "calibration.json", samples)
    write_json(output / "calibration-manifest.json", {
        "schema": "qwen35-bilingual-calibration-v1", "count": 32,
        "order": "alternate WikiText train and authored Chinese/mixed chat, 16 of each",
        "exclusion": "No development/test rows, test prompts, model-generated test outputs, or acceptance answers",
        "hashes": {"wiki_calibration": sha256(source), "authored_passages": sha256(passages_path),
                   "tokenizer.json": sha256(Path(args.model) / "tokenizer.json"),
                   "calibration.json": sha256(output / "calibration.json")},
        "samples": provenance,
    })
    print(f"Frozen {len(samples)} alternating bilingual training samples in {output}")


def reference(args):
    import torch
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    samples = checked_samples(args.samples)
    model = load_model(args)
    records = []
    with torch.inference_mode():
        for index, ids in enumerate(samples):
            started = time.monotonic()
            hidden = model.model(torch.tensor([ids]), use_cache=False).last_hidden_state[0]
            filename = f"hf_{index:03d}.f32"
            with open(output / filename, "wb") as handle:
                # The head is shared with embedding; stream rows instead of holding
                # an entire 256 x 248320 tensor or converting it all to float64.
                for row in range(0, len(ids), args.rows_per_chunk):
                    values = model.lm_head(hidden[row:row + args.rows_per_chunk]).float().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite reference logits for sample {index}")
                    values.astype("<f4", copy=False).tofile(handle)
            records.append({"sample": index, "mode": "hf", "file": filename,
                            "shape": [len(ids), model.config.vocab_size], "tokens": ids,
                            "sha256": sha256(output / filename)})
            print(f"reference {index + 1}/{len(samples)}: {len(ids)} rows, "
                  f"{time.monotonic() - started:.2f}s", flush=True)
    write_json(output / "manifest.json", {**source_metadata(args), "records": records,
                                         "samples_sha256": sha256(args.samples)})


def generate(args):
    import torch
    model, tok = load_model(args), tokenizer(args.model)
    records = []
    with torch.inference_mode():
        for case in read_json(args.prompts):
            ids = case["prompt_tokens"]
            budget = case["max_new_tokens"]
            if not ids or budget < 1 or len(ids) + budget > CONTEXT_LIMIT:
                raise ValueError("Generation prompt + budget must fit in 256 tokens")
            started = time.monotonic()
            output = model.generate(
                torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                max_new_tokens=budget, do_sample=False, temperature=None, top_p=None, top_k=None,
                eos_token_id=EOS_IDS, pad_token_id=PAD_ID, use_cache=True,
            )[0].tolist()[len(ids):]
            text = tok.decode(output, skip_special_tokens=True)
            expected = case.get("expected_substring")
            rubric = case.get("rubric", {"keyword_groups": [[expected]]} if expected else {})
            numerical_only = case.get("purpose") == "numerical" or case["name"] == "context256"
            record = {**case, "generated_tokens": output, "text": text,
                      "elapsed_seconds": time.monotonic() - started,
                      "stopped_on_eos": bool(output and output[-1] in EOS_IDS),
                      "content_check": None if numerical_only else content_check(text, rubric),
                      "manual_semantic_review_required": not numerical_only}
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    write_json(args.output, records)
    write_json(str(args.output) + ".metadata.json", source_metadata(args))
    if any(r["content_check"] is False for r in records):
        raise SystemExit("At least one reference generation failed the content check")


def dump(args):
    """Capture all decoder outputs plus primitive inputs/outputs for selected layers."""
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
    model = load_model(args)
    ids = checked_samples(args.samples)[args.sample]
    if len(ids) > 32:
        raise ValueError("Primitive dumps require <=32 tokens; use reference for full context")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tensors, handles, replacements = [], [], []

    def save(name, value):
        if isinstance(value, (tuple, list)):
            for i, item in enumerate(value):
                save(f"{name}.{i}", item)
            return
        if not isinstance(value, torch.Tensor):
            return
        array = value.detach().float().cpu().numpy().astype("<f4", copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite primitive: {name}")
        filename = name.replace("/", "_") + ".f32"
        array.tofile(output / filename)
        tensors.append({"name": name, "file": filename, "shape": list(array.shape),
                        "sha256": sha256(output / filename)})

    wanted = {f"model.layers.{i}" for i in range(model.config.num_hidden_layers)}
    wanted |= {"model.embed_tokens", "model.norm", "model.rotary_emb", "lm_head"}
    prefixes = tuple(f"model.layers.{i}." for i in args.primitive_layers)
    for name, module in model.named_modules():
        if name in wanted or name.startswith(prefixes):
            handles.append(module.register_forward_hook(
                lambda module, values, result, name=name: save(name + ".output", result)))
            if name.startswith(prefixes):
                handles.append(module.register_forward_pre_hook(
                    lambda module, values, name=name: save(name + ".input", values)))
    for index in args.primitive_layers:
        layer = model.model.layers[index]
        if not hasattr(layer, "linear_attn"):
            continue
        attention = layer.linear_attn
        original = attention.chunk_gated_delta_rule

        def traced(query, key, value, *, g, beta, original=original, index=index, **kwargs):
            prefix = f"model.layers.{index}.delta"
            for name, tensor in [("q", query), ("k", key), ("v", value), ("g", g), ("beta", beta)]:
                save(prefix + "." + name, tensor)
            result, final = original(query, key, value, g=g, beta=beta, **kwargs)
            q = l2norm(query.float()) / math.sqrt(query.shape[-1])
            k = l2norm(key.float())
            state = torch.zeros(query.shape[0], query.shape[2], query.shape[3], value.shape[3])
            states, deltas, memories, outputs = [], [], [], []
            for position in range(query.shape[1]):
                state = state * g[:, position].exp()[..., None, None]
                memory = (state * k[:, position, :, :, None]).sum(-2)
                delta = beta[:, position, :, None] * (value[:, position] - memory)
                state = state + k[:, position, :, :, None] * delta[..., None, :]
                outputs.append((state * q[:, position, :, :, None]).sum(-2))
                states.append(state.clone())
                memories.append(memory)
                deltas.append(delta)
            replay = torch.stack(outputs, dim=1)
            torch.testing.assert_close(replay, result, rtol=2e-4, atol=2e-5)
            for name, tensor in [("q_normalized_scaled", q), ("k_normalized", k),
                                 ("alpha", g.exp()), ("memory", torch.stack(memories, dim=1)),
                                 ("delta", torch.stack(deltas, dim=1)),
                                 ("state", torch.stack(states, dim=1)), ("output", result)]:
                save(prefix + "." + name, tensor)
            return result, final
        attention.chunk_gated_delta_rule = traced
        replacements.append((attention, original))
    try:
        with torch.inference_mode():
            result = model(torch.tensor([ids]), use_cache=True)
            save("logits", result.logits)
            for i in args.primitive_layers:
                if model.config.layer_types[i] == "linear_attention":
                    save(f"cache.{i}.conv", result.past_key_values.conv_states[i])
                    save(f"cache.{i}.recurrent", result.past_key_values.recurrent_states[i])
    finally:
        for handle in handles:
            handle.remove()
        for attention, original in replacements:
            attention.chunk_gated_delta_rule = original
    write_json(output / "tensor-manifest.json", {**source_metadata(args), "tokens": ids,
                                                "tensors": tensors})
    print(f"Saved {len(tensors)} tensors for {len(ids)} tokens to {output}")


def cache_check(args):
    import torch
    model = load_model(args)
    results = []
    with torch.inference_mode():
        for index, ids in enumerate(checked_samples(args.samples)):
            # Compare all rows, including the last input position. This tests
            # both convolution/recurrent states and rotary/KV cache positions.
            hidden = model.model(torch.tensor([ids]), use_cache=False).last_hidden_state
            cache, maximum, error, base, matches = None, 0.0, 0.0, 0.0, 0
            for position, token in enumerate(ids):
                out = model(torch.tensor([[token]]), past_key_values=cache, use_cache=True)
                cache = out.past_key_values
                ref = model.lm_head(hidden[:, position:position + 1])
                difference = out.logits - ref
                maximum = max(maximum, difference.abs().max().item())
                error += difference.square().sum().item()
                base += ref.square().sum().item()
                matches += int(out.logits.argmax(-1).item() == ref.argmax(-1).item())
            result = {"sample": index, "tokens": len(ids), "max_abs": maximum,
                      "relative_l2": math.sqrt(error / base), "top1": matches / len(ids)}
            result["passed"] = result["relative_l2"] <= 1e-4 and result["top1"] == 1.0
            results.append(result)
            print(json.dumps(result), flush=True)
    write_json(args.output, {"results": results, "passed": all(x["passed"] for x in results)})
    if not all(x["passed"] for x in results):
        raise SystemExit(1)


def checked_array(root, record):
    shape = record["shape"]
    if len(shape) != 2 or any(type(x) is not int or x <= 0 for x in shape):
        raise ValueError("Logits must have a positive 2D shape")
    root = Path(root).resolve()
    path = (root / record["file"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest file must remain inside its artifact directory")
    if path.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"Invalid float32 file size: {path}")
    if record.get("sha256") is not None and sha256(path) != record["sha256"]:
        raise ValueError(f"Logit digest mismatch: {path}")
    return np.memmap(path, dtype="<f4", mode="r", shape=tuple(shape))


def row_metrics(a, b, targets):
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite logits")
    a, b = a.astype(np.float64), b.astype(np.float64)
    cosine = np.einsum("ij,ij->i", a, b) / np.maximum(
        np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-30)
    def nll(values):
        values = values[:len(targets)]
        if not len(targets):
            return 0.0
        maximum = values.max(axis=1)
        return float((np.log(np.exp(values - maximum[:, None]).sum(axis=1)) + maximum -
                      values[np.arange(len(targets)), targets]).sum())
    return {"rows": len(a), "tokens": len(targets), "cosine": float(cosine.sum()),
            "matches": int((a.argmax(axis=1) == b.argmax(axis=1)).sum()),
            "squared_error": float(np.square(a - b).sum()), "squared_reference": float(np.square(a).sum()),
            "reference_nll": nll(a), "candidate_nll": nll(b), "max_abs": float(np.abs(a - b).max())}


def summarize(totals):
    if not totals["tokens"] or not totals["rows"] or not totals["squared_reference"]:
        raise ValueError("Comparison requires nonempty, nonzero logits and next-token targets")
    return {"tokens": totals["tokens"], "logit_rows": totals["rows"],
            "cosine": totals["cosine"] / totals["rows"],
            "top1": totals["matches"] / totals["rows"],
            "relative_l2": math.sqrt(totals["squared_error"] / totals["squared_reference"]),
            "max_abs": totals["max_abs"],
            "reference_ppl": math.exp(min(totals["reference_nll"] / totals["tokens"], 700)),
            "candidate_ppl": math.exp(min(totals["candidate_nll"] / totals["tokens"], 700)),
            "ppl_ratio": math.exp(min((totals["candidate_nll"] - totals["reference_nll"]) / totals["tokens"], 700))}


def compare(args):
    reference_dir, candidate_dir = Path(args.reference), Path(args.candidate)
    ref_manifest = read_json(reference_dir / "manifest.json")
    cand_manifest = read_json(candidate_dir / "manifest.json")
    identity_checked = False
    if ref_manifest.get("hashes") and cand_manifest.get("hashes"):
        # Every reference source must be bound by a candidate claiming identity.
        if any(cand_manifest["hashes"].get(k) != v for k, v in ref_manifest["hashes"].items()):
            raise ValueError("Reference and candidate model/tokenizer identity hashes differ")
        identity_checked = True
    if args.require_source_identity and not identity_checked:
        raise ValueError("Source identity hashes are required")
    quantization_hash = None
    if args.mode == "int":
        quantization_hash = sha256(candidate_dir / "quantization.json")
        if cand_manifest.get("quantization_sha256") != quantization_hash:
            raise ValueError("Candidate quantization manifest digest mismatch")
        if args.expected_quantization_sha256 and quantization_hash != args.expected_quantization_sha256:
            raise ValueError("Candidate differs from the frozen quantization digest")
    elif args.expected_quantization_sha256:
        raise ValueError("Frozen quantization hash requires --mode int")
    refs = ref_manifest["records"]
    records = [x for x in cand_manifest["records"] if x["mode"] == args.mode]
    candidates = {x["sample"]: x for x in records}
    if len(candidates) != len(records) or len({x["sample"] for x in refs}) != len(refs):
        raise ValueError("Duplicate sample IDs")
    if set(candidates) != {x["sample"] for x in refs}:
        raise ValueError("Candidate sample IDs must exactly match reference IDs")
    totals, samples = {}, []
    for ref in refs:
        cand = candidates[ref["sample"]]
        ids = ref["tokens"]
        candidate_history = cand["tokens"]
        if getattr(args, "candidate_generated_history", False):
            generated = cand.get("generated_tokens", [])
            if not generated or any(x in EOS_IDS for x in generated[:-1]):
                raise ValueError("Generated history is empty or continues after EOS")
            candidate_history = candidate_history + generated[:-1]
        if candidate_history != ids or len(ids) < 2 or ref["shape"][0] != len(ids):
            raise ValueError("Comparison requires identical token histories and one row per token")
        a, b = checked_array(reference_dir, ref), checked_array(candidate_dir, cand)
        if b.shape[0] < a.shape[0] or b.shape[1] < a.shape[1]:
            raise ValueError("Candidate shape is smaller than the reference")
        subtotal = {}
        for row in range(0, len(ids), args.rows_per_chunk):
            end = min(row + args.rows_per_chunk, len(ids))
            metrics = row_metrics(a[row:end], b[row:end, :a.shape[1]], ids[row + 1:min(end + 1, len(ids))])
            for key, value in metrics.items():
                subtotal[key] = max(subtotal.get(key, 0), value) if key == "max_abs" else subtotal.get(key, 0) + value
        sample = {"sample": ref["sample"], **summarize(subtotal)}
        samples.append(sample)
        for key, value in subtotal.items():
            totals[key] = max(totals.get(key, 0), value) if key == "max_abs" else totals.get(key, 0) + value
        print(json.dumps(sample), flush=True)
    result = {"mode": args.mode, **summarize(totals), "samples": samples,
              "source_identity_checked": identity_checked, "quantization_sha256": quantization_hash}
    result["passed"] = (result["cosine"] >= 0.99999 and result["relative_l2"] <= 0.001
                        if args.mode in ("float", "hf") else
                        result["cosine"] >= 0.99 and result["top1"] >= 0.90 and result["ppl_ratio"] <= 1.10)
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_DEFAULT)
    parser.add_argument("--threads", type=int, default=4)
    commands = parser.add_subparsers(dest="command", required=True)
    boundary = commands.add_parser("prepare-boundary")
    boundary.add_argument("--generation-reference", required=True)
    boundary.add_argument("--output", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--dataset-dir", required=True)
    prep.add_argument("--dataset-revision", default="b08601e04326c79dfdd32d625aee71d232d685c3")
    prep.add_argument("--output", required=True)
    prompts = commands.add_parser("prepare-prompts")
    prompts.add_argument("--output", required=True)
    calibration = commands.add_parser("prepare-calibration")
    calibration.add_argument("--data-dir", required=True)
    calibration.add_argument("--output", required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--samples", required=True)
    ref.add_argument("--output", required=True)
    ref.add_argument("--rows-per-chunk", type=int, default=16)
    gen = commands.add_parser("generate")
    gen.add_argument("--prompts", required=True)
    gen.add_argument("--output", required=True)
    trace = commands.add_parser("dump")
    trace.add_argument("--samples", required=True)
    trace.add_argument("--sample", type=int, default=0)
    trace.add_argument("--primitive-layers", type=int, nargs="+", default=[0, 3])
    trace.add_argument("--output", required=True)
    cache = commands.add_parser("cache-check")
    cache.add_argument("--samples", required=True)
    cache.add_argument("--output", required=True)
    comp = commands.add_parser("compare")
    comp.add_argument("--reference", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--mode", choices=["float", "int", "hf"], required=True)
    comp.add_argument("--expected-quantization-sha256")
    comp.add_argument("--require-source-identity", action="store_true")
    comp.add_argument("--candidate-generated-history", action="store_true",
                      help="Compare cached generated rows against a full HF teacher-forced history")
    comp.add_argument("--rows-per-chunk", type=int, default=8)
    comp.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.threads < 1 or getattr(args, "rows_per_chunk", 1) < 1:
        parser.error("Thread and chunk counts must be positive")
    # No accidental network/model downloads during validation.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    {"prepare": prepare, "prepare-boundary": prepare_boundary, "prepare-prompts": prepare_prompts, "prepare-calibration": prepare_calibration,
     "reference": reference, "generate": generate, "dump": dump,
     "cache-check": cache_check, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
