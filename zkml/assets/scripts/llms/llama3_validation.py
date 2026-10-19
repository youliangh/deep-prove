#!/usr/bin/env python3
"""Reproducible local Llama3 reference, fixtures and streaming accuracy checks.

Model files are always loaded locally. Only `prepare` downloads WikiText-2.
Rust extract-logits --output-dir writes the same row-major f32 file convention.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

import numpy as np


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, local_files_only=True)


def load_model(args):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    if getattr(args, "omit_epsilon", False):
        for module in model.modules():
            if hasattr(module, "variance_epsilon"):
                module.variance_epsilon = 0.0
    return model


def prepare(args):
    from datasets import load_dataset

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tok = tokenizer(args.model)
    manifest = {"dataset": "Salesforce/wikitext", "config": "wikitext-2-raw-v1",
                "length": 256, "selection": "first nonempty rows with >=256 tokens"}
    for split, count, label in [("train", 32, "calibration"), ("validation", 4, "development"),
                                ("test", 25, "test")]:
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split,
                               cache_dir=str(output / "dataset-cache"))
        samples, indices = [], []
        for index, row in enumerate(dataset):
            if not row["text"].strip():
                continue
            ids = tok.encode(row["text"], add_special_tokens=True)
            if len(ids) < 256:
                continue
            samples.append(ids[:256])
            indices.append(index)
            if len(samples) == count:
                break
        if len(samples) != count:
            raise RuntimeError(f"Insufficient {split} samples: {len(samples)}")
        write_json(output / f"{label}.json", samples)
        manifest[label] = {"split": split, "row_indices": indices,
                           "fingerprint": dataset._fingerprint}
    tests = read_json(output / "test.json")
    prompts = [
        {"name": "factual", "prompt_tokens": tok.encode("The capital of France is"),
         "max_new_tokens": 32},
        {"name": "story", "prompt_tokens": tok.encode("Once upon a time, in a small village,"),
         "max_new_tokens": 64},
        {"name": "context256", "prompt_tokens": tests[0][:192], "max_new_tokens": 64},
    ]
    write_json(output / "prompts.json", prompts)
    write_json(output / "smoke.json", [tok.encode("The capital of France is"),
                                       tok.encode("Once upon a time, in a small village,")])
    write_json(output / "data-manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def reference(args):
    import torch

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model = load_model(args)
    prefix = "hf_noeps" if args.omit_epsilon else "hf"
    records = []
    with torch.inference_mode():
        for index, ids in enumerate(read_json(args.samples)):
            values = model(torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite reference logits in sample {index}")
            filename = f"{prefix}_{index:03d}.f32"
            values.astype("<f4", copy=False).tofile(output / filename)
            records.append({"sample": index, "mode": prefix, "file": filename,
                            "shape": list(values.shape), "tokens": ids})
            print(f"reference {index + 1}: {values.shape}", flush=True)
    metadata = {"records": records, "model": str(Path(args.model).resolve()),
                "dtype": "float32", "attention": "eager", "threads": args.threads,
                "seed": 42, "omit_epsilon": args.omit_epsilon,
                "versions": {name: importlib.metadata.version(name) for name in
                             ["torch", "transformers", "tokenizers", "numpy", "safetensors"]},
                "hashes": {name: sha256(Path(args.model) / name) for name in
                           ["config.json", "tokenizer.json", "model.safetensors"]}}
    write_json(output / "manifest.json", metadata)


def generate(args):
    import torch

    model = load_model(args)
    tok = tokenizer(args.model)
    records = []
    with torch.inference_mode():
        for case in read_json(args.prompts):
            ids = case["prompt_tokens"]
            output = model.generate(
                torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                max_new_tokens=case["max_new_tokens"], do_sample=False,
                temperature=None, top_p=None, pad_token_id=tok.eos_token_id,
            )[0].tolist()
            records.append({**case, "generated_tokens": output[len(ids):],
                            "prompt": tok.decode(ids, skip_special_tokens=True),
                            "text": tok.decode(output[len(ids):], skip_special_tokens=True)})
            print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    write_json(args.output, records)


def nll(values, tokens):
    values = values[:len(tokens) - 1].astype(np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    return float((np.log(np.exp(shifted).sum(axis=1)) -
                  shifted[np.arange(len(tokens) - 1), tokens[1:]]).sum())


def compare(args):
    reference_dir, candidate_dir = Path(args.reference), Path(args.candidate)
    reference_manifest = read_json(reference_dir / "manifest.json")
    candidate_manifest = read_json(candidate_dir / "manifest.json")
    reference_hashes = reference_manifest.get("hashes")
    candidate_hashes = candidate_manifest.get("hashes")
    source_identity_checked = reference_hashes is not None and candidate_hashes is not None
    if source_identity_checked:
        for name in ["model.safetensors", "config.json", "tokenizer.json"]:
            if name not in reference_hashes or name not in candidate_hashes:
                raise ValueError(f"Incomplete source identity hashes: {name}")
            if reference_hashes[name] != candidate_hashes[name]:
                raise ValueError(f"Source identity hash mismatch: {name}")

    expected_quantization = getattr(args, "expected_quantization_sha256", None)
    if expected_quantization is not None:
        expected_quantization = expected_quantization.lower()
        if len(expected_quantization) != 64 or any(c not in "0123456789abcdef" for c in expected_quantization):
            raise ValueError("Expected quantization SHA256 must contain 64 hexadecimal digits")
        if args.mode != "int":
            raise ValueError("An expected quantization hash requires --mode int")
    quantization_hash = None
    if args.mode == "int":
        quantization_hash = sha256(candidate_dir / "quantization.json")
        if quantization_hash != candidate_manifest.get("quantization_sha256"):
            raise ValueError("Candidate quantization.json does not match its manifest SHA256")
        if expected_quantization is not None and quantization_hash != expected_quantization:
            raise ValueError("Candidate quantization SHA256 differs from the externally pinned calibration")

    reference_records = reference_manifest["records"]
    candidate_records = [r for r in candidate_manifest["records"]
                         if r["mode"] == args.mode]
    candidates = {r["sample"]: r for r in candidate_records}
    expected = {r["sample"] for r in reference_records}
    if len(expected) != len(reference_records) or len(candidates) != len(candidate_records):
        raise ValueError("Duplicate sample IDs")
    if set(candidates) != expected:
        raise ValueError("Candidate and reference sample IDs must match exactly")
    totals = {"tokens": 0, "logit_rows": 0, "cosine": 0.0, "matching": 0, "reference_nll": 0.0,
              "candidate_nll": 0.0, "squared_error": 0.0, "squared_reference": 0.0}
    samples = []
    for ref in reference_records:
        cand = candidates[ref["sample"]]
        if ref["tokens"] != cand["tokens"]:
            raise ValueError("Logits must be compared under identical token histories")
        rows, columns = ref["shape"]
        actual_rows, actual_columns = cand["shape"]
        if len(ref["tokens"]) < 2 or rows != len(ref["tokens"]):
            raise ValueError("Reference must contain one logit row per input token and >=2 tokens")
        if actual_rows < rows or actual_columns < columns:
            raise ValueError("Candidate shape smaller than the reference")
        for path, elements in [(reference_dir / ref["file"], rows * columns),
                               (candidate_dir / cand["file"], actual_rows * actual_columns)]:
            if path.stat().st_size != elements * 4:
                raise ValueError(f"Invalid float32 logit file size: {path}")
        a = np.memmap(reference_dir / ref["file"], dtype="<f4", mode="r", shape=(rows, columns))
        b = np.memmap(candidate_dir / cand["file"], dtype="<f4", mode="r",
                      shape=(actual_rows, actual_columns))[:rows, :columns]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Non-finite logits")
        valid = len(ref["tokens"]) - 1
        # Every real row predicts a next token, including the final prompt row.
        # Only NLL lacks a ground-truth target for that final row.
        af, bf = a.astype(np.float64), b.astype(np.float64)
        cosine = np.einsum("ij,ij->i", af, bf) / np.maximum(
            np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1), 1e-30)
        matches = int((af.argmax(axis=1) == bf.argmax(axis=1)).sum())
        rnll, cnll = nll(a, ref["tokens"]), nll(b, ref["tokens"])
        squared_error = float(np.square(af - bf).sum())
        squared_reference = float(np.square(af).sum())
        samples.append({"sample": ref["sample"], "tokens": valid, "logit_rows": rows,
                        "cosine": float(cosine.mean()), "top1": matches / rows,
                        "relative_l2": math.sqrt(squared_error / squared_reference),
                        "reference_ppl": math.exp(rnll / valid),
                        "candidate_ppl": math.exp(cnll / valid)})
        for key, value in [("tokens", valid), ("logit_rows", rows), ("cosine", float(cosine.sum())),
                           ("matching", matches), ("reference_nll", rnll),
                           ("candidate_nll", cnll), ("squared_error", squared_error),
                           ("squared_reference", squared_reference)]:
            totals[key] += value
        print(json.dumps(samples[-1]), flush=True)
    count = totals["tokens"]
    logit_rows = totals["logit_rows"]
    if not count:
        raise ValueError("No valid prediction tokens")
    result = {"mode": args.mode, "tokens": count, "logit_rows": logit_rows,
              "cosine": totals["cosine"] / logit_rows,
              "top1": totals["matching"] / logit_rows,
              "relative_l2": math.sqrt(totals["squared_error"] / totals["squared_reference"]),
              "reference_ppl": math.exp(totals["reference_nll"] / count),
              "candidate_ppl": math.exp(totals["candidate_nll"] / count), "samples": samples,
              "source_identity_checked": source_identity_checked,
              "reference_source_hashes": reference_hashes, "candidate_source_hashes": candidate_hashes,
              "quantization_sha256": quantization_hash,
              "expected_quantization_sha256": expected_quantization}
    result["ppl_ratio"] = result["candidate_ppl"] / result["reference_ppl"]
    is_float = args.mode == "float"
    result["passed"] = (result["cosine"] >= (0.99999 if is_float else 0.99) and
                         (result["relative_l2"] <= 0.001 if is_float else
                          result["top1"] >= 0.90 and result["ppl_ratio"] <= 1.10))
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/home/yliangh/models/meta-llama/Llama-3.2-1B")
    parser.add_argument("--threads", type=int, default=8)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--output", required=True)
    ref = commands.add_parser("reference")
    ref.add_argument("--samples", required=True)
    ref.add_argument("--output", required=True)
    ref.add_argument("--omit-epsilon", action="store_true")
    gen = commands.add_parser("generate")
    gen.add_argument("--prompts", required=True)
    gen.add_argument("--output", required=True)
    comp = commands.add_parser("compare")
    comp.add_argument("--reference", required=True)
    comp.add_argument("--candidate", required=True)
    comp.add_argument("--mode", choices=["float", "int", "hf_noeps"], required=True)
    comp.add_argument("--expected-quantization-sha256",
                      help="For int mode, require the quantization hash frozen during development")
    comp.add_argument("--output", required=True)
    args = parser.parse_args()
    {"prepare": prepare, "reference": reference, "generate": generate, "compare": compare}[args.command](args)


if __name__ == "__main__":
    main()
