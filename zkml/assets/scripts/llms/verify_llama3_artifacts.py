#!/usr/bin/env python3
"""Verify three frozen Llama artifacts, then reject isolated tampered copies.

No proving, weights, model download, or source changes are performed. All trust
anchors are explicit inputs from a separate setup/acceptance record. Relative
paths in the pins file are resolved against that file's parent directory.

Example pins JSON (repeat cases for all three accepted continuations):
{
  "schema": 1,
  "tokenizer": "/trusted/tokenizer.json",
  "tokenizer_sha256": "<64 hexadecimal digits from trusted setup>",
  "vocab_size": 128256,
  "eos_token": 128001,
  "cases": [{
    "name": "france",
    "artifacts_dir": "proofs/france",
    "verifier": "trusted/france-verifier.bin",
    "verifier_sha256": "<64 hexadecimal digits from trusted setup>",
    "prompt_tokens": [128000, 791, 6864, 315, 9822, 374],
    "expected_generated_tokens": [12345]
  }]
}

expected_generated_tokens is optional, but when present pins the previously
accepted continuation independently of its artifact report. Do not construct
trusted hashes from an untrusted artifact directory at verification time.
"""

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time


FILES = (
    "proof.bin", "io.bin", "prompt_tokens.json", "generated_tokens.json",
    "result.json", "generated.txt",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checked_digest(value):
    if not isinstance(value, str) or re.fullmatch("[0-9a-fA-F]{64}", value) is None:
        raise ValueError("Trust pins require an explicit 64-digit SHA256")
    return value.lower()


def resolve(base, value):
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def changed_hash(value):
    return ("0" if value[0] != "0" else "1") + value[1:]


def modify_json(directory, filename, callback):
    value = read_json(directory / filename)
    callback(value)
    write_json(directory / filename, value)


def change_generated(directory, callback, paired=True):
    tokens = read_json(directory / "generated_tokens.json")
    callback(tokens)
    write_json(directory / "generated_tokens.json", tokens)
    if paired:
        modify_json(directory, "result.json", lambda report: report.update(generated_tokens=tokens))


def change_prompt(directory, vocab):
    tokens = read_json(directory / "prompt_tokens.json")
    tokens[-1] = (tokens[-1] + 1) % vocab
    write_json(directory / "prompt_tokens.json", tokens)
    modify_json(directory, "result.json", lambda report: report.update(prompt_tokens=tokens))


def flip_last_byte(path):
    data = bytearray(path.read_bytes())
    if not data:
        raise ValueError(f"Cannot mutate empty binary: {path}")
    data[-1] ^= 1
    path.write_bytes(data)


def append_byte(path):
    with path.open("ab") as stream:
        stream.write(b"\0")


def mutation_cases(vocab, eos):
    def token_at(index):
        def mutate(tokens):
            position = index if index >= 0 else len(tokens) + index
            tokens[position] = (tokens[position] + 1) % vocab
        return mutate

    def replace_eos(tokens):
        tokens[-1] = eos if tokens[-1] != eos else (eos + 1) % vocab

    def text_pair(directory):
        modify_json(directory, "result.json", lambda report: report.update(
            generated_text=report["generated_text"] + " [tampered]"))
        text = (directory / "generated.txt").read_text()
        (directory / "generated.txt").write_text(text + " [tampered]")

    return [
        ("proof_byte", lambda d: flip_last_byte(d / "proof.bin"), None),
        ("proof_trailing_byte", lambda d: append_byte(d / "proof.bin"), "Trailing bytes"),
        ("io_byte", lambda d: flip_last_byte(d / "io.bin"), None),
        ("io_trailing_byte", lambda d: append_byte(d / "io.bin"), "Trailing bytes"),
        ("prompt_and_report", lambda d: change_prompt(d, vocab), "Artifact prompt differs"),
        ("generated_file_only", lambda d: change_generated(d, token_at(-1), False),
         "result.json generated tokens differ"),
        ("first_generated_and_report", lambda d: change_generated(d, token_at(0)),
         "Frozen generated tokens differ"),
        ("last_generated_and_report", lambda d: change_generated(d, token_at(-1)),
         "Frozen generated tokens differ"),
        ("eos_and_report", lambda d: change_generated(d, replace_eos),
         "Frozen generated tokens differ"),
        ("generated_length_and_report", lambda d: change_generated(d, lambda t: t.append(0)),
         "Frozen generated tokens differ"),
        ("claimed_text_and_text_file", text_pair, "generated text differs"),
        ("claimed_full_text", lambda d: modify_json(d, "result.json", lambda r: r.update(
            full_text=r["full_text"] + " [tampered]")), "full text differs"),
        ("generated_text_file", lambda d: (d / "generated.txt").write_text(
            (d / "generated.txt").read_text() + " [tampered]"), "generated.txt differs"),
        ("reported_verifier_hash", lambda d: modify_json(d, "result.json", lambda r: r.update(
            verifier_sha256="0" * 64)), "verifier fingerprint differs"),
    ]


def run_verifier(command, timeout, threads, expected_error=None, expect_success=False):
    started = time.monotonic()
    env = dict(os.environ, RAYON_NUM_THREADS=str(threads))
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, env=env)
    result = {
        "returncode": completed.returncode,
        "seconds": round(time.monotonic() - started, 3),
        "stderr": completed.stderr[-12000:],
    }
    if expect_success:
        if completed.returncode != 0:
            raise AssertionError(f"Original proof rejected: {result}")
        verified = json.loads(completed.stdout)
        if verified.get("verified") is not True or verified.get("text_verified") is not True:
            raise AssertionError("Verifier did not certify both the proof and decoded text")
        result["verified_statement"] = verified
    else:
        # A panic, signal, timeout, or CLI invocation error is not a rejection proof.
        if completed.returncode != 1 or "Error:" not in completed.stderr:
            raise AssertionError(f"Mutation was accepted or failed operationally: {result}")
        if expected_error and expected_error not in completed.stderr:
            raise AssertionError(f"Mutation failed for an unexpected reason ({expected_error}): {result}")
        result["rejected"] = True
    return result


def verify_case(case, pins, base, args):
    artifacts = resolve(base, case["artifacts_dir"])
    verifier = resolve(base, case["verifier"])
    tokenizer = resolve(base, pins["tokenizer"])
    vk_hash = checked_digest(case["verifier_sha256"])
    tokenizer_hash = checked_digest(pins["tokenizer_sha256"])
    prompt = case["prompt_tokens"]
    vocab = pins.get("vocab_size", 128256)
    eos = pins.get("eos_token", 128001)
    if not prompt or any(type(token) is not int or not 0 <= token < vocab for token in prompt):
        raise ValueError("Pinned prompt must contain valid nonempty token IDs")
    if digest(verifier) != vk_hash or digest(tokenizer) != tokenizer_hash:
        raise ValueError("External verifier/tokenizer trust pin does not match its file")
    for name in FILES:
        if not (artifacts / name).is_file():
            raise ValueError(f"Required artifact is missing: {artifacts / name}")
    if not read_json(artifacts / "generated_tokens.json"):
        raise ValueError("Acceptance case must contain at least one generated token")

    report = {
        "name": case["name"], "artifacts_dir": str(artifacts), "checks": [],
        "original_sha256": {name: digest(artifacts / name) for name in FILES},
        "verifier_sha256": vk_hash, "tokenizer_sha256": tokenizer_hash,
    }
    with tempfile.TemporaryDirectory(prefix="deep-prove-artifact-check-") as temporary:
        work = Path(temporary)
        expected_prompt = work / "trusted-prompt.json"
        write_json(expected_prompt, prompt)

        def command(directory, verifier_path=verifier, tokenizer_path=tokenizer,
                    expected_vk=vk_hash, expected_tokenizer=tokenizer_hash):
            return [str(args.verifier_bin.resolve()), "--artifacts-dir", str(directory),
                    "--verifier", str(verifier_path), "--verifier-sha256", expected_vk,
                    "--prompt-tokens", str(expected_prompt), "--tokenizer", str(tokenizer_path),
                    "--tokenizer-sha256", expected_tokenizer]

        baseline = run_verifier(command(artifacts), args.timeout, args.threads, expect_success=True)
        statement = baseline["verified_statement"]
        if statement["prompt_tokens"] != prompt:
            raise AssertionError("Certified prompt differs from externally pinned prompt")
        if "expected_generated_tokens" in case:
            if statement["generated_tokens"] != case["expected_generated_tokens"]:
                raise AssertionError("Proof does not certify the previously accepted continuation")
        report["checks"].append({"name": "original", **baseline})

        variants = mutation_cases(vocab, eos)
        if len(statement["generated_tokens"]) > 2:
            position = len(statement["generated_tokens"]) // 2
            variants.append(("middle_generated_and_report", lambda d: change_generated(
                d, lambda t: t.__setitem__(position, (t[position] + 1) % vocab)),
                "Frozen generated tokens differ"))
        for name, mutate, expected_error in variants:
            directory = work / name
            directory.mkdir()
            for filename in FILES:
                # Real copies: mutating a test must never modify the frozen original.
                shutil.copy2(artifacts / filename, directory / filename)
            mutate(directory)
            try:
                result = run_verifier(command(directory), args.timeout, args.threads, expected_error)
            except Exception as error:
                raise AssertionError(f"{case['name']}/{name}: {error}") from error
            report["checks"].append({"name": name, **result})
            shutil.rmtree(directory)

        for name, kwargs in [
            ("wrong_external_vk_pin", {"expected_vk": changed_hash(vk_hash)}),
            ("wrong_external_tokenizer_pin", {"expected_tokenizer": changed_hash(tokenizer_hash)}),
        ]:
            result = run_verifier(command(artifacts, **kwargs), args.timeout, args.threads, "SHA256 mismatch")
            report["checks"].append({"name": name, **result})

        for name, source, argument in [
            ("modified_trusted_vk", verifier, "verifier_path"),
            ("modified_trusted_tokenizer", tokenizer, "tokenizer_path"),
        ]:
            changed = work / name
            shutil.copy2(source, changed)
            append_byte(changed)
            result = run_verifier(command(artifacts, **{argument: changed}), args.timeout,
                                  args.threads, "SHA256 mismatch")
            report["checks"].append({"name": name, **result})
    if report["original_sha256"] != {name: digest(artifacts / name) for name in FILES}:
        raise AssertionError("Frozen original artifacts changed during verification")
    report["passed"] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pins", type=Path, required=True, help="External trusted setup/acceptance record")
    parser.add_argument("--verifier-bin", type=Path, default=Path("target/release/verify-llm"))
    parser.add_argument("--report", type=Path, default=Path("target/llama3/artifact-verification.json"))
    parser.add_argument("--require-cases", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600, help="Seconds per verification; timeout is a failure")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = {"schema": 1, "passed": False, "pins": str(args.pins.resolve()), "cases": []}
    try:
        pins = read_json(args.pins)
        cases = pins["cases"]
        if pins.get("schema") != 1 or len(cases) != args.require_cases:
            raise ValueError(f"Pins must use schema 1 and contain {args.require_cases} cases")
        if len({case["name"] for case in cases}) != len(cases):
            raise ValueError("Acceptance case names must be distinct")
        if args.threads < 1 or args.timeout <= 0:
            raise ValueError("Threads and timeout must be positive")
        report["pins_sha256"] = digest(args.pins)
        for case in cases:
            print(f"Verifying original and tampered copies: {case['name']}", flush=True)
            report["cases"].append(verify_case(copy.deepcopy(case), pins, args.pins.resolve().parent, args))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, report)
        report["passed"] = True
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.report, report)
    print(json.dumps({"passed": report["passed"], "report": str(args.report),
                      "error": report.get("error")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
