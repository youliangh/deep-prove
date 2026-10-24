#!/usr/bin/env python3
"""Benchmark exact Llama 3 input lengths in separate, sequential processes.

Each prompt is a prefix of the first row in --input-tokens. One predicted token
is certified as output, so the proved input has exactly the requested length,
including when that prediction is EOS. Calibration and model capacity are fixed
across lengths. GNU time records whole-process peak RSS, including model loading,
calibration, setup, inference, proving and verification.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[4]
SUMMARY_FIELDS = (
    "sequence_length", "status", "peak_rss_kib", "peak_rss_gib",
    "prover_seconds", "verifier_seconds", "proof_bytes", "proof_mib",
    "prove_high_water_rss_mib", "quantization_seconds", "setup_seconds",
    "inference_seconds", "wall_seconds", "returncode", "error",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def token_rows(path):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty JSON array of token arrays")
    for row in rows:
        if not isinstance(row, list) or not row or any(
            type(token) is not int or token < 0 for token in row
        ):
            raise ValueError(f"{path}: expected nonempty arrays of nonnegative token IDs")
    return rows


def hardware():
    result = {
        "platform": platform.platform(), "machine": platform.machine(),
        "logical_cpus": os.cpu_count(), "python": platform.python_version(),
    }
    if hasattr(os, "sched_getaffinity"):
        result["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    if hasattr(os, "getloadavg"):
        result["load_average_at_start"] = os.getloadavg()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                result["cpu_model"] = line.split(":", 1)[1].strip()
                break
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        result["memory_at_start"] = dict(
            line.split(":", 1) for line in meminfo.read_text().splitlines()
            if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:", "SwapFree:"))
        )
    return result


def summarize(output_dir, manifest):
    write_json(output_dir / "summary.json", {
        "schema": 1, "updated_at": timestamp(),
        "definitions": manifest["definitions"], "runs": manifest["runs"],
    })
    temporary = output_dir / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest["runs"])
    temporary.replace(output_dir / "summary.csv")
    write_json(output_dir / "manifest.json", manifest)


def validate_result(run_dir, expected_prompt, model_context, threads, calibration_samples, prior_scale):
    artifacts = run_dir / "artifacts"
    report = read_json(artifacts / "result.json")
    if report.get("verified") is not True:
        raise ValueError("Missing successful in-process cryptographic verification")
    if report.get("prompt_tokens") != expected_prompt:
        raise ValueError("Verified prompt differs from the exact requested input prefix")
    if read_json(artifacts / "prompt_tokens.json") != expected_prompt:
        raise ValueError("Saved prompt tokens differ from the requested input prefix")
    if report.get("certification_context") != len(expected_prompt):
        raise ValueError("Certification context differs from the requested sequence length")
    if report.get("model_context") != model_context:
        raise ValueError("Loaded model context differs between runs")
    if report.get("requested_max_new_tokens") != 1 or len(report.get("generated_tokens", [])) != 1:
        raise ValueError("Expected exactly one certified output token")
    if report.get("num_threads") != threads:
        raise ValueError("Benchmark did not use the requested thread count")
    if report.get("calibration_samples") != calibration_samples:
        raise ValueError("Unexpected calibration sample count")
    if digest(artifacts / "calibration_tokens.json") != report.get("calibration_tokens_sha256"):
        raise ValueError("Saved calibration token hash is inconsistent")
    scale = report.get("quantization_sha256")
    if not scale or scale != digest(artifacts / "quantization.json"):
        raise ValueError("Quantization snapshot hash is missing or inconsistent")
    if prior_scale is not None and scale != prior_scale:
        raise ValueError("Calibration scales changed between sequence lengths")
    with (run_dir / "bench.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("Expected exactly one completed benchmark CSV row")
    row = rows[0]
    if int(row["max_context"]) != len(expected_prompt) or int(row["min_user_len"]) != len(expected_prompt):
        raise ValueError("Benchmark CSV records an unexpected sequence or prompt length")
    proof_bytes = (artifacts / "proof.bin").stat().st_size
    if proof_bytes != report.get("proof_bytes") or proof_bytes != int(row["proof_size"]):
        raise ValueError("Serialized proof size differs from the reported size")
    memory = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                       (run_dir / "time.txt").read_text())
    if memory is None:
        raise ValueError("GNU time did not record peak resident memory")
    rss_kib = int(memory.group(1))
    result = {
        "peak_rss_kib": rss_kib, "peak_rss_gib": rss_kib / 1024 ** 2,
        "proof_bytes": proof_bytes, "proof_mib": proof_bytes / 1024 ** 2,
        "prove_high_water_rss_mib": int(row["prove_full_memory_peak"]),
        "quantization_sha256": scale,
        "model_fingerprints": report.get("model_fingerprints"),
        "calibration_tokens_sha256": report.get("calibration_tokens_sha256"),
    }
    for output, source in (
        ("prover_seconds", "prove_full"), ("verifier_seconds", "verify_full"),
        ("quantization_seconds", "quantization_time"),
        ("setup_seconds", "context_generation_time"), ("inference_seconds", "inference_time"),
    ):
        milliseconds = int(row[source])
        if milliseconds < 0:
            raise ValueError(f"Negative measurement: {source}")
        result[output] = milliseconds / 1000
    return result


def process_memory(pid):
    """Read Linux process RSS counters; a just-exited process has no sample."""
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except FileNotFoundError:
        return None
    counters = dict(line.split(":", 1) for line in lines if ":" in line)
    if "VmRSS" not in counters:  # A zombie has already released its memory.
        return None
    return {name: int(counters[name].split()[0])
            for name in ("VmRSS", "VmHWM", "VmSwap")}


def validate_memory_guard(limit):
    if limit is None:
        return
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("--max-rss-gib must be a finite positive number")
    if sys.platform != "linux":
        raise ValueError("--max-rss-gib requires Linux /proc process counters")
    try:
        if process_memory(os.getpid()) is None:
            raise ValueError("Current process has no readable RSS counters")
        Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text()
    except (OSError, KeyError) as error:
        raise ValueError(f"--max-rss-gib requires readable Linux /proc: {error}") from error


def run_process(command, directory, environment, length, max_rss_gib=None):
    """Keep progress visible and terminate the benchmark together with GNU time."""
    started = time.monotonic()
    heartbeat = started + 60
    monitor = None if max_rss_gib is None else {
        "schema": 1, "limit_gib": max_rss_gib, "poll_seconds": 0.25,
        "scope": "GNU time's benchmark child; Linux /proc PID status counters",
        "samples": 0, "peak_rss_kib": 0, "peak_hwm_kib": 0,
        "peak_swap_kib": 0, "triggered": False,
    }
    child_pid = None
    kill_at = None
    returncode = None
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen(command, cwd=directory, env=environment,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if monitor is not None:
                monitor["time_pid"] = process.pid
                write_json(directory / "memory-monitor.json", monitor)
            while True:
                returncode = process.poll()
                if returncode is not None:
                    return returncode, monitor
                if monitor is not None:
                    if child_pid is None:
                        try:
                            children = Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split()
                        except FileNotFoundError:
                            if process.poll() is not None:
                                continue
                            raise RuntimeError("Memory guard cannot read GNU time's /proc children")
                        if len(children) > 1:
                            raise RuntimeError("Memory guard expected one GNU time benchmark child")
                        if children:
                            child_pid = int(children[0])
                            monitor["benchmark_pid"] = child_pid
                    counters = process_memory(child_pid) if child_pid is not None else None
                    if counters is not None:
                        monitor["samples"] += 1
                        for output, source in (("peak_rss_kib", "VmRSS"), ("peak_hwm_kib", "VmHWM"),
                                               ("peak_swap_kib", "VmSwap")):
                            monitor[output] = max(monitor[output], counters[source])
                        if not monitor["triggered"] and counters["VmHWM"] >= max_rss_gib * 1024 ** 2:
                            monitor.update(triggered=True, triggered_at=timestamp(),
                                           trigger_elapsed_seconds=time.monotonic() - started,
                                           trigger_counters_kib=counters)
                            write_json(directory / "memory-monitor.json", monitor)
                            print(f"RSS guard reached {max_rss_gib:g} GiB; terminating benchmark PID {child_pid}",
                                  file=sys.stderr, flush=True)
                            try:
                                os.kill(child_pid, signal.SIGTERM)
                                kill_at = time.monotonic() + 5
                            except ProcessLookupError:
                                pass
                    if kill_at is not None and time.monotonic() >= kill_at:
                        try:
                            os.kill(child_pid, signal.SIGKILL)
                            monitor["sigkill_sent"] = True
                        except ProcessLookupError:
                            pass
                        kill_at = None
                if time.monotonic() >= heartbeat:
                    print(f"Running {length} input tokens: {time.monotonic() - started:.0f} s elapsed", flush=True)
                    heartbeat = time.monotonic() + 60
                    if monitor is not None:
                        write_json(directory / "memory-monitor.json", monitor)
                try:
                    process.wait(timeout=0.25 if monitor is not None else 60)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            if monitor is not None:
                monitor.update(wall_seconds=time.monotonic() - started, returncode=process.poll())
                write_json(directory / "memory-monitor.json", monitor)


def interrupted(signum, frame):
    raise KeyboardInterrupt(f"Received signal {signum}")


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--lengths", default="32,64,128,256", help="Comma-separated exact input token counts")
    result.add_argument("--threads", type=int, default=8, help="Rayon threads (default: 8)")
    result.add_argument("--matmul-threads", type=int, default=4, help="Matrix multiplication threads (default: 4)")
    result.add_argument("--output-dir", type=Path, required=True, help="New or empty results directory")
    result.add_argument("--binary", type=Path, default=REPO / "target/release/bench-llm")
    result.add_argument("--calibration-tokens", type=Path, default=REPO / "target/llama3/data/calibration.json")
    result.add_argument("--calibration-samples", type=int, default=8)
    result.add_argument("--input-tokens", type=Path, default=REPO / "target/llama3/data/test.json",
                        help="JSON row arrays; use prefixes of the first held-out sample")
    result.add_argument("--model-context", type=int, default=256, help="Fixed capacity for every process")
    result.add_argument("--max-rss-gib", type=float,
                        help="Optional Linux benchmark RSS limit; terminate and record an incomplete resource_limit run")
    result.add_argument("--dry-run", action="store_true", help="Print plan without writing files or running the model")
    return result


def main():
    args = parser().parse_args()
    try:
        lengths = [int(value) for value in args.lengths.split(",")]
        if not lengths or len(set(lengths)) != len(lengths) or min(lengths) < 1:
            raise ValueError("Sequence lengths must be unique positive integers")
        if min(args.threads, args.matmul_threads, args.model_context) < 1:
            raise ValueError("Thread counts and model context must be positive")
        validate_memory_guard(args.max_rss_gib)
        for name in ("model_dir", "binary", "calibration_tokens", "input_tokens", "output_dir"):
            setattr(args, name, getattr(args, name).expanduser().resolve())
        if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
            raise ValueError(f"Benchmark binary is not executable: {args.binary}")
        if not Path("/usr/bin/time").is_file():
            raise ValueError("GNU time is required at /usr/bin/time")
        for filename in ("model.safetensors", "config.json", "tokenizer.json"):
            if not (args.model_dir / filename).is_file():
                raise ValueError(f"Missing model file: {args.model_dir / filename}")
        samples = token_rows(args.calibration_tokens)
        input_tokens = token_rows(args.input_tokens)[0]
        if not 1 <= args.calibration_samples <= len(samples):
            raise ValueError("Calibration sample count is out of range")
        if max(lengths) > len(input_tokens):
            raise ValueError("First input sample is shorter than the longest requested sequence")
        if max(max(lengths), max(map(len, samples[:args.calibration_samples]))) > args.model_context:
            raise ValueError("Input or calibration sample exceeds the fixed model context")
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise ValueError(f"Output directory is not empty: {args.output_dir}")
        environment = {
            "RAYON_NUM_THREADS": str(args.threads), "MATMUL_NUM_THREADS": str(args.matmul_threads),
            "RUST_LOG": "info,zkml::poly_commit::context=debug,zkml::iop::prover=debug",
            "LC_ALL": "C", "TMPDIR": "/tmp",
        }
        manifest = {
            "schema": 1, "created_at": timestamp(), "hardware": hardware(),
            "model_dir": str(args.model_dir), "binary": str(args.binary),
            "model_context": args.model_context, "calibration_samples": args.calibration_samples,
            "requested_lengths": lengths, "repetitions": 1,
            "max_rss_gib": args.max_rss_gib,
            "calibration_source": str(args.calibration_tokens),
            "input_source": str(args.input_tokens), "input_sample_index": 0,
            "environment": environment,
            "definitions": {
                "sequence_length": "Actual unpadded model input rows; one final prediction is output only.",
                "peak_rss_kib": "GNU time whole-process high-water RSS, including calibration/setup; independent process per length.",
                "prover_seconds": "bench.csv prove_full / 1000; proof execution graph, excluding calibration, setup, inference and serialization.",
                "verifier_seconds": "bench.csv verify_full / 1000; in-process cryptographic verifier, excluding key/proof loading and outer token checks.",
                "proof_bytes": "Serialized proof.bin length (bincode 2 standard); excludes public IO and verifier key.",
                "prove_high_water_rss_mib": "Process high-water RSS read immediately after proving, including any earlier setup peak.",
                "padding": "Proof tensors pad sequence dimensions to powers of two; requested default lengths are powers of two.",
            },
            "runs": [],
        }
        for length in lengths:
            directory = args.output_dir / f"sequence-{length}"
            command = [
                "/usr/bin/time", "-v", "-o", str(directory / "time.txt"),
                str(args.binary), "--model", "llama3", "--hf-dir", str(args.model_dir),
                "--prompt-tokens", str(directory / "prompt_tokens.json"), "--max-new-tokens", "1",
                "--max-context", str(args.model_context),
                "--calibration-tokens", str(args.calibration_tokens),
                "--calibration-samples", str(args.calibration_samples), "--num-threads", str(args.threads),
                "--bench", str(directory / "bench.csv"), "--artifacts-dir", str(directory / "artifacts"),
            ]
            manifest["runs"].append({
                "sequence_length": length, "status": "pending", "directory": str(directory),
                "command": command, "shell_command": shlex.join(command),
            })
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
            return 0
        manifest["binary_sha256"] = digest(args.binary)
        manifest["script_sha256"] = digest(__file__)
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                  capture_output=True, text=True)
        manifest["git_commit"] = revision.stdout.strip() if revision.returncode == 0 else None
        manifest["calibration_source_sha256"] = digest(args.calibration_tokens)
        manifest["input_source_sha256"] = digest(args.input_tokens)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summarize(args.output_dir, manifest)
        scale = None
        calibration_hash = None
        model_fingerprints = None
        signal.signal(signal.SIGTERM, interrupted)
        for run in manifest["runs"]:
            directory = Path(run["directory"])
            directory.mkdir()
            prompt = input_tokens[:run["sequence_length"]]
            write_json(directory / "prompt_tokens.json", prompt)
            run.update(status="running", started_at=timestamp())
            summarize(args.output_dir, manifest)
            print(f"Starting {run['sequence_length']} input tokens; log: {directory / 'run.log'}", flush=True)
            started = time.monotonic()
            try:
                returncode, memory_monitor = run_process(
                    run["command"], directory, {**os.environ, **environment},
                    run["sequence_length"], args.max_rss_gib)
                run["returncode"] = returncode
                if memory_monitor is not None and memory_monitor["triggered"]:
                    run.update(status="resource_limit", error=f"Benchmark exceeded RSS limit of {args.max_rss_gib:g} GiB")
                    memory = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)",
                                       (directory / "time.txt").read_text())
                    if memory is not None:
                        rss_kib = int(memory.group(1))
                        run.update(peak_rss_kib=rss_kib, peak_rss_gib=rss_kib / 1024 ** 2)
                    print(run["error"], file=sys.stderr, flush=True)
                    return 1
                if returncode != 0:
                    raise RuntimeError(f"Benchmark exited with status {returncode}; see run.log")
                result = validate_result(directory, prompt, args.model_context, args.threads,
                                         args.calibration_samples, scale)
                current_calibration = result["calibration_tokens_sha256"]
                if not current_calibration or (calibration_hash is not None and current_calibration != calibration_hash):
                    raise ValueError("Calibration tokens changed between sequence lengths")
                if model_fingerprints is not None and result["model_fingerprints"] != model_fingerprints:
                    raise ValueError("Model fingerprints changed between sequence lengths")
                scale = result["quantization_sha256"]
                calibration_hash = current_calibration
                model_fingerprints = result["model_fingerprints"]
                run.update(result, status="completed")
            except (Exception, KeyboardInterrupt) as error:
                run.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           error=f"{type(error).__name__}: {error}")
                print(run["error"], file=sys.stderr, flush=True)
            finally:
                run.update(wall_seconds=time.monotonic() - started, finished_at=timestamp())
                summarize(args.output_dir, manifest)
            if run["status"] != "completed":
                return 130 if run["status"] == "interrupted" else 1
            print(f"Completed {run['sequence_length']}: prover {run['prover_seconds']:.3f} s, "
                  f"verifier {run['verifier_seconds']:.3f} s, "
                  f"peak RSS {run['peak_rss_gib']:.3f} GiB, proof {run['proof_bytes']} bytes", flush=True)
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
