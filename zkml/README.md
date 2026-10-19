# ZKML Inference Proving

**WARNING**: This codebase is not audited and not production ready and is provided as is. Use at your own risk.

## Overview

DeepProve is a framework for proving inference of neural networks using cryptographic techniques based on sumchecks and logup GKR. Proving time is sublinear in the size of the model, providing order-of-magnitude speedups compared to other inference proving frameworks.

**First-class LLM support**: DeepProve proves end-to-end inference for transformer-based LLMs including **GPT-2**, **Gemma 3**, **Llama 2**, and **Llama 3**, covering all transformer layer types from embeddings through to next-token argmax. MLP and CNN inference is also supported.

The framework quantizes weights and activations into a fixed zero-centered range. The default range is `[-2^11, 2^11-1]` (12-bit symmetric quantization); the bit width can be set via the `ZKML_BIT_LEN` environment variable (typical values: 8, 10, 12, 16).

## How DeepProve Works

DeepProve is built on a few simple ideas. If you skim only one section of this document, this is the one.

### 1. Certify, don't re-execute

A naïve ZK system would re-run each token's forward pass inside a circuit. For a sequence of length `t` and a per-token circuit of size `|C|`, that gives `O(t² · |C|)` prover work — unworkable on real-world LLMs.

DeepProve instead concatenates the input tokens and the claimed output tokens into a single sequence, and runs the model on that sequence **once**. Because of the causal attention mask and the deterministic argmax at the output, the model's prediction at position `t` certifies the token at position `t+1`. We prove a single forward pass; the structure of attention does the rest.

### 2. Pre-process the model once

A model file in SafeTensors, GGUF, or ONNX format is parsed into an operator graph, quantized into integer-only arithmetic (post-training static quantization, calibrated on a representative dataset), and committed to with a polynomial commitment scheme. The result is a reusable proving context (`*.pk` / `*.vk`) that amortizes across every subsequent inference proof.

### 3. Prove layer-by-layer, then chain

For every layer in the graph we run a specialised protocol:

- **Linear ops** (MatMul, Conv, RMSNorm, LayerNorm) — efficient sum-check protocols
- **Non-linear ops** (Softmax, GeLU, requantization) — LogUp-GKR lookup arguments

Per-layer claims chain back-to-front in a GKR-style composition, producing a single end-to-end proof.

The polynomial commitment scheme used in the public binaries is **HyperKZG**, built on top of [arkworks](https://arkworks.rs/) and therefore generic over any pairing-friendly curve. The CPU build defaults to BN254; the GPU build (`--features cuda`) is currently BN254-only.

## Documentation

Deeper technical write-ups — proof protocols for each layer (embeddings, QKV, multi-head attention, positional encoding, LayerNorm, RMSNorm, argmax), softmax, EinSum, ReLU, MaxPool, range checks, lookup tables, and commitment schemes — live in [`docs/`](docs/) and are built with [mdBook](https://github.com/rust-lang/mdBook) + [mdbook-katex](https://github.com/lzanini/mdbook-katex):

```bash
cargo install mdbook
cargo install mdbook-katex
mdbook build docs --open
```

## Supported Models & Layers

### Transformer (LLM)

The recommended path is **SafeTensors via Hugging Face**: pass `--hf <model-id>` and `bench-llm` will fetch and cache the weights automatically on first run. GGUF (`--gguf <path|url>`) and ONNX models are also supported.

| Model | Recommended loader | Notes |
|-------|--------------------|-------|
| **GPT-2** | `--hf openai-community/gpt2` | SafeTensors auto-downloaded from HF. GGUF also supported (`--gguf gpt2.Q2_K.gguf`) |
| **Gemma 3** | `--hf google/gemma-3-1b-it` | SafeTensors only |
| **Llama 2** | `--hf meta-llama/Llama-2-7b-hf` | SafeTensors only. `--accuracy` mode currently falls back to default quantization for Llama 2 |
| **Llama 3 / 3.2** | `--model llama3 --hf-dir /path/to/local/model` | SafeTensors; grouped-query attention, tied embeddings and Llama 3 scaled RoPE. Calibrated integer inference is always enabled. |

Supported transformer layers:

- **Embeddings** — provable token vocabulary lookup
- **Positional encoding** — Absolute and Rotary (RoPE)
- **Attention mask** — causal masking for auto-regressive generation
- **EinSum / Multi-head attention** — batched matrix multiplication for QK^T and OV projections
- **Softmax** — with provable range checks
- **LayerNorm / RMSNorm** — vector normalization with proofs
- **Argmax / Logits** — next-token prediction output layer

### Offline verification of saved LLM proofs

`bench-llm --artifacts-dir DIR` saves the verified proof, public IO, verifier
context, prompt and continuation token IDs, text, and a setup record in
`result.json`. For a text prompt, use `--prompt TEXT --max-new-tokens N`; for an
exact token sequence, use `--prompt-tokens prompt_tokens.json`. The certified
context contains `prompt_length + N - 1` input positions, including the position
which predicts the final generated token. Generation may stop earlier at EOS.

To prove several requests with one frozen calibration and setup, use
`--prompt-cases prompts.json`. This file is an array of objects with `name`,
`prompt_tokens`, and `max_new_tokens` (the same format produced by
`llama3_validation.py prepare`). Each request runs from reset caches and receives
its own complete proof. Artifacts are written under `DIR/NAME/`, with an index in
`DIR/cases.json`. Each saved verifier context binds that request's context limit;
pin the exact per-request `verifier.bin` when verifying it independently.

Build the standalone verifier with `cargo build --release -p zkml --bin verify-llm`.
It requires no model weights and makes no network requests:

```bash
target/release/verify-llm \
  --artifacts-dir /path/to/artifacts \
  --verifier /trusted/setup/verifier.bin \
  --verifier-sha256 "$TRUSTED_VERIFIER_SHA256" \
  --prompt-tokens /trusted/request/prompt_tokens.json \
  --tokenizer /trusted/model/tokenizer.json \
  --tokenizer-sha256 "$TRUSTED_TOKENIZER_SHA256"
```

The verifier checks the cryptographic proof, public autoregressive sequence,
prompt, and agreement of frozen token IDs with `result.json`. Supplying a
tokenizer also checks `generated.txt` and the recorded full text against the
certified IDs. Without `--tokenizer`, only token IDs are certified and the JSON
response explicitly reports `text_verified: false`. Without `--prompt-tokens`,
the prompt is read from the artifact directory and is reported as the verified
statement; supply the expected request separately when that distinction matters.

`--verifier` is mandatory: its association with the intended model, quantization
and setup must be established externally. The optional SHA256 pin should come
from a trusted setup record. `result.json` records the generated verifier's
SHA256 for reproducibility, but a hash supplied by an untrusted artifact
directory does not establish trust. Using that directory's own `verifier.bin`
performs a reproducible self-check; it does not independently establish which
model was committed. The same external trust requirement applies to the
tokenizer used to interpret certified IDs as text.

### Traditional (MLP / CNN)

- **Dense** — fully connected layers
- **Conv2D** — convolutions with FFT-based optimization
- **Activations** — ReLU, GELU
- **Pooling** — MaxPool
- **Reshape / Flatten / Add** — tensor manipulation and residual connections

## Installation

### Prerequisites

- **Rust + Cargo** (latest stable)
- **Python 3.10+** — only required if you plan to run the MLP/CNN benchmark harness `bench.py`
- **CUDA 12.x toolkit + NVIDIA driver** — only required for the GPU build

### Build (CPU)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
cargo build --release -p zkml --bin bench-llm
```

### Build (GPU)

DeepProve ships two GPU feature flags:

| Feature | Platform | Effect |
|---------|----------|--------|
| `cuda` | Linux + NVIDIA | Replaces the CPU PCS with `HyperKZGGpu` and runs inference on burn's CUDA backend — **fastest path** |
| `wgpu` | Linux / macOS / Windows | Cross-platform GPU inference; PCS still runs on CPU |

```bash
# Linux + NVIDIA: full GPU acceleration (inference + PCS)
cargo build --release --features cuda -p zkml --bin bench-llm

# Cross-platform GPU inference only
cargo build --release --features wgpu -p zkml --bin bench-llm
```

The default (no `--features` flag) is a CPU-only build using the `ndarray` backend.

## Quickstart: Prove GPT-2 Inference End-to-End

This walkthrough builds the binary, fetches GPT-2 from Hugging Face, and produces an end-to-end inference proof.

### 1. Build

```bash
cargo build --release -p zkml --bin bench-llm
```

### 2. Get a GPT-2 model

The recommended path is `--hf openai-community/gpt2`: `bench-llm` will fetch the SafeTensors weights and tokenizer from Hugging Face on first run and cache them locally for subsequent runs — no manual download required. SafeTensors is the most widely-used checkpoint format for transformer models, which is why we lead with it.

GGUF (`--gguf <path|url>`) and ONNX checkpoints are also supported; the GGUF path is convenient if you already have a pre-quantized file (e.g. `gpt2.Q2_K.gguf`).

### 3. Prove a single sequence

```bash
cargo run --release --bin bench-llm -- \
  --model gpt2 --hf openai-community/gpt2 --sequence 64
```

This builds the proving context, runs inference on a 64-token random prompt, generates a proof, and verifies it. Two CSV files are produced (see *Reading the output* below).

### 4. Sweep multiple sequence lengths

```bash
cargo run --release --bin bench-llm -- \
  --model gpt2 --hf openai-community/gpt2 --sequence 64,128,256,512
```

Each sequence length appends one row to the bench CSV.

### 5. Run on GPU

Linux + NVIDIA:

```bash
cargo run --release --features cuda --bin bench-llm -- \
  --model gpt2 --hf openai-community/gpt2 --sequence 64
```

The `cuda` feature switches the PCS to `HyperKZGGpu` and runs inference on the GPU backend.

### 6. Cache the proving context

The proving context (`pk` / `vk`) is the most expensive one-off step. Save it to disk once and reuse it across runs:

```bash
# First run: generate and save
cargo run --release --bin bench-llm -- \
  --model gpt2 --hf openai-community/gpt2 --sequence 64 \
  --save_params setup

# Subsequent runs: skip context generation
cargo run --release --bin bench-llm -- \
  --model gpt2 --hf openai-community/gpt2 --sequence 64 \
  --load_params setup
```

This writes/reads `setup.pk` and `setup.vk` in the current directory.

### 7. Higher-accuracy mode

```bash
cargo run --release --bin bench-llm -- \
  --model gpt2 --hf openai-community/gpt2 --sequence 64 --accuracy
```

`--accuracy` enables outlier smoothing and per-tensor activation tracking, improving cosine similarity / perplexity at the cost of additional proving time. Available for GPT-2 and Gemma 3.

### 8. Profile distributed proving locally

```bash
cargo run --release --bin bench-llm -- \
  --model gpt2 --hf openai-community/gpt2 --sequence 64 \
  --distributed --num_chunks 4
```

`--distributed` simulates the chunked distributed protocol on a single host (useful for measuring memory split across N workers without provisioning a cluster). Output goes to `bench_distributed.csv`.

### 9. Read the output CSV

`bench-llm` produces two CSV files in the current directory:

| File | Source | Contents |
|------|--------|----------|
| `bench-llm.csv` | `timed_core` | One row per timed function call across the run — useful for per-method profiling |
| `bench.csv` (or `bench_distributed.csv` with `--distributed`) | `measure` | One row per benchmark trial with aggregated metrics |

Columns in the per-run CSV (`bench.csv`):

| Column | Meaning |
|--------|---------|
| `model_name` | `gpt2` / `gemma3` / `llama2` |
| `num_threads` | Rayon thread count for this run |
| `quantization_time` | One-off cost of converting the model to integer arithmetic (ms) |
| `context_generation_time` | One-off cost of generating PK/VK (ms; `0` when `--load_params` is used) |
| `max_context` | Maximum context window for this trial (tokens) |
| `min_user_len` | Prompt length (tokens) |
| `inference_time` | Quantized inference wall-time (ms) |
| `prove_full` | Total proving wall-time (ms) |
| `proof_size` | Serialized proof size (bytes) |
| `prove_full_memory_peak` | Peak RSS during proving (MiB) |
| `token/sec` | Derived as `max_context / prove_full` |

### 10. Try Gemma 3 / Llama 2

```bash
# Gemma 3
cargo run --release --bin bench-llm -- \
  --model gemma3 --hf google/gemma-3-1b-it --sequence 128

# Llama 2
cargo run --release --bin bench-llm -- \
  --model llama2 --hf meta-llama/Llama-2-7b-hf --sequence 128
```

Hugging Face access requires `huggingface-cli login` for gated models (e.g. Llama 2).

### 11. Run the standalone code example

For a self-contained reference run with prompt `"The sky is"`:

```bash
cargo run --release --example chunk_llm
```

Run `bench-llm --help` for the full list of flags and options.

## Expected Performance

Reference numbers on a 24-core / 504 GB CPU machine (AMD EPYC 9254, 2.9 GHz), HyperKZG PCS, default quantization:

| Model   | Sequence | Prove time | Verify | Proof size | Throughput (tokens/s) | Throughput (tokens/min) |
|---------|---------:|-----------:|-------:|-----------:|----------------------:|------------------------:|
| GPT-2   |       64 |    2.35 min |  1.25 s |   7.95 MiB |                  0.45 |                      27 |
| GPT-2   |      128 |    3.02 min |  1.14 s |   8.82 MiB |                  0.71 |                      42 |
| GPT-2   |      256 |    4.60 min |  1.24 s |   9.77 MiB |                  0.93 |                      56 |
| GPT-2   |      512 |    7.64 min |  1.33 s |  10.71 MiB |                  1.12 |                      67 |
| Gemma 3 |       64 |    8.49 min |  3.69 s |  16.34 MiB |                  0.13 |                       8 |
| Gemma 3 |      128 |   10.27 min |  3.88 s |  18.02 MiB |                  0.21 |                      12 |
| Gemma 3 |      256 |   12.65 min |  4.09 s |  19.76 MiB |                  0.34 |                      20 |
| Gemma 3 |      512 |   18.95 min |  4.32 s |  21.73 MiB |                  0.45 |                      27 |

The CSV output column is `token/sec`; the per-minute number is provided here for convenience.

Llama 2 benchmarks land in the same regime as Gemma 3; rerun locally for current numbers. GPU builds (`--features cuda`) accelerate both inference and PCS opening; the relative speedup depends on hardware.

For full benchmark methodology and a comparison against published prior work, see the main [README](../README.md) and the DeepProve paper.

## CNN/MLP Inference

```bash
python bench.py --model-type cnn
python bench.py --configs 5,100:7,50 --repeats 5
```

Results are written to `zkml_*.csv` with columns: setup time, inference time, proving time, verification time, accuracy, proof size (KB).

## Repository Layout

```
zkml/
├── src/
│   ├── lib.rs               # Public API
│   ├── bin/bench/
│   │   ├── llm.rs           # bench-llm binary (LLM benchmarking)
│   │   └── cnn.rs           # bench binary (CNN/MLP benchmarking)
│   ├── layers/              # Layer implementations (CNN + transformer)
│   ├── model/               # Model runners and LLM driver
│   ├── parser/              # ONNX, GGUF, SafeTensors parsers
│   ├── tensor/              # Tensor operations
│   ├── graph/               # Computational graph representation
│   ├── iop/                 # Interactive oracle proofs and chunking
│   ├── lookup/              # Lookup table implementations
│   ├── quantization/        # Quantization strategies
│   └── commit/              # Commitment schemes
├── examples/
│   ├── chunk_llm.rs         # LLM inference + chunked proving (GPT-2)
│   └── chunk_cifar.rs       # CNN inference proving (CIFAR)
├── docs/                    # mdBook technical documentation
├── assets/scripts/          # Python scripts for model generation
└── bench.py                 # Python benchmark harness (MLP/CNN)
```

## Status & Roadmap

**Models & layers:**

- [x] LLM inference proving (GPT-2, Gemma 3, Llama 2)
- [x] Full transformer layer coverage (embeddings, attention, LayerNorm, RMSNorm, softmax, argmax)
- [x] MLP, CNN (Dense, Conv2D, ReLU/GELU, MaxPool)
- [x] Chunked / distributed proving
- [x] Unpadded inference
- [ ] Additional model families
- [ ] More layer types (BatchNorm, Dropout, etc.)

**Accuracy.** DeepProve preserves model behaviour to a high degree even after full integer quantization. Out of the box every layer uses a single scaling factor, which is fast and works well for most architectures. For GPT-2 and Gemma 3 you can opt into a higher-fidelity mode with `--accuracy`: this enables outlier smoothing and per-tensor activation tracking, pushing cosine similarity above 99.6% versus the floating-point baseline at 12-bit quantization (GPT-2) — well worth a try if you care about output quality. We are working on extending outlier smoothing to Llama 2 and adding row-wise per-layer quantization to squeeze out more accuracy on long-context models.

**Performance:**

- [x] HyperKZG PCS (CPU + GPU)
- [x] GPU acceleration via `--features cuda` (Linux + NVIDIA) or `--features wgpu` (cross-platform inference)
- [ ] GPU proving cluster
- [ ] Smaller, more frequent lookup tables
- [ ] Improved parallelism for logup, GKR, sumchecks

## Troubleshooting

- **Out of memory** — large LLMs at long context windows require substantial RAM. Reduce `--sequence` / `--max-context` or use a host with more memory.
- **Hugging Face downloads fail** — gated models (e.g. Llama 2) require `huggingface-cli login` first; otherwise check network access.
- **GGUF format mismatch** — ensure the GGUF file matches the expected model architecture (`--model gpt2` ↔ a GPT-2 GGUF).
- **CUDA build fails** — verify the CUDA toolkit (≥ 12.x) and NVIDIA driver are installed; the `cuda` feature is Linux-only.
- **macOS thread limiting** — CPU affinity is not supported on macOS; the binary proceeds without restrictions and prints a warning.

## License

This project is licensed under the [LICENSE](LICENSE) file.

## Acknowledgements

This project is built on top of the work from [scroll-tech/ceno](https://github.com/scroll-tech/ceno) — it re-uses the sumcheck and GKR implementation from that codebase.
