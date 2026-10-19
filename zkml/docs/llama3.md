# Llama 3.2 1B local inference and proofs

This implementation loads the complete Hugging Face `Llama-3.2-1B` checkpoint
directly. It uses all 16 decoder blocks, grouped-query attention (32 query heads,
8 KV heads), the Llama 3 frequency-dependent rotary embedding, SwiGLU, RMSNorm,
and the tied embedding/output projection. The supplied checkpoint is a base
language model; the examples are text completions, without an instruction-model
chat template.

The design follows the uploaded DeepProve paper (`deep-prove.pdf`, IACR ePrint
2026/1112): calibrate and freeze integer scales, evaluate the complete causal
sequence once after autoregressive generation, and prove the resulting graph
with DeepProve's sumcheck and LogUp-GKR machinery. Generated tokens are checked
against that final pass before proving. The verifier checks the prompt,
adjacent prediction/input linkage, the final prediction, EOS, valid token IDs,
padding, and the binding between full and split public IO. The argmax relation
certifies a maximum; it does not certify a particular tie-breaking rule.

The floating-point path retains the checkpoint's RMSNorm epsilon. The integer
path uses the paper's epsilon-free approximation. Its separately measured
effect and the new two-sided integer norm bounds are explained in
[llama3-rmsnorm.md](llama3-rmsnorm.md).

CPU integer matrix multiplication uses a cache-blocked Rayon implementation
with exact i64 wrapping multiply/add, so the complete model can be validated on
CPU. It supports batched and broadcast matrices, transpose/slice strides, and
padding metadata. Its results are checked element by element against the
original backend, including values above the exact-integer range of f64. The
float and GPU matrix multiplication paths remain the existing Burn operations.

The batched polynomial-opening expression uses a balanced addition tree. The
previous left fold grew in depth with the total number of claims, and the
dependency recursively expands that expression before sumcheck. A 12,000-claim
reproduction exhausted an 8 MiB main-thread stack. Balancing preserves claim
order, challenge indices, chunk coefficients, and the polynomial being proved,
while making traversal depth logarithmic. Prover and verifier use the same
constructor. The constructor also rejects witness counts beyond the dependency's
16-bit index limit rather than silently wrapping indices.

## GLU precision and integer range

The original observer selected its ordinary activation range before checking
the FFN activation node. Its special activation branch was nested under the
residual `Add` condition, so a SwiGLU node could never reach it. This mattered
for the supplied checkpoint: block 1 has a calibrated GLU output magnitude of
about 623.644. A signed 12-bit output grid then has a step of about 0.3046,
rounding an otherwise useful activation of 0.125 to zero. A controlled HF
ablation applying only this GLU output quantization reproduced the large
accuracy loss on the first development sequence; increasing the GLU output precision removed
that isolated loss.

The observer now handles FFN activations before residual additions and chooses
the largest output width that fits the following contraction and the existing
requantization chunks. With 12-bit weights, intermediate width 8,192, and the
actual 15-bit fixed-point multiplier scale, the available magnitude budget is
`63 - 13 - 12 - 15 = 23` bits. Thus the GLU output uses signed 24-bit integers,
`[-2^23, 2^23-1]`, represented by two existing 12-bit lookup chunks. Arbitrary
14- or 17-bit outputs are incompatible with the current requantization domain
constraint. The following down projection has a conservative 48-bit
intermediate budget; adding the 15-bit fixed multiplier reaches the existing
63-bit limit. The down projection itself is not requantized immediately: the
following residual addition adjusts its fixed multipliers to preserve its
integer headroom. Its bound now pairs each multiplier with its own input width,
then takes the larger product bound and retains the addition/sign margin.
Previously it combined the largest multiplier and the widest input even when
they belonged to different branches. With the wider GLU output, that excessive
bound would right-shift several down-projection multipliers all the way to
zero. The corrected bound retains both residual branches. A regression uses
the measured mixed-width scales, checks the domain extremes and rounding in
i128, and proves and verifies the complete Add-plus-Requant graph. Invalid
activation width/contraction combinations are rejected before any unsigned
budget subtraction can wrap.

The global weight width and RMSNorm output remain at 12 bits. A separate HF
ablation of RMSNorm output rounding did not reproduce the large GLU-related
loss. Increasing the global bit width would also reduce the precision budget
available inside the SiLU lookup, so this correction changes the GLU output
range specifically. Regression tests exercise a calibrated GLU outlier, a
small retained activation, actual integer requantization with two chunks, the
unchanged residual range, and valid/invalid bit-budget boundaries. Final
native integer evaluation passes the numerical gates below; complete-model
proof acceptance remains separate.

## Reproduce numerical validation

The commands below assume the repository root, the pinned Rust toolchain, and
the local model directory used for this implementation. No model download is
needed. `prepare` downloads WikiText-2; the other Python commands use local
weights. Install the CPU build of PyTorch and the pinned packages in
`zkml/assets/scripts/llms/requirements-llama3.txt` in a virtual environment.

```bash
cargo build --release --locked -p zkml \
  --bin extract-logits --bin bench-llm --bin verify-llm

python zkml/assets/scripts/llms/llama3_validation.py prepare \
  --output target/llama3/data
python zkml/assets/scripts/llms/llama3_validation.py reference \
  --samples target/llama3/data/test.json --output target/llama3/hf-test

RAYON_NUM_THREADS=8 target/release/extract-logits \
  --model /home/yliangh/models/meta-llama/Llama-3.2-1B \
  --tokens-file target/llama3/data/test.json --max-context 256 \
  --mode float --output-dir target/llama3/rust-float-test
python zkml/assets/scripts/llms/llama3_validation.py compare \
  --reference target/llama3/hf-test --candidate target/llama3/rust-float-test \
  --mode float --output target/llama3/float-test-metrics.json
```

The fixed data selection uses the first sufficiently long nonempty records in
each split, with exact row IDs and dataset fingerprints saved in
`data-manifest.json`: 32 training records available for calibration, 4 validation
records for development, and 25 held-out test records. Each contains 256 tokens.
Calibration starts with the first 8 training records; changes to calibration
are evaluated on development data before the held-out run. Cosine similarity,
top-1 agreement, and relative L2 cover all 6,400 real logit rows, including each
sequence's final row. Perplexity uses 6,375 next-token targets: the final row has
no target and is excluded only from NLL. Padding is excluded throughout.
Perplexity is `exp(total NLL / total prediction tokens)`. Both implementations
receive exactly the same token histories.

```bash
RAYON_NUM_THREADS=8 target/release/extract-logits \
  --model /home/yliangh/models/meta-llama/Llama-3.2-1B \
  --tokens-file target/llama3/data/development.json --max-context 256 \
  --calibration-tokens target/llama3/data/calibration.json \
  --calibration-samples 8 --mode int \
  --output-dir target/llama3/rust-int-development
python zkml/assets/scripts/llms/llama3_validation.py compare \
  --reference target/llama3/hf-development \
  --candidate target/llama3/rust-int-development \
  --mode int --output target/llama3/int-development-metrics.json
```

Generate `hf-development` with the same reference command using
`development.json`. For the held-out integer run, replace development paths
with test paths, and pass `--expected-quantization-sha256` with the development
run's accepted digest to pin its frozen scales. Integer comparisons always
check `quantization.json` against its manifest. When both manifests contain
source hashes, model weights, configuration, and tokenizer identities must
match; historical records without hashes explicitly report that this check
was unavailable. The comparison command exits unsuccessfully if the agreed
accuracy gates are missed: float mean cosine ≥ 0.99999 and relative L2 ≤ 0.001;
integer mean cosine ≥ 0.99, teacher-forcing top-1 agreement ≥ 0.90, and
perplexity ≤ 1.10 times the HF reference. Passing metrics are necessary in
addition to readable generated text and successful proof verification.

## Generate and prove the exact continuations

For integer text generation by itself, use the same local weights and frozen
calibration selection with `extract-logits`:

```bash
RAYON_NUM_THREADS=8 target/release/extract-logits \
  --model /home/yliangh/models/meta-llama/Llama-3.2-1B \
  --text "The capital of France is" --num-tokens 32 --max-context 256 \
  --calibration-tokens target/llama3/data/calibration.json \
  --calibration-samples 8 --mode int --output-dir target/llama3/completion
```

Its manifest contains generated token IDs and decoded text. The following
command also generates and verifies proofs for the three fixed cases:

```bash
RUST_LOG=info target/release/bench-llm \
  --model llama3 --hf-dir /home/yliangh/models/meta-llama/Llama-3.2-1B \
  --prompt-cases target/llama3/data/prompts.json --max-context 256 \
  --calibration-tokens target/llama3/data/calibration.json \
  --calibration-samples 8 --num-threads 8 \
  --artifacts-dir target/llama3/proofs
```

This loads and calibrates once, creates the proving contexts once, and proves
the three cases in sequence. They request 32, 64, and 64 new tokens; the final
case has a 192-token prompt. A prompt of length `p` and a request for `n` new
tokens require `p+n-1` input rows because the final prediction is an output.
Generation stops early at EOS. Each case directory contains the verified proof,
public IO, verifier parameters, prompt and generated token IDs, decoded text,
and a result record. `quantization.json` at the root records the frozen scales;
its SHA-256 must match the integer quality run's manifest.

Before proving, `generation-unverified.json` and `generation-unverified.txt`
retain the actual continuation and successful causal-replay check. They are
explicitly marked as unverified numerical diagnostics. Only the later
`result.json`, written after cryptographic verification, records proof success.

Retain the verifier key and its digest, the prompt, and tokenizer identity as
trusted inputs outside the proof directory. A hash read only from an untrusted
`result.json` does not establish which model was proved. The standalone verifier
does not require weights or recalibration:

```bash
target/release/verify-llm \
  --artifacts-dir target/llama3/proofs/factual \
  --verifier /path/to/trusted/factual-verifier.bin \
  --verifier-sha256 TRUSTED_VERIFIER_SHA256 \
  --prompt-tokens target/llama3/data/factual-prompt.json \
  --tokenizer /home/yliangh/models/meta-llama/Llama-3.2-1B/tokenizer.json \
  --tokenizer-sha256 TRUSTED_TOKENIZER_SHA256
```

The prompt file is the `prompt_tokens` array from the corresponding entry of
`prompts.json`. Provide the corresponding verifier key for each case, because
the certified length is part of its configuration. With a tokenizer supplied,
the verifier also reconstructs and checks the saved text. This local evaluation
uses the existing DeepProve setup mechanism; it does not constitute an audited
production trusted-setup ceremony.

## Validation record

`target/llama3/final-acceptance.json` is the final acceptance index, linking the
numerical reports, completed proof run, trusted pins, and independent artifact
verification with their SHA-256 hashes.

The float and integer numerical acceptance runs passed their fixed gates.
All three complete-model generation proofs (`factual`, `story`, and
`context256`) passed in-process and independent offline verification. The
artifact acceptance suite passed all 60 checks: three original proofs and
57 expected rejections of modified artifacts or trust pins. Completed
numerical measurements:

| Comparison | Logit rows / NLL targets | Mean cosine | Top-1 agreement | Relative L2 | Reference PPL | Candidate PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rust float vs HF FP32, test | 6,400 / 6,375 | 0.9999999999994863 | 1.0000 | 9.1779e-7 | 15.1206506342 | 15.1206503573 |
| HF without epsilon vs HF FP32, development | 1,024 / 1,020 | 0.9999665832 | 0.9941 | 0.0081822 | 12.2826799126 | 12.2959552401 |
| Rust integer vs HF FP32, development | 1,024 / 1,020 | 0.9987087372 | 0.9600 | 0.0449938 | 12.2826799126 | 12.3473677087 |
| Rust integer vs HF FP32, test | 6,400 / 6,375 | 0.9986906713 | 0.9540625 | 0.0458092 | 15.1206506342 | 15.1431646463 |

The full float run completed in 22m40.54s with peak RSS 10,873,196 KiB, while
other local validation/build tasks were running. These are observed resource
figures, not isolated throughput measurements. Detailed metrics are saved at
`target/llama3/float-test-metrics.json` and
`target/llama3/epsilon-ablation.json`.
The earlier float extraction predates candidate-source hashing; its report
therefore retains `source_identity_checked=false`. The integer development
and test reports contain matching weights, configuration, and tokenizer hashes.

The integer development run passed all quality gates using 8 calibration
samples, with a PPL ratio of 1.0052665865. It completed in 14m32.67s with peak RSS
20,805,588 KiB while the proof setup was running concurrently. Its complete
metrics are in `target/llama3/int-development-metrics.json`; source hashes and
frozen scales are checked. The independently calibrated proof run produced the
identical scale snapshot SHA-256:
`2ebfe9a192a728d7de8ff715b8873674d31cbe5c67d303cc7d1c9c5c48d1f4b3`.

The held-out integer test also passed all three gates with those same frozen
8-sample calibration parameters: mean cosine 0.9986906713258432, top-1 agreement
0.9540625, and PPL ratio 1.0014889578941346. Its 25 sequences contain 6,400 real
logit rows and 6,375 NLL targets. All token histories and shapes match the HF
reference, and the externally supplied quantization digest matches the actual
snapshot file, development run, and proof setup. The run exited successfully
after 43m38.44s with peak RSS 20,806,876 KiB, concurrently with proving work.
`target/llama3/int-test-metrics.json` contains the complete metrics;
`target/llama3/numerical-acceptance.json` summarizes the checked gates, source
identities, calibration pin, and evidence hashes. Existing raw execution and
comparison logs are copied under `target/llama3/numerical-logs/`.

The first complete-model proof attempt completed generation, causal replay,
layer proofs, and lookup-table claims, then aborted with a main-thread stack
overflow while entering the batched commitment opening. It produced no verified
proof artifacts. The failure is consistent with the independently reproduced
deep-expression failure above; a debugger backtrace was unavailable because
the environment prohibits ptrace. The attempt took 29m21.74s including
calibration/setup, with peak RSS 85,602,184 KiB. Its failed log is retained at
`target/llama3/proof-attempt-1.log`.

After the balanced-tree repair, all three cases completed full causal replay,
proof generation, and verification. The factual continuation starts with
`Paris.` for `The capital of France is`; the story continues with
`there lived a young man named John.` The exact token IDs and complete text are
retained in each case's `result.json` and verified against the proof's public
IO. Recorded measurements from `target/llama3/proof-bench-retry.csv` are:

| Case | Prompt tokens | Generated tokens | Certified input rows | Prove (s) | In-process verify (s) | Proof bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| factual | 6 | 32 | 37 | 519.407 | 20.602 | 28,328,372 |
| story | 11 | 64 | 74 | 669.705 | 22.437 | 31,148,963 |
| context256 | 192 | 64 | 255 | 1,034.949 | 28.284 | 34,024,947 |

The final case contains 256 prompt-plus-generated tokens; 255 input rows
suffice because the last token is the final prediction. All cases use the
complete 16-block model and the same accepted calibration snapshot. The
8-thread CPU run calibrated and set up once, then completed all three cases
in 1h06m57s with exit status 0. Calibration took 509.124 s and context/setup
generation took 802.827 s; those shared times are repeated in each CSV row,
not incurred three times. Peak process RSS was 200,762,952 KiB (191.46 GiB).
The per-case high-water readings were 113,924, 140,759, and 196,057 MiB,
respectively. These are cumulative process peaks including setup and retained
allocations, not per-case allocation deltas. Numerical acceptance and other
local checks overlapped this run, so the timings are not isolated throughput
measurements. The full execution record is
`target/llama3/proof-attempt-2.log`.

Each opening contained 14,156 claims. The factual, story, and context cases
used 29,341, 29,365, and 29,605 witness slots, respectively, within the supported
65,535-slot domain. All completed on the default 8 MiB main stack. For the
first case, expression construction took 11.064 ms and monomial expansion
took 197.622 ms. `target/llama3/opening-expression-checks.json` retains the
small-stack regression and first-case integration evidence; the full run log
records the later cases.

Separate `verify-llm` processes accepted all three original proofs and checked
prompt linkage, exact generated token IDs, and decoded text using external
trusted verifier files and pinned tokenizer identities. Each case also passed
19 negative checks covering proof/IO corruption, extra serialized bytes,
changed prompts and generated tokens, changed text, and changed trust pins or
trusted files. Negative checks require exit status 1 and an ordinary verifier
error. Checks with a prescribed rejection reason must also match that reason;
the raw proof/IO byte-corruption checks allow either decoding or verification
errors. A panic, signal, timeout, or invocation failure is not counted as a
successful rejection.

`target/llama3/artifact-verification.json` records all 60 results and the hashes
of each retained subset report, pins file, provenance record, and verification
binary. Its three subsets match `target/llama3/trusted-pins.json` exactly.
`target/llama3/first-proof-acceptance.json` preserves the earlier first-case
checkpoint, including proof/verifier/public-IO hashes; the final three-case
acceptance is recorded in the aggregate report.
