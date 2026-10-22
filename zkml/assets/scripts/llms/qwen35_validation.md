Qwen3.5 reference and acceptance tools
=====================================

Use a separate Python environment with `requirements-qwen35.txt`. The existing
Llama environment is independent. The reference requires PyTorch 2.6.0+cpu and
Transformers 5.3.0, runs CPU float32 eager attention, and loads only the text model
with `Qwen3_5ForCausalLM` and `Qwen3_5TextConfig`. It checks all loading diagnostics
and verifies the tied embedding/output head. Every command uses local files;
validation never downloads models or datasets.

The implementation reference is the [official pinned model source](https://github.com/huggingface/transformers/blob/v5.3.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
The tokenizer uses the checkpoint's serialized tokenizer JSON without replacing
its Unicode regex. Chat formatting disables thinking and preserves the original
chat template. Both `<|endoftext|>` (248044) and `<|im_end|>` (248046) terminate
text generation. These acceptance fixtures cap the final prompt-plus-output
sequence at 256 tokens. The Rust CLI's `--max-context` instead bounds computed
input rows: `prompt_len + max_new_tokens - 1 <= max_context`. The final predicted
token is an output and is not fed back into another forward pass. Thus a 192+64
fixture has 256 final tokens and 255 computed rows. Proof context sizes refer to
certified input rows, not final output sequence length.

Run from this directory, replacing the paths below with local paths. Large binary
artifacts belong outside the checkout. Global `--model` and `--threads` options
precede the subcommand.

```sh
python qwen35_validation.py prepare --dataset-dir /path/to/wikitext-2-raw-v1 --output /tmp/qwen/data
python qwen35_validation.py prepare-calibration --data-dir /tmp/qwen/data --output /tmp/qwen/bilingual
python qwen35_validation.py prepare-prompts --output /tmp/qwen/prompts
python qwen35_validation.py reference --samples /tmp/qwen/data/development.json --output /tmp/qwen/hf-dev
python qwen35_validation.py reference --samples /tmp/qwen/data/test.json --output /tmp/qwen/hf-test
python qwen35_validation.py generate --prompts /tmp/qwen/prompts/validation-prompts.json --output /tmp/qwen/hf-generation.json
python qwen35_validation.py cache-check --samples /tmp/qwen/data/smoke.json --output /tmp/qwen/cache.json
```

`prepare` selects 32 train, 4 development, and 25 test sequences from independent
raw WikiText splits and retokenizes with this checkpoint. It freezes source file
hashes, dataset revision, selection indices, and token IDs. `prepare-calibration`
interleaves the first 16 WikiText training sequences with 16 original Chinese or
mixed-language passages from `qwen35_calibration_zh.json`; 8/16/32-sample prefixes
are bilingual. These passages contain no held-out prompts or reference answers.
Development data is for tuning. Freeze quantization before using test data.

The native `extract-logits` binary accepts those token lists and writes a matching
`manifest.json` plus little-endian row-major float32 logits. Use `--mode float`
or `--mode int`, `--max-context 256`, `--tokens-file`, and `--output-dir`. For
integer evaluation also supply the frozen `--calibration-tokens` and
`--calibration-samples` settings. `--num-tokens 0` extracts all prompt rows;
`--num-tokens 64` generates continuations. Derive semantic token lists directly
from the `prompt_tokens` fields in `validation-prompts.json`.

```sh
python qwen35_validation.py compare --reference /tmp/qwen/hf-test --candidate /tmp/qwen/native-test --mode float --require-source-identity --output /tmp/qwen/float-metrics.json
python qwen35_validation.py compare --reference /tmp/qwen/hf-test --candidate /tmp/qwen/native-int-test --mode int --require-source-identity --expected-quantization-sha256 FROZEN_DIGEST --output /tmp/qwen/int-metrics.json
python qwen35_check_artifacts.py generation --prompts /tmp/qwen/prompts/validation-prompts.json --reference /tmp/qwen/hf-generation.json --candidate /tmp/qwen/native-generation --mode float --output /tmp/qwen/generation-check.json
python qwen35_check_artifacts.py generation --prompts /tmp/qwen/prompts/validation-prompts.json --reference /tmp/qwen/hf-generation.json --candidate /tmp/qwen/native-int-generation --mode int --expected-quantization-sha256 FROZEN_DIGEST --output /tmp/qwen/int-generation-check.json
```

Comparison streams every logit row, including the final prompt row. Next-token
NLL uses all available targets and excludes only the final row without a target.
It rejects token-history, model/tokenizer identity, digest, shape, or nonfinite
mismatches. Integer artifacts bind their complete quantization manifest; pass its
frozen digest when accepting test results. Acceptance thresholds are cosine
>=0.99999 and relative L2 <=0.001 for float; cosine >=0.99, top-1 agreement >=0.90,
and perplexity ratio <=1.10 for integer.

The nine semantic cases cover Chinese and English facts, arithmetic, translation,
short creation, and a 192-token chat prompt with a 64-token output budget. The
checker binds every generated token to the saved native logit argmax, checks
finite full logits and exact decoding, token budgets, EOS stopping, and a frozen
content rubric. Read every answer for coherence and responsiveness; keyword
checks alone do not certify meaning. Raw WikiText continuation is a numerical
stress case and is not counted as semantic success.

For layer diagnostics, `dump` accepts a token sample of at most 32 tokens and
captures every decoder block plus selected attention primitives. Native
`--dump-layer-inputs` captures graph inputs. Compare block outputs via the next
block's normalization input:

```sh
python qwen35_validation.py dump --samples /tmp/qwen/data/primitive.json --output /tmp/qwen/hf-trace
python qwen35_check_artifacts.py layers --reference /tmp/qwen/hf-trace --candidate /tmp/qwen/native/float-layers-000 --output /tmp/qwen/layers.json
python -m unittest discover -p 'test_qwen35*.py' -v
python qwen35_tiny_fixture.py --output /tmp/qwen/tiny
```

The tiny fixture is an independent seed-42 Hugging Face oracle with one linear
attention block and one full attention block. It checks checkpoint loading,
per-head query/gate layout, effective normalization weights, partial RoPE, tied
head behavior, intermediate outputs, and full logits without loading 0.8B weights.

The chat boundary case may stop early at EOS. To test an actual 192+64 cached
continuation, separately generate the numerical `context256` case in
`data/prompts.json`, then freeze its generated history:

```sh
python qwen35_validation.py generate --prompts /tmp/qwen/data/prompts.json --output /tmp/qwen/numerical-generation.json
python qwen35_validation.py prepare-boundary --generation-reference /tmp/qwen/numerical-generation.json --output /tmp/qwen/boundary
python qwen35_validation.py reference --samples /tmp/qwen/boundary/full-history.json --output /tmp/qwen/hf-boundary
# Run native generation with boundary/prompt-tokens.json and --num-tokens 64.
python qwen35_validation.py compare --reference /tmp/qwen/hf-boundary --candidate /tmp/qwen/native-boundary --mode float --require-source-identity --candidate-generated-history --output /tmp/qwen/cache-boundary.json
```

This compares all 255 computed rows from native cached generation against one
full Hugging Face forward pass, including every decode step. There are 256 final
tokens: the last generated token is predicted from the preceding 255 input tokens.
The history must match exactly, and generation after EOS is rejected. Repetitive
raw continuation remains numerical evidence only.

For integer development diagnostics, compare a short fixed sequence against both
HF block outputs and native float node inputs:

```sh
python qwen35_check_artifacts.py layers --reference /tmp/qwen/hf-trace --candidate /tmp/qwen/native-int/int-layers-000 --diagnostic --output /tmp/qwen/int-block-diagnostic.json
python qwen35_check_artifacts.py inputs --reference /tmp/qwen/native-float/float-layers-000 --candidate /tmp/qwen/native-int/int-layers-000 --output /tmp/qwen/int-node-diagnostic.json
```

`inputs` checks source identity and the quantization digest, compares each original
graph node and input port, and reports how many reference elements lie outside
that input's calibrated integer domain. Endpoint counts can indicate saturation;
a large error increase with no domain exceedance can instead indicate lookup or
rounding precision. These reports localize errors and are not held-out acceptance.
