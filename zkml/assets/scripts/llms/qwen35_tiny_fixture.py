#!/usr/bin/env python3
"""Generate the independent HF oracle for the native Qwen3.5 graph regression.

Use requirements-qwen35.txt; no network or downloaded model is needed.
  python qwen35_tiny_fixture.py --output /tmp/deep-prove-qwen35/tiny
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5ForCausalLM

from qwen35_validation import PINNED, sha256, write_json
import importlib.metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    versions = {name: importlib.metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise RuntimeError(f"Reference version mismatch: {versions}")
    path = Path(args.output)
    path.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    config = Qwen3_5TextConfig(
        hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=4,
        vocab_size=32, layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True, rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.,
                         "partial_rotary_factor": 0.5, "mrope_interleaved": True,
                         "mrope_section": [1, 0, 0]},
    )
    config._attn_implementation = "eager"
    model = Qwen3_5ForCausalLM(config).eval()
    with torch.no_grad():
        for name, value in model.named_parameters():
            if "norm.weight" in name and ".linear_attn.norm." not in name:
                value.copy_(torch.linspace(-0.2, 0.2, value.numel()).view_as(value))
    # Adding metadata after construction preserves the frozen RNG oracle.
    config.eos_token_id = 31
    config.attn_output_gate = True
    Qwen3_5Config(text_config=config.to_dict()).to_json_file(path / "config.json")
    weights = {name.replace("model.", "model.language_model.", 1): value.contiguous()
               for name, value in model.state_dict().items() if name != "lm_head.weight"}
    save_file(weights, str(path / "model.safetensors"))
    write_json(path / "tokens.json", [[1, 2, 3]])
    traces = {}
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(lambda module, inputs, result, i=i:
                                    traces.__setitem__(f"layer_{i}", result.detach().tolist()))
    with torch.inference_mode():
        logits = model(torch.tensor([[1, 2, 3]]), use_cache=False).logits[0]
    logits.numpy().astype("<f4").tofile(path / "hf_000.f32")
    write_json(path / "logits.json", logits.tolist())
    write_json(path / "traces.json", traces)
    write_json(path / "manifest.json", {
        "records": [{"sample": 0, "mode": "hf", "file": "hf_000.f32", "shape": [3, 32],
                     "tokens": [1, 2, 3], "sha256": sha256(path / "hf_000.f32")}],
        "versions": versions, "seed": 42, "source_script_sha256": sha256(__file__),
        "description": "Two-layer synthetic HF oracle; ordinary RMS weights linspace(-.2,.2)",
    })
    loaded, info = Qwen3_5ForCausalLM.from_pretrained(
        path, config=config, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager", output_loading_info=True,
    )
    if any(info.values()):
        raise RuntimeError(f"Synthetic checkpoint load mismatch: {info}")
    with torch.inference_mode():
        torch.testing.assert_close(loaded(torch.tensor([[1, 2, 3]]), use_cache=False).logits[0],
                                   logits, rtol=0, atol=0)
    print(json.dumps({"output": str(path), "tensors": len(weights),
                      "top1": logits.argmax(-1).tolist()}))


if __name__ == "__main__":
    main()
