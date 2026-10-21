"""--enable-unified-memory serves the same tokens as the static pools on a tiny
Muse Glimmer.

Muse models put full attention on every fourth layer counting from the last, so
with 10 layers 7 sliding-window layers share the buffers of 3 full-attention
layers in 3 page groups, the last group one layer short. Each prompt is longer
than the 128-token window, and the extended one reuses a cached prefix whose
early sliding-window pages were already released.
"""

import json
import os
import tempfile
import unittest

import requests
import torch
from safetensors.torch import save_file

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_tpu_ci(
    est_time=300,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "muse-glimmer", "unified-memory", "logprob"],
)

NUM_LAYERS = 10
SLIDING_WINDOW = 128
CONFIG = {
    "architectures": ["MuseGlimmerForCausalLM"],
    "model_type": "muse_glimmer",
    "vocab_size": 4096,
    "hidden_size": 1024,
    "intermediate_size": 2048,
    "num_hidden_layers": NUM_LAYERS,
    # The released attention shape: 2 KV heads of 128.
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "max_position_embeddings": 4096,
    "sliding_window": SLIDING_WINDOW,
    "layer_types": [
        "full_attention" if (NUM_LAYERS - 1 - i) % 4 == 0 else "sliding_attention"
        for i in range(NUM_LAYERS)
    ],
    "no_rope_layers": [
        0 if (NUM_LAYERS - 1 - i) % 4 == 0 else 1 for i in range(NUM_LAYERS)
    ],
    # Attention logits of unit spread on random weights; the released scale makes
    # them sharp enough that bf16 rounding flips near-tie tokens.
    "qk_scale_factor": 8.0,
    # As the released checkpoint's config translation sets it.
    "rope_is_neox_style": True,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "torch_dtype": "bfloat16",
}
MAX_TOTAL_TOKENS = 8192
SERVER_ARGS = [
    "--skip-tokenizer-init",
    "--tp-size",
    "8",
    "--attention-backend",
    "pallas_tpu_rpa_v3",
    "--kv-cache-dtype",
    "bfloat16",
    "--page-size",
    "64",
    "--context-length",
    "2048",
    "--chunked-prefill-size",
    "256",
    "--max-prefill-tokens",
    "256",
    "--max-running-requests",
    "4",
    "--max-total-tokens",
    str(MAX_TOTAL_TOKENS),
    "--grammar-backend",
    "none",
    # The TPU distributed top-k sampler can halt at TP8; greedy needs no top-k.
    "--sampling-backend",
    "pytorch",
    "--enable-torch-compile",
    "--torch-compile-max-bs",
    "4",
    "--tpu-compile-tokens",
    "256",
    "--tpu-compile-bs",
    "1",
    "4",
]
SERVER_LAUNCH_TIMEOUT = 1800
MAX_NEW_TOKENS = 32


def _make_checkpoint(model_dir: str) -> None:
    g = torch.Generator().manual_seed(0)
    c = CONFIG
    hidden, inter = c["hidden_size"], c["intermediate_size"]
    q_width = c["num_attention_heads"] * c["head_dim"]
    kv_width = c["num_key_value_heads"] * c["head_dim"]

    def randn(*shape, std):
        return (torch.randn(*shape, generator=g) * std).bfloat16()

    tensors = {
        "model.embed_tokens.weight": randn(c["vocab_size"], hidden, std=1.0),
        "model.norm.weight": 1.0 + randn(hidden, std=0.1),
        "lm_head.weight": randn(c["vocab_size"], hidden, std=0.5),
    }
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}"
        for name, rows in (
            ("q_proj", q_width),
            ("k_proj", kv_width),
            ("v_proj", kv_width),
            ("output_gate_proj", q_width),
        ):
            tensors[f"{p}.self_attn.{name}.weight"] = randn(
                rows, hidden, std=hidden**-0.5
            )
        tensors[f"{p}.self_attn.o_proj.weight"] = randn(
            hidden, q_width, std=q_width**-0.5
        )
        tensors[f"{p}.mlp.gate_proj.weight"] = randn(inter, hidden, std=hidden**-0.5)
        tensors[f"{p}.mlp.up_proj.weight"] = randn(inter, hidden, std=hidden**-0.5)
        tensors[f"{p}.mlp.down_proj.weight"] = randn(hidden, inter, std=inter**-0.5)
        # Stored as offsets from 1, as the released checkpoint does.
        for name in (
            "input_layernorm",
            "post_attn_norm",
            "post_attention_layernorm",
            "post_ffn_norm",
        ):
            tensors[f"{p}.{name}.weight"] = randn(hidden, std=0.1)
    save_file(tensors, os.path.join(model_dir, "model.safetensors"))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(CONFIG, f)


def _prompt(length: int, seed: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, CONFIG["vocab_size"], (length,), generator=g).tolist()


SHARED_PREFIX = _prompt(400, seed=3)
# Cold prompts, then a prompt whose first 400 tokens hit the radix cache.
PROMPTS = [
    _prompt(600, seed=1),
    _prompt(1000, seed=2),
    SHARED_PREFIX,
    SHARED_PREFIX + _prompt(300, seed=4),
]


def _serve(model_dir: str, extra_args: list[str]) -> list[dict]:
    process = popen_launch_server(
        model_dir,
        DEFAULT_URL_FOR_TEST,
        timeout=SERVER_LAUNCH_TIMEOUT,
        other_args=SERVER_ARGS + extra_args,
    )
    try:
        info = requests.get(DEFAULT_URL_FOR_TEST + "/server_info", timeout=60).json()
        outputs = [{"tpu_unified_kv_pool": bool(info.get("tpu_unified_kv_pool"))}]
        for ids in PROMPTS:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "input_ids": ids,
                    "sampling_params": {
                        "max_new_tokens": MAX_NEW_TOKENS,
                        "temperature": 0.0,
                        "ignore_eos": True,
                    },
                    "return_logprob": True,
                },
                timeout=600,
            )
            response.raise_for_status()
            meta = response.json()["meta_info"]
            outputs.append(
                {
                    "tokens": [t[1] for t in meta["output_token_logprobs"]],
                    "logprobs": [t[0] for t in meta["output_token_logprobs"]],
                    "cached_tokens": meta.get("cached_tokens", 0),
                }
            )
        return outputs
    finally:
        kill_process_tree(process.pid)


class TestMuseGlimmerUnifiedMemory(CustomTestCase):
    def test_unified_pool_serves_the_static_pool_tokens(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _make_checkpoint(model_dir)
            static = _serve(model_dir, [])
            unified = _serve(model_dir, ["--enable-unified-memory"])

        self.assertEqual(static[0], {"tpu_unified_kv_pool": False})
        self.assertEqual(unified[0], {"tpu_unified_kv_pool": True})
        for want, got in zip(static[1:], unified[1:]):
            self.assertEqual(got["tokens"], want["tokens"])
            gap = max(abs(a - b) for a, b in zip(got["logprobs"], want["logprobs"]))
            self.assertLess(gap, 1e-3)
        # The extended prompt reused the cached prefix in both arms.
        self.assertGreaterEqual(unified[-1]["cached_tokens"], 384)
        self.assertGreaterEqual(static[-1]["cached_tokens"], 384)


if __name__ == "__main__":
    unittest.main()
