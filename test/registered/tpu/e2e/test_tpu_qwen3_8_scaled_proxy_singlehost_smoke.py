"""Scaled Qwen3.8-2.4T-A95B-FP8 proxy TPU smoke & equivalence test on one v7x-8 host.

Builds a deterministic 4-layer (3 GDN + 1 Full-Attention) checkpoint matching the
exact structural layout of ``Qwen/Qwen3.8-2.4T-A95B-FP8`` at 1/4 head count:
  * ``architectures = ["Qwen3_5MoeForCausalLM"]``, ``model_type = "qwen3_5_moe_text``
  * GDN: ``linear_num_key_heads = 4``, ``linear_num_value_heads = 32``,
    ``linear_key_head_dim = 128``, ``linear_value_head_dim = 128``, split
    ``in_proj_qkv`` + ``in_proj_z`` BF16 weights
  * Full Attention: ``num_attention_heads = 16``, ``num_key_value_heads = 4``,
    ``head_dim = 256``, ``attn_output_gate = True``, ``partial_rotary_factor = 0.25``
  * MoE: ``64`` block-[128, 128] FP8 routed experts (``top_k = 10``) + ``1`` BF16
    shared expert + ``modules_to_not_convert`` matching the 2.4T checkpoint
  * Indexed ``mtp.*`` shard to exercise EP-local target skipping in both
    ``EpLocalExpertReader`` and ``EPLocalRunaiModelStreamerLoader``

On an 8-chiplet v7x-8 host, runs three 8-chiplet compiled servers and checks:
  1. ``TP=8, DP=2, EP=8`` (``attn_tp=4``: 1x GDN Q/K, ``DCP=1``) reference
  2. ``TP=8, DCP=2, EP=8`` (``attn_tp=8``: 2x GDN Q/K replication + exact 2.4T
     per-rank ``(2 Q, 1 KV, d=256)`` DCP + ``--load-format runai_streamer``)
  3. ``TP=8, DCP=4, EP=8`` (``attn_tp=8``: 2x GDN Q/K replication + 4-rank DCP LSE
     merge + ``--load-format safetensors --weight-loader-disable-mmap``)
asserting zero Dynamo recompilations after ``TpuGraphRunner precompile done`` and
prefill next-token top-k logprob agreement across all three configurations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import unittest

import requests
import safetensors.torch
import torch

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_tpu_ci(
    est_time=480,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "qwen3.8-max", "smoke", "dcp", "compile"],
)

VOCAB_SIZE = 4096
HIDDEN_SIZE = 1024
NUM_LAYERS = 4
LAYER_TYPES = [
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
]
NUM_ATTN_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
LINEAR_NUM_K_HEADS = 4
LINEAR_NUM_V_HEADS = 32
LINEAR_K_DIM = 128
LINEAR_V_DIM = 128
CONV_KERNEL = 4
NUM_EXPERTS = 64
TOP_K = 10
MOE_INTERMEDIATE_SIZE = 512
SHARED_EXPERT_INTERMEDIATE_SIZE = 512
BLOCK_SIZE = 128

PRECOMPILE_DONE = "TpuGraphRunner precompile done"
MAX_NEW_TOKENS = 8


def _prompt(length: int, seed: int) -> list[int]:
    return [
        (i * 7919 + seed * 104729) % (VOCAB_SIZE - 256) + 128 for i in range(length)
    ]


TEXT_BATCHES = [
    [_prompt(96, 1)],
    [_prompt(45, 2), _prompt(260, 3)],
]


def _quantize_block_fp8(
    weight_f32: torch.Tensor, block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize 2D ``[N, K]`` float32 tensor into ``(fp8_e4m3fn, weight_scale_inv)``."""
    n, k = weight_f32.shape
    assert n % block_size == 0 and k % block_size == 0
    blocks = weight_f32.view(
        n // block_size, block_size, k // block_size, block_size
    ).permute(0, 2, 1, 3)
    amax = blocks.abs().amax(dim=(-1, -2)).clamp(min=1e-6)
    fmax = torch.finfo(torch.float8_e4m3fn).max
    scale_inv = (amax / fmax).to(torch.float32)
    q = (blocks / scale_inv.unsqueeze(-1).unsqueeze(-1)).clamp(-fmax, fmax)
    fp8 = q.to(torch.float8_e4m3fn).permute(0, 2, 1, 3).contiguous().view(n, k)
    return fp8, scale_inv.contiguous()


def _write_minimal_fast_tokenizer(out_dir: str, vocab_size: int = VOCAB_SIZE) -> None:
    vocab = {"<unk>": 0, "<bos>": 1, "<eos>": 2}
    for i in range(3, vocab_size):
        vocab[f"tok_{i}"] = i
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {
                "id": 0,
                "content": "<unk>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 1,
                "content": "<bos>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
            {
                "id": 2,
                "content": "<eos>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            },
        ],
        "normalizer": None,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": None,
        "decoder": None,
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>",
        },
    }
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_json, f)
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "unk_token": "<unk>",
            },
            f,
        )


def build_scaled_qwen38_fp8_checkpoint(out_dir: str, seed: int = 20261118) -> str:
    """Write a deterministic 4-layer scaled Qwen3.8-2.4T-A95B-FP8 checkpoint."""
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator().manual_seed(seed)

    modules_to_not_convert = ["lm_head", "model.embed_tokens"]
    for layer_id, ltype in enumerate(LAYER_TYPES):
        prefix = f"model.layers.{layer_id}"
        if ltype == "linear_attention":
            modules_to_not_convert.extend(
                [
                    f"{prefix}.linear_attn.conv1d",
                    f"{prefix}.linear_attn.in_proj_a",
                    f"{prefix}.linear_attn.in_proj_b",
                    f"{prefix}.linear_attn.in_proj_qkv",
                    f"{prefix}.linear_attn.in_proj_z",
                    f"{prefix}.linear_attn.out_proj",
                ]
            )
        else:
            modules_to_not_convert.extend(
                [
                    f"{prefix}.self_attn.k_proj",
                    f"{prefix}.self_attn.o_proj",
                    f"{prefix}.self_attn.q_proj",
                    f"{prefix}.self_attn.v_proj",
                ]
            )
        modules_to_not_convert.extend(
            [
                f"{prefix}.mlp.gate",
                f"{prefix}.mlp.shared_expert.down_proj",
                f"{prefix}.mlp.shared_expert.gate_proj",
                f"{prefix}.mlp.shared_expert.up_proj",
                f"{prefix}.mlp.shared_expert_gate",
            ]
        )

    config = {
        "architectures": ["Qwen3_5MoeForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "attn_output_gate": True,
        "bos_token_id": 1,
        "dtype": "bfloat16",
        "eos_token_id": 2,
        "full_attention_interval": 4,
        "head_dim": HEAD_DIM,
        "hidden_act": "silu",
        "hidden_size": HIDDEN_SIZE,
        "initializer_range": 0.02,
        "layer_types": LAYER_TYPES,
        "linear_conv_kernel_dim": CONV_KERNEL,
        "linear_key_head_dim": LINEAR_K_DIM,
        "linear_num_key_heads": LINEAR_NUM_K_HEADS,
        "linear_num_value_heads": LINEAR_NUM_V_HEADS,
        "linear_value_head_dim": LINEAR_V_DIM,
        "mamba_ssm_dtype": "float32",
        "max_position_embeddings": 16384,
        "model_type": "qwen3_5_moe_text",
        "moe_intermediate_size": MOE_INTERMEDIATE_SIZE,
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": False,
        "norm_topk_prob": True,
        "num_attention_heads": NUM_ATTN_HEADS,
        "num_experts": NUM_EXPERTS,
        "num_experts_per_tok": TOP_K,
        "num_hidden_layers": NUM_LAYERS,
        "num_key_value_heads": NUM_KV_HEADS,
        "output_gate_type": "swish",
        "output_router_logits": False,
        "pad_token_id": None,
        "partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
        "shared_expert_intermediate_size": SHARED_EXPERT_INTERMEDIATE_SIZE,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": VOCAB_SIZE,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_per_tensor": False,
            "act_per_tensor": False,
            "weight_block_size": [BLOCK_SIZE, BLOCK_SIZE],
            "modules_to_not_convert": sorted(modules_to_not_convert),
        },
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    _write_minimal_fast_tokenizer(out_dir, VOCAB_SIZE)

    scale = 0.02
    shard1: dict[str, torch.Tensor] = {}
    shard2: dict[str, torch.Tensor] = {}

    shard1["model.embed_tokens.weight"] = (
        torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=g) * scale
    ).bfloat16()
    shard1["model.norm.weight"] = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16)
    shard1["lm_head.weight"] = (
        torch.randn(VOCAB_SIZE, HIDDEN_SIZE, generator=g) * scale
    ).bfloat16()

    key_dim = LINEAR_NUM_K_HEADS * LINEAR_K_DIM  # 512
    val_dim = LINEAR_NUM_V_HEADS * LINEAR_V_DIM  # 4096

    for layer_id, ltype in enumerate(LAYER_TYPES):
        target = shard1 if layer_id < 2 else shard2
        lp = f"model.layers.{layer_id}"
        target[f"{lp}.input_layernorm.weight"] = torch.zeros(
            HIDDEN_SIZE, dtype=torch.bfloat16
        )
        target[f"{lp}.post_attention_layernorm.weight"] = torch.zeros(
            HIDDEN_SIZE, dtype=torch.bfloat16
        )
        if ltype == "linear_attention":
            target[f"{lp}.linear_attn.in_proj_qkv.weight"] = (
                torch.randn(key_dim * 2 + val_dim, HIDDEN_SIZE, generator=g) * scale
            ).bfloat16()
            target[f"{lp}.linear_attn.in_proj_z.weight"] = (
                torch.randn(val_dim, HIDDEN_SIZE, generator=g) * scale
            ).bfloat16()
            target[f"{lp}.linear_attn.in_proj_b.weight"] = (
                torch.randn(LINEAR_NUM_V_HEADS, HIDDEN_SIZE, generator=g) * scale
            ).bfloat16()
            target[f"{lp}.linear_attn.in_proj_a.weight"] = (
                torch.randn(LINEAR_NUM_V_HEADS, HIDDEN_SIZE, generator=g) * scale
            ).bfloat16()
            target[f"{lp}.linear_attn.conv1d.weight"] = (
                torch.randn(key_dim * 2 + val_dim, 1, CONV_KERNEL, generator=g) * scale
            ).bfloat16()
            target[f"{lp}.linear_attn.A_log"] = torch.linspace(
                0.5, 2.0, LINEAR_NUM_V_HEADS, dtype=torch.float32
            )
            target[f"{lp}.linear_attn.dt_bias"] = torch.full(
                (LINEAR_NUM_V_HEADS,), 0.1, dtype=torch.bfloat16
            )
            target[f"{lp}.linear_attn.norm.weight"] = torch.ones(
                LINEAR_V_DIM, dtype=torch.bfloat16
            )
            target[f"{lp}.linear_attn.out_proj.weight"] = (
                torch.randn(HIDDEN_SIZE, val_dim, generator=g) * scale
            ).bfloat16()
        else:
            target[f"{lp}.self_attn.q_proj.weight"] = (
                torch.randn(NUM_ATTN_HEADS * 2 * HEAD_DIM, HIDDEN_SIZE, generator=g)
                * scale
            ).bfloat16()
            target[f"{lp}.self_attn.k_proj.weight"] = (
                torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE, generator=g) * scale
            ).bfloat16()
            target[f"{lp}.self_attn.v_proj.weight"] = (
                torch.randn(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE, generator=g) * scale
            ).bfloat16()
            target[f"{lp}.self_attn.o_proj.weight"] = (
                torch.randn(HIDDEN_SIZE, NUM_ATTN_HEADS * HEAD_DIM, generator=g) * scale
            ).bfloat16()
            target[f"{lp}.self_attn.q_norm.weight"] = torch.zeros(
                HEAD_DIM, dtype=torch.bfloat16
            )
            target[f"{lp}.self_attn.k_norm.weight"] = torch.zeros(
                HEAD_DIM, dtype=torch.bfloat16
            )

        target[f"{lp}.mlp.gate.weight"] = (
            torch.randn(NUM_EXPERTS, HIDDEN_SIZE, generator=g) * scale
        ).bfloat16()
        for proj_name, shape in (
            ("gate_proj", (SHARED_EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE)),
            ("up_proj", (SHARED_EXPERT_INTERMEDIATE_SIZE, HIDDEN_SIZE)),
            ("down_proj", (HIDDEN_SIZE, SHARED_EXPERT_INTERMEDIATE_SIZE)),
        ):
            target[f"{lp}.mlp.shared_expert.{proj_name}.weight"] = (
                torch.randn(*shape, generator=g) * scale
            ).bfloat16()
        target[f"{lp}.mlp.shared_expert_gate.weight"] = (
            torch.randn(1, HIDDEN_SIZE, generator=g) * scale
        ).bfloat16()

        for eid in range(NUM_EXPERTS):
            for proj_name, shape in (
                ("gate_proj", (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)),
                ("up_proj", (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)),
                ("down_proj", (HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE)),
            ):
                raw = torch.randn(*shape, generator=g) * scale
                fp8_w, scale_inv = _quantize_block_fp8(raw)
                key_base = f"{lp}.mlp.experts.{eid}.{proj_name}"
                target[f"{key_base}.weight"] = fp8_w
                target[f"{key_base}.weight_scale_inv"] = scale_inv

    # Unused MTP shard indexed in model.safetensors.index.json, mirroring 2.4T.
    shard3 = {
        "mtp.fc.weight": torch.zeros(
            HIDDEN_SIZE, HIDDEN_SIZE * 2, dtype=torch.bfloat16
        ),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": torch.zeros(
            HIDDEN_SIZE, MOE_INTERMEDIATE_SIZE, dtype=torch.float8_e4m3fn
        ),
        "mtp.layers.0.mlp.experts.0.down_proj.weight_scale_inv": torch.ones(
            HIDDEN_SIZE // BLOCK_SIZE,
            MOE_INTERMEDIATE_SIZE // BLOCK_SIZE,
            dtype=torch.float32,
        ),
    }

    shards = [
        ("model-00001-of-00003.safetensors", shard1),
        ("model-00002-of-00003.safetensors", shard2),
        ("model-00003-of-00003.safetensors", shard3),
    ]
    weight_map: dict[str, str] = {}
    for fname, tensors in shards:
        safetensors.torch.save_file(tensors, os.path.join(out_dir, fname))
        for k in tensors:
            weight_map[k] = fname

    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {"total_size": 0}, "weight_map": weight_map}, f, indent=2
        )
    return out_dir


def _query_batch(prompts: list[list[int]]) -> list[dict]:
    resp = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "input_ids": prompts,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": 5,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def _read_log(log_file) -> str:
    with open(log_file.name, errors="replace") as f:
        return f.read()


def _serve_leg(model_dir: str, extra_args: list[str]) -> tuple[list[list[dict]], str]:
    base_args = [
        "--device",
        "tpu",
        "--dtype",
        "bfloat16",
        "--quantization",
        "fp8",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--attention-backend",
        "pallas_tpu_rpa_batched",
        "--page-size",
        "256",
        "--context-length",
        "2048",
        "--chunked-prefill-size",
        "1024",
        "--max-prefill-tokens",
        "1024",
        "--max-running-requests",
        "4",
        "--mem-fraction-static",
        "0.45",
        "--disable-radix-cache",
        "--grammar-backend",
        "none",
        "--enable-torch-compile",
        "--torch-compile-max-bs",
        "4",
        "--tpu-compile-tokens",
        "1024",
        "--tpu-compile-bs",
        "1",
        "2",
        "4",
    ]
    log_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".log")
    process = popen_launch_server(
        model_dir,
        DEFAULT_URL_FOR_TEST,
        timeout=1200,
        other_args=base_args + extra_args,
        env={"TORCH_LOGS": "+recompiles"},
        return_stdout_stderr=(log_file, log_file),
        device="tpu",
    )
    try:
        deadline = time.monotonic() + 180
        while "froze torch.compile" not in _read_log(log_file):
            if time.monotonic() > deadline:
                break
            time.sleep(1)
        results = [_query_batch(batch) for batch in TEXT_BATCHES]
        return results, _read_log(log_file)
    except Exception:
        tail = "\n".join(_read_log(log_file).splitlines()[-120:])
        print(f"[scaled-proxy] server log tail before teardown:\n{tail}", flush=True)
        raise
    finally:
        terminate_and_kill_process_tree(process, wait_timeout=60)
        log_file.close()


def _max_top1_logprob_diff(
    ref_batches: list[list[dict]], cand_batches: list[list[dict]]
) -> float:
    max_diff = 0.0
    for ref_b, cand_b in zip(ref_batches, cand_batches):
        for r_item, c_item in zip(ref_b, cand_b):
            r_top = r_item["meta_info"]["output_top_logprobs"][0][0]
            c_top = c_item["meta_info"]["output_top_logprobs"][0][0]
            # Compare first generated token's top-1 logprob magnitude.
            diff = abs(float(r_top[0]) - float(c_top[0]))
            max_diff = max(max_diff, diff)
    return max_diff


class TestQwen38ScaledProxySinglehostSmoke(CustomTestCase):
    def test_qk_replication_and_dcp_legs_match_reference_without_recompiles(self):
        ckpt_tmp = tempfile.TemporaryDirectory(prefix="qwen38_scaled_proxy_")
        self.addCleanup(ckpt_tmp.cleanup)
        model_dir = build_scaled_qwen38_fp8_checkpoint(ckpt_tmp.name)

        # Leg A: TP=8, DP=2, EP=8 -> attn_tp=4 (1x GDN Q/K, no DCP).
        # Under DP=2, max_running_requests and chunked_prefill_size divide by 2 per DP group.
        ref_out, ref_log = _serve_leg(
            model_dir,
            [
                "--tp-size",
                "8",
                "--dp-size",
                "2",
                "--enable-dp-attention",
                "--enable-dp-lm-head",
                "--ep-size",
                "8",
                "--max-running-requests",
                "8",
                "--chunked-prefill-size",
                "2048",
                "--max-prefill-tokens",
                "2048",
                "--load-format",
                "safetensors",
                "--weight-loader-disable-mmap",
            ],
        )

        # Leg B: TP=8, DCP=2, EP=8 -> attn_tp=8 (2x GDN Q/K rep + (2 Q, 1 KV, d=256) DCP + Run:AI)
        dcp2_out, dcp2_log = _serve_leg(
            model_dir,
            [
                "--tp-size",
                "8",
                "--dcp-size",
                "2",
                "--dcp-comm-backend",
                "ag_rs",
                "--ep-size",
                "8",
                "--load-format",
                "runai_streamer",
            ],
        )

        # Leg C: TP=8, DCP=4, EP=8 -> attn_tp=8 (2x GDN Q/K rep + 4-rank DCP LSE merge)
        dcp4_out, dcp4_log = _serve_leg(
            model_dir,
            [
                "--tp-size",
                "8",
                "--dcp-size",
                "4",
                "--dcp-comm-backend",
                "ag_rs",
                "--ep-size",
                "8",
                "--load-format",
                "safetensors",
                "--weight-loader-disable-mmap",
            ],
        )

        for name, log in (("dp2_ref", ref_log), ("dcp2", dcp2_log), ("dcp4", dcp4_log)):
            idx = log.rfind(PRECOMPILE_DONE)
            self.assertGreaterEqual(idx, 0, f"{name}: {PRECOMPILE_DONE!r} missing")
            tail = log[idx:]
            self.assertNotIn("Recompiling function", tail, name)
            self.assertNotIn("Detected recompile", tail, name)

        for name, cand in (("dcp2", dcp2_out), ("dcp4", dcp4_out)):
            diff = _max_top1_logprob_diff(ref_out, cand)
            print(
                f"[scaled-proxy] {name} vs dp2_ref max first-step top1 logprob diff = {diff:.6f}"
            )
            self.assertFalse(math.isnan(diff), f"{name}: NaN logprob")
            self.assertLess(
                diff, 0.15, f"{name}: first-step top1 logprob diverged ({diff})"
            )


def _write_checkpoint_cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build scaled Qwen3.8 FP8 proxy checkpoint."
    )
    parser.add_argument(
        "--write-checkpoint",
        type=str,
        required=True,
        help="Write scaled Qwen3.8 FP8 checkpoint to directory and exit.",
    )
    args = parser.parse_args()
    path = build_scaled_qwen38_fp8_checkpoint(args.write_checkpoint)
    print(f"Wrote scaled Qwen3.8 FP8 checkpoint to {path}")


if __name__ == "__main__":
    if any(arg.startswith("--write-checkpoint") for arg in sys.argv[1:]):
        _write_checkpoint_cli()
    else:
        unittest.main()
