"""Kimi-K3 with the decode megakernel, four tpu7x hosts (tp32), one request at a time.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/server_mk.py

The megakernel runs all 93 decoder layers as one Pallas call per step and owns
their weights, so it serves only batch 1: one request in flight, no radix cache,
and one compiled decode batch size. Prefill chunks run through the same kernel
in 8-token row blocks, so the chunk size must be a multiple of 8.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3.server import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        "max_running_requests": 1,
        "torch_compile_max_bs": 1,
        "tpu_compile_bs": [1],
        "disable_radix_cache": True,
        # Only embed, the final norm and lm_head stay in sglang; the kernel's own
        # weights and VMEM-staged copies count as model memory.
        "mem_fraction_static": 0.9,
    }
)
# The kernel replaces the decoder layers these per-op kernels speed up; the
# measured runs left them off.
PER_OP_ONLY_ENV = (
    "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD",
    "SGLANG_TPU_K3_STATIC_FP8",
    "SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV",
    "SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE",
    "SGLANG_K3_MERGE_MOE_ALLREDUCE",
    "SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL",
    "SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER",
)
for key in PER_OP_ONLY_ENV:
    CONFIG["runtime_env"].pop(key)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL": "1",
        # fp8 dense weights, bf16 reduction transport, o_proj reduced through the
        # chip pair; GSM8K-checked (see the cookbook page).
        "SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX": "dense_fp8,pair_oproj,bf16_transport",
    }
)
