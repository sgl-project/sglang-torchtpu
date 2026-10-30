"""Kimi-K3 P/D on the megakernel programs (K3_PROGRAM), one 2x2x4 tpu7x slice
(tp32) per role, for InferenceX AgentX.

    CI_PD_ROLE=prefill python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/pd_program.py
    CI_PD_ROLE=decode  python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/pd_program.py

Each role runs one program (sglang_tpu/k3_programs): the prefill program on
sglang's chunked prefill, the decode program (DSpark with the MLA draft,
Inferact/Kimi-K3-DSpark) on its decode batches, both over the unified pools
Raiden moves between them. Weights come pre-sharded per host:

    python3 -m sglang_tpu.k3_programs.tools preshard k3 \\
        --checkpoint /weights/Kimi-K3 --output $SGLANG_TPU_K3_PROGRAM_PRESHARDED

Environment:
    CI_PD_ROLE                      prefill | decode
    K3_PROGRAM_CONCURRENCY          client concurrency (default 8); the decode
                                    admits twice that, as vLLM-TorchTPU's AgentX
                                    sweep does
    K3_PROGRAM_CONTEXT              context length (default 1048576)
    K3_PROGRAM_KV_DTYPE             fp8_e4m3 (default, as vLLM-TorchTPU's AgentX
                                    sweep) or bf16
    SGLANG_TPU_K3_PROGRAM_PRESHARDED pre-sharded weights directory
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3.server import CONFIG as SERVER_CONFIG

ROLE = os.environ.get("CI_PD_ROLE", "prefill")
CONCURRENCY = int(os.environ.get("K3_PROGRAM_CONCURRENCY", "8"))
CONTEXT_LENGTH = int(os.environ.get("K3_PROGRAM_CONTEXT", str(1 << 20)))
KV_DTYPE = os.environ.get("K3_PROGRAM_KV_DTYPE") or "fp8_e4m3"
# One logical block: 256 tokens on each of the 32 ranks in bf16, 512 in fp8.
PAGE_SIZE = 32 * (512 if KV_DTYPE.startswith("fp8") else 256)
# The decode admits twice the client concurrency, up to the 64 rows the
# programs' plain decode is checked to (without attention DP); the prefill
# holds what it is computing or transferring.
RUNNING = min(64, max(4, 2 * CONCURRENCY)) if ROLE == "decode" else max(4, CONCURRENCY)
# A prefill chunk is one page: every full chunk ends on a KDA checkpoint the
# prefix cache keeps (the programs' windows split it as they need).
PREFILL_CHUNK = PAGE_SIZE
# Prefill KDA checkpoints the prefix cache may hold (each three blocks of the
# 29 pools: about 28 MB a rank). The prefill program's weights take ~74 GB of
# a rank's 96 GB (experts 45, dense 6, its sequence-parallel copies of the
# latent and shared projections 22.5), so its pools share ~12 GB: 128
# checkpoints, 96 tails (38 MB each) and ~8M tokens of KV at 580 bytes.
PREFILL_CHECKPOINTS = int(os.environ.get("K3_PROGRAM_PREFILL_CHECKPOINTS", "128"))

CONFIG = deepcopy(SERVER_CONFIG)
server_args = CONFIG["server_args"]
# sglang never runs the (layerless) target model's forward: nothing to compile,
# and its own KDA state pool is replaced by the unified pools.
for key in (
    "tpu_compile_tokens",
    "tpu_compile_bs",
    "torch_compile_max_bs",
    "enable_torch_compile",
    "mamba_ssm_dtype",
):
    server_args.pop(key, None)
server_args.update(
    {
        "host": "0.0.0.0",
        "speculative_algorithm": "K3_PROGRAM",
        "disaggregation_mode": ROLE,
        "disaggregation_transfer_backend": "raiden",
        "kv_cache_dtype": KV_DTYPE,
        "page_size": PAGE_SIZE,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PREFILL_CHUNK,
        "max_prefill_tokens": PREFILL_CHUNK,
        "max_running_requests": RUNNING,
        "mem_fraction_static": 0.92,
        "skip_server_warmup": True,
    }
)
if ROLE == "decode":
    server_args.update(
        disable_radix_cache=True,
        # A KDA state slot and a full context of KV per running request, and
        # as many again for requests still arriving over the transfer: the
        # decode never retracts.
        max_mamba_cache_size=2 * RUNNING,
        max_total_tokens=2 * RUNNING * CONTEXT_LENGTH,
    )
else:
    # AgentX turns extend the session's earlier prompt: the prefill keeps them
    # in its radix cache, the KV pages and a KDA checkpoint per page-aligned
    # chunk, and sizes the KV pool from what the checkpoints leave.
    # Prompt ends past their last full page, for hits inside a page (sglang_tails).
    CONFIG["runtime_env"]["SGLANG_TPU_K3_PROGRAM_PREFIX_TAILS"] = os.environ.get(
        "K3_PROGRAM_PREFIX_TAILS", "96"
    )
    server_args.update(
        disable_radix_cache=False,
        mamba_radix_cache_strategy="extra_buffer",
        mamba_track_interval=PAGE_SIZE,
        max_mamba_cache_size=PREFILL_CHECKPOINTS,
        max_total_tokens=None,
    )
runtime_env = CONFIG["runtime_env"]
# The sglang decoder kernels' switches: the program replaces the decoder.
for key in (
    "SGLANG_TPU_K3_STATIC_FP8",
    "SGLANG_TPU_ENABLE_K3_HOST_SPLIT_GEMV",
    "SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE",
    "SGLANG_K3_MERGE_MOE_ALLREDUCE",
    "SGLANG_TPU_ENABLE_K3_ATTN_RES_KERNEL",
    "SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER",
    "SGLANG_TPU_FUSED_MOE_PERMUTE_DENSE_THRESHOLD",
):
    runtime_env.pop(key, None)
runtime_env.update(
    {
        # The program's custom ops run each as one device program.
        "SGLANG_TPU_EAGER_MODE": "DEFER_NEVER",
        "SGLANG_TPU_K3_PROGRAM_PRESHARDED": os.environ.get(
            "SGLANG_TPU_K3_PROGRAM_PRESHARDED", "/presharded"
        ),
        # InferenceX AgentX's synthetic acceptance for Kimi-K3 DSpark (mean
        # accepted length, bonus token included, per draft width).
        "SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS": "7:3.78,3:2.87",
        "SGLANG_RAIDEN_RESHARD": "0",
        # The MLA DSpark draft (both roles hold its KV).
        "SGLANG_TPU_K3_PROGRAM_DRAFT": "Inferact/Kimi-K3-DSpark",
        # At AgentX's pinned acceptance (3.78 at seven drafts, 2.87 at three),
        # measured with fp8 KV: seven drafts win only for a lone request; three
        # beat seven from two requests and beat plain steps through sixteen
        # (e.g. B8 1034 vs 851 tok/s, B16 1274 vs 995 at 131k context).
        "SGLANG_TPU_K3_PROGRAM_SPEC_SCHEDULE": "1:1:7,2:16:3",
        # Reading the pre-shard from GCS (gcsfuse): more concurrent ranges per file.
        "SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG": "read_streams=16",
    }
)
