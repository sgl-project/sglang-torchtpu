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
    K3_PROGRAM_DECODE_RUNNING       the decode's running requests (default
                                    twice the concurrency, up to 64)
    K3_PROGRAM_KERNEL_EXTRA         more kernel settings (a=b;c=d)
    K3_PROGRAM_ATTENTION_DP         1 (default) or 2: the decode's attention
                                    data parallelism (two DCP16 groups)
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
# A smaller decode batch keeps every step speculating: three drafts verify up
# to 16 requests (64 rows), and past that the decode steps plain. At AgentX
# concurrency 32 with 64 slots, ~18 requests decoded on average and 46% of
# steps ran 17-32 requests plain (ITL 17.4 ms); 16 requests at three drafts
# emit more tokens a second than 17-28 plain, and the others wait their turn
# with their KV already transferred.
if ROLE == "decode":
    RUNNING = int(os.environ.get("K3_PROGRAM_DECODE_RUNNING", RUNNING))
# Attention DP2 on the decode (k3_programs/sglang_dp.py): two DCP16 groups,
# three drafts up to 32 requests (16 a group).
ATTENTION_DP = (
    int(os.environ.get("K3_PROGRAM_ATTENTION_DP", "1")) if ROLE == "decode" else 1
)
# A prefill chunk is one page: every full chunk ends on a KDA checkpoint the
# prefix cache keeps (the programs' windows split it as they need).
PREFILL_CHUNK = PAGE_SIZE
# Prefill KDA checkpoints the prefix cache may hold (each three blocks of the
# 29 pools: about 28 MB a rank). The prefill program's weights take ~74 GB of
# a rank's 96 GB (experts 45, dense 6, its sequence-parallel copies of the
# latent and shared projections 22.5), so its pools share ~12 GB: 128
# checkpoints, 96 tails (38 MB each) and ~8M tokens of KV at 580 bytes.
# A prefix hit resumes only at a checkpoint, and running requests hold
# several slots each: at concurrency 32, 128 checkpoints let bursts evict
# cached prefixes (4.9M tokens in 9 minutes, effective hit ~81%) and the
# prefill queue grew to 32 requests. Eight a session, 128 to 384.
PREFILL_CHECKPOINTS = int(
    os.environ.get(
        "K3_PROGRAM_PREFILL_CHECKPOINTS", str(min(384, max(128, 8 * CONCURRENCY)))
    )
)

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
        # AgentX traces reach 990k tokens: a long prompt's max_tokens is cut to
        # what the context leaves, rather than the request (and the run) failing.
        "allow_auto_truncate": True,
        # usage.prompt_tokens_details.cached_tokens, for AgentX's cache metrics.
        "enable_cache_report": True,
    }
)
if ROLE == "decode":
    # The decode keeps finished turns' KV pages, and a session's next turn
    # transfers only the pages past that prefix (KV only: the transfer delivers
    # each request's KDA state; sglang_decode_cache). Under attention DP2 a hit
    # keeps the request in the group holding its pages.
    DECODE_PREFIX_CACHE = os.environ.get("K3_PROGRAM_DECODE_PREFIX_CACHE", "1") == "1"
    CONFIG["runtime_env"]["SGLANG_TPU_K3_PROGRAM_ENABLE_DECODE_PREFIX_CACHE"] = (
        "1" if DECODE_PREFIX_CACHE else "0"
    )
    server_args.update(
        disable_radix_cache=not DECODE_PREFIX_CACHE,
        # A KDA state slot and a full context of KV per running request, and
        # as many again for requests still arriving over the transfer: the
        # decode never retracts.
        max_mamba_cache_size=2 * RUNNING,
        max_total_tokens=2 * max(RUNNING, CONCURRENCY) * CONTEXT_LENGTH,
    )
    if DECODE_PREFIX_CACHE:
        server_args.update(
            disaggregation_decode_enable_radix_cache=True,
            # Validated before K3_PROGRAM's settings: the decode tracks nothing.
            mamba_radix_cache_strategy="extra_buffer",
            mamba_track_interval=PAGE_SIZE * (1 << 16),
            # Each request's slot and its extra buffer's two.
            max_mamba_cache_size=3 * 2 * RUNNING,
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
    # A host-DRAM tier for the prefix cache (k3_programs/sglang_hicache.py):
    # K3_PROGRAM_HICACHE_GB of host blocks a rank (8 ranks share a host), with
    # K3_PROGRAM_HICACHE_CHECKPOINTS KDA checkpoints among them. Off by default.
    HICACHE_GB = int(os.environ.get("K3_PROGRAM_HICACHE_GB", "0"))
    if HICACHE_GB > 0:
        server_args.update(
            enable_hierarchical_cache=True,
            hicache_io_backend="raiden",
            hicache_mem_layout="page_first_fused",
            hicache_write_policy=os.environ.get(
                "K3_PROGRAM_HICACHE_WRITE_POLICY", "write_through"
            ),
            hicache_size=HICACHE_GB,
        )
        CONFIG["runtime_env"]["SGLANG_TPU_K3_PROGRAM_HICACHE_CHECKPOINTS"] = (
            os.environ.get("K3_PROGRAM_HICACHE_CHECKPOINTS", "512")
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
        # Keep the compile cache (tmpfs) after warmup: an executable compiled
        # while serving is then loaded, not compiled again, after a restart.
        "SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS": "0",
        "SGLANG_TPU_K3_PROGRAM_PRESHARDED": os.environ.get(
            "SGLANG_TPU_K3_PROGRAM_PRESHARDED", "/presharded"
        ),
        # InferenceX AgentX's synthetic acceptance for Kimi-K3 DSpark (mean
        # accepted length, bonus token included, per draft width).
        "SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS": os.environ.get(
            "K3_PROGRAM_SIMULATE_ACC_LENGTHS", "7:3.78,3:2.87,1:1.84"
        ),
        "SGLANG_RAIDEN_RESHARD": "0",
        # The MLA DSpark draft (both roles hold its KV).
        "SGLANG_TPU_K3_PROGRAM_DRAFT": "Inferact/Kimi-K3-DSpark",
        # At AgentX's pinned acceptance (3.78 at seven drafts, 2.87 at three),
        # measured with fp8 KV: seven drafts win only for a lone request; three
        # beat seven from two requests and beat plain steps through sixteen
        # (e.g. B8 1034 vs 851 tok/s, B16 1274 vs 995 at 131k context). Past
        # sixteen, plain steps (vLLM's schedule carries the last range's count
        # into the tail); K3_PROGRAM_SPEC_SCHEDULE overrides it (three drafts
        # verify up to 24 requests).
        "SGLANG_TPU_K3_PROGRAM_SPEC_SCHEDULE": os.environ.get(
            "K3_PROGRAM_SPEC_SCHEDULE",
            "1:32:3,33:128:0" if ATTENTION_DP == 2 else "1:1:7,2:16:3,17:64:0",
        ),
        # Overlap scheduling on the decode (the prefill stays synchronous).
        "SGLANG_TPU_K3_PROGRAM_OPT_OVERLAP": os.environ.get("K3_PROGRAM_OVERLAP", "1"),
        # Reading the pre-shard from GCS (gcsfuse): more concurrent ranges per
        # file; K3_PROGRAM_KERNEL_EXTRA appends kernel settings (a=b;c=d).
        "SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG": ";".join(
            filter(
                None,
                (
                    "read_streams=16",
                    "attention_dp=2;dcp_group=16" if ATTENTION_DP == 2 else None,
                    os.environ.get("K3_PROGRAM_KERNEL_EXTRA"),
                ),
            )
        ),
        "SGLANG_TPU_K3_PROGRAM_ATTENTION_DP": str(ATTENTION_DP),
    }
)
