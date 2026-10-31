"""Kimi-K3 on the megakernel programs, both programs in one engine per 2x2x4
tpu7x slice (tp32), for InferenceX AgentX.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3/colocated_program.py

Each engine prefills and decodes (sglang_tpu/k3_programs/sglang_colocated.py):
the prefill program shares the decode program's routed experts (both at expert
parallelism 16), and a request goes from prefill to decode in place. Two
engines, one per slice, sit behind a cache-aware router, so a session's turns
reach the engine holding its prefix. Weights come pre-sharded per host (see
pd_program.py).

Environment:
    K3_PROGRAM_CONCURRENCY          this engine's share of the client concurrency
                                    (default 4)
    K3_PROGRAM_CONTEXT              context length (default 1048576)
    K3_PROGRAM_KV_DTYPE             fp8_e4m3 (default) or bf16
    K3_PROGRAM_EXPERT_PARALLEL      16 (default) or 8: both programs' expert
                                    parallelism (the shared experts' layout)
    K3_PROGRAM_PREFILL_DECODE_INTERVAL  decode steps after a prefill batch (16)
    K3_PROGRAM_SCHEDULE_POLICY      fcfs (default) or lpm
    K3_PROGRAM_PREFILL_SHARE        prefill's share of the engine while
                                    requests decode (0: the fixed interval)
    SGLANG_TPU_K3_PROGRAM_PRESHARDED pre-sharded weights directory
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3.server import CONFIG as SERVER_CONFIG

CONCURRENCY = int(os.environ.get("K3_PROGRAM_CONCURRENCY", "4"))
CONTEXT_LENGTH = int(os.environ.get("K3_PROGRAM_CONTEXT", str(1 << 20)))
KV_DTYPE = os.environ.get("K3_PROGRAM_KV_DTYPE") or "fp8_e4m3"
# One logical block: 256 tokens on each of the 32 ranks in bf16, 512 in fp8.
PAGE_SIZE = 32 * (512 if KV_DTYPE.startswith("fp8") else 256)
# Decoding requests, up to the 64 rows the programs' plain decode is checked
# to; the prefill takes its share of the same slots.
RUNNING = min(64, max(4, 2 * CONCURRENCY))
# Both programs' weights take ~80 GB of a rank's 96 GB (the shared experts 45,
# two sets of dense weights 12, the prefill's sequence-parallel projections
# 22.5): the pools are small, and every KDA state (a checkpoint, a running
# request's slot) takes three of their blocks, a tail four -- each block a
# page of KV. A prefix hit resumes only at a checkpoint, and running requests
# hold several slots each (sglang plans five), so a burst evicts checkpoints:
# at concurrency 8, 24 checkpoints left long sessions matching a few pages of
# their 100k-500k contexts (minute-long TTFTs) while 94% of 9.8M KV tokens sat
# idle; 64 checkpoints and 48 tails left only 1.2M tokens of KV, and sessions
# evicted each other's pages. Eight checkpoints and two tails a session, up to
# 64 checkpoints.
CHECKPOINTS = int(
    os.environ.get(
        "K3_PROGRAM_PREFILL_CHECKPOINTS", str(min(64, max(16, 8 * CONCURRENCY)))
    )
)
TAILS = os.environ.get("K3_PROGRAM_PREFIX_TAILS", str(max(8, 2 * CONCURRENCY)))

CONFIG = deepcopy(SERVER_CONFIG)
server_args = CONFIG["server_args"]
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
        "kv_cache_dtype": KV_DTYPE,
        "page_size": PAGE_SIZE,
        "context_length": CONTEXT_LENGTH,
        "chunked_prefill_size": PAGE_SIZE,
        "max_prefill_tokens": PAGE_SIZE,
        "max_running_requests": RUNNING,
        "mem_fraction_static": 0.92,
        "skip_server_warmup": True,
        # AgentX traces reach 990k tokens: a long prompt's max_tokens is cut to
        # what the context leaves, rather than the request (and the run) failing.
        "allow_auto_truncate": True,
        # usage.prompt_tokens_details.cached_tokens, for AgentX's cache metrics.
        "enable_cache_report": True,
        # The prefix cache: KV pages, a KDA checkpoint per page-aligned prefill
        # chunk, and tails past the last full page (sglang_tails). The running
        # requests' slots come out of the same state slots.
        "disable_radix_cache": False,
        "mamba_radix_cache_strategy": "extra_buffer",
        # Checkpoints at the prefill's page-sized chunks only (the decode
        # program keeps its requests' KDA state; K3ProgramAlgo sets the same).
        "mamba_track_interval": PAGE_SIZE * (1 << 16),
        "max_mamba_cache_size": CHECKPOINTS + 2 * RUNNING,
        "max_total_tokens": None,
        # Decode steps between prefill batches: a long prompt's chunks would
        # otherwise stop every decoding request on the engine until it ends.
        "prefill_decode_interval": int(
            os.environ.get("K3_PROGRAM_PREFILL_DECODE_INTERVAL", "16")
        ),
        # Waiting prefills by arrival (fcfs) or longest cached prefix first
        # (lpm: a session's next turn, a few pages of new work, before a cold
        # prompt that starts from scratch).
        "schedule_policy": os.environ.get("K3_PROGRAM_SCHEDULE_POLICY", "fcfs"),
    }
)
runtime_env = CONFIG["runtime_env"]
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
        "SGLANG_TPU_EAGER_MODE": "DEFER_NEVER",
        # Keep the compile cache (tmpfs) after warmup: an executable compiled
        # while serving is then loaded, not compiled again, after a restart.
        "SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS": "0",
        "SGLANG_TPU_K3_PROGRAM_PRESHARDED": os.environ.get(
            "SGLANG_TPU_K3_PROGRAM_PRESHARDED", "/presharded"
        ),
        "SGLANG_TPU_K3_PROGRAM_DRAFT": "Inferact/Kimi-K3-DSpark",
        # Both programs at one expert parallelism, so the prefill shares the
        # decode program's experts: 16 by default, the fused prefill MoE's
        # layout (the decode runs it as half experts); 8, the pre-shard's,
        # leaves the prefill on the unfused MoE (about 5x slower). More
        # concurrent ranges per file, reading the pre-shard from GCS.
        "SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG": (
            "prefill_expert_parallel={ep};decode_expert_parallel={ep};"
            "read_streams=16"
        ).format(ep=os.environ.get("K3_PROGRAM_EXPERT_PARALLEL", "16")),
        "SGLANG_TPU_K3_PROGRAM_PREFIX_TAILS": TAILS,
        "SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS": "7:3.78,3:2.87",
        "SGLANG_TPU_K3_PROGRAM_SPEC_SCHEDULE": "1:1:7,2:16:3,17:64:0",
        # Overlap scheduling: the decode program's next step launches while
        # the scheduler processes the last one's tokens.
        "SGLANG_TPU_K3_PROGRAM_OPT_OVERLAP": os.environ.get("K3_PROGRAM_OVERLAP", "1"),
        # Prefill's share of the engine while requests decode, each prefill
        # batch deferring the next by rounds sized from its cost (0: the fixed
        # prefill_decode_interval).
        "SGLANG_TPU_K3_PROGRAM_PREFILL_SHARE": os.environ.get(
            "K3_PROGRAM_PREFILL_SHARE", "0"
        ),
    }
)
