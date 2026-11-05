"""Kimi-K3.1 (Kimi-K3 + a DSA indexer on its 24 MLA layers) on four tpu7x hosts, tp32/ep8.

Native sglang path: no speculative decoding, no megakernel. Every host runs

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/server.py
    # node_rank / dist_init_addr come from TPU_WORKER_ID / TPU_WORKER_HOSTNAMES.
    # Add --dry-run to print the command.

The cookbook page renders this recipe with the long_128k_c64 profile, which
restates the sizing below.

kimi_k3/server.py plus the K3.1 sparse-MLA/DSA backend, fp8 MLA KV, radix cache,
the per-op decode flags, the host-side decode launch switches, the SparseCore gather offload and the DSA indexer token
split. Sized as measured on
16 chips (2026-11): 128K context (131072 in + 8192 out), up to 64 running
requests, one 4096-token prefill bucket.

Text only: SGLANG_TPU_VL_TEXT_ONLY_EXTEND compiles only the token-id EXTEND
graphs, and the server answers a request with image input with a 400.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3.server import CONFIG as K3_SERVER_CONFIG

CONCURRENCY = 64
# One prefill token bucket, as in kimi_k3/server.py; 4096 is the chunk the
# 16-chip runs measured (eight EXTEND buckets up to 8192 OOM-killed the hosts).
PREFILL_TOKENS = 4096
CONTEXT_LENGTH = 131072 + 8192
# Decode buckets: powers of two up to CONCURRENCY, so a small batch is not
# padded to 64.
DECODE_BS = [1, 2, 4, 8, 16, 32, CONCURRENCY]
# An upper bound only: the pool is sized by mem_fraction_static, about 532K
# tokens on 16 chips (MLA KV and DSA index keys are replicated on every rank).
MAX_TOTAL_TOKENS = 4000000


# Host-side decode launch switches and the right-sized Pallas VMEM limits. Set
# only on this recipe: they were validated on the Kimi-K3.1 per-op native decode,
# and the VMEM right-size changes Kimi-K3's greedy output (Mosaic tiles the
# single-token MoE kernel differently), so the Kimi-K3 recipes leave them off.
NATIVE_DECODE_HOST_ENV = {
    "SGLANG_TPU_DECODE_SKIP_GUARD_EVAL": "1",
    "SGLANG_TPU_AOT_LEAN_RUNTIME_WRAPPER": "1",
    "SGLANG_TPU_SPLIT_EXEC_DIRECT_CALL": "1",
    # XLA keeps weight prefetches in flight across the decode Pallas kernels.
    "SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE": "1",
}


def with_libtpu_flags(args: str, **flags: str) -> str:
    """Set each --name=value in a LIBTPU_INIT_ARGS string, replacing an existing setting.

    A replaced flag moves to the end; the 16-chip runs used this order.
    """
    kept = [f for f in args.split() if f.split("=", 1)[0].lstrip("-") not in flags]
    return " ".join(kept + [f"--{name}={value}" for name, value in flags.items()])


CONFIG = deepcopy(K3_SERVER_CONFIG)
CONFIG["server_args"].update(
    {
        "model_path": "moonshotai/Kimi-K3.1-Pro",
        # 24 MLA layers carry a DSA indexer (top-2048); fp8 split MLA KV.
        "attention_backend": "pallas_tpu_mla_sparse",
        "kv_cache_dtype": "fp8_e4m3",
        "context_length": CONTEXT_LENGTH,
        "max_running_requests": CONCURRENCY,
        "torch_compile_max_bs": CONCURRENCY,
        "tpu_compile_bs": DECODE_BS,
        "chunked_prefill_size": PREFILL_TOKENS,
        "max_prefill_tokens": PREFILL_TOKENS,
        "tpu_compile_tokens": [PREFILL_TOKENS],
        # Up to 8 requests share one prefill chunk.
        "prefill_max_requests": 8,
        "mem_fraction_static": 0.85,
        "max_total_tokens": MAX_TOTAL_TOKENS,
        "disable_radix_cache": False,
        "enable_cache_report": True,
        # The TPU backend compiles and warms every bucket at startup. The HTTP warmup
        # would send an image request to this VL model, which the text-only server
        # answers with a 400 (and startup then aborts).
        "skip_server_warmup": True,
    }
)
CONFIG["runtime_env"].update(
    {
        "LIBTPU_INIT_ARGS": with_libtpu_flags(
            CONFIG["runtime_env"].get("LIBTPU_INIT_ARGS", ""),
            # libtpu < 0.0.51 (#1137) corrupts SparseCore buffers with large KV pools.
            xla_tpu_buffer_assignment_enable_fast_merge_fallback="false",
            # The DSA top-k remap gathers run on the SparseCore (on the TensorCore a
            # 1024-token prefill chunk at 160K context spends ~100 ms per DSA layer in them).
            xla_tpu_offload_gather_to_sparsecore="true",
        ),
        "SGLANG_TPU_ENABLE_QBLOCK_SPARSE_MLA_PREFILL": "1",
        # Prefill: each attention-TP rank runs the DSA indexer top-k for its slice of
        # the chunk's rows and an all-reduce reassembles them. The rows equal the
        # unsplit op's on 93 layers / 16 chips (every layer and chunk checked); prompt
        # logprobs still move at the bf16 level because the prefill graph changes.
        # Cold TTFT 64K 17.6 -> 13.5 s, 128K 37.9 -> 27.0 s; GSM8K-200 0.995.
        "SGLANG_TPU_DSA_INDEXER_TOKEN_SPLIT": "1",
        # Text-only serving compiles only the token-id EXTEND graphs (host RAM at
        # startup); scheduler intake rejects image requests with a 400.
        "SGLANG_TPU_VL_TEXT_ONLY_EXTEND": "1",
        "SGLANG_SET_CPU_AFFINITY": "1",
        "SGLANG_TPU_ASYNC_COPY_WORKER": "0",
        # Per-op decode flags of the Kimi-K3 decode optimization PRs (each default off).
        "SGLANG_TPU_TWOLEVEL_ALLREDUCE_SPLIT_BARRIER": "1",
        "SGLANG_TPU_K3_QKVG_PREFETCH": "1",
        "SGLANG_TPU_K3_LATENT_UP_ATTN_RES": "1",
        "SGLANG_TPU_K3_OPROJ_ROW": "1",
        "SGLANG_TPU_DYNAMO_PARAMS_AS_GRAPH_ATTRS": "2",
        "SGLANG_TPU_ALLOC_DECODE_FAST_PATH": "1",
        # bs1 decode logits through the one-shot Pallas gather; every other logits
        # gather is an all-reduce (SGLANG_TPU_LOGITS_GATHER_ALLREDUCE, default on).
        "SGLANG_TPU_LOGITS_ONESHOT_GATHER": "1",
        **NATIVE_DECODE_HOST_ENV,
    }
)
