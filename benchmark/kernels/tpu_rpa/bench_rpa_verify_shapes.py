"""Microbenchmark for the Pallas RPA v3 kernel at GPT-OSS-120B verify shapes.

Drives ``ragged_paged_attention`` directly -- the same kernel the serving path
calls -- against a fused fp8 KV cache laid out exactly like
``TpuFusedMHATokenToKVPool``. No model, no server, no re-implementation: only
the kernel and its metadata.

Motivation. In the ``baseline-v0`` xprof (v7x, c64 EAGLE3 decode), one target
verify forward costs 43.6 ms of device time, and attention is 17.4 ms of it:

    RPAm full attention   18 layers x 723 us = 12.9 ms
    RPAm sliding window   18 layers x 251 us =  4.5 ms

The SWA layers read a 128-token window -- ~37x less KV than the full-attention
layers -- yet still cost 35% as much, which says most of a call is fixed cost
rather than streaming. Sweeping ``--kv-lens`` separates the two: the intercept
of time vs kv_len is the per-call floor, the slope is the achieved KV
bandwidth.

Shapes default to what the profiled run actually used, read off the HLO operand
types of ``RPAm-p_64-bq_128_128-bkv_2048_512``:

    q      bf16[256, 8, 128]              256 = bs 64 x 4 draft tokens, 8 q heads
    kv     f8e4m3fn[256, 2, 128]          2 kv heads
    cache  f8e4m3fn[4641, 64, 1, 4, 128]  page 64, kv_heads_x2 4, head_dim 128

head_dim is 64 in the checkpoint and padded to 128 by the pool, so half of
every KV byte the kernel streams is zero padding -- hence the separate
useful-bandwidth column.

Run on a TPU host with a free chip::

    python -m benchmark.kernels.tpu_rpa.bench_rpa_verify_shapes
    python -m benchmark.kernels.tpu_rpa.bench_rpa_verify_shapes --sliding-window 128
"""

import argparse
import time

import jax
import jax.numpy as jnp

from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)

# TpuFusedMHATokenToKVPool.HEAD_DIM_ALIGNMENT
KV_HEAD_DIM_ALIGNMENT = 128


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _build_cache(args, kv_packing: int, kv_dtype):
    num_kv_heads_x2 = _cdiv(args.num_kv_heads * 2, kv_packing) * kv_packing
    padded_head_dim = (
        _cdiv(args.head_dim, KV_HEAD_DIM_ALIGNMENT) * KV_HEAD_DIM_ALIGNMENT
    )
    num_blocks = args.batch_size * args.pages_per_seq + 1
    cache = jnp.zeros(
        (
            num_blocks,
            args.page_size,
            num_kv_heads_x2 // kv_packing,
            kv_packing,
            padded_head_dim,
        ),
        dtype=kv_dtype,
    )
    return cache, num_kv_heads_x2, padded_head_dim


def _build_metadata(args, kv_len: int, q_len_per_seq: int, is_decode: bool):
    bs = args.batch_size
    # Rectangular page table, as _build_metadata_host emits it.
    page_indices = jnp.arange(bs * args.pages_per_seq, dtype=jnp.int32) + 1
    kv_lens = jnp.full((bs,), kv_len, dtype=jnp.int32)
    cu_q_lens = jnp.arange(bs + 1, dtype=jnp.int32) * q_len_per_seq
    cu_kv_lens = jnp.arange(bs + 1, dtype=jnp.int32) * (
        args.pages_per_seq * args.page_size
    )
    # Decode -> [n, n, n]; TARGET_VERIFY takes the extend path, whose RPA v3
    # mask is (0, 1, 1) -> [0, n, n].
    slots = [bs, bs, bs] if is_decode else [0, bs, bs]
    return kv_lens, page_indices, cu_q_lens, cu_kv_lens, jnp.array(slots, jnp.int32)


def _make_kernel(args):
    sm_scale = args.head_dim**-0.5
    sliding_window = args.sliding_window
    m_block_sizes = tuple(args.m_block_sizes) if args.m_block_sizes else None

    chunk_prefill_size = args.chunk_prefill_size

    def run(q, k, v, cache, meta):
        kv_lens, page_indices, cu_q_lens, cu_kv_lens, distribution = meta
        return ragged_paged_attention(
            q,
            k,
            v,
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            None,  # custom_mask
            None,  # attention_sink
            causal=1,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            k_scale=1.0,
            v_scale=1.0,
            chunk_prefill_size=chunk_prefill_size,
            m_block_sizes=m_block_sizes,
        )

    # Donate the cache so the kernel updates it in place, matching the serving
    # path's input_output_aliases; without it XLA copies the whole pool per call.
    return jax.jit(run, donate_argnums=(3,))


def _time(fn, q, k, v, cache, meta, warmup: int, iters: int):
    """Mean device ms per call.

    Enqueues every call before blocking once at the end. Blocking per call
    would measure the host round-trip instead: at these shapes dispatch is
    ~0.25 ms, which is the same order as the kernel itself.
    """
    for _ in range(warmup):
        _, cache = jax.block_until_ready(fn(q, k, v, cache, meta))
    t0 = time.perf_counter()
    for _ in range(iters):
        _, cache = fn(q, k, v, cache, meta)
    jax.block_until_ready(cache)
    return (time.perf_counter() - t0) * 1e3 / iters, cache


def _compare_spec(args, cache, qkv, fn):
    """One verify call vs --num-steps plain decode calls, per kv_len.

    Both read the same KV per sequence, so if attention were bandwidth-bound
    the verify call would cost about what one decode costs and the ratio would
    approach --num-steps.
    """
    steps = args.num_steps
    q_v, k_v, v_v = qkv(args.batch_size * args.q_len_per_seq)
    q_d, k_d, v_d = qkv(args.batch_size)
    header = (
        f"{'kv_len':>8}{'verify':>11}{'1x decode':>12}"
        f"{str(steps) + 'x decode':>12}{'ratio':>8}"
    )
    print(header)
    print("-" * len(header))
    for kv_len in args.kv_lens:
        if kv_len > args.pages_per_seq * args.page_size:
            continue
        m_v = _build_metadata(args, kv_len, args.q_len_per_seq, is_decode=False)
        m_d = _build_metadata(args, kv_len, 1, is_decode=True)
        t_v, cache = _time(fn, q_v, k_v, v_v, cache, m_v, args.warmup, args.iters)
        t_d, cache = _time(fn, q_d, k_d, v_d, cache, m_d, args.warmup, args.iters)
        print(
            f"{kv_len:>8}{t_v:>10.3f}m{t_d:>11.3f}m"
            f"{steps * t_d:>11.3f}m{steps * t_d / t_v:>7.2f}x"
        )
    print("-" * len(header))
    print(
        f"ratio > 1 means one verify beats {steps} decodes on attention; "
        f"the end-to-end win also needs acceptance length > 1."
    )
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--q-len-per-seq", type=int, default=4, help="draft tokens per req")
    # Model totals (gpt-oss-120b/config.json), sharded by --tp the way
    # ModelConfig.get_num_kv_heads does.
    p.add_argument("--total-q-heads", type=int, default=64)
    p.add_argument("--total-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument(
        "--tp",
        type=int,
        default=8,
        help="baseline-v0 ran tp=8/ep=8: MoE gmm is g_16 (128/16 experts per "
        "rank) and attention is 8 q heads per rank.",
    )
    p.add_argument("--page-size", type=int, default=64)
    p.add_argument("--pages-per-seq", type=int, default=147)
    p.add_argument(
        "--kv-lens",
        type=int,
        nargs="+",
        default=[128, 512, 1024, 2048, 4096, 4736, 8192],
    )
    p.add_argument("--sliding-window", type=int, default=None)
    p.add_argument("--kv-dtype", choices=("fp8_e4m3", "bf16"), default="fp8_e4m3")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument(
        "--compare-spec",
        action="store_true",
        help="Contrast one verify call against --num-steps plain decode calls, "
        "the attention half of 'does speculative decoding pay off'.",
    )
    p.add_argument("--num-steps", type=int, default=3)
    p.add_argument(
        "--chunk-prefill-size",
        type=int,
        default=None,
        help="Route the verify rows to the PREFILL case with this static_q_len "
        "(what PR 711 does) instead of leaving them on MIXED.",
    )
    p.add_argument(
        "--m-block-sizes",
        type=int,
        nargs=4,
        metavar=("BQ", "BKV", "BQ_C", "BKV_C"),
        default=None,
        help="Override the MIXED-case blocks the heuristic picks (v7 gives "
        "bq=128 bkv=2048 bkv_c=512). DECODE sizes its bkv block to span the "
        "whole KV, which is where its bandwidth advantage comes from.",
    )
    args = p.parse_args()

    if args.total_q_heads % args.tp:
        raise SystemExit(f"tp={args.tp} does not divide {args.total_q_heads} q heads")
    args.num_q_heads = args.total_q_heads // args.tp
    # Fewer kv heads than ranks means they replicate, never shard below 1.
    args.num_kv_heads = max(1, args.total_kv_heads // args.tp)

    kv_packing = 4 if args.kv_dtype == "fp8_e4m3" else 2
    kv_dtype = jnp.float8_e4m3fn if args.kv_dtype == "fp8_e4m3" else jnp.bfloat16
    kv_elem = 1 if args.kv_dtype == "fp8_e4m3" else 2
    num_tokens = args.batch_size * args.q_len_per_seq

    cache, num_kv_heads_x2, padded_head_dim = _build_cache(args, kv_packing, kv_dtype)
    key = jax.random.PRNGKey(0)

    def qkv(n):
        return (
            jax.random.normal(key, (n, args.num_q_heads, args.head_dim), jnp.bfloat16),
            jnp.zeros((n, args.num_kv_heads, args.head_dim), kv_dtype),
            jnp.zeros((n, args.num_kv_heads, args.head_dim), kv_dtype),
        )

    q, k, v = qkv(num_tokens)
    fn = _make_kernel(args)

    print(
        f"device={jax.devices()[0].device_kind}  bs={args.batch_size} "
        f"q_tokens={num_tokens}  tp={args.tp}: "
        f"q_heads {args.total_q_heads}->{args.num_q_heads} "
        f"kv_heads {args.total_kv_heads}->{args.num_kv_heads} "
        f"head_dim={args.head_dim}->{padded_head_dim}\n"
        f"page={args.page_size} pages_per_seq={args.pages_per_seq} "
        f"kv={args.kv_dtype} sliding_window={args.sliding_window}\n"
        f"cache={cache.shape} ({cache.size * kv_elem / 1e9:.2f} GB)\n"
    )

    if args.compare_spec:
        _compare_spec(args, cache, qkv, fn)
        return

    header = f"{'kv_len':>8}{'time':>10}{'KV read':>11}{'GB/s':>9}{'useful GB/s':>13}"
    print(header)
    print("-" * len(header))

    for kv_len in args.kv_lens:
        if kv_len > args.pages_per_seq * args.page_size:
            print(f"{kv_len:>8}  skipped (exceeds page table capacity)")
            continue
        meta = _build_metadata(args, kv_len, args.q_len_per_seq, is_decode=False)
        span = (
            kv_len if args.sliding_window is None else min(kv_len, args.sliding_window)
        )
        t, cache = _time(fn, q, k, v, cache, meta, args.warmup, args.iters)

        tokens = args.batch_size * _cdiv(span, args.page_size) * args.page_size
        kv_bytes = tokens * num_kv_heads_x2 * padded_head_dim * kv_elem
        useful = tokens * (args.num_kv_heads * 2) * args.head_dim * kv_elem
        print(
            f"{kv_len:>8}{t:>9.3f}m{kv_bytes / 1e6:>9.1f}MB"
            f"{kv_bytes / (t * 1e-3) / 1e9:>9.0f}{useful / (t * 1e-3) / 1e9:>13.0f}"
        )

    print("-" * len(header))
    print("Intercept of time vs kv_len is the per-call fixed cost; slope is KV BW.")


if __name__ == "__main__":
    main()
