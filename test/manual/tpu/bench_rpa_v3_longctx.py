"""Eager microbenchmark of the RPA v3 decode / verify kernel at long context.

Runs ``ragged_paged_attention`` directly in JAX (no torch.compile, so no stale
compile-cache kernels) for a grid of kv_len x batch x query rows per sequence
at one geometry: 1 KV head, ``--q-heads`` q heads, head_dim 128, page 128,
fp8 e4m3 KV, learned rel bias + token sink as the Inkling2 target model uses.

TPU has no L2; the kernel streams the KV cache from HBM through VMEM every
call, so timing a loop over the same inputs is the serving pattern.

Usage on a TPU host (one chip is enough):

  PYTHONPATH=<repo>:<repo>/python python test/manual/tpu/bench_rpa_v3_longctx.py \
      --kv-lens 8192,32768,122880 --bs 1,8,16,32 --rows 1,8 \
      --blocks default,8:8192:8:512,8:16384:8:512
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import time

import jax
import jax.numpy as jnp
import numpy as np

import sglang_tpu.attention.pallas.ragged_paged_attention_v3 as rpa_mod
from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
    ragged_paged_attention,
)

PAGE_SIZE = 128
HEAD_DIM = 128
REL_DIM = 16
# Bytes per cached token per layer in the fused fp8 layout [p, 128, 1, 4, 128].
TOKEN_BYTES_FP8 = 512


def _parse_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _parse_blocks(s: str) -> list[tuple[int, ...] | None]:
    out: list[tuple[int, ...] | None] = []
    for item in s.split(","):
        if not item:
            continue
        if item == "default":
            out.append(None)
        else:
            out.append(tuple(int(x) for x in item.split(":")))
    return out


def build_inputs(
    *,
    bs: int,
    kv_len: int,
    rows: int,
    q_heads: int,
    kv_dtype,
    rel: bool,
    extent: int,
    layers: int = 1,
    seed: int = 0,
):
    """One batch of ``bs`` decode sequences, each at ``kv_len`` tokens of which
    the last ``rows`` are this step's new tokens (kernel writes them)."""
    pages_per_seq = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = bs * pages_per_seq
    max_num_tokens = max(8, ((bs * rows + 7) // 8) * 8)
    kv_packing = rpa_mod.get_dtype_packing(kv_dtype)
    key = jax.random.PRNGKey(seed)
    k_q, k_k, k_v, k_c, k_r, k_p, k_s = jax.random.split(key, 7)

    q = jax.random.normal(k_q, (max_num_tokens, q_heads, HEAD_DIM), jnp.bfloat16)
    new_k = jax.random.normal(k_k, (max_num_tokens, 1, HEAD_DIM), jnp.bfloat16)
    new_v = jax.random.normal(k_v, (max_num_tokens, 1, HEAD_DIM), jnp.bfloat16)
    new_k = new_k.astype(kv_dtype)
    new_v = new_v.astype(kv_dtype)
    slots = (2 + kv_packing - 1) // kv_packing * kv_packing
    kv_shape = (total_pages, PAGE_SIZE, slots // kv_packing, kv_packing, HEAD_DIM)
    # Random seed pages tiled to the full pool: 1M-token caches do not fit a bf16 temp.
    seed_pages = min(total_pages, 64)
    reps = (total_pages + seed_pages - 1) // seed_pages
    kv_fused = [
        jnp.tile(
            jax.random.normal(kk, (seed_pages,) + kv_shape[1:], jnp.bfloat16).astype(
                kv_dtype
            ),
            (reps, 1, 1, 1, 1),
        )[:total_pages]
        for kk in jax.random.split(k_c, layers)
    ]

    # Scatter the page table so neighbouring logical pages are not contiguous.
    page_indices = (np.arange(total_pages, dtype=np.int32) * 7 + 3) % total_pages
    kv_lens = jnp.full((bs,), kv_len, jnp.int32)
    cu_q_lens = jnp.asarray(np.arange(bs + 1, dtype=np.int32) * rows)
    cu_kv_lens = jnp.asarray(
        np.arange(bs + 1, dtype=np.int32) * pages_per_seq * PAGE_SIZE
    )
    inputs = dict(
        q=q,
        new_k=new_k,
        new_v=new_v,
        kv_fused=kv_fused,
        kv_lens=kv_lens,
        page_indices=jnp.asarray(page_indices),
        cu_q_lens=cu_q_lens,
        cu_kv_lens=cu_kv_lens,
        distribution=jnp.asarray([bs, bs, bs], jnp.int32),
        rel_r=None,
        rel_proj=None,
        token_sink=None,
    )
    if rel:
        inputs["rel_r"] = (
            jax.random.normal(k_r, (max_num_tokens, q_heads, REL_DIM), jnp.float32) * 0.5
        ).astype(jnp.bfloat16)
        inputs["rel_proj"] = jax.random.normal(
            k_p, (REL_DIM, extent), jnp.float32
        ).astype(jnp.bfloat16)
        inputs["token_sink"] = (
            jax.random.normal(k_s, (max_num_tokens, q_heads), jnp.float32) * 2.0
        ).astype(jnp.bfloat16)
    return inputs


# The caches are donated: without donation XLA copies each cache (1 GB at
# bs16 x 120k) to satisfy the kernel's input/output alias, 1 ms per layer.
@functools.partial(
    jax.jit,
    static_argnames=("rows", "blocks", "sliding_window", "k_scale", "v_scale"),
    donate_argnames=("q", "new_k", "new_v", "caches"),
)
def _run_layers(
    q, new_k, new_v, caches, kv_lens, page_indices, cu_q_lens, cu_kv_lens,
    distribution, rel_r, rel_proj, token_sink, *, rows, blocks, sliding_window,
    k_scale, v_scale,
):
    """One attention call per cache, as a model's layers run them; the outer
    jit pays host dispatch once for all of them."""
    out = None
    new_caches = []
    for cache in caches:
        out, cache = ragged_paged_attention(
            q,
            new_k,
            new_v,
            cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            None,  # custom_mask
            None,  # attention_sink
            rel_r,
            rel_proj,
            token_sink,
            sm_scale=1.0 / float(np.sqrt(HEAD_DIM)),
            sliding_window=sliding_window,
            k_scale=k_scale,
            v_scale=v_scale,
            decode_query_size=rows,
            d_block_sizes=blocks,
        )
        new_caches.append(cache)
    return out, new_caches


_last_default = [None]
_orig_default_block_sizes = rpa_mod.get_default_block_sizes


def _recording_default_block_sizes(*args, **kw):
    out = _orig_default_block_sizes(*args, **kw)
    if kw.get("case") == rpa_mod.RpaCase.DECODE:
        _last_default[0] = tuple(
            out[k] for k in ("bq_sz", "bkv_sz", "bq_csz", "bkv_csz")
        )
    return out


rpa_mod.get_default_block_sizes = _recording_default_block_sizes


def run_once(inputs, *, rows, blocks, sliding_window, k_scale, v_scale):
    out, caches = _run_layers(
        jnp.copy(inputs["q"]),
        jnp.copy(inputs["new_k"]),
        jnp.copy(inputs["new_v"]),
        inputs["kv_fused"],
        inputs["kv_lens"],
        inputs["page_indices"],
        inputs["cu_q_lens"],
        inputs["cu_kv_lens"],
        inputs["distribution"],
        inputs["rel_r"],
        inputs["rel_proj"],
        inputs["token_sink"],
        rows=rows,
        blocks=blocks,
        sliding_window=sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    inputs["kv_fused"] = caches
    return out


def bench(inputs, *, iters, warmup, **kw):
    for _ in range(warmup):
        out = run_once(inputs, **kw)
    out.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = run_once(inputs, **kw)
    out.block_until_ready()
    return (time.perf_counter() - t0) / iters, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-lens", default="8192,32768,122880")
    ap.add_argument("--bs", default="1,8,16,32")
    ap.add_argument("--rows", default="1,8")
    ap.add_argument("--blocks", default="default", help="comma list of default|bq:bkv:bqc:bkvc")
    ap.add_argument("--q-heads", type=int, default=6)
    ap.add_argument("--kv-dtype", default="fp8", choices=["fp8", "bf16"])
    ap.add_argument("--no-rel", action="store_true", help="no rel bias / token sink")
    ap.add_argument("--extent", type=int, default=1024)
    ap.add_argument("--sliding-window", type=int, default=None)
    ap.add_argument("--layers", type=int, default=1, help="caches per jitted call")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="compare outputs across block configs")
    a = ap.parse_args()

    src = open(rpa_mod.__file__, "rb").read()
    print(f"kernel {rpa_mod.__file__} sha1 {hashlib.sha1(src).hexdigest()[:12]}")
    print(f"device {jax.devices()[0]}")
    kv_dtype = jnp.float8_e4m3fn if a.kv_dtype == "fp8" else jnp.bfloat16
    scale = 1.0 if a.kv_dtype == "fp8" else None
    token_bytes = TOKEN_BYTES_FP8  # bf16 with 1 KV head is also 512 B/token
    blocks_list = _parse_blocks(a.blocks)

    print(
        f"{'kv_len':>7} {'bs':>3} {'rows':>4} {'blocks':>22} {'us/call':>9} "
        f"{'GB/s':>7} {'ns/tok':>7}  (per layer, {a.layers} layers per call)"
    )
    for kv_len in _parse_ints(a.kv_lens):
        for bs in _parse_ints(a.bs):
            for rows in _parse_ints(a.rows):
                inputs = build_inputs(
                    bs=bs,
                    kv_len=kv_len,
                    rows=rows,
                    q_heads=a.q_heads,
                    kv_dtype=kv_dtype,
                    rel=not a.no_rel,
                    extent=a.extent,
                    layers=a.layers,
                )
                ref = None
                for blocks in blocks_list:
                    try:
                        dt, out = bench(
                            inputs,
                            iters=a.iters,
                            warmup=a.warmup,
                            rows=rows,
                            blocks=blocks,
                            sliding_window=a.sliding_window,
                            k_scale=scale,
                            v_scale=scale,
                        )
                    except Exception as e:  # noqa: BLE001
                        msg = str(e).splitlines()[0][:70]
                        print(f"{kv_len:>7} {bs:>3} {rows:>4} {str(blocks):>22} FAIL {msg}")
                        continue
                    dt /= a.layers
                    tokens = bs * kv_len
                    gbs = tokens * token_bytes / dt / 1e9
                    note = f" ={_last_default[0]}" if blocks is None else ""
                    if a.check:
                        o = np.asarray(out[: bs * rows].astype(jnp.float32))
                        if ref is None:
                            ref = o
                        else:
                            note = f" maxdiff={np.abs(o - ref).max():.4f}"
                    print(
                        f"{kv_len:>7} {bs:>3} {rows:>4} {str(blocks):>22} {dt * 1e6:>9.1f} "
                        f"{gbs:>7.0f} {dt / tokens * 1e9:>7.2f}{note}"
                    )


if __name__ == "__main__":
    main()
