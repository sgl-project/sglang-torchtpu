"""Tune the TPU MLA kernel's block sizes for Kimi-K3 TP32 with an fp8 KV cache.

Runs on one tpu7x core and prints the rows of
sglang_tpu/attention/pallas/tuned_block_sizes_mla.py for 4 q heads per shard
(3 padded to 4), kv_lora 512, rope 64, page 128, bf16 queries and
float8_e4m3fn KV. Same rule as the bf16 Kimi-K3 rows: per token bucket, the
config with the smallest worst-case slowdown against the best config of each
workload.

    python3 benchmark/kernels/mla/tune_mla_fp8_k3_tpu.py
"""

import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas.mla_ragged_paged_attention_v2 import (
    get_kv_cache_shape,
    mla_ragged_paged_attention,
)

HEADS, LKV, R, PAGE = 4, 512, 64, 128
Q_DTYPE, KV_DTYPE = jnp.bfloat16, jnp.float8_e4m3fn
ITERS = 20
_SEEN = set()  # infeasible-config errors already reported


def build(q_lens, kv_lens):
    """Inputs for sequences with these query and kv lengths."""
    n = len(q_lens)
    tokens = int(sum(q_lens))
    aligned = [-(-k // PAGE) * PAGE for k in kv_lens]
    cu_kv = np.concatenate([[0], np.cumsum(aligned)]).astype(np.int32)
    cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    pages = int(cu_kv[-1] // PAGE)
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    ql = jax.random.normal(k1, (tokens, HEADS, LKV), jnp.float32).astype(Q_DTYPE)
    qpe = jax.random.normal(k2, (tokens, HEADS, R), jnp.float32).astype(Q_DTYPE)
    # New tokens arrive in the cache's dtype (the backend quantizes them).
    kvc = (jax.random.normal(k3, (tokens, LKV), jnp.float32) * 0.1).astype(KV_DTYPE)
    kpe = (jax.random.normal(k4, (tokens, R), jnp.float32) * 0.1).astype(KV_DTYPE)
    cache = (
        jax.random.normal(
            k5, get_kv_cache_shape(pages + 8, PAGE, LKV + R, KV_DTYPE), jnp.float32
        )
        * 0.1
    ).astype(KV_DTYPE)
    return dict(
        ql_nope=ql,
        q_pe=qpe,
        new_kv_c=kvc,
        new_k_pe=kpe,
        cache_kv=cache,
        kv_lens=jnp.asarray(kv_lens, jnp.int32),
        page_indices=jnp.arange(pages, dtype=jnp.int32),
        cu_q_lens=jnp.asarray(cu_q),
        cu_kv_lens=jnp.asarray(cu_kv),
    ), n


def timed(inputs, n, decode, bkv, bq, dbs):
    dist = jnp.asarray([n, n, n] if decode else [0, 0, n], jnp.int32)

    def call(ql):
        out = mla_ragged_paged_attention(
            ql,
            inputs["q_pe"],
            inputs["new_kv_c"],
            inputs["new_k_pe"],
            inputs["cache_kv"],
            inputs["kv_lens"],
            inputs["page_indices"],
            inputs["cu_q_lens"],
            inputs["cu_kv_lens"],
            dist,
            sm_scale=0.1,
            num_kv_pages_per_block=(bkv, 1, bkv),
            num_queries_per_block=(bq, 1, bq),
            decode_batch_size=dbs,
            segments="decode" if decode else "mixed",
        )
        return out[0] if isinstance(out, tuple) else out

    def loop(ql):
        def step(i, carry):
            q, acc = carry
            o = call(q)
            return q + (o[:1, :1, :1] * 0).astype(q.dtype), acc + o[0, 0, 0].astype(
                jnp.float32
            )

        return jax.lax.fori_loop(0, ITERS, step, (ql, jnp.float32(0)))[1]

    fn = jax.jit(loop)
    try:
        jax.block_until_ready(fn(inputs["ql_nope"]))
    except Exception as e:  # an infeasible config (VMEM, tiling)
        key = (decode, type(e).__name__)
        if key not in _SEEN:
            _SEEN.add(key)
            print(
                f"infeasible decode={decode} bkv={bkv} bq={bq} dbs={dbs}: {type(e).__name__}: {str(e)[:400]}",
                flush=True,
            )
        return None
    t = time.perf_counter()
    for _ in range(3):
        r = fn(inputs["ql_nope"])
    jax.block_until_ready(r)
    return (time.perf_counter() - t) / 3 / ITERS * 1e6


def choose(times):
    """times: {config: [t per workload or None]} -> (config, worst slowdown)."""
    workloads = len(next(iter(times.values())))
    best = [
        min(t[w] for t in times.values() if t[w] is not None) for w in range(workloads)
    ]
    scored = []
    for cfg, ts in times.items():
        if any(t is None for t in ts):
            continue
        scored.append((max(t / b for t, b in zip(ts, best)), cfg))
    scored.sort()
    return scored[0][1], scored[0][0], best


print("devices", jax.devices()[:1], flush=True)
rows = []
# Decode: bs sequences, one query each, at three kv lengths.
for bs in (1, 2, 4, 8, 16, 32, 64, 128):
    workloads = [build([1] * bs, [kv] * bs) for kv in (512, 2048, 8192)]
    times = {}
    for bkv, dbs in itertools.product((4, 8, 16, 32), (1, 2, 4, 8)):
        if dbs > bs:
            continue
        times[(bkv, 1, dbs)] = [
            timed(inp, n, True, bkv, 1, dbs) for inp, n in workloads
        ]
    cfg, worst, best = choose(times)
    print(
        f"decode bs={bs}: {cfg} worst={worst:.2f}x best_us={[round(b, 1) for b in best]}",
        flush=True,
    )
    rows.append(
        f'("decode", "bfloat16", "float8_e4m3fn", 4, 512, 64, 128, {bs}): {cfg},'
    )
# Mixed: one prefill chunk of t tokens, fresh and behind a 4k prefix.
for t in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
    workloads = [build([t], [t]), build([t], [t + 4096])]
    times = {}
    for bkv, bq in itertools.product((4, 8, 16), (16, 32, 64, 128, 256, 512, 1024)):
        if bq > max(16, t):
            continue
        times[(bkv, bq)] = [timed(inp, n, False, bkv, bq, 1) for inp, n in workloads]
    cfg, worst, best = choose(times)
    print(
        f"mixed t={t}: {cfg} worst={worst:.2f}x best_us={[round(b, 1) for b in best]}",
        flush=True,
    )
    rows.append(f'("mixed", "bfloat16", "float8_e4m3fn", 4, 512, 64, 128, {t}): {cfg},')
print("ROWS")
for r in rows:
    print(r)
print("DONE")
