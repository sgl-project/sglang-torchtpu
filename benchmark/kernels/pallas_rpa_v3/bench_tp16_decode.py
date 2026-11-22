"""Check and time one TP16 shard of Muse SWA decode. Use --interpret on CPU."""

import argparse
from functools import partial
from time import perf_counter
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.attention.pallas import ragged_paged_attention_v3 as v3
from sglang_tpu.attention.pallas import rpa_v3_decode_fast as fast


def run(batch, length, iters, interpret, window=2049):
    if batch < 2 or length < 1 or iters < 1:
        raise ValueError("need batch >= 2, length >= 1 and iters >= 1")
    if not interpret and jax.default_backend() != "tpu":
        raise RuntimeError("timing needs a TPU; pass --interpret for the CPU check")
    rng = np.random.default_rng(1119)
    page_size, width = 128, 64
    pages_per_seq = (length + page_size - 1) // page_size
    live = batch - 1  # check one inactive padding row
    lens = np.maximum(1, length - np.arange(batch) * 17).astype(np.int32)
    lens[-1] = 0
    pages = rng.permutation(batch * pages_per_seq).reshape(batch, -1)
    q = jnp.array(rng.normal(size=(batch, 8, width)), jnp.bfloat16)
    dtype = jnp.float8_e4m3fn
    k = jnp.array(rng.normal(size=(batch, 1, width)), dtype)
    v = jnp.array(rng.normal(size=(batch, 1, width)), dtype)
    # The CPU check poisons free, evicted and tail rows. Stock RPA can return
    # NaNs on evicted slots, so compare hardware timings with clean padding.
    cache = np.full(
        (pages.size, page_size, 1, 4, 128), np.nan if interpret else 0, dtype=dtype
    )
    expected = np.zeros(q.shape, np.float32)
    for row in range(live):
        end = lens[row] - 1
        positions = np.arange(max(0, end + 1 - window), end)
        cached = rng.normal(size=(len(positions), 2, width)).astype(dtype)
        p, t = pages[row, positions // page_size], positions % page_size
        cache[p, t] = 0
        cache[p, t, 0, :2, :width] = cached
        keys = np.concatenate([cached[:, 0], np.asarray(k[row])]).astype(np.float32)
        values = np.concatenate([cached[:, 1], np.asarray(v[row])]).astype(np.float32)
        scores = np.asarray(q[row], dtype=np.float32) @ keys.T * 0.125
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        expected[row] = (probs / probs.sum(axis=-1, keepdims=True)) @ values

    original = jnp.asarray(cache.copy())
    for row in range(live):
        end = lens[row] - 1
        dest = cache[pages[row, end // page_size], end % page_size, 0]
        dest[:] = 0
        dest[0, :width], dest[1, :width] = np.asarray(k[row, 0]), np.asarray(v[row, 0])
    args = (
        q,
        k,
        v,
        original,
        jnp.asarray(lens),
        jnp.array(pages.flatten(), jnp.int32),
        jnp.minimum(jnp.arange(batch + 1), live),
        jnp.arange(batch + 1) * pages_per_seq * page_size,
        jnp.full(3, live, jnp.int32),
    )

    for mode in ("tp16",) if interpret else ("", "tp16"):
        jit = jax.jit(
            partial(
                v3.ragged_paged_attention,
                custom_mask=None,
                sm_scale=0.125,
                sliding_window=window,
                _skip_mixed=True,
                _decode_fast=mode,
                _decode_window_aligned_kv=True,
                _decode_zero_kv_tail=True,
                d_block_sizes=(1, 2176, 1, 2176),
                vmem_limit_bytes=64 * 1024**2,
            ),
            donate_argnums=(3,),
        )
        # Exercise the public dispatch on CPU too. Stock RPA needs TPU lowering.
        with patch.object(
            fast,
            "run_decode_fast_rpa",
            partial(
                fast.run_decode_fast_rpa,
                interpret=interpret,
            ),
        ):
            output, updated = jax.block_until_ready(
                jit(*args[:3], original.copy(), *args[4:])
            )
            if mode == "tp16" and not interpret:
                assert (
                    f"RPAd_fast-s4-c256-stream0b-mp17-sw_{window}-tp16"
                    in jit.lower(*args).as_text()
                )
            out = np.asarray(output, dtype=np.float32)
            np.testing.assert_array_equal(
                np.asarray(updated).view(np.uint8), cache.view(np.uint8)
            )
            if mode == "tp16":
                np.testing.assert_array_equal(out[-1], np.zeros_like(out[-1]))
            rel = np.linalg.norm(out[:live] - expected[:live]) / np.linalg.norm(
                expected[:live]
            )
            assert np.isfinite(rel) and rel < 0.012, rel
            print(
                f"{mode or 'stock'} batch={batch} length={length} window={window} rel_l2={rel:.6f}",
                flush=True,
            )
            if not interpret:
                elapsed = []
                for _ in range(iters):
                    start = perf_counter()
                    _, updated = jax.block_until_ready(
                        jit(*args[:3], updated, *args[4:])
                    )
                    elapsed.append((perf_counter() - start) * 1e6)
                print(f"median_wall_us={np.median(elapsed):.1f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=2177)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--window", type=int, choices=(2048, 2049), default=2049)
    parser.add_argument("--interpret", action="store_true")
    opts = parser.parse_args()
    run(opts.batch, opts.length, opts.iters, opts.interpret, opts.window)
