"""Microbenchmark for the TPU Pallas fused-MoE ``gmm_v2`` kernel.

Times the MoE expert GEMMs **fused vs. unfused activation** in a single run,
isolating the change made in the "fuse MoE activation into GMM1" PR:

* **fused**   — ``gmm_v2(..., fuse_act=act)``: the [.., 2*intermediate] gate||up
  accumulator is reduced to ``act(gate) * up`` in-kernel before write-out.
* **unfused** — ``gmm_v2(...)`` producing the full [.., 2*intermediate] tensor,
  followed by a separate JAX ``act(gate) * up`` pass (the old path).

It reports two things per shape:

* **GMM1 only** — the precise op the PR changes (gate/up proj + activation).
* **Full MoE** — gather / GMM1(+act) / GMM2 / scatter-reduce, the end-to-end
  per-layer MoE cost.

This runs the real ``sglang.srt.layers.moe.pallas.gmm_v2`` kernel directly with
JAX tensors — no server, no attention, no torch shuttle — so the measured delta
is the MoE kernel itself, not serving overhead.

Requires JAX on a TPU and the *new* ``gmm_v2`` (with ``fuse_act``). Run e.g.::

    LD_PRELOAD=/path/to/abi_shim.so python -m \
      benchmark.kernels.pallas_fused_moe.bench_tpu_fused_moe \
      --num-tokens 1 16 64 256

Defaults model the Qwen3-30B-A3B MoE block (H=2048, I=768, E=128, top_k=8).
"""

import argparse
import statistics
import time

import jax
import jax.numpy as jnp

from sglang.srt.layers.moe.pallas.gmm_v2 import gmm_v2

# gmm_v2 requires the LHS row dim to be a multiple of the sublane tiling; the
# bridge pads to 32 (a multiple of 16 for bf16 and 8 for f32/i32). Mirror that.
M_ALIGN = 32


def _act(gate, up, name):
    if name == "silu":
        return jax.nn.silu(gate) * up
    if name == "gelu":
        return jax.nn.gelu(gate) * up
    raise ValueError(f"unsupported activation {name!r}")


def _build_inputs(m, hidden, inter, num_experts, top_k, dtype, seed=0):
    """Build routed/sorted MoE inputs at GMM shapes (matches fused_moe_op)."""
    k = jax.random.PRNGKey(seed)
    k_x, k_w1, k_w2, k_r = jax.random.split(k, 4)

    hidden_states = jax.random.normal(k_x, (m, hidden), dtype=dtype)
    # gmm_v2 (g, k, n) layout: w1 -> (E, hidden, 2*inter); w2 -> (E, inter, hidden)
    w1 = jax.random.normal(k_w1, (num_experts, hidden, 2 * inter), dtype=dtype) * 0.02
    w2 = jax.random.normal(k_w2, (num_experts, inter, hidden), dtype=dtype) * 0.02

    # Random top-k routing.
    logits = jax.random.normal(k_r, (m, num_experts), dtype=jnp.float32)
    topk_ids = jax.lax.top_k(logits, top_k)[1].astype(jnp.int32)  # [m, top_k]

    topk_ids_flat = topk_ids.reshape(-1)
    argsort_idx = jnp.argsort(topk_ids_flat)
    token_indices = jnp.repeat(jnp.arange(m, dtype=jnp.int32), top_k)
    sorted_token_ids = token_indices[argsort_idx]
    group_sizes = jax.nn.one_hot(topk_ids_flat, num_experts, dtype=jnp.int32).sum(
        axis=0
    )

    x_sorted = hidden_states[sorted_token_ids]  # [m*top_k, hidden]
    pad_m = (-x_sorted.shape[0]) % M_ALIGN
    if pad_m:
        x_sorted = jnp.pad(x_sorted, ((0, pad_m), (0, 0)))
        group_sizes = group_sizes.at[num_experts - 1].add(pad_m)

    return dict(
        x_sorted=x_sorted,
        w1=w1,
        w2=w2,
        group_sizes=group_sizes,
        argsort_idx=argsort_idx,
        m=m,
        top_k=top_k,
        hidden=hidden,
    )


def _make_gmm1(act_name):
    def fused(x_sorted, w1, group_sizes):
        return gmm_v2(lhs=x_sorted, rhs=w1, group_sizes=group_sizes, fuse_act=act_name)

    def unfused(x_sorted, w1, group_sizes):
        out = gmm_v2(lhs=x_sorted, rhs=w1, group_sizes=group_sizes)  # [M, 2*inter]
        n = out.shape[-1] // 2
        return _act(out[:, :n], out[:, n:], act_name)

    return jax.jit(fused), jax.jit(unfused)


def _make_full_moe(act_name, out_dtype):
    def fused(x_sorted, w1, w2, group_sizes):
        g1 = gmm_v2(lhs=x_sorted, rhs=w1, group_sizes=group_sizes, fuse_act=act_name)
        return gmm_v2(
            lhs=g1, rhs=w2, group_sizes=group_sizes, preferred_element_type=out_dtype
        )

    def unfused(x_sorted, w1, w2, group_sizes):
        g1 = gmm_v2(lhs=x_sorted, rhs=w1, group_sizes=group_sizes)
        n = g1.shape[-1] // 2
        act = _act(g1[:, :n], g1[:, n:], act_name)
        return gmm_v2(
            lhs=act, rhs=w2, group_sizes=group_sizes, preferred_element_type=out_dtype
        )

    return jax.jit(fused), jax.jit(unfused)


def _bench(fn, args, iters):
    out = fn(*args)
    jax.block_until_ready(out)  # compile
    for _ in range(3):
        jax.block_until_ready(fn(*args))  # warm
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - t) * 1e3)  # ms
    return statistics.median(samples)


def _max_abs_diff(a, b):
    return float(jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--num-tokens",
        type=int,
        nargs="+",
        default=[1, 16, 64, 256],
        help="batch sizes (decode rows before top_k expansion)",
    )
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--intermediate", type=int, default=768)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--activation", default="silu", choices=["silu", "gelu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    dtype = getattr(jnp, args.dtype)
    devs = jax.devices()
    if not any("TPU" in d.device_kind for d in devs):
        raise SystemExit("No TPU device visible to JAX; this benchmark is TPU-only.")
    print(f"device={devs[0].device_kind}  dtype={args.dtype}  act={args.activation}")
    print(
        f"H={args.hidden} I={args.intermediate} E={args.num_experts} "
        f"top_k={args.top_k}\n"
    )

    gmm1_fused, gmm1_unfused = _make_gmm1(args.activation)
    moe_fused, moe_unfused = _make_full_moe(args.activation, dtype)

    hdr = (
        f"{'tokens':>7} | {'GMM1 unfused':>13} {'GMM1 fused':>11} {'speedup':>8}"
        f" | {'MoE unfused':>12} {'MoE fused':>10} {'speedup':>8} | {'max|Δ|':>8}"
    )
    print(hdr)
    print("-" * len(hdr))

    for m in args.num_tokens:
        d = _build_inputs(
            m, args.hidden, args.intermediate, args.num_experts, args.top_k, dtype
        )
        g1_args = (d["x_sorted"], d["w1"], d["group_sizes"])
        moe_args = (d["x_sorted"], d["w1"], d["w2"], d["group_sizes"])

        # correctness: fused vs unfused GMM1 must match within bf16 tolerance
        diff = _max_abs_diff(gmm1_fused(*g1_args), gmm1_unfused(*g1_args))

        g1_u = _bench(gmm1_unfused, g1_args, args.iters)
        g1_f = _bench(gmm1_fused, g1_args, args.iters)
        moe_u = _bench(moe_unfused, moe_args, args.iters)
        moe_f = _bench(moe_fused, moe_args, args.iters)

        print(
            f"{m:>7} | {g1_u:>11.3f}ms {g1_f:>9.3f}ms {g1_u/g1_f:>7.2f}x"
            f" | {moe_u:>10.3f}ms {moe_f:>8.3f}ms {moe_u/moe_f:>7.2f}x"
            f" | {diff:>8.1e}"
        )


if __name__ == "__main__":
    main()
