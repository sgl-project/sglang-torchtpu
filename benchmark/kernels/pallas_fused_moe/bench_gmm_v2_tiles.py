"""Microbenchmark gmm_v2 tile sizes and MoE intermediate padding for GPT-OSS.

Times the TPU Pallas ``gmm_v2`` grouped matmul on one TPU core with raw JAX
(no server, no torch_tpu runtime). The defaults model the MoE of one
GPT-OSS-120B pipeline stage at TP1: 128 experts, top-4, hidden 2880,
intermediate 2880, bf16 weights (MXFP4 dequantized at load) with biases.

* ``gmm1``: ``x_sorted[m, K] @ w13[e, K, 2*N]`` with the swigluoai gate/up
  activation fused in-kernel (writes ``[m, N]``). ``K`` is hidden padded to a
  multiple of 128 (2944).
* ``gmm2``: ``h[m, N] @ w2[e, N, K]``.
* ``moe``: gmm1 then gmm2, both with the production tiling: the two expert
  GEMMs of ``_fused_moe_jax`` without the permute / un-permute around them.

``m`` = tokens * top_k routed rows. ``N`` (``--intermediates``) is the
intermediate width after load-time zero padding. The next multiple of 128
above 2880 is 2944 = 23 * 128. 23 is prime, so the only lane-aligned divisors
of 2944 are 128 and 2944, and with 64 MiB of VMEM ``make_gmm_configs`` shrinks
gmm1's tile_n from the 1024 ``calculate_tiling`` asks for to 128: 23 N tiles,
each re-streaming the whole lhs. The bf16 loader pads to
``gmm_v2.pad_size_n_to_fewest_tiles`` instead, 3072 there. By default the
script compares the 128-aligned width with the loader's. Pad rows and columns
are zero, so every width computes the same result; the script checks this
against the first width (``max_abs_diff_vs_base_n``).

Tile specs: ``auto`` is ``calculate_tiling`` (what production passes);
otherwise ``tile_m,tile_k,tile_n``, where ``K`` / ``N`` stand for the op's
full contraction / output width (per projection for gmm1). Rows report the
tiles the kernel resolves after ``make_gmm_configs``' divisor fix-up, and the
xprof scope name. A spec that resolves to tiles already measured is reported
as ``dup``; a spec that fails to compile (typically VMEM) is reported, not
fatal.

Each timed sample runs the op ``--iters`` times inside one jit
(``lax.fori_loop``; one element of each output is written into the next
call's lhs, so calls cannot be hoisted or overlapped), amortizing dispatch.

Run on a TPU host, e.g.::

    python -m benchmark.kernels.pallas_fused_moe.bench_gmm_v2_tiles \\
        --tokens 4096 8192 16384 --out results.jsonl
"""

import os

# A raw-JAX TPU process: pick the physical backend before JAX is imported
# (SGLang's platform discovery would otherwise select CPU-only JAX for export).
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import argparse  # noqa: E402
import functools  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402
from jax.experimental.pallas import tpu as pltpu  # noqa: E402


def _import_gmm_v2():
    """Import gmm_v2 without executing ``sglang_tpu.layers.moe.pallas``'s ``__init__``:
    that imports ``fused_moe_op``, which imports torch_tpu at module scope, and
    this process must not start the torch_tpu runtime next to JAX's own TPU
    client. gmm_v2 itself only needs JAX and ``sglang.srt.environ``."""
    for name in ("sglang_tpu.layers.moe", "sglang_tpu.layers.moe.pallas"):
        if name not in sys.modules:
            spec = importlib.util.find_spec(name)
            pkg = types.ModuleType(name)
            pkg.__path__ = list(spec.submodule_search_locations)
            sys.modules[name] = pkg
    return importlib.import_module("sglang_tpu.layers.moe.pallas.gmm_v2")


gmm = _import_gmm_v2()

BF16 = jnp.bfloat16
ACT = "swigluoai"
OPS = ("gmm1", "gmm2", "moe")
DEFAULT_GMM1_SPECS = [
    "auto",
    "128,K,128",
    "256,K,128",
    "512,K,128",
    "128,K,512",
    "256,K,512",
    "128,K,768",
    "256,K,768",
    "128,K,1024",
    "256,K,1024",
]
DEFAULT_GMM2_SPECS = ["auto", "256,K,N"]


def _align(x, a):
    return -(-x // a) * a


def _tile_info(spec, size_k, size_n):
    """``auto`` -> calculate_tiling; ``tm,tk,tn`` (``K``/``N`` = full dim)."""
    if spec == "auto":
        return gmm.calculate_tiling
    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(f"tile spec {spec!r} is not 'auto' or 'tile_m,tile_k,tile_n'")
    full = {"K": size_k, "N": size_n}
    tm, tk, tn = (full[p] if p in full else int(p) for p in parts)
    return gmm.TileSizes(tile_m=tm, tile_k=tk, tile_n=tn)


def _resolve(m, rhs, bias, tile_info, fuse_act):
    """The GmmConfigs gmm_v2 builds for this call (tiles after the fix-ups)."""
    sds = jax.ShapeDtypeStruct
    num_groups, size_k, _ = rhs.shape
    return gmm.make_gmm_configs(
        sds((m, size_k), BF16),
        sds(rhs.shape, rhs.dtype),
        None,
        sds(bias.shape, bias.dtype),
        sds((num_groups,), jnp.int32),
        sds((1,), jnp.int32),
        tile_info=tile_info,
        # What gmm_v2 uses when the caller passes vmem_limit_bytes=None.
        vmem_limit_bytes=gmm.default_vmem_limit_bytes(),
        out_dtype=BF16,
        acc_dtype=None,
        maybe_quantize_lhs=True,
        zero_initialize=True,
        fuse_act=fuse_act,
    )


def _describe(cfgs):
    d, t = cfgs.dims, cfgs.tiles
    num_n = -(-d.size_n // t.tile_n)
    rhs_cols = d.size_n * (2 if cfgs.fuse_act else 1)
    # The grid is (n, gm, k): the lhs is streamed once per N tile.
    est_bytes = (
        num_n * d.size_m * d.size_k * 2
        + d.size_group * d.size_k * rhs_cols * 2
        + d.size_m * d.size_n * 2
    )
    return dict(
        tiles=[t.tile_m, t.tile_k, t.tile_n],
        num_n=num_n,
        num_k=-(-d.size_k // t.tile_k),
        scope=gmm.get_scope_name(cfgs),
        est_hbm_gb=est_bytes / 1e9,
    )


def _real_model(key, *, num_experts, hidden, inter):
    """Unpadded weights; w13's last dim is [gate | up]."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    e = num_experts
    return dict(
        w13=jax.random.normal(k1, (e, hidden, 2 * inter), BF16) * 0.02,
        b13=jax.random.normal(k2, (e, 1, 2 * inter), jnp.float32) * 0.1,
        w2=jax.random.normal(k3, (e, inter, hidden), BF16) * 0.02,
        b2=jax.random.normal(k4, (e, 1, hidden), jnp.float32) * 0.1,
    )


def _padded_model(real, *, hidden, inter, k_pad, n_pad):
    """Zero padding as process_mxfp4_moe_weights_bf16 lays it out: gate and up
    halves each padded to n_pad; hidden padded to k_pad (w13 rows, w2 cols)."""

    def halves(t):
        pad = [(0, 0)] * (t.ndim - 1) + [(0, n_pad - inter)]
        gate, up = t[..., :inter], t[..., inter:]
        return jnp.concatenate([jnp.pad(gate, pad), jnp.pad(up, pad)], axis=-1)

    return dict(
        w13=jnp.pad(halves(real["w13"]), ((0, 0), (0, k_pad - hidden), (0, 0))),
        b13=halves(real["b13"]),
        w2=jnp.pad(real["w2"], ((0, 0), (0, n_pad - inter), (0, k_pad - hidden))),
        b2=jnp.pad(real["b2"], ((0, 0), (0, 0), (0, k_pad - hidden))),
    )


def _group_sizes(key, *, tokens, num_experts, top_k, skew):
    """Rows per expert from top-k routing of random logits; ``skew`` adds a
    per-expert logit bias with that standard deviation (hot experts)."""
    k_logits, k_bias = jax.random.split(key)
    logits = jax.random.normal(k_logits, (tokens, num_experts), jnp.float32)
    if skew:
        logits = logits + skew * jax.random.normal(k_bias, (num_experts,), jnp.float32)
    ids = lax.top_k(logits, top_k)[1].reshape(-1)
    return jnp.zeros((num_experts,), jnp.int32).at[ids].add(1)


def _lhs(key, *, m, width, real_width):
    x = jax.random.normal(key, (m, real_width), BF16)
    # The timing loop writes the previous output's [0, 0] here; the n=1 call
    # (whose output is checked) writes 0, so start from 0.
    x = x.at[0, 0].set(0)
    return jnp.pad(x, ((0, 0), (0, width - real_width)))


def _op_fn(op, tiles1, tiles2):
    def gmm1(lhs, w13, b13, group_sizes):
        return gmm.gmm_v2(
            lhs,
            w13,
            group_sizes,
            rhs_bias=b13,
            tile_info=tiles1,
            fuse_act=ACT,
            preferred_element_type=BF16,
        )

    def gmm2(lhs, w2, b2, group_sizes):
        return gmm.gmm_v2(
            lhs,
            w2,
            group_sizes,
            rhs_bias=b2,
            tile_info=tiles2,
            preferred_element_type=BF16,
        )

    def moe(lhs, w13, b13, w2, b2, group_sizes):
        return gmm2(gmm1(lhs, w13, b13, group_sizes), w2, b2, group_sizes)

    return {"gmm1": gmm1, "gmm2": gmm2, "moe": moe}[op]


def _reference(op):
    """XLA ragged_dot version of the op (f32 accumulate, bf16 between GEMMs)."""

    def g(x, w, b, group_sizes, fuse):
        m = x.shape[0]
        group_ids = jnp.repeat(
            jnp.arange(group_sizes.shape[0]), group_sizes, total_repeat_length=m
        )
        acc = lax.ragged_dot(x, w, group_sizes, preferred_element_type=jnp.float32)
        acc = acc + b[group_ids, 0]
        if fuse:
            n = acc.shape[-1] // 2
            acc = gmm.swigluoai(acc[:, :n], acc[:, n:])
        return acc.astype(BF16)

    def ref(lhs, *rest):
        if op == "gmm1":
            w13, b13, group_sizes = rest
            return g(lhs, w13, b13, group_sizes, True)
        if op == "gmm2":
            w2, b2, group_sizes = rest
            return g(lhs, w2, b2, group_sizes, False)
        w13, b13, w2, b2, group_sizes = rest
        return g(g(lhs, w13, b13, group_sizes, True), w2, b2, group_sizes, False)

    return jax.jit(ref)


def _make_runner(fn, out_shape):
    """``n`` back-to-back calls in one jit; ``n`` is traced, so one executable
    serves both the n=1 output check and the timed loop."""

    @functools.partial(jax.jit, donate_argnums=(0,))
    def run(lhs, rest, n):
        def body(_, carry):
            lhs, out = carry
            lhs = lax.dynamic_update_slice(lhs, out[:1, :1].astype(lhs.dtype), (0, 0))
            return lhs, fn(lhs, *rest)

        return lax.fori_loop(0, n, body, (lhs, jnp.zeros(out_shape, BF16)))

    return run


def _measure(fn, lhs, rest, out_shape, iters, repeats):
    run = _make_runner(fn, out_shape)

    def call(n):
        x = jax.block_until_ready(lhs.copy())
        t = time.perf_counter()
        out = jax.block_until_ready(run(x, rest, jnp.int32(n)))
        return out[1], time.perf_counter() - t

    out, compile_s = call(1)  # compile + one call; its output is checked
    samples = [call(iters)[1] * 1e3 / iters for _ in range(repeats)]
    return out, statistics.median(samples), min(samples), compile_s


def _max_abs_diff(a, b, cols):
    a = a[:, :cols].astype(jnp.float32)
    b = b[:, :cols].astype(jnp.float32)
    return float(jnp.max(jnp.abs(a - b)))


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--ops", nargs="+", default=list(OPS), choices=OPS)
    p.add_argument("--tokens", nargs="+", type=int, default=[4096, 8192, 16384])
    p.add_argument(
        "--routings",
        nargs="+",
        default=["uniform", "skewed"],
        choices=["uniform", "skewed"],
    )
    p.add_argument("--skew", type=float, default=1.0, help="expert logit bias stddev")
    p.add_argument(
        "--intermediates",
        nargs="+",
        type=int,
        default=None,
        help="padded intermediate widths; the first is the comparison base "
        "(default: the 128-aligned width, then the loader's)",
    )
    p.add_argument("--gmm1-specs", nargs="+", default=DEFAULT_GMM1_SPECS)
    p.add_argument("--gmm2-specs", nargs="+", default=DEFAULT_GMM2_SPECS)
    p.add_argument("--num-experts", type=int, default=128)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--hidden", type=int, default=2880)
    p.add_argument("--intermediate", type=int, default=2880, help="unpadded width")
    p.add_argument(
        "--ref-max-tokens",
        type=int,
        default=4096,
        help="check 'auto' rows against XLA ragged_dot up to this many tokens",
    )
    p.add_argument("--iters", type=int, default=10, help="op calls per timed sample")
    p.add_argument("--repeats", type=int, default=5, help="timed samples per config")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="append one JSON object per row")
    args = p.parse_args()

    dev = jax.devices()[0]
    if dev.platform != "tpu":
        raise SystemExit("No TPU device visible to JAX; this benchmark is TPU-only.")
    hidden, inter, top_k = args.hidden, args.intermediate, args.top_k
    k_pad = _align(hidden, 128)
    if args.intermediates is None:
        loader_n = gmm.pad_size_n_to_fewest_tiles(k_pad, inter, BF16, fuse_act=ACT)
        args.intermediates = sorted({_align(inter, 128), loader_n})
    for n_pad in args.intermediates:
        if n_pad < inter or n_pad % 128:
            raise SystemExit(
                f"--intermediates {n_pad}: need >= {inter} and a multiple of 128"
            )
    info = pltpu.get_tpu_info()
    print(
        f"device={dev.device_kind} jax={jax.__version__} "
        f"vmem={info.vmem_capacity_bytes >> 20}MiB E={args.num_experts} top_k={top_k} "
        f"hidden={hidden}->{k_pad} intermediate={inter}->{args.intermediates}",
        flush=True,
    )
    out_f = open(args.out, "a") if args.out else None

    def emit(row):
        tiles = row.get("tiles")
        head = (
            f"{row['op']:>4} t{row['tokens']:<6} {row['routing']:>7} N{row['intermediate']}"
            f" {row['spec']:>11} -> {str(tiles):>34}"
        )
        if row["status"] == "dup":
            print(f"{head} dup of {row['dup_of']}", flush=True)
        elif row["status"] != "ok":
            print(f"{head} FAIL {row['error'][:200]}", flush=True)
        else:
            print(
                f"{head} n_tiles={row['num_n']!s:>7} {row['ms']:8.3f} ms"
                f" (min {row['ms_min']:.3f}) {row['model_tflops']:6.1f} TF/s"
                f" x{row.get('speedup_vs_auto', float('nan')):.3f} vs auto"
                f" x{row.get('speedup_vs_base_n', float('nan')):.3f} vs N{row['base_n']}"
                f" |d|auto={row.get('max_abs_diff_vs_auto', float('nan')):.1e}"
                f" |d|N{row['base_n']}={row.get('max_abs_diff_vs_base_n', float('nan')):.1e}"
                f" |d|ref={row.get('max_abs_diff_vs_ref', float('nan')):.1e}"
                f" finite={row['finite']} compile={row['compile_s']:.1f}s",
                flush=True,
            )
        if out_f:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

    key = jax.random.PRNGKey(args.seed)
    k_model, k_route, k_x1, k_x2 = jax.random.split(key, 4)
    real = _real_model(
        k_model, num_experts=args.num_experts, hidden=hidden, inter=inter
    )
    base_n = args.intermediates[0]
    for tokens in args.tokens:
        m = tokens * top_k
        for routing in args.routings:
            skew = args.skew if routing == "skewed" else 0.0
            group_sizes = _group_sizes(
                k_route,
                tokens=tokens,
                num_experts=args.num_experts,
                top_k=top_k,
                skew=skew,
            )
            imbalance = float(group_sizes.max()) / (m / args.num_experts)
            x1 = _lhs(k_x1, m=m, width=k_pad, real_width=hidden)
            base_out = {}  # op -> (ms of the auto spec, output) at base_n
            for n_pad in args.intermediates:
                w = _padded_model(
                    real, hidden=hidden, inter=inter, k_pad=k_pad, n_pad=n_pad
                )
                x2 = _lhs(k_x2, m=m, width=n_pad, real_width=inter)
                # Useful work at the unpadded model width, so rows at different
                # padded widths compare as the same computation.
                model_flops = {
                    "gmm1": 2 * m * hidden * 2 * inter,
                    "gmm2": 2 * m * inter * hidden,
                }
                model_flops["moe"] = model_flops["gmm1"] + model_flops["gmm2"]
                for op in args.ops:
                    if op == "gmm1":
                        specs = args.gmm1_specs
                        lhs, rest = x1, (w["w13"], w["b13"], group_sizes)
                        out_shape, real_cols = (m, n_pad), inter
                    elif op == "gmm2":
                        specs = args.gmm2_specs
                        lhs, rest = x2, (w["w2"], w["b2"], group_sizes)
                        out_shape, real_cols = (m, k_pad), hidden
                    else:
                        specs = ["auto"]
                        lhs = x1
                        rest = (w["w13"], w["b13"], w["w2"], w["b2"], group_sizes)
                        out_shape, real_cols = (m, k_pad), hidden
                    seen = {}
                    auto = None  # (ms, output) of this op's auto spec
                    ref_out = None
                    for spec in specs:
                        row = dict(
                            op=op,
                            tokens=tokens,
                            m=m,
                            routing=routing,
                            skew=skew,
                            imbalance=imbalance,
                            intermediate=n_pad,
                            base_n=base_n,
                            spec=spec,
                            model_flops=model_flops[op],
                        )
                        try:
                            if op == "gmm1":
                                t1 = _tile_info(spec, k_pad, n_pad)
                                t2 = None
                                cfgs = {
                                    "gmm1": _resolve(m, w["w13"], w["b13"], t1, ACT)
                                }
                            elif op == "gmm2":
                                t1 = None
                                t2 = _tile_info(spec, n_pad, k_pad)
                                cfgs = {"gmm2": _resolve(m, w["w2"], w["b2"], t2, None)}
                            else:
                                t1 = t2 = gmm.calculate_tiling
                                cfgs = {
                                    "gmm1": _resolve(m, w["w13"], w["b13"], t1, ACT),
                                    "gmm2": _resolve(m, w["w2"], w["b2"], t2, None),
                                }
                        except Exception as e:  # noqa: BLE001 - report, go on
                            msg = f"{type(e).__name__}: {e}".replace("\n", " ")
                            row.update(status="fail", error=msg[:4000])
                            emit(row)
                            continue
                        desc = {name: _describe(c) for name, c in cfgs.items()}
                        if op == "moe":
                            row.update(
                                tiles={k: v["tiles"] for k, v in desc.items()},
                                num_n={k: v["num_n"] for k, v in desc.items()},
                                scope={k: v["scope"] for k, v in desc.items()},
                                est_hbm_gb=sum(v["est_hbm_gb"] for v in desc.values()),
                            )
                        else:
                            row.update(desc[op])
                        resolved = json.dumps(row["tiles"])
                        if resolved in seen:
                            row.update(status="dup", dup_of=seen[resolved])
                            emit(row)
                            continue
                        seen[resolved] = spec
                        try:
                            out, ms, ms_min, compile_s = _measure(
                                _op_fn(op, t1, t2),
                                lhs,
                                rest,
                                out_shape,
                                args.iters,
                                args.repeats,
                            )
                        except Exception as e:  # noqa: BLE001 - report, go on
                            msg = f"{type(e).__name__}: {e}".replace("\n", " ")
                            row.update(status="fail", error=msg[:4000])
                            emit(row)
                            continue
                        row.update(
                            status="ok",
                            ms=ms,
                            ms_min=ms_min,
                            compile_s=compile_s,
                            model_tflops=model_flops[op] / ms / 1e9,
                            finite=bool(jnp.isfinite(out).all()),
                        )
                        if out_shape[1] > real_cols:
                            # Pad columns must come out exactly zero.
                            row["pad_cols_max_abs"] = float(
                                jnp.max(jnp.abs(out[:, real_cols:].astype(jnp.float32)))
                            )
                        if spec == "auto":
                            auto = (ms, out)
                            if tokens <= args.ref_max_tokens:
                                ref_out = _reference(op)(lhs, *rest)
                                row["max_abs_diff_vs_ref"] = _max_abs_diff(
                                    out, ref_out, real_cols
                                )
                                row["ref_max_abs"] = float(
                                    jnp.max(jnp.abs(ref_out.astype(jnp.float32)))
                                )
                                ref_out = None
                            if n_pad == base_n:
                                base_out[op] = auto
                        if auto is not None:
                            row["speedup_vs_auto"] = auto[0] / ms
                            row["max_abs_diff_vs_auto"] = _max_abs_diff(
                                out, auto[1], real_cols
                            )
                        if op in base_out:
                            row["speedup_vs_base_n"] = base_out[op][0] / ms
                            row["max_abs_diff_vs_base_n"] = _max_abs_diff(
                                out, base_out[op][1], real_cols
                            )
                        del out
                        emit(row)
                    auto = None
                del w, x2
            base_out.clear()
            del x1
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
