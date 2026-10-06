"""Microbenchmark rpa_v3 prefill block sizes and the sliding-window chunk skip.

Times ``ragged_paged_attention`` (the pallas_tpu_rpa_v3 kernel) on one TPU core
with raw JAX: no server, no torch_tpu runtime. The defaults model one
GPT-OSS-120B pipeline stage at TP1 (all 64 query / 8 KV heads on one core,
head_dim 64, attention sinks) prefilling 16K-token chunks:

* ``full_*`` cases are the full-attention layers, ``swa_*`` the sliding-window
  (128-token) layers.
* Extend rows run on the kernel's MIXED range with no ``chunk_prefill_size``,
  as the sglang backend runs them, so block sizes go in via ``m_block_sizes``.
* ``default`` is what ``get_default_block_sizes`` picks for the shape; every
  other candidate is ``bq_sz,bkv_sz,bq_csz,bkv_csz``. A candidate that does not
  compile (typically VMEM) is reported, not fatal.
* Sliding-window cases run with the chunk skip on and off (the kernel's
  private ``_sliding_window_chunk_skip``); the outputs must be identical,
  which the script checks.

Each timed sample runs the kernel ``--iters`` times back to back inside one jit
(``lax.fori_loop``; each call's output is the next call's query), so dispatch
overhead is amortized. The time per call includes the wrapper's input/output
relayout; ``debug`` rows time the same call with ``debug_mode=True`` (DMAs and
attention skipped) as a floor for that relayout plus loop overhead.

Run on a TPU host, e.g.::

    python -m benchmark.kernels.pallas_rpa_v3.bench_rpa_v3_prefill_blocks \\
        --kv-dtypes fp8 bf16 --page-sizes 64 256 --out results.jsonl
"""

import os

# A raw-JAX TPU process: pick the physical backend before JAX is imported
# (SGLang's platform discovery would otherwise select CPU-only JAX for export).
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import argparse  # noqa: E402
import functools  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

from sglang.srt.layers.attention.pallas import (  # noqa: E402
    ragged_paged_attention_v3 as rpa,
)

KV_DTYPES = {"bf16": jnp.bfloat16, "fp8": jnp.float8_e4m3fn}

# name -> ([(kv_len, q_len), ...], sliding_window). kv_len includes the new
# q_len tokens. The kernel's sliding_window counts attended tokens including
# the current one: the backend passes sliding_window_size + 1 = 128.
CASES = {
    "full_16k": ([(16384, 16384)], None),
    "full_2x8k": ([(8192, 8192)] * 2, None),
    "full_4x4k": ([(4096, 4096)] * 4, None),
    "full_16k_prefix16k": ([(32768, 16384)], None),
    "swa_16k": ([(16384, 16384)], 128),
    "swa_4x4k": ([(4096, 4096)] * 4, 128),
    "swa_16k_prefix16k": ([(32768, 16384)], 128),
}
DEFAULT_CASES = ["full_16k", "full_4x4k", "full_16k_prefix16k", "swa_16k", "swa_4x4k"]
DEFAULT_BLOCKS = [
    "default",
    "64,2048,64,512",
    "64,2048,64,256",
    "64,1024,64,512",
    "128,512,128,256",
    "128,1024,128,256",
    "128,2048,128,256",
    "128,1024,128,512",
    "128,2048,128,512",
    "256,512,256,256",
    "256,1024,256,256",
]


def _tpu_generation():
    kind = jax.devices()[0].device_kind  # e.g. "TPU7x", "TPU v6 lite"
    digits = "".join(ch for ch in kind.split("TPU", 1)[-1] if ch.isdigit())
    if not digits:
        raise SystemExit(f"cannot parse a TPU generation from {kind!r}")
    return int(digits[0])


def _flops(seqs, sliding_window, num_q_heads, head_dim):
    """Useful attention FLOPs: QK^T + PV over visible keys at the model head_dim."""
    total = 0
    for kv_len, q_len in seqs:
        visible = np.arange(kv_len - q_len, kv_len, dtype=np.int64) + 1
        if sliding_window is not None:
            visible = np.minimum(visible, sliding_window)
        total += int(visible.sum())
    return 4 * head_dim * num_q_heads * total


def _build_inputs(
    seqs, *, num_q_heads, num_kv_heads, head_dim, page_size, context_len, kv_dtype
):
    num_seqs = len(seqs)
    pages_per_seq = -(-max([context_len] + [kv for kv, _ in seqs]) // page_size)
    total_pages = num_seqs * pages_per_seq
    total_q = sum(q for _, q in seqs)
    max_num_tokens = -(-total_q // 128) * 128
    head_dim_pad = -(-head_dim // 128) * 128
    kv_packing = 4 // jnp.dtype(kv_dtype).itemsize
    # Interleaved K/V heads padded up to the packing, as get_kv_cache_shape
    # lays out the pool (e.g. 1 KV head with FP8 KV: 2 heads -> one row of 4).
    heads_x2 = -(-num_kv_heads * 2 // kv_packing) * kv_packing

    kq, kk, kv, kc, ks, kp = jax.random.split(jax.random.PRNGKey(0), 6)
    bf16 = jnp.bfloat16
    q = jax.random.normal(kq, (max_num_tokens, num_q_heads, head_dim), bf16)
    # The backend quantizes new K/V to the cache dtype before the call.
    k_shape = (max_num_tokens, num_kv_heads, head_dim)
    k = jax.random.normal(kk, k_shape, bf16).astype(kv_dtype)
    v = jax.random.normal(kv, k_shape, bf16).astype(kv_dtype)
    # The TPU KV pool stores head_dim padded to 128 with zero padding lanes.
    cache = jax.random.normal(kc, (total_pages, page_size, heads_x2, head_dim), bf16)
    pad = ((0, 0), (0, 0), (0, 0), (0, head_dim_pad - head_dim))
    cache = jnp.pad(cache, pad).astype(kv_dtype)
    cache = cache.reshape(
        total_pages, page_size, heads_x2 // kv_packing, kv_packing, head_dim_pad
    )
    assert cache.shape == rpa.get_kv_cache_shape(
        total_pages, page_size, num_kv_heads, head_dim, kv_dtype
    ), cache.shape
    # Scattered physical pages, as a long-running pool hands them out.
    pages = np.asarray(jax.random.permutation(kp, total_pages), dtype=np.int32)
    cu_q = [0] + [int(x) for x in np.cumsum([ql for _, ql in seqs])]
    return dict(
        q=q,
        k=k,
        v=v,
        cache=cache,
        sinks=jax.random.normal(ks, (num_q_heads,), jnp.float32),
        kv_lens=jnp.asarray([kv for kv, _ in seqs], jnp.int32),
        page_indices=jnp.asarray(pages),
        cu_q_lens=jnp.asarray(cu_q, jnp.int32),
        cu_kv_lens=jnp.asarray(
            [i * pages_per_seq * page_size for i in range(num_seqs + 1)], jnp.int32
        ),
        # All rows on the MIXED range: [decode_end, prefill_end, mixed_end].
        distribution=jnp.asarray([0, 0, num_seqs], jnp.int32),
        pages_per_seq=pages_per_seq,
        max_num_tokens=max_num_tokens,
        total_q=total_q,
        num_seqs=num_seqs,
    )


def _default_blocks(d, *, hq, hkv, hd, page_size, kv_dtype, sliding_window):
    """Mirror of the wrapper's _prepare_block_sizes for the MIXED range."""
    b = rpa.get_default_block_sizes(
        jnp.bfloat16,
        kv_dtype,
        hq,
        hkv,
        hd,
        page_size,
        d["max_num_tokens"],
        d["num_seqs"],
        d["pages_per_seq"],
        case=rpa.RpaCase.MIXED,
        vmem_limit_bytes=rpa.get_vmem_limit(),
        use_custom_mask=False,
        sliding_window=sliding_window,
    )
    return (b["bq_sz"], b["bkv_sz"], b["bq_csz"], b["bkv_csz"])


def _make_runner(kw):
    """``n`` back-to-back kernel calls in one jit; ``n`` is traced, so one
    executable serves both the n=1 output check and the timed loop."""

    @functools.partial(jax.jit, donate_argnums=(0, 3))
    def run(q, k, v, cache, sinks, kv_lens, pages, cu_q, cu_kv, dist, n):
        def body(_, carry):
            q, cache = carry
            return rpa.ragged_paged_attention(
                q, k, v, cache, kv_lens, pages, cu_q, cu_kv, dist, None, sinks, **kw
            )

        return lax.fori_loop(0, n, body, (q, cache))

    return run


def _measure(d, kw, iters, repeats):
    run = _make_runner(kw)

    def call(n):
        args = jax.block_until_ready((d["q"].copy(), d["cache"].copy()))
        t = time.perf_counter()
        out = jax.block_until_ready(
            run(
                args[0],
                d["k"],
                d["v"],
                args[1],
                d["sinks"],
                d["kv_lens"],
                d["page_indices"],
                d["cu_q_lens"],
                d["cu_kv_lens"],
                d["distribution"],
                jnp.int32(n),
            )
        )
        return out[0], time.perf_counter() - t

    out, compile_s = call(1)  # compile + one call; its output is checked
    samples = [call(iters)[1] * 1e3 / iters for _ in range(repeats)]
    out = np.asarray(out[: d["total_q"]].astype(jnp.float32))
    return out, statistics.median(samples), min(samples), compile_s


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cases", nargs="+", default=DEFAULT_CASES, choices=sorted(CASES))
    p.add_argument(
        "--blocks",
        nargs="+",
        default=DEFAULT_BLOCKS,
        help='"default" or "bq_sz,bkv_sz,bq_csz,bkv_csz"',
    )
    p.add_argument(
        "--kv-dtypes", nargs="+", default=["fp8", "bf16"], choices=sorted(KV_DTYPES)
    )
    p.add_argument("--page-sizes", nargs="+", type=int, default=[64, 256])
    p.add_argument("--num-q-heads", type=int, default=64)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("--context-len", type=int, default=32768, help="sets pages_per_seq")
    p.add_argument(
        "--iters", type=int, default=10, help="kernel calls per timed sample"
    )
    p.add_argument("--repeats", type=int, default=5, help="timed samples per config")
    p.add_argument("--no-debug-floor", action="store_true", help="skip debug_mode rows")
    p.add_argument("--out", default=None, help="append one JSON object per row")
    args = p.parse_args()

    dev = jax.devices()[0]
    if dev.platform != "tpu":
        raise SystemExit("No TPU device visible to JAX; this benchmark is TPU-only.")
    # get_tpu_version() reads the device kind through torch_tpu; this raw-JAX
    # process must not start the torch_tpu runtime, so answer from JAX.
    generation = _tpu_generation()
    rpa.get_tpu_version = lambda: generation
    hq, hkv, hd = args.num_q_heads, args.num_kv_heads, args.head_dim
    sm_scale = 1.0 / float(np.sqrt(hd))
    print(
        f"device={dev.device_kind} gen={generation} jax={jax.__version__} "
        f"vmem_limit={rpa.get_vmem_limit()} heads={hq}/{hkv} head_dim={hd}",
        flush=True,
    )
    out_f = open(args.out, "a") if args.out else None

    def emit(row):
        head = (
            f"{row['case']:>18} {row['kv']:>4} p{row['page']:<4}"
            f" {row['arm']:>6} {str(tuple(row['blocks'])):>22}"
            f" vmem~{row['vmem_mib']:5.1f}MiB"
        )
        if row["status"] != "ok":
            print(f"{head} FAIL {row['error'][:200]}", flush=True)
        else:
            print(
                f"{head} {row['ms']:8.3f} ms (min {row['ms_min']:.3f})"
                f" {row['tflops']:6.1f} TF/s"
                f" x{row.get('speedup_vs_default', float('nan')):.3f}"
                f" max|d|={row.get('max_abs_diff_vs_default', float('nan')):.1e}"
                f" eq_noskip={row.get('equal_to_noskip', '-')}"
                f" finite={row.get('finite', '-')} compile={row['compile_s']:.1f}s",
                flush=True,
            )
        if out_f:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

    for page in args.page_sizes:
        for kv_name in args.kv_dtypes:
            kv_dtype = KV_DTYPES[kv_name]
            kv_scale = 1.0 if kv_name == "fp8" else None
            for case in args.cases:
                seqs, sw = CASES[case]
                d = _build_inputs(
                    seqs,
                    num_q_heads=hq,
                    num_kv_heads=hkv,
                    head_dim=hd,
                    page_size=page,
                    context_len=args.context_len,
                    kv_dtype=kv_dtype,
                )
                flops = _flops(seqs, sw, hq, hd)
                default = _default_blocks(
                    d,
                    hq=hq,
                    hkv=hkv,
                    hd=hd,
                    page_size=page,
                    kv_dtype=kv_dtype,
                    sliding_window=sw,
                )
                specs = []
                for spec in args.blocks:
                    blocks = (
                        default
                        if spec == "default"
                        else tuple(int(x) for x in spec.split(","))
                    )
                    if spec == "default" or blocks != default:
                        specs.append((spec, blocks))
                if not args.no_debug_floor:
                    specs.append(("debug", default))
                baseline = {}  # arm -> (ms, output) for the default blocks
                for spec, blocks in specs:
                    if spec == "debug":
                        arms = ["debug"]
                    else:
                        arms = ["skip", "noskip"] if sw is not None else ["full"]
                    outs = {}
                    rows = []
                    for arm in arms:
                        row = dict(
                            case=case,
                            kv=kv_name,
                            page=page,
                            arm=arm,
                            spec=spec,
                            blocks=list(blocks),
                            is_default=blocks == default,
                            seqs=seqs,
                            sliding_window=sw,
                            flops=flops,
                            vmem_mib=rpa.get_vmem_estimate_bytes(
                                hkv,
                                hq // hkv,
                                hd,
                                blocks[0],
                                blocks[1],
                                jnp.bfloat16,
                                kv_dtype,
                                bkv_csz=blocks[3],
                            )
                            / 2**20,
                        )
                        kw = dict(
                            sm_scale=sm_scale,
                            sliding_window=sw,
                            k_scale=kv_scale,
                            v_scale=kv_scale,
                            m_block_sizes=blocks,
                            _sliding_window_chunk_skip=arm != "noskip",
                            debug_mode=arm == "debug",
                        )
                        try:
                            out, ms, ms_min, compile_s = _measure(
                                d, kw, args.iters, args.repeats
                            )
                        except Exception as e:  # noqa: BLE001 - report, go on
                            msg = f"{type(e).__name__}: {e}".replace("\n", " ")
                            row.update(status="fail", error=msg[:4000])
                            rows.append(row)
                            continue
                        row.update(
                            status="ok",
                            ms=ms,
                            ms_min=ms_min,
                            compile_s=compile_s,
                            tflops=flops / ms / 1e9,
                        )
                        if arm != "debug":
                            outs[arm] = out
                            row["finite"] = bool(np.isfinite(out).all())
                            if spec == "default":
                                baseline[arm] = (ms, out)
                            if arm in baseline:
                                row["speedup_vs_default"] = baseline[arm][0] / ms
                                row["max_abs_diff_vs_default"] = float(
                                    np.abs(out - baseline[arm][1]).max()
                                )
                        rows.append(row)
                    if "skip" in outs and "noskip" in outs:
                        equal = bool(np.array_equal(outs["skip"], outs["noskip"]))
                        for row in rows:
                            row["equal_to_noskip"] = equal
                    for row in rows:
                        emit(row)
                del d
    if out_f:
        out_f.close()


if __name__ == "__main__":
    main()
