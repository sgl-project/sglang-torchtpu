"""TPU microbenchmark: GPT-OSS-120B TP8 decode megakernel step vs baselines (one 2x2x1 host).

Random weights at real dims (expert format per --w-dtype, default fp8), random KV pools, every
sequence at context `--ctx`.
Prints one JSON line per (mode, B). Modes:

  mk        one pallas_call per step (in-kernel all-reduce) + the in-place KV scatter kernel
  fallback  same kernel bodies split per layer/phase with XLA psum in between (2L+1 calls)
  xla       pure-JAX/XLA per-rank decode step with the same packed weights and pools, XLA psum
            (a "no custom kernels" proxy; NOT the stock sglang torch_tpu path)

The stock sglang decode forward itself (torch_tpu, RPA v3, gmm_v2 MXFP4 MoE) is not
reproduced here; compare against the T21 Phase-0 xprof forward-time numbers for that.
Note the stock path reads MXFP4 experts (0.57x the bytes of the fp8 expert format).

  python benchmark/kernels/gptoss_decode_megakernel/bench_decode_step.py \
      --batch 1 8 32 64 --modes mk fallback xla --ctx 1024
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from aot_compile import import_kernel  # noqa: E402


def xla_step_fn(c, mesh, MK, REF):
    """Pure-JAX per-rank decode step on the packed weights (baseline only)."""
    import jax
    import jax.numpy as jnp
    from jax import lax
    from jax.sharding import PartitionSpec as P

    BF16, F32 = jnp.bfloat16, jnp.float32
    kvdt = REF.kv_jdtype(c)

    def per_rank(rank, x, pos, pt_s, pt_f, W, kv_s, kv_f):
        W = {k: v[0] for k, v in W.items()}
        kv_s, kv_f = kv_s[0], kv_f[0]
        rank = rank[0]
        B = c.B
        cos, sin = REF.yarn_tables(c, pos)
        res = jnp.pad(x, ((0, 0), (0, c.Hp - c.H))).astype(F32)
        kvn_all = []
        for l in range(c.L):
            vec = W["vec"][l]
            sliding = c.sliding[l]
            pool, pt = (kv_s, pt_s) if sliding else (kv_f, pt_f)
            xf = res
            xn = (
                xf
                * lax.rsqrt(jnp.sum(xf * xf, -1, keepdims=True) / c.H + c.eps)
                * vec[0, : c.Hp]
            ).astype(BF16)
            qkv = (
                jnp.dot(xn, W["wqkv"][l], preferred_element_type=F32)
                + vec[3, : c.QKVWL]
            )
            q = REF.rope(qkv[:, : c.QWL].reshape(B, c.NQL, c.D), cos, sin)
            k = REF.rope(qkv[:, c.QWL : c.QWL + c.D].reshape(B, 1, c.D), cos, sin)[:, 0]
            v = qkv[:, c.QWL + 128 : c.QWL + 128 + c.D]
            kq, vq = k.astype(kvdt), v.astype(kvdt)
            if sliding:  # only the window's pages; positions relative to its first page
                plo = jnp.maximum(pos - (c.window - 1), 0) // c.page
                widx = jnp.minimum(
                    plo[:, None] + jnp.arange(c.NWP)[None, :], c.max_pages - 1
                )
                pt_l = jnp.take_along_axis(pt, widx, axis=1)
                pos_l = pos - plo * c.page
            else:
                pt_l, pos_l = pt, pos
            pages = pool[l // 2][pt_l]  # [B, n_pages, page, 1, packing, 128]
            kh = pages[:, :, :, 0, 0, : c.D].reshape(B, -1, c.D)
            vh = pages[:, :, :, 0, 1, : c.D].reshape(B, -1, c.D)
            o = REF.attention(
                c,
                q,
                kq.astype(F32)[:, None],
                vq.astype(F32)[:, None],
                kh.astype(F32)[:, :, None],
                vh.astype(F32)[:, :, None],
                pos_l,
                W["sinks"][l][:, 0],
                sliding,
            )
            part = jnp.dot(
                o.reshape(B, c.QWL).astype(BF16), W["wo"][l], preferred_element_type=F32
            )
            res = res + lax.psum(part.astype(BF16).astype(F32), "tp") + vec[2, : c.Hp]
            res = res.astype(BF16).astype(F32)
            xn = (
                res
                * lax.rsqrt(jnp.sum(res * res, -1, keepdims=True) / c.H + c.eps)
                * vec[1, : c.Hp]
            ).astype(BF16)
            lg = jnp.dot(xn, W["wr"][l], preferred_element_type=F32) + vec[4, :128]
            vals, ids = lax.top_k(lg, c.TOPK)
            wts = jax.nn.softmax(vals, -1)
            base = rank * c.EL if c.moe_mode == "ep" else 0
            loc = ids - base
            ok = (loc >= 0) & (loc < c.EL)
            loc = jnp.clip(loc, 0, c.EL - 1)
            # packed (chunk-tiled) layouts used directly, so XLA does not relayout weights
            if c.w_dtype == "fp8":
                small = B * c.TOPK < c.EL
                pick = (lambda a: a[loc]) if small else (lambda a: a)
                w13 = (pick(W["w13"][l]).astype(F32) * pick(W["c13"][l])).astype(BF16)
                w2 = (pick(W["w2"][l]).astype(F32) * pick(W["c2"][l])).astype(BF16)
            elif (
                c.w_dtype == "mxfp4"
            ):  # dequantize in XLA (only the gathered experts if small B)

                def deq(n, idx=None):
                    q, sc = W[n + "q"][l], W[n + "s"][l]
                    if idx is not None:
                        q, sc = q[idx], sc[idx]
                    return MK.mxfp4_unpack_xla(q, sc).astype(BF16)

                small = B * c.TOPK < c.EL
                w13 = deq("w13", loc if small else None)
                w2 = deq("w2", loc if small else None)
            else:
                w13, w2 = (
                    W["w13"][l],
                    W["w2"][l],
                )  # [EL, NJ, Hp, 2*tne], [EL, NJ, tne, Hp]
                if B * c.TOPK < c.EL:
                    w13, w2 = w13[loc], w2[loc]
            b13, b2 = (
                W["b13"][l][:, :, 0],
                W["b2"][l][:, 0],
            )  # [EL, NJ, 2*tne], [EL, Hp]
            flag = 1.0 if c.moe_mode == "ep" else (rank == 0).astype(F32)
            T_ = c.tne

            def act(h):
                g = jnp.minimum(h[..., :T_], c.limit)
                u = jnp.clip(h[..., T_:], -c.limit, c.limit)
                return ((u + 1.0) * (g * jax.nn.sigmoid(c.alpha * g))).astype(BF16)

            if B * c.TOPK < c.EL:  # gather the routed experts' weights
                h = jnp.einsum("bh,bkjhn->bkjn", xn, w13, preferred_element_type=F32)
                a = act(h + b13[loc])
                y = jnp.einsum("bkjt,bkjth->bkh", a, w2, preferred_element_type=F32)
                y = y + flag * b2[loc]
                part = jnp.sum(jnp.where(ok, wts, 0.0)[..., None] * y, 1)
            else:  # dense over all local experts (most are active at this batch)
                wfull = (
                    jnp.zeros((B, c.EL), F32)
                    .at[jnp.arange(B)[:, None], loc]
                    .add(jnp.where(ok, wts, 0.0))
                )
                h = jnp.einsum("bh,ejhn->ebjn", xn, w13, preferred_element_type=F32)
                a = act(h + b13[:, None])
                y = jnp.einsum("ebjt,ejth->ebh", a, w2, preferred_element_type=F32)
                y = y + flag * b2[:, None]
                part = jnp.einsum("be,ebh->bh", wfull, y)
            res = (
                (res + lax.psum(part.astype(BF16).astype(F32), "tp"))
                .astype(BF16)
                .astype(F32)
            )
            kvn_all.append((kq, vq))
        # write new K/V in place (XLA scatter)
        for l, (kq, vq) in enumerate(kvn_all):
            pt = pt_s if c.sliding[l] else pt_f
            page = pt[jnp.arange(B), pos // c.page]
            tok = pos % c.page
            slot = jnp.zeros((B, c.kv_packing, 128), kvdt)
            slot = slot.at[:, 0, : c.D].set(kq).at[:, 1, : c.D].set(vq)
            if c.sliding[l]:
                kv_s = kv_s.at[l // 2, page, tok, 0].set(slot)
            else:
                kv_f = kv_f.at[l // 2, page, tok, 0].set(slot)
        return res.astype(BF16)[:, : c.H], kv_s[None], kv_f[None]

    w_spec = {k: P("tp") for k in MK.w_names(c)}
    smap = jax.shard_map(
        per_rank,
        mesh=mesh,
        in_specs=(P("tp"), P(), P(), P(), P(), w_spec, P("tp"), P("tp")),
        out_specs=(P(), P("tp"), P("tp")),
        check_vma=False,
    )

    def step(x, pos, pt_swa, pt_full, W, kv_swa, kv_full):
        return smap(
            jnp.arange(c.R, dtype=jnp.int32),
            x,
            pos,
            pt_swa,
            pt_full,
            W,
            kv_swa,
            kv_full,
        )

    return step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--modes", nargs="+", default=["mk", "fallback", "xla"])
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument(
        "--ctx",
        type=int,
        default=1024,
        help="tokens already in every sequence (e.g. 1024, 4096, 8192, 32768)",
    )
    ap.add_argument("--kv", default="fp8", choices=["bf16", "fp8"])
    ap.add_argument("--moe", default="tp", choices=["tp", "ep"])
    ap.add_argument("--tne", type=int, default=384)
    ap.add_argument(
        "--npb", type=int, default=8, help="KV pages per attention item (full layers)"
    )
    ap.add_argument(
        "--kv-bufs", type=int, default=3, help="KV item buffers (prefetch depth + 1)"
    )
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--vmem-mib", type=int, default=60)
    ap.add_argument(
        "--mxfp4", action="store_true", help="MXFP4 expert weights (stock format)"
    )
    ap.add_argument(
        "--w-dtype",
        choices=["bf16", "mxfp4", "fp8"],
        default=None,
        help="expert weight format (default fp8 = MXFP4 converted to e4m3)",
    )
    ap.add_argument(
        "--ar-mode",
        choices=["auto", "oneshot", "rsag"],
        default="auto",
        help="all-reduce algorithm (auto: rsag for B >= 16)",
    )
    ap.add_argument(
        "--ablate",
        nargs="*",
        default=[],
        choices=["ar", "attn", "attn_kv", "moe", "experts"],
        help="mk/fallback timing ablations (outputs wrong); see config.ABLATIONS",
    )
    ap.add_argument(
        "--kv-write",
        choices=["scatter", "kernel"],
        default="scatter",
        help="kernel: megakernel writes new K/V into aliased pools (no scatter call)",
    )
    ap.add_argument(
        "--no-scatter",
        action="store_true",
        help="leave out the separate KV scatter call (pools not updated)",
    )
    ap.add_argument(
        "--steps-per-call",
        type=int,
        default=8,
        help="also time K steps inside one jit (ms_device); 1 disables",
    )
    ap.add_argument(
        "--no-ar-fold",
        action="store_true",
        help="all-reduce full [BR, Hp] rows even for B <= 4",
    )
    ap.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="tiny dims + Pallas interpret mode on 8 virtual CPU devices (script check)",
    )
    args = ap.parse_args()
    if args.cpu_smoke:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    config, MK = import_kernel()
    from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import (
        reference as REF,
    )

    devs = jax.devices()
    assert len(devs) >= 8, f"need 8 TPU devices, got {devs}"
    mesh = Mesh(np.array(devs[:8]), ("tp",))
    shd, rep = NamedSharding(mesh, P("tp")), NamedSharding(mesh, P())
    max_pages = -(-(args.ctx + 1) // 64)
    nb = max(args.batch) * max_pages + 1  # full pool: every page of every sequence
    swa_ring = 4  # > NWP pages: sliding layers only ever touch the window's pages
    nb_swa = max(args.batch) * swa_ring + 1

    def rand_pool(spec, key):
        """Finite random KV bits without an f32 intermediate (pools reach ~20 GiB/rank)."""

        def f(k):
            if spec.dtype == jnp.bfloat16:
                bits = jax.random.bits(k, spec.shape, jnp.uint16) & jnp.uint16(0xBE7F)
            else:  # fp8 e4m3fn: clearing bit 3 of the exponent avoids the NaN code
                bits = jax.random.bits(k, spec.shape, jnp.uint8) & jnp.uint8(0xB7)
            return jax.lax.bitcast_convert_type(bits, spec.dtype)

        return jax.jit(f, out_shardings=shd)(key)

    def new_pools(c):
        return [
            rand_pool(s, k)
            for s, k in zip(
                MK.pool_specs(c), jax.random.split(jax.random.PRNGKey(1), 2)
            )
        ]

    def rand(spec, key, scale):
        f = lambda k: (jax.random.normal(k, spec.shape, jnp.float32) * scale).astype(
            spec.dtype
        )
        return jax.jit(f, out_shardings=shd)(key)

    W = None
    for B in args.batch:
        mk_cfg = config.tiny if args.cpu_smoke else config.full
        extra = (
            {}
            if args.cpu_smoke
            else dict(L=args.layers, tne=args.tne, vmem_mib=args.vmem_mib)
        )
        c = mk_cfg(
            B,
            kv_dtype=args.kv,
            moe_mode=args.moe,
            npb=args.npb,
            nb_full=nb,
            w_dtype=args.w_dtype or ("mxfp4" if args.mxfp4 else "fp8"),
            ar_mode=args.ar_mode,
            ablate=tuple(args.ablate),
            kv_write=args.kv_write,
            ar_fold=not args.no_ar_fold,
            nb_swa=nb_swa,
            max_pages=max_pages,
            kv_bufs=args.kv_bufs,
            **extra,
        )
        if W is None:  # weights do not depend on B
            keys = jax.random.split(jax.random.PRNGKey(0), 16)
            W = {}
            for i, (k, s) in enumerate(MK.packed_specs(c).items()):
                if s.dtype == jnp.uint32:  # any nibble is a valid e2m1 code
                    W[k] = jax.jit(
                        lambda kk, s=s: jax.random.bits(kk, s.shape, jnp.uint32),
                        out_shardings=shd,
                    )(keys[i])
                elif s.dtype == jnp.float8_e4m3fn:  # finite e4m3, |w8| < 1
                    W[k] = jax.jit(
                        lambda kk, s=s: jax.lax.bitcast_convert_type(
                            jax.random.bits(kk, s.shape, jnp.uint8) & jnp.uint8(0xB7),
                            s.dtype,
                        ),
                        out_shardings=shd,
                    )(keys[i])
                elif k in ("c13", "c2"):
                    W[k] = jax.jit(
                        lambda s=s: jnp.full(s.shape, 2.0**-4, s.dtype),
                        out_shardings=shd,
                    )()
                elif k in ("w13s", "w2s"):  # fp4 magnitudes <= 6 -> |w| <= 0.05
                    W[k] = jax.jit(
                        lambda s=s: jnp.full(s.shape, 2.0**-7, s.dtype),
                        out_shardings=shd,
                    )()
                else:
                    scale = {"vec": 0.0, "sinks": 1.0}.get(k, 0.02)
                    W[k] = rand(s, keys[i], scale)
            # ln1 / ln2 = 1 so activations stay O(1)
            W["vec"] = W["vec"].at[:, :, 0:2, :].set(1.0)
        pools = new_pools(c)
        x = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(2), (B, c.H)).astype(jnp.bfloat16), rep
        )
        pos = jax.device_put(jnp.full((B,), args.ctx, jnp.int32), rep)
        pt = np.arange(1, 1 + B * max_pages, dtype=np.int32).reshape(B, max_pages)
        pt_f = jax.device_put(jnp.asarray(pt[::-1].copy()), rep)
        # SWA pool: page j of sequence b -> slot j % swa_ring of b's ring (window-distinct)
        ptw = (
            1
            + np.arange(B)[:, None] * swa_ring
            + np.arange(max_pages)[None, :] % swa_ring
        )
        pt_s = jax.device_put(jnp.asarray(ptw.astype(np.int32)), rep)
        for mode in args.modes:
            rec = dict(
                mode=mode,
                B=B,
                L=c.L,
                ctx=args.ctx,
                kv=c.kv_dtype,
                moe=c.moe_mode,
                scatter=not args.no_scatter,
                kv_write=args.kv_write,
                w=c.w_dtype,
                ablate=list(c.ablate),
                ar_fold=c.AR_FOLD,
                ar="rsag" if c.AR_RSAG else "oneshot",
                tne=c.tne,
                npb=c.npb,
                kv_bufs=c.kv_bufs,
            )
            try:
                if mode == "xla":
                    fn = xla_step_fn(c, mesh, MK, REF)
                else:
                    fn = MK.make_step(
                        c,
                        mesh,
                        xla_ar=(mode == "fallback"),
                        interpret=args.cpu_smoke,
                        scatter=not args.no_scatter,
                    )
                step = jax.jit(fn, donate_argnums=(5, 6))
                t0 = time.perf_counter()
                out, ps, pf = step(x, pos, pt_s, pt_f, W, *pools)
                out.block_until_ready()
                rec["first_call_s"] = round(time.perf_counter() - t0, 2)
                for _ in range(args.warmup):
                    out, ps, pf = step(x, pos, pt_s, pt_f, W, ps, pf)
                out.block_until_ready()
                ts = []
                for _ in range(args.iters):
                    t1 = time.perf_counter()
                    out, ps, pf = step(x, pos, pt_s, pt_f, W, ps, pf)
                    out.block_until_ready()
                    ts.append((time.perf_counter() - t1) * 1e3)
                ts = np.array(ts)
                rec.update(
                    ms_median=round(float(np.median(ts)), 3),
                    ms_p10=round(float(np.percentile(ts, 10)), 3),
                    ms_p90=round(float(np.percentile(ts, 90)), 3),
                    iters=len(ts),
                    finite=bool(jnp.all(jnp.isfinite(out.astype(jnp.float32)))),
                )
                # pipelined: back-to-back dispatch, one block -> hides host dispatch latency
                t1 = time.perf_counter()
                for _ in range(args.iters):
                    out, ps, pf = step(x, pos, pt_s, pt_f, W, ps, pf)
                out.block_until_ready()
                rec["ms_pipelined"] = round(
                    (time.perf_counter() - t1) * 1e3 / args.iters, 3
                )
                if args.steps_per_call > 1:
                    # device time: K steps inside one jitted call (fori over the step)
                    K = args.steps_per_call

                    def multi(x, pos, pt_s, pt_f, W, ps, pf):
                        def body(_, carry):
                            o, a, b_ = carry
                            o2, a, b_ = fn(x, pos, pt_s, pt_f, W, a, b_)
                            return o2, a, b_

                        return jax.lax.fori_loop(
                            0, K, body, (jnp.zeros_like(x), ps, pf)
                        )

                    mstep = jax.jit(multi, donate_argnums=(5, 6))
                    out, ps, pf = mstep(x, pos, pt_s, pt_f, W, ps, pf)
                    out.block_until_ready()
                    tk = []
                    for _ in range(max(3, args.iters // 4)):
                        t1 = time.perf_counter()
                        out, ps, pf = mstep(x, pos, pt_s, pt_f, W, ps, pf)
                        out.block_until_ready()
                        tk.append((time.perf_counter() - t1) * 1e3 / K)
                    rec["ms_device"] = round(float(np.median(tk)), 3)
                pools = [ps, pf]
            except Exception as e:  # noqa: BLE001
                rec["error"] = (type(e).__name__ + ": " + str(e))[:2000]
                pools = new_pools(c)
            print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
