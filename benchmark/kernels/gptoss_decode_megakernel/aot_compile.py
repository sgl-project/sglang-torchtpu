"""Ahead-of-time compile check of the GPT-OSS TP8 decode megakernel for tpu7x:2x2x1 on CPU.

Needs `jax[tpu]` (libtpu) installed but no TPU: the Mosaic/XLA TPU compiler runs on the host.
Prints one JSON line per (B, variant) with compile success, compile time, the kernel's VMEM
scratch estimate and XLA's memory analysis.

  python benchmark/kernels/gptoss_decode_megakernel/aot_compile.py --batch 1 8 --kv fp8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types


def import_kernel():
    """Import this tree's kernel package with stub parent packages (skips sglang's heavy
    package __init__ chain, so only jax is needed)."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "python")
    )
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.hardware_backend",
        "sglang.srt.hardware_backend.tpu",
        "sglang.srt.hardware_backend.tpu.kernels",
    ):
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(root, *name.split("."))]
        sys.modules[name] = m
    from sglang.srt.hardware_backend.tpu.kernels.gptoss_decode_megakernel import (
        config,
        megakernel,
    )

    return config, megakernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--kv", default="fp8", choices=["bf16", "fp8"])
    ap.add_argument("--moe", default="tp", choices=["tp", "ep"])
    ap.add_argument("--tne", type=int, default=384)
    ap.add_argument("--npb", type=int, default=8)
    ap.add_argument(
        "--nb", type=int, default=4096, help="pages per pool (per rank, per layer)"
    )
    ap.add_argument("--max-pages", type=int, default=128)
    ap.add_argument("--vmem-mib", type=int, default=60)
    ap.add_argument(
        "--xla-ar", action="store_true", help="compile the XLA-collective fallback"
    )
    ap.add_argument("--mxfp4", action="store_true", help="MXFP4 expert weights")
    ap.add_argument("--w-dtype", choices=["bf16", "mxfp4", "fp8"], default=None)
    ap.add_argument("--kv-write", choices=["scatter", "kernel"], default="scatter")
    ap.add_argument("--ar-mode", choices=["auto", "oneshot", "rsag"], default="auto")
    ap.add_argument("--ablate", nargs="*", default=[])
    ap.add_argument("--kv-bufs", type=int, default=3)
    ap.add_argument(
        "--no-scatter", action="store_true", help="leave out the KV scatter call"
    )
    ap.add_argument("--topology", default="tpu7x:2x2x1")
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    config, MK = import_kernel()
    topo = topologies.get_topology_desc(topology_name=args.topology, platform="tpu")
    mesh = Mesh(np.array(topo.devices[:8]), ("tp",))
    for B in args.batch:
        c = config.full(
            B,
            L=args.layers,
            kv_dtype=args.kv,
            moe_mode=args.moe,
            tne=args.tne,
            npb=args.npb,
            nb_full=args.nb,
            nb_swa=args.nb,
            max_pages=args.max_pages,
            vmem_mib=args.vmem_mib,
            kv_bufs=args.kv_bufs,
            w_dtype=args.w_dtype or ("mxfp4" if args.mxfp4 else "fp8"),
            ar_mode=args.ar_mode,
            ablate=tuple(args.ablate),
            kv_write=args.kv_write,
        )
        rec = dict(
            mode="aot_" + args.topology,
            B=B,
            L=c.L,
            kv=c.kv_dtype,
            moe=c.moe_mode,
            w=c.w_dtype,
            max_pages=c.max_pages,
            kv_write=c.kv_write,
            ar="rsag" if c.AR_RSAG else "oneshot",
            ablate=args.ablate,
            ar_fold=c.AR_FOLD,
            tne=c.tne,
            npb=c.npb,
            Hp=c.Hp,
            IL=c.IL,
            xla_ar=args.xla_ar,
            scatter=not args.no_scatter,
        )
        vm = MK.vmem_bytes(c, "mega")
        rec["vmem_scratch_mib"] = round(vm["total"] / 2**20, 2)
        rec["vmem_top"] = {
            k: round(v / 2**20, 2)
            for k, v in sorted(vm.items(), key=lambda kv: -kv[1])[1:7]
        }
        rep = NamedSharding(mesh, P())
        shd = NamedSharding(mesh, P("tp"))
        sds = jax.ShapeDtypeStruct
        W = {
            k: sds(v.shape, v.dtype, sharding=shd)
            for k, v in MK.packed_specs(c).items()
        }
        pools = tuple(sds(p.shape, p.dtype, sharding=shd) for p in MK.pool_specs(c))
        a = (
            sds((B, c.H), jnp.bfloat16, sharding=rep),
            sds((B,), jnp.int32, sharding=rep),
            sds((B, c.max_pages), jnp.int32, sharding=rep),
            sds((B, c.max_pages), jnp.int32, sharding=rep),
            W,
            *pools,
        )
        try:
            t0 = time.perf_counter()
            step = MK.make_step(
                c, mesh, xla_ar=args.xla_ar, scatter=not args.no_scatter
            )
            lowered = jax.jit(step, donate_argnums=(5, 6)).lower(*a)
            rec["lower_s"] = round(time.perf_counter() - t0, 2)
            t1 = time.perf_counter()
            compiled = lowered.compile()
            rec["compile_s"] = round(time.perf_counter() - t1, 2)
            rec["ok"] = True
            try:
                ma = compiled.memory_analysis()
                rec["hbm_temp_mib"] = round(ma.temp_size_in_bytes / 2**20, 2)
                rec["hbm_args_gib"] = round(ma.argument_size_in_bytes / 2**30, 2)
                rec["hbm_alias_gib"] = round(ma.alias_size_in_bytes / 2**30, 2)
            except Exception as e:  # noqa: BLE001
                rec["memory_analysis"] = repr(e)[:200]
            try:
                hlo = compiled.as_text()
            except Exception as e:  # noqa: BLE001  (seen with XLA async collectives)
                hlo = None
                rec["as_text_error"] = repr(e)[:200]
            if hlo is not None:
                copies = [
                    ln.strip()[:160]
                    for ln in hlo.splitlines()
                    if " copy(" in ln or " copy-start(" in ln
                ]
                rec["hlo_copy_ops"] = len(copies)
                # shapes of the copies: a pool-sized copy here would mean the in-place KV write
                # is not in place
                rec["hlo_copies"] = [
                    ln.split("=")[1].split(" copy")[0].strip() for ln in copies
                ]
                rec["hlo_custom_calls"] = hlo.count("custom-call(")
                rec["hlo_all_reduce"] = hlo.count("all-reduce(") + hlo.count(
                    "all-reduce-start("
                )
        except Exception as e:  # noqa: BLE001
            rec["ok"] = False
            rec["error"] = (type(e).__name__ + ": " + str(e))[:3000]
            traceback.print_exc(file=sys.stderr)
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
