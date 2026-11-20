"""Compare the packed-FP4 gmm2 default tiles with the generic default.

Runs the TPU Pallas ``gmm_v2`` grouped matmul on one TPU core with raw JAX (no
server): ``lhs[m, K] (bf16, quantized to FP8 in kernel) @ rhs[g, K, N]`` with a
uint8-packed NVFP4 rhs (block 512), as the MoE down projection (gmm2) runs it.
For each ``m`` it pins two tiles:

* ``generic``: ``get_default_gmm_block_sizes`` (the default before the
  packed-FP4 rule);
* ``packed_fp4``: ``get_default_packed_fp4_gmm_block_sizes`` with gmm_v2's
  default VMEM limit (what ``get_tuned_block_sizes`` returns for packed FP4).

and reports, per tile: ms per call (chained calls, difference of two chain
lengths), and with ``--check`` whether the output is bitwise equal to the
generic tile's, and with ``--compile`` the lower and compile seconds.

TPU has no hardware cache between HBM and VMEM, so chained calls stream the
rhs from HBM each time; the timing is not cache-warm.

``m`` is the routed row count (chunk tokens x top-k). Rows are split over the
groups either evenly (``rr``) or with a fixed geometric skew (``skew``, the
heaviest group gets about 8x the lightest).

  python benchmark/kernels/pallas_fused_moe/bench_packed_fp4_gmm2_tiles.py \\
      --m 16384 32768 65536 --check --compile

``--tile-module PATH`` loads ``tuned_block_sizes.py`` from a file instead of the
installed package (to compare a candidate rule against an installed kernel).
"""

from __future__ import annotations

import os

# A raw-JAX TPU process: pick the physical backend before JAX is imported.
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import argparse  # noqa: E402
import importlib  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import numpy as np  # noqa: E402


def _bypass_pallas_package_init():
    """Let ``sglang_tpu.moe.pallas`` submodules import without the package
    ``__init__``, which imports torch_tpu through ``fused_moe_op``; this process
    must not start the torch_tpu runtime next to JAX's own TPU client."""
    for name in ("sglang_tpu.moe", "sglang_tpu.moe.pallas"):
        if name not in sys.modules:
            spec = importlib.util.find_spec(name)
            pkg = types.ModuleType(name)
            pkg.__path__ = list(spec.submodule_search_locations)
            sys.modules[name] = pkg


def group_sizes(m: int, groups: int, routing: str) -> np.ndarray:
    if routing == "rr":
        weights = np.ones(groups)
    elif routing == "skew":
        weights = np.geomspace(8.0, 1.0, groups)
    else:
        raise ValueError(f"unknown routing {routing!r}")
    sizes = np.floor(weights / weights.sum() * m).astype(np.int32)
    sizes[0] += m - int(sizes.sum())
    return sizes


def load_tile_module(path: str | None):
    if path is None:
        return importlib.import_module("sglang_tpu.moe.pallas.tuned_block_sizes")
    spec = importlib.util.spec_from_file_location("candidate_tuned_block_sizes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--m", type=int, nargs="+", default=[16384, 32768, 65536])
    p.add_argument("--k", type=int, default=4096, help="contraction (intermediate)")
    p.add_argument("--n", type=int, default=4096, help="output (hidden)")
    p.add_argument("--groups", type=int, default=128, help="local experts")
    p.add_argument("--routing", nargs="+", default=["rr", "skew"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--check", action="store_true", help="bitwise vs the generic tile")
    p.add_argument("--compile", action="store_true", help="lower/compile seconds")
    p.add_argument("--tile-module", default=None)
    args = p.parse_args()

    import jax
    import jax.numpy as jnp
    from jax import lax

    _bypass_pallas_package_init()
    gmm = importlib.import_module("sglang_tpu.moe.pallas.gmm_v2")
    TileSizes, default_vmem_limit_bytes = gmm.TileSizes, gmm.default_vmem_limit_bytes

    tiles_mod = load_tile_module(args.tile_module)
    k, n, g = args.k, args.n, args.groups
    dev = jax.local_devices()[0]
    print(f"device {dev} vmem_limit={default_vmem_limit_bytes()}", flush=True)
    with jax.default_device(dev):
        key = jax.random.PRNGKey(3)
        rhs = jax.random.randint(key, (g, k // 2, n), 0, 255, jnp.int32).astype(
            jnp.uint8
        )
        scale = jnp.full((g, k // 512, 1, n), 0.01, jnp.float32)
        for m in args.m:
            lhs = (jax.random.normal(key, (m, k), jnp.float32) * 0.5).astype(
                jnp.bfloat16
            )
            arms = {
                "generic": tiles_mod.get_default_gmm_block_sizes(m, k, n, g),
                "packed_fp4": tiles_mod.get_default_packed_fp4_gmm_block_sizes(
                    m, k, n, g, default_vmem_limit_bytes()
                ),
            }
            for routing in args.routing:
                gs = jnp.asarray(group_sizes(m, g, routing))
                ref = None
                for arm, (tm, tk, tn) in arms.items():
                    kw = dict(
                        tile_info=TileSizes(tile_m=tm, tile_k=tk, tile_n=tn),
                        preferred_element_type=jnp.bfloat16,
                        packed_fp4=True,
                        maybe_quantize_lhs=True,
                        zero_initialize=True,
                    )
                    tag = f"m={m} routing={routing} arm={arm} tile={tm}x{tk}x{tn}"

                    # The weights are jit arguments, not closed-over constants.
                    def one(x, w, w_scale, kw=kw, gs=gs):
                        return gmm.gmm_v2(
                            lhs=x, rhs=w, rhs_scale=w_scale, group_sizes=gs, **kw
                        )

                    if args.compile:
                        t0 = time.perf_counter()
                        lowered = jax.jit(one).lower(lhs, rhs, scale)
                        t1 = time.perf_counter()
                        lowered.compile()
                        t2 = time.perf_counter()
                        print(
                            f"COMPILE {tag} lower_s={t1 - t0:.2f} compile_s={t2 - t1:.2f}",
                            flush=True,
                        )
                    if args.check:
                        out = np.asarray(
                            jax.jit(one)(lhs, rhs, scale).astype(jnp.float32)
                        )
                        if ref is None:
                            ref = out
                        else:
                            diff = np.abs(out - ref)
                            print(
                                f"CHECK {tag} bitwise={bool((diff == 0).all())} max_abs={float(diff.max()):.3e}",
                                flush=True,
                            )

                    def chain(cnt, x, w, w_scale, one=one):
                        def body(_, x):
                            return x + one(x, w, w_scale)[:, :1] * jnp.bfloat16(0)

                        return lax.fori_loop(0, cnt, body, x)

                    f1 = jax.jit(lambda x, w, ws, c=chain: c(args.iters, x, w, ws))
                    f2 = jax.jit(lambda x, w, ws, c=chain: c(2 * args.iters, x, w, ws))
                    x = f2(f1(lhs, rhs, scale), rhs, scale)
                    x.block_until_ready()
                    t_short, t_long = [], []
                    for _ in range(args.reps):
                        s = time.perf_counter()
                        x = f1(x, rhs, scale)
                        x.block_until_ready()
                        t_short.append(time.perf_counter() - s)
                        s = time.perf_counter()
                        x = f2(x, rhs, scale)
                        x.block_until_ready()
                        t_long.append(time.perf_counter() - s)
                    ms = (min(t_long) - min(t_short)) / args.iters * 1e3
                    print(f"TIME {tag} ms={ms:.3f}", flush=True)


if __name__ == "__main__":
    main()
