"""K3 MoE block (megakernel stage) on one host's 8 cores vs goldens captured
from sglang's eager TP8 path (13-layer real-weight model, decode bs=1).

Set K3MK_GOLDEN to a capture root (k3cap plugin output). Routing must match
exactly; the block output is compared in bf16 ulps.
"""

import os
import time
import unittest

import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase

_CKPT = os.environ.get("K3_CKPT", "/weights/Kimi-K3")
_GOLDEN = os.environ.get("K3MK_GOLDEN", "")

try:
    import jax

    _DEVICES = [d for d in jax.devices() if d.platform == "tpu"]
except Exception:  # pragma: no cover
    _DEVICES = []


def _ulp(a, b):
    ai = np.asarray(a, ml_dtypes.bfloat16).view(np.int16).astype(np.int32)
    bi = np.asarray(b, ml_dtypes.bfloat16).view(np.int16).astype(np.int32)
    return np.abs(ai - bi)


@unittest.skipUnless(
    len(_DEVICES) >= 8 and os.path.isdir(_GOLDEN), "needs 8 TPU cores and K3MK_GOLDEN"
)
class TestMoeStage(CustomTestCase):
    def test_layers_vs_sglang(self):
        import torch
        from jax.sharding import Mesh

        from sglang_tpu.k3_megakernel.config import (
            K3Dims,
            MoeLayout,
            Relax,
            Topology,
        )
        from sglang_tpu.k3_megakernel.kernel.stages.moe import MoeStage
        from sglang_tpu.k3_megakernel.weights import (
            Checkpoint,
            apply_dense_fp8,
            load_moe_layer,
        )

        topo = Topology(
            *[int(v) for v in os.environ.get("K3MK_TOPO", "1,8").split(",")]
        )
        layout = MoeLayout(K3Dims(), topo)
        ckpt = Checkpoint(_CKPT)
        mesh = Mesh(np.array(_DEVICES[: topo.world]), ("x",))
        relax = Relax.parse(os.environ.get("K3MK_RELAX", "").split(","))
        # K3MK_MOE_OPTS, e.g. "early_dma=1,pair_down=1,split_refill=1": schedule options, which
        # must reproduce the default kernel bit for bit.
        opts = {
            k: int(v)
            for k, _, v in (
                o.partition("=")
                for o in os.environ.get("K3MK_MOE_OPTS", "").split(",")
                if o
            )
        }
        opts = {k: (bool(v) if k != "ring" else v) for k, v in opts.items()}
        fn = MoeStage(layout, rows=1, relax=relax, **opts).build(mesh)
        exact_fn = (
            MoeStage(layout, rows=1).build(mesh) if relax != Relax() or opts else None
        )
        step = os.path.join(_GOLDEN, "rank0", os.environ.get("K3MK_STEP", "step1"))
        layers = [int(v) for v in os.environ.get("K3MK_LAYERS", "1,2,5,12").split(",")]
        names = (
            "router_bias",
            "latent_norm",
            "router",
            "latent_down",
            "shared_gu",
            "shared_down",
            "latent_up",
            "expert_gu",
            "expert_gu_scale",
            "expert_down",
            "expert_down_scale",
        )
        for layer in layers:
            g_in = torch.load(
                os.path.join(step, f"L{layer:02d}.mlp.in.pt"), weights_only=False
            )
            g_out = torch.load(
                os.path.join(step, f"L{layer:02d}.mlp.out.pt"), weights_only=False
            )["out"]
            g_topk = torch.load(
                os.path.join(step, f"L{layer:02d}.topk.pt"), weights_only=False
            )["out"]
            gold_ids = g_topk[1].numpy()[0]
            t0 = time.time()
            w = load_moe_layer(
                ckpt, layer, layout, ranks=range(topo.world), experts=gold_ids
            )
            load_s = time.time() - t0
            x = g_in["arg0"].float().numpy().astype(ml_dtypes.bfloat16)
            xs = np.broadcast_to(x[None], (topo.world,) + x.shape)
            from jax.sharding import NamedSharding
            from jax.sharding import PartitionSpec as P

            sharding = NamedSharding(mesh, P("x"))
            args = [
                jax.device_put(a, sharding)
                for a in [np.ascontiguousarray(xs)] + [w[k] for k in names]
            ]
            exact_out = exact_ids = None
            if exact_fn is not None:
                eo, ei, _ = exact_fn(*args)
                exact_out, exact_ids = (
                    np.asarray(eo)[0].astype(np.float32),
                    np.asarray(ei)[0],
                )
            if relax.dense_fp8:
                dense = (
                    "router",
                    "latent_down",
                    "shared_gu",
                    "shared_down",
                    "latent_up",
                )
                q = apply_dense_fp8({k: w[k] for k in dense})
                args = (
                    args[:3]
                    + [jax.device_put(q[k], sharding) for k in dense]
                    + args[8:]
                    + [jax.device_put(q[k + "_s"], sharding) for k in dense]
                )
            out, ids, wts = fn(*args)
            out, ids, wts = np.asarray(out), np.asarray(ids), np.asarray(wts)
            if os.environ.get("K3MK_SAVE"):
                np.save(
                    os.path.join(os.environ["K3MK_SAVE"], f"moe_L{layer}.npy"),
                    out[0].astype(np.float32),
                )
            for r in range(1, topo.world):
                np.testing.assert_array_equal(
                    out[r], out[0], err_msg=f"rank {r} differs"
                )
            if exact_out is not None and relax == Relax():
                np.testing.assert_array_equal(
                    out[0].astype(np.float32), exact_out, err_msg=f"L{layer} {opts}"
                )
                print(
                    f"   options {opts}: bit-identical to the default kernel",
                    flush=True,
                )
            elif exact_out is not None:
                same = sorted(ids[0].tolist()) == sorted(exact_ids.tolist())
                print(
                    f"   relax {relax}: out rel vs exact kernel "
                    f"{np.linalg.norm(out[0].astype(np.float32) - exact_out) / np.linalg.norm(exact_out):.2e}"
                    f"  routing {'same' if same else 'DIFFERS'} as exact",
                    flush=True,
                )
            else:
                self.assertEqual(
                    sorted(ids[0].tolist()),
                    sorted(gold_ids.tolist()),
                    f"L{layer} routing",
                )
            order = np.argsort(ids[0]), np.argsort(gold_ids)
            wdiff = np.abs(wts[0][order[0]] - g_topk[0].numpy()[0][order[1]]).max()
            gold = g_out.float().numpy()
            prefix = g_in["prefix_sum"].float().numpy() if "prefix_sum" in g_in else 0.0
            mine = out[0].astype(np.float32)
            with_prefix = (mine + prefix).astype(ml_dtypes.bfloat16)
            u0, u1 = _ulp(mine, gold), _ulp(with_prefix, gold)
            print(
                f"L{layer}: load {load_s:.1f}s  weights maxdiff {wdiff:.2e}  "
                f"ulp(out vs gold) max {u0.max()} mean {u0.mean():.3f} | "
                f"ulp(out+prefix vs gold) max {u1.max()} mean {u1.mean():.3f} >1: {(u1 > 1).mean():.4f}",
                flush=True,
            )

        if os.environ.get("K3MK_BENCH"):
            jax.block_until_ready(fn(*args))
            for n in (1, 50):
                t0 = time.perf_counter()
                outs = [fn(*args) for _ in range(n)]
                jax.block_until_ready(outs)
                print(
                    f"{n} back-to-back calls: {(time.perf_counter() - t0) / n * 1e6:.0f} us/call",
                    flush=True,
                )
            times = {}
            for rep in (1, 21):
                fr = MoeStage(layout, rows=1, relax=relax, repeat=rep).build(mesh)
                jax.block_until_ready(fr(*args))
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    jax.block_until_ready(fr(*args))
                    ts.append(time.perf_counter() - t0)
                times[rep] = min(ts)
            print(
                f"device time per MoE block (in-kernel repeat): {(times[21] - times[1]) / 20 * 1e6:.1f} us",
                flush=True,
            )

    def test_topk_tie_fallback(self):
        """Two experts with identical router columns and bias tie on every input; the fast
        top-k must fall back to the reference selection (lowest id first)."""
        import torch
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_megakernel.config import (
            K3Dims,
            MoeLayout,
            Topology,
        )
        from sglang_tpu.k3_megakernel.kernel.stages.moe import MoeStage
        from sglang_tpu.k3_megakernel.weights import (
            Checkpoint,
            load_moe_layer,
        )

        topo = Topology(1, 8)
        layout = MoeLayout(K3Dims(), topo)
        mesh = Mesh(np.array(_DEVICES[:8]), ("x",))
        sh = NamedSharding(mesh, P("x"))
        step = os.path.join(_GOLDEN, "rank0", "step1")
        g_in = torch.load(os.path.join(step, "L02.mlp.in.pt"), weights_only=False)
        gold_ids = torch.load(os.path.join(step, "L02.topk.pt"), weights_only=False)[
            "out"
        ][1].numpy()[0]
        w = load_moe_layer(Checkpoint(_CKPT), 2, layout, ranks=range(8), experts=())
        # Copy the 5th pick's router column and bias onto an expert of the other router half.
        src = int(gold_ids[4])
        rc, rp = layout.router_cols, layout.router_pad
        dst = (src + rc) % (2 * rc)
        sh_, sc_ = src // rc, src % rc
        dh, dc = dst // rc, dst % rc
        router, bias = w["router"].copy(), w["router_bias"].copy()
        for r in range(8):
            if (r & 1) == dh:
                router[r][:, dc] = w["router"][r ^ (sh_ != dh)][:, sc_]
            bias[r][0, dh * rp + dc] = w["router_bias"][r][0, sh_ * rp + sc_]
        li, eh = layout.lane_inter, layout.experts_per_host
        zeros = [
            np.zeros((8, eh, 3584, 2 * li), ml_dtypes.float8_e4m3fn),
            np.zeros((8, eh, 1, 2 * li), np.float32),
            np.zeros((8, eh, li // 2, 3584), np.uint8),
            np.zeros((8, eh, li // 32, 3584), np.uint8),
        ]
        x = g_in["arg0"].float().numpy().astype(ml_dtypes.bfloat16)
        args = [
            np.ascontiguousarray(np.broadcast_to(x[None], (8,) + x.shape)),
            bias,
            w["latent_norm"],
            router,
            w["latent_down"],
            w["shared_gu"],
            w["shared_down"],
            w["latent_up"],
        ] + zeros
        args = [jax.device_put(a, sh) for a in args]
        res = {}
        for fast in (True, False):
            _, ids, wts = MoeStage(layout, rows=1, fast_topk=fast).build(mesh)(*args)
            res[fast] = (np.asarray(ids)[0], np.asarray(wts)[0])
        print(f"tie: src {src} dst {dst}; fast ids {res[True][0].tolist()}", flush=True)
        self.assertIn(min(src, dst), res[True][0].tolist())
        np.testing.assert_array_equal(res[True][0], res[False][0])
        np.testing.assert_array_equal(res[True][1], res[False][1])


if __name__ == "__main__":
    unittest.main()
