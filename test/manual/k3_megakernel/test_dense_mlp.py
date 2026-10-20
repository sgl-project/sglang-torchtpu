"""K3 layer-0 dense MLP stage on one host's 8 cores vs the eager sglang golden
(L00.mlp.in -> L00.mlp.out, which includes the residual P2)."""

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


@unittest.skipUnless(
    len(_DEVICES) >= 8 and os.path.isdir(_GOLDEN), "needs 8 TPU cores and K3MK_GOLDEN"
)
class TestDenseMlp(CustomTestCase):
    def test_layer0(self):
        import torch
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_megakernel.config import K3Dims, Relax, Topology
        from sglang_tpu.k3_megakernel.kernel.stages.dense_mlp import DenseMlpStage
        from sglang_tpu.k3_megakernel.weights import (
            Checkpoint,
            apply_dense_fp8,
            load_dense_mlp,
        )

        topo = Topology(
            *[int(v) for v in os.environ.get("K3MK_TOPO", "1,8").split(",")]
        )
        dims = K3Dims()
        mesh = Mesh(np.array(_DEVICES[: topo.world]), ("x",))
        relax = Relax.parse(os.environ.get("K3MK_RELAX", "").split(","))
        stage = DenseMlpStage(dims, topo, rows=1, relax=relax)
        w = load_dense_mlp(Checkpoint(_CKPT), topo, stage.channels, range(topo.world))
        step = os.path.join(_GOLDEN, "rank0", "step1")
        g_in = torch.load(os.path.join(step, "L00.mlp.in.pt"), weights_only=False)
        gold = (
            torch.load(os.path.join(step, "L00.mlp.out.pt"), weights_only=False)["out"]
            .float()
            .numpy()
        )
        x = g_in["arg0"].float().numpy().astype(ml_dtypes.bfloat16)
        p2 = g_in["prefix_sum"].float().numpy().astype(ml_dtypes.bfloat16)
        sh = NamedSharding(mesh, P("x"))
        rep = lambda a: jax.device_put(
            np.ascontiguousarray(np.broadcast_to(a[None], (topo.world,) + a.shape)), sh
        )
        args = [
            rep(x),
            rep(p2),
            jax.device_put(w["dense_gu"], sh),
            jax.device_put(w["dense_down"], sh),
        ]
        exact_out = None
        if relax != Relax():
            exact_out = np.asarray(
                DenseMlpStage(dims, topo, rows=1).build(mesh)(*args)[0]
            )[0].astype(np.float32)
        if relax.dense_fp8:
            q = apply_dense_fp8(dict(w))
            args = args[:2] + [
                jax.device_put(q[k], sh)
                for k in ("dense_gu", "dense_down", "dense_gu_s", "dense_down_s")
            ]
        (out,) = stage.build(mesh)(*args)
        out = np.asarray(out)
        for r in range(1, topo.world):
            np.testing.assert_array_equal(out[r], out[0])
        mine = out[0].astype(np.float32)
        rel = np.linalg.norm(mine - gold) / np.linalg.norm(gold)
        eq = np.mean(mine.astype(ml_dtypes.bfloat16) == gold.astype(ml_dtypes.bfloat16))
        print(f"L00 dense: rel vs sglang {rel:.2e}, bit-equal {eq:.4f}", flush=True)
        if exact_out is not None:
            print(
                f"   relax {relax}: rel vs exact kernel {np.linalg.norm(mine - exact_out) / np.linalg.norm(exact_out):.2e}",
                flush=True,
            )
        else:
            self.assertLess(rel, 1e-2)
        if os.environ.get("K3MK_SAVE"):
            np.save(os.path.join(os.environ["K3MK_SAVE"], "dense_L0.npy"), mine)
        if os.environ.get("K3MK_BENCH"):
            times = {}
            for n in (1, 11):
                f = DenseMlpStage(dims, topo, rows=1, relax=relax, repeat=n).build(mesh)
                jax.block_until_ready(f(*args))
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    jax.block_until_ready(f(*args))
                    ts.append(time.perf_counter() - t0)
                times[n] = min(ts)
            print(
                f"device time per dense MLP: {(times[11] - times[1]) / 10 * 1e6:.1f} us",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
