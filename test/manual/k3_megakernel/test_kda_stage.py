"""K3 KDA stage on one host's 8 cores vs the eager sglang goldens (and the
numpy reference scripts/ref_kda.py semantics): new conv/recurrent states per
rank and the all-reduced module output."""

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
class TestKdaStage(CustomTestCase):
    def test_layers(self):
        import torch
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_megakernel.config import Relax, Topology
        from sglang_tpu.k3_megakernel.kernel.stages.kda import KdaStage
        from sglang_tpu.k3_megakernel.weights import (
            Checkpoint,
            apply_dense_fp8,
            load_kda_layer,
        )

        topo = Topology(
            *[int(v) for v in os.environ.get("K3MK_TOPO", "1,8").split(",")]
        )
        mesh = Mesh(np.array(_DEVICES[:8]), ("x",))
        sh = NamedSharding(mesh, P("x"))
        relax = Relax.parse(os.environ.get("K3MK_RELAX", "").split(","))
        stage = KdaStage(topo, relax=relax)
        fn = stage.build(mesh)
        exact_fn = KdaStage(topo).build(mesh) if relax != Relax() else None
        ckpt = Checkpoint(_CKPT)
        layers = [
            int(v) for v in os.environ.get("K3MK_LAYERS", "0,1,2,4,12").split(",")
        ]
        step = os.environ.get("K3MK_STEP", "step1")
        for layer in layers:
            gi = [
                torch.load(
                    os.path.join(_GOLDEN, f"rank{r}", step, f"L{layer:02d}.attn.in.pt"),
                    weights_only=False,
                )
                for r in range(8)
            ]
            go = [
                torch.load(
                    os.path.join(
                        _GOLDEN, f"rank{r}", step, f"L{layer:02d}.attn.out.pt"
                    ),
                    weights_only=False,
                )
                for r in range(8)
            ]
            w = load_kda_layer(ckpt, layer, topo, range(8), pair_oproj=relax.pair_oproj)
            x = gi[0]["hidden_states"].float().numpy().astype(ml_dtypes.bfloat16)
            conv = np.stack([g["state"]["conv"][0, :, 0, :].numpy() for g in gi])
            rec = np.stack([g["state"]["temporal"][0].numpy() for g in gi])
            put = lambda a: jax.device_put(np.ascontiguousarray(a), sh)
            args = [
                put(np.broadcast_to(x[None], (8,) + x.shape)),
                put(w["kda_fb"]),
                put(w["kda_conv"]),
                put(w["kda_dt"]),
                put(w["kda_alog"]),
                put(w["kda_onorm"]),
                put(conv),
                put(rec),
                put(w["kda_proj"]),
                put(w["kda_oproj"]),
            ]
            exact_out = None
            if exact_fn is not None:
                ex_args = args
                if relax.pair_oproj:
                    ex_args = args[:9] + [
                        put(load_kda_layer(ckpt, layer, topo, range(8))["kda_oproj"])
                    ]
                exact_out = np.asarray(exact_fn(*ex_args)[0])[0].astype(np.float32)
            if relax.dense_fp8:
                q = apply_dense_fp8({k: w[k] for k in ("kda_proj", "kda_oproj")})
                args = args[:8] + [
                    put(q[k])
                    for k in ("kda_proj", "kda_oproj", "kda_proj_s", "kda_oproj_s")
                ]
            out, new_conv, new_rec = (np.asarray(a) for a in fn(*args))
            g_conv = np.stack([g["state"]["conv"][0, :, 0, :].numpy() for g in go])
            g_rec = np.stack([g["state"]["temporal"][0].numpy() for g in go])
            gold = go[0]["out"].float().numpy()
            conv_eq = np.mean(new_conv == g_conv)
            rec_rel = np.linalg.norm(new_rec - g_rec) / np.linalg.norm(g_rec)
            mine = out[0].astype(np.float32)
            rel = np.linalg.norm(mine - gold) / np.linalg.norm(gold)
            ulp = np.abs(
                mine.astype(ml_dtypes.bfloat16).view(np.int16).astype(np.int32)
                - gold.astype(ml_dtypes.bfloat16).view(np.int16).astype(np.int32)
            )
            print(
                f"L{layer:02d}: conv bit-eq {conv_eq:.5f}  rec rel {rec_rel:.2e}  "
                f"out rel {rel:.2e}  bit-eq {np.mean(ulp == 0):.4f}  <=1ulp {np.mean(ulp <= 1):.4f}",
                flush=True,
            )
            if exact_out is not None:
                print(
                    f"   relax {relax}: out rel vs exact kernel {np.linalg.norm(mine - exact_out) / np.linalg.norm(exact_out):.2e}",
                    flush=True,
                )
            if os.environ.get("K3MK_SAVE"):
                np.save(
                    os.path.join(os.environ["K3MK_SAVE"], f"kda_L{layer}_{step}.npy"),
                    mine,
                )
            if exact_fn is None:
                self.assertLess(rec_rel, 1e-4)
        if os.environ.get("K3MK_BENCH"):
            times = {}
            for n in (1, 11):
                f = KdaStage(topo, relax=relax, repeat=n).build(mesh)
                jax.block_until_ready(f(*args))
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    jax.block_until_ready(f(*args))
                    ts.append(time.perf_counter() - t0)
                times[n] = min(ts)
            print(
                f"device time per KDA layer: {(times[11] - times[1]) / 10 * 1e6:.1f} us",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
