"""K3 MLA stage on one host's 8 cores vs the eager sglang goldens: the new
latent row written into the paged cache and the all-reduced module output."""

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
class TestMlaStage(CustomTestCase):
    def test_layers(self):
        import torch
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_megakernel.config import Relax, Topology
        from sglang_tpu.k3_megakernel.kernel.stages.mla import MlaStage
        from sglang_tpu.k3_megakernel.weights import (
            Checkpoint,
            apply_dense_fp8,
            load_mla_layer,
        )

        topo = Topology(1, 8)
        mesh = Mesh(np.array(_DEVICES[:8]), ("x",))
        sh = NamedSharding(mesh, P("x"))
        put = lambda a: jax.device_put(np.ascontiguousarray(a), sh)
        relax = Relax.parse(os.environ.get("K3MK_RELAX", "").split(","))
        stage = MlaStage(topo, relax=relax)
        exact_stage = MlaStage(topo) if relax != Relax() else None
        ckpt = Checkpoint(_CKPT)
        step = os.environ.get("K3MK_STEP", "step1")
        for layer in [
            int(v) for v in os.environ.get("K3MK_LAYERS", "3,7,11").split(",")
        ]:
            gi = torch.load(
                os.path.join(_GOLDEN, "rank0", step, f"L{layer:02d}.attn.in.pt"),
                weights_only=False,
            )
            go = torch.load(
                os.path.join(_GOLDEN, "rank0", step, f"L{layer:02d}.attn.out.pt"),
                weights_only=False,
            )
            st, st_out = gi["state"], go["state"]
            pages = st["page_indices"].numpy().astype(np.int32)
            kv_len = int(st["kv_lens"][0])
            n_pool = int(pages.max()) + 2
            pool = np.zeros((n_pool, 64, 2, 640), ml_dtypes.bfloat16)
            for i, p in enumerate(pages):
                if p:
                    pool[p] = st["kv_pages"][i].float().numpy()
            # Pages past the request's own are padding; give every block a full 16-entry window.
            pad = (-len(pages)) % 16
            pages_p = np.concatenate([pages, np.zeros(pad, np.int32)])
            w = load_mla_layer(ckpt, layer, topo, range(8), pair_oproj=relax.pair_oproj)
            x = gi["hidden_states"].float().numpy().astype(ml_dtypes.bfloat16)
            rep = lambda a: put(
                np.broadcast_to(np.asarray(a)[None], (8,) + np.asarray(a).shape)
            )
            args = [
                rep(np.array([kv_len], np.int32)),
                rep(pages_p),
                rep(x),
                put(w["mla_qnorm"]),
                put(w["mla_kvnorm"]),
                put(w["mla_wkc"]),
                put(w["mla_wvc"]),
                put(w["mla_w1"]),
                put(w["mla_qb"]),
                put(w["mla_oproj"]),
                rep(pool),
            ]
            exact_out = None
            if exact_stage is not None:
                ex_args = args
                if relax.pair_oproj:
                    ex_args = (
                        args[:9]
                        + [
                            put(
                                load_mla_layer(ckpt, layer, topo, range(8))["mla_oproj"]
                            )
                        ]
                        + args[10:]
                    )
                exact_out = np.asarray(
                    exact_stage.build(mesh, pool.shape)(*ex_args)[0]
                )[0].astype(np.float32)
            if relax.dense_fp8:
                q = apply_dense_fp8(
                    {k: w[k] for k in ("mla_w1", "mla_qb", "mla_oproj")}
                )
                args = (
                    args[:7]
                    + [
                        put(q[k])
                        for k in (
                            "mla_w1",
                            "mla_qb",
                            "mla_oproj",
                            "mla_w1_s",
                            "mla_qb_s",
                            "mla_oproj_s",
                        )
                    ]
                    + args[10:]
                )
            fn = stage.build(mesh, pool.shape)
            out, cache = (np.asarray(a) for a in fn(*args))
            t = kv_len - 1
            pg = pages[t // 128]
            new_row = cache[0, pg, (t % 128) // 2, t % 2].astype(np.float32)
            gold_page = st_out["kv_pages"][t // 128].float().numpy()
            gold_row = gold_page[(t % 128) // 2, t % 2]
            other = cache[0, pg].astype(np.float32)
            untouched = np.array_equal(
                np.delete(other.reshape(128, 640), t, 0),
                np.delete(gold_page.reshape(128, 640), t, 0),
            )
            d = np.abs(new_row - gold_row)
            print(
                f"   row diff: max {d.max():.3e} nz {np.count_nonzero(d)} | first 4 mine {new_row[:4]} gold {gold_row[:4]} "
                f"| kpe mine {new_row[512:516]} gold {gold_row[512:516]} | other slot gold {gold_page[(t % 128) // 2, 1 - t % 2][:4]}"
                f" | mine other slot {cache[0, pg, (t % 128) // 2, 1 - t % 2][:4].astype(np.float32)}",
                flush=True,
            )
            gold = go["out"].float().numpy()
            mine = out[0].astype(np.float32)
            rel = np.linalg.norm(mine - gold) / np.linalg.norm(gold)
            eq = np.mean(
                mine.astype(ml_dtypes.bfloat16) == gold.astype(ml_dtypes.bfloat16)
            )
            print(
                f"L{layer:02d}: new row bit-equal {np.array_equal(new_row, gold_row)} (rest of page unchanged: "
                f"{untouched})  out rel {rel:.2e} bit-eq {eq:.4f}",
                flush=True,
            )
            if exact_out is not None:
                print(
                    f"   relax {relax}: out rel vs exact kernel {np.linalg.norm(mine - exact_out) / np.linalg.norm(exact_out):.2e}",
                    flush=True,
                )
            if os.environ.get("K3MK_SAVE"):
                np.save(
                    os.path.join(os.environ["K3MK_SAVE"], f"mla_L{layer}_{step}.npy"),
                    mine,
                )
        if os.environ.get("K3MK_BENCH"):
            times = {}
            for n in (1, 11):
                f = MlaStage(topo, relax=relax, repeat=n).build(mesh, pool.shape)
                jax.block_until_ready(f(*args))
                ts = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    jax.block_until_ready(f(*args))
                    ts.append(time.perf_counter() - t0)
                times[n] = min(ts)
            print(
                f"device time per MLA layer (kv_len {kv_len}): {(times[11] - times[1]) / 10 * 1e6:.1f} us",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
