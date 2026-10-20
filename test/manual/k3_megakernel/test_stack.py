"""K3 megakernel decoder stack (13-layer truncated model, TP8 on one host) vs
the eager sglang goldens of one decode step: per-layer outputs, the final
hidden state, KDA states and the new MLA latent rows. The kernel reduces in
f32, sglang in bf16, so outputs are compared by relative error; chained
references live in scripts/ref_stack.py."""

import os
import time
import unittest

import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase

_CKPT = os.environ.get("K3_CKPT", "/weights/Kimi-K3")
_GOLDEN = os.environ.get("K3MK_GOLDEN", "")
_OUT = os.environ.get("K3MK_SAVE", "")

try:
    import jax

    _DEVICES = [d for d in jax.devices() if d.platform == "tpu"]
except Exception:  # pragma: no cover
    _DEVICES = []


def _load(step, rank, name):
    import torch

    return torch.load(
        os.path.join(_GOLDEN, f"rank{rank}", step, name + ".pt"), weights_only=False
    )


@unittest.skipUnless(
    len(_DEVICES) >= 8 and os.path.isdir(_GOLDEN), "needs 8 TPU cores and K3MK_GOLDEN"
)
class TestStack(CustomTestCase):
    def test_13_layers(self):
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_megakernel.config import Relax, Topology
        from sglang_tpu.k3_megakernel.kernel.stack import StackKernel, StackSpec
        from sglang_tpu.k3_megakernel.weights import Checkpoint, load_stack

        step = os.environ.get("K3MK_STEP", "step1")
        relax = Relax.parse(os.environ.get("K3MK_RELAX", "").split(","))
        topo = Topology(
            *[int(v) for v in os.environ.get("K3MK_TOPO", "1,8").split(",")]
        )
        spec = StackSpec(topo, 13, (3, 7, 11), relax=relax)
        world, L = 8, spec.num_layers
        mesh = Mesh(np.array(_DEVICES[:world]), ("x",))
        sh = NamedSharding(mesh, P("x"))
        ckpt = Checkpoint(_CKPT)

        # Experts sglang chose plus those of the f32-reduction chained reference (scripts/ref_stack.py):
        # routing can legitimately differ at near-ties once the trajectories separate.
        ref = os.environ.get("K3MK_REF_STACK", "")
        ref = np.load(ref) if ref and os.path.exists(ref) else None
        experts = {}
        for i in range(1, L):
            ids = set()
            for s in ("step1", "step2"):
                ids |= set(
                    _load(s, 0, f"L{i:02d}.topk")["out"][1].numpy().reshape(-1).tolist()
                )
            if ref is not None:
                ids |= set(ref[f"moe_ids_{i}"].reshape(-1).tolist())
            experts[i] = sorted(ids)
        t0 = time.time()
        w = load_stack(
            ckpt,
            spec,
            range(world),
            experts,
            expert_cache=os.environ.get("K3MK_EXPERT_CACHE", ""),
        )
        print(f"weights loaded in {time.time() - t0:.0f}s", flush=True)

        st = _load(step, 0, f"L{spec.mla_layers[0]:02d}.attn.in")["state"]
        pages = st["page_indices"].numpy().astype(np.int32)
        kv_len = int(st["kv_lens"][0])
        cache_pages = int(pages.max()) + 1
        pages = np.concatenate([pages, np.zeros((-len(pages)) % 16, np.int32)])
        states = {}
        for j, layer in enumerate(spec.mla_layers):
            kv = (
                _load(step, 0, f"L{layer:02d}.attn.in")["state"]["kv_pages"]
                .float()
                .numpy()
            )
            cache = np.zeros((cache_pages, 64, 2, 640), ml_dtypes.bfloat16)
            for i, p in enumerate(st["page_indices"].numpy()):
                if p:
                    cache[p] = kv[i]
            states[f"mla_cache_{j}"] = np.broadcast_to(
                cache[None], (world,) + cache.shape
            )
        for k, layer in enumerate(spec.kda_layers):
            # One pool slot (row 0) per rank, holding the golden state before this step.
            states[f"kda_conv_{k}"] = np.stack(
                [
                    _load(step, r, f"L{layer:02d}.attn.in")["state"]["conv"].numpy()
                    for r in range(world)
                ]
            )
            states[f"kda_rec_{k}"] = np.stack(
                [
                    _load(step, r, f"L{layer:02d}.attn.in")["state"]["temporal"].numpy()
                    for r in range(world)
                ]
            )
        stacked = os.environ.get("K3MK_STACKED", "0") == "1"
        if stacked:
            nk, nm = len(spec.kda_layers), len(spec.mla_layers)
            states = {
                "kda_conv": np.stack(
                    [states[f"kda_conv_{k}"] for k in range(nk)], axis=1
                ),
                "kda_rec": np.stack(
                    [states[f"kda_rec_{k}"] for k in range(nk)], axis=1
                ),
                "mla_cache": np.stack(
                    [states[f"mla_cache_{j}"] for j in range(nm)], axis=1
                ),
            }
        x0 = (
            _load(step, 0, "L00.layer.in")["hidden_states"]
            .float()
            .numpy()
            .astype(ml_dtypes.bfloat16)
        )

        rep = lambda a: np.broadcast_to(
            np.asarray(a)[None], (world,) + np.asarray(a).shape
        )
        inputs = dict(
            tables=rep(np.asarray(spec.tables(), np.int32)),
            meta=rep(np.array([kv_len, 0], np.int32)),
            pages=rep(pages),
            x0=rep(x0),
            **states,
            **w,
        )
        kernel_kw = {}
        for item in filter(None, os.environ.get("K3MK_KERNEL_OPTS", "").split(";")):
            key, _, value = item.partition("=")
            kernel_kw[key] = value
        kernel = StackKernel(
            spec, stacked_states=stacked, layer_out=True, fused_head=False, **kernel_kw
        )
        names = kernel.input_names()
        shapes = {n: (inputs[n].shape[1:], inputs[n].dtype) for n in names}
        fn = kernel.build(
            mesh, shapes, alias_states=os.environ.get("K3MK_ALIAS", "1") == "1"
        )
        args = [jax.device_put(np.ascontiguousarray(inputs[n]), sh) for n in names]
        t0 = time.time()
        outs = fn(*args)
        jax.block_until_ready(outs)
        print(f"first call (compile + run) {time.time() - t0:.0f}s", flush=True)
        final, layer_out, ids = (np.asarray(o) for o in outs[:3])

        for r in range(1, world):
            np.testing.assert_array_equal(final[r], final[0], err_msg=f"rank {r}")
        rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
        for i in range(L):
            gold = _load(step, 0, f"L{i:02d}.layer.out")["out"][0].float().numpy()
            mine = layer_out[0, i].astype(np.float32)
            extra = ""
            if i >= 1:
                gids = set(
                    _load(step, 0, f"L{i:02d}.topk")["out"][1]
                    .numpy()
                    .reshape(-1)
                    .tolist()
                )
                mids = set(ids[0, i].tolist())
                extra = f"  routing {'same' if gids == mids else f'DIFFERS ({len(gids & mids)}/16)'}"
                missing = mids - set(experts[i])
                if missing:
                    extra += f"  NOT LOADED {sorted(missing)}"
            vs_ref = ""
            if ref is not None:
                rr = ref[f"layer_out_{i}"].reshape(1, -1)
                eq = np.mean(
                    mine.astype(ml_dtypes.bfloat16) == rr.astype(ml_dtypes.bfloat16)
                )
                vs_ref = f" | vs chained ref rel {rel(mine, rr):.2e} bit-eq {eq:.4f}"
                if i >= 1:
                    vs_ref += (
                        " routing==ref"
                        if set(ids[0, i].tolist())
                        == set(ref[f"moe_ids_{i}"].reshape(-1).tolist())
                        else " ROUTING!=ref"
                    )
            print(
                f"L{i:02d}: layer out rel vs sglang {rel(mine, gold):.2e}{extra}{vs_ref}",
                flush=True,
            )
        cmp_path = os.environ.get("K3MK_COMPARE", "")
        if cmp_path:
            # A saved run of another configuration (e.g. the exact kernel vs a relaxed one).
            other = np.load(cmp_path)
            for i in range(L):
                same = (
                    set(ids[0, i].tolist()) == set(other["ids"][i].tolist())
                    if i
                    else True
                )
                print(
                    f"L{i:02d}: rel vs {os.path.basename(cmp_path)} "
                    f"{rel(layer_out[0, i].astype(np.float32), other['layer_out'][i]):.2e}"
                    f"{'' if same else '  routing differs'}",
                    flush=True,
                )
            print(
                f"final: rel vs {os.path.basename(cmp_path)} {rel(final[0].astype(np.float32), other['final']):.2e}",
                flush=True,
            )
        gold = _load(step, 0, "model.out").float().numpy()
        fr = (
            ""
            if ref is None
            else f" | vs chained ref rel {rel(final[0].astype(np.float32), ref['model_out'].reshape(1, -1)):.2e}"
        )
        print(
            f"final: rel vs sglang {rel(final[0].astype(np.float32), gold):.2e}{fr}",
            flush=True,
        )
        if _OUT:
            np.savez(
                os.path.join(_OUT, f"stack_{step}{os.environ.get('K3MK_TAG', '')}.npz"),
                final=final[0].astype(np.float32),
                layer_out=layer_out[0].astype(np.float32),
                ids=ids[0],
            )
        # Pool writes: new KDA states and MLA rows, read back from the (possibly aliased) inputs.
        state_names = [
            n for n in names if n.startswith(("kda_conv", "kda_rec", "mla_cache"))
        ]
        new = outs[3:] if len(outs) > 3 else [args[names.index(n)] for n in state_names]
        new = {n: np.asarray(a) for n, a in zip(state_names, new)}
        if stacked:
            for fam, n in (("kda_conv", nk), ("kda_rec", nk), ("mla_cache", nm)):
                for i in range(n):
                    new[f"{fam}_{i}"] = new[fam][:, i]
        for k, layer in enumerate(spec.kda_layers):
            g = _load(step, 0, f"L{layer:02d}.attn.out")["state"]
            c_rel = rel(new[f"kda_conv_{k}"][0, 0], g["conv"][0].numpy())
            r_rel = rel(new[f"kda_rec_{k}"][0, 0], g["temporal"][0].numpy())
            print(
                f"KDA L{layer:02d} rank0 new state rel vs sglang: conv {c_rel:.2e} rec {r_rel:.2e}",
                flush=True,
            )
        for j, layer in enumerate(spec.mla_layers):
            t = kv_len - 1
            g = (
                _load(step, 0, f"L{layer:02d}.attn.out")["state"]["kv_pages"][t // 128]
                .float()
                .numpy()
            )
            mine = new[f"mla_cache_{j}"][0, pages[t // 128]].astype(np.float32)
            print(
                f"MLA L{layer:02d} new row rel vs sglang {rel(mine[(t % 128) // 2, t % 2], g[(t % 128) // 2, t % 2]):.2e}",
                flush=True,
            )

        if os.environ.get("K3MK_BENCH"):
            n = 20
            jax.block_until_ready(fn(*args))
            t0 = time.perf_counter()
            res = [fn(*args) for _ in range(n)]
            jax.block_until_ready(res)
            print(
                f"{n} back-to-back calls: {(time.perf_counter() - t0) / n * 1e3:.3f} ms/call",
                flush=True,
            )


if __name__ == "__main__":
    unittest.main()
