"""K3 megakernel rows-mode prefill (StackKernel(verify_rows=R, verify_mode="rows",
prefill_blocks=B): a chunk of up to B * R tokens as R-row blocks through every stage) vs the
decode kernel called once per token on the same pools (13-layer truncated model, TP8 on one
host). Rows mode reorders f32 sums (M=R dots, grouped experts, row-scattered reduces), so
outputs and states are compared by relative error; MLA rows and KDA states past the prompt
must not be written into the request's slot (padding rows go to the null slot 0).

The prompt goes in as two chunks (fresh request with reset, then a continuation without),
and the KDA slot starts with garbage that the reset must ignore. Routed experts are zero
(the loop is what is tested, and a real expert set does not fit the host budget of a test)."""

import os
import time
import unittest

import ml_dtypes
import numpy as np

from sglang.test.test_utils import CustomTestCase

_CKPT = os.environ.get("K3_CKPT", "/weights/Kimi-K3")

try:
    import jax
    import jax.numpy as jnp

    _DEVICES = [d for d in jax.devices() if d.platform == "tpu"]
except Exception:  # pragma: no cover
    _DEVICES = []


@unittest.skipUnless(
    len(_DEVICES) >= 8 and os.path.isdir(_CKPT),
    "needs 8 TPU cores and the K3 checkpoint",
)
class TestStackPrefillRows(CustomTestCase):
    def test_chunks_match_decode_steps(self):
        from jax.sharding import Mesh, NamedSharding
        from jax.sharding import PartitionSpec as P

        from sglang_tpu.k3_megakernel.config import (
            MoeLayout,
            Relax,
            Topology,
        )
        from sglang_tpu.k3_megakernel.kernel.stack import (
            StackKernel,
            StackSpec,
            input_names,
        )
        from sglang_tpu.k3_megakernel.weights import (
            Checkpoint,
            _expert_family_shapes,
            load_stack,
        )

        relax = Relax.parse(os.environ.get("K3MK_RELAX", "").split(","))
        spec = StackSpec(Topology(1, 8), 13, (3, 7, 11), relax=relax)
        world, rows, blocks, slot = 8, 8, 2, 1
        chunks = [int(c) for c in os.environ.get("K3MK_CHUNKS", "11,5").split(",")]
        tokens = sum(chunks)
        mesh = Mesh(np.array(_DEVICES[:world]), ("x",))
        sh = NamedSharding(mesh, P("x"))
        ckpt = Checkpoint(_CKPT)
        t0 = time.time()
        w = load_stack(ckpt, spec, range(world), routed_experts=False)
        w = {n: jax.device_put(np.ascontiguousarray(a), sh) for n, a in w.items()}
        layout = MoeLayout(spec.dims, spec.topo, replicas=relax.expert_replicas)
        for n, (shape, dtype) in _expert_family_shapes(
            layout, world, spec.num_layers - 1
        ).items():
            dt = jnp.float8_e4m3fn if n == "expert_gu" else dtype
            w[n] = jax.jit(lambda s=shape, d=dt: jnp.zeros(s, d), out_shardings=sh)()
        print(f"weights loaded in {time.time() - t0:.0f}s", flush=True)

        rng = np.random.default_rng(0)
        ids = rng.integers(0, 150000, size=tokens)
        emb = np.asarray(ckpt.get("embed_tokens.weight")[np.sort(ids)]).astype(
            ml_dtypes.bfloat16
        )
        pages = np.zeros(16, np.int32)
        pages[0] = 1  # page 0 stays the null page

        opts = {}
        for item in filter(None, os.environ.get("K3MK_KERNEL_OPTS", "").split(";")):
            key, _, value = item.partition("=")
            opts[key] = value
        # The oracle feeds x0 and reads the final row itself: no fused embedding / head.
        decode_k = StackKernel(spec, **dict(opts, fused_head=False))
        loop_k = StackKernel(
            spec, verify_rows=rows, verify_mode="rows", prefill_blocks=blocks, **opts
        )
        kd = decode_k.kda
        rep = lambda a: np.broadcast_to(
            np.asarray(a)[None], (world,) + np.asarray(a).shape
        )

        def pools(garbage: bool):
            st = {}
            for k in range(len(spec.kda_layers)):
                conv = np.zeros((world, 2, 3, 1, 3 * kd.hd), np.float32)
                rec = np.zeros((world, 2, kd.heads, 128, 128), np.float32)
                if garbage:
                    conv[:, slot] = rng.standard_normal(conv[:, slot].shape)
                    rec[:, slot] = rng.standard_normal(rec[:, slot].shape)
                st[f"kda_conv_{k}"], st[f"kda_rec_{k}"] = conv, rec
            for j in range(len(spec.mla_layers)):
                st[f"mla_cache_{j}"] = np.zeros(
                    (world, 2, 64, 2, 640), ml_dtypes.bfloat16
                )
            return {n: jax.device_put(a, sh) for n, a in st.items()}

        names = input_names(spec)
        state_names = [
            n for n in names if n.startswith(("kda_conv_", "kda_rec_", "mla_cache_"))
        ]
        fns = {}
        consts = dict(
            tables=jax.device_put(rep(np.asarray(spec.tables(), np.int32)), sh),
            pages=jax.device_put(rep(pages), sh),
        )

        def run(kernel, meta, x0, st):
            inputs = dict(
                meta=jax.device_put(rep(np.asarray(meta, np.int32)), sh),
                x0=jax.device_put(rep(x0), sh),
                **consts,
                **st,
                **w,
            )
            if id(kernel) not in fns:
                shapes = {n: (inputs[n].shape[1:], inputs[n].dtype) for n in names}
                fns[id(kernel)] = kernel.build(mesh, shapes, alias_states=True)
            outs = fns[id(kernel)](*[inputs[n] for n in names])
            jax.block_until_ready(outs)
            return np.asarray(outs[0]), dict(zip(state_names, outs[3:]))

        t0 = time.time()
        st, loop_out, done = pools(garbage=True), [], 0
        for i, n in enumerate(chunks):
            x = np.zeros((blocks * rows, spec.dims.hidden), ml_dtypes.bfloat16)
            x[:n] = emb[done : done + n]
            x = x.reshape(blocks, rows, -1)
            # [kv_len of row 0, read row, write base, rows, write stride, reset]
            out, st = run(loop_k, [done + 1, slot, slot, n, 0, int(i == 0)], x, st)
            loop_out.append(out.reshape(world, blocks * rows, -1)[:, :n])
            done += n
            print(f"chunk {i} ({n} rows) call {time.time() - t0:.0f}s", flush=True)
        loop_out = np.concatenate(loop_out, axis=1)
        loop_st = {n: np.asarray(a) for n, a in st.items()}

        st = pools(garbage=False)
        dec_out = []
        for t in range(tokens):
            out, st = run(decode_k, [t + 1, slot, slot, 1, 0, 0], emb[t : t + 1], st)
            dec_out.append(out)
        dec_out = np.concatenate(dec_out, axis=1)

        rel = lambda a, b: float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
        for r in range(1, world):
            np.testing.assert_array_equal(loop_out[r], loop_out[0], err_msg=f"rank {r}")
        worst = 0.0
        for t in range(tokens):
            e = rel(loop_out[0, t].astype(np.float32), dec_out[0, t].astype(np.float32))
            worst = max(worst, e)
            print(f"token {t}: rel {e:.2e}", flush=True)
        for n in state_names:
            a, b = loop_st[n].astype(np.float32), np.asarray(st[n]).astype(np.float32)
            if n.startswith(
                "kda_"
            ):  # the request's slot; slot 0 (null) takes the padding rows
                e = rel(a[:, slot], b[:, slot])
            else:  # MLA rows of the prompt (page 1, tokens 0..tokens-1)
                e = rel(
                    a[:, 1].reshape(world, 128, 640)[:, :tokens],
                    b[:, 1].reshape(world, 128, 640)[:, :tokens],
                )
            print(f"{n}: rel {e:.2e}", flush=True)
            worst = max(worst, e)
        self.assertLess(worst, float(os.environ.get("K3MK_TOL", "5e-2")))
        print(f"rows prefill vs decode steps: worst rel {worst:.2e}", flush=True)


if __name__ == "__main__":
    unittest.main()
