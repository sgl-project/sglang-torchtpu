# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Wan2.2 megakernel hook (sglang_tpu.multimodal_gen.wan_megakernel.hook).

The hook's request state (cross-attention keys and values cached a request, the per-request op-call
log line and the step constants) with the kernel and torch replaced by stand-ins.
Skipped where JAX is unavailable.
"""

import os
import sys
import types
import unittest
import unittest.mock

# Host-CPU tests: in the TPU image the diffusion platform and JAX would
# otherwise select the TPU.
os.environ.setdefault("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=60,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

try:
    import jax.numpy  # noqa: F401

    from sglang_tpu.multimodal_gen.wan_megakernel import reference as ref
    from sglang_tpu.multimodal_gen.wan_megakernel.tiling import Tiling

    _HAS_JAX = True
except Exception:  # pragma: no cover - depends on JAX availability
    _HAS_JAX = False

_SKIP = "Requires JAX with Pallas TPU interpret mode"


def envs_mod():
    from sglang_tpu.envs import envs

    return envs


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestCrossKvCache(CustomTestCase):
    """hook._cross_kv: the text side of cross-attention, cached a request."""

    def setUp(self):
        import torch

        from sglang_tpu.multimodal_gen.wan_megakernel import hook

        self.hook, self.torch, self.calls, self.ctxs = hook, torch, [], {}
        self.model = types.SimpleNamespace(
            condition_embedder=types.SimpleNamespace(
                text_embedder=lambda t: (self.calls.append(t), t)[1]
            )
        )
        self.state = types.SimpleNamespace(
            cfg=None, tiling=Tiling(), kv_cache=[], kv_cache_size=2
        )
        self._pack = hook.pack.cross_kv, hook.pack.cross_keys
        hook.pack.cross_kv = lambda model, ctx, cfg: (ctx.float().sum(), ctx.shape)
        hook.pack.cross_keys = lambda k2, v2, counts: (counts, "weights")

    def tearDown(self):
        self.hook.pack.cross_kv, self.hook.pack.cross_keys = self._pack

    def kv(self, text, request="r", b=0, batch=1):
        ctx = self.ctxs.setdefault(request, types.SimpleNamespace(request_id=request))
        return self.hook._cross_kv(self.model, self.state, text, b, batch, ctx)

    def test_hits_only_the_same_live_tensor(self):
        pos, neg = self.torch.ones(1, 4, 8), self.torch.zeros(1, 4, 8)
        for _ in range(3):  # one rank serving both CFG branches, step after step
            self.assertEqual(float(self.kv(pos)[0]), 32.0)
            self.assertEqual(float(self.kv(neg)[0]), 0.0)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.kv(pos)[2:], ((64, 128, 256), "weights"))  # its keys
        del pos  # a new tensor (possibly at the freed address) never hits the old entry
        self.assertEqual(float(self.kv(self.torch.full((1, 4, 8), 2.0))[0]), 64.0)
        self.assertEqual(len(self.calls), 3)

    def test_new_request_and_batch_samples(self):
        text = self.torch.arange(16.0).reshape(2, 1, 8)
        a = [float(self.kv(text, b=b, batch=2)[0]) for b in range(2)]
        self.assertEqual(a, [28.0, 92.0])
        self.kv(text, b=1, batch=2)
        self.assertEqual(len(self.calls), 2)
        self.kv(text, request="other", b=1, batch=2)
        self.assertEqual(len(self.calls), 3)

    def test_no_request_id_no_cache(self):
        text = self.torch.ones(1, 4, 8)
        self.kv(text, request=None)
        self.kv(text, request=None)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.state.kv_cache, [])


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestOpCallLog(CustomTestCase):
    """hook._count_op_call: one log line a request and rank once every step ran the kernel."""

    def setUp(self):
        import torch

        from sglang_tpu.multimodal_gen.wan_megakernel import hook

        self.hook, self.torch = hook, torch
        hook._OP_CALLS.update(rid=None, n=0)
        self.state = types.SimpleNamespace()

    def ctx(self, rid, steps=40, warmup=False):
        return types.SimpleNamespace(
            request_id=rid, timesteps=self.torch.zeros(steps), is_warmup=warmup
        )

    def calls(self, ctx, n, batch=1):
        with self.assertLogs(self.hook.logger, level="INFO") as cm:
            self.hook.logger.info("start")
            for _ in range(n):
                self.hook._count_op_call(self.state, ctx, batch)
        return [
            r.getMessage() for r in cm.records if "block-stack op" in r.getMessage()
        ]

    def test_one_line_at_the_full_count(self):
        a = self.ctx("a")
        self.assertEqual(self.calls(a, 39), [])
        self.assertEqual(
            self.calls(a, 1),
            ["Wan megakernel: request a ran the block-stack op 40 times"],
        )

    def test_new_request_restarts(self):
        self.calls(self.ctx("a"), 25)  # a short request never logs a full count
        b = self.ctx("b")
        self.assertEqual(self.calls(b, 39), [])  # b counts from zero, not from 25
        self.assertEqual(
            self.calls(b, 1),
            ["Wan megakernel: request b ran the block-stack op 40 times"],
        )

    def test_samples_and_warmup(self):
        self.assertEqual(
            self.calls(self.ctx("w", steps=2, warmup=True), 4, batch=2),
            ["Wan megakernel: request w ran the block-stack op 4 times (warmup)"],
        )


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestStepConstants(CustomTestCase):
    """hook._forward_one with SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_STEP_CONSTANTS: the op
    gets the same inputs bit for bit and the model the same output, while the
    real-row count goes to the device once a shape without waiting instead of by a
    blocking copy every step."""

    H, C, L = 128, 16, 2

    def setUp(self):
        import torch
        from torch import nn

        import sglang.multimodal_gen.runtime.distributed as sgl_dist
        from sglang_tpu.multimodal_gen.wan_megakernel import hook

        self.hook, self.torch, self.sgl_dist = hook, torch, sgl_dist
        H, C, L = self.H, self.C, self.L
        g = torch.Generator().manual_seed(0)

        def rand(*shape):
            return torch.randn(*shape, generator=g)

        bf = torch.bfloat16
        conv = nn.Conv3d(C, H, kernel_size=(1, 2, 2), stride=(1, 2, 2)).to(bf)
        lin_t, lin_m, proj = (nn.Linear(H, n).to(bf) for n in (H, 6 * H, 4 * C))
        for m in (conv, lin_t, lin_m, proj):
            for prm in m.parameters():
                prm.data.copy_(rand(*prm.shape) * 0.05)
        freqs = torch.arange(H, dtype=torch.float32) * 0.01
        wrope = torch.arange(64, dtype=torch.float32) * 0.1
        norm = nn.LayerNorm(H, elementwise_affine=False, eps=1e-6)

        def norm_out(x, shift, scale):
            return (norm(x.float()) * (1 + scale) + shift).to(bf)

        def rope(pos):
            a = pos.float().sum(1, keepdim=True) * wrope
            return torch.cos(a), torch.sin(a)

        self.model = types.SimpleNamespace(
            patch_size=(1, 2, 2),
            patch_embedding=conv,
            condition_embedder=types.SimpleNamespace(
                time_embedder=lambda t, _: lin_t(
                    torch.sin(t[:, None].float() * freqs).to(bf)
                ),
                time_modulation=lambda temb: lin_m(torch.nn.functional.silu(temb)),
                text_embedder=lambda t: t,
            ),
            rotary_emb=types.SimpleNamespace(forward_uncached=rope),
            scale_shift_table=rand(1, 2, H) * 0.1,
            norm_out=norm_out,
            proj_out=lambda x: (proj(x), None),
        )
        self.cfg = ref.Config(layers=L, hidden=H, heads=1, head_dim=128, text_len=8)
        self.tables = rand(L, 6, H)
        k2 = rand(L, 1, 8, 128).to(bf)
        self._saved = (
            hook.pack.cross_kv,
            hook.pack.cross_keys,
            hook._op_for,
            sgl_dist.sequence_model_parallel_all_gather,
        )
        hook.pack.cross_kv = lambda model, ctx, cfg: (k2, k2 + 1)
        hook.pack.cross_keys = lambda a, b, counts: (
            torch.full((1,), 8, dtype=torch.int32),
            torch.ones(8, 1),
        )
        hook._op_for = lambda cfg, cfg_ranks, sp, tiling: self.op
        sgl_dist.sequence_model_parallel_all_gather = lambda x, dim: x

    def tearDown(self):
        (
            self.hook.pack.cross_kv,
            self.hook.pack.cross_keys,
            self.hook._op_for,
            self.sgl_dist.sequence_model_parallel_all_gather,
        ) = self._saved

    def op(self, xs, cos, sin, e, k2, v2, keys, key_weights, valid, *weights):
        self.seen.append([xs, cos, sin, e, k2, v2, keys, key_weights, valid])
        # Every input the kernel reads moves the output: a wrong one shows.
        mix = e[0, 0] + cos[:, :1] - sin[:, :1] + valid.float() * 1e-3
        return (xs.float() * 0.5 + mix).to(self.torch.bfloat16)

    def run_steps(self, enabled):
        """Three steps each of two latent shapes; returns the outputs, the op's
        inputs and the blocking (``device=``) and plain ``torch.tensor`` calls."""
        torch = self.torch
        flag = envs_mod().SGLANG_TPU_ENABLE_WAN_MEGAKERNEL_STEP_CONSTANTS
        with flag.override(enabled):
            weights = {name: torch.zeros(1) for name in self.hook.pack.WEIGHT_ORDER}
            state = self.hook._Megakernel(
                None, self.cfg, Tiling(), weights, None, self.tables, 1, 1
            )
        self.seen, outs, calls = [], [], []
        real = torch.tensor

        def counted(*args, **kwargs):
            calls.append("device" in kwargs)
            return real(*args, **kwargs)

        ctx = types.SimpleNamespace(
            request_id="r", timesteps=torch.zeros(3), negative_prompt_embeds=None
        )
        text = torch.ones(1, 8, self.H, dtype=torch.bfloat16)
        g = torch.Generator().manual_seed(1)
        with unittest.mock.patch.object(torch, "tensor", counted):
            for frames, h, w in ((3, 8, 12), (2, 16, 8)):
                for step in range(3):
                    lat = torch.randn(1, self.C, frames, h, w, generator=g)
                    t = torch.full((1,), 999.0 - 10 * step)
                    outs.append(
                        self.hook._forward_one(
                            self.model,
                            state,
                            lat.to(torch.bfloat16),
                            text,
                            0,
                            1,
                            t,
                            ctx,
                            1,
                            0,
                        )
                    )
        return outs, self.seen, calls

    def test_same_inputs_and_outputs(self):
        outs0, seen0, calls0 = self.run_steps(False)
        outs1, seen1, calls1 = self.run_steps(True)
        self.assertEqual(len(seen0), 6)
        for a, b in zip(outs0, outs1):
            self.assertTrue(self.torch.equal(a, b))
        for step0, step1 in zip(seen0, seen1):
            for a, b in zip(step0, step1):
                self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape))
                self.assertTrue(self.torch.equal(a, b))
        # Off: one blocking upload a step. On: one host tensor a shape, no blocking.
        self.assertEqual(calls0, [True] * 6)
        self.assertEqual(calls1, [False] * 2)
        valids = [step[-1] for step in seen1]
        self.assertTrue(all(v is valids[0] for v in valids[:3]))
        self.assertTrue(all(v is valids[3] for v in valids[3:]))
        self.assertIsNot(valids[0], valids[3])
        self.assertEqual([int(v[0]) for v in (valids[0], valids[3])], [72, 64])


@unittest.skipUnless(_HAS_JAX, _SKIP)
class TestRowEmbed(TestStepConstants):
    """hook._forward_one with SGLANG_TPU_OPT_WAN_MEGAKERNEL_ROW_EMBED: sglang's
    PatchEmbed (flatten off, as Wan builds it) gives its rows as the linear makes
    them and each rank pads only its own: the op gets the same rows bit for bit on
    every rank, whether its rows are all real, partly past the sequence's end or
    all padding."""

    def setUp(self):
        super().setUp()
        import torch

        from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed

        conv = self.model.patch_embedding
        pe = PatchEmbed(
            patch_size=(1, 2, 2), in_chans=self.C, embed_dim=self.H, flatten=False
        ).to(torch.bfloat16)
        pe.proj.weight.data.copy_(conv.weight.data)
        pe.proj.bias.data.copy_(conv.bias.data)
        self.model.patch_embedding = pe

    def forward(self, row_embed, sp, rank, lat):
        torch = self.torch
        with envs_mod().SGLANG_TPU_OPT_WAN_MEGAKERNEL_ROW_EMBED.override(row_embed):
            weights = {name: torch.zeros(1) for name in self.hook.pack.WEIGHT_ORDER}
            state = self.hook._Megakernel(
                None, self.cfg, Tiling(), weights, None, self.tables, 1, sp
            )
        self.sgl_dist.sequence_model_parallel_all_gather = lambda x, dim: torch.cat(
            [x] * sp, dim=dim
        )
        ctx = types.SimpleNamespace(
            request_id="r", timesteps=torch.zeros(3), negative_prompt_embeds=None
        )
        text = torch.ones(1, 8, self.H, dtype=torch.bfloat16)
        self.seen = []
        out = self.hook._forward_one(
            self.model,
            state,
            lat,
            text,
            0,
            1,
            torch.full((1,), 999.0),
            ctx,
            sp,
            rank,
        )
        return out, self.seen[0]

    def test_same_rows_every_rank(self):
        torch = self.torch
        g = torch.Generator().manual_seed(2)
        # 3 x 10 x 50 = 1,500 tokens: 1,024 rows a device at 2 and 4 devices (rank
        # 1 holds 476 real rows, ranks 2-3 none), 2,048 at 1 device.
        lat = torch.randn(1, self.C, 3, 20, 100, generator=g).to(torch.bfloat16)
        for sp in (1, 2, 4):
            for rank in range(sp):
                out0, seen0 = self.forward(False, sp, rank, lat)
                out1, seen1 = self.forward(True, sp, rank, lat)
                with self.subTest(sp=sp, rank=rank):
                    self.assertTrue(torch.equal(out0, out1))
                    for a, b in zip(seen0, seen1):
                        self.assertEqual((a.dtype, a.shape), (b.dtype, b.shape))
                        self.assertTrue(torch.equal(a, b))
                    real = max(
                        0, min(1500 - rank * seen1[0].shape[0], seen1[0].shape[0])
                    )
                    self.assertFalse(seen1[0][real:].any())


if __name__ == "__main__":
    unittest.main()
