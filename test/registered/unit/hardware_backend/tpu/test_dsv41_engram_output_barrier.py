# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402 -- production environment overrides must precede torch import.
"""Real Engram forward preserves image bypass through compiled TPU fencing.

The native closed-HC reproducer aborts on its BF16 where without the Engram
output fence. This component test checks the real registered hook, real gate,
real HC Sinkhorn/post functions, image selection, and replay values. Checkpoint
lookup/projection are bounded fixtures; no tokenizer, weights or server needed.
"""

import copy
import importlib.util
import os
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["SGLANG_PLATFORM"] = "tpu_oot"

from sglang_tpu.env_override import apply_tpu_env_overrides

apply_tpu_env_overrides()

import torch
from sglang.kernels.ops.layernorm.mhc import _hc_split_sinkhorn_torch
from sglang.srt.layers.engram import Engram
from sglang.srt.models.deepseek_v4 import DeepseekV4DecoderLayer
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.models.deepseek_v41.hooks import hook_specs

register_tpu_ci(
    est_time=240,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)


class _Lookup(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lookups = 0
        self.rows = torch.nn.Embedding(8, width, dtype=torch.bfloat16)
        with torch.no_grad():
            self.rows.weight.copy_(torch.randn_like(self.rows.weight) * 0.05)

    def forward(self, ids, forward_batch=None, *, cp_all_tokens=False):
        if not torch.compiler.is_compiling():
            self.lookups += 1
        return self.rows(ids[:, 0]).unsqueeze(1)


class _ProjectionBoundary(torch.nn.Module):
    def forward(self, rows):
        return rows, None


def _engram():
    module = Engram.__new__(Engram)
    torch.nn.Module.__init__(module)
    module.embed = _Lookup(5 * 5120)
    module.wkv = _ProjectionBoundary()
    module.q_weight = torch.nn.Parameter(torch.ones(4, 5120), requires_grad=False)
    module.k_weight = torch.nn.Parameter(torch.ones(4, 5120), requires_grad=False)
    module.eps = 1e-6
    module.clamp_value = 1e-6
    return module


class _Chain(torch.nn.Module):
    hc_mult = 4

    def __init__(self):
        super().__init__()
        self.engram = _engram()
        self.register_buffer("hc_fn", torch.randn(2, 24, 20480) * 0.002)
        self.register_buffer("hc_scale", torch.ones(3))
        self.register_buffer("hc_base", torch.zeros(24))

    def _weights(self, x, index):
        # hc_pre's distributed symmetric-allocation context needs a full runtime.
        # Retain its stat/linear DAG here, using the actual shared Sinkhorn op;
        # the assertions concern Engram's fence and image bypass, not HC math.
        flat = x.flatten(1).float()
        mixes = torch.nn.functional.linear(flat, self.hc_fn[index])
        mixes = mixes * torch.rsqrt(flat.square().mean(-1, keepdim=True) + 1e-6)
        pre, post, comb = _hc_split_sinkhorn_torch(
            mixes.unsqueeze(1), self.hc_scale, self.hc_base, 4, 20, 1e-6
        )
        return pre.squeeze(1), post.squeeze(1), comb.squeeze(1)

    def forward(self, x, values, hashes, input_ids):
        _, post, comb = self._weights(x, 0)
        before = DeepseekV4DecoderLayer.hc_post(self, values, x, post, comb)
        result = self.engram(before, hashes)
        selected = torch.where((input_ids == 129264)[:, None, None], before, result)
        pre, post, comb = self._weights(selected, 1)
        combined = (selected.float() * pre.unsqueeze(-1)).sum(1).bfloat16()
        output = DeepseekV4DecoderLayer.hc_post(self, combined, selected, post, comb)
        return before, result, selected, output


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires TorchTPU")
class TestEngramOutputBarrier(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import torch_tpu  # noqa: F401
        from sglang_tpu.patches import (
            _register_aten_fallbacks,
            _register_tpu_compile_backend,
        )
        from sglang.srt.plugins.hook_registry import HookRegistry

        _register_tpu_compile_backend()
        # Ordinary plugin initialization registers Engram's copysign.out path.
        _register_aten_fallbacks()
        target = "sglang.srt.layers.engram.Engram.forward"
        for spec in hook_specs():
            if spec[0] == target:
                HookRegistry.register(*spec)
        HookRegistry.apply_hooks()
        assert target in HookRegistry._patched

    def test_real_engram_image_bypass_and_replay(self):
        from torch._dynamo.backends.registry import lookup_backend

        from sglang.srt.environ import envs

        # Match the DeepseekV4ForCausalLM TPU override in hooks/dsv4.py: this
        # model uses the torch HC paths, not CUDA TileLang HC kernels.
        envs.SGLANG_OPT_USE_TILELANG_MHC_POST.set(False)
        envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.set(False)
        torch.manual_seed(17)
        torch.set_float32_matmul_precision("medium")
        cpu = _Chain().eval()
        # Idle ranks still execute the original lookup before returning x.
        empty = torch.empty(0, 4, 5120, dtype=torch.bfloat16)
        calls = cpu.engram.embed.lookups
        self.assertIs(cpu.engram(empty, torch.empty(0, 1, dtype=torch.int64)), empty)
        self.assertEqual(cpu.engram.embed.lookups, calls + 1)
        device = copy.deepcopy(cpu).to("tpu")
        traced = []
        backend = lookup_backend("tpu")

        def recording_backend(gm, inputs):
            traced.append(gm)
            return backend(gm, inputs)

        compiled = torch.compile(
            device, backend=recording_backend, fullgraph=True, dynamic=False
        )
        for tokens in (128, 1024):
            for replay in range(2):
                x = (torch.randn(tokens, 4, 5120) * 0.125).bfloat16()
                values = (torch.randn(tokens, 5120) * 0.125).bfloat16()
                hashes = (torch.arange(tokens) % 8).reshape(-1, 1)
                ids = torch.arange(tokens, dtype=torch.int64) + replay
                ids[::7] = 129264
                args = (x, values, hashes, ids)
                with torch.no_grad():
                    expected = cpu(*args)
                    actual = tuple(
                        t.cpu() for t in compiled(*(t.to("tpu") for t in args))
                    )
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(got, want, rtol=0.04, atol=0.015)
                image = ids == 129264
                self.assertTrue(torch.equal(actual[2][image], actual[0][image]))
                self.assertTrue(torch.equal(actual[2][~image], actual[1][~image]))
        # The fence must protect Engram's result consumed by where, not an
        # unrelated input. Its input is the real gate's final BF16 conversion.
        for gm in traced:
            barriers = [
                n for n in gm.graph.nodes if "optimization_barrier" in str(n.target)
            ]
            self.assertEqual(len(barriers), 1)
            gate = barriers[0].args[0][0]
            self.assertIn("to", str(gate.target))
            self.assertTrue(
                any(
                    "where" in str(user.target)
                    for node in barriers[0].users
                    for user in node.users
                )
            )


if __name__ == "__main__":
    unittest.main()
