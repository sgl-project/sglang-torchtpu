"""Repro: the GPT-OSS MXFP4 fused-MoE kernel produces NaN under torch.compile
at the real decode shape, while eager is finite.

Background
----------
gpt-oss-20b decode under ``torch.compile(backend="tpu", fullgraph=True)`` emits
all-NaN logits (→ token 0), while eager decode is correct. Bisection (disabling
sliding-window, attention sinks, the cache, precompile; and a Qwen3 control)
localized the NaN to the fused-MoE path at the *decode* token shape, and only
under the compiled (XLA) execution — eager never NaNs. This test reproduces it
in isolation: run ``fused_moe_call`` eager vs ``torch.compile(backend="tpu")``
at the **real** gpt-oss-20b MoE dims (hidden=2880, intermediate=2880, E=32) and
at the decode token counts (m=1 and m=16), and assert the compiled output is
NaN-free (it currently is NOT).

Run:
  python -m pytest test/manual/tpu/test_fused_moe_compile_nan.py -x -s
"""

from __future__ import annotations

import importlib.util
import os
import unittest

import torch

_THIS = os.path.dirname(__file__)
_NATIVE_TEST = os.path.join(_THIS, "test_mxfp4_native_moe.py")
_spec = importlib.util.spec_from_file_location("_mxfp4_native", _NATIVE_TEST)
_mxfp4_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mxfp4_native)

try:
    import torch_tpu  # noqa: F401

    _HAS_TPU = True
except Exception:  # pragma: no cover
    _HAS_TPU = False


def _has_tpu_device() -> bool:
    if not _HAS_TPU:
        return False
    try:
        torch.empty(1, device="tpu")
        return True
    except Exception:
        return False


# Real gpt-oss-20b MoE dims.
E, HIDDEN, INTERMEDIATE, TOPK = 32, 2880, 2880, 4


@unittest.skipUnless(_has_tpu_device(), "requires torch_tpu + TPU")
class TestFusedMoeCompileNaN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from sglang.srt.hardware_backend.tpu.patches import apply_tpu_patches

            apply_tpu_patches()
        except Exception:
            pass

    def _weights(self):
        from sglang.srt.layers.quantization.mxfp4_tpu_util import (
            process_mxfp4_moe_weights_native,
        )

        layer = _mxfp4_native._build_synthetic_hf_gpt_oss_layer(
            E, HIDDEN, INTERMEDIATE, seed=0
        )
        w = process_mxfp4_moe_weights_native(
            layer["gate_up_proj_blocks"],
            layer["gate_up_proj_scales"],
            layer["gate_up_proj_bias"],
            layer["down_proj_blocks"],
            layer["down_proj_scales"],
            layer["down_proj_bias"],
            w13_interleaved=True,
        )
        return [t.to("tpu") for t in w]

    def _inputs(self, m):
        g = torch.Generator().manual_seed(1)
        hs = (torch.randn(m, HIDDEN, generator=g) * 0.5).to(torch.bfloat16).to("tpu")
        ti = torch.randint(0, E, (m, TOPK), generator=g, dtype=torch.int32).to("tpu")
        tw = (
            torch.softmax(torch.randn(m, TOPK, generator=g).float(), -1)
            .to(torch.bfloat16)
            .to("tpu")
        )
        return hs, ti, tw

    def _check(self, m):
        from sglang.srt.layers.moe.pallas.fused_moe_op import fused_moe_call

        w = self._weights()

        def call(hs, tw, ti):
            return fused_moe_call(
                hidden_states=hs,
                w1=w[0],
                w1_scale=w[1],
                w1_bias=w[2],
                w2=w[3],
                w2_scale=w[4],
                w2_bias=w[5],
                topk_weights=tw,
                topk_ids=ti,
                topk=TOPK,
                activation="swigluoai",
            )

        hs, ti, tw = self._inputs(m)
        eager = call(hs, tw, ti).cpu().float()
        compiled_fn = torch.compile(call, backend="tpu", fullgraph=True, dynamic=False)
        compiled = compiled_fn(hs, tw, ti).cpu().float()

        eager_nan = bool(eager.isnan().any())
        comp_nan = bool(compiled.isnan().any())
        print(
            f"[m={m}] eager_nan={eager_nan} compiled_nan={comp_nan} "
            f"compiled_nan_frac={compiled.isnan().float().mean().item():.3f}"
        )
        self.assertFalse(eager_nan, f"eager fused_moe produced NaN at m={m}")
        self.assertFalse(
            comp_nan,
            f"torch.compile(backend=tpu) fused_moe produced NaN at m={m} "
            f"(real gpt-oss dims hidden={HIDDEN} intermediate={INTERMEDIATE})",
        )

    def test_decode_m1(self):
        self._check(1)

    def test_decode_m16(self):
        self._check(16)

    def test_prefill_m64(self):
        self._check(64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
