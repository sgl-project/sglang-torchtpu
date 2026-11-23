# SPDX-License-Identifier: Apache-2.0
"""TPU shifted mHC hooks and draft boundary state."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.models.deepseek_v41 import hooks
from sglang_tpu.models.deepseek_v41.ops import mhc as tpu_mhc

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepSeekV4MhcBoundary(CustomTestCase):
    def setUp(self):
        self.hc = mhc.HcSubLayer(
            mhc.HcConfig(2, 20, 1e-6, 1e-6, 2, True, False),
            torch.empty(8, 4),
            torch.empty(3),
            torch.empty(8),
            torch.nn.Identity(),
        )
        self.y = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        self.residual = torch.tensor([[[3.0, 4.0], [5.0, 6.0]]], dtype=torch.bfloat16)
        self.coefficients = (
            torch.tensor([[0.25, 0.75]]),
            torch.tensor([[1.0, 0.5]]),
            torch.tensor([[[0.75, 0.25], [0.25, 0.75]]]),
        )
        self.next = mhc.HcNextBoundary(self.hc.norm, False, True, self.hc)
        self.enterContext(envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False))
        self.enterContext(envs.SGLANG_OPT_USE_FLASHINFER_MHC.override(False))

    def test_non_tpu_hooks_call_original(self):
        original = Mock()
        hooks.tpu_dsv41_shifted_mhc(
            original, self.hc, self.y, self.residual, self.coefficients, self.next
        )
        original.assert_called_once_with(
            self.hc, self.y, self.residual, self.coefficients, self.next
        )
        original.reset_mock()
        model = object()
        hooks.tpu_dsv41_shifted_draft_stages(original, model, None, self.y, None)
        original.assert_called_once_with(model, None, self.y, None)

    def test_target_reads_live_sublayer_after_hook_binding(self):
        from weakref import ref

        owner = torch.nn.Module()
        owner.ffn_hc = self.hc
        setattr(self.next.norm, tpu_mhc._OWNER, (ref(owner), "ffn_hc"))
        owner.ffn_hc = self.hc._replace(scale=torch.ones(3))
        device_rows = SimpleNamespace(device=SimpleNamespace(type="tpu"))
        original = Mock()
        with patch.object(
            tpu_mhc, "fused_boundary", side_effect=[None, object()]
        ) as fused:
            args = (self.hc, device_rows, self.residual, self.coefficients, self.next)
            self.assertIs(
                hooks.tpu_dsv41_shifted_mhc(original, *args), original.return_value
            )
            self.assertIsNot(
                hooks.tpu_dsv41_shifted_mhc(original, *args), original.return_value
            )
            self.assertIs(fused.call_args.args[-1], owner.ffn_hc)
            original.assert_called_once_with(*args)

    def test_draft_consumes_boundary_inputs_and_gates_across_stages(self):
        width = 5120
        norm = torch.nn.Identity()
        norm.weight = torch.ones(width, dtype=torch.bfloat16)
        norm.cast_x_before_out_mul = False
        norm.variance_size_override = None
        hc = self.hc._replace(cfg=self.hc.cfg._replace(hidden=width), norm=norm)
        initial = torch.full((1, 2, width), 2.0, dtype=torch.bfloat16)
        attn_values = []
        ffn_values = []

        def attention(positions, rows, batch):
            attn_values.append(rows[0, 0].item())
            return rows + 1

        def feed_forward(rows, batch):
            ffn_values.append(rows[0, 0].item())
            return rows + 2

        def finish(rows, residual, post, comb):
            return mhc.post(hc.cfg, rows, residual, post, comb)

        def make_stage():
            attn = Mock(side_effect=attention)
            attn.maybe_use_decode_attn_tp.return_value = nullcontext()
            return SimpleNamespace(
                attn_hc=hc,
                ffn_hc=hc,
                hc_cfg=hc.cfg,
                self_attn=attn,
                _run_ffn=feed_forward,
                hc_post=finish,
            )

        def complete(prev_hc, rows, residual, coefficients, boundary, next_hc):
            pre, post, comb = coefficients
            updated = finish(rows, residual, post, comb)
            return mhc.HcState(
                updated, pre, mhc.HcNormed(rows + 10), (pre + 1, post + 1, comb)
            )

        stage0, stage1 = make_stage(), make_stage()
        batch = SimpleNamespace(
            forward_mode=SimpleNamespace(
                is_decode=lambda: False, is_target_verify=lambda: False
            )
        )
        with (
            patch.object(tpu_mhc, "fused_boundary", side_effect=complete),
            patch.object(mhc, "mix_stats", return_value=self.coefficients) as mix,
        ):
            residual, previous_pre = tpu_mhc.run_draft_stages(
                (stage0, stage1), None, initial, batch
            )

        self.assertEqual(attn_values, [2.0, 25.0])
        self.assertEqual(ffn_values, [13.0, 36.0])
        self.assertEqual(mix.call_count, 1)
        pre, post, comb = self.coefficients
        torch.testing.assert_close(previous_pre, pre + 3)
        expected_residual = initial
        for value, post_offset in ((3, 0), (15, 1), (26, 2), (38, 3)):
            rows = torch.full((1, width), value, dtype=torch.bfloat16)
            expected_residual = finish(
                rows, expected_residual, post + post_offset, comb
            )
        torch.testing.assert_close(residual, expected_residual)


if __name__ == "__main__":
    unittest.main()
