# SPDX-License-Identifier: Apache-2.0
"""The optional shifted-mHC boundary preserves the built-in post fallback."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakePlatform(SRTPlatform):
    _enum = PlatformEnum.OOT

    def get_dispatch_key_name(self):
        return "test_mhc"


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
        self.fused = Mock(return_value=None)
        self.enterContext(patch.dict(mhc._FUSED_HC_BOUNDARIES, clear=True))
        self.enterContext(envs.SGLANG_OPT_USE_TILELANG_MHC_POST.override(False))
        self.enterContext(envs.SGLANG_OPT_USE_FLASHINFER_MHC.override(False))

    def _post(self, boundary):
        return mhc._post_fusion(
            self.hc, self.y, self.residual, self.coefficients, boundary
        )

    def _assert_post_fallback(self, result):
        expected = torch.tensor([[[4.5, 6.5], [5.0, 6.5]]], dtype=torch.bfloat16)
        torch.testing.assert_close(
            result.residual.view(torch.uint8), expected.view(torch.uint8)
        )
        self.assertIs(result.pre, self.coefficients[0])
        self.assertIsNone(result.input)
        self.assertIsNone(result.stats)

    def test_unregistered_uses_original_post(self):
        self._assert_post_fallback(self._post(self.next))

    def test_in_tree_ignores_registration(self):
        mhc.register_fused_hc_boundary("native", self.fused)
        with patch("sglang.srt.platforms._current_platform", SRTPlatform()):
            self._assert_post_fallback(self._post(self.next))
        self.fused.assert_not_called()

    def test_oot_can_accept_or_decline_the_boundary(self):
        mhc.register_fused_hc_boundary("test_mhc", self.fused)
        next_hc = self.hc._replace(scale=torch.ones(3))
        boundary = self.next._replace(get_hc=lambda: next_hc)
        fused_state = mhc.HcState(self.residual, input=mhc.HcNormed(self.y))
        self.fused.side_effect = [None, fused_state]

        with patch("sglang.srt.platforms._current_platform", _FakePlatform()):
            self._assert_post_fallback(self._post(boundary))
            self.assertIs(self._post(boundary), fused_state)
        self.assertEqual(self.fused.call_count, 2)
        self.assertIs(self.fused.call_args.args[-1], next_hc)

    def test_oot_skips_missing_or_incompatible_boundary(self):
        mhc.register_fused_hc_boundary("test_mhc", self.fused)
        with patch("sglang.srt.platforms._current_platform", _FakePlatform()):
            for boundary in (
                None,
                self.next._replace(hc=None),
                self.next._replace(norm_fusable=False),
            ):
                self._assert_post_fallback(self._post(boundary))
        self.fused.assert_not_called()

    def test_draft_consumes_boundary_inputs_and_gates_across_stages(self):
        from sglang.srt.models.deepseek_v4_platform_stage import NonCudaDSparkMixin

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
        mhc.register_fused_hc_boundary("test_mhc", complete)
        run = NonCudaDSparkMixin._forward_hc_pre_from_prev
        with (
            patch("sglang.srt.platforms._current_platform", _FakePlatform()),
            patch.object(mhc, "mix_stats", return_value=self.coefficients) as mix,
        ):
            state = run(stage0, None, mhc.HcState(initial), batch, stage1)
            state = run(stage1, None, state, batch)

        self.assertEqual(attn_values, [2.0, 25.0])
        self.assertEqual(ffn_values, [13.0, 36.0])
        self.assertEqual(mix.call_count, 1)
        self.assertIsNone(state.stats)
        self.assertIsNone(state.input)
        pre, post, comb = self.coefficients
        torch.testing.assert_close(state.pre, pre + 3)
        expected_residual = initial
        for value, post_offset in ((3, 0), (15, 1), (26, 2), (38, 3)):
            rows = torch.full((1, width), value, dtype=torch.bfloat16)
            expected_residual = finish(
                rows, expected_residual, post + post_offset, comb
            )
        torch.testing.assert_close(state.residual, expected_residual)


if __name__ == "__main__":
    unittest.main()
