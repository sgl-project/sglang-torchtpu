# SPDX-License-Identifier: Apache-2.0
"""TPU shifted mHC hooks and draft boundary state."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch
from weakref import ref

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
        hooks.tpu_dsv41_shifted_draft_forward(original, model, self.y, None, None, None)
        original.assert_called_once_with(model, self.y, None, None, None)

    def test_target_reads_live_sublayer_after_hook_binding(self):
        owner = torch.nn.Module()
        owner.ffn_hc = self.hc
        setattr(self.next.norm, tpu_mhc._OWNER, (ref(owner), "ffn_hc"))
        owner.ffn_hc = self.hc._replace(scale=torch.ones(3))
        device_rows = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), shape=(96, 5120)
        )
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

    def test_serving_skips_small_and_unmeasured_buckets(self):
        owner = torch.nn.Module()
        owner.ffn_hc = self.hc
        setattr(self.next.norm, tpu_mhc._OWNER, (ref(owner), "ffn_hc"))
        original = Mock()
        for tokens in (8, 40, 48, 64, 80, 8208):
            with self.subTest(tokens=tokens):
                device_rows = SimpleNamespace(
                    device=SimpleNamespace(type="tpu"), shape=(tokens, 5120)
                )
                with patch.object(tpu_mhc, "fused_boundary") as fused:
                    result = hooks.tpu_dsv41_shifted_mhc(
                        original,
                        self.hc,
                        device_rows,
                        self.residual,
                        self.coefficients,
                        self.next,
                    )
                self.assertIs(result, original.return_value)
                fused.assert_not_called()

    def test_draft_small_bucket_calls_original_before_embedding(self):
        model = SimpleNamespace(
            hc_pre_from_prev_sublayer=True, forward_embed=Mock(), _expand_hc=Mock()
        )
        ids = SimpleNamespace(device=SimpleNamespace(type="tpu"), shape=(40,))
        original = Mock()
        result = hooks.tpu_dsv41_shifted_draft_forward(
            original, model, ids, None, None, None
        )
        original.assert_called_once_with(model, ids, None, None, None)
        self.assertIs(result, original.return_value)
        model.forward_embed.assert_not_called()
        model._expand_hc.assert_not_called()

    def test_draft_forward_hook_uses_embedding_and_head_contract(self):
        input_ids = SimpleNamespace(device=SimpleNamespace(type="tpu"), shape=(240,))
        flat_embeds = SimpleNamespace(
            device=input_ids.device, shape=(240, 5120), dim=lambda: 2
        )
        expanded_embeds = SimpleNamespace(
            device=input_ids.device, shape=(240, 4, 5120), dim=lambda: 3
        )
        model = SimpleNamespace(
            stages=object(),
            hc_mult=2,
            forward_embed=Mock(return_value=self.residual),
            _expand_hc=Mock(return_value=self.residual),
        )
        original = Mock()
        recorder = Mock()
        recorder.disable_this_region.side_effect = nullcontext
        for embeds in (None, flat_embeds, expanded_embeds):
            with self.subTest(embeds=embeds):
                model.forward_embed.reset_mock()
                model._expand_hc.reset_mock()
                with (
                    patch.object(
                        tpu_mhc, "can_run_draft", return_value=True
                    ) as can_run,
                    patch.object(
                        tpu_mhc,
                        "run_draft_stages",
                        return_value=(self.residual, self.coefficients[0]),
                    ) as run_stages,
                    patch(
                        "sglang.srt.eplb.expert_distribution.get_global_expert_distribution_recorder",
                        return_value=recorder,
                    ),
                    patch(
                        "sglang.kernels.ops.layernorm.mhc.mhc_combine",
                        return_value=self.y,
                    ) as combine,
                ):
                    out = hooks.tpu_dsv41_shifted_draft_forward(
                        original, model, input_ids, None, None, embeds
                    )
                rows = input_ids if embeds is None else embeds
                can_run.assert_called_once_with(
                    model, rows, has_input_embeds=embeds is not None
                )
                stage_input = (
                    expanded_embeds if embeds is expanded_embeds else self.residual
                )
                self.assertIs(run_stages.call_args.args[2], stage_input)
                self.assertIs(out.hidden_states, self.y)
                self.assertIsNone(out.next_token_logits)
                torch.testing.assert_close(
                    combine.call_args.args[0], self.residual.flatten(1)
                )
                self.assertIs(combine.call_args.args[1], self.coefficients[0])
                self.assertEqual(combine.call_args.args[2:], (2, torch.bfloat16))
                self.assertEqual(model.forward_embed.call_count, int(embeds is None))
                self.assertEqual(
                    model._expand_hc.call_count, int(embeds is flat_embeds)
                )
        original.assert_not_called()
        self.assertEqual(recorder.disable_this_region.call_count, 3)

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
