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

    def test_partial_load_initializes_missing_and_refreshes_only_touched_buffers(self):
        layer = torch.nn.Module()
        layer.hc_cfg = SimpleNamespace(
            pre_from_prev=True, cp_prefill=False, mult=4, hidden=5120
        )
        layer.hc_attn_fn = SimpleNamespace(
            device=SimpleNamespace(type="tpu"), dtype=torch.float32, shape=(24, 20480)
        )
        layer.attn_hc = SimpleNamespace(fn=layer.hc_attn_fn, norm=torch.nn.Module())
        layer.ffn_hc = SimpleNamespace(fn=layer.hc_attn_fn, norm=torch.nn.Module())

        def pack(hc):
            setattr(hc.norm, tpu_mhc._PACKED, object())

        with (
            patch(
                "sglang.srt.models.deepseek_v4.DeepseekV4PlatformDecoderLayer",
                torch.nn.Module,
            ),
            patch(
                "sglang_tpu.pallas_export.get_pallas_tpu_device_kind",
                return_value="TPU7x",
            ),
            patch.object(tpu_mhc, "register_shifted_mhc_op"),
            patch.object(
                tpu_mhc, "_refresh_packed_buffer", side_effect=pack
            ) as refresh,
        ):
            tpu_mhc.prepare_shifted_mhc(
                [layer], weight_names=["model.layers.0.hc_attn_fn"]
            )
            self.assertEqual(
                [call.args[0] for call in refresh.call_args_list],
                [layer.attn_hc, layer.ffn_hc],
            )
            refresh.reset_mock()
            tpu_mhc.prepare_shifted_mhc(
                [layer], weight_names=["model.layers.0.mlp.weight"]
            )
            refresh.assert_not_called()
            tpu_mhc.prepare_shifted_mhc(
                [layer], weight_names=["model.layers.0.hc_ffn_fn"]
            )
            refresh.assert_called_once_with(layer.ffn_hc)

    def test_custom_loader_is_rejected_before_mutation_only_for_packed_models(self):
        model = torch.nn.Module()
        model.norm = torch.nn.Module()
        model.attn_hc = self.hc
        updater = SimpleNamespace(
            custom_weight_loaders={"custom"}, get_model=lambda: model
        )
        original = Mock(return_value=(True, "Success"))
        run = hooks.tpu_dsv41_guard_shifted_mhc_custom_loader
        self.assertTrue(run(original, updater, [], "custom")[0])
        setattr(model.norm, tpu_mhc._OWNER, (ref(model), "attn_hc"))
        original.reset_mock()
        self.assertFalse(run(original, updater, [], "custom")[0])
        original.assert_not_called()
        for fmt in (None, "direct", "flattened_bucket"):
            self.assertTrue(run(original, updater, [], fmt)[0])

    def test_weight_checker_excludes_packed_buffer_and_repacks_after_reset(self):
        from sglang.srt.model_executor.weight_checker import WeightChecker
        from sglang.srt.plugins.hook_registry import HookType, _wrap_fn

        model = torch.nn.Module()
        model.mix = torch.nn.Parameter(torch.zeros(24, 20480), requires_grad=False)
        model.norm = torch.nn.Module()
        model.attn_hc = self.hc._replace(fn=model.mix, norm=model.norm)
        tpu_mhc._refresh_packed_buffer(model.attn_hc)
        retained = getattr(model.norm, tpu_mhc._PACKED)
        setattr(model.norm, tpu_mhc._OWNER, (ref(model), "attn_hc"))
        checker = WeightChecker.__new__(WeightChecker)
        checker._get_model = lambda: model
        checker._placement = dict(
            tp_rank=0, tp_size=1, dp_rank=0, dp_size=1, pp_rank=0, pp_size=1
        )
        with patch.object(
            WeightChecker,
            "_model_state",
            _wrap_fn(
                WeightChecker._model_state,
                hooks.tpu_dsv41_exclude_packed_mhc_from_weight_checks,
                HookType.AROUND,
            ),
        ):
            self.assertEqual(set(dict(checker._model_state())), {"mix"})
            raw = checker.handle("raw_checksum", role="target")
            self.assertEqual(set(raw["checksums"]), {"mix"})
            reset = _wrap_fn(
                WeightChecker._reset_tensors,
                hooks.tpu_dsv41_refresh_shifted_mhc_after_weight_reset,
                HookType.AFTER,
            )
            with patch(
                "sglang.srt.model_executor.weight_checker._random_like",
                side_effect=lambda tensor: torch.full_like(tensor, 0.1),
            ):
                reset(checker)
        self.assertIs(getattr(model.norm, tpu_mhc._PACKED), retained)
        torch.testing.assert_close(retained, tpu_mhc.pack_mix_weight(model.mix))

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
