# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash TPU vision hooks (sglang_tpu/hooks/glm5_next_vision.py)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import inspect
import pkgutil
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.compilation.support_torch_compile import (
    CompilePolicy,
    get_torch_compile_policy,
)
from sglang.srt.environ import envs
from sglang.srt.managers import mm_schedule
from sglang.srt.plugins.hook_registry import HookType, _wrap_fn
from sglang.srt.runtime_context import get_flags
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import mm_schedule_tpu
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import glm5_next_vision as hooks

_GLM5_VISION_HOOKS = [h for h in HOOKS if h[1].__module__ == hooks.__name__]


def _parameters(fn, skip: int) -> list[tuple]:
    params = list(inspect.signature(fn).parameters.values())[skip:]
    return [(p.name, p.kind, p.default) for p in params]


class _FakeGlm5NextVLDecoderTpuCompilePolicy(CompilePolicy):
    pass


class _FakeGlm5NextDecoderTpuCompilePolicy(CompilePolicy):
    pass


def _fake_decoder_policy_module() -> dict[str, types.ModuleType]:
    module = types.ModuleType("sglang_tpu.compilation.decoder_compile_policy")
    module.Glm5NextVLDecoderTpuCompilePolicy = _FakeGlm5NextVLDecoderTpuCompilePolicy
    glm = types.ModuleType("sglang_tpu.models.glm5_next_compile_policy")
    glm.Glm5NextDecoderTpuCompilePolicy = _FakeGlm5NextDecoderTpuCompilePolicy
    return {module.__name__: module, glm.__name__: glm}


class TestGlm5NextVisionHooks(CustomTestCase):
    def test_hooks_target_existing_glm5_next_methods(self):
        self.assertEqual(len(_GLM5_VISION_HOOKS), 8)
        for target, _, _ in _GLM5_VISION_HOOKS:
            with self.subTest(target=target):
                self.assertTrue(callable(pkgutil.resolve_name(target)))

    def test_around_hooks_mirror_target_signatures(self):
        arounds = [
            (hook, target)
            for target, hook, hook_type in _GLM5_VISION_HOOKS
            if hook_type is HookType.AROUND
        ]
        self.assertEqual(len(arounds), 5)
        self.assertEqual(
            {hook for hook, _ in arounds},
            {
                hooks.defer_host_staged_mm_feature_move,
                hooks.encode_with_glm5_next_vision_runner,
                hooks.encode_glm5_next_vision_on_host,
                hooks.forward_glm5_next_multimodal,
            },
        )
        for hook, target in arounds:
            with self.subTest(target=target):
                target_params = _parameters(pkgutil.resolve_name(target), skip=0)
                is_method = bool(target_params and target_params[0][0] == "self")
                self.assertEqual(
                    _parameters(hook, skip=2 if is_method else 1),
                    target_params[1:] if is_method else target_params,
                )

    def test_compile_decoder_only_annotates_multimodal_decoder(self):
        class _DummyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_tokens = torch.nn.Embedding(16, 8)
                self.layers_to_capture = [3, 22, 42]

        class _DummyOuter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.visual = torch.nn.Linear(2, 2)
                self.model = _DummyDecoder()
                self.lm_head = torch.nn.Linear(8, 16, bias=False)
                self.mm_config = SimpleNamespace(
                    language_only=False, language_model_only=False
                )
                self.use_data_parallel = False
                self.decode_embeds_in_language_model = False
                self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
                self.logits_processor = object()

            def get_input_embeddings(self):
                return self.model.embed_tokens

        model = _DummyOuter()
        expected_param_names = set(dict(model.named_parameters()).keys())
        expected_state_keys = set(model.state_dict().keys())

        with (
            get_flags().dp.override(enabled=False),
            patch.dict(sys.modules, _fake_decoder_policy_module()),
        ):
            hooks.compile_glm5_next_decoder_only(None, model)

        self.assertTrue(model.keeps_mm_features_on_host)
        self.assertTrue(model.decode_embeds_in_language_model)
        self.assertIsInstance(model.language_model, hooks._Glm5NextLanguageModel)
        self.assertIsNone(get_torch_compile_policy(model.model))
        self.assertIs(
            get_torch_compile_policy(model.language_model),
            _FakeGlm5NextVLDecoderTpuCompilePolicy,
        )
        self.assertIn("language_model", dict(model.named_modules()))
        self.assertEqual(
            set(dict(model.named_parameters()).keys()), expected_param_names
        )
        self.assertEqual(set(model.state_dict().keys()), expected_state_keys)
        self.assertIs(model.language_model.pp_group, model.pp_group)
        self.assertIs(model.language_model.logits_processor, model.logits_processor)
        self.assertEqual(model.language_model.layers_to_capture, [3, 22, 42])
        self.assertIs(
            model.language_model.get_input_embeddings(), model.model.embed_tokens
        )

    def test_compile_decoder_only_annotates_epd_language_only(self):
        class _DummyDecoder(torch.nn.Module):
            pass

        epd_language_only = SimpleNamespace(
            visual=None,
            model=_DummyDecoder(),
            mm_config=SimpleNamespace(language_only=True, language_model_only=False),
            decode_embeds_in_language_model=False,
        )
        with (
            get_flags().dp.override(enabled=False),
            patch.dict(sys.modules, _fake_decoder_policy_module()),
        ):
            hooks.compile_glm5_next_decoder_only(None, epd_language_only)

        self.assertFalse(hasattr(epd_language_only, "keeps_mm_features_on_host"))
        self.assertTrue(epd_language_only.decode_embeds_in_language_model)
        self.assertIsInstance(
            epd_language_only.language_model, hooks._Glm5NextLanguageModel
        )
        self.assertIsNone(get_torch_compile_policy(epd_language_only.model))
        self.assertIs(
            get_torch_compile_policy(epd_language_only.language_model),
            _FakeGlm5NextVLDecoderTpuCompilePolicy,
        )

    def test_compile_decoder_only_annotates_text_only_whole_model(self):
        """--language-model-only compiles the whole model, as the fallback does,
        under the GLM policy that declares it spmd_safe."""

        class _DummyDecoder(torch.nn.Module):
            pass

        def text_only():
            return SimpleNamespace(
                visual=None,
                model=_DummyDecoder(),
                mm_config=SimpleNamespace(language_only=True, language_model_only=True),
                decode_embeds_in_language_model=True,
            )

        language_model_only, dp_attention = text_only(), text_only()
        with patch.dict(sys.modules, _fake_decoder_policy_module()):
            with get_flags().dp.override(enabled=False):
                hooks.compile_glm5_next_decoder_only(None, language_model_only)
            with get_flags().dp.override(enabled=True):
                hooks.compile_glm5_next_decoder_only(None, dp_attention)

        self.assertIs(
            get_torch_compile_policy(language_model_only),
            _FakeGlm5NextDecoderTpuCompilePolicy,
        )
        self.assertIsNone(get_torch_compile_policy(language_model_only.model))
        self.assertFalse(hasattr(language_model_only, "language_model"))
        self.assertTrue(language_model_only.decode_embeds_in_language_model)
        # DP attention keeps TpuGraphRunner's whole-model fallback.
        self.assertIsNone(get_torch_compile_policy(dp_attention))
        self.assertIsNone(get_torch_compile_policy(dp_attention.model))
        self.assertFalse(hasattr(dp_attention, "language_model"))

    def test_compile_decoder_only_skips_encoder_only(self):
        encoder_only = SimpleNamespace(
            visual=torch.nn.Linear(2, 2),
            model=None,
            mm_config=SimpleNamespace(encoder_only=True, language_model_only=False),
            use_data_parallel=False,
        )
        with (
            get_flags().dp.override(enabled=False),
            patch.dict(sys.modules, _fake_decoder_policy_module()),
        ):
            hooks.compile_glm5_next_decoder_only(None, encoder_only)

        self.assertIsNone(get_torch_compile_policy(encoder_only))
        self.assertFalse(hasattr(encoder_only, "language_model"))
        self.assertTrue(encoder_only.keeps_mm_features_on_host)

    def test_compile_decoder_only_skips_when_dp_attention_enabled(self):
        class _DummyDecoder(torch.nn.Module):
            pass

        for name, visual, language_only in (
            ("multimodal", torch.nn.Linear(2, 2), False),
            ("epd_language_only", None, True),
        ):
            with self.subTest(mode=name):
                model = SimpleNamespace(
                    visual=visual,
                    model=_DummyDecoder(),
                    mm_config=SimpleNamespace(
                        language_only=language_only, language_model_only=False
                    ),
                    use_data_parallel=False,
                    decode_embeds_in_language_model=True,
                )
                with (
                    get_flags().dp.override(enabled=True),
                    patch.dict(sys.modules, _fake_decoder_policy_module()),
                    self.assertLogs(hooks.logger, level="WARNING") as logs,
                ):
                    hooks.compile_glm5_next_decoder_only(None, model)

                self.assertIsNone(get_torch_compile_policy(model.model))
                self.assertFalse(hasattr(model, "language_model"))
                self.assertTrue(model.decode_embeds_in_language_model)
                self.assertTrue(
                    any("--enable-dp-attention" in msg for msg in logs.output)
                )

    def test_keeps_mm_features_on_host_tracks_dp_encoder_through_around_hook(self):
        class _DummyGlm5Model:
            def __init__(self, *, model_dp: bool, visual_dp: bool):
                self.visual = SimpleNamespace(
                    device=torch.device("cpu"), use_data_parallel=visual_dp
                )
                self.model = None
                self.use_data_parallel = model_dp

            def get_image_feature(self, items):
                return items

            def get_video_feature(self, items):
                return items

            def get_audio_feature(self, items):
                return items

        _DummyGlm5Model.get_image_feature = _wrap_fn(
            _DummyGlm5Model.get_image_feature,
            hooks.encode_glm5_next_vision_on_host,
            HookType.AROUND,
        )
        _DummyGlm5Model.get_video_feature = _wrap_fn(
            _DummyGlm5Model.get_video_feature,
            hooks.encode_glm5_next_vision_on_host,
            HookType.AROUND,
        )

        wrapped_can_skip = _wrap_fn(
            mm_schedule._can_skip_pre_embed_feature_move,
            hooks.defer_host_staged_mm_feature_move,
            HookType.AROUND,
        )
        feature = torch.ones((4, 8))

        non_dp = _DummyGlm5Model(model_dp=False, visual_dp=False)
        hooks.compile_glm5_next_decoder_only(None, non_dp)
        self.assertTrue(non_dp.keeps_mm_features_on_host)

        with patch.object(
            mm_schedule, "_can_skip_pre_embed_feature_move", wrapped_can_skip
        ):
            for method_name in ("get_image_feature", "get_video_feature"):
                with self.subTest(method=method_name):
                    embed_fn = getattr(non_dp, method_name)
                    self.assertIs(getattr(embed_fn, "__self__", None), non_dp)
                    self.assertEqual(getattr(embed_fn, "__name__", None), method_name)

                    item = SimpleNamespace(feature=feature)
                    mm_schedule._move_items_to_device(
                        [item], torch.device("meta"), embed_fn
                    )
                    self.assertIs(item.feature, feature)

                    with patch.object(
                        mm_schedule_tpu, "_default_move_items_to_device"
                    ) as default:
                        mm_schedule_tpu._move_items_to_device(
                            [item], torch.device("meta"), embed_fn
                        )
                        default.assert_not_called()
                        self.assertIs(item.feature, feature)

            audio_item = SimpleNamespace(feature=feature)
            mm_schedule._move_items_to_device(
                [audio_item], torch.device("meta"), non_dp.get_audio_feature
            )
            self.assertEqual(audio_item.feature.device.type, "meta")

            for model_dp, visual_dp in ((True, False), (False, True)):
                with self.subTest(model_dp=model_dp, visual_dp=visual_dp):
                    dp_model = _DummyGlm5Model(model_dp=model_dp, visual_dp=visual_dp)
                    hooks.compile_glm5_next_decoder_only(None, dp_model)
                    self.assertFalse(dp_model.keeps_mm_features_on_host)

                    dp_item = SimpleNamespace(feature=feature)
                    mm_schedule._move_items_to_device(
                        [dp_item], torch.device("meta"), dp_model.get_image_feature
                    )
                    self.assertEqual(dp_item.feature.device.type, "meta")

                    tpu_item = SimpleNamespace(feature=feature)
                    with patch.object(
                        mm_schedule_tpu, "_default_move_items_to_device"
                    ) as default:
                        mm_schedule_tpu._move_items_to_device(
                            [tpu_item],
                            torch.device("meta"),
                            dp_model.get_image_feature,
                        )
                        default.assert_called_once_with(
                            [tpu_item],
                            torch.device("meta"),
                            dp_model.get_image_feature,
                        )

    def test_encode_with_vision_runner_falls_back_when_runner_declines(self):
        sentinel = torch.ones((2, 4))
        unready_visual = SimpleNamespace(graph_runners=SimpleNamespace(ready=False))
        got = hooks.encode_with_glm5_next_vision_runner(
            lambda _v, _x, _g: sentinel,
            unready_visual,
            torch.zeros((8, 4)),
            torch.tensor([[1, 2, 4]]),
        )
        self.assertIs(got, sentinel)

        declining_visual = SimpleNamespace(
            graph_runners=SimpleNamespace(ready=True, run_grid=lambda _x, _g: None)
        )
        got_fallback = hooks.encode_with_glm5_next_vision_runner(
            lambda _v, _x, _g: sentinel,
            declining_visual,
            torch.zeros((8, 4)),
            torch.tensor([[1, 2, 4]]),
        )
        self.assertIs(got_fallback, sentinel)

    def test_encode_vision_on_host_respects_mm_embed_on_device_env(self):
        raw = torch.ones((4, 8), dtype=torch.float32)
        model = SimpleNamespace(visual=SimpleNamespace(device=torch.device("cpu")))

        with envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False):
            out_host = hooks.encode_glm5_next_vision_on_host(
                lambda _m, _items: raw, model, []
            )
            self.assertEqual(out_host.device.type, "cpu")
            self.assertTrue(torch.equal(out_host, raw))

        meta_model = SimpleNamespace(
            visual=SimpleNamespace(device=torch.device("meta"))
        )
        with envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(True):
            out_dev = hooks.encode_glm5_next_vision_on_host(
                lambda _m, _items: raw, meta_model, []
            )
            self.assertEqual(out_dev.device.type, "meta")
            self.assertEqual(out_dev.shape, (4, 8))

        non_tensor = []
        self.assertIs(
            hooks.encode_glm5_next_vision_on_host(
                lambda _m, _items: non_tensor, model, []
            ),
            non_tensor,
        )

    def test_forward_glm5_next_multimodal_routes_decode_verify_and_extend(self):
        from sglang.srt.layers.layer_boundary.adapters.attention import (
            get_attn_tp_context,
        )
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardMode,
            LogitsOutputMode,
        )
        from sglang_tpu.envs import envs as tpu_envs
        from sglang_tpu.mm_embed_tpu import embed_mm_inputs_tpu

        fallback_out = object()
        unwrapped_model = SimpleNamespace(language_model=None)
        self.assertIs(
            hooks.forward_glm5_next_multimodal(
                lambda *a, **kw: fallback_out,
                unwrapped_model,
                torch.ones(4, dtype=torch.int64),
                torch.arange(4),
                SimpleNamespace(forward_mode=ForwardMode.EXTEND),
            ),
            fallback_out,
        )

        lm_calls = []

        def fake_lm(*args, **kwargs):
            lm_calls.append((args, kwargs, torch.is_grad_enabled()))
            return "lm_result"

        mrope_positions = torch.zeros((3, 4), dtype=torch.int64)
        input_ids = torch.ones(4, dtype=torch.int64)
        model = SimpleNamespace(
            language_model=fake_lm,
            is_mrope_enabled=True,
        )

        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            lm_calls.clear()
            fb = SimpleNamespace(
                forward_mode=mode,
                mrope_positions=mrope_positions,
                logits_output_mode=LogitsOutputMode.DISTRIBUTED_TOP_K,
            )
            got = hooks.forward_glm5_next_multimodal(
                lambda *a, **kw: self.fail("original_fn should not be called"),
                model,
                input_ids,
                torch.arange(4),
                fb,
            )
            self.assertEqual(got, "lm_result")
            self.assertEqual(len(lm_calls), 1)
            args, kwargs, grad_enabled = lm_calls[0]
            self.assertFalse(grad_enabled)
            self.assertEqual(fb.logits_output_mode, LogitsOutputMode.DISTRIBUTED_TOP_K)
            self.assertEqual(len(args), 3)
            self.assertIs(args[0], input_ids)
            self.assertIs(args[1], mrope_positions)
            self.assertIs(args[2], fb)
            self.assertEqual(kwargs, {"input_embeds": None, "pp_proxy_tensors": None})

        lm_calls.clear()
        extend_fb = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            mrope_positions=mrope_positions,
            logits_output_mode=LogitsOutputMode.DISTRIBUTED_TOP_K,
        )
        with tpu_envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND.override(True):
            got = hooks.forward_glm5_next_multimodal(
                lambda *a, **kw: self.fail("original_fn should not be called"),
                model,
                input_ids,
                torch.arange(4),
                extend_fb,
            )
        self.assertEqual(got, "lm_result")
        self.assertEqual(extend_fb.logits_output_mode, LogitsOutputMode.FULL_VOCAB)
        self.assertEqual(len(lm_calls), 1)
        self.assertIs(lm_calls[0][0][1], mrope_positions)

        mm_calls = []

        def fake_mm_routine(**kwargs):
            mm_calls.append(
                (kwargs, get_attn_tp_context().input_scattered, torch.is_grad_enabled())
            )
            return "mm_extend_result"

        extend_fb.logits_output_mode = LogitsOutputMode.DISTRIBUTED_TOP_K
        with (
            tpu_envs.SGLANG_TPU_VL_TEXT_ONLY_EXTEND.override(False),
            patch(
                "sglang.srt.managers.mm_utils.general_mm_embed_routine",
                fake_mm_routine,
            ),
        ):
            got = hooks.forward_glm5_next_multimodal(
                lambda *a, **kw: self.fail("original_fn should not be called"),
                model,
                input_ids,
                torch.arange(4),
                extend_fb,
            )
        self.assertEqual(got, "mm_extend_result")
        self.assertEqual(extend_fb.logits_output_mode, LogitsOutputMode.FULL_VOCAB)
        self.assertEqual(len(mm_calls), 1)
        mm_kwargs, was_scattered, grad_enabled = mm_calls[0]
        self.assertFalse(was_scattered)
        self.assertFalse(grad_enabled)
        self.assertIs(mm_kwargs["language_model"], fake_lm)
        self.assertIs(mm_kwargs["multimodal_model"], model)
        self.assertIs(mm_kwargs["positions"], mrope_positions)
        self.assertIs(mm_kwargs["mm_embedder"], embed_mm_inputs_tpu)

    def test_glm5_next_language_model_scales_scattered_input_embeds_to_unit_tp_sum(
        self,
    ):
        from sglang.srt.layers import dp_attention
        from sglang.srt.layers.layer_boundary.adapters.attention import (
            get_attn_tp_context,
        )
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.runtime_context import get_parallel

        attn_tp_ctx = get_attn_tp_context()
        base_embeds = torch.arange(24, dtype=torch.float32).reshape(6, 4) + 1.0
        aux_tensor = torch.ones((6, 4), dtype=torch.float32)
        seen_by_decoder = []
        logits_calls = []

        class _Owner(torch.nn.Module):
            def __init__(self, *, is_last_rank: bool, capture_aux: bool):
                super().__init__()
                self.capture_aux_hidden_states = capture_aux
                self.pp_group = SimpleNamespace(
                    is_first_rank=True, is_last_rank=is_last_rank
                )
                self.lm_head = torch.nn.Linear(4, 8, bias=False)
                self.model = self._run_decoder
                self.logits_processor = self._run_logits

            def _run_decoder(self, **kwargs):
                seen_by_decoder.append(kwargs["input_embeds"])
                out = kwargs["input_embeds"]
                return (out, [aux_tensor]) if self.capture_aux_hidden_states else out

            def _run_logits(
                self, input_ids, hidden_states, lm_head, forward_batch, aux_hidden
            ):
                logits_calls.append(
                    (input_ids, hidden_states, lm_head, forward_batch, aux_hidden)
                )
                return "logits_out"

        forward_batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            input_ids=torch.ones(6, dtype=torch.int64),
            can_run_tbo=False,
        )
        saved_allow = getattr(attn_tp_ctx, "allow_input_scattered", False)
        saved_rank_tensors = dp_attention._RANK_TENSORS.copy()
        try:
            attn_tp_ctx.allow_input_scattered = True
            for rank in range(8):
                dp_attention._RANK_TENSORS = {}
                owner = _Owner(is_last_rank=True, capture_aux=True)
                lm = hooks._Glm5NextLanguageModel(owner)
                with get_parallel().override(
                    tp_size=8,
                    tp_rank=rank,
                    attn_tp_size=8,
                    attn_tp_rank=rank,
                    attn_dp_size=1,
                    attn_dp_rank=0,
                ):
                    result = lm(
                        None,
                        torch.zeros((3, 6), dtype=torch.int64),
                        forward_batch,
                        input_embeds=base_embeds.clone(),
                    )
                self.assertEqual(result, "logits_out")

            self.assertEqual(len(seen_by_decoder), 8)
            tp_sum = sum(seen_by_decoder)
            torch.testing.assert_close(tp_sum, base_embeds)
            self.assertEqual(len(logits_calls), 8)
            self.assertEqual(logits_calls[0][4], [aux_tensor])

            non_last_owner = _Owner(is_last_rank=False, capture_aux=False)
            non_last_lm = hooks._Glm5NextLanguageModel(non_last_owner)
            attn_tp_ctx.allow_input_scattered = False
            raw_hidden = non_last_lm(
                None,
                torch.zeros((3, 6), dtype=torch.int64),
                forward_batch,
                input_embeds=base_embeds.clone(),
            )
            torch.testing.assert_close(raw_hidden, base_embeds)
        finally:
            attn_tp_ctx.allow_input_scattered = saved_allow
            dp_attention._RANK_TENSORS = saved_rank_tensors

    def test_attach_glm5_next_nextn_multimodal_forward_accepts_input_embeds(self):
        inner_calls = []
        logits_calls = []

        class _DummyDraft(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = torch.nn.Linear(4, 8, bias=False)
                self.model = self._run_model
                self.logits_processor = self._run_logits

            def _run_model(
                self, input_ids, positions, forward_batch, input_embeds=None
            ):
                inner_calls.append(
                    (
                        input_ids,
                        positions,
                        forward_batch,
                        input_embeds,
                        torch.is_grad_enabled(),
                    )
                )
                return (
                    input_embeds
                    if input_embeds is not None
                    else torch.zeros((input_ids.shape[0], 4))
                )

            def _run_logits(self, input_ids, hidden_states, lm_head, forward_batch):
                logits_calls.append((input_ids, hidden_states, lm_head, forward_batch))
                return "draft_logits"

        draft = _DummyDraft()
        expected_params = set(dict(draft.named_parameters()).keys())
        hooks.attach_glm5_next_nextn_multimodal_forward(None, draft)
        self.assertEqual(set(dict(draft.named_parameters()).keys()), expected_params)
        self.assertEqual(
            list(inspect.signature(draft.forward).parameters.keys()),
            [
                "input_ids",
                "positions",
                "forward_batch",
                "input_embeds",
                "pp_proxy_tensors",
            ],
        )

        ids = torch.ones(4, dtype=torch.int64)
        pos = torch.arange(4, dtype=torch.int64)
        fb = SimpleNamespace()
        embeds = torch.full((4, 4), 3.0)

        out_plain = draft.forward(ids, pos, fb)
        self.assertEqual(out_plain, "draft_logits")
        self.assertIsNone(inner_calls[-1][3])
        self.assertFalse(inner_calls[-1][4])

        out_mm = draft.forward(ids, pos, fb, input_embeds=embeds)
        self.assertEqual(out_mm, "draft_logits")
        self.assertIs(inner_calls[-1][3], embeds)
        self.assertIs(logits_calls[-1][1], embeds)


if __name__ == "__main__":
    unittest.main()
