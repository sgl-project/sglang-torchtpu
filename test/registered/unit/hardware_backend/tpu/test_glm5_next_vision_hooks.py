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


class _FakeQwen3VLDecoderTpuCompilePolicy(CompilePolicy):
    pass


class _FakeGlm5NextDecoderTpuCompilePolicy(CompilePolicy):
    pass


def _fake_decoder_policy_module() -> dict[str, types.ModuleType]:
    module = types.ModuleType("sglang_tpu.compilation.decoder_compile_policy")
    module.Qwen3VLDecoderTpuCompilePolicy = _FakeQwen3VLDecoderTpuCompilePolicy
    glm = types.ModuleType("sglang_tpu.models.glm5_next_compile_policy")
    glm.Glm5NextDecoderTpuCompilePolicy = _FakeGlm5NextDecoderTpuCompilePolicy
    return {module.__name__: module, glm.__name__: glm}


class TestGlm5NextVisionHooks(CustomTestCase):
    def test_hooks_target_existing_glm5_next_methods(self):
        self.assertEqual(len(_GLM5_VISION_HOOKS), 6)
        for target, _, _ in _GLM5_VISION_HOOKS:
            with self.subTest(target=target):
                self.assertTrue(callable(pkgutil.resolve_name(target)))

    def test_around_hooks_mirror_target_signatures(self):
        arounds = [
            (hook, target)
            for target, hook, hook_type in _GLM5_VISION_HOOKS
            if hook_type is HookType.AROUND
        ]
        self.assertEqual(len(arounds), 4)
        self.assertEqual(
            {hook for hook, _ in arounds},
            {
                hooks.defer_host_staged_mm_feature_move,
                hooks.encode_with_glm5_next_vision_runner,
                hooks.encode_glm5_next_vision_on_host,
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
            pass

        model = SimpleNamespace(
            visual=torch.nn.Linear(2, 2),
            model=_DummyDecoder(),
            mm_config=SimpleNamespace(language_only=False, language_model_only=False),
            use_data_parallel=False,
            decode_embeds_in_language_model=True,
        )
        with (
            get_flags().dp.override(enabled=False),
            patch.dict(sys.modules, _fake_decoder_policy_module()),
        ):
            hooks.compile_glm5_next_decoder_only(None, model)

        self.assertTrue(model.keeps_mm_features_on_host)
        self.assertFalse(model.decode_embeds_in_language_model)
        self.assertIs(
            get_torch_compile_policy(model.model),
            _FakeQwen3VLDecoderTpuCompilePolicy,
        )

    def test_compile_decoder_only_annotates_epd_language_only(self):
        class _DummyDecoder(torch.nn.Module):
            pass

        epd_language_only = SimpleNamespace(
            visual=None,
            model=_DummyDecoder(),
            mm_config=SimpleNamespace(language_only=True, language_model_only=False),
            decode_embeds_in_language_model=True,
        )
        with (
            get_flags().dp.override(enabled=False),
            patch.dict(sys.modules, _fake_decoder_policy_module()),
        ):
            hooks.compile_glm5_next_decoder_only(None, epd_language_only)

        self.assertFalse(hasattr(epd_language_only, "keeps_mm_features_on_host"))
        self.assertFalse(epd_language_only.decode_embeds_in_language_model)
        self.assertIs(
            get_torch_compile_policy(epd_language_only.model),
            _FakeQwen3VLDecoderTpuCompilePolicy,
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
        self.assertTrue(language_model_only.decode_embeds_in_language_model)
        # DP attention keeps TpuGraphRunner's whole-model fallback.
        self.assertIsNone(get_torch_compile_policy(dp_attention))
        self.assertIsNone(get_torch_compile_policy(dp_attention.model))

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


if __name__ == "__main__":
    unittest.main()
