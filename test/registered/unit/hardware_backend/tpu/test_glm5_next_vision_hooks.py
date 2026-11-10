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
from sglang.srt.plugins.hook_registry import HookType
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import glm5_next_vision as hooks

_GLM5_VISION_HOOKS = [h for h in HOOKS if h[1].__module__ == hooks.__name__]


def _parameters(fn, skip: int) -> list[tuple]:
    params = list(inspect.signature(fn).parameters.values())[skip:]
    return [(p.name, p.kind, p.default) for p in params]


class _FakeQwen3VLDecoderTpuCompilePolicy(CompilePolicy):
    pass


def _fake_decoder_policy_module() -> dict[str, types.ModuleType]:
    module = types.ModuleType("sglang_tpu.compilation.decoder_compile_policy")
    module.Qwen3VLDecoderTpuCompilePolicy = _FakeQwen3VLDecoderTpuCompilePolicy
    return {module.__name__: module}


class TestGlm5NextVisionHooks(CustomTestCase):
    def test_hooks_target_existing_glm5_next_methods(self):
        self.assertEqual(len(_GLM5_VISION_HOOKS), 5)
        for target, _, _ in _GLM5_VISION_HOOKS:
            with self.subTest(target=target):
                self.assertTrue(callable(pkgutil.resolve_name(target)))

    def test_around_hooks_mirror_target_signatures(self):
        arounds = [
            (hook, target)
            for target, hook, hook_type in _GLM5_VISION_HOOKS
            if hook_type is HookType.AROUND
        ]
        self.assertEqual(len(arounds), 3)
        self.assertEqual(
            {hook for hook, _ in arounds},
            {
                hooks.encode_with_glm5_next_vision_runner,
                hooks.encode_glm5_next_vision_on_host,
            },
        )
        for hook, target in arounds:
            with self.subTest(target=target):
                self.assertEqual(
                    _parameters(hook, skip=2),
                    _parameters(pkgutil.resolve_name(target), skip=1),
                )

    def test_compile_decoder_only_annotates_multimodal_decoder(self):
        class _DummyDecoder(torch.nn.Module):
            pass

        model = SimpleNamespace(
            visual=torch.nn.Linear(2, 2),
            model=_DummyDecoder(),
            decode_embeds_in_language_model=True,
        )
        with patch.dict(sys.modules, _fake_decoder_policy_module()):
            hooks.compile_glm5_next_decoder_only(None, model)

        self.assertFalse(model.decode_embeds_in_language_model)
        self.assertIs(
            get_torch_compile_policy(model.model),
            _FakeQwen3VLDecoderTpuCompilePolicy,
        )

    def test_compile_decoder_only_skips_language_only_and_encoder_only(self):
        class _DummyDecoder(torch.nn.Module):
            pass

        language_only = SimpleNamespace(visual=None, model=_DummyDecoder())
        encoder_only = SimpleNamespace(visual=torch.nn.Linear(2, 2), model=None)
        with patch.dict(sys.modules, _fake_decoder_policy_module()):
            hooks.compile_glm5_next_decoder_only(None, language_only)
            hooks.compile_glm5_next_decoder_only(None, encoder_only)

        self.assertIsNone(get_torch_compile_policy(language_only.model))

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

        meta_model = SimpleNamespace(visual=SimpleNamespace(device=torch.device("meta")))
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
