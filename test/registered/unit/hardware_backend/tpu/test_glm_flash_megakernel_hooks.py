"""CPU checks of the GLM-5.3-Flash megakernel's serving hooks."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.glm_flash_megakernel import ops
from sglang_tpu.hooks import glm_flash_megakernel as hook

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Kernel:
    def __init__(self, req):
        self.req = req

    def invalidate(self):
        self.req = None


def _target_runner():
    from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration

    # The route checks the exact model type; nothing it reads needs __init__.
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    model_runner = SimpleNamespace(
        server_args=SimpleNamespace(speculative_algorithm=None, model_path="glm"),
        model_config=SimpleNamespace(context_len=4096),
        attn_backend=SimpleNamespace(init_forward_metadata=lambda forward_batch: None),
    )
    return SimpleNamespace(model=model, model_runner=model_runner)


class TestGlmFlashMegakernelHooks(CustomTestCase):
    def test_freeing_the_kernels_request_drops_it(self):
        mk = _Kernel(req=3)
        with mock.patch.object(hook, "_KERNELS", [mk]):
            hook.drop_freed_glm_flash_megakernel_request(object(), [5, 3])
        self.assertIsNone(mk.req)

    def test_freeing_another_request_keeps_it(self):
        mk = _Kernel(req=3)
        with mock.patch.object(hook, "_KERNELS", [mk]):
            hook.drop_freed_glm_flash_megakernel_request(object(), [5])
        self.assertEqual(mk.req, 3)

    def test_route_requires_radix_cache_off(self):
        """Kernel steps write no radix snapshot of the KDA state, so a radix-on
        server fails at startup instead of caching states the kernel never wrote."""
        for disabled in (False, True):
            memory = SimpleNamespace(disable_radix_cache=disabled)
            with (
                self.subTest(disable_radix_cache=disabled),
                tpu_envs.SGLANG_TPU_GLM_FLASH_MEGAKERNEL.override(True),
                mock.patch(
                    "sglang.srt.runtime_context.get_memory", return_value=memory
                ),
                mock.patch.object(ops, "GlmFlashMegakernel") as kernel,
                mock.patch.object(hook, "_KERNELS", []),
            ):
                with (
                    contextlib.nullcontext()
                    if disabled
                    else self.assertRaisesRegex(ValueError, "--disable-radix-cache")
                ):
                    hook.route_glm_flash_megakernel(None, _target_runner())
                self.assertEqual(kernel.called, disabled)


if __name__ == "__main__":
    unittest.main()
