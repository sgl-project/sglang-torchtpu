"""GLM-5.3-Flash fused mHC hooks (sglang_tpu/hooks/mhc.py), with a stand-in for
the kernel module."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import inspect
import pkgutil
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.models.glm5_next import Glm5NextDecoderLayer
from sglang.srt.plugins.hook_registry import HookType
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import mhc as hooks

_FLAG = tpu_envs.SGLANG_TPU_ENABLE_GLM5_NEXT_FUSED_MHC
_MHC_HOOKS = [h for h in HOOKS if h[1].__module__ == hooks.__name__]


def _layer():
    layer = Glm5NextDecoderLayer.__new__(Glm5NextDecoderLayer)
    layer.__dict__["config"] = SimpleNamespace(
        mhc=True, hc_mult=4, rms_norm_eps=1e-5, hc_eps=1e-6, hc_sinkhorn_iters=20
    )
    for name in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
        layer.__dict__[name] = object()
    return layer


class TestGlm5NextMhcHooks(CustomTestCase):
    def test_around_hooks_mirror_target_signatures(self):
        """The AROUND hooks name their targets' parameters, so a core signature
        change fails here instead of misrouting arguments."""
        arounds = [h for h in _MHC_HOOKS if h[2] is HookType.AROUND]
        self.assertEqual(len(arounds), 3)
        for target, hook, _ in arounds:
            with self.subTest(target=target):
                params = list(inspect.signature(hook).parameters)
                original = inspect.signature(pkgutil.resolve_name(target))
                self.assertEqual(params[0], "original")
                self.assertEqual(params[1:], list(original.parameters))

    def test_post_pre_wired_only_for_glm_layers_with_the_flag(self):
        layer = _layer()
        other = SimpleNamespace(hc_ffn_pre=lambda: None)
        for flag, owner_pre, wired in (
            (True, layer.hc_ffn_pre, True),
            (False, layer.hc_ffn_pre, False),
            (True, other.hc_ffn_pre, False),
        ):
            with self.subTest(flag=flag, wired=wired), _FLAG.override(flag):
                _, kwargs = hooks.wire_hc_ffn_post_pre(
                    "state", hc_ffn_pre=owner_pre, hc_ffn_post_pre=None
                )
                post_pre = kwargs["hc_ffn_post_pre"]
                if wired:
                    self.assertEqual(post_pre, layer.hc_ffn_post_pre)
                else:
                    self.assertIsNone(post_pre)

    def test_layer_without_mhc_keeps_kernel_off(self):
        layer = SimpleNamespace(config=SimpleNamespace(mhc=False))
        with _FLAG.override(True), patch.object(hooks, "_kernel", None):
            hooks.register_mhc_ops(None, layer)
            self.assertIsNone(hooks._kernel)

    def test_routing(self):
        layer = _layer()
        original = MagicMock(return_value="torch")
        kernel = MagicMock()
        args = ("a", "b", "c", "d")

        with patch.object(hooks, "_kernel", None):
            self.assertEqual(hooks.hc_post(original, layer, *args), "torch")
            self.assertEqual(
                hooks.hc_ffn_post_pre(original, layer, *args, None, None), "torch"
            )

        with patch.object(hooks, "_kernel", kernel):
            kernel.hc_post.return_value = "kernel"
            self.assertEqual(hooks.hc_post(original, layer, *args), "kernel")
            kernel.hc_post.return_value = None
            self.assertEqual(hooks.hc_post(original, layer, *args), "torch")

            kernel.hc_pre.return_value = None
            self.assertEqual(hooks.hc_pre(original, layer, *args, None, None), "torch")
            # A declined fused site runs the unfused chain, not the GPU original.
            kernel.hc_post_pre.return_value = None
            original.reset_mock()
            self.assertIsNone(hooks.hc_ffn_post_pre(original, layer, *args, None, None))
            original.assert_not_called()


if __name__ == "__main__":
    unittest.main()
