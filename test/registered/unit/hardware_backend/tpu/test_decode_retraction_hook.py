"""The TPU plugin rejects host-pool decode retraction at startup."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

from sglang.srt.arg_groups import resolution_hooks
from sglang.srt.arg_groups.hicache_hook import handle_hicache_ratio_default
from sglang.srt.arg_groups.kv_cache_hook import handle_cache_compatibility
from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.hooks.decode_retraction import reject_host_pool_retraction


def _make_args(**overrides) -> ServerArgs:
    args = ServerArgs(model_path="dummy", **overrides)
    handle_hicache_ratio_default(args)
    return args


class TestRejectHostPoolRetraction(CustomTestCase):
    def setUp(self):
        hooks = patch.dict(resolution_hooks._HOOKS, clear=True)
        hooks.start()
        self.addCleanup(hooks.stop)
        resolution_hooks.register_resolution_hook("handle_cache_compatibility")(
            reject_host_pool_retraction
        )

    def _check(self, args):
        resolution_hooks.run_hook(handle_cache_compatibility, args)

    def test_rejects_host_pool_on_tpu(self):
        # Rejected whatever the hicache layout/io, instead of at the first
        # retraction restore.
        for hicache_kwargs in (
            {},
            {"hicache_io_backend": "raiden", "hicache_mem_layout": "page_first_fused"},
        ):
            with self.subTest(**hicache_kwargs):
                args = _make_args(
                    device="tpu",
                    disaggregation_mode="decode",
                    disaggregation_decode_retraction_backup="host_pool",
                    **hicache_kwargs,
                )
                with self.assertRaisesRegex(ValueError, "not supported on TPU"):
                    self._check(args)

    def test_host_receive_names_the_threshold(self):
        # Decode host receive resolves retraction to host_pool, so the error
        # names the threshold instead of suggesting cpu_tensor.
        args = _make_args(
            device="tpu",
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup="host_pool",
            disaggregation_decode_host_receive_threshold=0.5,
        )
        with self.assertRaisesRegex(ValueError, "Set the threshold to 0"):
            self._check(args)

    def test_host_receive_resolves_to_host_pool_and_is_rejected(self):
        # The real flow: the backup flag is unset and decode host receive
        # resolves it to host_pool.
        args = _make_args(
            device="tpu",
            disaggregation_mode="decode",
            disaggregation_decode_host_receive_threshold=0.5,
        )
        handle_pd_disaggregation(args)
        with self.assertRaisesRegex(ValueError, "Set the threshold to 0"):
            self._check(args)

    def test_allows_cpu_tensor_on_tpu(self):
        args = _make_args(
            device="tpu",
            disaggregation_mode="decode",
            disaggregation_decode_retraction_backup="cpu_tensor",
        )
        self._check(args)

    def test_core_checks_run_first(self):
        # The built-in step still rejects host_pool off a decode server.
        args = _make_args(
            device="tpu", disaggregation_decode_retraction_backup="host_pool"
        )
        with self.assertRaisesRegex(ValueError, "only supported on a PD decode"):
            self._check(args)


class TestRegisterWiresTheHook(CustomTestCase):
    def test_register_adds_the_resolution_hook_on_tpu(self):
        import sglang.srt.platforms as platforms
        import sglang_tpu.hooks as tpu_hooks

        with (
            patch.dict(resolution_hooks._HOOKS, clear=True),
            patch.object(tpu_hooks, "_registered", False),
            patch.object(platforms.current_platform, "is_tpu", return_value=True),
            patch("sglang.srt.plugins.hook_registry.HookRegistry.register"),
        ):
            tpu_hooks.register()
            self.assertIn(
                reject_host_pool_retraction,
                resolution_hooks._HOOKS.get("handle_cache_compatibility", []),
            )


if __name__ == "__main__":
    unittest.main()
