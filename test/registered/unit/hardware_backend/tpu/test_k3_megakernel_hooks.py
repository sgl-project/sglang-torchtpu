"""Kimi-K3 decode megakernel hooks (sglang_tpu/hooks/k3_megakernel.py): registration,
step routing and the startup checks, without a TPU or the kernel."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import contextlib
import pkgutil
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import k3_megakernel as hooks

_FLAG = tpu_envs.SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL


def _forward(model, mode, bs=1, tokens=1):
    batch = SimpleNamespace(forward_mode=mode, batch_size=bs, _tpu_real_b=bs)
    ids = torch.zeros(tokens, dtype=torch.long)
    return hooks.k3_megakernel_forward(
        MagicMock(return_value="sglang"), model, ids, None, batch
    )


def _model_with_kernel():
    runtime = MagicMock()
    runtime.forward.return_value = "kernel"
    return SimpleNamespace(_k3_megakernel=runtime, _k3_megakernel_owns_weights=True)


class TestK3MegakernelHooks(CustomTestCase):
    def test_hooks_target_existing_kimi_k3_methods(self):
        targets = {t for t, fn, _ in HOOKS if fn.__module__ == hooks.__name__}
        self.assertEqual(len(targets), 5)
        for target in targets:
            with self.subTest(target=target):
                self.assertTrue(callable(pkgutil.resolve_name(target)))

    def test_flag_off_leaves_sglang_untouched(self):
        with _FLAG.override(False):
            self.assertEqual(_forward(SimpleNamespace(), ForwardMode.DECODE), "sglang")
            causal_lm = SimpleNamespace(model=SimpleNamespace())
            hooks.attach_k3_megakernel(None, causal_lm)
            self.assertFalse(hasattr(causal_lm.model, "_k3_megakernel"))

    def test_owned_weights_build_placeholder_layers(self):
        from sglang.srt.layers.utils import PPMissingLayer

        def make_pp_layers(num_layers, layer_fn, prefix=""):
            return [layer_fn(i, f"{prefix}.{i}") for i in range(num_layers)]

        real = MagicMock()
        with _FLAG.override(True):
            layers = hooks.skip_k3_decoder_layers(make_pp_layers, 3, real, prefix="m")
        self.assertTrue(all(isinstance(x, PPMissingLayer) for x in layers))
        real.assert_not_called()

    def test_bs1_steps_run_the_kernel_and_others_raise(self):
        model = _model_with_kernel()
        self.assertEqual(_forward(model, ForwardMode.DECODE), "kernel")
        self.assertEqual(_forward(model, ForwardMode.EXTEND, tokens=17), "kernel")
        for mode, bs, tokens in (
            (ForwardMode.DECODE, 2, 2),
            (ForwardMode.EXTEND, 2, 9),
            (ForwardMode.TARGET_VERIFY, 1, 8),
        ):
            with self.subTest(mode=mode, bs=bs), self.assertRaises(NotImplementedError):
                _forward(model, mode, bs, tokens)

    def test_startup_rejects_configs_the_kernel_cannot_run(self):
        ok = dict(
            max_running_requests=1, disable_radix_cache=True, speculative_algorithm=None,
            enable_mixed_chunk=False, chunked_prefill_size=1024, page_size=128,
            pp_size=1, dp_size=1, attn_cp_size=1, dcp_size=1, kv_cache_dtype="auto",
            attention_backend="pallas_tpu_mla", tpu_compile_bs=[1],
        )  # fmt: skip

        def check(**kw):
            a = SimpleNamespace(**{**ok, **kw})
            bags = {
                "get_schedule": a, "get_memory": a, "get_spec": a, "get_parallel": a,
                "get_model": a,
                "get_exec": SimpleNamespace(kernel=a, graph=a),
            }  # fmt: skip
            with contextlib.ExitStack() as stack:
                for name, bag in bags.items():
                    stack.enter_context(
                        patch(f"sglang.srt.runtime_context.{name}", return_value=bag)
                    )
                hooks._check_server_args()

        check()
        for kw, needle in (
            ({"max_running_requests": 4}, "--max-running-requests 1"),
            ({"disable_radix_cache": False}, "--disable-radix-cache"),
            ({"speculative_algorithm": "DSPARK"}, "no speculative decoding"),
            ({"chunked_prefill_size": -1}, "--chunked-prefill-size > 0"),
            ({"kv_cache_dtype": "fp8_e4m3"}, "bf16 KV cache"),
            ({"tpu_compile_bs": [1, 4]}, "TPU compile batch sizes"),
        ):
            with self.subTest(**kw), self.assertRaisesRegex(ValueError, needle):
                check(**kw)

    def test_serving_refuses_debug_and_broken_kernel_options(self):
        from sglang_tpu.k3_megakernel.ops import SERVING_KERNEL_OPTIONS, kernel_options

        serving = dict(allowed=SERVING_KERNEL_OPTIONS)
        self.assertEqual(kernel_options("moe_ring=6", **serving), {"moe_ring": 6})
        for bad in ("moe_shared_after=1", "debug_repeat=2", "collective_id=7", "x=1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                kernel_options(bad, **serving)

    def test_prefill_rows_cover_every_extend(self):
        from sglang_tpu.k3_megakernel.ops import prefill_rows

        self.assertEqual(prefill_rows(1024, "auto"), 1024)
        for size, mode in ((-1, "auto"), (None, "auto"), (1020, "rows")):
            with self.subTest(size=size, mode=mode), self.assertRaises(ValueError):
                prefill_rows(size, mode)


if __name__ == "__main__":
    unittest.main()
