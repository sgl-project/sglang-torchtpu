"""Kimi-K3 decode megakernel hooks (sglang_tpu/hooks/k3_megakernel.py): registration,
routing of forward steps and the serving-argument checks, without a TPU or the kernel."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import contextlib
import inspect
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
_K3MK_HOOKS = [h for h in HOOKS if h[1].__module__ == hooks.__name__]


def _server_args(**kw):
    args = dict(
        max_running_requests=1,
        disable_radix_cache=True,
        speculative_algorithm=None,
        enable_mixed_chunk=False,
        chunked_prefill_size=1024,
        page_size=128,
        pp_size=1,
        dp_size=1,
        attn_cp_size=1,
        dcp_size=1,
        kv_cache_dtype="auto",
        attention_backend="pallas_tpu_mla",
        tpu_compile_bs=[1],
    )
    args.update(kw)
    return SimpleNamespace(**args)


def _bags(a):
    """The runtime-context config bags the startup check reads, from flat ``a``."""
    return {
        "get_schedule": SimpleNamespace(
            max_running_requests=a.max_running_requests,
            enable_mixed_chunk=a.enable_mixed_chunk,
            chunked_prefill_size=a.chunked_prefill_size,
            page_size=a.page_size,
        ),
        "get_memory": SimpleNamespace(disable_radix_cache=a.disable_radix_cache),
        "get_spec": SimpleNamespace(speculative_algorithm=a.speculative_algorithm),
        "get_parallel": SimpleNamespace(
            pp_size=a.pp_size,
            dp_size=a.dp_size,
            attn_cp_size=a.attn_cp_size,
            dcp_size=a.dcp_size,
        ),
        "get_model": SimpleNamespace(kv_cache_dtype=a.kv_cache_dtype),
        "get_exec": SimpleNamespace(
            kernel=SimpleNamespace(attention_backend=a.attention_backend),
            graph=SimpleNamespace(tpu_compile_bs=a.tpu_compile_bs),
        ),
    }


def _batch(mode, bs=1):
    return SimpleNamespace(forward_mode=mode, batch_size=bs, _tpu_real_b=bs)


def _model(owns=True):
    model = SimpleNamespace()
    model._k3_megakernel = MagicMock()
    model._k3_megakernel.forward = MagicMock(return_value="kernel")
    model._k3_megakernel_owns_weights = owns
    return model


class TestK3MegakernelHookWiring(CustomTestCase):
    def test_hooks_registered_on_existing_targets(self):
        targets = {t for t, _, _ in _K3MK_HOOKS}
        self.assertEqual(
            targets,
            {
                "sglang.srt.models.kimi_k3.make_pp_layers",
                "sglang.srt.models.kimi_k3.KimiK3LinearModel.forward",
                "sglang.srt.models.kimi_k3.KimiK3ForConditionalGeneration.load_weights",
                "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.load_weights",
                "sglang.srt.models.kimi_k3.KimiK3LinearForCausalLM.post_load_weights",
            },
        )
        for target in targets:
            with self.subTest(target=target):
                self.assertTrue(callable(pkgutil.resolve_name(target)))

    def test_forward_hook_mirrors_target_signature(self):
        target = "sglang.srt.models.kimi_k3.KimiK3LinearModel.forward"
        params = list(inspect.signature(hooks.k3_megakernel_forward).parameters)
        original = inspect.signature(pkgutil.resolve_name(target))
        self.assertEqual(params[:2], ["original", "model"])
        self.assertEqual(params[2:], list(original.parameters)[1:])

    def test_owned_weights_build_placeholder_layers(self):
        from sglang.srt.layers.utils import PPMissingLayer

        def make_pp_layers(num_layers, layer_fn, prefix=""):
            return [layer_fn(i, f"{prefix}.{i}") for i in range(num_layers)]

        real = MagicMock(return_value="decoder layer")
        with _FLAG.override(True):
            layers = hooks.skip_k3_decoder_layers(make_pp_layers, 3, real, prefix="m")
        self.assertTrue(all(isinstance(x, PPMissingLayer) for x in layers))
        real.assert_not_called()
        with _FLAG.override(False):
            self.assertEqual(
                hooks.skip_k3_decoder_layers(make_pp_layers, 2, real, prefix="m"),
                ["decoder layer"] * 2,
            )

    def test_attached_runtime_blocks_online_weight_updates(self):
        causal_lm = SimpleNamespace(model=SimpleNamespace())
        with (
            _FLAG.override(True),
            patch.object(hooks, "_check_server_args"),
            patch(
                "sglang_tpu.k3_megakernel.ops.build_runtime", return_value="runtime"
            ) as build,
        ):
            hooks.attach_k3_megakernel(None, causal_lm)
        build.assert_called_once_with(causal_lm, owns_weights=True)
        self.assertEqual(causal_lm.model._k3_megakernel, "runtime")
        self.assertIn(
            "Online weight updates", causal_lm.model._derived_weight_cache_error
        )

    def test_flag_off_passes_through(self):
        original = MagicMock(return_value="sglang")
        with _FLAG.override(False):
            self.assertFalse(hooks._owns_weights())
            # No runtime attached: the forward is sglang's.
            model = SimpleNamespace()
            out = hooks.k3_megakernel_forward(
                original,
                model,
                torch.zeros(1, dtype=torch.long),
                None,
                _batch(ForwardMode.DECODE),
            )
            self.assertEqual(out, "sglang")
            causal_lm = SimpleNamespace(model=SimpleNamespace())
            hooks.attach_k3_megakernel(None, causal_lm)
            self.assertFalse(hasattr(causal_lm.model, "_k3_megakernel"))


class TestK3MegakernelRouting(CustomTestCase):
    def _forward(self, model, mode, bs=1, tokens=1):
        original = MagicMock(return_value="sglang")
        ids = torch.zeros(tokens, dtype=torch.long)
        return hooks.k3_megakernel_forward(original, model, ids, None, _batch(mode, bs))

    def test_bs1_decode_and_extend_go_to_the_kernel(self):
        model = _model(owns=True)
        self.assertEqual(self._forward(model, ForwardMode.DECODE), "kernel")
        self.assertEqual(self._forward(model, ForwardMode.EXTEND, tokens=17), "kernel")

    def test_ineligible_steps_raise_when_the_kernel_owns_the_weights(self):
        model = _model(owns=True)
        for mode, bs, tokens in (
            (ForwardMode.DECODE, 2, 2),
            (ForwardMode.EXTEND, 2, 9),
            (ForwardMode.TARGET_VERIFY, 1, 8),
            (ForwardMode.MIXED, 1, 4),
        ):
            with self.subTest(mode=mode, bs=bs):
                with self.assertRaises(NotImplementedError):
                    self._forward(model, mode, bs, tokens)

    def test_kept_sglang_weights_run_prefill_on_sglang(self):
        model = _model(owns=False)
        self.assertEqual(self._forward(model, ForwardMode.DECODE), "kernel")
        self.assertEqual(self._forward(model, ForwardMode.EXTEND, tokens=5), "sglang")
        self.assertEqual(
            self._forward(model, ForwardMode.DECODE, bs=2, tokens=2), "sglang"
        )


class TestK3MegakernelServerArgs(CustomTestCase):
    def _check(self, args):
        with contextlib.ExitStack() as stack:
            for name, bag in _bags(args).items():
                stack.enter_context(
                    patch(f"sglang.srt.runtime_context.{name}", return_value=bag)
                )
            hooks._check_server_args()

    def test_serving_config_passes(self):
        self._check(_server_args())

    def test_each_unsupported_arg_is_reported(self):
        for kw, needle in (
            ({"max_running_requests": 4}, "--max-running-requests 1"),
            ({"disable_radix_cache": False}, "--disable-radix-cache"),
            ({"speculative_algorithm": "DSPARK"}, "no speculative decoding"),
            ({"enable_mixed_chunk": True}, "--enable-mixed-chunk"),
            ({"pp_size": 2}, "pp_size = dp_size = 1"),
            ({"chunked_prefill_size": -1}, "--chunked-prefill-size > 0"),
            ({"chunked_prefill_size": None}, "--chunked-prefill-size > 0"),
            ({"page_size": 64}, "--page-size 128"),
            ({"kv_cache_dtype": "fp8_e4m3"}, "bf16 KV cache"),
            ({"attention_backend": "fa3"}, "--attention-backend pallas_tpu_mla"),
            ({"attn_cp_size": 2}, "no context parallelism"),
            ({"dcp_size": 2}, "no context parallelism"),
            ({"tpu_compile_bs": [1, 2, 4]}, "TPU compile batch sizes"),
        ):
            with self.subTest(**kw):
                with self.assertRaisesRegex(ValueError, needle):
                    self._check(_server_args(**kw))


class TestK3MegakernelOptions(CustomTestCase):
    def test_kernel_options_parse_and_reject_unknown_keys(self):
        from sglang_tpu.k3_megakernel.ops import kernel_options

        self.assertEqual(kernel_options(""), {})
        # Unrestricted (benchmark) parsing keeps every StackKernel keyword.
        self.assertEqual(
            kernel_options("moe_ring=6; wire_relax=oproj_flat;fused_head=False"),
            {"moe_ring": 6, "wire_relax": "oproj_flat", "fused_head": False},
        )
        for bad in ("no_such_option=1", "verify_rows=8", "prefill_blocks=2"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                kernel_options(bad)

    def test_serving_refuses_debug_runtime_and_broken_options(self):
        from sglang_tpu.k3_megakernel.ops import (
            SERVING_KERNEL_OPTIONS,
            kernel_options,
        )

        serving = dict(allowed=SERVING_KERNEL_OPTIONS)
        self.assertEqual(
            kernel_options("moe_ring=6;wire_relax=oproj_flat", **serving),
            {"moe_ring": 6, "wire_relax": "oproj_flat"},
        )
        for bad in (
            "moe_shared_after=1",
            "debug_skip=frozenset({'attn'})",
            "debug_repeat=2",
            "moe_debug=no_expert_math",
            "trace=True",
            "layer_out=True",
            "wire_scope=attn",
            "fused_head=True",
            "collective_id=7",
            "capture_layers=(1,)",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                kernel_options(bad, **serving)

    def test_serving_options_are_kernel_keywords(self):
        from sglang_tpu.k3_megakernel.kernel.stack import StackKernel
        from sglang_tpu.k3_megakernel.ops import SERVING_KERNEL_OPTIONS

        accepted = set(inspect.signature(StackKernel.__init__).parameters)
        self.assertLessEqual(SERVING_KERNEL_OPTIONS, accepted)

    def test_prefill_rows_never_below_an_extend(self):
        from sglang_tpu.k3_megakernel.ops import prefill_rows

        self.assertEqual(prefill_rows(1024, "auto"), 1024)
        self.assertEqual(prefill_rows(1000, "loop"), 1000)
        for size, mode in ((-1, "auto"), (0, "auto"), (None, "auto"), (1020, "rows")):
            with self.subTest(size=size, mode=mode), self.assertRaises(ValueError):
                prefill_rows(size, mode)
        with self.assertRaises(ValueError):
            prefill_rows(1024, "token")

    def test_relax_from_env(self):
        from sglang_tpu.k3_megakernel.config import Relax

        self.assertEqual(Relax.parse(()), Relax())
        with tpu_envs.SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX.override(
            "dense_fp8,pair_oproj,bf16_transport,expert_replicas=48"
        ):
            relax = Relax.from_env()
        self.assertEqual(
            relax,
            Relax(
                dense_fp8=True, pair_oproj=True, bf16_transport=True, expert_replicas=48
            ),
        )
        with self.assertRaises(ValueError):
            Relax.parse(("fp4_everything",))


if __name__ == "__main__":
    unittest.main()
