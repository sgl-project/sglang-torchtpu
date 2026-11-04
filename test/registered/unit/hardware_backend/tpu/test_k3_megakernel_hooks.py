"""Kimi-K3 decode megakernel hooks (sglang_tpu/hooks/k3_megakernel.py): registration,
step routing and the startup checks, without a TPU or the kernel."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import contextlib
import importlib.util
import pkgutil
import sys
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
        self.assertEqual(len(targets), 7)
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
        self.assertEqual(_forward(model, ForwardMode.TARGET_VERIFY, tokens=8), "kernel")
        for mode, bs, tokens in (
            (ForwardMode.DECODE, 2, 2),
            (ForwardMode.EXTEND, 2, 9),
            (ForwardMode.TARGET_VERIFY, 2, 16),
        ):
            with self.subTest(mode=mode, bs=bs), self.assertRaises(NotImplementedError):
                _forward(model, mode, bs, tokens)

    def test_startup_rejects_configs_the_kernel_cannot_run(self):
        ok = dict(
            max_running_requests=1, disable_radix_cache=True, speculative_algorithm=None,
            enable_mixed_chunk=False, chunked_prefill_size=1024, page_size=128,
            pp_size=1, dp_size=1, attn_cp_size=1, dcp_size=1, kv_cache_dtype="auto",
            attention_backend="pallas_tpu_mla", tpu_compile_bs=[1],
            context_length=None,
        )  # fmt: skip

        def check(dsa=False, **kw):
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
                hooks._check_server_args(dsa=dsa)

        check()
        check(speculative_algorithm="DSPARK")
        check(speculative_algorithm="DFLASH")
        for kw, needle in (
            ({"max_running_requests": 4}, "--max-running-requests 1"),
            ({"disable_radix_cache": False}, "--disable-radix-cache"),
            ({"speculative_algorithm": "EAGLE"}, "other than DSPARK"),
            ({"chunked_prefill_size": -1}, "--chunked-prefill-size > 0"),
            ({"kv_cache_dtype": "fp8_e4m3"}, "bf16 KV cache"),
            ({"tpu_compile_bs": [1, 4]}, "TPU compile batch sizes"),
        ):
            with self.subTest(**kw), self.assertRaisesRegex(ValueError, needle):
                check(**kw)

        # Kimi-K3.1: the kernel owns its pools, sized by --context-length; sglang's pool is the
        # sparse backend's, whatever its dtype.
        k31 = dict(attention_backend="pallas_tpu_mla_sparse", context_length=8192)
        check(dsa=True, **k31)
        check(dsa=True, kv_cache_dtype="fp8_e4m3", **k31)
        for kw, needle in (
            ({"attention_backend": "pallas_tpu_mla"}, "pallas_tpu_mla_sparse"),
            ({"context_length": None}, "--context-length"),
        ):
            with (
                self.subTest(dsa=True, **kw),
                self.assertRaisesRegex(ValueError, needle),
            ):
                check(dsa=True, **{**k31, **kw})

    def test_k31_indexer_load_check_skipped_only_with_owned_weights(self):
        original = MagicMock(return_value="checked")
        with _FLAG.override(True):
            self.assertIsNone(hooks.skip_k31_indexer_load_check(original, None, set()))
        original.assert_not_called()
        with _FLAG.override(False):
            self.assertEqual(
                hooks.skip_k31_indexer_load_check(original, None, set()), "checked"
            )

    def test_k31_kv_len_bound_leaves_real_lengths_alone(self):
        """Kernel-owned pools bound only lengths no real request has: past the pools (warmup
        dummies carry sglang's page-table length) or shorter than the step's rows."""
        from sglang_tpu.k3_megakernel.ops import K3MegakernelRuntime

        bound = lambda capacity, kv, lo: int(  # noqa: E731
            K3MegakernelRuntime._bound_kv_len(
                SimpleNamespace(capacity=capacity, _kv_tail=8),
                torch.tensor(kv, dtype=torch.int32),
                lo,
            )
        )
        # A 3-token prompt (prefill lo = n = 3) keeps kv_len 3: prefix 0, fresh request.
        self.assertEqual(bound(4096, 3, torch.tensor(3, dtype=torch.int32)), 3)
        self.assertEqual(bound(4096, 2, 1), 2)  # decode after a 1-token prompt
        # Verify of 8 rows after 1 committed token.
        self.assertEqual(bound(4096, 9, 8), 9)
        self.assertEqual(bound(4096, 0, 8), 8)  # warmup dummies
        self.assertEqual(bound(4096, 10**6, 1), 4096 - 8)
        # Kimi-K3: sglang's pools, untouched.
        self.assertEqual(bound(0, 10**6, 1), 10**6)

    def test_dspark_capture_builds_the_capturing_kernels(self):
        runtime = MagicMock()
        causal_lm = SimpleNamespace(model=SimpleNamespace(_k3_megakernel=runtime))
        spec = SimpleNamespace(speculative_num_draft_tokens=8)
        with patch("sglang.srt.runtime_context.get_spec", return_value=spec):
            hooks.prepare_k3_capture_kernels(None, causal_lm, [51, 7, 23])
            hooks.prepare_k3_capture_kernels(
                None, SimpleNamespace(model=SimpleNamespace()), [7]
            )
        runtime.prepare.assert_called_once_with((7, 23, 51), verify_rows=8)

    def test_verify_rows_use_the_decode_down_projection(self):
        """Verify variants default to moe_down_dequant=False (a row's MoE output must not depend
        on later draft rows); an explicit KERNEL_OPTS value still wins."""
        from sglang_tpu.k3_megakernel.ops import K3MegakernelRuntime

        built = []
        stack = SimpleNamespace(StackKernel=lambda spec, **kw: built.append(kw))
        runtime = SimpleNamespace(
            spec=None, opts={}, verify_mode="rows", variants=[], _variant_index={}
        )
        with patch.dict(sys.modules, {"sglang_tpu.k3_megakernel.kernel.stack": stack}):
            K3MegakernelRuntime._add_variant(runtime, "verify", (7,), 8)
            runtime.opts = {"moe_down_dequant": True}
            K3MegakernelRuntime._add_variant(runtime, "verify", (7,), 4)
        self.assertEqual([kw["moe_down_dequant"] for kw in built], [False, True])

    @unittest.skipUnless(
        importlib.util.find_spec("jax"), "kernel_options checks StackKernel's signature"
    )
    def test_serving_refuses_debug_and_broken_kernel_options(self):
        from sglang_tpu.k3_megakernel.ops import SERVING_KERNEL_OPTIONS, kernel_options

        serving = dict(allowed=SERVING_KERNEL_OPTIONS)
        self.assertEqual(kernel_options("moe_ring=6", **serving), {"moe_ring": 6})
        for bad in ("moe_shared_after=1", "debug_repeat=2", "collective_id=7", "x=1"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                kernel_options(bad, **serving)

    def test_down_scale_family_is_padded_to_16_rows(self):
        """Serving uploads expert_down_scale with 16 rows (the kernel reads the first 12), the
        row count whose default torch_tpu layout is the row-major one the kernel reads."""
        import numpy as np

        from sglang_tpu.k3_megakernel.ops import _down_scale_rows, _pad_down_scale

        self.assertEqual((_down_scale_rows(12), _down_scale_rows(16)), (16, 16))
        scale = np.arange(2 * 12 * 4, dtype=np.uint8).reshape(2, 12, 4)
        other = np.ones((3,), np.uint8)
        out = _pad_down_scale({"expert_down_scale": scale, "expert_gu": other})
        self.assertEqual(out["expert_down_scale"].shape, (2, 16, 4))
        np.testing.assert_array_equal(out["expert_down_scale"][:, :12], scale)
        self.assertFalse(out["expert_down_scale"][:, 12:].any())
        self.assertIs(out["expert_gu"], other)

    def test_prefill_rows_cover_every_extend(self):
        from sglang_tpu.k3_megakernel.ops import prefill_rows

        self.assertEqual(prefill_rows(1024, "auto"), 1024)
        for size, mode in ((-1, "auto"), (None, "auto"), (1020, "rows")):
            with self.subTest(size=size, mode=mode), self.assertRaises(ValueError):
                prefill_rows(size, mode)

    def test_prefill_block(self):
        """Rows mode needs whole prefill blocks; the block is a positive multiple of 8."""
        from sglang_tpu.k3_megakernel.ops import _dsa_capacity, prefill_rows

        block = tpu_envs.SGLANG_TPU_K3_MEGAKERNEL_PREFILL_BLOCK
        with block.override(8):
            self.assertEqual(prefill_rows(1032, "rows"), 1032)
        with block.override(16), self.assertRaises(ValueError):
            prefill_rows(1032, "rows")
        with block.override(32):
            # The pools hold the context plus one whole prefill block of padding rows.
            self.assertEqual(_dsa_capacity(2048 - 32), 2048)
            self.assertEqual(_dsa_capacity(2048 - 31), 4096)
        for bad in (0, 12):
            with (
                block.override(bad),
                self.subTest(bad=bad),
                self.assertRaises(ValueError),
            ):
                prefill_rows(1024, "auto")

    def test_kda_rows_follow_the_gdn_pool_layout(self):
        """Extend / decode use the slot row; a verify reads the committed row and writes the
        request's window (as TpuGDNAttnBackend)."""
        from sglang.srt.configs.mamba_utils import (
            Mamba2CacheParams,
            Mamba2StateDType,
            Mamba2StateShape,
        )
        from sglang_tpu.k3_megakernel.ops import kda_state_rows
        from sglang_tpu.mamba_pool import TpuMambaPool

        shape = Mamba2StateShape.create(
            tp_world_size=1, intermediate_size=1024, n_groups=2, num_heads=8,
            head_dim=128, state_size=128, conv_kernel=4,
        )  # fmt: skip
        dtype = Mamba2StateDType(conv=torch.float32, temporal=torch.float32)
        pool = TpuMambaPool(
            size=3, spec_state_size=3, mamba_layer_ids=[1], device="cpu",
            cache_params=Mamba2CacheParams(dtype=dtype, layers=[1], shape=shape),
            speculative_num_draft_tokens=4, speculative_eagle_topk=1,
        )  # fmt: skip
        slot, req = 2, 3
        window = pool.window_start + req * pool.checkpoint_width

        def rows(mode):
            read, base = kda_state_rows(
                pool, mode, torch.tensor(slot, dtype=torch.int32), torch.tensor([req])
            )
            return int(read), int(base)

        self.assertEqual(rows(ForwardMode.DECODE), (slot, slot))
        self.assertEqual(rows(ForwardMode.TARGET_VERIFY), (slot, window))
        pool.select_checkpoints(torch.tensor([slot]), torch.tensor([2]))
        self.assertEqual(rows(ForwardMode.TARGET_VERIFY), (window + 2, window))


if __name__ == "__main__":
    unittest.main()
