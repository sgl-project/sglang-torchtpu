# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Qwen3.5 fused_moe_decode, >256 chunking, and text-only flags."""

import importlib
import importlib.util
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.qwen2_moe import Qwen2MoeSparseMoeBlock
from sglang.srt.plugins.hook_registry import HookType
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import hooks as tpu_hooks
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.decoder_compile_policy import Qwen3VLDecoderTpuCompilePolicy
from sglang_tpu.hooks.fused_moe_decode import bind_qwen_fused_moe_decode

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def _stub_module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)

    def _getattr(attr: str):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return MagicMock()

    mod.__getattr__ = _getattr
    return mod


def _load_fused_moe_decode_op():
    mod_name = "sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    stubs = {}
    if importlib.util.find_spec("jax") is None:
        jax_jnp = _stub_module("jax.numpy")
        jax_sharding = _stub_module(
            "jax.sharding",
            Mesh=object,
            NamedSharding=object,
            PartitionSpec=MagicMock(),
        )
        jax_mod = _stub_module(
            "jax",
            Array=object,
            ShapeDtypeStruct=object,
            numpy=jax_jnp,
            sharding=jax_sharding,
        )
        stubs.update(
            {
                "jax": jax_mod,
                "jax.numpy": jax_jnp,
                "jax.sharding": jax_sharding,
            }
        )
    if importlib.util.find_spec("torch_tpu") is None:
        torch_tpu_pallas = _stub_module(
            "torch_tpu._internal.pallas",
            jax_op=MagicMock(),
            pallas=MagicMock(),
        )
        torch_tpu_mesh = _stub_module(
            "torch_tpu._internal.distributed.device_mesh",
            get_device_assignment=MagicMock(return_value=[]),
        )
        torch_tpu_dist = _stub_module(
            "torch_tpu._internal.distributed",
            device_mesh=torch_tpu_mesh,
        )
        torch_tpu_internal = _stub_module(
            "torch_tpu._internal",
            pallas=torch_tpu_pallas,
            distributed=torch_tpu_dist,
        )
        torch_tpu = _stub_module("torch_tpu", _internal=torch_tpu_internal)
        stubs.update(
            {
                "torch_tpu": torch_tpu,
                "torch_tpu._internal": torch_tpu_internal,
                "torch_tpu._internal.pallas": torch_tpu_pallas,
                "torch_tpu._internal.distributed": torch_tpu_dist,
                "torch_tpu._internal.distributed.device_mesh": torch_tpu_mesh,
            }
        )
    if not stubs:
        return importlib.import_module(mod_name)
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        return importlib.import_module(mod_name)
    finally:
        for name, prev in saved.items():
            if prev is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prev


class TestQwen35FusedMoeDecode(CustomTestCase):
    def _make_moe_block(self) -> Qwen2MoeSparseMoeBlock:
        block = Qwen2MoeSparseMoeBlock.__new__(Qwen2MoeSparseMoeBlock)
        torch.nn.Module.__init__(block)
        return block

    def test_hook_registration_and_binding(self):
        expected = (
            "sglang.srt.models.qwen2_moe.Qwen2MoeSparseMoeBlock.__init__",
            bind_qwen_fused_moe_decode,
            HookType.AFTER,
        )
        self.assertIn(expected, tpu_hooks.HOOKS)

        mod_name = "sglang_tpu.moe.pallas.fused_moe_decode.fused_moe_decode_op"
        noop_block = self._make_moe_block()
        saved_mod = sys.modules.pop(mod_name, None)
        try:
            with patch("importlib.util.find_spec", return_value=None):
                bind_qwen_fused_moe_decode(None, noop_block)
            self.assertIsNone(noop_block._tpu_fused_moe_decode_should_use)
            self.assertIsNone(noop_block._tpu_fused_moe_decode_apply)
        finally:
            if saved_mod is not None:
                sys.modules[mod_name] = saved_mod

        fmd = _load_fused_moe_decode_op()
        block = self._make_moe_block()
        self.assertFalse(block.use_tpu_fused_moe_decode(torch.zeros(4, 8), None))
        bind_qwen_fused_moe_decode(None, block)
        self.assertIs(
            block._tpu_fused_moe_decode_should_use, fmd.should_use_fused_moe_decode
        )
        self.assertIs(block._tpu_fused_moe_decode_apply, fmd.apply_fused_moe_decode)

    def test_use_tpu_fused_moe_decode_delegates_and_guards_defer_finalize(self):
        block = self._make_moe_block()
        hidden = torch.randn(4, 8)
        batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

        self.assertFalse(block.use_tpu_fused_moe_decode(hidden, batch))

        should_use = MagicMock(return_value=True)
        block._tpu_fused_moe_decode_should_use = should_use
        self.assertTrue(block.use_tpu_fused_moe_decode(hidden, batch))
        should_use.assert_called_once_with(block, hidden, batch)

        should_use.reset_mock()
        self.assertFalse(
            block.use_tpu_fused_moe_decode(hidden, batch, defer_finalize=True)
        )
        should_use.assert_not_called()

    def test_qwen2_moe_forward_dispatches_before_mega_moe(self):
        block = self._make_moe_block()
        block._use_mega_moe = False
        block.gate = lambda h: (torch.ones(h.shape[0], 4), None)
        block.topk = lambda h, logits: (
            torch.ones(h.shape[0], 2),
            torch.zeros(h.shape[0], 2, dtype=torch.int32),
            None,
        )
        block._tpu_fused_moe_decode_should_use = lambda layer, h, fb: True
        expected = torch.full((6, 8), 3.0)
        apply_mock = MagicMock(return_value=expected)
        block._tpu_fused_moe_decode_apply = apply_mock

        out = block.forward(
            torch.randn(6, 8), SimpleNamespace(forward_mode=ForwardMode.DECODE)
        )
        torch.testing.assert_close(out, expected)
        apply_mock.assert_called_once()

    def test_apply_fused_moe_decode_chunks_above_256_and_passes_shared_gate(self):
        fmd = _load_fused_moe_decode_op()
        experts = SimpleNamespace(
            w13_weight=torch.zeros(1),
            _fmd_w1_scale=torch.ones(1),
            w2_weight=torch.zeros(1),
            _fmd_w2_scale=torch.ones(1),
            _fmd_sw1=torch.zeros(1),
            _fmd_s1s=torch.ones(1),
            _fmd_sw2=torch.zeros(1),
            _fmd_s2s=torch.ones(1),
            _fmd_sgw=torch.ones(8, 1),
            _fmd_rank=torch.zeros(1, dtype=torch.int32),
        )
        block = SimpleNamespace(
            experts=experts,
            _tpu_fused_moe_decode_world_size=1,
            _tpu_fused_moe_decode_routed_scale=1.0,
            _tpu_fused_moe_decode_swiglu_limit=None,
            _tpu_fused_moe_decode_has_shared=True,
            _tpu_fused_moe_decode_has_shared_gate=True,
            _tpu_fused_moe_decode_ep_rank=0,
        )

        recorded_calls = []

        def fake_op(*args):
            recorded_calls.append(args)
            return args[0] + 1.0

        hidden = torch.arange(512 * 8, dtype=torch.bfloat16).reshape(512, 8)
        topk_w = torch.ones(512, 2, dtype=torch.float32)
        topk_ids = torch.zeros(512, 2, dtype=torch.int32)

        with patch.object(fmd, "_get_op", return_value=fake_op) as get_op_mock:
            out = fmd.apply_fused_moe_decode(block, hidden, (topk_w, topk_ids, None))

        get_op_mock.assert_called_once_with(
            1, 1.0, None, has_shared=True, has_shared_gate=True
        )
        self.assertEqual(len(recorded_calls), 2)
        for call_args in recorded_calls:
            self.assertEqual(len(call_args), 13)
            self.assertEqual(call_args[0].shape, (256, 8))
            self.assertIs(call_args[11], experts._fmd_sgw)
        torch.testing.assert_close(out, hidden + 1.0)

    def test_serves_multimodal_respects_language_model_only_and_mixed_cases(self):
        outer = SimpleNamespace(
            get_image_feature=MagicMock(),
            get_input_embeddings=lambda: torch.nn.Embedding(4, 8),
            language_model_only=False,
        )
        mr = SimpleNamespace(
            model=outer,
            server_args=SimpleNamespace(language_model_only=False, language_only=False),
        )
        policy = Qwen3VLDecoderTpuCompilePolicy(
            model_runner=mr,
            t_bucketer=Bucketer(name="Token", buckets=[64, 512]),
            b_bucketer=Bucketer(name="Bs", buckets=[8, 32]),
            owner=outer,
        )
        policy.extend_bs = 8

        self.assertTrue(policy._serves_multimodal)
        self.assertTrue(policy._uses_input_embeds(outer))

        mr.server_args.language_model_only = True
        self.assertFalse(policy._serves_multimodal)
        self.assertFalse(policy._uses_input_embeds(outer))

        with patch(
            "sglang_tpu.compilation.decoder_compile_policy.get_schedule",
            return_value=SimpleNamespace(enable_mixed_chunk=True),
        ):
            cases = policy._extend_compile_cases()
        mixed_bs = {bs for _, bs, mode in cases if mode == ForwardMode.MIXED}
        self.assertEqual(mixed_bs, {8, 32})

    def test_language_model_only_architecture_allowlist_includes_qwen35(self):
        self.assertIn(
            "Qwen3_5ForConditionalGeneration",
            ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES,
        )
        self.assertIn(
            "Qwen3_5MoeForConditionalGeneration",
            ServerArgs.LANGUAGE_MODEL_ONLY_ARCHITECTURES,
        )


if __name__ == "__main__":
    unittest.main()
