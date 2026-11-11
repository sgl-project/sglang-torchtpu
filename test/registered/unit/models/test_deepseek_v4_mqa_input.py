"""Early MQA preparation must bypass projections without skipping source work."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.deepseek_v4 as model
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _FakePlatform:
    def __init__(self, oot):
        self.oot = oot

    def is_out_of_tree(self):
        return self.oot


class _Backend(AttentionBackend):
    def prepare_mqa_qkv(self, layer, x, q_lora, qkv_a, positions, q_out):
        # Stand-in for device operations; non-idempotent normalization exposes
        # accidental re-preparation of an already prepared result.
        normalized = q_lora / 2
        q = normalized.unsqueeze(1) + positions[:, None, None]
        kv = (qkv_a[:, 2:] if qkv_a is not None else x[:, 2:]) / 3
        if q_out is not None:
            q_out.copy_(q)
            q = q_out
        return normalized, q, kv


class _EarlyBackend(_Backend):
    def __init__(self, enabled=True, stored=False):
        self.enabled = enabled
        self.stored = stored

    def prepare_mqa_input(self, layer, x, positions, forward_batch, **buffers):
        if not self.enabled:
            return None
        assert forward_batch.forward_mode == ForwardMode.DECODE
        normalized = x[:, :2] / 2
        q = normalized.unsqueeze(1) + positions[:, None, None]
        kv = x[:, 2:] / 3
        if buffers["q_out"] is not None:
            buffers["q_out"].copy_(q)
            q = buffers["q_out"]
        return model.MQAInputPreparation(normalized, q, None if self.stored else kv)


def _layer(fused, forbid_projection=False):
    layer = model.MQALayer.__new__(model.MQALayer)
    torch.nn.Module.__init__(layer)
    layer.fuse_wqa_wkv = fused
    layer.q_lora_rank = 2
    layer.use_fused_qk_norm_rope = False
    layer.compress_ratio = 4
    layer.compressor = None
    layer.indexed = None

    def project(x):
        if forbid_projection:
            raise AssertionError("ordinary projection ran before early preparation")
        return x if fused else x[:, :2], None

    def indexer(**kwargs):
        layer.indexed = kwargs["q_lora"].clone()

    layer.wqkv_a = project
    layer.wq_a = project
    layer.indexer = indexer
    return layer


def _run(layer, backend, oot, x, x_quant=None, q_out=None):
    positions = torch.arange(x.shape[0])
    with (
        patch("sglang.srt.platforms.current_platform", _FakePlatform(oot)),
        patch.object(model, "is_cp_active", return_value=False),
        patch.object(model, "_is_npu", False),
        patch.object(env_gate, "is_unified_kv_triton", return_value=False),
        patch.object(env_gate, "is_unified_kv_fp8", return_value=False),
    ):
        return layer._forward_prepare(
            x,
            positions,
            SimpleNamespace(forward_mode=ForwardMode.DECODE),
            backend,
            x_quant=x_quant,
            q_out=q_out,
        )


class TestMQAInputPreparation(CustomTestCase):
    def test_fallback_and_in_tree_gate_preserve_projection_results(self):
        # Missing override, explicit decline, and in-tree platforms must all
        # retain fused/unfused projection and use x_quant rather than x.
        x = torch.zeros(3, 4)
        quant = torch.arange(12, dtype=torch.float32).view(3, 4)
        for fused in (False, True):
            for backend, oot in (
                (_Backend(), True),
                (_EarlyBackend(enabled=False), True),
                (_EarlyBackend(), False),
            ):
                with self.subTest(fused=fused, backend=type(backend), oot=oot):
                    if not oot:

                        def forbidden(*args, **kwargs):
                            raise AssertionError("in-tree platform used early hook")

                        backend.prepare_mqa_input = forbidden
                    layer = _layer(fused)
                    q, kv = _run(layer, backend, oot, x, x_quant=quant)
                    expected = quant[:, :2] / 2
                    torch.testing.assert_close(
                        q, expected.unsqueeze(1) + torch.arange(3)[:, None, None]
                    )
                    torch.testing.assert_close(kv, quant[:, 2:] / 3)
                    torch.testing.assert_close(layer.indexed, expected)

    def test_prepared_outputs_skip_projection_and_normalization_keep_indexer(self):
        x = torch.zeros(3, 4)
        quant = torch.arange(12, dtype=torch.float32).view(3, 4)
        reference = _layer(True)
        expected_q, expected_kv = _run(reference, _Backend(), False, quant)
        for stored in (False, True):
            with self.subTest(stored=stored):
                layer = _layer(True, forbid_projection=True)
                output = torch.empty_like(expected_q)
                q, kv = _run(
                    layer,
                    _EarlyBackend(stored=stored),
                    True,
                    x,
                    x_quant=quant,
                    q_out=output,
                )
                self.assertIs(q, output)
                torch.testing.assert_close(q, expected_q)
                if stored:
                    self.assertIsNone(kv)
                else:
                    torch.testing.assert_close(kv, expected_kv)
                torch.testing.assert_close(layer.indexed, reference.indexed)


if __name__ == "__main__":
    unittest.main()
