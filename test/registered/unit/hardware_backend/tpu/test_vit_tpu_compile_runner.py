import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    VisionTpuCompilePolicy,
    ViTTpuCompileRunner,
    _cu_seqlens_to_segment_ids,
)
from sglang.srt.layers.attention.vision import (
    VisionTpuAttention,
    _build_tpu_segment_ids,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")


class _SegmentBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SimpleNamespace(
            qkv_backend=SimpleNamespace(supports_segment_ids=True)
        )
        self.graph_cu_seqlens = None
        self.segment_ids = None

    def forward(
        self,
        x,
        cu_seqlens,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
        segment_ids=None,
    ):
        self.graph_cu_seqlens = cu_seqlens
        self.segment_ids = segment_ids
        return x


class _LegacyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = SimpleNamespace(
            qkv_backend=SimpleNamespace(supports_segment_ids=False)
        )
        self.cu_seqlens = None

    def forward(
        self,
        x,
        cu_seqlens,
        rotary_pos_emb_cos,
        rotary_pos_emb_sin,
    ):
        self.cu_seqlens = cu_seqlens
        return x


def _make_vit(block):
    return SimpleNamespace(
        blocks=[block],
        merger=nn.Identity(),
        deepstack_visual_indexes=[],
        deepstack_merger_list=None,
        spatial_merge_size=1,
    )


class TestViTTpuCompileRunner(unittest.TestCase):
    def test_policy_declares_tpu_platform(self):
        self.assertEqual(VisionTpuCompilePolicy.platform, "tpu")

    def test_segment_ids_are_built_from_host_cumulative_lengths(self):
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

        segment_ids = _cu_seqlens_to_segment_ids(
            cu_seqlens, total_patches=5, device=torch.device("cpu")
        )

        self.assertEqual(segment_ids.dtype, torch.int32)
        self.assertEqual(segment_ids.tolist(), [[0, 0, 1, 1, 1]])

    def test_shared_tpu_attention_builds_packed_segment_ids(self):
        segment_ids = _build_tpu_segment_ids(
            torch.tensor([0, 2, 5], dtype=torch.int32),
            bsz=1,
            seq_len=5,
            flatten_batch=True,
            device=torch.device("cpu"),
        )

        self.assertEqual(segment_ids.tolist(), [[0, 0, 1, 1, 1]])

        segment_ids = _build_tpu_segment_ids(
            torch.tensor([0, 0, 2, 2, 5], dtype=torch.int32),
            bsz=1,
            seq_len=5,
            flatten_batch=True,
            device=torch.device("cpu"),
        )

        self.assertEqual(segment_ids.tolist(), [[1, 1, 3, 3, 3]])

    def test_shared_tpu_attention_separates_dense_padding(self):
        segment_ids = _build_tpu_segment_ids(
            torch.tensor([0, 3, 5], dtype=torch.int32),
            bsz=2,
            seq_len=3,
            flatten_batch=False,
            device=torch.device("cpu"),
        )

        self.assertEqual(segment_ids.tolist(), [[0, 0, 0], [0, 0, 1]])

    def test_shared_tpu_attention_supports_batched_flattened_callers(self):
        flash_op = MagicMock(side_effect=lambda q, k, v, ids: torch.zeros_like(q))
        with (
            patch("sglang.srt.layers.attention.vision._is_tpu", True),
            patch(
                "sglang.srt.layers.attention.vision._get_vision_flash_op",
                return_value=flash_op,
            ),
        ):
            attention = VisionTpuAttention(
                head_dim=4,
                num_heads=1,
                num_kv_heads=1,
                flatten_batch=True,
            )
        q = torch.zeros(6, 1, 4)

        result = attention(q, q, q, bsz=2)

        self.assertEqual(tuple(result.shape), (6, 1, 4))
        segment_ids = flash_op.call_args.args[-1]
        self.assertEqual(tuple(segment_ids.shape), (2, 128))
        self.assertEqual(segment_ids[:, :3].tolist(), [[0, 0, 0], [0, 0, 0]])
        self.assertTrue(torch.all(segment_ids[:, 3:] == 1))

    def test_shared_tpu_attention_delegates_explicit_masks_to_sdpa(self):
        with (
            patch("sglang.srt.layers.attention.vision._is_tpu", True),
            patch(
                "sglang.srt.layers.attention.vision._get_vision_flash_op",
                return_value=MagicMock(),
            ),
        ):
            attention = VisionTpuAttention(
                head_dim=4,
                num_heads=2,
                num_kv_heads=1,
                flatten_batch=True,
            )
        expected = torch.zeros(6, 2, 4)
        attention._sdpa_fallback.forward = MagicMock(return_value=expected)
        q = torch.zeros(6, 2, 4)
        kv = torch.zeros(6, 1, 4)
        mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)

        result = attention(q, kv, kv, bsz=2, attention_mask=mask)

        self.assertIs(result, expected)
        attention._sdpa_fallback.forward.assert_called_once()
        _, expanded_k, expanded_v, _ = attention._sdpa_fallback.forward.call_args.args
        self.assertEqual(expanded_k.shape[1], 2)
        self.assertEqual(expanded_v.shape[1], 2)

        with self.assertRaisesRegex(ValueError, "cannot combine"):
            attention(
                q,
                kv,
                kv,
                bsz=2,
                attention_mask=mask,
                segment_ids=torch.zeros(2, 3, dtype=torch.int32),
            )

    def test_shared_tpu_attention_uses_pallas_for_fp32_softmax(self):
        flash_op = MagicMock(side_effect=lambda q, k, v, ids: torch.zeros_like(q))
        with (
            patch("sglang.srt.layers.attention.vision._is_tpu", True),
            patch(
                "sglang.srt.layers.attention.vision._get_vision_flash_op",
                return_value=flash_op,
            ),
        ):
            attention = VisionTpuAttention(
                head_dim=4,
                num_heads=1,
                num_kv_heads=1,
                flatten_batch=True,
                softmax_in_single_precision=True,
            )
        attention._sdpa_fallback.forward = MagicMock()
        q = torch.zeros(3, 1, 4)

        result = attention(q, q, q, bsz=1)

        self.assertEqual(tuple(result.shape), (3, 1, 4))
        flash_op.assert_called_once()
        attention._sdpa_fallback.forward.assert_not_called()

    def test_shared_tpu_attention_expands_kv_heads(self):
        flash_op = MagicMock(side_effect=lambda q, k, v, ids: torch.zeros_like(q))
        with (
            patch("sglang.srt.layers.attention.vision._is_tpu", True),
            patch(
                "sglang.srt.layers.attention.vision._get_vision_flash_op",
                return_value=flash_op,
            ),
        ):
            attention = VisionTpuAttention(
                head_dim=4,
                num_heads=2,
                num_kv_heads=1,
                flatten_batch=True,
            )
        q = torch.zeros(3, 2, 4)
        kv = torch.zeros(3, 1, 4)

        attention(q, kv, kv, bsz=1)

        _, k, v, _ = flash_op.call_args.args
        self.assertEqual(k.shape[1], 2)
        self.assertEqual(v.shape[1], 2)

    def test_eager_segment_backend_receives_segment_ids(self):
        block = _SegmentBlock()
        runner = ViTTpuCompileRunner(_make_vit(block))
        x = torch.zeros(5, 4)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        rotary = torch.zeros(5, 2)

        result = runner.run(
            x,
            cu_seqlens,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )

        self.assertEqual(tuple(result.shape), (5, 1, 4))
        self.assertEqual(block.graph_cu_seqlens.tolist(), [0, 5])
        self.assertEqual(block.segment_ids.tolist(), [[0, 0, 1, 1, 1]])

    def test_eager_legacy_backend_keeps_original_cumulative_lengths(self):
        block = _LegacyBlock()
        runner = ViTTpuCompileRunner(_make_vit(block))
        x = torch.zeros(5, 4)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        rotary = torch.zeros(5, 2)

        runner.run(
            x,
            cu_seqlens,
            rotary_pos_emb_cos=rotary,
            rotary_pos_emb_sin=rotary,
        )

        self.assertIs(block.cu_seqlens, cu_seqlens)


if __name__ == "__main__":
    unittest.main()
