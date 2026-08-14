import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    Qwen3VLVisionTpuCompilePolicy,
    VisionPaddingContext,
    VisionTpuCompilePolicy,
    ViTTpuCompileRunner,
    _cu_seqlens_to_segment_ids,
    _Qwen3VLRunContext,
)
from sglang.srt.layers.attention.vision import (
    VisionTpuAttention,
    _build_tpu_segment_ids,
)
from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit")


@contextlib.contextmanager
def _unset(field):
    """Run with ``field`` unset, restoring whatever was there before.

    ``override`` snapshots and restores on exit, so clearing inside it is safe;
    the Envs API has no override-to-unset of its own.
    """
    with field.override(""):
        field.clear()
        yield


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


class _FakeVisionPolicy(VisionTpuCompilePolicy):
    """Smallest subclass that exercises the base ``run`` seam."""

    def compute_patch_buckets(self):
        return [4]

    def make_warmup_inputs(self, patch_count):
        return (torch.zeros(patch_count),)

    def prepare_call(self, x):
        return (x, torch.tensor([1])), {"real_rows": 2}

    def unpad_outputs(self, result, ctx):
        return result[: ctx["real_rows"]]


class TestViTTpuCompileRunner(unittest.TestCase):
    def test_policy_declares_tpu_platform(self):
        self.assertEqual(VisionTpuCompilePolicy.platform, "tpu")

    def test_qwen3_vl_policy_extends_base_and_implements_every_hook(self):
        # A base hook left unimplemented here would not surface until the policy
        # is instantiated at install time, i.e. in a serving run.
        self.assertEqual(Qwen3VLVisionTpuCompilePolicy.platform, "tpu")
        self.assertTrue(
            issubclass(Qwen3VLVisionTpuCompilePolicy, VisionTpuCompilePolicy)
        )
        self.assertEqual(
            getattr(Qwen3VLVisionTpuCompilePolicy, "__abstractmethods__", frozenset()),
            frozenset(),
        )
        # The base gate must be permissive: an encoder that does not override it
        # should compile, not silently fall back to eager.
        self.assertTrue(
            VisionTpuCompilePolicy._runner_ready_for_compile(None, object())
        )

    def test_base_run_drives_prepare_then_compiled_fn_then_unpad(self):
        policy = _FakeVisionPolicy()
        with self.assertRaisesRegex(RuntimeError, "before install"):
            policy.run(torch.zeros(4))

        policy.compiled_fn = MagicMock(side_effect=lambda x, extra: x * 2)
        result = policy.run(torch.arange(5).float())

        # prepare_call's tuple is splatted into the compiled fn ...
        self.assertEqual(policy.compiled_fn.call_args.args[1].tolist(), [1])
        # ... and its ctx drives unpad_outputs on the way back.
        self.assertEqual(result.tolist(), [0.0, 2.0])

    def test_qwen3_vl_patch_buckets_honor_the_env_override(self):
        policy = Qwen3VLVisionTpuCompilePolicy(SimpleNamespace(spatial_merge_size=2))
        sizes = envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES

        with _unset(sizes), _unset(envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES):
            ladder = policy.compute_patch_buckets()
        self.assertEqual((ladder[0], ladder[-1]), (196, 4900))

        # Sorted and de-duplicated, and validated against the merge window.
        with sizes.override("324, 196 ,196"):
            self.assertEqual(policy.compute_patch_buckets(), [196, 324])
        # Set but empty disables precompile.
        with sizes.override(""):
            self.assertEqual(policy.compute_patch_buckets(), [])
        with sizes.override("200"), self.assertRaisesRegex(
            ValueError, "perfect squares"
        ):
            policy.compute_patch_buckets()

        # resolve_patch_buckets caches, so a later env change is not picked up.
        with sizes.override("196"):
            self.assertEqual(policy.resolve_patch_buckets(), [196])
        with sizes.override("324"):
            self.assertEqual(policy.resolve_patch_buckets(), [196])

    def test_qwen3_vl_unpad_trims_bucket_padding_and_selects_rows(self):
        policy = Qwen3VLVisionTpuCompilePolicy(SimpleNamespace(spatial_merge_size=2))
        padded = torch.arange(8).float()
        # 12 real patches in a 16 bucket, 2x2 merge => 3 real output rows.
        padding = VisionPaddingContext(
            real_seq_len=12, bucket_len=16, padded=True, merge_sq=4
        )

        trimmed = policy.unpad_outputs(
            padded, _Qwen3VLRunContext(padding=padding, output_indices=None)
        )
        self.assertEqual(trimmed.tolist(), [0.0, 1.0, 2.0])

        selected = policy.unpad_outputs(
            padded,
            _Qwen3VLRunContext(padding=padding, output_indices=torch.tensor([2, 0])),
        )
        self.assertEqual(selected.tolist(), [2.0, 0.0])

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
