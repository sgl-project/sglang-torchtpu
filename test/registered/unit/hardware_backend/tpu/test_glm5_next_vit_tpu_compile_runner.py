# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's TPU vision runner matches Glm5NextVisionModel.forward."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import importlib.util
import sys
import types

if importlib.util.find_spec("torch_tpu") is None:
    _torch_tpu = types.ModuleType("torch_tpu")
    _internal = types.ModuleType("torch_tpu._internal")
    _sync_pkg = types.ModuleType("torch_tpu._internal.sync")
    _sync_mod = types.ModuleType("torch_tpu._internal.sync.sync")
    _sync_mod.synchronize = lambda: None
    _compile_mod = types.ModuleType("torch_tpu._internal.compile")
    sys.modules.setdefault("torch_tpu", _torch_tpu)
    sys.modules.setdefault("torch_tpu._internal", _internal)
    sys.modules.setdefault("torch_tpu._internal.sync", _sync_pkg)
    sys.modules.setdefault("torch_tpu._internal.sync.sync", _sync_mod)
    sys.modules.setdefault("torch_tpu._internal.compile", _compile_mod)

from sglang.srt.environ import envs as sglang_envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.graph_runner.glm5_next_vit_tpu_compile_runner import (
    Glm5NextVisionTpuCompilePolicy,
    Glm5NextVitTpuCompileRunner,
    _rot_pos_ids,
    patch_buckets,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

HEADS, HEAD_DIM, HIDDEN, OUT_HIDDEN, INTERMEDIATE, PATCH = 2, 16, 32, 24, 48, 4


class _SegmentSdpaBackend(torch.nn.Module):
    """Segment-id attention stand-in for VisionTpuAttention on CPU."""

    supports_segment_ids = True

    def forward(self, q, k, v, bsz, cu_seqlens=None, segment_ids=None, **kwargs):
        del kwargs
        if segment_ids is None:
            from sglang.srt.layers.attention.vision import _build_tpu_segment_ids

            segment_ids = _build_tpu_segment_ids(
                cu_seqlens, bsz, q.shape[0], flatten_batch=True, device=q.device
            )
        ids = segment_ids[0]
        mask = ids[:, None] == ids[None, :]
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1),
            k.transpose(0, 1),
            v.transpose(0, 1),
            attn_mask=mask,
        )
        return out.transpose(0, 1)


def _unwrapped(fn):
    return getattr(fn, "__wrapped__", fn)


def _build_vit(use_data_parallel: bool = False):
    from sglang.srt.layers.rotary_embedding import base as rope_base
    from sglang.srt.models import glm5_next
    from sglang.srt.runtime_context import get_parallel
    from sglang.test.test_utils import publish_build_topology

    from sglang.srt.layers.attention import vision

    rope_base._is_tpu = True
    vision.apply_rotary_pos_emb = vision.apply_rotary_pos_emb_native_eager
    for k in list(vision.ROTARY_EMBED_CLASSES):
        vision.ROTARY_EMBED_CLASSES[k] = vision.apply_rotary_pos_emb_native_eager
    publish_build_topology(tp_size=1)
    get_parallel().override_permanently(tp_group=None, attn_tp_group=None)

    if hasattr(glm5_next.swiglu_clamped, "__wrapped__"):
        glm5_next.swiglu_clamped = glm5_next.swiglu_clamped.__wrapped__

    vision_config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_heads=HEADS,
        patch_size=PATCH,
        temporal_patch_size=1,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=OUT_HIDDEN,
        intermediate_size=INTERMEDIATE,
        projection_intermediate_size=INTERMEDIATE,
        depth=2,
        rms_norm_eps=1e-5,
        attention_bias=True,
        swiglu_limit=7.0,
    )
    torch.manual_seed(0)
    vit = glm5_next.Glm5NextVisionModel(
        vision_config,
        quant_config=None,
        prefix="visual",
        use_data_parallel=True,
    ).float()
    vit.use_data_parallel = use_data_parallel
    with torch.no_grad():
        for param in vit.parameters():
            param.copy_(torch.randn_like(param) * 0.2)
    _enable_segment_backends(vit)
    runner = Glm5NextVitTpuCompileRunner(vit)
    vit.graph_runners = runner
    return vit, runner


def _enable_segment_backends(vit) -> None:
    for block in vit.blocks:
        block.attn.qkv_backend = _SegmentSdpaBackend()


class TestGlm5NextVitTpuCompileRunner(CustomTestCase):
    def test_rot_pos_ids_matches_vision_model_rot_pos_emb(self):
        vit, _ = _build_vit()
        grids = torch.tensor([[1, 4, 6], [2, 2, 4]], dtype=torch.int32)
        _, _, want_ids = vit.rot_pos_emb(grids)
        got_ids = torch.cat(
            [_rot_pos_ids(4, 6, 2), _rot_pos_ids(2, 4, 2).repeat(2, 1)], dim=0
        )
        self.assertTrue(torch.equal(got_ids, want_ids.cpu()))

    def test_staged_runner_matches_unhooked_vision_model_across_padded_buckets(self):
        vit, runner = _build_vit()
        grids = torch.tensor(
            [[1, 4, 6], [1, 2, 2], [1, 8, 6]], dtype=torch.int32
        )
        total_patches = int(grids.prod(dim=1).sum())
        in_features = 3 * 1 * PATCH * PATCH
        pixels = torch.randn(total_patches, in_features, dtype=torch.float32)

        with torch.no_grad():
            want = _unwrapped(type(vit).forward)(vit, pixels, grids)

        _enable_segment_backends(vit)
        self.assertTrue(runner.ready)
        runner._patch_buckets = [16, 36, 64]

        with (
            torch.no_grad(),
            sglang_envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False),
        ):
            got = runner.run_grid(pixels, grids)

        self.assertIsNotNone(got)
        self.assertEqual(got.shape, (6 + 1 + 12, OUT_HIDDEN))
        self.assertEqual(got.device.type, "cpu")
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_video_grid_with_multiple_frames_matches_unhooked_vision_model(self):
        vit, runner = _build_vit()
        grids = torch.tensor([[2, 4, 4], [1, 2, 4]], dtype=torch.int32)
        total_patches = int(grids.prod(dim=1).sum())
        in_features = 3 * 1 * PATCH * PATCH
        pixels = torch.randn(total_patches, in_features, dtype=torch.float32)

        with torch.no_grad():
            want = _unwrapped(type(vit).forward)(vit, pixels, grids)

        _enable_segment_backends(vit)
        runner._patch_buckets = [16]

        with torch.no_grad():
            got = runner.run_grid(pixels, grids)

        self.assertIsNotNone(got)
        self.assertEqual(got.shape, (8 + 2, OUT_HIDDEN))
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    def test_oversized_or_unaligned_or_empty_frames(self):
        vit, runner = _build_vit()
        _enable_segment_backends(vit)
        runner._patch_buckets = [16]
        in_features = 3 * 1 * PATCH * PATCH
        pixels = torch.randn(24, in_features, dtype=torch.float32)
        self.assertIsNone(
            runner.run_grid(pixels, torch.tensor([[1, 4, 6]], dtype=torch.int32))
        )
        self.assertIsNone(
            runner.run_grid(pixels[:15], torch.tensor([[1, 3, 5]], dtype=torch.int32))
        )
        self.assertIsNone(
            runner.run_grid(pixels[:0], torch.tensor([[1, 0, 4]], dtype=torch.int32))
        )
        empty_out = runner.run_grid(
            pixels[:0], torch.zeros((0, 3), dtype=torch.int32)
        )
        self.assertIsNotNone(empty_out)
        self.assertEqual(empty_out.shape, (0, OUT_HIDDEN))
        with self.assertRaisesRegex(ValueError, "cannot stage"):
            runner.stage(pixels[:0], 0, 4, 16)

    def test_policy_warmup_inputs_and_run_grid(self):
        vit, runner = _build_vit()
        _enable_segment_backends(vit)
        policy = Glm5NextVisionTpuCompilePolicy(owner=vit)
        policy._patch_buckets = [16, 36]
        policy.compiled_fn = runner.forward
        policy._install_run_adapter(runner)

        warmup_args = policy.make_warmup_inputs(16)
        self.assertEqual(len(warmup_args), 4)
        self.assertEqual(warmup_args[0].shape, (16, 3 * PATCH * PATCH))
        self.assertEqual(warmup_args[1].shape, (16, HEAD_DIM))
        self.assertEqual(warmup_args[2].shape, (16, HEAD_DIM))
        self.assertEqual(warmup_args[3].shape, (1, 16))

        pixels = torch.randn(24, 3 * PATCH * PATCH, dtype=torch.float32)
        got = vit.graph_runners.run_grid(
            pixels, torch.tensor([[1, 4, 6]], dtype=torch.int32)
        )
        self.assertEqual(got.shape, (6, OUT_HIDDEN))

        oversize = vit.graph_runners.run_grid(
            torch.randn(64, 3 * PATCH * PATCH),
            torch.tensor([[1, 8, 8]], dtype=torch.int32),
        )
        self.assertIsNone(oversize)
        self.assertIn(64, policy._warned_oversize)

    def test_buckets_align_to_the_merger(self):
        with tpu_envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.override(1024):
            self.assertEqual(patch_buckets(), [g * g for g in range(14, 33, 2)])
        with tpu_envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("256, 36"):
            self.assertEqual(patch_buckets(), [36, 256])
        with tpu_envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("25"):
            with self.assertRaisesRegex(ValueError, "divisible by 4"):
                patch_buckets()


if __name__ == "__main__":
    unittest.main()
