# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash's TPU vision runner matches Glm5NextVisionModel.forward."""

import importlib.util
import os
import sys
import types
import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

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
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.runtime_context import get_flags, get_parallel
from sglang.srt.utils import is_tpu
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase, published_topology
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.graph_runner import vit_tpu_compile_runner
from sglang_tpu.graph_runner.glm5_next_vit_tpu_compile_runner import (
    Glm5NextVisionTpuCompilePolicy,
    Glm5NextVitTpuCompileRunner,
    _rot_pos_ids,
    patch_buckets,
)
from sglang_tpu.hooks.glm5_next_vision import encode_with_glm5_next_vision_runner

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

HEADS, HEAD_DIM, HIDDEN, OUT_HIDDEN, INTERMEDIATE, PATCH = 2, 16, 32, 24, 48, 4
# Norm-relative error per image for bf16 TPU compiled vs fp32 CPU / bf16 eager.
BF16_REL_TOL = 3e-2


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


@contextmanager
def _cpu_vit_env(use_data_parallel: bool = False):
    from sglang.srt.layers.attention import vision
    from sglang.srt.layers.rotary_embedding import base as rope_base
    from sglang.srt.models import glm5_next

    with ExitStack() as stack:
        stack.enter_context(patch.object(rope_base, "_is_tpu", True))
        stack.enter_context(
            patch.object(
                vision, "apply_rotary_pos_emb", vision.apply_rotary_pos_emb_native_eager
            )
        )
        stack.enter_context(
            patch.dict(
                vision.ROTARY_EMBED_CLASSES,
                {
                    k: vision.apply_rotary_pos_emb_native_eager
                    for k in list(vision.ROTARY_EMBED_CLASSES)
                },
            )
        )
        stack.enter_context(
            patch.object(
                glm5_next, "swiglu_clamped", _unwrapped(glm5_next.swiglu_clamped)
            )
        )
        # Keep CPU tests off VisionTpuAttention even when running on a TPU host.
        stack.enter_context(published_topology(tp_size=1, mm_attention_backend="sdpa"))
        stack.enter_context(get_parallel().override(tp_group=None, attn_tp_group=None))

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
        yield vit, runner


def _enable_segment_backends(vit) -> None:
    for block in vit.blocks:
        block.attn.qkv_backend = _SegmentSdpaBackend()


class TestGlm5NextVitTpuCompileRunner(CustomTestCase):
    def test_rot_pos_ids_matches_vision_model_rot_pos_emb(self):
        with _cpu_vit_env() as (vit, _):
            grids = torch.tensor([[1, 4, 6], [2, 2, 4]], dtype=torch.int32)
            _, _, want_ids = vit.rot_pos_emb(grids)
            got_ids = torch.cat(
                [_rot_pos_ids(4, 6, 2), _rot_pos_ids(2, 4, 2).repeat(2, 1)], dim=0
            )
            self.assertTrue(torch.equal(got_ids, want_ids.cpu()))

    def test_staged_runner_matches_unhooked_vision_model_across_padded_buckets(self):
        with _cpu_vit_env() as (vit, runner):
            grids = torch.tensor([[1, 4, 6], [1, 2, 2], [1, 8, 6]], dtype=torch.int32)
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
        with _cpu_vit_env() as (vit, runner):
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
        with _cpu_vit_env() as (vit, runner):
            _enable_segment_backends(vit)
            runner._patch_buckets = [16]
            in_features = 3 * 1 * PATCH * PATCH
            pixels = torch.randn(24, in_features, dtype=torch.float32)
            self.assertIsNone(
                runner.run_grid(pixels, torch.tensor([[1, 4, 6]], dtype=torch.int32))
            )
            self.assertIsNone(
                runner.run_grid(
                    pixels[:15], torch.tensor([[1, 3, 5]], dtype=torch.int32)
                )
            )
            self.assertIsNone(
                runner.run_grid(
                    pixels[:0], torch.tensor([[1, 0, 4]], dtype=torch.int32)
                )
            )
            empty_out = runner.run_grid(
                pixels[:0], torch.zeros((0, 3), dtype=torch.int32)
            )
            self.assertIsNotNone(empty_out)
            self.assertEqual(empty_out.shape, (0, OUT_HIDDEN))
            with self.assertRaisesRegex(ValueError, "cannot stage"):
                runner.stage(pixels[:0], 0, 4, 16)

    def test_policy_warmup_inputs_and_run_grid(self):
        with _cpu_vit_env() as (vit, runner):
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

    def test_policy_skips_compile_when_dp_attention_enabled(self):
        with _cpu_vit_env() as (vit, runner):
            policy = Glm5NextVisionTpuCompilePolicy(owner=vit)
            with get_flags().dp.override(enabled=False):
                self.assertTrue(policy._runner_ready_for_compile(runner))
            with get_flags().dp.override(enabled=True):
                self.assertFalse(policy._runner_ready_for_compile(runner))
                self.assertFalse(policy.install())
                self.assertIsNone(policy.compiled_fn)

    def test_buckets_align_to_the_merger(self):
        with patch.dict(os.environ):
            tpu_envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.clear()
            with tpu_envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.override(1024):
                self.assertEqual(patch_buckets(), [g * g for g in range(14, 33, 2)])
            with tpu_envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("256, 36"):
                self.assertEqual(patch_buckets(), [36, 256])
            with tpu_envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("25"):
                with self.assertRaisesRegex(ValueError, "divisible by 4"):
                    patch_buckets()


def _tpu_available() -> bool:
    """The TPU platform (what VisionTpuAttention requires) with a usable device.

    torch.accelerator alone is also true on CUDA hosts.
    """
    return (
        is_tpu() and hasattr(torch, "accelerator") and torch.accelerator.is_available()
    )


def _tpu_vision_config() -> SimpleNamespace:
    """Small 2-layer GLM-5.3-Flash ViT with the checkpoint's head/patch geometry."""
    return SimpleNamespace(
        hidden_size=256,
        num_heads=4,
        patch_size=14,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=256,
        intermediate_size=512,
        projection_intermediate_size=512,
        depth=2,
        rms_norm_eps=1e-5,
        attention_bias=True,
        swiglu_limit=10.0,
    )


@unittest.skipUnless(_tpu_available(), "requires the TPU platform and a TPU device")
class TestGlm5NextVitTpuCompileRunnerOnTpu(CustomTestCase):
    def test_compiled_buckets_and_oversize_fallback_on_tpu(self):
        from sglang.srt.layers.attention.vision import VisionTpuAttention
        from sglang.srt.models import glm5_next
        from sglang_tpu.graph_runner.tpu_graph_runner import (
            _resolve_fused_op_dispatch,
        )

        vision_config = _tpu_vision_config()
        in_features = (
            vision_config.in_channels
            * vision_config.temporal_patch_size
            * vision_config.patch_size**2
        )

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    glm5_next,
                    "swiglu_clamped",
                    _unwrapped(glm5_next.swiglu_clamped),
                )
            )
            stack.enter_context(
                patch.object(
                    vit_tpu_compile_runner,
                    "_precompiled_max_patch_count",
                    vit_tpu_compile_runner._precompiled_max_patch_count,
                )
            )

            with (
                published_topology(tp_size=1, mm_attention_backend="sdpa"),
                get_parallel().override(tp_group=None, attn_tp_group=None),
            ):
                torch.manual_seed(0)
                cpu_vit = glm5_next.Glm5NextVisionModel(
                    vision_config,
                    quant_config=None,
                    prefix="visual",
                    use_data_parallel=True,
                ).float()
                cpu_vit.use_data_parallel = False
                with torch.no_grad():
                    for param in cpu_vit.parameters():
                        scale = 0.1 if param.ndim >= 2 else 0.5
                        param.copy_(
                            (torch.randn_like(param) * scale).bfloat16().float()
                        )
                _enable_segment_backends(cpu_vit)
                state_dict = {k: v.clone() for k, v in cpu_vit.state_dict().items()}

            stack.enter_context(published_topology(tp_size=1))
            stack.enter_context(
                get_parallel().override(tp_group=None, attn_tp_group=None)
            )
            stack.enter_context(get_flags().dp.override(enabled=False))

            # Build under a bf16 default dtype, as the model loader does, rather
            # than casting every parameter: LayerNorm keeps the fp32 weights its
            # forward casts to, so its layer_norm sees one dtype.
            with set_default_torch_dtype(torch.bfloat16):
                tpu_vit = glm5_next.Glm5NextVisionModel(
                    vision_config,
                    quant_config=None,
                    prefix="visual",
                    use_data_parallel=True,
                )
            tpu_vit.use_data_parallel = False
            tpu_vit.load_state_dict(state_dict)
            tpu_vit = tpu_vit.to(device="tpu")
            for block in tpu_vit.blocks:
                self.assertIsInstance(block.attn.qkv_backend, VisionTpuAttention)

            _resolve_fused_op_dispatch(tpu_vit)
            runner = Glm5NextVitTpuCompileRunner(tpu_vit)
            tpu_vit.graph_runners = runner
            self.assertTrue(runner.ready)

            policy = Glm5NextVisionTpuCompilePolicy(owner=tpu_vit)
            stack.enter_context(
                tpu_envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("64,256")
            )
            self.assertTrue(policy.install())
            pending = policy.precompile()
            for call in pending or ():
                call.wait()

            # Non-square frames padded into distinct precompiled buckets:
            # 4x6=24 -> bucket 64, 12x16=192 -> bucket 256.
            grids = torch.tensor([[1, 4, 6], [1, 12, 16]], dtype=torch.int32)
            total_patches = int(grids.prod(dim=1).sum())
            gen = torch.Generator().manual_seed(1)
            pixels = torch.randn(total_patches, in_features, generator=gen).bfloat16()

            with torch.no_grad(), torch.compiler.set_stance("force_eager"):
                cpu_ref = _unwrapped(type(cpu_vit).forward)(
                    cpu_vit, pixels.float(), grids
                )
                tpu_eager = _unwrapped(type(tpu_vit).forward)(
                    tpu_vit, pixels, grids
                ).cpu()

            with (
                torch.no_grad(),
                sglang_envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.override(False),
                torch.compiler.set_stance("fail_on_recompile"),
            ):
                compiled = encode_with_glm5_next_vision_runner(
                    _unwrapped(type(tpu_vit).forward), tpu_vit, pixels, grids
                )

            merged_counts = [6, 48]
            self.assertEqual(compiled.shape, (sum(merged_counts), 256))
            self.assertEqual(compiled.device.type, "cpu")
            for grid, got, ref_cpu, ref_eager in zip(
                grids.tolist(),
                compiled.float().split(merged_counts),
                cpu_ref.split(merged_counts),
                tpu_eager.float().split(merged_counts),
            ):
                with self.subTest(grid=grid):
                    err_cpu = (got - ref_cpu).norm() / ref_cpu.norm()
                    err_eager = (got - ref_eager).norm() / ref_eager.norm()
                    self.assertLess(err_cpu.item(), BF16_REL_TOL)
                    self.assertLess(err_eager.item(), BF16_REL_TOL)

            # Oversize grid (18x16=288 > largest bucket 256) falls back to eager
            # without recompiling under fail_on_recompile.
            oversize_grid = torch.tensor([[1, 18, 16]], dtype=torch.int32)
            oversize_pixels = torch.randn(288, in_features, generator=gen).bfloat16()
            with torch.no_grad(), torch.compiler.set_stance("force_eager"):
                oversize_ref = _unwrapped(type(cpu_vit).forward)(
                    cpu_vit, oversize_pixels.float(), oversize_grid
                )
            with (
                torch.no_grad(),
                torch.compiler.set_stance("fail_on_recompile"),
            ):
                oversize_out = encode_with_glm5_next_vision_runner(
                    _unwrapped(type(tpu_vit).forward),
                    tpu_vit,
                    oversize_pixels,
                    oversize_grid,
                )

            self.assertIn(288, policy._warned_oversize)
            self.assertEqual(oversize_out.shape, (72, 256))
            oversize_err = (
                oversize_out.float().cpu() - oversize_ref
            ).norm() / oversize_ref.norm()
            self.assertLess(oversize_err.item(), BF16_REL_TOL)


if __name__ == "__main__":
    unittest.main()
