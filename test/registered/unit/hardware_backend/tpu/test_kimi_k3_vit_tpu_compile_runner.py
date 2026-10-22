"""Kimi-K3's TPU vision runner computes what the tower and projector compute.

The runner stages each image on the host (merge-window order,
position-embedding taps, RoPE tables, segment ids) at a fixed patch bucket and
runs patch embedding, encoder, merge and projector as one program. These
check it against the tower's own eager forward (interpolated position
embedding, per-image SDPA, complex RoPE, temporal-pool merger) and the
projector:

* on a small Kimi-K3 tower with several images, in both interpolation modes,
  plus the bucket ladder and the eager fall-back. CPU tensors; a
  segment-masked SDPA stands in for the Pallas flash kernel.
* on the TPU in bf16, with the real VisionTpuAttention, at the checkpoint's
  vision config (64x64 bilinear position embedding, two layers), against the
  tower and projector in fp32 on the CPU.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.graph_runner.kimi_k3_vit_tpu_compile_runner import (
    KimiK3VitTpuCompileRunner,
    merge_window_order,
    patch_buckets,
    rotate_interleaved,
)

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

HEADS, HEAD_DIM, HIDDEN, PATCH = 2, 16, 24, 4
# Norm-relative error per image. On _checkpoint_model bf16 rounding costs about
# 6e-3, while bicubic taps for its bilinear tower cost 0.2.
BF16_REL_TOL = 3e-2


class _SegmentSdpa:
    """VisionTpuAttention's contract on CPU: attend within a segment id."""

    supports_segment_ids = True

    def __init__(self, head_dim, num_heads, num_kv_heads):
        del head_dim, num_heads, num_kv_heads

    def __call__(self, q, k, v, bsz, segment_ids=None):
        ids = segment_ids[0]
        mask = ids[:, None] == ids[None, :]
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
        )
        return out.transpose(0, 1)


def _unwrapped(fn):
    # The TPU plugin may have wrapped these with its hooks; test the originals.
    return getattr(fn, "__wrapped__", fn)


def _model(interpolation_mode="bilinear"):
    from sglang.srt.models.kimi_k3_vl import (
        KimiK3MultiModalProjector,
        KimiK3VisionTower,
    )

    config = SimpleNamespace(
        merge_kernel_size=(2, 2),
        patch_size=PATCH,
        merge_type="sd2_tpool",
        hidden_size=HIDDEN,
        num_attention_heads=HEADS,
        num_hidden_layers=2,
        intermediate_size=48,
        init_pos_emb_height=6,
        init_pos_emb_width=5,
        init_pos_emb_time=4,
        pos_emb_type="divided_fixed",
        pos_emb_interpolation_mode=interpolation_mode,
        qkv_hidden_size=HEADS * HEAD_DIM,
        norm_type="rmsnorm",
        attn_bias=False,
        linear_bias=False,
        mm_hidden_size=HIDDEN,
        text_hidden_size=32,
        projector_ln_eps=1e-5,
    )
    torch.manual_seed(0)
    tower = KimiK3VisionTower(config).float()
    projector = KimiK3MultiModalProjector(config).float()
    with torch.no_grad():
        for module in (tower, projector):
            for param in module.parameters():
                param.copy_(torch.randn_like(param) * 0.2)
    runner = KimiK3VitTpuCompileRunner(tower.encoder, attention_cls=_SegmentSdpa)
    runner.bind(tower, projector)
    return tower, projector, runner


def _checkpoint_model(device="cpu", dtype=torch.float32):
    """The checkpoint's vision tower and projector (KimiK3VisionConfig's
    defaults, a 7168-wide text model) with two layers. Seeded and rounded to
    bf16, so every copy holds the same weights."""
    from sglang.srt.configs.kimi_k3 import KimiK3VisionConfig
    from sglang.srt.models.kimi_k3_vl import (
        KimiK3MultiModalProjector,
        KimiK3VisionTower,
    )

    config = KimiK3VisionConfig(vt_num_hidden_layers=2, text_hidden_size=7168)
    torch.manual_seed(0)
    tower = KimiK3VisionTower(config)
    projector = KimiK3MultiModalProjector(config)
    with torch.no_grad():
        # The one parameter left uninitialized. At unit scale, next to the
        # patch embedding's ~0.6, the interpolation shows in the output.
        tower.patch_embed.pos_emb.weight.normal_()
        for module in (tower, projector):
            for param in module.parameters():
                param.copy_(param.bfloat16())
    return tower.to(device, dtype), projector.to(device, dtype)


def _reference(tower, projector, pixels, grids):
    features = _unwrapped(type(tower)._forward_eager)(
        tower, pixels, grids, grid_thw_list=grids.tolist()
    )
    return _unwrapped(type(projector).forward)(projector, features)


class TestKimiK3VitRunner(CustomTestCase):
    def test_real_rope_matches_complex_rope(self):
        from sglang.srt.models.kimi_k3_vl import apply_rope

        x = torch.randn(10, HEADS, HEAD_DIM)
        angles = torch.randn(10, HEAD_DIM // 2)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        want, _ = apply_rope(x, x, freqs_cis)
        got = rotate_interleaved(x, freqs_cis.real, freqs_cis.imag)
        torch.testing.assert_close(got, want)

    def test_window_order_is_the_mergers_order(self):
        from sglang.srt.models.kimi_vl_moonvit import tpool_patch_merger

        h, w = 4, 6
        x = torch.arange(h * w, dtype=torch.float32)[:, None]
        (merged,) = tpool_patch_merger(x, torch.tensor([[1, h, w]]))
        torch.testing.assert_close(merged.reshape(-1), x[merge_window_order(h, w), 0])

    def test_staged_runner_matches_tower_and_projector(self):
        # Bilinear is the checkpoint's mode.
        for mode in ("bilinear", "bicubic"):
            with self.subTest(mode=mode):
                tower, projector, runner = _model(mode)
                runner._patch_buckets = [36, 64]
                # A grid that interpolates the 6x5 position embedding, one that
                # shrinks it, and padding in both buckets.
                grids = torch.tensor([[1, 4, 6], [1, 2, 2], [1, 8, 6]])
                n = int(grids.prod(dim=1).sum())
                pixels = torch.randn(n, 3, PATCH, PATCH)
                with torch.no_grad():
                    want = _reference(tower, projector, pixels, grids)
                    got = runner.run_images(pixels, [tuple(g) for g in grids.tolist()])
                self.assertEqual([t.shape[0] for t in got], [6, 1, 12])
                torch.testing.assert_close(torch.cat(got), want, rtol=1e-4, atol=1e-4)

    def test_only_reproduced_interpolation_modes_are_ready(self):
        for mode, ready in (("bilinear", True), ("bicubic", True), ("nearest", False)):
            with self.subTest(mode=mode):
                self.assertEqual(_model(mode)[2].ready, ready)

    def test_video_and_oversized_images_take_the_eager_path(self):
        _, _, runner = _model()
        runner._patch_buckets = [16]
        pixels = torch.randn(32, 3, PATCH, PATCH)
        self.assertIsNone(runner.run_images(pixels, [(2, 4, 4)]))
        self.assertIsNone(runner.run_images(pixels[:24], [(1, 4, 6)]))

    def test_buckets_align_to_the_merger(self):
        with envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.override(1024):
            self.assertEqual(patch_buckets(), [g * g for g in range(14, 33, 2)])
        with envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("256, 36"):
            self.assertEqual(patch_buckets(), [36, 256])
        with envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.override("25"):
            with self.assertRaisesRegex(ValueError, "divisible by 4"):
                patch_buckets()


@unittest.skipUnless(
    hasattr(torch, "accelerator") and torch.accelerator.is_available(), "requires TPU"
)
class TestKimiK3VitRunnerOnTpu(CustomTestCase):
    def test_bf16_runner_matches_fp32_tower_and_projector(self):
        tower, projector = _checkpoint_model()
        tpu_tower, tpu_projector = _checkpoint_model("tpu", torch.bfloat16)
        runner = KimiK3VitTpuCompileRunner(tpu_tower.encoder)
        runner.bind(tpu_tower, tpu_projector)
        self.assertTrue(runner.ready)
        runner._patch_buckets = [256, 576, 4900]
        # The 64x64 position embedding upsampled, shrunk, and a non-square
        # grid padded from 512 patches to the 576 bucket.
        grids = torch.tensor([[1, 70, 70], [1, 16, 16], [1, 16, 32]])
        pixels = torch.randn(int(grids.prod(dim=1).sum()), 3, 14, 14).bfloat16()
        with torch.no_grad():
            want = _reference(tower, projector, pixels.float(), grids)
            got = runner.run_images(pixels, [tuple(g) for g in grids.tolist()])
        merged = [h * w // 4 for _, h, w in grids.tolist()]
        self.assertEqual([t.shape[0] for t in got], merged)
        for grid, out, ref in zip(grids.tolist(), got, want.split(merged)):
            with self.subTest(grid=grid):
                error = (out.float() - ref).norm() / ref.norm()
                self.assertLess(error.item(), BF16_REL_TOL)


if __name__ == "__main__":
    unittest.main()
