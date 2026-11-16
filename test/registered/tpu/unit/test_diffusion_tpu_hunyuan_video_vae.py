# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's HunyuanVideo VAE decode (multimodal_gen/hunyuan_video_vae.py)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=180, suite="nightly-tpu-4-v7", nightly=True)

import contextlib
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.vaes.hunyuanvae import (
    HunyuanVAEArchConfig,
    HunyuanVAEConfig,
)
from sglang.multimodal_gen.runtime.distributed import group_coordinator, parallel_state
from sglang.multimodal_gen.runtime.models.vaes import common, hunyuanvae
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs
from sglang_tpu.multimodal_gen import hunyuan_video_vae as plugin

# Latent [T, H, W] sizes, decoded in 40 px tiles at a 24 px stride (5 and 3
# latent rows) and 17 frames at a 12-frame stride (5 and 3 latent frames).
# 8x7x8: a short second-to-last tile row (4 rows), 1-row and 2-column edges
# and a 2-frame last slab; 7x5x4: a last slab of one latent frame, which decodes
# to no frames; 2x3x3: one tile, so most of 8 ranks have none.
SHAPES = ((8, 7, 8), (7, 5, 4), (2, 3, 3))


def _vae(dtype, device="cpu"):
    torch.manual_seed(0)
    arch = HunyuanVAEArchConfig(
        latent_channels=4,
        block_out_channels=(4, 4, 8, 8),
        layers_per_block=1,
        norm_num_groups=2,
    )
    config = HunyuanVAEConfig(
        arch_config=arch,
        parallel_decode_mode="tiled",
        tile_sample_min_height=40,
        tile_sample_min_width=40,
        tile_sample_stride_height=24,
        tile_sample_stride_width=24,
    )
    config.load_encoder = False
    return hunyuanvae.AutoencoderKLHunyuanVideo(config).eval().to(device, dtype)


@contextlib.contextmanager
def _installed(enable_torch_compile=False, device_types=("cpu",)):
    with contextlib.ExitStack() as stack:
        vae = hunyuanvae.AutoencoderKLHunyuanVideo
        for name in ("parallel_tiled_decode", "blend_t", "blend_v", "blend_h"):
            stack.enter_context(patch.object(vae, name, getattr(vae, name)))
        stack.enter_context(patch.object(plugin, "_DEVICE_TYPES", device_types))
        args = SimpleNamespace(enable_torch_compile=enable_torch_compile)
        stack.enter_context(
            patch.object(server_args_module, "_global_server_args", args)
        )
        plugin.install()
        plugin.install()  # idempotent
        yield


@contextlib.contextmanager
def _stock_one_rank():
    """The stock decode as the only SP rank: its collectives copy."""
    dist = SimpleNamespace(
        all_gather=lambda outputs, input_: outputs[0].copy_(input_),
        all_gather_object=lambda outputs, obj: outputs.__setitem__(0, obj),
    )
    with (
        patch.object(common, "get_sp_world_size", lambda: 1),
        patch.object(common, "get_sp_parallel_rank", lambda: 0),
        patch.object(common, "dist", dist),
        patch.object(common, "all_gather_single", lambda out, in_: out.copy_(in_)),
    ):
        yield


class _DecodeGroup:
    """Rank ``rank`` of a ``world``-rank VAE decode group, gathering with
    GroupCoordinator.all_gather over the collective patched in by _split_decode."""

    all_gather = group_coordinator.GroupCoordinator.all_gather

    def __init__(self, world, rank):
        self.world_size = world
        self.rank_in_group = self.device_group = rank
        self.ranks = list(range(world))


def _split_decode(vae, z, world):
    """Rank 0's video when ``world`` ranks share the decode, and what each rank sent."""
    sent = {}

    def all_gather_single(output, input_, group):
        sent[group] = input_.clone()
        output.copy_(torch.cat([sent.get(r, input_) for r in range(world)]))

    for rank in reversed(range(world)):  # rank 0 last, once every rank has sent
        with (
            patch.object(parallel_state, "model_parallel_is_initialized", lambda: True),
            patch.object(
                parallel_state,
                "get_decode_parallel_group_coordinator",
                lambda: _DecodeGroup(world, rank),
            ),
            patch.object(group_coordinator, "is_shm_available", lambda *a: False),
            patch.object(group_coordinator, "all_gather_single", all_gather_single),
        ):
            video = vae.parallel_tiled_decode(z)
    return video, sent


class TestTiledDecode(CustomTestCase):
    def test_split_decode_equals_stock(self):
        for dtype in (torch.float32, torch.float16):
            vae = _vae(dtype)
            for shape in SHAPES:
                z = torch.randn(1, 4, *shape).to(dtype)
                vae.__dict__.pop("_tpu_tile_decode", None)
                with _stock_one_rank():
                    want = vae.parallel_tiled_decode(z)
                    with _installed(device_types=("tpu",)):  # CPU takes stock
                        self.assertTrue(torch.equal(vae.parallel_tiled_decode(z), want))
                self.assertNotIn("_tpu_tile_decode", vae.__dict__)
                for world in (1, 2, 3, 8):
                    with self.subTest(dtype=dtype, shape=shape, world=world):
                        with _installed():
                            got, sent = _split_decode(vae, z, world)
                        self.assertEqual(got.dtype, want.dtype)
                        self.assertTrue(torch.equal(got, want))
                        # One all-gather, the same flat shape on every rank.
                        ranks = list(range(world)) if world > 1 else []
                        self.assertEqual(sorted(sent), ranks)
                        self.assertLessEqual(len({t.shape for t in sent.values()}), 1)
                        self.assertTrue(all(t.dim() == 1 for t in sent.values()))

    def test_tiles_own_their_storage_and_compile_once(self):
        vae, z = _vae(torch.float32), torch.randn(1, 4, 8, 7, 8)
        tiles = []

        def compile_(fn, **_):
            return lambda latent: fn(tiles.append(latent) or latent)

        with (
            _installed(enable_torch_compile=True),
            patch.object(torch, "compile", side_effect=compile_) as compiled,
        ):
            for _ in range(2):  # two requests
                _split_decode(vae, z, 3)
            del vae.__dict__["_tpu_tile_decode"]
            with envs.SGLANG_TPU_ENABLE_HUNYUAN_VIDEO_DECODE_COMPILE.override(False):
                _split_decode(vae, z, 3)
        compiled.assert_called_once()
        self.assertEqual(len(tiles), 2 * 27)  # 3x3x3 tiles, each decoded once
        for t in tiles:  # tensors of their own: storage offset 0, contiguous
            self.assertTrue(t._base is None and not t.storage_offset())
            self.assertTrue(t.is_contiguous())
        # Under static shapes, one program per tile shape.
        frames, rows, cols = (5, 2), (5, 4, 1), (5, 2)
        want = {(1, 4, t, h, w) for t in frames for h in rows for w in cols}
        self.assertEqual({tuple(t.shape) for t in tiles}, want)


def _has_tpu() -> bool:
    try:
        return bool(importlib.util.find_spec("torch_tpu") and torch.tpu.device_count())
    except Exception:
        return False


@unittest.skipUnless(_has_tpu(), "needs a TPU device")
class TestTiledDecodeOnDevice(CustomTestCase):
    def test_compiled_tiles_match_eager_and_stock(self):
        import torch_tpu._internal.compile  # noqa: F401  (the "tpu" backend)

        from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

        TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())
        vae, z = _vae(torch.float16, "tpu"), torch.randn(1, 4, 4, 5, 5).half()
        compile_ = torch.compile
        with (
            _installed(enable_torch_compile=True, device_types=("tpu",)),
            patch.object(  # one frame slab of 2x2 tiles: four tile shapes
                torch, "compile", lambda fn, **_: compile_(fn, backend="tpu")
            ),
            patch.object(torch._dynamo.config, "automatic_dynamic_shapes", False),
        ):
            got = vae.parallel_tiled_decode(z.to("tpu")).cpu()
            del vae.__dict__["_tpu_tile_decode"]
            with envs.SGLANG_TPU_ENABLE_HUNYUAN_VIDEO_DECODE_COMPILE.override(False):
                eager = vae.parallel_tiled_decode(z.to("tpu")).cpu()
        with _stock_one_rank():
            want = vae.cpu().float().parallel_tiled_decode(z.float())
        torch.testing.assert_close(got, eager, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(eager.float(), want, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
