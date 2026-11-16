# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Hunyuan3D-2 volume decode (sglang_tpu.multimodal_gen.hunyuan3d)
and the Hunyuan3D-2 shape stages the TPU recipe runs."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import contextlib
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch

from sglang.multimodal_gen.runtime.distributed import group_coordinator, parallel_state
from sglang.multimodal_gen.runtime.models.vaes import hunyuan3d_vae as hv
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan3d import (
    shape,
)
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import hunyuan3d

# 17^3 = 4913 grid points: 24 chunks of 200 and a 113-point tail.
RESOLUTION, CHUNK, TAIL = 16, 200, 113


@contextlib.contextmanager
def _installed():
    with contextlib.ExitStack() as stack:
        decoder = hv.VanillaVolumeDecoder
        stack.enter_context(patch.object(decoder, "__call__", decoder.__call__))
        stack.enter_context(patch.object(hunyuan3d, "_DEVICE_TYPES", ("cpu",)))
        hunyuan3d.install()
        hunyuan3d.install()  # idempotent
        yield


def _vae(dtype):
    torch.manual_seed(0)
    vae = hv.ShapeVAE(
        num_latents=64,
        embed_dim=8,
        width=64,
        heads=4,
        num_decoder_layers=1,
        qkv_bias=False,
        qk_norm=True,
        include_pi=False,
    )
    vae = vae.eval().to(dtype)
    with torch.no_grad():
        return vae, vae(torch.randn(2, 64, 8, dtype=dtype))


def _decode(vae, latents):
    return vae.volume_decoder(
        latents,
        vae.geo_decoder,
        bounds=1.01,
        num_chunks=CHUNK,
        octree_resolution=RESOLUTION,
        enable_pbar=False,
    )


class _DecodeGroup:
    """Rank ``rank`` of a ``world``-rank VAE decode group, gathering with
    GroupCoordinator.all_gather over the collective patched in by _decode_ranks."""

    all_gather = group_coordinator.GroupCoordinator.all_gather

    def __init__(self, world, rank):
        self.world_size = world
        self.rank_in_group = self.device_group = rank
        self.ranks = list(range(world))


@contextlib.contextmanager
def _decode_ranks(world, rank, all_gather_single):
    """Run as rank ``rank`` of the decode group, in replica 1 of two DP replicas:
    the world group spans both replicas and must not be used."""

    def world_group():
        raise AssertionError("the world group spans the DP replicas")

    with (
        patch.object(parallel_state, "model_parallel_is_initialized", lambda: True),
        patch.object(
            parallel_state,
            "get_decode_parallel_group_coordinator",
            lambda: _DecodeGroup(world, rank),
        ),
        patch.object(parallel_state, "world_group_is_initialized", lambda: True),
        patch.object(parallel_state, "get_world_size", lambda: 2 * world),
        patch.object(parallel_state, "get_world_rank", lambda: world + rank),
        patch.object(parallel_state, "get_world_group", world_group),
        patch.object(group_coordinator, "is_shm_available", lambda *a: False),
        patch.object(group_coordinator, "all_gather_single", all_gather_single),
    ):
        yield


def _sharded_decode(vae, latents, world):
    """Rank 0's grid when ``world`` ranks share the decode."""
    sent = {}

    def all_gather_single(output, input_, group):
        # The device collective: every rank's tensor, in rank order.
        sent[group] = input_
        output.copy_(torch.cat([sent.get(r, input_) for r in range(world)]))

    for rank in reversed(range(world)):  # rank 0 last, once every rank has sent
        with _decode_ranks(world, rank, all_gather_single):
            grid = _decode(vae, latents)
    assert sorted(sent) == (list(range(world)) if world > 1 else []), sorted(sent)
    # An all-gather takes the same flat [batch, rows] shape from every rank.
    shapes = {t.shape for t in sent.values()}
    assert len(shapes) <= 1 and all(len(s) == 2 for s in shapes), shapes
    return grid


class TestVolumeDecode(CustomTestCase):
    def test_sharded_grid_equals_stock(self):
        for dtype in (torch.float32, torch.float16):
            vae, latents = _vae(dtype)
            want = _decode(vae, latents)
            with _installed():
                for world in (1, 3, 8, 16):
                    with self.subTest(dtype=dtype, world=world):
                        got = _sharded_decode(vae, latents, world)
                        self.assertEqual(got.dtype, torch.float32)
                        self.assertTrue(torch.equal(got, want))

    def test_rank_chunks_own_their_storage_and_are_built_once(self):
        vae, latents = _vae(torch.float32)
        c_kv = vae.geo_decoder.cross_attn_decoder.attn.c_kv

        def all_gather_single(output, input_, group):
            output.copy_(input_.repeat(3, 1))

        with (
            _installed(),
            patch.object(
                hv, "generate_dense_grid_points", wraps=hv.generate_dense_grid_points
            ) as build_grid,
            patch.object(c_kv, "forward", wraps=c_kv.forward) as kv,
            _decode_ranks(3, 2, all_gather_single),
        ):
            _decode(vae, latents)
            _decode(vae, latents)
        self.assertEqual(build_grid.call_count, 1)
        self.assertEqual(kv.call_count, 2)  # once per request
        # Rank 2 of 3 takes chunks 18-24 of 25: six full ones and the tail, as
        # flat [batch * points * 3] tensors.
        chunks = vae.geo_decoder.__dict__["_tpu_volume_chunks"][1][0]
        self.assertEqual(len(chunks), 7)
        self.assertEqual(
            {tuple(c.shape) for c in chunks}, {(2 * CHUNK * 3,), (2 * TAIL * 3,)}
        )
        self.assertTrue(all(c.storage_offset() == 0 for c in chunks))
        storages = {c.untyped_storage().data_ptr() for c in chunks}
        self.assertEqual(len(storages), len(chunks))

    def test_other_devices_and_decoders_take_the_stock_path(self):
        vae, latents = _vae(torch.float32)
        want = _decode(vae, latents)
        with _installed():
            with patch.object(hunyuan3d, "_DEVICE_TYPES", ("tpu",)):
                self.assertTrue(torch.equal(_decode(vae, latents), want))
            vae.geo_decoder.cross_attn_decoder.attn.kv_cache = True
            self.assertTrue(torch.equal(_decode(vae, latents), want))
        self.assertNotIn("_tpu_volume_chunks", vae.geo_decoder.__dict__)


class TestShapeStages(CustomTestCase):
    def test_the_first_rank_of_each_replica_writes_the_mesh(self):
        stage = shape.Hunyuan3DShapeSaveStage.__new__(shape.Hunyuan3DShapeSaveStage)
        stage.config = SimpleNamespace(paint_enable=False)
        # World rank 2 is replica 1's first rank under --dp-size 2 on 4 ranks.
        for world_rank, replica_rank in ((0, 0), (1, 1), (2, 0), (3, 1)):
            with (
                self.subTest(world_rank=world_rank),
                tempfile.TemporaryDirectory() as output_dir,
                patch.object(
                    parallel_state, "world_group_is_initialized", lambda: True
                ),
                patch.object(
                    parallel_state,
                    "get_replica_group",
                    lambda: SimpleNamespace(rank_in_group=replica_rank),
                ),
            ):
                mesh = Mock()
                path = os.path.join(output_dir, "mesh.obj")
                batch = SimpleNamespace(
                    extra={"shape_meshes": [mesh]},
                    is_warmup=False,
                    metrics=None,
                    output_file_path=lambda: path,
                )
                out = stage.forward(batch, None)
                self.assertIsInstance(out, OutputBatch)
                self.assertEqual(out.output_file_paths, [path])
                if replica_rank == 0:
                    mesh.export.assert_called_once_with(path)
                else:
                    mesh.export.assert_not_called()

    def test_latents_are_drawn_in_float32(self):
        # A float16 draw quantizes the normal; the latents are a float32 draw
        # from the request's generator, rounded.
        stage = shape.Hunyuan3DShapeBeforeDenoisingStage.__new__(
            shape.Hunyuan3DShapeBeforeDenoisingStage
        )
        stage.latent_shape = (64, 32)
        batch = SimpleNamespace(generator=None, seed=7, generator_device="cpu")
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(generator_device=None)
        )
        scheduler = SimpleNamespace(init_noise_sigma=1.0)
        cpu = PropertyMock(return_value=torch.device("cpu"))
        with patch.object(type(stage), "device", cpu):
            got = stage._prepare_latents(
                batch, server_args, 2, torch.float16, scheduler
            )
        want = torch.randn(
            2, 64, 32, generator=torch.Generator().manual_seed(7), dtype=torch.float32
        ).half()
        self.assertTrue(torch.equal(got, want))


if __name__ == "__main__":
    import unittest

    unittest.main()
