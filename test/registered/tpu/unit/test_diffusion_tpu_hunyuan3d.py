# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Hunyuan3D-2 patches (sglang_tpu.multimodal_gen.hunyuan3d),
the Hunyuan3D-2 stage and DiT hooks the TPU recipe runs, and the antialiased
resize fallback its image conditioner takes on TPU."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

import contextlib
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed import group_coordinator, parallel_state
from sglang.multimodal_gen.runtime.models.dits import hunyuan3d as dit
from sglang.multimodal_gen.runtime.models.vaes import hunyuan3d_vae as hv
from sglang.multimodal_gen.runtime.pipelines_core.executors import parallel_executor
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan3d import (
    shape,
)
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import patches
from sglang_tpu.envs import envs
from sglang_tpu.multimodal_gen import hunyuan3d

# 17^3 = 4913 grid points: 24 chunks of 200 and a 113-point tail.
RESOLUTION, CHUNK, TAIL = 16, 200, 113


def _server_args(**kwargs):
    return patch.object(
        server_args_module, "_global_server_args", SimpleNamespace(**kwargs)
    )


@contextlib.contextmanager
def _installed(enable_torch_compile=False):
    with contextlib.ExitStack() as stack:
        decoder = hv.VanillaVolumeDecoder
        stack.enter_context(patch.object(decoder, "__call__", decoder.__call__))
        stack.enter_context(patch.object(hunyuan3d, "_DEVICE_TYPES", ("cpu",)))
        stack.enter_context(_server_args(enable_torch_compile=enable_torch_compile))
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

    def test_torch_compile_compiles_the_chunk_step_once(self):
        vae, latents = _vae(torch.float32)
        want = _decode(vae, latents)
        with (
            _installed(enable_torch_compile=True),
            patch.object(torch, "compile", side_effect=lambda fn, **_: fn) as compile_,
        ):
            got = _decode(vae, latents)
            _decode(vae, latents)
        compile_.assert_called_once()
        self.assertTrue(torch.equal(got, want))

    def test_compile_switch_keeps_the_chunk_step_eager(self):
        vae, latents = _vae(torch.float32)
        with (
            _installed(enable_torch_compile=True),
            envs.SGLANG_TPU_ENABLE_HUNYUAN3D_DECODE_COMPILE.override(False),
            patch.object(torch, "compile", side_effect=lambda fn, **_: fn) as compile_,
        ):
            _decode(vae, latents)
        compile_.assert_not_called()


class TestAntialiasedResizeFallback(CustomTestCase):
    def _registered_kernels(self, tpu_kernels=()):
        """The fallbacks registered when torch_tpu has kernels for ``tpu_kernels``."""
        kernels = {}

        def register_kernel(op, device_types):
            self.assertEqual(device_types, "PrivateUse1")
            return lambda fn: kernels.setdefault(op, fn)

        def has_kernel(op, key):
            return key == "PrivateUse1" and op in tpu_kernels

        with (
            patch.object(patches, "_aten_fallbacks_registered", False),
            patch.object(torch.library, "register_kernel", register_kernel),
            patch.object(torch._C, "_dispatch_has_kernel_for_dispatch_key", has_kernel),
        ):
            patches._register_aten_fallbacks()
        return kernels

    def test_a_torch_tpu_kernel_is_kept(self):
        for op in (
            "aten::_upsample_bilinear2d_aa",
            "aten::_upsample_bilinear2d_aa.out",
        ):
            with self.subTest(op=op):
                kernels = self._registered_kernels(tpu_kernels=(op,))
                self.assertIn("aten::logaddexp", kernels)
                self.assertNotIn("aten::_upsample_bilinear2d_aa", kernels)

    def test_registered_kernel_runs_the_cpu_kernel(self):
        kernel = self._registered_kernels()["aten::_upsample_bilinear2d_aa"]
        image = torch.rand(1, 3, 512, 512, generator=torch.Generator().manual_seed(0))

        def resize(x):
            return F.interpolate(
                x, size=(518, 518), mode="bilinear", align_corners=False, antialias=True
            )

        self.assertTrue(torch.equal(kernel(image, [518, 518], False), resize(image)))
        half = kernel(image.half(), [518, 518], False, None, None)
        self.assertEqual(half.dtype, torch.float16)
        self.assertTrue(torch.equal(half, resize(image.half().float()).half()))


class TestShapePipelineHooks(CustomTestCase):
    def test_regional_compile_takes_every_dit_block(self):
        def compiled(name):
            return any(c(name, None) for c in dit.Hunyuan3D2DiT._compile_conditions)

        blocks = [f"double_blocks.{i}" for i in range(16)]
        blocks += [f"single_blocks.{i}" for i in range(32)]
        self.assertTrue(all(map(compiled, blocks)))
        others = [
            "double_blocks",
            "double_blocks.0.img_attn",
            "single_blocks.3.linear1",
        ]
        self.assertFalse(any(map(compiled, others + ["latent_in", "final_layer"])))

    def _save(self, mesh, world_rank, replica_rank, output_dir):
        """Run the save stage through ParallelExecutor on one rank of a server."""

        class Executor:
            def _should_use_stage_nvtx(self, batch, server_args):
                return False

            @contextlib.contextmanager
            def _component_residency_request(self, stages, batch, server_args):
                yield

            def _run_stage_with_executor_hooks(self, stage, index, batch, *args):
                server_args, run_stage = args[:2]
                return run_stage(stage, batch)

        def barrier(*args, **kwargs):
            # A rank that raised in a MAIN_RANK_ONLY stage never reaches it.
            raise AssertionError("barrier: the ranks fall out of step on errors")

        with _server_args():
            stage = shape.Hunyuan3DShapeSaveStage(SimpleNamespace(paint_enable=False))
        path = os.path.join(output_dir, "mesh.obj")
        batch = SimpleNamespace(
            extra={"shape_meshes": [mesh]},
            is_warmup=False,
            metrics=None,
            output_file_path=lambda: path,
        )
        with (
            patch.object(parallel_executor, "get_world_rank", lambda: world_rank),
            patch.object(parallel_executor, "get_cfg_group", lambda: None),
            patch.object(parallel_executor, "get_world_group", lambda: None),
            patch.object(torch.distributed, "barrier", barrier),
            patch.object(parallel_state, "world_group_is_initialized", lambda: True),
            patch.object(
                parallel_state,
                "get_replica_group",
                lambda: SimpleNamespace(rank_in_group=replica_rank),
            ),
        ):
            return parallel_executor.ParallelExecutor._execute_stages(
                Executor(),
                [stage],
                batch,
                SimpleNamespace(enable_cfg_parallel=False),
                lambda stage, batch: stage.forward(batch, None),
            )

    def test_every_rank_raises_for_a_failed_mesh(self):
        with tempfile.TemporaryDirectory() as output_dir:
            for world_rank, replica_rank in ((0, 0), (1, 1), (2, 0)):
                with self.subTest(world_rank=world_rank):
                    with self.assertRaisesRegex(RuntimeError, "Mesh generation failed"):
                        self._save(None, world_rank, replica_rank, output_dir)

    def test_the_first_rank_of_each_replica_writes_the_mesh(self):
        # World rank 2 is replica 1's first rank under --dp-size 2 on 4 ranks.
        with tempfile.TemporaryDirectory() as output_dir:
            for world_rank, replica_rank in ((0, 0), (1, 1), (2, 0), (3, 1)):
                with self.subTest(world_rank=world_rank):
                    mesh = Mock()
                    out = self._save(mesh, world_rank, replica_rank, output_dir)
                    path = os.path.join(output_dir, "mesh.obj")
                    self.assertIsInstance(out, OutputBatch)
                    self.assertEqual(out.output_file_paths, [path])
                    if replica_rank == 0:
                        mesh.export.assert_called_once_with(path)
                    else:
                        mesh.export.assert_not_called()

    def test_generator_device_seeds_the_generator(self):
        class Conditioner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))

            def forward(self, image, **kwargs):
                return {"main": image.new_zeros(1, 2, 4)}

            def unconditional_embedding(self, batch_size, **kwargs):
                return {"main": torch.zeros(batch_size, 2, 4, device="meta")}

        class Scheduler:
            init_noise_sigma = 1.0

            def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
                self.timesteps = torch.tensor(sigmas)

        stage_cls = shape.Hunyuan3DShapeBeforeDenoisingStage
        meta = torch.device("meta")  # the stage device, other than the CPU
        cpu_generator = torch.Generator
        # (request generator_device, pipeline config generator_device) -> device
        cases = {
            ("cpu", None): "cpu",
            (None, "cpu"): "cpu",
            ("cuda", "cpu"): "meta",
            (None, None): "meta",
        }
        for (request_device, config_device), want in cases.items():
            seeded = []

            def generator(device):
                seeded.append(torch.device(device).type)
                return cpu_generator("cpu")

            with (
                self.subTest(request=request_device, config=config_device),
                tempfile.NamedTemporaryFile(suffix=".png") as image,
                _server_args(),
                patch.object(stage_cls, "device", new_callable=PropertyMock) as device,
            ):
                device.return_value = meta
                stage = stage_cls(
                    image_processor=lambda path: {
                        "image": torch.zeros(1, 3, 4, 4),
                        "mask": torch.zeros(1, 1, 4, 4),
                    },
                    conditioner=Conditioner(),
                    scheduler=Scheduler(),
                    config=None,
                    latent_shape=(8, 4),
                    guidance_embed=False,
                )
                batch = SimpleNamespace(
                    image_path=image.name,
                    num_outputs_per_prompt=1,
                    guidance_scale=5.0,
                    num_inference_steps=2,
                    generator=None,
                    seed=7,
                    generator_device=request_device,
                    extra={},
                )
                server_args = SimpleNamespace(
                    pipeline_config=SimpleNamespace(generator_device=config_device)
                )
                with (
                    patch.object(torch, "Generator", generator),
                    patch.object(
                        stage, "_prepare_latents", wraps=stage._prepare_latents
                    ) as prepare,
                ):
                    stage.forward(batch, server_args)
                self.assertEqual(seeded, [want])
                self.assertEqual(prepare.call_args.args[3].initial_seed(), 7)
                self.assertEqual(batch.latents.device, meta)


if __name__ == "__main__":
    import unittest

    unittest.main()
