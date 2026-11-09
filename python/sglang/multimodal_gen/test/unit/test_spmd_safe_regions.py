"""Regional-compile hooks: the SPMD-safe opt-in contract, the rank-invariant
inputs Wan feeds its blocks, the fusion barrier and the per-request drain."""

import contextlib
import inspect
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._pytree import tree_leaves

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    fuse_scale_shift_kernel_native,
)
from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers import gpu_worker
from sglang.multimodal_gen.runtime.managers.forward_context import (
    set_forward_context,
)
from sglang.multimodal_gen.runtime.models.dits import wanvideo, zimage
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.platforms.interface import Platform
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    compile_matching_submodules,
)
from sglang.multimodal_gen.test.unit.test_usp_attention_kv_gather import (
    _RankInvariantMeta,
)

_COMPILE = "sglang.multimodal_gen.runtime.utils.torch_compile"


class _Region(nn.Module):
    def __init__(self):
        super().__init__()
        self.compiled = []  # (forward, compile kwargs) per compile call

    def compile(self, **kwargs):
        self.compiled.append((self.forward, kwargs))

    def forward(self, x):
        return x


class _RegionalModel(nn.Module):
    _compile_conditions = [lambda _name, module: isinstance(module, _Region)]

    def __init__(self, spmd_safe):
        super().__init__()
        self._compile_regions_spmd_safe = spmd_safe
        self.blocks = nn.ModuleList([_Region(), _Region()])


def _wrap_region(forward):
    def region(*args, **kwargs):
        return forward(*args, **kwargs)

    region.wrapped = forward
    return region


class TestSpmdSafeRegions(unittest.TestCase):
    def _compile(self, model, compile_kwargs):
        with patch(f"{_COMPILE}.current_platform") as platform:
            platform.prepare_spmd_region_wrapper.return_value = _wrap_region
            compile_matching_submodules(model, compile_kwargs=compile_kwargs)

    def test_marked_regions_are_wrapped_then_compiled_as_requested(self):
        model = _RegionalModel(spmd_safe=True)
        self._compile(model, {"backend": "tpu", "fullgraph": False})
        for block in model.blocks:
            ((forward, kwargs),) = block.compiled
            self.assertIs(forward.wrapped.__self__, block)
            # Not fullgraph: past the recompile limit a region runs eagerly.
            self.assertEqual(kwargs, {"backend": "tpu", "fullgraph": False})

    def test_unmarked_regions_compile_as_requested(self):
        model = _RegionalModel(spmd_safe=False)
        self._compile(model, {"backend": "tpu", "fullgraph": False})
        for block in model.blocks:
            ((forward, kwargs),) = block.compiled
            self.assertEqual(forward, block.forward)
            self.assertEqual(kwargs, {"backend": "tpu", "fullgraph": False})

    def test_regions_are_unwrapped_unless_the_model_opts_in(self):
        self.assertIsNone(Platform().prepare_spmd_region_wrapper())
        self.assertFalse(BaseDiT._compile_regions_spmd_safe)
        self.assertTrue(wanvideo.WanTransformer3DModel._compile_regions_spmd_safe)
        self.assertTrue(zimage.ZImageTransformer2DModel._compile_regions_spmd_safe)


class TestFusionBarrier(unittest.TestCase):
    def test_default_platform_returns_the_input(self):
        x = torch.arange(4.0)
        self.assertIs(Platform.fusion_barrier(x), x)

    def test_wan_ffn_runs_the_barrier_between_the_activation_and_fc_out(self):
        calls = []
        fc_out_inputs = []
        barrier_out = torch.full((2,), 7.0)

        def step(name, value):
            calls.append(name)
            return value

        mlp = SimpleNamespace(
            fc_in=lambda x: (step("fc_in", x), None),
            fc_out=lambda x: (fc_out_inputs.append(x), (step("fc_out", x), None))[1],
            _apply_activation=lambda x, bias, use_fused_bias_gelu: step("act", x),
        )
        with (
            patch.object(wanvideo, "fused_gelu_active", return_value=False),
            patch.object(wanvideo, "nvfp4_bias_gelu_active", return_value=False),
            patch.object(
                wanvideo.current_platform,
                "fusion_barrier",
                side_effect=lambda x: (step("barrier", x), barrier_out)[1],
            ),
        ):
            wanvideo._WanGELUMLP.forward(mlp, torch.ones(2))
        self.assertEqual(calls, ["fc_in", "act", "barrier", "fc_out"])
        # A compiler drops a barrier whose result is unused.
        self.assertIs(fc_out_inputs[0], barrier_out)


class _Stop(Exception):
    """Ends the forward once block 0's inputs are recorded."""


class _RecordingBlock(nn.Module):
    def forward(self, *args, attn_mask_meta=None, **kwargs):
        self.args = args
        # A program depends only on the meta entries it reads, and the
        # attention paths read the global pad span, never the rank-local
        # entries (the tests that use _RankInvariantMeta check them).
        meta = {
            key: value
            for key, value in (attn_mask_meta or {}).items()
            if key not in _RankInvariantMeta.rank_local_keys
        }
        leaves = tree_leaves((args, kwargs, meta))
        self.layouts = [
            (tuple(leaf.shape), leaf.stride(), leaf.storage_offset())
            for leaf in leaves
            if isinstance(leaf, torch.Tensor)
        ]
        # Dynamo specializes on Python scalars it reads, so a rank-dependent
        # scalar changes the program even when every tensor layout matches.
        self.scalars = [leaf for leaf in leaves if not isinstance(leaf, torch.Tensor)]
        raise _Stop


@contextlib.contextmanager
def _sp_rank(rank, world_size):
    tp_group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=-1: x
    )
    sp_group = SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        ulysses_world_size=world_size,
        ring_world_size=1,
    )
    with (
        patch.object(parallel_state, "_TP", tp_group),
        patch.object(parallel_state, "_SP", sp_group),
    ):
        yield


def _tiny_wan():
    arch = WanVideoArchConfig(
        num_attention_heads=2,
        attention_head_dim=8,
        in_channels=4,
        out_channels=4,
        text_dim=8,
        freq_dim=8,
        ffn_dim=16,
        num_layers=1,
    )
    model = wanvideo.WanTransformer3DModel(WanVideoConfig(arch_config=arch), {})
    torch.manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.02)
    return model


def _set_tiny_wan_server_args(test):
    previous = server_args_module._global_server_args
    server_args_module.set_global_server_args(
        SimpleNamespace(
            attention_backend=None,
            attention_backend_config=None,
            kv_gather_degree=1,
            sp_split_auto=False,
        )
    )
    test.addCleanup(server_args_module.set_global_server_args, previous)


class TestWanBlockInputsAreRankInvariant(unittest.TestCase):
    def setUp(self):
        _set_tiny_wan_server_args(self)

    def test_every_rank_feeds_blocks_offset_zero_shards(self):
        # Latents (frames, height, width), patch (1, 2, 2), at SP 4. 2 x 2 x 4 =
        # 16 tokens, 4 per rank: with batch 1 a rank's slice is a contiguous
        # view whose storage offset grows with the rank. 3 x 3 x 3 = 27 tokens,
        # 7 per rank: the last rank's shard ends in a pad token.
        world_size = 4
        forward_batch = SimpleNamespace(
            enable_sequence_shard=True, enable_teacache=False, enable_spectrum=False
        )
        context = SimpleNamespace(forward_batch=forward_batch, attn_metadata=None)
        for latent_shape, batch, per_token_timestep in (
            ((2, 4, 8), 1, False),
            ((2, 4, 8), 1, True),
            ((2, 4, 8), 2, True),
            ((3, 6, 6), 1, True),
            ((3, 6, 6), 2, True),
        ):
            seq_len = latent_shape[0] * latent_shape[1] * latent_shape[2] // 4
            local_len = -(-seq_len // world_size)
            num_pad = local_len * world_size - seq_len
            latents = torch.randn(batch, 4, *latent_shape)
            text = torch.randn(batch, 5, 8)
            timestep = (
                torch.full((batch, seq_len), 500.0)
                if per_token_timestep
                else torch.full((batch,), 500.0)
            )
            layouts, scalars = [], []
            for rank in range(world_size):
                case = dict(
                    tokens=seq_len, batch=batch, per_token=per_token_timestep, rank=rank
                )
                with self.subTest(**case), _sp_rank(rank, world_size):
                    model = _tiny_wan()
                    block = _RecordingBlock()
                    model.blocks = nn.ModuleList([block])
                    with (
                        patch.object(
                            wanvideo, "get_forward_context", return_value=context
                        ),
                        self.assertRaises(_Stop),
                    ):
                        model(latents, text, timestep)
                    layouts.append(block.layouts)
                    scalars.append(block.scalars)

                    # Shards of the sequence zero-padded at its tail.
                    hidden_states, _, temb, _ = block.args
                    rows = slice(rank * local_len, (rank + 1) * local_len)
                    tokens = model.patch_embedding(latents).flatten(2).transpose(1, 2)
                    torch.testing.assert_close(
                        hidden_states,
                        F.pad(tokens, (0, 0, 0, num_pad))[:, rows],
                        rtol=0,
                        atol=0,
                    )
                    sharded = [hidden_states]
                    if per_token_timestep:
                        _, proj, _, _ = model.condition_embedder(
                            timestep.flatten(), text, None, timestep_seq_len=seq_len
                        )
                        proj = F.pad(proj, (0, 0, 0, num_pad)).unflatten(2, (6, -1))
                        torch.testing.assert_close(temb, proj[:, rows], rtol=0, atol=0)
                        sharded.append(temb)
                    for tensor in sharded:
                        self.assertEqual(tensor.storage_offset(), 0)
                        self.assertTrue(tensor.is_contiguous())
            # Same shapes, strides, offsets and non-tensor arguments on every rank.
            for rank in range(world_size):
                self.assertEqual(layouts[rank], layouts[0], f"rank {rank}")
                self.assertEqual(scalars[rank], scalars[0], f"rank {rank}")


def _native_forward(op):
    # TPU's dispatch: forward_native, whose nested torch.compile is off on TPU.
    return inspect.unwrap(type(op).forward_native).__get__(op)


class TestWanBlockCompilesAsOneGraph(unittest.TestCase):
    """A graph break inside a Wan block splits its SPMD-safe region, so the
    region's collectives run in separate programs again."""

    def setUp(self):
        _set_tiny_wan_server_args(self)
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def test_wrapped_blocks_compile_with_fullgraph(self):
        forward_batch = SimpleNamespace(
            enable_sequence_shard=False, enable_teacache=False, enable_spectrum=False
        )
        with (
            _sp_rank(0, 1),
            patch.object(CustomOp, "dispatch_forward", _native_forward),
            # The fallback TPU selects; the CPU default calls sgl_kernel.
            patch.object(
                layernorm, "fuse_scale_shift_kernel", fuse_scale_shift_kernel_native
            ),
            set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=forward_batch
            ),
            torch.no_grad(),
        ):
            model = _tiny_wan()
            text = torch.randn(1, 5, 8)
            latents = torch.randn(1, 4, 2, 4, 8)
            timesteps = (torch.full((1,), 500.0), torch.full((1, 16), 500.0))
            eager = [model(latents, text, timestep) for timestep in timesteps]
            with patch(f"{_COMPILE}.current_platform") as platform:
                platform.prepare_spmd_region_wrapper.return_value = _wrap_region
                compile_matching_submodules(
                    model,
                    compile_kwargs=dict(backend="eager", fullgraph=True, dynamic=False),
                )
            for timestep, expected in zip(timesteps, eager):
                with self.subTest(timestep_shape=tuple(timestep.shape)):
                    torch.testing.assert_close(
                        model(latents, text, timestep), expected, rtol=0, atol=0
                    )


class TestDrainAfterRequest(unittest.TestCase):
    """When the platform asks, every rank drains its device queue after each
    request whose forward returned, warmups included: on TPU a skipped drain
    hung Wan2.2-A14B's second warmup request with 7 of its 8 ranks blocked."""

    def _serve(self, *, drain, forward, num_requests, output_rank=True, warmup=False):
        events = []
        worker = gpu_worker.GPUWorker.__new__(gpu_worker.GPUWorker)
        worker.pipeline = SimpleNamespace(
            forward=lambda req, server_args: (events.append("forward"), forward())[1]
        )
        worker.server_args = SimpleNamespace()
        worker.is_output_rank = output_rank
        worker._release_warmup_pool_before_serving = False
        worker._realtime_sessions = SimpleNamespace(attach=lambda _req: None)
        requests = [
            Req(
                sampling_params=SamplingParams(
                    request_id=f"req-{i}", save_output=False, suppress_logs=True
                )
            )
            for i in range(num_requests)
        ]
        if warmup:
            for req in requests:
                req.set_as_warmup()
        platform = gpu_worker.current_platform
        with (
            patch.object(platform, "is_cpu", return_value=True),
            patch.object(
                platform, "drain_device_queue_after_request", return_value=drain
            ),
            patch.object(
                torch.accelerator,
                "synchronize",
                side_effect=lambda: events.append("drain"),
            ),
        ):
            outputs = [worker.execute_forward([req]) for req in requests]
        return events, outputs

    def test_every_finished_forward_drains_when_the_platform_asks(self):
        for drain, output_rank, warmup in itertools.product(
            (True, False), (True, False), (False, True)
        ):
            with self.subTest(drain=drain, output_rank=output_rank, warmup=warmup):
                events, _ = self._serve(
                    drain=drain,
                    forward=OutputBatch,
                    num_requests=2,
                    output_rank=output_rank,
                    warmup=warmup,
                )
                expected = ["forward", "drain"] if drain else ["forward"]
                self.assertEqual(events, expected * 2)

    def test_a_failed_forward_returns_its_error_without_a_drain(self):
        def fail():
            raise RuntimeError("forward failed")

        with self.assertLogs(gpu_worker.logger, "ERROR"):
            events, (output,) = self._serve(drain=True, forward=fail, num_requests=1)
        self.assertEqual(events, ["forward"])
        self.assertIn("forward failed", output.error)


if __name__ == "__main__":
    unittest.main()
