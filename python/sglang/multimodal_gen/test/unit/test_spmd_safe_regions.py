"""Regional-compile hooks: the SPMD-safe opt-in contract, the rank-invariant
inputs Wan feeds its blocks, and the fusion barrier."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils._pytree import tree_leaves

from sglang.multimodal_gen.configs.models.dits.wanvideo import (
    WanVideoArchConfig,
    WanVideoConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.models.dits import wanvideo
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms.interface import Platform
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    compile_matching_submodules,
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


class TestFusionBarrier(unittest.TestCase):
    def test_default_platform_returns_the_input(self):
        x = torch.arange(4.0)
        self.assertIs(Platform.fusion_barrier(x), x)

    def test_wan_ffn_runs_the_barrier_between_the_activation_and_fc_out(self):
        calls = []

        def step(name, value):
            calls.append(name)
            return value

        mlp = SimpleNamespace(
            fc_in=lambda x: (step("fc_in", x), None),
            fc_out=lambda x: (step("fc_out", x), None),
            _apply_activation=lambda x, bias, use_fused_bias_gelu: step("act", x),
        )
        with (
            patch.object(wanvideo, "fused_gelu_active", return_value=False),
            patch.object(wanvideo, "nvfp4_bias_gelu_active", return_value=False),
            patch.object(
                wanvideo.current_platform,
                "fusion_barrier",
                side_effect=lambda x: step("barrier", x),
            ),
        ):
            wanvideo._WanGELUMLP.forward(mlp, torch.ones(2))
        self.assertEqual(calls, ["fc_in", "act", "barrier", "fc_out"])


class _Stop(Exception):
    """Ends the forward once block 0's inputs are recorded."""


# Rank-local entries of the shared tail-pad meta. Only the CUDA varlen-FA
# K/V-gather path reads it; the paths compiled in spmd_safe regions mask from
# the global pad_start/pad_end instead.
_RANK_LOCAL_META_KEYS = frozenset({"local_pad"})


class _RecordingBlock(nn.Module):
    def forward(self, *args, **kwargs):
        self.args = args
        self.layouts = [
            (tuple(leaf.shape), leaf.stride(), leaf.storage_offset())
            for leaf in tree_leaves((args, kwargs))
            if isinstance(leaf, torch.Tensor)
        ]
        # Dynamo specializes on Python scalars it reads, so a rank-dependent
        # scalar changes the program even when every tensor layout matches.
        meta = kwargs.get("attn_mask_meta") or {}
        rest = {k: v for k, v in kwargs.items() if k != "attn_mask_meta"}
        self.scalars = (
            [
                leaf
                for leaf in tree_leaves((args, rest))
                if not isinstance(leaf, torch.Tensor)
            ],
            {
                key: value
                for key, value in meta.items()
                if not isinstance(value, torch.Tensor)
                and key not in _RANK_LOCAL_META_KEYS
            },
        )
        raise _Stop


@contextlib.contextmanager
def _sp_rank(rank, world_size, ring_world_size=1):
    tp_group = SimpleNamespace(
        world_size=1, rank_in_group=0, all_gather=lambda x, dim=-1: x
    )
    sp_group = SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        ulysses_world_size=world_size // ring_world_size,
        ring_world_size=ring_world_size,
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


class TestWanBlockInputsAreRankInvariant(unittest.TestCase):
    def setUp(self):
        previous = server_args_module._global_server_args
        server_args_module.set_global_server_args(
            SimpleNamespace(
                attention_backend=None,
                attention_backend_config=None,
                kv_gather_degree=1,
                sp_split_auto=False,
            )
        )
        self.addCleanup(server_args_module.set_global_server_args, previous)

    def test_every_rank_feeds_blocks_offset_zero_shards(self):
        # 2 x 2 x 4 = 16 latent tokens, 4 per rank. With batch 1 a rank's slice
        # is a contiguous view whose storage offset grows with the rank.
        world_size, local_len = 4, 4
        forward_batch = SimpleNamespace(
            enable_sequence_shard=True, enable_teacache=False, enable_spectrum=False
        )
        context = SimpleNamespace(forward_batch=forward_batch, attn_metadata=None)
        for batch, per_token_timestep in ((1, False), (1, True), (2, True)):
            latents = torch.randn(batch, 4, 2, 4, 8)
            text = torch.randn(batch, 5, 8)
            timestep = (
                torch.full((batch, 16), 500.0)
                if per_token_timestep
                else torch.full((batch,), 500.0)
            )
            layouts = []
            for rank in range(world_size):
                case = dict(batch=batch, per_token=per_token_timestep, rank=rank)
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

                    hidden_states, _, temb, _ = block.args
                    rows = slice(rank * local_len, (rank + 1) * local_len)
                    tokens = model.patch_embedding(latents).flatten(2).transpose(1, 2)
                    torch.testing.assert_close(
                        hidden_states, tokens[:, rows], rtol=0, atol=0
                    )
                    sharded = [hidden_states]
                    if per_token_timestep:
                        _, proj, _, _ = model.condition_embedder(
                            timestep.flatten(), text, None, timestep_seq_len=16
                        )
                        torch.testing.assert_close(
                            temb, proj.unflatten(2, (6, -1))[:, rows], rtol=0, atol=0
                        )
                        sharded.append(temb)
                    for tensor in sharded:
                        self.assertEqual(tensor.storage_offset(), 0)
                        self.assertTrue(tensor.is_contiguous())
            # Same shapes, strides and offsets for every tensor on every rank.
            for rank, layout in enumerate(layouts):
                self.assertEqual(layout, layouts[0], f"rank {rank}")


if __name__ == "__main__":
    unittest.main()
