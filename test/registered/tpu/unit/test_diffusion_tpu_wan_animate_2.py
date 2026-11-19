# SPDX-License-Identifier: Apache-2.0
"""Wan-Animate-2 on TPU: sglang_tpu.multimodal_gen.wan_animate_2 against the stock
flex path, the compiled blocks, the rank shards and the config patches."""

import contextlib
import importlib.util
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    fuse_scale_shift_kernel_native,
)
from sglang.multimodal_gen.configs.models.dits.wan_animate_2 import (
    WanAnimate2ArchConfig,
    WanAnimate2Config,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan_Animate_2_14B_Config,
    Wan_Animate_2_14B_Distilled_Config,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import wan_animate_2 as dit
from sglang.multimodal_gen.runtime.models.dits import wan_animate_2_block as block
from sglang.multimodal_gen.runtime.models.dits.wan_animate_2_clip_conditioning import (
    WanAnimate2ClipConditioning,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    compile_matching_submodules,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import is_in_ci
from sglang_tpu.multimodal_gen import platform as tpu_platform
from sglang_tpu.multimodal_gen import wan_animate_2

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=60, suite="nightly-tpu-4-v7", nightly=True)

_COMPILE = "sglang.multimodal_gen.runtime.utils.torch_compile"
Block = block.WanAnimate2TransformerBlock
Layout = block._InContextLayout

# (generation grid, reference grid, full-clip grid): a full clip, a short last clip,
# reference frames one column narrower (the CI inputs) and wider than the generation
# frames, and a short clip with a wider reference.
LAYOUTS = (
    ((3, 4, 8), (2, 4, 8), (3, 4, 8)),
    ((2, 4, 8), (1, 4, 8), (4, 4, 8)),
    ((3, 4, 8), (2, 4, 7), (3, 4, 8)),
    ((3, 4, 7), (2, 4, 8), (3, 4, 7)),
    ((2, 4, 7), (1, 5, 7), (5, 4, 7)),
)


@contextlib.contextmanager
def _installed(device_types=("cpu",)):
    """The TPU paths on ``device_types``, with every patched name restored."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (Block, "_in_context_attention"),
            (Block, "__call__"),
            (dit.WanAnimate2Transformer3DModel, "_get_block_mask_from_layout"),
            (block, "_apply_rope_interleaved"),
            (Wan_Animate_2_14B_Config, "validate_server_args"),
            (Wan_Animate_2_14B_Config, "get_model_deployment_config"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(wan_animate_2, "_DEVICE_TYPES", device_types))
        wan_animate_2.install()
        wan_animate_2.install()  # idempotent
        yield


def _inputs(layout, sp_pad=0, heads=2, dim=16, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    n, nr = layout.generation_video_num_tokens, layout.reference_video_num_tokens
    q, k, v = (torch.randn(1, n + sp_pad, heads, dim).to(dtype) for _ in range(3))
    rk, rv = (
        block._pad_reference_video_kv(torch.randn(1, nr, heads, dim).to(dtype), layout)
        for _ in range(2)
    )
    return q, k, v, rk, rv


def _stock_flex(layout, log_scale, q, k, v, rk, rv):
    """The stock in-context attention, flex_attention eager on the CPU in float32."""
    mask = create_block_mask(
        layout.block_mask_fn(),
        B=None,
        H=None,
        Q_LEN=layout.padded_generation_video_len,
        KV_LEN=layout.padded_kv_len,
        device="cpu",
        _compile=False,
    )
    with patch.object(block, "_flex_attention_compiled", flex_attention):
        return Block._in_context_attention(
            SimpleNamespace(log_scale=log_scale),
            *(t.cpu().float() for t in (q, k, v, rk, rv)),
            mask,
            layout,
        )


class TestInContextAttention(CustomTestCase):
    def test_matches_stock_flex(self):
        for grids in LAYOUTS:
            for log_scale in (0.0, -1.3):
                for sp_pad in (0, 3):
                    with self.subTest(grids=grids, log_scale=log_scale, sp_pad=sp_pad):
                        layout = Layout(*grids)
                        inputs = _inputs(layout, sp_pad)
                        torch.testing.assert_close(
                            wan_animate_2.in_context_attention(
                                *inputs, layout, log_scale
                            ),
                            _stock_flex(layout, log_scale, *inputs),
                            rtol=0,
                            atol=2e-6,
                        )

    def test_tpu_tensors_take_the_split_attention(self):
        layout = Layout(*LAYOUTS[1])
        q, k, v, rk, rv = _inputs(layout)
        stub = SimpleNamespace(log_scale=0.0)
        with _installed(), patch.object(block, "_flex_attention_compiled") as flex:
            out = Block._in_context_attention(stub, q, k, v, rk, rv, None, layout)
        flex.assert_not_called()
        torch.testing.assert_close(out, _stock_flex(layout, 0.0, q, k, v, rk, rv))
        with (
            _installed(device_types=()),
            patch.object(
                block, "_flex_attention_compiled", side_effect=RuntimeError("flex")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "flex"):
                Block._in_context_attention(stub, q, k, v, rk, rv, None, layout)

    def test_no_block_mask_on_tpu(self):
        stub = SimpleNamespace(_device=torch.device("cpu"))
        with _installed(), patch.object(dit, "create_block_mask") as create:
            mask = dit.WanAnimate2Transformer3DModel._get_block_mask_from_layout(
                stub, Layout(*LAYOUTS[0])
            )
        self.assertIsNone(mask)
        create.assert_not_called()


class TestOnTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec("torch_tpu") is None:
            if is_in_ci():
                raise RuntimeError("TPU CI requires torch_tpu and a TPU device")
            raise unittest.SkipTest("torch_tpu is not installed")
        from sglang_tpu.env_override import apply_tpu_env_overrides

        apply_tpu_env_overrides()
        import torch_tpu  # noqa: F401

        torch.zeros(1, device="tpu").cpu()

    def test_kernel_matches_stock_flex(self):
        # Shapes the flash kernel takes: over 512 tokens per frame, head dim 128.
        layouts = (
            Layout((3, 16, 40), (2, 16, 40), (3, 16, 40)),
            Layout((2, 16, 40), (1, 16, 39), (4, 16, 40)),
        )
        for layout in layouts:
            for log_scale in (0.0, -1.3):
                with self.subTest(layout=layout.max_tokens_per_frame, ls=log_scale):
                    inputs = _inputs(layout, heads=2, dim=128, dtype=torch.bfloat16)
                    expected = _stock_flex(layout, log_scale, *inputs)
                    with patch.object(
                        wan_animate_2.splash,
                        "_attention_with_lse_reference",
                        side_effect=AssertionError("kernel expected"),
                    ):
                        actual = wan_animate_2.in_context_attention(
                            *(t.to("tpu") for t in inputs), layout, log_scale
                        )
                    actual = actual.cpu().float()
                    # bf16 output: one rounding of the float32 result, plus the
                    # kernel's bf16 probabilities.
                    err = (actual - expected).abs().max().item()
                    self.assertLess(err, 2e-2 * expected.abs().max().item())

    def test_pair_swap_rope_matches_stock(self):
        torch.manual_seed(0)
        x = torch.randn(1, 300, 4, 128).bfloat16()
        angles = torch.rand(300, 64) * 30
        cos, sin = angles.cos(), angles.sin()
        expected = block._apply_rope_interleaved(x, cos, sin).float()
        with _installed(device_types=("tpu",)):
            actual = block._apply_rope_interleaved(
                *(t.to("tpu") for t in (x, cos, sin))
            )
        actual = actual.cpu().float()
        # Only float32 rounding-order ties (FMA or not) may flip a bf16 rounding.
        ulp = expected.abs().clamp_min(2**-126) * 2**-7
        self.assertTrue(bool(((actual - expected).abs() <= ulp).all()))
        self.assertLess((actual != expected).float().mean().item(), 1e-3)


@contextlib.contextmanager
def _tiny_model_env(rank=0, world_size=1):
    tp = SimpleNamespace(world_size=1, rank_in_group=0, all_gather=lambda x, dim=-1: x)
    sp = SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        ulysses_world_size=world_size,
        ring_world_size=1,
    )
    previous = server_args_module._global_server_args
    server_args_module.set_global_server_args(
        SimpleNamespace(
            attention_backend=None,
            attention_backend_config=None,
            kv_gather_degree=1,
            sp_split_auto=False,
        )
    )
    forward_batch = SimpleNamespace(
        enable_sequence_shard=False,
        enable_teacache=False,
        enable_spectrum=False,
        extra={},
    )
    try:
        with (
            patch.object(parallel_state, "_TP", tp),
            patch.object(parallel_state, "_SP", sp),
            patch.object(
                CustomOp,
                "dispatch_forward",
                # TPU's dispatch: forward_native, whose nested torch.compile is off on TPU.
                lambda op: inspect.unwrap(type(op).forward_native).__get__(op),
            ),
            patch.object(
                layernorm, "fuse_scale_shift_kernel", fuse_scale_shift_kernel_native
            ),
            set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=forward_batch
            ),
            torch.no_grad(),
        ):
            yield
    finally:
        server_args_module.set_global_server_args(previous)


def _tiny_model(log_scale=0.0):
    arch = WanAnimate2ArchConfig(
        num_attention_heads=2,
        attention_head_dim=8,
        text_dim=8,
        freq_dim=8,
        ffn_dim=16,
        num_layers=10,
        image_dim=8,
        added_kv_proj_dim=16,
        log_scale=log_scale,
    )
    arch.text_len = 6
    model = dit.WanAnimate2Transformer3DModel(WanAnimate2Config(arch_config=arch), {})
    torch.manual_seed(0)
    for parameter in model.parameters():
        parameter.normal_(std=0.05)
    return model


def _clip(frames=3, full_frames=4, h=2, w=4):
    """A short clip (``frames`` of ``full_frames`` latent frames) and its noise."""
    H, W = 2 * h, 2 * w
    ref = torch.randn(16, frames - 1, H, W)
    cond = WanAnimate2ClipConditioning(
        reference_video_latents=ref,
        generation_video_grid_sizes=(frames, h, w),
        reference_video_frame_0_image_embeddings=torch.randn(1, 4, 8),
        reference_video_condition=torch.cat([torch.ones(4, frames - 1, H, W), ref]),
        prompt_ref_embeddings=torch.randn(3, 8),
        generation_condition=torch.randn(20, frames, H, W),
        reference_video_grid_sizes=(frames - 1, h, w),
        full_clip_grid_sizes=(full_frames, h, w),
        num_frames=4 * (frames - 2) + 1,
        init_noise=torch.zeros(1),
    )
    return cond, torch.randn(16, frames, H, W)


def _denoise_without_reference(model, cond, latent):
    return model(
        latent,
        torch.randn(5, 8),
        torch.full((1,), 900.0),
        encoder_hidden_states_image=torch.randn(1, 4, 8),
        reference_kv=SimpleNamespace(k={}, v={}, k_scale=None, v_scale=None),
        clip_cond=cond,
    )


def _denoise(model, cond, latent, unconditional):
    kv = model.build_reference_kv(clip_cond=cond)
    return model(
        latent,
        torch.randn(5, 8),
        torch.full((1,), 900.0),
        encoder_hidden_states_image=torch.randn(1, 4, 8),
        reference_kv=kv,
        clip_cond=cond,
        is_unconditional=unconditional,
    )


class TestCompiledBlocks(CustomTestCase):
    def setUp(self):
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        # As TpuOOTPlatform sets it: a new int is a recompile, not a dynamic int.
        dynamo_config = patch.object(
            torch._dynamo.config, "automatic_dynamic_shapes", False
        )
        dynamo_config.start()
        self.addCleanup(dynamo_config.stop)

    def test_every_block_shares_one_graph_per_pass(self):
        graphs = []

        def backend(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        with _tiny_model_env(), _installed():
            model = _tiny_model()
            cond, latent = _clip()
            torch.manual_seed(1)
            eager = [_denoise(model, cond, latent, u) for u in (False, True)]
            with patch(f"{_COMPILE}.current_platform") as platform:
                platform.prepare_spmd_region_wrapper.return_value = None
                compile_matching_submodules(
                    model,
                    compile_kwargs=dict(backend=backend, fullgraph=True, dynamic=False),
                )
            for _ in range(2):  # a second request compiles nothing
                torch.manual_seed(1)
                compiled = [_denoise(model, cond, latent, u) for u in (False, True)]
                for actual, expected in zip(compiled, eager):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The reference pass and the generation pass, for all 10 blocks.
        self.assertEqual(len(graphs), 2)


class _Stop(Exception):
    pass


class _FirstBlock(torch.nn.Module):
    def forward(self, **kwargs):
        self.hidden_states = kwargs["hidden_states"]
        raise _Stop


class TestRankShards(CustomTestCase):
    def test_every_rank_feeds_blocks_an_offset_zero_shard(self):
        # 3 x 2 x 4 = 24 generation and 16 reference tokens over 4 ranks: a rank's
        # slice is a view whose storage offset grows with the rank.
        cond, latent = _clip()
        for rank in range(4):
            with self.subTest(rank=rank), _tiny_model_env(rank, 4), _installed():
                model = _tiny_model()
                first = _FirstBlock()
                model.blocks = torch.nn.ModuleList([first])
                with self.assertRaises(_Stop):
                    model.build_reference_kv(clip_cond=cond)
                shards = [first.hidden_states]
                with self.assertRaises(_Stop):
                    _denoise_without_reference(model, cond, latent)
                shards.append(first.hidden_states)
                for shard, tokens in zip(shards, (16, 24)):
                    self.assertEqual(shard.shape[1], tokens // 4)
                    self.assertEqual(shard.storage_offset(), 0)
                    self.assertTrue(shard.is_contiguous())
                    self.assertEqual(shard.stride(0), shard[0].numel())


def _server_args(backend, component=None):
    return SimpleNamespace(
        tp_size=1,
        sp_degree=4,
        ulysses_degree=4,
        ring_degree=1,
        enable_cfg_parallel=True,
        use_fsdp_inference=False,
        attention_backend=backend,
        is_arg_explicitly_set=lambda name: name == "attention_backend",
        requested_component_attention_backend=lambda name: (component or {}).get(name),
    )


class TestServerArgs(CustomTestCase):
    def test_tpu_takes_the_platform_attention_backend(self):
        def on_tpu(value):
            return patch.object(type(current_platform), "is_tpu", lambda self: value)

        with _installed():
            for config in (
                Wan_Animate_2_14B_Config(),
                Wan_Animate_2_14B_Distilled_Config(),
            ):
                with on_tpu(False):
                    with self.assertRaisesRegex(ValueError, "flex_attention only"):
                        config.validate_server_args(_server_args("oot"))
                with on_tpu(True):
                    args = _server_args("oot")
                    config.validate_server_args(args)
                    self.assertEqual(args.attention_backend, "oot")
                    for rejected in (
                        _server_args("fa"),
                        _server_args("oot", {"transformer": "torch_sdpa"}),
                    ):
                        with self.assertRaisesRegex(ValueError, "flex_attention only"):
                            config.validate_server_args(rejected)

    def test_cfg_parallel_is_refused(self):
        # Two CFG halves running SP subgroups at once are not reproducible on
        # TPU, and the unconditional branch skips block 9.
        with _installed():
            for config in (
                Wan_Animate_2_14B_Config(),
                Wan_Animate_2_14B_Distilled_Config(),
            ):
                deployment = config.get_model_deployment_config()
                self.assertFalse(deployment.supports_cfg_parallel)
                self.assertFalse(deployment.auto_enable_cfg_parallel)
                self.assertEqual(deployment.keep_resident_components, ("dit",))

    def test_platform_installs_the_config_in_the_server_process(self):
        config = Wan_Animate_2_14B_Config
        with (
            patch.object(config, "validate_server_args", config.validate_server_args),
            patch.object(
                config,
                "get_model_deployment_config",
                config.get_model_deployment_config,
            ),
        ):
            tpu_platform.TpuOOTPlatform().apply_server_args_defaults(
                SimpleNamespace(
                    is_arg_explicitly_set=lambda name: False,
                    offload_during_compile=False,
                )
            )
            self.assertIs(
                config.validate_server_args, wan_animate_2._validate_server_args
            )
            self.assertIs(
                config.get_model_deployment_config,
                wan_animate_2._get_model_deployment_config,
            )


if __name__ == "__main__":
    unittest.main()
