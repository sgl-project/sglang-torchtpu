# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's HunyuanVideo patches (sglang_tpu.multimodal_gen.hunyuan_video)."""

import contextlib
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.compiler.config as compiler_config
import torch.nn.functional as F
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.dits.hunyuanvideo import (
    HunyuanVideoArchConfig,
    HunyuanVideoConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs import hunyuan as config
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.layers.rotary_embedding import mrope
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import hunyuanvideo as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages import latent_preparation
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.utils import torch_compile
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import hunyuan_video, worker
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=60, suite="nightly-tpu-4-v7", nightly=True)


@contextlib.contextmanager
def _installed():
    """The TPU paths on CPU, with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (config.HunyuanConfig, "tokenize_prompt"),
            (dit.HunyuanVideoTransformer3DModel, "forward"),
            (dit.HunyuanVideoTransformer3DModel, "__call__"),
            (dit.HunyuanVideoTransformer3DModel, "_compile_conditions"),
            (dit.HunyuanVideoTransformer3DModel, "_compile_regions_spmd_safe"),
            (USPAttention, "forward"),
            (latent_preparation, "randn_tensor"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(hunyuan_video, "_DEVICE_TYPES", ("cpu",)))
        hunyuan_video.install()
        hunyuan_video.install()  # idempotent
        yield


def _model():
    torch.manual_seed(0)
    arch = HunyuanVideoArchConfig(
        num_attention_heads=2,
        attention_head_dim=32,
        num_layers=1,
        num_single_layers=1,
        num_refiner_layers=1,
        rope_axes_dim=(8, 12, 12),
        text_embed_dim=48,
        pooled_projection_dim=16,
        guidance_embeds=True,
    )
    with patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_tpu):
        model = dit.HunyuanVideoTransformer3DModel(
            config=HunyuanVideoConfig(arch_config=arch), hf_config={}
        )
    for param in model.parameters():
        param.data.normal_(0, 0.05)
    return model.float().eval()


def _inputs(text_len, padded_len=None):
    """One request's DiT kwargs; ``padded_len`` right-pads the prompt as the TPU tokenizer does."""
    gen = torch.Generator().manual_seed(text_len)
    text = torch.randn(1, text_len, 48, generator=gen)
    mask = torch.ones(1, text_len, dtype=torch.bool)
    if padded_len is not None:
        text = F.pad(text, (0, 0, 0, padded_len - text_len), value=0.5)
        mask = F.pad(mask, (0, padded_len - text_len))
    return dict(
        hidden_states=torch.randn(1, 16, 2, 8, 8, generator=gen),
        encoder_hidden_states=text,
        timestep=torch.full((1,), 500.0),
        guidance=torch.full((1,), 6016.0),
        pooled_projections=torch.randn(1, 16, generator=gen),
        encoder_attention_mask=mask,
    )


def _run(model, kwargs):
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        return model(**kwargs)


def _leaves(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _leaves(item)
    else:
        yield value


def _region_inputs(model, kwargs, sp, rank):
    """Each block's inputs as rank ``rank`` of ``sp`` Ulysses ranks: tensor
    shapes, strides, storage offsets and dtypes, and the other arguments."""
    calls = []

    def record(module, args):
        calls.append(
            [
                (t.shape, t.stride(), t.storage_offset(), t.dtype)
                if isinstance(t, torch.Tensor)
                else t
                for t in _leaves(args)
            ]
        )

    def input_all_to_all(t, head_dim):  # [B, S, H, D] -> [B, sp * S, H / sp, D]
        return t[:, :, : t.shape[2] // sp].repeat(1, sp, 1, 1)

    def output_all_to_all(t, head_dim):  # [B, sp * S, H / sp, D] -> [B, S, H, D]
        return t[:, : t.shape[1] // sp].repeat(1, 1, sp, 1)

    blocks = (*model.double_blocks, *model.single_blocks)
    hooks = [block.register_forward_pre_hook(record) for block in blocks]
    group = SimpleNamespace(world_size=sp, rank_in_group=rank)
    try:
        with (
            patch.multiple(
                hunyuan_video,
                get_sp_world_size=lambda: sp,
                get_sp_parallel_rank=lambda: rank,
                _usp_input_all_to_all=input_all_to_all,
                _usp_output_all_to_all=output_all_to_all,
            ),
            patch.object(mrope, "get_sp_group", lambda: group),
        ):
            _run(model, kwargs)
    finally:
        for hook in hooks:
            hook.remove()
    return calls


class _TestBase(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.diffusion.common import fallback_torch
        from sglang.multimodal_gen.runtime.layers import layernorm

        # The fused scale-shift kernel TPU workers select (the CPU one needs sgl_kernel).
        native = patch.object(
            layernorm,
            "fuse_scale_shift_kernel",
            fallback_torch.fuse_scale_shift_kernel_native,
        )
        native.start()
        self.addCleanup(native.stop)
        self._previous_args = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            with patch.object(worker, "install_runtime_patches"):
                maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1, sp_size=1
                )

    def tearDown(self):
        set_global_server_args(self._previous_args)


class TestHunyuanVideoDiT(_TestBase):
    def test_bucketed_prompt_matches_stock(self):
        model = _model()
        for text_len in (5, 40):
            expected = _run(model, _inputs(text_len))
            with _installed():
                actual = _run(model, _inputs(text_len, padded_len=161))
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-4)

    def test_tpu_latents_never_take_the_stock_forward(self):
        # Its uneven prompt shards would hang the spmd-safe blocks.
        kwargs = dict(_inputs(5, padded_len=161), pooled_projections=None)
        with _installed(), self.assertRaises(NotImplementedError):
            _run(_model(), kwargs)

    def test_refiner_on_a_padded_prompt_matches_stock(self):
        refiner = _model().txt_in
        gen = torch.Generator().manual_seed(3)
        x, t = torch.randn(1, 7, 48, generator=gen), torch.full((1,), 500.0)
        keys = torch.tensor([0, 7], dtype=torch.int32)
        padded = F.pad(x, (0, 0, 0, 512 - 7), value=0.5)
        with set_forward_context(current_timestep=0, attn_metadata=None):
            with torch.no_grad():
                expected = refiner(x, t)
                actual = hunyuan_video._refine(refiner, padded, t, keys)[:, :7]
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_region_inputs_match_across_ranks(self):
        # The spmd-safe contract: every rank runs each block with the same
        # shapes, strides, storage offsets and non-tensor arguments.
        model = _model()
        with _installed():
            for text_len in (5, 40):
                kwargs = _inputs(text_len, padded_len=161)
                ranks = [_region_inputs(model, kwargs, 2, r) for r in range(2)]
                self.assertEqual(len(ranks[0]), 2)  # one double, one single block
                self.assertEqual(ranks[0], ranks[1])
                calls = ranks[0] + ranks[1]
                offsets = [x[2] for call in calls for x in call if isinstance(x, tuple)]
                self.assertTrue(offsets and not any(offsets))

    def test_each_block_compiles_once_marked_spmd_safe(self):
        model, graphs = _model(), []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        registry = torch_compile.CompiledModuleRegistry()
        with (
            _installed(),
            patch.object(torch_compile, "current_platform", TpuOOTPlatform()),
            compiler_config.patch(cache_key_tag=""),
        ):
            _run(model, _inputs(5, padded_len=161))  # eager first, as above
            compile_kwargs = dict(backend=counting, dynamic=False)
            # The double- and single-stream blocks, not the token refiner.
            self.assertEqual(
                registry.compile_regions_once(model, compile_kwargs=compile_kwargs), 2
            )
            for text_len in (5, 40, 160):
                _run(model, _inputs(text_len, padded_len=161))
        self.assertEqual(len(graphs), 2)
        for graph in graphs:
            for node in graph.graph.nodes:
                if node.op == "call_function":
                    self.assertEqual(node.meta.get("custom"), {"spmd_safe": True})

    def test_whole_dit_compiles_once_across_prompt_lengths(self):
        model = _model()
        graphs = []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        with _installed():
            # An eager call first: fused ops resolve their backend lazily.
            _run(model, _inputs(5, padded_len=161))
            model.compile(backend=counting, dynamic=False)
            for text_len in (5, 40, 160):
                _run(model, _inputs(text_len, padded_len=161))
        self.assertEqual(len(graphs), 1)


class TestHunyuanVideoNoise(CustomTestCase):
    def test_initial_latents_are_drawn_in_float32(self):
        drawn = []

        def stock(shape, generator=None, device=None, dtype=None):
            drawn.append(dtype)
            return torch.zeros(shape, dtype=dtype)

        hunyuan, other = config.HunyuanConfig(), SimpleNamespace()
        with _installed(), patch.object(hunyuan_video, "_STOCK_RANDN", stock):
            for pipeline, dtype, draw in (
                (hunyuan, torch.float16, torch.float32),
                (hunyuan, torch.float32, torch.float32),
                (other, torch.float16, torch.float16),
            ):
                args = SimpleNamespace(pipeline_config=pipeline)
                with patch.object(
                    server_args_module._server_args, "_global_server_args", args
                ):
                    noise = latent_preparation.randn_tensor(
                        (2, 3), generator=None, device=torch.device("cpu"), dtype=dtype
                    )
                self.assertEqual((noise.dtype, drawn[-1]), (dtype, draw))
            # Other devices draw as stock.
            with patch.object(hunyuan_video, "_DEVICE_TYPES", ("tpu",)):
                latent_preparation.randn_tensor((2,), device="cpu", dtype=torch.float16)
        self.assertEqual(drawn[-1], torch.float16)


class TestHunyuanVideoTokens(CustomTestCase):
    def test_llama_tokens_pad_to_max_length(self):
        def tokenizer(prompt, **kwargs):
            return BatchEncoding(
                {
                    "input_ids": torch.tensor([[5, 6, 7]]),
                    "attention_mask": torch.ones(1, 3, dtype=torch.long),
                }
            )

        llama = type("LlamaTokenizerFast", (), {"pad_token_id": 0, "eos_token_id": 2})
        clip = type("CLIPTokenizer", (), {"pad_token_id": 0, "eos_token_id": 2})
        with _installed():
            for tok, expected in ((llama(), 8), (clip(), 3)):
                with patch.object(
                    hunyuan_video,
                    "_STOCK_TOKENIZE",
                    lambda self, prompts, t, kw: tokenizer(prompts),
                ):
                    out = config.HunyuanConfig.tokenize_prompt(
                        None, ["p"], tok, {"max_length": 8}
                    )
                self.assertEqual(out["input_ids"].shape[1], expected)
                self.assertEqual(int(out["attention_mask"].sum()), 3)


def _has_tpu() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        return torch.tpu.device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(_has_tpu(), "requires a TPU")
class TestHunyuanVideoOnTpu(CustomTestCase):
    def test_segment_kernel_steps_match_reference(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=24, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        attn = SimpleNamespace(attn_impl=impl, sp_attention_mode="ulysses")
        gen = torch.Generator().manual_seed(0)
        image, text, real_text = 2048, 512, 140
        q, k, v = (
            torch.randn(1, image + text, 24, 128, generator=gen) for _ in range(3)
        )
        to_tpu = lambda t: t.bfloat16().to("tpu")  # noqa: E731
        segments = [
            (n, torch.tensor([0, real], dtype=torch.int32).to("tpu"))
            for n, real in ((image, image), (text, real_text))
        ]
        out = (
            hunyuan_video.segment_attention(
                attn, *map(to_tpu, (q, k, v)), segments, local=True
            )
            .float()
            .cpu()
        )
        keys = torch.cat((k[:, :image], k[:, image : image + real_text]), 1)
        values = torch.cat((v[:, :image], v[:, image : image + real_text]), 1)
        rounded = lambda t: t.bfloat16().float().transpose(1, 2)  # noqa: E731
        expected = F.scaled_dot_product_attention(
            rounded(q), rounded(keys), rounded(values), scale=128**-0.5
        ).transpose(1, 2)
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
