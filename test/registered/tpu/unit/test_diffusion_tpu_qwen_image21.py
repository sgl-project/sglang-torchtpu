# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Qwen-Image 2.1 patches (sglang_tpu.multimodal_gen.qwen_image21)."""

import contextlib
import importlib.util
import inspect
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.compiler.config as compiler_config
from transformers import BatchFeature

from sglang.multimodal_gen.configs.models.dits.qwenimage21 import (
    QwenImage21ArchConfig,
    QwenImage21DitConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import qwen_image21 as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    qwen_image21 as stage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import qwen_image21, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit")

# The stock builder, even if a worker init already installed the patches.
_STOCK_BUILD_LAYOUT = inspect.unwrap(stage.build_layout)
_AXES = (16, 56, 56)
_TARGET = (1, 4, 4)


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (stage, "build_layout"),
            (stage.QwenImage21EncodingStage, "__init__"),
            (dit.QwenImage21Attention, "forward"),
            (dit, "apply_rope"),
            (dit.QwenImage21TransformerBlock, "forward"),
            (dit.QwenImage21TransformerBlock, "__call__"),
            (dit.QwenImage21Transformer2DModel, "forward"),
            (dit.QwenImage21Transformer2DModel, "__call__"),
            (compiler_config, "cache_key_tag"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        qwen_image21.install()
        qwen_image21.install()  # idempotent
        yield


def _model():
    torch.manual_seed(0)
    arch = QwenImage21ArchConfig(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=128,
        context_in_dim=64,
        in_channels=16,
        axes_dims_rope=_AXES,
    )
    model = dit.QwenImage21Transformer2DModel(
        config=QwenImage21DitConfig(arch_config=arch), hf_config={}
    )
    for param in model.parameters():
        param.data.normal_(0, 0.05)
    return model.float().eval()


def _slots(text_len, with_image):
    slots = [False] * text_len
    if with_image:
        slots[2] = True  # one condition image after two text tokens
    return slots


def _denoise(model, layout, text_len, with_image):
    """Prefill and one cached step; returns both outputs and the prefix K/V."""
    gen = torch.Generator().manual_seed(text_len)
    x = torch.randn(1, 16, 16, generator=gen)
    enc = torch.randn(1, text_len, 64, generator=gen)
    cond = torch.randn(1, 4, 16, generator=gen) if with_image else None
    caches = [[{} for _ in range(model.num_layers)]]
    outputs = []
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        for t in (0.9, 0.5):
            outputs.append(
                model(
                    hidden_states=x,
                    encoder_hidden_states=enc,
                    timestep=torch.tensor([t * 1000]),
                    layouts=[layout],
                    condition_latents=cond,
                    prefix_caches=caches,
                )
            )
    return outputs, caches[0]


class _TestBase(CustomTestCase):
    def setUp(self):
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


class TestQwenImage21Layout(_TestBase):
    def test_prefix_pads_to_512_with_real_length_on_device(self):
        shapes = [(1, 2, 2), _TARGET]
        stock = _STOCK_BUILD_LAYOUT(_slots(7, True), shapes, _AXES, "cpu")
        with _installed():
            padded = stage.build_layout(_slots(7, True), shapes, _AXES, "cpu")
        prefix_len = stock["prefix_rope"].shape[0]
        self.assertEqual(prefix_len, 10)  # 6 text rows + a 2x2 image
        self.assertEqual(padded["prefix_rope"].shape, (512, 2, 128))
        self.assertEqual(padded["prefix_keys"].tolist(), [0, prefix_len])
        self.assertEqual(padded["target_keys"].tolist(), [0, 16])
        self.assertFalse(padded["text_only"])
        self.assertEqual(padded["segments"], stock["segments"])
        self.assertTrue(
            torch.equal(padded["text_indices"][:prefix_len], stock["text_indices"])
        )
        self.assertTrue((padded["text_indices"][prefix_len:] == 0).all())
        cos, sin = padded["prefix_rope"][:, 0, ::2], padded["prefix_rope"][:, 1, ::2]
        self.assertTrue(torch.equal(cos[:prefix_len], stock["prefix_rope"].real))
        self.assertTrue(torch.equal(sin[:prefix_len], stock["prefix_rope"].imag))
        self.assertTrue((cos[prefix_len:] == 1).all() and (sin[prefix_len:] == 0).all())

    def test_pair_swap_rope_is_the_complex_rotation(self):
        layout = _STOCK_BUILD_LAYOUT(_slots(3, False), [_TARGET], _AXES, "cpu")
        rope = layout["target_rope"]
        x = torch.randn(1, rope.shape[0], 2, 128).bfloat16()
        torch.testing.assert_close(
            qwen_image21.apply_rope_pair_swap(x, qwen_image21._real_rope(rope)),
            dit.apply_rope(x, rope),
            rtol=0,
            atol=0,
        )


class TestQwenImage21Denoise(_TestBase):
    def test_padded_prefix_matches_stock(self):
        model = _model()
        for text_len, with_image in ((5, False), (37, False), (9, True)):
            shapes = ([(1, 2, 2)] if with_image else []) + [_TARGET]
            slots = _slots(text_len, with_image)
            stock = _STOCK_BUILD_LAYOUT(slots, shapes, _AXES, "cpu")
            want, want_cache = _denoise(model, stock, text_len, with_image)
            with _installed():
                layout = stage.build_layout(slots, shapes, _AXES, "cpu")
                got, got_cache = _denoise(model, layout, text_len, with_image)
            for step in range(2):
                torch.testing.assert_close(got[step], want[step], rtol=1e-4, atol=1e-5)
            prefix_len = stock["prefix_rope"].shape[0]
            for layer in range(model.num_layers):
                torch.testing.assert_close(
                    got_cache[layer]["key"][:, :prefix_len],
                    want_cache[layer]["key"],
                    rtol=1e-4,
                    atol=1e-5,
                )

    def test_blocks_compile_once_across_prompt_lengths(self):
        model = _model()
        graphs = []

        def counting_backend(graph_module, example_inputs):
            graphs.append(graph_module)
            return graph_module.forward

        counts = []
        with _installed():
            # In place, as regional compile does (``submodule.compile``).
            for block in model.transformer_blocks:
                block.compile(backend=counting_backend, fullgraph=True, dynamic=False)
            # The first request also settles a lazily resolved kernel-dispatch
            # global (sglang.kernels.fused_op), which recompiles one graph once.
            for text_len in (5, 5, 37, 120):
                layout = stage.build_layout(
                    _slots(text_len, False), [_TARGET], _AXES, "cpu"
                )
                _denoise(model, layout, text_len, False)
                counts.append(len(graphs))
        # One prefill and one cached graph serve both blocks (not one per
        # layer), and new prompt lengths within the bucket reuse them.
        self.assertLessEqual(counts[1], 3)
        self.assertEqual(counts[2:], [counts[1]] * 2)


class TestQwenImage21WholeModelCompile(_TestBase):
    def test_whole_dit_compiles_once_across_prompt_lengths(self):
        model = _model()
        graphs = []

        def counting_backend(graph_module, example_inputs):
            graphs.append(graph_module)
            return graph_module.forward

        counts = []
        with _installed():
            # The TPU default (no --regional-compile) compiles the whole DiT.
            model.compile(backend=counting_backend, dynamic=False)
            for text_len in (5, 5, 37, 120):
                layout = stage.build_layout(
                    _slots(text_len, False), [_TARGET], _AXES, "cpu"
                )
                _denoise(model, layout, text_len, False)
                counts.append(len(graphs))
        self.assertEqual(counts[2:], [counts[1]] * 2)


class _CausalEncoder:
    """Per-token hidden rows that see only earlier tokens, like the text encoder."""

    def __init__(self):
        self.model = SimpleNamespace(
            visual=SimpleNamespace(rotary_pos_emb=SimpleNamespace()),
            get_rope_index=lambda ids, *args, **kwargs: (
                torch.arange(ids.shape[1]).view(1, 1, -1).expand(3, 1, -1),
                None,
            ),
        )
        self._encoder_tp_group = None

    def __call__(self, input_ids, **kwargs):
        hidden = input_ids[..., None].float().cumsum(1).expand(-1, -1, 4)
        return SimpleNamespace(hidden_states=(hidden,))


class TestQwenImage21PromptPadding(_TestBase):
    def _stage(self, ids):
        processor = Mock(
            side_effect=lambda **kw: BatchFeature(
                data={"input_ids": ids.clone(), "attention_mask": torch.ones_like(ids)}
            )
        )
        processor.tokenizer.convert_tokens_to_ids.return_value = 99
        processor.tokenizer.pad_token_id = 0
        processor.apply_chat_template.return_value = [[1]]
        encoder = _CausalEncoder()
        built = stage.QwenImage21EncodingStage(encoder, processor, None, None)
        built.use_declared_component = Mock(return_value=nullcontext(encoder))
        return built

    def test_prompt_tokens_pad_to_256_and_keep_real_rows(self):
        ids = torch.tensor([[5, 6, 99, 99, 7, 8, 9]])
        want, want_slots = self._stage(ids)._encode_prompt("p", [], "cpu")
        with _installed():
            padded = self._stage(ids)
            self.assertEqual(padded.processor(text=["p"])["input_ids"].shape, (1, 256))
            got, got_slots = padded._encode_prompt("p", [], "cpu")
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(torch.equal(got_slots, want_slots))


def _has_tpu() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        return torch.tpu.device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(_has_tpu(), "requires a TPU")
class TestQwenImage21KernelAttention(_TestBase):
    def test_prefix_then_target_kernel_matches_reference(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        usp = SimpleNamespace(skip_sequence_parallel=True, attn_impl=impl)
        gen = torch.Generator().manual_seed(0)
        for prefix_len, target_len in ((37, 2048), (500, 4096)):
            q, k, v = (
                torch.randn(1, target_len, 2, 128, generator=gen) for _ in range(3)
            )
            kp, vp = (torch.randn(1, 512, 2, 128, generator=gen) for _ in range(2))
            layout = {
                "prefix_keys": torch.tensor([0, prefix_len], dtype=torch.int32),
                "target_keys": torch.tensor([0, target_len], dtype=torch.int32),
            }
            to_tpu = lambda t: t.bfloat16().to("tpu")  # noqa: E731
            q_heads = to_tpu(q).transpose(1, 2)
            self.assertTrue(
                qwen_image21._uses_kernel(
                    impl, q_heads, to_tpu(kp).transpose(1, 2), to_tpu(k).transpose(1, 2)
                )
            )
            got = qwen_image21._target_attention(
                usp,
                *map(to_tpu, (q, k, v, kp, vp)),
                {n: t.to("tpu") for n, t in layout.items()},
            ).cpu()
            keys = torch.cat((kp[:, :prefix_len], k), 1)
            values = torch.cat((vp[:, :prefix_len], v), 1)
            want = torch.nn.functional.scaled_dot_product_attention(
                *(t.bfloat16().float().transpose(1, 2) for t in (q, keys, values))
            ).transpose(1, 2)
            self.assertLess((got.float() - want).abs().max().item(), 2e-2)


if __name__ == "__main__":
    unittest.main()
