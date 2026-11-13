# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Qwen-Image patches (sglang_tpu.multimodal_gen.qwen_image)."""

import contextlib
import importlib.util
import inspect
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.compiler.config as compiler_config
from transformers import BatchEncoding

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    fuse_scale_shift_kernel_native,
)
from sglang.multimodal_gen.configs.models.dits.qwenimage import (
    QwenImageArchConfig,
    QwenImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs import qwen_image as config
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers import elementwise, layernorm
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.layers.rotary_embedding.utils import (
    _apply_rotary_emb_complex,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import qwen_image as dit
from sglang.multimodal_gen.runtime.models.encoders import qwen2_5vl
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import qwen_image, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=120, suite="stage-a-test-tpu-unit")

_T2I = [(1, 4, 4)]
_EDIT = [(1, 4, 4), (1, 2, 3)]  # the noisy image, then one condition image


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (config.QwenImagePipelineConfig, "tokenize_prompt"),
            (config.QwenImagePipelineConfig, "get_freqs_cis"),
            (ImageEncodingStage, "__init__"),
            (dit.QwenImageCrossAttention, "forward"),
            (dit.QwenImageTransformer2DModel, "__call__"),
            (dit.QwenImageTransformerBlock, "forward"),
            (compiler_config, "cache_key_tag"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        qwen_image.install()
        qwen_image.install()  # idempotent
        yield


def _model():
    torch.manual_seed(0)
    arch = QwenImageArchConfig(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=128,
        joint_attention_dim=64,
        in_channels=16,
    )
    # The TPU dispatch; forward_cpu needs sgl_kernel's CPU ops.
    with patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_tpu):
        model = dit.QwenImageTransformer2DModel(
            config=QwenImageDitConfig(arch_config=arch), hf_config={}
        )
    for param in model.parameters():
        param.data.normal_(0, 0.05)
    return model.float().eval()


def _inputs(model, text_len, shapes):
    """The DiT kwargs of one request, RoPE from the (possibly patched) config."""
    gen = torch.Generator().manual_seed(text_len)
    image_len = sum(math.prod(shape) for shape in shapes)
    freqs_cis, freqs_complex = config.QwenImagePipelineConfig.get_freqs_cis(
        [shapes], [text_len], model.rotary_emb, torch.device("cpu"), torch.float32
    )
    return dict(
        hidden_states=torch.randn(1, image_len, 16, generator=gen),
        encoder_hidden_states=torch.randn(1, text_len, 64, generator=gen),
        encoder_hidden_states_mask=None,
        timestep=torch.tensor([500.0]),
        img_shapes=[shapes],
        txt_seq_lens=[text_len],
        freqs_cis=freqs_cis,
        freqs_complex=freqs_complex,
    )


def _run(model, kwargs):
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        return model(**kwargs)


class _TestBase(CustomTestCase):
    def setUp(self):
        # The TPU kernel selection; the CPU one needs sgl_kernel's CPU ops.
        for module in (layernorm, elementwise):
            self.enterContext(
                patch.object(
                    module, "fuse_scale_shift_kernel", fuse_scale_shift_kernel_native
                )
            )
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


class TestQwenImageRope(_TestBase):
    def test_host_tables_rotate_as_the_stock_complex_rope(self):
        rope = dit.QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
        args = ([_EDIT], [37], rope, torch.device("cpu"), torch.float32)
        _, (img, txt) = config.QwenImagePipelineConfig.get_freqs_cis(*args)
        with _installed():
            (img_table, txt_table), slot2 = (
                config.QwenImagePipelineConfig.get_freqs_cis(*args)
            )
        self.assertIs(slot2[0], img_table)
        self.assertEqual(tuple(txt_table.shape), (37, 256))
        for table, freqs in ((img_table, img), (txt_table, txt)):
            for dtype in (torch.bfloat16, torch.float32):
                x = torch.randn(1, table.shape[0], 2, 128).to(dtype)
                torch.testing.assert_close(
                    qwen_image.apply_rope_pair_swap(x, table),
                    _apply_rotary_emb_complex(x, freqs.unsqueeze(1)),
                    rtol=0,
                    atol=0,
                )


class TestQwenImageDenoise(_TestBase):
    def test_bucketed_prompt_matches_stock(self):
        model = _model()
        for shapes in (_T2I, _EDIT):
            for text_len in (5, 37, 300):
                want = _run(model, _inputs(model, text_len, shapes))
                with _installed():
                    got = _run(model, _inputs(model, text_len, shapes))
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)

    def test_nan_in_padded_prompt_rows_stays_there(self):
        model = _model()
        forward = dit.QwenImageTransformer2DModel.forward

        def poisoned(self, *args, encoder_hidden_states, **kwargs):
            encoder_hidden_states = encoder_hidden_states.clone()
            encoder_hidden_states[:, 37:] = float("nan")
            return forward(
                self, *args, encoder_hidden_states=encoder_hidden_states, **kwargs
            )

        with _installed():
            want = _run(model, _inputs(model, 37, _EDIT))
            with patch.object(dit.QwenImageTransformer2DModel, "forward", poisoned):
                got = _run(model, _inputs(model, 37, _EDIT))
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_batched_prompts_of_different_lengths_are_rejected(self):
        model = _model()
        kwargs = _inputs(model, 5, _T2I)
        kwargs["txt_seq_lens"] = [5, 3]
        with _installed(), self.assertRaises(NotImplementedError):
            _run(model, kwargs)

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
            # The first request also settles a lazily resolved kernel-dispatch
            # global (sglang.kernels.fused_op), which recompiles one graph once.
            for text_len in (5, 5, 37, 300):
                _run(model, _inputs(model, text_len, _T2I))
                counts.append(len(graphs))
        self.assertEqual(counts[2:], [counts[1]] * 2)


def _causal_hidden(input_ids):
    """Per-token rows that see only earlier tokens, like the text encoder."""
    return SimpleNamespace(hidden_states=(input_ids[..., None].float().cumsum(1),))


class TestQwenImagePromptPadding(_TestBase):
    def test_prompt_tokens_pad_to_256_and_keep_real_rows(self):
        ids = torch.arange(1, 41)[None]
        tokenizer = Mock(
            side_effect=lambda prompts, **kwargs: BatchEncoding(
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            ),
            pad_token_id=0,
        )
        pipeline_config = config.QwenImagePipelineConfig()
        stock = pipeline_config.tokenize_prompt(["p"], tokenizer, {})
        with _installed():
            padded = pipeline_config.tokenize_prompt(["p"], tokenizer, {})
        self.assertEqual(tuple(padded["input_ids"].shape), (1, 256))
        want, got = (
            config.qwen_image_postprocess_text(_causal_hidden(t["input_ids"]), t)
            for t in (stock, padded)
        )
        self.assertTrue(torch.equal(got, want))

    def test_edit_processor_pads_and_keeps_its_named_parameters(self):
        ids = torch.arange(1, 81)[None]

        def process(images=None, text=None, videos=None, **kwargs):
            return BatchEncoding(
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            )

        processor = Mock(side_effect=process)
        processor.tokenizer.pad_token_id = 0
        encoder = SimpleNamespace()
        with (
            _installed(),
            patch.object(
                qwen2_5vl, "Qwen2_5_VLForConditionalGeneration", SimpleNamespace
            ),
        ):
            stage = ImageEncodingStage(processor, text_encoder=encoder)
        self.assertEqual(
            list(inspect.signature(stage.image_processor.__call__).parameters),
            ["images", "text", "videos", "kwargs"],
        )
        padded = stage.image_processor(images=[None], text=["p"])
        self.assertEqual(tuple(padded["input_ids"].shape), (1, 256))
        self.assertEqual(padded["attention_mask"].sum().item(), 80)
        want, got = (
            config.qwen_image_edit_postprocess_text(_causal_hidden(t["input_ids"]), t)
            for t in (process(), padded)
        )
        self.assertTrue(torch.equal(got, want))


def _has_tpu() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        return torch.tpu.device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(_has_tpu(), "requires a TPU")
class TestQwenImageKernelAttention(_TestBase):
    def test_prompt_then_image_kernel_steps_match_reference(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        usp = SimpleNamespace(skip_sequence_parallel=True, attn_impl=impl)
        gen = torch.Generator().manual_seed(0)
        to_tpu = lambda t: t.bfloat16().to("tpu")  # noqa: E731
        for text_len, image_lens in ((37, [1024]), (500, [4096, 1024])):
            rows = [512, *image_lens]
            q, k, v = (
                torch.randn(1, sum(rows), 2, 128, generator=gen) for _ in range(3)
            )
            segments = [
                (n, torch.tensor([0, real], dtype=torch.int32).to("tpu"))
                for n, real in zip(rows, [text_len, *image_lens], strict=True)
            ]
            with patch.object(
                impl, "forward_ring_resume", wraps=impl.forward_ring_resume
            ) as resume:
                got = qwen_image._joint_attention(
                    usp, *map(to_tpu, (q, k, v)), segments, False
                ).cpu()
            self.assertEqual(resume.call_count, len(segments))
            real = torch.ones(sum(rows), dtype=torch.bool)
            real[text_len:512] = False
            want = torch.nn.functional.scaled_dot_product_attention(
                *(
                    t.bfloat16().float().transpose(1, 2)
                    for t in (q, k[:, real], v[:, real])
                )
            ).transpose(1, 2)
            self.assertLess((got.float() - want).abs().max().item(), 2e-2)


if __name__ == "__main__":
    unittest.main()
