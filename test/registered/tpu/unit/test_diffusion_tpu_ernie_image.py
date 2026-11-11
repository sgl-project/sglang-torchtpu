# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's ERNIE-Image patches (sglang_tpu.multimodal_gen.ernie_image)."""

import contextlib
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.compiler.config as compiler_config
from transformers import BatchEncoding, MistralConfig

from sglang.multimodal_gen.configs.models.dits.ernie_image import (
    ErnieImageArchConfig,
    ErnieImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs import ernie_image as config
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import ernie_image as dit
from sglang.multimodal_gen.runtime.models.encoders.mistral_3 import MistralModel
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import ernie_image, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=40, suite="stage-a-test-tpu-unit")


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (config.ErnieImagePipelineConfig, "tokenize_prompt"),
            (config.ErnieImagePipelineConfig, "shard_latents_for_sp"),
            (dit.ErnieImageSelfAttention, "forward"),
            (dit.ErnieImageTransformer2DModel, "forward"),
            (dit.ErnieImageTransformer2DModel, "__call__"),
            (dit.ErnieImageSharedAdaLNBlock, "forward"),
            (compiler_config, "cache_key_tag"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        ernie_image.install()
        ernie_image.install()  # idempotent
        yield


def _model():
    torch.manual_seed(0)
    arch = ErnieImageArchConfig(
        num_layers=2,
        num_attention_heads=2,
        attention_head_dim=64,
        ffn_hidden_size=128,
        text_in_dim=48,
        in_channels=16,
        out_channels=16,
        rope_axes_dim=(16, 24, 24),
    )
    # The TPU dispatch; forward_cpu needs sgl_kernel's CPU ops.
    with patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_tpu):
        model = dit.ErnieImageTransformer2DModel(
            config=ErnieImageDitConfig(arch_config=arch), hf_config={}
        )
    for param in model.parameters():
        param.data.normal_(0, 0.05)
    # Unit QK-norm gains, as trained ones are: scores, and so RoPE, matter.
    for layer in model.layers:
        layer.self_attention.norm_q.weight.data.fill_(1.0)
        layer.self_attention.norm_k.weight.data.fill_(1.0)
    return model.float().eval()


def _inputs(text_len, height=4, width=4, batch=1):
    """The DiT kwargs of one request, as the pipeline config builds them."""
    gen = torch.Generator().manual_seed(text_len)
    return dict(
        hidden_states=torch.randn(batch, 16, height, width, generator=gen),
        encoder_hidden_states=[torch.randn(batch, text_len, 48, generator=gen)],
        timestep=torch.full((batch,), 500.0),
        img_shapes=[[(1, height, width)]] * batch,
        txt_seq_lens=[text_len] * batch,
        freqs_cis=None,
        encoder_hidden_states_mask=None,
    )


def _run(model, kwargs):
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        return model(**kwargs)


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


class TestErnieImageDenoise(_TestBase):
    def test_bucketed_prompt_matches_stock(self):
        model = _model()
        for size, batch in (((4, 4), 1), ((3, 5), 1), ((4, 4), 2)):
            for text_len in (1, 37, 300, 600):
                want = _run(model, _inputs(text_len, *size, batch))
                with _installed():
                    got = _run(model, _inputs(text_len, *size, batch))
                torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)

    def test_nan_in_padded_prompt_rows_stays_there(self):
        model = _model()
        with _installed():
            forward = dit.ErnieImageTransformer2DModel.forward

            def poisoned(self, *args, encoder_hidden_states, **kwargs):
                encoder_hidden_states = encoder_hidden_states.clone()
                encoder_hidden_states[:, 37:] = float("nan")
                return forward(
                    self, *args, encoder_hidden_states=encoder_hidden_states, **kwargs
                )

            want = _run(model, _inputs(37, 3, 5))
            with patch.object(dit.ErnieImageTransformer2DModel, "forward", poisoned):
                got = _run(model, _inputs(37, 3, 5))
        self.assertTrue(torch.isfinite(got).all())
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_batched_prompts_of_different_lengths_are_rejected(self):
        model = _model()
        kwargs = _inputs(5, batch=2)
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
            for text_len in (5, 5, 37, 300):
                _run(model, _inputs(text_len))
                counts.append(len(graphs))
        self.assertEqual(counts[2:], [counts[1]] * 2)


def _causal_hidden(input_ids):
    """Per-token rows that see only earlier tokens, like the text encoder."""
    rows = input_ids[..., None].float().cumsum(1)
    return SimpleNamespace(hidden_states=(rows, rows))


class TestErnieImageHostConditioning(_TestBase):
    def test_postprocess_takes_real_rows_by_host_indices(self):
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
        hidden = torch.randn(2, 5, 3)
        got = config.ernie_image_postprocess_text(
            SimpleNamespace(hidden_states=(hidden, None)),
            SimpleNamespace(attention_mask=mask),
        )
        self.assertEqual(got.prompt_seq_lens, [3, 5])
        self.assertTrue(torch.equal(got.prompt_embeds[0, :3], hidden[0, :3]))
        self.assertTrue(torch.equal(got.prompt_embeds[1], hidden[1]))
        # A device without data-dependent shapes: only index_select reaches it.
        on_meta = config.ernie_image_postprocess_text(
            SimpleNamespace(hidden_states=(hidden.to("meta"), None)),
            SimpleNamespace(attention_mask=mask),
        )
        self.assertEqual(tuple(on_meta.prompt_embeds.shape), (2, 5, 3))
        self.assertEqual(on_meta.prompt_seq_lens, [3, 5])

    def test_prompt_tokens_pad_to_256_and_keep_real_rows(self):
        ids = torch.arange(12, 52)[None]
        tokenizer = Mock(
            side_effect=lambda prompts, **kwargs: BatchEncoding(
                {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            ),
            pad_token_id=11,
        )
        pipeline_config = config.ErnieImagePipelineConfig()
        stock = pipeline_config.tokenize_prompt(["p"], tokenizer, {})
        with _installed():
            padded = pipeline_config.tokenize_prompt(["p"], tokenizer, {})
        self.assertEqual(tuple(padded["input_ids"].shape), (1, 256))
        self.assertTrue((padded["input_ids"][0, 40:] == 11).all())
        want, got = (
            config.ernie_image_postprocess_text(_causal_hidden(t["input_ids"]), t)
            for t in (stock, padded)
        )
        self.assertTrue(torch.equal(got.prompt_embeds, want.prompt_embeds))
        self.assertEqual(got.prompt_seq_lens, [40])

    def test_text_encoder_real_rows_ignore_the_padding(self):
        torch.manual_seed(0)
        encoder = MistralModel(
            MistralConfig(
                vocab_size=64,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                pad_token_id=11,
            )
        ).eval()
        ids = torch.randint(12, 64, (1, 40))
        stock = BatchEncoding(
            {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        )
        padded = ernie_image._pad_tokens(BatchEncoding(dict(stock)), encoder.config)

        def encode(inputs):
            with set_forward_context(current_timestep=0, attn_metadata=None):
                with torch.no_grad():
                    outputs = encoder(**inputs, output_hidden_states=True)
            return config.ernie_image_postprocess_text(outputs, inputs)

        want, got = encode(stock), encode(padded)
        self.assertEqual(got.prompt_seq_lens, [40])
        torch.testing.assert_close(
            got.prompt_embeds, want.prompt_embeds, rtol=1e-5, atol=1e-5
        )


def _has_tpu() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        return torch.tpu.device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(_has_tpu(), "requires a TPU")
class TestErnieImageKernelAttention(_TestBase):
    def test_image_then_prompt_kernel_steps_match_reference(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=2, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        usp = SimpleNamespace(attn_impl=impl)
        gen = torch.Generator().manual_seed(0)
        to_tpu = lambda t: t.bfloat16().to("tpu")  # noqa: E731
        for image_len, text_len in ((1024, 37), (4096, 500)):
            rows = [image_len, 512]
            q, k, v = (
                torch.randn(1, sum(rows), 2, 128, generator=gen) for _ in range(3)
            )
            segments = [
                (n, torch.tensor([0, real], dtype=torch.int32).to("tpu"))
                for n, real in zip(rows, [image_len, text_len], strict=True)
            ]
            with patch.object(
                impl, "forward_ring_resume", wraps=impl.forward_ring_resume
            ) as resume:
                got = ernie_image._joint_attention(
                    usp, *map(to_tpu, (q, k, v)), segments
                ).cpu()
            self.assertEqual(resume.call_count, len(segments))
            real = torch.ones(sum(rows), dtype=torch.bool)
            real[image_len + text_len :] = False
            want = torch.nn.functional.scaled_dot_product_attention(
                *(
                    t.bfloat16().float().transpose(1, 2)
                    for t in (q, k[:, real], v[:, real])
                )
            ).transpose(1, 2)
            self.assertLess((got.float() - want).abs().max().item(), 2e-2)


if __name__ == "__main__":
    unittest.main()
