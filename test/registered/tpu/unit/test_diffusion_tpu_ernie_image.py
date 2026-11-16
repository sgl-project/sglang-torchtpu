# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's ERNIE-Image patches (sglang_tpu.multimodal_gen.ernie_image)."""

import contextlib
import functools
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from transformers import BatchEncoding, MistralConfig

from sglang.multimodal_gen.configs.models.dits.ernie_image import (
    ErnieImageArchConfig,
    ErnieImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs import ernie_image as config
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed import parallel_state
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import ernie_image as dit
from sglang.multimodal_gen.runtime.models.encoders.mistral_3 import MistralModel
from sglang.multimodal_gen.runtime.pipelines import ernie_image as pipeline
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.utils import torch_compile
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import ernie_image

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)


@contextlib.contextmanager
def _sp(rank=0, world_size=1):
    """Tensor parallel 1 and one Ulysses rank, without process groups."""
    tp = SimpleNamespace(world_size=1, rank_in_group=0, all_gather=lambda x, dim: x)
    sp = SimpleNamespace(
        world_size=world_size,
        rank_in_group=rank,
        ulysses_world_size=world_size,
        ring_world_size=1,
    )
    with (
        patch.object(parallel_state, "_TP", tp),
        patch.object(parallel_state, "_SP", sp),
    ):
        yield


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (pipeline.ErnieImagePipeline, "load_modules"),
            (config.ErnieImagePipelineConfig, "tokenize_prompt"),
            (config.ErnieImagePipelineConfig, "shard_latents_for_sp"),
            (config.ErnieImagePipelineConfig, "supports_dynamic_batching_for_request"),
            (dit.ErnieImageSelfAttention, "forward"),
            (dit.ErnieImageTransformer2DModel, "forward"),
            (dit.ErnieImageTransformer2DModel, "__call__"),
            (dit.ErnieImageTransformer2DModel, "_compile_conditions"),
            (dit.ErnieImageTransformer2DModel, "_compile_regions_spmd_safe"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        ernie_image.install()
        ernie_image.install()  # idempotent
        yield


def _model(heads=2):
    torch.manual_seed(0)
    arch = ErnieImageArchConfig(
        num_layers=2,
        num_attention_heads=heads,
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
        self.addCleanup(set_global_server_args, server_args_module._global_server_args)
        set_global_server_args(_make_unit_server_args())
        self.enterContext(_sp())


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
        with _installed():
            # The scheduler never batches requests, whose prompt lengths differ.
            pipeline_config = config.ErnieImagePipelineConfig()
            self.assertFalse(
                pipeline_config.supports_dynamic_batching_for_request(None)
            )
            with self.assertRaises(NotImplementedError):
                _run(model, kwargs)

    def test_cfg_parallel_is_refused_at_load(self):
        args = SimpleNamespace(enable_cfg_parallel=True)
        with _installed(), self.assertRaisesRegex(ValueError, "--ulysses-degree"):
            pipeline.ErnieImagePipeline.load_modules(None, args)

    def test_dit_compiles_once_across_prompt_lengths(self):
        wrapped = []

        def wrap_region(forward):
            wrapped.append(forward)
            return forward

        for regional in (False, True):
            model, graphs, counts = _model(), [], []

            def counting_backend(graph_module, example_inputs):
                graphs.append(graph_module)
                return graph_module.forward

            kwargs = dict(backend=counting_backend, dynamic=False)
            with (
                self.subTest(regional=regional),
                _installed(),
                patch.object(
                    torch_compile.current_platform,
                    "prepare_spmd_region_wrapper",
                    return_value=wrap_region,
                ),
            ):
                if regional:  # --regional-compile: each block, spmd-safe
                    torch_compile.compile_matching_submodules(
                        model, compile_kwargs=kwargs
                    )
                else:  # the TPU default compiles the whole DiT
                    model.compile(**kwargs)
                for text_len in (5, 5, 37, 300):
                    _run(model, _inputs(text_len))
                    counts.append(len(graphs))
                self.assertEqual(counts[2:], [counts[1]] * 2)
        self.assertEqual(len(wrapped), len(model.layers))

    def test_one_program_serves_every_rank(self):
        # Ulysses 4 without process groups: stand-ins keep the collectives'
        # shapes, so each rank runs the compiled DiT on its own rows.
        sp, model, graphs = 4, _model(heads=4), []

        def counting_backend(graph_module, example_inputs):
            graphs.append(graph_module)
            return graph_module.forward

        with (
            _installed(),
            patch.multiple(
                ernie_image,
                _usp_input_all_to_all=lambda t, head_dim: t.repeat(1, sp, 1, 1)[
                    :, :, : t.shape[2] // sp
                ],
                _usp_output_all_to_all=lambda t, head_dim: t[
                    :, : t.shape[1] // sp
                ].repeat(1, 1, sp, 1),
                gather_seq=lambda t, n: t.repeat(1, sp, 1)[:, :n],
            ),
        ):
            model.compile(backend=counting_backend, dynamic=False)
            for rank in range(sp):
                with _sp(rank, sp):
                    _run(model, _inputs(37, 3, 5))
        self.assertEqual(len(graphs), 1)

    def test_rank_rows_tile_the_whole_layout(self):
        pos_embed, shape = _model(heads=4).pos_embed, (3, 5, 37, 512, 1)
        layout = functools.partial(
            ernie_image._layout, pos_embed, dtype=torch.float32, device="cpu"
        )
        whole = layout((*shape, 1, 0))[0]
        shares = [layout((*shape, 4, rank))[0] for rank in range(4)]
        # 15 image rows pad to 4 per rank; Ulysses gathers the image shares,
        # then the prompt shares, in rank order.
        for i, identity in enumerate((1.0, 0.0)):
            image, text = whole[i].split((15, 512))
            image = F.pad(image, (0, 0, 0, 1), value=identity)
            got = [share[i][:4] for share in shares] + [
                share[i][4:] for share in shares
            ]
            torch.testing.assert_close(
                torch.cat(got), torch.cat((image, text)), rtol=0, atol=0
            )


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

    def test_prompt_tokens_pad_to_256_and_real_rows_ignore_the_padding(self):
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
        tokenizer = Mock(return_value=BatchEncoding(dict(stock)), pad_token_id=11)
        with _installed():
            pipeline_config = config.ErnieImagePipelineConfig()
            padded = pipeline_config.tokenize_prompt(["p"], tokenizer, {})
        self.assertEqual(tuple(padded["input_ids"].shape), (1, 256))

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
