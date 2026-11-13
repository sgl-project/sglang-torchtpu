# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's GLM-Image prior-token generation (sglang_tpu.multimodal_gen.glm_image_ar)."""

import contextlib
from unittest.mock import patch

import torch
from transformers import GlmImageConfig, GlmImageForConditionalGeneration
from transformers.generation import logits_process as lp

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import glm_image_ar

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=30, suite="nightly-tpu-4-v7", nightly=True)

IMAGE_START, IMAGE_END, IMAGE_TOKEN = 290, 291, 292


@contextlib.contextmanager
def _installed(backend="eager"):
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(
                GlmImageForConditionalGeneration,
                "generate",
                GlmImageForConditionalGeneration.generate,
            )
        )
        stack.enter_context(patch.object(glm_image_ar, "_DEVICE_TYPES", ("cpu",)))
        stack.enter_context(
            patch.object(
                glm_image_ar,
                "build_torch_compile_kwargs",
                lambda mode=None: {"backend": backend},
            )
        )
        glm_image_ar.install()
        glm_image_ar.install()  # idempotent
        yield


def _model():
    torch.manual_seed(0)
    config = GlmImageConfig(
        text_config=dict(
            vocab_size=300,
            vision_vocab_size=40,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=4096,
            pad_token_id=0,
            eos_token_id=IMAGE_END,
            rope_parameters=dict(
                rope_type="default",
                rope_theta=10000.0,
                partial_rotary_factor=0.5,
                mrope_section=[2, 1, 1],
            ),
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2),
        vq_config=dict(embed_dim=8, num_embeddings=40, latent_channels=8),
        image_start_token_id=IMAGE_START,
        image_end_token_id=IMAGE_END,
        image_token_id=IMAGE_TOKEN,
    )
    config.text_config._attn_implementation = "sdpa"
    model = GlmImageForConditionalGeneration(config).eval()
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    # Sharp attention, so the RoPE positions decide the picks.
    for layer in model.model.language_model.layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.k_proj):
            proj.weight.data.mul_(5)
    return model


# A preview grid, then the target grid, as the GLM-Image processor emits them.
GRIDS = torch.tensor([[1, 2, 2], [1, 3, 4]])


def _prompt(n, seed):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(1, 280, (1, n), generator=gen)
    ids[0, -1] = IMAGE_START
    return ids


def _generate(model, ids, **kwargs):
    with torch.no_grad():
        return model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            image_grid_thw=GRIDS,
            max_new_tokens=int(GRIDS.prod(-1).sum()) + 1,
            do_sample=True,
            **kwargs,
        )


class TestGlmImagePriorTokens(CustomTestCase):
    def test_static_cache_decode_matches_transformers(self):
        # top_k 1 makes the draw deterministic on both paths.
        model = _model()
        graphs = []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        for n, seed in ((9, 0), (300, 1)):
            ids = _prompt(n, seed)
            expected = _generate(model, ids, temperature=0.9, top_k=1, top_p=0.75)
            with _installed(counting):
                actual = _generate(model, ids, temperature=0.9, top_k=1, top_p=0.75)
            self.assertTrue(torch.equal(actual, expected), (n, actual, expected))
        # A prompt chunk, one token, and the draw, across prompt lengths.
        self.assertEqual(len(graphs), 3)

    def test_candidate_probabilities_match_transformers_processors(self):
        processors = lp.LogitsProcessorList(
            [
                lp.TemperatureLogitsWarper(0.9),
                lp.TopKLogitsWarper(50),
                lp.TopPLogitsWarper(0.75),
            ]
        )
        self.assertEqual(glm_image_ar.device_sampling(processors), (0.9, 50, 0.75))
        gen = torch.Generator().manual_seed(0)
        for _ in range(5):
            logits = torch.randn(1, 300, generator=gen) * 3
            want = processors(torch.zeros(1, 1, dtype=torch.long), logits).softmax(-1)
            probs, ids = glm_image_ar.candidate_probs(logits, 0.9, 50, 0.75)
            got = torch.zeros_like(want).scatter(-1, ids, probs)
            torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)

    def test_other_processors_use_the_full_row(self):
        processors = lp.LogitsProcessorList(
            [lp.TemperatureLogitsWarper(0.9), lp.MinPLogitsWarper(0.1)]
        )
        self.assertIsNone(glm_image_ar.device_sampling(processors))

    def test_draws_follow_the_candidate_distribution(self):
        logits = torch.tensor([[2.0, 1.5, 1.0, 0.2, -1.0, -3.0]])
        probs, ids = glm_image_ar.candidate_probs(logits, 1.0, 4, 0.9)
        decoder = glm_image_ar._Decoder.__new__(glm_image_ar._Decoder)
        torch.manual_seed(0)
        counts = torch.zeros(6)
        for _ in range(4000):
            noise = torch.empty(1, 4).exponential_()
            counts[int(decoder._draw(logits, noise, (1.0, 4, 0.9)))] += 1
        want = torch.zeros(6).scatter(0, ids[0], probs[0])
        torch.testing.assert_close(counts / 4000, want, atol=0.03, rtol=0)


if __name__ == "__main__":
    import unittest

    unittest.main()
