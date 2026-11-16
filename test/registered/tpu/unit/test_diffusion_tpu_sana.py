# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's SANA patches (sglang_tpu.multimodal_gen.sana)."""

import unittest
from unittest.mock import patch

import torch
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.configs.pipeline_configs import sana as config
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import sana

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

# The text-encoding stage's SANA tokenizer kwargs; max_length is Gemma2's text_len.
TOK_KWARGS = (
    Gemma2Config().tokenizer_kwargs
    | config.SanaPipelineConfig().text_encoder_extra_args[0]
)
MAX_LENGTH = TOK_KWARGS["max_length"]


def _stock_tokenize(lengths):
    """Stock ``tokenize_prompt``: rows padded to the longest prompt."""
    longest = max(lengths)

    def tokenize(self, prompts, tokenizer, tok_kwargs):
        assert tok_kwargs == TOK_KWARGS
        return BatchEncoding(
            {
                "input_ids": torch.tensor(
                    [[5] * n + [0] * (longest - n) for n in lengths]
                ),
                "attention_mask": torch.tensor(
                    [[1] * n + [0] * (longest - n) for n in lengths]
                ),
            }
        )

    return tokenize


class TestSanaTokens(CustomTestCase):
    def test_tokens_pad_to_the_tokenizer_max_length(self):
        self.assertEqual(MAX_LENGTH, 300)
        tokenizer = type("Tok", (), {"pad_token_id": 0, "eos_token_id": 1})()
        owner = config.SanaPipelineConfig
        for lengths in ([3, 2], [280], [MAX_LENGTH]):
            with (
                patch.object(sana, "_STOCK_TOKENIZE", _stock_tokenize(lengths)),
                patch.object(owner, "tokenize_prompt", owner.tokenize_prompt),
            ):
                sana.install()
                out = owner.tokenize_prompt(
                    None, ["p"] * len(lengths), tokenizer, dict(TOK_KWARGS)
                )
            self.assertEqual(tuple(out["input_ids"].shape), (len(lengths), MAX_LENGTH))
            self.assertEqual(out["attention_mask"].sum(1).tolist(), lengths)
            self.assertFalse(out["input_ids"][:, max(lengths) :].any())


if __name__ == "__main__":
    unittest.main()
