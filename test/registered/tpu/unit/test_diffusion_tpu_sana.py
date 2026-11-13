# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's SANA patches (sglang_tpu.multimodal_gen.sana)."""

import contextlib
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import BatchEncoding

from sglang.multimodal_gen.configs.models.dits.sana import SanaArchConfig, SanaConfig
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.configs.pipeline_configs import sana as config
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import sana as dit
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import sana, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=20, suite="nightly-tpu-4-v7", nightly=True)

# The text-encoding stage's SANA tokenizer kwargs; max_length is Gemma2's text_len.
TOK_KWARGS = (
    Gemma2Config().tokenizer_kwargs
    | config.SanaPipelineConfig().text_encoder_extra_args[0]
)
MAX_LENGTH = TOK_KWARGS["max_length"]


@contextlib.contextmanager
def _installed():
    """The TPU tokenizer patch, restored afterwards."""
    owner = config.SanaPipelineConfig
    with patch.object(owner, "tokenize_prompt", owner.tokenize_prompt):
        sana.install()
        sana.install()  # idempotent
        yield


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
        for lengths in ([3, 2], [280], [MAX_LENGTH]):
            stock = _stock_tokenize(lengths)
            with patch.object(config.SanaPipelineConfig, "tokenize_prompt", stock):
                sana.install()
                out = config.SanaPipelineConfig.tokenize_prompt(
                    None, ["p"] * len(lengths), tokenizer, dict(TOK_KWARGS)
                )
            self.assertEqual(tuple(out["input_ids"].shape), (len(lengths), MAX_LENGTH))
            self.assertEqual(out["attention_mask"].sum(1).tolist(), lengths)
            self.assertFalse(out["input_ids"][:, max(lengths) :].any())


def _model():
    torch.manual_seed(0)
    arch = SanaArchConfig(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=4,
        num_cross_attention_heads=2,
        cross_attention_head_dim=16,
        cross_attention_dim=32,
        caption_channels=24,
        sample_size=8,
    )
    model = dit.SanaTransformer2DModel(
        config=SanaConfig(arch_config=arch), hf_config={}
    )
    for param in model.parameters():
        param.data.normal_(0, 0.1)
    return model.float().eval()


def _inputs(text_len, padded_len=None):
    """DiT kwargs; ``padded_len`` right-pads the prompt as the TPU tokenizer does."""
    gen = torch.Generator().manual_seed(text_len)
    text = torch.randn(1, text_len, 24, generator=gen)
    mask = torch.ones(1, text_len, dtype=torch.long)
    if padded_len is not None:
        # Padded rows hold the encoder's (finite) outputs for pad tokens.
        text = F.pad(text, (0, 0, 0, padded_len - text_len), value=0.5)
        mask = F.pad(mask, (0, padded_len - text_len))
    return dict(
        hidden_states=torch.randn(1, 4, 8, 8, generator=gen),
        encoder_hidden_states=text,
        timestep=torch.full((1,), 500.0),
        encoder_attention_mask=mask,
    )


def _run(model, kwargs):
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        return model(**kwargs)


class TestSanaDiT(CustomTestCase):
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

    def test_padded_prompt_matches_stock(self):
        model = _model()
        for text_len in (5, 40, 280):
            expected = _run(model, _inputs(text_len))
            actual = _run(model, _inputs(text_len, padded_len=MAX_LENGTH))
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_dit_compiles_once_across_padded_prompt_lengths(self):
        model = _model()
        graphs = []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        model.compile(backend=counting, dynamic=False)
        for text_len in (5, 40, 280):
            _run(model, _inputs(text_len, padded_len=MAX_LENGTH))
        self.assertEqual(len(graphs), 1)


if __name__ == "__main__":
    import unittest

    unittest.main()
