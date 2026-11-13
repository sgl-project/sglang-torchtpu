# SPDX-License-Identifier: Apache-2.0
"""SANA on TPU workers: one compiled DiT per image size.

Runtime patch, tracked debt:
- Why: torch_tpu compiles static shapes. The Gemma2 prompt tokens pad to the
  batch's longest prompt, so the DiT's cross-attention key count changes with
  every prompt length and the whole DiT (the TPU default compiles the whole
  module) compiles again.
- How: :func:`install` right-pads the prompt tokens to the tokenizer's
  ``max_length`` (Gemma2's ``text_len``, 300), as diffusers' SANA pipeline does
  with ``padding="max_length"``. The text encoder is causal, so real tokens
  never see the padding, and the DiT's cross-attention mask drops the padded
  keys: the outputs are the stock ones.
- Upstream PR or issue: none.
- Remove when: SANA's tokenizer pads to ``max_length``.
"""

from __future__ import annotations

import functools

import torch.nn.functional as F

from sglang.multimodal_gen.configs.pipeline_configs import sana as config


def _pad_tokens(inputs, tokenizer, max_length):
    """``inputs`` with its token rows right-padded to ``max_length``."""
    extra = (max_length or 0) - inputs["input_ids"].shape[1]
    if extra > 0:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, extra), value=pad_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, extra))
    return inputs


def _padded_tokenize(tokenize):
    @functools.wraps(tokenize)
    def wrapped(self, prompts, tokenizer, tok_kwargs):
        inputs = tokenize(self, prompts, tokenizer, tok_kwargs)
        return _pad_tokens(inputs, tokenizer, tok_kwargs.get("max_length"))

    wrapped.__wrapped_for_tpu__ = True
    return wrapped


def install() -> None:
    pipeline_config = config.SanaPipelineConfig
    if not getattr(pipeline_config.tokenize_prompt, "__wrapped_for_tpu__", False):
        pipeline_config.tokenize_prompt = _padded_tokenize(
            pipeline_config.tokenize_prompt
        )
