"""Unit tests for letting the language model embed on decode steps.

`Qwen3VLForConditionalGeneration.forward` picks between two ways of reaching the
language model: `general_mm_embed_routine`, which embeds first and passes
`input_embeds`, or a direct call that passes `input_ids` and lets the model embed
inside its own (compiled) graph. `_embeds_in_language_model` decides, gated on the
per-wrapper `decode_embeds_in_language_model` flag.

What these guard: the routine is what splices image embeddings, so taking the
direct path anywhere other than decode would silently drop them; and the flag
defaulting on would change every Qwen3-VL subclass at once.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5MoeForConditionalGeneration,
)
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.platforms import current_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, HIDDEN = 2, 8


class _RecordingLM:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return torch.full((BS, HIDDEN), 3.0)

    def get_input_embeddings(self):
        return lambda ids: torch.zeros(ids.shape[0], HIDDEN)


def _forward_batch(is_decode=True):
    return SimpleNamespace(
        forward_mode=SimpleNamespace(
            is_decode=lambda: is_decode,
            is_target_verify=lambda: False,
        ),
        mrope_positions=torch.full((3, BS), 7, dtype=torch.int64),
        input_embeds=None,
        mm_inputs=None,
        contains_mm_inputs=lambda: False,
        contains_image_inputs=lambda: False,
        extend_prefix_lens_cpu=[0] * BS,
        extend_seq_lens_cpu=[1] * BS,
    )


def _run(stub, forward_batch, *, get_embedding=False, pp_proxy_tensors=None):
    """Drive the real base forward, with the embed routine faked minimally."""

    def fake_routine(*, input_ids, forward_batch, language_model, **kwargs):
        embeds = language_model.get_input_embeddings()(input_ids)
        return language_model(
            input_ids=None,
            forward_batch=forward_batch,
            input_embeds=embeds,
            **{k: v for k, v in kwargs.items() if k != "multimodal_model"},
        )

    with patch("sglang.srt.models.qwen3_vl.general_mm_embed_routine", fake_routine):
        return Qwen3VLForConditionalGeneration.forward(
            stub,
            torch.arange(BS),
            torch.zeros(BS, dtype=torch.int64),
            forward_batch,
            get_embedding,
            pp_proxy_tensors,
        )


def _make(lm, *, embeds_in_lm, **kw):
    """Stub whose `_embeds_in_language_model` is the real method."""
    stub = SimpleNamespace(
        decode_embeds_in_language_model=embeds_in_lm,
        language_model_only=kw.get("language_model_only", False),
        is_mrope_enabled=kw.get("is_mrope_enabled", False),
        capture_aux_hidden_states=False,
        use_deepstack=False,
        model=lm,
        lm_head="LM_HEAD",
        pp_group=SimpleNamespace(
            is_last_rank=kw.get("is_last_rank", True), is_first_rank=True
        ),
        logits_processor=lambda *args: ("logits", args),
        pooler=lambda hidden, fb: ("pooled", hidden),
    )
    stub._embeds_in_language_model = (
        lambda fb: Qwen3VLForConditionalGeneration._embeds_in_language_model(stub, fb)
    )
    return stub


class TestEmbedsInLanguageModel(CustomTestCase):
    def test_decode_passes_input_ids_when_enabled(self):
        lm = _RecordingLM()

        _run(_make(lm, embeds_in_lm=True), _forward_batch(is_decode=True))

        call = lm.calls[0]
        self.assertTrue(torch.equal(call["input_ids"], torch.arange(BS)))
        self.assertNotIn("input_embeds", call)

    def test_decode_still_uses_the_routine_when_disabled(self):
        # The default for every other Qwen3-VL wrapper.
        lm = _RecordingLM()

        _run(_make(lm, embeds_in_lm=False), _forward_batch(is_decode=True))

        call = lm.calls[0]
        self.assertIsNone(call["input_ids"])
        self.assertIsNotNone(call["input_embeds"])

    def test_prefill_always_uses_the_routine(self):
        # Prefill is where image embeddings get spliced -- taking the direct path
        # there would drop them.
        lm = _RecordingLM()

        _run(_make(lm, embeds_in_lm=True), _forward_batch(is_decode=False))

        call = lm.calls[0]
        self.assertIsNone(call["input_ids"])
        self.assertIsNotNone(call["input_embeds"])

    def test_pipeline_proxy_is_forwarded_on_the_direct_path(self):
        # The direct path must hand the proxy through, exactly as the
        # language_model_only branch does; a non-first stage consumes it instead
        # of embeddings.
        lm = _RecordingLM()
        proxy = object()

        _run(
            _make(lm, embeds_in_lm=True),
            _forward_batch(is_decode=True),
            pp_proxy_tensors=proxy,
        )

        self.assertIs(lm.calls[0]["pp_proxy_tensors"], proxy)

    def test_non_last_stage_returns_hidden_states(self):
        # Shared tail handles the rank; the direct path must not assume last.
        lm = _RecordingLM()

        out = _run(
            _make(lm, embeds_in_lm=True, is_last_rank=False),
            _forward_batch(is_decode=True),
        )

        self.assertTrue(torch.equal(out, torch.full((BS, HIDDEN), 3.0)))

    def test_embedding_request_still_reaches_the_pooler(self):
        out = _run(
            _make(_RecordingLM(), embeds_in_lm=True),
            _forward_batch(is_decode=True),
            get_embedding=True,
        )

        self.assertEqual(out[0], "pooled")


class TestWrapperOptIn(CustomTestCase):
    def test_qwen3_5_wrappers_opt_in_only_on_tpu(self):
        """The opt-in is platform-gated, not hardcoded on.

        The win needs the compiled boundary to sit *between* the embed routine and
        the language model, which is the TPU arrangement: only the annotated
        module is compiled, so the routine runs eagerly in front of it. Under a
        CUDA graph the whole forward is captured, routine included, so there is
        nothing to move -- and taking this path there would also skip the
        routine's copy into the pre-allocated ``forward_batch.input_embeds``
        buffer that the capture depends on.
        """
        expected = current_platform.is_tpu()
        self.assertEqual(
            Qwen3_5ForConditionalGeneration.decode_embeds_in_language_model, expected
        )
        self.assertEqual(
            Qwen3_5MoeForConditionalGeneration.decode_embeds_in_language_model, expected
        )

    def test_base_wrapper_stays_opted_out(self):
        # Flipping this default would change every Qwen3-VL subclass at once,
        # including backends where the language model is not a compiled boundary
        # and the routine's input_embeds buffer copy still matters.
        self.assertFalse(
            Qwen3VLForConditionalGeneration.decode_embeds_in_language_model
        )


if __name__ == "__main__":
    unittest.main()
