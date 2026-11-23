# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Ministral 3 patches (sglang_tpu.multimodal_gen.ministral3)."""

import contextlib
import importlib.util
import unittest
from unittest.mock import Mock, patch

import torch
from transformers import Ministral3Config
from transformers.generation import logits_process as lp
from transformers.masking_utils import create_causal_mask
from transformers.models.ministral3.modeling_ministral3 import (
    Ministral3ForCausalLM as HFMinistral3ForCausalLM,
)

from sglang.multimodal_gen.runtime.layers.attention.selector import (
    global_force_attn_backend_context_manager,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.encoders import mistral_3
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    ernie_image_pe,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import ministral3

register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)


def _models(vocab=256):
    """Transformers' Ministral 3 and sglang's, with the same random weights."""
    torch.manual_seed(7)
    config = Ministral3Config(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1_000_000.0,
            "factor": 4.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
            "original_max_position_embeddings": 16,
        },
        tie_word_embeddings=True,
        eos_token_id=2,
        pad_token_id=11,
    )
    config._attn_implementation = "sdpa"
    with global_force_attn_backend_context_manager(AttentionBackendEnum.TORCH_SDPA):
        reference = HFMinistral3ForCausalLM(config).eval()
        native = mistral_3.Ministral3ForCausalLM(config).eval()
    native.load_state_dict(reference.state_dict(), strict=True)
    return reference, native


@contextlib.contextmanager
def _installed(rank=0, broadcast=lambda obj, src: obj):
    """install() on one world rank, with eager compiles."""
    names = ("_can_use_unmasked_causal_attention", "create_causal_mask")
    world = Mock(rank_in_group=rank, broadcast_object=Mock(side_effect=broadcast))
    with contextlib.ExitStack() as stack:
        for owner, name in [(mistral_3, n) for n in names] + [
            (mistral_3.Ministral3ForCausalLM, "generate"),
            (ernie_image_pe.PromptEnhancementStage, "forward"),
        ]:
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(ministral3, "get_world_group", lambda: world))
        stack.enter_context(
            patch.object(
                ministral3,
                "build_torch_compile_kwargs",
                lambda mode: {"backend": "eager"},
            )
        )
        ministral3.install()
        ministral3.install()  # idempotent
        yield world


def _stage(generate):
    """A prompt-enhancement stage whose model calls ``generate()``."""
    # Its __init__ reads the global server args, which forward does not use.
    stage = object.__new__(ernie_image_pe.PromptEnhancementStage)
    stage.pe_model, stage.pe_tokenizer = Mock(), Mock()
    stage.pe_model.generate = lambda prompt, sampling_params: {"text": generate()}
    stage.pe_tokenizer.apply_chat_template.return_value = ""
    return stage


_STAGE_ARGS = Mock(pipeline_config=Mock(pe_model_max_length=16))


def _prompt(n, vocab=256):
    return (torch.arange(n)[None] * 7 + 3) % vocab


class TestMinistral3Decode(CustomTestCase):
    def test_generate_matches_transformers_generate(self):
        reference, native = _models()
        for n, sampling in (
            (5, dict(do_sample=False)),
            (300, dict(do_sample=True, top_k=1)),
        ):
            with self.subTest(n=n, **sampling), _installed(), torch.no_grad():
                ids = _prompt(n)
                expected = reference.generate(ids, max_new_tokens=12, **sampling)
                actual = ministral3._Decoder(native).generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    max_new_tokens=12,
                    **sampling,
                )
                torch.testing.assert_close(actual, expected)

    def test_one_compiled_graph_per_shape(self):
        _, native = _models()
        graphs = []

        def counting(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        with (
            patch.object(
                ministral3,
                "build_torch_compile_kwargs",
                lambda mode: {"backend": counting},
            ),
            torch.no_grad(),
        ):
            decoder = ministral3._Decoder(native)
            for n in (5, 40, 300):
                decoder.generate(input_ids=_prompt(n), max_new_tokens=6, do_sample=True)
        self.assertEqual(len(graphs), 2)  # a prompt chunk and one token

    def test_same_seed_draws_the_same_tokens(self):
        _, native = _models()
        outputs = []

        def generate():
            outputs.append(
                decoder.generate(
                    input_ids=_prompt(9), max_new_tokens=16, do_sample=True
                )
            )
            return ""

        with _installed(), torch.no_grad():
            decoder = ministral3._Decoder(native)
            for seed in (3, 3, 4):
                request = Mock(prompt="p", seed=seed, height=64, width=64)
                _stage(generate).forward(request, _STAGE_ARGS)
        self.assertIsNone(ministral3._SEED.get())
        torch.testing.assert_close(outputs[0], outputs[1])
        self.assertFalse(torch.equal(outputs[0], outputs[2]))

    def test_rank_0_enhances_and_every_rank_takes_its_prompt(self):
        generate = Mock(return_value="enhanced")
        sent_by_rank_0 = (lambda obj, src: obj, lambda obj, src: ("enhanced", None))
        for rank, broadcast in enumerate(sent_by_rank_0):
            request = Mock(prompt="p", seed=3, height=64, width=64)
            with _installed(rank, broadcast) as world:
                _stage(generate).forward(request, _STAGE_ARGS)
            self.assertEqual(request.prompt, "enhanced")
            self.assertEqual(world.broadcast_object.call_args.kwargs, {"src": 0})
        generate.assert_called_once()  # on rank 0 only
        # A failure on rank 0 fails the other ranks instead of leaving them waiting.
        with (
            _installed(1, lambda obj, src: (None, "OSError()")),
            self.assertRaisesRegex(RuntimeError, "OSError"),
        ):
            _stage(generate).forward(Mock(prompt="p", seed=3), _STAGE_ARGS)


class TestCandidateDraws(CustomTestCase):
    def test_candidates_give_the_full_row_distribution(self):
        gen = torch.Generator().manual_seed(0)
        # bf16-rounded logits, as the model gives: ties around the k-th are common.
        logits = (torch.randn(1, 4096, generator=gen) * 2).bfloat16().float()
        ids = torch.zeros(1, 3, dtype=torch.long)
        for processors in (
            [lp.TopKLogitsWarper(50)],
            [
                lp.TemperatureLogitsWarper(0.7),
                lp.TopKLogitsWarper(50),
                lp.TopPLogitsWarper(0.9),
            ],
            [lp.TopKLogitsWarper(20), lp.MinPLogitsWarper(0.05)],
        ):
            processors = lp.LogitsProcessorList(processors)
            k = ministral3.candidate_top_k(processors)
            values, candidates = torch.topk(logits, ministral3.CANDIDATES)
            self.assertLess(values[0, -1], values[0, k - 1])
            sparse = torch.zeros_like(logits)
            sparse[0, candidates[0]] = processors(ids, values).softmax(-1)[0]
            torch.testing.assert_close(sparse, processors(ids, logits).softmax(-1))

    def test_unsupported_processors_and_ties_take_the_full_row(self):
        for processors in (
            [lp.TopKLogitsWarper(64)],
            [lp.RepetitionPenaltyLogitsProcessor(1.1), lp.TopKLogitsWarper(50)],
            [lp.TopKLogitsWarper(50), lp.TypicalLogitsWarper(0.9)],
            [lp.TopPLogitsWarper(0.9)],
        ):
            self.assertIsNone(ministral3.candidate_top_k(processors))
        logits = torch.zeros(1, 4096)
        logits[0, :70] = 1.0  # 70 logits tie with the 50th
        processors = Mock(wraps=lp.LogitsProcessorList([lp.TopKLogitsWarper(50)]))
        token = ministral3.next_token(
            processors, None, logits, torch.topk(logits, 64), 50, True, None
        )
        self.assertEqual(processors.call_args.args[1].shape, (1, 4096))
        self.assertLess(int(token), 70)


class TestEncoderMask(CustomTestCase):
    def test_host_mask_matches_transformers(self):
        _, native = _models()
        mask = torch.tensor([[1] * 5 + [0] * 3, [1] * 8])
        # The arguments sglang's MistralModel passes.
        expected = create_causal_mask(
            config=native.config,
            attention_mask=mask,
            past_key_values=None,
            position_ids=torch.arange(8)[None],
            **{mistral_3._CREATE_CAUSAL_MASK_ARG: torch.zeros(2, 8, 32)},
        )
        torch.testing.assert_close(ministral3.causal_padding_mask(mask, 8), expected)

    def test_padded_encoder_rows_match_stock(self):
        _, native = _models()
        ids = _prompt(12).repeat(2, 1)
        mask = torch.ones_like(ids)
        mask[0, 7:] = 0
        context = set_forward_context(current_timestep=0, attn_metadata=None)
        with torch.no_grad(), context:
            expected = native.model(
                input_ids=ids, attention_mask=mask
            ).last_hidden_state
            with (
                _installed(),
                patch.object(
                    ministral3, "_STOCK_CAUSAL_MASK", side_effect=AssertionError
                ),
            ):
                actual = native.model(
                    input_ids=ids, attention_mask=mask
                ).last_hidden_state
        torch.testing.assert_close(actual, expected)


def _has_tpu() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        return torch.tpu.device_count() > 0
    except Exception:
        return False


@unittest.skipUnless(_has_tpu(), "requires a TPU")
class TestMinistral3OnTpu(CustomTestCase):
    def test_compiled_decode_and_padded_encoder_on_tpu(self):
        # As the worker sets it: float32 matmuls in full precision on TPU too.
        precision = torch.get_float32_matmul_precision()
        self.addCleanup(torch.set_float32_matmul_precision, precision)
        torch.set_float32_matmul_precision("highest")
        _, native = _models()
        ids, mask = _prompt(12).repeat(2, 1), torch.ones(2, 12, dtype=torch.long)
        mask[0, 7:] = 0
        context = set_forward_context(current_timestep=0, attn_metadata=None)
        with torch.no_grad(), context:
            cpu = ministral3._Decoder(native)
            cpu.forward = cpu._forward
            expected_ids = cpu.generate(
                _prompt(40), max_new_tokens=8, do_sample=True, top_k=1
            )
            expected = native.model(input_ids=ids, attention_mask=mask)
            native.to("tpu")
            with (
                patch.object(
                    mistral_3, "create_causal_mask", ministral3._causal_mask_on_host
                ),
                patch.object(
                    mistral_3,
                    "_can_use_unmasked_causal_attention",
                    ministral3._unmasked_on_host,
                ),
            ):
                actual = native.model(
                    input_ids=ids.to("tpu"), attention_mask=mask.to("tpu")
                )
            tpu_ids = ministral3._Decoder(native).generate(
                _prompt(40).to("tpu"), max_new_tokens=8, do_sample=True, top_k=1
            )
        torch.testing.assert_close(
            actual.last_hidden_state.cpu(),
            expected.last_hidden_state,
            atol=1e-4,
            rtol=1e-4,
        )
        torch.testing.assert_close(tpu_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
