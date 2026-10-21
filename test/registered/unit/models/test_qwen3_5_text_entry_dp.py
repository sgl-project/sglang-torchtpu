"""Qwen3_5TextForCausalLM under DP attention.

Under DP attention the embedding is replicated while a DP LM head shards over
the attention-TP group, so a tied head can alias the embedding only when that
group is one rank; otherwise it is a separate head loaded from the embedding
tensor. The fused GDN in-projections must be in packed_modules_mapping for the
fp8 skip list to match them.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import DEFAULT, MagicMock, patch

import torch

import sglang.srt.models.qwen3_5 as qwen3_5
from sglang.srt.layers.quantization.utils import is_layer_skipped
from sglang.srt.models.qwen3_5 import Qwen3_5TextForCausalLM
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VL_EMBED = "model.language_model.embed_tokens.weight"
TEXT_EMBED = "model.embed_tokens.weight"


def _loader_stub(*, tied: bool, head_is_embedding: bool) -> SimpleNamespace:
    """Only what load_weights touches: config, the head, the backbone."""
    stub = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=tied), received=[]
    )
    embedding = SimpleNamespace(weight=SimpleNamespace(weight_loader=MagicMock()))
    stub.model = MagicMock(embed_tokens=embedding)
    # The side effect drains the lazy generator the backbone would consume.
    stub.model.load_weights.side_effect = lambda names: stub.received.extend(names)
    stub.lm_head = (
        embedding
        if head_is_embedding
        else SimpleNamespace(weight=SimpleNamespace(weight_loader=MagicMock()))
    )
    return stub


class TestQwen3_5TextEntryLmHead(CustomTestCase):
    def _build(
        self,
        *,
        tied: bool,
        dp_attention: bool,
        enable_dp_lm_head: bool = True,
        attn_tp_size: int = 1,
    ):
        """Construct the entry with backbone, head and logits stubbed out."""
        override = get_context().override_server_args(
            enable_dp_lm_head=enable_dp_lm_head
        )
        override.install()
        self.addCleanup(override.restore)
        config = SimpleNamespace(tie_word_embeddings=tied, vocab_size=32, hidden_size=8)
        with (
            get_parallel().override(attn_tp_size=attn_tp_size),
            patch.object(qwen3_5, "is_dp_attention_enabled", return_value=dp_attention),
            patch.multiple(
                qwen3_5,
                Qwen3_5ForCausalLM=DEFAULT,
                LogitsProcessor=DEFAULT,
                ParallelLMHead=DEFAULT,
            ) as mocks,
        ):
            model = Qwen3_5TextForCausalLM(config)
        return model, mocks["ParallelLMHead"]

    def test_tied_head_aliases_the_embedding_when_layouts_agree(self):
        for dp_attention, attn_tp_size in ((False, 8), (True, 1)):
            with self.subTest(dp_attention=dp_attention, attn_tp_size=attn_tp_size):
                model, head_cls = self._build(
                    tied=True, dp_attention=dp_attention, attn_tp_size=attn_tp_size
                )
                self.assertIs(model.lm_head, model.model.embed_tokens)
                head_cls.assert_not_called()

    def test_tied_head_is_a_separate_shard_under_attention_tp(self):
        # Replicated embedding vs. a head sharded over 2 attention-TP ranks.
        model, head_cls = self._build(tied=True, dp_attention=True, attn_tp_size=2)

        self.assertIsNot(model.lm_head, model.model.embed_tokens)
        self.assertTrue(head_cls.call_args.kwargs["use_attn_tp_group"])

    def test_untied_head_shards_over_the_group_the_logits_gather_uses(self):
        for enable_dp_lm_head in (True, False):
            with self.subTest(enable_dp_lm_head=enable_dp_lm_head):
                _, head_cls = self._build(
                    tied=False, dp_attention=True, enable_dp_lm_head=enable_dp_lm_head
                )
                self.assertEqual(
                    head_cls.call_args.kwargs["use_attn_tp_group"], enable_dp_lm_head
                )

    def test_separate_tied_head_loads_from_the_embedding_under_either_layout(self):
        # Both the VL and the text-only key layouts reach the head after the
        # rename; a serialized lm_head is ignored because the embedding wins.
        for embed_key in (VL_EMBED, TEXT_EMBED):
            with self.subTest(embed_key=embed_key):
                stub = _loader_stub(tied=True, head_is_embedding=False)
                embed = torch.arange(6.0).view(3, 2)

                Qwen3_5TextForCausalLM.load_weights(
                    stub, [(embed_key, embed), ("lm_head.weight", torch.zeros(3, 2))]
                )

                head_loader = stub.lm_head.weight.weight_loader
                head_loader.assert_called_once()
                self.assertIs(head_loader.call_args.args[1], embed)
                self.assertEqual([n for n, _ in stub.received], ["embed_tokens.weight"])

    def test_aliased_tied_head_is_loaded_only_through_the_embedding(self):
        stub = _loader_stub(tied=True, head_is_embedding=True)

        Qwen3_5TextForCausalLM.load_weights(
            stub, [(VL_EMBED, torch.ones(3, 2)), ("lm_head.weight", torch.zeros(3, 2))]
        )

        stub.lm_head.weight.weight_loader.assert_not_called()
        self.assertEqual([n for n, _ in stub.received], ["embed_tokens.weight"])

    def test_untied_head_loads_its_own_weight(self):
        stub = _loader_stub(tied=False, head_is_embedding=False)
        head = torch.zeros(3, 2)

        Qwen3_5TextForCausalLM.load_weights(
            stub, [(VL_EMBED, torch.ones(3, 2)), ("lm_head.weight", head)]
        )

        head_loader = stub.lm_head.weight.weight_loader
        head_loader.assert_called_once()
        self.assertIs(head_loader.call_args.args[1], head)

    def test_fp8_skip_list_matches_the_fused_gdn_in_projection(self):
        # The checkpoint lists in_proj_a / in_proj_b; the module is in_proj_ba.
        ignored = [
            "model.language_model.layers.0.linear_attn.in_proj_a",
            "model.language_model.layers.0.linear_attn.in_proj_b",
        ]

        self.assertTrue(
            is_layer_skipped(
                "model.language_model.layers.0.linear_attn.in_proj_ba",
                ignored,
                fused_mapping=Qwen3_5TextForCausalLM.packed_modules_mapping,
            )
        )


if __name__ == "__main__":
    unittest.main()
