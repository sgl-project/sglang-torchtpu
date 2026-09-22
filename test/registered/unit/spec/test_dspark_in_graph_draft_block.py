"""The in-graph greedy draft block must match the eager proposal tail."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.dspark import (
    DSparkDraftMixin,
    GatedMarkovHead,
    RNNHead,
    VanillaMarkov,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    resolve_greedy_mask,
    sample_draft_block,
)
from sglang.srt.speculative.spec_tp_sync import SpecTpSync
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_VOCAB = 64
_RANK = 8
_HIDDEN = 16
_GAMMA = 3
_BS = 5


class _StubDSparkDraft:
    """Carries the mixin hook over a real markov head and a plain lm_head."""

    sample_draft_tokens_tpu = DSparkDraftMixin.sample_draft_tokens_tpu

    def __init__(self, *, markov_head, lm_head_weight, sample_from_anchor):
        self.gamma = _GAMMA
        self.sample_from_anchor = sample_from_anchor
        self.query_token_num = _GAMMA if sample_from_anchor else _GAMMA + 1
        self.markov_head = markov_head
        self.lm_head_weight = lm_head_weight

    def compute_base_logits(self, hidden):
        return torch.matmul(hidden, self.lm_head_weight.T), None


def _build_markov_head(head_type: str):
    if head_type == "vanilla":
        return VanillaMarkov(vocab_size=_VOCAB, markov_rank=_RANK)
    if head_type == "gated":
        return GatedMarkovHead(
            vocab_size=_VOCAB, markov_rank=_RANK, hidden_size=_HIDDEN
        )
    return RNNHead(vocab_size=_VOCAB, markov_rank=_RANK, hidden_size=_HIDDEN)


class TestDSparkInGraphDraftBlock(CustomTestCase):
    """The fold moves the semi-AR chain from the eager tail into the draft
    graph, so it has to pick the same hidden rows and the same anchor token;
    getting either wrong only shows up as a silent acceptance-rate drop."""

    def _assert_parity(self, *, head_type: str, sample_from_anchor: bool):
        torch.manual_seed(0)
        markov_head = _build_markov_head(head_type).eval()
        lm_head_weight = torch.randn(_VOCAB, _HIDDEN)
        draft = _StubDSparkDraft(
            markov_head=markov_head,
            lm_head_weight=lm_head_weight,
            sample_from_anchor=sample_from_anchor,
        )
        query_token_num = draft.query_token_num
        hidden_states = torch.randn(_BS * query_token_num, _HIDDEN)
        input_ids = torch.randint(0, _VOCAB, (_BS * query_token_num,))

        with torch.no_grad():
            folded = draft.sample_draft_tokens_tpu(hidden_states, input_ids)

            hidden_by_query = hidden_states.view(_BS, query_token_num, _HIDDEN)
            draft_hidden = (
                hidden_by_query if sample_from_anchor else hidden_by_query[:, 1:, :]
            )
            base_logits = torch.matmul(
                draft_hidden.reshape(_BS * _GAMMA, _HIDDEN), lm_head_weight.T
            ).view(_BS, _GAMMA, -1)
            eager = sample_draft_block(
                base_logits=base_logits,
                anchor_tokens=input_ids.view(_BS, query_token_num)[:, 0],
                draft_hidden=draft_hidden,
                sampling_info=None,
                markov_head=markov_head,
                device=torch.device("cpu"),
                tp_sync=SpecTpSync(SimpleNamespace(world_size=1, rank_in_group=0)),
            )

        self.assertEqual(tuple(folded.shape), (_BS, _GAMMA))
        torch.testing.assert_close(folded, eager.draft_tokens)

    def test_vanilla_head_matches_eager_tail(self):
        for sample_from_anchor in (True, False):
            with self.subTest(sample_from_anchor=sample_from_anchor):
                self._assert_parity(
                    head_type="vanilla", sample_from_anchor=sample_from_anchor
                )

    def test_hidden_dependent_heads_match_eager_tail(self):
        for head_type in ("gated", "rnn"):
            with self.subTest(head_type=head_type):
                self._assert_parity(head_type=head_type, sample_from_anchor=True)


def _sampling_info(top_ks):
    return SimpleNamespace(
        top_ks=torch.tensor(top_ks, dtype=torch.int64),
        is_all_greedy=all(k <= 1 for k in top_ks),
    )


class TestResolveGreedyMask(CustomTestCase):
    """All-greedy batches take a shared all-True buffer instead of a per-batch
    compare; the shortcut has to keep agreeing with top_ks <= 1, and the buffer
    has to grow rather than hand back a short or stale row range."""

    def test_all_greedy_agrees_with_the_compare(self):
        info = _sampling_info([1, 1, 1])
        mask = resolve_greedy_mask(bs=3, sampling_info=info, device=torch.device("cpu"))
        torch.testing.assert_close(mask, (info.top_ks <= 1).view(-1))

    def test_mixed_batch_keeps_the_per_row_compare(self):
        info = _sampling_info([1, 50, 1])
        mask = resolve_greedy_mask(bs=3, sampling_info=info, device=torch.device("cpu"))
        torch.testing.assert_close(mask, torch.tensor([True, False, True]))

    def test_buffer_grows_past_the_first_batch(self):
        small = resolve_greedy_mask(
            bs=2, sampling_info=None, device=torch.device("cpu")
        )
        big = resolve_greedy_mask(
            bs=257, sampling_info=None, device=torch.device("cpu")
        )
        self.assertEqual(tuple(small.shape), (2,))
        self.assertEqual(tuple(big.shape), (257,))
        self.assertTrue(bool(big.all()))


if __name__ == "__main__":
    unittest.main()
