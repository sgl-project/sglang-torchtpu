"""CPU parity of the TPU DFlash2 selector with core's CandidateSelector."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch._dynamo.config

from sglang.srt.models.dflash import CandidateSelector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.experimental.models.dflash2 import (
    build_lattice,
    sample_draft_tokens,
    sample_drafts,
    walk_lattice,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

BS, GAMMA, TOP_K, HIDDEN, VOCAB, RANK = 3, 4, 5, 8, 32, 6


def _model():
    torch.manual_seed(0)
    selector = CandidateSelector(
        hidden_size=HIDDEN, vocab_size=VOCAB, state_rank=RANK, top_k=TOP_K
    )
    with torch.no_grad():
        for param in selector.parameters():
            param.normal_()
    head = torch.randn(VOCAB, HIDDEN)

    def compute_candidates(hidden):
        vals, ids = torch.topk(hidden @ head.T, TOP_K, dim=-1)
        return ids, vals

    return SimpleNamespace(
        candidate_selector=selector,
        compute_candidates=compute_candidates,
        block_size=GAMMA + 1,
    )


def _core_lattice(model, hidden, anchors):
    candidate_ids, unary = model.compute_candidates(hidden.reshape(BS * GAMMA, -1))
    candidate_ids = candidate_ids.view(BS, GAMMA, -1)
    scores = model.candidate_selector.build_lattice(
        candidate_ids=candidate_ids,
        unary_logits=unary.view(BS, GAMMA, -1),
        hidden_states=hidden,
        anchor_token_ids=anchors,
    )
    return candidate_ids, scores


class TestExperimentalDflash2Selector(CustomTestCase):
    def setUp(self):
        # Core wraps its lattice and walk in torch.compile; compare them eagerly.
        patcher = torch._dynamo.config.patch(disable=True)
        patcher.__enter__()
        self.addCleanup(patcher.__exit__, None, None, None)
        self.model = _model()
        self.hidden = torch.randn(BS, GAMMA, HIDDEN)
        self.anchors = torch.randint(0, VOCAB, (BS,))

    def test_lattice_matches_core(self):
        with torch.no_grad():
            ids, scores = build_lattice(self.model, self.hidden, self.anchors)
            core_ids, core_scores = _core_lattice(self.model, self.hidden, self.anchors)
        torch.testing.assert_close(ids, core_ids)
        torch.testing.assert_close(scores, core_scores)

    def test_walk_matches_core_sampling_and_greedy(self):
        with torch.no_grad():
            ids, scores = _core_lattice(self.model, self.hidden, self.anchors)
        uniforms = torch.rand(BS, GAMMA)
        temperatures = torch.tensor([0.7, 1.0, 1.3])
        greedy_mask = torch.tensor([False, True, False])

        def inverse_cdf(row, slot):
            probs = torch.softmax(row / temperatures[:, None], dim=-1)
            drawn = (uniforms[:, slot : slot + 1] >= probs.cumsum(-1)).sum(-1)
            return torch.where(greedy_mask, row.argmax(-1), drawn.clamp_max(TOP_K - 1))

        tokens = walk_lattice(candidate_ids=ids, scores=scores, choose=inverse_cdf)
        core_tokens, _ = self.model.candidate_selector.sample_path(
            candidate_ids=ids,
            scores=scores,
            uniforms=uniforms,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
        )
        torch.testing.assert_close(tokens, core_tokens)

    def test_dense_logits_carry_core_q(self):
        temperatures = torch.tensor([0.7, 1.0, 1.3])
        with torch.no_grad():
            ids, scores = _core_lattice(self.model, self.hidden, self.anchors)
        core_tokens, core_q = self.model.candidate_selector.sample_path(
            candidate_ids=ids,
            scores=scores,
            uniforms=torch.rand(BS, GAMMA),
            temperatures=temperatures,
            greedy_mask=torch.zeros(BS, dtype=torch.bool),
        )
        with torch.no_grad():
            tokens, dense = sample_drafts(
                draft_model=self.model,
                draft_hidden=self.hidden,
                anchor_tokens=self.anchors,
                vocab_size=VOCAB,
                sampler=lambda logits, slot: core_tokens[:, slot],
            )
        torch.testing.assert_close(tokens, core_tokens)
        # The verify reads q as softmax(dense / T); off-candidate mass must be zero.
        q = torch.softmax(dense / temperatures[:, None, None], dim=-1)
        torch.testing.assert_close(q.gather(-1, ids), core_q)
        torch.testing.assert_close(q.gather(-1, ids).sum(-1), torch.ones(BS, GAMMA))

    def test_graph_hook_walks_greedy_path_from_block(self):
        block = GAMMA + 1
        block_hidden = torch.randn(BS, block, HIDDEN)
        input_ids = torch.randint(0, VOCAB, (BS, block))
        with torch.no_grad():
            tokens = sample_draft_tokens(
                self.model, block_hidden.reshape(BS * block, -1), input_ids.view(-1)
            )
            ids, scores = _core_lattice(
                self.model, block_hidden[:, 1:], input_ids[:, 0]
            )
        core_tokens, _ = self.model.candidate_selector.sample_path(
            candidate_ids=ids,
            scores=scores,
            uniforms=torch.zeros(BS, GAMMA),
            temperatures=torch.ones(BS),
            greedy_mask=torch.ones(BS, dtype=torch.bool),
        )
        torch.testing.assert_close(tokens, core_tokens)


if __name__ == "__main__":
    unittest.main()
