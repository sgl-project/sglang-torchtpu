"""Dense DFlash2 selector: K target-head candidates per slot, one path walked over their lattice.

score[b, j, p, c] = unary[b, j, c] + <A[cand[b, j-1, p]] * proj(h[b, j]), B[cand[b, j, c]]>,
with the anchor as slot 0's only predecessor; slot j draws from score[b, j, i_{j-1}].
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from sglang.srt.models.dflash import CandidateSelector, DFlash2DraftModel
from sglang_tpu.experimental.speculative.spec_ops import pick

# (row_scores [pbs, K], slot) -> chosen candidate index [pbs].
CandidateChooser = Callable[[torch.Tensor, int], torch.Tensor]
# (dense_logits [pbs, V], slot) -> sampled [pbs] tokens; block_draft_worker.DraftSampler.
StepSampler = Callable[[torch.Tensor, int], torch.Tensor]


def build_lattice(
    model: DFlash2DraftModel, draft_hidden: torch.Tensor, anchor_tokens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Candidate ids [pbs, gamma, K] and scores [pbs, gamma, K, K] from [pbs, gamma, H] hidden."""
    bs, gamma = draft_hidden.shape[:2]
    candidate_ids, unary_logits = model.compute_candidates(
        draft_hidden.reshape(bs * gamma, -1)
    )
    candidate_ids = candidate_ids.view(bs, gamma, -1)
    selector = model.candidate_selector
    assert isinstance(selector, CandidateSelector)
    # Core's _score_edges without its torch.compile / mark_static, which the TPU trace
    # cannot take; table lookups go through embedding, not advanced indexing.
    hidden = selector.hidden_projection(draft_hidden)
    predecessor_ids = torch.cat(
        [
            anchor_tokens[:, None, None].expand(-1, 1, selector.top_k),
            candidate_ids[:, :-1],
        ],
        dim=1,
    )
    predecessors = F.embedding(predecessor_ids, selector.predecessor_codebook)
    successors = F.embedding(candidate_ids, selector.successor_codebook)
    scores = unary_logits.view(bs, gamma, 1, -1) + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )
    return candidate_ids, scores


def walk_lattice(
    *, candidate_ids: torch.Tensor, scores: torch.Tensor, choose: CandidateChooser
) -> torch.Tensor:
    """Drafts [pbs, gamma]: each slot chooses from the row its predecessor selects."""
    # Slot 0's K rows all score from the anchor, so any row will do.
    index = torch.zeros(
        candidate_ids.shape[0], dtype=torch.int64, device=candidate_ids.device
    )
    tokens = []
    for slot in range(candidate_ids.shape[1]):
        index = choose(pick(scores[:, slot], index), slot)
        tokens.append(pick(candidate_ids[:, slot], index))
    return torch.stack(tokens, dim=1)


def sample_drafts(
    *,
    draft_model: DFlash2DraftModel,
    draft_hidden: torch.Tensor,
    anchor_tokens: torch.Tensor,
    vocab_size: int,
    sampler: StepSampler,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sampled drafts [pbs, gamma] and the dense [pbs, gamma, V] logits q is drawn from."""
    with torch.no_grad():
        candidate_ids, scores = build_lattice(draft_model, draft_hidden, anchor_tokens)
    dense_rows: list[torch.Tensor] = []

    def choose(row: torch.Tensor, slot: int) -> torch.Tensor:
        slot_ids = candidate_ids[:, slot]
        # -inf off the candidates, so softmax(dense / T) is the selector's q.
        dense = torch.full(
            (row.shape[0], vocab_size),
            float("-inf"),
            dtype=row.dtype,
            device=row.device,
        ).scatter(-1, slot_ids, row)
        dense_rows.append(dense)
        token = sampler(dense, slot)
        return (slot_ids == token[:, None]).to(torch.int32).argmax(dim=-1)

    draft_tokens = walk_lattice(
        candidate_ids=candidate_ids, scores=scores, choose=choose
    )
    return draft_tokens, torch.stack(dense_rows, dim=1)


def sample_draft_tokens(
    self: DFlash2DraftModel, hidden_states: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Greedy selector path [bs, gamma], traced into the TPU draft graph."""
    block = self.block_size
    hidden_3d = hidden_states.view(-1, block, hidden_states.shape[-1])
    candidate_ids, scores = build_lattice(
        self, hidden_3d[:, 1:], input_ids.view(-1, block)[:, 0]
    )
    return walk_lattice(
        candidate_ids=candidate_ids,
        scores=scores,
        choose=lambda row, slot: torch.argmax(row, dim=-1),
    )
