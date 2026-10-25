"""Dense DFlash2 selector: K target-head candidates per slot, one path walked over their lattice.

score[b, j, p, c] = unary[b, j, c] + <A[cand[b, j-1, p]] * proj(h[b, j]), B[cand[b, j, c]]>,
with the anchor as slot 0's only predecessor; slot j draws from score[b, j, i_{j-1}].
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.models.dflash import CandidateSelector, DFlash2DraftModel
from sglang_tpu.experimental.speculative.spec_ops import (
    STREAM_DRAFT,
    hash_uniforms,
    pick,
    verify_tokens_of,
)

# (row_scores [pbs, K], slot) -> chosen candidate index [pbs].
CandidateChooser = Callable[[torch.Tensor, int], torch.Tensor]
# (candidate_ids [pbs, gamma, K], scores [pbs, gamma, K, K]).
DraftLattice = tuple[torch.Tensor, torch.Tensor]


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


def walk_lattice_sampled(
    *,
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    uniforms: torch.Tensor,
    temperatures: torch.Tensor,
    greedy_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drafts [pbs, gamma] by core's sample_path inverse-CDF walk over the K candidates,
    and the [pbs, gamma, K] float32 score rows q was drawn from (lattice_path_rows).

    Every predecessor row draws at once; one int32 pass follows the indices via
    one-hot picks, as int64 lanes are emulated on TPU.
    """
    top_k = scores.shape[-1]
    temps = temperatures.view(-1).float().clamp_min(1e-5)[:, None, None, None]
    probs = torch.softmax(scores.float() / temps, dim=-1)
    maps = (uniforms[:, :, None, None] >= probs.cumsum(dim=-1)).sum(
        dim=-1, dtype=torch.int32
    )
    maps = torch.where(
        greedy_mask.view(-1, 1, 1),
        scores.argmax(dim=-1).to(torch.int32),
        maps.clamp_max(top_k - 1),
    )
    ids = candidate_ids.to(torch.int32)
    # Slot 0's K rows all score from the anchor, so any row will do.
    index = torch.zeros(ids.shape[0], dtype=torch.int32, device=ids.device)
    tokens, rows = [], []
    for slot in range(ids.shape[1]):
        rows.append(pick(scores[:, slot], index).float())
        index = pick(maps[:, slot], index)
        tokens.append(pick(ids[:, slot], index))
    return torch.stack(tokens, dim=1).to(candidate_ids.dtype), torch.stack(rows, dim=1)


def lattice_path_rows(
    candidate_ids: torch.Tensor, scores: torch.Tensor, draft_tokens: torch.Tensor
) -> torch.Tensor:
    """[pbs, gamma, K] float32 score rows q was drawn from along draft_tokens' path."""
    index = torch.zeros(
        candidate_ids.shape[0], dtype=torch.int64, device=candidate_ids.device
    )
    rows = []
    for slot in range(candidate_ids.shape[1]):
        rows.append(pick(scores[:, slot], index).float())
        slot_ids = candidate_ids[:, slot]
        index = (slot_ids == draft_tokens[:, slot, None]).to(torch.int32).argmax(dim=-1)
    return torch.stack(rows, dim=1)


def dense_logits(
    candidate_ids: torch.Tensor, rows: torch.Tensor, vocab_size: int
) -> torch.Tensor:
    """rows [pbs, gamma, K] scattered to [pbs, gamma, V], -inf off the candidates."""
    # softmax(dense / T) is then the selector's q.
    return torch.full(
        (*rows.shape[:2], vocab_size),
        float("-inf"),
        dtype=rows.dtype,
        device=rows.device,
    ).scatter(-1, candidate_ids, rows)


@compiled_op()
def sample_drafts(
    candidate_ids: torch.Tensor,
    scores: torch.Tensor,
    bonus_tokens: torch.Tensor,
    temperatures: torch.Tensor,
    top_ks: torch.Tensor,
    rng_step: torch.Tensor,
    vocab_size: int,
    *,
    dense: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sampled drafts [pbs, gamma], their q logits and verify tokens, in one program.

    q logits are the path's candidate rows [pbs, gamma, K] (over candidate_ids),
    or scattered to [pbs, gamma, V] when dense. hash_uniforms gives every TP
    rank the same walk, so nothing is synced. Rows with top_k <= 1 walk the argmax.
    """
    uniforms = hash_uniforms(rng_step, STREAM_DRAFT, tuple(scores.shape[:2]))
    draft_tokens, rows = walk_lattice_sampled(
        candidate_ids=candidate_ids,
        scores=scores,
        uniforms=uniforms,
        temperatures=temperatures,
        greedy_mask=top_ks.view(-1) <= 1,
    )
    q_logits = dense_logits(candidate_ids, rows, vocab_size) if dense else rows
    return draft_tokens, q_logits, verify_tokens_of(bonus_tokens, draft_tokens)


def sample_draft_tokens(
    self: DFlash2DraftModel, hidden_states: torch.Tensor, input_ids: torch.Tensor
) -> tuple[torch.Tensor, DraftLattice]:
    """Greedy selector path [bs, gamma] and its lattice, traced into the TPU draft graph."""
    block = self.block_size
    hidden_3d = hidden_states.view(-1, block, hidden_states.shape[-1])
    candidate_ids, scores = build_lattice(
        self, hidden_3d[:, 1:], input_ids.view(-1, block)[:, 0]
    )
    tokens = walk_lattice(
        candidate_ids=candidate_ids,
        scores=scores,
        choose=lambda row, slot: torch.argmax(row, dim=-1),
    )
    return tokens, (candidate_ids, scores)
