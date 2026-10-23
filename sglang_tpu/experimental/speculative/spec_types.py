"""Algorithm-neutral data types for TpuSpecWorker.

Device containers are NamedTuple because they are built inside the compiled step;
TpuSpecVerifyInput is host-side addressing built once per step outside it. Every
device tensor keeps the batch-bucket row count pbs; only the scheduler-facing
result is cut back to the live bs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import msgspec
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput
from sglang_tpu.experimental.speculative.spec_utils import pad_rows, padded_h2d


class SpecState(NamedTuple):
    """State carried between steps, rows padded to the batch bucket pbs.

    Attributes:
      bonus_tokens: [pbs] root token r of the next draft chain, not yet in target KV.
      seq_lens: [pbs] committed prefix length n; r sits at position n. 0 on padded rows.
      req_pool_indices: [pbs] handle to the per-request KV. 0 (reserved) on padded rows.
      draft_carry: Draft conditioning not held in a KV cache, as a NamedTuple of
        row-aligned [pbs, ...] tensors or absent fields, e.g. EAGLE3's first
        draft token and draft hidden state. None when the draft context lives
        in draft KV (DSpark).
    """

    bonus_tokens: torch.Tensor
    seq_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    draft_carry: Any = None


class CommitResult(NamedTuple):
    """commit's output.

    Attributes:
      state: The advanced SpecState.
      kv_fence_tail: A small device tensor computed after the step's last KV
        write; the TPU KV free fence's result copy waits on it.
    """

    state: SpecState
    kv_fence_tail: torch.Tensor


def map_carry(carry: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
    """Apply fn to every row-aligned tensor of a draft_carry NamedTuple."""
    # A list, not a generator: Dynamo cannot reconstruct a generator in pad_state_rows.
    return (
        None
        if carry is None
        else carry._make([fn(t) if t is not None else None for t in carry])
    )


class BucketState(NamedTuple):
    """Decode state padded to the batch bucket, reused while the batch composition holds.

    Attributes:
      key: batch.req_pool_indices of the producing step. The scheduler hands it
        back as spec_info.future_indices; filter/merge rebind it.
      state: The padded SpecState.
      live_mask: [pbs] bool, False on padded rows.
      temperatures: [pbs] float32, 1 on padded rows; None when all greedy.
      top_ks: [pbs] int32, 1 on padded rows; None when all greedy.
      top_ps: [pbs] float32, 1 on padded rows; None when all greedy.
      min_ps: [pbs] float32, 0 on padded rows; None when all greedy.
      need_top_k: Some request restricts top_k; the target is filtered for it.
      need_top_p: Some request restricts top_p; the target is filtered for it.
      need_min_p: Some request restricts min_p; the target is filtered for it.
    """

    key: torch.Tensor
    state: SpecState
    live_mask: torch.Tensor
    temperatures: torch.Tensor | None
    top_ks: torch.Tensor | None
    top_ps: torch.Tensor | None = None
    min_ps: torch.Tensor | None = None
    need_top_k: bool = False
    need_top_p: bool = False
    need_min_p: bool = False

    @property
    def need_filters(self) -> bool:
        return self.need_top_k or self.need_top_p or self.need_min_p


class VerifyRows(NamedTuple):
    """Positions and KV slots of the N + 1 verify rows [r, d_1, ..., d_N].

    The scheduler pre-allocates the slots; the draft block forward and the
    target verify write the same slots.

    Attributes:
      positions: [pbs, N + 1] int64, seq_lens + 0..N.
      cache_locs: [pbs, N + 1] int64 KV slots; dummy slot 0 on padded rows.
      flat_positions: [pbs * (N + 1)] positions as its own graph output.
      flat_cache_locs: [pbs * (N + 1)] cache_locs as its own graph output.
    """

    positions: torch.Tensor
    cache_locs: torch.Tensor
    # A reshape view fed to direct_compile costs an eager as_strided program per bucket.
    flat_positions: torch.Tensor
    flat_cache_locs: torch.Tensor


class DraftResult(NamedTuple):
    """Draft chain d_1..d_N and the draft distribution q that sampled it.

    Verify rows are [r, d_1, ..., d_N]; target logits/hidden for them come back
    as LogitsProcessorOutput rows [pbs * (N + 1), ...].

    Attributes:
      draft_tokens: [pbs, N].
      draft_logits: [pbs, N, V] logits q was sampled from (before temperature);
        None only when the batch is all greedy.
      num_verify_drafts: [pbs] int32 in [0, N]; only the first k drafts are
        verified (DSpark budget). Equals N when no pruning applies.
      verify_tokens: Optional flat [pbs * (N + 1)] [r, d_1, ..., d_N] rows when
        the draft already built them; verify concatenates them otherwise.
    """

    draft_tokens: torch.Tensor
    draft_logits: torch.Tensor | None
    num_verify_drafts: torch.Tensor
    verify_tokens: torch.Tensor | None = None


class AcceptResult(NamedTuple):
    """Lossless acceptance outcome, [pbs] rows.

    Attributes:
      num_correct_drafts: [pbs] accepted drafts, bonus excluded.
      bonus_tokens: [pbs] token sampled at the last accepted row.
      accept_tokens: [pbs, N + 1] correct drafts then bonus, padded with -1.
      accept_lens: [pbs] int32 committed tokens, 0 on padded rows.
      next_seq_lens: [pbs] seq_lens + accept_lens.
      window_tokens: flat [pbs * (N + 1)] accept_tokens with 0 for -1, for a
        commit that rewrites the whole verify window (EAGLE3 draft extend).
      window_seq_lens: [pbs] seq_lens + N + 1, the length once that window is written.
      live_mask: [pbs] bool, False on padded rows.
      last_steps: [pbs] int64 accept_lens - 1, the verify step each row's
        linear-attention state commits (-1 on padded rows).
    """

    num_correct_drafts: torch.Tensor
    bonus_tokens: torch.Tensor
    accept_tokens: torch.Tensor
    accept_lens: torch.Tensor
    next_seq_lens: torch.Tensor
    window_tokens: torch.Tensor
    window_seq_lens: torch.Tensor
    live_mask: torch.Tensor
    last_steps: torch.Tensor


class DraftBlockInputs(NamedTuple):
    """Device fields of a draft forward over width verify rows, [pbs] rows.

    Attributes:
      input_ids: [pbs * width] tokens of the block.
      positions: [pbs * width] int64.
      cache_locs: [pbs * width] int64 KV slots.
      seq_lens: [pbs] length handed to the forward, 0 on padded rows: post-block
        for the block drafter, pre-row for the chain drafter (RPA adds the width).
      extend_prefix_lens: [pbs] int32 pre-block lengths, 0 on padded rows.
      extend_seq_lens: [pbs] int32, width on live rows and 0 on padded rows.
    """

    input_ids: torch.Tensor
    positions: torch.Tensor
    cache_locs: torch.Tensor
    seq_lens: torch.Tensor
    extend_prefix_lens: torch.Tensor
    extend_seq_lens: torch.Tensor


class TpuSpecVerifyInput(msgspec.Struct, frozen=True):
    """Addressing for one verify step; the scheduler's plan already padded it to the bucket.

    Attributes:
      verify_cache_candidates: [pbs, 2 * (N + 1)] device int64 KV slots of
        [base, base + 2 * (N + 1)); the verify rows are picked without a D2H sync.
      committed_lens: [pbs] device int64 committed base.
      req_pool_indices: [pbs] device pool rows. 0 (reserved) marks padded rows
        in all three device fields.
      req_pool_indices_cpu: [pbs] host pool rows for host-built metadata.
      seq_lens_cpu: [bs] host committed base.
      host_len_slack: Rows the host lengths may trail the device by.
      future_indices: The scheduler's spec_info.future_indices, None before the
        first overlap relay.
    """

    verify_cache_candidates: torch.Tensor
    committed_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    req_pool_indices_cpu: torch.Tensor
    seq_lens_cpu: torch.Tensor
    host_len_slack: int
    future_indices: torch.Tensor | None

    @classmethod
    def from_batch(
        cls, batch: ScheduleBatch, num_verify_rows: int, padded_bs: int
    ) -> TpuSpecVerifyInput:
        """Convert the algorithm's batch.spec_info; raise for types without a TPU window."""
        draft_input = batch.spec_info
        if not isinstance(draft_input, HostPaddedDraftInput):
            raise NotImplementedError(
                f"No TpuSpecVerifyInput conversion from {type(draft_input)}."
            )
        candidates = draft_input.tpu_verify_cache_candidates
        committed_lens = draft_input.tpu_committed_lens
        req_pool_indices = draft_input.tpu_req_pool_indices
        if candidates is None or committed_lens is None or req_pool_indices is None:
            raise NotImplementedError(
                "TpuSpecWorker needs the verify window (overlap schedule, "
                "host-resident req_to_token, no grammar/logprob/LoRA)."
            )
        req_pool_indices_cpu = batch.req_pool_indices_cpu
        seq_lens_cpu = batch.seq_lens_cpu
        if not isinstance(req_pool_indices_cpu, torch.Tensor) or not isinstance(
            seq_lens_cpu, torch.Tensor
        ):
            raise RuntimeError(
                "TpuSpecWorker needs the host pool-row and length mirrors."
            )
        assert req_pool_indices_cpu.device.type == "cpu", req_pool_indices_cpu.device
        assert seq_lens_cpu.device.type == "cpu", seq_lens_cpu.device
        assert req_pool_indices.device.type == "cpu", req_pool_indices.device
        # With the window the host mirror holds the committed base, one unread
        # commit (<= N + 1 tokens) behind the device length. The plan stays on
        # host until the forward fixes its bucket; 0 is the reserved row and slot.
        return cls(
            verify_cache_candidates=padded_h2d(candidates, padded_bs, batch.device),
            committed_lens=padded_h2d(committed_lens, padded_bs, batch.device),
            req_pool_indices=padded_h2d(req_pool_indices, padded_bs, batch.device),
            req_pool_indices_cpu=pad_rows(req_pool_indices_cpu, padded_bs),
            seq_lens_cpu=seq_lens_cpu,
            host_len_slack=(
                num_verify_rows if draft_input.tpu_host_seq_lens_is_committed else 0
            ),
            future_indices=draft_input.future_indices,
        )

    @classmethod
    def idle(
        cls, *, num_verify_rows: int, padded_bs: int, device: str
    ) -> TpuSpecVerifyInput:
        """All-padding addressing for a DP-attention rank with no requests.

        Every row holds pool row 0, slot 0 and host length 0, so RPA runs none.
        """
        zeros = torch.zeros((padded_bs,), dtype=torch.int64)
        return cls(
            verify_cache_candidates=torch.zeros(
                (padded_bs, 2 * num_verify_rows), dtype=torch.int64
            ).to(device, non_blocking=True),
            committed_lens=zeros.to(device, non_blocking=True),
            req_pool_indices=zeros.to(device, non_blocking=True),
            req_pool_indices_cpu=zeros,
            seq_lens_cpu=torch.zeros((0,), dtype=torch.int64),
            host_len_slack=0,
            future_indices=None,
        )

    def kv_lens_after_rows(self, num_rows: int) -> tuple[torch.Tensor, int]:
        """Host [bs] KV length bound once num_rows rows past seq_lens are written."""
        kv_lens_cpu = self.seq_lens_cpu + num_rows + self.host_len_slack
        return kv_lens_cpu, int(kv_lens_cpu.sum())

    def padded_kv_lens_after_rows(self, num_rows: int) -> tuple[torch.Tensor, int]:
        """kv_lens_after_rows on [pbs] rows; 0 on padded rows keeps them inactive in RPA."""
        kv_lens_cpu, kv_lens_sum = self.kv_lens_after_rows(num_rows)
        return pad_rows(kv_lens_cpu, self.req_pool_indices_cpu.shape[0]), kv_lens_sum
