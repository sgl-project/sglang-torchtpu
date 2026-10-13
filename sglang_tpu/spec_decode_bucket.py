"""Run an EAGLE3 decode step at its batch bucket from scheduling to results.

Every eager op on TPU compiles once per input shape. The draft, verify and
draft-extend forwards already run at the batch bucket, but the bookkeeping
between them (FutureMap gathers and stores, cache-window selection, the pads
into each forward and the trims out of it) ran at the real batch size, so each
new running-request count compiled a few dozen programs while serving.

Instead, the decode prepare picks the bucket on the host and uploads padded
copies of the host-built inputs: the request rows, the host length shadow and
the candidate cache window (or, with no acceptance pending, the window built
from the lengths the resolve reads back anyway). Padding rows use request slot
0 and KV page 0, the same rows every other padded forward writes to. FutureMap
rows are read and written through the padded rows, nothing is trimmed between
phases, and the scheduler drops the padding rows when it reads the results on
the host.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

# num_reqs -> decode batch bucket, or None to keep the real batch size.
_decode_batch_bucket: Callable[[int], int | None] | None = None


def register_decode_batch_bucket(bucket: Callable[[int], int | None]) -> None:
    global _decode_batch_bucket
    _decode_batch_bucket = bucket


def decode_batch_bucket(num_reqs: int) -> int | None:
    if _decode_batch_bucket is None:
        return None
    return _decode_batch_bucket(num_reqs)


@dataclass
class DecodeBucket:
    """Bucket-sized inputs of one EAGLE3 decode step.

    Built by the decode prepare. ``seq_lens`` is filled when the FutureMap
    resolves the step, and the worker swaps all of them into the batch for the
    forward (the scheduler restores its own fields afterwards).
    """

    logical_bs: int
    rows: torch.Tensor
    rows_cpu: torch.Tensor
    # 1 for real rows, 0 for padding rows.
    valid: torch.Tensor
    # Host lengths, 0 for padding rows: the host shadow under device accept
    # routing, else read back when the FutureMap resolves the step.
    seq_lens_cpu: torch.Tensor | None = None
    seq_lens: torch.Tensor | None = None

    @property
    def size(self) -> int:
        return int(self.rows_cpu.shape[0])


def pad_host_rows(t: torch.Tensor, size: int) -> torch.Tensor:
    """Zero-pad a host tensor along dim 0."""
    if t.shape[0] == size:
        return t
    padded = t.new_zeros((size, *t.shape[1:]))
    padded[: t.shape[0]] = t
    return padded


def bucket_eligible(batch) -> bool:
    """Whether every consumer of this decode step tolerates padding rows."""
    sampling_info = batch.sampling_info
    return bool(
        sampling_info is not None
        # Non-greedy verify reads per-request sampling tensors at the real size.
        and sampling_info.is_all_greedy
        and not (
            sampling_info.penalizer_orchestrator is not None
            and sampling_info.penalizer_orchestrator.is_required
        )
        and sampling_info.logit_bias is None
        and not sampling_info.has_custom_logit_processor
        and not batch.return_hidden_states
    )


def enter_decode_bucket(batch) -> None:
    """Swap a resolved decode bucket into the batch for this forward.

    The scheduler snapshots and restores the batch fields around the forward,
    so the swap does not reach scheduler-side bookkeeping.
    """
    draft_input = batch.spec_info
    bucket = getattr(draft_input, "decode_bucket", None)
    if bucket is None or bucket.seq_lens is None:
        return
    assert bucket.seq_lens_cpu is not None
    batch.req_pool_indices = bucket.rows
    batch.req_pool_indices_cpu = bucket.rows_cpu
    batch.seq_lens = bucket.seq_lens
    batch.seq_lens_cpu = bucket.seq_lens_cpu
    # Read by no TPU decode forward (the compiled model ignores it); padding
    # its real-sized rows would compile once per batch size.
    batch.orig_seq_lens = None
    # Per-request M-RoPE deltas are read for every row; padding rows have none.
    if batch.multimodal_inputs is not None:
        pad = bucket.size - len(batch.multimodal_inputs)
        batch.multimodal_inputs = list(batch.multimodal_inputs) + [None] * pad
