# SPDX-License-Identifier: Apache-2.0
"""Host twin of a spec step's verify rows, so host-built attention metadata reads no device value.

On TorchTPU a device read waits for every program dispatched before it. The
verify's attention metadata is built after the draft is dispatched, so its reads
of the verify positions, KV slots and pool rows held the host until the draft
finished, and the rest of the verify preparation then ran with the device idle.
The rows only need the step's device seq_lens, which the previous verify's
accept produced and the draft's preparation waits for anyway: one read there
gives the host rows of both the draft and the verify forward.
"""

from typing import NamedTuple

import torch


class HostVerifyRows(NamedTuple):
    """Host int64 copies of the verify forward's addressing at the bucket.

    positions: [pbs * (N + 1)], as ForwardBatch.positions.
    out_cache_loc: [pbs * (N + 1)], as VerifyRows.flat_cache_locs.
    req_pool_indices: [pbs], as the bucket state's pool rows.
    """

    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    req_pool_indices: torch.Tensor


def _pad_rows(rows: torch.Tensor, num_rows: int) -> torch.Tensor:
    out = torch.zeros((num_rows, *rows.shape[1:]), dtype=torch.int64)
    out[: rows.shape[0]] = rows.to(torch.int64)
    return out


def host_verify_rows(
    *,
    seq_lens: torch.Tensor,
    candidates_cpu: torch.Tensor,
    committed_lens_cpu: torch.Tensor,
    req_pool_indices_cpu: torch.Tensor,
    num_rows: int,
) -> HostVerifyRows:
    """select_verify_rows on the host: the same positions and slots, bucket rows included.

    seq_lens is the [pbs] device seq_lens' host copy; candidates_cpu [bs, 2 * cap]
    and committed_lens_cpu [bs] hold the live rows, zero-padded to pbs as their
    device uploads are.
    """
    padded_bs = seq_lens.shape[0]
    seq_lens = seq_lens.to(torch.int64)
    candidates = _pad_rows(candidates_cpu, padded_bs)
    committed = _pad_rows(committed_lens_cpu, padded_bs)
    offsets = (seq_lens - committed).clamp_(0, candidates.shape[1] // 2)
    steps = torch.arange(num_rows, dtype=torch.int64)
    columns = offsets[:, None] + steps
    rows = torch.arange(padded_bs, dtype=torch.int64)
    cache_locs = candidates[rows[:, None], columns]
    positions = seq_lens[:, None] + steps
    return HostVerifyRows(
        positions=positions.reshape(-1),
        out_cache_loc=cache_locs.reshape(-1),
        req_pool_indices=_pad_rows(req_pool_indices_cpu, padded_bs),
    )


def record_host_rows(forward_batch, rows: HostVerifyRows | None) -> None:
    """Hand the rows to host-built attention metadata as the batch's host copies.

    host_input_mirror answers only while the batch still holds these device
    tensors, so a later rebind of a field falls back to the device read.
    """
    if rows is None:
        return
    from sglang_tpu.compilation import host_input_mirror

    for name, host in (
        ("positions", rows.positions),
        ("out_cache_loc", rows.out_cache_loc),
        ("req_pool_indices", rows.req_pool_indices),
    ):
        host_input_mirror.record_upload(
            forward_batch, name, host, getattr(forward_batch, name)
        )


class StepHostRows:
    """One spec step's host rows, from one read of its device seq_lens on first use.

    The draft forward asks first, right before its own metadata build, which
    would read the device anyway (the read waits for the previous verify); the
    verify reuses the rows. seq_lens_cpu skips the read (idle DP ranks: zeros).
    """

    def __init__(
        self,
        *,
        seq_lens: torch.Tensor,
        verify_input,
        num_rows: int,
        seq_lens_cpu: torch.Tensor | None = None,
    ) -> None:
        self._seq_lens = seq_lens
        self._seq_lens_cpu = seq_lens_cpu
        self._verify_input = verify_input
        self._num_rows = num_rows
        self._rows: HostVerifyRows | None = None
        self._built = False

    def verify_rows(self) -> HostVerifyRows | None:
        if not self._built:
            self._built = True
            self._rows = self._build()
        return self._rows

    def draft_rows(self, width: int) -> HostVerifyRows | None:
        """The first width verify rows of each request, as the draft block runs them."""
        rows = self.verify_rows()
        if rows is None:
            return None
        padded_bs = rows.req_pool_indices.shape[0]
        return HostVerifyRows(
            positions=rows.positions.view(padded_bs, -1)[:, :width].reshape(-1),
            out_cache_loc=rows.out_cache_loc.view(padded_bs, -1)[:, :width].reshape(-1),
            req_pool_indices=rows.req_pool_indices,
        )

    def _build(self) -> HostVerifyRows | None:
        candidates_cpu = self._verify_input.verify_cache_candidates_cpu
        committed_lens_cpu = self._verify_input.committed_lens_cpu
        if candidates_cpu is None or committed_lens_cpu is None:
            return None
        seq_lens_cpu = self._seq_lens_cpu
        if seq_lens_cpu is None:
            seq_lens_cpu = self._seq_lens.to("cpu")
        return host_verify_rows(
            seq_lens=seq_lens_cpu,
            candidates_cpu=candidates_cpu,
            committed_lens_cpu=committed_lens_cpu,
            req_pool_indices_cpu=self._verify_input.req_pool_indices_cpu,
            num_rows=self._num_rows,
        )
