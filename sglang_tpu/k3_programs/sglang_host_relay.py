# SPDX-License-Identifier: Apache-2.0
"""sglang's per-step decode bookkeeping on the host under the K3 programs.

Under overlap scheduling sglang relays each decode step's results to the next
step on the device. After the program's draft op the worker counts the accept
lengths, gathers each request's last token and adds the new lengths (eight
small ops), the future map stores the lengths and the last tokens (two index
copies), the result converts the tokens and the last tokens to int64 (two
copies), and the next step gathers both stores back (two gathers; a torch_tpu
trace shows an index_put_ on the same buffer shape before each). Each is its
own device program of a few microseconds plus a launch gap, between the draft
op and the next verify: about 0.16 ms of a 9.2 ms single-request step on
tpu7x.

None of them feeds the programs. The decode program keeps each request's next
input on the device and tracks its position itself, so the relayed lengths
and last tokens have no reader (the next step's lengths feed only the worker's
next sum). The scheduler's own bookkeeping -- committed lengths, KV
allocation, finish detection, streaming, the prefix cache -- reads the host
copy of each step's tokens and accept lengths.

With ``SGLANG_TPU_K3_PROGRAM_OPT_HOST_BOOKKEEPING`` the K3 algorithm's future
map is :class:`HostRelayFutureMap`, which stores and resolves nothing, and
the decode worker returns the program's token rows as they are
(:class:`HostRowsResult`): the scheduler copies them out (one transfer
instead of two) and the flat tokens and accept lengths its output processor
reads are built from the copy on the host. No device op runs between the
program's draft op and its next step.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.utils.common import async_d2h


def accept_counts(rows: np.ndarray) -> np.ndarray:
    """Each ``-1`` padded token row's token count (bonus included) as int32:
    the count the device relay took with ``(rows >= 0).sum(1)``."""
    return np.count_nonzero(np.asarray(rows) >= 0, axis=1).astype(np.int32)


class HostRelayFutureMap(FutureMap):
    """sglang's future map with nothing relayed on the device: the K3
    programs read none of it (module docstring). A resolve keeps sglang's
    host-side effects; the batch's device lengths stay as the batch was
    formed (only the device relay's own worker sum read them)."""

    def publish(
        self,
        future_indices: torch.Tensor,
        new_seq_lens: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        return None

    def stash(self, future_indices: torch.Tensor, payload: Any) -> None:
        return None

    def stash_bonus_tokens(
        self, indices: torch.Tensor, bonus_tokens: torch.Tensor
    ) -> None:
        return None

    def resolve_seq_lens_cpu(self, batch: Any) -> None:
        spec = batch.spec_info
        if spec is None or spec.future_indices is None:
            return
        # As sglang's resolve without a host mirror (needs_cpu_seq_lens off,
        # create_future_map's condition): the scheduler sums lazily.
        batch.seq_lens_cpu = None
        batch.seq_lens_sum = None

    def _resolve_spec_extras(self, batch: Any) -> None:
        spec = batch.spec_info
        if spec is not None:
            spec.dsa_topk_indices = None

    def resolve_mixed_spec_tails(self, batch: Any) -> None:
        raise NotImplementedError(
            "K3_PROGRAM: mixed prefill/decode batches read decode tokens the "
            "host-bookkeeping relay does not store"
        )


class HostRowsResult(GenerationBatchResult):
    """A decode step's result that keeps the program's ``[>= requests,
    WIDTH]`` int32 token rows (-1 padded) as they are: ``copy_to_cpu`` copies
    them out, and the flat ``[requests * WIDTH]`` tokens and ``[requests]``
    accept lengths sglang's output processor reads are built from the copy on
    the host when first read (every reader waits on ``copy_done`` first).
    Until the copy, ``next_token_ids`` is the device rows and
    ``accept_lens`` None; after it both read the copy."""

    def __init__(self, rows: torch.Tensor, requests: int, **kwargs: Any) -> None:
        self._requests = requests
        self._host_rows: Optional[torch.Tensor] = None
        self._host_fields: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        super().__init__(next_token_ids=rows, **kwargs)

    def _built(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self._host_rows is None:
            return None
        if self._host_fields is None:
            rows = self._host_rows.numpy()[: self._requests]
            self._host_fields = (
                torch.from_numpy(rows.reshape(-1).astype(np.int64)),
                torch.from_numpy(accept_counts(rows)),
            )
        return self._host_fields

    @property
    def next_token_ids(self) -> Optional[torch.Tensor]:
        built = self._built()
        return built[0] if built is not None else self.__dict__.get("_device_rows")

    @next_token_ids.setter
    def next_token_ids(self, value: torch.Tensor) -> None:
        self.__dict__["_device_rows"] = value

    @property
    def accept_lens(self) -> Optional[torch.Tensor]:
        built = self._built()
        return built[1] if built is not None else self.__dict__.get("_accept_lens")

    @accept_lens.setter
    def accept_lens(self, value: Optional[torch.Tensor]) -> None:
        self.__dict__["_accept_lens"] = value

    @property
    def has_sampled_token_ids(self) -> bool:
        # Without reading the fields (built once, from the landed copy).
        return True

    @torch.profiler.record_function("copy_result_to_cpu")
    def copy_to_cpu(self, return_logprob: bool, return_hidden_states: bool = True):
        if return_logprob or self.logits_output is not None:
            raise NotImplementedError("K3_PROGRAM results carry tokens only")
        # As sglang's copy, the result keeps the host copy only.
        self._host_rows = async_d2h(self.__dict__.pop("_device_rows"))
        self.copy_done.record()
