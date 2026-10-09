"""Build a host-padded EXTEND forward's metadata at the extend bucket.

When the target compile policy registers an extend bucket
(``model_executor/host_token_padding.py``), the prefill ``input_ids`` arrive
already padded to it. ``ForwardBatch.init_new`` would still upload the rest of
the extend metadata at the real token and request counts, and the compiled
forward shim would then pad it on device: one program per prompt length.

``ForwardBatch.init_new`` and ``compute_position_torch`` are patched (see
``patches.py``) to build that metadata from the host lists instead:

- positions and ``extend_start_loc`` come from the host length lists, padded,
  so the position patch needs no ``.tolist()`` device read;
- ``extend_seq/prefix_lens``, ``seq_lens``, ``req_pool_indices``,
  ``orig_seq_lens`` and ``out_cache_loc`` are uploaded padded from their host
  mirrors;
- the non-padded token count is the real one, not the padded input length.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.model_executor.host_token_padding import (
    extend_batch_bucket,
    extend_token_bucket,
    pad_host_list,
    pad_host_tokens,
)
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass(frozen=True)
class ExtendPadPlan:
    """Host lengths and bucket sizes of one host-padded EXTEND forward."""

    prefix_lens: list[int]
    seq_lens: list[int]
    num_tokens: int
    padded_num_tokens: int | None
    padded_batch_size: int | None


# Set by the patched ForwardBatch.init_new for the duration of the original
# call, so the patched compute_position_torch can read the host lists.
_current_plan: ContextVar[ExtendPadPlan | None] = ContextVar(
    "tpu_extend_pad_plan", default=None
)


def plan_extend_padding(batch: ScheduleBatch) -> ExtendPadPlan | None:
    """The padding a plain EXTEND batch gets, or None to leave it unpadded."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    seq_lens = batch.extend_lens
    prefix_lens = batch.prefix_lens
    if (
        batch.forward_mode != ForwardMode.EXTEND
        or not isinstance(seq_lens, list)
        or not isinstance(prefix_lens, list)
        or not seq_lens
        or not batch.extend_num_tokens
    ):
        return None
    num_reqs = len(seq_lens)
    padded_num_tokens = extend_token_bucket(batch.extend_num_tokens, num_reqs)
    padded_batch_size = extend_batch_bucket(num_reqs)
    if padded_num_tokens is None and padded_batch_size is None:
        return None
    return ExtendPadPlan(
        prefix_lens=list(prefix_lens),
        seq_lens=list(seq_lens),
        num_tokens=batch.extend_num_tokens,
        padded_num_tokens=padded_num_tokens,
        padded_batch_size=padded_batch_size,
    )


def current_extend_padding() -> ExtendPadPlan | None:
    return _current_plan.get()


def set_extend_padding(plan: ExtendPadPlan | None):
    """Make ``plan`` current; returns the token to pass to ``reset``."""
    return _current_plan.set(plan)


def reset_extend_padding(token) -> None:
    _current_plan.reset(token)


def host_positions(
    prefix_lens: list[int],
    seq_lens: list[int],
    padded_num_tokens: int | None,
    padded_batch_size: int | None,
    start_loc_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions and extend_start_loc, built on the host and zero-padded."""
    positions = torch.cat(
        [torch.arange(p, p + s) for p, s in zip(prefix_lens, seq_lens)], axis=0
    ).to(torch.int64)
    start_loc = [0]
    for s in seq_lens[:-1]:
        start_loc.append(start_loc[-1] + s)
    return (
        pad_host_tokens(positions, padded_num_tokens),
        torch.tensor(
            pad_host_list(start_loc, padded_batch_size), dtype=start_loc_dtype
        ),
    )


def apply_extend_padding(
    forward_batch: ForwardBatch, batch: ScheduleBatch, plan: ExtendPadPlan
) -> None:
    """Re-upload the real-sized extend fields padded from their host mirrors.

    Host-to-device copies compile nothing, unlike the device pads they replace.
    """
    device = forward_batch.input_ids.device
    pin_memory = is_pin_memory_available(device)

    def upload(values: torch.Tensor) -> torch.Tensor:
        return values.to(device, non_blocking=True)

    forward_batch.global_num_token_non_padded_cpu = plan.num_tokens
    if forward_batch.global_num_token_non_padded is not None:
        forward_batch.global_num_token_non_padded = upload(
            torch.tensor(plan.num_tokens, dtype=torch.int32, pin_memory=pin_memory)
        )

    out_cache_loc_cpu = batch.out_cache_loc_cpu
    if (
        plan.padded_num_tokens is not None
        and out_cache_loc_cpu is not None
        and out_cache_loc_cpu.shape[0] == plan.num_tokens
    ):
        # 0 is the reserved padding page.
        forward_batch.out_cache_loc = upload(
            pad_host_tokens(
                out_cache_loc_cpu.to(batch.out_cache_loc.dtype),
                plan.padded_num_tokens,
            )
        )

    padded_bs = plan.padded_batch_size
    if padded_bs is None:
        return
    for field, values in (
        ("extend_seq_lens", plan.seq_lens),
        ("extend_prefix_lens", plan.prefix_lens),
    ):
        setattr(
            forward_batch,
            field,
            upload(
                torch.tensor(
                    pad_host_list(values, padded_bs),
                    dtype=torch.int32,
                    pin_memory=pin_memory,
                )
            ),
        )
    if batch.seq_lens_cpu is not None:
        forward_batch.seq_lens = upload(
            pad_host_tokens(batch.seq_lens_cpu.to(batch.seq_lens.dtype), padded_bs)
        )
    if batch.req_pool_indices_cpu is not None:
        forward_batch.req_pool_indices = upload(
            pad_host_tokens(
                batch.req_pool_indices_cpu.to(batch.req_pool_indices.dtype),
                padded_bs,
            )
        )
    if (
        batch.orig_seq_lens_cpu is not None
        and len(batch.orig_seq_lens_cpu) == forward_batch.batch_size
    ):
        forward_batch.orig_seq_lens = upload(
            torch.tensor(
                pad_host_list(batch.orig_seq_lens_cpu, padded_bs),
                dtype=batch.orig_seq_lens.dtype,
                pin_memory=pin_memory,
            )
        )
