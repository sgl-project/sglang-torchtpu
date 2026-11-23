# SPDX-License-Identifier: Apache-2.0
"""Host planning for DeepSeek-V4.1 decoder SWA bounded replay."""

from typing import NamedTuple


class DecoderReplayPlan(NamedTuple):
    """The intersection of each extend with its prompt's decoder window."""

    row_indices: tuple[int, ...]
    lengths: tuple[int, ...]
    swa_floors: tuple[int, ...]


def plan_decoder_replay(
    *,
    prefix_lengths: tuple[int, ...],
    extend_lengths: tuple[int, ...],
    prompt_ends: tuple[int, ...],
    whole_extend: tuple[bool, ...],
    window: int,
) -> DecoderReplayPlan:
    """Plan against folded prefixes so historical replay rows can rebuild SWA.

    For requests that read every row, keep the ordinary SWA history. A trimmed
    request may read ring entries only at or above its prompt-window start.
    """
    if window <= 0:
        raise ValueError("The decoder SWA window must be positive")
    indices, lengths, floors = [], [], []
    offset = 0
    for prefix, count, prompt_end, whole in zip(
        prefix_lengths, extend_lengths, prompt_ends, whole_extend, strict=True
    ):
        if prefix < 0 or count < 0 or prompt_end < 0:
            raise ValueError("Decoder replay lengths must be nonnegative")
        floor = decoder_swa_floor(prompt_end, whole_extend=whole, window=window)
        skip = min(count, max(0, floor - prefix))
        indices.extend(range(offset + skip, offset + count))
        lengths.append(count - skip)
        floors.append(floor)
        offset += count
    return DecoderReplayPlan(tuple(indices), tuple(lengths), tuple(floors))


def decoder_prompt_bounds(
    batch, *, keep_hidden: bool
) -> tuple[tuple[int, ...], tuple[bool, ...]]:
    # Keep the ordinary decoder history for every chunk of a request with a
    # prompt reader, including chunks before its first requested logprob.
    return (
        tuple(
            len(req.full_untruncated_fill_ids)
            if req is batch.chunked_req
            else req.extend_end
            for req in batch.reqs
        ),
        tuple(
            keep_hidden or req.return_logprob or req.return_hidden_states_mode.is_full()
            for req in batch.reqs
        ),
    )


def decoder_swa_floor(prompt_end: int, *, whole_extend: bool, window: int) -> int:
    return 0 if whole_extend else max(0, prompt_end - window)
