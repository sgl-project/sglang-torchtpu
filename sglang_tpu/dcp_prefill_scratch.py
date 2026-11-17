"""Slot bookkeeping of the sparse MLA DCP prefill scratch (host side).

A slot is an unsharded copy of one request's latents and index keys that every
rank of a DCP group keeps while the request is in prefill
(SGLANG_TPU_DCP_PREFILL_SCRATCH).
"""

from __future__ import annotations

from typing import Optional, Sequence

# Per slot: (request id, tokens stored), or None when free.
SlotState = list[Optional[tuple[str, int]]]


def plan_scratch_slots(
    state: SlotState,
    *,
    rids: Sequence[str],
    prefix_lens: Sequence[int],
    seq_lens: Sequence[int],
) -> Optional[list[int]]:
    """Scratch slot of each prefill row, or None when the batch must gather.

    A row keeps its slot while it continues the request stored there (its
    prefix lies within the tokens stored) and takes a free slot when it starts
    at token 0; a slot is free once its request is absent from a prefill batch.
    Anything else (a radix prefix hit, more rows than slots) returns None and
    drops the batch's slots, so those requests gather from then on. ``state``
    is updated in place.
    """
    live = set(rids)
    slots: list[int] = []
    for rid, prefix in zip(rids, prefix_lens):
        slot = next((s for s, held in enumerate(state) if held and held[0] == rid), None)
        if slot is None and prefix == 0:
            slot = next(
                (
                    s
                    for s, held in enumerate(state)
                    if s not in slots and (held is None or held[0] not in live)
                ),
                None,
            )
        held = None if slot is None else state[slot]
        if slot is None or (held and held[0] == rid and prefix > held[1]):
            for s, other in enumerate(state):
                if other and other[0] in live:
                    state[s] = None
            return None
        slots.append(slot)
    for slot, rid, seq_len in zip(slots, rids, seq_lens):
        state[slot] = (rid, int(seq_len))
    return slots
