"""Slot bookkeeping of the sparse MLA DCP prefill scratch (host side).

A slot is an unsharded copy of one request's latents and index keys that every
rank of a DCP group keeps while the request is in prefill
(SGLANG_TPU_DCP_PREFILL_SCRATCH).
"""

from __future__ import annotations

from typing import Optional, Sequence

# Per slot: (request/attempt ownership token, tokens stored), or None when free.
# Retaining this small token prevents its identity from being recycled while
# the slot still contains that request's history; it must not retain the Req.
SlotState = list[Optional[tuple[object, int]]]


def plan_scratch_slots(
    state: SlotState,
    *,
    owners: Sequence[object],
    prefix_lens: Sequence[int],
    seq_lens: Sequence[int],
) -> Optional[list[int]]:
    """Scratch slot of each prefill row, or None when the batch must gather.

    A row keeps its slot while it continues the request stored there (its
    prefix lies within the tokens stored) and takes a free slot when it starts
    at token 0; a slot is free once its request is absent from a prefill batch.
    Anything else (a radix prefix hit, more rows than slots) returns None and
    drops the batch's slots, so those requests gather from then on. ``state``
    is updated in place. Owners are compared by identity: two completed and
    new requests can have equal public IDs and equal-valued cache handles.
    """
    live = {id(owner) for owner in owners}
    slots: list[int] = []
    for owner, prefix in zip(owners, prefix_lens):
        slot = next(
            (s for s, held in enumerate(state) if held and held[0] is owner), None
        )
        if slot is None and prefix == 0:
            slot = next(
                (
                    s
                    for s, held in enumerate(state)
                    if s not in slots and (held is None or id(held[0]) not in live)
                ),
                None,
            )
        held = None if slot is None else state[slot]
        if slot is None or (held and held[0] is owner and prefix > held[1]):
            for s, other in enumerate(state):
                if other and id(other[0]) in live:
                    state[s] = None
            return None
        slots.append(slot)
    for slot, owner, seq_len in zip(slots, owners, seq_lens):
        state[slot] = (owner, int(seq_len))
    return slots
