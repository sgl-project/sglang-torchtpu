"""Role-specific protocol state for the Raiden disaggregation backend.

These records own no locks. Access remains protected by the manager's existing
locks. Native I/O leases, abort tombstones, and completion tracking stay on the
manager because they can survive protocol-state cleanup.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple


@dataclass
class PrefillState:
    """Prefill protocol bookkeeping for one manager."""

    announce_pages: int

    # Accumulated page ids and aux data, keyed by room.
    send_blocks: Dict[int, List[int]] = field(default_factory=dict)
    # The pages of send_blocks, for the repeated-page check.
    send_block_set: Dict[int, Set[int]] = field(default_factory=dict)
    send_swa_blocks: Dict[int, List[int]] = field(default_factory=dict)
    # Committed state row/pages of the room, on the state engine.
    send_state_rows: Dict[int, List[int]] = field(default_factory=dict)
    send_state_meta: Dict[int, bytes] = field(default_factory=dict)
    send_aux: Dict[int, bytes] = field(default_factory=dict)

    # Decode reply address and the page counts it allocated.
    decode_addr: Dict[int, Tuple[str, int]] = field(default_factory=dict)
    replied: Set[int] = field(default_factory=set)

    # Chunks held until a pull or reshard request identifies the peer.
    held_chunks: Dict[int, List[Tuple[int, List[int]]]] = field(default_factory=dict)
    send_total: Dict[int, int] = field(default_factory=dict)
    decode_pages: Dict[int, int] = field(default_factory=dict)
    decode_swa_pages: Dict[int, int] = field(default_factory=dict)

    # Announced reshard chunks retained until the decode requests their push.
    reshard_chunks: Dict[int, Dict[int, Tuple[int, List[int]]]] = field(
        default_factory=dict
    )

    # Pages requested by RESHARD_READY for the closing push.
    reshard_ready_pages: Dict[int, List[int]] = field(default_factory=dict)

    # Rooms whose closing reshard push has completed.
    reshard_final_pushed: Set[int] = field(default_factory=set)

    # Committed pages waiting to reach the chunk announcement threshold.
    pending_offer: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)
    chunk_seq: Dict[int, int] = field(default_factory=dict)

    # Decode metadata received in RESHARD_REQ.
    reshard_req: Dict[int, dict] = field(default_factory=dict)

    # Rooms whose send accumulation is complete.
    reshard_finalized: Set[int] = field(default_factory=set)

    # Only acknowledged chunks leave the closing transfer.
    reshard_chunk_acked: Dict[int, Set[int]] = field(default_factory=dict)

    # Rooms whose RESHARD_BLOCKS reply has been sent.
    reshard_replied: Set[int] = field(default_factory=set)

    # Rooms cleared by the decode to push.
    reshard_ready: Set[int] = field(default_factory=set)

    # Time of the RESHARD_BLOCKS reply, used to detect decode silence.
    reshard_replied_at: Dict[int, float] = field(default_factory=dict)
    reshard_pushed: Set[int] = field(default_factory=set)

    # Decode-held prefix pull (prefix_pull.py). When the decode's PULL_REQ
    # promised an offer; offers received (empty blocks withdraw) and handed to
    # the puller; the range asked for; READY replies; the read in flight; and
    # rooms whose pull is settled either way.
    prefix_offer_since: Dict[int, float] = field(default_factory=dict)
    prefix_offers: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)
    prefix_offer_taken: Set[int] = field(default_factory=set)
    prefix_pull_asked: Dict[int, Tuple[int, int]] = field(default_factory=dict)
    prefix_ready: Dict[int, Tuple[str, int, str, int, int]] = field(
        default_factory=dict
    )
    prefix_pull_reads: Dict[int, str] = field(default_factory=dict)
    prefix_resolved: Set[int] = field(default_factory=set)
    # Host offers/registrations use the HiCache managers, not device staging.
    prefix_host_offers: Set[int] = field(default_factory=set)
    prefix_host_ready: Set[int] = field(default_factory=set)
    # Hybrid sliding-window models: the offer's (swa_first_page, window page
    # ids) and READY's (swa_req_id, swa_uuid, swa_endpoint, first, pages).
    prefix_swa_offers: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)
    prefix_swa_ready: Dict[int, Tuple[str, int, str, int, int]] = field(
        default_factory=dict
    )
    # Victim push: offers awaiting the decode, and its answers, by push seq.
    victim_open: Set[int] = field(default_factory=set)
    victim_acks: Dict[int, Any] = field(default_factory=dict)


@dataclass
class DecodeState:
    """Decode protocol bookkeeping for one manager."""

    # Local and remote page ids, endpoints, and aux slots, keyed by room.
    local_blocks: Dict[int, List[int]] = field(default_factory=dict)
    local_swa_blocks: Dict[int, List[int]] = field(default_factory=dict)
    local_state_rows: Dict[int, List[int]] = field(default_factory=dict)
    decode_aux_index: Dict[int, int] = field(default_factory=dict)
    remote_blocks: Dict[int, List[int]] = field(default_factory=dict)
    remote_swa_blocks: Dict[int, List[int]] = field(default_factory=dict)
    remote_state_rows: Dict[int, List[int]] = field(default_factory=dict)
    remote_endpoint: Dict[int, str] = field(default_factory=dict)
    remote_swa_endpoint: Dict[int, str] = field(default_factory=dict)
    remote_state_endpoint: Dict[int, str] = field(default_factory=dict)

    # Announcement time by read id; expired reads must never be issued.
    announced_at: Dict[str, float] = field(default_factory=dict)

    # Latest announcement time by registered room; restarts its waiting deadline.
    last_announced_at: Dict[int, float] = field(default_factory=dict)

    # Announced chunks waiting for admission and reads issued for each room.
    chunk_queue: Dict[int, List[tuple]] = field(default_factory=dict)
    remote_total: Dict[int, int] = field(default_factory=dict)

    # Announced page spans may have holes when chunks cross on the wire.
    announced_spans: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    chunk_reads: Dict[int, List[str]] = field(default_factory=dict)
    final_seen: Set[int] = field(default_factory=set)

    # Chunk announcements from each prefill source, by room and chunk.
    reshard_chunk_sources: Dict[int, Dict[int, Dict[int, Tuple[int, List[int]]]]] = (
        field(default_factory=dict)
    )

    # In-flight chunk registrations that the closing plan must wait for.
    reshard_chunk_registering: Dict[int, Set[int]] = field(default_factory=dict)

    # Registered chunk spans; the closing plan covers their complement.
    reshard_covered: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)

    # Per-room block ids from each source rank, for full and SWA KV.
    reshard_sources: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)
    reshard_swa_sources: Dict[int, Dict[int, List[int]]] = field(default_factory=dict)
    reshard_meta: Dict[int, dict] = field(default_factory=dict)
    reshard_registered: Set[int] = field(default_factory=set)

    # Decode-held prefix pull: tokens promised in PULL_REQ, rooms whose offer
    # is still owed, the offered page ids, and the prefill rank to answer.
    prefix_tokens: Dict[int, int] = field(default_factory=dict)
    prefix_offer_pending: Set[int] = field(default_factory=set)
    prefix_blocks: Dict[int, List[int]] = field(default_factory=dict)
    prefill_addr: Dict[int, str] = field(default_factory=dict)
    # Hybrid sliding-window models: (swa_first_page, window page ids) offered.
    prefix_swa_blocks: Dict[int, Tuple[int, List[int]]] = field(default_factory=dict)
    prefix_host_offers: Set[int] = field(default_factory=set)
    # Victim push: offers and done notices received and not yet taken,
    # by (prefill, seq).
    victim_pushes: Dict[Tuple[str, int], Any] = field(default_factory=dict)
    victim_dones: Dict[Tuple[str, int], Any] = field(default_factory=dict)
