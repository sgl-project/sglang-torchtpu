"""A second pool family of a pool transfer through a second native engine.

tpu-sync's KVCacheManager requires every storage it wraps to hold the same
number of blocks, because they share one block-id space. A pool family whose
block count differs from the full-attention pages' therefore fails at startup
in one engine with them. As the non-pipeline Raiden path does, each worker
wraps such a family in a second engine instead. There are two families, and a
worker carries one at most: the sliding window, a hybrid pool's sliding-window
layers, which hold far fewer pages than its full-attention layers; and
linear-attention (GDN/KDA) state, one row per request.

Terms used here and at the call sites in pool_transfer.py:

- family (Family; SwaFamily, StateFamily): what the second engine carries, and
  how its blocks are sized, sliced into spans, placed on the destination and
  announced.
- link (SideLink): one worker's second engine, with its controller, its work
  unit and the state of its twins.
- twin: the native transfer ``swa:<req_id>`` or ``state:<req_id>`` that moves
  the family's blocks (the window pages, the state row) of a closing transfer
  or a stage group between the second engines; early parts carry none and have
  no twin. A stage registers the twin
  together with the transfer's full-attention pages and announces both in one
  READY; decode rank 0 coordinates the twin on threads of its own,
  concurrently with the full-attention transfer.
- half: either transfer of such a pair, the full-attention one or its twin.
- book (TwinBook): reports a transfer settled only once both of its halves
  settled natively, and as failed if either failed, so neither side releases
  pages the other half may still move.
- declare: a decode rank telling the book whether a transfer it accepted has
  a twin. Rank 0 coordinates for every attention-TP rank, so a half can land
  on another rank before that rank has read the READYs that say whether a
  twin follows; the book holds such a landing until the rank declares.

The second engines register their work units with a second reshard store on
rank 0: a separate controller, so they keep transfer ranks from zero and the
worker ids the controller dispatches to. tpu-sync serves the first controller
and the first engine's worker service from process-wide servers; the second
store and engines each get a private one (private_native_servers).

Functions run under PoolTransfer.lock unless they say otherwise.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import inspect
import logging
import os
import time
from types import SimpleNamespace
from typing import Any, Optional

from sglang_tpu.disaggregation.raiden.host_pool import slot_host_bytes
from sglang_tpu.disaggregation.raiden.pool_spans import (
    _derived_uuid,
    _new_transfer_state,
    is_state_pool,
    is_swa_pool,
    parse_req_id,
    part_destination_blocks,
    part_spans,
    stage_req_id,
    state_destination,
    state_shard_spans,
)

logger = logging.getLogger(__name__)

_PRIVATE_WORKER_ENV = "RAIDEN_DISABLE_SINGLETON_WORKER"
# How to run without the window engine: one KV pool, one block count.
DISABLE_HYBRID_SWA = "run prefill and decode with --disable-hybrid-swa-memory"


def _key_req_id(room: int, key: tuple) -> str:
    """The id of the transfer a twin belongs to: key () is the closing
    transfer, ("g", group) a stage group."""
    return stage_req_id(room, key[1]) if key else str(room)


class Family:
    """What a second engine carries: the pools of one kind of block. A family
    says how its blocks are sized, sliced into spans, placed on the
    destination and announced."""

    # The family's half in a TwinBook, and what it is and moves in messages.
    name: str
    label: str
    cargo: str
    # Native id prefix of a twin, the data name of the family's reshard store
    # and the rank's reserved control port for its engine (conn.py).
    twin_prefix: str
    store_name: str
    port_offset: int
    # READY metadata keys: a transfer's block count in this family, and the
    # controller and work unit of the announcing worker's second engine.
    count_key: str
    controller_key: str
    unit_key: str
    # How to run without this family's engine, and how to make both roles agree.
    remedy: str
    agreement: str
    # Why a transfer and its twin cannot both be placed on a decode.
    no_side_block: str
    no_full_page: str

    def twin_req_id(self, req_id: str) -> str:
        return self.twin_prefix + req_id

    def twin_uuid(self, req_id: str) -> int:
        return _derived_uuid(self.twin_req_id(req_id))

    def twin_id(self, room: int, key: tuple) -> str:
        """The native id of the twin of transfer ``key`` of the room."""
        return self.twin_req_id(_key_req_id(room, key))


class SwaFamily(Family):
    """The sliding window: a hybrid pool's sliding-window layers, whose pools
    hold the attention window's pages only."""

    name = "swa"
    label = "sliding-window"
    cargo = "sliding-window KV"
    twin_prefix = "swa:"
    store_name = "pipeline-swa"
    port_offset = 2
    count_key = "swa"
    controller_key = "swa_controller"
    unit_key = "swa_unit"
    remedy = DISABLE_HYBRID_SWA
    agreement = f"use hybrid SWA memory on both, or {DISABLE_HYBRID_SWA}"
    no_side_block = "the decode holds no page for the producer's sliding window"
    no_full_page = "the decode holds no full-attention page for a window twin"

    @staticmethod
    def owns(tag: str) -> bool:
        return is_swa_pool(tag)

    @staticmethod
    def tensors(groups) -> list[Any]:
        return list(groups.swa)

    @staticmethod
    def max_blocks(manager) -> int:
        return manager._swa_max_request_blocks()

    @staticmethod
    def sent_blocks(manager, room: int) -> list[int]:
        """The producer's blocks of the room on the second engine."""
        return list(manager._prefill.send_swa_blocks.get(room, []))

    @staticmethod
    def spans(transfer, link, blocks: list[int], dst_attn_tp: int) -> list[Any]:
        """Spans of ``blocks``, a transfer's window pages on the producer."""
        return part_spans(
            link.pools,
            link.tensors,
            [],
            blocks,
            dst_attn_tp,
            int(transfer.manager.kv_args.total_kv_head_num),
            draft_total_kv_heads=transfer.draft_kv_heads,
            draft_swa_total_kv_heads=transfer.draft_swa_kv_heads,
            with_swa=True,
        )

    @staticmethod
    def destination(link, blocks: list[int], count: int, only_tags=None) -> dict:
        """Destination pages of a twin of ``count`` source blocks among the
        decode's window pages ``blocks``."""
        return part_destination_blocks(
            link.pools,
            [],
            blocks,
            start=0,
            count=0,
            source_swa_count=count,
            only_tags=only_tags,
        )


class StateFamily(Family):
    """Linear-attention (GDN/KDA) state: one row per request in every state
    tensor. An attention-TP rank holds its heads' share of the row, so the
    twin gathers the producer ranks' shares into the decode's full-width row."""

    name = "state"
    label = "linear-attention state"
    cargo = "linear-attention state"
    twin_prefix = "state:"
    store_name = "pipeline-state"
    port_offset = 4
    count_key = "state"
    controller_key = "state_controller"
    unit_key = "state_unit"
    remedy = "serve the model with the prefill's attention TP on the decode"
    agreement = "serve the same model on both"
    no_side_block = "the decode holds no row for the producer's linear-attention state"
    no_full_page = (
        "the decode already holds every KV page of the request, and its "
        "linear-attention state does not move without them; run the decode "
        "without a prefix cache"
    )

    @staticmethod
    def owns(tag: str) -> bool:
        return is_state_pool(tag)

    @staticmethod
    def tensors(groups) -> list[Any]:
        return list(groups.state)

    @staticmethod
    def max_blocks(manager) -> int:
        return manager._state_max_blocks

    @staticmethod
    def sent_blocks(manager, room: int) -> list[int]:
        return list(manager._prefill.send_state_rows.get(room, []))

    @staticmethod
    def spans(transfer, link, blocks: list[int], dst_attn_tp: int) -> list[Any]:
        """Spans of ``blocks``, the request's state row on the producer: this
        rank's head shard of the full-width row."""
        manager = transfer.manager
        if dst_attn_tp != 1:
            raise ValueError(
                "Raiden pipeline transfer moves linear-attention state to an "
                f"attention-TP 1 consumer only; got attention TP {dst_attn_tp}"
            )
        return state_shard_spans(
            link.pools, blocks, rank=manager.attn_tp_rank, workers=manager.attn_tp_size
        )

    @staticmethod
    def destination(link, blocks: list[int], count: int, only_tags=None) -> dict:
        """Destination rows of a twin of ``count`` source rows among the
        decode's state rows ``blocks``."""
        return state_destination(link.pools, blocks[:count])


SWA = SwaFamily()
STATE = StateFamily()
# Every family a second engine can carry; a worker's pools belong to one at most.
FAMILIES = (SWA, STATE)


def family_of(pools: list[dict]):
    """The family the second engine's ``pools`` belong to, or None without any."""
    families = {
        family for pool in pools for family in FAMILIES if family.owns(pool["tag"])
    }
    if len(families) > 1:
        raise NotImplementedError(
            "Raiden pipeline transfer runs one second engine per worker, but the "
            f"pools need {sorted(family.label for family in families)}"
        )
    return families.pop() if families else None


@contextlib.contextmanager
def private_native_servers():
    """Build the window store or engine on private tpu-sync gRPC servers.

    A RaidenController registers workers through the process-wide
    ControllerServer, and a KVCacheManager with a controller serves sender
    programs through the process-wide WorkerServiceServer. Started a second
    time, either one keeps its port and swaps in the newcomer's registry or
    engine: every worker of the full-attention store would then land in the
    window store's registry, and its dispatch would fail with "worker
    'worker_0' is not registered with this controller". Each constructor
    reads RAIDEN_DISABLE_SINGLETON_WORKER once; the full-attention store and
    engine keep the shared servers.
    """
    # TODO: drop this process-wide environment switch once tpu-sync offers a
    # constructor option for private servers.
    previous = os.environ.get(_PRIVATE_WORKER_ENV)
    os.environ[_PRIVATE_WORKER_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_PRIVATE_WORKER_ENV, None)
        else:
            os.environ[_PRIVATE_WORKER_ENV] = previous


def split_manifest(pools: list[dict]) -> tuple[list[dict], list[dict]]:
    """The full-attention pools of one manifest and the pools of its second
    engine, each with storage indices local to the engine that wraps it."""

    def side(pool: dict) -> bool:
        return any(family.owns(pool["tag"]) for family in FAMILIES)

    full = [dict(pool) for pool in pools if not side(pool)]
    second = [dict(pool) for pool in pools if side(pool)]
    for kind in (full, second):
        for index, pool in enumerate(kind):
            pool["storage_index"] = index
    return full, second


def _may_have_twin(req_id: str) -> bool:
    """Closing transfers and stage groups can have a twin; early parts never."""
    return ".p" not in req_id


class TwinBook:
    """Pairs each transfer with its twin before either is reported.

    ``half`` is "fa" or the family's name, and ``kind`` the native list the
    half settled in: "sent", "received" or "failed". A transfer without a twin
    passes straight through, except that a decode (``hold_undeclared``) holds
    a closing or stage-group half until expect() says whether a twin follows.
    """

    def __init__(self, *, hold_undeclared: bool):
        self._hold_undeclared = hold_undeclared
        # req_id -> dict(twin=bool | None, fa=kind | None, side=kind | None)
        self._halves: dict[str, dict] = {}
        self._ready: list[tuple[str, str]] = []

    def expect(self, req_id: str, twin: bool) -> None:
        entry = self._entry(req_id)
        entry["twin"] = twin
        self._settle(req_id, entry)

    def feed(self, req_id: str, half: str, kind: str) -> None:
        if half == "fa" and req_id not in self._halves:
            if not (self._hold_undeclared and _may_have_twin(req_id)):
                self._ready.append((req_id, kind))
                return
        entry = self._entry(req_id)
        half = "fa" if half == "fa" else "side"
        if entry[half] is None:
            entry[half] = kind
        self._settle(req_id, entry)

    def drain(self) -> tuple[list, list, list]:
        """The transfers settled since the last drain, as native poll lists."""
        lists = dict(sent=[], received=[], failed=[])
        for req_id, kind in self._ready:
            lists[kind].append(SimpleNamespace(req_id=req_id, completed_ns=None))
        self._ready = []
        return lists["sent"], lists["received"], lists["failed"]

    def forget(self, room: int) -> None:
        for req_id in [r for r in self._halves if parse_req_id(r)[0] == room]:
            del self._halves[req_id]

    def _entry(self, req_id: str) -> dict:
        return self._halves.setdefault(req_id, dict(twin=None, fa=None, side=None))

    def _settle(self, req_id: str, entry: dict) -> None:
        if entry["twin"] is None or entry["fa"] is None:
            return
        if entry["twin"] and entry["side"] is None:
            return
        del self._halves[req_id]
        failed = entry["fa"] == "failed" or entry["side"] == "failed"
        self._ready.append((req_id, "failed" if failed else entry["fa"]))


class SideLink:
    """One worker's second engine, its work unit and its twins."""

    def __init__(
        self,
        *,
        family: Family,
        pools: list[dict],
        tensors: list[Any],
        controller: str,
        unit_fields: dict,
        dst_units: list,
        max_blocks: int,
        slots: int,
        hold_undeclared: bool,
    ):
        self.family = family
        self.pools = pools
        self.tensors = tensors
        self.controller = controller
        self.unit_fields = unit_fields
        self.unit = SimpleNamespace(**unit_fields)
        self.dst_units = dst_units
        self.max_blocks = max_blocks
        self.slots = slots
        self.book = TwinBook(hold_undeclared=hold_undeclared)
        # Producer: req ids whose twin is registered and neither completed nor
        # cancelled yet.
        self.registered: set[str] = set()
        # Decode: req ids whose twin settled natively on this rank, landed or
        # failed, until the room is forgotten; a twin can settle here before
        # this rank registers the room or reaches the transfer.
        self.settled: set[str] = set()
        self.engine = None
        self.client = None
        self.executor: Optional[concurrent.futures.ThreadPoolExecutor] = None

    @property
    def tags(self) -> list[str]:
        return [pool["tag"] for pool in self.pools]


# ----- construction ------------------------------------------------------


def require_bounded_staging(family: Family) -> None:
    """Refuse, before any engine is built, a tpu-sync whose engines cannot
    stage per-pool block counts, which the second engine needs."""
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

    parameters = inspect.signature(KVCacheManager.register_pools).parameters
    if not {"staging_leases", "staging_blocks_per_pool"} <= parameters.keys():
        raise RuntimeError(
            f"The Raiden pipeline transfer moves {family.cargo} through a "
            "second engine, which needs a tpu-sync with bounded pool staging; "
            f"upgrade tpu-sync or {family.remedy}"
        )


def plan_slots(
    manager,
    family: Family,
    tensors: list[Any],
    full_tensors: list[Any],
    full_max_blocks: int,
) -> tuple[int, int, int]:
    """(slots, max_blocks, host bytes) of a worker's second engine. A twin and
    its transfer take a slot on each engine, so it gets one slot per pair the
    rank's staging budget affords, and the full-attention engine plans with
    the rest."""
    max_blocks = family.max_blocks(manager)
    slots = manager._num_slots(
        tensors,
        max_blocks,
        paired_slot_bytes=slot_host_bytes(full_tensors, full_max_blocks),
    )
    return slots, max_blocks, slots * slot_host_bytes(tensors, max_blocks)


def build(
    transfer,
    *,
    family: Family,
    pools: list[dict],
    tensors: list[Any],
    addresses: tuple[str, str],
    slots: int,
    max_blocks: int,
) -> SideLink:
    """The second-engine link of a worker whose full-attention engine is up:
    started, with its pools and work unit registered."""
    from tpu_sync.api.torch.reshard_client import ReshardClient

    controller, dispatch = addresses
    unit_fields = transfer.unit_fields | {"job_replica_id": controller}
    link = SideLink(
        family=family,
        pools=pools,
        tensors=tensors,
        controller=controller,
        unit_fields=unit_fields,
        dst_units=[
            SimpleNamespace(
                **(unit_fields | {"data_replica_idx": unit.data_replica_idx})
            )
            for unit in transfer.dst_units
        ],
        max_blocks=max_blocks,
        slots=slots,
        hold_undeclared=not transfer.producer,
    )
    link.client = ReshardClient(controller)
    if not transfer.producer:
        link.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=transfer.coord_workers,
            thread_name_prefix=f"raiden-pipeline-{family.name}",
        )
    with private_native_servers():
        # The rank's reserved control port for this family (conn.py).
        link.engine = transfer._start_engine(
            tensors=tensors,
            max_blocks=max_blocks,
            slots=slots,
            dispatch=dispatch,
            port_offset=family.port_offset,
        )
    fingerprint = transfer._register_pools(
        engine=link.engine,
        client=link.client,
        unit=link.unit,
        pools=pools,
        tensors=tensors,
        staging_options=dict(
            staging_leases=slots, staging_blocks_per_pool=[max_blocks] * len(pools)
        ),
    )
    logger.info(
        "Raiden pipeline %s engine ready: rank=%d slots=%d "
        "max_blocks=%d pools=%d controller=%s layout_fingerprint=%s",
        family.label,
        transfer.rank,
        slots,
        max_blocks,
        len(pools),
        controller,
        fingerprint,
    )
    return link


def check_agreement(transfer, meta: dict) -> None:
    """A READY from a producer whose second engine differs from this decode's."""
    announced = next((f for f in FAMILIES if f.unit_key in meta), None)
    local = None if transfer.side is None else transfer.side.family
    if announced is not local:
        family = announced or local
        raise ValueError(
            f"Raiden pipeline producer and decode disagree on the {family.label} "
            f"engine: {family.agreement}"
        )


def source_fields(transfer) -> dict:
    link = transfer.side
    if link is None:
        return {}
    family = link.family
    return {family.controller_key: link.controller, family.unit_key: link.unit_fields}


def _family(transfer):
    """The family of a worker's second engine; without one, the window's."""
    return SWA if transfer.side is None else transfer.side.family


def count_key(transfer) -> str:
    """READY metadata key of a transfer's block count on the second engine; a
    worker without one still announces the window's, as zero."""
    return _family(transfer).count_key


def sent_blocks(transfer, room: int) -> list[int]:
    """The producer's blocks of the room on the second engine."""
    return _family(transfer).sent_blocks(transfer.manager, room)


# ----- producer ----------------------------------------------------------


def register_twin(
    transfer, *, req_id: str, uuid: int, blocks: list[int], dst_attn_tp: int
) -> None:
    """Register the second engine's ``blocks`` of ``req_id``, if any, as its
    twin. A twin that cannot register cancels the transfer's own registration,
    which nothing has claimed yet, and raises."""
    link = transfer.side
    if link is None or not blocks:
        return
    family = link.family
    try:
        link.client.register_request_blocks(
            req_id=family.twin_req_id(req_id),
            uuid=family.twin_uuid(req_id),
            unit=link.unit,
            block_ids=list(blocks),
            pool_spans=family.spans(transfer, link, blocks, dst_attn_tp),
        )
    except Exception:
        try:
            transfer.client.cancel_request_blocks_if_unclaimed(req_id, uuid)
        except Exception as error:
            logger.error(
                "Raiden pipeline rank%d: cancelling %s after its %s twin failed "
                "to register also failed: %s",
                transfer.rank,
                req_id,
                family.label,
                error,
            )
        raise
    link.registered.add(req_id)
    link.book.expect(req_id, True)


def settle_half(transfer, req_id: str) -> bool:
    """Report the full-attention half of ``req_id`` as sent, for a producer
    with no page in it: its engine reports none. False when the transfer has
    no twin registered, so nothing is left to settle."""
    link = transfer.side
    if link is None or req_id not in link.registered:
        return False
    link.book.feed(req_id, "fa", "sent")
    return True


def complete_twin(transfer, req_id: str) -> None:
    """complete_request_blocks for the twin of ``req_id``, if it has one
    registered; the twin counts as registered until the call succeeds."""
    link = transfer.side
    if link is None or req_id not in link.registered:
        return
    link.client.complete_request_blocks(
        link.family.twin_req_id(req_id), link.family.twin_uuid(req_id), link.unit
    )
    link.registered.discard(req_id)


def cancel_unclaimed(transfer, req_id: str, uuid: int, *, registered: bool) -> bool:
    """cancel_request_blocks_if_unclaimed over a transfer and its twin: True
    once neither can be claimed any more. Each unclaimed half is cancelled
    even when the other is claimed; a claimed half settles natively, and the
    pair then settles as failed."""
    link = transfer.side
    twin_claimed = False
    if link is not None and req_id in link.registered:
        family = link.family
        if link.client.cancel_request_blocks_if_unclaimed(
            family.twin_req_id(req_id), family.twin_uuid(req_id)
        ):
            link.registered.discard(req_id)
            link.book.feed(req_id, family.name, "failed")
        else:
            twin_claimed = True
    cancelled = not registered or transfer.client.cancel_request_blocks_if_unclaimed(
        req_id, uuid
    )
    if cancelled and twin_claimed:
        link.book.feed(req_id, "fa", "failed")
    return cancelled and not twin_claimed


# ----- native completion -------------------------------------------------


def merge(transfer, status, side_status) -> tuple[list, list, list] | None:
    """Both engines' settled transfers as the pipeline's: each one reported
    once its twin settled too."""
    link = transfer.side
    family = link.family
    mgr = transfer.manager
    for halves, half in ((status, "fa"), (side_status, family.name)):
        if halves is None:
            continue
        for kind, settled in zip(("sent", "received", "failed"), halves):
            for t in settled:
                req_id = mgr._norm_req_id(t.req_id)
                if half != "fa":
                    req_id = req_id.removeprefix(family.twin_prefix)
                    if not transfer.producer:
                        _twin_settled(transfer, req_id)
                link.book.feed(req_id, half, kind)
    sent, received, failed = link.book.drain()
    if not (sent or received or failed):
        return None
    return sent, received, failed


def _twin_settled(transfer, req_id: str) -> None:
    """Decode: the twin of ``req_id`` settled natively on this rank, so no
    push of it can land any more."""
    transfer.side.settled.add(req_id)
    room, key = parse_req_id(req_id)
    request = transfer.receives.get(room)
    state = None if request is None else request.twins.get(_twin_key(key))
    if state is not None:
        state["done"] = True
        state["armed_until"] = 0.0


def _twin_key(key) -> tuple:
    """The twin key of a parsed native transfer key."""
    return () if key is None else ("g", key[1])


# ----- decode ------------------------------------------------------------


def twin_destination(transfer, request, *, count: int, only_tags=None) -> dict | None:
    """Destination blocks of a transfer's twin of ``count`` source blocks: None
    when it has none."""
    link = transfer.side
    if link is None or not count:
        return None
    return link.family.destination(link, request.side, count, only_tags)


def unplaceable(transfer, args: dict, twin: dict | None) -> str | None:
    """Why a transfer and its twin cannot both move, or None. The producer
    registered both halves, so the decode must claim both."""
    if twin is None:
        return None
    family = transfer.side.family
    if not twin["transfer_pool_tags"]:
        return family.no_side_block
    if not args["transfer_pool_tags"]:
        return family.no_full_page
    return None


def twin_live(request, *, key: tuple) -> bool:
    """Whether the twin of transfer ``key`` is accepted or coordinating and
    has not settled: it shares that transfer's staging reservation."""
    state = request.twins.get(key)
    return (
        state is not None
        and (state["accepted"] or state["future"] is not None)
        and not state["done"]
    )


def start_twin(
    transfer,
    room: int,
    request,
    *,
    key: tuple,
    twin: dict | None,
    metas: list[dict],
    ranks: list[int],
) -> None:
    """Declare whether the transfer ``key`` of the room has a twin and start
    it: rank 0 coordinates it on the link's threads; another attention-TP
    rank waits for its engine to report the landing. ``metas`` are the READY
    metadata of the transfer's stages by rank, ``ranks`` the source transfer
    ranks it names, as for its full-attention transfer."""
    link = transfer.side
    if link is None:
        return
    family = link.family
    link.book.expect(_key_req_id(room, key), twin is not None)
    if twin is None:
        return
    state = request.twins.get(key)
    if state is None:
        state = request.twins[key] = _new_transfer_state()
        state["done"] = _key_req_id(room, key) in link.settled
    if transfer.manager.attn_tp_rank != 0:
        state["accepted"] = True
        return
    if state["accepted"] or state["future"] is not None or state["done"]:
        return
    controllers = {meta[family.controller_key] for meta in metas}
    if len(controllers) != 1:
        raise ValueError(
            f"Raiden pipeline sources disagree on the {family.label} controller"
        )
    controller = controllers.pop()
    req_id = _key_req_id(room, key)
    state["kwargs"] = dict(
        src_units=[
            SimpleNamespace(**(metas[0][family.unit_key] | {"data_replica_idx": rank}))
            for rank in ranks
        ],
        dst_units=link.dst_units,
        req_id=family.twin_req_id(req_id),
        uuid=family.twin_uuid(req_id),
        use_block_chunks=True,
        is_sender=True,
        dst_mem_type=transfer.hbm_mem_type,
        src_controller_address=controller,
        dst_controller_address=link.controller,
        **twin,
    )
    _submit(transfer, state)


def resume_twin(transfer, request, *, key: tuple) -> None:
    """Coordinate a twin again whose coordination asked for a retry, once its
    full-attention transfer has moved on."""
    state = request.twins.get(key)
    if (
        state is None
        or "kwargs" not in state
        or state["accepted"]
        or state["future"] is not None
        or state["done"]
        or time.monotonic() < state["next_attempt"]
    ):
        return
    _submit(transfer, state)


def _submit(transfer, state: dict) -> None:
    kwargs = state["kwargs"]
    state["future"] = transfer.side.executor.submit(
        transfer._coordinate, kwargs["src_controller_address"], **kwargs
    )


# ----- lifetime ----------------------------------------------------------


def forget(transfer, room: int) -> None:
    link = transfer.side
    if link is None:
        return
    link.book.forget(room)
    link.registered = {r for r in link.registered if parse_req_id(r)[0] != room}
    link.settled = {r for r in link.settled if parse_req_id(r)[0] != room}
