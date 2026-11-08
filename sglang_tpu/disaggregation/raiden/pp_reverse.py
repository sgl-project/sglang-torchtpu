"""Reverse Raiden pipeline transfer: a PP prefill stage pulls a cached prefix
from a tensor-parallel decode (P/D prefix pull).

The forward transfer moves a prompt from TP=1 prefill stages into every rank of a
TP decode, each rank taking its kv-head slice. The reverse one fans in: each of
the decode's src_tp ranks sends all of its heads of a page, and one prefill
stage assembles the whole page. The decode's controller plans it, as the
source-side controller plans every pool reshard, and each prefill stage
coordinates its own transfer: a plan needs one destination manifest, and the
stages hold different layers.

Per room and stage the decode registers its pages of the prefix [start, end)
under pull_req_id(room, stage), and the window pages ending at end under that
id's window twin on the sliding-window engine.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import logging
import time
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from sglang.srt.utils.network import NetworkAddress
from sglang_tpu.disaggregation.raiden import pp_swa
from sglang_tpu.disaggregation.raiden.pp_common import (
    _BLOCK_MAJOR_KV_RANK,
    _HEAD_MAJOR_KV_RANK,
    _kv_rank,
    is_swa_pool,
    pool_is_seq_along_lane,
    pool_total_kv_heads,
    seq_along_lane_head_run_bytes,
)
from sglang_tpu.disaggregation.raiden.reshard_schedule import (
    ReshardGeometry,
    head_bytes,
    head_slice_params,
)
from sglang_tpu.disaggregation.raiden.serde import (
    MessageTag,
    PrefixPullReady,
    PrefixPullReq,
)

logger = logging.getLogger(__name__)

# Native ids of reverse transfers never contain ".p" or ".s", which
# pp_common.parse_req_id reads as forward early parts and stage groups.
_PULL_MARK = "~pull"


def pull_req_id(room: int, stage: int) -> str:
    return f"{room}{_PULL_MARK}{stage}"


def is_pull_req_id(req_id: str) -> bool:
    return _PULL_MARK in req_id


def parse_pull_req_id(req_id: str) -> tuple[int, int]:
    """(room, stage) of a reverse transfer id."""
    room, mark, stage = req_id.partition(_PULL_MARK)
    if not mark:
        raise ValueError(f"{req_id!r} is not a prefix-pull transfer id")
    return int(room), int(stage)


def pull_uuid(room: int, stage: int) -> int:
    """A positive signed 64-bit native uuid per (room, stage)."""
    digest = hashlib.sha256(pull_req_id(room, stage).encode()).digest()
    return (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1


def split_native_status(status, norm: Callable[[str], str] = str):
    """Split (sent, received, failed) native transfers into forward and reverse;
    either half is None when it holds nothing, as a poll that settled none."""
    if status is None:
        return None, None
    forward, reverse = [], []
    for transfers in status:
        forward.append([t for t in transfers if not is_pull_req_id(norm(t.req_id))])
        reverse.append([t for t in transfers if is_pull_req_id(norm(t.req_id))])
    return (
        tuple(forward) if any(forward) else None,
        tuple(reverse) if any(reverse) else None,
    )


def fan_in_spans(
    pools: Sequence[dict],
    tensors: Sequence[Any],
    full: list[int],
    swa: list[int],
    *,
    src_tp: int,
    src_rank: int,
    total_kv_heads: int,
    seq_along_lane: bool,
    draft_total_kv_heads: int | None = None,
    draft_swa_total_kv_heads: int | None = None,
) -> list[Any]:
    """Source spans of decode rank src_rank into one TP=1 prefill stage: all of
    the rank's heads of every page, placed at its heads' slot of the whole page.

    pools and tensors are the decode rank's, restricted to the stage's layers;
    full and swa its page ids, in token order. Offsets are in the destination's
    global byte space (dst_space_version 1), whose page stride is a whole page of
    every head.
    """
    result = []
    for pool, tensor in zip(pools, tensors):
        tag = pool["tag"]
        if _kv_rank(tensor) == _BLOCK_MAJOR_KV_RANK:
            raise ValueError(
                "A prefix pull slices per-layer kv-heads; a block-major KV bundle "
                "cannot feed a TP=1 prefill stage from a TP decode"
            )
        sal = pool_is_seq_along_lane(tag, seq_along_lane)
        pool_heads = pool_total_kv_heads(
            tag,
            total_kv_heads,
            draft_total_kv_heads=draft_total_kv_heads,
            draft_swa_total_kv_heads=draft_swa_total_kv_heads,
            producer_kv_heads=lambda: _unknown_draft_heads(tag),
        )
        if pool_heads % src_tp:
            # Replicated heads would cover the same destination bytes twice.
            raise ValueError(
                f"Prefix pull of {tag}: {pool_heads} kv-heads do not split evenly "
                f"over the decode's attention TP {src_tp}"
            )
        blocks = swa if is_swa_pool(tag) else full
        if sal or _kv_rank(tensor) == _HEAD_MAJOR_KV_RANK:
            spans = _head_run_fan_in(
                tag=tag,
                tensor=tensor,
                sal=sal,
                num_blocks=len(blocks),
                src_tp=src_tp,
                src_rank=src_rank,
                pool_heads=pool_heads,
            )
        else:
            spans = _fused_fan_in(
                tensor=tensor,
                num_blocks=len(blocks),
                src_tp=src_tp,
                src_rank=src_rank,
                pool_heads=pool_heads,
            )
        result.append(
            SimpleNamespace(
                tag=tag,
                block_ids=list(blocks),
                declared_bytes=sum(span.size_bytes * span.count for span in spans),
                dst_space_version=1,
                spans=spans,
            )
        )
    return result


def _unknown_draft_heads(tag: str) -> int:
    raise ValueError(
        f"Prefix pull of {tag} needs the draft's total kv-head count; a TP decode "
        "rank holds only its slice"
    )


def _head_run_fan_in(
    *,
    tag: str,
    tensor: Any,
    sal: bool,
    num_blocks: int,
    src_tp: int,
    src_rank: int,
    pool_heads: int,
) -> list[Any]:
    if sal:
        run = seq_along_lane_head_run_bytes(tensor)
    else:
        run = 2 * int(tensor.shape[2]) * int(tensor.shape[3]) * tensor.element_size()
    _, num_heads, dst_head, src_heads, dst_heads = head_slice_params(
        src_tp=src_tp,
        dst_tp=1,
        src_rank=src_rank,
        dst_rank=0,
        total_kv_heads=pool_heads,
    )
    if int(tensor.shape[1]) != 2 * src_heads:
        raise ValueError(
            f"Decode pool {tag} of shape {tuple(tensor.shape)} does not hold "
            f"{src_heads} of {pool_heads} kv-heads"
        )
    return [
        SimpleNamespace(
            src_block_ordinal=index,
            src_offset_bytes=0,
            dst_block_index=0,
            dst_offset_bytes=(index * dst_heads + dst_head) * run,
            size_bytes=num_heads * run,
            src_stride_bytes=0,
            dst_stride_bytes=0,
            count=1,
            dst_unit_ordinal=0,
        )
        for index in range(num_blocks)
    ]


def _fused_fan_in(
    *, tensor: Any, num_blocks: int, src_tp: int, src_rank: int, pool_heads: int
) -> list[Any]:
    geometry = ReshardGeometry(
        src_tp=src_tp,
        dst_tp=1,
        hp_bytes=head_bytes(tensor),
        total_kv_heads=pool_heads,
        kv_packing=int(tensor.shape[3]),
        element_size=tensor.element_size(),
    )
    page_tokens = int(tensor.shape[1])
    src_offset, size, src_stride, dst_offset, dst_stride = geometry.entry_bytes(
        src_rank, 0
    )
    return [
        SimpleNamespace(
            src_block_ordinal=index,
            src_offset_bytes=src_offset,
            dst_block_index=0,
            dst_offset_bytes=index * dst_stride * page_tokens + dst_offset,
            size_bytes=size,
            src_stride_bytes=src_stride,
            dst_stride_bytes=dst_stride,
            count=page_tokens,
            dst_unit_ordinal=0,
        )
        for index in range(num_blocks)
    ]


def stage_pools(
    pools: Sequence[dict], tensors: Sequence[Any], stage_tags: Sequence[str]
) -> tuple[list[dict], list[Any]]:
    """The decode pools a stage holds, in the decode's manifest order. Every
    stage tag must be one the decode registers."""
    wanted = set(stage_tags)
    picked = [(p, t) for p, t in zip(pools, tensors) if p["tag"] in wanted]
    missing = wanted - {p["tag"] for p, _ in picked}
    if missing:
        raise ValueError(
            f"Prefix pull: the decode holds no pools for stage tags {sorted(missing)}"
        )
    return [p for p, _ in picked], [t for _, t in picked]


def destination_args(
    stage_tags: Sequence[str], full: list[int], swa: list[int]
) -> dict:
    """coordinate_transfer page arguments of one stage: every tag lands the same
    page ids, full pages for full-attention tags and window pages for the rest."""
    tags, blocks, counts = [], [], []
    for tag in stage_tags:
        local = swa if is_swa_pool(tag) else full
        if not local:
            continue
        tags.append(tag)
        blocks.extend(local)
        counts.append(len(local))
    return dict(
        transfer_pool_tags=tags, dst_device_block_ids=blocks, dst_block_counts=counts
    )


# ----- protocol ------------------------------------------------------------
#
#   prefill stage s                      decode rank r (of attn_tp)
#   start(room)  -- PREFIX_PULL_REQ -->  offer: admit staging, register its
#                                        pages and window twin on the decode's
#                                        controllers under pull_req_id(room, s)
#                <-- PREFIX_PULL_READY -- controllers and work units (or refusal)
#   every rank ready: coordinate_transfer on the decode's controller, src = the
#   attn_tp decode units, dst = this stage's unit; the twin likewise on the
#   window controllers.
#   native completion: the stage sees "received", each decode rank "sent";
#   the decode completes (or, unclaimed past its deadline, cancels) its
#   registrations.

PENDING = "pending"
DONE = "done"
FAILED = "failed"
# Failure kinds, the reason label of the prefix-pull counters.
DECODE_REFUSED = "decode_refused"
TIMEOUT = "timeout"
COORDINATE_FAILED = "coordinate_failed"
NATIVE_FAILED = "native_failed"
# Seconds between a refused (staging busy) or declined attempt and the next.
_RETRY_S = 0.05


class ReversePull:
    """One pipeline worker's side of the reverse transfer: a decode rank's
    offers or a prefill stage's pulls. Called under transfer.lock except for
    on_message, which takes it."""

    def __init__(
        self,
        transfer: Any,
        held_prefix: Callable[[int], Any] | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.transfer = transfer
        self.held_prefix = held_prefix
        self.clock = clock
        # Decode: (room, stage) -> offer state.
        self.offers: dict[tuple[int, int], dict] = {}
        # Prefill: room -> this stage's pull state.
        self.pulls: dict[int, dict] = {}
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None

    # ----- messages ------------------------------------------------------

    def on_message(self, msg: list[bytes]) -> None:
        tag = bytes(msg[0])
        with self.transfer.lock:
            if tag == MessageTag.PREFIX_PULL_REQ:
                request = PrefixPullReq.from_frames(msg)
                reply = self._offer(request)
                self.transfer._send(request.endpoint, reply.to_frames())
            elif tag == MessageTag.PREFIX_PULL_READY:
                self._on_ready(PrefixPullReady.from_frames(msg))
            else:
                raise ValueError(f"not a prefix-pull message: {tag!r}")

    # ----- decode: offers ------------------------------------------------

    def _refuse(self, request, reason: str, *, retry: bool = False):
        mgr = self.transfer.manager
        if not retry:
            logger.warning(
                "Prefix pull room=%d stage=%d refused by decode rank %d: %s",
                request.room,
                request.stage,
                mgr.attn_tp_rank,
                reason,
            )
        return PrefixPullReady(
            room=request.room,
            stage=request.stage,
            rank=mgr.attn_tp_rank,
            attn_tp_size=mgr.attn_tp_size,
            ok=False,
            retry=retry,
            reason=reason,
        )

    def _ready(self, request, offer: dict):
        t = self.transfer
        mgr = t.manager
        return PrefixPullReady(
            room=request.room,
            stage=request.stage,
            rank=mgr.attn_tp_rank,
            attn_tp_size=mgr.attn_tp_size,
            ok=True,
            controller=t.controller,
            unit=dict(t.unit_fields),
            swa_controller=t.swa.controller if t.swa is not None else "",
            swa_unit=dict(t.swa.unit_fields) if t.swa is not None else {},
            full_pages=len(offer["full"]),
            swa_pages=len(offer["swa"]),
        )

    def _offer(self, request):
        """Register this rank's pages for one stage; idempotent per (room, stage)."""
        t = self.transfer
        key = (request.room, request.stage)
        offer = self.offers.get(key)
        if offer is not None:
            return self._ready(request, offer)
        if t.producer or self.held_prefix is None:
            return self._refuse(request, "only a decode serves prefix pulls")
        held = self.held_prefix(request.room)
        if held is None:
            return self._refuse(request, "the decode holds no prefix for the room")
        if request.end != held.end or request.start % held.page_size:
            return self._refuse(
                request,
                f"range [{request.start}, {request.end}) does not end at the held "
                f"prefix {held.end} or start on a page",
            )
        full = held.full_pages_from(request.start)
        swa = list(held.swa_pages) if request.swa_tags else []
        if request.swa_tags and t.swa is None:
            return self._refuse(request, "the decode has no sliding-window engine")
        # The stages' transfers read disjoint layers of one room, so the room
        # takes one staging slot whichever stages pull it.
        admission_key = ("pull", request.room)
        swa_slot = t.swa.max_blocks if swa else 0
        if admission_key not in t.admission and (
            t.admission.waiting is not None
            or not t.admission.admit(admission_key, t.max_blocks, swa_slot)
        ):
            t.admission.forget(admission_key)
            return self._refuse(request, "decode staging busy", retry=True)
        req_id = pull_req_id(request.room, request.stage)
        registered = []
        try:
            self._register(
                client=t.client,
                unit=t.unit,
                req_id=req_id,
                uuid=pull_uuid(request.room, request.stage),
                pools=t.pools,
                tensors=t.tensors,
                tags=request.full_tags,
                full=full,
                swa=[],
            )
            registered.append(
                (t.client, req_id, pull_uuid(request.room, request.stage))
            )
            if swa:
                twin = pp_swa.twin_req_id(req_id)
                self._register(
                    client=t.swa.client,
                    unit=t.swa.unit,
                    req_id=twin,
                    uuid=pp_swa.twin_uuid(req_id),
                    pools=t.swa.pools,
                    tensors=t.swa.tensors,
                    tags=request.swa_tags,
                    full=[],
                    swa=swa,
                )
        except Exception as error:
            for client, rid, uuid in registered:
                client.cancel_request_blocks_if_unclaimed(rid, uuid)
            self._release_room_admission(request.room, admission_key)
            return self._refuse(request, f"registration failed: {error}")
        offer = self.offers[key] = dict(
            full=full,
            swa=swa,
            req_id=req_id,
            admission_key=admission_key,
            settled={"fa": None, "swa": None if swa else "sent"},
            deadline=self.clock() + t.manager._timeout_s,
        )
        return self._ready(request, offer)

    def _register(self, *, client, unit, req_id, uuid, pools, tensors, tags, full, swa):
        t = self.transfer
        picked, picked_tensors = stage_pools(pools, tensors, tags)
        spans = fan_in_spans(
            picked,
            picked_tensors,
            full,
            swa,
            src_tp=t.manager.attn_tp_size,
            src_rank=t.manager.attn_tp_rank,
            total_kv_heads=int(t.manager.kv_args.total_kv_head_num),
            seq_along_lane=t.seq_along_lane,
            draft_total_kv_heads=t.draft_kv_heads,
            draft_swa_total_kv_heads=t.draft_swa_kv_heads,
        )
        client.register_request_blocks(
            req_id=req_id,
            uuid=uuid,
            unit=unit,
            block_ids=full or swa,
            pool_spans=spans,
        )

    def _retire_offer(self, key: tuple[int, int], *, failed: bool) -> None:
        t = self.transfer
        offer = self.offers.pop(key)
        room, stage = key
        try:
            t.client.complete_request_blocks(
                offer["req_id"], pull_uuid(room, stage), t.unit
            )
            if offer["swa"]:
                t.swa.client.complete_request_blocks(
                    pp_swa.twin_req_id(offer["req_id"]),
                    pp_swa.twin_uuid(offer["req_id"]),
                    t.swa.unit,
                )
        finally:
            self._release_room_admission(room, offer["admission_key"])
        if failed:
            logger.error(
                "Prefix pull room=%d stage=%d failed natively on decode rank %d",
                room,
                stage,
                t.manager.attn_tp_rank,
            )

    def _release_room_admission(self, room: int, admission_key) -> None:
        if not any(key[0] == room for key in self.offers):
            self.transfer.admission.release(admission_key)

    def _expire_offers(self) -> None:
        """Cancel offers the prefill never claimed: past their deadline, or whose
        hold the decode released (the request's KV landed or it was freed)."""
        t = self.transfer
        now = self.clock()
        for key, offer in list(self.offers.items()):
            room, stage = key
            held = self.held_prefix(room) if self.held_prefix is not None else None
            if now < offer["deadline"] and held is not None:
                continue
            cancelled = t.client.cancel_request_blocks_if_unclaimed(
                offer["req_id"], pull_uuid(room, stage)
            )
            if offer["swa"]:
                cancelled = (
                    t.swa.client.cancel_request_blocks_if_unclaimed(
                        pp_swa.twin_req_id(offer["req_id"]),
                        pp_swa.twin_uuid(offer["req_id"]),
                    )
                    and cancelled
                )
            if cancelled:
                self.offers.pop(key)
                self._release_room_admission(room, offer["admission_key"])
            elif held is None:
                logger.error(
                    "Prefix pull room=%d stage=%d still reads decode pages its hold "
                    "released; native completion settles it",
                    room,
                    stage,
                )

    # ----- prefill: pulls -------------------------------------------------

    def start(
        self, room: int, *, start: int, end: int, full: list[int], swa: list[int]
    ) -> None:
        """Pull [start, end) of the room's prompt into this stage's pages full
        (full-attention, in token order) and swa (the window ending at end)."""
        t = self.transfer
        if not t.producer:
            raise ValueError("Only a prefill stage pulls a prefix")
        if room in self.pulls:
            raise ValueError(f"Prefix pull of room {room} already started")
        consumers = t.replies.get(room)
        if not consumers:
            raise ValueError(f"Prefix pull of room {room}: the decode has not asked")
        widths = {int(meta["attn_tp_size"]) for meta in consumers.values()}
        if len(widths) != 1 or len(consumers) != widths.copy().pop():
            raise ValueError(
                f"Prefix pull of room {room}: expected one request per decode rank, "
                f"got {len(consumers)} for attention TP {sorted(widths)}"
            )
        swa_tags = t.swa.tags if (t.swa is not None and swa) else []
        self.pulls[room] = dict(
            start=start,
            end=end,
            full=list(full),
            swa=list(swa),
            full_tags=[pool["tag"] for pool in t.pools],
            swa_tags=list(swa_tags),
            endpoints=list(consumers),
            attn_tp=widths.pop(),
            ready={},
            state=PENDING,
            reason="",
            resend_at=None,
            kind="",
            fa=_half(),
            swa_half=_half() if swa_tags else None,
            admitted=False,
            deadline=self.clock() + t.manager._timeout_s,
        )
        self._send_requests(room)

    def status(self, room: int) -> tuple[str, str]:
        pull = self.pulls.get(room)
        if pull is None:
            return FAILED, "no prefix pull for the room"
        return pull["state"], pull["reason"]

    def failure_kind(self, room: int) -> str:
        return self.pulls[room]["kind"]

    def in_flight(self, room: int) -> bool:
        """Whether an accepted half of the room's pull may still write its pages."""
        pull = self.pulls.get(room)
        if pull is None:
            return False
        halves = [h for h in (pull["fa"], pull["swa_half"]) if h is not None]
        return any(
            (h["accepted"] or h["future"] is not None) and h["settled"] is None
            for h in halves
        )

    def forget(self, room: int) -> None:
        pull = self.pulls.pop(room, None)
        if pull is not None and pull["admitted"]:
            self.transfer.admission.release(("pull", room))

    def _send_requests(self, room: int) -> None:
        t = self.transfer
        pull = self.pulls[room]
        endpoint = NetworkAddress(t.manager.local_ip, t.manager.rank_port).to_tcp()
        request = PrefixPullReq(
            room=room,
            stage=t.manager.pp_rank,
            endpoint=endpoint,
            full_tags=pull["full_tags"],
            swa_tags=pull["swa_tags"],
            start=pull["start"],
            end=pull["end"],
        ).to_frames()
        for peer in pull["endpoints"]:
            t._send(peer, request)

    def _fail(self, room: int, kind: str, reason: str) -> None:
        pull = self.pulls[room]
        if pull["state"] != PENDING:
            return
        pull["state"], pull["kind"], pull["reason"] = FAILED, kind, reason
        logger.warning(
            "Prefix pull room=%d stage=%d failed: %s",
            room,
            self.transfer.manager.pp_rank,
            reason,
        )

    def _on_ready(self, ready) -> None:
        pull = self.pulls.get(ready.room)
        if pull is None or pull["state"] != PENDING:
            return
        if ready.stage != self.transfer.manager.pp_rank:
            raise ValueError(
                f"Prefix pull READY for stage {ready.stage} reached stage "
                f"{self.transfer.manager.pp_rank}"
            )
        if not ready.ok:
            if ready.retry:
                pull["resend_at"] = self.clock() + _RETRY_S
            else:
                self._fail(
                    ready.room,
                    DECODE_REFUSED,
                    f"decode rank {ready.rank}: {ready.reason}",
                )
            return
        if ready.full_pages != len(pull["full"]) or ready.swa_pages != len(pull["swa"]):
            self._fail(
                ready.room,
                DECODE_REFUSED,
                f"decode rank {ready.rank} offers {ready.full_pages} + "
                f"{ready.swa_pages} pages for {len(pull['full'])} + "
                f"{len(pull['swa'])}",
            )
            return
        pull["ready"][ready.rank] = ready

    def _coordinate_args(self, room: int, pull: dict, *, window: bool) -> dict:
        t = self.transfer
        readies = [pull["ready"][rank] for rank in range(pull["attn_tp"])]
        if window:
            unit_key, controller, dst_unit, dst_controller = (
                "swa_unit",
                readies[0].swa_controller,
                t.swa.unit,
                t.swa.controller,
            )
            req_id = pp_swa.twin_req_id(pull_req_id(room, t.manager.pp_rank))
            uuid = pp_swa.twin_uuid(pull_req_id(room, t.manager.pp_rank))
            pages = destination_args(pull["swa_tags"], [], pull["swa"])
        else:
            unit_key, controller, dst_unit, dst_controller = (
                "unit",
                readies[0].controller,
                t.unit,
                t.controller,
            )
            req_id = pull_req_id(room, t.manager.pp_rank)
            uuid = pull_uuid(room, t.manager.pp_rank)
            pages = destination_args(pull["full_tags"], pull["full"], [])
        controllers = {r.swa_controller if window else r.controller for r in readies}
        if len(controllers) != 1:
            raise ValueError("Prefix pull: decode ranks disagree on their controller")
        return dict(
            src_units=[SimpleNamespace(**getattr(r, unit_key)) for r in readies],
            dst_units=[dst_unit],
            req_id=req_id,
            uuid=uuid,
            use_block_chunks=True,
            is_sender=True,
            dst_mem_type=t.hbm_mem_type,
            src_controller_address=controller,
            dst_controller_address=dst_controller,
            **pages,
        )

    def _submit(self, room: int, pull: dict, half: dict, *, window: bool) -> None:
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="raiden-prefix-pull"
            )
        kwargs = half.get("kwargs") or self._coordinate_args(room, pull, window=window)
        half["kwargs"] = kwargs
        half["future"] = self._executor.submit(
            self.transfer._coordinate, kwargs["src_controller_address"], **kwargs
        )

    def _harvest(self, room: int, pull: dict) -> None:
        """Collect finished coordinate calls, also of a failed pull: an accepted
        transfer keeps its pages in flight until native completion."""
        for half in (pull["fa"], pull["swa_half"]):
            if half is None or half["future"] is None or not half["future"].done():
                continue
            future, half["future"] = half["future"], None
            try:
                accepted = future.result()
            except Exception as error:
                self._fail(
                    room, COORDINATE_FAILED, f"coordinate_transfer raised: {error}"
                )
                continue
            if accepted:
                half["accepted"] = True
            else:
                half["next_attempt"] = self.clock() + _RETRY_S

    def _advance(self, room: int, pull: dict) -> None:
        t = self.transfer
        now = self.clock()
        expired = now >= pull["deadline"]
        if expired:
            # An accepted transfer may still write the pages: native completion,
            # bounded by its own deadline, settles it; past twice ours, give up.
            if not self.in_flight(room):
                self._fail(room, TIMEOUT, "timed out")
                return
            if now >= pull["deadline"] + t.manager._timeout_s:
                logger.error(
                    "Prefix pull room=%d: an accepted transfer never settled", room
                )
                self._fail(room, TIMEOUT, "accepted transfer never settled")
            return
        if pull["resend_at"] is not None and now >= pull["resend_at"]:
            pull["resend_at"] = None
            self._send_requests(room)
        if len(pull["ready"]) < pull["attn_tp"]:
            return
        if not pull["admitted"]:
            window_pages = t.swa.max_blocks if pull["swa_half"] is not None else 0
            if not t.admission.admit(("pull", room), t.max_blocks, window_pages):
                return
            pull["admitted"] = True
        for half, window in ((pull["fa"], False), (pull["swa_half"], True)):
            if (
                half is not None
                and not half["accepted"]
                and half["future"] is None
                and now >= half["next_attempt"]
            ):
                self._submit(room, pull, half, window=window)

    # ----- native completion ------------------------------------------------

    def record(self, full_status, swa_status) -> None:
        """Reverse transfers each engine settled, split off the forward ones."""
        norm = self.transfer.manager._norm_req_id
        for status, half in ((full_status, "fa"), (swa_status, "swa")):
            if status is None:
                continue
            for kind, settled in zip(("sent", "received", "failed"), status):
                for t in settled:
                    req_id = norm(t.req_id).removeprefix(pp_swa.TWIN_PREFIX)
                    self._settle(*parse_pull_req_id(req_id), half=half, kind=kind)

    def _settle(self, room: int, stage: int, *, half: str, kind: str) -> None:
        offer = self.offers.get((room, stage))
        if offer is not None:
            offer["settled"][half] = kind
            if None not in offer["settled"].values():
                failed = "failed" in offer["settled"].values()
                self._retire_offer((room, stage), failed=failed)
            return
        pull = self.pulls.get(room)
        if pull is None or stage != self.transfer.manager.pp_rank:
            return
        target = pull["fa"] if half == "fa" else pull["swa_half"]
        if target is None:
            return
        target["settled"] = kind
        if kind == "failed":
            self._fail(room, NATIVE_FAILED, f"native {half} transfer failed")
            return
        halves = [h for h in (pull["fa"], pull["swa_half"]) if h is not None]
        if all(h["settled"] == "received" for h in halves) and pull["state"] == PENDING:
            pull["state"] = DONE

    def poll(self) -> None:
        if self.offers:
            self._expire_offers()
        for room, pull in list(self.pulls.items()):
            self._harvest(room, pull)
            if pull["state"] == PENDING:
                self._advance(room, pull)


def _half() -> dict:
    return dict(future=None, accepted=False, next_attempt=0.0, settled=None)
