"""Decide ``--max-queued-requests`` for a PD pair at the decode's door (#1188, option 2).

Both halves of a PD pair receive a request at the same time. Upstream applies
the queue limit only at tokenizer intake, which a decode's requests bypass,
and enforcing it when transferred requests join the waiting queue would
discard a finished prefill. Here the decode decides in ``_init_receiver``,
once the prefill ranks of the room are known and before anything is
pre-allocated: the client gets a 503 from the decode, and every prefill rank
gets the room's ABORT carrying the reason, so the twin is finished with the
same 503 while it still waits for the handshake. The PD router relays the
prefill's status, which is why the prefill side answers 503 as well.

Under raiden the prefill starts computing only once the decode asks to pull
the room's KV, which a refused request never does, so ``init()`` is safe to
call before refusing.
"""

from __future__ import annotations

from http import HTTPStatus
from typing import TYPE_CHECKING

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.common.conn import AbortNotification
from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.prefill import (
    SchedulerDisaggregationPrefillMixin,
    maybe_release_metadata_buffer,
)
from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestOutcome
from sglang.srt.mem_cache.common import release_kv_cache
from sglang_tpu._logging import get_logger

if TYPE_CHECKING:
    from sglang.srt.disaggregation.decode import DecodeRequest
    from sglang.srt.managers.schedule_batch import Req

logger = get_logger(__name__)

QUEUE_FULL = "The request queue is full."
#: Extra ABORT frame; the raiden prefill records it as the room's failure reason.
REASON_FRAME = b"reason:" + QUEUE_FULL.encode("ascii")

_is_installed = False
_upstream_init_receiver = DecodePreallocQueue._init_receiver
_upstream_handle_bootstrap_failure = (
    SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure
)


def backlog(queue: DecodePreallocQueue) -> int:
    """Requests ahead of the one being initialised that cannot start next step."""
    scheduler = queue.scheduler
    pending = (
        len(queue.queue)
        - 1
        + len(scheduler.disagg_decode_transfer_queue.queue)
        + len(scheduler.waiting_queue)
        + len(queue.retracted_queue)
    )
    free_slots = max(
        0,
        min(scheduler.req_to_token_pool.size, scheduler.max_running_requests)
        - scheduler.running_batch.batch_size(),
    )
    return pending - free_slots


def notify_prefill(receiver) -> None:
    """Send the room's ABORT with the reason to every prefill rank, then fail it locally."""
    kv_mgr = receiver.kv_mgr
    frames = AbortNotification(
        room=receiver.bootstrap_room,
        decode_ip=kv_mgr.local_ip,
        decode_port=kv_mgr.rank_port,
    ).to_zmq() + [REASON_FRAME]
    for info in receiver.bootstrap_infos or ():
        try:
            sock, lock = receiver._connect_to_bootstrap_server(info)
            with lock:
                sock.send_multipart(frames)
        except Exception as error:  # best effort, as abort() itself
            logger.debug(
                "door refusal: ABORT for room %s not delivered to %s: %s",
                receiver.bootstrap_room,
                info,
                error,
            )
    kv_mgr.record_failure(receiver.bootstrap_room, QUEUE_FULL)
    kv_mgr.update_status(receiver.bootstrap_room, KVPoll.Failed)
    receiver.conclude_state = KVPoll.Failed
    receiver.abort_notified = True


def _init_receiver(
    self: DecodePreallocQueue, decode_req: DecodeRequest, prefill_dp_rank: int
) -> None:
    limit = self.scheduler.max_queued_requests
    if limit is None or decode_req.is_rebootstrap or backlog(self) + 1 <= limit:
        return _upstream_init_receiver(self, decode_req, prefill_dp_rank)
    receiver = decode_req.kv_receiver
    logger.info(
        "door refusal: room %s rid %s (backlog %d, limit %d)",
        receiver.bootstrap_room,
        decode_req.req.rid,
        backlog(self),
        limit,
    )
    receiver.init(prefill_dp_rank)
    if receiver.conclude_state != KVPoll.Failed:
        notify_prefill(receiver)
    if decode_req in self.queue:
        self.queue.remove(decode_req)
    prepare_abort(
        decode_req.req, QUEUE_FULL, status_code=HTTPStatus.SERVICE_UNAVAILABLE
    )
    self.scheduler.retire_unadmitted_request(decode_req.req)


def handle_bootstrap_failure(self, req: Req) -> None:
    """A room the decode refused at its door ends with the same 503, quietly."""
    kv_mgr = getattr(req.disagg_kv_sender, "kv_mgr", None)
    records = getattr(kv_mgr, "failure_records", None)
    if records is None or records.get(req.bootstrap_room) != QUEUE_FULL:
        return _upstream_handle_bootstrap_failure(self, req)
    logger.info(
        "decode refused room %s (%s), %s", req.bootstrap_room, QUEUE_FULL, req.rid
    )
    req.time_stats.trace_ctx.abort(abort_info={"reason": QUEUE_FULL})
    if req.kv.holds_kv or req.kv.holds_mamba:
        release_kv_cache(req, self.tree_cache, is_insert=False)
    maybe_release_metadata_buffer(req, self.req_to_metadata_buffer_idx_allocator)
    req.pending_bootstrap = False
    prepare_abort(req, QUEUE_FULL, status_code=HTTPStatus.SERVICE_UNAVAILABLE)
    self.output_streamer.stream_output([req], req.return_logprob)
    self.tree_cache.finish(req.cache_request_handle, CacheRequestOutcome.ABORT)


def install() -> None:
    """Replace the decode's receiver init and the prefill's bootstrap-failure handler, once per process."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    logger.info(
        "door admission installed (PD --max-queued-requests decided at the decode's door)"
    )
    DecodePreallocQueue._init_receiver = _init_receiver
    SchedulerDisaggregationPrefillMixin.handle_bootstrap_failure = (
        handle_bootstrap_failure
    )
