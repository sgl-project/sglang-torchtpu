"""Abort a PD decode resume whose raiden restore DMA failed, on every rank.

With ``SGLANG_RAIDEN_RETRACT_HOST_GB`` set, a retracted request's KV is
restored by a raiden DMA per TP rank. A failed DMA used to raise into the
scheduler on only the ranks it hit, killing the server. ``load_cpu_copy`` now
marks the request instead (``raiden_kv_backup.restore_or_mark_failed``); this
wrapper all-reduces the marks over the attention TP/CP group -- every rank
resumed the same requests in the same order -- and every rank frees and aborts
the marked requests, so only those fail.
"""

from __future__ import annotations

import os
from http import HTTPStatus
from typing import List, Optional

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.decode import DecodePreallocQueue
from sglang.srt.disaggregation.utils import prepare_abort
from sglang.srt.hardware_backend.tpu import raiden_kv_backup
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.common import release_kv_cache

_is_installed = False


def _backup_enabled() -> bool:
    return float(os.environ.get("SGLANG_RAIDEN_RETRACT_HOST_GB", "0") or 0) > 0


def _abort_failed_restores(queue: DecodePreallocQueue, resumed: List[Req]) -> List[Req]:
    failed = raiden_kv_backup.take_failed_restores()
    if not resumed or not _backup_enabled():
        return resumed
    flags = torch.tensor(
        [int(req.kv.req_pool_idx in failed) for req in resumed], dtype=torch.int32
    )
    for group in (queue.gloo_group, queue.attn_cp_cpu_group):
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX, group=group)
    if not flags.any():
        return resumed
    if queue.pp_size > 1:
        # Other pipeline stages restored their layers and would keep the request.
        raise RuntimeError("raiden retraction restore DMA failed under PP")

    kept = []
    for req, bad in zip(resumed, flags.tolist()):
        if not bad:
            kept.append(req)
            continue
        message = f"KV restore after retraction failed for request {req.rid}"
        release_kv_cache(req, queue.tree_cache, is_insert=False)
        prepare_abort(req, message, status_code=HTTPStatus.INTERNAL_SERVER_ERROR)
        queue.scheduler.output_streamer.stream_output([req], req.return_logprob)
    return kept


def install() -> None:
    """Wrap ``DecodePreallocQueue.resume_retracted_reqs``, once per process."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    upstream = DecodePreallocQueue.resume_retracted_reqs

    def resume_retracted_reqs(
        self: DecodePreallocQueue, rids_to_check: Optional[List[str]] = None
    ) -> List[Req]:
        return _abort_failed_restores(self, upstream(self, rids_to_check))

    DecodePreallocQueue.resume_retracted_reqs = resume_retracted_reqs
