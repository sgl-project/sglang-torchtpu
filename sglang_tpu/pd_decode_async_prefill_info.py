"""Fetch PD prefill parallel info off the decode scheduler thread.

Upstream ``DecodePreallocQueue._ensure_prefill_info`` calls
``try_ensure_parallel_info`` -- a blocking HTTP GET with a 5 s timeout -- on the
scheduler thread, retried up to 15 times. A prefill that accepts the connection
but never answers stalls every decode batch for each attempt (~76 s per cycle).

This replacement submits the fetch to the existing prefill-recompute executor
and polls it each pass. Cached info still resolves immediately; the retry
interval, attempt limit and abort path are upstream's. TP ranks stay in step
without a new collective: each request stays in the decode queue on every rank
and the per-pass status all-reduce holds it until every rank has its info.
"""

from __future__ import annotations

import time
from concurrent.futures import Future
from typing import Dict, List, Tuple

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeRequest
from sglang_tpu._logging import get_logger

logger = get_logger(__name__)

_is_installed = False


def _ensure_prefill_info(
    queue: DecodePreallocQueue, addr_to_reqs: Dict[str, List[DecodeRequest]]
) -> Tuple[Dict[str, List[DecodeRequest]], List[DecodeRequest]]:
    """Non-blocking ensure parallel info for each addr.
    Returns (ready_addrs, remaining_reqs)."""
    inflight: Dict[str, Future] = queue.__dict__.setdefault("_ensure_inflight", {})
    ready: Dict[str, List[DecodeRequest]] = {}
    remaining: List[DecodeRequest] = []

    now = time.monotonic()
    for bootstrap_addr, reqs in addr_to_reqs.items():
        fetch = inflight.get(bootstrap_addr)
        if fetch is None:
            if bootstrap_addr not in queue.kv_manager.prefill_info_table:
                last_attempt = queue._ensure_last_attempt_time.get(bootstrap_addr)
                if last_attempt is None or (
                    now - last_attempt >= queue._ensure_retry_interval
                ):
                    queue._ensure_last_attempt_time[bootstrap_addr] = now
                    inflight[bootstrap_addr] = (
                        queue.kv_manager._ensure_prefill_recompute_executor().submit(
                            queue.kv_manager.try_ensure_parallel_info, bootstrap_addr
                        )
                    )
                remaining.extend(reqs)
                continue
            fetched = True
        elif not fetch.done():
            remaining.extend(reqs)
            continue
        else:
            del inflight[bootstrap_addr]
            # Re-raises a prefill/decode config mismatch on this thread.
            fetched = fetch.result()

        if fetched:
            queue._ensure_retry_count.pop(bootstrap_addr, None)
            queue._ensure_last_attempt_time.pop(bootstrap_addr, None)
            ready[bootstrap_addr] = reqs
            continue

        count = queue._ensure_retry_count.get(bootstrap_addr, 0) + 1
        queue._ensure_retry_count[bootstrap_addr] = count

        if count >= queue._max_ensure_retries:
            logger.error(
                f"Could not fetch prefill parallel info from {bootstrap_addr} "
                f"after {count} attempts"
            )
            for decode_req in reqs:
                # kv_receiver may be None from a prior queue cleanup
                if decode_req.kv_receiver is not None:
                    decode_req.kv_receiver.abort()
            queue._ensure_retry_count.pop(bootstrap_addr, None)
            queue._ensure_last_attempt_time.pop(bootstrap_addr, None)
        else:
            remaining.extend(reqs)

    return ready, remaining


def install() -> None:
    """Replace ``DecodePreallocQueue._ensure_prefill_info``, once per process."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    DecodePreallocQueue._ensure_prefill_info = _ensure_prefill_info
