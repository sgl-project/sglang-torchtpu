"""Name a decode request's raw ring page for the Raiden pool transfer.

The pooled DSA indexer's raw ring (SGLANG_TPU_DSA_KPOOL_RAW_RING) keeps a
request's raw key|gate rows by request slot, so a P/D transfer cannot land
them by page id. The pool transfer moves the prefill's last page of raw rows,
which holds the open pool, the only rows read back, into the ring page of the
request's last token (pool_spans.last_page_spans). Core's decode hands the
receiver page ids and state rows only, so this wraps
``DecodePreallocQueue._send_kv_metadata`` to name that ring page first.
"""

from __future__ import annotations

import functools
from typing import Any, Callable

from sglang.srt.disaggregation.decode import DecodePreallocQueue

_is_installed = False


def _send_kv_metadata(
    upstream: Callable, queue: DecodePreallocQueue, decode_req: Any, *args, **kwargs
) -> Any:
    from sglang_tpu.attention_pallas_mla_sparse import _split_pool

    pool = _split_pool(queue.token_to_kv_pool)
    rows = kwargs.get("receive_rows")
    if getattr(pool, "raw_ring_pages_cnt", 0) and rows is not None:
        _, fill_len = rows
        decode_req.kv_receiver.raw_ring_page = pool.raw_ring_page(
            int(decode_req.req.kv.req_pool_idx), int(fill_len) - 1
        )
    return upstream(queue, decode_req, *args, **kwargs)


def install() -> None:
    global _is_installed
    if _is_installed:
        return
    _is_installed = True
    upstream = DecodePreallocQueue._send_kv_metadata

    @functools.wraps(upstream)
    def wrapper(*args, **kwargs):
        return _send_kv_metadata(upstream, *args, **kwargs)

    DecodePreallocQueue._send_kv_metadata = wrapper
