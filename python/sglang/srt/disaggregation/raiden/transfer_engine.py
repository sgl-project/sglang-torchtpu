from __future__ import annotations

import logging
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)


class RaidenTransferEngine:
    """Thin adapter over tpu-raiden's ``KVCacheManager``.

    Raiden owns the prefill<->decode network transport (its own control+data
    TCP servers), host staging pool, D2H/H2D DMA, and completion tracking. This
    adapter only normalizes method names and defers the (TPU-only) import of
    ``tpu_raiden`` so the rest of the disaggregation stack imports cleanly on
    hosts without raiden installed.

    Producer (prefill) advertises blocks with ``register_send``; consumer
    (decode) pulls them with ``submit_load``; both drain completion with
    ``poll_finished`` -> (done_sending, done_recving, failed_recving).
    """

    def __init__(
        self,
        kv_caches: List[Any],
        node_id: int,
        local_control_port: int,
        max_blocks: int,
        num_slots: int,
        timeout_s: float = 120.0,
    ):
        from tpu_raiden.api.torch.kv_cache_manager import KVCacheManager

        self._mgr = KVCacheManager(
            kv_caches=kv_caches,
            node_id=node_id,
            local_control_port=local_control_port,
            max_blocks=max_blocks,
            num_slots=num_slots,
            timeout_s=timeout_s,
        )

    @property
    def local_control_port(self) -> int:
        return self._mgr.local_control_port

    @property
    def node_id(self) -> int:
        return self._mgr.node_id

    def register_send(self, req_id: str, uuid: int, block_ids: List[int]) -> bool:
        """Producer: advertise that ``block_ids`` are ready to be read."""
        return bool(self._mgr.register_read(req_id, uuid, block_ids))

    def submit_load(
        self,
        req_id: str,
        uuid: int,
        remote_endpoint: str,
        remote_block_ids: List[int],
        local_block_ids: List[int],
    ) -> int:
        """Consumer: start an async pull of remote blocks into local blocks."""
        result = self._mgr.start_read(
            req_id,
            uuid,
            remote_endpoint,
            remote_block_ids,
            local_block_ids,
        )
        return int(result) if result is not None else 0

    def poll_finished(self) -> Tuple[List[str], List[str], List[str]]:
        """Return (done_sending, done_recving, failed_recving) req-id lists."""
        return self._mgr.poll_stats()
