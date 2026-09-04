"""A thin wrapper for the underlying KV cache transfer engine."""

from __future__ import annotations

import logging
from typing import Any

import msgspec

logger = logging.getLogger(__name__)


class RaidenEngineConfig(msgspec.Struct, frozen=True):
    """Complete construction inputs for one page-index namespace."""

    kv_caches: list[Any]
    node_id: int
    local_control_port: int
    endpoint: str
    max_blocks: int
    num_slots: int
    timeout_s: float
    label: str
    rank: int


class RaidenEngine:
    """Stateful wrapper around exactly one native Raiden KV cache manager."""

    def __init__(self, native: Any, *, config: RaidenEngineConfig) -> None:
        self._kv_caches = list(config.kv_caches)
        self.endpoint = config.endpoint
        self.max_blocks = config.max_blocks
        self.num_slots = config.num_slots
        self.timeout_s = config.timeout_s
        self._engine = native
        logger.info(
            "RaidenEngine rank%d %s up | endpoint=%s max_blocks=%d num_slots=%d",
            config.rank,
            config.label,
            config.endpoint,
            config.max_blocks,
            config.num_slots,
        )

    @classmethod
    def create(cls, config: RaidenEngineConfig) -> RaidenEngine:
        if not config.kv_caches:
            raise RuntimeError("Raiden requires at least one KV cache tensor")

        from tpu_sync.api.torch.kv_cache_manager import KVCacheManager

        native = KVCacheManager(
            kv_caches=list(config.kv_caches),
            node_id=config.node_id,
            local_control_port=config.local_control_port,
            max_blocks=config.max_blocks,
            num_slots=config.num_slots,
            timeout_s=config.timeout_s,
        )
        return cls(native, config=config)

    @property
    def transfer_address(self) -> str:
        return self._engine.transfer_address

    @property
    def kv_layer_count(self) -> int:
        return len(self._kv_caches)

    def kv_tensor(self, index: int) -> Any:
        return self._kv_caches[index]

    def register_read(self, req_id, uuid, block_ids) -> bool:
        return bool(self._engine.register_read(req_id, uuid, block_ids))

    def start_read(self, req_id, uuid, endpoint, remote_blocks, local_blocks) -> None:
        self._engine.start_read(req_id, uuid, endpoint, remote_blocks, local_blocks)

    def register_plan(self, uuid, request, *, is_sender: bool) -> None:
        self._engine.register_active_plan(uuid, request, is_sender=is_sender)

    def plan_host_blocks(self, uuid, blocks):
        return self._engine.plan_host_blocks(uuid, blocks)

    def stage_to_host(self, blocks, host_blocks, live):
        return self._engine.d2h(blocks, host_blocks, live)

    def push_plan(self, uuid, endpoint, source, destination, layer_index) -> None:
        self._engine.push_registered_plan(
            uuid, endpoint, source, destination, layer_index
        )

    def poll(self) -> tuple[list[str], list[str], list[str]]:
        return self._engine.poll_stats()

    def unregister_plan(self, uuid: int) -> None:
        """Drop a settled plan, tolerating native timeout cleanup."""
        try:
            self._engine.unregister_active_plan(uuid)
        except RuntimeError as exc:
            if "not registered" not in str(exc):
                raise
