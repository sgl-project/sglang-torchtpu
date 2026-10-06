"""Thin adapter for the native Raiden KV cache transfer engine.

Construction and transfer lifecycle decisions stay with ``RaidenKVManager``.
This adapter forwards arguments, return values, and exceptions unchanged.
"""

from __future__ import annotations

from typing import Any


class RaidenEngine:
    """Wrap one native KVCacheManager for one page-index namespace."""

    def __init__(self, native: Any) -> None:
        self._engine = native

    @property
    def transfer_address(self):
        return self._engine.transfer_address

    def register_read(self, req_id, uuid, block_ids):
        return self._engine.register_read(req_id, uuid, block_ids)

    def start_read(self, req_id, uuid, endpoint, remote_blocks, local_blocks):
        return self._engine.start_read(
            req_id, uuid, endpoint, remote_blocks, local_blocks
        )

    def register_active_plan(self, uuid, request, *, is_sender):
        return self._engine.register_active_plan(uuid, request, is_sender=is_sender)

    def plan_host_blocks(self, uuid, blocks):
        return self._engine.plan_host_blocks(uuid, blocks)

    def d2h(self, blocks, host_blocks, live):
        return self._engine.d2h(blocks, host_blocks, live)

    def push_registered_plan(self, uuid, endpoint, source, destination, layer_index):
        return self._engine.push_registered_plan(
            uuid, endpoint, source, destination, layer_index
        )

    def poll_stats(self):
        return self._engine.poll_stats()

    def unregister_active_plan(self, uuid):
        return self._engine.unregister_active_plan(uuid)
