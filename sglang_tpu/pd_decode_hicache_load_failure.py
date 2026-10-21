"""Clean up a PD decode HiCache load-back whose raiden DMA failed.

Instead of stopping the server, every rank settles the failure together
(``resolve_load_back``), the requests bound to the load are aborted through the
decode transfer queue's failure path, and the nodes the load brought back are
demoted to host-only once nothing else holds them. Freeing their pages relies
on a failed DMA meaning its copies have ended (google/tpu-sync#1228). Prefill
servers are unchanged: a failed load still stops the server.
"""

from __future__ import annotations

import logging
from typing import List

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    HiCacheRestoreResult,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang_tpu.raiden_hybrid_cache_controller import TpuRaidenHybridCacheController
from sglang_tpu.raiden_transfer import LoadBackEvent

logger = logging.getLogger(__name__)

_is_installed = False


def _raiden_controller(tree_cache):
    controller = getattr(tree_cache, "cache_controller", None)
    if isinstance(controller, TpuRaidenHybridCacheController):
        return controller
    return None


def _loaded_nodes(tree_core) -> dict:
    """Anchor node id -> the nodes its load-back brings back, ancestors first."""
    return tree_core.__dict__.setdefault("_tpu_load_back_nodes", {})


def _fail_restores_of_failed_loads(transfer_queue, decode_reqs) -> None:
    """A restore whose load failed is FAILED, not READY; the transfer queue then
    aborts the request. A load's counter slot is not reused before its ack is
    reaped (``has_free_load_back_slot``), so the slot still holds its event."""
    events = transfer_queue.tree_cache.cache_controller.layer_done_counter.events
    for dr in decode_reqs:
        if dr.hicache_restore_status != HiCacheRestoreResult.READY:
            continue
        if dr.hicache_load_consumer_index < 0:
            continue
        event = events[dr.hicache_load_consumer_index]
        if isinstance(event, LoadBackEvent) and event.failed:
            dr.hicache_restore_status = HiCacheRestoreResult.FAILED


def _release_failed_loads(cache, controller) -> None:
    """Settle the head load acks; release a failed one for acking once only the
    load still locks its nodes. A collective: every rank walks the same acks
    and stops at the same one (the tree is the same on every rank)."""
    loaded = _loaded_nodes(cache.tree_core)
    for ack in controller.ack_load_queue:
        event = ack.finish_event
        if not controller.resolve_load_back(event, cache._all_reduce):
            return
        if not isinstance(event, LoadBackEvent) or not event.failed or event.released:
            continue
        # Only the Rust tree core reports lock counts; without it this raises
        # and a failed load stays fatal.
        binding = cache.tree_core._binding
        nodes = [n for anchor in ack.node_ids for n in loaded.get(anchor, [anchor])]
        if any(
            binding.inspect_get_component_device_lock_ref(n, int(ComponentType.FULL))
            != 1
            for n in nodes
        ):
            return  # an aborted request has not dropped its lock yet
        event.released = True


def _roll_back(cache, anchor: int, nodes: List[int]) -> None:
    """Demote the nodes a failed load brought back, deepest first."""
    binding = cache.tree_core._binding
    tracker: dict = {}
    for node_id in reversed(nodes):
        if not binding.inspect_is_device_evictable_leaf(node_id):
            raise RuntimeError(
                f"raiden HiCache load-back of node {anchor} failed and node "
                f"{node_id} is still in use, so its device copy cannot be dropped"
            )
        cache._demote(node_id, tracker)
    logger.warning(
        "raiden HiCache load-back of node %d failed: %d node(s) rolled back to "
        "host-only",
        anchor,
        len(nodes),
    )


def _loading_check_with_cleanup(orig_loading_check, cache, finish_count) -> None:
    """Upstream ``loading_check``, with a failed load held, acked, then rolled back."""
    controller = cache.cache_controller
    _release_failed_loads(cache, controller)
    pending = list(controller.ack_load_queue)
    orig_loading_check(cache, finish_count)
    # Upstream acks loads from the head of the queue.
    acked = pending[: len(pending) - len(controller.ack_load_queue)]
    loaded = _loaded_nodes(cache.tree_core)
    for ack in acked:
        for anchor in ack.node_ids:
            nodes = loaded.pop(anchor, None) or [anchor]
            if isinstance(ack.finish_event, LoadBackEvent) and ack.finish_event.failed:
                _roll_back(cache, anchor, nodes)
    # Loads that never started leave a record behind; keep only live ones.
    for anchor in [a for a in loaded if a not in cache.ongoing_load_back]:
        del loaded[anchor]


def install() -> None:
    """Patch the decode restore poll, ``loading_check`` and load-back recording."""
    global _is_installed
    if _is_installed:
        return
    _is_installed = True

    orig_process = DecodeHiCacheTransferMixin._process_hicache_local_restores
    orig_loading_check = UnifiedRadixCache.loading_check

    def _process_hicache_local_restores(self, decode_reqs):
        controller = _raiden_controller(self.tree_cache)
        if controller is None:
            return orig_process(self, decode_reqs)
        # Set before the poll's phase C issues a load, so it can fail without
        # raising and its nodes are recorded for the rollback.
        controller.handles_load_back_failures = True
        self.tree_cache.tree_core.__dict__["_tpu_record_load_back_nodes"] = True
        orig_process(self, decode_reqs)
        _fail_restores_of_failed_loads(self, decode_reqs)

    def loading_check(self, finish_count=None):
        controller = _raiden_controller(self)
        if controller is None or not controller.handles_load_back_failures:
            return orig_loading_check(self, finish_count)
        _loading_check_with_cleanup(orig_loading_check, self, finish_count)

    DecodeHiCacheTransferMixin._process_hicache_local_restores = (
        _process_hicache_local_restores
    )
    UnifiedRadixCache.loading_check = loading_check
    _record_load_back_nodes()


def _record_load_back_nodes() -> None:
    """Record which nodes each load-back brings back, for the rollback.

    Importing the Rust tree core builds its extension; without it, a failed
    load stays fatal.
    """
    try:
        from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
    except (ImportError, RuntimeError) as e:
        logger.info("Rust tree core unavailable (%s); failed load-backs stay fatal", e)
        return
    orig_build_load_back_spec = RustUnifiedTreeCore.build_load_back_spec

    def build_load_back_spec(self, node_id, req=None):
        kv_xfer, comp_xfers = orig_build_load_back_spec(self, node_id, req=req)
        if self.__dict__.get("_tpu_record_load_back_nodes"):
            _loaded_nodes(self)[node_id] = list(kv_xfer.nodes_to_load or [node_id])
        return kv_xfer, comp_xfers

    RustUnifiedTreeCore.build_load_back_spec = build_load_back_spec
