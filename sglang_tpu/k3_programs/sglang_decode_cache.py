# SPDX-License-Identifier: Apache-2.0
"""A P/D decode server's prefix cache under K3_PROGRAM: KV pages only.

vLLM-TorchTPU's 1P1D decode keeps each request's pages when it finishes and,
for the session's next turn, pulls only the suffix past its local prefix hit
(``TPU_RAIDEN_PREFIX_AWARE_LOAD``). sglang's decode radix cache does the same
through ``decode_prefix_len``: the decode matches a prefix in its own tree and
registers only the pages past it, which Raiden pairs with the prefill's last
pages (``RaidenKVManager._pair_blocks``).

For a hybrid model sglang keeps Mamba state in the tree (its ``MambaComponent``
matches, locks and copies it), but the decode program needs none: the P/D
transfer delivers every request's KDA state after its whole prompt into its
slot, and the program keeps the live state in its own slots. A state copied
from the tree into the slot would also race the transfer writing it. So the
decode tree's Mamba component here holds no states: every node is a valid
match boundary, a hit copies nothing, and a finished request's slot goes back
to the pool.
"""

from __future__ import annotations

from typing import Any, Optional

from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
    EvictLayer,
    TreeComponent,
)


class K3DecodeStateComponent(TreeComponent):
    """The decode tree's Mamba component: no states (see the module docstring)."""

    component_type = ComponentType.MAMBA

    def create_match_validator(self, match_device_only: bool = False):
        return lambda node: True

    def refresh_lru(self, phase, node, root_node) -> None:
        return None

    def finalize_match_result_in_cache(self, params, result):
        return result

    def redistribute_on_node_split(self, new_parent, child) -> None:
        return None

    def evict_component(
        self, node, device_frees, host_frees, target=EvictLayer.DEVICE
    ) -> tuple[int, int]:
        return 0, 0

    def _evict_device_start(self, request_cnt: int) -> None:
        return None

    def _evict_device_next_node(self, tracker, device_frees, host_frees):
        return None

    def _evict_device_end(self) -> None:
        return None

    def acquire_component_lock(self, node, result, lock_host: bool = False):
        return result

    def release_component_lock(self, node, params, lock_host: bool = False) -> None:
        return None

    def _dec_session_coverage(self, session_id: str, leaf) -> None:
        return None

    def _advance_session_coverage(self, session_id: str, leaf, old_ancestor) -> None:
        return None

    def _recede_session_coverage(self, session_id: str, leaf, fallback) -> None:
        return None

    def prepare_for_caching_req(
        self, req, insert_params, token_ids_len: int, is_finished: bool
    ) -> Optional[int]:
        return None  # the KV prefix decides the cached length

    def cleanup_after_caching_req(
        self,
        req,
        is_finished: bool,
        insert_result: Any = None,
        insert_params: Any = None,
    ) -> None:
        if is_finished and req.kv.mamba_pool_idx is not None:
            self.cache.req_to_token_pool.free_mamba_cache(req)


def install(params: Any) -> None:
    """Make a unified tree built from ``params`` use the stateless component."""
    params.component_registry_override = {
        **(getattr(params, "component_registry_override", None) or {}),
        ComponentType.MAMBA: K3DecodeStateComponent,
    }
