"""TpuHiCacheL3Backend — HiCacheStorage L3 backend for TPU host pools.

    --hicache-storage-backend tpu_l3
    --hicache-storage-backend-extra-config '{
        "engine": "posix",
        "engine_config": {
            "root_dir": "/mnt/nvme/hicache_l3",
            "num_threads": 4,
            "max_bytes": 0
        }
    }'

Zero-copy v1 contract (cache_controller.py):
    batch_set_v1(keys, host_indices, extra_info) -> List[bool]
    batch_get_v1(keys, host_indices, extra_info) -> List[bool]
host_indices is an int64 CPU tensor of token-slot indices, len(keys) *
page_size long; page i's first token slot is host_indices[i * page_size].
For the TPU page_first_fused host pool, page_id = slot // page_size and the
page is the contiguous kv_buffer[page_id] slab.

The backend never touches device memory and never calls PJRT; it operates
purely on the registered host pool's CPU tensor.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, List, Optional

import torch  # guaranteed wherever sglang runs; engines stay torch-free

from sglang.srt.mem_cache.hicache_storage import HiCacheStorage
from sglang_tpu.storage.tpu_l3.posix_engine import PosixFileEngine

logger = logging.getLogger(__name__)


def _build_engine(engine: str, engine_config: dict):
    if engine == "posix":
        return PosixFileEngine(**engine_config)
    raise ValueError(f"unknown tpu_l3 engine {engine!r}")


class TpuHiCacheL3Backend(HiCacheStorage):
    """HiCacheStorage over whole-page opaque host-pool pages."""

    # Opts into the controller's zero-copy v1 page path (batch_set_v1 /
    # batch_get_v1) without a core-side backend-name list entry.
    supports_zero_copy = True

    def __init__(self, storage_config: Any, kwargs: Optional[dict] = None):
        self.storage_config = storage_config
        cfg = dict(getattr(storage_config, "extra_config", None) or {})
        cfg.update(kwargs or {})
        self.page_size: Optional[int] = None
        self._engine = None
        self._engine_spec = (cfg.get("engine", "posix"), cfg.get("engine_config", {}))
        self._lock = threading.Lock()

        # Every TP rank holds a different KV shard of the same logical page,
        # so ranks must not share storage keys: identical keys with different
        # content would race last-writer-wins and restore another rank's
        # shard. Mirror HiCacheFile's key suffixing exactly.
        model_name = storage_config.model_name
        model_name = "-".join(model_name.split("/")) if model_name else ""
        self.key_suffix = f"_{model_name}"
        if not storage_config.is_mla_model:
            self.key_suffix += f"_{storage_config.tp_rank}_{storage_config.tp_size}"
        if storage_config.pp_size > 1:
            self.key_suffix += f"_{storage_config.pp_size}_{storage_config.pp_rank}"
        if storage_config.attn_cp_size > 1:
            self.key_suffix += (
                f"_cp{storage_config.attn_cp_rank}_{storage_config.attn_cp_size}"
            )

        if cfg.get("eager_engine_init", False):
            self._ensure_engine()

    def _suffix_keys(self, keys: List[str]) -> List[str]:
        return [k + self.key_suffix for k in keys]

    # -- sglang wiring ---------------------------------------------------- #

    def register_mem_pool_host(self, mem_pool_host) -> None:
        # Pages are addressed as kv_buffer[slot // page_size]; that only holds
        # for the TPU fused page-first host pool. Refuse anything else rather
        # than silently backing up and restoring the wrong bytes.
        layout = getattr(mem_pool_host, "layout", None)
        if layout != "page_first_fused":
            raise ValueError(
                f"tpu_l3 requires a page_first_fused host pool, got layout={layout!r}"
            )
        super().register_mem_pool_host(mem_pool_host)
        self.page_size = int(mem_pool_host.page_size)
        self._ensure_engine()

    def _ensure_engine(self) -> None:
        with self._lock:
            if self._engine is None:
                engine, engine_config = self._engine_spec
                # Namespace each rank's files under its key suffix: ranks
                # sharing one root_dir must not index, account, or evict each
                # other's pages. Ranks with identical suffixes (MLA) share the
                # namespace and see each other's files via the stat fallback.
                engine_config = dict(engine_config)
                engine_config["root_dir"] = os.path.join(
                    engine_config["root_dir"], self.key_suffix.lstrip("_") or "default"
                )
                self._engine = _build_engine(engine, engine_config)

    # -- page views -------------------------------------------------------- #

    def _page_view(self, token_slot: int) -> memoryview:
        """Contiguous writable memoryview of the host page owning token_slot."""
        pool = self.mem_pool_host
        page_id = int(token_slot) // int(self.page_size)
        page = pool.kv_buffer[page_id]  # [layer_num, *device_page_shape]
        if not page.is_contiguous():
            raise RuntimeError(
                f"host pool page {page_id} is not contiguous; "
                "page_first_fused layout invariant broken"
            )
        return memoryview(page.view(torch.uint8).reshape(-1).numpy())

    # -- HiCacheStorage v1 (zero-copy) -------------------------------------- #

    def batch_get_v1(
        self, keys: List[str], host_indices, extra_info=None
    ) -> List[bool]:
        self._ensure_engine()
        views = [
            self._page_view(host_indices[i * self.page_size].item())
            for i in range(len(keys))
        ]
        return self._engine.get_pages(self._suffix_keys(keys), views)

    def batch_set_v1(
        self, keys: List[str], host_indices, extra_info=None
    ) -> List[bool]:
        self._ensure_engine()
        views = [
            self._page_view(host_indices[i * self.page_size].item())
            for i in range(len(keys))
        ]
        return self._engine.set_pages(self._suffix_keys(keys), views)

    def batch_exists(self, keys: List[str], extra_info=None) -> int:
        self._ensure_engine()
        return self._engine.exists(self._suffix_keys(keys))

    # -- legacy/generic path (kept functional; controller prefers v1) ------ #

    def get(self, key, target_location=None, target_sizes=None):
        dummy = self.mem_pool_host.get_dummy_flat_data_page()
        view = memoryview(dummy.view(torch.uint8).reshape(-1).numpy())
        if self._engine.get_pages(self._suffix_keys([key]), [view])[0]:
            return dummy
        return None

    def batch_get(self, keys, target_locations=None, target_sizes=None):
        views = [
            memoryview(t.view(torch.uint8).reshape(-1).numpy())
            for t in target_locations
        ]
        results = self._engine.get_pages(self._suffix_keys(keys), views)
        return [t if ok else None for t, ok in zip(target_locations, results)]

    def set(self, key, value=None, target_location=None, target_sizes=None):
        view = memoryview(value.view(torch.uint8).reshape(-1).numpy())
        return self._engine.set_pages(self._suffix_keys([key]), [view])[0]

    def batch_set(self, keys, values=None, target_locations=None, target_sizes=None):
        views = [memoryview(v.view(torch.uint8).reshape(-1).numpy()) for v in values]
        return all(self._engine.set_pages(self._suffix_keys(keys), views))

    def exists(self, key) -> bool:
        return self._engine.exists(self._suffix_keys([key])) == 1

    def clear(self) -> bool:
        # Delete this rank's namespace (index + files). Other ranks' files
        # live under their own suffix directories and are untouched.
        self._ensure_engine()
        return self._engine.clear()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
