# SPDX-License-Identifier: Apache-2.0
"""Start the checkpoint page-cache prefetch in the worker bootstrap, before PJRT
and distributed init, so the reads overlap that init instead of waiting for it.

The core loader starts its own prefetch once the model is built; by then the
files this rank prefetched are in page cache and that pass re-reads from memory.
"""

from __future__ import annotations

import glob
import os

from sglang.srt.model_loader.weight_utils import (
    CheckpointFilePrefetchHandle,
    _prefetch_all_checkpoints,
    filter_duplicate_safetensors_files,
)
from sglang_tpu._logging import get_logger

logger = get_logger(__name__)

_handle: CheckpointFilePrefetchHandle | None = None


def start_early_checkpoint_prefetch(cfg) -> CheckpointFilePrefetchHandle | None:
    """``cfg`` is the resolving view of the server args. Returns the running
    handle, or None when the loader's prefetch is off, the load format does not
    read safetensors through mmap, or the path is not a local directory (a repo
    id is downloaded later by the loader)."""
    global _handle
    if _handle is not None:
        return _handle
    if (
        not cfg.weight_loader_prefetch_checkpoints
        or cfg.weight_loader_disable_mmap
        or cfg.load_format not in ("auto", "safetensors")
        or not os.path.isdir(cfg.model_path)
    ):
        return None
    files = sorted(glob.glob(os.path.join(cfg.model_path, "*.safetensors")))
    if not files:
        return None
    files = sorted(
        filter_duplicate_safetensors_files(
            files, cfg.model_path, "model.safetensors.index.json"
        )
    )
    # One stripe per local worker. RANK, not LOCAL_RANK:
    # P/D co-location rewrites LOCAL_RANK to the physical chip.
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("RANK", "0")) % local_world_size
    my_files = files[local_rank::local_world_size]
    if not my_files:
        return None
    # torch.distributed is not up yet, so the helper sees one rank
    # and reads exactly the files it is given.
    _handle = _prefetch_all_checkpoints(
        my_files, num_threads=cfg.weight_loader_prefetch_num_threads
    )
    logger.info(
        "Early checkpoint prefetch started: %d/%d shards, %d threads",
        len(my_files),
        len(files),
        cfg.weight_loader_prefetch_num_threads,
    )
    return _handle


def cancel_early_checkpoint_prefetch() -> None:
    """Cancel the early pass once the loader is done with the files; the daemon
    thread exits at its next block, so this never waits or raises."""
    global _handle
    if _handle is not None:
        _handle.cancel()
        _handle = None
