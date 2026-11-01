# SPDX-License-Identifier: Apache-2.0
"""Load local checkpoints through the host's weight feeder.

Why: every TPU rank reads the whole checkpoint, so per-rank reads fetch each
     shard once per rank. sglang_tpu.weight_feeder fetches it once per host.
How: AROUND the safetensors iterators, for local paths.
     BEFORE Engine._launch_scheduler_processes, to fetch while ranks start up.
Upstream PR or issue: <to be filed>.
Remove when: core exposes a platform seam for the checkpoint weight iterators and
     for work that runs ahead of the ranks.
"""

import logging
from collections.abc import Iterator

import torch

logger = logging.getLogger(__name__)


def weight_feeder_load_local_weights(
    original, hf_weights_files: list[str], *args, **kwargs
) -> Iterator[tuple[str, torch.Tensor]]:
    """Read local checkpoints through the feeder when staging can hold them."""
    from sglang_tpu.envs import envs
    from sglang_tpu.weight_feeder import (
        can_use_shared_staging,
        shared_weights_iterator,
    )

    num_shards = len(hf_weights_files)
    enabled = envs.SGLANG_TPU_WEIGHT_FEEDER_ENABLED.get()
    if hf_weights_files and enabled and can_use_shared_staging(hf_weights_files):
        logger.info("Loading %d local shards through the weight feeder", num_shards)
        return shared_weights_iterator(
            hf_weights_files, lambda files: original(files, *args, **kwargs)
        )
    logger.info(
        "Loading %d local shards with the core iterator (feeder enabled: %s)",
        num_shards,
        enabled,
    )
    return original(hf_weights_files, *args, **kwargs)


def weight_feeder_prefetch(*args, **kwargs) -> None:
    """Start fetching weights before the ranks launch.

    Rank startup is slow and does not need the weights. Fetching during it has shards
    staged by the time ranks load, which shortens engine startup. If this fails, ranks
    start their own feeder.
    """
    from sglang_tpu.weight_feeder import start_weight_prefetch

    try:
        start_weight_prefetch()
    except Exception:
        logger.exception("Could not start the weight feeder early")
