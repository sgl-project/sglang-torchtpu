"""What an engine worker sets up for RL training, and the weight transports it serves.

The TPU platform calls ``setup_worker`` from ``init_backend`` and serves
``weight_update_handlers`` through /plugin_weight_update; a new weight transport adds
its handler here.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional

import msgspec

from sglang_tpu.rl import shared_slice
from sglang_tpu.rl.ici_receiver import ICI_RECEIVE_HANDLER, receive_bucket


class RlSetup(msgspec.Struct, frozen=True):
    # the TPU slice this worker shares with an RL trainer (SGLANG_TPU_SHARED_SLICE)
    shared_slice: Optional[shared_slice.SharedSlice]


def setup_worker(*, job_rank: int, job_size: int) -> RlSetup:
    """Must run after importing torch_tpu and before the worker's first TPU op."""
    spec = shared_slice.SharedSliceSpec.from_env()
    return RlSetup(
        shared_slice=(
            None
            if spec is None
            else shared_slice.join_shared_slice(
                spec, job_rank=job_rank, job_size=job_size
            )
        )
    )


def weight_update_handlers(setup: RlSetup) -> dict[str, Callable[..., Any]]:
    """The weight transports this worker serves, by /plugin_weight_update handler name."""
    handlers = {}
    if setup.shared_slice is not None:
        handlers[ICI_RECEIVE_HANDLER] = functools.partial(
            receive_bucket, shared_slice=setup.shared_slice
        )
    return handlers
