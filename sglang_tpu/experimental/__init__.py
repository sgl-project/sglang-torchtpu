from __future__ import annotations

from typing import Callable

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.envs import envs as tpu_envs

_host_batch_padder: Callable[[int], int] | None = None


def register_host_batch_padder(padder: Callable[[int], int]) -> None:
    """Set the batch-size bucketing HOST_PADDED batches pad sampling params to."""
    global _host_batch_padder
    _host_batch_padder = padder


def get_host_batch_padder() -> Callable[[int], int]:
    assert _host_batch_padder is not None, "TpuSpecWorker has not registered a padder"
    return _host_batch_padder


def is_experimental_dspark(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether DSPARK is served by sglang_tpu.experimental.TpuDsparkWorker."""
    return (
        spec_algorithm.is_dspark()
        and tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.get()
    )
