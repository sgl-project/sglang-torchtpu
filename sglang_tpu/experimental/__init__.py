from __future__ import annotations

from typing import Callable

from sglang.srt.environ import envs
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


def experimental_spec_enabled() -> bool:
    """Either flag opts in: host-staged metadata without a D2H copy is what this worker serves."""
    return (
        tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.get()
        or envs.SGLANG_TPU_HOST_PADDING_NO_D2H.get()
    )


def is_experimental_spec_worker(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether DFLASH/DSPARK is served by sglang_tpu.experimental.TpuBlockSpecWorker."""
    return spec_algorithm.is_dflash_family() and experimental_spec_enabled()


def is_experimental_eagle3(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether EAGLE3 is served by sglang_tpu.experimental.TpuEagle3Worker."""
    return spec_algorithm.is_eagle3() and experimental_spec_enabled()


def is_experimental_spec(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether a TpuSpecWorker serves this algorithm (host-padded batches)."""
    return is_experimental_spec_worker(spec_algorithm) or is_experimental_eagle3(
        spec_algorithm
    )
