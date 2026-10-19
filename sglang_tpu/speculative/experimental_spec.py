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


def uses_host_batch(batch) -> bool:
    """Whether batch is a HOST_PADDED TpuSpecWorker batch (the init_new hook set its padder)."""
    return getattr(batch, "host_pad_bs", None) is not None


def get_host_batch_padder() -> Callable[[int], int] | None:
    """The TpuSpecWorker's batch bucketing, or None when no TpuSpecWorker serves this process."""
    return _host_batch_padder


def experimental_spec_enabled() -> bool:
    """Either flag opts in: host-staged metadata without a D2H copy is what this worker serves."""
    return (
        tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.get()
        or envs.SGLANG_TPU_HOST_PADDING_NO_D2H.get()
    )


def is_experimental_spec_worker(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether DFLASH/DSPARK is served by sglang_tpu.speculative.block_spec_worker.TpuBlockSpecWorker."""
    return spec_algorithm.is_dflash_family() and experimental_spec_enabled()


def is_experimental_mtp(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether MTP is served by sglang_tpu.speculative.chain_spec_worker.TpuChainSpecWorker."""
    # Core resolves NEXTN (MTP) to EAGLE; exact, since is_eagle() also matches EAGLE3.
    return spec_algorithm == SpeculativeAlgorithm.EAGLE and experimental_spec_enabled()


def is_experimental_chain(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether TpuChainSpecWorker serves this algorithm."""
    return is_experimental_mtp(spec_algorithm)


def is_experimental_eagle3(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether EAGLE3 is served by sglang_tpu.speculative.eagle.eagle_worker.TpuEagle3Worker."""
    return spec_algorithm.is_eagle3() and experimental_spec_enabled()
