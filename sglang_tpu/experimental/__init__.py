from __future__ import annotations

from typing import Callable

import torch

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.envs import envs as tpu_envs

# (host req_pool_indices [bs], host draft hidden [bs, H]) -> None.
PdCarrySeeder = Callable[[torch.Tensor, torch.Tensor], None]

_host_batch_padder: Callable[[int], int] | None = None
_pd_carry_seeder: PdCarrySeeder | None = None


def register_host_batch_padder(padder: Callable[[int], int]) -> None:
    """Set the batch-size bucketing HOST_PADDED batches pad sampling params to."""
    global _host_batch_padder
    _host_batch_padder = padder


def register_pd_carry_seeder(seeder: PdCarrySeeder) -> None:
    """Set the writer a PD decode server seeds the chain draft carry with."""
    global _pd_carry_seeder
    _pd_carry_seeder = seeder


def get_pd_carry_seeder() -> PdCarrySeeder:
    """The PD decode server's chain carry writer; only TpuChainSpecWorker registers one."""
    if _pd_carry_seeder is None:
        raise RuntimeError("No TpuChainSpecWorker registered a PD carry seeder.")
    return _pd_carry_seeder


def pd_decode_draft_prefill() -> bool:
    """Whether a PD decode, not its prefill, runs the draft over each prompt."""
    return tpu_envs.SGLANG_TPU_PD_DECODE_DRAFT_PREFILL.get()


def runs_draft_prompt_pass(disagg_mode: str) -> bool:
    """Whether this server runs the draft over prompts: a PD decode only under the flag."""
    if disagg_mode == "decode":
        return pd_decode_draft_prefill()
    if disagg_mode == "prefill":
        return not pd_decode_draft_prefill()
    return True


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
        or tpu_envs.SGLANG_TPU_HOST_PADDING_NO_D2H.get()
    )


def is_experimental_spec_worker(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether DFLASH/DSPARK is served by sglang_tpu.experimental.TpuBlockSpecWorker."""
    return spec_algorithm.is_dflash_family() and experimental_spec_enabled()


def is_experimental_mtp(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether MTP is served by sglang_tpu.experimental.TpuChainSpecWorker."""
    # Core resolves NEXTN (MTP) to EAGLE; exact, since is_eagle() also matches EAGLE3.
    return spec_algorithm == SpeculativeAlgorithm.EAGLE and experimental_spec_enabled()


def is_experimental_eagle3(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether EAGLE3 is served by sglang_tpu.experimental.TpuChainSpecWorker."""
    return spec_algorithm.is_eagle3() and experimental_spec_enabled()


def is_experimental_chain(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether TpuChainSpecWorker serves this algorithm (MTP or EAGLE3)."""
    return is_experimental_mtp(spec_algorithm) or is_experimental_eagle3(spec_algorithm)


def is_experimental_spec(spec_algorithm: SpeculativeAlgorithm) -> bool:
    """Whether a TpuSpecWorker serves this algorithm (host-padded batches)."""
    return is_experimental_spec_worker(spec_algorithm) or is_experimental_chain(
        spec_algorithm
    )
