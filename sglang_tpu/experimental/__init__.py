from __future__ import annotations

from typing import Callable

import torch

from sglang_tpu.spec_execution import (  # noqa: F401 - retained public predicates
    experimental_spec_enabled,
    is_experimental_chain,
    is_experimental_eagle3,
    is_experimental_mtp,
    is_experimental_spec,
    is_experimental_spec_worker,
)

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


def uses_host_batch(batch) -> bool:
    """Whether batch is a HOST_PADDED TpuSpecWorker batch (the init_new hook set its padder)."""
    return getattr(batch, "host_pad_bs", None) is not None


def get_host_batch_padder() -> Callable[[int], int] | None:
    """The TpuSpecWorker's batch bucketing, or None when no TpuSpecWorker serves this process."""
    return _host_batch_padder
