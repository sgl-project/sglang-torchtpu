# SPDX-License-Identifier: Apache-2.0
"""Lazy worker/policy bindings shared by TPU serving and compilation."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import msgspec

from sglang.srt.environ import envs
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _load_class(path: str) -> type:
    module, name = path.split(":")
    return getattr(import_module(module), name)


class SpecExecutionBinding(msgspec.Struct, frozen=True):
    worker_path: str
    policy_path: str | None = None
    host_padded: bool = False
    block_worker: bool = False
    chain_worker: bool = False

    def worker_cls(self, *, default: type) -> type:
        if not self.host_padded:
            from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

            # Compatibility adapts the shared V2 driver only; other defaults stay intact.
            if default is not EAGLEWorkerV2:
                return default
        return _load_class(self.worker_path)

    def compile_policy_mixin(self) -> type | None:
        return _load_class(self.policy_path) if self.policy_path is not None else None


_COMPAT_WORKER = "sglang_tpu.experimental.speculative.eagle.draft_worker:TpuEagleWorker"
_COMPATIBILITY = SpecExecutionBinding(worker_path=_COMPAT_WORKER)
_EAGLE = SpecExecutionBinding(
    worker_path=_COMPAT_WORKER,
    policy_path="sglang_tpu.compilation.spec_decode_compile_policy_eagle3:Eagle3SpecDecodeCompilePolicyMixin",
)
_DFLASH = SpecExecutionBinding(
    worker_path=_COMPAT_WORKER,
    policy_path="sglang_tpu.compilation.spec_decode_compile_policy_dflash:DFlashSpecDecodeCompilePolicyMixin",
)
_BLOCK = SpecExecutionBinding(
    worker_path="sglang_tpu.experimental.speculative.block_spec_worker:TpuBlockSpecWorker",
    policy_path="sglang_tpu.experimental.compilation.block_compile_policy:TpuBlockSpecDecodeCompilePolicyMixin",
    host_padded=True,
    block_worker=True,
)
_CHAIN = SpecExecutionBinding(
    worker_path="sglang_tpu.experimental.speculative.chain_spec_worker:TpuChainSpecWorker",
    policy_path="sglang_tpu.experimental.compilation.chain_compile_policy:TpuChainSpecDecodeCompilePolicyMixin",
    host_padded=True,
    chain_worker=True,
)


def experimental_spec_enabled() -> bool:
    return (
        tpu_envs.SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER.get()
        or envs.SGLANG_TPU_HOST_PADDING_NO_D2H.get()
    )


def _is_mtp(algorithm: SpeculativeAlgorithm) -> bool:
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    # Core resolves NEXTN (MTP) to EAGLE; exact, since is_eagle() also matches EAGLE3.
    return algorithm is SpeculativeAlgorithm.EAGLE


def resolve_spec_execution(
    algorithm: SpeculativeAlgorithm, *, pp_size: int
) -> SpecExecutionBinding:
    # Use the configured target width, never draft_pp_context's temporary PP=1 group.
    if pp_size == 1 and experimental_spec_enabled():
        if algorithm.is_dflash_family():
            return _BLOCK
        if algorithm.is_eagle3() or _is_mtp(algorithm):
            return _CHAIN
    if algorithm.is_dflash():
        return _DFLASH
    if algorithm.is_eagle() and not algorithm.is_frozen_kv_mtp():
        return _EAGLE
    return _COMPATIBILITY


def is_experimental_spec_worker(algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether DFLASH/DSPARK is served by TpuBlockSpecWorker."""
    return resolve_spec_execution(algorithm, pp_size=pp_size).block_worker


def is_experimental_chain(algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether TpuChainSpecWorker serves this algorithm (MTP or EAGLE3)."""
    return resolve_spec_execution(algorithm, pp_size=pp_size).chain_worker


def is_experimental_mtp(algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether MTP is served by TpuChainSpecWorker."""
    return _is_mtp(algorithm) and is_experimental_chain(algorithm, pp_size=pp_size)


def is_experimental_eagle3(algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether EAGLE3 is served by TpuChainSpecWorker."""
    return algorithm.is_eagle3() and is_experimental_chain(algorithm, pp_size=pp_size)


def is_experimental_spec(algorithm: SpeculativeAlgorithm, *, pp_size=1) -> bool:
    """Whether a TpuSpecWorker serves this algorithm (host-padded batches)."""
    return resolve_spec_execution(algorithm, pp_size=pp_size).host_padded
