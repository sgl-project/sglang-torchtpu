# SPDX-License-Identifier: Apache-2.0
"""The out-of-tree TPU platform (``SGLANG_PLATFORM=tpu_oot``).

Inherits everything from the in-tree :class:`TpuSRTPlatform`; adds the OOT
identity and the methods the core tree's ``is_out_of_tree()`` gates call.
"""

from __future__ import annotations

import functools
import logging
import os

import torch

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang_tpu.tpu_platform import TpuSRTPlatform

logger = logging.getLogger(__name__)

# What the TPU nightly lanes test. DSPARK is in flight and deliberately absent.
_SUPPORTED_SPEC_ALGORITHMS = frozenset({"EAGLE", "EAGLE3", "DFLASH"})

# The draft backends speculative/draft_utils.py can construct on TPU.
_SUPPORTED_DRAFT_ATTENTION_BACKENDS = frozenset(
    {"pallas_tpu_rpa_v3", "pallas_tpu_rpa_batched"}
)
_DEFAULT_DRAFT_ATTENTION_BACKEND = "pallas_tpu_rpa_v3"

# Set by worker_env.monkey_patch_torch_tpu_env in the scheduler worker;
# unlike WORLD_SIZE, no launcher exports it into the parent environment.
_WORKER_BOOTSTRAP_MARKER = "SGLANG_TPU_CACHE_DP_SCOPE"


@functools.cache
def _with_index_head_dim(pool_cls: type) -> type:
    """``pool_cls`` taking the ``index_head_dim`` the core OOT MLA builder passes."""

    class _Pool(pool_cls):
        def __init__(self, *args, index_head_dim=None, **kwargs):
            # Only the DSA pool takes an indexer, and mode B does not build it
            # (see get_dsa_kv_pool_cls).
            if index_head_dim is not None:
                raise NotImplementedError("The TPU MLA KV pool has no DSA indexer.")
            super().__init__(*args, **kwargs)

    _Pool.__name__ = pool_cls.__name__
    _Pool.__qualname__ = pool_cls.__qualname__
    return _Pool


class TpuOOTPlatform(TpuSRTPlatform):
    """TPU presented to SGLang as an out-of-tree platform."""

    _enum = PlatformEnum.OOT

    def is_tpu(self) -> bool:
        # The core tree's is_tpu() sites must keep seeing True under the OOT enum.
        return True

    def init_backend(self) -> None:
        # model_runner.py calls this at module import on the OOT path, in every
        # process and before the per-rank TPU env exists; the worker bootstrap
        # (initialize_torch_tpu_worker) calls it again, and that call is the real one.
        if os.environ.get(_WORKER_BOOTSTRAP_MARKER) is None:
            logger.debug("Deferring TPU backend init to the worker bootstrap.")
            return
        super().init_backend()

    def get_dsa_kv_pool_cls(self) -> type:
        # Reached from kv_cache_configurator._build_oot_dsa_kv_pool. That builder
        # does not say which layers run their own indexer, which the TPU DSA pool
        # needs, so sparse attention is served under SGLANG_PLATFORM=tpu only.
        # TODO(sglang-torchtpu#1072): build the DSA pool in this mode as well.
        raise NotImplementedError(
            "DSA / sparse attention (DeepSeek V3.2 family) is not available on "
            "TPU with SGLANG_PLATFORM=tpu_oot yet. Use SGLANG_PLATFORM=tpu."
        )

    def get_mla_kv_pool_cls(self) -> type:
        # kv_cache_configurator._build_oot_mla_kv_pool passes index_head_dim for DSA only.
        return _with_index_head_dim(super().get_mla_kv_pool_cls())

    def get_full_graph_backend_cls(self) -> None:
        # The upstream default; TPU compiles whole graphs through TpuGraphRunner.
        return None

    def supports_speculative_algorithm(self, algorithm: str) -> bool:
        name = str(algorithm).upper()
        if name == "DSPARK":
            # Served by sglang_tpu.experimental.TpuBlockSpecWorker.
            from sglang_tpu.experimental import experimental_spec_enabled

            return experimental_spec_enabled()
        return name in _SUPPORTED_SPEC_ALGORITHMS

    def supports_speculative_draft_attention_backend(
        self, algorithm: str, backend: str
    ) -> bool:
        if not self.supports_speculative_algorithm(algorithm):
            return False
        return backend in _SUPPORTED_DRAFT_ATTENTION_BACKENDS

    def get_default_speculative_draft_attention_backend(self, algorithm: str) -> str:
        if not self.supports_speculative_algorithm(algorithm):
            raise ValueError(
                f"Speculative algorithm {algorithm!r} is not supported on TPU; "
                f"supported: {sorted(_SUPPORTED_SPEC_ALGORITHMS)}."
            )
        return _DEFAULT_DRAFT_ATTENTION_BACKEND

    def get_torch_profiler_activity_str(self) -> str:
        # The name profiler_manager routes to xprof; not a torch.profiler attribute,
        # so the torch activity list stays as in mode A.
        return "TPU"

    def get_torch_profiler_activity(self) -> torch.profiler.ProfilerActivity:
        return torch.profiler.ProfilerActivity.PrivateUse1
