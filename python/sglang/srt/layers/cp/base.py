# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Base types and process-wide helpers for context parallel strategies.

The strategy implementation is split across:

* ``base.py``: base ABC, base metadata struct, enums, and singleton helpers.
* ``zigzag.py``: in-seq-split strategy and zigzag metadata.
* ``interleave.py``: round-robin-split strategy and interleave metadata.
* ``utils.py``: public re-exports, ``is_cp_v2_active``, ``prepare_cp_forward``,
  ``cp_split_before_forward`` / ``cp_gather_after_forward``.

Downstream PRs in the PCP series add:

* ``pcp_streaming.py``: TPU Prefill-Context-Parallel via the PCP
  streaming Pallas kernel; strategy + attention driver.
* ``pcp_sequence_layout.py``: host-side rank-major compact-Q layout planner
  consumed by the PCP streaming path.
* ``pcp_streaming_op_registry.py``: process-scoped registry that caches one
  torch custom op per (sm_scale, sliding_window, ...) tuple for the PCP path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import msgspec

from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.server_args import ServerArgs


class ContextParallelStrategyKind(IntEnum):
    """Context parallel strategy identifiers."""

    NONE = 0
    ZIGZAG = 1
    INTERLEAVE = 2
    # PCP_STREAMING: TPU Prefill-Context-Parallel via the PCP
    # streaming RPA kernel. KV cache striped per-page
    # across CP ranks (i % cp_size == rank); each rank keeps rank-major
    # packed Q rows; KV pages stream through the CP ring in-kernel via
    # ``pltpu.make_async_remote_copy`` while the kernel runs online-
    # softmax on the locally-owned Q rows. Single Pallas call per
    # attention layer; handles both prefill and decode via the standard
    # ``distribution=[decode_end, prefill_end, mixed_end]`` metadata.
    # See :mod:`sglang.srt.layers.cp.pcp_streaming`.
    PCP_STREAMING = 3

    @classmethod
    def from_string(cls, value: str) -> ContextParallelStrategyKind:
        if value == "zigzag":
            return cls.ZIGZAG
        if value == "interleave":
            return cls.INTERLEAVE
        if value == "pcp":
            return cls.PCP_STREAMING
        raise ValueError(
            f"Unknown cp_strategy={value!r}; expected one of "
            "{'zigzag', 'interleave', 'pcp'}"
        )

    @property
    def cli_value(self) -> str:
        return {
            ContextParallelStrategyKind.NONE: "none",
            ContextParallelStrategyKind.ZIGZAG: "zigzag",
            ContextParallelStrategyKind.INTERLEAVE: "interleave",
            ContextParallelStrategyKind.PCP_STREAMING: "pcp",
        }[self]


class CPAttentionBackendKind(IntEnum):
    """Attention backend calling convention used by CP strategy dispatch."""

    FLASH_ATTENTION = 0
    # PALLAS_TPU_RPA_V3_CP: sglang's Pallas RPA v3 CP backend on torch_tpu.
    # PCP_STREAMING strategy uses this — the backend subclass
    # ``TpuPallasRpaV3CpAttnBackend._forward_pcp`` invokes the PCP streaming
    # kernel via ``pcp_streaming_attention``.
    PALLAS_TPU_RPA_V3_CP = 1

    @classmethod
    def from_string(cls, value: str) -> CPAttentionBackendKind:
        if value in ("fa3", "flashinfer"):
            return cls.FLASH_ATTENTION
        if value == "pallas_tpu_rpa_v3_cp":
            return cls.PALLAS_TPU_RPA_V3_CP
        raise ValueError(
            f"Unsupported attention_backend={value!r} for CP strategy; expected one "
            "of {'fa3', 'flashinfer', 'pallas_tpu_rpa_v3_cp'}"
        )


class BaseContextParallelMetadata(msgspec.Struct, kw_only=True):
    total_seq_lens: int = 0
    bs: int = 1


class ContextParallelStrategy(ABC):
    """Owns process-wide policy for one context parallel layout."""

    name: str
    kind: ContextParallelStrategyKind

    def __init__(self, cp_size: int):
        self.cp_size = cp_size

    @property
    def cp_rank(self) -> int:

        return get_parallel().attn_cp_rank

    @property
    def per_layer_attn_cp_comm(self) -> bool:
        return _is_dsa_active()

    @property
    def local_compile_token_multiple(self) -> int:
        """Kernel lockstep floor for per-rank token counts. Only PCP overrides."""
        raise NotImplementedError(
            f"{self.name} strategy does not expose local_compile_token_multiple"
        )

    @abstractmethod
    def can_apply(self, num_tokens: int, forward_batch: ForwardBatch) -> bool:
        """Return True if this strategy can shard the current forward."""

    @abstractmethod
    def build_metadata(
        self,
        num_tokens: int,
        seqs_len: Optional[List[int]],
        extend_seqs_len: Optional[List[int]] = None,
    ) -> BaseContextParallelMetadata:
        """Build per-forward metadata for this strategy."""

    def build_extra_metadata(
        self,
        forward_batch: ForwardBatch,
        seq_lens_cpu: Optional[List[int]],
        extend_lens_cpu: Optional[List[int]],
    ) -> None:
        """Optional per-forward preparation beyond ``build_metadata``.

        PCP overrides this to materialise the sequence-layout plan and
        the Dynamo-safe plain-scalar mirrors on ForwardBatch. Zigzag /
        Interleave leave it as a no-op.
        """
        del forward_batch, seq_lens_cpu, extend_lens_cpu

    # ------------------------------------------------------------------
    # Optional per-strategy operations. Overridden only where the
    # strategy actually uses them; the base raises so a wrongly-routed
    # call surfaces a clear error instead of a silent no-op.
    # ------------------------------------------------------------------

    def shard_hidden_states(self, x: Any, forward_batch: ForwardBatch) -> Any:
        """Shard hidden states to the current CP rank, usually at the first layer."""
        raise NotImplementedError(
            f"{self.name} strategy does not implement shard_hidden_states"
        )

    def shard_position_ids(self, positions: Any, forward_batch: ForwardBatch) -> Any:
        """Shard KV-cache slot position IDs for each token to the current CP rank."""
        raise NotImplementedError(
            f"{self.name} strategy does not implement shard_position_ids"
        )

    def gather_hidden_states(
        self,
        x: Any,
        forward_batch: ForwardBatch,
        stream: Optional[Any] = None,
    ) -> Any:
        """Gather rank-local hidden states, usually at the last layer."""
        raise NotImplementedError(
            f"{self.name} strategy does not implement gather_hidden_states"
        )

    def gather_kv_cache(
        self,
        x: Any,
        forward_batch: ForwardBatch,
        stream: Optional[Any] = None,
    ) -> Any:
        """Gather rank-local KV payloads back to full token order."""
        raise NotImplementedError(
            f"{self.name} strategy does not implement gather_kv_cache"
        )

    def shard_per_request(
        self,
        extend_seqs_cpu: List[int],
        extend_seqs: Any,
    ) -> Tuple[List[int], Any, List[int], Any]:
        raise NotImplementedError(
            f"{self.name} strategy does not support per-request sharding"
        )

    def run_attention(
        self,
        q: Any,
        forward_batch: ForwardBatch,
        device: Any,
        attn_fn: Callable[[Any, Any, Any, int], Any],
        attention_backend: CPAttentionBackendKind = CPAttentionBackendKind.FLASH_ATTENTION,
    ) -> Any:
        """Dispatch CP attention using the selected backend convention."""
        raise NotImplementedError(
            f"{self.name} strategy does not implement run_attention"
        )

    def materialize_full_kv(
        self,
        forward_batch: ForwardBatch,
        layer: Any,
        k: Any,
        v: Any,
        swa_loc: Optional[Any] = None,
    ) -> None:
        """Write full-layout K/V to the backend cache if needed."""
        raise NotImplementedError(
            f"{self.name} strategy does not implement materialize_full_kv"
        )

    def reindex_attn_metadata(self, core_attn_metadata: Any) -> None:
        """Optional attention metadata rewrite for strategies that need it."""
        return None


def _is_dsa_active() -> bool:
    from sglang.srt.server_args import get_global_server_args

    sa = get_global_server_args()
    return bool(
        getattr(sa, "enable_prefill_cp", False)
        and getattr(sa, "_is_dsa_model_arch", False)
    )


_STRATEGY: Optional[ContextParallelStrategy] = None


def init_cp_strategy(server_args: ServerArgs) -> None:
    """Bind the configured CP strategy for this process."""
    global _STRATEGY

    if not getattr(server_args, "enable_prefill_cp", False):
        _STRATEGY = None
        return

    cp_size = getattr(server_args, "attn_cp_size", 1)
    if cp_size <= 1:
        _STRATEGY = None
        return

    kind = ContextParallelStrategyKind.from_string(server_args.cp_strategy)
    if kind == ContextParallelStrategyKind.ZIGZAG:
        from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

        _STRATEGY = ZigzagCPStrategy(cp_size=cp_size)
    elif kind == ContextParallelStrategyKind.INTERLEAVE:
        from sglang.srt.layers.cp.interleave import InterleaveCPStrategy

        _STRATEGY = InterleaveCPStrategy(cp_size=cp_size)
    elif kind == ContextParallelStrategyKind.PCP_STREAMING:
        # The PCP strategy module lands with the PCP infrastructure PR.
        # On a checkout that lacks it, raise with a clear pointer rather
        # than a bare ImportError from the module load.
        try:
            from sglang.srt.layers.cp.pcp_streaming import PCPStreamingCPStrategy
        except ImportError as e:
            raise NotImplementedError(
                "cp_strategy='pcp' requires the PCP strategy module "
                "(sglang.srt.layers.cp.pcp_streaming), which lands with "
                "the PCP infrastructure PR."
            ) from e

        _STRATEGY = PCPStreamingCPStrategy(cp_size=cp_size)
    else:
        raise ValueError(
            f"Unsupported cp_strategy kind {kind} for "
            f"cp_strategy={server_args.cp_strategy!r}"
        )


def get_cp_strategy() -> Optional[ContextParallelStrategy]:
    """Return the configured strategy, initializing lazily on first call.

    Subprocesses re-import this module with ``_STRATEGY = None`` and never
    re-run ``ServerArgs.__post_init__`` because the pickled instance bypasses
    ``__init__``. Lazy init lets worker processes recover the singleton from
    global server args.
    """
    global _STRATEGY

    if _STRATEGY is None:
        from sglang.srt.server_args import get_global_server_args

        try:
            server_args = get_global_server_args()
        except ValueError:
            return None
        if server_args is not None and getattr(server_args, "enable_prefill_cp", False):
            init_cp_strategy(server_args)
    return _STRATEGY


def get_cp_strategy_kind() -> ContextParallelStrategyKind:
    strategy = get_cp_strategy()
    if strategy is None:
        return ContextParallelStrategyKind.NONE
    return strategy.kind


def is_cp_enabled() -> bool:
    return get_cp_strategy() is not None


def is_zigzag() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.ZIGZAG


def is_interleave() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.INTERLEAVE


def is_pcp() -> bool:
    return get_cp_strategy_kind() == ContextParallelStrategyKind.PCP_STREAMING
