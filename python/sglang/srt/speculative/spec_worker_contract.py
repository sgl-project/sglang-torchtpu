"""What core reads from a speculative worker and from the draft input it relays.

These Protocols are the interface; ``BaseSpecWorker`` is one optional
implementation that also carries reusable defaults. Core depends on the
Protocol, so a platform worker can satisfy it by composition instead of
inheriting behaviour it never reads.

The scheduler, its components, FutureMap and P/D touch a spec worker and its
``next_draft_input`` only through the members named here. A worker returned by
``SRTPlatform.get_spec_worker_cls`` is safe to swap in when it exposes all of
them; ``test/registered/unit/spec/test_spec_worker_contract.py`` checks the
in-tree workers and the TPU plugin workers against these Protocols.

Deliberately absent, because only one algorithm or platform reaches them: the
DSpark-only commands behind ``is_dspark()`` / ``decide_needs_confidence_relay``
(``dump_info_records``, ``clear_info_records``, ``set_dspark_forced_budget_frac``,
``block_accept_estimate_log_suffix``, ``get_confidence_budget_prepare``), the TPU
spec-op precompile (``TpuSchedulerExtension.precompile_spec_ops``), and the
NGRAM relay fields read by ``RelayPayload.from_ngram``.

Known gap: ``SchedulerWeightUpdaterManager.save_remote_model`` still reads
``draft_worker.model_runner``, which not every worker defines (EAGLE-family
workers do not; DFlash's is the target runner).
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Callable,
    Optional,
    Protocol,
    TypeGuard,
    runtime_checkable,
)

if TYPE_CHECKING:
    import torch

    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.mem_cache.memory_pool import KVCache
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
    from sglang.srt.model_executor.model_runner import (
        ModelRunner,
        SamplingPrewarmResult,
    )
    from sglang.srt.speculative.base_spec_worker import HiCacheDraftPlan

# Scheduler.maybe_init_draft_worker constructs the worker with these keywords.
SPEC_WORKER_INIT_KWARGS = ("server_args", "gpu_id", "nccl_port", "target_worker")


class SpecWorkerContract(Protocol):
    """The spec-worker members core reads; ``BaseSpecWorker`` implements every one.

    Core tells a spec worker from a plain ``TpModelWorker`` draft with
    ``is_spec_worker``, not ``isinstance`` against this Protocol.
    """

    # ---- Scheduler init: memory, backends, graphs (scheduler.py) ----
    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None: ...

    def init_hicache_draft_plan(self) -> None: ...

    def init_attention_backends(self) -> None: ...

    def init_cuda_graphs(self) -> None: ...

    def prewarm_sampling(self) -> SamplingPrewarmResult: ...

    # Read by post_capture_resize_kv_pool through the scheduler.
    def _draft_model_runners(self) -> tuple[ModelRunner, ...]: ...

    def clear_cache_pool(self) -> None: ...

    @property
    def hicache_draft_plan(self) -> HiCacheDraftPlan: ...

    @property
    def retraction_draft_kv_pools(self) -> tuple[KVCache, ...]: ...

    @property
    def primary_draft_kv_pool(self) -> Optional[object]: ...

    # Read through getattr with a target-only fallback (scheduler.py).
    @property
    def spec_v2_attn_backends(self) -> tuple: ...

    # The scheduler builds the DP-attention adapter on the target model.
    @property
    def target_worker(self) -> TpModelWorker: ...

    # ---- Startup accounting and metrics (scheduler.py) ----
    @property
    def preloaded_weights_bytes(self) -> int: ...

    @property
    def weight_load_time(self) -> float: ...

    @property
    def graph_memory_usage(self) -> dict[str, float]: ...

    @property
    def graph_time_usage(self) -> dict[str, float]: ...

    @property
    def compile_timing_steps(self) -> list[dict]: ...

    # ---- Per step: run loop, WAR barrier, result processor ----
    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: Optional[Callable[[torch.Tensor], None]] = None,
        grammar_barrier: Optional[Callable[[], None]] = None,
        *,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> GenerationBatchResult: ...

    @property
    def last_shared_read_runner(self) -> ModelRunner: ...

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None: ...

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None: ...

    # ---- Weight updates (scheduler_components/weight_updater.py) ----
    def weight_update_runners(self) -> list[tuple[str, ModelRunner]]: ...


@runtime_checkable
class _SpecWorkerLifecycle(Protocol):
    """Contract methods a plain ``TpModelWorker`` draft does not have.

    Methods only: ``isinstance`` against a Protocol with property members calls
    every getter before Python 3.12, so its cost and result would depend on the
    worker's state and the interpreter version.
    """

    def init_hicache_draft_plan(self) -> None: ...

    def clear_cache_pool(self) -> None: ...

    def note_request_finished(self, *, rid: str, natural_stop: bool) -> None: ...


def is_spec_worker(worker: object) -> TypeGuard[SpecWorkerContract]:
    """Whether core drives ``worker`` as a spec worker rather than a plain draft."""
    return isinstance(worker, _SpecWorkerLifecycle)


class RelayedDraftInput(Protocol):
    """``GenerationBatchResult.next_draft_input`` as core reads it.

    The scheduler stores it as ``batch.spec_info`` and sets ``future_indices``;
    ScheduleBatch filters and merges it; ``RelayPayload.from_draft_input`` and
    ``FutureMap._resolve_spec_extras`` relay the tensors; P/D prefill reads the
    EAGLE ones. Fields an algorithm does not relay hold ``None``.
    """

    bonus_tokens: Optional[torch.Tensor]
    topk_p: Optional[torch.Tensor]
    topk_index: Optional[torch.Tensor]
    hidden_states: Optional[torch.Tensor]
    dsa_topk_indices: Optional[torch.Tensor]
    future_indices: Optional[torch.Tensor]
    future_dsa_topk_indices_available: bool

    def filter_batch(
        self, new_indices: torch.Tensor, new_indices_cpu: Optional[list[int]] = None
    ) -> None: ...

    def merge_batch(self, spec_info: RelayedDraftInput, /) -> None: ...
