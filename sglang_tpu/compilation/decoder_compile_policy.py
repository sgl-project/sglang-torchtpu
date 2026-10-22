"""TPU decoder compile policies for annotation-driven torch.compile."""

from __future__ import annotations

import dataclasses
import functools
import inspect
import time
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal

import msgspec
import torch
import tqdm
from torch.utils._pytree import tree_leaves

from sglang.srt.compilation import torch_compile_bundle
from sglang.srt.compilation.support_torch_compile import DecoderCompilePolicy
from sglang.srt.environ import envs
from sglang.srt.layers.aux_hidden_states import (
    AUX_HIDDEN_STATES_PROXY_KEY,
    AuxHiddenStateList,
)
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.cp.utils import is_cp_active, prepare_cp_forward
from sglang.srt.layers.dp_attention import dp_gather_width, is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    LogitsOutputMode,
    PPProxyTensors,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.observability.compile_timing import CompileTimingRecorder
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_parallel,
    get_schedule,
    max_prefill_buffer_tokens,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.spec_info import SpecInput
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.attention_rank_inputs import (
    get_attention_rank_tensors,
    use_attention_rank_tensors,
)
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.compiled_forward_shim import (
    _BS_FIELDS,
    _MAMBA_TRACK_FIELDS,
    _REAL_B_KEY,
    _REAL_Q_KEY,
    _TOKEN_FIELDS,
    CompiledForwardShim,
    _extend_batch_cap,
    _pad_forward_batch_inputs,
    _restore_forward_batch_inputs,
    build_bucketers,
)
from sglang_tpu.compilation.frame_stack import (
    call_with_reserved_frame_stack,
)
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.cp_forward import (
    install_cp_logits_forward,
    prepare_tpu_cp_forward_batch,
    wrap_cp_embedding_forward,
)
from sglang_tpu.cp_layout import warmup_request_lengths
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded
from sglang_tpu.host_padding import (
    check_host_padding_config,
    ordinary_compiled_forward,
)
from sglang_tpu.pp_wave import num_relayed_aux_slots

logger = get_logger(__name__)


def _guard_filter_options(server_args: Any) -> dict[str, Any] | None:
    """torch.compile options that skip guards on nn.Module state, or None.

    Every decode step evaluates the decoder graph's Dynamo guards on the host
    before the graph runs: about 9k guards and 0.7 ms per step for a 13-layer
    Kimi-K3, growing with depth. Most guard module attributes and parameters,
    which are fixed once warmup ends: TPU serving freezes compilation with
    fail_on_recompile, so a guard miss can only raise. Skipping them keeps the
    guards on the forward inputs and globals, which choose between the
    precompiled graphs. In exchange, a module attribute changed after warmup
    is no longer noticed, so the filter stays off for features that rebuild
    module state at runtime.
    """
    if not tpu_envs.SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER.get():
        return None
    enable_lora = getattr(server_args, "enable_lora", None)
    blockers = []
    if enable_lora or (
        enable_lora is None and getattr(server_args, "lora_paths", None)
    ):
        blockers.append("LoRA")
    if getattr(server_args, "enable_eplb", False):
        blockers.append("EPLB")
    if getattr(server_args, "elastic_ep_backend", None) is not None:
        blockers.append("elastic EP")
    if blockers:
        logger.warning(
            "[TPU] SGLANG_TPU_ENABLE_NN_MODULE_GUARD_FILTER ignored: %s changes "
            "module state after warmup.",
            ", ".join(blockers),
        )
        return None
    logger.info("[TPU] Skipping Dynamo guards on nn.Module state for the decoder.")
    return {"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe}


# Pure-text progress format (no animated bar) so log files and
# multi-process wrappers don't garble the output.
_COMPILE_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% Completed | "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)

# XLA TopK has a static output width. Warm the physically qualified serving
# width so the common candidate path does not compile on its first request.
_DISTRIBUTED_TOP_K_WARMUP_K = 50


def max_extend_tokens(*, context_len: int) -> int:
    """The largest flattened EXTEND token dimension one scheduled batch can
    reach: the prefill-buffer ceiling, else the larger of max_prefill_tokens
    and the context length (tokens sum across requests, so a per-request
    context length must not cap it)."""
    return max_prefill_buffer_tokens() or max(
        get_schedule().max_prefill_tokens or 0, context_len
    )


def _materialize_out(out: Any, attn_backend: Any, token_to_kv_pool: Any) -> None:
    """Sync returned tensors and backend-marked KV side effects.

    attn_backend / token_to_kv_pool come from the model_runner: upstream moved
    both off ForwardBatch (attn_backend now flows through ForwardContext, and the
    KV pool is held by the attention backend / model_runner), so warmup reads them
    from the runner rather than the batch.
    """
    # Inner decoder boundaries can return raw hidden states or tensor pytrees.
    # Each queued warmup must finish before launching the next collective graph.
    tensors = [leaf for leaf in tree_leaves(out) if isinstance(leaf, torch.Tensor)]
    if not tensors:
        for field in (
            "next_token_logits",
            "next_token_global_top_k_logits",
            "next_token_global_top_k_token_ids",
            "next_draft_tokens",
        ):
            value = getattr(out, field, None)
            if isinstance(value, torch.Tensor):
                tensors.append(value)

    if isinstance(out, PPProxyTensors):
        # A non-final pipeline stage returns its hidden states and residual.
        tensors.extend(
            value for value in out.tensors.values() if isinstance(value, torch.Tensor)
        )

    if getattr(attn_backend, "requires_kv_sync_after_compile", False):
        pool = token_to_kv_pool
        for layer_id in range(pool.start_layer, pool.start_layer + pool.layer_num):
            tensors.append(pool.get_fused_kv_buffer(layer_id))

    if not tensors:
        return
    from torch_tpu._internal.sync.sync import synchronize as tpu_sync

    tpu_sync(tensors, wait=True)


def _get_hbm_usage_mb() -> tuple[float, float] | None:
    """Return (used_mb, total_mb) for the current TPU device, or None."""
    try:
        info = torch.accelerator.get_memory_info(torch.device("tpu"))
        free = info["free"] if isinstance(info, dict) else info[0]
        total = info["total"] if isinstance(info, dict) else info[1]
        used = total - free
        return (used / 1024 / 1024, total / 1024 / 1024)
    except Exception:
        return None


def _hbm_delta_str(
    before: tuple[float, float] | None,
    after: tuple[float, float] | None,
) -> str:
    """Format HBM delta + total as a suffix string, or empty if unavailable."""
    if before is None or after is None:
        return " HBM: N/A"
    delta = after[0] - before[0]
    return f" HBM: +{delta:.0f} MB ({after[0]:.0f}/{after[1]:.0f} MB)"


def _warmup_progress(
    warmup: PendingCall | None,
    start: float,
    hbm_before: tuple[float, float] | None,
) -> tuple[str, float, str]:
    """Return accurate progress details for synchronous or queued warmups."""
    elapsed = time.perf_counter() - start
    if warmup is not None:
        return "Queued", elapsed, ""
    return "Compiled", elapsed, _hbm_delta_str(hbm_before, _get_hbm_usage_mb())


def _unpad_decoder_outputs(
    out: LogitsProcessorOutput | Any,
    forward_batch: ForwardBatch,
) -> Any:
    real_q = getattr(forward_batch, _REAL_Q_KEY, None)
    real_b = getattr(forward_batch, _REAL_B_KEY, None)
    if real_q is None or real_b is None:
        return out

    if isinstance(out, LogitsProcessorOutput):
        out.trim_next_token_outputs(real_b)
        if (
            out.hidden_states is not None
            and isinstance(out.hidden_states, torch.Tensor)
            and real_q < out.hidden_states.shape[0]
        ):
            out.hidden_states = out.hidden_states[:real_q]

    # Restore real token/batch metadata so eager outer-model processing and
    # downstream consumers never observe static-shape ghost slots.
    _restore_forward_batch_inputs(forward_batch, real_q, real_b)
    return out


class _DummyForwardBatchFields(msgspec.Struct, frozen=True):
    """Mode-specific fields used to construct a dummy precompile ForwardBatch."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    out_cache_loc: torch.Tensor
    seq_lens: torch.Tensor
    orig_seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    req_pool_indices: torch.Tensor
    req_pool_indices_cpu: torch.Tensor
    seq_lens_sum: int
    extend_num_tokens: int | None
    extend_seq_lens: torch.Tensor | None
    extend_prefix_lens: torch.Tensor | None
    extend_start_loc: torch.Tensor | None
    extend_seq_lens_cpu: list[int] | None
    extend_prefix_lens_cpu: list[int] | None
    extend_logprob_start_lens_cpu: list[int] | None

    @staticmethod
    def _extend_metadata_fields(
        is_extend: bool,
        bs: int,
        tokens_per_req: int,
        prefix: int,
        device: str | torch.device,
    ) -> dict[str, Any]:
        if not is_extend:
            return {
                "extend_num_tokens": None,
                "extend_seq_lens": None,
                "extend_prefix_lens": None,
                "extend_start_loc": None,
                "extend_seq_lens_cpu": None,
                "extend_prefix_lens_cpu": None,
                "extend_logprob_start_lens_cpu": None,
            }
        return {
            "extend_num_tokens": bs * tokens_per_req,
            "extend_seq_lens": torch.full(
                (bs,), tokens_per_req, dtype=torch.int32, device=device
            ),
            "extend_prefix_lens": torch.full(
                (bs,), prefix, dtype=torch.int32, device=device
            ),
            "extend_start_loc": torch.arange(
                0, bs * tokens_per_req, tokens_per_req, dtype=torch.int32, device=device
            ),
            "extend_seq_lens_cpu": [tokens_per_req] * bs,
            "extend_prefix_lens_cpu": [prefix] * bs,
            "extend_logprob_start_lens_cpu": [0] * bs,
        }

    @classmethod
    def default(
        cls,
        num_tokens: int,
        bs: int,
        is_extend: bool,
        tokens_per_req: int,
        *,
        device: str | torch.device,
        slot: int,
        out_cache_loc_dtype: torch.dtype,
    ) -> _DummyForwardBatchFields:
        """Reserved-slot layout for DECODE / EXTEND / TARGET_VERIFY."""
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        input_ids[:bs] = 1
        seq_lens = torch.ones(bs, dtype=torch.int64, device=device)
        return cls(
            input_ids=input_ids,
            positions=torch.zeros(num_tokens, dtype=torch.int64, device=device),
            out_cache_loc=torch.full(
                (num_tokens,), slot, dtype=out_cache_loc_dtype, device=device
            ),
            seq_lens=seq_lens,
            orig_seq_lens=seq_lens,
            seq_lens_cpu=torch.ones(bs, dtype=torch.int64),
            req_pool_indices=torch.full((bs,), slot, dtype=torch.int64, device=device),
            req_pool_indices_cpu=torch.full((bs,), slot, dtype=torch.int64),
            seq_lens_sum=bs,
            **cls._extend_metadata_fields(is_extend, bs, tokens_per_req, 0, device),
        )

    @classmethod
    def draft_extend(
        cls,
        num_tokens: int,
        bs: int,
        tokens_per_req: int,
        *,
        device: str | torch.device,
        page_size: int,
        out_cache_loc_dtype: torch.dtype,
    ) -> _DummyForwardBatchFields:
        """Realistic per-request layout for DRAFT_EXTEND_V2."""
        num_context_pages = max(1, (tokens_per_req + page_size - 1) // page_size)
        context_len = num_context_pages * page_size  # 1 or more pages of context
        prefix_len = context_len - tokens_per_req

        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        input_ids[:bs] = 1
        # query positions per request: [prefix, prefix + tokens_per_req)
        positions = prefix_len + torch.arange(tokens_per_req, device=device).repeat(
            bs
        ).to(torch.int64)
        # distinct KV write locations: request r (1..bs) uses a disjoint page
        # span, with extend tokens at the tail of that span.
        r = torch.arange(1, bs + 1, device=device)
        base = (r * context_len + prefix_len).view(bs, 1)
        out_cache_loc = (
            (base + torch.arange(tokens_per_req, device=device).view(1, -1))
            .reshape(-1)
            .to(out_cache_loc_dtype)
        )
        return cls(
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=out_cache_loc,
            seq_lens=torch.full((bs,), context_len, dtype=torch.int64, device=device),
            orig_seq_lens=torch.full(
                (bs,), context_len, dtype=torch.int32, device=device
            ),
            seq_lens_cpu=torch.full((bs,), context_len, dtype=torch.int64),
            req_pool_indices=torch.arange(1, bs + 1, dtype=torch.int64, device=device),
            req_pool_indices_cpu=torch.arange(1, bs + 1, dtype=torch.int64),
            seq_lens_sum=context_len * bs,
            # DRAFT_EXTEND_V2 always carries extend metadata (prefix = prefix_len).
            **cls._extend_metadata_fields(True, bs, tokens_per_req, prefix_len, device),
        )

    @classmethod
    def cp_extend(
        cls,
        request_lens: list[int],
        *,
        device: str | torch.device,
        slot: int,
        out_cache_loc_dtype: torch.dtype,
    ) -> _DummyForwardBatchFields:
        """Reserved-slot EXTEND with one prefix-free request per entry of
        ``request_lens`` (a context-parallel warmup batch)."""
        bs = len(request_lens)
        num_tokens = sum(request_lens)
        lens = torch.tensor(request_lens, dtype=torch.int64)
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        input_ids[:bs] = 1
        positions = torch.cat(
            [torch.arange(length, dtype=torch.int64) for length in request_lens]
        ).to(device)
        extend_start_loc = torch.zeros(bs, dtype=torch.int32)
        extend_start_loc[1:] = torch.cumsum(lens, dim=0)[:-1].to(torch.int32)
        return cls(
            input_ids=input_ids,
            positions=positions,
            out_cache_loc=torch.full(
                (num_tokens,), slot, dtype=out_cache_loc_dtype, device=device
            ),
            seq_lens=lens.to(device),
            orig_seq_lens=lens.to(device),
            seq_lens_cpu=lens.clone(),
            req_pool_indices=torch.full((bs,), slot, dtype=torch.int64, device=device),
            req_pool_indices_cpu=torch.full((bs,), slot, dtype=torch.int64),
            seq_lens_sum=num_tokens,
            extend_num_tokens=num_tokens,
            extend_seq_lens=lens.to(torch.int32).to(device),
            extend_prefix_lens=torch.zeros(bs, dtype=torch.int32, device=device),
            extend_start_loc=extend_start_loc.to(device),
            extend_seq_lens_cpu=list(request_lens),
            extend_prefix_lens_cpu=[0] * bs,
            extend_logprob_start_lens_cpu=[0] * bs,
        )


# Per-DP-rank token counts the compiled forward reads as device tensors.
_DP_COUNT_FIELDS = ("global_num_tokens_gpu", "global_num_tokens_for_logprob_gpu")


class EagerLogitsCase(msgspec.Struct, frozen=True):
    """One shape of the eager outer-model logits that follows a compiled decoder forward."""

    num_tokens: int
    bs: int
    mode: ForwardMode
    capture_hidden_mode: CaptureHiddenMode
    spec_info: SpecInput | None = None


class DecoderTpuCompilePolicy(DecoderCompilePolicy):
    """Compile, pad, unpad, and precompile one decoder boundary."""

    model_runner: ModelRunner
    platform = "tpu"
    fullgraph = True
    dynamic = False
    model_extra_tensor_attrs: ClassVar[tuple[str, ...]] = ()

    def get_dp_padded_token_counts(
        self,
        forward_batch: ForwardBatch,
        model_runner: Any,
        global_num_tokens: list[int],
        global_num_tokens_for_logprob: list[int],
        *,
        attn_tp_size: int,
        cp_align_size: int,
    ) -> tuple[list[int], list[int]]:
        (
            global_num_tokens,
            global_num_tokens_for_logprob,
        ) = super().get_dp_padded_token_counts(
            forward_batch,
            model_runner,
            global_num_tokens,
            global_num_tokens_for_logprob,
            attn_tp_size=attn_tp_size,
            cp_align_size=cp_align_size,
        )

        # TPU DPA gather derives its expected local rows from the DP-global
        # count, so logprob counts must use the same attention-TP alignment.
        if attn_tp_size > 1:
            global_num_tokens_for_logprob = [
                (value + attn_tp_size - 1) // attn_tp_size * attn_tp_size
                for value in global_num_tokens_for_logprob
            ]
        return global_num_tokens, global_num_tokens_for_logprob

    def __init__(
        self,
        *,
        model_runner: Any | None = None,
        t_bucketer: Bucketer | None = None,
        b_bucketer: Bucketer | None = None,
        owner: Any | None = None,
        method_name: str | None = None,
        method: Callable[..., Any] | None = None,
        name: str | None = None,
        label: str | None = None,
        tracer: Literal["dynamo", "direct"] = "dynamo",
        compile_timing_recorder: CompileTimingRecorder | None = None,
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            owner=owner,
            method_name=method_name or "forward",
            method=method,
            name=name,
            label=label,
            compile_timing_recorder=compile_timing_recorder,
        )
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer
        self.tracer = tracer
        self.extend_bs: int | None = None
        self.compiled_forward_shim: CompiledForwardShim | None = None
        self._logged_vl_warmup_variants: set[str] = set()
        # Variant tags actually warmed, so precompile() can tell a family that
        # was compiled from one an override dropped.
        self._compiled_extend_variants: set[str] = set()

    @property
    def _cp_strategy(self) -> Any:
        """Prefill context-parallel strategy, or None when not configured; not
        cached, since the strategy may initialize after this policy is built."""
        return get_cp_strategy()

    def _record_compile_step(
        self,
        *,
        phase: str,
        duration_seconds: float,
        pending_call: PendingCall | None = None,
        batch_size: int | None = None,
        num_tokens: int | None = None,
        num_draft_tokens: int | None = None,
        draft_step: int | None = None,
        input_variant: str | None = None,
        logits_output_mode: str | None = None,
        capture_hidden_mode: str | None = None,
        has_spec_info: bool | None = None,
    ) -> PendingCall | None:
        if self.compile_timing_recorder is None:
            return pending_call
        assert self.model_runner is not None
        record = functools.partial(
            self.compile_timing_recorder.record,
            component="draft" if self.model_runner.is_draft_worker else "target",
            phase=phase,
            policy=type(self).__name__,
            batch_size=batch_size,
            num_tokens=num_tokens,
            num_draft_tokens=num_draft_tokens,
            draft_step=draft_step,
            model_index=self.model_runner.draft_model_idx,
            input_variant=input_variant,
            logits_output_mode=logits_output_mode,
            capture_hidden_mode=capture_hidden_mode,
            has_spec_info=has_spec_info,
        )
        if pending_call is None:
            record(duration_seconds=duration_seconds)
            return None

        def wait_and_record():
            start = time.perf_counter()
            result = pending_call.wait()
            # Count submission plus blocking wait/execution, excluding time
            # spent submitting other shapes while native compilation overlaps.
            record(duration_seconds=duration_seconds + time.perf_counter() - start)
            return result

        return PendingCall(wait_and_record)

    def _extend_compile_cap(self) -> int:
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        return _extend_batch_cap(self.t_bucketer, self.b_bucketer, self.extend_bs)

    def max_padded_tokens(self) -> int:
        """The most token rows a forward is padded to: the last token bucket,
        or the last batch bucket for a decode batch."""
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        return max(self.t_bucketer.max_val, self.b_bucketer.max_val)

    def _extend_compile_cases(self) -> list[tuple[int, int, ForwardMode]]:
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None

        mode_caps = [(ForwardMode.EXTEND, self._extend_compile_cap())]
        if get_schedule().enable_mixed_chunk:
            mode_caps.append(
                (
                    ForwardMode.MIXED,
                    _extend_batch_cap(
                        self.t_bucketer,
                        self.b_bucketer,
                        self.b_bucketer.max_val,
                    ),
                )
            )

        return [
            (num_tokens, batch_size, mode)
            for mode, batch_size in mode_caps
            for num_tokens in self.t_bucketer.buckets
            if num_tokens >= batch_size
        ]

    def _extend_cp_compile_cases(self) -> list[tuple[int, int]]:
        """(local_tokens, batch_cap) of the context-parallel extend family."""
        if self._cp_strategy is None:
            return []
        assert self.t_bucketer is not None
        cap = self._extend_compile_cap()
        return [
            (num_tokens, cap)
            for num_tokens in self.t_bucketer.buckets
            if num_tokens >= cap
        ]

    def _extend_compile_shapes(self) -> list[tuple[int, int]]:
        return list(
            dict.fromkeys(
                (num_tokens, batch_size)
                for num_tokens, batch_size, _ in self._extend_compile_cases()
            )
        )

    def install(self) -> bool:
        """Install runtime compile wrappers for the annotated decoder target."""
        if self.model_runner is None or self.owner is None or self.method_name is None:
            logger.warning(
                "[decoder compile policy] missing model_runner/owner/method; "
                "cannot install."
            )
            return False

        fn = getattr(self.owner, self.method_name, None)
        if fn is None:
            logger.warning(
                "[decoder compile policy] %s target %s has no method %s; skipping.",
                self.name,
                type(self.owner).__name__,
                self.method_name,
            )
            return False

        server_args = self.model_runner.server_args
        is_draft_worker = self.model_runner.is_draft_worker
        split_draft = self._uses_eager_draft_coordinator()
        compile_owner = (
            getattr(self.owner, "model", self.owner)
            if is_draft_worker and (split_draft or not self._supports_async_warmup())
            else self.owner
        )
        compile_fn = compile_owner.forward if is_draft_worker else fn

        cp_logits_installed = False
        if self._compiles_logits_processor():
            logits_processor = getattr(
                self.model_runner.model, "logits_processor", None
            )
            if logits_processor is not None:
                # TPUs don't support returning input logprobs -- this errors elsewhere.
                logits_processor.compute_input_logprobs = False
                if self._cp_strategy is not None:
                    install_cp_logits_forward(logits_processor)
                    cp_logits_installed = True

        if self._cp_strategy is not None and not is_draft_worker:
            pp_group = self.model_runner.pp_group
            if not cp_logits_installed and pp_group.is_last_rank:
                raise NotImplementedError(
                    "Prefill context parallelism on TPU reduces the last-token rows "
                    "inside the compiled logits step, but "
                    f"{type(self.model_runner.model).__name__} runs its logits "
                    "processor outside the compiled forward."
                )
            if pp_group.is_first_rank:
                compile_fn = wrap_cp_embedding_forward(
                    compile_fn, self.model_runner.model.get_input_embeddings()
                )
        if self.tracer == "direct":
            compiled, warmup_async = self._direct_compile(compile_fn)
        else:
            compiled = self._compile(compile_fn)
            warmup_async = None
        compiled.__signature__ = inspect.signature(compile_fn)
        setattr(
            compile_owner,
            "forward" if is_draft_worker else self.method_name,
            compiled,
        )
        if split_draft:
            logits_fn = self.owner.logits_processor.forward
            compiled_logits = torch.compile(
                torch.no_grad()(logits_fn),
                backend=self.backend or current_platform.get_compile_backend(),
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
            )
            compiled_logits.__signature__ = inspect.signature(logits_fn)
            self.owner.logits_processor.forward = compiled_logits
        max_t = max_extend_tokens(
            context_len=self.model_runner.model_config.context_len
        )
        # max_b is the reconciled compile-batch ceiling, already grown to cover
        # scheduler concurrency in ServerArgs.__post_init__ via
        # reconcile_tpu_batch_limits, so the grid covers every decode batch the
        # scheduler can form.
        max_b = get_exec().graph.torch_compile_max_bs
        t_bucketer, b_bucketer = build_bucketers(server_args, max_t=max_t, max_b=max_b)
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer
        if is_dp_attention_enabled():
            self.extend_bs = b_bucketer.max_val
        else:
            self.extend_bs = min(
                get_schedule().prefill_max_requests or b_bucketer.max_val,
                b_bucketer.max_val,
            )

        outer_model = self.model_runner.model
        existing_prep = getattr(outer_model, "prepare_forward_batch", None)

        def _prepare(forward_batch: ForwardBatch) -> None:
            if existing_prep is not None:
                existing_prep(forward_batch)
            self.prepare_forward_batch(forward_batch)

        outer_model.prepare_forward_batch = _prepare
        self._install_forward_shim(
            fn=fn,
            compiled=compiled,
            warmup_async=warmup_async,
        )
        self.model_runner.compile_policy = self

        if is_dp_attention_enabled():
            self._install_dp_idle_as_decode()

        logger.info(
            "[decoder compile policy] Installed %s on %s.%s "
            "(t-bucketer=%s, b-bucketer=%s).",
            self.name,
            type(self.owner).__name__,
            self.method_name,
            self.t_bucketer.name,
            self.b_bucketer.name,
        )
        return True

    def _uses_eager_draft_coordinator(self) -> bool:
        return self.model_runner.is_draft_worker and hasattr(
            self.owner, "logits_processor"
        )

    def _supports_async_warmup(self) -> bool:
        return not self.model_runner.is_draft_worker

    def _install_forward_shim(
        self,
        fn: Callable[..., Any],
        compiled: Callable[..., Any],
        warmup_async: Callable[..., PendingCall] | None = None,
    ) -> None:
        """Wrap and install the compiled forward shim on owner.

        EAGLE3 splits backbone and logits_processor into separate compiled regions,
        leaving outer forward eager as a coordinator. DFlash samples via target lm_head
        without logits_processor, so its entire forward is compiled.

        Args:
          fn: Original uncompiled forward function.
          compiled: Compiled forward function.
        """
        is_eagle3_draft = self._uses_eager_draft_coordinator()
        shim_target = fn if is_eagle3_draft else compiled
        logger.info(
            "[decoder compile policy] _install_forward_shim for %s (is_draft=%s, is_eagle3_draft=%s, shim_target=%s)",
            type(self.owner).__name__,
            self.model_runner.is_draft_worker,
            is_eagle3_draft,
            getattr(shim_target, "__name__", type(shim_target).__name__),
        )
        # EAGLE3's async callable covers only the backbone, not the outer
        # forward coordinated by the shim. Other direct-compiled targets can
        # safely use the queued warmup path.
        shim_warmup = None if is_eagle3_draft else warmup_async
        self.compiled_forward_shim = CompiledForwardShim(
            shim_target, policy=self, warmup_async_fn=shim_warmup
        )
        setattr(self.owner, self.method_name, self.compiled_forward_shim)

    def _compile(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        server_args = self.model_runner.server_args
        use_bundle = (
            server_args.unsafe_save_torch_compile_bundle is not None
            or server_args.unsafe_load_torch_compile_bundle is not None
        )
        compiler = torch_compile_bundle.Compiler if use_bundle else torch.compile
        # The bundle Compiler installs its own guard filter for restorable guards.
        options = None if use_bundle else _guard_filter_options(server_args)
        return compiler(
            torch.no_grad()(fn),
            backend=self.backend or current_platform.get_compile_backend(),
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
            **({"options": options} if options else {}),
        )

    def _direct_compile(
        self, fn: Callable[..., Any]
    ) -> tuple[Callable[..., Any], Callable[..., PendingCall]]:
        """Compile directly through TorchTPU without Dynamo or AOTAutograd."""
        from sglang_tpu.compilation.direct_compile import (
            direct_compile,
        )
        from sglang_tpu.compilation.hybrid_linear_direct_compile import (
            bind_traced_attention_metadata,
            bind_traced_conv_weights,
            traced_attention_metadata,
        )

        fn = torch.no_grad()(fn)
        model = self.model_runner.model

        def trace_forward(
            attention_metadata: Any,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: Any,
            *args: Any,
            attention_rank_tensors: tuple[torch.Tensor, ...],
            **kwargs: Any,
        ) -> Any:
            backend = get_forward_context().attn_backend
            with (
                bind_traced_attention_metadata(
                    backend, attention_metadata, forward_batch.forward_mode
                ),
                bind_traced_conv_weights(model),
                use_attention_rank_tensors(attention_rank_tensors),
            ):
                return fn(input_ids, positions, forward_batch, *args, **kwargs)

        compiled = direct_compile(
            trace_forward,
            specialization_key=self._direct_compile_specialization_key,
            name=self.name,
            spmd_safe=self.spmd_safe,
            lift_rank=(
                torch.distributed.is_initialized()
                and get_parallel().tp_group.world_size
                == torch.distributed.get_world_size()
            ),
            model=model,
            model_extra_tensor_attrs=self.get_model_extra_tensor_attrs(),
            kv_pool=self.get_direct_compile_kv_pool(),
            ignored_argnames=self._ignored_argnames(),
        )

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            backend = get_forward_context().attn_backend
            return compiled(
                traced_attention_metadata(backend),
                *args,
                attention_rank_tensors=get_attention_rank_tensors(
                    torch.device(self.model_runner.device)
                ),
                **kwargs,
            )

        def warmup_async(*args: Any, **kwargs: Any) -> PendingCall:
            backend = get_forward_context().attn_backend
            return compiled.warmup_async(
                traced_attention_metadata(backend),
                *args,
                attention_rank_tensors=get_attention_rank_tensors(
                    torch.device(self.model_runner.device)
                ),
                **kwargs,
            )

        return call, warmup_async

    def get_direct_compile_kv_pool(self):
        kv_pool = self.model_runner.token_to_kv_pool
        if not isinstance(kv_pool, HybridLinearKVPool):
            return kv_pool
        from sglang_tpu.compilation.hybrid_linear_direct_compile import (
            has_linear_attention,
            hybrid_linear_kv_pool_state,
        )

        # An MTP draft shares the target's GDN state but runs no GDN layer; lifting it
        # swaps in new buffers per forward and Raiden keeps the stale registered ones.
        if not has_linear_attention(self.model_runner.model):
            return kv_pool.full_kv_pool
        return hybrid_linear_kv_pool_state(
            kv_pool=kv_pool, req_to_token_pool=self.model_runner.req_to_token_pool
        )

    def _ignored_argnames(self) -> tuple[str, ...]:
        return (
            "forward_batch.sampling_info",
            "forward_batch.orig_seq_lens",
            # Prompt-logprob inputs are rejected on compiled TPU. Ignore this
            # request-only tensor before direct compile flattens ForwardBatch.
            "forward_batch.extend_input_logprob_token_ids_gpu",
            # These host mirrors are consumed before the compiled model.
            "forward_batch.req_pool_indices_cpu",
            "forward_batch.seq_lens_cpu",
            "forward_batch.mm_input_ids_cpu",
            *self._extra_ignored_argnames(),
        )

    def _extra_ignored_argnames(self) -> tuple[str, ...]:
        return ()

    def get_model_extra_tensor_attrs(self) -> tuple[str, ...]:
        """Returns paths to non-param non-buffer Tensor attrs on the model.

        This is only used for direct_compile. direct_compile accepts a model
        and needs to know where all of the Tensors in the model are. It knows
        that Tensors can be a part of the parameters and buffers but
        we need to explicitly tell it where all of the non-parameter non-buffer
        Tensors are.

        Example: ("foo.bar.baz",).
        """
        return self.model_extra_tensor_attrs

    def _direct_compile_specialization_key(
        self,
        attention_metadata: Any,
        input_ids: torch.Tensor,
        _positions: torch.Tensor,
        forward_batch: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[Any, ...]:
        logits_output_mode = (
            forward_batch.logits_output_mode
            if self._compiles_logits_processor()
            else None
        )
        key = (
            forward_batch.forward_mode,
            forward_batch.capture_hidden_mode,
            forward_batch.batch_size,
            input_ids.shape[0],
            logits_output_mode,
        )
        return key

    def _compiles_logits_processor(self) -> bool:
        """Whether this compiled callable includes target-model logits processing."""

        pp_group = self.model_runner.pp_group
        if pp_group is not None and not pp_group.is_last_rank:
            # Only the final pipeline stage runs the logits processor.
            return False
        runner_model = self.model_runner.model
        # A wrapper may compile its language model with the logits processor
        # inside (Kimi); one that keeps the logits eager (Gemma4) does not.
        owns_logits = self.owner is runner_model or (
            self.owner is getattr(runner_model, "language_model", None)
            and getattr(self.owner, "logits_processor", None) is not None
        )
        return owns_logits and not bool(
            getattr(self.model_runner, "is_draft_worker", False)
        )

    def _pp_warmup_kwargs(self, num_tokens: int) -> dict[str, Any]:
        """The ``pp_proxy_tensors`` kwarg a non-first pipeline stage warms up with.

        The proxy carries the previous stage's hidden states and residual at the
        padded token bucket, in the model dtype, on the device; the first stage
        (and a non-pipelined model) gets no proxy.
        """
        pp_group = self.model_runner.pp_group
        if pp_group is None or pp_group.is_first_rank:
            return {}
        hidden_size = self.model_runner.model_config.hidden_size
        # Serving trims the wave's max-bucket buffers to the batch's bucket
        # (_fit_pp_proxy_rows), so the program takes row views of them; warm
        # with the same views, or each bucket copies its view on first use.
        rows = self.max_padded_tokens()

        def zeros(shape: tuple[int, int]) -> torch.Tensor:
            buffer = torch.zeros(
                (rows, shape[1]),
                dtype=self.model_runner.dtype,
                device=self.model_runner.device,
            )
            return buffer[: shape[0]]

        proxy = {
            "hidden_states": zeros((num_tokens, hidden_size)),
            "residual": zeros((num_tokens, hidden_size)),
        }
        # Serving hands the aux capture slots on with the activations, so
        # trace them too or the graph signature differs from the real one.
        num_aux = num_relayed_aux_slots(
            self.model_runner.spec_aux_config, self.model_runner.model
        )
        if num_aux:
            proxy[AUX_HIDDEN_STATES_PROXY_KEY] = zeros(
                (num_tokens, hidden_size * num_aux)
            )
        return {"pp_proxy_tensors": PPProxyTensors(proxy)}

    def _distributed_top_k_width(self) -> int | None:
        if not self._compiles_logits_processor():
            return None
        from sglang_tpu.sampler import (
            distributed_top_k_candidate_width_for_model,
        )

        return distributed_top_k_candidate_width_for_model(self.model_runner.model)

    def _warmup_logits_output_modes(
        self, batch_size: int
    ) -> tuple[LogitsOutputMode, ...]:
        """Return every logits mode reachable at this physical batch size."""

        if self._distributed_top_k_width() is None:
            return (LogitsOutputMode.FULL_VOCAB,)
        spec_algorithm = getattr(self.model_runner, "spec_algorithm", None)
        if spec_algorithm is not None and not spec_algorithm.is_none():
            return (LogitsOutputMode.FULL_VOCAB,)
        from sglang_tpu.sampler import (
            distributed_top_k_min_batch_size,
        )

        if batch_size < distributed_top_k_min_batch_size():
            return (LogitsOutputMode.FULL_VOCAB,)
        return (
            LogitsOutputMode.DISTRIBUTED_TOP_K,
            LogitsOutputMode.FULL_VOCAB,
        )

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Bucket a normal or target forward before attention planning."""
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        if self._cp_batch_active(forward_batch):
            real_q, real_b = prepare_tpu_cp_forward_batch(
                forward_batch,
                t_bucketer=self.t_bucketer,
                b_bucketer=self.b_bucketer,
                extend_bs=self.extend_bs,
                cp_size=self._cp_strategy.cp_size,
                cp_rank=self._cp_strategy.cp_rank,
                device=self.model_runner.device,
            )
        else:
            real_q, real_b = _pad_forward_batch_inputs(
                forward_batch,
                self.t_bucketer,
                self.b_bucketer,
                extend_bs=self.extend_bs,
            )
            self._upload_host_padded_inputs(forward_batch)
        self._upload_host_inputs(forward_batch)
        setattr(forward_batch, _REAL_Q_KEY, real_q)
        setattr(forward_batch, _REAL_B_KEY, real_b)
        spec_info = forward_batch.spec_info
        if spec_info is not None:
            spec_info.record_padding(
                logical_batch_size=real_b,
                physical_batch_size=int(forward_batch.seq_lens.shape[0]),
                logical_num_tokens=real_q,
                physical_num_tokens=int(forward_batch.input_ids.shape[0]),
            )

    def _supports_padded_sampling_outputs(self) -> bool:
        return self._compiles_logits_processor()

    def keeps_padded_mm_embeds(self, batch: ForwardBatch) -> bool:
        """Whether the shim hands mm_input_embeds back at the token bucket."""
        return False

    def keep_outputs_padded(self, batch: ForwardBatch) -> bool:
        from sglang_tpu.sampler import supports_padded_sampling

        # EAGLE warmup and serving have different sampling_info values. Avoid
        # guarding that metadata in compiled paths which cannot use this mode.
        if not tpu_envs.SGLANG_TPU_HOST_PADDING.get() or batch.spec_info is not None:
            return False
        info = getattr(batch, "sampling_info", None)
        if info is None or not supports_padded_sampling(info):
            return False
        observer = getattr(self.model_runner, "sampling_observer", None)
        return (
            # Beam-member tails are sliced by the worker before sampling.
            len(info) == getattr(batch, _REAL_B_KEY, batch.seq_lens.shape[0])
            and (
                batch.forward_mode.is_decode()
                or batch.forward_mode.is_extend_without_speculative()
            )
            and not batch.is_prefill_only
            and not batch.return_logprob
            and batch.capture_hidden_mode == CaptureHiddenMode.NULL
            and not batch.return_pooled_hidden_states
            and not batch.return_hidden_states_before_norm
            and ordinary_compiled_forward(self.model_runner.model_config)
            and (observer is None or not observer.is_active(info))
            and self._supports_padded_sampling_outputs()
        )

    def _upload_host_inputs(self, forward_batch: ForwardBatch) -> None:
        if (
            not tpu_envs.SGLANG_TPU_HOST_PADDING.get()
            or forward_batch.spec_info is not None
        ):
            return
        # Each field is checked on its own: tokens, lengths and extend metadata
        # are placed by separate decisions, so one device-side field does not
        # mean the others were uploaded.
        # Only staged forward inputs move here; CPU mirrors stay on the host.
        # Mamba tracking followed seq_lens.device during padding, so host-owned
        # lengths defer those uploads to this boundary as well.
        for name in (
            *_MAMBA_TRACK_FIELDS,
            "seq_lens",
            "orig_seq_lens",
            "req_pool_indices",
            "input_ids",
            "positions",
            "mrope_positions",
            "out_cache_loc",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
        ):
            value = getattr(forward_batch, name, None)
            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                setattr(
                    forward_batch,
                    name,
                    value.to(self.model_runner.device, non_blocking=True),
                )

    def _upload_host_padded_inputs(self, forward_batch: ForwardBatch) -> None:
        """HOST_PADDED inputs were padded on host above; upload them at the bucket shape."""
        sampling_info = forward_batch.sampling_info
        if sampling_info is None or not is_host_padded(sampling_info):
            return
        # DP attention's bucket metadata is built on input_ids' (host) device too.
        for name in (*_TOKEN_FIELDS, *_BS_FIELDS, *_DP_COUNT_FIELDS):
            if name == "req_pool_indices_cpu":
                continue
            value = getattr(forward_batch, name, None)
            if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                setattr(
                    forward_batch,
                    name,
                    value.to(self.model_runner.device, non_blocking=True),
                )

    def _cp_batch_active(self, forward_batch: ForwardBatch) -> bool:
        """Whether this extend batch runs the context-parallel program."""
        return (
            self._cp_strategy is not None
            and forward_batch.attn_cp_metadata is not None
            and is_cp_active(forward_batch)
        )

    def _install_dp_idle_as_decode(self) -> None:
        """Normalize idle DP batches before model execution.

        An idle rank may take a different code path than decode, but with DP
        attention all ranks must run the same code. The base policy relabels an
        idle batch as DECODE; specialized policies may override the phase.

        Upstream removed ``ModelRunner.forward_idle`` (idle now flows through
        ``ModelRunner.forward`` -> the eager runner), so hook ``forward`` and
        relabel idle batches at entry; the ``is_idle()`` guard keeps decode /
        extend batches untouched.
        """
        model_runner = self.model_runner
        # Idempotent: a re-install (tests, warmups) must not nest a second
        # wrapper around an already-wrapped forward.
        if getattr(model_runner.forward, "_tpu_idle_as_decode", False):
            return
        orig_forward = model_runner.forward

        def forward(forward_batch, *args, **kwargs):
            # Relabel only; forward()'s decode path (eager runner) runs
            # prepare_forward_batch exactly once. Do NOT prepare here — the
            # shim's _prepare is not idempotent, and a second call overwrites
            # _REAL_Q/_REAL_B with the already-padded size, so outputs never get
            # unpadded (garbage decode under DP attention).
            if (
                forward_batch.forward_mode.is_idle()
                and is_dp_attention_enabled()
                # Any DP-synced idle (incl. all-idle): an IDLE mode reaching
                # the compiled forward can't match any variant's mode guard.
                and forward_batch.global_num_tokens_cpu
            ):
                self._normalize_dp_idle_forward(forward_batch)
            return orig_forward(forward_batch, *args, **kwargs)

        forward._tpu_idle_as_decode = True
        model_runner.forward = forward

    def _normalize_dp_idle_forward(self, forward_batch) -> None:
        """Relabel an ordinary idle DP batch for the existing decode pad path."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        forward_batch.forward_mode = ForwardMode.DECODE
        # Scheduler idle locations are int64; active decode and warmup use
        # the allocator dtype. Normalize before padding materializes KV slots.
        forward_batch.out_cache_loc = forward_batch.out_cache_loc.to(
            dtype=self._dummy_out_cache_loc_dtype(ForwardMode.DECODE)
        )
        self._match_busiest_dp_decode_rows(forward_batch)

    def _match_busiest_dp_decode_rows(self, forward_batch: ForwardBatch) -> None:
        """Give an idle decode rank as many rows as the busiest DP rank.

        The shim unpads decode outputs to this rank's entry of
        ``original_global_num_tokens_cpu``. At zero rows the eager tail skips
        work the busy ranks run (an LM head outside the compiled decoder skips
        its attention-TP gather), and TPU collectives pair up by issue order
        across every rank, so the busy ranks would wait forever.
        """
        # With a peer extending, the counts are tokens, not rows, and
        # prepare_mlp_sync_batch materializes this rank as a dummy extend.
        if forward_batch.is_extend_in_batch or forward_batch.seq_lens.numel() > 0:
            return
        counts = forward_batch.original_global_num_tokens_cpu
        rows = max(counts)
        if rows == 0:
            return

        self._materialize_dp_idle_decode(forward_batch, rows=rows)
        # Rebind a copy: the list object is shared with the ScheduleBatch.
        counts = list(counts)
        counts[get_parallel().attn_dp_rank] = rows
        forward_batch.original_global_num_tokens_cpu = counts

    def _materialize_dp_idle_decode(
        self, forward_batch: ForwardBatch, *, rows: int = 1
    ) -> None:
        """Represent an idle decode rank with physical dummy sequences."""
        if forward_batch.seq_lens.numel() > 0:
            return

        rows = max(int(rows), 1)
        device = forward_batch.input_ids.device
        seq_dtype = forward_batch.seq_lens.dtype
        forward_batch.batch_size = rows
        forward_batch.seq_lens = torch.ones(rows, dtype=seq_dtype, device=device)
        forward_batch.seq_lens_cpu = torch.ones(rows, dtype=seq_dtype)
        forward_batch.seq_lens_sum = rows
        forward_batch.input_ids = torch.zeros(
            rows, dtype=forward_batch.input_ids.dtype, device=device
        )
        forward_batch.positions = torch.zeros(rows, dtype=torch.int64, device=device)
        if isinstance(forward_batch.out_cache_loc, torch.Tensor):
            forward_batch.out_cache_loc = torch.zeros(
                rows, dtype=forward_batch.out_cache_loc.dtype, device=device
            )
        if isinstance(forward_batch.req_pool_indices, torch.Tensor):
            pool_dtype = forward_batch.req_pool_indices.dtype
            forward_batch.req_pool_indices = torch.zeros(
                rows, dtype=pool_dtype, device=device
            )
            forward_batch.req_pool_indices_cpu = torch.zeros(rows, dtype=pool_dtype)

    def materialize_dp_idle_extend(
        self, forward_batch: ForwardBatch, num_tokens: int
    ) -> bool:
        """Materialize an empty DP rank as one TPU extend request.

        This belongs to the decoder policy rather than the speculative policy:
        normal and speculative extend/prefill forwards both need every DP rank
        to execute the same compiled shape. Keeping the construction here also
        avoids embedding TPU-specific compile inputs in ``ForwardBatch``.
        """
        if forward_batch.seq_lens.shape[0] != 0:
            return False

        device = forward_batch.seq_lens.device
        seq_dtype = forward_batch.seq_lens.dtype
        forward_batch.extend_num_tokens = num_tokens
        forward_batch.extend_seq_lens = torch.tensor(
            [num_tokens], dtype=torch.int32, device=device
        )
        forward_batch.extend_prefix_lens = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        forward_batch.extend_start_loc = torch.zeros(
            1, dtype=torch.int32, device=device
        )
        forward_batch.seq_lens = torch.tensor(
            [num_tokens], dtype=seq_dtype, device=device
        )
        forward_batch.seq_lens_sum = int(num_tokens)
        if forward_batch.seq_lens_cpu is not None:
            forward_batch.seq_lens_cpu = torch.tensor([num_tokens], dtype=seq_dtype)
        forward_batch.extend_prefix_lens_cpu = [0]
        forward_batch.extend_seq_lens_cpu = [int(num_tokens)]
        forward_batch.extend_logprob_start_lens_cpu = [0]
        forward_batch.batch_size = 1
        forward_batch.input_ids = torch.zeros(
            num_tokens, dtype=forward_batch.input_ids.dtype, device=device
        )
        forward_batch.positions = torch.arange(
            num_tokens, dtype=torch.int64, device=device
        )
        if isinstance(forward_batch.out_cache_loc, torch.Tensor):
            forward_batch.out_cache_loc = torch.zeros(
                num_tokens,
                # Scheduler idle batches carry empty int64 locations even when
                # the live allocator and precompiled extend graph use int32.
                dtype=self._dummy_out_cache_loc_dtype(forward_batch.forward_mode),
                device=device,
            )
        if isinstance(forward_batch.req_pool_indices, torch.Tensor):
            pool_dtype = forward_batch.req_pool_indices.dtype
            forward_batch.req_pool_indices = torch.zeros(
                1, dtype=pool_dtype, device=device
            )
            if isinstance(forward_batch.req_pool_indices_cpu, torch.Tensor):
                forward_batch.req_pool_indices_cpu = torch.zeros(1, dtype=pool_dtype)
        return True

    def set_recompilation_checker(self, checker: Any, model_label: str = "") -> None:
        """Forward serving-time recompilation checks to the installed shim."""
        self.compiled_forward_shim.set_recompilation_checker(
            checker, model_label=model_label
        )

    def precompile(self) -> list[PendingCall]:
        """Schedule decoder warmups for this policy's bucket grid."""
        overall = time.perf_counter()
        disagg_mode = get_disagg().disaggregation_mode
        pending = []
        if disagg_mode != "decode":
            pending += self._precompile_extend()
            self._assert_cp_family_compiled()
        if disagg_mode != "prefill":
            pending += self._precompile_decode()
        spec_algorithm = getattr(self.model_runner, "spec_algorithm", None)
        host_padded_prefill = (
            tpu_envs.SGLANG_TPU_HOST_PADDING.get()
            and spec_algorithm is not None
            and spec_algorithm.is_none()
        )
        if host_padded_prefill:
            check_host_padding_config(self.model_runner)
        if (
            host_padded_prefill
            and get_exec().kernel.sampling_backend == "tpu"
            and ordinary_compiled_forward(self.model_runner.model_config)
        ):
            # Execute after queued model warmups, before the checker is armed.
            pending.append(PendingCall(self._warmup_host_padding))
        logger.info(
            "[decoder compile policy] queued precompile in %.1fs.",
            time.perf_counter() - overall,
        )
        return pending

    def _warmup_host_padding(self) -> None:
        from sglang.srt.arg_groups.overrides import resolving_view
        from sglang_tpu.compilation.host_padding_warmup import (
            warmup_host_relay,
            warmup_sequence_lengths,
        )
        from sglang_tpu.sampling.warmup import warmup_host_sampling

        runner = self.model_runner
        if (
            not ordinary_compiled_forward(runner.model_config)
            or resolving_view(runner.server_args).dllm_algorithm is not None
            or runner.req_to_token_pool.req_to_token.device.type != "cpu"
        ):
            return
        buckets = sorted(set(self.b_bucketer.buckets) | {self._extend_compile_cap()})
        backends = (
            runner.attn_backend,
            runner.decode_attn_backend,
            *(runner.decode_attn_backend_group or ()),
        )
        max_live = min(runner.max_running_requests, max(buckets))
        runner.req_to_token_pool._tpu_relay_bucketer = self.b_bucketer
        from sglang_tpu.allocator_tpu import host_return_allocators

        for allocator in host_return_allocators(runner.token_to_kv_pool_allocator):
            allocator.defer_decode_upload = True
        warmup_sequence_lengths(runner.device, buckets)
        for backend in {id(value): value for value in backends}.values():
            warmup = getattr(backend, "warmup_distributions", None)
            if warmup is not None:
                warmup(max_live, runner.device)
        warmup_host_sampling(
            runner.sampler,
            buckets,
            runner.model_config.vocab_size,
            runner.device,
            make_logits=self._sampling_warmup_logits,
        )
        warmup_host_relay(
            runner,
            self.b_bucketer,
            min(max_live, self.b_bucketer.max_val),
            extend_cap=self._extend_compile_cap(),
        )

    def _sampling_warmup_logits(self, rows: int) -> torch.Tensor:
        return torch.zeros(
            (rows, self.model_runner.model_config.vocab_size), dtype=torch.float32
        ).to(self.model_runner.device)

    def _assert_cp_family_compiled(self) -> None:
        """Raise if prefill context parallelism is on but warmup compiled no
        context-parallel extend graphs."""
        if self._cp_strategy is None or "cp" in self._compiled_extend_variants:
            return
        raise RuntimeError(
            f"{type(self).__name__} compiled no context-parallel extend graphs "
            "while prefill context parallelism is enabled. A _precompile_extend "
            "override must add _extend_cp_compile_items() to its warmup grid; "
            "without it every context-parallel batch recompiles at serve time."
        )

    def _extend_input_variants(self) -> tuple[str, ...]:
        """Input signatures an extend graph must be compiled for."""
        return ("plain",)

    def _extend_serving_variants(self) -> tuple[str, ...]:
        """Every input signature serving hands the extend graph, for each warmup grid."""
        return self._extend_input_variants()

    def _extend_cp_compile_items(self) -> list[tuple[int, int, ForwardMode, str, Any]]:
        """Warmup items of the context-parallel extend family.

        Separate from the plain grid so a ``_precompile_extend`` override can
        add the family back without restating how it is shaped.
        """
        return [
            (num_tokens, bs, ForwardMode.EXTEND, "cp", logits_output_mode)
            for num_tokens, bs in self._extend_cp_compile_cases()
            for logits_output_mode in self._warmup_logits_output_modes(bs)
        ]

    def _precompile_extend(self) -> list[PendingCall]:
        """Compile extend at each token bucket and the fixed batch cap."""
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None

        compile_cases = self._extend_compile_cases()
        if not compile_cases:
            raise ValueError(
                "No TPU extend compile shapes are valid: token buckets "
                f"{list(self.t_bucketer.buckets)} are all smaller than the "
                f"extend batch cap {self._extend_compile_cap()}."
            )

        compile_items = [
            (num_tokens, bs, mode, "plain", logits_output_mode)
            for num_tokens, bs, mode in compile_cases
            for logits_output_mode in self._warmup_logits_output_modes(bs)
        ]
        cp_items = self._extend_cp_compile_items()
        compile_items += cp_items
        # Input (prompt) logprobs need every prefill row, so they warm the
        # FULL-capture variant instead of the NULL one -- a replacement, not an
        # addition, because the shim then presents FULL on every extend and a
        # single variant covers both. Off by default, which leaves this exactly
        # the extend graph the backend has always compiled.
        extend_capture_hidden_mode = (
            CaptureHiddenMode.FULL
            if envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.get()
            else CaptureHiddenMode.NULL
        )
        logger.info(
            "[decoder compile policy] EXTEND/MIXED: %d graph variants "
            "(mode_caps=%s, token_buckets=%s, context_parallel_cases=%d)",
            len(compile_items),
            sorted({(mode.name, batch_size) for _, batch_size, mode in compile_cases}),
            list(self.t_bucketer.buckets),
            len(cp_items),
        )
        compile_range = (
            tqdm.tqdm(
                compile_items,
                desc="Compiling EXTEND/MIXED",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else compile_items
        )
        pending = []
        for num_tokens, bs, mode, variant, logits_output_mode in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            warmup = self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=mode,
                variant=variant,
                logits_output_mode=logits_output_mode,
                capture_hidden_mode=extend_capture_hidden_mode,
            )
            elapsed = time.perf_counter() - start
            warmup = self._record_compile_step(
                pending_call=warmup,
                phase=mode.name.lower(),
                duration_seconds=elapsed,
                batch_size=bs,
                num_tokens=num_tokens,
                input_variant=None if variant == "plain" else variant,
                logits_output_mode=logits_output_mode.name.lower(),
                capture_hidden_mode=extend_capture_hidden_mode.name.lower(),
            )
            if warmup is not None:
                pending.append(warmup)
            if get_parallel().tp_rank == 0:
                status, elapsed, hbm_str = _warmup_progress(warmup, start, hbm_before)
                compile_range.set_description(
                    f"{status} {mode.name} (tokens={num_tokens}, bs={bs}, "
                    f"variant={variant}, logits={logits_output_mode.name}) "
                    f"in {elapsed:.1f}s{hbm_str}"
                )
        return pending

    def _precompile_decode(self) -> list[PendingCall]:
        """Compile decode mode: vary batch size (tokens == batch size)."""
        assert self.b_bucketer is not None

        bs_buckets = list(self.b_bucketer.buckets)
        compile_items = [
            (bs, logits_output_mode)
            for bs in bs_buckets
            for logits_output_mode in self._warmup_logits_output_modes(bs)
        ]
        logger.info(
            "[decoder compile policy] DECODE: %d graph variants (bs=%s)",
            len(compile_items),
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                compile_items,
                desc="Compiling DECODE",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else compile_items
        )
        pending = []
        for bs, logits_output_mode in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            warmup = self._run_one(
                num_tokens=bs,
                bs=bs,
                mode=ForwardMode.DECODE,
                logits_output_mode=logits_output_mode,
            )
            elapsed = time.perf_counter() - start
            warmup = self._record_compile_step(
                pending_call=warmup,
                phase="decode",
                duration_seconds=elapsed,
                batch_size=bs,
                num_tokens=bs,
                logits_output_mode=logits_output_mode.name.lower(),
            )
            if warmup is not None:
                pending.append(warmup)
            if get_parallel().tp_rank == 0:
                status, elapsed, hbm_str = _warmup_progress(warmup, start, hbm_before)
                compile_range.set_description(
                    f"{status} DECODE (bs={bs}, logits={logits_output_mode.name}) "
                    f"in {elapsed:.1f}s{hbm_str}"
                )
        return pending

    def _make_warmup_inputs(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        variant: str = "plain",
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: SpecInput | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Build generic text decoder warmup inputs."""
        if variant == "cp":
            forward_batch = self._make_cp_dummy_forward_batch(
                local_tokens=num_tokens, bs=bs, capture_hidden_mode=capture_hidden_mode
            )
        else:
            forward_batch = self._make_dummy_forward_batch(
                num_tokens=num_tokens,
                bs=bs,
                mode=mode,
                capture_hidden_mode=capture_hidden_mode,
                spec_info=spec_info,
            )
        return (
            (forward_batch.input_ids, forward_batch.positions, forward_batch),
            self._pp_warmup_kwargs(num_tokens),
        )

    def _make_cp_dummy_forward_batch(
        self, *, local_tokens: int, bs: int, capture_hidden_mode: CaptureHiddenMode
    ) -> ForwardBatch:
        """EXTEND dummy whose zigzag layout fills exactly ``local_tokens``
        rows on every context-parallel rank, with the CP-v2 metadata built
        the way serving builds it."""
        assert self._cp_strategy is not None
        request_lens = warmup_request_lengths(
            local_rows=local_tokens, cp_size=self._cp_strategy.cp_size, max_reqs=bs
        )
        layout = _DummyForwardBatchFields.cp_extend(
            request_lens,
            device=self.model_runner.device,
            slot=self._dummy_kv_slot(),
            out_cache_loc_dtype=self._dummy_out_cache_loc_dtype(ForwardMode.EXTEND),
        )
        forward_batch = self._forward_batch_from_layout(
            layout,
            bs=len(request_lens),
            mode=ForwardMode.EXTEND,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=None,
        )
        if not is_cp_active(forward_batch):
            raise RuntimeError(
                "context-parallel warmup batch is not context-parallel active; "
                "check the zigzag strategy configuration."
            )
        prepare_cp_forward(forward_batch)
        return forward_batch

    def _run_one(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        variant: str = "plain",
        logits_output_mode: LogitsOutputMode = LogitsOutputMode.FULL_VOCAB,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: SpecInput | None = None,
        attn_backend: Any | None = None,
        profile_prefix: int | None = None,
    ) -> PendingCall | float | None:
        """Run one warmup, defer async execution, or time synchronous profiling."""
        self._compiled_extend_variants.add(variant)
        args, kwargs = self._make_warmup_inputs(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            variant=variant,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        forward_batch = (
            kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
        )
        prepared_batch = self.model_runner.prepare_dummy_forward_batch(forward_batch)
        if prepared_batch is not forward_batch:
            if "forward_batch" in kwargs:
                kwargs["forward_batch"] = prepared_batch
            else:
                args = (*args[:2], prepared_batch, *args[3:])
            forward_batch = prepared_batch
        if profile_prefix is not None:
            from sglang_tpu.pp_profile import set_profile_inputs

            args, kwargs = set_profile_inputs(
                self, forward_batch, args, kwargs, num_tokens, profile_prefix
            )

        if spec_info is None:
            if logits_output_mode is LogitsOutputMode.DISTRIBUTED_TOP_K:
                forward_batch.sampling_info = self._make_top_k_warmup_sampling_info(bs)
            elif logits_output_mode is LogitsOutputMode.FULL_VOCAB:
                forward_batch.sampling_info = (
                    self._make_full_vocab_warmup_sampling_info(bs)
                )
            else:
                raise ValueError(f"Invalid warmup logits mode: {logits_output_mode}")
        self._seed_dummy_req_to_token(forward_batch)

        model = self.model_runner.model
        # Set the DP metadata before the pad shim runs, just like serving does,
        # so precompile traces the same DP graph serving uses. Without it,
        # precompile traces a different graph.
        if is_dp_attention_enabled():
            self.set_dp_attention_metadata(forward_batch, num_tokens, bs, mode)
        model.prepare_forward_batch(forward_batch)
        forward_batch.logits_output_mode = logits_output_mode

        forward_attn_backend = attn_backend or self.model_runner.attn_backend
        forward_attn_backend.init_forward_metadata(forward_batch)

        # The compiled forward resolves its attention backend from the forward
        # context (upstream moved attn_backend off ForwardBatch into a
        # ForwardContext). Serving establishes it in ModelRunner.forward, but the
        # warmup path invokes the compiled forward directly, so set it here too.
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        # Profiling enters at the outer model: only that wrapper passes
        # input_embeds and the proxy positionally, and direct_compile resolves
        # its recorded tensor paths against that exact call structure.
        entry = model if profile_prefix is not None else self.owner

        # Match TpuGraphRunner's no_grad wrapper for Dynamo cache reuse.
        with (
            forward_context(ForwardContext(attn_backend=forward_attn_backend)),
            torch.no_grad(),
        ):
            started = time.perf_counter()
            async_warmup = getattr(self.owner.forward, "warmup_async_fn", None)
            if (
                profile_prefix is None
                and self.tracer == "direct"
                and envs.SGLANG_DIRECT_COMPILE_ASYNC.get()
                and self._supports_async_warmup()
                and async_warmup is not None
            ):
                out = call_with_reserved_frame_stack(
                    self.owner.forward.warmup_async, *args, **kwargs
                )
            else:
                out = call_with_reserved_frame_stack(entry.forward, *args, **kwargs)
            if isinstance(out, PendingCall):
                return out.then(
                    lambda result: _materialize_out(
                        result,
                        forward_attn_backend,
                        self.model_runner.token_to_kv_pool,
                    )
                )
            _materialize_out(
                out,
                forward_attn_backend,
                self.model_runner.token_to_kv_pool,
            )

        if profile_prefix is not None:
            return (time.perf_counter() - started) * 1000
        return None

    def finalize_forward_batch(self, outputs: Any, forward_batch: ForwardBatch) -> Any:
        return _unpad_decoder_outputs(outputs, forward_batch)

    def warmup_eager_logits(self, cases: Sequence[EagerLogitsCase]) -> None:
        """Warm the logits a wrapper runs eagerly after its compiled decoder, one run per case."""

    def set_dp_attention_metadata(
        self, forward_batch: ForwardBatch, num_tokens: int, bs: int, mode: ForwardMode
    ) -> None:
        """Set the DP metadata in the same way as serving.

        Serving fills in the DP padding mode and global per-rank token counts
        before the shim runs. Precompile must set the same values so the shim
        traces the identical DP graph. All ranks compile the same bucket in
        lockstep.
        """
        from sglang.srt.layers.dp_attention import DpPaddingMode

        # Draft scopes narrow attention-DP to one but retain the target's
        # gathered token counts. Match that gather, not the local topology.
        dp_size = dp_gather_width()
        device = self.model_runner.device
        global_num_tokens = [num_tokens] * dp_size
        forward_batch.dp_padding_mode = DpPaddingMode.MAX_LEN
        forward_batch.is_extend_in_batch = mode == ForwardMode.EXTEND
        forward_batch.global_num_tokens_cpu = list(global_num_tokens)
        forward_batch.original_global_num_tokens_cpu = list(global_num_tokens)
        forward_batch.global_num_tokens_gpu = torch.tensor(
            global_num_tokens, dtype=torch.int64, device=device
        )
        forward_batch.global_num_tokens_for_logprob_cpu = [bs] * dp_size
        forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
            [bs] * dp_size, dtype=torch.int64, device=device
        )

    def _dummy_kv_slot(self) -> int:
        """KV-pool slot for dummy ``req_pool_indices``/``out_cache_loc``."""
        return 0

    def _dummy_tokens_per_req(
        self, mode: ForwardMode, spec_info: SpecInput | None
    ) -> int:
        """Extend tokens per request in the dummy batch (1 for plain extend)."""
        return 1

    def _dummy_mm_inputs(self, bs: int) -> Any:
        """``mm_inputs`` field for the dummy batch (``None`` for plain decoders)."""
        return None

    def live_cache_loc_dtype(self) -> torch.dtype:
        """Dtype of the extend cache locations the live allocator hands out."""
        return self._dummy_out_cache_loc_dtype(ForwardMode.EXTEND)

    def _dummy_out_cache_loc_dtype(self, mode: ForwardMode) -> torch.dtype:
        """Match cache-location indices produced by the live allocator."""
        allocator = self.model_runner.token_to_kv_pool_allocator
        if isinstance(allocator, SWATokenToKVPoolAllocator):
            allocator = allocator.full_attn_allocator
        if not isinstance(allocator.free_pages, torch.Tensor):
            raise RuntimeError(
                "The live KV allocator must be initialized before TPU precompile."
            )
        return allocator.free_pages.dtype

    def _seed_dummy_req_to_token(self, forward_batch: ForwardBatch) -> None:
        """Seed req_to_token rows for the dummy batch.

        No-op for plain decoders: row 0 stays all-zeros (page 0, the reserved
        padding page), which is exactly the harmless state padded batches expect.
        The spec policy overrides this to seed valid page indices.
        """
        return

    @property
    def _model_is_mrope(self) -> bool:
        return self.model_runner.model_config.model_is_mrope

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
    ) -> ForwardBatch:
        """Construct a dummy ForwardBatch at the target bucket shape."""
        tokens_per_req = self._dummy_tokens_per_req(mode, spec_info)
        layout = self._dummy_layout(mode, num_tokens, bs, tokens_per_req)
        return self._forward_batch_from_layout(
            layout,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )

    def _forward_batch_from_layout(
        self,
        layout: _DummyForwardBatchFields,
        *,
        bs: int,
        mode: ForwardMode,
        capture_hidden_mode: CaptureHiddenMode,
        spec_info: Any | None,
    ) -> ForwardBatch:
        # MROPE runtime positions are real [3, num_tokens] tensors. Keep strides
        # identical here so Dynamo does not guard-miss after precompile.
        mrope_positions = None
        if self._model_is_mrope and mode.is_target_verify():
            # torch_tpu fuses the eager ops before a compiled call into one program;
            # mirror ForwardBatch.compute_spec_mrope_positions so verify reuses it.
            mrope_positions = (
                layout.positions.to(torch.int64).flatten().unsqueeze(0).repeat(3, 1)
            )
        elif self._model_is_mrope:
            mrope_positions = layout.positions.unsqueeze(0).expand(3, -1).contiguous()

        sampling_info = (
            self._make_top_k_warmup_sampling_info(bs) if spec_info is None else None
        )

        batch = ForwardBatch(
            forward_mode=mode,
            batch_size=bs,
            input_ids=layout.input_ids,
            req_pool_indices=layout.req_pool_indices,
            req_pool_indices_cpu=layout.req_pool_indices_cpu,
            seq_lens=layout.seq_lens,
            seq_lens_sum=layout.seq_lens_sum,
            orig_seq_lens=layout.orig_seq_lens,
            out_cache_loc=layout.out_cache_loc,
            seq_lens_cpu=layout.seq_lens_cpu,
            positions=layout.positions,
            return_logprob=False,
            extend_num_tokens=layout.extend_num_tokens,
            extend_seq_lens=layout.extend_seq_lens,
            extend_prefix_lens=layout.extend_prefix_lens,
            extend_start_loc=layout.extend_start_loc,
            extend_seq_lens_cpu=layout.extend_seq_lens_cpu,
            extend_prefix_lens_cpu=layout.extend_prefix_lens_cpu,
            extend_logprob_start_lens_cpu=layout.extend_logprob_start_lens_cpu,
            mrope_positions=mrope_positions,
            # Model accesses ``capture_hidden_mode.need_capture()``; default
            # of None would raise AttributeError inside the compiled region.
            capture_hidden_mode=capture_hidden_mode,
            sampling_info=sampling_info,
            spec_info=spec_info,
            # Serving copies this process-wide value from ModelRunner. Keep
            # warmup on the same speculative/non-speculative selector branch.
            spec_algorithm=getattr(self.model_runner, "spec_algorithm", None),
            mm_inputs=self._dummy_mm_inputs(bs),
        )
        self._add_dummy_mamba_tracking(batch, bs=bs, mode=mode)
        if enable_num_token_non_padded():
            # Serving sets these in init_new and ModelRunner.forward; the MoE
            # path guards on the field being present, so warmup must carry it.
            from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
            from sglang.srt.layers.cp.utils import is_mla_cp_enabled

            num_tokens = int(layout.input_ids.shape[0])
            batch.global_num_token_non_padded = torch.tensor(
                num_tokens, dtype=torch.int32, device=layout.input_ids.device
            )
            batch.global_num_token_non_padded_cpu = num_tokens
            batch.set_local_num_token_non_padded(
                sharded=(
                    batch.attn_tp_sequence_sharded
                    and not is_dsa_enable_prefill_cp()
                    and not is_mla_cp_enabled()
                ),
            )
        return batch

    def _add_dummy_mamba_tracking(
        self, batch: ForwardBatch, *, bs: int, mode: ForwardMode
    ) -> None:
        """Give warmup batches the radix tracking plan serving extend/decode carry.

        Their upload and the GDN backend's tracking metadata are eager ops keyed on
        the bucket; an all-False mask writes no snapshot.
        """
        if (
            self.model_runner.is_draft_worker
            or not isinstance(self.model_runner.req_to_token_pool, HybridReqToTokenPool)
            or not (mode.is_extend() or mode.is_decode())
            or mode.is_target_verify()
            or not get_exec().mamba.enable_mamba_extra_buffer
        ):
            return
        # The scheduler builds these on the host, as int64 / bool.
        batch.mamba_track_indices = torch.zeros(bs, dtype=torch.int64)
        batch.mamba_track_mask = torch.zeros(bs, dtype=torch.bool)
        if mode.is_extend():
            batch.mamba_track_seqlens = torch.zeros(bs, dtype=torch.int64)

    def _make_top_k_warmup_sampling_info(self, bs: int) -> SamplingBatchInfo:
        """Mirror an eligible serving batch for the common static TopK width.
        This warms candidate graphs instead of an unused no-sampling specialization."""

        device = self.model_runner.device
        top_k = min(
            _DISTRIBUTED_TOP_K_WARMUP_K,
            self._distributed_top_k_width() or _DISTRIBUTED_TOP_K_WARMUP_K,
        )
        model_config = getattr(self.model_runner, "model_config", None)
        vocab_size = int(getattr(model_config, "vocab_size", top_k))
        return SamplingBatchInfo(
            temperatures=torch.ones((bs, 1), dtype=torch.float32, device=device),
            top_ps=torch.ones(bs, dtype=torch.float32, device=device),
            top_ks=torch.full(
                (bs,),
                top_k,
                dtype=torch.int32,
                device=device,
            ),
            min_ps=torch.zeros(bs, dtype=torch.float32, device=device),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=True,
            need_min_p_sampling=False,
            vocab_size=vocab_size,
            return_sampling_masks=[False] * bs,
            device=str(device),
            uniform_top_k=top_k,
        )

    def _make_full_vocab_warmup_sampling_info(self, bs: int) -> SamplingBatchInfo:
        device = self.model_runner.device
        model_config = getattr(self.model_runner, "model_config", None)
        vocab_size = int(getattr(model_config, "vocab_size", 1))
        return SamplingBatchInfo(
            temperatures=torch.ones((bs, 1), dtype=torch.float32, device=device),
            top_ps=torch.ones(bs, dtype=torch.float32, device=device),
            top_ks=torch.full((bs,), TOP_K_ALL, dtype=torch.int32, device=device),
            min_ps=torch.zeros(bs, dtype=torch.float32, device=device),
            is_all_greedy=False,
            is_any_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=vocab_size,
            return_sampling_masks=[False] * bs,
            device=str(device),
            uniform_top_k=TOP_K_ALL,
        )

    def _dummy_layout(
        self,
        mode: ForwardMode,
        num_tokens: int,
        bs: int,
        tokens_per_req: int,
    ) -> _DummyForwardBatchFields:
        """Pick the mode-varying layout for the dummy batch."""
        device = self.model_runner.device
        if mode.is_draft_extend_v2():
            return _DummyForwardBatchFields.draft_extend(
                num_tokens,
                bs,
                tokens_per_req,
                device=device,
                page_size=get_schedule().page_size,
                out_cache_loc_dtype=self._dummy_out_cache_loc_dtype(mode),
            )
        return _DummyForwardBatchFields.default(
            num_tokens,
            bs,
            mode.is_extend() and not mode.is_target_verify(),
            tokens_per_req,
            device=device,
            slot=self._dummy_kv_slot(),
            out_cache_loc_dtype=self._dummy_out_cache_loc_dtype(mode),
        )


class Qwen3VLDecoderTpuCompilePolicy(DecoderTpuCompilePolicy):
    """Qwen3-VL decoder policy that warms multimodal prefill call shapes."""

    def _warmup_host_padding(self) -> None:
        # The compiled inner decoder excludes eager prompt embedding and logits.
        if not self._compiles_logits_processor():
            with torch.no_grad():
                for tokens, rows, mode in self._extend_compile_cases():
                    embedding = self.owner.get_input_embeddings()
                    ids = torch.zeros(tokens, dtype=torch.int64).to(
                        self.model_runner.device
                    )
                    ids.clamp_(min=0, max=embedding.num_embeddings - 1)
                    embedding(ids).cpu()
                    self._eager_warmup_logits(tokens, rows, mode).cpu()
        super()._warmup_host_padding()

    def _sampling_warmup_logits(self, rows: int) -> torch.Tensor:
        if self._compiles_logits_processor():
            return super()._sampling_warmup_logits(rows)
        return self._eager_warmup_logits(rows, rows, ForwardMode.DECODE)

    def warmup_eager_logits(self, cases: Sequence[EagerLogitsCase]) -> None:
        if self._compiles_logits_processor():
            return
        with torch.no_grad():
            for case in cases:
                self._eager_warmup_logits(
                    case.num_tokens,
                    case.bs,
                    case.mode,
                    capture_hidden_mode=case.capture_hidden_mode,
                    spec_info=case.spec_info,
                ).cpu()

    def _eager_warmup_logits(
        self,
        tokens: int,
        rows: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: SpecInput | None = None,
    ) -> torch.Tensor:
        runner = self.model_runner
        batch = self._make_dummy_forward_batch(
            tokens,
            rows,
            mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        batch.logits_output_mode = LogitsOutputMode.FULL_VOCAB

        def hidden() -> torch.Tensor:
            return torch.zeros(
                (tokens, runner.model_config.hidden_size), dtype=runner.dtype
            ).to(runner.device)

        # EAGLE3/DFlash capture hands the logits processor one hidden per captured layer.
        aux_hidden_states = (
            AuxHiddenStateList(hidden() for _ in self.owner.layers_to_capture)
            if runner.model.capture_aux_hidden_states
            else None
        )
        hidden_states = hidden()
        # torch_tpu fuses pending eager ops into one program; serving starts the
        # logits right after the compiled call, so flush the setup ops first.
        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            if isinstance(value, torch.Tensor) and value.device.type == "tpu":
                value.cpu()
        for tensor in (hidden_states, *(aux_hidden_states or ())):
            tensor.cpu()
        return runner.model.logits_processor(
            batch.input_ids,
            hidden_states,
            runner.model.lm_head,
            batch,
            aux_hidden_states,
        ).next_token_logits

    def install(self) -> bool:
        outer = getattr(self.model_runner, "model", None)
        if (
            hasattr(outer, "get_image_feature")
            and is_dp_attention_enabled()
            # A wrapper that routes every rank through one input_embeds program
            # on image steps and compiles its logits processor (Kimi-K2.5).
            and not getattr(outer, "serves_dp_attention", False)
        ):
            raise NotImplementedError(
                f"{type(outer).__name__}: TPU decoder compilation of this "
                "multimodal wrapper does not support DP attention; disable "
                "--enable-torch-compile or --enable-dp-attention."
            )
        return super().install()

    @property
    def _num_deepstack(self) -> int:
        model = getattr(self.model_runner, "model", None)
        return int(getattr(model, "num_deepstack_embeddings", 0) or 0)

    def _uses_input_embeds(self, target: Any) -> bool:
        """Whether to warm ``target`` via the Qwen3-VL input_embeds path."""
        outer = getattr(self.model_runner, "model", None)
        is_multimodal = hasattr(outer, "get_image_feature")
        return is_multimodal and callable(getattr(target, "get_input_embeddings", None))

    def _extend_input_variants(self) -> tuple[str, ...]:
        """Split text from deepstack: deepstack adds a Dynamo-guarded kwarg."""
        if not self._uses_input_embeds(self.owner):
            return super()._extend_input_variants()
        if self._num_deepstack > 0:
            return ("deepstack", "text_only")
        return ("text_only",)

    def _extend_serving_variants(self) -> tuple[str, ...]:
        if not self._uses_input_embeds(self.owner):
            return super()._extend_serving_variants()
        # "text_ids" warms positional token-ID calls used by text-only serving,
        # in addition to the input_embeds calls used by multimodal serving.
        return (*self._extend_input_variants(), "text_ids")

    def _direct_compile_specialization_key(
        self,
        attention_metadata: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: Any,
        *,
        input_embeds: torch.Tensor | None = None,
        input_deepstack_embeds: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> tuple[Any, ...]:
        logits_output_mode = (
            forward_batch.logits_output_mode
            if self._compiles_logits_processor()
            else None
        )
        key = (
            forward_batch.forward_mode,
            forward_batch.capture_hidden_mode,
            forward_batch.batch_size,
            (input_embeds if input_embeds is not None else input_ids).shape[0],
            input_embeds is not None,
            input_deepstack_embeds is not None,
            logits_output_mode,
        )
        return key

    def _decode_uses_input_ids(self) -> bool:
        """Whether the wrapper hands decode / verify raw ``input_ids``, skipping embed.

        Wrappers that set ``decode_embeds_in_language_model`` bypass
        ``general_mm_embed_routine`` in decode / verify so the embedding lands inside the
        compiled graph instead of running as eager device programs in front of
        it.
        """
        return self.model_runner.model.decode_embeds_in_language_model

    @staticmethod
    def _trim_q(t: torch.Tensor, real_q: int) -> torch.Tensor:
        """Trim the token axis, but skip the slice when it would be a no-op.

        ``prepare_forward_batch`` pads to the *next* bucket, so a batch whose size
        already lands on a bucket boundary is not padded at all. In steady-state
        decode ``real_q`` is just the running-request count and serving
        concurrencies sit on the bucket values, so this is the common case.

        A no-op slice is still a strided view, and torch_tpu tensors are opaque
        dense buffers with no stride metadata, so the backend materializes every
        view -- here as a same-shape identity copy in its own
        ``tt_jit_as_strided`` program. This hook runs on the compiled forward's
        output, outside the graph, where the per-program host cost is serial and
        unhidden.
        """
        return t if t.shape[0] == real_q else t[:real_q]

    def _supports_padded_sampling_outputs(self) -> bool:
        # Qwen-VL runs its logits processor outside the inner decoder. Both
        # hidden states and ForwardBatch must retain matching physical rows.
        return True

    def finalize_forward_batch(self, outputs: Any, forward_batch: Any) -> Any:
        """Slice the inner VL decoder output before outer logits processing."""
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        sampling_info = forward_batch.sampling_info
        if (
            sampling_info is not None
            and is_host_padded(sampling_info)
            and not forward_batch.forward_mode.is_target_verify()
        ):
            # HOST_PADDED extend samples and writes draft KV at the buckets; trimming
            # here would run the eager logits and the draft writer per prompt length.
            return outputs
        out = super().finalize_forward_batch(outputs, forward_batch)
        if real_q is None:
            return out
        if isinstance(out, torch.Tensor):
            return self._trim_q(out, real_q)
        if isinstance(out, PPProxyTensors):
            # Intermediate pipeline stages must transmit the padded proxy. The
            # next stage pads ForwardBatch again but consumes this proxy as-is.
            return out
        if isinstance(out, tuple) and len(out) == 2:
            hidden_states, aux_hidden_states = out
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = self._trim_q(hidden_states, real_q)
            if isinstance(aux_hidden_states, (list, tuple)):
                # One slice per captured layer. Empty unless EAGLE3 / DFLASH
                # populated ``layers_to_capture``, so this branch multiplies the
                # identity copies only under speculative decoding.
                aux_hidden_states = type(aux_hidden_states)(
                    (
                        self._trim_q(value, real_q)
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for value in aux_hidden_states
                )
            return hidden_states, aux_hidden_states
        return out

    def _precompile_extend(self) -> list[PendingCall]:
        if not self._uses_input_embeds(self.owner):
            return super()._precompile_extend()

        assert self.t_bucketer is not None
        assert self.b_bucketer is not None

        compile_cases = self._extend_compile_cases()
        if not compile_cases:
            raise ValueError(
                "No TPU extend compile shapes are valid: token buckets "
                f"{list(self.t_bucketer.buckets)} are all smaller than the "
                f"extend batch cap {self._extend_compile_cap()}."
            )

        variants = self._extend_serving_variants()

        compile_items = [
            (num_tokens, bs, mode, variant, logits_output_mode)
            for num_tokens, bs, mode in compile_cases
            for variant in variants
            for logits_output_mode in self._warmup_logits_output_modes(bs)
        ]
        logger.info(
            "[decoder compile policy] VL EXTEND: %d graph variants "
            "(batch_caps=%s, token_buckets=%s, variants=%s)",
            len(compile_items),
            sorted({(mode.name, batch_size) for _, batch_size, mode in compile_cases}),
            list(self.t_bucketer.buckets),
            variants,
        )
        compile_range = (
            tqdm.tqdm(
                compile_items,
                desc="Compiling VL EXTEND",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else compile_items
        )
        pending = []
        for num_tokens, bs, mode, variant, logits_output_mode in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            warmup = self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=mode,
                variant=variant,
                logits_output_mode=logits_output_mode,
            )
            elapsed = time.perf_counter() - start
            warmup = self._record_compile_step(
                pending_call=warmup,
                phase=mode.name.lower(),
                duration_seconds=elapsed,
                batch_size=bs,
                num_tokens=num_tokens,
                input_variant=variant,
                logits_output_mode=logits_output_mode.name.lower(),
            )
            if warmup is not None:
                pending.append(warmup)
            if get_parallel().tp_rank == 0:
                status, elapsed, hbm_str = _warmup_progress(warmup, start, hbm_before)
                compile_range.set_description(
                    f"{status} VL {mode.name} (tokens={num_tokens}, bs={bs}, "
                    f"variant={variant}, logits={logits_output_mode.name}) "
                    f"in {elapsed:.1f}s{hbm_str}"
                )
        self._precompile_mm_embed_placement()
        return pending

    def _precompile_mm_embed_placement(self) -> None:
        """Warm the device select that merges host-placed multimodal rows.

        It runs eagerly outside the compiled graph, so it JITs on first use at
        each token bucket unless warmed here. The deepstack grid needs no warm:
        it is bound straight from its host matrix, which is a plain transfer.
        """
        from sglang_tpu.mm_embed_tpu import apply_host_mm_embeds

        assert self.t_bucketer is not None

        embed_tokens = self.owner.get_input_embeddings()
        hidden = embed_tokens.embedding_dim
        dtype = embed_tokens.weight.dtype
        device = self.model_runner.device
        buckets = list(self.t_bucketer.buckets)

        start = time.perf_counter()
        with torch.no_grad():
            for num_tokens in buckets:
                apply_host_mm_embeds(
                    input_embeds=torch.zeros(
                        (num_tokens, hidden), dtype=dtype, device=device
                    ),
                    host_src=torch.zeros((num_tokens, hidden), dtype=dtype),
                    host_mask=torch.zeros((num_tokens, 1), dtype=torch.bool),
                )
        elapsed = time.perf_counter() - start
        logger.info(
            "[decoder compile policy] VL MM EMBED PLACEMENT: %d token buckets in %.1fs",
            len(buckets),
            elapsed,
        )
        self._record_compile_step(
            phase="mm_embed_placement",
            duration_seconds=elapsed,
            input_variant="host_placement",
        )

    def _precompile_decode(self) -> list[PendingCall]:
        if not self._uses_input_embeds(self.owner):
            return super()._precompile_decode()

        assert self.b_bucketer is not None

        bs_buckets = list(self.b_bucketer.buckets)
        # (bs, variant, logits_output_mode): warm the embeds convention VL
        # requests use and the positional one text-only serving actually
        # takes (mirrors the base policy).
        # In-graph embedding uses "text_only" (input_ids + mrope positions);
        # "text_ids" would warm unused 1-D positions.
        variants = (
            ("text_only",)
            if self._decode_uses_input_ids()
            else ("text_only", "text_ids")
        )
        compile_items = [
            (bs, variant, logits_output_mode)
            for bs in bs_buckets
            for variant in variants
            for logits_output_mode in self._warmup_logits_output_modes(bs)
        ]
        logger.info(
            "[decoder compile policy] VL DECODE: %d graph variants (bs=%s)",
            len(compile_items),
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                compile_items,
                desc="Compiling VL DECODE",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else compile_items
        )
        pending = []
        for bs, variant, logits_output_mode in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            warmup = self._run_one(
                num_tokens=bs,
                bs=bs,
                mode=ForwardMode.DECODE,
                variant=variant,
                logits_output_mode=logits_output_mode,
            )
            elapsed = time.perf_counter() - start
            warmup = self._record_compile_step(
                pending_call=warmup,
                phase="decode",
                duration_seconds=elapsed,
                batch_size=bs,
                num_tokens=bs,
                input_variant=variant,
                logits_output_mode=logits_output_mode.name.lower(),
            )
            if warmup is not None:
                pending.append(warmup)
            if get_parallel().tp_rank == 0:
                status, elapsed, hbm_str = _warmup_progress(warmup, start, hbm_before)
                compile_range.set_description(
                    f"{status} VL DECODE (bs={bs}, variant={variant}, "
                    f"logits={logits_output_mode.name}) in {elapsed:.1f}s{hbm_str}"
                )
        return pending

    def _make_warmup_inputs(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        variant: str = "plain",
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if variant == "text_ids" or not self._uses_input_embeds(self.owner):
            return super()._make_warmup_inputs(
                num_tokens=num_tokens,
                bs=bs,
                mode=mode,
                variant="plain" if variant == "text_ids" else variant,
                capture_hidden_mode=capture_hidden_mode,
                spec_info=spec_info,
            )

        forward_batch = self._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )

        positions = (
            forward_batch.mrope_positions
            if self._model_is_mrope
            else forward_batch.positions
        )

        if (
            mode.is_decode() or mode.is_target_verify()
        ) and self._decode_uses_input_ids():
            # Positions stay mrope-shaped; only the embedding moves into the graph.
            return (
                (forward_batch.input_ids, positions, forward_batch),
                self._pp_warmup_kwargs(num_tokens),
            )

        embed_tokens = self.owner.get_input_embeddings()
        input_embeds = embed_tokens(forward_batch.input_ids)

        call_kwargs: dict[str, Any] = self._pp_warmup_kwargs(num_tokens)
        num_deepstack = self._num_deepstack
        if variant == "deepstack" and mode == ForwardMode.EXTEND and num_deepstack > 0:
            call_kwargs["input_deepstack_embeds"] = torch.zeros(
                input_embeds.shape[0],
                input_embeds.shape[1] * num_deepstack,
                dtype=input_embeds.dtype,
                device=input_embeds.device,
            )

        logged_variant = (
            "deepstack" if "input_deepstack_embeds" in call_kwargs else "text_only"
        )
        if logged_variant not in self._logged_vl_warmup_variants:
            self._logged_vl_warmup_variants.add(logged_variant)
            logger.info(
                "[decoder padding] VL warmup (%s) uses input_embeds path: "
                "input_ids=None, input_embeds=%s, input_deepstack_embeds=%s",
                logged_variant,
                tuple(input_embeds.shape),
                (
                    tuple(call_kwargs["input_deepstack_embeds"].shape)
                    if "input_deepstack_embeds" in call_kwargs
                    else None
                ),
            )

        call_kwargs.update(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        return (), call_kwargs


# Kimi-K3's multimodal wrapper compiles its language model with the Qwen3-VL
# policy unchanged.
KimiK3VLDecoderTpuCompilePolicy = Qwen3VLDecoderTpuCompilePolicy
