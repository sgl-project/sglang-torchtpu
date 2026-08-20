"""TPU decoder compile policies for annotation-driven torch.compile."""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, ClassVar, Literal, Optional

import msgspec
import torch
import tqdm

from sglang.srt.compilation import torch_compile_bundle
from sglang.srt.compilation.support_torch_compile import DecoderCompilePolicy
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
    CompiledForwardShim,
    _extend_batch_cap,
    _pad_forward_batch_inputs,
    _restore_forward_batch_inputs,
    build_bucketers,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpecInput

logger = get_logger(__name__)

# Pure-text progress format (no animated bar) so log files and
# multi-process wrappers don't garble the output.
_COMPILE_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% Completed | "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)

# XLA TopK has a static output width. Warm the physically qualified serving
# width so the common candidate path does not compile on its first request.
_DISTRIBUTED_TOP_K_WARMUP_K = 50


def _materialize_out(out: Any, attn_backend: Any, token_to_kv_pool: Any) -> None:
    """Wait for the precompile forward's PJRT execution to finish.

    attn_backend / token_to_kv_pool come from the model_runner: upstream moved
    both off ForwardBatch (attn_backend now flows through ForwardContext, and the
    KV pool is held by the attention backend / model_runner), so warmup reads them
    from the runner rather than the batch.

    Under torch.compile the XLA compile itself is synchronous — AOTAutograd's
    SerializableAOTDispatchCompiler runs the backend during graph tracing and
    writes the compiled PjRtLoadedExecutable into the Dynamo cache. What is
    async is the PJRT execution the compiled graph dispatches when invoked:
    ``TorchTpuCompiledExecutable.__call__`` just enqueues the execution and
    returns handles. Without a ``synchronize(wait=True)`` here the precompile
    loop would keep launching bucket-N+1's forward before bucket-N's PJRT
    execution completes, which (a) inflates peak HBM (multiple in-flight
    graphs holding activation buffers) and (b) makes the per-bucket tqdm
    timings meaningless. This applies whether ``out`` is a raw ``torch.Tensor``
    (CP-v2 precompile calls ``model.model(...)`` and gets back the inner
    hidden_states directly) or a ``LogitsProcessorOutput`` (non-CP path
    through ``Qwen3MoeForCausalLM.forward``).
    """
    tensors: list[torch.Tensor] = []

    if isinstance(out, torch.Tensor):
        tensors.append(out)
    elif isinstance(out, PPProxyTensors):
        for t in out.tensors.values():
            if isinstance(t, torch.Tensor):
                tensors.append(t)
    else:
        for field in (
            "next_token_logits",
            "next_token_global_top_k_logits",
            "next_token_global_top_k_token_ids",
        ):
            value = getattr(out, field, None)
            if isinstance(value, torch.Tensor):
                tensors.append(value)

    if getattr(attn_backend, "requires_kv_sync_after_compile", False):
        pool = token_to_kv_pool
        for layer_id in range(pool.start_layer, pool.start_layer + pool.layer_num):
            tensors.append(pool.get_fused_kv_buffer(layer_id))

    if not tensors:
        return
    from torch_tpu._internal.sync.sync import synchronize as tpu_sync

    tpu_sync(tensors, wait=True)


def _get_hbm_usage_mb() -> Optional[tuple[float, float]]:
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
    before: Optional[tuple[float, float]],
    after: Optional[tuple[float, float]],
) -> str:
    """Format HBM delta + total as a suffix string, or empty if unavailable.

    Also emits a WARNING when the after-bucket HBM usage exceeds 90% of
    total — PCP precompile bakes many shapes and a runaway warmup would
    otherwise OOM mid-loop without an actionable message.
    """
    if before is None or after is None:
        return " HBM: N/A"
    delta = after[0] - before[0]
    used_pct = after[0] / max(after[1], 1.0)
    if used_pct > 0.9:
        logger.warning(
            "[decoder compile policy] HBM usage %.1f%% (%.0f/%.0f MB) — "
            "warmup nearing OOM. Consider SGLANG_TPU_PCP_ENABLE_NONCP_COMPILE=0 "
            "if it's set, or reducing --mem-fraction-static.",
            100.0 * used_pct,
            after[0],
            after[1],
        )
    return f" HBM: +{delta:.0f} MB ({after[0]:.0f}/{after[1]:.0f} MB)"


def _unpad_decoder_outputs(out: Any, forward_batch: Any) -> Any:
    real_q = getattr(forward_batch, _REAL_Q_KEY, None)
    real_b = getattr(forward_batch, _REAL_B_KEY, None)
    if real_q is None or real_b is None:
        return out

    # Handle raw Tensor return (from ``Qwen3MoeModel.forward`` when a
    # compile boundary is installed on it — e.g. under PCP CP-v2, where
    # the eager runner calls ``model.model(...)`` directly). Without this
    # branch the padded ghost slots leak past the shim into the CP-gather.
    if isinstance(out, torch.Tensor):
        # Under PCP CP-v2 the compiled model.model returns a per-rank
        # rank-major-packed tensor of shape ``(local_padded_num_tokens,
        # hidden_dim)``. Real tokens and padding are interleaved per the
        # ``PcpSequenceLayoutPlan.token_order`` reorder (rank r owns positions
        # ``[k*cycle + r*interleave, k*cycle + (r+1)*interleave)``), so the
        # padding is NOT contiguous at the tail. ``out[:real_q]`` would blindly
        # take the first ``real_q`` rows, mixing padding into the sample and
        # violating the ``pcp_size * local_padded_num_tokens`` invariant that
        # :func:`sglang.srt.layers.cp.pcp_sequence_layout.gather_and_unpack_pcp`
        # relies on to reverse the pack (via ``inverse_order``). Skip the
        # tensor-level unpad in that case — the downstream cp_gather does the
        # real unpacking. Other CP-v2 strategies (Zigzag/Interleave) do not
        # set ``pcp_attn_active``, so they keep the original truncation.
        is_pcp_active = forward_batch.pcp_attn_active
        if not is_pcp_active and real_q < out.shape[0]:
            out = out[:real_q]
        _restore_forward_batch_inputs(forward_batch, real_q, real_b)
        return out

    trim_next_token_outputs = getattr(out, "trim_next_token_outputs", None)
    if trim_next_token_outputs is not None:
        trim_next_token_outputs(real_b)
    else:
        next_token_logits = getattr(out, "next_token_logits", None)
        if next_token_logits is not None and real_b < next_token_logits.shape[0]:
            out.next_token_logits = next_token_logits[:real_b]

    hidden_states = getattr(out, "hidden_states", None)
    if hidden_states is not None and real_q < hidden_states.shape[0]:
        out.hidden_states = hidden_states[:real_q]

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
    extend_num_tokens: Optional[int]
    extend_seq_lens: Optional[torch.Tensor]
    extend_prefix_lens: Optional[torch.Tensor]
    extend_start_loc: Optional[torch.Tensor]
    extend_seq_lens_cpu: Optional[list[int]]
    extend_prefix_lens_cpu: Optional[list[int]]
    extend_logprob_start_lens_cpu: Optional[list[int]]

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


class DecoderTpuCompilePolicy(DecoderCompilePolicy):
    """Compile, pad, unpad, and precompile one decoder boundary."""

    platform = "tpu"
    fullgraph = True
    dynamic = False
    model_extra_tensor_attrs: ClassVar[tuple[str, ...]] = ()

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
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            owner=owner,
            method_name=method_name or "forward",
            method=method,
            name=name,
            label=label,
        )
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer
        self.tracer = tracer
        self.extend_bs: int | None = None
        self.compiled_forward_shim: CompiledForwardShim | None = None
        self._logged_vl_warmup_variants: set[str] = set()

    def _extend_compile_cap(self) -> int:
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        return _extend_batch_cap(self.t_bucketer, self.b_bucketer, self.extend_bs)

    def _extend_compile_cases(self) -> list[tuple[int, int, ForwardMode]]:
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None

        mode_caps = [(ForwardMode.EXTEND, self._extend_compile_cap())]
        server_args = getattr(self.model_runner, "server_args", None)
        if getattr(server_args, "enable_mixed_chunk", False):
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
        is_draft_worker = bool(getattr(self.model_runner, "is_draft_worker", False))
        compile_owner = self.owner.model if is_draft_worker else self.owner
        compile_fn = compile_owner.forward if is_draft_worker else fn

        compiled = self._compile(compile_fn)
        compiled.__signature__ = inspect.signature(compile_fn)
        setattr(
            compile_owner,
            "forward" if is_draft_worker else self.method_name,
            compiled,
        )
        if is_draft_worker:
            logits_fn = self.owner.logits_processor.forward
            compiled_logits = torch.compile(
                torch.no_grad()(logits_fn),
                backend=self.backend or current_platform.get_compile_backend(),
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
            )
            compiled_logits.__signature__ = inspect.signature(logits_fn)
            self.owner.logits_processor.forward = compiled_logits
        # max_t = smallest positive prefill/context limit (extend never runs
        # longer). max_b = the reconciled compile-batch ceiling, already grown to
        # cover the scheduler concurrency in ServerArgs.__post_init__ via
        # reconcile_tpu_batch_limits, so the grid covers every decode batch the
        # scheduler can form (Task #40).
        t_candidates = [
            c
            for c in (
                server_args.chunked_prefill_size,
                server_args.max_prefill_tokens,
                self.model_runner.model_config.context_len,
            )
            if c and c > 0
        ]
        if not t_candidates:
            raise ValueError(
                "Cannot determine max_t for TPU compile bucketing: "
                "chunked_prefill_size, max_prefill_tokens, and context_len are all unset."
            )
        max_t = min(t_candidates)
        max_b = server_args.torch_compile_max_bs
        t_bucketer, b_bucketer = build_bucketers(server_args, max_t=max_t, max_b=max_b)
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer
        if is_dp_attention_enabled():
            self.extend_bs = b_bucketer.max_val
        else:
            self.extend_bs = min(
                server_args.prefill_max_requests or b_bucketer.max_val,
                b_bucketer.max_val,
            )

        outer_model = self.model_runner.model
        existing_prep = getattr(outer_model, "prepare_forward_batch", None)

        def _prepare(forward_batch: ForwardBatch) -> None:
            if existing_prep is not None:
                existing_prep(forward_batch)
            self.prepare_forward_batch(forward_batch)

        outer_model.prepare_forward_batch = _prepare

        # Keep draft logits out of the compiled model-forward graph. The outer
        # forward calls the backbone and logits processor as independent
        # compiled regions.
        shim_target = fn if is_draft_worker else compiled
        self.compiled_forward_shim = CompiledForwardShim(shim_target, policy=self)
        setattr(self.owner, self.method_name, self.compiled_forward_shim)
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

    def _compile(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        if self.tracer == "direct":
            return self._direct_compile(fn)

        server_args = self.model_runner.server_args
        use_bundle = (
            server_args.unsafe_save_torch_compile_bundle is not None
            or server_args.unsafe_load_torch_compile_bundle is not None
        )
        compiler = torch_compile_bundle.Compiler if use_bundle else torch.compile
        return compiler(
            torch.no_grad()(fn),
            backend=self.backend or current_platform.get_compile_backend(),
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )

    def _direct_compile(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Compile directly through TorchTPU without Dynamo or AOTAutograd."""
        from sglang.srt.hardware_backend.tpu.compilation.direct_compile import (
            direct_compile,
        )

        fn = torch.no_grad()(fn)
        model = self.model_runner.model
        ignored_spec_inputs = (
            () if self.model_runner.is_draft_worker else ("forward_batch.spec_info",)
        )

        def trace_forward(
            attention_metadata: Any,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            backend = get_forward_context().attn_backend
            previous_metadata = backend.forward_metadata
            # Attention reads metadata through the backend during tracing.
            backend.forward_metadata = attention_metadata
            try:
                return fn(input_ids, positions, forward_batch, *args, **kwargs)
            finally:
                backend.forward_metadata = previous_metadata

        compiled = direct_compile(
            trace_forward,
            specialization_key=self._direct_compile_specialization_key,
            model=model,
            model_extra_tensor_attrs=self.get_model_extra_tensor_attrs(),
            kv_pool=self.model_runner.token_to_kv_pool,
            ignored_argnames=(
                "forward_batch.sampling_info",
                "forward_batch.orig_seq_lens",
                # Pallas consumes these host mirrors before the compiled model.
                "forward_batch.req_pool_indices_cpu",
                "forward_batch.seq_lens_cpu",
                *ignored_spec_inputs,
            ),
        )

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            backend = get_forward_context().attn_backend
            return compiled(
                backend.forward_metadata,
                *args,
                **kwargs,
            )

        return call

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
        _attention_metadata: Any,
        input_ids: torch.Tensor,
        _positions: torch.Tensor,
        forward_batch: Any,
        *_args: Any,
        **_kwargs: Any,
    ) -> tuple[Any, ...]:
        # Direct tracing omits sampling_info tensor inputs, although the logits
        # processor branches on a host-only static k. Standard models expose
        # their final model/layout capability so all full-logit cases share one
        # executable. Unknown wrappers retain a conservative per-k key.
        model = self.model_runner.model
        logits_processor = getattr(model, "logits_processor", None)
        lm_head = getattr(model, "lm_head", None)
        selector = getattr(
            logits_processor, "get_tpu_distributed_top_k_for_batch", None
        )
        if selector is not None and lm_head is not None:
            distributed_top_k = selector(forward_batch, lm_head)
        else:
            from sglang.srt.hardware_backend.tpu.sampler import (
                distributed_top_k_for_batch,
            )

            distributed_top_k = distributed_top_k_for_batch(
                forward_batch,
                sampling_backend=self.model_runner.server_args.sampling_backend,
            )
        return (
            forward_batch.forward_mode,
            forward_batch.capture_hidden_mode,
            forward_batch.batch_size,
            input_ids.shape[0],
            distributed_top_k,
        )

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Bucket a normal or target forward before attention planning."""
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        real_q, real_b = _pad_forward_batch_inputs(
            forward_batch,
            self.t_bucketer,
            self.b_bucketer,
            extend_bs=self.extend_bs,
        )
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

    def _install_dp_idle_as_decode(self) -> None:
        """Run idle DP batches as DECODE.

        An idle rank may take a different code path than decode, but with DP
        attention all ranks must run the same code. Relabel idle as DECODE.

        Upstream removed ``ModelRunner.forward_idle`` (idle now flows through
        ``ModelRunner.forward`` -> the eager runner), so hook ``forward`` and
        relabel idle batches at entry; the ``is_idle()`` guard keeps decode /
        extend batches untouched.
        """
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

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
                forward_batch.forward_mode = ForwardMode.DECODE
            return orig_forward(forward_batch, *args, **kwargs)

        forward._tpu_idle_as_decode = True
        model_runner.forward = forward

    def set_recompilation_checker(self, checker: Any, model_label: str = "") -> None:
        """Forward serving-time recompilation checks to the installed shim."""
        self.compiled_forward_shim.set_recompilation_checker(
            checker, model_label=model_label
        )

    def precompile(self) -> None:
        """Warm decoder graphs for this policy's bucket grid.

        Under ``--enable-prefill-cp`` with the PCP strategy every extend /
        decode routes through ``eager_runner``'s CP-v2 branch, which
        dispatches ``model.model(input_ids, positions, forward_batch,
        input_embeds=<sharded>, ...)``. The non-CP entry point uses
        ``input_embeds=None``; Dynamo caches on that type mismatch so the
        non-CP graphs are unreachable at serve time. Skip them — each
        compile is 30-60s on 30B / minutes on 480B and leaves dead XLA
        program cache behind. Set ``SGLANG_TPU_PCP_ENABLE_NONCP_COMPILE=1``
        to retain them (mixed workloads that route some traffic through
        the non-CP path).
        """
        from sglang.srt.environ import envs

        overall = time.perf_counter()
        pcp_active = self._pcp_streaming_enabled()
        keep_noncp = bool(envs.SGLANG_TPU_PCP_ENABLE_NONCP_COMPILE.get())
        if not pcp_active or keep_noncp:
            self._precompile_extend()
            self._precompile_decode()
        if pcp_active:
            self._precompile_extend_pcp()
            self._precompile_decode_pcp()
        logger.info(
            "[decoder compile policy] precompile done in %.1fs.",
            time.perf_counter() - overall,
        )

    def _pcp_streaming_enabled(self) -> bool:
        """True iff the CP-v2 PCP streaming branch will fire at serve time."""
        from sglang.srt.layers.cp.base import is_pcp
        from sglang.srt.layers.cp.utils import enable_cp_v2

        sa = self.model_runner.server_args
        return enable_cp_v2() and (sa.attn_cp_size or 1) > 1 and is_pcp()

    def _precompile_extend_pcp(self) -> None:
        """Compile extend at each PCP-aligned token bucket + the extend cap."""
        assert self.t_bucketer is not None and self.b_bucketer is not None
        pcp_size = int(self.model_runner.server_args.attn_cp_size or 1)
        bs = _extend_batch_cap(self.t_bucketer, self.b_bucketer)
        # PCP kernel requires ``padded_num_tokens % (pcp_size * q_block_size)
        # == 0`` (per-rank shard must be a multiple of the kernel's lockstep
        # floor). Rounding only to ``pcp_size`` here would let a bucket like
        # 1032 (=8*129) pass and blow up at Pallas trace time. Filter with a
        # visible log line rather than silently dropping shapes.
        from sglang.srt.layers.cp.base import get_cp_strategy

        strategy = get_cp_strategy()
        kernel_bucket = pcp_size * int(strategy.local_compile_token_multiple)
        # Also require num_tokens % bs == 0 — the dummy CP forward splits
        # tokens equally across the ``bs`` dummy requests and any leftover
        # produces a shape mismatch between ``extend_num_tokens`` (bs*floor)
        # and ``input_ids.shape[0]`` (num_tokens). Drop mis-aligned buckets
        # here with a visible log line so we never dispatch one.
        valid_shapes = [
            (t, bs)
            for t in self.t_bucketer.buckets
            if t >= bs and t % kernel_bucket == 0 and t % bs == 0
        ]
        dropped = [
            t
            for t in self.t_bucketer.buckets
            if t >= bs and not (t % kernel_bucket == 0 and t % bs == 0)
        ]
        if dropped:
            # WARNING (not INFO) because dropped buckets = ignored
            # user-configured shapes. Operators should see this by default.
            logger.warning(
                "[decoder compile policy] EXTEND-PCP: dropping %d buckets "
                "not aligned to pcp_size*q_block_size=%d or bs=%d: %s",
                len(dropped),
                kernel_bucket,
                bs,
                dropped,
            )
        if not valid_shapes:
            raise ValueError(
                "No PCP-aligned extend buckets: token buckets "
                f"{list(self.t_bucketer.buckets)} yield no shape with "
                f"t>={bs}, t%{kernel_bucket}==0, and t%{bs}==0."
            )
        logger.info(
            "[decoder compile policy] EXTEND-PCP: %d shapes (bs=%d, pcp=%d)",
            len(valid_shapes),
            bs,
            pcp_size,
        )
        compile_range = (
            tqdm.tqdm(
                valid_shapes,
                desc="Compiling EXTEND-PCP",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else valid_shapes
        )
        for num_tokens, bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one_pcp(num_tokens=num_tokens, bs=bs)
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND-PCP (tokens={num_tokens}, bs={bs}) "
                    f"in {elapsed:.1f}s{hbm_str}"
                )

    def _run_one_pcp(
        self, *, num_tokens: int, bs: int, mode: ForwardMode = ForwardMode.EXTEND
    ) -> None:
        """One dummy CP-v2 forward, mirroring eager_runner's CP branch.

        The eager CP branch calls ``model_runner.model.model(...)`` with
        rank-sharded ``input_embeds`` and ``positions`` — not
        ``Qwen3MoeForCausalLM.forward``. The compile policy still installs
        the shim on ``Qwen3MoeModel.forward`` (models/qwen3_moe.py), so
        ``self.owner.forward`` (owner is ``Qwen3MoeModel``) enters the same
        compiled graph the eager CP path hits at serve time.

        Handles both extend (``mode=EXTEND``) and decode (``mode=DECODE``);
        decode reuses the same rank-sharded dispatch since PCP now routes
        decode through the streaming kernel too (see is_cp_v2_active in
        layers/cp/utils.py).
        """
        from sglang.srt.layers.cp.utils import (
            cp_split_before_forward,
            is_cp_v2_active,
            prepare_cp_forward,
        )
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        args, _ = self._make_warmup_inputs(num_tokens=num_tokens, bs=bs, mode=mode)
        forward_batch = args[2]
        # For EXTEND: rewrite the dummy batch's extend_* fields so the
        # per-req length reflects num_tokens // bs (rather than the base
        # 1-token-per-req dummy). Without this,
        # ``PCPStreamingCPStrategy.build_extra_metadata`` sizes
        # ``plan.inverse_order`` at (bs,) and torch_tpu re-JITs the
        # cp_gather + logits_processor + argmax paths at (num_tokens,) on
        # the first real request.
        # For DECODE: extend_* fields stay None (that's how the base
        # dummy factory constructs them); ``build_extra_metadata``
        # branches on ``extend_lens_cpu is None`` and takes the pure-
        # decode path with scheduled=ones(bs), which is what serve time
        # decodes look like — so no rewrite is needed.
        if mode.is_extend(include_draft_extend_v2=True):
            if num_tokens % bs != 0:
                # Splitting num_tokens into equal per-req lengths requires
                # exact divisibility — otherwise ``extend_num_tokens`` (=
                # bs*floor) would not equal input_ids.shape[0], producing a
                # shape mismatch inside the model that shows up as either a
                # Pallas trace assert or (worse) a silently wrong compiled
                # graph shape. Filter buckets in _precompile_extend_pcp.
                raise ValueError(
                    "_run_one_pcp: num_tokens must be divisible by bs "
                    f"(got num_tokens={num_tokens}, bs={bs}). Filter the "
                    "bucket at _precompile_extend_pcp before dispatch."
                )
            _tokens_per_req = num_tokens // bs
            _dev = forward_batch.input_ids.device
            forward_batch.extend_seq_lens = torch.full(
                (bs,), _tokens_per_req, dtype=torch.int32, device=_dev
            )
            forward_batch.extend_num_tokens = int(bs * _tokens_per_req)
            forward_batch.extend_seq_lens_cpu = [_tokens_per_req] * bs
            forward_batch.extend_start_loc = torch.arange(
                0,
                bs * _tokens_per_req,
                _tokens_per_req,
                dtype=torch.int32,
                device=_dev,
            )
        self._seed_dummy_req_to_token(num_tokens)
        if is_dp_attention_enabled():
            self._set_dp_attention_metadata(forward_batch, num_tokens, bs, mode)
        self.model_runner.model.prepare_forward_batch(forward_batch)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)
        assert is_cp_v2_active(forward_batch), (
            f"PCP precompile: is_cp_v2_active(dummy_batch) is False; check "
            f"SGLANG_ENABLE_CP_V2, cp_strategy=='pcp', attn_cp_size>1, and "
            f"that the current mode ({mode.name}) satisfies "
            f"is_context_parallel_extend() or is_decode()."
        )
        prepare_cp_forward(forward_batch)
        embed_layer = self.model_runner.model.get_input_embeddings()
        complete_hidden = embed_layer(forward_batch.input_ids)
        sharded_hidden, sharded_positions = cp_split_before_forward(
            complete_hidden,
            forward_batch.positions,
            forward_batch,
        )
        from sglang.srt.layers.cp.utils import cp_gather_after_forward

        with forward_context(
            ForwardContext(attn_backend=self.model_runner.attn_backend)
        ), torch.no_grad():
            hidden_states = self.owner.forward(
                forward_batch.input_ids,
                sharded_positions,
                forward_batch,
                input_embeds=sharded_hidden,
                pp_proxy_tensors=None,
            )
            # Warm the post-forward pipeline (cp_gather + logits_processor +
            # final norm + LM head projection + argmax). Without this, the
            # first CP serve request pays ~1.8s of TPU-idle time while the
            # host JIT-compiles these eager-mode kernels
            # (``tt_jit_argmax``, ``tt_jit_layernorm_L527C0``,
            # ``tt_jit_all``, ``tt_jit_index``, etc. — verified via TPU
            # profile diff).
            model = self.model_runner.model
            # capture_aux_hidden_states=True (EAGLE-3): the model forward
            # returns (hidden_states, aux_hidden_states) tuple instead of
            # a single tensor. Mirror the eager_runner CP-v2 branch unpack
            # so the precompile shape matches serve time.
            aux_hidden_states = None
            if model.capture_aux_hidden_states:
                hidden_states, aux_hidden_states = hidden_states
            if model.pp_group.is_last_rank:
                gathered = cp_gather_after_forward(
                    hidden_states,
                    forward_batch,
                    stream=None,
                )
                out = model.logits_processor(
                    forward_batch.input_ids,
                    gathered,
                    model.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
                # Warm the sampler-side kernels (argmax + validity check)
                # that fire in every serve request but aren't triggered
                # by the model forward chain. Without this, the first
                # serve pays ~185ms of ``tt_jit_argmax`` JIT compile.
                # LogitsProcessorOutput.next_token_logits is Optional; the
                # isinstance narrows the None branch (LM head can return
                # None when spec-decode is off; see logits_processor.py).
                if isinstance(out.next_token_logits, torch.Tensor):
                    _ = torch.argmax(out.next_token_logits, dim=-1)
                    _ = torch.all(torch.isfinite(out.next_token_logits))
            else:
                out = hidden_states
            _materialize_out(
                out,
                self.model_runner.attn_backend,
                self.model_runner.token_to_kv_pool,
            )

    def _precompile_extend(self) -> None:
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

        logger.info(
            "[decoder compile policy] EXTEND/MIXED: %d cases "
            "(mode_caps=%s, token_buckets=%s)",
            len(compile_cases),
            sorted({(mode.name, batch_size) for _, batch_size, mode in compile_cases}),
            list(self.t_bucketer.buckets),
        )
        compile_range = (
            tqdm.tqdm(
                compile_cases,
                desc="Compiling EXTEND/MIXED",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else compile_cases
        )
        for num_tokens, bs, mode in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(num_tokens=num_tokens, bs=bs, mode=mode)
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled {mode.name} (tokens={num_tokens}, bs={bs}) "
                    f"in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_decode_pcp(self) -> None:
        """Warm decode on the PCP CP-v2 path (mirrors _precompile_extend_pcp).

        For decode, num_tokens == bs (one token per request). PCP requires
        the global bucket to be a multiple of pcp_size * q_block_size, so
        the effective PCP dispatch pads up to that minimum regardless of
        the caller-visible bucket. Warming at each configured decode bs
        ensures the first served decode step doesn't retrace.
        """
        assert self.b_bucketer is not None
        bs_buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[decoder compile policy] DECODE-PCP: %d shapes (bs=%s)",
            len(bs_buckets),
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling DECODE-PCP",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else bs_buckets
        )
        for bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one_pcp(num_tokens=bs, bs=bs, mode=ForwardMode.DECODE)
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled DECODE-PCP (bs={bs}) in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_decode(self) -> None:
        """Compile decode mode: vary batch size (tokens == batch size)."""
        assert self.b_bucketer is not None

        bs_buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[decoder compile policy] DECODE: %d shapes (bs=%s)",
            len(bs_buckets),
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling DECODE",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else bs_buckets
        )
        for bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(num_tokens=bs, bs=bs, mode=ForwardMode.DECODE)
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled DECODE (bs={bs}) in {elapsed:.1f}s{hbm_str}"
                )

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
        """Build generic text decoder warmup inputs.

        The 4-positional + ``pp_proxy_tensors=None`` kwarg shape matches
        the dispatch that ``Qwen2/3MoeForCausalLM.forward`` produces when
        it calls ``self.model(input_ids, positions, forward_batch,
        input_embeds, pp_proxy_tensors=pp_proxy_tensors)``. The compile
        shim is installed on the INNER model (``Qwen3MoeModel.forward``),
        so serve-time non-CP dispatch enters the shim with ``len(args)
        == 4`` and ``pp_proxy_tensors`` as the only kwarg. Precompiling
        the same shape prevents a serve-time recompile (Dynamo guards
        on len(args) and on the kwarg dict identity).
        """
        forward_batch = self._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        return (
            forward_batch.input_ids,
            forward_batch.positions,
            forward_batch,
            None,  # input_embeds — positional (None for non-CP)
        ), {"pp_proxy_tensors": None}

    def _run_one(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        variant: str = "plain",
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: SpecInput | None = None,
        attn_backend: Any | None = None,
    ) -> None:
        """Build and execute one dummy forward pass for compilation."""
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
        self._seed_dummy_req_to_token(forward_batch)

        model = self.model_runner.model
        # Set the DP metadata before the pad shim runs, just like serving does,
        # so precompile traces the same DP graph serving uses. Without it,
        # precompile traces a different graph.
        if is_dp_attention_enabled():
            self._set_dp_attention_metadata(forward_batch, num_tokens, bs, mode)
        model.prepare_forward_batch(forward_batch)

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

        # Match TpuGraphRunner's no_grad wrapper for Dynamo cache reuse.
        with (
            forward_context(ForwardContext(attn_backend=forward_attn_backend)),
            torch.no_grad(),
        ):
            out = self.owner.forward(*args, **kwargs)
            # Materialize lazy outputs and backend-marked side effects.
            _materialize_out(
                out,
                forward_attn_backend,
                self.model_runner.token_to_kv_pool,
            )

    def finalize_forward_batch(self, outputs: Any, forward_batch: ForwardBatch) -> Any:
        return _unpad_decoder_outputs(outputs, forward_batch)

    def _set_dp_attention_metadata(
        self, forward_batch: ForwardBatch, num_tokens: int, bs: int, mode: ForwardMode
    ) -> None:
        """Set the DP metadata in the same way as serving.

        Serving fills in the DP padding mode and global per-rank token counts
        before the shim runs. Precompile must set the same values so the shim
        traces the identical DP graph. All ranks compile the same bucket in
        lockstep.
        """
        from sglang.srt.layers.dp_attention import DpPaddingMode

        dp_size = self.model_runner.server_args.dp_size
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

        # MROPE runtime positions are real [3, num_tokens] tensors. Keep strides
        # identical here so Dynamo does not guard-miss after precompile.
        mrope_positions = None
        if getattr(self.model_runner, "model_is_mrope", False):
            mrope_positions = layout.positions.unsqueeze(0).expand(3, -1).contiguous()

        sampling_info = (
            self._make_top_k_warmup_sampling_info(bs) if spec_info is None else None
        )

        return ForwardBatch(
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
            mm_inputs=self._dummy_mm_inputs(bs),
        )

    def _make_top_k_warmup_sampling_info(self, bs: int) -> SamplingBatchInfo:
        """Mirror an eligible serving batch for the common static TopK width.
        This warms candidate graphs instead of an unused no-sampling specialization."""

        device = self.model_runner.device
        model_config = getattr(self.model_runner, "model_config", None)
        vocab_size = int(
            getattr(model_config, "vocab_size", _DISTRIBUTED_TOP_K_WARMUP_K)
        )
        return SamplingBatchInfo(
            temperatures=torch.ones((bs, 1), dtype=torch.float32, device=device),
            top_ps=torch.ones(bs, dtype=torch.float32, device=device),
            top_ks=torch.full(
                (bs,),
                _DISTRIBUTED_TOP_K_WARMUP_K,
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
            uniform_top_k=_DISTRIBUTED_TOP_K_WARMUP_K,
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
                page_size=self.model_runner.server_args.page_size,
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

    def install(self) -> bool:
        if is_dp_attention_enabled():
            raise NotImplementedError(
                "Qwen3-VL TPU decoder compilation does not support DP attention; "
                "disable --enable-torch-compile for this configuration."
            )
        return super().install()

    @property
    def _num_deepstack(self) -> int:
        model = getattr(self.model_runner, "model", None)
        return int(getattr(model, "num_deepstack_embeddings", 0) or 0)

    @property
    def _model_is_mrope(self) -> bool:
        return bool(getattr(self.model_runner, "model_is_mrope", False))

    def _uses_input_embeds(self, target: Any) -> bool:
        """Whether to warm ``target`` via the Qwen3-VL input_embeds path."""
        outer = getattr(self.model_runner, "model", None)
        is_multimodal = hasattr(outer, "get_image_feature")
        return is_multimodal and callable(getattr(target, "get_input_embeddings", None))

    def _direct_compile_specialization_key(
        self,
        _attention_metadata: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: Any,
        *,
        input_embeds: torch.Tensor,
        input_deepstack_embeds: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> tuple[Any, ...]:
        return (
            forward_batch.forward_mode,
            forward_batch.capture_hidden_mode,
            forward_batch.batch_size,
            input_embeds.shape[0],
            input_deepstack_embeds is not None,
        )

    def finalize_forward_batch(self, outputs: Any, forward_batch: Any) -> Any:
        """Slice the inner VL decoder output before outer logits processing."""
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        out = super().finalize_forward_batch(outputs, forward_batch)
        if real_q is None:
            return out
        if isinstance(out, torch.Tensor):
            return out[:real_q]
        if isinstance(out, PPProxyTensors):
            # Intermediate pipeline stages must transmit the padded proxy. The
            # next stage pads ForwardBatch again but consumes this proxy as-is.
            return out
        if isinstance(out, tuple) and len(out) == 2:
            hidden_states, aux_hidden_states = out
            if isinstance(hidden_states, torch.Tensor):
                hidden_states = hidden_states[:real_q]
            if isinstance(aux_hidden_states, (list, tuple)):
                aux_hidden_states = type(aux_hidden_states)(
                    value[:real_q] if isinstance(value, torch.Tensor) else value
                    for value in aux_hidden_states
                )
            return hidden_states, aux_hidden_states
        return out

    def _precompile_extend(self) -> None:
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

        variants = ["text_only"]
        if self._num_deepstack > 0:
            variants = ["deepstack", "text_only"]

        logger.info(
            "[decoder compile policy] VL EXTEND: %d shapes x %d variants "
            "(batch_caps=%s, token_buckets=%s)",
            len(compile_cases),
            len(variants),
            sorted({(mode.name, batch_size) for _, batch_size, mode in compile_cases}),
            list(self.t_bucketer.buckets),
        )
        compile_items = [
            (num_tokens, bs, mode, variant)
            for num_tokens, bs, mode in compile_cases
            for variant in variants
        ]
        compile_range = (
            tqdm.tqdm(
                compile_items,
                desc="Compiling VL EXTEND",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else compile_items
        )
        for num_tokens, bs, mode, variant in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=mode,
                variant=variant,
            )
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled VL {mode.name} (tokens={num_tokens}, bs={bs}, "
                    f"variant={variant}) in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_decode(self) -> None:
        if not self._uses_input_embeds(self.owner):
            return super()._precompile_decode()

        assert self.b_bucketer is not None

        bs_buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[decoder compile policy] VL DECODE: %d shapes (bs=%s)",
            len(bs_buckets),
            bs_buckets,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling VL DECODE",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else bs_buckets
        )
        for bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=bs,
                bs=bs,
                mode=ForwardMode.DECODE,
                variant="text_only",
            )
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled VL DECODE (bs={bs}) in {elapsed:.1f}s{hbm_str}"
                )

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
        if not self._uses_input_embeds(self.owner):
            return super()._make_warmup_inputs(
                num_tokens=num_tokens,
                bs=bs,
                mode=mode,
                variant=variant,
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

        embed_tokens = self.owner.get_input_embeddings()
        input_embeds = embed_tokens(forward_batch.input_ids)

        call_kwargs: dict[str, Any] = {}
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
