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
    """Sync returned tensors and backend-marked KV side effects.

    attn_backend / token_to_kv_pool come from the model_runner: upstream moved
    both off ForwardBatch (attn_backend now flows through ForwardContext, and the
    KV pool is held by the attention backend / model_runner), so warmup reads them
    from the runner rather than the batch.
    """
    tensors: list[torch.Tensor] = []

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
    """Format HBM delta + total as a suffix string, or empty if unavailable."""
    if before is None or after is None:
        return " HBM: N/A"
    delta = after[0] - before[0]
    return f" HBM: +{delta:.0f} MB ({after[0]:.0f}/{after[1]:.0f} MB)"


def _unpad_decoder_outputs(out: Any, forward_batch: Any) -> Any:
    real_q = getattr(forward_batch, _REAL_Q_KEY, None)
    real_b = getattr(forward_batch, _REAL_B_KEY, None)
    if real_q is None or real_b is None:
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
        """Warm decoder graphs for this policy's bucket grid."""
        overall = time.perf_counter()
        self._precompile_extend()
        self._precompile_decode()
        logger.info(
            "[decoder compile policy] precompile done in %.1fs.",
            time.perf_counter() - overall,
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
        """Build generic text decoder warmup inputs."""
        forward_batch = self._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        return (forward_batch.input_ids, forward_batch.positions, forward_batch), {}

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
            # Serving copies this process-wide value from ModelRunner. Keep
            # warmup on the same speculative/non-speculative selector branch.
            spec_algorithm=getattr(self.model_runner, "spec_algorithm", None),
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
