"""TPU decoder compile policies for annotation-driven torch.compile."""

from __future__ import annotations

import inspect
import time
from typing import Any, Callable, Optional

import torch
import tqdm

from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketer import Bucketer
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
    CompiledForwardShim,
    _pad_forward_batch_inputs,
    _restore_forward_batch_inputs,
    build_bucketers,
    resolve_compile_caps,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.platforms import current_platform

logger = get_logger(__name__)

# Pure-text progress format (no animated bar) so log files and
# multi-process wrappers don't garble the output.
_COMPILE_BAR_FORMAT = (
    "{desc}: {percentage:3.0f}% Completed | "
    "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
)


def _materialize_out(out: Any, attn_backend: Any, token_to_kv_pool: Any) -> None:
    """Sync returned tensors and backend-marked KV side effects.

    attn_backend / token_to_kv_pool come from the model_runner: upstream moved
    both off ForwardBatch (attn_backend now flows through ForwardContext, and the
    KV pool is held by the attention backend / model_runner), so warmup reads them
    from the runner rather than the batch.
    """
    tensors: list[torch.Tensor] = []

    next_token_logits = getattr(out, "next_token_logits", None)
    if isinstance(next_token_logits, torch.Tensor):
        tensors.append(next_token_logits)

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


class DecoderTpuCompilePolicy(CompilePolicy):
    """Compile, pad, unpad, and precompile one decoder boundary."""

    platform = "tpu"
    fullgraph = True
    dynamic = False

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
        self._logged_vl_warmup_variants: set[str] = set()

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

        signature = inspect.signature(fn)
        backend = self.backend or current_platform.get_compile_backend()
        compiled = torch.compile(
            torch.no_grad()(fn),
            backend=backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        compiled.__signature__ = signature
        setattr(self.owner, self.method_name, compiled)

        server_args = self.model_runner.server_args
        max_t, max_b = resolve_compile_caps(
            chunked_prefill_size=server_args.chunked_prefill_size,
            max_prefill_tokens=server_args.max_prefill_tokens,
            context_len=self.model_runner.model_config.context_len,
            torch_compile_max_bs=getattr(server_args, "torch_compile_max_bs", None),
            max_running_requests=self.model_runner.max_running_requests,
        )
        t_bucketer, b_bucketer = build_bucketers(server_args, max_t=max_t, max_b=max_b)
        self.t_bucketer = t_bucketer
        self.b_bucketer = b_bucketer

        outer_model = self.model_runner.model
        existing_prep = getattr(outer_model, "prepare_forward_batch", None)

        def _prepare(forward_batch: ForwardBatch) -> None:
            if existing_prep is not None:
                existing_prep(forward_batch)
            real_q, real_b = _pad_forward_batch_inputs(
                forward_batch, t_bucketer, b_bucketer
            )
            setattr(forward_batch, _REAL_Q_KEY, real_q)
            setattr(forward_batch, _REAL_B_KEY, real_b)

        outer_model.prepare_forward_batch = _prepare

        self.compiled_forward_shim = CompiledForwardShim(compiled, policy=self)
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
                and forward_batch.batch_size > 0
                and is_dp_attention_enabled()
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

        bs = self.b_bucketer.max_val
        valid_shapes = [(t, bs) for t in self.t_bucketer.buckets if t >= bs]
        if not valid_shapes:
            raise ValueError(
                "No TPU extend compile shapes are valid: token buckets "
                f"{list(self.t_bucketer.buckets)} are all smaller than the "
                f"extend batch cap {bs}."
            )

        logger.info(
            "[decoder compile policy] EXTEND: %d shapes (bs=%d, token_buckets=%s)",
            len(valid_shapes),
            bs,
            list(self.t_bucketer.buckets),
        )
        compile_range = (
            tqdm.tqdm(
                valid_shapes,
                desc="Compiling EXTEND",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else valid_shapes
        )
        for num_tokens, bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(num_tokens=num_tokens, bs=bs, mode=ForwardMode.EXTEND)
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND (tokens={num_tokens}, bs={bs}) "
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
            if get_tensor_model_parallel_rank() == 0
            else bs_buckets
        )
        for bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(num_tokens=bs, bs=bs, mode=ForwardMode.DECODE)
            if get_tensor_model_parallel_rank() == 0:
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
        spec_info: Any | None = None,
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
        spec_info: Any | None = None,
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
        forward_batch = args[2]
        self._seed_dummy_req_to_token(num_tokens)

        model = self.model_runner.model
        # Set the DP metadata before the pad shim runs, just like serving does,
        # so precompile traces the same DP graph serving uses. Without it,
        # precompile traces a different graph.
        if is_dp_attention_enabled():
            self._set_dp_attention_metadata(forward_batch, num_tokens, bs, mode)
        model.prepare_forward_batch(forward_batch)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        # The compiled forward resolves its attention backend from the forward
        # context (upstream moved attn_backend off ForwardBatch into a
        # ForwardContext). Serving establishes it in ModelRunner.forward, but the
        # warmup path invokes the compiled forward directly, so set it here too.
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )

        # Match TpuGraphRunner's no_grad wrapper for Dynamo cache reuse.
        with forward_context(
            ForwardContext(attn_backend=self.model_runner.attn_backend)
        ), torch.no_grad():
            out = self.owner.forward(*args, **kwargs)
            # Materialize lazy outputs and backend-marked side effects.
            _materialize_out(
                out,
                self.model_runner.attn_backend,
                self.model_runner.token_to_kv_pool,
            )

    def unpad_outputs(self, out: Any, forward_batch: Any) -> Any:
        return _unpad_decoder_outputs(out, forward_batch)

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

    def _dummy_tokens_per_req(self, mode: ForwardMode, spec_info: Any | None) -> int:
        """Extend tokens per request in the dummy batch (1 for plain extend)."""
        return 1

    def _dummy_mm_inputs(self, bs: int) -> Any:
        """``mm_inputs`` field for the dummy batch (``None`` for plain decoders)."""
        return None

    def _seed_dummy_req_to_token(self, num_tokens: int) -> None:
        """Seed req_to_token rows for the dummy batch.

        No-op for plain decoders: row 0 stays all-zeros (page 0, the reserved
        padding page), which is exactly the harmless state padded batches expect.
        The spec policy overrides this to seed slot 1 with a valid page index.
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
        device = self.model_runner.device
        is_extend = mode.is_extend()
        slot = self._dummy_kv_slot()

        # Token-level inputs
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        input_ids[:bs] = 1
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        out_cache_loc = torch.full(
            (num_tokens,), slot, dtype=torch.int64, device=device
        )

        # MROPE runtime positions are real [3, num_tokens] tensors. Keep strides
        # identical here so Dynamo does not guard-miss after precompile.
        mrope_positions = None
        if getattr(self.model_runner, "model_is_mrope", False):
            mrope_positions = positions.unsqueeze(0).expand(3, -1).contiguous()

        # Batch-level inputs
        seq_lens = torch.ones(bs, dtype=torch.int64, device=device)
        seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
        req_pool_indices = torch.full((bs,), slot, dtype=torch.int64, device=device)

        tokens_per_req = self._dummy_tokens_per_req(mode, spec_info)
        extend_seq_lens = (
            torch.full((bs,), tokens_per_req, dtype=torch.int32, device=device)
            if is_extend
            else None
        )
        extend_prefix_lens = (
            torch.zeros(bs, dtype=torch.int32, device=device) if is_extend else None
        )
        extend_start_loc = (
            torch.arange(
                0, bs * tokens_per_req, tokens_per_req, dtype=torch.int32, device=device
            )
            if is_extend
            else None
        )
        extend_num_tokens = bs * tokens_per_req if is_extend else None
        extend_seq_lens_cpu = [tokens_per_req] * bs if is_extend else None
        extend_prefix_lens_cpu = [0] * bs if is_extend else None
        # Match runtime extend metadata; None vs list creates guard misses.
        extend_logprob_start_lens_cpu = [0] * bs if is_extend else None

        return ForwardBatch(
            forward_mode=mode,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=bs,
            orig_seq_lens=seq_lens,
            out_cache_loc=out_cache_loc,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            return_logprob=False,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
            mrope_positions=mrope_positions,
            # Model accesses ``capture_hidden_mode.need_capture()``; default
            # of None would raise AttributeError inside the compiled region.
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
            mm_inputs=self._dummy_mm_inputs(bs),
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

    def unpad_outputs(self, out: Any, forward_batch: Any) -> Any:
        """Slice the inner VL decoder output before outer logits processing."""
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        out = super().unpad_outputs(out, forward_batch)
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

        bs = self.b_bucketer.max_val
        valid_shapes = [(t, bs) for t in self.t_bucketer.buckets if t >= bs]
        if not valid_shapes:
            raise ValueError(
                "No TPU extend compile shapes are valid: token buckets "
                f"{list(self.t_bucketer.buckets)} are all smaller than the "
                f"extend batch cap {bs}."
            )

        variants = ["text_only"]
        if self._num_deepstack > 0:
            variants = ["deepstack", "text_only"]

        logger.info(
            "[decoder compile policy] VL EXTEND: %d shapes x %d variants "
            "(bs=%d, token_buckets=%s)",
            len(valid_shapes),
            len(variants),
            bs,
            list(self.t_bucketer.buckets),
        )
        compile_items = [
            (num_tokens, bs, variant)
            for num_tokens, bs in valid_shapes
            for variant in variants
        ]
        compile_range = (
            tqdm.tqdm(
                compile_items,
                desc="Compiling VL EXTEND",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_tensor_model_parallel_rank() == 0
            else compile_items
        )
        for num_tokens, bs, variant in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.EXTEND,
                variant=variant,
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled VL EXTEND (tokens={num_tokens}, bs={bs}, "
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
            if get_tensor_model_parallel_rank() == 0
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
            if get_tensor_model_parallel_rank() == 0:
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

        call_kwargs["input_embeds"] = input_embeds
        return (None, positions, forward_batch), call_kwargs
