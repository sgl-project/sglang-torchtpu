"""Speculative-decoding TPU compile policy."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
import tqdm

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
    _restore_forward_batch_inputs,
)
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    _COMPILE_BAR_FORMAT,
    DecoderTpuCompilePolicy,
    _get_hbm_usage_mb,
    _hbm_delta_str,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecPaddingState

logger = get_logger(__name__)


# Safe values for padded sampling rows.
_TEMPERATURE_FILL = 1.0
_TOP_K_FILL = 1
_TOP_P_FILL = 1.0
# -1 keeps padded results out of real request slots.
_DRAFT_TOKEN_FILL = 0
_RETRIEVE_INDEX_FILL = -1


@dataclass(frozen=True)
class _SpecVerifyPadding:
    """Original tensors needed to restore a logical verify batch."""

    state: SpecPaddingState
    batch: Any
    verify_input: SpecInput
    seq_lens: torch.Tensor
    draft_token: torch.Tensor
    retrieve_index: torch.Tensor
    temperatures: torch.Tensor | None
    top_ks: torch.Tensor | None
    top_ps: torch.Tensor | None


def _pad_rows(tensor: torch.Tensor, pad_rows: int, fill: float) -> torch.Tensor:
    """Append ``pad_rows`` filler requests to a per-request tensor."""
    pad_spec = [0, 0] * (tensor.dim() - 1) + [0, pad_rows]
    return F.pad(tensor, pad_spec, value=fill)


def _deferred_padding_state(spec_info: SpecInput, logits_rows: int):
    """Report the padding still present in the target logits, if any."""
    state = spec_info.padding_state
    if state is None or not spec_info.keep_outputs_padded:
        return None
    if state.physical_num_tokens == state.logical_num_tokens:
        return None
    if logits_rows != state.physical_num_tokens:
        raise ValueError(
            "Deferred verify logits do not match the recorded padding: "
            f"state={state}, logits rows={logits_rows}."
        )
    return state


def _pad_first_dim(t: torch.Tensor | None, target_rows: int) -> torch.Tensor | None:
    """Zero-pad a tensor's first (row) dim to target_rows (any ndim).

    None (an unset field) passes through unchanged.
    """
    if t is None or t.shape[0] >= target_rows:
        return t
    pad = t.new_zeros((target_rows - t.shape[0], *t.shape[1:]))
    return torch.cat([t, pad], dim=0)


class SpecDecodeCompilePolicyMixin:
    """Marker base for TPU speculative-decoding compile-policy mixins."""


class Eagle3CompilePolicyMixin(SpecDecodeCompilePolicyMixin):
    """Add EAGLE3 warmup and batch shaping to a decoder compile policy."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        server_args = self.model_runner.server_args
        self.speculative_num_steps = server_args.speculative_num_steps or 0
        self.speculative_algorithm = server_args.speculative_algorithm
        self.speculative_eagle_topk = server_args.speculative_eagle_topk or 1
        self.is_draft_worker = self.model_runner.is_draft_worker

        if self.speculative_algorithm and self.speculative_algorithm != "EAGLE3":
            raise ValueError(
                f"TPU torch.compile precompilation only supports EAGLE3, "
                f"got {self.speculative_algorithm!r}."
            )

        # Draft tokens verified per request = the TARGET_VERIFY width. This
        # equals speculative_num_steps + 1 only for a topk=1 chain; use the
        # configured value so tree drafts (topk > 1) size correctly.
        self.draft_token_num = server_args.speculative_num_draft_tokens or (
            self.speculative_num_steps + 1 if self.speculative_num_steps else 0
        )
        # Width of ``spec_info.hidden_states`` the EAGLE3 draft model expects.
        # 0 for non-draft workers (the target never reads it).
        self.eagle_hidden_size = self._infer_eagle_hidden_size()

    def prepare_spec_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Bucket a draft forward before constructing attention metadata."""
        mode = forward_batch.forward_mode
        if mode.is_idle():
            return
        if mode.is_decode():
            forward_batch.spec_info.keep_outputs_padded = True
            self._pad_draft_decode(forward_batch)
        elif mode.is_draft_extend_v2():
            self._pad_draft_extend(forward_batch)

    def _infer_eagle_hidden_size(self) -> int:
        """Derive the hidden-state width the EAGLE3 draft model's forward expects.

        EAGLE3 draft models expose an ``fc`` layer whose ``in_features`` is
        ``target_hidden_size * num_aux_hidden_states``. Fall back to
        ``model_config.hidden_size``. Returns 0 for non-draft workers.
        """
        if not self.is_draft_worker:
            return 0
        model = self.model_runner.model
        fc = getattr(getattr(model, "model", None), "fc", None)
        if fc is not None and hasattr(fc, "in_features"):
            return fc.in_features
        return getattr(self.model_runner.model_config, "hidden_size", 0) or 0

    def precompile(self) -> None:
        """Warm target graphs during graph-runner initialization.

        Draft warmup is deferred to ``precompile_draft()`` because its custom
        attention backends are initialized later by the EAGLE worker.
        """
        if self.is_draft_worker:
            logger.info(
                "[spec compile policy] DRAFT: deferring warmup until its "
                "attention backends are initialized."
            )
            return
        # Run the base policy's own warmup first (base decoder EXTEND/DECODE, or a
        # model-specific policy's variants when this mixin is composed over one),
        # then layer the spec-only forward modes on top.
        super().precompile()
        self._precompile_target_verify()
        self._precompile_extend_full()

    def _precompile_target_verify(self) -> None:
        """Compile TARGET_VERIFY across the batch-size buckets."""
        assert self.b_bucketer is not None
        draft_token_num = self.draft_token_num
        bs_buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[spec compile policy] TARGET_VERIFY: %d shapes "
            "(bs=%s, draft_token_num=%d)",
            len(bs_buckets),
            bs_buckets,
            draft_token_num,
        )
        compile_range = (
            tqdm.tqdm(
                bs_buckets,
                desc="Compiling TARGET_VERIFY",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else bs_buckets
        )
        for bs in compile_range:
            num_tokens = bs * draft_token_num
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.TARGET_VERIFY,
                # The spec worker sets CaptureHiddenMode.FULL during verify so
                # the target captures all hidden states for the draft model.
                # Precompile with the same mode so the Dynamo graph matches.
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=self._make_verify_spec_info(),
            )
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled TARGET_VERIFY (bs={bs}, tokens={num_tokens})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_extend_full(self) -> None:
        """Compile EXTEND with CaptureHiddenMode.FULL at the fixed batch cap."""
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        bs = self._extend_compile_cap()
        valid_shapes = [(t, bs) for t in self.t_bucketer.buckets if t >= bs]
        logger.info(
            "[spec compile policy] EXTEND(FULL): %d shapes "
            "(bs=%d, token_buckets=%s)",
            len(valid_shapes),
            bs,
            list(self.t_bucketer.buckets),
        )
        compile_range = (
            tqdm.tqdm(
                valid_shapes,
                desc="Compiling EXTEND(FULL)",
                bar_format=_COMPILE_BAR_FORMAT,
            )
            if get_parallel().tp_rank == 0
            else valid_shapes
        )
        for num_tokens, bs in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=self._make_null_spec_info(),
            )
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND(FULL) (tokens={num_tokens}, bs={bs})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def precompile_draft(
        self,
        draft_attn_backend: Any,
        draft_extend_attn_backend: Any,
    ) -> None:
        """Compile every DRAFT-model forward family at server warmup.

        Invoked by the EAGLE worker after its custom draft attention backends
        are wired, which don't exist yet when the graph runner calls
        ``precompile()``, so the draft precompile is deferred to here. It covers
        the three forward-mode families the draft model runs at serving:

        * ``EXTEND(LAST)`` -- draft prefill right after target prefill
        * ``DECODE(LAST)`` -- the multi-step draft loop (per step backend)
        * ``DRAFT_EXTEND_V2(FULL)`` -- post-verify draft extend

        Each warmup runs the outer draft forward, exercising both independently
        compiled regions: the draft backbone and the draft logits processor.
        """
        if not self.is_draft_worker or self.b_bucketer is None:
            return
        overall = time.perf_counter()
        self._precompile_draft_prefill_extend()
        self._precompile_draft_decode(draft_attn_backend)
        self._precompile_draft_extend(draft_extend_attn_backend)
        logger.info(
            "[spec compile policy] DRAFT precompile done in %.1fs.",
            time.perf_counter() - overall,
        )
        # The serving-time recompilation checker snapped its baseline at
        # graph-runner init, before this deferred precompile; re-snapshot so
        # the graphs compiled here aren't reported as serving-time recompiles
        # on the first draft forward.
        shim = self.compiled_forward_shim
        checker = shim._recompilation_checker if shim is not None else None
        if checker is not None:
            checker.snapshot_baseline()

    def _precompile_shapes(
        self, desc: str, items: list[Any], run: Callable[[Any], str]
    ) -> None:
        compile_range = (
            tqdm.tqdm(items, desc=f"Compiling {desc}", bar_format=_COMPILE_BAR_FORMAT)
            if get_parallel().tp_rank == 0
            else items
        )
        for item in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            shape_desc = run(item)
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_str = _hbm_delta_str(hbm_before, _get_hbm_usage_mb())
                compile_range.set_description(
                    f"Compiled {desc} ({shape_desc}) in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_draft_prefill_extend(self) -> None:
        """Compile draft prefill EXTEND(LAST) token buckets at the batch cap.

        Runtime EXTEND padding uses this same fixed batch cap to avoid compiling
        the token-bucket by batch-bucket cross product.
        """
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        bs = self.b_bucketer.max_val
        valid_shapes = [(t, bs) for t in self.t_bucketer.buckets if t >= bs]
        logger.info(
            "[spec compile policy] DRAFT EXTEND(LAST): %d shapes "
            "(bs=%d, token_buckets=%s)",
            len(valid_shapes),
            bs,
            list(self.t_bucketer.buckets),
        )

        def run(shape: tuple[int, int]) -> str:
            num_tokens, bs = shape
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=self._make_draft_prefill_spec_info(),
            )
            return f"tokens={num_tokens}, bs={bs}"

        self._precompile_shapes("DRAFT EXTEND(LAST)", valid_shapes, run)

    def _precompile_draft_decode(self, draft_attn_backend: Any) -> None:
        """Compile the multi-step draft DECODE(LAST) across the bs buckets."""
        if draft_attn_backend is None:
            return
        assert self.b_bucketer is not None
        step_backends = list(draft_attn_backend.attn_backends)
        bs_buckets = list(self.b_bucketer.buckets)
        shapes = [(bs, step) for bs in bs_buckets for step in range(len(step_backends))]
        logger.info(
            "[spec compile policy] DRAFT DECODE: %d shapes (bs=%s, steps=%d, topk=%d)",
            len(shapes),
            bs_buckets,
            len(step_backends),
            self.speculative_eagle_topk,
        )

        def run(shape: tuple[int, int]) -> str:
            bs, step = shape
            self._run_one(
                num_tokens=bs * self.speculative_eagle_topk,
                bs=bs,
                mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=self._make_draft_decode_spec_info(),
                attn_backend=step_backends[step],
            )
            return f"bs={bs}, step={step}"

        self._precompile_shapes("DRAFT DECODE", shapes, run)

    def _precompile_draft_extend(self, attn_backend: Any) -> None:
        """Compile DRAFT_EXTEND_V2(FULL) across the bs buckets."""
        assert self.b_bucketer is not None
        bs_buckets = list(self.b_bucketer.buckets)
        tokens_per_req = self.draft_token_num
        logger.info(
            "[spec compile policy] DRAFT_EXTEND_V2(FULL): %d shapes "
            "(bs=%s, tokens_per_req=%d)",
            len(bs_buckets),
            bs_buckets,
            tokens_per_req,
        )

        def run(bs: int) -> str:
            num_tokens = bs * tokens_per_req
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.DRAFT_EXTEND_V2,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=self._make_draft_extend_spec_info(),
                attn_backend=attn_backend,
            )
            return f"bs={bs}, tokens={num_tokens}"

        self._precompile_shapes("DRAFT_EXTEND_V2(FULL)", bs_buckets, run)

    def _make_draft_prefill_spec_info(self) -> EagleDraftExtendInput:
        """``EagleDraftExtendInput`` stub for draft prefill EXTEND(LAST)."""
        spec_info = EagleDraftExtendInput.create_idle_input(
            device=self.model_runner.device,
            hidden_size=self.eagle_hidden_size or 1,
            dtype=self.model_runner.dtype,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = 1
        spec_info.num_tokens_for_logprob_per_req = 1
        return spec_info

    def _make_draft_decode_spec_info(self) -> EagleDraftInput:
        """``EagleDraftInput`` stub for the multi-step draft DECODE(LAST)."""
        model_config = self.model_runner.model_config
        hidden_size = (
            getattr(model_config, "spec_hidden_size", 0) or model_config.hidden_size
        )
        spec_info = self._make_idle_draft_input(
            hidden_size=hidden_size,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = self.speculative_eagle_topk
        spec_info.num_tokens_for_logprob_per_req = self.speculative_eagle_topk
        return spec_info

    def _make_draft_extend_spec_info(self) -> EagleDraftExtendInput:
        """``EagleDraftExtendInput`` stub for DRAFT_EXTEND_V2 precompilation."""
        spec_info = EagleDraftExtendInput.create_idle_input(
            device=self.model_runner.device,
            hidden_size=self.eagle_hidden_size or 1,
            dtype=self.model_runner.dtype,
            # Runtime ``prepare_for_draft_extend`` sets FULL on ForwardBatch,
            # but the EagleDraftExtendInput itself keeps its LAST default.
            # Dynamo guards both fields independently, so preserve that split.
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        # Draft extend fills the whole tree width per request; mirror what
        # ``_draft_extend_for_decode`` sets on the serving input.
        spec_info.num_tokens_per_req = self.draft_token_num
        spec_info.num_tokens_for_logprob_per_req = self.draft_token_num
        return spec_info

    def _make_null_spec_info(self) -> Any:
        """Build a minimal idle spec_info stub for non-spec forward modes.

        Carries the attributes Dynamo probes without affecting the forward
        result, so both branches of ``if spec_info is not None`` are traced.
        """
        return self._make_idle_draft_input(
            hidden_size=self.eagle_hidden_size or 1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def _make_idle_draft_input(
        self, *, hidden_size: int, capture_hidden_mode: CaptureHiddenMode
    ) -> EagleDraftInput:
        """Build a warmup stub."""
        device = self.model_runner.device
        return EagleDraftInput(
            bonus_tokens=torch.empty(0, device=device, dtype=torch.int32),
            hidden_states=torch.empty(
                (0, hidden_size), device=device, dtype=self.model_runner.dtype
            ),
            topk_p=torch.empty(
                (0, self.speculative_eagle_topk),
                device=device,
                dtype=torch.float32,
            ),
            topk_index=torch.empty(
                (0, self.speculative_eagle_topk),
                device=device,
                dtype=torch.int64,
            ),
            draft_probs=None,
            capture_hidden_mode=capture_hidden_mode,
        )

    def _make_verify_spec_info(self) -> Any:
        """Build an ``EagleVerifyInput`` for TARGET_VERIFY precompilation."""
        return EagleVerifyInput.create_idle_input(
            topk=self.speculative_eagle_topk,
            spec_steps=self.speculative_num_steps,
            num_verify_tokens=self.draft_token_num,
            device=self.model_runner.device,
        )

    def _dummy_kv_slot(self) -> int:
        # Row 0 of the req_to_token pool is the reserved padding row: free_slots
        # start at 1, so it is never allocated to a real request and its
        # contents persist for the whole run. Padded/ghost batch entries default
        # req_pool_indices to 0 and read row 0, expecting page 0. Spec precompile
        # must seed a *live* page index (see _seed_dummy_req_to_token), so it
        # keeps that off row 0 -- otherwise every padded read would permanently
        # point at a real KV page. Slot 1 is allocatable, so the seed is
        # transient (overwritten by the first real request).
        return 1

    def _dummy_tokens_per_req(
        self, mode: ForwardMode, spec_info: SpecInput | None
    ) -> int:
        # TARGET_VERIFY verifies draft_token_num tokens per request;
        # DRAFT_EXTEND_V2 extends by the same width. Other extend modes use 1.
        if mode.is_target_verify() and spec_info is not None:
            return spec_info.draft_token_num
        if mode.is_draft_extend_v2() and self.draft_token_num:
            return self.draft_token_num
        return 1

    def _dummy_mm_inputs(self, bs: int) -> list[None]:
        # Text-only spec requests produce mm_inputs=[None]*bs at runtime.
        return [None] * bs

    def _dummy_out_cache_loc_dtype(self, mode: ForwardMode) -> torch.dtype:
        # Spec V2 explicitly creates int64 locations for verify, draft decode,
        # and draft extend. Prefill-shaped EXTEND still uses the live allocator.
        if (
            mode.is_target_verify()
            or mode.is_draft_extend_v2()
            or (self.is_draft_worker and mode.is_decode())
        ):
            return torch.int64
        return super()._dummy_out_cache_loc_dtype(mode)

    def _seed_dummy_req_to_token(self, forward_batch: ForwardBatch) -> None:
        # Seed the page table so attention reads valid, in-range KV token indices.
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        if forward_batch.forward_mode.is_draft_extend_v2():
            rows = forward_batch.req_pool_indices.to("cpu")
            ctx = int(forward_batch.seq_lens[0].item())
            cols = torch.arange(ctx, dtype=req_to_token.dtype)
            vals = rows.to(req_to_token.dtype).view(-1, 1) * ctx + cols.view(1, -1)
            req_to_token[rows, :ctx] = vals
        else:
            num_tokens = int(forward_batch.input_ids.shape[0])
            req_to_token[1, :num_tokens] = 1

    def finalize_forward_batch(self, outputs: Any, forward_batch: ForwardBatch) -> Any:
        """Slice padded spec outputs back to real request shapes.

        Differs from the base decoder unpad: TARGET_VERIFY emits one logit row
        per draft token (``real_q``), and CaptureHiddenMode.LAST (draft extend)
        emits one hidden row per sequence (``real_b``).
        """
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        real_b = getattr(forward_batch, _REAL_B_KEY, None)
        if real_q is None or real_b is None:
            return outputs

        next_token_logits = getattr(outputs, "next_token_logits", None)
        hidden_states = getattr(outputs, "hidden_states", None)
        if next_token_logits is None and hidden_states is None:
            # Inner decoder boundaries return raw tensors/tuples. Delegate
            # cooperatively so a model-specific policy later in the MRO (for
            # example Qwen3-VL) can slice those outputs and restore metadata.
            return super().finalize_forward_batch(outputs, forward_batch)
        if next_token_logits is not None and real_b < next_token_logits.shape[0]:
            # TARGET_VERIFY outputs bs * draft_token_num logits (one per draft
            # token across all requests); slicing to real_b would drop verify
            # tokens, so use real_q instead.
            fm = getattr(forward_batch, "forward_mode", None)
            logit_rows = (
                real_q if (fm is not None and fm.is_target_verify()) else real_b
            )
            if logit_rows < next_token_logits.shape[0]:
                outputs.next_token_logits = next_token_logits[:logit_rows]

        if hidden_states is not None:
            # Capture-LAST (draft extend) emits one hidden row per
            # sequence (batch-dim), so slice to real_b. Every other capture mode
            # emits one row per token (real_q).
            cap = getattr(forward_batch, "capture_hidden_mode", None)
            n_real = real_b if (cap is not None and cap.is_last()) else real_q
            if n_real < hidden_states.shape[0]:
                outputs.hidden_states = hidden_states[:n_real]

        _restore_forward_batch_inputs(forward_batch, real_q, real_b)
        spec_info = forward_batch.spec_info
        if spec_info is not None:
            spec_info.keep_outputs_padded = False
        return outputs

    def _pad_draft_decode(self, forward_batch: ForwardBatch) -> None:
        """Pad the speculative draft-decode batch to a torch.compile bs bucket."""
        b_bucketer = self.b_bucketer
        if b_bucketer is None:
            return
        real_bs = int(forward_batch.seq_lens.shape[0])
        pad_b = b_bucketer.next_bucket(real_bs)
        if pad_b == real_bs:
            return

        forward_batch.seq_lens = _pad_first_dim(forward_batch.seq_lens, pad_b)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if isinstance(seq_lens_cpu, torch.Tensor):
            forward_batch.seq_lens_cpu = _pad_first_dim(seq_lens_cpu, pad_b)
        forward_batch.req_pool_indices = _pad_first_dim(
            forward_batch.req_pool_indices, pad_b
        )
        forward_batch.req_pool_indices_cpu = _pad_first_dim(
            forward_batch.req_pool_indices_cpu, pad_b
        )
        forward_batch.positions = _pad_first_dim(
            forward_batch.positions,
            forward_batch.positions.shape[0] // real_bs * pad_b,
        )
        mrope_positions = forward_batch.mrope_positions
        if (
            mrope_positions is not None
            and mrope_positions.shape[-1] < forward_batch.positions.shape[0]
        ):
            pad_tokens = forward_batch.positions.shape[0] - mrope_positions.shape[-1]
            forward_batch.mrope_positions = torch.cat(
                [
                    mrope_positions,
                    mrope_positions.new_zeros(
                        (*mrope_positions.shape[:-1], pad_tokens)
                    ),
                ],
                dim=-1,
            )
        forward_batch.out_cache_loc = _pad_first_dim(
            forward_batch.out_cache_loc,
            forward_batch.out_cache_loc.shape[0] // real_bs * pad_b,
        )

        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if isinstance(mm_inputs, list) and len(mm_inputs) < pad_b:
            forward_batch.mm_inputs = mm_inputs + [None] * (pad_b - len(mm_inputs))

        # EagleDraftInput seeds consumed by select_top_k_tokens.
        spec_info = forward_batch.spec_info
        if spec_info is not None:
            spec_info.topk_p = _pad_first_dim(spec_info.topk_p, pad_b)
            spec_info.topk_index = _pad_first_dim(spec_info.topk_index, pad_b)
            spec_info.hidden_states = _pad_first_dim(spec_info.hidden_states, pad_b)

            spec_info.record_padding(
                logical_batch_size=real_bs,
                physical_batch_size=pad_b,
                logical_num_tokens=real_bs * self.speculative_eagle_topk,
                physical_num_tokens=pad_b * self.speculative_eagle_topk,
            )

        forward_batch.batch_size = pad_b
        setattr(forward_batch, _REAL_B_KEY, real_bs)

    def finalize_spec_forward_batch(
        self, outputs: Any, forward_batch: ForwardBatch
    ) -> Any:
        """Trim draft outputs and restore the logical batch state."""
        real_bs = getattr(forward_batch, _REAL_B_KEY, None)
        spec_info = forward_batch.spec_info
        if real_bs is None:
            if spec_info is not None:
                spec_info.keep_outputs_padded = False
            return outputs
        forward_batch.batch_size = real_bs
        if spec_info is not None:
            spec_info.keep_outputs_padded = False
        # draft_probs is absent when rejection sampling is disabled.
        return tuple(t[:real_bs] if t is not None else None for t in outputs)

    def _pad_draft_extend(self, forward_batch: ForwardBatch) -> None:
        """Bucket V2 draft-extend before attention metadata is constructed."""
        if self.b_bucketer is None:
            return
        draft_extend_input = forward_batch.spec_info
        real_bs = int(forward_batch.seq_lens.shape[0])
        real_tokens = int(forward_batch.input_ids.shape[0])
        pad_b = self.b_bucketer.next_bucket(real_bs)
        pad_tokens = pad_b * self.draft_token_num
        if pad_b != real_bs:
            # Draft extend has a fixed width, so it needs no token bucket.
            # This also grows its owned token- and request-axis tensors.
            forward_batch._pad_inputs_to_size(
                model_runner=self.model_runner,
                num_tokens=pad_tokens,
                bs=pad_b,
            )
            forward_batch.batch_size = pad_b
            # Intentionally leave _REAL_Q_KEY unset. draft extend consumes the
            # output later using its own select-index gather instead of generic
            # prefix unpadding.
            setattr(forward_batch, _REAL_B_KEY, real_bs)

        # ForwardBatch.init_new may carry CPU extend metadata padded for the
        # previous prefill cap. Draft extend bypasses model.prepare_forward_batch
        # after attention metadata is marked ready, so normalize it here even
        # when the live batch already lands exactly on a batch bucket.
        for field in (
            "extend_seq_lens_cpu",
            "extend_prefix_lens_cpu",
            "extend_logprob_start_lens_cpu",
        ):
            value = getattr(forward_batch, field, None)
            if value is None:
                value = []
            elif isinstance(value, torch.Tensor):
                value = value.tolist()
            setattr(forward_batch, field, (value + [0] * pad_b)[:pad_b])

        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if isinstance(mm_inputs, list):
            forward_batch.mm_inputs = (mm_inputs + [None] * pad_b)[:pad_b]

        if isinstance(draft_extend_input, EagleDraftExtendInput):
            draft_extend_input.record_padding(
                logical_batch_size=real_bs,
                physical_batch_size=pad_b,
                logical_num_tokens=real_tokens,
                physical_num_tokens=pad_tokens,
            )

    def should_defer_verify_output_unpadding(
        self, batch: Any, verify_input: SpecInput
    ) -> bool:
        """Keep compiled verify outputs bucketed through sampling helpers."""
        verify_input.keep_outputs_padded = not batch.forward_mode.is_idle()
        return verify_input.keep_outputs_padded

    def prepare_spec_verify(
        self,
        batch: Any,
        verify_input: SpecInput,
        next_token_logits: torch.Tensor,
    ) -> _SpecVerifyPadding | None:
        """Temporarily pad scheduler inputs to match deferred target logits."""
        if batch.forward_mode.is_idle():
            return None
        state = _deferred_padding_state(verify_input, int(next_token_logits.shape[0]))
        if state is None:
            return None

        pad_bs = state.physical_batch_size - state.logical_batch_size
        sampling_info = batch.sampling_info

        # Preserve original tensors so aliases can be restored exactly.
        padding = _SpecVerifyPadding(
            state=state,
            batch=batch,
            verify_input=verify_input,
            seq_lens=batch.seq_lens,
            draft_token=verify_input.draft_token,
            retrieve_index=verify_input.retrieve_index,
            temperatures=sampling_info.temperatures,
            top_ks=sampling_info.top_ks,
            top_ps=sampling_info.top_ps,
        )

        # Build every replacement before publishing any of them, so an error
        # cannot leave the caller with a partially padded batch.
        seq_lens = _pad_rows(padding.seq_lens, pad_bs, 0)
        draft_token = _pad_rows(
            padding.draft_token,
            pad_bs * verify_input.draft_token_num,
            _DRAFT_TOKEN_FILL,
        )
        retrieve_index = _pad_rows(padding.retrieve_index, pad_bs, _RETRIEVE_INDEX_FILL)
        temperatures = (
            _pad_rows(padding.temperatures, pad_bs, _TEMPERATURE_FILL)
            if padding.temperatures is not None
            else None
        )
        top_ks = (
            _pad_rows(padding.top_ks, pad_bs, _TOP_K_FILL)
            if padding.top_ks is not None
            else None
        )
        top_ps = (
            _pad_rows(padding.top_ps, pad_bs, _TOP_P_FILL)
            if padding.top_ps is not None
            else None
        )

        # Publish padded tensors only while eagle_sample runs.
        batch.seq_lens = seq_lens
        verify_input.draft_token = draft_token
        verify_input.retrieve_index = retrieve_index
        sampling_info.temperatures = temperatures
        sampling_info.top_ks = top_ks
        sampling_info.top_ps = top_ps
        # Penalties, logit bias, and grammar data stay logical and are applied
        # only to the logical logits prefix. retrieve_next_token /
        # retrieve_next_sibling are sgl_kernel-only arguments, and draft_probs
        # is accepted but ignored by the torch rejection sampler.
        return padding

    def finalize_spec_verify(
        self,
        padding: _SpecVerifyPadding | None,
        verify_result: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ],
        bookkeeping_result: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[tuple[Any, Any, Any], tuple[Any, Any]]:
        """Restore logical inputs and trim all padded verification outputs."""
        if padding is None:
            return verify_result, bookkeeping_result
        predict, accept_lens, accept_index = verify_result
        new_seq_lens, bonus_tokens = bookkeeping_result
        sampling_info = padding.batch.sampling_info
        padding.batch.seq_lens = padding.seq_lens
        padding.verify_input.draft_token = padding.draft_token
        padding.verify_input.retrieve_index = padding.retrieve_index
        sampling_info.temperatures = padding.temperatures
        sampling_info.top_ks = padding.top_ks
        sampling_info.top_ps = padding.top_ps
        state = padding.state
        logical_bs = state.logical_batch_size
        return (
            (
                predict[: state.logical_num_tokens] if predict is not None else None,
                accept_lens[:logical_bs] if accept_lens is not None else None,
                accept_index[:logical_bs] if accept_index is not None else None,
            ),
            (
                new_seq_lens[:logical_bs] if new_seq_lens is not None else None,
                bonus_tokens[:logical_bs] if bonus_tokens is not None else None,
            ),
        )


class SpecDecoderTpuCompilePolicy(
    Eagle3CompilePolicyMixin, DecoderTpuCompilePolicy
):
    """Speculative-decoding policy over the base decoder policy.

    The common composition (unannotated spec target models). The runner composes
    ``Eagle3CompilePolicyMixin`` over other annotated base policies
    dynamically via ``TpuGraphRunner._augment_spec_policy``.
    """
