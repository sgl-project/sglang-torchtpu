"""Speculative-decoding TPU compile policy."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
import tqdm

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_disagg, get_parallel, get_spec
from sglang.srt.speculative.spec_info import SpecInput
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
    _restore_forward_batch_inputs,
)
from sglang_tpu.compilation.decoder_compile_policy import (
    _COMPILE_BAR_FORMAT,
    _get_hbm_usage_mb,
    _hbm_delta_str,
    _warmup_progress,
)
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded

logger = get_logger(__name__)


def _pad_first_dim(t: torch.Tensor | None, target_rows: int) -> torch.Tensor | None:
    """Zero-pad a tensor's first (row) dim to target_rows (any ndim).

    Args:
      t: Input tensor to pad or None.
      target_rows: Target number of rows.

    Returns:
      Padded tensor or None if input was None.
    """
    if t is None or t.shape[0] >= target_rows:
        return t
    pad = t.new_zeros((target_rows - t.shape[0], *t.shape[1:]))
    return torch.cat([t, pad], dim=0)


class BaseSpecDecodeCompilePolicyMixin:
    """Base mixin adding speculative decoding warmup and batch shaping to a decoder compile policy."""

    model_runner: ModelRunner

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        spec = get_spec()
        self.speculative_num_steps = spec.speculative_num_steps or 0
        self.speculative_algorithm = spec.speculative_algorithm
        self.is_draft_worker = self.model_runner.is_draft_worker

        # Draft tokens verified per request = the TARGET_VERIFY width.
        self.draft_token_num = spec.speculative_num_draft_tokens or (
            self.speculative_num_steps + 1 if self.speculative_num_steps else 0
        )

    def _extra_ignored_argnames(self) -> tuple[str, ...]:
        if self.is_draft_worker:
            return ()
        return ("forward_batch.spec_info",)

    def precompile(self) -> list[PendingCall]:
        """Schedule target warmups during graph-runner initialization.

        Draft warmup is deferred to ``precompile_draft()`` because its custom
        attention backends are initialized later by the EAGLE worker.
        """
        if self.is_draft_worker:
            logger.info(
                "[spec compile policy] DRAFT: deferring warmup until its "
                "attention backends are initialized."
            )
            return []
        pending = super().precompile()
        # A PD prefill pool has no SpeculativeState, so a TARGET_VERIFY dummy
        # forward would trip the backend's pool-type assert; a PD decode server
        # never runs prefill.
        disagg_mode = get_disagg().disaggregation_mode
        if disagg_mode != "prefill":
            pending += self._precompile_target_verify()
        if disagg_mode != "decode":
            pending += self._precompile_extend_full()
        return pending

    def _precompile_target_verify(self) -> list[PendingCall]:
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
        pending = []
        for bs in compile_range:
            num_tokens = bs * draft_token_num
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            warmup = self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=self.make_verify_spec_info(),
            )
            if warmup is not None:
                pending.append(warmup)
            if get_parallel().tp_rank == 0:
                status, elapsed, hbm_str = _warmup_progress(warmup, start, hbm_before)
                compile_range.set_description(
                    f"{status} TARGET_VERIFY (bs={bs}, tokens={num_tokens})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )
        return pending

    def _precompile_extend_full(self) -> list[PendingCall]:
        """Compile target prefill with and without speculative input state."""
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        bs = self._extend_compile_cap()
        variants = self._extend_input_variants()
        valid_shapes = [
            (t, bs, has_spec_info, variant)
            for t in self.t_bucketer.buckets
            if t >= bs
            for has_spec_info in (False, True)
            for variant in variants
        ]
        logger.info(
            "[spec compile policy] EXTEND(FULL): %d shapes (bs=%d, token_buckets=%s, "
            "variants=%s)",
            len(valid_shapes),
            bs,
            list(self.t_bucketer.buckets),
            list(variants),
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
        pending = []
        for num_tokens, bs, has_spec_info, variant in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            warmup = self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.EXTEND,
                variant=variant,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=self.make_null_spec_info() if has_spec_info else None,
            )
            if warmup is not None:
                pending.append(warmup)
            if get_parallel().tp_rank == 0:
                status, elapsed, hbm_str = _warmup_progress(warmup, start, hbm_before)
                compile_range.set_description(
                    f"{status} EXTEND(FULL) (tokens={num_tokens}, bs={bs}, "
                    f"variant={variant}, "
                    f"spec_info={'set' if has_spec_info else 'none'})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )
        return pending

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

    def make_null_spec_info(self) -> SpecInput:
        """Create null speculative input metadata for extend compilation.

        Raises:
          NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def make_verify_spec_info(self) -> SpecInput:
        """Create dummy speculative verify input metadata for verify compilation.

        Raises:
          NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def precompile_draft(
        self,
        draft_attn_backend: Any = None,
        draft_extend_attn_backend: Any = None,
        fused_draft_forward: Callable[[ForwardBatch], Any] | None = None,
    ) -> None:
        """Compile draft-model forward families at server warmup (subclass hook)."""

    def _dummy_kv_slot(self) -> int:
        return 1

    def _dummy_tokens_per_req(
        self, mode: ForwardMode, spec_info: SpecInput | None
    ) -> int:
        if mode.is_target_verify() and spec_info is not None:
            return spec_info.draft_token_num
        if mode.is_draft_extend_v2() and self.draft_token_num:
            return self.draft_token_num
        return 1

    def _dummy_mm_inputs(self, bs: int) -> list[None]:
        return [None] * bs

    def _dummy_out_cache_loc_dtype(self, mode: ForwardMode) -> torch.dtype:
        if (
            mode.is_target_verify()
            or mode.is_draft_extend_v2()
            or (self.is_draft_worker and mode.is_decode())
        ):
            return torch.int64
        return super()._dummy_out_cache_loc_dtype(mode)

    def _seed_dummy_req_to_token(self, forward_batch: ForwardBatch) -> None:
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
        """Slice padded spec outputs back to real request shapes."""
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        real_b = getattr(forward_batch, _REAL_B_KEY, None)
        if real_q is None or real_b is None:
            return outputs

        next_token_logits = getattr(outputs, "next_token_logits", None)
        next_draft_tokens = getattr(outputs, "next_draft_tokens", None)
        hidden_states = getattr(outputs, "hidden_states", None)
        if (
            next_token_logits is None
            and next_draft_tokens is None
            and hidden_states is None
        ):
            return super().finalize_forward_batch(outputs, forward_batch)

        fm = getattr(forward_batch, "forward_mode", None)
        sampling_info = forward_batch.sampling_info
        host_padded = sampling_info is not None and is_host_padded(sampling_info)
        is_verify = fm is not None and fm.is_target_verify()
        if next_token_logits is not None and real_b < next_token_logits.shape[0]:
            if is_verify:
                logit_rows = real_q
            elif host_padded:
                # HOST_PADDED sampling rows are padded to the batch bucket; sample at that shape.
                logit_rows = sampling_info.temperatures.shape[0]
                assert logit_rows <= next_token_logits.shape[0], (
                    logit_rows,
                    next_token_logits.shape,
                )
            else:
                logit_rows = real_b
            if logit_rows < next_token_logits.shape[0]:
                outputs.next_token_logits = next_token_logits[:logit_rows]

        if next_draft_tokens is not None and real_b < next_draft_tokens.shape[0]:
            outputs.next_draft_tokens = next_draft_tokens[:real_b]

        cap = getattr(forward_batch, "capture_hidden_mode", None)
        n_real = real_b if (cap is not None and cap.is_last()) else real_q
        # HOST_PADDED extend hands its hidden to the draft KV writer at the token
        # bucket; a [:real_q] trim would re-pad per prompt length.
        keep_hidden_padded = host_padded and not is_verify

        if (
            hidden_states is not None
            and n_real < hidden_states.shape[0]
            and not keep_hidden_padded
        ):
            outputs.hidden_states = hidden_states[:n_real]

        _restore_forward_batch_inputs(forward_batch, real_q, real_b)
        spec_info = forward_batch.spec_info
        if spec_info is not None:
            spec_info.keep_outputs_padded = False
        return outputs

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
        return tuple(t[:real_bs] if t is not None else None for t in outputs)
