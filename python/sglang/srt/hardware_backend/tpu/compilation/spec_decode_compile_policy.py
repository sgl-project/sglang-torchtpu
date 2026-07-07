"""Speculative-decoding TPU compile policy."""

from __future__ import annotations

import time
from typing import Any

import torch
import tqdm

from sglang.srt.distributed import get_tensor_model_parallel_rank
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
)
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    _COMPILE_BAR_FORMAT,
    DecoderTpuCompilePolicy,
    _get_hbm_usage_mb,
    _hbm_delta_str,
    _materialize_out,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

logger = get_logger(__name__)


def _pad_first_dim(t: torch.Tensor | None, target_rows: int) -> torch.Tensor | None:
    """Zero-pad a tensor's first (row) dim to target_rows (any ndim).

    None (an unset field) passes through unchanged.
    """
    if t is None or t.shape[0] >= target_rows:
        return t
    pad = t.new_zeros((target_rows - t.shape[0], *t.shape[1:]))
    return torch.cat([t, pad], dim=0)


class SpecDecodeCompilePolicyMixin:
    """Adds EAGLE3 spec-decode precompile + draft-decode pad/unpad to a base decoder policy."""

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
        """Warm target graphs for all spec forward modes.

        Only the **target** model is precompiled; draft shapes are compiled by
        the startup server warmup that drives the full speculative loop.
        """
        if self.is_draft_worker:
            logger.info(
                "[spec compile policy] DRAFT: skipping precompilation "
                "(startup server warmup compiles draft shapes before readiness)."
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
            if get_tensor_model_parallel_rank() == 0
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
            if get_tensor_model_parallel_rank() == 0:
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
        bs = self.b_bucketer.max_val
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
            if get_tensor_model_parallel_rank() == 0
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
            )
            if get_tensor_model_parallel_rank() == 0:
                elapsed = time.perf_counter() - start
                hbm_after = _get_hbm_usage_mb()
                hbm_str = _hbm_delta_str(hbm_before, hbm_after)
                compile_range.set_description(
                    f"Compiled EXTEND(FULL) (tokens={num_tokens}, bs={bs})"
                    f" in {elapsed:.1f}s{hbm_str}"
                )

    def _make_null_spec_info(self) -> Any:
        """Build a minimal idle spec_info stub for non-spec forward modes.

        Carries the attributes Dynamo probes without affecting the forward
        result, so both branches of ``if spec_info is not None`` are traced.
        """
        return EagleDraftInput.create_idle_input(
            device=self.model_runner.device,
            hidden_size=self.eagle_hidden_size or 1,
            dtype=torch.bfloat16,
            topk=self.speculative_eagle_topk,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def _make_verify_spec_info(self) -> Any:
        """Build an ``EagleVerifyInput`` for TARGET_VERIFY precompilation."""
        return EagleVerifyInput.create_idle_input(
            topk=self.speculative_eagle_topk,
            spec_steps=self.speculative_num_steps,
            num_verify_tokens=self.draft_token_num,
            device=self.model_runner.device,
        )

    def _run_one(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
        variant: str = "plain",
    ) -> None:
        """Build and execute one dummy spec forward pass for compilation."""
        if spec_info is None:
            spec_info = self._make_null_spec_info()
        forward_batch = self._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        # Seed req_to_token row 1 with valid page indices for this model's KV
        # pool. The dummy batch uses req_pool_indices=1 and out_cache_loc=1 to
        # avoid conflicts with row/slot 0 (which may hold stale state from
        # another model's precompilation, e.g. a smaller draft KV pool).
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        req_to_token[1, :num_tokens] = 1

        model = self.model_runner.model
        model.prepare_forward_batch(forward_batch)
        self.model_runner.attn_backend.init_forward_metadata(forward_batch)

        with torch.no_grad():
            out = self.owner.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
            )
            _materialize_out(out, forward_batch)

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
    ) -> ForwardBatch:
        """Construct a dummy spec ForwardBatch at the target bucket shape."""
        device = self.model_runner.device
        # TARGET_VERIFY is extend-like (is_extend() returns True) so it needs
        # the extend metadata fields populated.
        is_extend = mode.is_extend()

        # Token-level inputs
        input_ids = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        input_ids[:bs] = 1
        positions = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        # Use slot 1 (not 0) for out_cache_loc. Slot 0 is the permanent padding
        # slot shared between target and draft models; writing target page
        # indices there could be out-of-bounds for the draft's smaller KV pool.
        out_cache_loc = torch.ones(num_tokens, dtype=torch.int64, device=device)

        # Batch-level inputs -- req_pool_indices=1 (not 0) for the same reason.
        seq_lens = torch.ones(bs, dtype=torch.int64, device=device)
        seq_lens_cpu = torch.ones(bs, dtype=torch.int64)
        req_pool_indices = torch.ones(bs, dtype=torch.int64, device=device)

        if is_extend:
            if mode.is_target_verify() and spec_info is not None:
                tokens_per_req = spec_info.draft_token_num
            elif mode.is_draft_extend() and self.draft_token_num:
                tokens_per_req = self.draft_token_num
            else:
                tokens_per_req = 1
            extend_seq_lens = torch.full(
                (bs,), tokens_per_req, dtype=torch.int32, device=device
            )
            extend_prefix_lens = torch.zeros(bs, dtype=torch.int32, device=device)
            extend_start_loc = torch.arange(
                0, bs * tokens_per_req, tokens_per_req, dtype=torch.int32, device=device
            )
            extend_num_tokens = bs * tokens_per_req
            extend_seq_lens_cpu = [tokens_per_req] * bs
            extend_prefix_lens_cpu = [0] * bs
            extend_logprob_start_lens_cpu = [0] * bs
        else:
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None
            extend_num_tokens = None
            extend_seq_lens_cpu = None
            extend_prefix_lens_cpu = None
            extend_logprob_start_lens_cpu = None

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
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool=self.model_runner.token_to_kv_pool,
            attn_backend=self.model_runner.attn_backend,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
            # Text-only requests produce mm_inputs=[None]*bs at runtime, NOT
            # None. Dynamo guards on ``mm_inputs is None`` so match it.
            mm_inputs=[None] * bs,
        )

    def unpad_outputs(self, out: Any, forward_batch: Any) -> Any:
        """Slice padded spec outputs back to real request shapes.

        Differs from the base decoder unpad: TARGET_VERIFY emits one logit row
        per draft token (``real_q``), and CaptureHiddenMode.LAST (draft extend)
        emits one hidden row per sequence (``real_b``).
        """
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        real_b = getattr(forward_batch, _REAL_B_KEY, None)
        if real_q is None or real_b is None:
            return out

        next_token_logits = getattr(out, "next_token_logits", None)
        if next_token_logits is not None and real_b < next_token_logits.shape[0]:
            # TARGET_VERIFY outputs bs * draft_token_num logits (one per draft
            # token across all requests); slicing to real_b would drop verify
            # tokens, so use real_q instead.
            fm = getattr(forward_batch, "forward_mode", None)
            logit_rows = (
                real_q if (fm is not None and fm.is_target_verify()) else real_b
            )
            if logit_rows < next_token_logits.shape[0]:
                out.next_token_logits = next_token_logits[:logit_rows]

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is not None:
            # Capture-LAST (draft extend) emits one hidden row per
            # sequence (batch-dim), so slice to real_b. Every other capture mode
            # emits one row per token (real_q).
            cap = getattr(forward_batch, "capture_hidden_mode", None)
            n_real = real_b if (cap is not None and cap.is_last()) else real_q
            if n_real < hidden_states.shape[0]:
                out.hidden_states = hidden_states[:n_real]

        # Restore real batch_size so downstream consumers (sampler, logprob,
        # scheduler hooks) don't iterate over ghost slots.
        forward_batch.batch_size = real_b
        return out

    def pad_draft_decode(self, forward_batch: Any) -> None:
        """Pad the speculative draft-decode batch to a torch.compile bs bucket.

        Stashes the real (pre-pad) batch size on ``forward_batch`` (``_REAL_B_KEY``),
        which ``unpad_draft_outputs`` reads to trim the draft outputs. No-op when
        this policy has no batch bucketer (torch.compile off) or the batch is
        already at a bucket.
        """
        b_bucketer = self.b_bucketer
        if b_bucketer is None:
            return
        real_bs = int(forward_batch.seq_lens.shape[0])
        pad_b = b_bucketer.next_bucket(real_bs)
        if pad_b == real_bs:
            return

        forward_batch.seq_lens = _pad_first_dim(forward_batch.seq_lens, pad_b)
        forward_batch.req_pool_indices = _pad_first_dim(
            forward_batch.req_pool_indices, pad_b
        )
        forward_batch.positions = _pad_first_dim(
            forward_batch.positions,
            forward_batch.positions.shape[0] // real_bs * pad_b,
        )
        forward_batch.out_cache_loc = _pad_first_dim(
            forward_batch.out_cache_loc,
            forward_batch.out_cache_loc.shape[0] // real_bs * pad_b,
        )

        # EagleDraftInput seeds consumed by select_top_k_tokens.
        spec_info = forward_batch.spec_info
        if spec_info is not None:
            spec_info.topk_p = _pad_first_dim(spec_info.topk_p, pad_b)
            spec_info.topk_index = _pad_first_dim(spec_info.topk_index, pad_b)
            spec_info.hidden_states = _pad_first_dim(spec_info.hidden_states, pad_b)

        forward_batch.batch_size = pad_b
        setattr(forward_batch, _REAL_B_KEY, real_bs)

    def unpad_draft_outputs(self, outputs: tuple, forward_batch: Any) -> tuple:
        """Trim draft-decode outputs back to the real (pre-pad) batch size.

        ``pad_draft_decode`` stashes the real batch size on ``forward_batch``
        (``_REAL_B_KEY``); trim back to it here. No-op when the batch was not
        padded (torch.compile off, or already at a bucket).
        """
        real_bs = getattr(forward_batch, _REAL_B_KEY, None)
        if real_bs is None:
            return outputs
        return tuple(t[:real_bs] for t in outputs)


class SpecDecoderTpuCompilePolicy(
    SpecDecodeCompilePolicyMixin, DecoderTpuCompilePolicy
):
    """Speculative-decoding policy over the base decoder policy.

    The common composition (unannotated spec target models). The runner composes
    ``SpecDecodeCompilePolicyMixin`` over other annotated base policies
    dynamically via ``TpuGraphRunner._augment_spec_policy``.
    """
