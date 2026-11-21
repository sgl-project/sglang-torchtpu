"""TPU compile policy for TpuBlockSpecWorker's target and draft runners."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    LogitsOutputMode,
)
from sglang.srt.runtime_context import get_disagg
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.recompilation_checker import (
    install_recompilation_checker,
)
from sglang_tpu.compilation.warmup_queue import current_warmup_queue
from sglang_tpu.dflash_utils_tpu import register_argmax_draft_top_k
from sglang_tpu.experimental import runs_draft_prompt_pass
from sglang_tpu.experimental.compilation.spec_compile_policy import (
    TpuSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.speculative.spec_types import uniform_ragged_layout
from sglang_tpu.experimental.speculative.verify_width import (
    verify_width_candidates,
    verify_width_table,
)

logger = get_logger(__name__)


class TpuBlockSpecDecodeCompilePolicyMixin(TpuSpecDecodeCompilePolicyMixin):
    """The draft warms its block forward and KV writer.

    DFlash and DSpark drafts share the spec inputs and K/V layout, so the target
    verify grid and the draft KV writer carry over. The target scores
    [r, d_1..d_gamma] (draft_token_num = gamma + 1 rows); the draft forwards
    [r, m, ..., m] (the drafter's block_width rows) and never runs EXTEND.
    """

    def target_verify_widths(self) -> tuple[int, ...]:
        # Shared with TpuBlockSpecWorker, which validates the table. compact packs
        # bucket x width tokens per verify; cap-accept verifies at full width.
        assert self.b_bucketer is not None
        if read_ragged_verify_mode() is RaggedVerifyMode.CAP_ACCEPT:
            return (self.draft_token_num,)
        candidates = verify_width_candidates(self.draft_token_num)
        if candidates is not None:
            return candidates
        table = verify_width_table(self.draft_token_num)
        if table is None:
            return (self.draft_token_num,)
        return table.reachable_widths(self.b_bucketer.buckets)

    def draft_commit_widths(self) -> tuple[int, ...]:
        # compact unpacks the verify outputs back to the full window before commit.
        if read_ragged_verify_mode() is RaggedVerifyMode.COMPACT:
            return (self.draft_token_num,)
        return self.target_verify_widths()

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: DFlashVerifyInput | None = None,
    ) -> ForwardBatch:
        batch = super()._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        if (
            not self.is_draft_worker
            and mode.is_target_verify()
            and read_ragged_verify_mode() is RaggedVerifyMode.COMPACT
            # A full-width table entry verifies on the plain decode-tile path.
            and num_tokens < bs * self.draft_token_num
        ):
            # Narrower buckets verify through the ragged layout.
            assert isinstance(batch.spec_info, DFlashVerifyInput)
            batch.spec_info.ragged_verify_layout = uniform_ragged_layout(
                seq_lens=batch.seq_lens, num_tokens=num_tokens
            )
        return batch

    def profile_target_verify(
        self, shapes: list[tuple[int, int]], *, repeats: int = 3
    ) -> dict[tuple[int, int], float]:
        """Seconds of the warmed TARGET_VERIFY forward per (bucket, width), fastest of repeats.

        Runs the programs warmup compiled (compact buckets narrower than the window
        through the ragged layout), never a new shape. The dummy batch's short
        contexts understate attention, so callers treat the result as a prior.
        """
        logits_output_mode = self.target_verify_logits_output_modes()[-1]
        profile = {}
        for bs, width in shapes:
            times = []
            for _ in range(repeats + 1):
                started = time.perf_counter()
                out = self._run_one(
                    num_tokens=bs * width,
                    bs=bs,
                    mode=ForwardMode.TARGET_VERIFY,
                    logits_output_mode=logits_output_mode,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    spec_info=self.make_verify_spec_info(width=width),
                )
                if isinstance(out, PendingCall):
                    out.wait()
                times.append(time.perf_counter() - started)
            # The first run pays one-off host work (dummy inputs, caches).
            profile[(bs, width)] = min(times[1:])
        return profile

    def target_verify_logits_output_modes(self) -> tuple[LogitsOutputMode, ...]:
        # TpuBlockSpecWorker.verify_top_k_width reads the same negotiated width.
        if self.is_draft_worker or self._distributed_top_k_width() is None:
            return (LogitsOutputMode.FULL_VOCAB,)
        return (LogitsOutputMode.FULL_VOCAB, LogitsOutputMode.DISTRIBUTED_TOP_K)

    def precompile_draft(
        self,
        draft_attn_backend: AttentionBackend,
        *,
        kv_writer: Callable[..., torch.Tensor],
        block_width: int,
    ) -> None:
        """Compile the draft block forward and the draft KV writer."""
        if not self.is_draft_worker:
            raise ValueError("precompile_draft runs on the draft runner only.")
        if self.b_bucketer is None:
            raise ValueError("b_bucketer must be initialized before precompile_draft.")
        start = time.perf_counter()
        if self._compiled_kv_appender is None:
            self._compiled_kv_appender = self._compile_kv_appender(kv_writer)
        disagg_mode = get_disagg().disaggregation_mode
        pending: list[PendingCall] = []
        if disagg_mode != "prefill":
            pending = self._precompile_draft_block(
                draft_attn_backend, block_width=block_width
            )
            self._precompile_draft_append_kv_verify()
        if runs_draft_prompt_pass(disagg_mode):
            self._precompile_draft_append_kv_extend()
        queue = current_warmup_queue()
        if pending and queue is not None:
            # Async block warmups compile with the target's (shared under SPMD);
            # the checker baselines only once they have run.
            queue.calls.extend(pending)
            queue.completions.append(self._install_draft_recompilation_checker)
            logger.info(
                "[spec compile policy] %s DRAFT queued %d warmups in %.1fs.",
                self.algo_label,
                len(pending),
                time.perf_counter() - start,
            )
            return
        for warmup in pending:
            warmup.wait()
        logger.info(
            "[spec compile policy] %s DRAFT precompile done in %.1fs.",
            self.algo_label,
            time.perf_counter() - start,
        )
        self._install_draft_recompilation_checker()

    def _install_draft_recompilation_checker(self) -> None:
        install_recompilation_checker(
            policies=(self,),
            model_label=f"draft {type(self.model_runner.model).__name__}",
        )

    def _precompile_draft_block(
        self, draft_attn_backend: AttentionBackend, *, block_width: int
    ) -> list[PendingCall]:
        assert self.b_bucketer is not None
        width = block_width
        lm_head = self.model_runner.model.lm_head
        if lm_head is not None:
            register_argmax_draft_top_k(lm_head)
        shapes = [(bs, bs * width) for bs in self.b_bucketer.buckets]
        logger.info(
            "[spec compile policy] %s DRAFT BLOCK: %d shapes (bs=%s, width=%d)",
            self.algo_label,
            len(shapes),
            list(self.b_bucketer.buckets),
            width,
        )

        pending: list[PendingCall] = []

        def run(shape: tuple[int, int]) -> str:
            bs, num_tokens = shape
            warmup = self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                # Same structure as TpuBlockSpecWorker's draft block input, so the trace matches.
                spec_info=make_draft_block_spec_info(
                    draft_token_num=width, device=self.model_runner.device
                ),
                attn_backend=draft_attn_backend,
            )
            if isinstance(warmup, PendingCall):
                pending.append(warmup)
            return f"bs={bs}, tokens={num_tokens}"

        self._precompile_shapes(f"{self.algo_label} DRAFT BLOCK", shapes, run)
        return pending
