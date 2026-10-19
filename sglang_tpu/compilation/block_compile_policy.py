"""TPU compile policy for TpuBlockSpecWorker's target and draft runners."""

from __future__ import annotations

import time
from collections.abc import Callable

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
)
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_disagg
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.recompilation_checker import (
    install_recompilation_checker,
)
from sglang_tpu.compilation.spec_compile_policy import (
    TpuSpecDecodeCompilePolicyMixin,
)

logger = get_logger(__name__)


def block_draft_width(model: DFlashDraftModel, num_verify_rows: int) -> int:
    """Rows per request in the draft block forward [r, m, ..., m]."""
    # Anchor DSpark drafts read all gamma rows, so the block drops the last mask row.
    if isinstance(model, DSparkDraftModel) and model.sample_from_anchor:
        return num_verify_rows - 1
    return num_verify_rows


class TpuBlockSpecDecodeCompilePolicyMixin(TpuSpecDecodeCompilePolicyMixin):
    """The draft warms its block forward and KV writer.

    DFlash and DSpark drafts share the spec inputs and K/V layout, so the target
    verify grid and the draft KV writer carry over. The target scores
    [r, d_1..d_gamma] (draft_token_num = gamma + 1 rows); the draft forwards
    [r, m, ..., m] (draft_block_width rows) and never runs EXTEND.
    """

    @property
    def draft_block_width(self) -> int:
        model = self.model_runner.model
        if not isinstance(model, DFlashDraftModel):
            raise TypeError(f"Expected a DFlashDraftModel draft, got {type(model)}.")
        return block_draft_width(model, self.draft_token_num)

    def make_null_spec_info(self) -> DFlashVerifyInput:
        # Same structure as TpuBlockSpecWorker's draft block input, so the draft trace matches.
        width = self.draft_block_width if self.is_draft_worker else self.draft_token_num
        return make_draft_block_spec_info(
            draft_token_num=width, device=self.model_runner.device
        )

    def precompile_draft(
        self,
        draft_attn_backend: AttentionBackend,
        *,
        kv_writer: Callable[..., None],
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
        if disagg_mode != "prefill":
            self._precompile_draft_block(draft_attn_backend)
            self._precompile_draft_append_kv_verify()
        if disagg_mode != "decode":
            self._precompile_draft_append_kv_extend()
        logger.info(
            "[spec compile policy] %s DRAFT precompile done in %.1fs.",
            self.algo_label,
            time.perf_counter() - start,
        )
        install_recompilation_checker(
            policies=(self,),
            model_label=f"draft {type(self.model_runner.model).__name__}",
        )

    def _precompile_draft_block(self, draft_attn_backend: AttentionBackend) -> None:
        assert self.b_bucketer is not None
        width = self.draft_block_width
        shapes = [(bs, bs * width) for bs in self.b_bucketer.buckets]
        logger.info(
            "[spec compile policy] %s DRAFT BLOCK: %d shapes (bs=%s, width=%d)",
            self.algo_label,
            len(shapes),
            list(self.b_bucketer.buckets),
            width,
        )

        def run(shape: tuple[int, int]) -> str:
            bs, num_tokens = shape
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                spec_info=self.make_null_spec_info(),
                attn_backend=draft_attn_backend,
            )
            return f"bs={bs}, tokens={num_tokens}"

        self._precompile_shapes(f"{self.algo_label} DRAFT BLOCK", shapes, run)
