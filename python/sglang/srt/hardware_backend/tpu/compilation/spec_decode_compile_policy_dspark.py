"""DSpark speculative-decoding TPU compile policy."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_disagg
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dspark_components.dspark_draft import draft_owns_embed
from sglang.srt.speculative.dspark_components.dspark_worker_v2 import DSparkWorkerV2

logger = get_logger(__name__)


class DsparkSpecDecodeCompilePolicyMixin(DFlashSpecDecodeCompilePolicyMixin):
    """DSpark speculative decoding compile policy mixin.

    DSpark shares the DFlash draft architecture and spec-input types, so the
    target verify/extend enumeration and the draft KV append shapes carry over.
    Only two things differ: the draft forward is gamma wide against a gamma + 1
    verify window, and the KV write goes through TargetHiddenKvInjector rather
    than a method on the worker.
    """

    algo_label = "DSPARK"

    @property
    def draft_query_token_num(self) -> int:
        """Row count per request in the draft forward (the target verify window
        is self.draft_token_num = gamma + 1)."""
        gamma = self.draft_token_num - 1
        if gamma < 1:
            raise ValueError(
                f"DSpark draft_token_num must be >= 2 (= gamma + 1), got "
                f"{self.draft_token_num}."
            )
        return gamma if self.model_runner.model.sample_from_anchor else gamma + 1

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
        # DFlash always feeds the draft input_embeds; DSpark's draft embeds
        # input_ids in-graph, and the two trace different graphs.
        if self.is_draft_worker and draft_owns_embed(self.model_runner.model):
            batch.input_embeds = None
        return batch

    def make_null_spec_info(self) -> DFlashVerifyInput:
        if not self.is_draft_worker:
            return super().make_null_spec_info()
        device = self.model_runner.device
        return DFlashVerifyInput(
            draft_token=torch.empty(0, device=device, dtype=torch.int64),
            positions=torch.empty(0, device=device, dtype=torch.int64),
            draft_token_num=self.draft_query_token_num,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def _ensure_kv_appender(self, dspark_worker: DSparkWorkerV2 | None = None) -> None:
        if dspark_worker is None:
            raise ValueError("DSpark worker cannot be None")
        injector = dspark_worker.kv_injector
        if self._compiled_kv_appender is None:
            self._compiled_kv_appender = self._compile_kv_appender(
                injector.inject_target_hidden
            )
        injector.inject_target_hidden = self.append_target_hidden_to_draft_kv_by_loc_tpu

    def append_target_hidden_to_draft_kv_by_loc_tpu(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: torch.Tensor | None = None,
        commit_lens: torch.Tensor | None = None,
        state_slot: torch.Tensor | None = None,
        final_pos: torch.Tensor | None = None,
    ) -> None:
        if state_slot is not None or final_pos is not None:
            # Only unified-kv MLA injection threads these; that path has no TPU
            # kernel today and the arg hook rejects it before we get here.
            raise NotImplementedError(
                "DSpark unified-kv draft KV injection is not supported on TPU."
            )
        super().append_target_hidden_to_draft_kv_by_loc_tpu(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )

    def precompile_draft(
        self,
        draft_attn_backend: AttentionBackend | None = None,
        draft_extend_attn_backend: AttentionBackend | None = None,
        fused_draft_forward: Callable[[ForwardBatch], object] | None = None,
        dspark_worker: DSparkWorkerV2 | None = None,
    ) -> None:
        """Compile draft-model forward passes during server warmup."""
        del draft_extend_attn_backend, fused_draft_forward
        if not self.is_draft_worker or self.b_bucketer is None:
            return
        if dspark_worker is None:
            raise ValueError("DSpark worker cannot be None")
        overall = time.perf_counter()
        disagg_mode = get_disagg().disaggregation_mode
        if disagg_mode != "prefill":
            self._precompile_draft_verify(draft_attn_backend)
        self._ensure_kv_appender(dspark_worker)
        if disagg_mode != "prefill":
            self._precompile_draft_append_kv_verify()
        if disagg_mode != "decode":
            self._precompile_draft_append_kv_extend()
        logger.info(
            "[spec compile policy] DSPARK DRAFT precompile done in %.1fs.",
            time.perf_counter() - overall,
        )
        # Snapshot only once every warmup graph exists, else they are reported
        # as serving-time recompiles on the first draft forward.
        self.snapshot_recompilation_baseline()

    def _precompile_draft_verify(
        self, draft_attn_backend: AttentionBackend | None = None
    ) -> None:
        """Compile the DSpark draft block forward across batch size buckets."""
        if self.b_bucketer is None:
            raise ValueError(
                "b_bucketer must be initialized before precompiling draft verify."
            )
        bs_buckets = list(self.b_bucketer.buckets)
        query_token_num = self.draft_query_token_num
        shapes = [(bs, bs * query_token_num) for bs in bs_buckets]
        logger.info(
            "[spec compile policy] DSPARK DRAFT VERIFY: %d shapes (bs=%s, query_token_num=%d)",
            len(shapes),
            bs_buckets,
            query_token_num,
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

        self._precompile_shapes("DSPARK DRAFT VERIFY", shapes, run)


class DsparkSpecDecoderTpuCompilePolicy(
    DsparkSpecDecodeCompilePolicyMixin, DecoderTpuCompilePolicy
):
    """DSpark speculative-decoding policy over base decoder policy."""
