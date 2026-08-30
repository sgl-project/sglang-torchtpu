"""DFlash speculative-decoding TPU compile policy."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    BaseSpecDecodeCompilePolicyMixin,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

logger = get_logger(__name__)


class DFlashSpecDecodeCompilePolicyMixin(BaseSpecDecodeCompilePolicyMixin):
    """DFlash speculative decoding compile policy mixin."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not self.draft_token_num:
            self.draft_token_num = self._infer_dflash_block_size()
        self.dflash_hidden_size = self._infer_dflash_hidden_size()

    def _infer_dflash_block_size(self) -> int:
        """Infer DFlash block size from server_args or draft config."""
        server_args = self.model_runner.server_args
        if server_args.speculative_num_draft_tokens is not None:
            return int(server_args.speculative_num_draft_tokens)
        if server_args.speculative_draft_model_path:
            try:
                draft_model_config = ModelConfig.from_server_args(
                    server_args,
                    model_path=server_args.speculative_draft_model_path,
                    model_revision=server_args.speculative_draft_model_revision,
                    is_draft_model=True,
                )
                dflash_cfg = parse_dflash_draft_config(
                    draft_hf_config=draft_model_config.hf_config
                )
                block_size = dflash_cfg.resolve_block_size(default=16)
                if block_size is not None:
                    return int(block_size)
            except (ValueError, KeyError, AttributeError, RuntimeError) as e:
                logger.warning(
                    "Failed to infer DFlash block_size in compile policy: %s; defaulting to 16.",
                    e,
                )
        return 16

    def _infer_dflash_hidden_size(self) -> int:
        """Derive hidden size for draft worker input embeddings."""
        if not self.is_draft_worker:
            return 0
        model_config = getattr(self.model_runner, "model_config", None)
        return getattr(model_config, "hidden_size", 0) or 0

    def make_null_spec_info(self) -> DFlashVerifyInput:
        """Create null DFlash verify input metadata for compilation.

        Returns:
          DFlashVerifyInput with CaptureHiddenMode.NULL.
        """
        return DFlashVerifyInput(
            draft_token=torch.empty(
                0, device=self.model_runner.device, dtype=torch.int64
            ),
            positions=torch.empty(
                0, device=self.model_runner.device, dtype=torch.int64
            ),
            draft_token_num=self.draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def make_verify_spec_info(self) -> DFlashVerifyInput:
        """Create dummy DFlash verify input metadata for TARGET_VERIFY compilation.

        Returns:
          DFlashVerifyInput with CaptureHiddenMode.FULL.
        """
        return DFlashVerifyInput(
            draft_token=torch.empty(
                0, device=self.model_runner.device, dtype=torch.int64
            ),
            positions=torch.empty(
                0, device=self.model_runner.device, dtype=torch.int64
            ),
            draft_token_num=self.draft_token_num,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: Any | None = None,
    ) -> ForwardBatch:
        batch = super()._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        if self.is_draft_worker:
            device = self.model_runner.device
            hidden_size = self.dflash_hidden_size or 1
            batch.input_embeds = torch.zeros(
                (num_tokens, hidden_size),
                dtype=self.model_runner.dtype,
                device=device,
            )
            tokens_per_req = self._dummy_tokens_per_req(mode, spec_info)
            batch.extend_num_tokens = bs * tokens_per_req
            batch.extend_seq_lens = torch.full(
                (bs,), tokens_per_req, dtype=torch.int32, device=device
            )
            batch.extend_prefix_lens = torch.zeros(
                (bs,), dtype=torch.int32, device=device
            )
            batch.extend_seq_lens_cpu = [tokens_per_req] * bs
            batch.extend_prefix_lens_cpu = [0] * bs
        return batch

    def precompile_draft(
        self,
        draft_attn_backend: Any = None,
        draft_extend_attn_backend: Any = None,
        fused_draft_forward: Callable[[ForwardBatch], Any] | None = None,
    ) -> None:
        """Compile draft-model forward passes during server warmup."""
        del draft_extend_attn_backend, fused_draft_forward
        if not self.is_draft_worker or self.b_bucketer is None:
            return
        overall = time.perf_counter()
        self._precompile_draft_verify(draft_attn_backend)
        logger.info(
            "[spec compile policy] DFLASH DRAFT precompile done in %.1fs.",
            time.perf_counter() - overall,
        )
        shim = self.compiled_forward_shim
        checker = shim._recompilation_checker if shim is not None else None
        if checker is not None:
            checker.snapshot_baseline()

    def _precompile_draft_verify(self, draft_attn_backend: Any = None) -> None:
        """Compile DFLASH draft model forward across batch size buckets."""
        assert self.b_bucketer is not None
        bs_buckets = list(self.b_bucketer.buckets)
        draft_token_num = self.draft_token_num
        shapes = [(bs, bs * draft_token_num) for bs in bs_buckets]
        logger.info(
            "[spec compile policy] DFLASH DRAFT VERIFY: %d shapes (bs=%s, draft_token_num=%d)",
            len(shapes),
            bs_buckets,
            draft_token_num,
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

        self._precompile_shapes("DFLASH DRAFT VERIFY", shapes, run)


class DFlashSpecDecoderTpuCompilePolicy(
    DFlashSpecDecodeCompilePolicyMixin, DecoderTpuCompilePolicy
):
    """DFlash speculative-decoding policy over base decoder policy."""
