"""DFlash speculative-decoding TPU compile policy."""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    BaseSpecDecodeCompilePolicyMixin,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.platforms import current_platform
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

logger = get_logger(__name__)


def _kv_appender_specialization_key(
    *,
    target_hidden: torch.Tensor,
    cache_loc: torch.Tensor,
    positions: torch.Tensor,
    cache_loc_2d: torch.Tensor | None,
    commit_lens: torch.Tensor | None,
) -> tuple[object, ...]:
    """Key direct-compile artifacts on input shapes and the verify/extend split."""
    return tuple(
        None if t is None else (tuple(t.shape), t.dtype)
        for t in (target_hidden, cache_loc, positions, cache_loc_2d, commit_lens)
    )


class DFlashSpecDecodeCompilePolicyMixin(BaseSpecDecodeCompilePolicyMixin):
    """DFlash speculative decoding compile policy mixin."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if not self.draft_token_num:
            self.draft_token_num = self._infer_dflash_block_size()
        self.dflash_hidden_size = self._infer_dflash_hidden_size()
        self.dflash_target_hidden_size = self._infer_dflash_target_hidden_size()
        self._dflash_worker: DFlashWorkerV2 | None = None
        self._compiled_kv_appender: Callable[..., None] | None = None

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
        return int(self.model_runner.model_config.hidden_size)

    def _infer_dflash_target_hidden_size(self) -> int:
        """Infer target hidden dimension for DFlash project_target_hidden."""
        model = self.model_runner.model
        fc = getattr(model, "fc", None)
        if fc is not None:
            return int(fc.in_features)
        return self.dflash_hidden_size or 1

    def _ensure_kv_appender(self, dflash_worker: DFlashWorkerV2 | None = None) -> None:
        """Initialize, compile and install the draft KV appender on the worker."""
        if dflash_worker is None:
            raise ValueError("DFlash worker cannot be None")
        self._dflash_worker = dflash_worker
        if self._compiled_kv_appender is None:
            self._compiled_kv_appender = self._compile_kv_appender(
                dflash_worker.append_target_hidden_to_draft_kv_by_loc_tpu
            )
        dflash_worker.append_target_hidden_to_draft_kv_by_loc_tpu = (
            self.append_target_hidden_to_draft_kv_by_loc_tpu
        )

    def _compile_kv_appender(self, fn: Callable[..., None]) -> Callable[..., None]:
        """Compile the draft KV appender through this policy's active tracer.

        The appender reads draft attention weights and writes the draft KV pool,
        which are exactly the two tensor sources direct_compile lifts for us.
        """
        fn = torch.no_grad()(fn)
        if self.tracer == "direct":
            from sglang.srt.hardware_backend.tpu.compilation.direct_compile import (
                direct_compile,
            )

            return direct_compile(
                fn,
                specialization_key=_kv_appender_specialization_key,
                model=self.model_runner.model,
                model_extra_tensor_attrs=self.get_model_extra_tensor_attrs(),
                kv_pool=self.model_runner.token_to_kv_pool,
            )
        return torch.compile(
            fn,
            backend=self.backend or current_platform.get_compile_backend(),
            dynamic=self.dynamic,
        )

    def pad_append_target_hidden_tpu(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: torch.Tensor | None = None,
        commit_lens: torch.Tensor | None = None,
    ) -> (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
        | None
    ):
        """Pad inputs to static compile bucket sizes for TPU draft KV materialization.

        Args:
          target_hidden: Context features [total_tokens, in_features] from target model.
          cache_loc: 1D cache slot locations [total_tokens].
          positions: Token position IDs [total_tokens].
          cache_loc_2d: 2D cache slots [bs, block_size] during verify; None during prefill/extend.
          commit_lens: Accept token counts [bs] during verify; None during prefill/extend.

        Returns:
          Tuple of padded (target_hidden, cache_loc, positions, cache_loc_2d, commit_lens),
          or None if inputs are empty (no-op).

        Examples:
          Prefill / Extend (pad by token bucket, e.g. next bucket = 4096):
            Inputs:
              target_hidden=(3605, 14400), cache_loc=(3605,), positions=(3605,),
              cache_loc_2d=None, commit_lens=None
            Outputs:
              target_hidden=(4096, 14400), cache_loc=(4096,), positions=(4096,),
              cache_loc_2d=None, commit_lens=None

          Verify / Decode (pad by batch size bucket, e.g. bs=1 -> pad_bs=4, block_size=10):
            Inputs:
              target_hidden=(10, 14400), cache_loc=(10,), positions=(10,),
              cache_loc_2d=(1, 10), commit_lens=(1,)
            Outputs:
              target_hidden=(40, 14400), cache_loc=(40,), positions=(40,),
              cache_loc_2d=(4, 10), commit_lens=(4,)
        """
        if cache_loc_2d is not None:
            if commit_lens is None:
                raise ValueError("DFLASH cache_loc_2d requires commit_lens.")
            bs = int(commit_lens.shape[0])
            if int(cache_loc_2d.shape[0]) != bs:
                raise ValueError(
                    "DFLASH cache_loc_2d batch size mismatch: "
                    f"cache_loc_2d={tuple(cache_loc_2d.shape)}, commit_lens={tuple(commit_lens.shape)}."
                )
            if bs == 0:
                return None
            pad_bs = (
                self.b_bucketer.next_bucket(bs) if self.b_bucketer is not None else bs
            )
            block_size = int(cache_loc_2d.shape[1])
            if pad_bs > bs:
                pad_rows = pad_bs - bs
                cache_loc_2d = torch.nn.functional.pad(
                    cache_loc_2d, (0, 0, 0, pad_rows)
                )
                commit_lens = torch.nn.functional.pad(commit_lens, (0, pad_rows))
                target_hidden = torch.nn.functional.pad(
                    target_hidden, (0, 0, 0, pad_rows * block_size)
                )
                positions = torch.nn.functional.pad(
                    positions, (0, pad_rows * block_size)
                )
                cache_loc = torch.nn.functional.pad(
                    cache_loc, (0, pad_rows * block_size)
                )
            return target_hidden, cache_loc, positions, cache_loc_2d, commit_lens

        num_tokens = int(target_hidden.shape[0])
        if num_tokens == 0:
            return None
        pad_num_tokens = (
            self.t_bucketer.next_bucket(num_tokens)
            if self.t_bucketer is not None
            else num_tokens
        )
        if pad_num_tokens > num_tokens:
            pad_count = pad_num_tokens - num_tokens
            target_hidden = torch.nn.functional.pad(target_hidden, (0, 0, 0, pad_count))
            positions = torch.nn.functional.pad(positions, (0, pad_count))
            cache_loc = torch.nn.functional.pad(cache_loc, (0, pad_count))
        return target_hidden, cache_loc, positions, None, None

    def append_target_hidden_to_draft_kv_by_loc_tpu(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: torch.Tensor | None = None,
        commit_lens: torch.Tensor | None = None,
    ) -> None:
        """Pad inputs and execute compiled draft KV materialization on TPU."""
        padded = self.pad_append_target_hidden_tpu(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )
        if padded is None:
            return
        (
            target_hidden,
            cache_loc,
            positions,
            cache_loc_2d,
            commit_lens,
        ) = padded

        with torch.no_grad():
            if self._compiled_kv_appender is None:
                raise RuntimeError("DFLASH draft KV appender is not compiled.")
            self._compiled_kv_appender(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )

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
        spec_info: DFlashVerifyInput | None = None,
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
        draft_attn_backend: AttentionBackend | None = None,
        draft_extend_attn_backend: AttentionBackend | None = None,
        fused_draft_forward: Callable[[ForwardBatch], object] | None = None,
        dflash_worker: DFlashWorkerV2 | None = None,
    ) -> None:
        """Compile draft-model forward passes during server warmup."""
        del draft_extend_attn_backend, fused_draft_forward
        if not self.is_draft_worker or self.b_bucketer is None:
            return
        if dflash_worker is None:
            raise ValueError("DFlash worker cannot be None")
        overall = time.perf_counter()
        self._precompile_draft_verify(draft_attn_backend)
        self._precompile_append_target_hidden_to_draft_kv_by_loc_tpu(dflash_worker)
        logger.info(
            "[spec compile policy] DFLASH DRAFT precompile done in %.1fs.",
            time.perf_counter() - overall,
        )
        # Snapshot only once every warmup graph exists, else they are reported
        # as serving-time recompiles on the first draft forward.
        self.snapshot_recompilation_baseline()

    def _precompile_append_target_hidden_to_draft_kv_by_loc_tpu(
        self, dflash_worker: DFlashWorkerV2
    ) -> None:
        """Warm up DFlash draft KV append across token buckets and batch size buckets."""
        self._ensure_kv_appender(dflash_worker)
        self._precompile_draft_append_kv_verify()
        self._precompile_draft_append_kv_extend()

    def _precompile_draft_append_kv_verify(self) -> None:
        """Compile DFLASH draft KV append across batch size buckets (verify/decode)."""
        if self.b_bucketer is None:
            raise ValueError(
                "b_bucketer must be initialized before precompiling draft verify."
            )
        bs_buckets = list(self.b_bucketer.buckets)
        draft_token_num = self.draft_token_num
        target_hidden_size = self.dflash_target_hidden_size
        device = self.model_runner.device
        dtype = self.model_runner.dtype
        cache_loc_dtype = self._dummy_out_cache_loc_dtype(ForwardMode.TARGET_VERIFY)
        logger.info(
            "[spec compile policy] DFLASH DRAFT APPEND KV (VERIFY): %d shapes (bs=%s, draft_token_num=%d)",
            len(bs_buckets),
            bs_buckets,
            draft_token_num,
        )

        def run(bs: int) -> str:
            num_tokens = bs * draft_token_num
            target_hidden = torch.zeros(
                (num_tokens, target_hidden_size), dtype=dtype, device=device
            )
            cache_loc = torch.zeros((num_tokens,), dtype=cache_loc_dtype, device=device)
            positions = torch.zeros((num_tokens,), dtype=torch.int64, device=device)
            cache_loc_2d = torch.zeros(
                (bs, draft_token_num), dtype=cache_loc_dtype, device=device
            )
            commit_lens = torch.zeros((bs,), dtype=torch.int32, device=device)
            self.append_target_hidden_to_draft_kv_by_loc_tpu(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=commit_lens,
            )
            return f"bs={bs}, tokens={num_tokens}"

        self._precompile_shapes("DFLASH DRAFT APPEND KV (VERIFY)", bs_buckets, run)

    def _precompile_draft_append_kv_extend(self) -> None:
        """Compile DFLASH draft KV append across token buckets (extend/prefill)."""
        if self.t_bucketer is None:
            raise ValueError(
                "t_bucketer must be initialized before precompiling draft extend."
            )
        token_buckets = list(self.t_bucketer.buckets)
        logger.info(
            "[spec compile policy] DFLASH DRAFT APPEND KV (EXTEND): %d shapes (token_buckets=%s)",
            len(token_buckets),
            token_buckets,
        )
        target_hidden_size = self.dflash_target_hidden_size
        device = self.model_runner.device
        dtype = self.model_runner.dtype
        # Extend feeds batch.out_cache_loc straight from the live KV allocator.
        cache_loc_dtype = self._dummy_out_cache_loc_dtype(ForwardMode.EXTEND)

        def run(num_tokens: int) -> str:
            target_hidden = torch.zeros(
                (num_tokens, target_hidden_size), dtype=dtype, device=device
            )
            cache_loc = torch.zeros((num_tokens,), dtype=cache_loc_dtype, device=device)
            positions = torch.zeros((num_tokens,), dtype=torch.int64, device=device)
            self.append_target_hidden_to_draft_kv_by_loc_tpu(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
            )
            return f"tokens={num_tokens}"

        self._precompile_shapes("DFLASH DRAFT APPEND KV (EXTEND)", token_buckets, run)

    def _precompile_draft_verify(
        self, draft_attn_backend: AttentionBackend | None = None
    ) -> None:
        """Compile DFLASH draft model forward across batch size buckets."""
        if self.b_bucketer is None:
            raise ValueError(
                "b_bucketer must be initialized before precompiling draft verify."
            )
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
