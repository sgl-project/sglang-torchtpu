"""DFlash speculative-decoding TPU compile policy."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import msgspec
import torch
import torch.nn.functional as F

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.linear import LinearBase
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_disagg, get_spec
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpecInput, SpecPaddingState
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.compiled_forward_shim import (
    _REAL_B_KEY,
    _REAL_Q_KEY,
    _restore_forward_batch_inputs,
)
from sglang_tpu.compilation.decoder_compile_policy import (
    DecoderTpuCompilePolicy,
)
from sglang_tpu.compilation.recompilation_checker import (
    install_recompilation_checker,
)
from sglang_tpu.compilation.spec_decode_compile_policy import (
    BaseSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.envs import envs as tpu_envs

logger = get_logger(__name__)

# Neutral values for the padded sampling rows; their outputs are discarded.
_TEMPERATURE_FILL = 1.0
_TOP_K_FILL = 1
_TOP_P_FILL = 1.0


class _DFlashVerifyPadding(msgspec.Struct, frozen=True):
    """Logical tensors saved while a verify batch runs on padded rows."""

    state: SpecPaddingState
    batch: Any
    verify_input: SpecInput
    seq_lens: torch.Tensor
    temperatures: torch.Tensor | None
    top_ks: torch.Tensor | None
    top_ps: torch.Tensor | None


def _pad_rows(tensor: torch.Tensor, pad_rows: int, fill: float) -> torch.Tensor:
    """Append filler requests to a per-request tensor."""
    pad_spec = [0, 0] * (tensor.dim() - 1) + [0, pad_rows]
    return F.pad(tensor, pad_spec, value=fill)


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


def _block_embedder_specialization_key(
    *, block_ids: torch.Tensor
) -> tuple[object, ...]:
    """Key block-embedder artifacts on the padded [bs, block_size] input shape."""
    return (tuple(block_ids.shape), block_ids.dtype)


class DFlashSpecDecodeCompilePolicyMixin(BaseSpecDecodeCompilePolicyMixin):
    """DFlash speculative decoding compile policy mixin."""

    # Prefixes the draft precompile progress labels; DSpark reuses these stages.
    algo_label = "DFLASH"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if not self.draft_token_num:
            self.draft_token_num = self._infer_dflash_block_size()
        self.dflash_hidden_size = self._infer_dflash_hidden_size()
        self.dflash_target_hidden_size = self._infer_dflash_target_hidden_size()
        self._dflash_worker: DFlashWorkerV2 | None = None
        self._compiled_kv_appender: Callable[..., None] | None = None
        self._compiled_block_embedder: Callable[..., torch.Tensor] | None = None

    def _infer_dflash_block_size(self) -> int:
        """Infer DFlash block size from the spec config or the draft config."""
        spec = get_spec()
        if spec.speculative_num_draft_tokens is not None:
            return int(spec.speculative_num_draft_tokens)
        if spec.speculative_draft_model_path:
            try:
                draft_model_config = ModelConfig.from_server_args(
                    self.model_runner.server_args,
                    model_path=spec.speculative_draft_model_path,
                    model_revision=spec.speculative_draft_model_revision,
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
            # LinearBase (row-parallel included) reports the full input width.
            return int(fc.input_size if isinstance(fc, LinearBase) else fc.in_features)
        return self.dflash_hidden_size or 1

    def install(self) -> bool:
        if self.is_draft_worker:
            self._rewrite_draft_layers()
        return super().install()

    def _rewrite_draft_layers(self) -> None:
        """Swap draft layers for their TPU decode variants before the forward is traced."""
        from sglang.srt.environ import envs
        from sglang.srt.runtime_context import get_parallel

        # The TPU KV appender projects target hidden states outside draft_tp_context,
        # so a row-parallel fc would reduce over the wrong group under DP attention.
        if envs.SGLANG_DFLASH_ROW_PARALLEL_FC.get() and get_parallel().attn_dp_enabled:
            raise NotImplementedError(
                "SGLANG_DFLASH_ROW_PARALLEL_FC is not supported with DP attention on TPU"
            )
        model = self.model_runner.model
        if tpu_envs.SGLANG_TPU_DFLASH_FP8_DRAFT.get():
            from sglang_tpu.fp8_weight_linear import (
                install_fp8_linears,
                install_fp8_plain_linears,
            )

            num_fp8 = install_fp8_linears(model)
            num_fp8 += install_fp8_plain_linears(model, ("kernel_projection",))
            logger.info(
                "[spec compile policy] DFLASH draft: %d linears as fp8 weight-only.",
                num_fp8,
            )

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
            from sglang_tpu.compilation.direct_compile import (
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

    def should_defer_verify_output_unpadding(
        self, batch: Any, verify_input: Any
    ) -> bool:
        """Keep target verify outputs padded so accept runs on bucket shapes."""
        # IDLE never reaches verify: DFlashWorkerV2.forward_batch_generation
        # short-circuits it, so there is no DP-peer case to cover here.
        return not batch.forward_mode.is_idle()

    def prepare_spec_verify(
        self, batch: Any, verify_input: Any, next_token_logits: torch.Tensor
    ) -> _DFlashVerifyPadding | None:
        """Widen the per-request verify inputs to match the padded logits."""
        state = verify_input.padding_state
        if state is None or not verify_input.keep_outputs_padded:
            return None
        if state.physical_num_tokens == state.logical_num_tokens:
            return None
        rows = int(next_token_logits.shape[0])
        if rows == state.logical_num_tokens:
            # Something already restored the outputs before we got here.
            verify_input.keep_outputs_padded = False
            return None
        if rows != state.physical_num_tokens:
            raise ValueError(
                "Deferred DFLASH verify logits do not match the recorded padding: "
                f"state={state}, logits rows={rows}."
            )

        sampling_info = batch.sampling_info
        padding = _DFlashVerifyPadding(
            state=state,
            batch=batch,
            verify_input=verify_input,
            seq_lens=batch.seq_lens,
            temperatures=None if sampling_info is None else sampling_info.temperatures,
            top_ks=None if sampling_info is None else sampling_info.top_ks,
            top_ps=None if sampling_info is None else sampling_info.top_ps,
        )
        pad_rows = state.physical_batch_size - state.logical_batch_size
        # draft_token is not read after the forward, so it stays logical.
        batch.seq_lens = _pad_rows(batch.seq_lens, pad_rows, 0)
        if sampling_info is not None:
            sampling_info.temperatures = _pad_rows(
                sampling_info.temperatures, pad_rows, _TEMPERATURE_FILL
            )
            sampling_info.top_ks = _pad_rows(
                sampling_info.top_ks, pad_rows, _TOP_K_FILL
            )
            sampling_info.top_ps = _pad_rows(
                sampling_info.top_ps, pad_rows, _TOP_P_FILL
            )
        return padding

    def finalize_dflash_verify(
        self,
        padding: _DFlashVerifyPadding,
        results: tuple[Any, ...],
        logits_output: Any,
    ) -> tuple[Any, ...]:
        """Restore the logical verify inputs and trim every deferred output.

        The token-dim trim is done here off padding_state rather than left to
        finalize_forward_batch: that path keys off the _tpu_real_q stamp, which
        only exists when prepare_forward_batch ran, and the padded rows are fatal
        downstream (a later view folds them into the feature dim).
        """
        batch = padding.batch
        batch.seq_lens = padding.seq_lens
        sampling_info = batch.sampling_info
        if sampling_info is not None:
            sampling_info.temperatures = padding.temperatures
            sampling_info.top_ks = padding.top_ks
            sampling_info.top_ps = padding.top_ps
        padding.verify_input.keep_outputs_padded = False

        num_tokens = padding.state.logical_num_tokens
        for field in ("next_token_logits", "hidden_states"):
            value = getattr(logits_output, field, None)
            if isinstance(value, torch.Tensor) and value.shape[0] > num_tokens:
                setattr(logits_output, field, value[:num_tokens])

        logical_bs = padding.state.logical_batch_size
        return tuple(t[:logical_bs] if t is not None else None for t in results)

    def finalize_forward_batch(self, outputs: Any, forward_batch: ForwardBatch) -> Any:
        """Skip the output trim while a deferred verify still needs padded rows."""
        spec_info = forward_batch.spec_info
        if (
            forward_batch.forward_mode.is_target_verify()
            and spec_info is not None
            and spec_info.keep_outputs_padded
        ):
            # Inputs are restored even when outputs stay padded; the worker calls
            # back here once accept is done.
            real_q = getattr(forward_batch, _REAL_Q_KEY, None)
            real_b = getattr(forward_batch, _REAL_B_KEY, None)
            if real_q is not None and real_b is not None:
                _restore_forward_batch_inputs(forward_batch, real_q, real_b)
            return outputs
        return super().finalize_forward_batch(outputs, forward_batch)

    def _ensure_block_embedder(
        self, dflash_worker: DFlashWorkerV2 | None = None
    ) -> None:
        """Compile and install the drafted-block embedder on the worker.

        Run eager, the vocab-parallel mask fans out into a dozen separately
        dispatched kernels plus a standalone all-reduce; compiling collapses them
        into one launch. The seam is an attribute on the worker, never on the
        embedding module itself -- the target's own forward traces that module
        inline, and a CompiledFunction there would launch mid-trace.
        """
        if dflash_worker is None:
            raise ValueError("DFlash worker cannot be None")
        self._dflash_worker = dflash_worker
        if self._compiled_block_embedder is None:
            self._compiled_block_embedder = self._compile_block_embedder(
                dflash_worker.embed_draft_block_ids,
                embed_module=dflash_worker.ensure_embed_module(),
            )
        dflash_worker.embed_draft_block_ids = self._compiled_block_embedder

    def _compile_block_embedder(
        self, fn: Callable[..., torch.Tensor], *, embed_module: torch.nn.Module
    ) -> Callable[..., torch.Tensor]:
        """Compile the block embedder against the embedding module's own weight."""
        if not isinstance(embed_module.weight, torch.nn.Parameter):
            raise TypeError(
                "DFLASH block embedder requires embed_module.weight to be a Parameter "
                f"so direct_compile can lift it; got {type(embed_module.weight)}."
            )
        fn = torch.no_grad()(fn)
        if self.tracer == "direct":
            from sglang_tpu.compilation.direct_compile import (
                direct_compile,
            )

            return direct_compile(
                fn,
                specialization_key=_block_embedder_specialization_key,
                model=embed_module,
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
        # Device/dtype normalization happens inside the compiled appender.
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
            # The live draft batch sets only the local field; direct_compile pins
            # each input's type at trace time.
            batch.global_num_token_non_padded = None
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
        if get_disagg().disaggregation_mode != "prefill":
            self._precompile_draft_verify(draft_attn_backend)
        self._precompile_append_target_hidden_to_draft_kv_by_loc_tpu(dflash_worker)
        self._precompile_embed_draft_block_ids(dflash_worker)
        logger.info(
            "[spec compile policy] DFLASH DRAFT precompile done in %.1fs.",
            time.perf_counter() - overall,
        )
        # Include draft warmup in the shared target/draft baseline. Later TPU
        # speculative-op and KV-offload warmup refresh it when they finish.
        install_recompilation_checker(
            policies=(self,),
            model_label=f"draft {type(self.model_runner.model).__name__}",
        )

    def _precompile_append_target_hidden_to_draft_kv_by_loc_tpu(
        self, dflash_worker: DFlashWorkerV2
    ) -> None:
        """Warm up DFlash draft KV append across token buckets and batch size buckets."""
        self._ensure_kv_appender(dflash_worker)
        disagg_mode = get_disagg().disaggregation_mode
        if disagg_mode != "prefill":
            self._precompile_draft_append_kv_verify()
        if disagg_mode != "decode":
            self._precompile_draft_append_kv_extend()

    @torch.no_grad()
    def _precompile_embed_draft_block_ids(self, dflash_worker: DFlashWorkerV2) -> None:
        """Compile the drafted-block embedder across batch size buckets."""
        # Dynamo guards the caller's grad mode before entering the compiled
        # no_grad wrapper, so warmup must match the scheduler's no_grad context.
        self._ensure_block_embedder(dflash_worker)
        if self.b_bucketer is None:
            raise ValueError(
                "b_bucketer must be initialized before precompiling block embed."
            )
        bs_buckets = list(self.b_bucketer.buckets)
        draft_token_num = self.draft_token_num
        device = self.model_runner.device
        logger.info(
            "[spec compile policy] DFLASH DRAFT BLOCK EMBED: %d shapes (bs=%s, draft_token_num=%d)",
            len(bs_buckets),
            bs_buckets,
            draft_token_num,
        )

        def run(bs: int) -> str:
            # direct_compile does not normalize call structure, so this must match
            # the serving call exactly: keyword arg, long dtype, [bs, block_size].
            block_ids = torch.zeros(
                (bs, draft_token_num), dtype=torch.long, device=device
            )
            dflash_worker.embed_draft_block_ids(block_ids=block_ids)
            return f"bs={bs}"

        self._precompile_shapes("DFLASH DRAFT BLOCK EMBED", bs_buckets, run)

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
            "[spec compile policy] %s DRAFT APPEND KV (VERIFY): %d shapes (bs=%s, draft_token_num=%d)",
            self.algo_label,
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

        self._precompile_shapes(
            f"{self.algo_label} DRAFT APPEND KV (VERIFY)", bs_buckets, run
        )

    def _precompile_draft_append_kv_extend(self) -> None:
        """Compile DFLASH draft KV append across token buckets (extend/prefill)."""
        if self.t_bucketer is None:
            raise ValueError(
                "t_bucketer must be initialized before precompiling draft extend."
            )
        token_buckets = list(self.t_bucketer.buckets)
        logger.info(
            "[spec compile policy] %s DRAFT APPEND KV (EXTEND): %d shapes (token_buckets=%s)",
            self.algo_label,
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

        self._precompile_shapes(
            f"{self.algo_label} DRAFT APPEND KV (EXTEND)", token_buckets, run
        )

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
