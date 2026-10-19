"""TPU compile policy for TpuChainSpecWorker's draft runner."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_disagg
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.attention_rank_inputs import (
    get_attention_rank_tensors,
    use_attention_rank_tensors,
)
from sglang_tpu.compilation.recompilation_checker import (
    install_recompilation_checker,
)
from sglang_tpu.compilation.spec_compile_policy import (
    TpuSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.compilation.spec_decode_compile_policy_dflash import (
    DFlashSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.speculative.chain_draft_input import (
    chain_draft_forward_input,
    register_chain_draft_pytree,
)

logger = get_logger(__name__)


class TpuChainSpecDecodeCompilePolicyMixin(TpuSpecDecodeCompilePolicyMixin):
    """The draft warms its chain forwards per batch bucket.

    The whole draft chain is one compiled program (compile_fused_draft); commit
    forwards the N + 1 verify rows on target features, and extend forwards the
    prompt on target features at the extend cap.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._eager_backbone_forward: Callable[..., Any] | None = None
        self._eager_model_forward: Callable[..., Any] | None = None
        self._eager_logits_forward: Callable[..., Any] | None = None

    def install(self) -> bool:
        """Keep the draft's eager forwards; the draft programs trace them inline."""
        if self.is_draft_worker and self.owner is not None:
            self._eager_backbone_forward = self.owner.model.forward
            self._eager_model_forward = self.owner.forward
            self._eager_logits_forward = self.owner.logits_processor.forward
        installed = super().install()
        if installed and self.is_draft_worker and self.tracer == "direct":
            # The shim's eager coordinator would launch the embedding, norms and fc
            # op by op on every draft-rows and prompt-extend forward.
            assert self.compiled_forward_shim is not None
            self.compiled_forward_shim.fn = self._direct_compile_draft_forward()
        return installed

    @property
    def _model_is_mrope(self) -> bool:
        # The MTP draft ropes the 1D positions it is called with, never mrope_positions.
        return not self.is_draft_worker and super()._model_is_mrope

    @contextmanager
    def _trace_eager_draft(
        self,
        attention_metadata: Any,
        forward_mode: ForwardMode,
        attention_rank_tensors: tuple[torch.Tensor, ...],
    ) -> Iterator[None]:
        """Swap in the eager draft forwards; installed ones launch nested programs."""
        from sglang_tpu.compilation.hybrid_linear_direct_compile import (
            bind_traced_attention_metadata,
        )

        eager_backbone = self._eager_backbone_forward
        eager_model = self._eager_model_forward
        eager_logits = self._eager_logits_forward
        if eager_backbone is None or eager_model is None or eager_logits is None:
            raise RuntimeError("Draft direct compile needs the installed draft policy.")
        owner = self.owner
        backbone = owner.model
        logits_processor = owner.logits_processor
        saved = (owner.forward, backbone.forward, logits_processor.forward)
        owner.forward = eager_model
        backbone.forward = eager_backbone
        logits_processor.forward = eager_logits
        try:
            with (
                bind_traced_attention_metadata(
                    get_forward_context().attn_backend, attention_metadata, forward_mode
                ),
                use_attention_rank_tensors(attention_rank_tensors),
            ):
                yield
        finally:
            owner.forward, backbone.forward, logits_processor.forward = saved

    def _direct_compile_draft(
        self,
        trace: Callable[..., Any],
        specialization_key: Callable[..., tuple[Any, ...]],
    ) -> Callable[..., Any]:
        """direct_compile trace(attention_metadata, ..., attention_rank_tensors=...)."""
        from sglang_tpu.compilation.direct_compile import direct_compile
        from sglang_tpu.compilation.hybrid_linear_direct_compile import (
            traced_attention_metadata,
        )

        register_chain_draft_pytree()
        # trace names forward_batch itself: ignored_argnames resolve by parameter name.
        compiled = direct_compile(
            trace,
            specialization_key=specialization_key,
            model=self.model_runner.model,
            model_extra_tensor_attrs=self.get_model_extra_tensor_attrs(),
            kv_pool=self.get_direct_compile_kv_pool(),
            ignored_argnames=self._ignored_argnames(),
        )

        def call(*args: Any, **kwargs: Any) -> Any:
            return compiled(
                traced_attention_metadata(get_forward_context().attn_backend),
                *args,
                attention_rank_tensors=get_attention_rank_tensors(
                    torch.device(self.model_runner.device)
                ),
                **kwargs,
            )

        return call

    def _direct_compile_draft_forward(self) -> Callable[..., Any]:
        """The whole draft forward, embedding to logits, as one program per shape."""
        assert self._eager_model_forward is not None
        eager_model = torch.no_grad()(self._eager_model_forward)

        def trace(
            attention_metadata: Any,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            forward_batch: ForwardBatch,
            *args: Any,
            attention_rank_tensors: tuple[torch.Tensor, ...],
            **kwargs: Any,
        ) -> Any:
            with self._trace_eager_draft(
                attention_metadata, forward_batch.forward_mode, attention_rank_tensors
            ):
                if not forward_batch.forward_mode.is_target_verify():
                    return eager_model(
                        input_ids, positions, forward_batch, *args, **kwargs
                    )
                # Draft TARGET_VERIFY is commit's draft rows, which read only the
                # hidden; skip the lm_head and its full-vocab all-gather over every row.
                logits_processor = self.owner.logits_processor
                logits_processor.forward = _hidden_only_logits
                return eager_model(input_ids, positions, forward_batch, *args, **kwargs)

        return self._direct_compile_draft(
            trace, specialization_key=self._direct_compile_specialization_key
        )

    def compile_fused_draft(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """One program per (batch bucket, sampled) for the whole draft chain."""
        if self.tracer != "direct":
            return torch.compile(
                fn,
                backend=self.backend or current_platform.get_compile_backend(),
                fullgraph=False,
                dynamic=False,
            )
        fn = torch.no_grad()(fn)

        def trace(
            attention_metadata: Any,
            forward_batch: ForwardBatch,
            *args: Any,
            attention_rank_tensors: tuple[torch.Tensor, ...],
        ) -> Any:
            with self._trace_eager_draft(
                attention_metadata, forward_batch.forward_mode, attention_rank_tensors
            ):
                return fn(forward_batch, *args)

        def specialization_key(
            _attention_metadata: Any, forward_batch: ForwardBatch, *args: Any, **_: Any
        ) -> tuple[Any, ...]:
            # args[5] is temperatures: None for an all-greedy batch.
            return (forward_batch.batch_size, args[5] is None)

        call = self._direct_compile_draft(trace, specialization_key=specialization_key)
        return functools.wraps(fn)(call)

    def _make_dummy_forward_batch(
        self,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        *,
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: DFlashVerifyInput | None = None,
    ) -> ForwardBatch:
        if not (
            self.is_draft_worker and mode.is_extend() and not mode.is_target_verify()
        ):
            return super()._make_dummy_forward_batch(
                num_tokens=num_tokens,
                bs=bs,
                mode=mode,
                capture_hidden_mode=capture_hidden_mode,
                spec_info=spec_info,
            )
        # The DFlash override shapes every draft batch as verify rows; the chain
        # draft prompt extend is a plain extend.
        batch = super(
            DFlashSpecDecodeCompilePolicyMixin, self
        )._make_dummy_forward_batch(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        batch.input_embeds = None
        return batch

    def precompile_draft(
        self,
        draft_attn_backend: AttentionBackend,
        *,
        target_feature_size: int,
        fused_draft_warmup: Callable[[int, bool], None],
    ) -> None:
        """Compile the fused draft, commit's draft rows and the draft prompt extend."""
        if not self.is_draft_worker:
            raise ValueError("precompile_draft runs on the draft runner only.")
        if self.b_bucketer is None or self.t_bucketer is None:
            raise ValueError("Bucketers must be initialized before precompile_draft.")
        start = time.perf_counter()
        disagg_mode = get_disagg().disaggregation_mode
        if disagg_mode != "prefill":
            self._precompile_fused_draft(fused_draft_warmup)
            self._precompile_draft_rows(
                draft_attn_backend, target_feature_size=target_feature_size
            )
        if disagg_mode != "decode":
            self._precompile_draft_prompt_extend(target_feature_size)
        logger.info(
            "[spec compile policy] %s DRAFT precompile done in %.1fs.",
            self.algo_label,
            time.perf_counter() - start,
        )
        install_recompilation_checker(
            policies=(self,),
            model_label=f"draft {type(self.model_runner.model).__name__}",
        )

    def _precompile_fused_draft(
        self, fused_draft_warmup: Callable[[int, bool], None]
    ) -> None:
        assert self.b_bucketer is not None
        shapes = [
            (bs, sampled) for bs in self.b_bucketer.buckets for sampled in (False, True)
        ]
        logger.info(
            "[spec compile policy] %s FUSED DRAFT: %d shapes (bs=%s, greedy and sampled)",
            self.algo_label,
            len(shapes),
            list(self.b_bucketer.buckets),
        )

        def run(shape: tuple[int, bool]) -> str:
            bs, sampled = shape
            fused_draft_warmup(bs, sampled)
            return f"bs={bs}, sampled={sampled}"

        self._precompile_shapes(f"{self.algo_label} FUSED DRAFT", shapes, run)

    def _precompile_draft_rows(
        self, draft_attn_backend: AttentionBackend, *, target_feature_size: int
    ) -> None:
        """commit's draft forward over the N + 1 verify rows on target features."""
        assert self.b_bucketer is not None
        width = self.draft_token_num
        buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[spec compile policy] %s DRAFT ROWS: %d shapes (bs=%s, verify rows=%d)",
            self.algo_label,
            len(buckets),
            buckets,
            width,
        )

        def run(bs: int) -> str:
            self._run_one(
                num_tokens=bs * width,
                bs=bs,
                mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=chain_draft_forward_input(
                    hidden_states=self._dummy_hidden(bs * width, target_feature_size),
                    width=width,
                ),
                attn_backend=draft_attn_backend,
            )
            return f"bs={bs}, rows={width}"

        self._precompile_shapes(f"{self.algo_label} DRAFT ROWS", buckets, run)

    def _precompile_draft_prompt_extend(self, target_feature_size: int) -> None:
        assert self.t_bucketer is not None
        bs = self._extend_compile_cap()
        token_buckets = [t for t in self.t_bucketer.buckets if t >= bs]
        logger.info(
            "[spec compile policy] %s DRAFT EXTEND(LAST): %d shapes (bs=%d, tokens=%s)",
            self.algo_label,
            len(token_buckets),
            bs,
            token_buckets,
        )

        def run(num_tokens: int) -> str:
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.EXTEND,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=chain_draft_forward_input(
                    hidden_states=self._dummy_hidden(num_tokens, target_feature_size),
                    width=1,
                ),
            )
            return f"tokens={num_tokens}, bs={bs}"

        self._precompile_shapes(
            f"{self.algo_label} DRAFT EXTEND(LAST)", token_buckets, run
        )

    def _dummy_hidden(self, rows: int, width: int) -> torch.Tensor:
        return torch.zeros(
            (rows, width),
            dtype=self.model_runner.dtype,
            device=self.model_runner.device,
        )


def _hidden_only_logits(
    input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    logits_metadata: Any,
    *args: Any,
    **kwargs: Any,
) -> LogitsProcessorOutput:
    """FULL capture's hidden, as LogitsProcessor returns it, without the logits."""
    del input_ids, lm_head, logits_metadata, args, kwargs
    return LogitsProcessorOutput(next_token_logits=None, hidden_states=hidden_states)
