# SPDX-License-Identifier: Apache-2.0
"""TPU compile policy of ``DeepseekV4ForCausalLM`` and its bundled DSpark draft
``DeepseekV4ForCausalLMDSpark`` (attached in
``sglang_tpu.compilation.core_compile_annotations``).

After the batch is padded to its bucket and before attention planning, it
prepares eager DP token embeddings, and right before the compiled forward its
``CompiledForwardShim`` subclass stages host Engram rows:

- Token embedding, under attention DP: looked up eagerly (with MM/negative
  sentinel clamping) and passed in as ``input_embeds``. Attention DP builds the
  embedding unsharded (``enable_tp=not is_dp_attention_enabled()``), and inside
  the TP1/DP8 compiled graph that BF16 gather returns wrong rows, with int64
  ids too, while the same lookup outside the graph is exact (as vllm-torchtpu
  #1848 does). Running it in ``prepare_forward_batch`` also overlaps the TPU
  gather with attention metadata setup and host Engram overlap resolution.
  Remove once torch_tpu compiles that gather correctly.
- Engram, when the model has Engram layers: ``_EngramCompiledForwardShim``
  resolves any deferred overlap tokens, packs rows with single-threaded NumPy
  and uploads via double-buffered async raw DMA (``engram_host``), records a
  TPU stream ``Event`` (or output tensors on non-event devices) to gate DMA
  slot reuse without retaining forward output buffers in HBM (safe on IDLE DP
  ranks and async warmup ``PendingCall``), and triggers next-chunk prefill
  prefetch after dispatching the compiled forward.

Warmup pads through the same ``prepare_forward_batch`` and shim, so precompiled
and serving graphs share one signature.

Under ``--tpu-compile-tracer direct`` (V4.1): the policy first registers the
model's plain tensor attributes and prunes unused inputs
(``direct_tracer.prepare_direct_compile``), and the staged Engram rows ride
on the attention metadata, because the tracer rebuilds ``ForwardBatch`` from
its dataclass fields only. V4.1's graphs are ``spmd_safe``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.models.deepseek_v41 import engram_host


def _stage_host_engram(coordinator, hasher, input_ids, forward_batch):
    """Pack and stage this rank's Engram rows on the host via ``coordinator``."""
    padded_tokens = input_ids.shape[0]
    device = input_ids.device
    engram_host.move_hasher_to_host(hasher)
    # DSpark verify commits wait here, after the next draft is dispatched; both
    # paths below read or write hasher.history.
    engram_host.flush_engram_commits(hasher)
    staged_result = None
    if coordinator.started:
        staged_result = coordinator.stage_all(
            padded_tokens, forward_batch=forward_batch
        )
    elif forward_batch.forward_mode.is_target_verify():
        staged_result = coordinator.stage_verify(input_ids, forward_batch)
    if staged_result is None:
        with torch.no_grad():
            hash_ids = engram_host.host_hash_ids(hasher, input_ids, forward_batch)
        staged_result = coordinator.stage_from_hashes(hash_ids, padded_tokens)
    staged, uploads, num_cols = staged_result
    forward_batch.tpu_engram_staged = staged
    forward_batch.tpu_engram_local_ids = engram_host.local_row_ids(
        padded_tokens, len(staged), num_cols, device
    )
    return uploads


class _EngramCompiledForwardShim(CompiledForwardShim):
    """CompiledForwardShim that stages double-buffered host Engram uploads."""

    def _forward(
        self,
        fn: Callable[..., Any],
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        forward_batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        hasher = self.policy._engram_hasher
        if hasher is None or torch.compiler.is_compiling():
            return super()._forward(
                fn, input_ids, positions, forward_batch, *args, **kwargs
            )
        model = self.policy.model_runner.model
        device = (
            input_ids.device
            if isinstance(input_ids, torch.Tensor)
            else self.policy.model_runner.device
        )
        coordinator = engram_host.get_or_create_coordinator(model, hasher, device)
        if coordinator is None:
            return super()._forward(
                fn, input_ids, positions, forward_batch, *args, **kwargs
            )
        saved_ids = forward_batch.tpu_engram_local_ids
        saved_staged = forward_batch.tpu_engram_staged
        active_uploads = None
        padded_tokens = input_ids.shape[0] if isinstance(input_ids, torch.Tensor) else 0
        try:
            active_uploads = _stage_host_engram(
                coordinator, hasher, input_ids, forward_batch
            )
            if not active_uploads:
                return super()._forward(
                    fn, input_ids, positions, forward_batch, *args, **kwargs
                )
            raw_consumers: list[torch.Tensor] = []
            recorded_events: list[Any] = []

            def wrapped_fn(*f_args, **f_kwargs):
                raw_out = fn(*f_args, **f_kwargs)
                if not isinstance(raw_out, PendingCall):
                    event = coordinator.make_consumer_event()
                    if event is not None:
                        recorded_events.append(event)
                    else:
                        raw_consumers.extend(
                            engram_host.collect_output_tensors(raw_out, device=device)
                        )
                    return raw_out

                def _capture_pending(res):
                    if res is None:
                        return res
                    event = coordinator.make_consumer_event()
                    if event is not None:
                        recorded_events.append(event)
                    else:
                        raw_consumers.extend(
                            engram_host.collect_output_tensors(res, device=device)
                        )
                    return res

                return raw_out.then(_capture_pending)

            out = super()._forward(
                wrapped_fn, input_ids, positions, forward_batch, *args, **kwargs
            )
            if isinstance(out, PendingCall):
                # Deferred warmup/precompile call: clear needs_consumer now so
                # multiple queued warmup variants on the same token bucket can
                # stage without clashing, and re-pin staged fields at wait().
                coordinator.record_consumers(active_uploads, [])
                staged_ids = forward_batch.tpu_engram_local_ids
                staged_rows = forward_batch.tpu_engram_staged
                uploads_for_call = active_uploads
                active_uploads = None

                def _run_pending():
                    prev_ids = forward_batch.tpu_engram_local_ids
                    prev_staged = forward_batch.tpu_engram_staged
                    forward_batch.tpu_engram_local_ids = staged_ids
                    forward_batch.tpu_engram_staged = staged_rows
                    try:
                        res = out.wait()
                    finally:
                        forward_batch.tpu_engram_local_ids = prev_ids
                        forward_batch.tpu_engram_staged = prev_staged
                    if recorded_events:
                        coordinator.bind_consumer_event(
                            uploads_for_call, recorded_events[-1]
                        )
                    else:
                        consumers = raw_consumers or engram_host.collect_output_tensors(
                            res, device=device
                        )
                        if consumers:
                            for upload in uploads_for_call:
                                upload.consumers = consumers
                                upload.needs_consumer = False
                    return res

                return PendingCall(_run_pending)
            if recorded_events:
                coordinator.record_consumer_event(active_uploads, recorded_events[-1])
                active_uploads = None
                if coordinator.has_next_prompt_prefetch():
                    # event.record() already flushed the compiled forward to PJRT.
                    coordinator.finish_forward(padded_tokens)
            else:
                consumers = raw_consumers or engram_host.collect_output_tensors(
                    out, device=device
                )
                coordinator.record_consumers(active_uploads, consumers)
                active_uploads = None
                if coordinator.has_next_prompt_prefetch():
                    coordinator.flush_consumers(consumers)
                    coordinator.finish_forward(padded_tokens)
            return out
        finally:
            if active_uploads:
                coordinator.record_consumers(active_uploads, [])
            forward_batch.tpu_engram_local_ids = saved_ids
            forward_batch.tpu_engram_staged = saved_staged


def is_deepseek_v41(model_runner) -> bool:
    config = getattr(model_runner.model_config, "hf_config", None)
    return getattr(config, "model_type", None) == "deepseek_v41"


class DeepseekV4DecoderTpuCompilePolicy(DecoderTpuCompilePolicy):
    """Decoder policy that stages host Engram rows and DP token embeddings.

    For V4.1, ``spmd_safe`` declares the traced graphs identical across ranks,
    so ``--use-spmd-compile`` compiles each warmup once and broadcasts it.
    Rank-dependent values must reach the graph as rank inputs: attention DP and
    MoE EP offsets go through the lifted rank tensors, and every per-rank
    batch quantity (tokens, staged Engram rows, attention metadata) is a graph
    input. Verify with ``VERIFY_SPMD_SAFE=1`` before a recipe enables the
    flag: a rank-dependent constant would ship the owner rank's executable to
    every rank with no error. V4 was not verified and keeps the defaults.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.is_v41 = is_deepseek_v41(self.model_runner)
        self.spmd_safe = self.is_v41
        model = self.model_runner.model
        # No Engram layers (DeepSeek-V4) or a NextN draft, which has no hasher.
        self._engram_hasher = (
            model.model.engram_hasher
            if self.model_runner.model_config.use_engram
            and not self.model_runner.is_draft_worker
            else None
        )
        self._embeds_outside_graph = get_parallel().attn_dp_size > 1

    def _install_forward_shim(
        self,
        fn: Callable[..., Any],
        compiled: Callable[..., Any],
        warmup_async: Callable[..., PendingCall] | None = None,
    ) -> None:
        super()._install_forward_shim(
            fn=fn, compiled=compiled, warmup_async=warmup_async
        )
        if self._engram_hasher is None:
            return
        base = self.compiled_forward_shim
        assert base is not None
        self.compiled_forward_shim = _EngramCompiledForwardShim(
            base.fn, policy=self, warmup_async_fn=base.warmup_async_fn
        )
        setattr(self.owner, self.method_name, self.compiled_forward_shim)

    def _embed_outside_graph(self, forward_batch: ForwardBatch) -> None:
        model = self.model_runner.model
        has_mm_inputs = (
            not self.model_runner.is_draft_worker
            and model.vision is not None
            and not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.mm_inputs is not None
            and any(x is not None for x in forward_batch.mm_inputs)
        )
        if (
            not self._embeds_outside_graph
            or forward_batch.input_embeds is not None
            or has_mm_inputs
            or not isinstance(forward_batch.input_ids, torch.Tensor)
            or forward_batch.input_ids.is_cuda
        ):
            return
        config = self.model_runner.model_config.hf_config
        image_token_id = (
            config.image_token_id
            if config.model_type == "deepseek_v41" and config.vision_n_layers > 0
            else None
        )
        fallback_id = (
            image_token_id
            if image_token_id is not None and 0 <= image_token_id < config.vocab_size
            else config.engram_pad_token_id
        )
        with torch.no_grad():
            ids = forward_batch.input_ids
            safe_ids = torch.where(
                (ids < 0) | (ids >= engram_host._MM_PAD_SHIFT_VALUE),
                fallback_id,
                ids,
            )
            forward_batch.input_embeds = self._embed(safe_ids)

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        super().prepare_forward_batch(forward_batch)
        # Enqueue outside-graph embedding on TPU before attention planning and
        # host Engram staging so the TPU gather overlaps with overlap-token
        # resolution and NumPy row packing.
        self._embed_outside_graph(forward_batch)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        model = self.model_runner.model
        if isinstance(model, DeepseekV4ForCausalLMDSpark):
            # As forward_embed: the draft's stages read hc_mult copies per token.
            return (
                model.embed_tokens(input_ids).unsqueeze(1).repeat(1, model.hc_mult, 1)
            )
        return model.get_input_embeddings()(input_ids)

    def _supports_async_warmup(self) -> bool:
        # The DSpark draft forward is one compiled program (no eager logits
        # coordinator), so its block warmups can join the shared SPMD compile.
        return super()._supports_async_warmup() or (
            self.is_v41 and self.model_runner.is_draft_worker
        )

    def _run_one(self, **kwargs):
        backend = kwargs.get("attn_backend") or self.model_runner.attn_backend
        if (
            not getattr(backend, "decoder_replay_enabled", False)
            or not kwargs["mode"].is_extend_without_speculative()
            or kwargs.get("profile_prefix") is not None
        ):
            return super()._run_one(**kwargs)

        pending = []
        total = kwargs["num_tokens"]
        buckets = [0, *(rows for rows in self.t_bucketer.buckets if rows <= total)]
        try:
            for rows in buckets:
                backend.decoder_warmup_rows = rows
                warmup = super()._run_one(**kwargs)
                if isinstance(warmup, PendingCall):
                    pending.append(warmup)
        finally:
            backend.decoder_warmup_rows = None
        if pending:

            def wait_all():
                for warmup in pending:
                    warmup.wait()

            return PendingCall(wait_all)
        return None

    def _direct_compile_specialization_key(
        self, attention_metadata, input_ids, positions, forward_batch, *args, **kwargs
    ):
        key = super()._direct_compile_specialization_key(
            attention_metadata, input_ids, positions, forward_batch, *args, **kwargs
        )
        tail = getattr(attention_metadata, "late_layer_tail", None)
        if tail is None:
            return key
        # All DP peers share the same phase and tail bucket. Individual ranks'
        # live rows and selectors travel as tensors.
        return (*key, tail.token_indices.shape[0])

    def _direct_compile(self, fn):
        if self.is_v41:
            from sglang_tpu.models.deepseek_v41.direct_tracer import (
                prepare_direct_compile,
            )

            prepare_direct_compile(self)
        call, warmup_async = super()._direct_compile(fn)
        if self._engram_hasher is None or not engram_host.shared_host_tables(
            self.model_runner.model
        ):
            return call, warmup_async
        return _carrying_engram(call), _carrying_engram(warmup_async)


def _carrying_engram(run):
    """Put the batch's staged Engram on the attention metadata, a pytree the
    direct tracer lifts as graph inputs (engram_host.staged_inputs), before
    ``run(input_ids, positions, forward_batch, ...)`` reads the metadata."""

    @functools.wraps(run)
    def carried(input_ids, positions, forward_batch, *args, **kwargs):
        metadata = get_forward_context().attn_backend.forward_metadata
        if metadata is None or not hasattr(metadata, "engram"):
            raise RuntimeError(
                "DeepSeek-V4.1 direct tracer: the attention metadata must be built "
                "before the compiled forward to carry the staged Engram rows"
            )
        staged = forward_batch.tpu_engram_staged
        metadata.engram = (
            forward_batch.tpu_engram_local_ids,
            tuple(
                item if isinstance(item, torch.Tensor) else tuple(item)
                for item in staged
            ),
        )
        return run(input_ids, positions, forward_batch, *args, **kwargs)

    return carried
