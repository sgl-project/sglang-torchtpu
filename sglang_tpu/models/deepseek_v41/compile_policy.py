# SPDX-License-Identifier: Apache-2.0
"""TPU compile policy of ``DeepseekV4ForCausalLM`` (attached in
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
  and uploads via double-buffered async raw DMA (``engram_host``), captures
  un-sliced forward outputs as DMA slot consumers (safe on IDLE DP ranks and
  async warmup ``PendingCall``), and triggers next-chunk prefill prefetch after
  dispatching the compiled forward.

Warmup pads through the same ``prepare_forward_batch`` and shim, so precompiled
and serving graphs share one signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.compilation.compiled_forward_shim import CompiledForwardShim
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.models.deepseek_v41 import engram_host

_ENGRAM_STAGED_FIELDS = ("tpu_engram_local_ids", "tpu_engram_staged")


def _stage_host_engram(coordinator, hasher, input_ids, forward_batch):
    """Pack and stage this rank's Engram rows on the host via ``coordinator``."""
    padded_tokens = input_ids.shape[0]
    device = input_ids.device
    if coordinator.started:
        staged, uploads, num_cols = coordinator.stage_all(padded_tokens)
    else:
        engram_host.move_hasher_to_host(hasher)
        with torch.no_grad():
            hash_ids = engram_host.host_hash_ids(hasher, input_ids, forward_batch)
        staged, uploads, num_cols = coordinator.stage_from_hashes(
            hash_ids, padded_tokens
        )
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
        hasher = getattr(self.policy, "_engram_hasher", None)
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
        saved = {name: getattr(forward_batch, name) for name in _ENGRAM_STAGED_FIELDS}
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

            def wrapped_fn(*f_args, **f_kwargs):
                raw_out = fn(*f_args, **f_kwargs)
                if not isinstance(raw_out, PendingCall):
                    raw_consumers.extend(
                        engram_host.collect_output_tensors(raw_out, device=device)
                    )
                    return raw_out

                def _capture_pending(res):
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
                staged_fields = {
                    name: getattr(forward_batch, name) for name in _ENGRAM_STAGED_FIELDS
                }
                uploads_for_call = active_uploads
                active_uploads = None

                def _run_pending():
                    prev = {
                        name: getattr(forward_batch, name) for name in staged_fields
                    }
                    for name, val in staged_fields.items():
                        setattr(forward_batch, name, val)
                    try:
                        res = out.wait()
                    finally:
                        for name, val in prev.items():
                            setattr(forward_batch, name, val)
                    consumers = raw_consumers or engram_host.collect_output_tensors(
                        res, device=device
                    )
                    if consumers:
                        for upload in uploads_for_call:
                            upload.consumers = consumers
                            upload.needs_consumer = False
                    return res

                return PendingCall(_run_pending)
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
            for name, value in saved.items():
                setattr(forward_batch, name, value)


class DeepseekV4DecoderTpuCompilePolicy(DecoderTpuCompilePolicy):
    """Decoder policy that stages host Engram rows and DP token embeddings."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
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
            getattr(model, "vision", None) is not None
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
        image_token_id = getattr(config, "image_token_id", None)
        vocab_size = getattr(config, "vocab_size", 0)
        fallback_id = (
            image_token_id
            if image_token_id is not None and 0 <= image_token_id < vocab_size
            else getattr(config, "engram_pad_token_id", 0)
        )
        embed_tokens = model.get_input_embeddings()
        with torch.no_grad():
            ids = forward_batch.input_ids
            safe_ids = torch.where(
                (ids < 0) | (ids >= engram_host._MM_PAD_SHIFT_VALUE),
                fallback_id,
                ids,
            )
            forward_batch.input_embeds = embed_tokens(safe_ids)

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        super().prepare_forward_batch(forward_batch)
        # Enqueue outside-graph embedding on TPU before attention planning and
        # host Engram staging so the TPU gather overlaps with overlap-token
        # resolution and NumPy row packing.
        self._embed_outside_graph(forward_batch)
