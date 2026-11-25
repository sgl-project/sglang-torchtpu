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
from sglang_tpu.experimental.sampling.sampling_batch_info import is_host_padded
from sglang_tpu.models.deepseek_v41 import engram_host


def _stage_host_engram(coordinator, hasher, input_ids, forward_batch):
    """Pack and stage this rank's Engram rows on the host via ``coordinator``."""
    padded_tokens = input_ids.shape[0]
    device = input_ids.device
    engram_host.move_hasher_to_host(hasher)
    # DSpark verify commits wait here, after the next draft is dispatched; both
    # paths below read or write hasher.history. A verify flushes in stage_verify,
    # where its draft tokens come back in the same read.
    verify = forward_batch.forward_mode.is_target_verify() and not coordinator.started
    if verify:
        dev_verify = coordinator.get_device_verify()
        if dev_verify is not None:
            dev_res = dev_verify.stage_verify(input_ids, forward_batch)
            if dev_res is not None:
                staged, num_cols = dev_res
                forward_batch.tpu_engram_staged = staged
                forward_batch.tpu_engram_local_ids = engram_host.local_row_ids(
                    padded_tokens, len(staged), num_cols, device
                )
                return None
    if not verify:
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


def _local_decoder_rows(forward_batch: ForwardBatch) -> int:
    """Decoder rows of this rank's (folded) extend, as the backend will plan
    them (without DP there is no MLP-sync count)."""
    from sglang_tpu.models.deepseek_v41.decoder_replay import (
        fill_lens_or_ends,
        plan_decoder_rows,
    )

    starts = list(forward_batch.extend_prefix_lens_cpu)
    ext = list(forward_batch.extend_seq_lens_cpu)
    row_floor = forward_batch.encoder_swa_row_floor
    floors = [0] * len(starts)
    if row_floor is not None:
        first = 0
        for b, n in enumerate(ext):
            if n:
                floors[b] = int(row_floor[first])
            first += n
    return plan_decoder_rows(
        starts=starts,
        extend_lens=ext,
        fill_lens=fill_lens_or_ends(
            getattr(forward_batch, "decoder_fill_lens", None), starts, ext
        ),
        floors=floors,
    ).num_rows


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
            model = self.model_runner.model
            embed_mod = (
                model.embed_tokens
                if isinstance(model, DeepseekV4ForCausalLMDSpark)
                else model.get_input_embeddings()
            )
            if (
                ids.device.type == "tpu"
                and getattr(embed_mod, "tp_size", 0) == 1
                and getattr(embed_mod, "output_dtype", None) is None
                and hasattr(embed_mod, "weight")
            ):
                from sglang_tpu.models.deepseek_v41.engram_device import (
                    fused_safe_embed_tpu,
                )

                hc_mult = (
                    int(model.hc_mult)
                    if isinstance(model, DeepseekV4ForCausalLMDSpark)
                    else 1
                )
                forward_batch.input_embeds = fused_safe_embed_tpu(
                    ids, embed_mod.weight, fallback_id=fallback_id, hc_mult=hc_mult
                )
            else:
                safe_ids = torch.where(
                    (ids < 0) | (ids >= engram_host._MM_PAD_SHIFT_VALUE),
                    fallback_id,
                    ids,
                )
                forward_batch.input_embeds = self._embed(safe_ids)

    # ------------------------------------------------------------------ #
    # Decoder SWA bounded replay: one shared decoder row bucket per forward
    # ------------------------------------------------------------------ #

    def _decoder_replay_on(self) -> bool:
        backend = self.model_runner.attn_backend
        return bool(getattr(backend, "decoder_replay", False))

    def _pick_decoder_rows_bucket(self, forward_batch: ForwardBatch) -> None:
        """Every DP rank runs the decoder layers on the same row bucket: the
        smallest holding the busiest rank's decoder rows (MLP-sync counts)."""
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            decoder_rows_bucket,
        )

        if (
            forward_batch.forward_mode != ForwardMode.EXTEND
            or getattr(forward_batch, "decoder_rows_bucket", None) is not None
            or not self._decoder_replay_on()
        ):
            return
        token_bucket = int(forward_batch.input_ids.shape[0])
        rows = getattr(forward_batch, "global_decoder_rows", None)
        if not rows:
            if get_parallel().attn_dp_size > 1:
                # No MLP-sync counts (a dummy or warmup extend): the token bucket
                # holds any rank's rows and is the same on every rank.
                forward_batch.decoder_rows_bucket = token_bucket
                return
            rows = [_local_decoder_rows(forward_batch)]
        forward_batch.decoder_rows_bucket = decoder_rows_bucket(max(rows), token_bucket)

    def _extend_serving_variants(self) -> tuple[str, ...]:
        variants = super()._extend_serving_variants()
        if not self._decoder_replay_on():
            return variants
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            DECODER_ROW_BUCKETS,
        )

        # Every pure extend runs one decoder bucket; "plain" has no replay.
        decoder = ("dec0", *(f"dec{b}" for b in DECODER_ROW_BUCKETS), "decT")
        return tuple(
            v
            for variant in variants
            for v in (decoder if variant == "plain" else (variant,))
        )

    def _make_warmup_inputs(self, *, num_tokens: int, variant: str = "plain", **kwargs):
        if not variant.startswith("dec"):
            return super()._make_warmup_inputs(
                num_tokens=num_tokens, variant=variant, **kwargs
            )
        args, kw = super()._make_warmup_inputs(
            num_tokens=num_tokens, variant="plain", **kwargs
        )
        forward_batch = kw["forward_batch"] if "forward_batch" in kw else args[2]
        bucket = variant[len("dec") :]
        forward_batch.decoder_rows_bucket = (
            num_tokens if bucket == "T" else min(int(bucket), num_tokens)
        )
        # The dummy's rows fill the bucket at most (compile shape only).
        forward_batch.decoder_replay_warmup = True
        return args, kw

    def _direct_compile_specialization_key(
        self, attention_metadata: Any, *args: Any, **kwargs: Any
    ) -> tuple[Any, ...]:
        key = super()._direct_compile_specialization_key(
            attention_metadata, *args, **kwargs
        )
        if not self._decoder_replay_on():
            return key
        # The decoder row bucket (None: no decoder replay in this forward).
        return (*key, getattr(attention_metadata, "tail_rows", None))

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        super().prepare_forward_batch(forward_batch)
        self._pick_decoder_rows_bucket(forward_batch)
        # Enqueue outside-graph embedding on TPU before attention planning and
        # host Engram staging so the TPU gather overlaps with overlap-token
        # resolution and NumPy row packing.
        self._embed_outside_graph(forward_batch)

    def _dp_idle_extend_device(self, forward_batch: ForwardBatch) -> torch.device:
        # An idle DP rank's dummy request spans the busiest rank's live tokens;
        # built on the device, it compiles per extend length on every idle rank.
        # A HOST_PADDED batch pads it on the host and uploads it at the bucket
        # (_upload_host_padded_inputs), as the busy ranks' inputs.
        sampling_info = forward_batch.sampling_info
        if sampling_info is not None and is_host_padded(sampling_info):
            return torch.device("cpu")
        return super()._dp_idle_extend_device(forward_batch)

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
