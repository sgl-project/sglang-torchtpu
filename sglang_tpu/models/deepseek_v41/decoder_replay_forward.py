# SPDX-License-Identifier: Apache-2.0
"""Model forward for DeepSeek-V4.1 TPU prompt-tail replay."""

import torch


def configure_decoder_replay(original, model, config, *args, **kwargs):
    from sglang.srt.runtime_context import get_platform

    original(model, config, *args, **kwargs)
    if (
        get_platform().is_tpu
        and config.model_type == "deepseek_v41"
        and model.late_layer_start is not None
    ):
        if not model.hc_pre_from_prev_sublayer:
            raise ValueError("TPU decoder replay requires shifted mHC")
        # Reuse the source-layer validation in the constructor. The TPU hook
        # owns replay, so ordinary forwards bypass the upstream tail layout.
        model.tpu_decoder_replay_start = model.late_layer_start
        model.late_layer_start = None


@torch.no_grad()
def forward_decoder_replay(
    original,
    model,
    input_ids,
    positions,
    forward_batch,
    input_embeds,
    pp_proxy_tensors=None,
):
    from sglang.srt.model_executor.forward_context import get_attn_backend

    start = getattr(model, "tpu_decoder_replay_start", None)
    if start is not None and forward_batch.forward_mode.is_extend_without_speculative():
        backend = get_attn_backend()
        metadata = (
            backend.tail_forward_metadata if backend.decoder_replay_enabled else None
        )
        if metadata is not None:
            return _forward_prefill(
                model,
                backend,
                metadata.late_layer_tail,
                start,
                input_ids,
                positions,
                forward_batch,
                input_embeds,
            )
    return original(
        model, input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
    )


def _gather_ids(input_ids, forward_batch):
    from sglang.srt.layers.dp_attention import (
        dp_gather_replicate,
        get_global_dp_buffer_len,
    )

    gathered = input_ids.new_empty((get_global_dp_buffer_len(), 1))
    dp_gather_replicate(gathered, input_ids[:, None].clone(), forward_batch)
    return gathered.squeeze(-1)


def _run_layers(
    model,
    layer_ids,
    state,
    positions,
    input_ids,
    input_ids_global,
    hash_ids,
    forward_batch,
    aux,
    *,
    capture_rows=None,
):
    from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean
    from sglang.srt.eplb.expert_distribution import (
        get_global_expert_distribution_recorder,
    )

    recorder = get_global_expert_distribution_recorder()
    for layer_id in layer_ids:
        layer = model.layers[layer_id]
        engram = layer.engram
        if engram is not None:
            state = state.with_residual(
                engram(
                    state.residual,
                    hash_ids[:, engram.layer_hash_index],
                    forward_batch,
                    input_ids=input_ids,
                    cp_all_tokens=False,
                )
            )
        if aux is not None and layer_id in model.dspark_layers_to_capture:
            captured = state.residual
            if capture_rows is not None:
                captured = capture_rows(captured)
            mhc_stream_mean(captured, out=aux.slot(captured[:, 0]))
            del captured
        with recorder.with_current_layer(layer_id):
            state = layer.forward_hc_pre_from_prev(
                positions=positions,
                state=state,
                input_ids=input_ids,
                forward_batch=forward_batch,
                input_ids_global=input_ids_global,
            )
    return state


def _forward_prefill(
    model, backend, tail, start, input_ids, positions, forward_batch, input_embeds
):
    from sglang.kernels.ops.layernorm.mhc import hc_broadcast, mhc_combine
    from sglang.srt.layers.aux_hidden_states import AuxHiddenStatePacker
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.runtime_context import get_forward, get_parallel

    assert model.pp_group.world_size == 1 and not get_forward().sp_active
    for attr in ("freqs_cis_c4", "freqs_cis_c128"):
        if hasattr(forward_batch, attr):
            delattr(forward_batch, attr)

    capture_layers = model.dspark_layers_to_capture
    aux = (
        AuxHiddenStatePacker(len(capture_layers))
        if capture_layers is not None and not tail.skip_decoder
        else None
    )
    state = mhc.HcState(
        hc_broadcast(
            model.embed_tokens(input_ids) if input_embeds is None else input_embeds,
            model.hc_mult,
        )
    )
    gather_ids = get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none()
    input_ids_global = (
        _gather_ids(input_ids, forward_batch)
        if gather_ids
        else getattr(forward_batch, "input_ids_global", input_ids)
    )
    hash_ids = (
        model.engram_hasher(input_ids, forward_batch)
        if model.engram_hasher is not None
        else None
    )
    state = _run_layers(
        model,
        range(model.start_layer, start),
        state,
        positions,
        input_ids,
        input_ids_global,
        hash_ids,
        forward_batch,
        aux,
        capture_rows=tail.rows,
    )

    source = model.layers[start]
    backend.write_tail_source_kv(
        layer=source.self_attn,
        x=mhc.combine(source.attn_hc, state),
        positions=positions,
        forward_batch=forward_batch,
    )
    num_tokens = input_ids.shape[0]
    if tail.skip_decoder:
        residual = state.residual
        empty = (
            residual.new_zeros((num_tokens, model.hidden_size)),
            residual.new_zeros((num_tokens, model.hc_mult * model.hidden_size)),
        )
        if capture_layers is not None:
            return empty, residual.new_zeros(
                (num_tokens, len(capture_layers) * model.hidden_size)
            )
        return empty

    saved = backend.enter_late_layer_tail(forward_batch)
    try:
        state = state.take_rows(tail.rows)
        input_ids = tail.rows(input_ids)
        # Re-gather in the compact DP layout installed by the backend.
        input_ids_global = (
            _gather_ids(input_ids, forward_batch)
            if gather_ids
            else tail.rows(input_ids_global)
        )
        state = _run_layers(
            model,
            range(start, model.end_layer),
            state,
            tail.positions,
            input_ids,
            input_ids_global,
            None if hash_ids is None else tail.rows(hash_ids),
            forward_batch,
            aux,
        )
        state = state.materialized(model.layers[model.end_layer - 1].hc_cfg)
    finally:
        backend.exit_late_layer_tail(saved, forward_batch)

    pre_head = state.residual.flatten(1)
    hidden = model.norm(mhc_combine(pre_head, state.pre, model.hc_mult, pre_head.dtype))
    output = (
        tail.restore_rows(hidden),
        tail.restore_rows(pre_head),
    )
    if aux is not None:
        return output, tail.restore_rows(aux.finalize())
    return output
