# SPDX-License-Identifier: Apache-2.0
"""Causal depthwise ShortConv for the TPU per-layer state layout.

Inputs and outputs are token-major [tokens, channels]. History is stored oldest
first in [slots, width - 1, 1, channels], and weight[:, -1] multiplies the current
token. Arithmetic and history use FP32; outputs retain the input dtype before
the model applies its output gate. All loops depend only on compiled shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardMode
    from sglang_tpu.attention_short_conv import TpuShortConvMetadata


def _write_history(
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    history: torch.Tensor,
    active: torch.Tensor,
) -> None:
    # Several padded requests can share slot 0. A batched index_copy_ would
    # have duplicate destinations, so serialize the fixed-size batch rows.
    # Inactive rows preserve their destination, including a dirty null slot.
    history = history.unsqueeze(2).to(conv_state.dtype)
    for row in range(slots.shape[0]):
        index = slots[row : row + 1]
        previous = conv_state.index_select(0, index)
        update = torch.where(
            active[row : row + 1].reshape(1, 1, 1, 1),
            history[row : row + 1],
            previous,
        )
        conv_state.index_copy_(0, index, update)


def short_conv_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    slots: torch.Tensor,
) -> torch.Tensor:
    """One token per request, including padded requests mapped to slot 0."""
    history = conv_state.index_select(0, slots).squeeze(2)
    window = torch.cat((history, x.float().unsqueeze(1)), dim=1)
    compute_weight = weight.float()
    output = torch.zeros_like(x, dtype=torch.float32)
    for k in range(weight.shape[1]):
        output = output + window[:, k] * compute_weight[:, k]
    if bias is not None:
        output = output + bias.float()

    active = slots > 0
    _write_history(conv_state, slots, window[:, 1:], active)
    return torch.where(active[:, None], output, 0).to(x.dtype)


def short_conv_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    conv_state: torch.Tensor,
    slots: torch.Tensor,
    query_start_loc: torch.Tensor,
    has_initial_state: torch.Tensor,
    token_request_indices: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> torch.Tensor:
    """Packed prefill, with zero-length requests and trailing token padding.

    The caller builds token_request_indices once per forward, shared by all
    layers. Invalid tokens use request 0 and are masked before returning.
    """
    history = conv_state.index_select(0, slots).squeeze(2)
    history = torch.where(has_initial_state[:, None, None], history, 0)
    batch_size, history_len, channels = history.shape
    history_rows = batch_size * history_len
    bank = torch.cat((history.reshape(-1, channels), x.float()), dim=0)
    starts = query_start_loc[:-1]
    lengths = query_start_loc[1:] - starts
    positions = torch.arange(x.shape[0], dtype=torch.int64, device=x.device)
    positions = torch.where(valid_tokens, positions, 0)
    token_starts = starts.index_select(0, token_request_indices)
    compute_weight = weight.float()
    output = torch.zeros_like(x, dtype=torch.float32)

    for k in range(weight.shape[1]):
        relative = positions - token_starts - history_len + k
        indices = torch.where(
            relative < 0,
            token_request_indices * history_len + relative + history_len,
            history_rows + positions - history_len + k,
        )
        output = output + bank.index_select(0, indices) * compute_weight[:, k]
    if bias is not None:
        output = output + bias.float()

    tail_positions = (
        lengths[:, None]
        - history_len
        + torch.arange(history_len, dtype=torch.int64, device=x.device)
    )
    request_base = (
        torch.arange(batch_size, dtype=torch.int64, device=x.device)[:, None]
        * history_len
    )
    tail_indices = torch.where(
        tail_positions < 0,
        request_base + tail_positions + history_len,
        history_rows + starts[:, None] + tail_positions,
    )
    next_history = bank.index_select(0, tail_indices.reshape(-1)).reshape(
        batch_size, history_len, channels
    )
    _write_history(conv_state, slots, next_history, (slots > 0) & (lengths > 0))
    active_tokens = valid_tokens & (slots.index_select(0, token_request_indices) > 0)
    return torch.where(active_tokens[:, None], output, 0).to(x.dtype)


def short_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    metadata: TpuShortConvMetadata,
    forward_mode: ForwardMode,
) -> torch.Tensor:
    """Platform kernel entry point used between LFM2's two gates."""
    conv_state = metadata.layer_cache.conv[0]
    if forward_mode.is_decode():
        return short_conv_decode(x, weight, bias, conv_state, metadata.cache_indices)
    return short_conv_prefill(
        x,
        weight,
        bias,
        conv_state,
        metadata.cache_indices,
        metadata.query_start_loc,
        metadata.has_initial_state,
        metadata.token_request_indices,
        metadata.valid_tokens,
    )
