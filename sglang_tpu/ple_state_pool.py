# SPDX-License-Identifier: Apache-2.0
"""TPU Qwen4-Exp PLE short-conv state: one root tensor per PLE layer.

Core's :class:`~sglang.srt.mem_cache.ple_state_pool.ShortConvPool` keeps every
PLE layer in one ``[layers, slots + 1, channels, state_len]`` buffer and hands
out ``conv_state[layer]`` views. Raiden registers each state tensor as a pool
whose leading axis is the block (here, the slot) and refuses a view that folds
the buffer's leading axes: "Tensor's leading (block) dimension does not match
its base storage buffer: the tensor's shape [97, 10240, 9] folds leading
dimensions of the buffer's [1, 97, 10240, 9]". This pool allocates each
layer's ``[slots + 1, channels, state_len]`` state as its own tensor, as
:class:`~sglang_tpu.mamba_pool.TpuMambaPool` does for GDN state.

Speculative decoding's intermediate state is not supported here.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.ple_state_pool import ShortConvPool


class TpuShortConvPool(ShortConvPool):
    def __init__(
        self,
        *,
        size: int,
        state_shape: Optional[Tuple[int, int]],
        layer_ids: List[int],
        dtype: torch.dtype,
        device: str,
        spec_state_size: int = 0,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
    ):
        if speculative_num_draft_tokens is not None and layer_ids:
            raise NotImplementedError(
                "TPU PLE short-conv state does not keep speculative intermediate state"
            )
        self.size = size
        self.device = device
        self.layer_map = {layer_id: i for i, layer_id in enumerate(layer_ids)}
        # Core's fields, unused here: there is no stacked buffer.
        self.conv_state = None
        self.intermediate_conv_state = None
        self.layer_states: List[torch.Tensor] = []
        if not layer_ids or state_shape is None:
            return
        self.layer_states = [
            torch.zeros((size + 1, *state_shape), dtype=dtype, device=device)
            for _ in layer_ids
        ]

    @property
    def enabled(self) -> bool:
        return bool(self.layer_states)

    def layer_cache(self, layer_id: int) -> torch.Tensor:
        return self.layer_states[self.layer_map[layer_id]]

    def layer_intermediate_cache(self, layer_id: int) -> Optional[torch.Tensor]:
        return None

    def clear(self):
        for state in self.layer_states:
            state.zero_()

    def reset_slots(self, indices: torch.Tensor) -> None:
        if indices.numel() == 0:
            return
        for state in self.layer_states:
            state[indices] = 0

    def copy_slots(self, src_index: torch.Tensor, dst_index: torch.Tensor) -> None:
        for state in self.layer_states:
            state[dst_index] = state[src_index]

    def get_cpu_slots(self, indices: torch.Tensor) -> Any:
        if not self.layer_states:
            return None
        # Core's layout: [layers, slots, ...].
        return torch.stack([state[indices].to("cpu") for state in self.layer_states])

    def load_cpu_slots(self, data: Any, indices: torch.Tensor) -> None:
        if not self.layer_states or data is None:
            return
        for state, rows in zip(self.layer_states, data):
            state[indices] = rows.to(state.device)

    def iter_transfer_state_entries(self):
        """Yield each layer's replicated root state tensor for PD transfer."""
        for layer_id, index in self.layer_map.items():
            yield "ple_short_conv", self.layer_states[index], None, layer_id
