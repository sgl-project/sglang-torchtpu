"""direct_compile inputs for hybrid full + linear attention (GDN/KDA) models.

HybridLinearAttnBackend has no forward_metadata of its own and the linear state
lives outside the KV pool, so both children's metadata and the TpuMambaPool
tensors are lifted here as graph inputs, with the request-pool tensors a
Qwen4-Exp PLE layer reads in its forward (slot mapping, short-conv state,
n-gram context).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import (
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang_tpu.attention_gdn_pallas import GdnTracedMetadata, TpuGDNAttnBackend
from sglang_tpu.mamba_pool import TpuMambaPool


# Request-pool tensors a model reads inside its forward, as (owner, attribute)
# paths from the request pool: the Qwen4-Exp PLE layer maps request rows to
# state slots there, and reads and writes its short-conv state and n-gram
# context (HybridReqToTokenPool.short_conv_pool / ngram_pool).
_REQ_POOL_SIDE_TENSORS = (
    ((), "req_index_to_mamba_index_mapping"),
    (("short_conv_pool",), "conv_state"),
    (("ngram_pool",), "context"),
)


def _req_pool_side_tensors(req_to_token_pool: ReqToTokenPool) -> list[tuple]:
    """(owner, attribute) of each side tensor the request pool holds, only for
    models with PLE side state (others never read them inside the graph)."""
    short_conv = getattr(req_to_token_pool, "short_conv_pool", None)
    ngram = getattr(req_to_token_pool, "ngram_pool", None)
    if not (
        (short_conv is not None and short_conv.enabled)
        or (ngram is not None and ngram.enabled)
    ):
        return []
    owners = []
    for path, attr in _REQ_POOL_SIDE_TENSORS:
        owner = req_to_token_pool
        for name in path:
            owner = getattr(owner, name)
        if isinstance(getattr(owner, attr, None), torch.Tensor):
            owners.append((owner, attr))
    return owners


class HybridLinearKVPoolState:
    """KVPool over the full-attention KV buffers followed by the linear-attention
    state and the request-pool side tensors the forward reads (PLE)."""

    def __init__(
        self,
        *,
        full_kv_pool: MHATokenToKVPool,
        mamba_pool: TpuMambaPool,
        side_tensors: list[tuple] = (),
    ) -> None:
        self._full_kv_pool = full_kv_pool
        self._mamba_pool = mamba_pool
        self._side_tensors = list(side_tensors)

    def get_state(self) -> list[torch.Tensor]:
        return [
            *self._full_kv_pool.get_state(),
            *self._mamba_pool.get_state(),
            *(getattr(owner, attr) for owner, attr in self._side_tensors),
        ]

    def set_state(self, state: list[torch.Tensor]) -> None:
        num_full = len(self._full_kv_pool.get_state())
        num_mamba = len(self._mamba_pool.get_state())
        self._full_kv_pool.set_state(state[:num_full])
        self._mamba_pool.set_state(state[num_full : num_full + num_mamba])
        side = state[num_full + num_mamba :]
        assert len(side) == len(self._side_tensors), (len(side), self._side_tensors)
        for (owner, attr), tensor in zip(self._side_tensors, side):
            setattr(owner, attr, tensor)


def hybrid_linear_kv_pool_state(
    *, kv_pool: HybridLinearKVPool, req_to_token_pool: ReqToTokenPool
) -> HybridLinearKVPoolState:
    # The GDN layers read state through the request pool, so lift that one.
    assert isinstance(req_to_token_pool, HybridReqToTokenPool)
    mamba_pool = req_to_token_pool.mamba_pool
    assert isinstance(mamba_pool, TpuMambaPool), type(mamba_pool)
    full_kv_pool = kv_pool.full_kv_pool
    assert isinstance(full_kv_pool, MHATokenToKVPool), type(full_kv_pool)
    return HybridLinearKVPoolState(
        full_kv_pool=full_kv_pool,
        mamba_pool=mamba_pool,
        side_tensors=_req_pool_side_tensors(req_to_token_pool),
    )


def has_linear_attention(model: nn.Module) -> bool:
    return any(isinstance(module, RadixLinearAttention) for module in model.modules())


def _conv_weight_owners(
    model: nn.Module,
) -> list[tuple[RadixLinearAttention, nn.Module]]:
    """Each RadixLinearAttention with the conv1d module its conv_weights views."""
    modules = dict(model.named_modules())
    owners = []
    for name, module in modules.items():
        if not isinstance(module, RadixLinearAttention):
            continue
        conv1d = modules[name.rpartition(".")[0]].get_submodule("conv1d")
        # conv1d.weight is already the trace placeholder here, so match by size.
        if (
            not isinstance(module.conv_weights, torch.Tensor)
            or module.conv_weights._base is None
            or module.conv_weights.numel() != conv1d.weight.numel()
        ):
            raise NotImplementedError(
                f"direct_compile expects {name}.conv_weights to view conv1d.weight."
            )
        owners.append((module, conv1d))
    return owners


@contextmanager
def bind_traced_conv_weights(model: nn.Module) -> Iterator[None]:
    """Re-derive conv_weights from the traced conv1d weight, so no view enters the graph."""
    owners = _conv_weight_owners(model)
    eager_views = [attn.conv_weights for attn, _ in owners]
    for attn, conv1d in owners:
        attn.conv_weights = conv1d.weight.view(attn.conv_weights.shape)
    try:
        yield
    finally:
        for (attn, _), view in zip(owners, eager_views):
            attn.conv_weights = view


def _linear_backend(backend: HybridLinearAttnBackend) -> TpuGDNAttnBackend:
    linear = backend.linear_attn_backend
    assert isinstance(linear, TpuGDNAttnBackend), type(linear)
    return linear


def traced_attention_metadata(backend: AttentionBackend) -> Any:
    """The attention metadata direct_compile passes as graph inputs."""
    if isinstance(backend, HybridLinearAttnBackend):
        return (
            backend.full_attn_backend.forward_metadata,
            _linear_backend(backend).traced_metadata(),
        )
    return backend.forward_metadata


@contextmanager
def bind_traced_attention_metadata(
    backend: AttentionBackend, metadata: Any, forward_mode: ForwardMode
) -> Iterator[None]:
    """Install traced_attention_metadata's graph inputs while tracing."""
    if not isinstance(backend, HybridLinearAttnBackend):
        with backend.bind_forward_metadata(metadata, forward_mode):
            yield
        return
    full_metadata, linear_metadata = metadata
    assert isinstance(linear_metadata, GdnTracedMetadata)
    with (
        backend.full_attn_backend.bind_forward_metadata(full_metadata, forward_mode),
        _linear_backend(backend).bind_traced_metadata(linear_metadata),
    ):
        yield
