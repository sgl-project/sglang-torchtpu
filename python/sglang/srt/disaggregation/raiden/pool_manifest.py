"""Raiden pool manifest for the fused TPU KV cache.

Raiden addresses staged bytes as ``(pool, block)``. A *storage* is one whole
tensor handed to ``KVCacheManager``; a *pool* is an array of equally strided
blocks inside one storage, carrying an opaque ``tag`` and a map of the live
byte regions inside a block. The fused TPU KV pool keeps one
``[num_blocks, page_size, kv_heads, kv_packing, padded_head_dim]`` tensor per
layer, so each layer is one storage holding exactly one pool, and all layers
share the ``kv`` tag: a transfer that names the tag moves every layer.

The manifest also carries the *layout fingerprint*, the identity the controller
compares before it emits a plan. It covers only the properties that must be
equal on both sides of a transfer -- the per-rank kv-head count deliberately
stays out, since resharding exists precisely because it differs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Sequence

import msgspec

TAG_KV = "kv"

LAYOUT_FINGERPRINT_SCHEMA = "sglang-fused-tpu-kv-v1"


class PoolManifest(msgspec.Struct, frozen=True):
    """Pool table for one rank's KV storages, plus its layout identity."""

    pools: tuple[Any, ...]
    layout_fingerprint: str
    page_tokens: int
    head_pair_bytes: int

    def pool_dicts(self) -> List[Dict[str, Any]]:
        return [pool.to_dict() for pool in self.pools]


def build_pool_manifest(
    kv_caches: Sequence[Any],
    *,
    total_kv_heads: int,
) -> PoolManifest:
    """Describe ``kv_caches`` (one fused tensor per layer) as raiden pools.

    ``kv_caches`` must be the same list, in the same order, that is handed to
    ``KVCacheManager``: pool ``storage_index`` is that position.
    """
    from tpu_raiden.api.torch.pool_layout import PoolSpec, RegionSpec

    if not kv_caches:
        raise ValueError("Raiden pool manifest requires at least one KV tensor")

    pools = []
    page_tokens = 0
    head_pair_bytes = 0
    dtype_tag = ""
    for storage_index, tensor in enumerate(kv_caches):
        shape = tensor.shape
        if len(shape) != 5:
            raise ValueError(
                "Raiden pool manifest requires the fused KV layout "
                f"[num_blocks, page_size, kv_heads, kv_packing, head_dim]; got "
                f"{tuple(int(d) for d in shape)}"
            )
        num_blocks, page_size, kv_heads, kv_packing, head_dim = (int(d) for d in shape)
        itemsize = tensor.element_size()
        # One kv-head's K and V are a single contiguous run, so this granule is
        # independent of the per-rank head count and matches across ranks.
        layer_head_pair_bytes = kv_packing * head_dim * itemsize
        token_bytes = kv_heads * layer_head_pair_bytes
        block_bytes = page_size * token_bytes
        layer_dtype_tag = str(tensor.dtype)

        if storage_index == 0:
            page_tokens = page_size
            head_pair_bytes = layer_head_pair_bytes
            dtype_tag = layer_dtype_tag
        elif (
            page_size != page_tokens
            or layer_head_pair_bytes != head_pair_bytes
            or layer_dtype_tag != dtype_tag
        ):
            raise ValueError(
                "Raiden pool manifest requires a uniform KV geometry across "
                f"layers; layer {storage_index} differs from layer 0"
            )

        pools.append(
            PoolSpec(
                tag=TAG_KV,
                storage_index=storage_index,
                base_offset_bytes=0,
                block_stride_bytes=block_bytes,
                num_blocks=num_blocks,
                regions=(
                    RegionSpec(
                        name="token",
                        offset_bytes=0,
                        stride_bytes=token_bytes,
                        unit_bytes=token_bytes,
                        num_units=page_size,
                    ),
                ),
                dtype_tag=layer_dtype_tag,
            )
        )

    fingerprint = layout_fingerprint(
        dtype_tag=dtype_tag,
        page_tokens=page_tokens,
        head_pair_bytes=head_pair_bytes,
        num_layers=len(pools),
        total_kv_heads=total_kv_heads,
    )
    return PoolManifest(
        pools=tuple(pools),
        layout_fingerprint=fingerprint,
        page_tokens=page_tokens,
        head_pair_bytes=head_pair_bytes,
    )


def layout_fingerprint(
    *,
    dtype_tag: str,
    page_tokens: int,
    head_pair_bytes: int,
    num_layers: int,
    total_kv_heads: int,
) -> str:
    """Stable identity for a KV layout that two peers must share to transfer.

    Deliberately excludes the per-rank kv-head count and block count: those
    differ between a prefill and a decode rank of a resharding pair.
    """
    payload = {
        "schema": LAYOUT_FINGERPRINT_SCHEMA,
        "dtype": dtype_tag,
        "page_tokens": int(page_tokens),
        "head_pair_bytes": int(head_pair_bytes),
        "num_layers": int(num_layers),
        "total_kv_heads": int(total_kv_heads),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
