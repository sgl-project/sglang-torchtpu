# SPDX-License-Identifier: Apache-2.0
"""The megakernel's operands, built in torch from sglang's MiniMax-H3 DiT.

:func:`pack_blocks` turns the loaded blocks' projections into the kernel's
tiled, layer-stacked weights (``reference.pack_weights``'s layout) on their
device and frees each original once it is copied, staging the blocks through
host memory so the stacks fit where the blocks were. The per-step builders
turn a forward's AdaLN input, combined indices and RoPE cache into
``tables``, ``onehot``, ``cos`` and ``sin``; the tables come from each
block's own resident ``adaln_proj`` on every call (sharded: each rank's
blocks' projections, then one all-gather of the shares). Only
:func:`config_from_arch` and :func:`tile_widths` import JAX (for
``reference.Config`` and the tiling module).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref

# A token's AdaLN row is a one-hot of this many lanes times a table, so the
# lookup is one exact bf16 matmul (reference.TABLE_LANES).
TABLE_LANES = 128
# Table rows are stored padded to a bf16 sublane tile (reference._TABLE_ROW_TILE).
TABLE_ROW_TILE = 16
# The packed weights in the kernel's operand order (kernel._WEIGHT_ORDER).
WEIGHT_ORDER = ("norm1", "norm2", "q_norm", "k_norm", "wqkv", "wo", "w13", "w2")
_PROJECTIONS = ("wqkv", "wo", "w13", "w2")
_NORMS = ("norm1", "norm2", "q_norm", "k_norm")


def config_from_arch(arch, layers: int) -> ref.Config:
    """The kernel's ``Config`` for sglang's ``MiniMaxH3DiTArchConfig``."""
    from sglang_tpu.multimodal_gen.h3_megakernel import reference as ref

    return ref.Config(
        hidden=arch.hidden_size,
        heads=arch.num_attention_heads,
        head_dim=arch.attention_head_dim,
        ffn=arch.ffn_hidden_size,
        layers=layers,
        rope_dims=6 * arch.rope_inv_freq_len,
        eps=arch.norm_eps,
        qk_eps=arch.qk_norm_eps,
    )


def tile_widths(config: ref.Config) -> dict[str, int]:
    """``reference.pack_weights``'s tile widths, cut down to divide ``config``
    (H3's own widths keep the defaults: 2 heads, 768, 256 and 256 columns)."""
    from sglang_tpu.multimodal_gen.h3_megakernel import tiling as tl

    return {
        "qkv_heads": 2 if config.heads % 2 == 0 else 1,
        "out_cols": tl.divisor(config.hidden, 768, 128),
        "ffn_cols": tl.divisor(config.ffn, 256, 128),
        "down_cols": tl.divisor(config.hidden, 256, 128),
    }


def _projection_param(block: nn.Module, name: str) -> nn.Parameter:
    return {
        "wqkv": block.attn.qkv_proj.weight,
        "wo": block.attn.out_proj.weight,
        "w13": block.mlp.fc1.weight,
        "w2": block.mlp.fc2.weight,
    }[name]


def _packed_projection(
    block: nn.Module, name: str, config: ref.Config, tiles: dict[str, int]
) -> torch.Tensor:
    """One block's projection in the kernel's tiled layout (a view)."""
    hidden, ffn, dim, heads = config.hidden, config.ffn, config.head_dim, config.heads
    weight = _projection_param(block, name)
    if name == "wqkv":
        # sglang keeps [Q | K | V] (every head's q rows, then k, then v); a
        # tile holds g heads, each one's q | k | v columns together.
        group = tiles["qkv_heads"]
        qkv = weight.view(3, heads // group, group, dim, hidden)
        return qkv.permute(1, 4, 2, 0, 3).reshape(heads // group, hidden, -1)
    if name == "wo":
        cols = tiles["out_cols"]
        return weight.view(hidden // cols, cols, config.width).permute(0, 2, 1)
    if name == "w13":
        # fc1 rows are [gate | up]; a tile holds ffn_cols gate columns, then
        # the same ffn_cols up columns.
        cols = tiles["ffn_cols"]
        fc1 = weight.view(2, ffn // cols, cols, hidden)
        return fc1.permute(1, 3, 0, 2).reshape(ffn // cols, hidden, 2 * cols)
    cols = tiles["down_cols"]
    return weight.view(hidden // cols, cols, ffn).permute(0, 2, 1)


def _norm_weight(block: nn.Module, name: str) -> torch.Tensor:
    if name in ("norm1", "norm2"):
        return getattr(block, name).weight
    return getattr(block.attn, name).weight


def _sync(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize(device)


def device_bytes(blocks) -> int:
    """Bytes of ``blocks``' parameters and buffers off the CPU."""
    return sum(
        tensor.numel() * tensor.element_size()
        for block in blocks
        for tensor in (*block.parameters(), *block.buffers())
        if tensor.device.type != "cpu"
    )


@torch.no_grad()
def pack_blocks(
    blocks,
    config: ref.Config,
    *,
    free_originals: bool = True,
    stage_on_host: bool = False,
) -> dict[str, torch.Tensor]:
    """Stack ``blocks``' weights in the kernel's layout, on their device.

    One weight kind at a time: its stacked buffer is allocated, then filled
    layer by layer, each layer's original freed (``free_originals``) as soon
    as it is copied.

    ``stage_on_host`` first moves every block tensor to host memory, then
    allocates the stacks, then moves the blocks' remaining tensors (the AdaLN
    projections and norms) back. Each block's projections sit between its
    other tensors, so freeing them one by one leaves holes no stack fits in:
    without staging the device needs the blocks plus all four stacks (H3:
    about 65 + 38.6 GB a device). Staged, it needs the larger of the blocks
    and the stacks plus the AdaLN projections (about 65 GB), and the host
    holds one copy of the blocks meanwhile.
    """
    blocks = list(blocks)
    tiles = tile_widths(config)
    device = blocks[0].attn.qkv_proj.weight.device
    if stage_on_host:
        for block in blocks:
            block.to("cpu")
        _sync(device)
    packed = {
        name: torch.stack(
            [_norm_weight(block, name).detach().reshape(1, -1) for block in blocks]
        )
        .to(torch.bfloat16)
        .to(device)
        for name in _NORMS
    }
    for name in _PROJECTIONS:
        shape = _packed_projection(blocks[0], name, config, tiles).shape
        stacked = torch.empty(
            (len(blocks), *shape), dtype=torch.bfloat16, device=device
        )
        for index, block in enumerate(blocks):
            source = _packed_projection(block, name, config, tiles)
            if source.device != device:
                # One contiguous host buffer a layer, uploaded in one copy.
                source = source.contiguous()
            stacked[index].copy_(source)
            if free_originals:
                _sync(device)
                param = _projection_param(block, name)
                # Same device as the parameter: staged ones are on the host.
                param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        packed[name] = stacked
    if stage_on_host:
        for block in blocks:
            block.to(device)
        _sync(device)
    return packed


def adaln_tables(adaln_projs, adaln_input: torch.Tensor) -> torch.Tensor:
    """``[L, 6, R, H]`` bf16: table ``k`` row ``3 m + modality`` of each block.

    Row ``r`` of table ``k`` is the ``k``-th of the block's ``adaln_proj``
    outputs (six ``[3 M, H]`` chunks) at row ``r``, the row sglang's
    ``combined_indices`` select; rows ``3 M`` and up are zero, to a multiple
    of 16. Each call runs every block's projection on this forward's
    ``adaln_input``: nothing carries over between steps or requests.
    """
    tables = torch.stack([torch.stack(proj(adaln_input)) for proj in adaln_projs]).to(
        torch.bfloat16
    )
    rows = tables.shape[2]
    pad = -rows % TABLE_ROW_TILE
    if rows + pad > TABLE_LANES:
        raise ValueError(
            f"MiniMax H3 megakernel takes at most {TABLE_LANES} AdaLN rows "
            f"(3 x unique timesteps); got {rows}"
        )
    return nn.functional.pad(tables, (0, 0, 0, pad))


def adaln_block_range(layers: int, devices: int, rank: int) -> tuple[int, int, int]:
    """``(first, end, per_rank)``: the blocks ``[first, end)`` whose AdaLN
    projections rank ``rank`` of ``devices`` holds under sharding.

    Every rank but the last ones holds ``per_rank = ceil(layers / devices)``
    consecutive blocks in rank order (H3's 50: 25 at 2 ranks; 13, 13, 13, 11
    at 4; 7 each and 1 at 8); a rank past the last block holds none.
    """
    per_rank = -(-layers // devices)
    first = min(rank * per_rank, layers)
    return first, min(first + per_rank, layers), per_rank


def adaln_share(
    adaln_projs,
    adaln_input: torch.Tensor,
    *,
    per_rank: int,
    table_shape: tuple[int, int, int],
) -> torch.Tensor:
    """This rank's ``[per_rank, 6, R, H]`` bf16 share of the tables.

    ``adaln_projs`` are the projections of this rank's blocks
    (:func:`adaln_block_range`). Their tables are :func:`adaln_tables` of
    them: per block the same projection on the same input as the replicated
    path, so the same bits. Zero tables fill them to ``per_rank``, so every
    rank's share has one shape. ``table_shape`` (the projections'
    ``expand_ratio``, ``modality_num`` and ``hidden_size``) shapes the zero
    tables of a rank that holds no block.
    """
    if adaln_projs:
        local = adaln_tables(adaln_projs, adaln_input)
    else:
        tables, modalities, hidden = table_shape
        rows = modalities * adaln_input.shape[0]
        pad = -rows % TABLE_ROW_TILE
        if rows + pad > TABLE_LANES:
            raise ValueError(
                f"MiniMax H3 megakernel takes at most {TABLE_LANES} AdaLN rows "
                f"(3 x unique timesteps); got {rows}"
            )
        local = torch.zeros(
            (0, tables, rows + pad, hidden),
            dtype=torch.bfloat16,
            device=adaln_input.device,
        )
    missing = per_rank - local.shape[0]
    if missing:
        local = torch.cat((local, local.new_zeros((missing, *local.shape[1:]))))
    return local


def sharded_adaln_tables(
    adaln_projs,
    adaln_input: torch.Tensor,
    *,
    per_rank: int,
    layers: int,
    table_shape: tuple[int, int, int],
    gather,
) -> torch.Tensor:
    """:func:`adaln_tables` of all ``layers`` blocks, from this rank's share.

    Every rank passes ``gather`` its :func:`adaln_share`; ``gather``
    concatenates every rank's along dim 0 in rank order. The ranks' blocks are
    consecutive, so its first ``layers`` tables are the blocks' in order and
    the zero tables are what follows. Each block's table comes from the rank
    that holds it, so this equals the replicated path's tables bit for bit
    when ``adaln_input`` is the same on every rank (sglang's H3 forward
    computes it on every rank from the same timesteps and weights).
    """
    local = adaln_share(
        adaln_projs, adaln_input, per_rank=per_rank, table_shape=table_shape
    )
    return gather(local)[:layers]


def table_onehot(combined_indices: torch.Tensor) -> torch.Tensor:
    """``[T]`` AdaLN row per token -> ``[T, 128]`` bf16 one-hot."""
    lanes = torch.arange(TABLE_LANES, device=combined_indices.device)
    return (combined_indices.view(-1, 1) == lanes).to(torch.bfloat16)


def rope_tables(
    cos_sin_cache: torch.Tensor, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """sglang's ``[T, R]`` cos | sin cache -> the kernel's ``[T, D]`` tables.

    sglang's non-CUDA RoPE rotates the first ``R`` head dims with
    ``cat(cos, cos)`` and ``cat(sin, sin)`` and passes the rest through:
    over the whole head that is cos ``(cos, cos, 1)`` and sin ``(sin, sin,
    0)``, sglang's own bf16 values.
    """
    rows, width = cos_sin_cache.shape
    cos, sin = cos_sin_cache.to(torch.bfloat16).split(width // 2, dim=-1)
    ones = cos.new_ones(rows, head_dim - width)
    return (
        torch.cat((cos, cos, ones), dim=-1),
        torch.cat((sin, sin, torch.zeros_like(ones)), dim=-1),
    )
