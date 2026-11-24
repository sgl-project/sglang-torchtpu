# SPDX-License-Identifier: Apache-2.0
"""The megakernel's operands, built in torch from sglang's ``WanTransformer3DModel``.

:func:`pack_blocks` turns the loaded blocks into the kernel's tiled,
layer-stacked weights (``kernel.pack_kernel_weights``' layout) and frees each
block's projections once copied. :func:`cross_kv` with :func:`cross_keys`, and
:func:`modulation` build the per-request and per-step inputs. Only the packer
needs JAX (for the layouts); the rest is torch.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang_tpu.multimodal_gen.wan_megakernel import reference as ref

# Kernel operand order (kernel.KERNEL_WEIGHTS).
WEIGHT_ORDER = ("wqkv", "wo", "wq2", "wo2", "w1", "w2", "vecs")
# kernel._VEC_ORDER: one f32 row of a layer's vectors (s*: fp8 column scales).
VEC_ORDER = (
    "bqkv",
    "qn",
    "kn",
    "bo",
    "n2w",
    "n2b",
    "bq2",
    "qn2",
    "bo2",
    "b2",
    "b1",
    "sqkv",
    "so",
    "sq2",
    "so2",
    "s2",
    "s1",
)
FP8_MAX, FP8_INV = ref.FP8_MAX, ref.FP8_INV


def config_from_model(model) -> ref.Config:
    blk = model.blocks[0]
    hidden = blk.to_q.weight.shape[1]
    heads = model.num_attention_heads
    return ref.Config(
        layers=len(model.blocks),
        hidden=hidden,
        heads=heads,
        head_dim=hidden // heads,
        ffn=blk.ffn.fc_in.weight.shape[0],
        text_len=model.text_len,
        eps=blk.norm_q.variance_epsilon,
    )


def _qk_order(cfg: ref.Config, device) -> torch.Tensor:
    return torch.as_tensor(ref.qk_order(cfg), device=device)


def _cols(w: torch.Tensor, t: int) -> torch.Tensor:
    """``[out, in]`` -> ``[out/t, in, t]`` (contiguous column tiles of ``w.T``)."""
    out, inp = w.shape
    return w.reshape(out // t, t, inp).permute(0, 2, 1)


def _quantize_out_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[out, in]`` -> e4m3 rows (one per output column of ``w.T``) and their f32
    scales ``max |row| / 448`` (``kernel.quantize_columns`` on the transpose)."""
    wf = w.float()
    amax = wf.abs().amax(dim=1, keepdim=True)
    scale = torch.where(amax > 0, amax * FP8_INV, torch.ones_like(amax))
    q = (wf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.reshape(-1)


def _layer_tensors(
    blk: nn.Module,
    cfg: ref.Config,
    t: int,
    order: torch.Tensor,
    down_cols: int | None = None,
    fp8: bool = False,
) -> dict[str, torch.Tensor]:
    """One block's kernel arrays (views where possible), from its parameters."""
    H = cfg.hidden
    wq = blk.to_q.weight.index_select(0, order)
    wk = blk.to_k.weight.index_select(0, order)
    mats = {
        "wqkv": torch.cat([wq, wk, blk.to_v.weight], dim=0),  # [3H, H] (out, in)
        "wo": blk.to_out.weight,
        "wq2": blk.attn2.to_q.weight,
        "wo2": blk.attn2.to_out.weight,
        "w1": blk.ffn.fc_in.weight,
        "w2": blk.ffn.fc_out.weight,
    }
    scales = {}
    for name, sname in (
        ("wqkv", "sqkv"),
        ("wo", "so"),
        ("wq2", "sq2"),
        ("wo2", "so2"),
        ("w1", "s1"),
        ("w2", "s2"),
    ):
        if fp8:
            mats[name], scales[sname] = _quantize_out_rows(mats[name])
        else:
            scales[sname] = torch.ones(
                mats[name].shape[0], dtype=torch.float32, device=mats[name].device
            )
    return {
        "wqkv": _cols(mats["wqkv"], t),
        "wo": _cols(mats["wo"], t),
        "wq2": _cols(mats["wq2"], t),
        "wo2": _cols(mats["wo2"], t),
        "w1": _cols(mats["w1"], t),
        "w2": (
            _cols(mats["w2"], down_cols)
            if down_cols
            else mats["w2"].t().reshape(cfg.ffn // t, t, H)
        ),
        "vecs": torch.cat(
            [
                torch.cat(
                    [
                        blk.to_q.bias.index_select(0, order),
                        blk.to_k.bias.index_select(0, order),
                        blk.to_v.bias,
                    ]
                ),
                blk.norm_q.weight.index_select(0, order),
                blk.norm_k.weight.index_select(0, order),
                blk.to_out.bias,
                blk.self_attn_residual_norm.norm.weight,
                blk.self_attn_residual_norm.norm.bias,
                blk.attn2.to_q.bias,
                blk.attn2.norm_q.weight,
                blk.attn2.to_out.bias,
                blk.ffn.fc_out.bias,
                blk.ffn.fc_in.bias,
            ]
        )
        .float()
        .reshape(1, -1),
        "svecs": torch.cat(
            [scales[n] for n in ("sqkv", "so", "sq2", "so2", "s2", "s1")]
        )
        .float()
        .reshape(1, -1),
    }


_PROJ_PARAMS = (
    ("to_q", "weight"),
    ("to_k", "weight"),
    ("to_v", "weight"),
    ("to_out", "weight"),
    ("attn2.to_q", "weight"),
    ("attn2.to_out", "weight"),
    ("ffn.fc_in", "weight"),
    ("ffn.fc_out", "weight"),
)


def _sync(device: torch.device) -> None:
    if device.type != "cpu":
        torch.accelerator.synchronize()


def _get(module: nn.Module, path: str) -> nn.Module:
    for part in path.split("."):
        module = getattr(module, part)
    return module


@torch.no_grad()
def pack_blocks(
    model,
    cfg: ref.Config,
    cols: int,
    *,
    free_originals: bool = True,
    stage_on_host: bool = True,
    down_cols: int | None = None,
    fp8: bool = False,
) -> dict[str, torch.Tensor]:
    """Stack the blocks' weights in the kernel's layout on their device.

    ``stage_on_host``: the blocks move to host memory first, the stacks are
    built from there one layer at a time, then the blocks' remaining tensors
    (norms, biases, tables, cross K/V projections) move back. On device that
    needs max(blocks, stacks) instead of their sum.
    """
    blocks = list(model.blocks)
    device = blocks[0].to_q.weight.device
    if stage_on_host and device.type != "cpu":
        for b in blocks:
            b.to("cpu")
        _sync(device)
    host_order = _qk_order(cfg, blocks[0].to_q.weight.device)
    first = _layer_tensors(blocks[0], cfg, cols, host_order, down_cols, fp8)
    first["vecs"] = torch.cat([first["vecs"], first.pop("svecs")], dim=1)
    packed = {
        name: torch.empty(
            (len(blocks), *t.shape),
            dtype=torch.float32
            if name == "vecs"
            else (torch.float8_e4m3fn if fp8 else torch.bfloat16),
            device=device,
        )
        for name, t in first.items()
    }
    for i, b in enumerate(blocks):
        if i == 0:
            layer = first
        else:
            layer = _layer_tensors(b, cfg, cols, host_order, down_cols, fp8)
            layer["vecs"] = torch.cat([layer["vecs"], layer.pop("svecs")], dim=1)
        for name, t in layer.items():
            packed[name][i].copy_(t.contiguous().to(packed[name].dtype))
        del layer
        if free_originals:
            _sync(device)
            for path, attr in _PROJ_PARAMS:
                p = getattr(_get(b, path), attr)
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        if i == 0:
            first = None
    if stage_on_host and device.type != "cpu":
        for b in blocks:
            b.to(device)
        _sync(device)
    return {name: packed[name] for name in WEIGHT_ORDER}


@torch.no_grad()
def modulation_tables(model) -> torch.Tensor:
    """``[L, 6, H]`` f32: every block's ``scale_shift_table``."""
    return torch.stack(
        [b.scale_shift_table.detach().float().reshape(6, -1) for b in model.blocks]
    )


@torch.no_grad()
def modulation(tables: torch.Tensor, timestep_proj: torch.Tensor) -> torch.Tensor:
    """``[L, 8, H]`` f32 rows ``table + temb`` (diffusers: ``table + temb.float()``),
    rows 6-7 zero. ``timestep_proj`` is ``[1, 6, H]``."""
    e = tables + timestep_proj.float().reshape(1, 6, -1)
    return torch.nn.functional.pad(e, (0, 0, 0, 2))


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()


@torch.no_grad()
def cross_kv(
    model, ctx: torch.Tensor, cfg: ref.Config
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(k2, v2)`` ``[L, heads, T, D]`` bf16 (``reference.cross_kv`` per layer,
    head-major) from the embedded text ``ctx [T, H]`` bf16."""
    ks, vs = [], []
    for b in model.blocks:
        k, _ = b.attn2.to_k(ctx)
        k = _rms(k, b.attn2.norm_k.weight, cfg.eps).to(torch.bfloat16)
        v, _ = b.attn2.to_v(ctx)
        ks.append(k)
        vs.append(v.to(torch.bfloat16))

    def heads_major(xs):
        x = torch.stack(xs)  # [L, T, H]
        return (
            x.reshape(len(xs), cfg.text_len, cfg.heads, cfg.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    return heads_major(ks), heads_major(vs)


@torch.no_grad()
def cross_keys(
    k2: torch.Tensor, v2: torch.Tensor, counts: tuple[int, ...]
) -> tuple[torch.Tensor, torch.Tensor]:
    """``kernel.stack``'s ``keys`` (``[1]`` int32) and ``key_weights`` (``[T, 1]``
    f32) for ``k2, v2 [L, heads, T, D]``.

    sglang zero-pads the prompt's T5 embedding to T rows and Wan's cross-attention
    has no mask, so the padding rows' K/V are one row repeated (every layer and
    head). With rows ``n..T-1`` equal to the last row, the softmax over all T rows
    equals the one over the first ``keys`` rows (``keys``: the smallest of
    ``counts`` above ``n``, else T) where each row below ``n`` weighs 1 and the
    ``keys - n`` copies of the repeated row together weigh ``T - n``. The copies
    take the binary digits of ``T - n``, one each and 0 for the rest, when there
    are enough copies: each weighted term ``2^j p`` rounds to bf16 exactly as
    ``p`` does, so the result differs from the all-rows one only by the f32
    rounding of the sums. Otherwise they share it equally. With ``keys = T``
    every weight is 1. Rows are compared exactly, so this holds for any K/V; it
    runs on the device (no host sync).
    """
    T = k2.shape[2]
    dev = k2.device
    idx = torch.arange(T, device=dev)
    differs = (k2 != k2[:, :, -1:]).any(dim=-1) | (v2 != v2[:, :, -1:]).any(dim=-1)
    differs = differs.reshape(-1, T).any(dim=0)
    n = torch.where(differs, idx, -1).max() + 1  # the repeated last row's first row
    sizes = torch.tensor([c for c in counts if c < T] + [T], device=dev)
    keys = torch.where(sizes > n, sizes, T).min()
    m = T - n  # rows the copies stand for
    bit = torch.arange(T.bit_length(), device=dev)
    digit = (m // 2**bit) % 2
    row = n + (digit[None, :] * (bit[None, :] < bit[:, None])).sum(dim=1)
    binary = ((idx[:, None] == row[None, :]) * digit * 2**bit).sum(dim=1)
    copies = torch.where(
        digit.sum() <= keys - n, binary.float(), m.float() / (keys - n).float()
    )
    weights = torch.where((idx < n) | (keys == T), 1.0, copies).to(torch.float32)
    return keys.reshape(1).to(torch.int32), weights.reshape(T, 1)
