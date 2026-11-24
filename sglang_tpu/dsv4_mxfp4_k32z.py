# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 block-32 MXFP4 routed experts -> K32Z operands (lossless).

K32Z is the fused EP expert format (selected by
``SGLANG_TPU_DSV41_EXPERT_FORMAT=k32z``; the default ``k512`` keeps the gmm
preparation in ``dsv4_mxfp4_moe.py``). It keeps the checkpoint's E2M1 codes
and E8M0 exponents instead of requantizing to block 512:

* codes as in the checkpoint, with code 8 (-0) rewritten to 0 (+0, same value);
* per output column, a base ``b`` = the minimum exponent over the column's
  K groups (32 rows each), and per (group, column) a shift ``d = e - b``.

The kernel converts a code to e5m2 in hardware, adds ``4 * d`` to each byte
(``d`` to the exponent field) and applies the column scale ``2^(b - 127)``
once after the K sum, so every nonzero weight is exact. A zero code in a
group with ``d >= 1`` decodes to ``2^(d - 15)`` times the column scale
(2^-14 of the group's smallest nonzero magnitude): that is the K32Z contract.
The operands below are a lossless re-layout of the checkpoint
(:func:`k32z_to_mxfp4` inverts them exactly, up to -0 -> +0).

Operands of E local experts, hidden H, intermediate I (no padding columns):

* ``w13`` uint8 ``[E, H/2, 2I]``: codes packed along K (byte ``[e, j, n]``
  holds K row ``2j`` in its low nibble and ``2j+1`` in its high nibble, the
  gmm layout's packing), gate columns ``[0, I)`` then up ``[I, 2I)``
  (checkpoint ``w1`` then ``w3``).
* ``w2`` uint8 ``[E, I/2, H]``: the same for down.
* ``w13_shift`` uint8 ``[E, G1P, 2I * B1 / 8]`` and ``w2_shift`` uint8
  ``[E, G2P, H * B2 / 8]``: shift fields of ``B1`` (gate/up, default 2) and
  ``B2`` (down, 4) bits. The shift of (group ``g``, column ``n``) sits in
  byte ``(g, (T // F) * 128 + l)`` at bit ``B * (T % F)``, with column tile
  ``T = n // 128``, lane ``l = n % 128`` and ``F = 8 // B`` fields per byte.
  ``G1 = H/32`` and ``G2 = I/32`` group rows, padded to the u8 sublane tile:
  ``G1P = roundup(G1, 32)``, ``G2P = roundup(G2 + 4, 32)``.
* Base bytes ride in ``w2_shift``'s padding rows: row ``G2 + T % 2`` holds the
  down bases of column tile ``T`` at byte columns ``(T // 2) * 128 + l``, row
  ``G2 + 2 + T % 2`` the gate/up bases the same way. The kernel derives the
  f32 column scale as the bit pattern ``b << 23``, so ``1 <= b <= 254``.

The fused EP serving path uses SERVING_GATE_UP_BITS (4) for the gate/up fields:
the V4.1 Flash checkpoint's gate/up shifts reach 6, so 2-bit fields (shifts up
to 3) hold only some layers.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

K512 = "k512"
K32Z = "k32z"
EXPERT_FORMATS = (K512, K32Z)

MXFP4_GROUP = 32  # K rows per checkpoint E8M0 exponent
LANE_TILE = 128  # columns per shift-field tile
U8_SUBLANE_TILE = 32  # u8 VMEM sublane tile; shift tables pad their rows to it
GATE_UP_SHIFT_BITS = 2
# The gate/up field width the fused EP op serves (one width for every rank and
# layer: it sets the gate/up table's shape, which the op's operands share).
SERVING_GATE_UP_BITS = 4
DOWN_SHIFT_BITS = 4
BASE_ROWS = 4  # down base (even/odd tiles) + gate/up base (even/odd tiles)
# The nonzero E2M1 magnitudes have e5m2 exponent fields 14..17; adding d must
# keep the field <= 30 (31 is inf/NaN). Zero codes decode to field d.
E5M2_MAX_SHIFT = 13
EXPERTS_PER_CHUNK = 8
# Set on a FusedMoE whose gmm preparation was deferred to the fused EP
# preparation (K32Z selected); its checkpoint tensors are still in place.
PENDING_ATTR = "_tpu_dsv41_k32z_pending"
# "k32z" on a FusedMoE that holds K32Z operands.
FORMAT_ATTR = "tpu_expert_format"

_E8M0_MIN_SUBNORMAL_BITS = 0x00400000  # f32 bits of 2^-127 (E8M0 exponent 0)


class K32ZGeometry(NamedTuple):
    hidden: int
    inter: int
    gate_up_bits: int
    down_bits: int

    @property
    def g1(self) -> int:
        return self.hidden // MXFP4_GROUP

    @property
    def g2(self) -> int:
        return self.inter // MXFP4_GROUP

    @property
    def g1p(self) -> int:
        return _round_up(self.g1, U8_SUBLANE_TILE)

    @property
    def g2p(self) -> int:
        return _round_up(self.g2 + BASE_ROWS, U8_SUBLANE_TILE)

    @property
    def w13_shift_cols(self) -> int:
        return 2 * self.inter * self.gate_up_bits // 8

    @property
    def w2_shift_cols(self) -> int:
        return self.hidden * self.down_bits // 8

    def max_shift(self, which: int) -> int:
        """Largest shift the fields of matmul ``which`` (1 gate/up, 2 down) hold."""
        bits = self.gate_up_bits if which == 1 else self.down_bits
        return min((1 << bits) - 1, E5M2_MAX_SHIFT)


class K32ZExperts(NamedTuple):
    """One layer's K32Z operands (see the module docstring) and the largest
    shift each matmul's experts use."""

    w13: torch.Tensor
    w13_shift: torch.Tensor
    w2: torch.Tensor
    w2_shift: torch.Tensor
    max_shift: tuple


def _round_up(v: int, m: int) -> int:
    return -(-v // m) * m


def k32z_geometry(
    hidden: int,
    inter: int,
    *,
    gate_up_bits: int = GATE_UP_SHIFT_BITS,
    down_bits: int = DOWN_SHIFT_BITS,
) -> K32ZGeometry:
    """The K32Z layout for an expert shape; raises where the shape cannot take it."""
    if gate_up_bits not in (2, 4) or down_bits != 4:
        raise ValueError(
            f"K32Z shift fields are 2 or 4 bits for gate/up and 4 bits for down; "
            f"got {gate_up_bits} and {down_bits}"
        )
    geo = K32ZGeometry(int(hidden), int(inter), gate_up_bits, down_bits)
    f1, f2 = 8 // gate_up_bits, 8 // down_bits
    problems = []
    if hidden % MXFP4_GROUP or inter % MXFP4_GROUP:
        problems.append(f"hidden and inter must be multiples of {MXFP4_GROUP}")
    if (2 * inter) % (LANE_TILE * f1):
        problems.append(f"2*inter must be a multiple of {LANE_TILE * f1}")
    if hidden % (LANE_TILE * max(f2, 2)):
        problems.append(f"hidden must be a multiple of {LANE_TILE * max(f2, 2)}")
    # Gate/up bases take (2I / 256) * 128 = I byte columns of w2_shift.
    if inter > geo.w2_shift_cols:
        problems.append(
            f"the gate/up base row needs {inter} byte columns, w2_shift has "
            f"{geo.w2_shift_cols}"
        )
    if problems:
        raise ValueError(
            f"K32Z cannot lay out experts of hidden={hidden}, inter={inter}: "
            + "; ".join(problems)
        )
    return geo


# --------------------------------------------------------------------------- #
# The switch.
# --------------------------------------------------------------------------- #
def selected_expert_format() -> str:
    """``SGLANG_TPU_DSV41_EXPERT_FORMAT``, validated: "k512" or "k32z"."""
    from sglang_tpu.envs import envs as tpu_envs

    value = str(tpu_envs.SGLANG_TPU_DSV41_EXPERT_FORMAT.get()).strip().lower()
    if value not in EXPERT_FORMATS:
        raise ValueError(
            f"SGLANG_TPU_DSV41_EXPERT_FORMAT must be one of {EXPERT_FORMATS}, "
            f"got {value!r}"
        )
    return value


def check_expert_format_env() -> None:
    """Startup check: K32Z exists only as the fused EP kernel's format, and its
    conversion runs in the fused EP preparation."""
    from sglang_tpu.envs import envs as tpu_envs

    if (
        selected_expert_format() == K32Z
        and not tpu_envs.SGLANG_TPU_DSV41_FUSED_EP.get()
    ):
        raise ValueError(
            "SGLANG_TPU_DSV41_EXPERT_FORMAT=k32z needs SGLANG_TPU_DSV41_FUSED_EP=1 "
            "(gmm_v2 has no K32Z path)."
        )


def defer_to_fused_ep(layer: torch.nn.Module) -> bool:
    """Called where the gmm preparation would run. With K32Z selected, leave the
    checkpoint experts in place and mark the layer: the fused EP preparation
    converts it to K32Z, or to the gmm layout where the layer keeps gmm_v2
    (hash-routed, NextN). Returns whether the gmm preparation is deferred."""
    if selected_expert_format() != K32Z:
        return False
    check_expert_format_env()
    setattr(layer, PENDING_ATTR, True)
    return True


def is_deferred(layer: torch.nn.Module) -> bool:
    return bool(getattr(layer, PENDING_ATTR, False))


def finish_deferred_as_k512(module: torch.nn.Module) -> int:
    """Run the deferred gmm preparation on every still-pending FusedMoE under
    ``module`` (layers that keep gmm_v2). Returns how many were converted."""
    from sglang_tpu.dsv4_mxfp4_moe import prepare_tpu_dsv4_mxfp4_moe_weights

    done = 0
    for m in module.modules():
        if is_deferred(m):
            prepare_tpu_dsv4_mxfp4_moe_weights(m)
            setattr(m, PENDING_ATTR, False)
            done += 1
    return done


# --------------------------------------------------------------------------- #
# Conversion (torch; runs on the tensors' own device).
# --------------------------------------------------------------------------- #
def e8m0_exponents(scale: torch.Tensor) -> torch.Tensor:
    """E8M0 scales -> int32 exponents ``e`` with ``scale = 2^(e - 127)``.

    Takes the raw E8M0 bytes (uint8 / float8_e8m0fnu) or the f32 powers of two
    SGLang's loader fills the scale parameter with (``param.copy_`` of the
    float8_e8m0fnu tensor; exponent 0 is the f32 subnormal 2^-127). Anything
    else (NaN, a non-power of two) is not an E8M0 scale and is refused, so the
    conversion cannot silently lose a value.
    """
    if scale.dtype in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)):
        e = scale.view(torch.uint8).to(torch.int32)
        if bool((e == 255).any()):
            raise ValueError("E8M0 scale 0xFF (NaN) in the checkpoint experts")
        return e
    if scale.dtype == torch.bfloat16:
        scale = scale.to(torch.float32)
    if scale.dtype != torch.float32:
        raise TypeError(f"expected E8M0 bytes or f32 powers of two, got {scale.dtype}")
    bits = scale.contiguous().view(torch.int32)
    e = bits >> 23
    normal = ((bits & ~0x7F800000) == 0) & (e >= 1) & (e <= 254)
    subnormal = bits == _E8M0_MIN_SUBNORMAL_BITS
    if not bool((normal | subnormal).all()):
        raise ValueError(
            "a checkpoint expert scale is not an E8M0 power of two 2^(e - 127)"
        )
    return torch.where(subnormal, torch.zeros_like(e), e)


def _kpacked_canonical_codes(packed_nk: torch.Tensor) -> torch.Tensor:
    """Checkpoint codes ``[c, N, K/2]`` (int8/uint8, low nibble = even K) ->
    uint8 ``[c, K/2, N]`` packed along K, code 8 (-0) -> 0 (+0)."""
    w = packed_nk.view(torch.uint8) if packed_nk.dtype == torch.int8 else packed_nk
    if w.dtype != torch.uint8:
        raise TypeError(f"expected packed E2M1 codes as uint8/int8, got {w.dtype}")
    w = w.transpose(1, 2).contiguous()
    lo = w & 0x0F
    hi = w & 0xF0
    lo = torch.where(lo == 0x08, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 0x80, torch.zeros_like(hi), hi)
    return lo | hi


def _pack_fields(d: torch.Tensor, bits: int, rows: int) -> torch.Tensor:
    """int32 shifts ``[c, G, N]`` -> uint8 ``[c, rows, N * bits / 8]`` fields
    (tile ``T``'s field in byte column ``(T // F) * 128 + l`` at bit
    ``bits * (T % F)``); rows ``>= G`` are zero."""
    c, g, n = d.shape
    per = 8 // bits
    x = d.reshape(c, g, n // (LANE_TILE * per), per, LANE_TILE)
    packed = x[:, :, :, 0, :]
    for j in range(1, per):
        packed = packed | (x[:, :, :, j, :] << (bits * j))
    out = torch.zeros((c, rows, n // per), dtype=torch.uint8, device=d.device)
    out[:, :g] = packed.reshape(c, g, n // per).to(torch.uint8)
    return out


def _base_rows(b: torch.Tensor) -> torch.Tensor:
    """int32 bases ``[c, N]`` -> uint8 ``[c, 2, N/2]``: row ``T % 2`` holds column
    tile ``T`` at byte columns ``(T // 2) * 128 + l``."""
    c, n = b.shape
    x = b.reshape(c, n // (2 * LANE_TILE), 2, LANE_TILE)
    return x.permute(0, 2, 1, 3).reshape(c, 2, n // 2).to(torch.uint8)


def _shift_and_base(scale: torch.Tensor, max_shift: int, what: str):
    """Checkpoint scales ``[c, N, G]`` -> (shifts int32 ``[c, G, N]``, bases
    int32 ``[c, N]``, largest shift), refusing what the fields cannot hold."""
    e = e8m0_exponents(scale)
    b = e.amin(dim=-1)
    d = (e - b.unsqueeze(-1)).transpose(1, 2).contiguous()
    d_max = int(d.max()) if d.numel() else 0
    if d_max > max_shift:
        raise ValueError(
            f"K32Z {what} shift {d_max} exceeds the {max_shift} its fields hold "
            "(4-bit gate/up fields hold more layers than 2-bit; "
            "SGLANG_TPU_DSV41_EXPERT_FORMAT=k512 keeps the gmm path)"
        )
    if bool((b < 1).any()) or bool((b > 254).any()):
        raise ValueError(
            f"K32Z {what} column base exponent outside [1, 254] (the kernel builds "
            "the column scale as the f32 bits b << 23)"
        )
    return d, b, d_max


def convert_mxfp4_experts_to_k32z(
    w13_packed: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_packed: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    experts_per_chunk: int = EXPERTS_PER_CHUNK,
    gate_up_bits: int = GATE_UP_SHIFT_BITS,
    down_bits: int = DOWN_SHIFT_BITS,
) -> K32ZExperts:
    """Checkpoint block-32 MXFP4 experts -> K32Z operands, on the tensors' own
    device, ``experts_per_chunk`` experts at a time (each output is allocated
    once and filled per chunk, so the transients are one chunk's).

    Inputs as SGLang's loader leaves the FusedMoE parameters: ``w13_packed``
    ``[E, 2I, H/2]`` (w1 rows then w3 rows), ``w13_scale`` ``[E, 2I, H/32]``,
    ``w2_packed`` ``[E, H, I/2]``, ``w2_scale`` ``[E, H, I/32]``; scales as
    E8M0 bytes or f32 powers of two (:func:`e8m0_exponents`).
    """
    num_experts, two_inter, hidden_half = w13_packed.shape
    hidden, inter = 2 * hidden_half, two_inter // 2
    geo = k32z_geometry(hidden, inter, gate_up_bits=gate_up_bits, down_bits=down_bits)
    want = {
        "w13_scale": (num_experts, two_inter, geo.g1),
        "w2_packed": (num_experts, hidden, inter // 2),
        "w2_scale": (num_experts, hidden, geo.g2),
    }
    got = {
        "w13_scale": tuple(w13_scale.shape),
        "w2_packed": tuple(w2_packed.shape),
        "w2_scale": tuple(w2_scale.shape),
    }
    if two_inter % 2 or got != want:
        raise ValueError(f"checkpoint expert shapes {got} do not match {want}")
    if experts_per_chunk < 1:
        raise ValueError(f"experts_per_chunk must be >= 1, got {experts_per_chunk}")

    device = w13_packed.device
    u8 = dict(dtype=torch.uint8, device=device)
    w13 = torch.empty((num_experts, hidden // 2, two_inter), **u8)
    w2 = torch.empty((num_experts, inter // 2, hidden), **u8)
    s13 = torch.empty((num_experts, geo.g1p, geo.w13_shift_cols), **u8)
    s2 = torch.empty((num_experts, geo.g2p, geo.w2_shift_cols), **u8)
    max13 = max2 = 0
    g2 = geo.g2
    for c0 in range(0, num_experts, experts_per_chunk):
        c1 = min(c0 + experts_per_chunk, num_experts)
        w13[c0:c1] = _kpacked_canonical_codes(w13_packed[c0:c1])
        w2[c0:c1] = _kpacked_canonical_codes(w2_packed[c0:c1])
        d1, b1, m1 = _shift_and_base(w13_scale[c0:c1], geo.max_shift(1), "gate/up")
        d2, b2, m2 = _shift_and_base(w2_scale[c0:c1], geo.max_shift(2), "down")
        s13[c0:c1] = _pack_fields(d1, gate_up_bits, geo.g1p)
        chunk = _pack_fields(d2, down_bits, geo.g2p)
        chunk[:, g2 : g2 + 2] = _base_rows(b2)
        chunk[:, g2 + 2 : g2 + 4, :inter] = _base_rows(b1)
        s2[c0:c1] = chunk
        max13, max2 = max(max13, m1), max(max2, m2)
        del d1, b1, d2, b2, chunk
    return K32ZExperts(w13, s13, w2, s2, (max13, max2))


def _unpack_fields(table: torch.Tensor, bits: int, groups: int, n: int) -> torch.Tensor:
    """Inverse of :func:`_pack_fields`: uint8 ``[E, rows, n*bits/8]`` -> int32
    ``[E, groups, n]``."""
    per = 8 // bits
    mask = (1 << bits) - 1
    x = table[:, :groups].to(torch.int32)
    e = x.shape[0]
    x = x.reshape(e, groups, n // (LANE_TILE * per), LANE_TILE)
    parts = [(x >> (bits * j)) & mask for j in range(per)]
    return torch.stack(parts, dim=3).reshape(e, groups, n)


def _unbase_rows(rows: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse of :func:`_base_rows`: ``[E, 2, >= n/2]`` -> int32 ``[E, n]``."""
    e = rows.shape[0]
    x = (
        rows[:, :, : n // 2]
        .to(torch.int32)
        .reshape(e, 2, n // (2 * LANE_TILE), LANE_TILE)
    )
    return x.permute(0, 2, 1, 3).reshape(e, n)


def k32z_to_mxfp4(ops: K32ZExperts, *, gate_up_bits: int = GATE_UP_SHIFT_BITS):
    """The checkpoint experts back from K32Z operands: ``(w13 uint8 [E, 2I, H/2],
    e13 uint8 [E, 2I, H/32], w2 uint8 [E, H, I/2], e2 uint8 [E, H, I/32])``,
    E8M0 exponents as bytes and codes with -0 as +0. Exact: it is the lossless
    witness the CPU tests check the conversion with."""
    num_experts, hidden_half, two_inter = ops.w13.shape
    geo = k32z_geometry(2 * hidden_half, two_inter // 2, gate_up_bits=gate_up_bits)
    g2 = geo.g2
    d1 = _unpack_fields(ops.w13_shift, gate_up_bits, geo.g1, two_inter)
    d2 = _unpack_fields(ops.w2_shift, geo.down_bits, g2, geo.hidden)
    b2 = _unbase_rows(ops.w2_shift[:, g2 : g2 + 2], geo.hidden)
    b1 = _unbase_rows(ops.w2_shift[:, g2 + 2 : g2 + 4], two_inter)
    e13 = (d1 + b1.unsqueeze(1)).transpose(1, 2).to(torch.uint8).contiguous()
    e2 = (d2 + b2.unsqueeze(1)).transpose(1, 2).to(torch.uint8).contiguous()
    w13 = ops.w13.transpose(1, 2).contiguous()
    w2 = ops.w2.transpose(1, 2).contiguous()
    return w13, e13, w2, e2


# --------------------------------------------------------------------------- #
# Layer preparation.
# --------------------------------------------------------------------------- #
def checkpoint_layout_ok(experts: torch.nn.Module, hidden: int, inter: int) -> bool:
    """A FusedMoE still holding the checkpoint's FP4 experts of this geometry."""
    names = ("w13_weight", "w13_weight_scale_inv", "w2_weight", "w2_weight_scale_inv")
    w13, s13, w2, s2 = (getattr(experts, n, None) for n in names)
    if any(t is None for t in (w13, s13, w2, s2)):
        return False
    if w13.dtype not in (torch.uint8, torch.int8) or w2.dtype != w13.dtype:
        return False
    if w13.dim() != 3:
        return False
    e = w13.shape[0]
    return (
        tuple(w13.shape) == (e, 2 * inter, hidden // 2)
        and tuple(s13.shape) == (e, 2 * inter, hidden // MXFP4_GROUP)
        and tuple(w2.shape) == (e, hidden, inter // 2)
        and tuple(s2.shape) == (e, hidden, inter // MXFP4_GROUP)
    )


def _free(param: torch.nn.Parameter) -> None:
    """Point ``param`` at an empty tensor and free its old buffer."""
    old = param.data
    param.data = torch.empty(0, dtype=old.dtype, device=old.device)
    if old.untyped_storage().data_ptr() != param.data.untyped_storage().data_ptr():
        try:
            old.untyped_storage().resize_(0)
        except RuntimeError:  # not resizable (e.g. numpy-backed); dropped below
            pass
    del old


def prepare_tpu_dsv4_mxfp4_k32z_weights(
    layer: torch.nn.Module,
    *,
    experts_per_chunk: int = EXPERTS_PER_CHUNK,
    gate_up_bits: int = GATE_UP_SHIFT_BITS,
) -> K32ZExperts:
    """Replace a FusedMoE's checkpoint FP4 experts by K32Z operands, in place.

    ``w13_weight`` / ``w2_weight`` keep their Parameter objects and take the
    K-packed codes, ``w13_weight_shift`` / ``w2_weight_shift`` hold the shift
    tables and the ``*_weight_scale_inv`` parameters are cleared. The old
    buffers are freed once the new ones exist, as the gmm preparation does, so
    the load holds one layer's checkpoint and K32Z tensors at most.
    """
    from torch.nn import Parameter

    ops = convert_mxfp4_experts_to_k32z(
        layer.w13_weight.data,
        layer.w13_weight_scale_inv.data,
        layer.w2_weight.data,
        layer.w2_weight_scale_inv.data,
        experts_per_chunk=experts_per_chunk,
        gate_up_bits=gate_up_bits,
    )
    for name in (
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        "w2_weight_scale_inv",
    ):
        _free(getattr(layer, name))
    layer.w13_weight.data = ops.w13
    layer.w2_weight.data = ops.w2
    layer.w13_weight_scale_inv = None
    layer.w2_weight_scale_inv = None
    layer.w13_weight_shift = Parameter(ops.w13_shift, requires_grad=False)
    layer.w2_weight_shift = Parameter(ops.w2_shift, requires_grad=False)
    setattr(layer, FORMAT_ATTR, K32Z)
    setattr(layer, PENDING_ATTR, False)
    layer.k32z_max_shift = ops.max_shift
    return ops
