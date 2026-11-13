# SPDX-License-Identifier: Apache-2.0
"""Load-time weight permutations and fp8 normalization of NVFP4 expert slabs."""

import jax
import jax.numpy as jnp

from sglang_tpu.muse_megakernel.mk.consts import LANE, SMALL


def perm_256(t, axis: int):
    """Permute each 256-slice along ``axis`` from (16, 16) to transposed (16, 16)."""
    axis = axis % t.ndim
    d = t.shape[axis]
    assert d % 256 == 0, f"perm_256 requires multiple of 256, got {d}"
    dt = t.dtype
    work = t.astype(jnp.float8_e4m3fn) if dt == jnp.float4_e2m1fn else t
    sh = work.shape
    work = work.reshape(*sh[:axis], d // 256, 16, 16, *sh[axis + 1 :])
    work = jnp.swapaxes(work, axis + 1, axis + 2).reshape(sh)
    return work.astype(dt) if dt == jnp.float4_e2m1fn else work


def permute_weights_kperm(weights: tuple, kinds: tuple, tp: int, hel: int, he: int):
    """Permute wpool, wpool_s, and pre_expert_w on fp4 layers so expert x arrives in kperm order."""
    L = len(kinds)
    if not any(k == "fp4" for k in kinds):
        return weights
    fp4_mask = jnp.array([k == "fp4" for k in kinds], dtype=jnp.bool_)
    sm, wqkvg, wo, wpool, wrouter, wunpool = weights

    def _perm_sm(s):
        s4 = s.reshape(tp, L, *s.shape[1:])
        wps = s4[:, :, SMALL["wpool_s"], :hel]
        s4 = s4.at[:, :, SMALL["wpool_s"], :hel].set(
            jnp.where(fp4_mask[None, :, None], perm_256(wps, -1), wps)
        )
        pew = s4[:, :, SMALL["pre_expert_w"], :he]
        s4 = s4.at[:, :, SMALL["pre_expert_w"], :he].set(
            jnp.where(fp4_mask[None, :, None], perm_256(pew, -1), pew)
        )
        return s4.reshape(s.shape)

    def _perm_wp(wp):
        wp4 = wp.reshape(tp, L, *wp.shape[1:])
        wp4 = jnp.where(fp4_mask[None, :, None, None], perm_256(wp4, -1), wp4)
        return wp4.reshape(wp.shape)

    sm_p = jax.block_until_ready(jax.jit(_perm_sm, out_shardings=sm.sharding)(sm))
    wp_p = jax.block_until_ready(jax.jit(_perm_wp, out_shardings=wpool.sharding)(wpool))
    return (sm_p, wqkvg, wo, wp_p, wrouter, wunpool)


def permute_expert_slab_kperm(slab: tuple):
    """Permute an fp4 expert slab (w13, w13s, w2, w2s) for zero-shuffle kperm dequant."""
    w13, w13s, w2, w2s = slab
    w13_p = jax.block_until_ready(
        jax.jit(lambda x: perm_256(perm_256(x, -2), -1), out_shardings=w13.sharding)(
            w13
        )
    )
    w13s_p = jax.block_until_ready(
        jax.jit(lambda x: perm_256(x, -1), out_shardings=w13s.sharding)(w13s)
    )
    w2_p = jax.block_until_ready(
        jax.jit(lambda x: perm_256(x, -2), out_shardings=w2.sharding)(w2)
    )
    return (w13_p, w13s_p, w2_p, w2s)


# ex_kperm >= 3: largest dequantized fp4 weight after the block scales are normalized (e4m3
# keeps its 3 mantissa bits down to 2^-6, so the weights stay in its normal range)
WFP8_TARGET_MAX = 32.0
_FP4_MAX = 6.0


def expert_wfp8_globals(slab: tuple, target_max: float = WFP8_TARGET_MAX):
    """Per-(rank, expert) (gate, up, down) globals of an fp4 slab: [rows] f32 each, such that

    block scale / global keeps every dequantized weight at most target_max.
    """
    _, w13s, _, w2s = slab
    il = w13s.shape[-1] // 2

    def f(s13, s2):
        a13 = jnp.abs(s13.astype(jnp.float32)) * _FP4_MAX
        a2 = jnp.abs(s2.astype(jnp.float32)) * _FP4_MAX
        # Snap per-expert globals to exact powers of 2 so s / global preserves
        # every e4m3 block scale's 3-bit mantissa without rounding error.
        half_target = target_max * 0.5
        return tuple(
            jnp.exp2(jnp.ceil(jnp.log2(jnp.maximum(x, 1e-30) / half_target)))
            for x in (
                jnp.max(a13[..., :il], axis=(1, 2)),
                jnp.max(a13[..., il:], axis=(1, 2)),
                jnp.max(a2, axis=(1, 2)),
            )
        )

    shd = jax.sharding.NamedSharding(
        w13s.sharding.mesh, jax.sharding.PartitionSpec(w13s.sharding.spec[0])
    )
    return jax.block_until_ready(jax.jit(f, out_shardings=(shd, shd, shd))(w13s, w2s))


def normalize_expert_slab_wfp8(slab: tuple, gs: tuple):
    """Divides an fp4 slab's block scales by its (gate, up, down) globals (ex_kperm >= 3)."""
    w13, w13s, w2, w2s = slab
    il = w13s.shape[-1] // 2

    def f(s13, s2, gg, gu, gd):
        inv13 = jnp.concatenate(
            [
                jnp.broadcast_to((1.0 / gg)[:, None, None], s13[..., :il].shape),
                jnp.broadcast_to((1.0 / gu)[:, None, None], s13[..., il:].shape),
            ],
            axis=-1,
        )
        s13n = (s13.astype(jnp.float32) * inv13).astype(s13.dtype)
        s2n = (s2.astype(jnp.float32) / gd[:, None, None]).astype(s2.dtype)
        return s13n, s2n

    s13n, s2n = jax.block_until_ready(
        jax.jit(f, out_shardings=(w13s.sharding, w2s.sharding))(w13s, w2s, *gs)
    )
    return (w13, s13n, w2, s2n)


def apply_expert_gs_to_smalls(smalls, gs: tuple, kinds: tuple, tp: int, n_experts: int):
    """Writes the (gate, up, down) globals into the fp4 layers' smalls rows expert_gs*."""
    L = len(kinds)
    E = n_experts
    Ep = -(-E // LANE) * LANE
    fp4 = jnp.array([k == "fp4" for k in kinds], dtype=jnp.bool_)

    def f(sm, gg, gu, gd):
        s4 = sm.reshape(tp, L, *sm.shape[1:])
        r_up, r_dn = SMALL["expert_gs"], SMALL["expert_gs_down"]
        bc = lambda g: jnp.broadcast_to(g.reshape(tp, 1, E), (tp, L, E))
        up = s4[:, :, r_up, :]
        new_up = up.at[:, :, 0:E].set(bc(gg)).at[:, :, Ep : Ep + E].set(bc(gu))
        dn = s4[:, :, r_dn, :]
        new_dn = dn.at[:, :, 0:E].set(bc(gd))
        s4 = s4.at[:, :, r_up, :].set(jnp.where(fp4[None, :, None], new_up, up))
        s4 = s4.at[:, :, r_dn, :].set(jnp.where(fp4[None, :, None], new_dn, dn))
        return s4.reshape(sm.shape)

    return jax.block_until_ready(jax.jit(f, out_shardings=smalls.sharding)(smalls, *gs))
