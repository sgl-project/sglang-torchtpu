"""Unit tests for Step (c) 2-phase batched RMW write_decode_kv_merged_jax (kv_pack_dense2=True)."""

import numpy as np
import pytest


def test_step_c_write_decode_kv_merged_dense2_interpret_parity_and_aot():
    """Verify 2-phase batched RMW write_decode_kv_merged_jax for kv_pack_dense2=True."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import write_kv_only_rpa_v3
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        write_decode_kv_merged_jax,
    )

    T = 64
    max_seqs = 64
    phys_rows = 256
    page_size = phys_rows * 2  # 512
    pps = 4
    num_blocks = max_seqs * pps + 8
    rng = np.random.default_rng(20261219)
    pg_np = rng.permutation(num_blocks)[: max_seqs * pps].astype(np.int32)

    # Mix q_len=1 (even & odd across pages), q_len>1 (even/odd kv_start & page-boundary),
    # q_len=0 mid-batch, and kv_len < q_len inactive guard.
    q_lens_np = np.zeros((max_seqs,), dtype=np.int32)
    kv_lens_np = np.zeros((max_seqs,), dtype=np.int32)
    for i in range(50):
        q_lens_np[i] = 1
        kv_lens_np[i] = 18 + (i & 1) + 512 * (i % 3)
    q_lens_np[50], kv_lens_np[50] = 2, 512  # even start (510..511)
    q_lens_np[51], kv_lens_np[51] = 2, 513  # odd start, crosses 512 boundary (511..512)
    q_lens_np[52], kv_lens_np[52] = 2, 21   # odd start ending on odd row (19..20)
    q_lens_np[53], kv_lens_np[53] = 3, 514  # odd start, 3 tokens across 512 (511..513)
    q_lens_np[54], kv_lens_np[54] = 3, 515  # even start, 3 tokens across 512 (512..514)
    q_lens_np[57], kv_lens_np[57] = 1, 0    # inactive (kv_len < q_len)
    cu_q_np = np.concatenate([[0], np.cumsum(q_lens_np)]).astype(np.int32)
    assert int(cu_q_np[-1]) <= T
    n_act = 58

    # Poison cache with 0x7F (FP8 NaN) and pre-populate historical tokens cleanly.
    hist_u8 = np.full((num_blocks, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
    hist_d2 = jax.lax.bitcast_convert_type(jnp.asarray(hist_u8, jnp.uint8), jnp.float8_e4m3fn)
    old_lens_np = np.maximum(kv_lens_np - q_lens_np, 0).astype(np.int32)
    tot_old = int(old_lens_np.sum())
    old_u8 = np.zeros((tot_old, 1, 4, 128), dtype=np.uint8)
    old_u8[:, 0, :2, :] = np.asarray(
        jax.lax.bitcast_convert_type(
            jnp.asarray((rng.standard_normal((tot_old, 2, 128)) * 0.25).astype(np.float32), jnp.float8_e4m3fn),
            jnp.uint8,
        )
    )
    old_kv = jax.lax.bitcast_convert_type(jnp.asarray(old_u8, jnp.uint8), jnp.float8_e4m3fn)
    cu_kv_j = jnp.arange(max_seqs + 1, dtype=jnp.int32) * (pps * page_size)
    dist_j = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)
    pg_j = jnp.asarray(pg_np, dtype=jnp.int32)

    hist_d2 = write_kv_only_rpa_v3(
        old_kv,
        hist_d2,
        jnp.asarray(old_lens_np, dtype=jnp.int32),
        pg_j,
        jnp.asarray(np.concatenate([[0], np.cumsum(old_lens_np)]).astype(np.int32)),
        cu_kv_j,
        dist_j,
        interpret=True,
        kv_pack_dense2=True,
    )

    kv_new_u8 = np.full((T, 1, 4, 128), 0x3C, dtype=np.uint8)
    kv_new_u8[: int(cu_q_np[-1]), 0, :2, :] = np.asarray(
        jax.lax.bitcast_convert_type(
            jnp.asarray((rng.standard_normal((int(cu_q_np[-1]), 2, 128)) * 0.35).astype(np.float32), jnp.float8_e4m3fn),
            jnp.uint8,
        )
    )
    kv_new_j = jax.lax.bitcast_convert_type(jnp.asarray(kv_new_u8, jnp.uint8), jnp.float8_e4m3fn)
    kvl_j = jnp.asarray(kv_lens_np, dtype=jnp.int32)
    cu_q_j = jnp.asarray(cu_q_np, dtype=jnp.int32)

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    ref_out = write_kv_only_rpa_v3(
        kv_new_j,
        hist_d2,
        kvl_j,
        pg_j,
        cu_q_j,
        cu_kv_j,
        dist_j,
        interpret=True,
        kv_pack_dense2=True,
    )
    fast_out = write_decode_kv_merged_jax(
        kv_new_j,
        hist_d2,
        kvl_j,
        pg_j,
        cu_q_j,
        cu_kv_j,
        dist_j,
        kv_pack_dense2=True,
        interpret=ip_race,
    )

    r_u8 = np.asarray(ref_out).view(np.uint8)
    f_u8 = np.asarray(fast_out).view(np.uint8)
    assert np.array_equal(r_u8 == 0x7F, f_u8 == 0x7F), "dense2 0x7F NaN mask mismatch"
    assert np.array_equal(r_u8, f_u8), f"dense2 bit mismatch: {np.sum(r_u8 != f_u8)} bytes"

    # Verify fallback preserves correctness on odd T (e.g. T=4) where _load_new_kv_1row_u16 cannot 8-tile.
    fb_ref = write_kv_only_rpa_v3(
        kv_new_j[:4],
        hist_d2,
        kvl_j[:4],
        pg_j,
        cu_q_j[:5],
        cu_kv_j[:5],
        jnp.asarray([4, 4, 4], dtype=jnp.int32),
        interpret=True,
        kv_pack_dense2=True,
    )
    fb_out = write_decode_kv_merged_jax(
        kv_new_j[:4],
        hist_d2,
        kvl_j[:4],
        pg_j,
        cu_q_j[:5],
        cu_kv_j[:5],
        jnp.asarray([4, 4, 4], dtype=jnp.int32),
        kv_pack_dense2=True,
        interpret=True,
    )
    assert np.array_equal(np.asarray(fb_ref).view(np.uint8), np.asarray(fb_out).view(np.uint8))

    # TPU7x 2x2x1 AOT compile + HLO zero-copy / kernel name verification.
    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )
    abstract = tuple(jax.ShapeDtypeStruct(x.shape, x.dtype) for x in (kv_new_j, hist_d2, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j))

    def _aot_write_d2(kv_n, kv_c, kvl, pg, cq, ckv, d):
        return write_decode_kv_merged_jax(
            kv_n, kv_c, kvl, pg, cq, ckv, d, kv_pack_dense2=True, interpret=False
        )

    with mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
        lowered = jax.jit(
            jax.shard_map(
                _aot_write_d2,
                mesh=mesh_aot,
                in_specs=(P(), P(), P(), P(), P(), P(), P()),
                out_specs=P(),
                check_vma=False,
            ),
            donate_argnums=(1,),
        ).lower(*abstract)
    hlo_txt = lowered.as_text()
    assert "StepC_write_kv_only_d2-p_512" in hlo_txt
    compiled = lowered.compile()
    assert compiled is not None
    compiled_hlo = compiled.as_text()
    assert "input_output_alias={ {}: (1, {}, may-alias) }" in compiled_hlo
    kv_copy_lines = [
        ln for ln in compiled_hlo.splitlines()
        if "f8e4m3fn[264,256,1,4,128]" in ln and "= copy(" in ln
    ]
    assert not kv_copy_lines, f"Unexpected kv_cache copy in dense2 HLO: {kv_copy_lines}"


def test_step_c_write_decode_kv_merged_dense2_tpu7x_bd4_unchanged():
    """Sanity check StepC_write_kv_only_bd4 naming and TPU7x AOT alongside dense2."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        write_decode_kv_merged_jax,
    )

    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )
    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    abstract = (
        s((64, 1, 4, 128), jnp.float8_e4m3fn),
        s((128, 128, 1, 4, 128), jnp.float8_e4m3fn),
        s((64,), jnp.int32),
        s((256,), jnp.int32),
        s((65,), jnp.int32),
        s((65,), jnp.int32),
        s((3,), jnp.int32),
    )
    with mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
        lowered = jax.jit(
            jax.shard_map(
                lambda kv_n, kv_c, kvl, pg, cq, ckv, d: write_decode_kv_merged_jax(
                    kv_n, kv_c, kvl, pg, cq, ckv, d, kv_pack_bd4=True, interpret=False
                ),
                mesh=mesh_aot,
                in_specs=(P(), P(), P(), P(), P(), P(), P()),
                out_specs=P(),
                check_vma=False,
            ),
            donate_argnums=(1,),
        ).lower(*abstract)
    assert "StepC_write_kv_only_bd4-p_512" in lowered.as_text()
    assert lowered.compile() is not None


def _make_poisoned_hist_and_kv(
    rng: np.random.Generator,
    *,
    num_blocks: int,
    phys_rows: int,
    page_size: int,
    pps: int,
    lens_np: np.ndarray,
    q_lens_np: np.ndarray,
    T: int,
    n_act: int,
    dense2: bool,
    bd4: bool,
):
    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        write_kv_only_rpa_v3,
    )

    max_seqs = lens_np.shape[0]
    pg_np = rng.permutation(num_blocks)[: max_seqs * pps].astype(np.int32)
    hist_u8 = np.full((num_blocks, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
    hist_kv = jax.lax.bitcast_convert_type(
        jnp.asarray(hist_u8, jnp.uint8), jnp.float8_e4m3fn
    )
    old_lens_np = np.maximum(lens_np - q_lens_np, 0).astype(np.int32)
    tot_old = int(old_lens_np.sum())
    cu_kv_j = jnp.arange(max_seqs + 1, dtype=jnp.int32) * (pps * page_size)
    dist_j = jnp.asarray([n_act, n_act, n_act], dtype=jnp.int32)
    pg_j = jnp.asarray(pg_np, dtype=jnp.int32)

    if tot_old > 0:
        old_u8 = np.zeros((tot_old, 1, 4, 128), dtype=np.uint8)
        old_u8[:, 0, :2, :] = np.asarray(
            jax.lax.bitcast_convert_type(
                jnp.asarray(
                    (rng.standard_normal((tot_old, 2, 128)) * 0.25).astype(np.float32),
                    jnp.float8_e4m3fn,
                ),
                jnp.uint8,
            )
        )
        old_kv = jax.lax.bitcast_convert_type(
            jnp.asarray(old_u8, jnp.uint8), jnp.float8_e4m3fn
        )
        hist_kv = write_kv_only_rpa_v3(
            old_kv,
            hist_kv,
            jnp.asarray(old_lens_np, dtype=jnp.int32),
            pg_j,
            jnp.asarray(
                np.concatenate([[0], np.cumsum(old_lens_np)]).astype(np.int32)
            ),
            cu_kv_j,
            dist_j,
            interpret=True,
            kv_pack_dense2=dense2,
            kv_pack_bd4=bd4,
        )

    cu_q_np = np.concatenate([[0], np.cumsum(q_lens_np)]).astype(np.int32)
    kv_new_u8 = np.full((T, 1, 4, 128), 0x3C, dtype=np.uint8)
    if int(cu_q_np[-1]) > 0:
        kv_new_u8[: int(cu_q_np[-1]), 0, :2, :] = np.asarray(
            jax.lax.bitcast_convert_type(
                jnp.asarray(
                    (
                        rng.standard_normal((int(cu_q_np[-1]), 2, 128)) * 0.35
                    ).astype(np.float32),
                    jnp.float8_e4m3fn,
                ),
                jnp.uint8,
            )
        )
    kv_new_j = jax.lax.bitcast_convert_type(
        jnp.asarray(kv_new_u8, jnp.uint8), jnp.float8_e4m3fn
    )
    return (
        hist_kv,
        kv_new_j,
        jnp.asarray(lens_np, dtype=jnp.int32),
        pg_j,
        jnp.asarray(cu_q_np, dtype=jnp.int32),
        cu_kv_j,
        dist_j,
    )


def test_wave10_step_c_rpa_plan_full_attn_unlock():
    """Verify non-shared Full-Attn unlocks Step (c) for BD4 & DENSE2 while shared/512B stay gated."""
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        resolve_step_c_rpa_plan,
    )

    mode = "w8,s4,batchfull,g4k,foldcol,shared16"
    cfg_bd4, one_bd4, _, grp_bd4, ok_bd4 = resolve_step_c_rpa_plan(
        mode,
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=False,
    )
    assert ok_bd4 is True and one_bd4 is False and grp_bd4 is True
    assert cfg_bd4.loop_kind == "ring4k" and cfg_bd4.ring_group_size == 4 and cfg_bd4.shared_group_size == 0

    cfg_d2, one_d2, _, grp_d2, ok_d2 = resolve_step_c_rpa_plan(
        mode,
        sliding_window=None,
        pages_per_seq=40,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=False,
    )
    assert ok_d2 is True and one_d2 is False and grp_d2 is True
    assert cfg_d2.loop_kind == "ring4k" and cfg_d2.ring_group_size == 4 and cfg_d2.shared_group_size == 0

    _, one_ring4k, _, _, ok_ring4k = resolve_step_c_rpa_plan(
        mode,
        sliding_window=None,
        pages_per_seq=48,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=False,
    )
    assert ok_ring4k is True and one_ring4k is False

    cfg_512, one_512, slot_512, grp_512, ok_512 = resolve_step_c_rpa_plan(
        mode,
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=512,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=False,
    )
    assert ok_512 is True and one_512 is False and grp_512 is False
    assert (cfg_512.loop_kind, cfg_512.num_slots, cfg_512.chunk_sz, slot_512) == (
        "ring4k",
        3,
        4096,
        8,
    )

    _, _, _, _, ok_too_big = resolve_step_c_rpa_plan(
        "w2,stream0b",
        sliding_window=None,
        pages_per_seq=128,
        phys_page_rows=512,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=False,
    )
    assert ok_too_big is False

    _, _, _, _, ok_lc_full = resolve_step_c_rpa_plan(
        mode,
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=True,
    )
    assert ok_lc_full is False


@pytest.mark.parametrize("fmt", ["dense2", "bd4"])
def test_wave10_write_decode_kv_merged_b256_interpret_and_aot(fmt: str):
    """Verify write_decode_kv_merged_jax supports B=256 with race-detect interpret & 0-copy HLO."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        write_kv_only_rpa_v3,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        write_decode_kv_merged_jax,
    )

    bd4 = fmt == "bd4"
    dense2 = fmt == "dense2"
    phys_rows = 128 if bd4 else 256
    page_size = 512
    B, pps = 256, 3
    nb = B * pps + 4
    rng = np.random.default_rng(202621 + (1 if bd4 else 0))
    q_lens = np.ones((B,), dtype=np.int32)
    q_lens[100] = 0
    q_lens[210] = 0
    lens = np.array([15 + (i & 3) + 512 * (i % 2) for i in range(B)], dtype=np.int32)
    lens[210] = 0

    hist, kv_new, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j = _make_poisoned_hist_and_kv(
        rng,
        num_blocks=nb,
        phys_rows=phys_rows,
        page_size=page_size,
        pps=pps,
        lens_np=lens,
        q_lens_np=q_lens,
        T=B,
        n_act=B,
        dense2=dense2,
        bd4=bd4,
    )
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    ref_out = write_kv_only_rpa_v3(
        kv_new, hist, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j,
        interpret=True, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
    )
    fast_out = write_decode_kv_merged_jax(
        kv_new, hist, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j,
        kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_race,
    )
    r_u8 = np.asarray(ref_out).view(np.uint8)
    f_u8 = np.asarray(fast_out).view(np.uint8)
    assert np.array_equal(r_u8 == 0x7F, f_u8 == 0x7F)
    assert np.array_equal(r_u8, f_u8)

    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )
    abstract = tuple(
        jax.ShapeDtypeStruct(x.shape, x.dtype)
        for x in (kv_new, hist, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j)
    )
    with mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
        compiled = (
            jax.jit(
                jax.shard_map(
                    lambda a, b, c, d, e, f, g: write_decode_kv_merged_jax(
                        a, b, c, d, e, f, g, kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=False
                    ),
                    mesh=mesh_aot,
                    in_specs=(P(), P(), P(), P(), P(), P(), P()),
                    out_specs=P(),
                    check_vma=False,
                ),
                donate_argnums=(1,),
            )
            .lower(*abstract)
            .compile()
        )
    hlo = compiled.as_text()
    assert "input_output_alias={ {}: (1, {}, may-alias) }" in hlo
    assert not [ln for ln in hlo.splitlines() if f"f8e4m3fn[{nb},{phys_rows},1,4,128]" in ln and "= copy(" in ln]


@pytest.mark.parametrize("fmt", ["512B", "dense2", "bd4"])
def test_wave10_write_decode_kv_multi_layer_6d_interpret_and_aot(fmt: str):
    """Verify batched 6D bundle multi-layer KV write parity (including q_len>1 fallback) & HLO alias."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        write_kv_only_rpa_v3,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        write_decode_kv_multi_layer_jax,
    )

    bd4 = fmt == "bd4"
    dense2 = fmt == "dense2"
    phys_rows = 128 if bd4 else (256 if dense2 else 512)
    page_size = 512
    n_pool_layers, L = 7, 5
    layer_ids = jnp.asarray([0, 2, 3, 5, 6], dtype=jnp.int32)
    T, max_seqs, pps = 32, 32, 3
    nb = max_seqs * pps + 4
    rng = np.random.default_rng(202622 + len(fmt))

    q_lens = np.ones((max_seqs,), dtype=np.int32)
    q_lens[11] = 0
    q_lens[27] = 1
    lens = np.array([17 + (i & 3) + 512 * (i % 2) for i in range(max_seqs)], dtype=np.int32)
    lens[27] = 0  # inactive

    pg_np = rng.permutation(nb)[: max_seqs * pps].astype(np.int32)
    pg_j = jnp.asarray(pg_np, dtype=jnp.int32)
    cu_kv_j = jnp.arange(max_seqs + 1, dtype=jnp.int32) * (pps * page_size)
    dist_j = jnp.asarray([max_seqs, max_seqs, max_seqs], dtype=jnp.int32)

    bundle_u8 = np.full((nb, n_pool_layers, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
    bundle_j = jax.lax.bitcast_convert_type(jnp.asarray(bundle_u8, jnp.uint8), jnp.float8_e4m3fn)
    cu_q_np = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
    old_lens_np = np.maximum(lens - q_lens, 0).astype(np.int32)
    tot_old = int(old_lens_np.sum())
    cu_old_j = jnp.asarray(np.concatenate([[0], np.cumsum(old_lens_np)]).astype(np.int32))
    kvl_j = jnp.asarray(lens, dtype=jnp.int32)
    cu_q_j = jnp.asarray(cu_q_np, dtype=jnp.int32)

    kv_layers_list = []
    ref_bundle = bundle_j
    init_bundle = bundle_j
    for lid in [0, 2, 3, 5, 6]:
        old_u8 = np.zeros((tot_old, 1, 4, 128), dtype=np.uint8)
        old_u8[:, 0, :2, :] = np.asarray(
            jax.lax.bitcast_convert_type(
                jnp.asarray((rng.standard_normal((tot_old, 2, 128)) * 0.25).astype(np.float32), jnp.float8_e4m3fn),
                jnp.uint8,
            )
        )
        old_kv = jax.lax.bitcast_convert_type(jnp.asarray(old_u8, jnp.uint8), jnp.float8_e4m3fn)
        flat_init = init_bundle.reshape((nb * n_pool_layers, phys_rows, 1, 4, 128))
        flat_init = write_kv_only_rpa_v3(
            old_kv, flat_init, jnp.asarray(old_lens_np, dtype=jnp.int32),
            pg_j * n_pool_layers + lid, cu_old_j, cu_kv_j, dist_j,
            interpret=True, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
        )
        init_bundle = flat_init.reshape(bundle_j.shape)

        kv_n_u8 = np.full((T, 1, 4, 128), 0x3C, dtype=np.uint8)
        kv_n_u8[: int(cu_q_np[-1]), 0, :2, :] = np.asarray(
            jax.lax.bitcast_convert_type(
                jnp.asarray((rng.standard_normal((int(cu_q_np[-1]), 2, 128)) * 0.35).astype(np.float32), jnp.float8_e4m3fn),
                jnp.uint8,
            )
        )
        kv_n = jax.lax.bitcast_convert_type(jnp.asarray(kv_n_u8, jnp.uint8), jnp.float8_e4m3fn)
        kv_layers_list.append(kv_n)
        ref_bundle = init_bundle if lid == 0 else ref_bundle

    ref_flat = init_bundle.reshape((nb * n_pool_layers, phys_rows, 1, 4, 128))
    for i, lid in enumerate([0, 2, 3, 5, 6]):
        ref_flat = write_kv_only_rpa_v3(
            kv_layers_list[i], ref_flat, kvl_j, pg_j * n_pool_layers + lid,
            cu_q_j, cu_kv_j, dist_j, interpret=True, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
        )
    ref_out = ref_flat.reshape(bundle_j.shape)
    kv_layers_j = jnp.stack(kv_layers_list, axis=0)

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    fast_out = write_decode_kv_multi_layer_jax(
        kv_layers_j, init_bundle, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j,
        layer_ids=layer_ids, kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_race,
    )
    r_u8 = np.asarray(ref_out).view(np.uint8)
    f_u8 = np.asarray(fast_out).view(np.uint8)
    assert np.array_equal(r_u8 == 0x7F, f_u8 == 0x7F)
    assert np.array_equal(r_u8, f_u8)

    # Also verify q_len > 1 dynamic fallback executes correctly.
    q_lens_mq = np.zeros((max_seqs,), dtype=np.int32)
    q_lens_mq[0], q_lens_mq[1] = 2, 3
    lens_mq = np.zeros((max_seqs,), dtype=np.int32)
    lens_mq[0], lens_mq[1] = 513, 21
    cu_q_mq = jnp.asarray(np.concatenate([[0], np.cumsum(q_lens_mq)]).astype(np.int32))
    kvl_mq = jnp.asarray(lens_mq, dtype=jnp.int32)
    ref_mq_flat = init_bundle.reshape((nb * n_pool_layers, phys_rows, 1, 4, 128))
    for i, lid in enumerate([0, 2, 3, 5, 6]):
        ref_mq_flat = write_kv_only_rpa_v3(
            kv_layers_list[i], ref_mq_flat, kvl_mq, pg_j * n_pool_layers + lid,
            cu_q_mq, cu_kv_j, dist_j, interpret=True, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
        )
    fast_mq = write_decode_kv_multi_layer_jax(
        kv_layers_j, init_bundle, kvl_mq, pg_j, cu_q_mq, cu_kv_j, dist_j,
        layer_ids=layer_ids, kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=True,
    )
    assert np.array_equal(
        np.asarray(ref_mq_flat.reshape(bundle_j.shape)).view(np.uint8),
        np.asarray(fast_mq).view(np.uint8),
    )

    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )
    abstract = tuple(
        jax.ShapeDtypeStruct(x.shape, x.dtype)
        for x in (kv_layers_j, init_bundle, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j, layer_ids)
    )
    with mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
        lowered = jax.jit(
            jax.shard_map(
                lambda kvl_l, kc, kvl_, pg_, cq_, ckv_, dt_, lids_: write_decode_kv_multi_layer_jax(
                    kvl_l, kc, kvl_, pg_, cq_, ckv_, dt_, layer_ids=lids_,
                    kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=False,
                ),
                mesh=mesh_aot,
                in_specs=(P(),) * 8,
                out_specs=P(),
                check_vma=False,
            ),
            donate_argnums=(1,),
        ).lower(*abstract)
    assert "StepC_write_kv_multi_" in lowered.as_text()
    hlo = lowered.compile().as_text()
    assert "input_output_alias={ {}: (1, {}, may-alias) }" in hlo
    assert not [ln for ln in hlo.splitlines() if f"f8e4m3fn[{nb},{n_pool_layers},{phys_rows},1,4,128]" in ln and "= copy(" in ln]


@pytest.mark.parametrize("fmt", ["512B", "dense2", "bd4"])
def test_wave10_write_decode_kv_multi_layer_5d_interpret_and_aot(fmt: str):
    """Verify batched 5D unified multi-layer KV write parity & zero-copy HLO aliasing (M=4 & M=8)."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas.ragged_paged_attention_v3 import (
        write_kv_only_rpa_v3,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        write_decode_kv_multi_layer_jax,
    )

    bd4 = fmt == "bd4"
    dense2 = fmt == "dense2"
    phys_rows = 128 if bd4 else (256 if dense2 else 512)
    page_size = 512
    M = 4
    T, max_seqs, pps = 32, 32, 3
    nb = max_seqs * pps + 4
    rng = np.random.default_rng(202623 + len(fmt))

    q_lens = np.ones((max_seqs,), dtype=np.int32)
    q_lens[7] = 0
    lens = np.array([25 + (i & 3) + 512 * (i % 2) for i in range(max_seqs)], dtype=np.int32)
    hist_base, _, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j = _make_poisoned_hist_and_kv(
        rng,
        num_blocks=nb,
        phys_rows=phys_rows,
        page_size=page_size,
        pps=pps,
        lens_np=lens,
        q_lens_np=q_lens,
        T=T,
        n_act=max_seqs,
        dense2=dense2,
        bd4=bd4,
    )
    caches_in = tuple(hist_base.copy() for _ in range(M))
    kv_list = []
    for _ in range(M):
        kv_u8 = np.full((T, 1, 4, 128), 0x3C, dtype=np.uint8)
        kv_u8[: int(cu_q_j[-1]), 0, :2, :] = np.asarray(
            jax.lax.bitcast_convert_type(
                jnp.asarray((rng.standard_normal((int(cu_q_j[-1]), 2, 128)) * 0.35).astype(np.float32), jnp.float8_e4m3fn),
                jnp.uint8,
            )
        )
        kv_list.append(jax.lax.bitcast_convert_type(jnp.asarray(kv_u8, jnp.uint8), jnp.float8_e4m3fn))
    kv_layers_j = jnp.stack(kv_list, axis=0)
    refs_out = tuple(
        write_kv_only_rpa_v3(
            kv_list[i], caches_in[i], kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j,
            interpret=True, kv_pack_dense2=dense2, kv_pack_bd4=bd4,
        )
        for i in range(M)
    )
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    fast_out = write_decode_kv_multi_layer_jax(
        kv_layers_j, caches_in, kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j,
        layer_ids=None, kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_race,
    )
    assert isinstance(fast_out, tuple) and len(fast_out) == M
    for i in range(M):
        r_u8 = np.asarray(refs_out[i]).view(np.uint8)
        f_u8 = np.asarray(fast_out[i]).view(np.uint8)
        assert np.array_equal(r_u8 == 0x7F, f_u8 == 0x7F)
        assert np.array_equal(r_u8, f_u8)

    # AOT compile M=8 chunk on TPU7x verifying every output buffer is aliased zero-copy.
    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
    mesh_aot = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )
    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    M8 = 8
    c_sh = (nb, phys_rows, 1, 4, 128)
    abs_args = (
        (s((M8, T, 1, 4, 128), jnp.float8_e4m3fn),)
        + (s(c_sh, jnp.float8_e4m3fn),) * M8
        + tuple(s(x.shape, x.dtype) for x in (kvl_j, pg_j, cu_q_j, cu_kv_j, dist_j))
    )

    def _aot_5d(*args):
        return write_decode_kv_multi_layer_jax(
            args[0],
            args[1 : 1 + M8],
            args[1 + M8],
            args[2 + M8],
            args[3 + M8],
            args[4 + M8],
            args[5 + M8],
            layer_ids=None,
            kv_pack_dense2=dense2,
            kv_pack_bd4=bd4,
            interpret=False,
        )

    with mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
        compiled = (
            jax.jit(
                jax.shard_map(
                    _aot_5d,
                    mesh=mesh_aot,
                    in_specs=(P(),) * len(abs_args),
                    out_specs=(P(),) * M8,
                    check_vma=False,
                ),
                donate_argnums=tuple(range(1, 1 + M8)),
            )
            .lower(*abs_args)
            .compile()
        )
    hlo = compiled.as_text()
    for out_i in range(M8):
        assert f"{{{out_i}}}: ({1 + out_i}, {{}}, may-alias)" in hlo
    assert not [ln for ln in hlo.splitlines() if f"f8e4m3fn[{nb},{phys_rows},1,4,128]" in ln and "= copy(" in ln]


def test_wave10_flush_pending_step_c_kv_writes_grouping():
    """Verify flush_pending_step_c_kv_writes groups by pool/page_indices/pack flags and chunks 5D <=8."""
    from unittest import mock

    import torch

    from sglang_tpu.muse_megakernel.decode import b1_moe_ops

    # Check custom op registration for both 6D and 5D.
    op_6d = b1_moe_ops.get_step_c_kv_write_multi_6d_op(kv_pack_bd4=True)
    op_5d = b1_moe_ops.get_step_c_kv_write_multi_5d_op(4, kv_pack_dense2=True)
    assert callable(op_6d) and callable(op_5d)

    kvl = torch.zeros((32,), dtype=torch.int32)
    cq = torch.zeros((33,), dtype=torch.int32)
    ckv = torch.zeros((33,), dtype=torch.int32)
    swa_group_page_indices = torch.zeros((2, 128), dtype=torch.int32)
    pg_swa_b = torch.zeros((128,), dtype=torch.int32)
    pg_full = torch.zeros((1024,), dtype=torch.int32)
    dist_swa = torch.zeros((3,), dtype=torch.int32)
    dist_full = torch.zeros((3,), dtype=torch.int32)
    bundle_6d = torch.zeros((8, 6, 128, 1, 4, 128), dtype=torch.float8_e4m3fn)

    pending = []
    # 2 companion SWA group-1 layers with fresh slice views swa_group_page_indices[0]
    # plus 1 SWA group-0 layer (6D) -> must batch companion views into 1 m6d + 1 single op.
    for layer_idx, lid in ((10, 0), (12, 1)):
        pending.append((
            bundle_6d, kvl, swa_group_page_indices[0], cq, ckv, dist_swa,
            torch.tensor(lid, dtype=torch.int32),
            torch.zeros((32, 1, 4, 128), dtype=torch.float8_e4m3fn),
            False, True, (True, True, 1, False, True), layer_idx,
        ))
    pending.append((
        bundle_6d, kvl, pg_swa_b, cq, ckv, dist_swa,
        torch.tensor(2, dtype=torch.int32),
        torch.zeros((32, 1, 4, 128), dtype=torch.float8_e4m3fn),
        False, True, (True, True, 0, False, True), 14,
    ))
    # 11 Full unified 5D layers -> stays in a single m5d(11) call (<=32 limit, <=16 prewarmed).
    for i in range(11):
        pending.append((
            torch.zeros((8, 256, 1, 4, 128), dtype=torch.float8_e4m3fn),
            kvl, pg_full, cq, ckv, dist_full, None,
            torch.zeros((32, 1, 4, 128), dtype=torch.float8_e4m3fn),
            True, False, (False, False, -1, True, False), 20 + i,
        ))

    calls: list[tuple] = []
    with (
        mock.patch.object(b1_moe_ops, "get_step_c_kv_write_op", side_effect=lambda **kw: lambda *a: calls.append(("single", kw, len(a)))),
        mock.patch.object(b1_moe_ops, "get_step_c_kv_write_multi_6d_op", side_effect=lambda **kw: lambda *a: calls.append(("m6d", kw, tuple(a[0].shape), tuple(a[-1].tolist())))),
        mock.patch.object(b1_moe_ops, "get_step_c_kv_write_multi_5d_op", side_effect=lambda k, **kw: lambda *a: calls.append(("m5d", k, kw, tuple(x is pending[i + 3][7] for i, x in enumerate(a[:k])), tuple(a[0].shape), len(a)))),
    ):
        b1_moe_ops.flush_pending_step_c_kv_writes(pending)

    assert calls == [
        ("m6d", {"kv_pack_dense2": False, "kv_pack_bd4": True}, (2, 32, 1, 4, 128), (0, 1)),
        ("single", {"kv_pack_dense2": False, "kv_pack_bd4": True}, 8),
        ("m5d", 11, {"kv_pack_dense2": True, "kv_pack_bd4": False}, (True,) * 11, (32, 1, 4, 128), 27),
    ]

    # Verify 45 layers chunk as [29, 16], 46 layers (L0 + 45 SWA) chunk as [29, 17], and 27 as [16, 11].
    for n_chk, exp_k in ((45, [29, 16]), (46, [29, 17]), (27, [16, 11])):
        p_big = [
            (
                torch.zeros((8, 128, 1, 4, 128), dtype=torch.float8_e4m3fn),
                kvl, pg_full, cq, ckv, dist_full, None,
                torch.zeros((32, 1, 4, 128), dtype=torch.float8_e4m3fn),
                False, True, (False, False, -1, False, True), 100 + i,
            )
            for i in range(n_chk)
        ]
        chk_calls: list[int] = []
        with mock.patch.object(
            b1_moe_ops, "get_step_c_kv_write_multi_5d_op",
            side_effect=lambda k, **_kw: lambda *_a, _k=k: chk_calls.append(_k),
        ):
            b1_moe_ops.flush_pending_step_c_kv_writes(p_big)
        assert chk_calls == exp_k

    # Verify torch.compile(..., fullgraph=True) traces flush_pending_step_c_kv_writes with 0 id() failures.
    b1_moe_ops.get_step_c_kv_write_multi_6d_op(kv_pack_dense2=False, kv_pack_bd4=True)
    b1_moe_ops.get_step_c_kv_write_multi_5d_op(2, kv_pack_dense2=True, kv_pack_bd4=False)
    compiled_calls: list[str] = []

    def _compiled_flush(k0, k1, c5a, c5b, b6d, swa_2d, pg_f, kvl_, cq_, ckv_, ds_, df_):
        p = [
            (b6d, kvl_, swa_2d[0], cq_, ckv_, ds_, torch.zeros((), dtype=torch.int32), k0, False, True, (True, True, 1, False, True), 0),
            (b6d, kvl_, swa_2d[0], cq_, ckv_, ds_, torch.ones((), dtype=torch.int32), k1, False, True, (True, True, 1, False, True), 2),
            (c5a, kvl_, pg_f, cq_, ckv_, df_, None, k0, True, False, (False, False, -1, True, False), 3),
            (c5b, kvl_, pg_f, cq_, ckv_, df_, None, k1, True, False, (False, False, -1, True, False), 5),
        ]
        b1_moe_ops.flush_pending_step_c_kv_writes(p)
        return k0

    def _stub_m6d(*_a):
        compiled_calls.append("m6d")

    def _stub_m5d_2(*_a):
        compiled_calls.append("m5d_2")

    with (
        mock.patch.dict(b1_moe_ops._STEP_C_KV_WRITE_MULTI_6D_OPS, {(False, True): _stub_m6d}),
        mock.patch.dict(b1_moe_ops._STEP_C_KV_WRITE_MULTI_5D_OPS, {(2, True, False): _stub_m5d_2}),
    ):
        fn_c = torch.compile(_compiled_flush, backend="eager", fullgraph=True)
        out_c = fn_c(
            torch.zeros((32, 1, 4, 128), dtype=torch.float8_e4m3fn),
            torch.zeros((32, 1, 4, 128), dtype=torch.float8_e4m3fn),
            torch.zeros((8, 256, 1, 4, 128), dtype=torch.float8_e4m3fn),
            torch.zeros((8, 256, 1, 4, 128), dtype=torch.float8_e4m3fn),
            bundle_6d,
            swa_group_page_indices,
            pg_full,
            kvl,
            cq,
            ckv,
            dist_swa,
            dist_full,
        )
    assert out_c.shape == (32, 1, 4, 128)
    assert compiled_calls == ["m6d", "m5d_2"]


def test_wave16_step_c_kv_write_multi_5d_op_schema_and_compile():
    """Verify get_step_c_kv_write_multi_5d_op custom_op schema, mutates_args, and FakeTensor dispatch."""
    from unittest import mock

    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode
    from sglang_tpu.muse_megakernel.decode import b1_moe_ops

    with pytest.raises(ValueError, match="2 <= num_layers <= 29"):
        b1_moe_ops.get_step_c_kv_write_multi_5d_op(1)
    with pytest.raises(ValueError, match="2 <= num_layers <= 29"):
        b1_moe_ops.get_step_c_kv_write_multi_5d_op(30)

    fake_calls: list[tuple[int, int]] = []
    with (
        mock.patch.dict(b1_moe_ops._STEP_C_KV_WRITE_MULTI_5D_OPS, {}, clear=True),
        mock.patch(
            "torch_tpu._internal.pallas.custom_jax_kernel",
            side_effect=lambda fn, **kw: (
                fake_calls.append((len(kw["donate_argnums"]), len(kw["input_output_aliases"]))) or fn
            ),
        ),
    ):
        op2 = b1_moe_ops.get_step_c_kv_write_multi_5d_op(2, kv_pack_dense2=True)
        op8 = b1_moe_ops.get_step_c_kv_write_multi_5d_op(8, kv_pack_bd4=True)
        assert fake_calls == [(2, 2), (8, 8)]

        schema2 = op2._opoverload._schema
        assert len(schema2.arguments) == 9  # 2 kv + 2 cache + 5 meta
        assert [a.alias_info.is_write if a.alias_info else False for a in schema2.arguments] == [
            False, False, True, True, False, False, False, False, False
        ]
        schema8 = op8._opoverload._schema
        assert len(schema8.arguments) == 21  # 8 kv + 8 cache + 5 meta
        assert [a.alias_info.is_write if a.alias_info else False for a in schema8.arguments] == (
            [False] * 8 + [True] * 8 + [False] * 5
        )

        with FakeTensorMode():
            kvs = [torch.empty((16, 1, 4, 128), dtype=torch.float8_e4m3fn) for _ in range(2)]
            cs = [torch.empty((20, 256, 1, 4, 128), dtype=torch.float8_e4m3fn) for _ in range(2)]
            meta = (
                torch.empty((16,), dtype=torch.int32),
                torch.empty((32,), dtype=torch.int32),
                torch.empty((17,), dtype=torch.int32),
                torch.empty((17,), dtype=torch.int32),
                torch.empty((3,), dtype=torch.int32),
            )
            assert op2(*kvs, *cs, *meta) is None


@pytest.mark.parametrize("fmt", ["512B", "dense2", "bd4"])
def test_wave16_step_c_write_kv_multi_5d_zero_stack_on_wait_and_aot(fmt: str):
    """Verify zero-stack 4D-tuple 5D multi-layer KV write under on_wait+detect_races and TPU7x HLO."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        write_decode_kv_merged_jax,
        write_decode_kv_multi_layer_jax,
    )

    bd4 = fmt == "bd4"
    dense2 = fmt == "dense2"
    pack_factor = 4 if bd4 else (2 if dense2 else 1)
    phys_rows = 512 // pack_factor
    B, n_layers, pps = 16, 5, 4
    nb = B * pps + 4
    n_live = 11
    rng = np.random.default_rng(20262700 + len(fmt))

    kvc_u8 = np.full((n_layers, nb, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
    kvl_np = np.array(
        [10, 11, 12, 13, 512, 513, 514, 515, 256, 257, 300, 0, 0, 0, 0, 0],
        dtype=np.int32,
    )
    pg_mat = rng.permutation(nb)[: B * pps].reshape(B, pps).astype(np.int32)
    for b_i in range(n_live):
        tok = int(kvl_np[b_i]) - 1
        pg = int(pg_mat[b_i, tok // 512])
        pr = (tok % 512) // pack_factor
        for s_prev in range(tok & (pack_factor - 1)):
            c_pair = (0 if s_prev < 2 else 2) if bd4 else (0 if s_prev == 0 else 2)
            l0, l1 = ((0, 64) if (s_prev & 1) == 0 else (64, 128)) if bd4 else (0, 128)
            kvc_u8[:, pg, pr, 0, c_pair : c_pair + 2, l0:l1] = rng.integers(
                0x30, 0x3D, size=(n_layers, 2, l1 - l0), dtype=np.uint8
            )
    caches_5d = tuple(
        jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8[li], jnp.uint8), jnp.float8_e4m3fn)
        for li in range(n_layers)
    )
    kv_new_list = []
    cols_valid = 64 if bd4 else 128
    for _ in range(n_layers):
        kv_u8 = np.full((B, 1, 4, 128), 0x7F, dtype=np.uint8)
        kv_u8[:n_live, 0, :2, :cols_valid] = rng.integers(
            0x30, 0x3D, size=(n_live, 2, cols_valid), dtype=np.uint8
        )
        kv_new_list.append(
            jax.lax.bitcast_convert_type(jnp.asarray(kv_u8, jnp.uint8), jnp.float8_e4m3fn)
        )
    kv_tuple = tuple(kv_new_list)

    kvl_j = jnp.asarray(kvl_np, jnp.int32)
    pg_j = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
    cq = jnp.asarray(
        np.concatenate([np.arange(n_live + 1), np.full((B - n_live,), n_live)], dtype=np.int32)
    )
    ckv = jnp.arange(B + 1, dtype=jnp.int32) * (pps * 512)
    dist = jnp.asarray([n_live, n_live, n_live], jnp.int32)

    ip_on_wait = pltpu.InterpretParams(
        dma_execution_mode="on_wait",
        detect_races=True,
        uninitialized_memory="nan",
    )
    got = write_decode_kv_multi_layer_jax(
        kv_tuple, caches_5d, kvl_j, pg_j, cq, ckv, dist,
        kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_on_wait,
    )
    assert isinstance(got, tuple) and len(got) == n_layers
    for li in range(n_layers):
        ref_li = write_decode_kv_merged_jax(
            kv_tuple[li], caches_5d[li], kvl_j, pg_j, cq, ckv, dist,
            kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_on_wait,
        )
        assert np.array_equal(
            np.asarray(ref_li).view(np.uint8), np.asarray(got[li]).view(np.uint8)
        ), f"Zero-stack 5D on_wait mismatch at layer {li} (fmt={fmt})"

    # Also verify q_len > 1 dynamic fallback with tuple input.
    cq_mq = cq.at[1:].set(cq[1:] + 1)
    kvl_mq = kvl_j.at[0].set(102)
    got_mq = write_decode_kv_multi_layer_jax(
        kv_tuple, caches_5d, kvl_mq, pg_j, cq_mq, ckv, dist,
        kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_on_wait,
    )
    ref_mq0 = write_decode_kv_merged_jax(
        kv_tuple[0], caches_5d[0], kvl_mq, pg_j, cq_mq, ckv, dist,
        kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_on_wait,
    )
    assert np.array_equal(np.asarray(ref_mq0).view(np.uint8), np.asarray(got_mq[0]).view(np.uint8))

    # TPU7x AOT HLO check for M=8 separate 4D inputs (matches get_step_c_kv_write_multi_5d_op):
    # every cache output must may-alias and zero stack/DUS/concatenate/copy may exist on kv_new.
    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x2" if bd4 else "tpu7x:2x2x1", platform="tpu").devices
    mesh = Mesh(np.array(tpu_devs[:1]), ("x",))
    M8, T_aot = 8, 64
    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    abs_args = (
        tuple(s((T_aot, 1, 4, 128), jnp.float8_e4m3fn) for _ in range(M8)),
        tuple(s((nb, phys_rows, 1, 4, 128), jnp.float8_e4m3fn) for _ in range(M8)),
        s((T_aot,), jnp.int32),
        s((T_aot * pps,), jnp.int32),
        s((T_aot + 1,), jnp.int32),
        s((T_aot + 1,), jnp.int32),
        s((3,), jnp.int32),
    )
    with mock.patch("sglang_tpu.attention.pallas.util.get_tpu_version", return_value=7):
        compiled = jax.jit(
            jax.shard_map(
                lambda kvs, cs, kl, p, q, ck, d: write_decode_kv_multi_layer_jax(
                    kvs, cs, kl, p, q, ck, d, kv_pack_dense2=dense2, kv_pack_bd4=bd4
                ),
                mesh=mesh,
                in_specs=(
                    tuple(P() for _ in range(M8)),
                    tuple(P() for _ in range(M8)),
                    P(), P(), P(), P(), P(),
                ),
                out_specs=tuple(P() for _ in range(M8)),
                check_vma=False,
            ),
            donate_argnums=(1,),
        ).lower(*abs_args).compile()
    hlo = compiled.as_text()
    for out_i in range(M8):
        assert f"{{{out_i}}}: ({M8 + out_i}, {{}}, may-alias)" in hlo
    bad_ops = [
        ln
        for ln in hlo.splitlines()
        if any(op in ln.split("backend_config=")[0] for op in ("= dynamic-update-slice(", "= concatenate(", "= copy("))
        and (f"f8e4m3fn[{M8},{T_aot},1,4,128]" in ln or f"f8e4m3fn[{T_aot},1,4,128]" in ln)
    ]
    assert not bad_ops, f"Unexpected stack/DUS/copy HLO ops: {bad_ops}"


@pytest.mark.parametrize("tp", [8, 16])
def test_wave10_step_c_full_attn_adaptive_1k_and_8k1k_aot(tp: int):
    """Verify Step (c) Full-Attn (BD4 TP=16 & DENSE2 TP=8) with adaptive 1K lowers & compiles <64 MiB VMEM."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_rpad_chip_sharded_jax,
    )

    bd4 = tp == 16
    dense2 = tp == 8
    phys_rows = 128 if bd4 else 256
    B, pps = 64, 32
    r_pair = B // (tp // 2)
    qkv_cols = 640 if tp == 16 else 1280
    gate_cols = 512 if tp == 16 else 1024
    hidden, h_pre, scale_block = 8192, 4096, 512
    nb = B * pps + 4

    topo_str = "tpu7x:2x2x2" if tp == 16 else "tpu7x:2x2x1"
    tpu_devs = topologies.get_topology_desc(topo_str, platform="tpu").devices
    mesh = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )

    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    in_specs = (
        P("model"), P("model", None), P(None, "model"),
        P("model", None), P(), P(), P(), P(),
        P(None, "model"), P("model"), P(), P(), P(),
        P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None),
        P(), P("model", None), P(), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(),
        P(), P(), P(), P(), P(), P(),
        P(), P(),
    )
    abstract = (
        s((tp,), jnp.int32), s((tp * r_pair, hidden), jnp.float32), s((B, tp * gate_cols), jnp.bfloat16),
        s((tp * gate_cols, hidden), jnp.float8_e4m3fn), s((hidden,), jnp.float32),
        s((hidden,), jnp.float32), s((hidden,), jnp.float32), s((hidden,), jnp.float32),
        s((hidden, tp * (h_pre // 2)), jnp.float8_e4m3fn), s((tp * (h_pre // 2),), jnp.float32),
        s((hidden, 256), jnp.bfloat16), s((h_pre,), jnp.float32), s((1, 256), jnp.float32),
        s((tp * 32, 512, 8192), jnp.uint32), s((tp * 32, 8, 8192), jnp.bfloat16),
        s((tp * 32, 512, 4096), jnp.uint32), s((tp * 32, 8, 4096), jnp.bfloat16),
        s((h_pre,), jnp.float32), s((tp * h_pre, h_pre), jnp.float8_e4m3fn), s((h_pre,), jnp.float32),
        s((hidden,), jnp.float32), s((hidden,), jnp.float32), s((hidden,), jnp.float32), s((hidden,), jnp.float32),
        s((tp * hidden, qkv_cols), jnp.float8_e4m3fn), s((qkv_cols,), jnp.float32),
        s((tp * hidden, gate_cols), jnp.float8_e4m3fn), s((gate_cols,), jnp.float32),
        s((nb, phys_rows, 1, 4, 128), jnp.float8_e4m3fn),
        s((B,), jnp.int32), s((B * pps,), jnp.int32), s((B + 1,), jnp.int32), s((B + 1,), jnp.int32),
        s((3,), jnp.int32),
        s((B, 2, 128), jnp.bfloat16),
        s((4,), jnp.float32),
    )

    def _run_aot(*args):
        (
            rk, st, gt, wo, wos, pa, pb, pfg, wp, sp, wr, pe, bs,
            w1, s1, w2, s2, nm, wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg, sg,
            kvc, kvl, pi, cq, ckv, dt, cs, fp,
        ) = args
        return b1_moe_b2_rpad_chip_sharded_jax(
            rk, st, gt, wo, wos, pa, pb, pfg, wp, sp, wr, pe, bs,
            w1, s1, w2, s2, nm, wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg, sg,
            kvc, kvl, pi, cq, ckv, dt,
            cos_sin_step=cs, free_params=fp,
            world_size=tp,
            rpa_sm_scale=64.0**-0.5,
            rpa_sliding_window=-1,
            rpa_mode_str="w8,s4,batchfull,g4k,foldcol,shared16",
            kv_pack_dense2=dense2,
            kv_pack_bd4=bd4,
            replica_cost_mode="tiles",
            skip_kv_write=True,
        )

    with mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
        lowered = jax.jit(
            jax.shard_map(
                _run_aot,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=(P("model", None), P("model"), P()),
                check_vma=False,
            )
        ).lower(*abstract)
    hlo_txt = lowered.as_text()
    assert f"muse_w9_b1_moe_b2_rpad_b64_tp{tp}_" in hlo_txt
    assert "_c1024_" not in hlo_txt, f"Unexpected adaptive 1K lax.cond split in lowered HLO for TP={tp}"
    assert lowered.compile() is not None


def _compile_step_c_aot(
    *,
    tp: int,
    B: int,
    pps: int,
    sw: int,
    mode: str,
    dense2: bool,
    bd4: bool,
    dist_len: int = 3,
    has_rope: bool = False,
    allow_large_b: bool | None = None,
):
    """Lower and AOT-compile b1_moe_b2_rpad_chip_sharded_jax for TPU7x and assert <63 MiB VMEM."""
    import gc
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_rpad_chip_sharded_jax,
    )

    phys_rows = 128 if bd4 else (256 if dense2 else 512)
    r_pair = B // (tp // 2)
    qkv_cols = 640 if tp == 16 else 1280
    gate_cols = 512 if tp == 16 else 1024
    hidden, h_pre = 8192, 4096
    nb = B * pps + 4

    topo_str = "tpu7x:2x2x2" if tp == 16 else "tpu7x:2x2x1"
    tpu_devs = topologies.get_topology_desc(topo_str, platform="tpu").devices
    mesh = Mesh(
        np.array(sorted(tpu_devs, key=lambda dv: (dv.coords[1], dv.coords[0], dv.core_on_chip))),
        ("model",),
    )

    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    in_specs = (
        P("model"), P("model", None), P(None, "model"),
        P("model", None), P(), P(), P(), P(),
        P(None, "model"), P("model"), P(), P(), P(),
        P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None),
        P(), P("model", None), P(), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(),
        P(), P(), P(), P(), P(), P(),
        *((P(),) if has_rope else ()),
        P(),
    )
    abstract = (
        s((tp,), jnp.int32), s((tp * r_pair, hidden), jnp.float32), s((B, tp * gate_cols), jnp.bfloat16),
        s((tp * gate_cols, hidden), jnp.float8_e4m3fn), s((hidden,), jnp.float32),
        s((hidden,), jnp.float32), s((hidden,), jnp.float32), s((hidden,), jnp.float32),
        s((hidden, tp * (h_pre // 2)), jnp.float8_e4m3fn), s((tp * (h_pre // 2),), jnp.float32),
        s((hidden, 256), jnp.bfloat16), s((h_pre,), jnp.float32), s((1, 256), jnp.float32),
        s((tp * 32, 512, 8192), jnp.uint32), s((tp * 32, 8, 8192), jnp.bfloat16),
        s((tp * 32, 512, 4096), jnp.uint32), s((tp * 32, 8, 4096), jnp.bfloat16),
        s((h_pre,), jnp.float32), s((tp * h_pre, h_pre), jnp.float8_e4m3fn), s((h_pre,), jnp.float32),
        s((hidden,), jnp.float32), s((hidden,), jnp.float32), s((hidden,), jnp.float32), s((hidden,), jnp.float32),
        s((tp * hidden, qkv_cols), jnp.float8_e4m3fn), s((qkv_cols,), jnp.float32),
        s((tp * hidden, gate_cols), jnp.float8_e4m3fn), s((gate_cols,), jnp.float32),
        s((nb, phys_rows, 1, 4, 128), jnp.float8_e4m3fn),
        s((B,), jnp.int32), s((B * pps,), jnp.int32), s((B + 1,), jnp.int32), s((B + 1,), jnp.int32),
        s((dist_len,), jnp.int32),
        *((s((B, 2, 128), jnp.bfloat16),) if has_rope else ()),
        s((4,), jnp.float32),
    )

    def _run_aot(*args):
        rk, st, gt, wo, wos, pa, pb, pfg, wp, sp, wr, pe, bs, w1, s1, w2, s2, nm, wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg, sg, kvc, kvl, pi, cq, ckv, dt = args[:34]
        cs = args[34] if has_rope else None
        fp = args[35] if has_rope else args[34]
        return b1_moe_b2_rpad_chip_sharded_jax(
            rk, st, gt, wo, wos, pa, pb, pfg, wp, sp, wr, pe, bs,
            w1, s1, w2, s2, nm, wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg, sg,
            kvc, kvl, pi, cq, ckv, dt,
            cos_sin_step=cs, free_params=fp,
            world_size=tp,
            rpa_sm_scale=64.0**-0.5,
            rpa_sliding_window=sw,
            rpa_mode_str=mode,
            kv_pack_dense2=dense2,
            kv_pack_bd4=bd4,
            replica_cost_mode="tiles",
            skip_kv_write=True,
            allow_large_b=allow_large_b,
        )

    with mock.patch("sglang_tpu.attention.pallas.util.get_torch_tpu_device_kind", return_value="TPU v7"):
        lowered = jax.jit(
            jax.shard_map(
                _run_aot,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=(P("model", None), P("model"), P()),
                check_vma=False,
            )
        ).lower(*abstract)
        hlo = lowered.as_text()
        comp = lowered.compile()
        assert comp is not None
        del comp, lowered
        jax.clear_caches()
        gc.collect()
    return hlo


@pytest.mark.parametrize(
    "tp,B,dense2,bd4,has_rope",
    [
        (8, 128, True, False, True),
        (16, 128, False, True, True),
        (16, 256, False, True, False),
        (16, 256, False, True, True),
    ],
)
def test_wave11_step_c_b128_b256_3scope_swa_aot(
    tp: int, B: int, dense2: bool, bd4: bool, has_rope: bool
):
    """Verify 3-scope Step (c) compiles <63 MiB VMEM on TPU7x (BD4 b2 by default; DENSE2 explicit b1 opt-in)."""
    hlo = _compile_step_c_aot(
        tp=tp,
        B=B,
        pps=4,
        sw=2049,
        mode="w8,s4,b1,batchfull,g4k,foldcol,shared16" if dense2 else "w8,s4,batchfull,g4k,foldcol,shared16",
        dense2=dense2,
        bd4=bd4,
        dist_len=3,
        has_rope=has_rope,
        allow_large_b=dense2,
    )
    assert f"muse_w9_b1_moe_b2_rpad_b{B}_tp{tp}_" in hlo


@pytest.mark.parametrize(
    "tp,dense2,bd4,mode,dist_len",
    [
        (8, True, False, "w8,s4,c2048,shared64,coal,noz,pbf16,q1024,shsplit,shfp8,batchfull,foldcol", 4),
        (16, False, True, "w8,s4,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol", 4),
    ],
)
def test_wave11_step_c_lc_shared_full_attn_aot(
    tp: int, dense2: bool, bd4: bool, mode: str, dist_len: int
):
    """Verify Step (c) supports LC shared64 Full-Attn (DENSE2 TP=8 & BD4 TP=16) <63 MiB VMEM."""
    hlo = _compile_step_c_aot(
        tp=tp,
        B=64,
        pps=512,
        sw=-1,
        mode=mode,
        dense2=dense2,
        bd4=bd4,
        dist_len=dist_len,
        has_rope=False,
    )
    assert f"muse_w9_b1_moe_b2_rpad_b64_tp{tp}_" in hlo


@pytest.mark.parametrize("case", ["b128_swa_3scope", "b64_lc_shared16"])
def test_wave11_step_c_interpret_parity_vs_step_b(case: str):
    """Verify eager race/NaN-detect InterpretParams parity of 3-scope B=128 Step (c) and LC shared16 Step (c)."""
    import dataclasses
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_chip_sharded_jax,
        b1_moe_b2_rpad_chip_sharded_jax,
        write_decode_kv_merged_jax,
    )

    import gc

    world_size = 8
    devs = jax.devices("cpu")
    if len(devs) < world_size:
        pytest.skip(f"Requires XLA_FLAGS=--xla_force_host_platform_device_count={world_size}")
    mesh = Mesh(np.array(devs[:world_size]), ("model",))

    is_lc = case == "b64_lc_shared16"
    B = 64 if is_lc else 128
    hidden, h_pre, scale_block, qkv_cols, gate_cols = 8192, 4096, 512, 1280, 1024
    r_pair = B // (world_size // 2)
    rng = np.random.default_rng(20262210 + (1 if is_lc else 0))

    cfg = dataclasses.replace(
        default_sparse_decode_config(B, world_size, scale_block),
        weight_mode="half_n" if B > 64 else "stream3",
        pair_rs=True,
        bridge_ag=True,
        m_buckets=(32, 64) if B > 64 else (16, 32),
        route_chunk_rows=32,
        static_holder_mask=True,
        route_fast=True,
        fast_plan=2,
        unroll_strips=True,
        fast_gs=True,
        spec_prime_parts=0,
    )
    _rf8 = lambda sh: jnp.asarray((rng.standard_normal(sh) * 0.15).astype(np.float32), jnp.float8_e4m3fn)
    ranks = jnp.arange(world_size, dtype=jnp.int32)
    stream_chip = jnp.asarray((rng.standard_normal((world_size * r_pair, hidden)) * 0.1).astype(np.float32))
    gated = jnp.asarray((rng.standard_normal((B, world_size * gate_cols)) * 0.1).astype(np.float32), jnp.bfloat16)
    wo, wos = _rf8((world_size * gate_cols, hidden)), jnp.ones((hidden,), jnp.float32) * 0.05
    ones_h = jnp.ones((hidden,), jnp.float32)
    w_pre, s_pre = _rf8((hidden, world_size * (h_pre // 2))), jnp.ones((world_size * (h_pre // 2),), jnp.float32) * 0.05
    w_router = jnp.asarray((rng.standard_normal((hidden, 256)) * 0.02).astype(np.float32), jnp.bfloat16)
    bias_np = np.zeros((1, 256), np.float32)
    bias_np[0, 0:256:32] = 12.0
    pre_exp, bias = jnp.ones((h_pre,), jnp.float32), jnp.asarray(bias_np)
    # 8 biased experts (32..256 step 32 -> 1 per rank) with non-zero weights on held ranks.
    w1_u = np.zeros((world_size * 32, 512, 8192), dtype=np.uint32)
    w2_u = np.zeros((world_size * 32, 512, 4096), dtype=np.uint32)
    w1_u[::32] = rng.integers(0, 0x11111111, size=(world_size, 512, 8192), dtype=np.uint32)
    w2_u[::32] = rng.integers(0, 0x11111111, size=(world_size, 512, 4096), dtype=np.uint32)
    w1, s1 = jnp.asarray(w1_u, jnp.uint32), jnp.ones((world_size * 32, 8, 8192), jnp.bfloat16) * 0.02
    w2, s2 = jnp.asarray(w2_u, jnp.uint32), jnp.ones((world_size * 32, 8, 4096), jnp.bfloat16) * 0.02
    del w1_u, w2_u, bias_np
    gc.collect()
    post_exp = jnp.ones((h_pre,), jnp.float32)
    w_post, s_post = _rf8((world_size * h_pre, h_pre)), jnp.ones((h_pre,), jnp.float32) * 0.05
    w_qkv, s_qkv = _rf8((world_size * hidden, qkv_cols)), jnp.ones((qkv_cols,), jnp.float32) * 0.05
    w_gate, s_gate = _rf8((world_size * hidden, gate_cols)), jnp.ones((gate_cols,), jnp.float32) * 0.05
    fp = jnp.asarray([0.84, 2.0, 0.20, 1.0], jnp.float32)

    pps = 6 if is_lc else 4
    phys_rows = 256
    nb = B * pps + 4
    if is_lc:
        sh_pgs = 3
        kvl_np = np.array([sh_pgs * 512 + 30 + (i & 31) for i in range(B)], dtype=np.int32)
        pg_mat = np.zeros((B, pps), dtype=np.int32)
        nxt = 0
        for g in range(B // 16):
            sh = np.arange(nxt, nxt + sh_pgs, dtype=np.int32)
            nxt += sh_pgs
            for j in range(16):
                row = g * 16 + j
                pg_mat[row, :sh_pgs] = sh
                pg_mat[row, sh_pgs:] = np.arange(nxt, nxt + (pps - sh_pgs), dtype=np.int32)
                nxt += pps - sh_pgs
        pg_j = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
        nb = nxt + 4
        shared_p = jnp.full((B // 16,), sh_pgs, dtype=jnp.int32)
        dist = jnp.concatenate([jnp.asarray([B, B, B], jnp.int32), shared_p])
        sw = -1
        mode = "w8,s4,c2048,shared16,coal,noz,pbf16,q512,shsplit,shfp8,batchfull,foldcol"
    else:
        kvl_np = np.array([150 + (i & 63) + 512 * (i % 3) for i in range(B)], dtype=np.int32)
        pg_j = jnp.asarray(rng.permutation(nb)[: B * pps].astype(np.int32), jnp.int32)
        shared_p = None
        dist = jnp.asarray([B, B, B], jnp.int32)
        sw = 2049
        mode = "w8,s4,b1,batchfull,g4k,foldcol"

    kvc_base = jax.lax.bitcast_convert_type(
        jnp.asarray(np.full((nb, phys_rows, 1, 4, 128), 0x38, dtype=np.uint8), jnp.uint8),
        jnp.float8_e4m3fn,
    )
    kvl_j = jnp.asarray(kvl_np, jnp.int32)
    cuq = jnp.arange(B + 1, dtype=jnp.int32)
    cukv = jnp.arange(B + 1, dtype=jnp.int32) * (pps * 512)

    base_args = (
        ranks, stream_chip, gated, wo, wos, ones_h, ones_h, ones_h, w_pre, s_pre, w_router, pre_exp, bias,
        w1, s1, w2, s2, post_exp, w_post, s_post, ones_h, ones_h, ones_h, ones_h, w_qkv, s_qkv, w_gate, s_gate,
        kvc_base, kvl_j, pg_j, cuq, cukv, dist, fp,
    )
    in_specs = (
        P("model"), P("model", None), P(None, "model"),
        P("model", None), P(), P(), P(), P(),
        P(None, "model"), P("model"), P(), P(), P(),
        P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None),
        P(), P("model", None), P(), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(),
        P(), P(), P(), P(), P(), P(), P(),
    )

    def _ref_b(*args):
        rk, st, gt, wo_, wos_, pa, pb, pfg, wp, sp, wr, pe, bs, w1_, s1_, w2_, s2_, nm, wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg_, sg_, kvc_, kvl_, pi_, cq_, ckv_, dt_, fp_ = args
        st_o, q_5d, kvm, gt_o = b1_moe_b2_chip_sharded_jax(
            rk, st, gt, wo_, wos_, pa, pb, pfg, wp, sp, wr, pe, bs, w1_, s1_, w2_, s2_, nm,
            wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg_, sg_, cos_sin_step=None, free_params=fp_,
            world_size=world_size, fuse_qkv_epi=True, replica_cost_mode="tiles", config=cfg, interpret=True,
        )
        out_g, kvc_o = run_decode_fast_rpa(
            q_5d, kvm, None, kvc_, kvl_, pi_, cq_, ckv_, dt_[:3],
            sm_scale=64.0**-0.5, sliding_window=(None if sw <= 0 else sw),
            k_scale=1.0, v_scale=1.0, mode_str=mode, decode_query_size=1,
            fused_out_pack=2, prepared_inputs=True, gate=gt_o, out_norm_eps=1e-5,
            kv_pack_dense2=True, shared_pages=shared_p, interpret=True,
        )
        return st_o, out_g, kvc_o

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )

    def _fast_c(*args):
        rk, st, gt, wo_, wos_, pa, pb, pfg, wp, sp, wr, pe, bs, w1_, s1_, w2_, s2_, nm, wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg_, sg_, kvc_, kvl_, pi_, cq_, ckv_, dt_, fp_ = args
        st_o, out_g, kvm = b1_moe_b2_rpad_chip_sharded_jax(
            rk, st, gt, wo_, wos_, pa, pb, pfg, wp, sp, wr, pe, bs, w1_, s1_, w2_, s2_, nm,
            wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg_, sg_, kvc_, kvl_, pi_, cq_, ckv_, dt_,
            cos_sin_step=None, free_params=fp_, world_size=world_size,
            rpa_sm_scale=64.0**-0.5, rpa_sliding_window=sw,
            rpa_mode_str=mode, kv_pack_dense2=True, replica_cost_mode="tiles",
            config=cfg, skip_kv_write=True, interpret=ip_race,
            allow_large_b=(not is_lc),
        )
        kvc_o = write_decode_kv_merged_jax(
            kvm, kvc_, kvl_, pi_, cq_, ckv_, dt_, kv_pack_dense2=True, interpret=ip_race,
        )
        return st_o, out_g, kvc_o

    ref_res = jax.jit(
        jax.shard_map(_ref_b, mesh=mesh, in_specs=in_specs, out_specs=(P("model", None), P(), P()), check_vma=False)
    )(*base_args)
    ref_st_np = np.asarray(ref_res[0], dtype=np.float64).ravel()
    ref_g_np = np.asarray(ref_res[1], dtype=np.float64).ravel()
    ref_kv_u8 = np.asarray(ref_res[2]).view(np.uint8).copy()
    del ref_res
    jax.clear_caches()
    gc.collect()

    fast_res = jax.jit(
        jax.shard_map(_fast_c, mesh=mesh, in_specs=in_specs, out_specs=(P("model", None), P(), P()), check_vma=False)
    )(*base_args)
    fast_st_np = np.asarray(fast_res[0], dtype=np.float64).ravel()
    fast_g_np = np.asarray(fast_res[1], dtype=np.float64).ravel()
    fast_kv_u8 = np.asarray(fast_res[2]).view(np.uint8).copy()
    del fast_res, base_args, w1, w2, s1, s2, kvc_base
    jax.clear_caches()
    gc.collect()

    for x, y, nm in [(ref_st_np, fast_st_np, "stream"), (ref_g_np, fast_g_np, "gated")]:
        cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        assert cos > 0.9999, f"{case} {nm} cos={cos}"
    assert np.array_equal(ref_kv_u8, fast_kv_u8)


def test_wave11_step_c_rpa_plan_shared_and_large_b():
    """Verify resolve_step_c_rpa_plan keeps BD4 B=256 b2 ON, gates DENSE2 B>64 off, and blocks grp=1 downgrade."""
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        resolve_step_c_rpa_plan,
    )

    assert tpu_envs.SGLANG_TPU_MUSE_MK_STEP_C_LARGE_B.default is False

    # TP=16 B=256 BD4 SWA (batch4) upgrades from s4_b2 to s3_b4 (7,680 <= 8,192): keeps Step (c) ON at full b4.
    cfg_swa_256_bd4, _, slot_swa_256_bd4, grp_swa_256_bd4, ok_swa_256_bd4 = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4",
        sliding_window=2049,
        pages_per_seq=4,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=256,
        max_num_seqs=256,
    )
    assert ok_swa_256_bd4 is True and grp_swa_256_bd4 is True
    assert cfg_swa_256_bd4.num_slots == 3 and cfg_swa_256_bd4.swabatch_group_size == 4
    assert slot_swa_256_bd4 == 20

    # TP=8 B=128 DENSE2 stays gated to 2-Scope Step (b) by default (#4962/#5010); LARGE_B=1 unlocks
    # 40*512 bkv (SWA batch4 fits cleanly, while >40*512 SWA still refuses silent grp=1 downgrade).
    _, _, _, _, ok_swa_128_def = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4",
        sliding_window=2049,
        pages_per_seq=4,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=128,
        max_num_seqs=128,
    )
    assert ok_swa_128_def is False

    cfg_swa_128_b4, _, slot_swa_128_b4, grp_swa_128_b4, ok_swa_128_b4 = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4,d2grp",
        sliding_window=2049,
        pages_per_seq=32,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=128,
        max_num_seqs=128,
        allow_large_b=True,
    )
    assert ok_swa_128_b4 is True and grp_swa_128_b4 is True and cfg_swa_128_b4.swabatch_group_size == 4
    assert slot_swa_128_b4 == 20

    _, _, _, _, ok_swa_128_over = resolve_step_c_rpa_plan(
        "w8,s12,batchfull,g4k,foldcol,batch4,d2grp",
        sliding_window=2049,
        pages_per_seq=32,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=128,
        max_num_seqs=128,
        allow_large_b=True,
    )
    assert ok_swa_128_over is False

    _, _, _, _, ok_swa_128_b1 = resolve_step_c_rpa_plan(
        "w8,s4,b1,batchfull,g4k,foldcol",
        sliding_window=2049,
        pages_per_seq=4,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=128,
        max_num_seqs=128,
        allow_large_b=True,
    )
    assert ok_swa_128_b1 is True

    # BD4 B=256 Full 8K/1K (pps=32) upgrades to Wave-17 group-vectorized ring4k (s3, c4096, rg=4, nocoal, 12,288 rows <= 24*512);
    # DENSE2 B=128 Full stays gated by default; 512B B=256 and pps=512 B=256 stay gated to Step (b).
    cfg_full_256, one_full_256, slot_full_256, grp_full_256, ok_full_256 = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol",
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=256,
        max_num_seqs=256,
    )
    assert ok_full_256 is True and one_full_256 is False and grp_full_256 is True
    assert (
        cfg_full_256.loop_kind,
        cfg_full_256.num_slots,
        cfg_full_256.chunk_sz,
        cfg_full_256.ring_group_size,
        slot_full_256,
    ) == (
        "ring4k",
        3,
        4096,
        4,
        32,
    )
    assert cfg_full_256.coalesce_dma is False

    _, _, _, _, ok_d2_full_128 = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol",
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=128,
        max_num_seqs=128,
    )
    assert ok_d2_full_128 is False

    _, _, _, _, ok_512_256 = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol",
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=512,
        b_tokens=256,
        max_num_seqs=256,
    )
    assert ok_512_256 is False

    _, _, _, _, ok_512_256_lb = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol",
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=512,
        b_tokens=256,
        max_num_seqs=256,
        allow_large_b=True,
    )
    assert ok_512_256_lb is True

    _, _, _, _, ok_smem_over = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol",
        sliding_window=None,
        pages_per_seq=1024,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=256,
        max_num_seqs=256,
    )
    assert ok_smem_over is False

    # LC shared64 accepted ONLY when explicit shared_group_size>0 and divisible.
    cfg_sh, _, _, _, ok_sh64 = resolve_step_c_rpa_plan(
        "w8,s4,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
        sliding_window=None,
        pages_per_seq=512,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
        shared_group_size=64,
        has_shared_prefix=True,
    )
    assert ok_sh64 is True and cfg_sh.shared_group_size == 64

    _, _, _, _, ok_sh_missing = resolve_step_c_rpa_plan(
        "shared16",
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
        shared_group_size=0,
        has_shared_prefix=True,
    )
    assert ok_sh_missing is False


def test_wave11_muse_spark_step_c_b128_b256_and_lc_full_dispatch():
    """Verify MuseSparkDecoderLayer Step (c) dispatches B=128/256 SWA and gates LC Full on SGLANG_TPU_MUSE_MK_STEP_C_LC_FULL."""
    import os
    import types
    from unittest import mock

    import torch
    from sglang.srt.model_executor.forward_batch_info import (
        CaptureHiddenMode,
        ForwardMode,
    )

    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.muse_spark import MuseSparkDecoderLayer

    assert tpu_envs.SGLANG_TPU_MUSE_MK_STEP_C_LC_FULL.default is False

    h_dim, h_pre = 8192, 4096

    def _run_case(
        *,
        tp: int,
        batch: int,
        sw: int | None,
        pps: int,
        dist_len: int,
        rpa_mode: str,
        lc_full_env: str,
        lc_swa_env: str = "1",
        large_b_env: str = "0",
    ):
        r_pair = batch // (tp // 2)
        qkv_cols = 640 if tp == 16 else 1280
        gate_cols = 512 if tp == 16 else 1024
        dense2 = tp == 8
        bd4 = tp == 16
        phys_rows = 128 if bd4 else 256
        nb = batch * pps + 4

        norm_stub = types.SimpleNamespace(
            eps=1e-5, scale_input=False, gain=lambda: torch.ones((h_dim,), dtype=torch.float32)
        )
        gate_stub = types.SimpleNamespace(
            alpha=torch.ones((h_dim,), dtype=torch.float32),
            beta=torch.ones((h_dim,), dtype=torch.float32),
        )
        mlp = types.SimpleNamespace(
            _fused_decode_mode="sparse",
            _fused_decode_op=object(),
            _fused_decode_batches=(32, 64, 128, 256),
            _fused_decode_rank=torch.zeros((tp,), dtype=torch.int32),
            _fused_decode_num_replicas=0,
            _fused_decode_has_spec_slot=False,
            _fused_decode_replica_cost_mode="tiles",
            router_multiplier=1.0,
            pre_expert_norm=norm_stub,
            _bias_2d=lambda: torch.zeros((1, 256), dtype=torch.float32),
            _norm_2d=lambda: torch.ones((1, h_pre), dtype=torch.float32),
            _free_params_for_batch=lambda _b: torch.ones((4,), dtype=torch.float32),
            _buffers={
                "_w7_wrouter_km": torch.zeros((h_dim, 256), dtype=torch.bfloat16),
                "_w8_wpre_pair_w8k": torch.zeros((h_dim, h_pre // 2), dtype=torch.float8_e4m3fn),
                "_w8_spre_pair": torch.ones((h_pre // 2,), dtype=torch.float32),
                "_w9_wpost_pair_w8k": torch.zeros((h_pre, h_pre), dtype=torch.float8_e4m3fn),
                "_w9_spost_pair": torch.ones((h_pre,), dtype=torch.float32),
            },
            experts=types.SimpleNamespace(
                w13_weight=torch.zeros((1, 1, 1), dtype=torch.float4_e2m1fn_x2).expand(256 // tp, 4096, 4096),
                w13_weight_scale=torch.ones((1, 8, 1), dtype=torch.bfloat16).expand(256 // tp, 8, 4096),
                w2_weight=torch.zeros((1, 1, 1), dtype=torch.float4_e2m1fn_x2).expand(256 // tp, 2048, 4096),
                w2_weight_scale=torch.ones((1, 8, 1), dtype=torch.bfloat16).expand(256 // tp, 8, 4096),
            ),
        )
        self_attn = types.SimpleNamespace(
            _o_proj_w8k=torch.zeros((gate_cols, h_dim), dtype=torch.float8_e4m3fn),
            _o_proj_scale=torch.ones((h_dim,), dtype=torch.float32),
        )
        next_layer_attn = types.SimpleNamespace(
            decode_attn_gate_fuse=True,
            rotary_emb=object() if sw is not None else None,
            norm_eps=1e-5,
            query_scale=1.0,
            _buffers={
                "_w7_wqkv_w8k": torch.zeros((h_dim, qkv_cols), dtype=torch.float8_e4m3fn),
                "_w7_sqkv": torch.ones((qkv_cols,), dtype=torch.float32),
                "_w7_wgate_w8k": torch.zeros((h_dim, gate_cols), dtype=torch.float8_e4m3fn),
                "_w7_sgate": torch.ones((gate_cols,), dtype=torch.float32),
            },
            attn=types.SimpleNamespace(
                layer_id=1,
                sliding_window_size=(sw - 1) if sw is not None else None,
                scaling=64.0**-0.5,
                k_scale_float=1.0,
                v_scale_float=1.0,
            ),
        )
        layer = types.SimpleNamespace(
            _w9_b1_moe_b2_superkernel=True,
            _w8_bridge_qkv_epi=True,
            mlp=mlp,
            self_attn=self_attn,
            post_attention_layernorm=norm_stub,
            pre_feedforward_layernorm=norm_stub,
            post_feedforward_layernorm=norm_stub,
            post_attention_residual_gate=gate_stub,
            post_feedforward_residual_gate=gate_stub,
            _can_run_b1_moe_superkernel=lambda n, *, tp, is_dec_step, replicated=False: MuseSparkDecoderLayer._can_run_b1_moe_superkernel(
                layer, n, tp=tp, is_dec_step=is_dec_step, replicated=replicated
            ),
            _epi_scalars=lambda next_ln, next_at: MuseSparkDecoderLayer._epi_scalars(
                layer, next_ln, next_at
            ),
        )
        md = types.SimpleNamespace(
            decode_query_size=1,
            cu_q_lens=torch.arange(batch + 1, dtype=torch.int32),
            cu_kv_lens=torch.arange(batch + 1, dtype=torch.int32) * (pps * 512),
            distribution=torch.asarray([batch, batch, batch], dtype=torch.int32),
            full_decode_distribution=(
                torch.asarray([batch, batch, batch] + [500] * (dist_len - 3), dtype=torch.int32)
                if dist_len > 3
                else None
            ),
            swa_page_indices=torch.zeros((batch * pps,), dtype=torch.int32) if sw is not None else None,
            swa_group_page_indices=None,
        )
        backend_stub = types.SimpleNamespace(
            _dcp=1,
            _decode_prepared_gated_op=object(),
            _kv_pack_dense2=dense2,
            _kv_pack_bd4=bd4,
            _swa_layer_groups={1: 0},
            forward_metadata=md,
            token_to_kv_pool=types.SimpleNamespace(
                swa_kv_pool=types.SimpleNamespace(kv_pack_dense2=dense2, kv_pack_bd4=bd4),
                full_kv_pool=types.SimpleNamespace(kv_pack_dense2=dense2, kv_pack_bd4=bd4),
            ),
            can_forward_decode_prepared=lambda _a, _fb: True,
            _kv_lens=lambda _fb: torch.full((batch,), 256, dtype=torch.int32),
            _layer_page_indices=lambda _lid, is_swa_layer=False: torch.zeros((batch * pps,), dtype=torch.int32),
            _layer_distribution=lambda is_swa_layer=False: (
                md.distribution
                if (is_swa_layer or md.full_decode_distribution is None)
                else md.full_decode_distribution
            ),
        )
        fb = types.SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            attn_backend=backend_stub,
        )
        calls: list[str] = []
        pending: list = []
        env_map = {
            "SGLANG_TPU_MUSE_DECODE_STEP_C_SUPERKERNEL": "1",
            "SGLANG_TPU_MUSE_DECODE_TP16_QKV_EPI": "1" if tp == 16 else "0",
            "SGLANG_TPU_MUSE_MK_STEP_C_LC_SWA": lc_swa_env,
            "SGLANG_TPU_MUSE_MK_STEP_C_LC_FULL": lc_full_env,
            "SGLANG_TPU_MUSE_MK_STEP_C_LARGE_B": large_b_env,
            "SGLANG_TPU_RPA_V3_DECODE_FAST": rpa_mode,
        }
        with (
            mock.patch.dict(os.environ, env_map),
            mock.patch("sglang_tpu.models.muse_spark._can_use_bridge_qkv_epi", return_value=True),
            mock.patch("sglang_tpu.decode_qkv_epilogue_tpu._resolve_decode_attn_backend", return_value=backend_stub),
            mock.patch(
                "sglang_tpu.attention_pallas_rpa_base._fused_kv_operands",
                return_value=(torch.zeros((nb, phys_rows, 1, 4, 128), dtype=torch.float8_e4m3fn), None),
            ),
            mock.patch(
                "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_b2_rpad_chip_sharded_op",
                side_effect=lambda *_a, **_kw: (
                    lambda *_args: (
                        calls.append("step_c")
                        or (
                            torch.zeros((r_pair, h_dim), dtype=torch.float32),
                            torch.zeros((batch, gate_cols), dtype=torch.bfloat16),
                            torch.zeros((batch, 1, 4, 128), dtype=torch.float8_e4m3fn),
                        )
                    )
                ),
            ),
            mock.patch(
                "sglang_tpu.muse_megakernel.decode.b1_moe_ops.get_b1_moe_b2_chip_sharded_op",
                side_effect=lambda *_a, **_kw: (
                    lambda *_args: (
                        calls.append("step_b")
                        or (
                            torch.zeros((r_pair, h_dim), dtype=torch.float32),
                            torch.zeros((batch, 1, qkv_cols // 128, 2, 128), dtype=torch.bfloat16),
                            torch.zeros((batch, 1, 4, 128), dtype=torch.float8_e4m3fn),
                            torch.zeros((batch, gate_cols), dtype=torch.bfloat16),
                        )
                    )
                ),
            ),
        ):
            MuseSparkDecoderLayer._try_b1_moe_superkernel(
                layer,
                positions=torch.arange(batch, dtype=torch.int32),
                stream=torch.zeros((r_pair, h_dim), dtype=torch.float32),
                attn_input=None,
                forward_batch=fb,
                n_tok=batch,
                tp=tp,
                b2_kind=3,
                is_dec_step=True,
                pa_alpha_arg=gate_stub.alpha,
                pa_beta_arg=gate_stub.beta,
                pre_ff_gain=torch.ones((h_dim,), dtype=torch.float32),
                pre_exp_gain=torch.ones((h_pre,), dtype=torch.float32),
                cos_sin_step=torch.zeros((batch, 2, 128), dtype=torch.bfloat16) if sw is not None else None,
                cs_kw={},
                precomputed_qkv_gate=None,
                precomputed_prepared_qkvg=None,
                precomputed_gated=torch.zeros((batch, gate_cols), dtype=torch.bfloat16),
                next_input_layernorm=norm_stub,
                next_layer_attn=next_layer_attn,
                final_layernorm=None,
                pending_step_c_kv_writes=pending,
            )
        return calls, len(pending)

    # Wave 15: B=128 TP=8 DENSE2 stays gated by default; LARGE_B=1 unlocks SWA batch4 (40*512) while >40*512 refuses grp=1 downgrade;
    # B=256 TP=16 BD4 keeps Step (c) via b2 (5,120 <= 8,192).
    assert _run_case(tp=8, batch=128, sw=2049, pps=32, dist_len=3, rpa_mode="w8,s4,batchfull,g4k,foldcol,batch4", lc_full_env="0") == (["step_b"], 0)
    assert _run_case(tp=16, batch=256, sw=2049, pps=32, dist_len=3, rpa_mode="w8,s4,batchfull,g4k,foldcol,batch4", lc_full_env="0") == (["step_c"], 1)
    assert _run_case(tp=8, batch=128, sw=2049, pps=32, dist_len=3, rpa_mode="w8,s4,batchfull,g4k,foldcol,batch4,d2grp", lc_full_env="0", large_b_env="1") == (["step_c"], 1)
    assert _run_case(tp=8, batch=128, sw=2049, pps=32, dist_len=3, rpa_mode="w8,s12,batchfull,g4k,foldcol,batch4,d2grp", lc_full_env="0", large_b_env="1") == (["step_b"], 0)
    assert _run_case(tp=8, batch=128, sw=2049, pps=32, dist_len=3, rpa_mode="w8,s4,b1,batchfull,g4k,foldcol", lc_full_env="0", large_b_env="1") == (["step_c"], 1)

    # Target 2: LC Full shared64 stays gated to Step (b) by default (LC_FULL=0) and enters Step (c) only with LC_FULL=1.
    lc_mode = "w8,s4,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol"
    assert _run_case(tp=16, batch=64, sw=None, pps=512, dist_len=4, rpa_mode=lc_mode, lc_full_env="0") == (["step_b"], 0)
    assert _run_case(tp=16, batch=64, sw=None, pps=512, dist_len=4, rpa_mode=lc_mode, lc_full_env="1") == (["step_c"], 1)

    # Wave 12/13: 512B B=64 pps=32 & BD4 B=256 pps=32 Full-Attn enter Step (c) via ring4k auto-clamp.
    assert _run_case(tp=16, batch=64, sw=None, pps=32, dist_len=3, rpa_mode="w8,s4,batchfull,g4k,foldcol", lc_full_env="0") == (["step_c"], 1)
    assert _run_case(tp=16, batch=256, sw=None, pps=32, dist_len=3, rpa_mode="w8,s4,batchfull,g4k,foldcol", lc_full_env="0") == (["step_c"], 1)


@pytest.mark.parametrize(
    "tp,B,dense2,bd4,pps",
    [
        (8, 64, False, False, 32),
        (16, 64, False, False, 32),
        (16, 256, False, True, 32),
        (8, 128, True, False, 32),
    ],
)
def test_wave12_step_c_ring4k_autoclamp_and_adaptive_1k_aot(
    tp: int, B: int, dense2: bool, bd4: bool, pps: int
):
    """Verify ring4k auto-clamp compiles <63 MiB VMEM on TPU7x without lax.cond _c1024_ split."""
    hlo = _compile_step_c_aot(
        tp=tp,
        B=B,
        pps=pps,
        sw=-1,
        mode="w8,s4,b1,batchfull,g4k,foldcol,shared16" if dense2 else "w8,s4,batchfull,g4k,foldcol,shared16",
        dense2=dense2,
        bd4=bd4,
        dist_len=3,
        has_rope=False,
        allow_large_b=dense2,
    )
    assert f"muse_w9_b1_moe_b2_rpad_b{B}_tp{tp}_" in hlo
    assert "_c1024_" not in hlo


@pytest.mark.parametrize("case", ["b64_512b_adaptive_1k", "b64_512b_ring4k_8k", "b128_d2_ring4k_8k"])
def test_wave12_step_c_ring4k_and_adaptive_1k_interpret_parity(case: str):
    """Verify eager race/NaN InterpretParams parity for ring4k auto-clamp and adaptive _c1024 Full-Attn Step (c)."""
    import dataclasses
    import gc
    import jax
    import jax.numpy as jnp
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas.rpa_v3_decode_fast import run_decode_fast_rpa
    from sglang_tpu.moe.pallas.fused_rs_muse.fused_decode_sparse import (
        default_sparse_decode_config,
    )
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_b2_chip_sharded_jax,
        b1_moe_b2_rpad_chip_sharded_jax,
        write_decode_kv_merged_jax,
    )

    world_size = 8
    devs = jax.devices("cpu")
    if len(devs) < world_size:
        pytest.skip(f"Requires XLA_FLAGS=--xla_force_host_platform_device_count={world_size}")
    mesh = Mesh(np.array(devs[:world_size]), ("model",))

    is_b128 = case == "b128_d2_ring4k_8k"
    is_short = case == "b64_512b_adaptive_1k"
    dense2 = is_b128
    B = 128 if is_b128 else 64
    pps = 32
    phys_rows = 256 if dense2 else 512
    hidden, h_pre, scale_block, qkv_cols, gate_cols = 8192, 4096, 512, 1280, 1024
    r_pair = B // (world_size // 2)
    rng = np.random.default_rng(20262300 + len(case))

    cfg = dataclasses.replace(
        default_sparse_decode_config(B, world_size, scale_block),
        weight_mode="half_n" if B > 64 else "stream3",
        pair_rs=True,
        bridge_ag=True,
        m_buckets=(32, 64) if B > 64 else (16, 32),
        route_chunk_rows=32,
        static_holder_mask=True,
        route_fast=True,
        fast_plan=2,
        unroll_strips=True,
        fast_gs=True,
        spec_prime_parts=0,
    )
    _rf8 = lambda sh: jnp.asarray((rng.standard_normal(sh) * 0.15).astype(np.float32), jnp.float8_e4m3fn)
    ranks = jnp.arange(world_size, dtype=jnp.int32)
    stream_chip = jnp.asarray((rng.standard_normal((world_size * r_pair, hidden)) * 0.1).astype(np.float32))
    gated = jnp.asarray((rng.standard_normal((B, world_size * gate_cols)) * 0.1).astype(np.float32), jnp.bfloat16)
    wo, wos = _rf8((world_size * gate_cols, hidden)), jnp.ones((hidden,), jnp.float32) * 0.05
    ones_h = jnp.ones((hidden,), jnp.float32)
    w_pre, s_pre = _rf8((hidden, world_size * (h_pre // 2))), jnp.ones((world_size * (h_pre // 2),), jnp.float32) * 0.05
    w_router = jnp.asarray((rng.standard_normal((hidden, 256)) * 0.02).astype(np.float32), jnp.bfloat16)
    bias_np = np.zeros((1, 256), np.float32)
    bias_np[0, 0:256:32] = 12.0
    pre_exp, bias = jnp.ones((h_pre,), jnp.float32), jnp.asarray(bias_np)
    w1_u = np.zeros((world_size * 32, 512, 8192), dtype=np.uint32)
    w2_u = np.zeros((world_size * 32, 512, 4096), dtype=np.uint32)
    w1_u[::32] = rng.integers(0, 0x11111111, size=(world_size, 512, 8192), dtype=np.uint32)
    w2_u[::32] = rng.integers(0, 0x11111111, size=(world_size, 512, 4096), dtype=np.uint32)
    w1, s1 = jnp.asarray(w1_u, jnp.uint32), jnp.ones((world_size * 32, 8, 8192), jnp.bfloat16) * 0.02
    w2, s2 = jnp.asarray(w2_u, jnp.uint32), jnp.ones((world_size * 32, 8, 4096), jnp.bfloat16) * 0.02
    del w1_u, w2_u, bias_np
    gc.collect()

    post_exp = jnp.ones((h_pre,), jnp.float32)
    w_post, s_post = _rf8((world_size * h_pre, h_pre)), jnp.ones((h_pre,), jnp.float32) * 0.05
    w_qkv, s_qkv = _rf8((world_size * hidden, qkv_cols)), jnp.ones((qkv_cols,), jnp.float32) * 0.05
    w_gate, s_gate = _rf8((world_size * hidden, gate_cols)), jnp.ones((gate_cols,), jnp.float32) * 0.05
    fp = jnp.asarray([0.84, 2.0, 0.20, 1.0], jnp.float32)

    # Use 32 shared read blocks with per-seq permutations + B disjoint tail write blocks
    # so pps=32 exercises full multi-chunk ring4k / _c1024 within the 32 GiB container limit.
    nb = 32 + B
    kvc_u8 = rng.integers(0x30, 0x3D, size=(nb, phys_rows, 1, 4, 128), dtype=np.uint8)
    kvc_base = jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8, jnp.uint8), jnp.float8_e4m3fn)
    del kvc_u8
    kvl_np = (
        np.array([200 + (i * 11) % 800 for i in range(B)], dtype=np.int32)
        if is_short
        else np.array([6200 + (i * 73) % 4100 for i in range(B)], dtype=np.int32)
    )
    kvl_j = jnp.asarray(kvl_np, jnp.int32)
    pg_mat = np.zeros((B, pps), dtype=np.int32)
    for i in range(B):
        pg_mat[i, :] = (np.arange(pps, dtype=np.int32) + i) % 32
        pg_mat[i, int(kvl_np[i]) // 512] = 32 + i
    pg_j = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
    cuq = jnp.arange(B + 1, dtype=jnp.int32)
    cukv = jnp.arange(B + 1, dtype=jnp.int32) * (pps * 512)
    dist = jnp.asarray([B, B, B], jnp.int32)
    mode_sc = "w8,s4,b1,batchfull,g4k,foldcol,shared16" if is_b128 else "w8,s4,batchfull,g4k,foldcol,shared16"
    mode_sb = "w8,s3,ring4k,c4096,batchfull,g4k,foldcol"

    base_args = (
        ranks, stream_chip, gated, wo, wos, ones_h, ones_h, ones_h, w_pre, s_pre, w_router, pre_exp, bias,
        w1, s1, w2, s2, post_exp, w_post, s_post, ones_h, ones_h, ones_h, ones_h, w_qkv, s_qkv, w_gate, s_gate,
        kvc_base, kvl_j, pg_j, cuq, cukv, dist, fp,
    )
    in_specs = (
        P("model"), P("model", None), P(None, "model"),
        P("model", None), P(), P(), P(), P(),
        P(None, "model"), P("model"), P(), P(), P(),
        P("model", None, None), P("model", None, None), P("model", None, None), P("model", None, None),
        P(), P("model", None), P(), P(), P(), P(), P(),
        P("model", None), P(), P("model", None), P(),
        P(), P(), P(), P(), P(), P(), P(),
    )
    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )

    def _ref_b(*args):
        rk, st, gt, wo_, wos_, pa, pb, pfg, wp, sp, wr, pe, bs, w1_, s1_, w2_, s2_, nm, wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg_, sg_, kvc_, kvl_, pi_, cq_, ckv_, dt_, fp_ = args
        st_o, q_5d, kvm, gt_o = b1_moe_b2_chip_sharded_jax(
            rk, st, gt, wo_, wos_, pa, pb, pfg, wp, sp, wr, pe, bs, w1_, s1_, w2_, s2_, nm,
            wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg_, sg_, cos_sin_step=None, free_params=fp_,
            world_size=world_size, fuse_qkv_epi=True, replica_cost_mode="tiles", config=cfg, interpret=True,
        )
        out_g, kvc_o = run_decode_fast_rpa(
            q_5d, kvm, None, kvc_, kvl_, pi_, cq_, ckv_, dt_,
            sm_scale=64.0**-0.5, sliding_window=None,
            k_scale=1.0, v_scale=1.0, mode_str=mode_sb, decode_query_size=1,
            fused_out_pack=2, prepared_inputs=True, gate=gt_o, out_norm_eps=1e-5,
            kv_pack_dense2=dense2, interpret=True,
        )
        return st_o, out_g, kvc_o

    def _fast_c(*args):
        rk, st, gt, wo_, wos_, pa, pb, pfg, wp, sp, wr, pe, bs, w1_, s1_, w2_, s2_, nm, wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg_, sg_, kvc_, kvl_, pi_, cq_, ckv_, dt_, fp_ = args
        st_o, out_g, kvm = b1_moe_b2_rpad_chip_sharded_jax(
            rk, st, gt, wo_, wos_, pa, pb, pfg, wp, sp, wr, pe, bs, w1_, s1_, w2_, s2_, nm,
            wpt, spt, pff, pfa, pfb, nig, wqk, sqk, wg_, sg_, kvc_, kvl_, pi_, cq_, ckv_, dt_,
            cos_sin_step=None, free_params=fp_, world_size=world_size,
            rpa_sm_scale=64.0**-0.5, rpa_sliding_window=-1,
            rpa_mode_str=mode_sc, kv_pack_dense2=dense2, replica_cost_mode="tiles",
            config=cfg, skip_kv_write=True, interpret=ip_race,
            allow_large_b=is_b128,
        )
        kvc_o = write_decode_kv_merged_jax(
            kvm, kvc_, kvl_, pi_, cq_, ckv_, dt_, kv_pack_dense2=dense2, interpret=ip_race,
        )
        return st_o, out_g, kvc_o

    ref_res = jax.jit(
        jax.shard_map(_ref_b, mesh=mesh, in_specs=in_specs, out_specs=(P("model", None), P(), P()), check_vma=False)
    )(*base_args)
    ref_st_np = np.asarray(ref_res[0], dtype=np.float64).ravel()
    ref_g_np = np.asarray(ref_res[1], dtype=np.float64).ravel()
    ref_kv_u8 = np.asarray(ref_res[2]).view(np.uint8).copy()
    del ref_res
    jax.clear_caches()
    gc.collect()

    fast_res = jax.jit(
        jax.shard_map(_fast_c, mesh=mesh, in_specs=in_specs, out_specs=(P("model", None), P(), P()), check_vma=False)
    )(*base_args)
    fast_st_np = np.asarray(fast_res[0], dtype=np.float64).ravel()
    fast_g_np = np.asarray(fast_res[1], dtype=np.float64).ravel()
    fast_kv_u8 = np.asarray(fast_res[2]).view(np.uint8).copy()
    del fast_res, base_args, w1, w2, s1, s2, kvc_base
    jax.clear_caches()
    gc.collect()

    for x, y, nm in [(ref_st_np, fast_st_np, "stream"), (ref_g_np, fast_g_np, "gated")]:
        cos = float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-12))
        assert cos > 0.9999, f"{case} {nm} cos={cos}"
    assert np.array_equal(ref_kv_u8, fast_kv_u8)


@pytest.mark.parametrize(
    "n_layers,bd4,dense2",
    [
        (8, True, False),
        (5, True, False),
        (8, False, True),
    ],
)
def test_wave13_step_c_write_kv_multi_5d_sublane8_parity_and_aot(
    n_layers: int, bd4: bool, dense2: bool
):
    """Verify sublane-vectorized 5D StepC_write_kv_multi eager race/NaN Interpret parity & TPU7x AOT aliasing."""
    from unittest import mock

    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        write_decode_kv_merged_jax,
        write_decode_kv_multi_layer_jax,
    )

    ip_race = pltpu.InterpretParams(
        dma_execution_mode="eager",
        detect_races=True,
        uninitialized_memory="nan",
    )
    rng = np.random.default_rng(20262400 + n_layers + (10 if bd4 else 0))
    B, pps = 16, 4
    pack_factor = 4 if bd4 else 2
    phys_rows = 512 // pack_factor
    nb = B * pps + 4
    n_live = 11

    kvc_u8 = np.full((n_layers, nb, phys_rows, 1, 4, 128), 0x7F, dtype=np.uint8)
    kvl_np = np.array([10, 11, 12, 13, 512, 513, 514, 515, 256, 257, 300, 0, 0, 0, 0, 0], dtype=np.int32)
    pg_mat = rng.permutation(nb)[: B * pps].reshape(B, pps).astype(np.int32)
    for b_i in range(n_live):
        tok = int(kvl_np[b_i]) - 1
        pg = int(pg_mat[b_i, tok // 512])
        pr = (tok % 512) // pack_factor
        for s_prev in range(tok & (pack_factor - 1)):
            c_pair = (0 if s_prev < 2 else 2) if bd4 else (0 if s_prev == 0 else 2)
            l0, l1 = ((0, 64) if (s_prev & 1) == 0 else (64, 128)) if bd4 else (0, 128)
            kvc_u8[:, pg, pr, 0, c_pair : c_pair + 2, l0:l1] = rng.integers(
                0x30, 0x3D, size=(n_layers, 2, l1 - l0), dtype=np.uint8
            )
    caches_5d = tuple(
        jax.lax.bitcast_convert_type(jnp.asarray(kvc_u8[li], jnp.uint8), jnp.float8_e4m3fn)
        for li in range(n_layers)
    )
    kv_new_u8 = np.full((n_layers, B, 1, 4, 128), 0x7F, dtype=np.uint8)
    cols_valid = 64 if bd4 else 128
    kv_new_u8[:, :n_live, 0, :2, :cols_valid] = rng.integers(
        0x30, 0x3D, size=(n_layers, n_live, 2, cols_valid), dtype=np.uint8
    )
    kv_new_layers = jax.lax.bitcast_convert_type(
        jnp.asarray(kv_new_u8, jnp.uint8), jnp.float8_e4m3fn
    )
    kvl_j = jnp.asarray(kvl_np, jnp.int32)
    pg_j = jnp.asarray(pg_mat.reshape(-1), jnp.int32)
    cq = jnp.asarray(
        np.concatenate([np.arange(n_live + 1), np.full((B - n_live,), n_live)], dtype=np.int32)
    )
    ckv = jnp.arange(B + 1, dtype=jnp.int32) * (pps * 512)
    dist = jnp.asarray([n_live, n_live, n_live], jnp.int32)

    got = write_decode_kv_multi_layer_jax(
        kv_new_layers, caches_5d, kvl_j, pg_j, cq, ckv, dist,
        kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_race,
    )
    for li in range(n_layers):
        ref_li = write_decode_kv_merged_jax(
            kv_new_layers[li], caches_5d[li], kvl_j, pg_j, cq, ckv, dist,
            kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=True,
        )
        assert np.array_equal(
            np.asarray(ref_li).view(np.uint8), np.asarray(got[li]).view(np.uint8)
        ), f"5D sublane8 mismatch at layer {li} (n_layers={n_layers}, bd4={bd4})"

    # Verify q_len > 1 fallback parity cleanly matches per-layer writer.
    cq_mq = cq.at[1:].set(cq[1:] + 1)
    kvl_mq = kvl_j.at[0].set(102)
    got_mq = write_decode_kv_multi_layer_jax(
        kv_new_layers, caches_5d, kvl_mq, pg_j, cq_mq, ckv, dist,
        kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=ip_race,
    )
    ref_mq0 = write_decode_kv_merged_jax(
        kv_new_layers[0], caches_5d[0], kvl_mq, pg_j, cq_mq, ckv, dist,
        kv_pack_dense2=dense2, kv_pack_bd4=bd4, interpret=True,
    )
    assert np.array_equal(np.asarray(ref_mq0).view(np.uint8), np.asarray(got_mq[0]).view(np.uint8))

    # TPU7x AOT HLO input-output aliasing + zero cache copies for n_layers in {5, 8}.
    tpu_devs = topologies.get_topology_desc("tpu7x:2x2x2" if bd4 else "tpu7x:2x2x1", platform="tpu").devices
    mesh = Mesh(np.array(tpu_devs[:1]), ("x",))
    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    abs_args = (
        s((n_layers, 64, 1, 4, 128), jnp.float8_e4m3fn),
        tuple(s((nb, phys_rows, 1, 4, 128), jnp.float8_e4m3fn) for _ in range(n_layers)),
        s((64,), jnp.int32),
        s((64 * pps,), jnp.int32),
        s((65,), jnp.int32),
        s((65,), jnp.int32),
        s((3,), jnp.int32),
    )
    with mock.patch("sglang_tpu.attention.pallas.util.get_tpu_version", return_value=7):
        compiled = jax.jit(
            jax.shard_map(
                lambda kn, cs, kl, p, q, ck, d: write_decode_kv_multi_layer_jax(
                    kn, cs, kl, p, q, ck, d, kv_pack_dense2=dense2, kv_pack_bd4=bd4
                ),
                mesh=mesh,
                in_specs=(P(), tuple(P() for _ in range(n_layers)), P(), P(), P(), P(), P()),
                out_specs=tuple(P() for _ in range(n_layers)),
                check_vma=False,
            ),
            donate_argnums=(1,),
        ).lower(*abs_args).compile()
    hlo = compiled.as_text()
    for out_i in range(n_layers):
        assert f"{{{out_i}}}: ({1 + out_i}, {{}}, may-alias)" in hlo


def test_wave15_step_c_adaptive_1k_opt_in():
    """Verify SGLANG_TPU_MUSE_MK_ADAPTIVE_1K defaults False and enables _c1024_ strictly for B=64 512B."""
    import os
    from unittest import mock

    from sglang_tpu.envs import envs as tpu_envs

    assert tpu_envs.SGLANG_TPU_MUSE_MK_ADAPTIVE_1K.default is False

    hlo_def = _compile_step_c_aot(
        tp=8,
        B=64,
        pps=32,
        sw=-1,
        mode="w8,s4,batchfull,g4k,foldcol,batch4",
        dense2=False,
        bd4=False,
        dist_len=3,
        has_rope=False,
    )
    assert "_c1024_" not in hlo_def

    with mock.patch.dict(os.environ, {"SGLANG_TPU_MUSE_MK_ADAPTIVE_1K": "1"}):
        hlo_b64_512 = _compile_step_c_aot(
            tp=8,
            B=64,
            pps=32,
            sw=-1,
            mode="w8,s4,batchfull,g4k,foldcol,batch4",
            dense2=False,
            bd4=False,
            dist_len=3,
            has_rope=False,
        )
        assert "_c1024_" in hlo_b64_512

        hlo_b256_bd4 = _compile_step_c_aot(
            tp=16,
            B=256,
            pps=32,
            sw=-1,
            mode="w8,s4,batchfull,g4k,foldcol,batch4",
            dense2=False,
            bd4=True,
            dist_len=3,
            has_rope=False,
        )
        assert "_c1024_" not in hlo_b256_bd4

        hlo_b128_d2 = _compile_step_c_aot(
            tp=8,
            B=128,
            pps=32,
            sw=-1,
            mode="w8,s4,b1,batchfull,g4k,foldcol",
            dense2=True,
            bd4=False,
            dist_len=3,
            has_rope=False,
            allow_large_b=True,
        )
        assert "_c1024_" not in hlo_b128_d2


def test_wave16_step_c_bd4_swa_s3_b4_and_streamdyn_ring8k():
    """Verify Wave-16 Step (c) BD4 SWA s3_b4, streamdyn->ring8k auto-clamp, & shared64 s4 TPU7x AOT."""
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        resolve_step_c_rpa_plan,
    )

    # TP=16 B=256 BD4 8k1k ("tp16,w8,c1024,batchfull,dyn4k", pps=20) upgrades in Wave-17 to
    # group-vectorized Step (c) ring4k s3_b4 (3 * 4 * 8 * 128 == 12,288 <= 24*512).
    cfg_dyn_bd4, one_blk_dyn, slot_dyn, grp_dyn, ok_dyn = resolve_step_c_rpa_plan(
        "tp16,w8,c1024,batchfull,dyn4k",
        sliding_window=None,
        pages_per_seq=20,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=256,
        max_num_seqs=256,
    )
    assert (ok_dyn, one_blk_dyn, grp_dyn) == (True, False, True)
    assert (
        cfg_dyn_bd4.loop_kind,
        cfg_dyn_bd4.num_slots,
        cfg_dyn_bd4.chunk_sz,
        cfg_dyn_bd4.ring_group_size,
        slot_dyn,
    ) == (
        "ring4k",
        3,
        4096,
        4,
        32,
    )

    # TPU7x AOT (<63 MiB VMEM) for BD4 B=256 SWA s3_b4, BD4 B=256 dyn4k->ring8k, & D2 B=64 shared64 s4.
    hlo_swa_s3_b4 = _compile_step_c_aot(
        tp=16,
        B=256,
        pps=32,
        sw=2049,
        mode="tp16,w8,c1024,batchfull,dyn4k,batch4",
        dense2=False,
        bd4=True,
        dist_len=3,
        has_rope=True,
    )
    assert "muse_w9_b1_moe_b2_rpad_b256_tp16_" in hlo_swa_s3_b4

    hlo_dyn_ring8k = _compile_step_c_aot(
        tp=16,
        B=256,
        pps=20,
        sw=-1,
        mode="tp16,w8,c1024,batchfull,dyn4k",
        dense2=False,
        bd4=True,
        dist_len=3,
        has_rope=False,
    )
    assert "muse_w9_b1_moe_b2_rpad_b256_tp16_" in hlo_dyn_ring8k

    hlo_sh64_s4 = _compile_step_c_aot(
        tp=8,
        B=64,
        pps=516,
        sw=-1,
        mode="w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
        dense2=True,
        bd4=False,
        dist_len=4,
        has_rope=False,
    )
    assert "muse_w9_b1_moe_b2_rpad_b64_tp8_" in hlo_sh64_s4


def test_wave18_step_c_d2_swa_g4_and_shared64_shsub_aot():
    """Verify Wave-18 Step (c) D2/BD4 SWA grp=4 & shared64 q512/shsub2k routing + TPU7x <60/63 MiB AOT."""
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        resolve_step_c_rpa_plan,
    )

    cfg_swa_d2, one_swa_d2, slot_swa_d2, grp_swa_d2, ok_swa_d2 = resolve_step_c_rpa_plan(
        "w8,c1024,batchfull,dyn4k",
        sliding_window=2049,
        pages_per_seq=32,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=64,
        max_num_seqs=64,
    )
    assert (ok_swa_d2, one_swa_d2, grp_swa_d2) == (True, True, True)
    assert (cfg_swa_d2.loop_kind, cfg_swa_d2.num_slots, cfg_swa_d2.swabatch_group_size, cfg_swa_d2.fold_col, slot_swa_d2) == (
        "swabatch", 4, 4, True, 20
    )

    cfg_swa_bd4, _, _, grp_swa_bd4, ok_swa_bd4 = resolve_step_c_rpa_plan(
        "tp16,w8,c1024,batchfull,dyn4k",
        sliding_window=2049,
        pages_per_seq=32,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=256,
        max_num_seqs=256,
    )
    assert (ok_swa_bd4, grp_swa_bd4, cfg_swa_bd4.num_slots, cfg_swa_bd4.swabatch_group_size, cfg_swa_bd4.fold_col) == (
        True, True, 3, 4, True
    )

    cfg_sh_d2, _, _, _, ok_sh_d2 = resolve_step_c_rpa_plan(
        "w8,c4096,shared64,shsplit,pbf16,batchfull",
        sliding_window=None,
        pages_per_seq=516,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=64,
        max_num_seqs=64,
        shared_group_size=64,
        has_shared_prefix=True,
    )
    assert (ok_sh_d2, cfg_sh_d2.chunk_sz, cfg_sh_d2.shared_q_tile, cfg_sh_d2.sh_sub_chunk, cfg_sh_d2.fold_col, cfg_sh_d2.priv_group_size, cfg_sh_d2.hoist_new_kv) == (
        True, 8192, 1024, 2048, True, 1, True
    )

    cfg_sh_bd4, _, slot_sh_bd4, _, ok_sh_bd4 = resolve_step_c_rpa_plan(
        "tp16,w8,c4096,shared64,q256,pbf16,batchfull",
        sliding_window=None,
        pages_per_seq=516,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
        shared_group_size=64,
        has_shared_prefix=True,
    )
    assert (ok_sh_bd4, cfg_sh_bd4.chunk_sz, cfg_sh_bd4.shared_q_tile, cfg_sh_bd4.sh_sub_chunk, cfg_sh_bd4.fold_col, cfg_sh_bd4.priv_group_size, cfg_sh_bd4.hoist_new_kv, slot_sh_bd4) == (
        True, 8192, 512, 2048, True, 4, True, 32
    )

    for tp_a, b_a, pps_a, sw_a, d2_a, bd4_a, dist_a, rope_a, mode_a in (
        (8,  64, 32, 2049, True, False, 3, True, "w8,c1024,batchfull,dyn4k"),
        (16, 64, 32, 2049, False, True, 3, True, "tp16,w8,c1024,batchfull,dyn4k"),
        (16, 256, 32, 2049, False, True, 3, True, "tp16,w8,c1024,batchfull,dyn4k"),
        (8,  64, 516, -1, True, False, 4, False, "w8,c4096,shared64,shsplit,pbf16,batchfull"),
        (16, 64, 516, -1, False, True, 4, False, "tp16,w8,c4096,shared64,q256,pbf16,batchfull"),
    ):
        hlo = _compile_step_c_aot(
            tp=tp_a,
            B=b_a,
            pps=pps_a,
            sw=sw_a,
            mode=mode_a,
            dense2=d2_a,
            bd4=bd4_a,
            dist_len=dist_a,
            has_rope=rope_a,
        )
        assert f"muse_w9_b1_moe_b2_rpad_b{b_a}_tp{tp_a}_" in hlo


def test_wave17_step_c_ring_grp_and_priv_grp_aot_and_interpret():
    """Verify Wave-17 Step (c) ring4k_b4 / ring8k_b4 & shared64 priv_grp=4 routing + TPU7x <63 MiB AOT."""
    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        resolve_step_c_rpa_plan,
    )

    cfg_64_d2, one_64_d2, slot_64_d2, grp_64_d2, ok_64_d2 = resolve_step_c_rpa_plan(
        "w8,c1024,batchfull,dyn4k",
        sliding_window=None,
        pages_per_seq=20,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=64,
        max_num_seqs=64,
    )
    assert (ok_64_d2, one_64_d2, grp_64_d2) == (True, False, True)
    assert (cfg_64_d2.loop_kind, cfg_64_d2.num_slots, cfg_64_d2.chunk_sz, cfg_64_d2.ring_group_size, slot_64_d2) == (
        "ring4k", 4, 4096, 4, 32
    )

    cfg_64_bd4, one_64_bd4, slot_64_bd4, grp_64_bd4, ok_64_bd4 = resolve_step_c_rpa_plan(
        "tp16,w8,c1024,batchfull,dyn4k",
        sliding_window=None,
        pages_per_seq=20,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
    )
    assert (ok_64_bd4, one_64_bd4, grp_64_bd4) == (True, False, True)
    assert (cfg_64_bd4.loop_kind, cfg_64_bd4.num_slots, cfg_64_bd4.chunk_sz, cfg_64_bd4.ring_group_size, slot_64_bd4) == (
        "ring4k", 3, 8192, 4, 64
    )

    cfg_8_d2, _, slot_8_d2, grp_8_d2, ok_8_d2 = resolve_step_c_rpa_plan(
        "w8,c1024,batchfull,dyn4k",
        sliding_window=None,
        pages_per_seq=20,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=8,
        max_num_seqs=8,
    )
    assert ok_8_d2 is True and grp_8_d2 is False and cfg_8_d2.ring_group_size == 1 and slot_8_d2 == 24

    cfg_sh_bd4, _, slot_sh_bd4, _, ok_sh_bd4 = resolve_step_c_rpa_plan(
        "tp16,w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
        sliding_window=None,
        pages_per_seq=516,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
        shared_group_size=64,
        has_shared_prefix=True,
    )
    assert ok_sh_bd4 is True and (cfg_sh_bd4.priv_group_size, cfg_sh_bd4.priv_blk_pages, slot_sh_bd4) == (4, 8, 32)

    cfg_sh_d2, _, slot_sh_d2, _, ok_sh_d2 = resolve_step_c_rpa_plan(
        "w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol",
        sliding_window=None,
        pages_per_seq=516,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=64,
        max_num_seqs=64,
        shared_group_size=64,
        has_shared_prefix=True,
    )
    assert ok_sh_d2 is True and (cfg_sh_d2.priv_group_size, cfg_sh_d2.priv_blk_pages, slot_sh_d2) == (1, 0, 16)

    for tp_a, b_a, pps_a, d2_a, bd4_a, dist_a, mode_a in (
        (8, 64, 20, True, False, 3, "w8,c1024,batchfull,dyn4k"),
        (16, 64, 20, False, True, 3, "tp16,w8,c1024,batchfull,dyn4k"),
        (16, 64, 516, False, True, 4, "tp16,w8,c4096,shared64,coal,noz,pbf16,q1024,shfp8,batchfull,foldcol"),
    ):
        hlo = _compile_step_c_aot(
            tp=tp_a,
            B=b_a,
            pps=pps_a,
            sw=-1,
            mode=mode_a,
            dense2=d2_a,
            bd4=bd4_a,
            dist_len=dist_a,
            has_rope=False,
        )
        assert f"muse_w9_b1_moe_b2_rpad_b{b_a}_tp{tp_a}_" in hlo



def test_wave18_embed_l0_rpad_interpret_parity_and_tpu7x_aot():
    """Verify Fused L0 (chip-sharded & replicated B=8/16) under on_wait+races+nan and TPU7x AOT."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental import topologies
    from jax.experimental.pallas import tpu as pltpu
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.attention.pallas import rpa_v3_decode_fast as rpad
    from sglang_tpu.kernels import decode_collective_bridges_w8 as dcb8
    from sglang_tpu.muse_megakernel.decode.embed_l0_rpad_kernel import (
        embed_l0_rpad_chip_sharded_jax,
        embed_l0_rpad_replicated_b16_jax,
    )

    cpu_devs = jax.devices("cpu")
    if len(cpu_devs) >= 8:
        mesh_cpu = Mesh(np.array(cpu_devs[:8]), ("model",))
        ip = pltpu.InterpretParams(
            dma_execution_mode="on_wait",
            detect_races=True,
            uninitialized_memory="nan",
        )
        ws, hidden = 8, 1024
        in_sp = (
            P(), P("model", None), P(), P("model", None), P(), P("model", None), P(),
            P(), P(), P(), P(), P(), P(), P(),
        )
        rmode = "w8,s2,c2048,coal,noz,pbf16,batchfull,b2"

        for B, is_rep, v_shard, d2 in (
            (32, False, 18, True),
            (64, False, 24, True),
            (8, True, 18, True),
            (16, True, 24, False),
        ):
            rng = np.random.default_rng(20261800 + B + v_shard)
            ids_np = rng.integers(0, ws * v_shard, size=(B,), dtype=np.int32)
            ids_np[0] = v_shard - 1
            ids_np[1] = v_shard - 2
            ids_np[2] = 3 * v_shard + (v_shard - 1)
            ids_np[3] = 7 * v_shard + (v_shard - 2)
            emb_bf16 = jnp.asarray(
                rng.standard_normal((ws * v_shard, hidden)) * 0.1, dtype=jnp.bfloat16
            )
            l0_gain = jnp.asarray(rng.uniform(0.8, 1.2, (hidden,)), dtype=jnp.float32)
            w_qkv = jnp.asarray(
                rng.standard_normal((ws * hidden, 1280)) * 0.05, dtype=jnp.float8_e4m3fn
            )
            s_qkv = jnp.full((1280,), 0.02, dtype=jnp.float32)
            w_gate = jnp.asarray(
                rng.standard_normal((ws * hidden, 1024)) * 0.05, dtype=jnp.float8_e4m3fn
            )
            s_gate = jnp.full((1024,), 0.02, dtype=jnp.float32)
            cs = jnp.asarray(rng.standard_normal((B, 2, 128)) * 0.5, dtype=jnp.float32)
            phys = 256 if d2 else 512
            kv_c = jnp.asarray(
                rng.standard_normal((B * 2, phys, 1, 4, 128)) * 0.05,
                dtype=jnp.float8_e4m3fn,
            )
            kvl = jnp.full((B,), 64, dtype=jnp.int32)
            pidx = jnp.arange(B * 2, dtype=jnp.int32)
            cuq = jnp.arange(B + 1, dtype=jnp.int32)
            cukv = jnp.arange(B + 1, dtype=jnp.int32) * 64
            dist = jnp.asarray([B, B, B], dtype=jnp.int32)
            ids_j = jnp.asarray(ids_np, dtype=jnp.int32)

            def _ref(ids, emb, gain, wq, sq, wg, sg, kvc, kl, pi, cq, ckv, dt, csin):
                if is_rep:
                    st_r = dcb8.decode_vocab_embed_norm_jax(
                        ids, emb, world_size=ws, norm_eps=1e-5, interpret=ip
                    )
                    inv1 = jax.lax.rsqrt(jnp.mean(st_r * st_r, axis=-1, keepdims=True) + 1e-5)
                    ain = ((st_r * inv1) * gain[None, :]).astype(jnp.bfloat16)
                else:
                    st_r, ain = dcb8.decode_vocab_embed_chip_sharded_jax(
                        ids, emb, gain, world_size=ws, embed_eps=1e-5, l0_eps=1e-5,
                        l0_scale_input=False, interpret=ip,
                    )
                qkv_r = (
                    jnp.matmul(ain.astype(jnp.float32), wq.astype(jnp.float32)) * sq[None, :]
                ).astype(jnp.bfloat16)
                g_r = (
                    jnp.matmul(ain.astype(jnp.float32), wg.astype(jnp.float32)) * sg[None, :]
                ).astype(jnp.bfloat16)

                def _ek(qr, csr, qor, kvor):
                    dcb8._run_in_vmem_qkv_epilogue_dispatch(
                        qr, csr, qor, kvor, norm_eps=1e-5, query_scale=1.0,
                        k_scale=1.0, v_scale=1.0, has_rope=True, interpret_mode=True,
                    )

                q_e, kv_e = pl.pallas_call(
                    _ek,
                    out_shape=(
                        jax.ShapeDtypeStruct((B, 16, 128), jnp.bfloat16),
                        jax.ShapeDtypeStruct((B, 1, 4, 128), jnp.float8_e4m3fn),
                    ),
                    interpret=ip,
                )(qkv_r, csin)
                out_r, _ = rpad.run_decode_fast_rpa(
                    q_e.reshape(1, B, 8, 2, 128), kv_e, None, kvc, kl, pi, cq, ckv, dt,
                    sm_scale=0.125, sliding_window=2048, k_scale=1.0, v_scale=1.0,
                    mode_str=rmode, kv_pack_dense2=d2, gate=g_r, out_norm_eps=1e-5,
                    fused_out_pack=2, prepared_inputs=True, interpret=True,
                )
                return st_r, out_r, kv_e

            def _fus(ids, emb, gain, wq, sq, wg, sg, kvc, kl, pi, cq, ckv, dt, csin):
                fn = embed_l0_rpad_replicated_b16_jax if is_rep else embed_l0_rpad_chip_sharded_jax
                return fn(
                    ids, emb, gain, wq, sq, wg, sg, kvc, kl, pi, cq, ckv, dt, csin,
                    world_size=ws, embed_eps=1e-5, l0_eps=1e-5, l0_scale_input=False,
                    qkv_norm_eps=1e-5, query_scale=1.0, k_scale=1.0, v_scale=1.0,
                    rpa_sm_scale=0.125, rpa_sliding_window=2048, rpa_k_scale=1.0,
                    rpa_v_scale=1.0, rpa_out_norm_eps=1e-5, rpa_mode_str=rmode,
                    kv_pack_dense2=d2, kv_pack_bd4=False, skip_kv_write=True, interpret=ip,
                )

            out_sp = (P() if is_rep else P("model", None), P(), P())
            args = (ids_j, emb_bf16, l0_gain, w_qkv, s_qkv, w_gate, s_gate, kv_c, kvl, pidx, cuq, cukv, dist, cs)
            r_st, r_g, r_kv = jax.jit(
                jax.shard_map(_ref, mesh=mesh_cpu, in_specs=in_sp, out_specs=out_sp, check_vma=False)
            )(*args)
            f_st, f_g, f_kv = jax.jit(
                jax.shard_map(_fus, mesh=mesh_cpu, in_specs=in_sp, out_specs=out_sp, check_vma=False)
            )(*args)
            assert np.max(np.abs(np.asarray(r_st, np.float32) - np.asarray(f_st, np.float32))) == 0.0
            rf, ff = np.asarray(r_g, np.float64).ravel(), np.asarray(f_g, np.float64).ravel()
            g_cos = float(np.dot(rf, ff) / (np.linalg.norm(rf) * np.linalg.norm(ff) + 1e-30))
            assert g_cos > 0.99999, f"Fused L0 gated_cos={g_cos} for B={B}, rep={is_rep}"
            assert np.array_equal(np.asarray(r_kv).view(np.uint8), np.asarray(f_kv).view(np.uint8))

    # TPU7x AOT sweep across TP=16 (BD4) and TP=8 (D2/512B), B=8..256, incl. tail v_shard % 8 == 2 and 6D layer_idx.
    tpu_222 = topologies.get_topology_desc("tpu7x:2x2x2", platform="tpu").devices
    tpu_221 = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
    mesh16 = Mesh(np.array(sorted(tpu_222, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))), ("model",))
    mesh8 = Mesh(np.array(sorted(tpu_221, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))), ("model",))
    for ws, mesh, B, is_rep, d2, bd4, v_sh, rmode in (
        (16, mesh16, 256, False, False, True, 16384, "w8,s4,c2048,coal,noz,pbf16,batchfull,b4"),
        (16, mesh16, 64, False, False, True, 16386, "w8,s4,c2048,coal,noz,pbf16,batchfull,b4"),
        (16, mesh16, 16, True, False, True, 16384, "w8,s4,c2048,coal,noz,pbf16,batchfull,b4"),
        (8, mesh8, 256, False, True, False, 32768, "w8,s3,c2048,coal,noz,pbf16,batchfull,b2"),
        (8, mesh8, 128, False, True, False, 32768, "w8,s4,c2048,coal,noz,pbf16,batchfull,b4"),
        (8, mesh8, 32, False, True, False, 32770, "w8,s4,c2048,coal,noz,pbf16,batchfull,b4"),
        (8, mesh8, 8, True, False, False, 32768, "w8,s4,c2048,coal,noz,pbf16,batchfull,b4"),
    ):
        qkv_c = 1280 if ws == 8 else 640
        gate_c = 1024 if ws == 8 else 512
        phys = 256 if d2 else 512
        shapes = (
            jax.ShapeDtypeStruct((B,), jnp.int32),
            jax.ShapeDtypeStruct((v_sh, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((8192,), jnp.float32),
            jax.ShapeDtypeStruct((8192, qkv_c), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((qkv_c,), jnp.float32),
            jax.ShapeDtypeStruct((8192, gate_c), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((gate_c,), jnp.float32),
            jax.ShapeDtypeStruct((32, 45, phys, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((B,), jnp.int32),
            jax.ShapeDtypeStruct((B * 8,), jnp.int32),
            jax.ShapeDtypeStruct((B + 1,), jnp.int32),
            jax.ShapeDtypeStruct((B + 1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
            jax.ShapeDtypeStruct((B, 2, 128), jnp.float32),
            jax.ShapeDtypeStruct((), jnp.int32),
        )
        fn = embed_l0_rpad_replicated_b16_jax if is_rep else embed_l0_rpad_chip_sharded_jax
        sm = jax.jit(
            jax.shard_map(
                lambda *a, _fn=fn, _ws=ws, _rm=rmode, _d2=d2, _bd4=bd4: _fn(
                    *a[:14], layer_idx=a[14], world_size=_ws, rpa_sliding_window=2048,
                    rpa_mode_str=_rm, kv_pack_dense2=_d2, kv_pack_bd4=_bd4, skip_kv_write=True,
                ),
                mesh=mesh, in_specs=(P(),) * len(shapes), out_specs=(P(), P(), P()), check_vma=False,
            )
        )
        assert sm.trace(*shapes).lower(lowering_platforms=("tpu",)).compile() is not None


def test_wave18_tp8_b256_stepc_split34_l61_xy_ag_and_b8_shared8_aot():
    """Verify TP=8 B=256 Step(c) (flat SWA D2 + nested Scope-3/4 Full D2 & SWA 512B), L61 XY AG, & B=8 shared8 AOT."""
    import jax
    import jax.numpy as jnp
    from jax.experimental import topologies
    from jax.sharding import Mesh, PartitionSpec as P

    from sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel import (
        b1_moe_fnorm_chip_sharded_jax,
        resolve_step_c_rpa_plan,
    )

    # Planner checks: TP=8 B=256 SWA D2, Full D2 ring4k, SWA 512B, and B=8 shared8 c8192 (512B/D2/BD4).
    _, _, _, _, ok_256_swa_d2 = resolve_step_c_rpa_plan(
        "w8,s3,c2048,coal,noz,pbf16,batchfull,b2",
        sliding_window=2049, pages_per_seq=8, phys_page_rows=256,
        kv_pack_dense2=True, b_tokens=256, max_num_seqs=256, allow_large_b=True,
    )
    assert ok_256_swa_d2 is True
    cfg_256_full_d2, _, slot_256_full_d2, grp_256_full_d2, ok_256_full_d2 = resolve_step_c_rpa_plan(
        "w8,s4,c4096,coal,noz,pbf16,ring4k,rb4",
        sliding_window=None, pages_per_seq=20, phys_page_rows=256,
        kv_pack_dense2=True, b_tokens=256, max_num_seqs=256, allow_large_b=True,
    )
    assert (
        ok_256_full_d2 is True
        and grp_256_full_d2 is True
        and cfg_256_full_d2.ring_group_size == 2
        and cfg_256_full_d2.num_slots * slot_256_full_d2 * 256 > 16 * 512
    )
    _, _, _, _, ok_256_swa_512 = resolve_step_c_rpa_plan(
        "w8,s2,c2048,coal,noz,pbf16,batchfull,b2",
        sliding_window=2049, pages_per_seq=8, phys_page_rows=512,
        kv_pack_dense2=False, b_tokens=256, max_num_seqs=256, allow_large_b=True,
    )
    assert ok_256_swa_512 is True
    for d2_sh, bd4_sh, phys_sh in ((False, False, 512), (True, False, 256), (False, True, 128)):
        _, _, _, _, ok_sh8 = resolve_step_c_rpa_plan(
            "w8,s4,c8192,shared8,coal,noz,pbf16,shfp8,batchfull",
            sliding_window=None, pages_per_seq=516, phys_page_rows=phys_sh,
            kv_pack_dense2=d2_sh, kv_pack_bd4=bd4_sh, b_tokens=8, max_num_seqs=8,
            shared_group_size=8, has_shared_prefix=True,
        )
        assert ok_sh8 is True

    # TPU7x AOT compile: TP=8 B=256 flat SWA D2, nested Scope-3/4 Full D2 ring4k rb4, nested Scope-3/4 SWA 512B.
    for sw_a, d2_a, mode_a, pps_a in (
        (2049, True, "w8,s3,c2048,coal,noz,pbf16,batchfull,b2", 8),
        (-1, True, "w8,s4,c4096,coal,noz,pbf16,ring4k,rb4", 20),
        (2049, False, "w8,s2,c2048,coal,noz,pbf16,batchfull,b2", 8),
    ):
        hlo = _compile_step_c_aot(
            tp=8, B=256, pps=pps_a, sw=sw_a, mode=mode_a, dense2=d2_a, bd4=False,
            dist_len=3, has_rope=(sw_a > 0), allow_large_b=True,
        )
        assert "muse_w9_b1_moe_b2_rpad_b256_tp8_" in hlo

    # TPU7x AOT compile: TP=8 B=128 & B=256 Fused L61 (2D-XY AG 2-sem borrow + out_hidden_ref Scope-1 alias).
    tpu_221 = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
    mesh8 = Mesh(np.array(sorted(tpu_221, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))), ("model",))
    for B_l61 in (128, 256):
        r_pair = B_l61 // 4
        shapes = (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((r_pair, 8192), jnp.float32),
            jax.ShapeDtypeStruct((B_l61, 1024), jnp.bfloat16),
            jax.ShapeDtypeStruct((1024, 8192), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((8192, 2048), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 2048), jnp.float32),
            jax.ShapeDtypeStruct((8192, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((1, 256), jnp.float32),
            jax.ShapeDtypeStruct((32, 4096, 8192), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((32, 8, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((32, 4096, 4096), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((32, 8, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((4096, 4096), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
        )
        sm = jax.jit(
            jax.shard_map(
                lambda *a: b1_moe_fnorm_chip_sharded_jax(
                    *a[:24], free_params=a[24], world_size=8, replica_cost_mode="tiles"
                ),
                mesh=mesh8, in_specs=(P(),) * len(shapes), out_specs=P(), check_vma=False,
            )
        )
        assert sm.trace(*shapes).lower(lowering_platforms=("tpu",)).compile() is not None


def test_wave18_fuse_l0_and_fuse_l61_ops_and_killswitches():
    """Verify FUSE_L0/FUSE_L61 env defaults, kill-switch gating, L0 op prewarm, and 45-layer [29, 16] flush."""
    from unittest import mock

    import torch
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.models.muse_spark import MuseSparkDecoderLayer, MuseSparkTextModel
    from sglang_tpu.muse_megakernel.decode import b1_moe_ops

    assert bool(tpu_envs.SGLANG_TPU_MUSE_DECODE_FUSE_L61.get()) is True
    assert bool(tpu_envs.SGLANG_TPU_MUSE_DECODE_FUSE_L0.get()) is True

    # Verify _maybe_forward_embed_l0_rpad cleanly returns None when kill-switch or ready flag is off.
    m = object.__new__(MuseSparkTextModel)
    m.__dict__["_w18_fuse_l0_rpad_ready"] = True
    m.__dict__["_w18_fuse_l0_rep_b16_ready"] = True
    with mock.patch.object(tpu_envs.SGLANG_TPU_MUSE_DECODE_FUSE_L0, "get", return_value=False):
        assert m._maybe_forward_embed_l0_rpad(
            input_ids=torch.zeros((64,), dtype=torch.int32),
            positions=torch.zeros((64,), dtype=torch.int32),
            forward_batch=None,
            cos_sin_step=None,
            n_tok=64,
            tp=8,
            replicated=False,
            pending_step_c_kv_writes=[],
        ) is None

    # Verify _can_run_b1_moe_superkernel allows TP=8 B=256 when sparse MoE batches include 256.
    from types import SimpleNamespace

    lyr = object.__new__(MuseSparkDecoderLayer)
    lyr.__dict__["_w9_b1_moe_b2_superkernel"] = True
    fake_mlp = type("FakeMLP", (), {})()
    fake_mlp.__dict__["_fused_decode_mode"] = "sparse"
    fake_mlp.__dict__["_fused_decode_op"] = object()
    fake_mlp.__dict__["_fused_decode_batches"] = (32, 64, 128, 256)
    fake_mlp._buffers = {
        "_w8_wpre_pair_w8k": torch.empty((1,)),
        "_w7_wrouter_km": torch.empty((1,)),
        "_decode_fp4_w1": SimpleNamespace(dtype=torch.float4_e2m1fn_x2, ndim=3, shape=(32, 4096, 4096)),
        "_decode_fp4_s1": SimpleNamespace(dtype=torch.bfloat16, ndim=3, shape=(32, 8, 8192)),
        "_decode_fp4_w2": SimpleNamespace(dtype=torch.float4_e2m1fn_x2, ndim=3, shape=(32, 4096, 2048)),
        "_decode_fp4_s2": SimpleNamespace(dtype=torch.bfloat16, ndim=3, shape=(32, 8, 4096)),
    }
    fake_mlp._bias_2d = lambda: SimpleNamespace(dtype=torch.float32)
    fake_mlp._norm_2d = lambda: SimpleNamespace(dtype=torch.bfloat16)
    fake_mlp.experts = None
    lyr.__dict__["_modules"] = {"mlp": fake_mlp}
    assert lyr._can_run_b1_moe_superkernel(256, tp=8, is_dec_step=True, replicated=False) is True

    # Verify FUSE_L61=False disables both replicated L61 and TP=8 B=256 chip-sharded L61 cleanly
    # (never falling into TP=8 B<=128 Step (a) b1_moe_kernel).
    fake_mlp._buffers["_w7_wpost_w8k"] = torch.empty((1,))
    fake_mlp._buffers["_w9_wpost_pair_w8k"] = torch.empty((1,))
    fake_mlp._free_params_for_batch = lambda _b: None
    with mock.patch.object(tpu_envs.SGLANG_TPU_MUSE_DECODE_FUSE_L61, "get", return_value=False):
        assert lyr._try_b1_moe_replicated_b16_superkernel(
            positions=None, stream=None, attn_input=None, forward_batch=None,
            n_tok=8, tp=8, b2_kind=1, is_dec_step=True,
            pa_alpha_arg=None, pa_beta_arg=None, pre_ff_gain=None, pre_exp_gain=None,
            cos_sin_step=None, cs_kw={}, precomputed_qkv_gate=None,
            precomputed_prepared_qkvg=None, precomputed_gated=None,
            next_input_layernorm=None, next_layer_attn=None, final_layernorm=object(),
        ) is None
        assert lyr._try_b1_moe_superkernel(
            positions=None, stream=None, attn_input=None, forward_batch=None,
            n_tok=256, tp=8, b2_kind=1, is_dec_step=True,
            pa_alpha_arg=None, pa_beta_arg=None, pre_ff_gain=None, pre_exp_gain=None,
            cos_sin_step=None, cs_kw={}, precomputed_qkv_gate=None,
            precomputed_prepared_qkvg=None, precomputed_gated=None,
            next_input_layernorm=None, next_layer_attn=None, final_layernorm=object(),
        ) is None

    # Verify prewarm_b1_moe_superkernel_ops populates both L0 caches when include_fuse_l0=True.
    exported: list[str] = []
    with (
        mock.patch.dict(b1_moe_ops._EMBED_L0_RPAD_OPS, {}, clear=True),
        mock.patch.dict(b1_moe_ops._EMBED_L0_RPAD_B16_OPS, {}, clear=True),
        mock.patch.object(
            b1_moe_ops,
            "_export_superkernel_op",
            side_effect=lambda name, *a, **kw: exported.append(name) or (lambda *x: x),
        ),
        mock.patch.object(b1_moe_ops, "get_step_c_kv_write_op", return_value=lambda *x: None),
        mock.patch.object(
            b1_moe_ops,
            "get_step_c_kv_write_multi_5d_op",
            side_effect=lambda k, **kw: (exported.append(f"kv5d_{k}") or (lambda *x: None)),
        ),
        mock.patch.object(b1_moe_ops, "get_step_c_kv_write_multi_6d_op", return_value=lambda *x: None),
    ):
        b1_moe_ops.prewarm_b1_moe_superkernel_ops(
            16,
            [(0, False, "tiles", True)],
            bridge_qkv_epi=True,
            has_rope_variants=(True,),
            include_rep_b16=True,
            include_fuse_l0=True,
            step_c_configs=((False, False, "w8,s4", True),),
        )
        assert len(b1_moe_ops._EMBED_L0_RPAD_OPS) == 1
        assert len(b1_moe_ops._EMBED_L0_RPAD_B16_OPS) == 1
        assert any("muse_w18_embed_l0_rpad_tp16_bd4" in n for n in exported)
        assert any("muse_w18_embed_l0_rpad_rep_b16_tp16_bd4" in n for n in exported)
        assert "kv5d_17" in exported and "kv5d_29" in exported

    # Verify 45 SWA layers flush as [29, 16] and 46 SWA layers (Fused L0 + 45 Step c SWA) flush as [29, 17].
    meta = tuple(torch.zeros((1,), dtype=torch.int32) for _ in range(5))
    kv0 = torch.zeros((16, 1, 4, 128), dtype=torch.float8_e4m3fn)
    for n_swa, want_chunks in ((45, [29, 16]), (46, [29, 17])):
        chunk_sizes: list[int] = []
        entries = [
            (torch.zeros((2, 128, 1, 4, 128), dtype=torch.float8_e4m3fn), *meta, None, kv0, False, True, (False, True, 0, False, True), li)
            for li in range(n_swa)
        ]
        with mock.patch.object(
            b1_moe_ops,
            "get_step_c_kv_write_multi_5d_op",
            side_effect=lambda k, **kw: (chunk_sizes.append(k) or (lambda *a: None)),
        ):
            b1_moe_ops.flush_pending_step_c_kv_writes(entries)
        assert chunk_sizes == want_chunks



def test_wave18_w31_safe_wm_stream3_and_tp8_b256_hoistkv_split34_aot():
    """Verify safe_wm delegates stream3/half_n from default_sparse_decode_config & TP=8 B=256 hoistkv Split34 AOT."""
    import dataclasses
    from unittest import mock

    import jax
    import jax.numpy as jnp

    from sglang_tpu.attention.pallas import rpa_v3_decode_fast as rpad
    from sglang_tpu.muse_megakernel.decode import b1_moe_b2_kernel as mkb

    # 1. Verify _prepare_chip_b1_moe_b2_setup directly honors base_cfg.weight_mode in ("stream3", "half_n")
    #    at B=128/256 (unlocking dec-moe-w29 stream3 inside Step (c)) while guarding unsupported modes.
    orig_def_cfg = mkb.default_sparse_decode_config
    s = lambda sh, dt: jax.ShapeDtypeStruct(sh, dt)
    for B_wm, ws_wm, want_wm, mock_wm in (
        (128, 8, "stream3", "stream3"),
        (256, 16, "stream3", "stream3"),
        (256, 16, "half_n", "half_n"),
        (128, 8, "half_n", "stream3_k2"),
        (64, 8, "stream3", "stream3_k2"),
    ):
        r_p = B_wm // (ws_wm // 2)
        g_c = 512 if ws_wm == 16 else 1024
        q_c = 640 if ws_wm == 16 else 1280
        e_l = 256 // ws_wm
        with mock.patch.object(
            mkb,
            "default_sparse_decode_config",
            side_effect=lambda b, ep, sb, _m=mock_wm: dataclasses.replace(
                orig_def_cfg(b, ep, sb), weight_mode=_m
            ),
        ):
            seen_wm: list[str] = []
            args_abs = (
                s((1,), jnp.int32), s((r_p, 8192), jnp.float32), s((B_wm, g_c), jnp.bfloat16),
                s((g_c, 8192), jnp.float8_e4m3fn), s((8192,), jnp.float32),
                s((8192,), jnp.float32), s((8192,), jnp.float32), s((8192,), jnp.float32),
                s((8192, 2048), jnp.float8_e4m3fn), s((2048,), jnp.float32),
                s((8192, 256), jnp.bfloat16), s((4096,), jnp.float32),
                s((1, 256), jnp.float32),
                s((e_l, 512, 8192), jnp.uint32), s((e_l, 8, 8192), jnp.bfloat16),
                s((e_l, 512, 4096), jnp.uint32), s((e_l, 8, 4096), jnp.bfloat16),
                s((4096,), jnp.float32),
                s((4096, 4096), jnp.float8_e4m3fn), s((4096,), jnp.float32),
                s((8192,), jnp.float32), s((8192,), jnp.float32), s((8192,), jnp.float32),
                s((8192,), jnp.float32),
                s((8192, q_c), jnp.float8_e4m3fn), s((q_c,), jnp.float32),
                s((8192, g_c), jnp.float8_e4m3fn), s((g_c,), jnp.float32),
            )

            def _trace_setup(*a, _ws=ws_wm):
                st = mkb._prepare_chip_b1_moe_b2_setup(
                    *a, None, None, None, None, None, None, None, None,
                    world_size=_ws, fuse_qkv_epi=False, config=None, interpret=False,
                )
                seen_wm.append(st["config"].weight_mode)
                return jnp.int32(0)

            jax.eval_shape(_trace_setup, *args_abs)
            assert seen_wm == [want_wm]

    # 2. TPU7x AOT compile <63 MiB:
    #    (a) TP=16 B=256 BD4 Step (c) SWA 2K & Full prod_mode (hoistkv) with weight_mode="stream3"
    #    (b) TP=8 B=256 Full D2 & SWA D2 with production hoistkv (w8,noadapt,hoistkv,g4k,foldcol,batch2,batch4),
    #        verifying split_b2_rpad avoids both Full hoistkv (67.35 MiB) and SWA s=4 (63.01 MiB) VMEM overflows.
    orig_parse = rpad.parse_decode_fast_config
    prod_mode = "w8,noadapt,hoistkv,g4k,foldcol,batch2,batch4"
    with mock.patch.object(
        mkb,
        "default_sparse_decode_config",
        side_effect=lambda b, ep, sb: dataclasses.replace(
            orig_def_cfg(b, ep, sb), weight_mode="stream3"
        ),
    ):
        hlo_16_s3 = _compile_step_c_aot(
            tp=16, B=256, pps=8, sw=2049,
            mode="w8,s4,c2048,coal,noz,pbf16,batchfull,b4",
            dense2=False, bd4=True, dist_len=3, has_rope=True, allow_large_b=True,
        )
        assert "muse_w9_b1_moe_b2_rpad_b256_tp16_" in hlo_16_s3
        with mock.patch.object(
            rpad,
            "parse_decode_fast_config",
            side_effect=lambda *a, **kw: orig_parse(*a, **{**kw, "b_tokens": 16}),
        ):
            hlo_16_full_s3_hk = _compile_step_c_aot(
                tp=16, B=256, pps=32, sw=-1, mode=prod_mode,
                dense2=False, bd4=True, dist_len=3, has_rope=False, allow_large_b=True,
            )
            assert "muse_w9_b1_moe_b2_rpad_b256_tp16_" in hlo_16_full_s3_hk

    with mock.patch.object(
        rpad,
        "parse_decode_fast_config",
        side_effect=lambda *a, **kw: orig_parse(*a, **{**kw, "b_tokens": 16}),
    ):
        hlo_8_full_hk = _compile_step_c_aot(
            tp=8, B=256, pps=32, sw=-1, mode=prod_mode,
            dense2=True, bd4=False, dist_len=3, has_rope=False, allow_large_b=True,
        )
        assert "muse_w9_b1_moe_b2_rpad_b256_tp8_" in hlo_8_full_hk

    hlo_8_swa_hk = _compile_step_c_aot(
        tp=8, B=256, pps=8, sw=2049, mode=prod_mode,
        dense2=True, bd4=False, dist_len=3, has_rope=True, allow_large_b=True,
    )
    assert "muse_w9_b1_moe_b2_rpad_b256_tp8_" in hlo_8_swa_hk


def test_wave19_w32_b3_s12_wpre_hoist_unsplit_and_bkv_in_ici_aot():
    """Verify 3-scope s12 scoping, w_pre/w_qkv/w_gate prestart hoisting, exact Scope-3 slicing, and 10 MiB split guard."""
    import dataclasses
    from unittest import mock

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import sglang_tpu.attention.pallas.rpa_v3_decode_fast as rpad
    import sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel as mkb

    orig_def_cfg = mkb.default_sparse_decode_config
    orig_parse = rpad.parse_decode_fast_config
    prod_mode = "w8,noadapt,hoistkv,g4k,foldcol,batch2,batch4"

    # 1. Verify s12 tiering, w_pre/w_gate hoisting, and exact Scope-3 VMEM sizing (no exp_3d / mate_recv bloat).
    for ws, qkv_c, gate_c, fnorm in ((8, 1280, 1024, False), (16, 640, 512, False), (8, 0, 0, True)):
        r_pair = 256 // (ws // 2)
        o_in = 1024 if ws == 8 else 512
        e_loc = 256 // ws
        args_abs = (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((r_pair, 8192), jnp.float32),
            jax.ShapeDtypeStruct((256, o_in), jnp.bfloat16),
            jax.ShapeDtypeStruct((o_in, 8192), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((8192, 2048), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 2048), jnp.float32),
            jax.ShapeDtypeStruct((8192, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((1, 256), jnp.float32),
            jax.ShapeDtypeStruct((e_loc, 4096, 8192), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((e_loc, 8, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((e_loc, 4096, 4096), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((e_loc, 8, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((4096, 4096), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
        )
        if not fnorm:
            args_abs = args_abs + (
                jax.ShapeDtypeStruct((8192, qkv_c), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((1, qkv_c), jnp.float32),
                jax.ShapeDtypeStruct((8192, gate_c), jnp.float8_e4m3fn),
                jax.ShapeDtypeStruct((1, gate_c), jnp.float32),
            )
        st_box = []

        def _trace_b3(*a, _ws=ws, _fn=fnorm):
            tail = (None, None, None, None) if _fn else a[24:28]
            st_box.append(
                mkb._prepare_chip_b1_moe_b2_setup(
                    *a[:24], *tail, None, None, None, None, None, None, None, None,
                    world_size=_ws, fuse_qkv_epi=(not _fn), config=None, interpret=False,
                    has_rpad=(not _fn), final_norm_only=_fn,
                )
            )
            return jnp.int32(0)

        jax.eval_shape(_trace_b3, *args_abs)
        st = st_box[0]
        want_hoist = (not fnorm) and (qkv_c + gate_c <= 2048)
        assert st["use_3scope"] is True
        assert st["step_b3_hoist_w_pre"] is want_hoist
        assert st["step_b3_hoist_w_gate"] is want_hoist
        assert 1 in st["step_b3_moe_scope_tier"] and len(st["step_b3_s12_shapes"]) >= 5
        s3_shapes = [tuple(sp.shape) for sp in st["step_b3_scope3_shapes"]]
        if fnorm:
            assert s3_shapes[:2] == [(r_pair, 4096), (256 // ws, 4096)]
            assert len(s3_shapes) == 5
        else:
            assert s3_shapes[:4] == [(256, 8192), (256, qkv_c), (r_pair, 4096), (256 // ws, 4096)]
            assert len(s3_shapes) == (7 if want_hoist else 9)  # no exp_3d; w_qkv/w_gate hoisted on TP=16

    # 2. TPU7x AOT <63 MiB across unsplit (<=10 MiB bkv+init_sv) and split (>10 MiB, pps=18 8K hk) paths + Step (b) B=256.
    with mock.patch.object(
        mkb,
        "default_sparse_decode_config",
        side_effect=lambda b, ep, sb: dataclasses.replace(orig_def_cfg(b, ep, sb), weight_mode="stream3"),
    ):
        with mock.patch.object(
            rpad,
            "parse_decode_fast_config",
            side_effect=lambda *a, **kw: orig_parse(*a, **{**kw, "b_tokens": 16}),
        ):
            # pps=6 (bkv=4M + init_sv=2M = 6M <= 10M -> unsplit + bkv_in_ici)
            # pps=18 (bkv=10M + init_sv=2M = 12M > 10M -> split_b2_rpad + prestart w_qkv/w_gate, avoids Mosaic spill)
            for pps_t in (6, 18):
                hlo = _compile_step_c_aot(
                    tp=8, B=256, pps=pps_t, sw=-1, mode=prod_mode,
                    dense2=True, bd4=False, dist_len=3, has_rope=False, allow_large_b=True,
                )
                assert "muse_w9_b1_moe_b2_rpad_b256_tp8_" in hlo

        tpu8 = topologies.get_topology_desc("tpu7x:2x2x1", platform="tpu").devices
        mesh8 = Mesh(np.array(sorted(tpu8, key=lambda d: (d.coords[1], d.coords[0], d.core_on_chip))), ("model",))
        sb_shapes = (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((64, 8192), jnp.float32),
            jax.ShapeDtypeStruct((256, 1024), jnp.bfloat16),
            jax.ShapeDtypeStruct((1024, 8192), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((8192, 2048), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 2048), jnp.float32),
            jax.ShapeDtypeStruct((8192, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((1, 256), jnp.float32),
            jax.ShapeDtypeStruct((32, 4096, 8192), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((32, 8, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((32, 4096, 4096), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((32, 8, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((4096, 4096), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((1, 8192), jnp.float32),
            jax.ShapeDtypeStruct((8192, 1280), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 1280), jnp.float32),
            jax.ShapeDtypeStruct((8192, 1024), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((1, 1024), jnp.float32),
        )
        in_sp = (P(),) * len(sb_shapes)
        fn_sb = jax.jit(jax.shard_map(
            lambda *a: mkb.b1_moe_b2_chip_sharded_jax(*a, world_size=8, fuse_qkv_epi=True),
            mesh=mesh8, in_specs=in_sp, out_specs=(P(), P(), P(), P()), check_vma=False,
        ))
        with jax.set_mesh(mesh8):
            sh_a = tuple(jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=NamedSharding(mesh8, P())) for s in sb_shapes)
            hlo_sb = fn_sb.lower(*sh_a).compile().as_text()
        assert "muse_w9_b1_moe_b2_b256_tp8_" in hlo_sb


def test_wave19_w33_swabatch_g16_g8_step_c_plan_and_aot():
    """Verify resolve_step_c_rpa_plan caps wide swabatch at g8 inside Step (c) & TPU7x AOT <60/63 MiB."""
    import dataclasses
    from unittest import mock

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.experimental import topologies
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import sglang_tpu.attention.pallas.rpa_v3_decode_fast as rpad
    import sglang_tpu.muse_megakernel.decode.b1_moe_b2_kernel as mkb

    orig_parse = rpad.parse_decode_fast_config
    prod_mode = "w8,noadapt,hoistkv,g4k,foldcol,batch2,batch4"
    b4_only_mode = "w8,noadapt,hoistkv,g4k,foldcol,batch4"

    # 1. Planner routing when parse_decode_fast_config emits wide swabatch_group_size (16/8) as in dec-rpa-w30:
    # Step (c) caps g16 -> g8 (silicon-optimal across B=64/128/256 BD4 and D2; -50% BD4 bkv VMEM).
    plan_cases = [
        # (label, B, sw, pps, phys, d2, bd4, req_g, want_s, want_g)
        ("tp16_b64_bd4_1k_cap_g16_to_g8", 64, None, 3, 128, False, True, 16, 3, 8),
        ("tp16_b64_bd4_swa_cap_g16_to_g8", 64, 2049, 8, 128, False, True, 16, 3, 8),
        ("tp16_b128_bd4_swa_cap_g16_to_g8", 128, 2049, 8, 128, False, True, 16, 3, 8),
        ("tp16_b256_bd4_swa_cap_g16_to_g8", 256, 2049, 8, 128, False, True, 16, 3, 8),
        ("tp8_b64_d2_1k_cap_g16_to_g8", 64, None, 3, 256, True, False, 16, 4, 8),
        ("tp8_b64_d2_swa_g8", 64, 2049, 8, 256, True, False, 8, 3, 8),
        ("tp8_b128_d2_swa_g8", 128, 2049, 8, 256, True, False, 8, 2, 8),
        ("tp8_b256_d2_swa_g8", 256, 2049, 8, 256, True, False, 8, 2, 8),
    ]
    for label, B, sw, pps, phys, d2, bd4, req_g, want_s, want_g in plan_cases:
        max_rp = 5 if sw is not None else pps
        with mock.patch.object(
            rpad,
            "parse_decode_fast_config",
            side_effect=lambda *a, _rg=req_g, _mrp=max_rp, **kw: dataclasses.replace(
                orig_parse(*a, **kw),
                loop_kind="swabatch",
                num_slots=4,
                swabatch_group_size=_rg,
                max_ring_pages=_mrp,
                hoist_new_kv=False,
            ),
        ):
            cfg, ob, spg, gv, ok = mkb.resolve_step_c_rpa_plan(
                prod_mode,
                sliding_window=sw,
                pages_per_seq=pps,
                phys_page_rows=phys,
                kv_pack_dense2=d2,
                kv_pack_bd4=bd4,
                b_tokens=B,
                max_num_seqs=B,
                allow_large_b=(B > 64),
            )
        assert ok is True and gv is True and ob is True, label
        assert (cfg.num_slots, cfg.swabatch_group_size, spg) == (want_s, want_g, want_g * max_rp), (
            label, cfg.num_slots, cfg.swabatch_group_size, spg,
        )

    # 2. Explicit batch4-only mode stays at swabatch_group_size == 4 across cand_w9_29 and dec-rpa-w30.
    cfg4, _, _, gv4, ok4 = mkb.resolve_step_c_rpa_plan(
        b4_only_mode, sliding_window=2049, pages_per_seq=8, phys_page_rows=128,
        kv_pack_bd4=True, b_tokens=256, max_num_seqs=256, allow_large_b=True,
    )
    assert ok4 is True and gv4 is True and (cfg4.num_slots, cfg4.swabatch_group_size) == (3, 4)

    # 3. TPU7x AOT <60/63 MiB for Step (c) wide swabatch g8 paths (incl. TP=16 B=128 & B=256 BD4 g8 s3 with _dfp).
    aot_cases = [
        (16, 64, 3, -1, False, True, False, 16),
        (16, 64, 8, 2049, False, True, True, 16),
        (8, 64, 3, -1, True, False, False, 16),
        (8, 128, 8, 2049, True, False, True, 8),
    ]
    for tp, B, pps, sw, d2, bd4, rope, req_g in aot_cases:
        max_rp = 5 if sw > 0 else pps
        with mock.patch.object(
            rpad,
            "parse_decode_fast_config",
            side_effect=lambda *a, _rg=req_g, _mrp=max_rp, **kw: dataclasses.replace(
                orig_parse(*a, **kw),
                loop_kind="swabatch",
                num_slots=4,
                swabatch_group_size=_rg,
                max_ring_pages=_mrp,
                hoist_new_kv=False,
            ),
        ):
            hlo = _compile_step_c_aot(
                tp=tp, B=B, pps=pps, sw=sw, mode=prod_mode,
                dense2=d2, bd4=bd4, dist_len=3, has_rope=rope, allow_large_b=(B > 64),
            )
            assert f"muse_w9_b1_moe_b2_rpad_b{B}_tp{tp}_" in hlo

    # Explicit TP=16 B=128 & B=256 BD4 SWA 2K g8 s3 (capped from g16) AOT with free_params (_dfp).
    tpu16 = topologies.get_topology_desc("tpu7x:2x2x2", platform="tpu").devices
    mesh16 = Mesh(np.array(sorted(tpu16, key=lambda d: (d.coords[2], d.coords[1], d.coords[0], d.core_on_chip))), ("model",))
    H, pps = 8192, 8
    for B in (128, 256):
        n_pg = B * pps
        r_pair = B // 8
        s16 = (
            jax.ShapeDtypeStruct((1,), jnp.int32),
            jax.ShapeDtypeStruct((r_pair, H), jnp.float32),
            jax.ShapeDtypeStruct((B, 512), jnp.bfloat16),
            jax.ShapeDtypeStruct((512, H), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((H,), jnp.float32),
            jax.ShapeDtypeStruct((H,), jnp.float32),
            jax.ShapeDtypeStruct((H,), jnp.float32),
            jax.ShapeDtypeStruct((H,), jnp.float32),
            jax.ShapeDtypeStruct((H, 2048), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((2048,), jnp.float32),
            jax.ShapeDtypeStruct((H, 256), jnp.bfloat16),
            jax.ShapeDtypeStruct((4096,), jnp.float32),
            jax.ShapeDtypeStruct((1, 256), jnp.float32),
            jax.ShapeDtypeStruct((16, 4096, 8192), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((16, 8, 8192), jnp.bfloat16),
            jax.ShapeDtypeStruct((16, 4096, 4096), jnp.float4_e2m1fn),
            jax.ShapeDtypeStruct((16, 8, 4096), jnp.bfloat16),
            jax.ShapeDtypeStruct((1, 4096), jnp.float32),
            jax.ShapeDtypeStruct((4096, 4096), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((4096,), jnp.float32),
            jax.ShapeDtypeStruct((H,), jnp.float32),
            jax.ShapeDtypeStruct((H,), jnp.float32),
            jax.ShapeDtypeStruct((H,), jnp.float32),
            jax.ShapeDtypeStruct((H,), jnp.float32),
            jax.ShapeDtypeStruct((H, 640), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((640,), jnp.float32),
            jax.ShapeDtypeStruct((H, 512), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((512,), jnp.float32),
            jax.ShapeDtypeStruct((B, 2, 128), jnp.float32),
            jax.ShapeDtypeStruct((4,), jnp.float32),
            jax.ShapeDtypeStruct((n_pg, 128, 1, 4, 128), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((B,), jnp.int32),
            jax.ShapeDtypeStruct((n_pg,), jnp.int32),
            jax.ShapeDtypeStruct((B + 1,), jnp.int32),
            jax.ShapeDtypeStruct((B + 1,), jnp.int32),
            jax.ShapeDtypeStruct((3,), jnp.int32),
        )
        with mock.patch.object(
            rpad,
            "parse_decode_fast_config",
            side_effect=lambda *a, **kw: dataclasses.replace(
                orig_parse(*a, **kw),
                loop_kind="swabatch",
                num_slots=4,
                swabatch_group_size=16,
                max_ring_pages=5,
                hoist_new_kv=False,
            ),
        ):
            fn16 = jax.jit(jax.shard_map(
                lambda *a: mkb.b1_moe_b2_rpad_chip_sharded_jax(
                    *a[:28], a[30], a[31], a[32], a[33], a[34], a[35],
                    cos_sin_step=a[28], free_params=a[29],
                    world_size=16, rpa_sliding_window=2049, rpa_mode_str=prod_mode,
                    kv_pack_bd4=True, skip_kv_write=True, allow_large_b=True,
                    replica_cost_mode="tiles",
                ),
                mesh=mesh16, in_specs=(P(),) * len(s16), out_specs=(P(), P(), P()), check_vma=False,
            ))
            with jax.set_mesh(mesh16):
                sh16 = tuple(jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=NamedSharding(mesh16, P())) for s in s16)
                hlo16 = fn16.lower(*sh16).compile().as_text()
            assert f"muse_w9_b1_moe_b2_rpad_b{B}_tp16_" in hlo16
