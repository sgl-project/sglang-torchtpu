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
    cfg_bd4, one_bd4, _, _, ok_bd4 = resolve_step_c_rpa_plan(
        mode,
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=False,
    )
    assert ok_bd4 is True and one_bd4 is True
    assert cfg_bd4.loop_kind == "w7auto" and cfg_bd4.shared_group_size == 0

    cfg_d2, one_d2, _, _, ok_d2 = resolve_step_c_rpa_plan(
        mode,
        sliding_window=None,
        pages_per_seq=40,
        phys_page_rows=256,
        kv_pack_dense2=True,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=False,
    )
    assert ok_d2 is True and one_d2 is True
    assert cfg_d2.loop_kind == "w7auto" and cfg_d2.shared_group_size == 0

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
    # 11 Full unified 5D layers -> must split cleanly into chunks [8, 3] via pure layer_num.
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
        mock.patch.object(b1_moe_ops, "get_step_c_kv_write_multi_5d_op", side_effect=lambda k, **kw: lambda *a: calls.append(("m5d", k, kw, tuple(a[0].shape)))),
    ):
        b1_moe_ops.flush_pending_step_c_kv_writes(pending)

    assert calls == [
        ("m6d", {"kv_pack_dense2": False, "kv_pack_bd4": True}, (2, 32, 1, 4, 128), (0, 1)),
        ("single", {"kv_pack_dense2": False, "kv_pack_bd4": True}, 8),
        ("m5d", 8, {"kv_pack_dense2": True, "kv_pack_bd4": False}, (8, 32, 1, 4, 128)),
        ("m5d", 3, {"kv_pack_dense2": True, "kv_pack_bd4": False}, (3, 32, 1, 4, 128)),
    ]

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

    # TP=16 B=256 BD4 SWA (batch4) fits b2 (5,120 <= 8,192): keeps Step (c) ON by default (#5010).
    cfg_swa_256_bd4, _, slot_swa_256_bd4, grp_swa_256_bd4, ok_swa_256_bd4 = resolve_step_c_rpa_plan(
        "w8,s4,batchfull,g4k,foldcol,batch4",
        sliding_window=2049,
        pages_per_seq=4,
        phys_page_rows=128,
        kv_pack_bd4=True,
        b_tokens=256,
        max_num_seqs=256,
    )
    assert ok_swa_256_bd4 is True and grp_swa_256_bd4 is True and cfg_swa_256_bd4.swabatch_group_size == 2
    assert slot_swa_256_bd4 == 10

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

    # BD4 B=256 Full 8K/1K (pps=32) auto-clamps to ring8k (s4, c8192, nocoal) cleanly without LARGE_B;
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
    assert ok_full_256 is True and one_full_256 is False and grp_full_256 is False
    assert (cfg_full_256.loop_kind, cfg_full_256.num_slots, cfg_full_256.chunk_sz, slot_full_256) == (
        "ring4k",
        4,
        8192,
        16,
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
