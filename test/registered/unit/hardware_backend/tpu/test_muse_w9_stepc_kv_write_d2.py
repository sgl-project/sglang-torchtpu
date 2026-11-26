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

    _, _, _, _, ok_512 = resolve_step_c_rpa_plan(
        mode,
        sliding_window=None,
        pages_per_seq=32,
        phys_page_rows=512,
        b_tokens=64,
        max_num_seqs=64,
        has_shared_prefix=False,
    )
    assert ok_512 is False

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
    assert "_c1024_rope" in hlo_txt, f"Missing adaptive 1K branch in lowered HLO for TP={tp}"
    assert lowered.compile() is not None
