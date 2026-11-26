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
