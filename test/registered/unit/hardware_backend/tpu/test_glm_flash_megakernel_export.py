# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3-Flash megakernel step: the exported program has no XLA partition-id.

Serving exports the step for an SPMD-partitioned compile, which refuses a partition-id
outside the Pallas kernels ("partitionId instruction is not supported for SPMD
partitioning"). A chip's mesh position must come from inside the kernels. Exported on
host CPUs (8 virtual devices) for tpu7x; no TPU needed.
"""

import os
import subprocess
import sys
import textwrap
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(est_time=60, suite="stage-a-test-tpu-unit", chips=1)

_SCRIPT = textwrap.dedent(
    """
    import importlib.util, os, sys
    import jax, jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from jax._src import tpu_info as core_tpu_info
    from jax._src.pallas.mosaic import tpu_info
    from jax.experimental.pallas import tpu as pltpu

    info = tpu_info.get_tpu_info_for_chip(tpu_info.chip_version_from_device_kind("TPU7x"), 1)
    pltpu.get_tpu_info = lambda: info
    for m in (tpu_info, core_tpu_info):
        m.get_tpu_info = lambda: info
        m.get_device_kind = lambda: "TPU7x"
        m.is_tpu_device = lambda: True

    from sglang_tpu.glm_flash_megakernel.build import slab_maps
    from sglang_tpu.glm_flash_megakernel.kernel import decode as D
    from sglang_tpu.glm_flash_megakernel.kernel import runner as R
    from sglang_tpu.glm_flash_megakernel.layout import KernelLayout

    spec_path = sys.argv[1]
    s = importlib.util.spec_from_file_location("_cfg", spec_path)
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    spec = mod._spec()
    tp, F8, BF, F32 = 8, jnp.float8_e4m3fn, jnp.bfloat16, jnp.float32
    layout = KernelLayout(spec, tp)
    mesh = Mesh(jax.devices()[:tp], (D.AXIS,))
    cfg = D.KernelConfig(max_len=4096, n_query=4, n_req=1, n_layers=4, embed_2d=True,
                         km_tile=256, context_parallel=True, mla_shards=tp)
    run = R.make_decode(spec, layout, *slab_maps(spec), mesh, cfg)
    H = spec.hidden_size
    n_dsa, n_kda = len(spec.dsa_layers), len(spec.kda_layers)
    n_moe = sum(t == "sparse" for t in spec.mlp_layer_types)
    nh, qsh = spec.num_attention_heads // tp, spec.kda_qkv_dim // tp
    hk, dh, idh = spec.linear_num_heads // tp, spec.linear_head_dim, spec.index_head_dim
    ms, fs, ql, kvl = spec.moe_intermediate_size // tp, spec.intermediate_size // tp, spec.q_lora_rank, spec.kv_lora_rank
    qk, vd, L, ne, k1 = spec.qk_head_dim, spec.v_head_dim, spec.num_hidden_layers, spec.n_routed_experts, spec.linear_conv_kernel_dim
    sd = jax.ShapeDtypeStruct
    pk = lambda shape, dt, rows: jax.eval_shape(lambda: D.pack_kmajor(jnp.zeros(shape, dt), rows, cfg.km_cols))
    W = dict(
        embed_tokens=sd((spec.vocab_size, H // tp), BF), lm_head=pk((1, layout.vocab_padded // tp, H), BF, H),
        final_norm=sd((1, H), BF), smalls=sd((L + 1, layout.small_rows(), H), BF), smalls_f32=sd((L + 1, 1, H), F32),
        kda_qkv=pk((n_kda, 3 * qsh, H), F8, 2 * H), kda_o=pk((n_kda, H, qsh), F8, 2 * H),
        kda_fg_b=sd((n_kda, 2 * qsh, dh), BF), kda_dt_bias=sd((n_kda, 1, qsh), F32),
        kda_conv=sd((n_kda, k1, 1, 3 * qsh), BF), kda_b=sd((n_kda, hk, H), BF), kda_a_log=sd((n_kda, 1, hk), F32),
        kda_scale=sd((n_kda, 1, 3 * qsh + H), F32), dsa_qakv=pk((n_dsa, ql + kvl, H), F8, 2 * H),
        dsa_q_b=pk((n_dsa, nh * qk, ql), F8, 2 * H), dsa_kv_b=sd((n_dsa, nh * (qk + vd), kvl), BF),
        dsa_o=pk((n_dsa, H, nh * vd), F8, 2 * H), dsa_scale=sd((n_dsa, 1, D.proj_scale_cols(spec, tp)), F32),
        indexer_wq_b=pk((n_dsa, spec.index_n_heads * idh, ql), BF, H), moe_router=sd((n_moe, ne, H), BF),
        expert_gate_up=[sd((ne, H, 2 * ms), F8)] * n_moe, expert_gate_up_scale=sd((n_moe * ne, 2 * ms), F32),
        expert_down=[sd((ne, ms, H), F8)] * n_moe, expert_down_scale=sd((n_moe * ne, H), F32),
        shared_gate_up=sd((n_moe, H, 2 * ms), F8), shared_gate_up_scale=sd((n_moe, 2 * ms), F32),
        shared_down=sd((n_moe, ms, H), F8), shared_down_scale=sd((n_moe, H), F32),
        dense_gate_up=pk((3, 2 * fs, H), F8, 2 * H), dense_down=pk((3, H, fs), F8, 2 * H),
        dense_scale=sd((3, 1, D.proj_scale_cols(spec, tp)), F32),
    )
    pages, ps = 64, cfg.page_size
    pools = dict(
        nope=[sd((pages, ps // tp, 4, 128), jnp.uint8)] * n_dsa, raw=[sd((pages, ps, 2 * idh), BF)] * n_dsa,
        records=[sd((pages, ps // spec.index_kpool // 4, 4, 256), jnp.uint8)] * n_dsa,
        temporal=[sd((8, hk, dh, dh), F32)] * n_kda, conv=[sd((8, k1 - 1, 1, 3 * qsh), F32)] * n_kda,
    )

    def glob(x, sp):
        shape = tuple(d * (tp if p == D.AXIS else 1) for d, p in zip(x.shape, tuple(sp) + (None,) * 8))
        return sd(shape, x.dtype, sharding=NamedSharding(mesh, sp))

    W = {n: ([glob(x, run.wspec[n][0]) for x in v] if n in R.EXPERT_LISTS else glob(v, run.wspec[n])) for n, v in W.items()}
    pools = {n: [glob(x, run.pspec[n][0]) for x in v] for n, v in pools.items()}
    sm = NamedSharding(mesh, P())
    scal = [sd(sh, jnp.int32, sharding=sm) for sh in ((4,), (1,), (8,), (cfg.pages_per_seq,), (1,), (4,))]
    ks = sd((n_dsa,), F32, sharding=sm)
    init = sd((cfg.rows * spec.hc_mult, H), BF, sharding=NamedSharding(mesh, P(None, None)))
    exp = jax.export.export(jax.jit(run.shard), platforms=("tpu",))(*scal, ks, init, W, pools)
    print("PARTITION_ID_OPS", exp.mlir_module().count("partition_id"))
    """
)


class TestGlmFlashMegakernelExport(CustomTestCase):
    def test_split_step_has_no_xla_partition_id(self):
        env = dict(os.environ)
        env.update(
            JAX_PLATFORMS="cpu", XLA_FLAGS="--xla_force_host_platform_device_count=8"
        )
        cfg_test = os.path.join(
            os.path.dirname(__file__), "test_glm_flash_megakernel_config.py"
        )
        out = subprocess.run(
            [sys.executable, "-c", _SCRIPT, cfg_test],
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        lines = [l for l in out.stdout.splitlines() if l.startswith("PARTITION_ID_OPS")]
        self.assertTrue(lines, out.stderr[-3000:])
        self.assertEqual(lines[-1], "PARTITION_ID_OPS 0")


if __name__ == "__main__":
    unittest.main()
