"""Fused EP MoE v2 combine: the fused-scale build matches the XLA scale lookup bit for bit."""

import os
import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "deepseek-v4.1-flash", "accuracy", "compile"],
)

os.environ.setdefault("JAX_PLATFORMS", "tpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.moe.pallas.fused_moe_v2 import layer


@pytest.mark.parametrize("t_local", [256, 512, 1024])
def test_fused_scale_combine_is_bit_identical(t_local):
    if jax.default_backend() != "tpu":
        pytest.skip("Mosaic combine kernel requires TPU")
    topk, lane_blocks, tile_m = 6, 40, 128
    recv_rows = (
        8 * t_local * topk
    )  # every shard routing every pair here (the no-drop bound)
    sublanes = pl_sublanes = -(-recv_rows // tile_m) + 1
    rng = np.random.default_rng(t_local)
    arrivals = jnp.asarray(
        rng.normal(size=(recv_rows, lane_blocks, 128)).astype(np.float32)
    ).astype(jnp.float8_e4m3fn)
    scales = rng.uniform(0.5, 2.0, (pl_sublanes, tile_m)).astype(np.float32)
    scales[-1, :] = np.nan  # unwritten mirror entries: never addressed below
    pos = rng.integers(0, recv_rows, (t_local, topk)).astype(np.int32)
    mirror_pos = rng.integers(0, (sublanes - 1) * tile_m, (t_local, topk)).astype(
        np.int32
    )
    weights = rng.uniform(0.0, 1.0, (t_local, topk)).astype(np.float32)
    args = (
        arrivals,
        jnp.asarray(scales),
        jnp.asarray(pos),
        jnp.asarray(mirror_pos),
        jnp.asarray(weights),
    )

    def run(fuse):
        with tpu_envs.SGLANG_TPU_FUSED_EP_V2_COMBINE_FUSE_SCALES.override(fuse):
            fn = jax.jit(lambda *a: layer._combine_arrivals(*a, jnp.bfloat16))
            return np.asarray(fn(*args).view(jnp.uint16))

    np.testing.assert_array_equal(run(True), run(False))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
