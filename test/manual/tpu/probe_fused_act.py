# Isolate: gmm_v2 fused silu(gate)*up with multi-block group scales vs
# computing the same thing unfused (two matmuls + act in JAX).
import os

os.environ["TPU_VISIBLE_CHIPS"] = "0"
os.environ["TPU_CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_HOST_BOUNDS"] = "1,1,1"
os.environ["HOST_BOUNDS"] = "1,1,1"
os.environ["CHIPS_PER_HOST_BOUNDS"] = "1,1,1"
os.environ["TPU_WORKER_ID"] = "0"
os.environ["WRAP"] = "false,false,false"
for _v in (
    "TPU_WORKER_HOSTNAMES",
    "TPU_PROCESS_ADDRESSES",
    "TPU_PROCESS_PORT",
    "TORCH_TPU_SLICEBUILDER_ADDRESSES",
    "TORCH_TPU_TOPOLOGY",
):
    os.environ.pop(_v, None)

from sglang.srt.platforms import current_platform

current_platform.init_backend()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from sglang.srt.layers.moe.pallas.gmm_v2 import gmm_v2  # noqa: E402

rng = np.random.default_rng(0)
K, N, M = 256, 128, 64  # per-projection N; w13 has 2N cols
GS = 32
q = rng.integers(-8, 8, size=(1, K, 2 * N)).astype(np.int64)
s = (rng.random((1, K // GS, 2 * N)) * 0.05 + 0.01).astype(np.float32)

stored = (q + 8).astype(np.uint64).reshape(1, K // 8, 8, 2 * N)
words = np.zeros((1, K // 8, 2 * N), dtype=np.uint64)
for i in range(8):
    words |= stored[:, :, i, :] << (4 * i)
packed = ((words & 0xFFFFFFFF).astype(np.uint32) ^ np.uint32(0x88888888)).view(np.int32)

lhs = (rng.standard_normal((M, K)) * 0.5).astype(np.float32)
lhs_j = jnp.asarray(lhs, dtype=jnp.bfloat16)
rhs_j = jnp.asarray(packed)
scale_j = jnp.asarray(s[:, :, None, :])  # [1, B, 1, 2N]
group_sizes = jnp.array([M], dtype=jnp.int32)

# fused path
fused = gmm_v2(
    lhs_j,
    rhs_j,
    group_sizes,
    rhs_scale=scale_j,
    maybe_quantize_lhs=False,
    rhs_quant_dtype=jnp.int4,
    fuse_act="silu",
)
# unfused path (same kernel, no fuse_act) + act in JAX
raw = gmm_v2(
    lhs_j,
    rhs_j,
    group_sizes,
    rhs_scale=scale_j,
    maybe_quantize_lhs=False,
    rhs_quant_dtype=jnp.int4,
)
unfused = jax.nn.silu(raw[:, :N]) * raw[:, N:]

# float reference
wref = q.astype(np.float32) * np.repeat(s, GS, axis=1)
ref_raw = lhs @ wref[0]
ref = ref_raw[:, :N] * (1 / (1 + np.exp(-ref_raw[:, :N])))
ref = ref * ref_raw[:, N:]


def stats(name, a):
    a = np.asarray(a, dtype=np.float32)
    cos = float(
        (a.ravel() @ ref.ravel())
        / (np.linalg.norm(a.ravel()) * np.linalg.norm(ref.ravel()) + 1e-9)
    )
    print(f"{name}: cosine_vs_ref={cos:.6f}")


stats("raw-vs... (sanity: unfused)", unfused)
stats("fused                      ", fused)
