# Probe pltpu's int4 unpack order: one-hot lhs rows + unit scales make the
# MoE output equal the int4 values the kernel sees at each (k, n).
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

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from sglang.srt.layers.moe.pallas.gmm_v2 import gmm_v2  # noqa: E402

K, N = 256, 128
# q[k, n] = a value depending ONLY on k (mod 16), so any k-permutation is visible
q = ((np.arange(K) % 16) - 8).astype(np.int64)  # [-8..7] repeating
qkn = np.tile(q[:, None], (1, N))  # [K, N]

# pack the compressed-tensors way: offset binary, LSB-first, consecutive
stored = (qkn + 8).astype(np.uint64)
words = np.zeros((K // 8, N), dtype=np.uint64)
for i in range(8):
    words |= stored[i::8, :][: K // 8] * 0  # placeholder, replaced below
words = np.zeros((K // 8, N), dtype=np.uint64)
resh = stored.reshape(K // 8, 8, N)
for i in range(8):
    words |= resh[:, i, :] << (4 * i)
packed = (words & 0xFFFFFFFF).astype(np.uint32).view(np.int32)  # [K//8, N]

INT4_SIGN_XOR = np.uint32(0x88888888)
packed_x = (packed.view(np.uint32) ^ INT4_SIGN_XOR).view(np.int32)

# rhs [1, K//8, N] int32, scale [1, K//32, 1, N] = 1.0
rhs = jnp.asarray(packed_x[None])
scale = jnp.ones((1, K // 32, 1, N), dtype=jnp.float32)
lhs = jnp.eye(K, dtype=jnp.bfloat16)  # M=K one-hots
group_sizes = jnp.array([K], dtype=jnp.int32)

out = gmm_v2(
    lhs,
    rhs,
    group_sizes,
    rhs_scale=scale,
    maybe_quantize_lhs=False,
    rhs_quant_dtype=jnp.int4,
)
seen = np.asarray(out[:, 0]).astype(np.int64)  # value at (k, n=0)
expect = q
print("k : expect seen (first 24)")
for k in range(24):
    print(f"{k:3d}: {expect[k]:4d} {seen[k]:4d}")
match = (seen == expect).mean()
print(f"match fraction: {match:.3f}")
# derive permutation within the first word if mismatched
if match < 1.0:
    print("word0 expect:", expect[:8].tolist())
    print("word0 seen  :", seen[:8].tolist())
    print("word1 expect:", expect[8:16].tolist())
    print("word1 seen  :", seen[8:16].tolist())
