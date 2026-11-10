"""The EP decode MoE kernel (muse_megakernel/moe_ep_decode.py) against its XLA reference on
one TPU device: idle experts, experts split over several work items, padding slots.
    python3 test/manual/tpu/test_moe_ep_decode.py
Prints MOE_EP_DECODE_PASS when every received slot is within tolerance.
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from sglang_tpu.muse_megakernel import moe_ep_decode as M

# per-row ||diff|| / ||ref||: block-diagonal arms are exact in f32 on fp8 codes; dense arms
# round code x scale to bf16 (~4e-3)
TOL_ROW_REL = 1e-2


def main() -> None:
    c = M.EPMoEConfig(
        slots=96, experts=5, hidden=1024, inter=1024, chunk=512, item_rows=16
    )
    rng = np.random.default_rng(0)
    # 3 slots (4-slot block-diagonal arm), none, 40 (items of 16, 16, 8 rows: dense arms),
    # 9 (16-row dense arm), 2 (2-slot arm); padding
    counts = [3, 0, 40, 9, 2]
    n_recv = sum(counts)
    local_e = np.concatenate(
        [np.full(n, e) for e, n in enumerate(counts)] + [np.zeros(c.slots - n_recv)]
    ).astype(np.int32)
    x = rng.standard_normal((c.slots, c.hidden)).astype(np.float32)
    he, i2 = c.hidden, 2 * c.inter
    w13 = rng.integers(0, 2**32, (c.experts, he // 8, i2), dtype=np.uint32)
    w2 = rng.integers(0, 2**32, (c.experts, c.inter // 8, he), dtype=np.uint32)
    s13 = (rng.uniform(0.5, 1.5, (c.experts, he // 16, i2)) / (2.0 * he**0.5)).astype(
        jnp.bfloat16
    )
    s2 = (
        rng.uniform(0.5, 1.5, (c.experts, c.inter // 16, he)) / (2.0 * c.inter**0.5)
    ).astype(jnp.bfloat16)
    args = [jnp.asarray(a) for a in (x, w13, s13, w2, s2)]
    le = jnp.asarray(local_e)
    items, rows = M.layout(c, le, jnp.int32(n_recv))
    xk = jnp.zeros((c.rows, c.hidden), jnp.float32).at[rows].set(M.kernel_x(args[0]))
    wk = M.kernel_weights(*args[1:])
    y = jax.jit(M.ep_moe_call(c))(items, xk, *wk)[rows][:n_recv]
    ref = jax.jit(lambda *a: M.reference(c, le, jnp.int32(n_recv), *a))(*args)[:n_recv]
    rel = np.linalg.norm(np.asarray(y) - np.asarray(ref), axis=1) / np.maximum(
        np.linalg.norm(np.asarray(ref), axis=1), 1e-30
    )
    print(
        f"items {int(items[0])}; per-row rel err max {rel.max():.2e} mean {rel.mean():.2e}; "
        f"ref rms {float(np.sqrt(np.mean(np.asarray(ref) ** 2))):.3f}"
    )
    ok = bool(rel.max() <= TOL_ROW_REL and np.isfinite(np.asarray(y)).all())
    print("MOE_EP_DECODE_PASS" if ok else "MOE_EP_DECODE_FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
