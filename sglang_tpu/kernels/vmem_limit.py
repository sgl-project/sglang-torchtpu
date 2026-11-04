# SPDX-License-Identifier: Apache-2.0
"""Right-sized scoped VMEM limits for decode Pallas kernels (TPU).

XLA reserves a Mosaic kernel's whole ``vmem_limit_bytes`` as scoped VMEM for
the duration of the call, so a 40-64 MiB limit on a 64 MiB TensorCore leaves
XLA's memory-space assignment no room to keep weight prefetches in flight
across the kernel: in the Kimi-K3 TP32 decode step the DSA o_proj weight is
read from HBM at 1.3 TB/s and the DSA MoE-front weight copy is exposed.
With SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE the kernels ask for their working set
(computed from their block / scratch shapes) plus a margin instead. The
kernel source is unchanged, but Mosaic may tile a kernel differently under a
smaller limit (the single-token MoE kernel on Kimi-K3 sums in another order),
so the switch is off by default and set by the Kimi-K3.1 recipe.
"""

_MIB = 1 << 20


def padded_bytes(shape, itemsize: int) -> int:
    """Bytes of a VMEM array in (sublane, 128-lane) tiles: the last dim pads
    to 128 and the second-last to the dtype's sublane tile (8 rows of 32 bits)."""
    shape = tuple(int(d) for d in shape) or (1,)
    lanes = -(-shape[-1] // 128) * 128
    sub = 8 * max(1, 4 // itemsize)
    rows = -(-(shape[-2] if len(shape) > 1 else 1) // sub) * sub
    lead = 1
    for d in shape[:-2]:
        lead *= d
    return lead * rows * lanes * itemsize


def _enabled() -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    return tpu_envs.SGLANG_TPU_PALLAS_VMEM_RIGHTSIZE.get()


def scoped_limit(
    default: int, need: int, ratio: float = 1.25, margin: int = 4 * _MIB
) -> int:
    """``default`` unless right-sizing is on; then need * ratio + margin,
    rounded up to a MiB and never above ``default``."""
    if not _enabled():
        return default
    limit = -(-int(need * ratio + margin) // _MIB) * _MIB
    return min(default, limit)
