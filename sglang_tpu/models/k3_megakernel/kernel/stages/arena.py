# SPDX-License-Identifier: Apache-2.0
"""Typed views over one raw VMEM scratch buffer.

Buffers whose lifetimes never overlap (e.g. layer 0's dense-MLP stream and the
routed-expert ring, or KDA's and MLA's attention buffers) share one ``Arena``.
A view is a byte range of the arena reinterpreted as a ``(shape, dtype)`` memref
with the native tiled VMEM layout (tile ``(8 * packing, 128)``), so DMAs and
loads/stores on it behave like on an ordinary scratch buffer of that type.

Pallas has no public way to reinterpret tiled memory as a different shape: a
view is a ``ReshapeTransform`` subclass whose lowering emits
``tpu.reinterpret_cast`` with an explicit tiled layout (the stock reshape
lowering is wrapped once, per process). Every view must be written (by DMA or
store) before it is read; nothing is zero-initialized.
"""

import dataclasses
import math

import jax.numpy as jnp
from jax import tree_util
from jax._src.lib import tpu as _tpu
from jax._src.lib.mlir import ir
from jax._src.pallas.mosaic import lowering as _lowering
from jax._src.state import types as _state_types
from jax.experimental.pallas import tpu as pltpu

_ROW_BYTES = 128 * 4  # the arena is int32 [rows, 128]
_ALIGN = 8 * _ROW_BYTES  # one int32 (8, 128) tile


def _tile_rows(dtype) -> tuple[int, int]:
    """(tile rows, packing) of the native VMEM layout of a >= 2D memref."""
    packing = 4 // jnp.dtype(dtype).itemsize
    return 8 * packing, packing


def storage_bytes(shape, dtype) -> int:
    """VMEM bytes of a ``(shape, dtype)`` buffer in its native tiled layout."""
    rows, _ = _tile_rows(dtype)
    assert len(shape) >= 2 and shape[-1] % 128 == 0, shape
    return math.prod(shape[:-2]) * -(-shape[-2] // rows) * (shape[-1] // 128) * 4096


# A dataclass because JAX's ref transforms are registered pytree dataclasses.
@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, slots=True)
class _View(_state_types.ReshapeTransform):
    """Marks a reshape that reinterprets raw arena memory (see module docstring)."""

    def _validate_shape(self, input_shape):
        # The source slice spans the view's padded storage, which may exceed its elements.
        if math.prod(self.shape) > math.prod(input_shape):
            raise ValueError(
                f"view {self.shape} larger than its arena slice {input_shape}"
            )


def _layout(shape, dtype) -> ir.Attribute:
    rows, packing = _tile_rows(dtype)
    col_tiles = shape[-1] // 128
    strides = [col_tiles, 1]
    acc = -(-shape[-2] // rows) * col_tiles
    for dim in reversed(shape[:-2]):
        strides.insert(0, acc)
        acc *= dim
    inner = f"({packing},1)" if packing > 1 else ""
    return ir.Attribute.parse(
        f"#tpu.tiled<({rows},128){inner},[{','.join(map(str, strides))}]>"
    )


def _reshape_memref(ref, reshaper, ref_aval, ref_block_shape):
    if not isinstance(reshaper, _View):
        return _lowering._k3mk_stock_reshape_memref(
            ref, reshaper, ref_aval, ref_block_shape
        )
    shape = tuple(reshaper.shape)
    src = ir.MemRefType(ref.type)
    typed = ir.MemRefType.get(
        shape,
        src.element_type,
        layout=_layout(shape, ref_aval.dtype),
        memory_space=src.memory_space,
    )
    cast = _tpu.reinterpret_cast(typed, ref)
    plain = ir.MemRefType.get(shape, src.element_type, memory_space=src.memory_space)
    return _tpu.erase_memref_layout(plain, cast), shape


if not hasattr(_lowering, "_k3mk_stock_reshape_memref"):
    _lowering._k3mk_stock_reshape_memref = _lowering._reshape_memref
_lowering._reshape_memref = _reshape_memref


class Arena:
    """Static allocator of views; ``size`` grows to the largest end offset requested."""

    def __init__(self):
        self.size = 0

    def place(self, offset: int, shape, dtype) -> tuple[int, int]:
        """Reserve ``(shape, dtype)`` at byte ``offset`` (rounded up to a tile);
        returns (offset, end)."""
        offset = -(-offset // _ALIGN) * _ALIGN
        end = offset + -(-storage_bytes(shape, dtype) // _ALIGN) * _ALIGN
        self.size = max(self.size, end)
        return offset, end

    def scratch_shape(self):
        return pltpu.VMEM((max(self.size, _ALIGN) // _ROW_BYTES, 128), jnp.int32)

    @staticmethod
    def view(ref, offset: int, shape, dtype):
        """``ref``: the bound arena scratch ref; ``offset`` from ``place``."""
        shape = tuple(shape)
        rows = storage_bytes(shape, dtype) // _ROW_BYTES
        sub = ref.at[offset // _ROW_BYTES : offset // _ROW_BYTES + rows]
        if jnp.dtype(dtype) != jnp.dtype(jnp.int32):
            sub = sub.bitcast(dtype)
        return _state_types.TransformedRef(sub.ref, (*sub.transforms, _View(shape)))
