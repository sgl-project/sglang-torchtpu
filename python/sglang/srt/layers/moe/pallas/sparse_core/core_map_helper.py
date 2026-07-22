# Ported from tpu-inference (https://github.com/vllm-project/tpu-inference)
#   tpu_inference/kernels/sparse_core/core_map_helper.py @ 4367dcff1 (2026-07-19).

"""Run single-mesh SparseCore kernels through ``core_map``.

``pl.kernel`` lowers through ``mpmd_map``, which is slower than the ``core_map``
path for these single-mesh kernels on TPU. ``kernel`` here is a drop-in
replacement for ``pl.kernel`` that uses ``core_map`` directly, so the SparseCore
kernels keep that lowering without depending on the jax-side ``pl.kernel``
implementation.
"""

from jax._src import api
from jax._src import core as jax_core
from jax._src import lax, tree_util
from jax._src.pallas import core as pl_core


def _empty_out_ref(out_type):
    aval = pl_core._convert_out_shape_to_aval(out_type)
    memory_space = (
        None
        if isinstance(aval.memory_space, jax_core.MemorySpace)
        else aval.memory_space
    )
    value = lax.empty(aval.shape, aval.dtype, out_sharding=aval.sharding)
    return jax_core.new_ref(value, memory_space=memory_space)


def kernel(
    body,
    *,
    out_type,
    mesh,
    scratch_types=(),
    compiler_params=None,
    interpret=False,
    cost_estimate=None,
    debug=False,
    name=None,
    metadata=None
):
    """Drop-in replacement for ``pl.kernel`` that lowers via ``core_map``."""
    single_output = not isinstance(out_type, (tuple, list))
    out_types = (out_type,) if single_output else out_type

    @api.jit
    def run(*operands):
        arg_refs = tree_util.tree_map(jax_core.new_ref, operands)
        out_refs = tree_util.tree_map(_empty_out_ref, out_types)

        @pl_core.core_map(
            mesh,
            scratch_shapes=scratch_types,
            compiler_params=compiler_params,
            interpret=interpret,
            cost_estimate=cost_estimate,
            debug=debug,
            name=name,
            metadata=metadata,
        )
        def _(*scratch_refs, **scratch_kwrefs):
            return body(*arg_refs, *out_refs, *scratch_refs, **scratch_kwrefs)

        outs = tree_util.tree_map(lambda ref: ref[...], out_refs)
        return outs[0] if single_output else outs

    return run
