"""What direct_compile asks the lowered-graph store for, and what it gets back.

The store, and when it is used, is ``lowered_graph_cache``. A stored graph is
what a shared-compile owner hands its consumers: the executable, serialized as
for shared compilation (``spmd_compiler.serialize_executable``), its subslice
origin and its ``ExecutableInterface``.

This module imports neither TorchTPU nor ``direct_compile`` at runtime (the
shared types are annotations here).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang_tpu.compilation.spmd_compiler import ExecutableInterface, InputSpec

# ``ExecutableInterface.callback_factories`` of every stored graph: graphs with
# inserted callbacks are never stored, and the owner of a shared graph without
# callbacks pickles the same empty tuple.
NO_CALLBACKS = pickle.dumps(())


@dataclass(frozen=True)
class LoweredGraphRequest:
    """What ``direct_compile`` knows about one specialization before tracing.

    ``name`` is the compiled function's ``direct_compile(name=...)``, and
    ``graph`` the specialization's identity (``graph_identity.graph_name``),
    the string shared SPMD compilation matches across ranks. The other fields
    describe the traced function's inputs, not their values. The store
    combines them with process-wide facts (code version, configuration, the compiler's
    cache key) that ``direct_compile`` cannot see.

    Tensors are described by ``spmd_compiler.InputSpec``. ``input_tensors``
    pairs each explicit tensor input with its path in the call. ``model_inputs``
    has one entry per unique model tensor, in graph-input order, named by its
    first registered name. ``model_input_names`` maps every registered name and
    ``model_extra_tensor_attrs`` path to that index, so aliasing changes alter
    the request. ``kv_inputs`` lists the KV-pool state in order.
    ``argument_layouts`` describes the device layout every flat input compiles
    for before pruning, in graph-input order.
    """

    name: str
    graph: str
    input_treedef: str
    input_tensors: tuple[tuple[str, InputSpec], ...]
    static_inputs: tuple[tuple[str, str, str], ...]
    model_inputs: tuple[tuple[str, InputSpec], ...]
    model_input_names: tuple[tuple[str, int], ...]
    kv_inputs: tuple[InputSpec, ...]
    argument_layouts: tuple[str, ...]
    prune_unused_inputs: bool


@dataclass(frozen=True)
class LoweredGraph:
    """Everything needed to rebuild a compiled artifact without tracing.

    ``interface`` is the ``ExecutableInterface`` a shared-compile owner would
    publish for this graph (``callback_factories`` is always
    ``NO_CALLBACKS``), and ``direct_compile`` checks it against the live inputs
    exactly as a consumer checks its owner's. Its arguments are the retained
    model inputs, the KV-pool state and the explicit tensor inputs, in that
    order; its results are the output leaves followed by the updated KV-pool
    state. ``executable`` is ``spmd_compiler.serialize_executable`` of the
    compiled graph, and ``fingerprint`` its ``fingerprint_executable()``; a
    loaded executable with another fingerprint is not reused.
    ``subslice_origin`` is its ``subslice_origin()``, the TPU subslice's chip
    coordinates, which the serialized bytes omit; ``direct_compile`` loads the
    executable with it.
    """

    interface: ExecutableInterface
    executable: bytes
    fingerprint: str
    subslice_origin: Any
