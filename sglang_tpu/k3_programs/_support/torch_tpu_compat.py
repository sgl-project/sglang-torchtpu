# SPDX-License-Identifier: Apache-2.0
"""torch_tpu names that moved between the releases the programs run on."""

from __future__ import annotations


def compile_fx_graph_mode():
    """The eager mode that traces an op into an FX graph for compilation:
    ``INTERNAL_COMPILE_FX_GRAPH`` in torch-tpu 0.1, ``_INTERNAL_COMPILE_FX_GRAPH``
    in 0.2."""
    from torch_tpu._internal import execution_mode

    modes = execution_mode.EagerMode
    for name in ("_INTERNAL_COMPILE_FX_GRAPH", "INTERNAL_COMPILE_FX_GRAPH"):
        if hasattr(modes, name):
            return getattr(modes, name)
    raise AttributeError("torch_tpu has no FX-graph compile eager mode")
