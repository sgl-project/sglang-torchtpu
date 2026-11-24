# SPDX-License-Identifier: Apache-2.0
"""MiniMax-H3's DiT block stack as one Pallas megakernel call per forward.

``reference`` is the specification (plain JAX), ``kernel`` the Pallas
implementation of all 50 blocks of a forward in one ``pallas_call`` (Ulysses
exchange by remote DMA inside it), ``tiling`` its tile sizes. ``kernel``'s
``stack`` also takes a ``static_structure`` (see ``loop_structure``) and the
``cut_kv_compute``, ``split_items`` and ``epilogue_chunk`` options, each off
by default, which keeps the program unchanged. This module imports neither
JAX nor the kernel.
"""
