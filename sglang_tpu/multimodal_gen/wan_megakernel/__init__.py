# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-A14B's DiT block stack as one Pallas megakernel call per forward.

``reference`` is the specification (plain JAX, diffusers' rounding points),
``kernel`` the Pallas implementation of every block of a forward in one
``pallas_call`` (Ulysses exchange by remote DMA inside it), ``tiling`` its
tile sizes. This module imports neither JAX nor the kernel.
"""
