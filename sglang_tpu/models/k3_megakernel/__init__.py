# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 decode megakernel: one Pallas call per device per decode/verify step.

The layer loop, the cross-rank reductions (in-kernel remote DMA) and the
next-layer weight prefetch all live inside the kernel. Gated by
``SGLANG_TPU_ENABLE_K3_DECODE_MEGAKERNEL``. Layout: ``kernel/`` (the Pallas stack: stages,
collectives), ``weights.py`` (checkpoint -> kernel layouts), ``ops.py`` (sglang serving: torch
custom op and runtime), ``model_io.py`` / ``mesh.py`` (JAX-side loading), ``server/`` (standalone
OpenAI-compatible server).
"""
