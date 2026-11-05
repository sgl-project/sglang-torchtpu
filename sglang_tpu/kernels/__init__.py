# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas kernels for the SGLang torch_tpu backend.

Callable kernels live in ``sglang_tpu.kernels.ops.<group>``, one package per
computation, following the upstream ``sglang.kernels.ops`` layout (RFC #29630).
Shared kernel helpers such as ``vmem_limit`` sit at this level.
"""
