# SPDX-License-Identifier: Apache-2.0
"""GPT-OSS decode megakernel integration for sglang-torchtpu serving (T21 Phase 2).

Opt-in with ``SGLANG_TPU_GPTOSS_DECODE_MK=1``. See
``python/sglang/srt/hardware_backend/tpu/kernels/gptoss_decode_megakernel/INTEGRATION.md``.

- ``weights``: checkpoint (HF MXFP4) -> per-rank megakernel weight layout (torch, CPU).
- ``adapter``: JAX glue around the kernel: sglang page tables / lengths / rope cache ->
  kernel operands, the per-rank step, and the per-layer in-place KV write kernel.
- ``runtime``: torch side: enablement, the load-time weight tap, torch custom ops and the
  decode forward that replaces the layer stack.
"""
