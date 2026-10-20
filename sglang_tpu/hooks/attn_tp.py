# SPDX-License-Identifier: Apache-2.0
"""Sequence-parallel (input-scattered) attention-TP inputs on TPU.

device_supports_input_scattered
  Why: AttnTpContext allows --enable-attn-tp-input-scattered on CUDA and NPU
       only; the layer-boundary reduce-scatter / all-gather moves it relies on
       also lower on TPU.
  How: AROUND layer_boundary.adapters.attention._device_supports_input_scattered:
       True. The fused mHC sites fall back to torch on scattered batches
       (hooks/mhc.py).
  Upstream PR or issue: none (the seam is core's _device_supports_input_scattered).
  Remove when: never; this is the TPU answer for that seam.
"""

from __future__ import annotations


def device_supports_input_scattered(original) -> bool:
    return True
