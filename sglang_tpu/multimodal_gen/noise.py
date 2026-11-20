# SPDX-License-Identifier: Apache-2.0
"""Low-precision random normals on TPU devices drawn in float32 and rounded.

Runtime patch, tracked debt:
- Why: torch_tpu builds a bfloat16 normal from 7 random bits (128 values in
  [-2.89, 2.52], mean -0.011) and a float16 normal from 10. Diffusion
  pipelines draw their noise in the latent dtype on the device of the
  request's generator, a TPU generator under the HTTP default
  (``generator_device="cuda"``): diffusers' ``randn_tensor`` in the latent
  stages (bfloat16 for FLUX.2, Qwen-Image, LTX-2, SANA, ERNIE-Image and MOVA,
  float16 for Hunyuan3D-2) and ``torch.randn`` in stage and model code
  (Cosmos3, SenseNova-U1, the LTX-2 stage-2 noise).
- How: :func:`install` wraps ``torch.randn``: a bfloat16 or float16 draw on a
  TPU device is drawn in float32 from the same generator and rounded, as CUDA
  draws them. Draws on the host (CPU generators) are unchanged, as on other
  platforms.
- Upstream PR or issue: none.
- Remove when: torch_tpu draws low-precision normals from float32 samples.
"""

from __future__ import annotations

import functools

import torch
from torch.utils._device import _device_constructors

_LOW_PRECISION = (torch.bfloat16, torch.float16)
_DEVICE_TYPE = "tpu"


def _float32_randn(stock):
    @functools.wraps(stock)
    def randn(*size, dtype=None, device=None, **kwargs):
        # ``out=`` and ``requires_grad`` keep the stock call: the rounded copy
        # would be neither the given tensor nor a leaf.
        if (
            dtype in _LOW_PRECISION
            and isinstance(device, (str, torch.device))
            and torch.device(device).type == _DEVICE_TYPE
            and kwargs.get("out") is None
            and not kwargs.get("requires_grad", False)
        ):
            return stock(*size, dtype=torch.float32, device=device, **kwargs).to(dtype)
        return stock(*size, dtype=dtype, device=device, **kwargs)

    randn.stock = stock
    return randn


def install() -> None:
    if hasattr(torch.randn, "stock"):
        return
    # ``with torch.device(...)`` routes the constructors in this cached set; fill
    # it first so it holds the function the wrapper calls.
    _device_constructors()
    torch.randn = _float32_randn(torch.randn)
