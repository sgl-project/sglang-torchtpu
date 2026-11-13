# SPDX-License-Identifier: Apache-2.0
"""SenseNova-U1's starting noise drawn on the host in float32, for TPU workers.

Runtime patch, tracked debt:
- Why: ``t2i_generate`` draws its pixel-space noise with ``torch.randn`` in the
  model dtype on the model device. torch_tpu builds a bfloat16 normal from 7
  random bits: 128 values in [-2.89, 2.52] with mean -0.011, and from that
  noise the checkpoint renders near-black images.
- How: :func:`install` rebinds ``modeling_neo_chat._randn_with_seed`` (called by
  ``t2i_generate``). For a non-CPU device it takes the stock CPU path in float32
  with the same seeds, rounds to the requested dtype on the host and copies the
  result to the device: the noise of a CPU float32 run, rounded.
- Upstream PR or issue: none.
- Remove when: torch_tpu draws bfloat16 normals from float32 samples.
"""

from __future__ import annotations

import functools

import torch

_CPU = torch.device("cpu")


def install() -> None:
    from sglang.multimodal_gen.runtime.models.sensenova_u1.neo_unify import (
        modeling_neo_chat,
    )

    stock = modeling_neo_chat._randn_with_seed
    stock = getattr(stock, "__wrapped__", stock)

    @functools.wraps(stock)
    def _randn_with_seed(shape, *, device, dtype, seed):
        if torch.device(device).type == _CPU.type:
            return stock(shape, device=device, dtype=dtype, seed=seed)
        noise = stock(shape, device=_CPU, dtype=torch.float32, seed=seed)
        return noise.to(dtype).to(device)

    modeling_neo_chat._randn_with_seed = _randn_with_seed
