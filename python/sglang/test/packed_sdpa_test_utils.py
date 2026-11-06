# SPDX-License-Identifier: Apache-2.0
"""Synthetic packed-attention checks, independent of model startup and weights."""

import importlib.util
import logging
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

import sglang


@lru_cache(maxsize=1)
def load_sdpa_module():
    """Load the real SDPA/base classes without the serving package initializer.

    Only logging and platform enum imports are isolated. Tensor operations and
    attention dispatch are real; this does not test platform/backend selection.
    """
    directory = (
        Path(sglang.__file__).parent
        / "multimodal_gen/runtime/layers/attention/backends"
    )
    platforms = ModuleType("sglang.multimodal_gen.runtime.platforms")
    platforms.AttentionBackendEnum = SimpleNamespace(
        TORCH_SDPA="torch_sdpa",
        TORCH_CUDNN_SDPA="torch_cudnn_sdpa",
        DYNAMIC_CUDNN_SDPA="dynamic_cudnn_sdpa",
    )
    logging_utils = ModuleType("sglang.multimodal_gen.runtime.utils.logging_utils")
    logging_utils.init_logger = logging.getLogger

    def load(name, filename):
        spec = importlib.util.spec_from_file_location(name, directory / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    with patch.dict(
        sys.modules,
        {platforms.__name__: platforms, logging_utils.__name__: logging_utils},
    ):
        base = load("_packed_sdpa_base_under_test", "attention_backend.py")
        with patch.dict(
            sys.modules,
            {
                "sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend": base,
            },
        ):
            module = load("_packed_sdpa_under_test", "sdpa.py")
    # Keep the unique modules available to Dynamo after patch.dict restores
    # the package imports. Do not replace any real serving modules globally.
    sys.modules[base.__name__] = base
    sys.modules[module.__name__] = module
    return module


def packed_reference(query, key, value, bounds, *, scale, causal=False):
    """FP64 CPU matmul/softmax with a block-diagonal mask, not an SDPA call."""
    q, k, v = (tensor.detach().cpu().double() for tensor in (query, key, value))
    if q.shape[0] == 0:
        return q.new_empty((0, q.shape[1], v.shape[-1]))
    repeats = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(repeats, dim=1)
    v = v.repeat_interleave(repeats, dim=1)
    allowed = torch.zeros(q.shape[0], k.shape[0], dtype=torch.bool)
    for start, stop in zip(bounds[:-1], bounds[1:]):
        allowed[start:stop, start:stop] = True
    if causal:
        allowed = allowed.tril()
    scores = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2)) * scale
    scores.masked_fill_(~allowed, -float("inf"))
    return (scores.softmax(dim=-1) @ v.transpose(0, 1)).transpose(0, 1)


def packed_inputs(bounds, heads, kv_heads, dim, value_dim, dtype, *, strided=False):
    generator = torch.Generator().manual_seed(42)
    tensors = []
    for num_heads, width in ((heads, dim), (kv_heads, dim), (kv_heads, value_dim)):
        tensor = torch.randn(
            bounds[-1],
            num_heads,
            width * (2 if strided else 1),
            dtype=dtype,
            generator=generator,
        )
        tensors.append(tensor[..., ::2] if strided else tensor)
    return tuple(tensors)
