"""Decorator-driven precompile for fixed-shape TPU Spec V2 helper ops."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence

import torch

from sglang.srt.compilation.compiled_op import get_compiled_ops
from sglang.srt.hardware_backend.tpu.spec_op_warmup import SpecOpWarmupContext
from sglang.srt.speculative import eagle_utils as _eagle_utils  # noqa: F401

logger = logging.getLogger(__name__)


def precompile_spec_v2_ops(
    *,
    device: torch.device,
    physical_batch_buckets: Sequence[int],
    num_verify_tokens: int,
    vocab_size: int,
    draft_vocab_size: int,
    hidden_size: int,
    hidden_dtype: torch.dtype,
) -> None:
    """Warm up Spec V2 helper graphs for every physical batch bucket."""
    shapes = tuple(
        (physical_bs, physical_bs * num_verify_tokens)
        for physical_bs in physical_batch_buckets
    )
    if not shapes:
        return

    warmup_contexts = [
        SpecOpWarmupContext(
            device=device,
            physical_bs=physical_bs,
            num_verify_tokens=num_verify_tokens,
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
            hidden_size=hidden_size,
            hidden_dtype=hidden_dtype,
        )
        for physical_bs, _ in shapes
    ]
    registered_ops = get_compiled_ops()

    start = time.perf_counter()
    with torch.no_grad():
        for context in warmup_contexts:
            for registered_op in registered_ops:
                args = registered_op.make_warmup_args(context)
                registered_op.op(*args)
    if device.type == "tpu":
        torch.get_device_module(device).synchronize()
    logger.info(
        "[spec op precompile] precompiled %d ops for shapes %s in %.1fs",
        len(registered_ops),
        shapes,
        time.perf_counter() - start,
    )
