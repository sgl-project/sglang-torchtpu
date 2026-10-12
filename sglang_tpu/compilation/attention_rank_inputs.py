"""Bind shared attention rank state to explicit TPU direct-tracing inputs."""

from contextlib import contextmanager

import torch

from sglang.srt.layers import dp_attention


def get_attention_rank_tensors(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Expose initialized rank state as explicit direct-compiler inputs."""
    if not dp_attention._RANK_TENSORS:
        return ()
    return tuple(
        dp_attention._rank_tensor(axis, device) for axis in ("attn_tp", "attn_dp")
    )


@contextmanager
def use_attention_rank_tensors(tensors: tuple[torch.Tensor, ...]):
    """Bind traced placeholders for rank selectors, restoring eager state on exit.

    Dynamo lifts the global rank tensors automatically; direct tracing needs
    explicit inputs and must make the selectors read their placeholders.
    Like attention metadata, this scope runs synchronously in a worker.
    """
    previous = dp_attention._RANK_TENSORS
    dp_attention._RANK_TENSORS = previous.copy()
    try:
        for axis, tensor in zip(("attn_tp", "attn_dp"), tensors):
            dp_attention._RANK_TENSORS[(axis, tensor.device)] = tensor
            dp_attention._RANK_TENSORS[(axis, torch.device(tensor.device.type))] = (
                tensor
            )
        yield
    finally:
        dp_attention._RANK_TENSORS = previous
