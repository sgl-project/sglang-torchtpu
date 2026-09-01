# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/utils.py

from typing import Any

import torch


def set_weight_attrs(
    weight: torch.Tensor,
    weight_attrs: dict[str, Any] | None,
):
    """Set attributes on a weight tensor without overwriting existing ones."""
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f"Overwriting existing tensor attribute: {key}"

        setattr(weight, key, value)
