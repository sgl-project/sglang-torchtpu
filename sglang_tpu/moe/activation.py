# SPDX-License-Identifier: Apache-2.0
"""Pallas MoE activation names from a FusedMoE layer's runner config."""

from __future__ import annotations

import torch


def pallas_moe_activation(activation: str, layer: torch.nn.Module) -> str:
    """SiTU carries its (beta, linear_beta), the runner's gemm1 alpha / clamp, inline."""
    if activation != "situ":
        return activation
    config = layer.moe_runner_config
    if config.gemm1_alpha is None:
        raise ValueError("SiTU MoE requires gemm1_alpha (SiTU beta).")
    linear_beta = (
        "none" if config.gemm1_clamp_limit is None else config.gemm1_clamp_limit
    )
    return f"situ:{config.gemm1_alpha}:{linear_beta}"
