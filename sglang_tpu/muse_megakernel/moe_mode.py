# SPDX-License-Identifier: Apache-2.0
"""The Muse MoE expert layout: one mode drives the decode megakernel and the prefill program."""

import os
from typing import Literal

from sglang_tpu.envs import envs as tpu_envs

# tp: 1/tp of every expert per rank (expert-TP megakernel, fused-RS expert-TP prefill MoE);
# ep: whole experts per rank (ep_mk megakernel, EP prefill MoE).
MoeMode = Literal["tp", "ep"]

_LEGACY_ENV = "MUSE_PF_MOE_EP"


def check_no_legacy_env() -> None:
    if _LEGACY_ENV in os.environ:
        raise ValueError(
            f"{_LEGACY_ENV} is replaced by SGLANG_TPU_MUSE_MOE_MODE (tp | ep | auto); unset it"
        )


def parse_moe_mode(value: str, ep_size: int) -> MoeMode:
    """auto picks the layout from the server's EP size, never from a step's rows."""
    check_no_legacy_env()
    if value == "auto":
        return "ep" if ep_size > 1 else "tp"
    if value == "tp":
        return "tp"
    if value == "ep":
        return "ep"
    raise ValueError(f"SGLANG_TPU_MUSE_MOE_MODE={value!r}: tp, ep or auto")


def resolve_moe_mode(ep_size: int) -> MoeMode:
    return parse_moe_mode(tpu_envs.SGLANG_TPU_MUSE_MOE_MODE.get(), ep_size)
