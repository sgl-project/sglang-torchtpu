# SPDX-License-Identifier: Apache-2.0
"""Switch and startup checks for prefill-driven decode KV reservation.

Kept apart from hooks/prefill_driven.py so server-argument validation does not
import the hook modules and the scheduler code they load.
"""

from __future__ import annotations

from typing import Any

from sglang.srt.environ import envs
from sglang_tpu.envs import envs as tpu_envs


def is_enabled() -> bool:
    return tpu_envs.SGLANG_ENABLE_DISAGG_PREFILL_DRIVEN.get()


def validate(cfg: Any) -> None:
    """Startup checks. A pair that differs in attention TP fails per request,
    in RaidenKVReceiver, since neither side knows the other at startup."""
    if not is_enabled():
        return
    if tpu_envs.SGLANG_DISAGG_PREFILL_DRIVEN_SLOTS.get() < 0:
        raise ValueError("SGLANG_DISAGG_PREFILL_DRIVEN_SLOTS must be nonnegative")
    unsupported = []
    if cfg.enable_priority_scheduling and not cfg.disable_priority_preemption:
        unsupported.append("priority preemption (set --disable-priority-preemption)")
    if cfg.enable_lora_overlap_loading:
        unsupported.append("LoRA overlap loading")
    if cfg.disaggregation_mode not in ("prefill", "decode"):
        unsupported.append("aggregated serving")
    if cfg.disaggregation_transfer_backend != "raiden":
        unsupported.append(
            f"the {cfg.disaggregation_transfer_backend} transfer backend"
        )
    if cfg.disaggregation_decode_allocation_policy != "early":
        unsupported.append(
            f"decode allocation policy {cfg.disaggregation_decode_allocation_policy}"
        )
    if cfg.pp_size != 1:
        unsupported.append("pipeline parallelism")
    if cfg.enable_pdmux or cfg.decoupled_spec_role != "null":
        unsupported.append("PD multiplexing or decoupled speculation")
    if cfg.enable_pd_role_switch:
        unsupported.append("runtime PD role switching")
    # DSPARK retraction re-runs the prefill through /generate, which would
    # need a new reservation outside the supported request lifecycle.
    if (cfg.speculative_algorithm or "").upper() == "DSPARK":
        unsupported.append("DSPARK speculative decoding")
    if cfg.encoder_only or cfg.language_only:
        unsupported.append("encoder disaggregation")
    if envs.SGLANG_DISAGG_STAGING_BUFFER.get():
        unsupported.append("staging buffers")
    if envs.SGLANG_RUST_SERVER.get():
        unsupported.append("the Rust bootstrap registry")
    if unsupported:
        raise ValueError(
            "SGLANG_ENABLE_DISAGG_PREFILL_DRIVEN does not yet support "
            + ", ".join(unsupported)
        )
