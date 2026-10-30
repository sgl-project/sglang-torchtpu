"""tpu-sync's weight-update handlers, served through /plugin_weight_update."""

from __future__ import annotations

from typing import Any

from sglang.srt.managers.scheduler_components.weight_updater import (
    PluginWeightUpdateContext,
)
from sglang.srt.utils.network import get_local_ip_auto
from sglang_tpu.weight_sync.tpu_sync_receiver import TpuSyncReceiver

_RECEIVER = "tpu_sync_receiver"


def init_receiver(
    *, context: PluginWeightUpdateContext, payload: dict[str, Any]
) -> None:
    """Register this TP rank's staging buffers with the trainer's controller, replacing earlier ones."""
    if context.has_draft_model:
        raise ValueError("tpu-sync weight sync does not cover draft models")
    # free the old staging buffers before allocating new ones
    context.state.pop(_RECEIVER, None)
    context.state[_RECEIVER] = TpuSyncReceiver(
        model=context.model,
        controller_address=payload["controller_address"],
        job_name=payload["job_name"],
        weight_shapes=payload["weight_shapes"],
        parallelism=payload["parallelism"],
        tp_size=context.tp_size,
        tp_rank=context.tp_rank,
        num_kv_heads=context.model_config.get_total_num_kv_heads(),
        bind_ip=get_local_ip_auto(),
    )


def install(*, context: PluginWeightUpdateContext, payload: dict[str, Any]) -> int:
    """Copy the last transfer into the model; returns the parameters it changed when ``check_parity`` is set."""
    if not context.session_open or _RECEIVER not in context.state:
        raise RuntimeError(
            "tpu_sync.install needs tpu_sync.init_receiver and an open begin_weight_update() session"
        )
    return context.state[_RECEIVER].install(
        context.model, check_parity=payload.get("check_parity", False)
    )


WEIGHT_UPDATE_HANDLERS = {
    "tpu_sync.init_receiver": init_receiver,
    "tpu_sync.install": install,
}
