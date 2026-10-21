# SPDX-License-Identifier: Apache-2.0
"""TPU decode retraction checks.

reject_host_pool_retraction
  Why: Host-pool decode retraction moves KV through the generic L2 transfer
       engine, which bypasses the TPU cache controllers. The restore needs a
       per-layer load that neither TPU host pool implements (the raiden pool
       raises NotImplementedError, the torch_tpu pool fails on is_draft), and the
       raiden backup's DMA future is never awaited before the device pages are
       freed. The first retraction restore would kill the scheduler.
  How: resolution hook on handle_cache_compatibility; runs the built-in step,
       then rejects host_pool on a TPU decode server.
  Upstream PR or issue: none; the gap is in the sglang_tpu host pools, not upstream.
  Remove when: host-pool retraction goes through the TPU cache controllers
       (all-layer restore, awaited raiden backup).
"""

from __future__ import annotations

from typing import Any, Callable

from sglang.srt.arg_groups.overrides import resolving_view


def reject_host_pool_retraction(
    server_args: Any, previous: Callable[[Any], None]
) -> None:
    previous(server_args)
    cfg = resolving_view(server_args)
    if (
        cfg.disaggregation_decode_retraction_backup != "host_pool"
        or cfg.device != "tpu"
    ):
        return
    if cfg.disaggregation_decode_host_receive_threshold > 0:
        raise ValueError(
            "Decode host receive (--disaggregation-decode-host-receive-"
            "threshold > 0) requires host_pool retraction, which is not "
            "supported on TPU. Set the threshold to 0."
        )
    raise ValueError(
        "--disaggregation-decode-retraction-backup=host_pool is not "
        "supported on TPU: the TPU host pools cannot restore through the "
        "host-pool retraction path. Use cpu_tensor (the TPU default); set "
        "SGLANG_RAIDEN_RETRACT_HOST_GB to stage whole pages through raiden."
    )
