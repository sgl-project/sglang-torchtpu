# SPDX-License-Identifier: Apache-2.0
"""Bootstrap readiness across the stages of a pipeline-parallel prefill."""

from sglang.srt.disaggregation.base.conn import KVPoll


def poll_with_pp_consensus(sender) -> KVPoll:
    """PP stage 0's poll of a bootstrapping room on a pipeline-parallel prefill.

    Only stage 0 polls rooms while they bootstrap, so every stage marks a room
    WaitingForInput when its decode request arrives, without being polled. This
    is stage 0's own poll held at the least advanced status any stage reports:
    the room starts once every stage has its decode request, and fails if any
    stage failed it.
    """
    status = sender.poll()
    if status != KVPoll.WaitingForInput:
        return status
    return min(status, sender.kv_mgr.check_status_pp_consensus(sender.bootstrap_room))
