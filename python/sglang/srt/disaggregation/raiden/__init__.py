from sglang.srt.disaggregation.raiden.conn import (
    RaidenKVBootstrapServer,
    RaidenKVManager,
)
from sglang.srt.disaggregation.raiden.decode import (
    RaidenDecodeKVManager,
    RaidenKVReceiver,
)
from sglang.srt.disaggregation.raiden.prefill import (
    RaidenKVSender,
    RaidenPrefillKVManager,
)
from sglang.srt.disaggregation.utils import DisaggregationMode


def create_raiden_kv_manager(
    args, disaggregation_mode, server_args, is_mla_backend=False
):
    manager_cls = (
        RaidenPrefillKVManager
        if disaggregation_mode == DisaggregationMode.PREFILL
        else RaidenDecodeKVManager
    )
    return manager_cls(args, disaggregation_mode, server_args, is_mla_backend)


__all__ = [
    "RaidenKVManager",
    "RaidenPrefillKVManager",
    "RaidenDecodeKVManager",
    "create_raiden_kv_manager",
    "RaidenKVSender",
    "RaidenKVReceiver",
    "RaidenKVBootstrapServer",
]
