"""TPU P/D disaggregation transfer backends.

``base`` holds the backend-agnostic control-plane scaffolding; ``conn`` is the
default host ZMQ-push backend (the ``tpu`` transfer backend). A future opt-in
backend (e.g. ``tpu_sync``) subclasses the same base classes and implements
only the data-plane seam — see ``base.py`` for the contract.
"""

from sglang.srt.disaggregation.tpu.base import (
    KVTransferError,
    TpuKVBootstrapServerBase,
    TpuKVManagerBase,
    TpuKVReceiverBase,
    TpuKVSenderBase,
)
from sglang.srt.disaggregation.tpu.conn import (
    TpuKVBootstrapServer,
    TpuKVManager,
    TpuKVReceiver,
    TpuKVSender,
)

__all__ = [
    # Default host backend (the ``tpu`` transfer backend).
    "TpuKVManager",
    "TpuKVSender",
    "TpuKVReceiver",
    "TpuKVBootstrapServer",
    "KVTransferError",
    # Backend-agnostic bases (the seam a new backend plugs into).
    "TpuKVManagerBase",
    "TpuKVSenderBase",
    "TpuKVReceiverBase",
    "TpuKVBootstrapServerBase",
]
