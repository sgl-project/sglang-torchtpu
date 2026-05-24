"""No-op TPU graph runner stub for Phase 1 (eager mode only).

Phase 1 disables graph capture via support_cuda_graph() -> False,
so this class is never instantiated. It exists to satisfy the
factory method contract in TpuSRTPlatform.
"""

from __future__ import annotations

from sglang.srt.hardware_backend.tpu._logging import get_logger

logger = get_logger(__name__)


class TpuGraphRunner:

    def __init__(self, model_runner):
        raise NotImplementedError(
            "TPU graph runner not yet implemented. "
            "Ensure --disable-cuda-graph is set."
        )
