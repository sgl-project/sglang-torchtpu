# SPDX-License-Identifier: Apache-2.0
"""Environment variables owned by the TPU plugin.

Declare new TPU-only variables here with the descriptor types from
`sglang.srt.environ`. Existing ``SGLANG_TPU_*`` entries stay in core until their
core readers move; never declare one name in both registries.
"""

from __future__ import annotations

from sglang.srt.environ import EnvBool, EnvField, EnvFloat, EnvInt

__all__ = ["TpuEnvs", "envs"]

# Core closes EnvField.__set_name__ after its own Envs, reopen it for this class.
EnvField._allow_set_name = True


class TpuEnvs:
    # Host heap (see host_heap_trim.py)
    SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED = EnvBool(True)
    # Seconds between periodic trims. 0 turns the periodic trim off.
    SGLANG_TPU_MALLOC_TRIM_INTERVAL_SEC = EnvFloat(0.0)

    # P/D /health_generate: 503 once in-flight requests produced nothing for
    # WINDOW seconds; COLD_WINDOW applies until the first output after start.
    SGLANG_TPU_PD_HEALTH_READINESS_ENABLED = EnvBool(True)
    SGLANG_TPU_PD_HEALTH_WINDOW_S = EnvFloat(600.0)
    SGLANG_TPU_PD_HEALTH_COLD_WINDOW_S = EnvFloat(1200.0)
    # Per-peer judgement on a decode: a prefill peer is failing after STREAK
    # consecutive engine aborts of its requests with no success in between;
    # the decode is unhealthy once failing peers reach RATIO of the peers active
    # within the window, with at least MIN active. RATIO 0 turns it off.
    SGLANG_TPU_PD_HEALTH_PEER_FAILURE_RATIO = EnvFloat(0.5)
    SGLANG_TPU_PD_HEALTH_PEER_FAILURE_STREAK = EnvInt(3)
    SGLANG_TPU_PD_HEALTH_PEER_MIN = EnvInt(1)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
