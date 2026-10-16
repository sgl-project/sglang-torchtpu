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
    # GLM-5.3-Flash: attend densely (every causal key, on pallas_tpu_mla) with
    # contexts up to this many tokens, instead of DeepSeek Sparse Attention.
    # Exact up to index_topk; past it, outputs no longer match the model.
    # Unset = DSA. See sglang_tpu/models/glm5_next_args.py.
    SGLANG_TPU_GLM_FLASH_DENSE_MAX_CONTEXT = EnvInt(None)

    @classmethod
    def names(cls) -> list[str]:
        return sorted(
            name for name, field in vars(cls).items() if isinstance(field, EnvField)
        )


envs = TpuEnvs()
EnvField._allow_set_name = False
