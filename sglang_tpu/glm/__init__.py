# SPDX-License-Identifier: Apache-2.0
"""GLM-specific TPU loading capabilities."""


def register_loading_capabilities() -> None:
    from sglang_tpu.ep_local_expert_reader import register_ep_local_expert_read_model

    # Both use DeepseekV2WeightLoaderMixin's per-expert mapping. NEXTN remaps
    # model.layers.<N> to model.decoder without changing expert IDs. Shared
    # experts are separate keys; the reader rejects fused shared experts/EPLB.
    for name in ("GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"):
        register_ep_local_expert_read_model(f"sglang.srt.models.glm4_moe.{name}")
