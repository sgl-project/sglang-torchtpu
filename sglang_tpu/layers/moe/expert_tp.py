# SPDX-License-Identifier: Apache-2.0
"""Expert-TP selection for the fused reduce-scatter MoE; importable without torch_tpu or jax."""


def is_fused_moe_rs_expert_tp(moe_ep_size: int, moe_tp_size: int, tp_size: int) -> bool:
    """Whether the MoE shards every expert's intermediate dim over all TP ranks.

    Fused-RS then runs each rank's slice of all experts as virtual experts and
    sums the partial outputs (``expert_tp=True``) instead of expert parallelism.
    Only an explicit ``--moe-runner-backend pallas_tpu_fused_rs`` selects it, so
    TP-only MoE models on the auto backend keep their own path.
    """
    from sglang.srt.layers.moe import get_moe_runner_backend

    return (
        moe_ep_size == 1
        and moe_tp_size == tp_size > 1
        and get_moe_runner_backend().is_pallas_tpu_fused_rs()
    )
