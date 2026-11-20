# SPDX-License-Identifier: Apache-2.0
"""The EP decode megakernel: muse_megakernel.decode.tp.kernel's row-tiled program with an EP FFN section
on 32 whole experts per rank, and the layout of its weights. Two arms (config.ep_arm): a2a
(EPExperts: sequence-parallel, two ragged all-to-alls) and gather (EPGatherExperts: every row
on every rank, small steps)."""

from sglang_tpu.muse_megakernel_v2.common.config import MKConfig
from sglang_tpu.muse_megakernel_v2.decode.spep.config import EPArm, ep_arm, ep_mk_config
from sglang_tpu.muse_megakernel_v2.decode.spep.gather import EPGatherExperts
from sglang_tpu.muse_megakernel_v2.decode.spep.layer import EPExperts
from sglang_tpu.muse_megakernel_v2.decode.tp.kernel import megakernel_call

__all__ = ["EPArm", "ep_arm", "ep_megakernel_call"]


def ep_megakernel_call(cfg: MKConfig, axis: str = "model", *, arm: EPArm):
    """megakernel_call's f(meta, pos, pages_full, pages_swa, emb, *weights, *experts, *kv), with
    each layer's experts as config.ep_expert_shapes (common.ep_layout.ep_expert_weights),
    wpool / wunpool replicated ([L, H, H_e] and [L, H_e, H] fp8, their scales whole in the
    smalls rows) and the pooled width in kperm order (wpool's columns, wpool_s, pre_expert_w).
    An arm that cannot serve the step raises."""
    cfg = ep_mk_config(cfg)
    if arm is EPArm.A2A:
        hook: EPExperts | EPGatherExperts = EPExperts(cfg)
    elif arm is EPArm.GATHER:
        hook = EPGatherExperts(cfg)
    else:
        raise ValueError(f"EP arm {arm!r}: a2a or gather")
    return megakernel_call(cfg, axis, ep=hook)
