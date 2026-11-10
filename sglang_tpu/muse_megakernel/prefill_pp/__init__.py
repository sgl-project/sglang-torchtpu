# SPDX-License-Identifier: Apache-2.0
"""Muse Spark prefill, pipeline-parallel: one stage's layers whole on one device, no collectives.

A stage holds its layers whole (every head, all experts, the full pooler and unpooler) on one
device, so nothing in it talks to another device; the residual stream moves between stages
outside. Per layer slot:

  attn_in    (fused Pallas) qkvg projection, qk norms, rope -> q, k, v, gate
  attention  sglang_tpu's batched RPA prefill kernel, as the P/D prefill serves it: the
             chunk's K/V into the slot's fp8 SEQ_ALONG_LANE pool, attention over the pool
             (cached context and the causal chunk), each head's output RMS-normalized
  route      (fused Pallas) gated o projection and residual, router top-k, expert projection
  experts    sglang_tpu's fused MoE (sort, gather, gmm_v2 x2, un-permute), as served:
             NVFP4 W4A16, or fp8 W8A8 on the stage's fp8 layer
  combine    (fused Pallas) post-expert norm per slot, weighted slot sum, unpooler, gated
             residual

Every stage has the same slot layout (cfg.slots layers of dense weights, global attention at
cfg.global_slots, NVFP4 experts per slot and one fp8 expert set), so one program serves every
stage; a per-stage slot table says which slots hold a layer and which take the fp8 experts.
"""

from sglang_tpu.muse_megakernel.prefill_pp.config import (
    BF16,
    F32,
    F_FP8,
    F_GLOBAL,
    F_LAYER,
    F_VALID,
    FP8,
    H_LAYERS,
    H_STAGE,
    HDR,
    NF,
    PHASES,
    V_ROWS,
    WEIGHT_ORDER,
    PPConfig,
    V,
    expert_shapes,
    pool_shape,
    skipped_phases,
    slot_table,
    stage_layers,
    weight_shapes,
)
from sglang_tpu.muse_megakernel.prefill_pp.stage import stage_forward

__all__ = [
    "BF16", "F32", "F_FP8", "F_GLOBAL", "F_LAYER", "F_VALID", "FP8", "H_LAYERS", "H_STAGE",
    "HDR", "NF", "PHASES", "V", "V_ROWS", "WEIGHT_ORDER", "PPConfig", "expert_shapes",
    "pool_shape", "skipped_phases", "slot_table", "stage_forward", "stage_layers",
    "weight_shapes",
]  # fmt: skip
