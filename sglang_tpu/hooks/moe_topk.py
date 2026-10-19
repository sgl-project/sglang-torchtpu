# SPDX-License-Identifier: Apache-2.0
"""Fused router top-k for small batches on TPU.

Why: topk.biased_topk_impl (sigmoid + correction bias + top-k) lowers on TPU
     to an XLA sort over every expert plus a gather and reductions, about
     7 us per MoE layer for Kimi-K3's top-16 of 896 at batch 1.
How: AROUND topk.biased_topk_impl and topk.biased_grouped_topk_impl (single
     expert group only, which Kimi-K3 uses): batches of at most 64 tokens
     with sigmoid scoring, a shared f32 bias vector, no fused shared experts
     and a validated expert count (896, Kimi-K3) run one Pallas kernel
     (sglang_tpu/layers/pallas/router_topk.py); everything else calls the
     original. SGLANG_TPU_ENABLE_ROUTER_TOPK_KERNEL=0 turns it off.
     AFTER TopK.__init__ registers the kernel's custom op before
     torch.compile traces the model.
Upstream PR or issue: sgl-project/sglang-torchtpu#1442 ([Plugin seam]
     moe/topk: backend override for biased top-k routing).
Remove when: topk.py consults a router backend registry filled from
     init_backend() (or BaseFusedOp honors register_oot_forward on in-tree
     platforms), per #1442.
"""


def register_router_topk(result, *args, **kwargs):
    from sglang_tpu.layers.moe.router_topk_tpu import register_router_topk_op

    register_router_topk_op()
    return result


def biased_topk(original, *args, **kwargs):
    from sglang_tpu.layers.moe.router_topk_tpu import biased_topk_around

    return biased_topk_around(original, *args, **kwargs)


def biased_grouped_topk(original, *args, **kwargs):
    from sglang_tpu.layers.moe.router_topk_tpu import biased_grouped_topk_around

    return biased_grouped_topk_around(original, *args, **kwargs)
