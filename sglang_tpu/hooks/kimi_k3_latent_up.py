# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 MoE tail add inside the deferred latent up-proj kernel (opt-in).

kimi_k3_latent_up_tail_add3
  Why: with SGLANG_TPU_K3_LATENT_UP_ATTN_RES the host-split latent up-proj of a
       single-token MoE layer is deferred, and the layer's tail add
       bf16(out + shared) [+ prefix_sum] runs inside that GEMV's Pallas kernel
       together with the next layer's attention-residual aggregation
       (sglang_tpu.kimi_k3.latent_up_tail). Core has no seam to hand that add
       to a backend.
  How: AROUND sglang.srt.models.kimi_k3._add3. Calls the original unless a
       deferred up-proj is pending (only set by the plugin's binding, which is
       off unless the flag is set when the model loads).
  Upstream PR or issue: none yet; the seam would be a backend registry for the
       MoE tail add, like the attention-residual aggregate registry of #1428.
  Remove when: core lets a backend replace the Kimi-K3 MoE tail add.
"""


def kimi_k3_latent_up_tail_add3(original, *args, **kwargs):
    from sglang_tpu.kimi_k3.latent_up_tail import add3_around

    return add3_around(original, *args, **kwargs)
