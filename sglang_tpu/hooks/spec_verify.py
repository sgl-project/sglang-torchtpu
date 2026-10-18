# SPDX-License-Identifier: Apache-2.0
"""Fused greedy verify of a top-1 EAGLE / NEXTN draft chain.

fused_greedy_chain_verify
  Why: eagle_sample's generic greedy path is about ten eager launches per
       round on TPU (buffer preallocation, slice copies, simulated-acceptance
       ops) plus the bookkeeping and draft-extend input ops after it; the TPU
       round is host-bound, so each launch is round latency.
  How: AROUND eagle_utils.fused_greedy_chain_verify (None in core): one
       compiled op (eagle_utils_tpu.greedy_verify_chain_tpu) returns the
       verify outputs and the commit outputs.
  Upstream PR or issue: none (the seam is core's fused_greedy_chain_verify).
  Remove when: never; this is the TPU implementation of that seam.
"""

from __future__ import annotations


def fused_greedy_chain_verify(original, verify_input, next_token_logits, seq_lens):
    from sglang_tpu.eagle_utils_tpu import fused_greedy_chain_verify_tpu

    return fused_greedy_chain_verify_tpu(verify_input, next_token_logits, seq_lens)
