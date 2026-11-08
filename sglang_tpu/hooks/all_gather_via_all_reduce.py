# SPDX-License-Identifier: Apache-2.0
"""Logits all-gather through the in-place all-reduce in Dynamo-traced TPU graphs.

logits gathers
  Why: under the Dynamo tracer the reg_all_gather_into_tensor custom op returns
       wrong blocks from cross-chip peers on a TP group wider than one chip
       (tpu7x TP >= 4). The LM head's vocabulary gather then corrupts most of
       the logits, which shows up as a wrong first token after prefill
       (sgl-project/sglang-torchtpu#1700). The in-place all-reduce compiles
       correctly.
  How: AFTER LogitsProcessor.__init__ wraps _logits_gatherer (TP logits
       gather) and AROUND LogitsProcessor._gather_attn_tp_logits (attention-TP
       logits gather with --enable-dp-lm-head). While Dynamo traces
       (torch.compiler.is_dynamo_compiling()) on a multi-rank group, the gather
       is rebuilt from one all-reduce over a zero-padded [world, *block] buffer
       (sglang_tpu.all_gather_via_all_reduce). GroupCoordinator collectives
       stay native, so per-layer gathers keep their cost (#2013).
  Upstream PR or issue: sgl-project/sglang-torchtpu#1700 (TorchTPU compiled
       all_gather_into_tensor across chips).
  Remove when: the compiled all_gather_into_tensor returns the same blocks as
       eager on a cross-chip TP group, checked with the first-token sweep in
       #1700 at TP 8.
"""

from sglang_tpu.all_gather_via_all_reduce import (
    gather_attn_tp_logits_around as gather_attn_tp_logits,
)
from sglang_tpu.all_gather_via_all_reduce import wrap_logits_gatherer

__all__ = ["gather_attn_tp_logits", "wrap_logits_gatherer"]
