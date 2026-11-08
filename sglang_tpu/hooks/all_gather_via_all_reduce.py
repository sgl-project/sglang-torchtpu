# SPDX-License-Identifier: Apache-2.0
"""All-gather through the in-place all-reduce in Dynamo-traced TPU graphs.

all_gather
  Why: under the Dynamo tracer the reg_all_gather_into_tensor custom op returns
       wrong blocks from cross-chip peers on a TP group wider than one chip
       (tpu7x TP >= 4). The LM head's vocabulary gather then corrupts most of
       the logits, which shows up as a wrong first token after prefill
       (sgl-project/sglang-torchtpu#1700). The in-place all-reduce compiles
       correctly.
  How: AROUND GroupCoordinator.all_gather (the concat-style API behind
       tensor_model_parallel_all_gather and the logits gather); while Dynamo
       traces (torch.compiler.is_dynamo_compiling()), the gather is rebuilt
       from one all-reduce over a zero-padded [world, *block] buffer
       (sglang_tpu.all_gather_via_all_reduce). Eager calls, direct-tracer
       graphs and direct all_gather_into_tensor callers (DP-attention, attn-TP
       and CP gathers) use the native collective (#2013).
  Upstream PR or issue: sgl-project/sglang-torchtpu#1700 (TorchTPU compiled
       all_gather_into_tensor across chips).
  Remove when: the compiled all_gather_into_tensor returns the same blocks as
       eager on a cross-chip TP group, checked with the first-token sweep in
       #1700 at TP 8.
"""

from sglang_tpu.all_gather_via_all_reduce import all_gather_around as all_gather

__all__ = ["all_gather"]
