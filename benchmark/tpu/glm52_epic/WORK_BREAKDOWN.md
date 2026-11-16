# GLM5.2 draft snapshot — do not merge

This PR is an inventory for independent work extraction, not a monolithic merge.
Original experiment branch `epic/glm52-channelwise-port-20261113` at `ca8dbe8e62`
retains all historical evidence. Draft rebased on main
`97afaea93765ce3546806481cab3b5c3bc0beb09` and then removed historical diagnostic
code, observer/dump capture, custom admission, raw experiment archives, and the
unrelated qblock heuristic. Untracked request-table drafts remain untouched in
the original worktree.

## Production diff relative to main

Only three production files differ:

- `sglang_tpu/glm52_channelwise_fp8.py`: explicit opt-in checkpoint ABI, output-channel
  scales, dense/expert quant-method adapters. Not a JAX kernel-parity implementation.
- `sglang_tpu/tpu_platform.py`: TPU FP8 config factory delegates ordinary layouts
  upstream and enables explicit specialized layout.
- `sglang_tpu/ep_local_expert_reader.py`: exact GLM class opt-in to existing safe
  routed-expert range selection, preserving existing exclusions.

Two unit files cover ABI and reader changes. Remaining benchmark code is only
the compiled staged runner/bootstrap, exact-range prep, and native text/long HTTP
clients. `README.md` documents the execution and original hardware results.
`KERNEL_ROUTE_COMPARISON.md` provides concrete source routes and precision gaps.

## Proposed extraction order

| Work item | Scope | Independent acceptance |
|---|---|---|
| 1. Checkpoint ABI | plugin FP8 factory + scale loading + unit tests | ordinary FP8 unchanged; exact dense/expert scale/shard oracles; checkpoint E2E |
| 2. Host loading | exact GLM EP eligibility + staged range prep | index hash, exact ranges, rank coverage, bounded memory, all-rank startup |
| 3. Dense precision alignment | JAX W8A16 dense versus Torch W8A8 | matched layer inputs/weights/scales, logits before speed claims |
| 4. Attention/indexer parity | per-query Torch versus private FP8-QK/PV and BF16 indexer | explicit operand/route contract, prefill/decode/cache-boundary numerics |
| 5. MoE numeric policy | scale and GMM accumulator precision | same routed weights/inputs, tolerance justified before fusion changes |
| 6. Distributed MoE fusion | current GMM/SC versus fused EP/shared experts | dispatch/combine, collectives, shared/reduction ownership |
| 7. Packed norm/RoPE | independently shape-gated optimizations | ordinary fallback preserved, numeric tests and hardware validation |
| 8. Native benchmark/accuracy | minimal clients and runner | actual chat + reference accuracy, matched cache/active cadence, host-time labeling |

## Validation boundaries

Original pinned commits passed 20/20 chat checks and sixteen routed cold long
requests. See README for IDs and timings. This is not matched JAX performance or
full accuracy certification; cold versus cached greedy outputs differ.

Rebased/cleaned head has NOT been rerun on TPU. Source/config parsing, local HTTP
client checks and core TPU-debt ratchet are local checks only. Full Torch loader
and OOT tests cannot run in the current local interpreter without Torch.
Historical CPU/TPU results must not be reused as new-head validation.
