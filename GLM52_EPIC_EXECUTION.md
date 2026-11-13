# GLM-5.2 channelwise epic execution ledger

## Scope and baseline

- Started: 2026-11-13 evening, Asia/Shanghai.
- Branch: `epic/glm52-channelwise-port-20261113`.
- Base: origin/main `d6f54cc2899b2f9b23dde8e6e0605cac6d405120`.
- Worktree: `/Users/jiongxuan/workspace/sgl-wt/glm52-epic-20261113`.
- Do not modify the original working tree or other teams' worktrees.
- Design: https://outline.infiscale-tech.com/doc/glm52-sglang-torchtpu-he-ru-6K5ujkPbhn
- Scope: GLM-5.2 channelwise, current TorchTPU precision first. Eight physical v7x chips first; no 16-chip/DCP work before the eight-chip milestone.
- Performance target: private JAX reference, not merely improvement over Torch baseline. Different precision/topology results must be labeled.

## Execution order

1. Freeze exact runnable private JAX reference, runtime, model mount and token inputs; reproduce eight-chip prefill without profiler.
2. Audit latest-main model loading and native two-host TPU startup; construct immutable Torch baseline runner/config, check FP8 channelwise interpretation and memory capacity before full traffic.
3. Measure same inputs and comparable stage boundaries; compile/warmup separate from steady latency/throughput. Establish eight-chip NEXTN decode paired reference after prefill.
4. Implement only measured missing capabilities, in independent commits: loader/platform support, MoE integration, sparse adapter, fusion/tuning. No broad private-branch merge.
5. Run independent correctness checks and applicable Flash regressions before accepting shared-path changes. Collect source/runtime/config/input hashes, logs, dispatch and metrics.
6. Preserve PR-sized commits on epic; do not open/merge downstream PRs automatically. Record untested states rather than claiming milestone completion.

## Reference run

- Historical source experiment: `exp-s31bemg2me`, fixed source `3695225808f0293cae422ea7adf55eddacadd43d`.
- Runtime: JAX/JAXLIB 0.11.1, libtpu 0.0.46.1; image `jax0.9.0-rev1`, runtime installed by pinned runner.
- Workload: 8 physical chips / two hosts, C32, unique prefix49152+extend12288, chunk2048, max-prefill/compile bucket32768, output16, no MTP, FP8 QK/PV.
- Reproduction submitted: `exp-266i34jcm1`, job `job-j41ovz6guo`, artifact `art-2964t9rlq6`; initially PENDING, not performance evidence yet.
- Historical manifest exported using Falcon clone; redacted numeric token/bucket envs restored to32768 from the pinned source's `benchmark/glm52/delivery/chunk2k_8chip_env.sh`.
- Runner timeout bounded to6600 seconds; selected cluster has two-hour running policy. No resource/lease owned by others was changed.
- Historical ~1241.417ms/chunk is a reference, not a reproduced result.

## Latest-main compatibility findings (provisional)

- Current support documentation marks full GLM DSA experimental. Mode A TPU path required; do not claim full OOT support.
- `--attn-dp-size` is now canonical; `--dp-size` denotes whole-model replicas. Deprecated launcher DP_SIZE translation must not be confused with JAX TP=DP=EP semantics.
- Eight physical chips means two hosts; TP16 is not automatically equivalent to private's attention-tensor=1 layout.

## Flash isolation

- Keep Flash recipes/default dispatch, kpool/open-tail, zero-RoPE, KDA/mHC and megakernel consumers unchanged initially.
- New scale/cache ABI requires explicit spec/version and all consumers; never reinterpret existing Flash pool bytes.
- Fused EP MoE kernel API changes and serving backend cutover are separate commits/gates.
- Shared-path changes need corresponding plain/pooled, qblock, cache alias, NEXTN/KDA checkpoint and MK regression. CPU checks alone cannot establish device correctness/performance.

## Status

- Epic worktree created from latest fetched main.
- JAX reference submitted; terminal status and metrics pending.
- Torch baseline runner audit completed; no kernel changes made. Latest main documents full GLM DSA experimental and sparse pool mode-A only. Eight-chip two-host support is not established by existing Flash tests.
- Admission currently reports four more TPU chips needed and a CPU flavor topology warning; keep the submitted experiment, do not duplicate or modify others' resources.
- Immediate next task: validate private checkpoint config/tensor scale shapes before Torch load. `--tpu-fp8-runtime channelwise` converts block checkpoint scales at runtime; it is NOT proof that the preconverted private channelwise checkpoint ABI is supported. Keep any loader compatibility fix separate from kernel changes.
- Native Torch launcher supports two hosts with explicit nnodes/rank/dist address and TP16; attention DP uses `--attn-dp-size`. Build a GLM5.2-only recipe rather than reuse Flash flags.
- This ledger is updated with actual evidence as the work proceeds.
