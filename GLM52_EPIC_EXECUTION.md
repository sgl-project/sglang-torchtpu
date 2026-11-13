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
- Metadata inspector added and checked against a synthetic FP8+per-output-channel safetensors header fixture; Python compile and git diff checks passed. This does not validate the real checkpoint.
- Native launcher candidate dry-run passed for both node ranks using TEST-NET rendezvous addresses; no TPU/model loading was performed. Candidate config is explicitly marked unvalidated in the benchmark README.
- Changes through `107e03b28f` touch only this ledger and `benchmark/tpu/glm52_epic/`; no Flash or production runtime paths changed.
- Real checkpoint CPU preflight SUCCEEDED: `exp-2gq0zsmhs0`, 118629 tensors, 59044 output-channel scale pairs, no missing indexed tensors or wrong shard assignments. Config is full GLM DSA (not Flash), hidden6144/78 layers/256 experts. Summary and config hashes committed. This is header evidence, not weight-content or TPU validation.
- Declared artifact extraction analyses `an-dqplhgxwpb` and `an-zpjlrgogt0` failed before readable diagnostics; task summary is available via Falcon exp logs, full header output remains remote. Platform issue https://github.com/primatrix/Falcon/issues/1525 tracks missing failed-analyzer logs. No bucket/provider bypass used.
- Epic pushed to origin; baseline/scaffolding commits remain separable. Plugin-only channelwise loader implementation is now being prepared behind explicit config opt-in; default/Flash behavior must fall through unchanged.
- Opt-in loader committed `1192e6c8be`, candidate JSON wired via HF quantization override `635b971e0c`, six CPU contract tests `322a967f78`. Syntax checks passed; actual imports/execution pending.
- CPU validation `exp-9dwxb55hll` could not pull private CI base image and was explicitly aborted. Public release-image CPU replacement `exp-3f6ay1a6ek` submitted with same pinned test source; records actual runtime versions and is not TPU-stack equivalence evidence.
- JAX reference remains pending; latest admission reports eight more TPU chips needed. No duplicate TPU experiment submitted.
- Real CPU execution SUCCEEDED: `exp-tl4zrfks0x`, exact source `173364f89dc6e07428349092c1ea6cc3c548ebd3`, six tests / 0.054s / exit0. Covers standard FP8 factory isolation, explicit ABI validation, dense reconstruction and stale-alias rejection, MoE reconstruction using real existing formatter. An actual import failure in prior run was fixed by importing LinearMethodBase from quantization.base_config. Logs and fixed manifest committed. Runtime torch2.11.0+cpu/JAX0.10.2; this is NOT on-device or current-stack validation.
- Public-image first run failed before tests because `SGLANG_PLATFORM=cpu` is not a plugin name. Correct CPU invocation unsets that variable and uses existing `SGLANG_USE_CPU_ENGINE=1`.
- Remaining hard gates: plugin/OOT regressions, real TPU loading/compile/numerical checks, matched W-P steady-state performance. No performance equivalence claimed.
- IMPORTANT CORRECTION: successful default-runtime analyzer `an-iis65psswb` retrieved complete headers. ALL 59044 scales are serialized rank1 [N]. Earlier output_channel_2d_weight classification refers to paired WEIGHT rank; original rank2 synthetic tests did not prove checkpoint compatibility. Production corrected in `1997351eb9`; real_scale_abi.json records representative actual headers and every shard hash. CPU tests being corrected/re-executed before claiming ABI acceptance.
- OOT suite `exp-wdvzn14s92` failed due missing sgl_kernel in public TPU image; rerun with established optional GPU import stub `exp-5v13xkhwf0` still fails missing registered CPU fused_sigmoid_mul_cpu. This runtime cannot establish full OOT regression; do not treat as pass or attributed feature regression without main control.
- Corrected real rank1 ABI CPU rerun SUCCEEDED: `exp-nh7bghd5dt`, source `396701fd7590f817cdbc2a58d298d618beb1c867`, seven tests / 0.067s / exit0. Now exercises actual legacy column/row TP2 loaders and real MoE channel scale loader before formatter. Supersedes original six synthetic-rank2 tests; still not device acceptance.
- Native chunk audit fixed a second preflight mismatch: supplied chunk must be GLOBAL32768 to resolve LOCAL2048 under attnDP16. Compile token buckets are local[2048]; max_prefill32768 stays global. Previous global2048/local128 padded to2048 candidate must not be used as matched baseline.
- Shared-core TPU debt ratchet passed unmodified baseline. Default analyzer runtime successfully retrieves full checkpoint evidence; earlier custom python image failure is not a missing artifact.
- OOT attribution control `exp-d7m6k8cufz` on exact main d6f54cc2899b also ran39 tests and hit the SAME four errors as epic exp-5v13xkhwf0, all missing registered CPU fused_sigmoid_mul_cpu in release image. Comparison saved; this is environment/baseline evidence, not full regression acceptance. No test weakened/disabled.
- Current pinned runtime wheel CPU probe `exp-lw6e0evey6` failed HTTP401 on private torch-tpu package index (pip then EOF prompted). Private base image earlier inaccessible. Therefore public CPU image cannot be promoted to device baseline, and no unverified runtime substitution is made.
- Metadata inspector now explicitly reports scale_rank in summary; synthetic rank1/rank2 fixture verified distinct counts to prevent repeating the ABI misread. Two-host startup runner preparation is independent of scarce TPU admission.
- JAX reference admitted and RUNNING at 14:01:36Z; exact private source3695225808f0 starts both hosts, JAX distributed world synchronized and model reached precompile at14:07. Actual in-venv versions verified via Falcon cp+exec: JAX/JAXLIB0.11.1/libtpu0.0.46.1. DSA logs confirm local_q2048, FP8Q/PV, packed split pipeline. No steady-state result yet; original wait remains attached.
- Extended metadata CPU `exp-20ka8bjztj` SUCCEEDED: actual dtype summary BF16=465,F32=59120,F8_E4M3=59044; NO FP8 weights without paired scales; embedding/head/indexer weights_proj all actual BF16. Evidence and exact inspector manifest committed.
- Protocol audit establishes historical steady_chunk_extend_ms is consecutive host-observed prefill-result completion cadence, NOT device-forward latency or HTTP TTFT. Exact source/data/filter boundaries documented; matched Torch observer still needed.
- Model-loader integration CPU run `exp-yx8wvm6hhg` initially 8/9 pass; earlykv_b test fixture omitted attn_mha required by actual TPU reader. Fixture corrected without mocking TPU path; rerun within bounded existing CPU allocation pending.
- This ledger is updated with actual evidence as the work proceeds.
