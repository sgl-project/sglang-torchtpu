# sglang_tpu: SGLang's TPU backend as an out-of-tree plugin

All TPU-only code of `sglang-torchtpu` lives in this package (dist `sglang-tpu`).
The sglang fork in `python/` shrinks toward a pinned upstream commit as its TPU
edits move here or go upstream as seams. This file is the operational summary.

## Install

Both distributions install editable, side by side:

```bash
pip install -e python/            # the sglang fork (dist "sglang")
pip install --no-deps -e .        # this plugin (dist "sglang-tpu")
```

`--no-deps` keeps pip from re-resolving the fork's dependency tree; the editable
fork already satisfies the plugin's `sglang` requirement.

## Two platform identities, one code base

| Mode | Select with | Platform class | `is_out_of_tree()` | Status |
|---|---|---|---|---|
| A (default) | `SGLANG_PLATFORM=tpu` or unset | `sglang_tpu.tpu_platform.TpuSRTPlatform` (`PlatformEnum.TPU`) | False | the shipped path |
| B | `SGLANG_PLATFORM=tpu_oot` | `sglang_tpu.platform.TpuOOTPlatform` (`PlatformEnum.OOT`, `is_tpu()` True) | True | what the PR gate and the single-host nightly run; opt-in for users until the divergences below close |

`tpu_oot` only activates when named: with `SGLANG_PLATFORM` unset the loader
calls every installed `activate()` and errors when two answer.

## Mode-B divergences (gates that take a different arm under OOT)

Measured against the fork at the wave-0 move; re-audit after every sync.

Closed: hybrid-SWA MHA models (the OOT pool and allocator arms mirror the TPU
arms); the import-time `init_backend()` call (`TpuOOTPlatform.init_backend`
defers to the worker bootstrap); MLA pool construction (the OOT builder's
`index_head_dim` kwarg); `get_device_memory_capacity`, which reports nothing
for TPU in both modes.

Open: sparse attention (DSA) models run in mode A only. Mode B refuses them at
pool construction (`TpuOOTPlatform.get_dsa_kv_pool_cls`), because the OOT DSA
builder does not say which layers run their own indexer.

Different arm, same outcome for every configuration that runs:

- `--prefill-only-disable-kv-cache`: the pool-family check refuses OOT pools,
  but argument resolution already rejects the flag on TPU in both modes (it
  needs the fa3/fa4 prefill backend).
- DFLASH draft backend: mode B falls back to `pallas_tpu_rpa_v3` for a backend
  `speculative/draft_utils.py` cannot build on TPU; mode A fails at construction.
- `SGLANG_PROFILE_V2=1` with the `"TPU"` activity: mode B also records a torch
  `PrivateUse1` trace, mode A records none; neither reaches xprof on that path.

Unreachable today: MLA with hybrid SWA or an FP4 KV cache takes the plain OOT
MLA builder in mode B and the generic builders in mode A; no TPU model uses
either.

## Layout

```
sglang_tpu/
  __init__.py            activate() for the tpu_oot entry point
  platform.py            TpuOOTPlatform (identity + the gate-required methods)
  tpu_platform.py        TpuSRTPlatform, the in-tree identity (moved here as-is)
  envs.py  hooks/  compat/
  attention/pallas/      Pallas attention kernels (was layers/attention/pallas)
  moe/pallas/            Pallas MoE kernels (was layers/moe/pallas)
  disaggregation/{tpu,raiden}/
  graph_runner/          decode graph runner + ViT compile runners
  <everything else>      was python/sglang/srt/hardware_backend/tpu/
```

The old paths under `python/sglang/srt/` are one-file re-export shims so
in-flight branches keep importing; they are deleted once the open-PR backlog
drains. Do not add code to them.

`tools/plugin/inventory.py` lists every hook and runtime patch with the core
attributes it replaces; `tools/plugin/scoreboard.py` prints every debt counter
of the migration. The TPU Plugin Scoreboard workflow publishes both daily.

## Rules for new TPU code

1. New TPU-only modules go here, never under `python/sglang/srt/`.
2. Core never imports this package at module scope. A function-scope import
   inside an `is_tpu()` guard is tolerated during the migration and counted by
   the ratchet lint (`tools/plugin/check_tpu_branches.py`); the baseline may
   only go down.
3. Prefer, in order: an existing platform method or capability flag; a registry
   populated from `init_backend()` (`register_attention_backend`,
   `BaseFusedOp.register_oot_forward`); a resolution hook; subclass-and-override
   through a factory; a seam PR upstream carried as a `hooks/` entry until it
   merges. A new `is_tpu()` branch in core needs a linked upstream seam issue.
4. Env vars: declare new TPU-only variables in `sglang_tpu/envs.py`. The
   existing `SGLANG_TPU_*` entries stay in `sglang.srt.environ` until their
   core readers move; never declare one name in both registries.

## Hooks

`hooks.register` is declared by both distributions (`tpu_hooks` in `python/`,
`tpu_oot_hooks` here): `SGLANG_PLATFORM=tpu` skips every general plugin of
`sglang-tpu`, so mode A needs its own entry point. `register()` runs once per
process and only on TPU, since every host installs the plugin.

## Tests

`test/registered/tpu/**` and `test/registered/unit/hardware_backend/tpu/**`
stay where they are (`test/run_suite.py` only collects `test/registered/**`).
The e2e smoke suites honour an inherited `SGLANG_PLATFORM`, so mode B is
`SGLANG_PLATFORM=tpu_oot python test/registered/tpu/e2e/<file>.py`.
CI picks the mode per lane (`render_jobset.py --platform`, the `platform`
input of `_tpu-test.yml`), so each test runs in one mode only: the PR gate
(stage-a, stage-b) and the single-host nightly (`nightly-tpu-4-v7`) run mode B;
the multi-host, P/D, accuracy, perf, combo and external lanes stay on mode A.

## Muse Spark multimodal

`models/muse_spark.py` is the text decoder; `models/muse_spark_omni.py` adds the
NaViT vision tower (adapted from upstream SGLang PR
[#44666](https://github.com/sgl-project/sglang/pull/44666), commit
`26305be7f410`) and is the registered `MuseSparkForConditionalGeneration`.
Image and video serving needs `--enable-multimodal`,
`SGLANG_EXTERNAL_MM_MODEL_ARCH=MuseSparkForConditionalGeneration` and
`SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE=sglang_tpu.multimodal`.

The vision encoder is replicated and complete images are distributed across
attention-TP ranks; CPU features stay on the host until their encoder rank
prepares patches. Live check:
[`test_tpu_muse_spark_mm_pd_multihost.py`](../test/registered/tpu/e2e/test_tpu_muse_spark_mm_pd_multihost.py)
(images and video over P/D), with `SGLANG_TPU_USE_LIVE_SERVER=1` and
`BASE_URL` pointing at the router.

## Startup cleanup

Before the scheduler-ready handshake, all ranks join pending compilation saves
and release startup file-cache pages. This also covers direct `Engine` and Rust
launches without a `/freeze_gc` request. Each host scans its files once per cleanup
pass; all ranks finish before the handshake returns.

The existing `/freeze_gc` request repeats file-cache release to cover files used
or created by subsequent warmup, then collects garbage, freezes GC and trims freed
host heap pages. Python HTTP startup waits for this second pass before reporting
readiness. Direct `Engine` callers can also call `freeze_gc()` after their own
warmup. Keeping cleanup in this lifecycle ensures temporary objects are collected
before `gc.freeze()` retains the remaining objects. Explicit `/freeze_gc` calls
also run the cleanup hooks.

Attention DP with shared MLP scheduling requires global control broadcasts:
`--enable-dp-attention-local-control-broadcast` and elastic EP scaling are rejected
because independently received freeze requests can deadlock with scheduling
collectives. DWDP's independent scheduling is exempt from this check.

`SGLANG_TPU_RELEASE_CHECKPOINT_PAGE_CACHE=0` keeps target and draft checkpoint
shards cached. `SGLANG_TPU_RELEASE_COMPILE_PAGE_CACHE=0` keeps the native compiler
cache pages. Both releases default on. The compiler cache namespace comes from
`TORCH_TPU_TIER2_COMPILATION_CACHE` (legacy:
`TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE`); empty or unset selects `default`,
and `disabled` skips it. Other namespaces are untouched. Disk-backed files are
advised with `POSIX_FADV_DONTNEED` and remain on disk; tmpfs/ramfs are skipped.
Advice is not a measurement of resident memory freed.

Checkpoint pages remain eligible for kernel reclaim during startup, but explicit
eviction starts at the scheduler-ready handshake. This can increase peak host
memory relative to evicting immediately after weight loading. Reusing files
after cleanup can require storage reads. After freeze cleanup, one rank per host
logs cgroup `memory.current` in bytes; this diagnostic does not gate readiness.
