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

## P/D generation health

Point the router or Kubernetes health check at `/health_generate`.

- `SGLANG_TPU_PD_HEALTH_WINDOW_S` sets the progress window **T** (default
  **200 seconds**). Choose T above the longest expected chunk, remote prefill,
  or KV-transfer wait, and below both `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`
  and `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` (default **300 seconds** each).
  `0` disables the progress window and uses only the core probe and worker crashes.
- `SGLANG_HEALTH_CHECK_TIMEOUT` sets the core probe wait (default **20 seconds**).
  Set the external health-check timeout above this, for example
  `--health-check-endpoint /health_generate --health-check-timeout-secs 30`
  on sgl-model-gateway.

**Healthy (200):** no confirmed worker crash, and pending work completes chunks
or produces tokens within T, including buffered tokens. A new request gets T
from dispatch before its first progress. Active work within this window stays
healthy even if the shorter core probe times out. An idle server must pass the
core probe.

**Unhealthy (503):** a confirmed worker crash (immediately), pending work with
no completed chunk or token for longer than T, or a failed core probe while idle.
Synthetic probe replies and aborts do not count as progress. A single healthy
step exceeding T also returns 503, so size T for supported workloads.

## Tests

`test/registered/tpu/**` and `test/registered/unit/hardware_backend/tpu/**`
stay where they are (`test/run_suite.py` only collects `test/registered/**`).
The e2e smoke suites honour an inherited `SGLANG_PLATFORM`, so mode B is
`SGLANG_PLATFORM=tpu_oot python test/registered/tpu/e2e/<file>.py`.
CI picks the mode per lane (`render_jobset.py --platform`, the `platform`
input of `_tpu-test.yml`), so each test runs in one mode only: the PR gate
(stage-a, stage-b) and the single-host nightly (`nightly-tpu-4-v7`) run mode B;
the multi-host, P/D, accuracy, perf, combo and external lanes stay on mode A.
