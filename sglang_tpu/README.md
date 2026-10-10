# sglang_tpu: SGLang's TPU backend as an out-of-tree plugin

All TPU-only code of `sglang-torchtpu` lives in this package (dist `sglang-tpu`).
The sglang fork in `python/` shrinks toward a pinned upstream commit as its TPU
edits move here or go upstream as seams. The plan is `plugin-refactor-proposal.md`
(the launch-plan doc set); this file is the operational summary.

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
| B | `SGLANG_PLATFORM=tpu_oot` | `sglang_tpu.platform.TpuOOTPlatform` (`PlatformEnum.OOT`, `is_tpu()` True) | True | opt-in until the divergences below close |

`tpu_oot` only activates when named: with `SGLANG_PLATFORM` unset the loader
calls every installed `activate()` and errors when two answer.

## Known mode-B divergences (gates that take a different arm under OOT)

Measured against the fork at the wave-0 move; re-audit after every sync.

- `kv_cache_configurator.py`, `--prefill-only-disable-kv-cache`: refused for any
  OOT pool family. Mode A accepts it on TPU.
- `utils/common.py:get_device_memory_capacity`: returns the real HBM size after
  `init_backend()`; mode A returns `None` on TPU. Changes the pre-capture
  activation reserve for mamba/GDN models (`kv_pool_runtime.py`).
- `arg_groups/speculative_hook.py`: the DFLASH draft backend check answers only
  `pallas_tpu_rpa_v3` / `pallas_tpu_rpa_batched`; mode A accepts any name in
  `DRAFT_ATTENTION_BACKEND_CHOICES` and fails later at construction.
- `utils/profile_utils.py`: a caller passing `"TPU"` in `activities` gets an
  extra torch profiler on `ProfilerActivity.PrivateUse1`; the scheduler's
  profiler path does not.
- MLA models: the OOT pool arm builds through `_build_oot_mla_kv_pool`, mode A
  through the shared MLA arm. Same pool class (`get_mla_kv_pool_cls()`), one
  extra `index_head_dim=None` kwarg; not validated end to end.

Closed in wave 0: hybrid-SWA MHA models (the OOT pool and allocator arms now
mirror the TPU arms) and the import-time `init_backend()` call
(`TpuOOTPlatform.init_backend` defers to the worker bootstrap).

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

## Tests

`test/registered/tpu/**` and `test/registered/unit/hardware_backend/tpu/**`
stay where they are (`test/run_suite.py` only collects `test/registered/**`).
The e2e smoke suites honour an inherited `SGLANG_PLATFORM`, so mode B is
`SGLANG_PLATFORM=tpu_oot python test/registered/tpu/e2e/<file>.py`.
