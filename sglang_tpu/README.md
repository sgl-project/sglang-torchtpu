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

## P/D victim prefix cache

Prefill stays the primary prefix cache. Before host pressure discards a cold
cached branch, prefill exports its FULL KV and ending SWA window to the decode
that served the branch; the decode keeps pushed victims and does not cache its
own generation requests. When the branch returns and misses on prefill, prefill
recalls the FULL KV and its window directly from decode DRAM into its own host
pages through the Raiden prefix exchange; normal admission then loads them into
HBM, with no decode HBM restore or device staging. Branches of a shared system
prompt, cache salts and adapter namespaces keep independent ownership.

Enable HiCache on both roles with `--enable-hierarchical-cache
--hicache-io-backend raiden` and decode radix caching with
`--disaggregation-decode-enable-radix-cache`, then make the decode's DRAM the
prefill's KV offload tier with one switch on both roles:

```bash
SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM=1
```

The switch stands for the component flags below, which remain available for
partial setups and keep any value set explicitly:

| Flag | Role | What it enables |
|---|---|---|
| `SGLANG_DISAGG_PREFIX_VICTIM_PUSH=1` | both | export cold cached branches to the decode's host pools |
| `SGLANG_DISAGG_PREFILL_PULL_DECODE_PREFIX=1` | both | recall decode-held prefixes on a prefill miss |
| `SGLANG_TPU_PD_SWA_HOST_WINDOWS=<count>` | both | hybrid SWA models: a host tier of SWA windows on each role, one per cached path, so the window travels with a pushed victim and comes back with a recalled prefix (the switch defaults it to 128; size it with the host budget; zero keeps the full-only host tiers) |

Requirements: the Python unified radix tree, matching P/D
attention-TP and context-parallel geometry, equal page sizes and layouts, no
pipeline or GDN/Mamba state transfer, and session affinity when several
decodes serve one prefill.

A spill moves RAM to RAM through the HiCache host-pool managers: prefill
registers its pinned host pages for a native host read and offers them; the
decode reserves host pages, reads into them and inserts the victim as a
host-only node. No HBM or staging slot is used, and at most four victims per
engine are in flight. Every rank agrees on branch, namespace, length and
outcome before allocation or insertion, and offers carry a per-prefill
incarnation so restarts cannot alias. Source pins and destination pages stay
reserved until the native transfer reports completion (bounded by
`SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S`), also after a refused or expired
offer; a cache reset or memory release waits for accepted reads to drain and
can be retried.

The native session deadline does not interrupt a blocked TCP operation and the
socket I/O timeouts default to disabled, so set them on both roles (either role
sends KV). This keeps existing values and otherwise gives each I/O 60 seconds:

```bash
for raiden_io_timeout in \
  TPU_RAIDEN_PREFILL_HANDSHAKE_ACK_READ_TIMEOUT_S \
  TPU_RAIDEN_PREFILL_FINAL_ACK_READ_TIMEOUT_S \
  TPU_RAIDEN_PREFILL_PAYLOAD_READ_TIMEOUT_S \
  TPU_RAIDEN_PREFILL_HANDSHAKE_WRITE_TIMEOUT_S \
  TPU_RAIDEN_PREFILL_PAYLOAD_WRITE_TIMEOUT_S \
  TPU_RAIDEN_DECODE_HANDSHAKE_READ_TIMEOUT_S \
  TPU_RAIDEN_DECODE_PAYLOAD_READ_TIMEOUT_S \
  TPU_RAIDEN_DECODE_ACK_WRITE_TIMEOUT_S \
  TPU_RAIDEN_DECODE_PAYLOAD_WRITE_TIMEOUT_S
do
  export "${raiden_io_timeout}=${!raiden_io_timeout:-60}"
done
```

These bound single stalled calls, not a slow transfer that keeps progressing.
Pages are reused only after native completion, and memory-release APIs must
wait until all native reads have drained and the scheduler is idle.

Both roles need matching tpu-sync Python and native artifacts from the pinned
[`ram2ram` fork branch](https://github.com/alisachen-google/h2h/tree/ram2ram)
(see its [build instructions](https://github.com/alisachen-google/h2h/blob/ram2ram/README.md#option-2-building-from-source);
`./build.sh torch` against the installed `torch_tpu` ABI). Select the checkout
on both roles and check the API:

```bash
export TPU_SYNC_SRC=/path/to/tpu-sync
export PYTHONPATH="$TPU_SYNC_SRC${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
import inspect
from tpu_sync.api.torch import kv_cache_manager as k
assert "source" in inspect.signature(k.KVCacheManager.register_read).parameters
assert "destination" in inspect.signature(k.KVCacheManager.start_read).parameters
assert "local_host_block_ids" in inspect.signature(k.KVCacheManager.start_read).parameters
PY
```

The native extension must come from the same build. Each victim-enabled
host-pool manager opens a control port (port zero means cache-only), and offers
publish that control endpoint rather than the block-data endpoint.

FULL pages and the trailing window move through their own host pools under one
producer-owned UUID. A victim is inserted only when every rank has received
both and the producer reported success; failed transfers release both landings
once accepted native reads settle, and shared-prefix insertions release
duplicate FULL and SWA pages independently.

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
or created by subsequent warmup, clears JAX tracing caches, then collects garbage, freezes GC and trims freed
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
shards cached. `SGLANG_TPU_RELEASE_COMPILE_ARTIFACTS=0` keeps the native compiler
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

JAX cache clearing leaves loaded serving executables alive, but new shapes may
need retracing. `SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED=0` skips export-record
clearing, JAX cache clearing, CPU-backend release, garbage collection before
freezing and startup heap trimming. File-cache release retains its separate
opt-outs.

## Fixed-cohort benchmark gate

- Enable with `SGLANG_TPU_BENCHMARK_BATCH_SIZE=N` (default: `0`, disabled).
- Send a native batch with base request ID `hourly-batch:<unique-cohort>`;
  normalization appends `_0` through `_N-1`. Use `x-rid` if the router overrides
  request IDs. Untagged requests bypass the gate.
- Unified and P/D emit first tokens as ready. Decode waits for all first tokens
  and, for P/D, KV transfers. Unified disables decode mixing during prefill.
- Decode timing runs from the first decode forward to the last request completion.
  Token counts exclude the prefill token and clip SBD output at the stop boundary.
- `/server_info` exposes `benchmark_batch_gate`: cohort ID, decode duration,
  decode output tokens, gate wait duration, and TPOT.
- Requires DP1/attention-DP1/CP1/PP1, device-resident decoder KV, and capacity for
  the full batch. Mixed cohorts, retraction, invalid completion, or timeout fail
  the benchmark. `SGLANG_TPU_BENCHMARK_TIMEOUT_SEC` defaults to `600`.

## Diffusion (multimodal_gen)

`multimodal_gen/` holds the diffusion TPU code. The `tpu_oot` entry point of
`sglang.multimodal_gen.platforms` selects `multimodal_gen.platform.TpuOOTPlatform`
whenever `torch.tpu` is registered; `SGLANG_DIFFUSION_PLATFORM_OVERRIDE=tpu`
selects the fork's built-in TPU platform instead. `TpuOOTPlatform` subclasses that
built-in platform with `PlatformEnum.OOT`, keeps `is_tpu()` true, and carries the
TPU behavior (`--attention-backend tpu_splash_attn`, the `AttentionBackendEnum.OOT`
backend).

Gates that take the OOT arm, with the same outcome as the built-in platform:

- `CustomOp` dispatch: the dispatch key is `tpu`, so ops reach `forward_tpu`
  where they define it and `forward_native` otherwise.
- `get_torch_library_dispatch_key()`: `PrivateUse1` instead of `CUDA`; its only
  caller registers the CUDA flash-attention op, which TPU never selects.
- `get_all_to_all_communicator_cls()`: answered explicitly with the base
  communicator the built-in platform uses.
