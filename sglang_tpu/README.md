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

The LFM2 causal ShortConv implementation is wired through the OOT platform
interfaces in mode B only (`SGLANG_PLATFORM=tpu_oot`). Mode A still selects the
in-tree ShortConv sidecar, which expects stacked GPU-layout state. The mode-B
implementation includes packed/chunked prefill, decode, padding and per-request
state updates, including lifting the state and metadata into `direct_compile`.
It currently requires `--disable-radix-cache` and no speculative decoding;
ZAYA1 and the separate LFM2-MoE ShortConv implementation are outside this wiring.

The TPU path now uses `sglang_tpu.kernels.mamba.short_conv_pallas` for the
causal convolution itself. Prefill streams 64-token tiles and their K-1 halo
through VMEM double buffers; decode packs eight requests into the vector
sublanes. The custom op donates the complete per-layer
`[slots, K-1, 1, channels]` FP32 tensor from `TpuMambaPool`, so a layer update
does not copy or re-layout the pool. Slot 0 remains a read-only padding slot.
The operator still preserves LFM2's existing gate placement (`B*x` before the
convolution and `C*` after it), FP32 accumulation, output dtype, and bias
semantics.

Run the registered numerical test on a TPU with
`SGLANG_PLATFORM=tpu_oot python test/registered/kernels/ops/mamba/test_tpu_short_conv.py`.
For a warmup-separated comparison against the current torch reference path,
use `python test/manual/kernels/bench_lfm2_short_conv_tpu.py --mode both`.
Local validation covers CPU Pallas-interpret and offline Mosaic lowering;
TPU device correctness, xprof timings, and speedup numbers must still be
collected on the target TPU generation.

The following full-model numbers describe the previous torch implementation,
before Pallas integration. Manual validation completed on one TPU v5e chip with BF16 weights
and KV cache, TP=1, single-request concurrency, and `torch.compile` disabled.
GSM8K accuracy was 72.00% on 200 examples (`max_tokens=1536`), with no request
errors and a 1.50% truncation rate. Serving benchmarks measured 22.08 and 22.93
output tokens/s for 128/128 and 1024/128 input/output tokens, respectively
(median of three runs of 32 requests after warmup).
Compiled full-model execution and registered full-model E2E coverage remain
pending.

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

`core_hooks.register` is declared by both distributions (`tpu_hooks` in `python/`,
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

## P/D KV offload from prefill DRAM to decode DRAM

Prefill stays the primary prefix cache. Before host pressure discards a cold
cached branch, prefill pushes its full-attention KV from its own DRAM into the
DRAM of the decode that served the branch. When the branch returns and misses
on prefill, prefill recalls it from decode DRAM into its own DRAM, and normal
admission loads it into HBM. No HBM page and no staging slot is used on either
role, and both HiCache write policies are supported. The decode keeps pushed
branches only; it does not cache its own generation requests.

Enable HiCache on both roles with `--enable-hierarchical-cache
--hicache-io-backend raiden` and decode radix caching with
`--disaggregation-decode-enable-radix-cache`, then set one switch on both
roles:

```bash
SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM=1
```

Unsupported settings are refused, never run without the offload. An engine
stops at start, naming the setting, with: a GDN/Mamba model, the unified KV
pool, pipeline parallelism (`--pp-size` above 1) or the Raiden pipeline
transfer, context or data parallelism,
speculative decoding, `--sleep-on-idle`, a transfer or HiCache I/O backend
other than raiden, an
L3 storage backend, HiCache or the radix cache off, the Rust radix tree, or a
tpu-sync build without the host-to-host read API
([google/tpu-sync@32568d35](https://github.com/google/tpu-sync/commit/32568d35)
or later). A pair fails its requests with the reason when the switch is on for
one role only or when the roles differ in attention-TP size (resharding).

On a hybrid sliding-window model (GPT-OSS) the offload also moves the last
sliding window of each branch, through an SWA host pool on both roles. The
pool is sized as upstream sizes a hybrid host stack, by `--hicache-ratio` or
a `--hicache-size` split by device bytes, and the model needs `--page-size`
above 1. Such a model resumes at a length only with the window
that ends there, and a branch carries one window, at its end: a prompt that
extends an offloaded branch recalls it, a prompt that shares only its beginning
does not. Under `write_back` a branch is offloaded once, whole, so requests
that share a prefix and then diverge get no recall from decode DRAM; the
prefill's own host tier still serves them while it holds the prefix, since it
keeps upstream's window of every evicted node. Under `write_through` a growing
branch is offloaded at successive lengths and they recall from decode DRAM too.

A decode whose DRAM is full declines a push; that is back-pressure, counted in
the `[victim-push]` log line. A prefill starts pushes only while it has a
request queued or running, so an engine left alone settles its transfers (a
declined or unanswered push within 60 s) and goes idle. A cache flush is
refused while transfers are in flight and can be retried. The Raiden socket I/O
timeouts default to disabled and the native session deadline does not interrupt
a blocked TCP operation: without them, a peer host that vanishes mid-transfer
without closing its connections keeps this engine's pages pinned and the engine
non-idle. Set them on both roles (either role sends KV):

```bash
export TPU_RAIDEN_PREFILL_{HANDSHAKE_ACK_READ,FINAL_ACK_READ,PAYLOAD_READ,HANDSHAKE_WRITE,PAYLOAD_WRITE}_TIMEOUT_S=60
export TPU_RAIDEN_DECODE_{HANDSHAKE_READ,PAYLOAD_READ,ACK_WRITE,PAYLOAD_WRITE}_TIMEOUT_S=60
```

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

The existing `/freeze_gc` request repeats compiler-artifact cleanup to cover files
used or created by subsequent warmup, then clears export records and JAX caches,
collects garbage, freezes GC and trims freed host heap pages. Checkpoint eviction
is completed by scheduler readiness and is not repeated during this pass. Python
HTTP startup waits for this second pass before reporting readiness. Direct
`Engine` callers can also call `freeze_gc()` after their own warmup. Keeping cleanup
in this lifecycle ensures temporary objects are collected before `gc.freeze()`
retains the remaining objects. Explicit `/freeze_gc` calls also run the cleanup
hooks. Under PP the stages dispatch the freeze in different loop iterations, so
each rank cleans up without a world barrier: only the lock-guarded native compiler
cache is removed, and the reply does not wait for the other stages.

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
advised with `POSIX_FADV_DONTNEED` and remain on disk. Disposable native, AOT and
JAX compiler files on tmpfs/ramfs are removed; lock files and directories remain.
Advice is not a measurement of resident memory freed.

After serial weight loading, checkpoint eviction runs in the background during
graph warmup and is joined before scheduler readiness. Overlapped weight loading
uses synchronous eviction at readiness, after all readers finish. Reusing files
after cleanup can require storage reads. After freeze cleanup, one rank per host
logs cgroup `memory.current` in bytes; this diagnostic does not gate readiness.

JAX cache clearing leaves loaded serving executables alive, but new shapes may
need retracing. `SGLANG_TPU_MALLOC_TRIM_AFTER_WARMUP_ENABLED=0` skips export-record
clearing, JAX cache clearing, CPU-backend release, garbage collection before
freezing and startup heap trimming. File-cache release retains its separate
opt-outs.

## Pipeline stage hand-off

Pipeline stages hand activations on in lockstep by default (`pp_wave.py`): one
collective-permute that every device launches in the same order, so each round
lasts as long as the slowest forward in flight and the pipeline drains with
zero launches.

- `SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF=1` selects the buffered hand-off
  (`pp_buffered.py`). Each stage sends a forward's output to the same column of
  the next stage with a point-to-point send as soon as the forward is launched,
  so a stage waits only for its own input.
- `SGLANG_TPU_PP_BUFFERED_SEND_DEPTH` (default `2`) caps the forwards a stage
  queues for sending before it blocks.
- Needs torch_tpu 0.2.0 and libtpu 0.0.50 or later; a start-up self-test checks
  the routing and the bytes and fails fast otherwise.
- HBM: a device holds up to `8 + SGLANG_TPU_PP_BUFFERED_SEND_DEPTH` hand-off
  buffers of the largest token bucket x hidden size (plus the aux capture slots
  under Eagle3 or DFlash) in the model dtype, about 94 MiB each for GPT-OSS-120B
  at 16384 rows, and one zero residual per bucket. The KV budget reserves this
  worst case, and each stage checks at start-up that it fits next to the KV
  cache, failing with the size it needs if not.
- Not supported with DP attention or prefill context parallelism.

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
built-in platform with `PlatformEnum.OOT`, keeps `is_tpu()` true, and adds:

- `--attention-backend tpu_splash_attn`, the `AttentionBackendEnum.OOT` backend.
- torch.compile with the `tpu` backend and static shapes; compile-cache tags for
  the rank assignment and the kernel inputs; TeaCache dropped and the VAE left
  eager under compile; `--offload-during-compile` off.
- Regional-compile regions of a model marked SPMD-safe (Wan, Z-Image) keep their
  collectives in one program (torch_tpu's `spmd_safe`, with its own cache tag);
  `fusion_barrier` is XLA's optimization barrier; every rank drains its device
  queue at the end of each request.
- Checkpoints read on the host, then moved (safetensors cannot open `tpu:N`);
  the Ulysses output all-to-all exchanges heads-major outputs in head slabs.
- `--ring-degree` (non-causal, by K/V all-gather) on the splash backend, its
  default under ring; an automatic selection may give it to any layer that takes
  `torch_sdpa`. The built-in platform has no ring-capable backend.
- Runtime patches applied at worker init (`multimodal_gen/worker.py`), with
  the debt header: `rope.py` (RoPE without stride-2 lane access); worker init
  also registers srt's ATen fallbacks (`patches._register_aten_fallbacks`, e.g.
  `logaddexp`).

Gates that take the OOT arm, with the same outcome as the built-in platform:

- `CustomOp` dispatch: the dispatch key is `tpu`, so ops reach `forward_tpu`
  where they define it and `forward_native` otherwise.
- `get_torch_library_dispatch_key()`: `PrivateUse1` instead of `CUDA`; its only
  caller registers the CUDA flash-attention op, which TPU never selects.
- `get_all_to_all_communicator_cls()`: answered explicitly with the base
  communicator the built-in platform uses.

Gates whose OOT arm differs from the built-in platform:

- `build_torch_compile_kwargs()`: the OOT arm compiles with `get_compile_backend()`
  (`tpu`) and drops the mode; the built-in arm passes the mode to the default
  backend (inductor). Worker init turns off dynamo's automatic dynamic shapes,
  as `dynamic=False` would.
