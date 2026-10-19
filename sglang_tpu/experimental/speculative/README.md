# Experimental TPU spec worker

Every algorithm is written as draft -> verify -> commit over static, bucketed
shapes. Enable it with `SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1`. It requires
overlap scheduling. For an example launch config, see
`benchmark/tpu/recipe/configs/qwen3_8_27b/mtp_gate.py`.

Two rules:

- **No serving-time recompilation.** This is a requirement, not a performance
  goal. One XLA compile mid-serving stalls every request in the batch for
  seconds. A change that compiles after warmup is a bug, however fast its
  steady state.
- **Decoupled from the CUDA and other device spec workers.** Do not reuse,
  patch or branch inside core's EAGLE / DFlash workers to make them run on TPU.
  Their dynamic shapes, device-side metadata reads and per-step allocation are
  what break the first rule, and their GPU behavior must not change. Share only
  the model definitions and backend-neutral math.

## Class hierarchy

Workers split by how the draft is produced (the family). Drafters split by checkpoint.

```
BaseSpecWorker (core)
└── TpuSpecWorker                       shared verify, bucketing, warmups
    ├── TpuChainSpecWorker              autoregressive: one draft forward per token
    │     └─ has-a TpuChainDraftWorker
    │                 └── TpuMtpDraftWorker      MTP (NEXTN)
    ├── TpuBlockSpecWorker              block-parallel: one forward over [r, m, ..., m]
    │     └─ has-a TpuBlockDraftWorker
    │                 ├── TpuDflashDraftWorker   DFlash and DFlash2
    │                 └── TpuDsparkDraftWorker   DSpark
    └── TpuEagle3Worker                 EAGLE3   (TODO: move under TpuChainSpecWorker)
          └─ has-a EagleDraftWorker (core)

TpuDraftWorker (ABC)  ── TpuChainDraftWorker, TpuBlockDraftWorker

SpecInput (core) ── HostPaddedDraftInput ── HostPaddedChainDraftInput
                                          ├─ HostPaddedBlockDraftInput
                                          └─ HostPaddedEagleDraftInput
```

The worker *has a* drafter instead of *being* one: the family fixes the step
structure, and each checkpoint only fills in sampling. `_DRAFT_WORKER_CLS` in each
family worker maps the algorithm to its drafter.

- **DFlash2 is-a DFlash.** Core's `DFlash2DraftModel` subclasses `DFlashDraftModel`
  and keeps the same block forward, KV injection and verify. Only sampling
  differs: it walks a K-candidate lattice (`../models/dflash2.py`). So it is
  served as `--speculative-algorithm DFLASH` by `TpuDflashDraftWorker`, which
  branches on the model type in `sample_drafts`.
- **DSpark is a sibling of DFlash, not a subclass.** Its model also subclasses
  `DFlashDraftModel`, but its sampling (Markov bias on the base logits) shares
  no code with DFlash's, so both derive from `TpuBlockDraftWorker`.
- **EAGLE3 is-a chain (TODO).** EAGLE3 drafts one token per forward from the
  carried hidden, the same as MTP, so it belongs under `TpuChainSpecWorker` with
  a `TpuChainDraftWorker` drafter. Today it subclasses `TpuSpecWorker` directly
  and drives core's `EagleDraftWorker`, so its draft loop, commit and compile
  policy (`TpuEagle3SpecDecodeCompilePolicyMixin`) duplicate the chain family's.
  Do not add fixes to it; move it to the chain family.

Inheritance kept for reuse, not because it is-a (do not extend the pattern):

- `TpuSpecDecodeCompilePolicyMixin(DFlashSpecDecodeCompilePolicyMixin)`: every
  family, chain included, reuses the DFlash policy's padded TARGET_VERIFY handling.
- `ChainVerifyInput(DFlashVerifyInput)` and `ChainDraftForwardInput(ChainVerifyInput)`:
  the chain verify and draft forwards reuse the DFlash input type so that RPA and
  direct_compile accept them; they add the verify KV lengths and the draft hidden.

## Why it is built this way

- **One shared verify.** Every family runs the same TARGET_VERIFY over
  `[r, d_1..d_N]` and the same lossless acceptance (`spec_ops`). A new algorithm
  only writes draft, commit and extend.
- **Host-padded, bucketed state.** Batches are padded to a batch bucket on the
  host, and `SpecState` stays at that bucket across steps. Only the result
  handed to the scheduler is at the live batch size. No metadata is read back
  from the device, and one program per bucket covers every live size.
- **Every serving shape is warmed, no fallback.** `init_cuda_graphs` /
  `precompile_draft` warm every shape seen while serving. An uncovered input
  raises: falling back to eager would turn the recompile into a silent
  slowdown. torch_tpu fuses the eager ops
  that feed a compiled call into one program, so fuse new eager ops into one
  warmed `compiled_op`.
- **Plugin only.** Core reaches this code through hooks (`../../hooks/spec_worker.py`
  selects the worker, `../../hooks/experimental_spec.py` adapts host-padded
  batches), and every hook delegates unchanged for non-experimental batches.

## Adding a drafter

1. Subclass `TpuChainDraftWorker` or `TpuBlockDraftWorker`. Validate the model in
   `__init__` and raise on anything unsupported. `mtp/mtp_draft_worker.py` is the
   smallest example. A variant whose model is-a an existing one branches inside
   that drafter, as DFlash2 does.
2. Register it in `_DRAFT_WORKER_CLS`. If the algorithm is new to this worker,
   route it in `../__init__.py` (`is_experimental_chain` / `is_experimental_spec_worker`).
   That routes the worker, the compile policy and the platform check.
3. Sampled drafts return the logits q was drawn from. Restricted support is
   `-inf` on dense logits, so the shared `accept_drafts_sampled*` stays lossless.

A new family subclasses `TpuSpecWorker` (the abstract method docstrings say when
each one may be a no-op), adds a `HostPaddedDraftInput` subclass and a compile
policy mixin, and routes all three.

## Validation

- CPU: `test/registered/unit/plugins/test_experimental_*.py`. Check the drafter
  against core's reference (e.g. `test_experimental_dflash2_selector.py`).
- Recompiles: serve with `SGLANG_TPU_CHECK_RECOMPILATION=1` from a fresh cache and
  drain the batch through every live size. The output must contain no
  `at serving time` line. `test_tpu_qwen3_8_27b_singlehost_mtp_recompile.py` is
  the template. `..._SITES=1` locates the source line of a compile, but never
  benchmark with it on.
- Accuracy: GSM8K with spec on matches spec off. Performance: A/B on identical
  server flags.

## Not supported

Draft trees (MTP and EAGLE3 need `--speculative-eagle-topk 1`), PP and mixed prefill/decode DP steps. Neither family supports DP
attention, and the chain family does not support PD disaggregation or a draft
sliding window. Each raises at startup.
