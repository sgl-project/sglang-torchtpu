# Experimental TPU spec worker

Every algorithm is written as draft -> verify -> commit over static, bucketed
shapes. Enable it with `SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER=1`. It requires
overlap scheduling (except on a P/D prefill). For an example launch config, see
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
    │                 ├── TpuMtpDraftWorker      MTP (NEXTN)
    │                 └── TpuEagle3DraftWorker   EAGLE3
    └── TpuBlockSpecWorker              block-parallel: one forward over [r, m, ..., m]
          └─ has-a TpuBlockDraftWorker
                      ├── TpuDflashDraftWorker   DFlash and DFlash2
                      └── TpuDsparkDraftWorker   DSpark

TpuDraftWorker (ABC)  ── TpuChainDraftWorker, TpuBlockDraftWorker

SpecInput (core) ── HostPaddedDraftInput ── HostPaddedChainDraftInput
                                          └─ HostPaddedBlockDraftInput
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
- **EAGLE3 is-a chain.** EAGLE3 drafts one token per forward from the carried
  hidden, the same as MTP, so `TpuChainSpecWorker` serves it with a
  `TpuEagle3DraftWorker`. Only the drafter differs: its carry is the draft's
  pre-norm hidden, and committed rows read the target's concatenated aux hidden
  states through the draft's fc.

Inheritance kept for reuse, not because it is-a (do not extend the pattern):

- `TpuSpecDecodeCompilePolicyMixin(DFlashSpecDecodeCompilePolicyMixin)`: every
  family, chain included, reuses the DFlash policy's padded TARGET_VERIFY handling.
- `ChainDraftForwardInput(DFlashVerifyInput)`: the chain draft forward reuses the
  DFlash input type so that RPA and direct_compile accept it.

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
- P/D: `test_tpu_qwen3_8b_pd_singlehost_dspark_recompile.py` and
  `test_tpu_qwen3_8_27b_pd_singlehost_{mtp,dflash2}_recompile.py` run the same
  check on both engines.
- Accuracy: GSM8K with spec on matches spec off. Performance: A/B on identical
  server flags.

## DP attention

Both families serve `--enable-dp-attention`. Drafts are greedy under DP attention,
so an idle DP rank can mirror its peers' draft syncs; sampled requests take the
target-only accept.

Startup checks:

- `--enable-dp-lm-head` when the draft's head is sharded over a group other than
  attention TP: always for MTP, DFlash and DSpark (they share the target head;
  `tpu_platform.py`), and for an EAGLE3 draft without its own head
  (`TpuSpecWorker` checks the bound head). TPU requires it anyway when attention
  TP > 1.
- `SGLANG_SPEC_TP_SYNC` syncs both spec-accept sites or neither when attention
  TP > 1 (e.g. `rng,spec-accept-greedy` for DFlash).

Known gap: DFlash with HiCache under DP attention passes GSM8K, but the KV
offload probe sees one verify forward fewer on a host reload than on a device hit.

## Not supported

Draft trees (MTP and EAGLE3 need `--speculative-eagle-topk 1`), PP outside a P/D
prefill and mixed prefill/decode DP steps. The chain family (MTP) does not
support a draft sliding window. Each raises at startup.

## P/D disaggregation

Both raiden roles serve every algorithm with the same spec flags.

- Decode seed: `build_disagg_*_input` (`../../hooks/experimental_spec.py`) builds
  it on host; core's builders stack it on device at the live bs, one compile per
  bs. MTP ships its carry as the EAGLE hidden seed, and decode writes it back
  into the carry table.
- Decode retraction: a cpu_tensor backup carries the draft KV with the target
  KV (`TpuSpecWorker.retraction_draft_kv_pools`); a draft pool without a CPU
  copy raises at startup. On resume the MTP/EAGLE3 carry is reseeded from the
  PD seed, as core EAGLE reseeds its hidden.
- Hybrid GDN targets: raiden moves each request's committed GDN state row on a
  third engine next to the KV, and the decode seed resets the slot's
  checkpoint read offset.
- PP prefill: only the last stage hosts the draft (`TpuSpecWorker`); earlier
  stages run `TpuPipelineStageWorker`, which only forwards the target. The
  stages relay the EAGLE3 / DFlash aux captures to the last one (`../../pp_wave.py`),
  and PP runs without overlap, which a prefill-only server allows.
- Draft prompt pass on the decode (`SGLANG_TPU_PD_DECODE_DRAFT_PREFILL`, off by
  default): the prefill's last stage skips the draft and writes each prompt
  token's draft-input hidden into `pd_hidden_pool.py`, a paged buffer Raiden
  moves with the target KV in place of the draft KV; the decode reads it back
  and runs the draft over the prompt on each prebuilt batch (`pd_draft_prefill.py`).
  The buffer pages like the full KV pool, so it scales with the prefill's
  `--max-total-tokens`. Needs the Raiden pipeline transfer; no DP attention.
- DP attention on the decode: the decode ranks pull their own requests' KV, which
  relieves the PP transfer that bounds a non-DP decode (`pd_singlehost_pp_spec.py`
  `decode_dp`; it adds `--enable-dp-lm-head`).
  `test_tpu_gpt_oss_120b_pd_singlehost_pp_{eagle3,dflash}_perf.py` run PP8 EAGLE3
  and DFlash; the `_dpa_perf.py` variants run the decode at DP8.
- Decode radix + HiCache on a sliding-window target (`../../pd_decode_swa_hicache.py`):
  EAGLE3 and DFlash are core's opt-ins; DSpark opts in on this worker (hook on
  `supports_pd_decode_swa_hicache`), since its draft KV at position i is projected
  from the target hidden at i. Every DFlash-family draft keeps its KV in the
  HiCache draft sidecar.
- KV free fence: commit writes draft KV after the accept outputs the result
  copies, so commit and extend return an output of their last KV write as
  `kv_fence_tail` (MTP's draft hidden, EAGLE3's seed token, the block KV
  writer's own output), and the async result copy copies it. A separate read of
  the KV buffer would add a program per step.
