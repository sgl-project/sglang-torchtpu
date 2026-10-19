## Code Review Rules

SGLang runs on Google TPUs through `torch_tpu`: PyTorch PrivateUse1
(`device="tpu"`) lowers to XLA HLO executed by PJRT/libtpu. Pay particular
attention to host/device boundaries and compiled-shape stability.

Apply the checks relevant to the changed execution paths.

### Review evidence

- Trace the changed behavior through callers, configuration, and consumers.
  Verify surrounding code at the PR's head commit; a stale checkout is not
  evidence of a missing caller or incompatible override.
- Prioritize actionable correctness, performance, compatibility, and design
  problems. Give the triggering scenario, consequence, and affected file/line.
  Keep uncertain risks separate from confirmed defects; skip stylistic nits.

### Design and shared backends

- Check separation of concerns and extensibility. Flag workarounds that hide an
  unexplained failure, duplicate behavior, or hard-code one model/configuration
  where the interface supports more. Explain the concrete maintenance or
  behavioral risk rather than asking for abstraction for its own sake.
- When a design is ambiguous, compare equivalent SGLang GPU. Consistent designs are useful precedent;
  differences need a backend-specific reason, not automatic rejection.
- Changes in shared code such as `mem_cache/`, `layers/`, `model_executor/`,
  managers, and speculative workers can affect CUDA, NPU, CPU, and TPU. Inspect
  callers and backend overrides; preserve base-class defaults and subclass
  invariants. Keep TPU imports and device assumptions out of generic paths.
- Check that overrides preserve the parent contract and do not unnecessarily
  duplicate it. Trace deleted behavior to its replacement and remaining callers.
  Flag new dependencies on deprecated paths and risky compatibility changes.
- Suggest splitting independent migrations or unrelated dtype/logic changes
  when combining them obscures regressions or makes rollback unsafe. Keep a
  shared-interface change with the backend adaptation needed to use it.
- Require comments or self-explaining code when the logic is not intuitive

### TPU execution and performance

- Inspect new device-to-host reads (`.item()`, `.tolist()`, `.cpu()`, Python
  conditions on tensors), transfers, and synchronization in prefill/decode hot
  paths. Check the actual tensor device and execution frequency before claiming
  a stall; CPU metadata and necessary completion boundaries are valid.
- Check whether data-dependent output sizes, Python scalar specialization, or
  new shape/dtype variants break compilation or trigger repeated compilation.
  Verify that bucketing, padding, masks, and compile-cache selection agree with
  runtime inputs, including partially filled batches and boundary lengths.
- Follow tensors through allocations, indexing, custom ops, and outputs. Check
  device placement, supported lowering, dtype conversions, and index bounds.
  Look for accidental CPU fallback, repeated allocations/copies, and per-token
  work that grows with the whole cache or batch capacity.
- In overlap and asynchronous copy paths, ensure consumers wait for completion
  and buffers remain alive until their last reader/writer finishes. Treat
  removed synchronization as a correctness change as well as an optimization.

### Runtime behavior and compatibility

- Follow KV-cache ownership through allocation, reuse, eviction, cancellation,
  and failure cleanup. Check leaks, double frees, stale mappings, and capacity
  accounting; include rejected speculative tokens when that path changes.
- For speculative decoding, check acceptance lengths, token positions, masks,
  request ordering, and draft/target state after partial acceptance or stopping.
- For distributed changes, check rank groups, collective order, tensor shapes,
  and divergent branches, especially empty or uneven batches that can hang.
- Check model/config, quantization scales, and attention-backend assumptions.
  Preserve request/response and streaming semantics; keep CLI defaults, validation,
  docs, and diagnostic messages consistent with the implemented behavior.

### Plugin migration

The TPU backend is becoming an out-of-tree plugin (`sglang_tpu/`); the fork in
`python/` shrinks until it equals a pinned upstream release. Every review of a
change that touches `python/sglang/srt/` or `sglang_tpu/` checks the migration
does not move backwards.

- Placement: new TPU-only code lands under `sglang_tpu/`. Flag anything added
  under `python/sglang/srt/` or inside one of the re-export shims there.
- Core debt: flag any new `is_tpu()` / `_is_tpu` branch, `sglang_tpu` import,
  TPU-named module, or TPU default in shared code. Accept one only when the PR
  links the upstream seam issue or PR that will remove it and the ratchet
  baseline change (`tools/plugin/tpu_debt_baseline.json`) is explicit.
- Mechanism: for a core change TPU needs, verify the lowest rung that fits was
  used, in order: an existing `SRTPlatform` method or capability flag; a
  registry filled from `init_backend()`; a `register_resolution_hook`; a
  subclass returned through a `get_*_cls()` factory; a seam PR upstream carried
  as a `sglang_tpu/hooks/` entry. Ask for the simpler rung when it fits.
- Hooks and patches (`sglang_tpu/hooks/`, `sglang_tpu/patches.py`) are tracked
  debt: require the header (Why / How / Upstream PR or issue / Remove when).
  A REPLACE on a method longer than ~30 lines means "write the seam instead".
- Seam PRs: a change meant for upstream carries no TPU code or naming, keeps
  the in-tree default byte-for-byte, gates differing behavior on
  `is_out_of_tree()`, stays under ~300 lines, and ships a CPU unit test with a
  fake platform. Flag a PR that bundles the seam with the feature using it;
  ask for the split.
- Cleanup: once a seam has merged upstream and arrived through the sync, the
  follow-up must delete the hook and the core branch. A pin bump that leaves
  them in place is incomplete.
- Mode A/B: a change to a gate that `is_out_of_tree()` reaches must keep both
  arms at the same outcome for every configuration that runs, or document the
  divergence in `sglang_tpu/README.md`. The OOT contract test stays green.
- Env vars: new TPU-only names in `sglang_tpu/envs.py`; never one name in both
  registries.
- Tests: a registered TPU test carries topology, model, and feature labels from
  `scripts/ci/tpu/support_matrix_vocab.json`; `scripts/ci/tpu/support_matrix.py
  --strict` passes; a disabled test states a reason with a link. A feature is
  not "supported" in docs until a registered test backs it.

### Validation

- Check that tests guard the changed core behavior and relevant boundary cases.
  Device-dependent changes need TPU coverage; CPU mocks alone cannot establish
  lowering, compilation, numerical, or asynchronous correctness. Check shared
  behavior on affected backends too. State hardware validation that was not run.
- For performance claims, inspect comparable before/after workloads, model,
  hardware, precision, and commands. Separate compilation/warmup from steady-state
  latency and throughput; target benchmark requests to the suspected regression.

### Review output

- Have a summary of the motivation and solution of the PR
- Point out high-level directional problem first if any
- Keep normal PR findings concise and ordered by severity(P0 - functional, correctness,
  reliability, critical performance, critical design problem. P1 - Reachable wrongs,
  possible performance bottleneck, debatable choices, unclear trade-offs, readability, extendibility problem.
  P2 - nits). Mark severity and the category in the front of the comments. Explain a failure
  scenario; do not inflate uncertainty into a
  defect. If no actionable issue is found, say so and note material test gaps.
- Call out points for manual review if ambiguous or risky or have considerable blast radius
- Call out verified non-issues
- High-level comments should be split into sections, each having a title. Low-level comments should be inline

## Where TPU code goes

`sglang_tpu/` at the repo root is the TPU backend, installed as the plugin
`sglang-tpu` next to the fork in `python/` (`sglang_tpu/README.md`).

- New TPU-only modules go in `sglang_tpu/`, never under `python/sglang/srt/`,
  in the subpackage that mirrors the core path they extend (`mem_cache/`,
  `layers/attention/`, ...); the package root holds only plugin plumbing.
  The vacated paths hold re-export shims; do not add code to them.
- Core must not import `sglang_tpu` or grow new `is_tpu()` branches.
  `tools/plugin/check_tpu_branches.py` (a pre-commit hook) ratchets both
  counts per file; raising the baseline needs a linked upstream seam issue.
- Reach core from the plugin, in order: an existing platform method or
  capability flag; a registry filled from `init_backend()`; a
  `register_resolution_hook`; a subclass handed back through a `get_*_cls()`
  factory; a seam PR upstream, carried as a `sglang_tpu/hooks/` entry until it
  merges.
- `test/registered/unit/plugins/test_tpu_oot_platform_contract.py` must stay
  green: every platform method an `is_out_of_tree()` gate reaches needs an
  answer on `TpuOOTPlatform`.
