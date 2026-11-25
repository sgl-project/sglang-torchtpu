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
- Write comments and self-explaining code for a first-time reader when the logic is not intuitive:
  - Name magic numbers, hardware budgets (such as VMEM windows), tile sizes, and cost-model coefficients with units and any headroom rationale (for example, `32 MiB, leaving 1/40 headroom`); prefer `if / elif` over nested inline ternaries.
  - Include an `Args:` docstring on non-trivial kernel and cost-model helpers so units and static-vs-dynamic shapes (tokens, sequence slots, pages, element widths) are explicit.
  - For multi-step formulas, lowering models, or staged pipelines, explain each step inline, in the docstring, or via a link to a self-contained design doc rather than leaving bare algebra.
  - Do not cite internal compiler source filenames (such as `.cc` files) in comments; state the hardware or lowering behavior directly.

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
- When a change gates between kernels or lowerings with a threshold or cost
  model, check that the gate is not overfit to a single workload (`8k/1k`),
  concurrency (`max_seqs`), or default server flag (such as
  `chunked_prefill_size`). Verify how the crossover moves across token buckets,
  sequence slots, cache pages, and page/chunk sizes.
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
- Use parameterized tests (`@pytest.mark.parametrize` or `parameterized`) rather
  than `for` loops inside one test method. In correctness and benchmark sweeps,
  vary one axis at a time instead of confounding parameters (such as changing
  `max_seqs` inside a cache-size sweep), verify synthetic inputs actually
  populate the tested dimension rather than slicing a shorter list or padding,
  and sweep configurable server flags (such as chunk size) rather than a single
  derived constant.
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

- New TPU-only modules go in `sglang_tpu/`, never under `python/sglang/srt/`.
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

## Writing a PR description

How to write the description of a pull request to `sgl-project/sglang-torchtpu`.
A reviewer should be able to read it in two minutes, run what it shows, and get
the same result.

### Rules

1. **Keep the template's sections**, in this order: Motivation, Modifications,
   Accuracy Tests, Speed Tests and Profiling, Checklist. Add a short "Notes for
   reviewers" only if there is something a reviewer must know.

2. **Keep it short.** Aim for what a reviewer reads in two minutes, usually
   under 70 lines. If a draft feels complete, cut it to about a third.

3. **Describe the PR as it is at its head, not its history.** The description
   becomes the squash-merge commit message. When review changes the PR,
   rewrite the description so it is still true. Do not add a "Review changes"
   list, notes per commit, test runs on earlier commits, or before/after tables
   of what an earlier version of the PR did. Those go in commit messages and
   review replies.

4. **Describe what the PR does, not what it does not do.** Do not list limits
   or caveats that the code does not have, and do not add "left to a
   follow-up" notes. The one exception is the "Not run" line in Notes for
   reviewers.

5. **Modifications: one entry per concept.** Use a short lettered list, one
   entry per main part (for example: a. index op, b. sparse attention op,
   c. KV pool), with two or three sentences each.
   - Do not list files, classes, or methods. Reviewers read the diff.
   - Spend words on the design point a reviewer needs before reading the code,
     for example why the cache is not copied, or where a dispatch gate or
     threshold comes from across shapes (or link the explainer doc or sweep
     test). A small table is fine for that.
   - Say what the diff does not show: a new startup rejection, a changed
     default, a lazy import.
   - Put supporting changes (argument resolution, config, recipe) in one
     closing sentence.

6. **Accuracy Tests: show a run a reviewer can repeat.** Three things:
   - the command that starts the server;
   - one simple prompt sent with `curl`, such as "What is the capital of
     France?";
   - the output it returned.

   Give any other check one line: what ran and the verdict, for example
   "kernel tests pass on TPU" or "outputs match the dense backend". No pass
   counts, lists of test files, timings, logprob tables, or `<details>` blocks
   of evidence. Those belong in the tests, in review replies, or in a linked
   doc.

7. **Speed Tests: give the exact benchmark command, then one table.**
   - Use the standard client, `benchmark/tpu/bench_serving.sh`, and paste the
     command line that was run.
   - Use 8k input tokens and 1k output tokens (`ISL=8192 OSL=1024`) on the
     recipe's default context as the default baseline. When the PR targets a
     different workload (such as long-context or high prefix-cache hit in
     AgentX, a specific concurrency, or decode-only), benchmark that target
     regime as well (or instead, when 8k/1k does not exercise the change) and
     state why.
   - Report the client's own summary numbers: output throughput, TTFT, TPOT.
   - Show one table per benchmarked workload. No analysis of where the time
     goes, and no restating a configuration that the named recipe already
     holds.
   - Do not compare with other backends or other models. Compare with the code
     before the PR only when the PR is a performance change.

8. **Say which commit the results come from.** Name the commit the tests and
   numbers ran at, normally the head, and say whether CI agrees.

9. **Serve with the recipe as shipped.** Use the model's recipe under
   `benchmark/tpu/recipe/`. If the demo needs a different setting, use a knob
   that the recipe or script already has, and show it in the command: an
   environment variable the recipe reads (such as
   `GLM_FLASH_DENSE_MAX_CONTEXT`) or a `bench_serving.sh` variable. Also show
   the environment variables the feature needs. Do not add an override that
   has no knob; if the default is wrong for the demo, fix the recipe in the PR.

10. **Link what you mention.** A function, a line, a commit, a PR, or a review
    thread gets a link: a permalink at a fixed commit with line anchors
    (`.../blob/<sha>/path/to/file.py#L10-L20`), or the thread's
    `#discussion_r<id>` URL.

11. **Write in plain English.** Full sentences, ordinary words, one idea per
    sentence. Say "the indexer picks the keys" rather than a compressed phrase
    a reader has to decode.

12. **No tool attribution.** Do not add "Generated with ..." footers or
    co-author lines for coding tools to the description or the commit
    messages. (Vendored files keep their "Adapted from" header.)

### Template

````markdown
## Motivation

<Two or three sentences: what was missing, and what this PR makes work.>

## Modifications

**a. <Main part>**

<Two or three sentences on what it does.>

**b. <Main part>**

<Two or three sentences on what it does.>

**c. <Main part>**

<Two or three sentences. If one design point needs explaining, explain it
here, with a small table if that helps.>

Smaller changes that go with these: <one sentence>.

## Accuracy Tests

`<model>` on `<hardware>`, at `<commit>`.

Serve, from the repo root:

```bash
python -m benchmark.tpu.launch_server --config recipe/configs/<model>/server.py
# The recipe starts the server with: <the key flags>
```

Send a prompt:

```bash
curl -s http://127.0.0.1:30000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "default",
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "max_tokens": 64, "temperature": 0
}'
```

Output:

```json
{"choices": [{"message": {"role": "assistant", "content": "The capital of France is Paris."}}]}
```

<One line per other check: what ran, and the verdict.>

## Speed Tests and Profiling

```bash
cd benchmark/tpu
MODEL=<model> DATASET_NAME=random-ids TOKENIZE_PROMPT=1 \
    ISL=8192 OSL=1024 BATCH_SIZE=<n> NUM_PROMPTS=<n> ./bench_serving.sh
```

| Input / output tokens | Concurrency | Output throughput | Mean TTFT | Mean TPOT |
|---|---|---|---|---|
| 8,192 / 1,024 | <n> | <tok/s> | <ms> | <ms> |

## Checklist

<the template's checklist, ticked honestly>

## Notes for reviewers

- Not run: <hardware or paths this PR was not tested on>.
- <Anything else a reviewer must know.>
````
