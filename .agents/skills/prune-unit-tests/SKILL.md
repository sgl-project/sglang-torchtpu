---
name: prune-unit-tests
description: Find and remove unit tests that are change detectors or stale, without lowering the bar or losing coverage. Use when asked to prune or clean up a test tree by deleting low-value tests, or to audit tests for change detectors, mirror tests, disconnected proofs, dead or disabled tests, or tests of deleted code.
---

# Prune unit tests

Goal: delete tests that cost maintenance and CI time but guard nothing. Not a goal:
fewer tests, faster CI by dropping coverage, or "simplifying" a test someone will
have to rewrite. **Unresolved doubt means keep.**

Read [`unit-test-admission`](../../../.claude/rules/unit-test-admission.md) first.
It's the source of truth for what a good test is, and this skill applies it in
reverse. Where the two disagree, the rule wins. The wider audit catalog is
[`ci-test-audit`](../ci-test-audit/SKILL.md).

Task constraints override broader audit catalogs. Don't delete coverage because a
surface is rarely used, rarely changed, expensive, or weakly connected. A catalog
suggestion to relocate, consolidate, repair, or reschedule a test doesn't authorize
deleting its coverage.

## Scope in this repo

- **Only tests this repo's CI runs.** These are files with a non-disabled
  registration in a suite the fork's CI executes:
  - `base-a-test-cpu`, run by PR Test Base.
  - `stage-a-test-tpu-unit` and `stage-b-test-tpu-4-v7`, run by PR Test (TPU).
  - `nightly-tpu-*`, run by the TPU nightly and feature-validation workflows.
- **Workflow files aren't proof.** Most are inherited from upstream. The CUDA,
  AMD, NPU, XPU, and MUSA jobs either skip or pass their change check without
  running tests. Confirm with run history (`gh run list` / `gh run view --json
  jobs`) when in doubt. `test/manual/` and files registered only to non-running
  suites are out of scope.
- **Only fork-owned tests.** These are files absent from the upstream revision
  most recently merged into `origin/main`. Find it as the second parent of the
  newest `Merge remote-tracking branch 'upstream/main'` commit
  (`git log --merges --grep="upstream/main" -1 --format=%P origin/main`). Don't
  take the second parent of the outer `Merge pull request ... sync-from-upstream`
  commit; that parent is the fork's sync branch and contains fork files. Unmerged
  `bot/sync-from-upstream-*` branches don't count.
- **Upstream-deleted files.** Files upstream deleted after that sync are still
  upstream-owned. The next sync removes them, so leave them alone.

Running tests locally:

- Use `PYTHONPATH=$PWD/python:$PWD`, because `sglang_tpu` lives at the repo root.
- CI runs each file as `python <file>`, so the file's `__main__` block decides
  what gets collected. Under `unittest.main`, module-level pytest functions never
  run; that's a "repair needed" finding, not dead code. Under
  `pytest.main([__file__])` they run normally. Read each file's entry point
  before calling anything uncollected.
- A file that imports `torch_tpu` runs only on the TPU lane. Judge it by reading,
  or run the case bodies in isolation with JAX at the version pinned in
  `python/pyproject_tpu.toml`.
- `CustomTestCase` reports assertion failures as errors.

Checkers that must stay green:

- `python scripts/ci/tpu/support_matrix.py --strict --output <path>.json`.
  Compare the JSON before and after, ignoring `commit` and `generated_at`.
- `python scripts/lint/check_registered_tests.py`
- `test/registered/unit/tools/test_run_suite_registrations.py`
- `test/registered/unit/plugins/test_tpu_oot_platform_contract.py`
- `pre-commit run --files <touched files>`

## The one question

For every test case, answer: **"Which plausible regression in production code would
turn this red, and would that regression slip past every surviving test if this
case were gone?"**

- There's a concrete production-only mutation that turns it red, and no surviving
  test covers it: **keep**.
- The only diffs that turn it red are behavior-preserving refactors or edits to the
  test itself: it's a **change detector**, so remove it.
- Its subject provably no longer exists: it's **stale**, so remove it.
- A surviving case covers its *entire* guarded contract (see "Redundancy"): it's
  **duplicate**, so remove it.

Answer per case, not per file. A file can lose three of eight cases.

The shapes listed below are **screening signals**. Matching a shape is never enough
on its own. Each removal needs the evidence the shape's paragraph asks for.

## Remove: change detectors

1. **Mirrors without an independent contract.** Similar formulas or branch
   structures are screening signals only. Identify the contract, where the oracle
   comes from, and a plausible production-only mutation. Keep independent reference
   implementations, precision, dtype and rounding checks, and boundary calculations
   when they'd catch a real behavioral error. Delete only when the assertion repeats
   an isolated implementation choice with no dependent contract. Example:
   `assertEqual(get_x(), 16)` where the body is `return 16`, it's the repo's own
   choice, and nothing reads it as a contract. Extracting or reading production code
   through AST in order to *run* it is not a source-text assertion.
2. **Incidental interaction choreography.** Mocks and `assert_called*` can be the
   contract. Internal allocation, ownership, routing, ordering, synchronization,
   registration, and required side effects are often the only observable effect of
   a method. Delete only choreography whose alteration preserves every relevant
   effect. Replacing a call assertion with a return-value assertion is valid only if
   that value detects the same failure modes. Otherwise keep the interaction
   assertion or use a stateful fake.
3. **Spelling-pinning source-text assertions.** `inspect.getsource`, reading a `.py`
   file, or a regex that asserts a line or name literally appears, where the failure
   it guards is only "someone reworded the code". Preserve source, AST, signature,
   registry, and inheritance checks that guard a concrete compatibility or
   bookkeeping failure (see Keep).
4. **Repo-owned constant and default echoes.** A literal that's this repo's free
   choice, pinned by nothing else, and not a documented user contract.
5. **Private-structure snapshots.** The exact set of private attributes, internal
   dict keys, or an internal repr, where no consumer depends on that shape.
6. **Tautologies.** Asserting fields the test just set, `isinstance` of a
   constructor's own return value, or that a mock returns what it was configured to
   return, with nothing else exercised. Before calling a type check a tautology,
   search for registrations of that type (pytree, serialization, dispatch tables,
   compile boundaries). A same-fields object of another type can pass every field
   assertion and still fail there.
7. **Disconnected proofs and demonstrations.** The asserted result depends only on
   test-local formulas or data, so it would stay green if the production
   implementation were deleted or broken. Classify it as lacking a production guard
   unless you can show a connection. This doesn't extend to neighboring cases that
   call production helpers, or to independent references compared against
   production outputs.

## Remove: stale

Not being run or not being collected doesn't by itself prove the behavior is
obsolete. Stale requires evidence about the *subject*, not just the test's
mechanics.

1. **Obsolete subject.** Remove only after tracing the current implementation,
   callers, replacement paths, and compatibility imports, and showing that the
   guarded behavior no longer exists. A rename, move, re-export, import failure, or
   missing local dependency is not enough. If the subject moved, the test should
   follow it rather than be deleted. **Unused is not obsolete**: if the code still
   ships but nothing calls it, the test stays. Report the dead code, and delete the
   code and its test together in a separate change.
2. **Upstream-deleted tests are not a category here.** A test upstream deleted
   after the last merged sync is still upstream-owned. Leave it alone; the next
   sync removes it. An upstream deletion is also no evidence about a fork-owned
   test. Judge that one under "Obsolete subject" alone.
3. **Disabled or uncollected cases.** Treat these as execution defects to
   investigate, not automatic deletions. Check supported hardware, dependencies,
   runner selection, imported test classes, and direct-script execution (many
   runners execute files as scripts through `__main__`). A closed issue may mean the
   test should be re-enabled. A missing reason or expiry is a metadata defect.
   Delete only when the subject is obsolete or the case is shown to have no
   regression-detection value. Otherwise keep it and report the repair needed.
   Hardware and platform gates are valid.
4. **No effective failure oracle.** Follow execution through imported or inherited
   cases, helpers, subprocesses, expected exceptions, and compile or import smoke
   checks. No local `assert` doesn't mean always-pass: an unexpected exception,
   nonzero subprocess status, timeout, or compile failure can be the oracle. A
   registration-only bridge file or worker script is not redundant if removing it
   removes execution of its cases. Delete only after identifying the actual oracle
   and showing it can't detect a relevant regression. Examples: swallowed failures,
   empty parameter sets or loops, shadowed test definitions, assertion helpers never
   invoked, the subject itself mocked out, a tolerance that can't fail.

## Redundancy

Redundancy requires equivalence of the **entire guarded contract**. Name the
surviving case and compare: inputs and boundary cases, execution path, backend,
dtype, shape and topology, platform mode, fixtures, assertions, tolerances, and
effective CI execution (runner, cadence, enabled state). The survivor has to cover
every meaningful failure mode of the removed case, in an equivalent environment and
cadence. Shared helpers, matching outputs, or catching one common bug aren't enough.
A different batch size, a singleton vs multi-element container, a different top-k
or other parameter value, or a weaker comparison (casting before comparing, a
looser tolerance) all break equivalence.
CPU mocks don't replace device execution, eager checks don't replace compiled checks,
and unit checks don't replace integration wiring. If equivalence is uncertain, keep.

Redundancy is a reason to delete only for proven duplicates (copy-paste leftovers,
split remnants), not a general test-count reduction.

## Keep, even if it looks like a change detector

- **Bug regressions** whose setup still reaches the failure mechanism and whose
  assertion observes its consequence. A documented bug is strong evidence of value,
  not proof the test still detects it. If it no longer does and the bug's code path
  is live, repair it or list it as "replace later". Don't just delete it.
- **Derived properties** checked against production code: boundary and alignment
  math, round-trips, invariants, protocol semantics.
- **Completeness, bookkeeping, and compatibility guards**: "every field in A is
  copied in B", registry completeness, hook-target and signature checks, ratchets,
  and checks that validate test doubles against the real interface. The fact that a
  behavioral test *could* catch the same failure doesn't make the guard redundant.
  Only an actual surviving test does. A missing explanatory failure message is a
  diagnostic improvement, not grounds for deletion.
- **Boundary tripwires** on code the repo doesn't own (a synced upstream, a
  dependency) that guard a concrete assumption here.
- **External-source literals**: values dictated by an outside spec, protocol, file
  format, or vendor API.
- **Negative branches**: "a non-matching input does not trigger", "the default
  applies when absent".
- **Statistical numerical tests** that check production output against a target
  distribution. These aren't ineffective stress tests.
- **Device, compiled, and numerical checks** you can't run locally. Being unable to
  run a test is no evidence against it. If deletion safety depends on execution
  evidence you can't get, keep it.

## Rules that are never relaxed

- Never remove a test because it fails, flakes, or is slow.
- If a change-detector case is the *only* thing exercising a real behavior, replace
  it in the same change with a behavioral assertion, or keep it and list it as
  "replace later". Never leave a behavior unguarded. A replacement regression test
  must fail on the pre-fix code (or a faithful targeted mutation) and pass on the
  fix.
- Don't weaken assertions, widen tolerances, or add skips in tests you keep.
- Don't edit production code in a pruning change. That includes shims,
  re-exports, and hooks that look orphaned. Report them separately.
- Don't relax contract tests, ratchets, registry checks, or matrix vocabulary to make
  a pruning change pass. Don't delete documented support or a matrix cell that loses
  its only backing test. Keep the test instead.
- Tests inherited from upstream are out of scope unless told otherwise. Deleting
  them only produces conflicts at the next sync.

## Procedure

1. **Scope.** Record the HEAD, the upstream comparison revision, and the manifest of
   in-scope files (fork-owned and CI-run, as above). The manifest is the deletion
   boundary.
   Classify each entry's role (unit test, bridge, worker script, manual repro,
   tooling test) before judging it. Search outside the manifest for callers,
   importers, workers, and covering tests, but don't expand deletion scope.
2. **Baseline.** Before editing, capture an inventory of CI registrations by backend,
   suite, runner or topology, enabled or disabled state, and labels, plus collected
   case identities where collection is available. Save it outside the repo.
3. **Screen, then read everything.** Grep for the shapes above, then read every
   in-scope file anyway. Grep finds candidates, but most findings come from reading.
4. **Judge each case** with the one question and the evidence each category asks
   for. **Write the ledger row before deleting.**
5. **Edit.** Delete cases or whole files. Clean up leftovers: unused imports,
   helpers, fixtures, empty classes, registration calls in now-empty files. Search
   tracked and hidden config for the file path, module import name, case names, and
   build labels. Check build targets (for example `BUILD.bazel` `py_test`) and
   their dependents, bridge imports, subprocess worker references, suite lists,
   workflow path filters, and CI naming or dashboard consumers. Regenerate derived
   docs and artifacts through their generator, not by hand.
6. **Verify.**
   - Re-take the inventory and reconcile every removed registration and case
     against the ledger. Investigate anything else that disappeared. Compare
     support-matrix coverage per model x topology x feature, not just "the label
     still appears somewhere".
   - Every edited file compiles (`python -m py_compile`). Run the surviving cases in
     partially edited files and any affected bridge or importer files through their
     real runner entry points. Record passed, failed, skipped, and collected counts.
     Zero executed cases is not a pass.
   - Run every checker listed under "Scope in this repo".
   - State explicitly which hardware validation wasn't run.
7. **Ledger.** For each removed case (including parametrized variants and imported
   cases affected by a whole-file removal), record:
   - identity
   - category
   - production subject and consumer
   - intended contract
   - concrete evidence of obsolescence or lack of guard value
   - surviving coverage (for duplicates)
   - registration impact
   - validation done

   Name any unverified assumption, and remember that an unresolved assumption means
   keep. Whole-file rows enumerate the affected cases and the file's runner or worker
   role. Also list "kept but borderline", "replace later", "repair needed" (for
   disabled tests), and "orphaned production code" items.
8. **Commit** in reviewable units (per subsystem or per category). Messages
   summarize counts by category. The ledger goes where the task instructions say.

## Calibration examples

- A set-equality check that every `input_*` field defined on an output type is
  copied by a transfer function, built by scanning that function's source. **Keep**:
  it's a completeness guard.
- `assertIn("self.some_flag", inspect.getsource(Cls.__init__))`. **Remove** only if a
  surviving behavioral test exercises the flag. Otherwise replace it or keep it as
  "replace later".
- A test whose only assertions are mock calls showing eviction and bookkeeping still
  happen on a skip path. **Keep**: the side effects are the contract.
- A file with no asserts that imports `TestCase` classes from elsewhere so they run
  on a different runner. **Keep**: it's a bridge.
- Two test methods asserting that a formula equals a position set, both computed
  inside the test, next to a method that calls the production helper. **Remove the
  two** (disconnected proofs), **keep the third**.
- `assertEqual(Config().page_size, 16)`, a repo-owned default no consumer pins.
  **Remove.**
- A fork-owned test whose subject the fork has deleted, kept alive only by a
  try/except import skip. **Remove** (obsolete subject). If the subject's code
  still ships, even unused: **keep**, and report the dead code.
- `@unittest.skip("flaky")` on live code. **Keep** and report "repair needed".
  The same skip on a test whose subject is gone: **remove**.
