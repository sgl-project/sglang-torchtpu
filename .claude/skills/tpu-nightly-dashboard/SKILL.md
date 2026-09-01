---
name: tpu-nightly-dashboard
description: The TPU nightly CI & dashboard contract — test taxonomy, required MODEL_CONFIG fields and metric labels, the stable-id rule, smoke legs and the ✕ record, and the publish opt-in. Use when touching test/registered/tpu/nightly-*/**, the TPU nightly workflows (nightly-test-tpu.yml, feature-validation-tpu.yml, accuracy-test-tpu.yml, perf-test-tpu.yml), or .github/tpu-ci/**.
---

# TPU Nightly CI & Dashboard Contract

Full design: [.github/tpu-ci/nightly-and-dashboard-redesign.md](../../../.github/tpu-ci/nightly-and-dashboard-redesign.md).
Enforced by the pr-gate lints in `test/registered/unit/test_tpu_dashboard_contract.py`
(plus `test_tpu_bench_serving_gate.py`) — run them locally before pushing:

```bash
python3 -m pytest test/registered/unit/test_tpu_dashboard_contract.py -q
```

## Test taxonomy — pick the right category first

| category | where | protocol |
|---|---|---|
| **Feature validation** | `test/registered/tpu/e2e/`, `integration/` | Tests a *feature* on some model. `register_tpu_ci(suite=..., nightly=True)` is the whole dashboard contract — pass/fail comes from the TIMINGS record, no metric dump needed. One job ↔ one test file. |
| **Accuracy** | `test/registered/tpu/nightly-accuracy/` | Model backbone only: basic serving config, **no feature flags**. One file, one suite per model. |
| **Perf** | `test/registered/tpu/nightly-perf/` | Specific model + serving config, through `RecipeLauncher` + `bench_serving.sh` **only** (that is what puts the server/bench config panels on the dashboard). |

The nightly (`nightly-test-tpu.yml`) builds once, then fans out the three stage
workflows **in parallel** — slice contention is Kueue's job, and a red stage costs the
others nothing. Each stage workflow is also dispatchable alone.

## Required MODEL_CONFIG fields and labels (label lint fails the PR otherwise)

| test kind | required `MODEL_CONFIG` fields | required labels on every metric |
|---|---|---|
| feature validation | — (no dump) | — |
| accuracy | `id`, `display_name`, `model` | `kind: accuracy`, `model` |
| perf | `id`, `display_name`, `model` | `kind: performance`, `model`, `workload` |

The `model_config_main` machinery stamps the labels from the config — declare the
fields and don't hand-roll `dump_metric` labels in nightly tests.

## Rules that are easy to break

- **`id` is never renamed.** It is the dashboard's series join key. `display_name` is
  the free-to-change half. A forced id change is recorded once in
  `.github/tpu-ci/dashboard_aliases.json` (old → new), never by editing the test.
- **`kind` takes only `accuracy` / `performance` / `info`.** Anything else is stored
  but charted nowhere. `info` is the escape hatch: not plotted, shown in the point's
  details panel — use it for bench detail fields, probes, tuning notes.
- **Labels are dimensions, not payloads.** Low-cardinality, stable values only; bulky
  one-off data goes in a `kind: info` record's value. Extra labels are always welcome
  (they become hover detail + filter dimensions) as long as they don't reuse the
  reserved keys: `kind`, `id`, `display_name`, `model`, `workload`, `eval`, `reason`.
- **Smoke legs precede the sweep.** A perf gate with `"smoke": True` needs a
  self-gating `runner` and must be listed before every `sweep` gate. On smoke failure
  (or an incomplete measurement) the sweep dumps a `smoke_failed` record per point —
  the ✕ with `reason` on hover — instead of numbers. Never plot partial numbers.
- **One job ↔ one test file.** A new nightly registration must be reachable from
  exactly one job: single-host via `discover_tests.py` (automatic), multi-host via a
  `MULTIHOST_LAUNCH` dict (automatic, one row per file), accuracy/perf via one suite
  per model wired into the stage workflow's matrix, static P/D via a hand-written job
  in `feature-validation-tpu.yml`.
- **Publishing:** scheduled runs always ingest; a dispatch ingests only with
  `update-dashboard: true` (stamps `[publish]` into the run name) and only on main.
  Reruns replace datapoints — records are keyed `(run_id, attempt)`.

## Before waiting a night

Check your log against the keel parser (README § Dashboard sanity-check commands):

```bash
export GITHUB_TOKEN=$(gh auth token)   # from a radixark/keel checkout
python3 ci-monitor/src/tpu_ci.py sync --limit 3 --db-dir /tmp/tpu-ci
python3 ci-monitor/src/tpu_dashboard.py --db-dir /tmp/tpu-ci --out /tmp/tpu-ci.html
```
