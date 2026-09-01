# TPU Nightly CI & Dashboard Redesign (Proposal)


## Motivation

Each point is a **Problem** / **Change** pair. Two kinds: requests raised by
others, and improvements proposed here.

### Requests

1. **Problem:** the dashboard cannot be updated manually — a fix on main stays
   invisible until the next scheduled nightly.
   **Change:** `update-dashboard` dispatch input stamps `[publish]`; keel
   ingests such runs on main, keyed `(run_id, attempt)` so reruns replace
   datapoints (keel unit test).

2. **Problem:** debug information on the dashboard is hard to navigate.
   **Change:** living **Q&A section** (§4 below), updated as suggestions come in.

3. **Problem:** a red job does not clearly display which test failed.
   **Change:** one GitHub job ↔ one test case; already landed, a pr-gate
   unit test enforces it going forward.

4. **Problem:** renaming a test splits its plot in two.
   **Change:** series join on a stable `id` (never renamed; `display_name`
   is free); forced id changes recorded once in `dashboard_aliases.json`
   (pr-gate lint).

### Improvements

5. **Problem:** a failed perf run is hidden instead of explained.
   **Change:** plot a **✕** with the reason on hover, server/bench config
   still attached (keel unit test).

6. **Problem:** no way to filter the charts down to what you need.
   **Change:** required labels group charts by model; every extra label
   becomes a filter / group-by tag (pr-gate label lint).

7. **Problem:** new information later would require dashboard changes.
   **Change:** the JSON dump accepts any extra non-reserved field/label;
   keel stores it, parsing added only when needed (reserved-key lint).

8. **Problem:** an inaccurately measured perf number still looks like a
   normal datapoint.
   **Change:** optional smoke test cases before the sweep validate the
   measurement; a failing one withholds the number and the ✕ carries the
   reason (pr-gate ordering lint).

---

## 1. Test categories

| category | protocol |
|---|---|
| **Feature validation** | Tests a *feature* (P/D, spec decoding, KV offload, …) on some model. Register into a nightly suite → tracked (Overview on failure, History for 7 days). One job ↔ one test file. No metric dump needed — pass/fail comes from the TIMINGS record. |
| **Accuracy** | Model backbone only: basic serving config, **no feature flags**. One file, one suite per model. Dumps the required accuracy fields/labels below. Stable `id`, never renamed. |
| **Perf** | Specific model + specific serving config, built on `RecipeLauncher` + `bench_serving.sh` **only** (that is what puts the server/bench config panels on the dashboard). Optional smoke legs first. Dumps the required perf fields/labels below. Stable `id`, never renamed. |

Required dump fields and labels — a pr-gate unit test (label lint) fails any
PR whose test is missing one:

| test kind | required `MODEL_CONFIG` fields | required labels on every metric |
|---|---|---|
| feature validation | — (no dump; TIMINGS record only) | — |
| accuracy | `id`, `display_name`, `model` | `kind: accuracy`, `model` |
| perf | `id`, `display_name`, `model` | `kind: performance`, `model`, `workload` |

Short examples:

**Feature validation** — registration is the whole dashboard contract:

```python
# test/registered/tpu/e2e/test_tpu_mymodel_multihost_spec.py
register_tpu_ci(est_time=3000, suite="nightly-tpu-multihost", nightly=True)

MULTIHOST_LAUNCH = {  # test-owned launch; identical specs share one server
    "hosts": 2, "launch_script": "recipe/mymodel_spec_mh.sh",
    "weights_pvc": "model-weights-rox",
}

def main():
    ...  # assert the FEATURE against the live server; raise on failure
```

**Accuracy** — eval gates against one basic-config server boot:

```python
# test/registered/tpu/nightly-accuracy/test_tpu_mymodel_singlehost_smoke.py
register_tpu_ci(est_time=2400, suite="nightly-tpu-accuracy-mymodel", nightly=True)

MODEL_CONFIG = {
    "id": "accuracy-mymodel",        # series join key — never renamed
    "display_name": "MyModel 120B",  # cosmetic, free to change
    "model": "org/mymodel-120b",     # canonical HF id
    "launch": {"recipe": "recipe/mymodel/server.sh", "env": {}},  # BASIC config only
    "gates": [{"dataset": "mmlu", "label": "mmlu_low",
               "baseline": 0.859, "atol": 0.03}],
}
# each gate score → dump_metric("mmlu_low", score,
#     labels={"kind": "accuracy", "model": MODEL, "eval": "mmlu"})

model_config_main(MODEL_CONFIG, base_url=BASE_URL)
```

**Perf** — a `bench_serving` sweep gate, optionally smoke-guarded:

```python
# test/registered/tpu/nightly-perf/test_tpu_perf_mymodel.py
register_tpu_ci(est_time=1800, suite="nightly-tpu-perf-mymodel", nightly=True)

MODEL_CONFIG = {
    "id": "perf-mymodel",            # series join key — never renamed
    "model": "org/mymodel-120b",
    "launch": {"recipe": "recipe/mymodel/server.sh",  # RecipeLauncher ⇒ config panels
               "env": {"ISL": "8192", "OSL": "1024", "CONCUR": "16"}},
    "gates": [
        # OPTIONAL smoke legs may precede the sweep; if one fails, no numbers
        # are plotted — a smoke_failed record renders ✕ + reason instead.
        {"dataset": "bench_serving", "label": "8k1k",
         "sweep": [{"workload": "8k1k", "isl": 8192, "osl": 1024,
                    "concurrency": 16}]},
    ],
}
# success → dump_metric("perf-mymodel.8k1k.total_throughput", tok_s,
#     labels={"kind": "performance", "model": MODEL, "workload": "8k1k", "c": 16})

model_config_main(MODEL_CONFIG, base_url=BASE_URL)
```

---

## 2. Workflow

```
 schedule (daily) ──▶ nightly-test-tpu.yml ◀── workflow_dispatch
                            │                  (any stage or single test;
                            ▼                   update-dashboard ⇒ [publish])
                     build image (once)
                            ▼
 [1] feature validation │ [2] accuracy │ [3] perf   — in parallel:
     slice contention is Kueue's job (the nightly lane's quota queues the
     jobs), and a red stage costs the others nothing;
     each stage is also its own dispatchable workflow
     (feature-validation-tpu.yml / accuracy-test-tpu.yml / perf-test-tpu.yml)
                            ▼
     keel ingest: schedule runs + main dispatch [publish] runs;
     keyed (run_id, attempt) — reruns replace datapoints
                            ▼
     dashboard:  Overview │ History │ Accuracy │ Performance
```

---

## 3. Test hierarchy in the nightly

```
test/registered/tpu/
├── e2e/                        feature validation — multi-host / P/D e2e
│   └── test_tpu_<model>_<feature>.py      (owns its MULTIHOST_LAUNCH)
├── integration/                feature validation — cross-component gates
├── unit/                       pr-gate unit tests (incl. the dashboard-contract checks)
├── legacy/                     not tracked by the nightly dashboard
├── nightly-accuracy/           accuracy — one file, one suite per model
│   └── test_tpu_<model>_singlehost_smoke.py
└── nightly-perf/               perf — one file, one suite per model
    └── test_tpu_perf_<model>.py

benchmark/tpu/
├── recipe/<model>/server.sh    serving recipes (every launch goes through one)
└── bench_serving.sh            the only allowed perf measurement path

.github/
├── workflows/                  nightly-test-tpu.yml + the three stage workflows
└── tpu-ci/
    └── dashboard_aliases.json  id rename registry
```

---

## 4. Q&A (living section)

---

## Appendix

Reference detail behind the sections above.

### A. Manual dashboard update — mechanism

`on: schedule` runs being the only dashboard updates stays the **default**
(an experiment on a branch must not pollute the trend), but every trigger
workflow gains a publish opt-in:

- New `workflow_dispatch` input: `update-dashboard` (boolean, default `false`).
- Publishing is only honored on **`ref == main`** — the ingester enforces this,
  not just the workflow, so a branch run can never write the main trend.
- All tests in the run can be rerun; the plots update accordingly. The keel
  ingester keys records by run and is idempotent; it must key by
  `(run_id, attempt)` and keep the **latest attempt**, so a rerun replaces the
  datapoint instead of duplicating it.

> **Mechanism** the GitHub runs API does not expose
> dispatch inputs, so the cheapest reliable signal is to stamp it into the run
> itself:
>
> ```yaml
> run-name: "Nightly Test (TPU) ${{ inputs.update-dashboard && '[publish]' || '' }}"
> ```
>
> The ingester then ingests `event == schedule`, plus
> `event == workflow_dispatch && head_branch == main && display_title contains [publish]`.
> This keeps keel's only input "things visible on the run", same as today.

The dashboard sync job (keel `tpu_ci.py sync`) can also be triggered
manually, but only ever syncs from **main**-eligible runs as defined above.

---

### B. Data contract (what a test must dump)

The dashboard never pattern-matches names. A record lands somewhere because
the test **said so** via labels. Everything below rides on the existing
`dump_metric` v1 JSONL schema (`filename`, `test_case`, `metric_name`,
`value`, `ts`, `labels`).

#### B.1 Required labels by category

| category | required labels | plotted where |
|---|---|---|
| feature validation | *(none — pass/fail from the TIMINGS record)* | Overview + History |
| accuracy | `kind: accuracy`, `model` | Accuracy tab, grouped by `model` |
| perf | `kind: performance`, `model`, `workload` | Performance tab, grouped by `model` |

The implicit inference rows in today's resolution order (`workload` ⇒
performance, `eval` ⇒ accuracy) are kept for backward compatibility during
migration, then **removed**: after the migration window, a metric with no
explicit `kind` is a build **warning → error** (§C.2).

Required labels are a **floor, not a ceiling** — any record may carry
additional custom labels, and the dashboard must tolerate them (§B.5).

#### B.2 Series identity and rename tolerance

The problem: charts join datapoints across nights by a name. If someone
renames a test (or a metric, or a model key), the trend silently splits in
two.

**Decided: stable `id` + repo-level alias registry** (supersedes the earlier
per-dump `legacy_names` idea):

1. **Split identity from display.** Each series carries a stable slug `id`
   (the join key, never shown, never "improved") and a `display_name` (free
   to change at will, cosmetic only). Most renames are display renames, and
   with this split they stop being data events entirely. This mirrors what
   the perf test already learned the hard way with `model`: the HF id keys
   the series precisely because the mount path was unstable.
2. **When an `id` truly must change, record it centrally, not in the test.**
   A single checked-in file, `.github/tpu-ci/dashboard_aliases.json`:

   ```json
   { "aliases": { "old-series-id": "new-series-id" } }
   ```

   The dashboard applies the map at render time.

Why central instead of per-dump `legacy_names`:

- The alias lives in **one** reviewable place with git history, instead of
  being scattered across test files forever (a test carrying its own rename
  archaeology never gets to drop it — deleting the field re-splits the
  trend).
- It works **retroactively**: render-time merging means old records need no
  rewrite, and a wrong alias is fixable by editing one file.
- Chained renames (a→b→c) resolve in one map; per-dump lists have to keep
  accumulating.
- A pr-gate lint can verify the map (no cycles, no alias to a nonexistent
  id, no id claimed by two tests) — impossible to check when the mapping is
  spread across dumps.

#### B.3 Perf smoke gating and the cross mark

Contract for a perf suite:

1. Smoke legs are **optional**; when declared, they run first (server
   healthy, short sanity request, …). Measurement completeness is checked
   unconditionally by `bench_serving_runner`, and its failure takes the same
   path as a smoke failure below.
2. On smoke failure the suite dumps a **failure record instead of numbers**:

   ```python
   dump_metric("smoke_failed", 1, labels={
       "kind": "performance", "model": MODEL, "workload": WORKLOAD,
       # free text, taken from the actual failure: a health-check timeout,
       # a failed sanity request, an incomplete measurement, ...
       "reason": "server health check timed out after 1800s",
   })
   ```

   Only the shape is the contract — the `smoke_failed` name and the
   `kind`/`model`/`workload` labels are what route the ✕ to the right
   chart; `reason` is whatever the failing leg can say about itself.

3. The dashboard renders that night's point as a **✕** on the chart; hovering
   shows `reason`. No partial/garbage numbers are ever plotted.
4. Server config + bench config panels stay attached to the ✕ point too, so
   the investigation (what config failed to even smoke?) happens on the
   dashboard.

#### B.4 Feature-validation rows

No metric dump needed — the `Test Summary` / `TIMINGS` block (or the
multi-host per-file records) is the row. The requirements are the existing
log-contract ones: the `== test files: ... ==` marker precedes anything that
can fail, P/D emits exactly one TIMINGS record (prefill leader), and every job
maps to exactly one test file so a red job names its test.

#### B.5 Custom information dump (extensibility)

The contract above pins the minimum; it must not become a straitjacket. The
schema is open in both directions — extra labels and extra record fields:

1. **Open field set.** Beyond the reserved v1 record fields (`filename`,
   `test_case`, `metric_name`, `value`, `ts`, `labels`), a test may add
   **any extra top-level field** to the JSON dump. keel stores unknown
   fields untouched and ignores them until someone needs them — at which
   point parsing/rendering is added on the keel side with no test changes,
   and the historical records already carry the data.

2. **Open label set.** `labels` is already a free-form dict in `dump_metric`
   v1; a test may attach any extra labels it wants
   (`variant`, `attention_backend`, `quant`, `ep_size`, …). The dashboard
   treats every non-reserved label as an opaque **tag**: shown in the
   datapoint's hover/details, and offered as a group-by / filter dimension in
   the UI. Unknown labels are **never** an error and never dropped — that is
   the forward-compatibility guarantee. A new axis of analysis is one extra
   label in the test, zero keel changes.

3. **Reserved keys live here.** The contract owns exactly:
   `kind`, `id`, `display_name`, `model`, `workload`, `eval`, `reason`.
   Everything else is custom. A future contract key is added to this list
   (and the §C.2 lint) *first*, so it can never silently collide with a label
   some test already uses.

4. **Labels are dimensions, not payloads.** A label keys/filters series, so
   its values must be low-cardinality and stable (a quant mode, a backend
   name — not a timestamp, request id, or full config blob). Bulky one-off
   information goes in a **`kind: info`** record instead:

   ```python
   # Not plotted; rendered verbatim in the test's details panel on the
   # dashboard. `value` may be a string or JSON-serializable dict.
   dump_metric("tuning_notes", json.dumps({"compile_buckets": [8192],
                                           "kv_dtype": "fp8"}),
               labels={"kind": "info", "model": MODEL})
   ```

   `kind: info` is the extension escape hatch: anything a future test wants
   visible on the dashboard but has no chart for (environment fingerprints,
   artifact links, tuning notes) ships as an info record today and can be
   promoted to first-class rendering later — the data is already flowing.

---

### C. Enforcement

Consistency comes from three layers, so nobody has to remember this doc:

#### C.1 Documentation

This file, plus the "Dashboard — making a new test show up" section of
[`README.md`](README.md), updated to the new contract once implemented.

#### C.2 pr-gate unit-test protocol

A fast, hermetic check (no TPU) in the pr-gate that fails a PR violating the
contract:

- **1:1 job↔test:** render each nightly workflow's discovery
  (`discover_tests.py`, `discover_multihost.py`, the static P/D jobs) and
  assert every emitted job carries exactly one test file, and every
  nightly-registered test is reachable from exactly one job.
- **Label lint:** AST-scan `test/registered/tpu/nightly-perf/**` and
  `nightly-accuracy/**` for the required `MODEL_CONFIG` fields (`id`,
  `display_name`, `model`) and the required labels (`kind: accuracy` /
  `kind: performance`, `model`, and `workload` for perf) in their
  `dump_metric` calls / `MODEL_CONFIG` gates; scan that perf tests go
  through `RecipeLauncher` + `bench_serving.sh`, and that any declared smoke
  legs precede the sweep.
- **Alias registry lint** (§B.2): ids unique, alias
  map acyclic, targets exist.
- **Reserved-label lint:** custom labels must not use the §B.5 reserved keys
  with off-contract meanings (e.g. a hand-rolled `kind` value other than
  `accuracy`/`performance`/`info`).
- **No-kind escalation:** after migration, "metric carries no kind" is an
  error, not a warning.

#### C.3 Claude skill

A `.claude/skills/tpu-nightly-dashboard/SKILL.md` (registered in
`.claude/rules/modify-component-must-read.md`) that triggers on touching
`test/registered/tpu/nightly-*/**`, the TPU nightly workflows, or
`.github/tpu-ci/**`, and encodes: the taxonomy, the required labels, the
stable-id rule, the smoke-leg contract, and "check your log against the keel
parser before waiting a night" (README §Dashboard sanity-check commands).
