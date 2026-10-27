# TPU CI (GKE, tpu7x)

SGLang's TPU tests run from GitHub Actions on `tpu-cluster` (project `sglang-torchtpu`,
`us-central1-c`). Every test runs as a Kubernetes JobSet, and Kueue decides when it gets
TPU hosts. Auth is keyless: workflows exchange GitHub's OIDC token for the `tpu-ci@`
service account through Workload Identity Federation.

There are four pipelines. The first three share one runner workflow and one Kueue setup;
the external one has its own.

| Pipeline | Workflow | When | Runs |
|---|---|---|---|
| PR gate | `pr-test-tpu.yml` | on demand per PR | stage A unit tests, stage B e2e tests |
| Nightly | `nightly-test-tpu.yml` | 01:00 PT | feature validation, perf |
| Accuracy | `accuracy-test-tpu.yml` | 10:00 and 17:00 PT | one job per benchmark |
| External | `nightly-test-external-tpu.yml` | 10:30 UTC | `external-nightly-*` suites on the external cluster, no Kueue; see [`external/README.md`](external/README.md) |

```
pr-test-tpu.yml ─┐                                  ┌─ render_jobset.py → JobSet (suspended)
nightly ─────────┼─ _tpu-build.yml (one image) ─────┤─ Kueue admits it onto a slice
accuracy ────────┘   _tpu-test.yml (one per job) ───┤─ pods run run_suite / recipe / command
                                                    └─ runner streams logs, reads the verdict
dashboard ◄── radixark/ci-monitor scrapes the job logs (12:00, 16:00, 21:00 UTC)
```

## Capacity: pools, lanes and queues

A **lane** is who a job is for: `pr-gate` or `nightly`. A **CI pool** is any node pool
whose nodes carry a `ci-tpu-pool` label (a unique name per pool); its **shape** is its
`cloud.google.com/gke-tpu-topology` label. **Every pool is shared by every lane**: no pool
belongs to a lane, and `topology.yaml` holds only the lanes, the priority lane and, per
shape, its hosts per slice and each lane's `min_hosts`.

`reconcile_quota.py` builds the Kueue objects from that and the live nodes:
- one ResourceFlavor per CI pool (selecting its `ci-tpu-pool`);
- per shape, one ClusterQueue + LocalQueue per lane, named `tpu-<lane>-<hosts>h`
  (`tpu-prgate-1h`, `tpu-nightly-2h`, ...), all in cohort `ci-<shape>` and each listing
  every pool of the shape.

CI pools today:

| Node pool | `ci-tpu-pool` | Shape | Hosts |
|---|---|---|---|
| `ci-tpu7x-2x2x1` | `pr-gate` | 2x2x1 | 5 |
| `ci-tpu7x-2x2x1-nightly` | `nightly-1h` | 2x2x1 | 3 |
| `ci-tpu7x-2x2x2` | `nightly` | 2x2x2 | 2 (one slice) |

The `ci-tpu-pool` values are only names; they do not tie a pool to a lane.

**How a shape's slices are split** (`priority: nightly`, `min_hosts: {pr-gate: 5}` on 2x2x1):
- Each lane keeps its `min_hosts`, in whole slices, and never lends them, so a PR never
  waits behind a nightly job for those 5 hosts.
- The priority lane (nightly) holds every other slice. When one frees up, a waiting
  nightly job gets it before a PR job that would have to borrow it.
- While nightly is idle, PR jobs may also use nightly's slices. Running jobs are never
  preempted; nightly waits for borrowed PR jobs to finish (at most one PR job's runtime).

In Kueue terms: reserved quota has `lendingLimit: 0`, the priority lane's quota is fully
lendable, queues prefer a pool where the job fits their own quota before borrowing
(`flavorFungibility.whenCanBorrow: TryNextFlavor`), and `reclaimWithinCohort: Never`.

The CI pools share one TPU reservation with the developer pools (tpuq, tpod, Haolin's).
Moving chips between them is a node-pool change by hand.

**Slice rules.** `tpu7x-standard-4t` has 4 chips per host: a `2x2x1` slice is 1 host, a
`2x2x2` slice is 2, a `2x2x4` slice is 4.
- A **2x2x1** pool can hold any number of hosts, because every host is a whole slice. It
  can be resized in place.
- A **multi-host** pool must hold exactly one slice. tpu7x nodes have no per-slice label, so
  `ci-tpu-pool` is the only way to keep a JobSet's pods inside one slice; two slices behind
  one label let pods straddle them and hang at ICI rendezvous. Add capacity as a new pool
  with a new `ci-tpu-pool` value. GKE also refuses to resize a multi-host pool.

## PR gate — `pr-test-tpu.yml`

```
pull_request ─▶ decide ─▶ build ─┬─▶ unit  (stage-a-test-tpu-unit) ─┐
                                 └─▶ e2e   (stage-b-test-tpu-4-v7) ─┴─▶ pr-test-tpu-finish
```

1. **`decide`.** Runs on `opened`, `synchronize`, `reopened`, `ready_for_review` and
   `labeled`. A PR is TPU-relevant if it touches the path list in the workflow or carries
   the `tpu` label. On the first attempt a TPU-relevant PR **fails on purpose** and posts a
   comment: TPU time is spent only when someone asks.
2. **Trigger.** Any rerun runs for real: `/tag-and-rerun-ci` or `/rerun-failed-ci` from a
   user in `CI_PERMISSIONS.json` or from the PR author, or "Re-run failed jobs" in the
   Actions UI. A new push cancels the running TPU run (one concurrency group per PR), and
   the new run starts red again.
3. **`build`.** One image for the PR head commit, through Cloud Build
   (`_tpu-build.yml`), tagged `YYYYMMDD-<actor>-<sha12>-<run_id>.<attempt>` in
   `us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu`. The base
   image is rebuilt only when `pyproject_tpu.toml` or the base Dockerfile changes.
4. **`unit` and `e2e` run in parallel**, each as one JobSet in `tpu-prgate-1h` (the
   shared 2x2x1 pool), with `SGLANG_PLATFORM=tpu_oot`:
   - `unit` runs every file registered to `stage-a-test-tpu-unit`. Files registered with
     `chips=1` run four at a time, one per chip; the rest run one after another.
   - `e2e` runs every file registered to `stage-b-test-tpu-4-v7`, with the shared model
     weights mounted.
   - A PR run stops at its first failing file.
5. **`pr-test-tpu-finish`** is the required check. It fails if any job before it failed or
   was cancelled. `/rerun-failed-ci` reruns only the failed stage, on the same image.

Tests are found by `run_suite.py` inside the pod, from `register_tpu_ci(suite=...)`; the
workflow does not list them.

## Nightly — `nightly-test-tpu.yml`

Runs at 01:00 PT (DST-aware). It builds one image, then calls two stage workflows in
parallel, all on the `nightly` lane. Kueue sorts out which job runs when.

```
build ─┬─▶ feature-validation-tpu.yml
       │     ├─ single-host:  one job per file registered to nightly-tpu-4-v7 (tpu_oot)
       │     ├─ pd-sh-qwen3-vl-30b: one hand-wired 1-prefill + 1-decode job
       │     └─ multi-host:   one job per file registered to nightly-tpu-multihost
       │                      with a MULTIHOST_LAUNCH dict (2x2x2 lane)
       └─▶ perf-test-tpu.yml: one job per perf suite, listed in the workflow (2x2x1 lane)
```

- **One job per test file.** A red job names its test, which is what the dashboard keys
  on. `discover_tests.py` builds the single-host matrix and `discover_multihost.py` the
  multi-host one; both read the `register_tpu_ci` calls.
- **Multi-host launch.** A multi-host test declares its own server in a module-level
  `MULTIHOST_LAUNCH` dict (`launch_script` under `benchmark/tpu/`, `hosts`, optional
  `weights_pvc`). Every host runs the recipe; rank 0 runs the test file against the live
  server (`SGLANG_TPU_USE_LIVE_SERVER=1`).
- **Dispatch.** `gh workflow run nightly-test-tpu.yml` takes `only` (one stage, or one
  leg), `test_filter` (a path substring) and `update-dashboard` (publish to the dashboard,
  main only). Each stage workflow can also be dispatched alone, with its own build.

## Accuracy — `accuracy-test-tpu.yml`

Runs at 10:00 and 17:00 PT. A `plan` job maps the cron that fired to a selection
(`day-10am` or `day-5pm`); dispatch takes `model` instead.

| Slot | 1-host lane | 2-host lane |
|---|---|---|
| `day-10am` | GPT-OSS-120B AIME / GPQA / GSM8K / MMLU-Pro; GPT-OSS-20B; Gemma 4 AIME / GPQA; Qwen3.8 AIME / GPQA | Kimi-K2.6 MMMU shards |
| `day-5pm` | Gemma 4 / Qwen3.5 / Qwen3.8 MMMU-Pro | DeepSeek R1 / V3.1 accuracy |

- **Single-host:** one job per benchmark (one suite, one file), each written out in the
  workflow with its suite, `run_name`, timeout and the selections it belongs to.
- **Multi-host:** files registered to `nightly-tpu-multihost-accuracy` are discovered, and
  a name pattern splits them between the two slots.
- **`collect-full-evaluations`** merges the Kimi MMMU shards' results and checks that all
  1,730 MMMU-Pro questions were answered exactly once.

## How one TPU job runs — `_tpu-test.yml`

Every in-project job calls this workflow; its work is one shell step. In order:

1. **Auth.** Workload Identity Federation as `tpu-ci@`, then credentials for
   `tpu-cluster`. The step re-mints the one-hour cluster token itself every 40 minutes.
2. **Kueue sync.** `reconcile_quota.py` creates any missing flavor, queue or LocalQueue
   and sets every CI queue's quota from the live Ready hosts (see Capacity). A multi-host
   pool holding more than one slice gets no quota. Any error fails the job rather than
   carrying on with stale quota.
3. **Render.** `render_jobset.py` fills `jobset-tpu.yaml` (or `jobset-tpu-pd.yaml` for
   P/D) for one of four modes:
   - **suite:** one host runs `run_suite.py --suite S`;
   - **launch:** every host runs the recipe, rank 0 runs the files against it;
   - **exec:** every host runs one command;
   - **P/D:** prefill and decode replicated jobs, with `CI_PD_ROLE` set.
4. **Placement.** The pod's nodeAffinity allows any CI pool (`ci-tpu-pool` exists) of the
   job's shape; the flavor Kueue admits it to pins the one pool.
5. **Submit and admission.** `kubectl create` the JobSet (suspended, labelled with the
   queue), then check that Kueue created a Workload for it; no Workload fails the job.
   Wait for admission (pods appear) for up to `admit_s`; running out is
   **`NOT ADMITTED`**.
6. **Run.** Wait up to 900 s for a pod to reach Running (only a warning after that), then
   stream every pod's log to the runner, re-attaching if the stream drops.
7. **Verdict.** The JobSet's `Completed` or `Failed` condition decides PASSED or FAILED; on
   failure the step dumps pods, a log tail, workloads and events. Running out of `wait_s`
   is `TIMED OUT`. A benchmark's JSON result is copied to a run artifact and the job
   summary.
8. **Cleanup.** The step deletes the JobSet when it exits. A finished JobSet is also
   deleted by Kubernetes after 2 hours.

**Budgets** (seconds). The runner lives at most 6 hours (21,600 s), GitHub's cap:

| Mode | JobSet deadline | Runtime wait `wait_s` |
|---|---|---|
| suite | 1.5 × the files' `est_time` (at least 360), less with chip workers | deadline + 600 |
| launch | 3600 + files × `timeout_per_file` | the same + 600 |
| exec | 7200 | 7800 |
| P/D | 8700 | 9300 |

`admit_s = 21600 − wait_s − 1500`: time in the queue comes out of the same 6 hours, so a
long job gets less of it. `est_time` therefore matters: it sizes the deadline, the wait
and the admission budget.

**Weights.** Jobs that need models mount a read-only PVC at `/weights`. The key
`model-weights` maps to the claim named in `weights.yaml` (today the Filestore claim
`model-weights-filestore`). JAX's compile cache is a gcsfuse mount of
`gs://sglang-torchtpu-ci-compile-cache`; torch_tpu's tier-2/3 caches live in `/dev/shm`.

## Kueue

- **Admission:** a JobSet waits suspended until its ClusterQueue has quota for every pod.
  A PodSet gets exactly one flavor, which is what keeps a multi-host job on one slice.
  Queues are `BestEffortFIFO`.
- **Gangs:** `waitForPodsReady` (5 min, `blockAdmission: true`) evicts and requeues a job
  whose pods cannot all start. While it waits, Kueue admits nothing else cluster-wide.
- **Quota follows the pools** through `reconcile_quota.py`, before every CI job. Quota for
  chips that do not exist admits jobs that hang; missing quota leaves live hosts idle.
- **Sharing:** every shape's queues form one cohort, `ci-<shape>`, and may use each
  other's idle, lendable quota. No preemption takes it back.
- **Only labelled JobSets are managed.** A hand-launched JobSet in `tpu-ci` without a
  `kueue.x-k8s.io/queue-name` label bypasses admission and can deadlock a slice against
  CI. Label it with the lane's queue, `tpu-<lane>-<hosts>h`.

## Adding things

| To add | Do |
|---|---|
| A PR-gate test | `register_tpu_ci(est_time=..., suite="stage-a-test-tpu-unit")` (or `stage-b-test-tpu-4-v7`). Nothing else. |
| A single-host nightly test | `register_tpu_ci(..., suite="nightly-tpu-4-v7", nightly=True)`. Discovery adds the job. |
| A multi-host nightly test | Register to `nightly-tpu-multihost` and add a `MULTIHOST_LAUNCH` dict plus its recipe. Discovery adds the job. |
| A perf suite | Register it, add the suite to `run_suite.py`, and add the entry to both lists in `perf-test-tpu.yml` (the `model` choice and the default list), plus the `model` dispatch option. |
| An accuracy benchmark | Register it, add the suite to `run_suite.py`, and add a job to `accuracy-test-tpu.yml` with its slot list. Keep `run_name` to 34 characters: it prefixes pod names, which are capped at 63. |
| A hand-wired P/D or exec test | A job in `feature-validation-tpu.yml` with `files:`, plus the `only` dispatch option in both nightly workflows. |
| A 2x2x1 host | Resize any 2x2x1 CI node pool. Nothing else; the priority lane's share grows. |
| A slice of a listed shape | A new node pool with a unique `ci-tpu-pool` label (multi-host: `--tpu-topology` and a workload policy). Nothing else. |
| A new shape | One line under `shapes` in `topology.yaml`, plus the pool. |
| A lane's guaranteed hosts | `min_hosts` under the shape in `topology.yaml`, in whole slices. |

`est_time` is read by the CI, not decoration; see Budgets.

## Files

| Path | Purpose |
|---|---|
| `.github/workflows/pr-test-tpu.yml` | PR gate: decide, build, stage A and B, finish |
| `.github/workflows/nightly-test-tpu.yml` | 01:00 PT: build once, then feature validation and perf |
| `.github/workflows/feature-validation-tpu.yml` | Nightly stage: single-host and multi-host matrices, the static P/D job |
| `.github/workflows/perf-test-tpu.yml` | Nightly stage: one job per perf suite |
| `.github/workflows/accuracy-test-tpu.yml` | 10:00 / 17:00 PT accuracy slots |
| `.github/workflows/combo-test-tpu.yml` | Dispatch-only trigger for the GPT-OSS-120B 2P+2D test |
| `.github/workflows/_tpu-build.yml` | Reusable: build the image once, output its digest |
| `.github/workflows/_tpu-test.yml` | Reusable: run one job on TPU (steps above) |
| `.github/tpu-ci/topology.yaml` | Lanes, the priority lane, and per shape its hosts and each lane's `min_hosts` |
| `.github/tpu-ci/ci_topology.py` | Loads `topology.yaml`; queue and cohort names |
| `.github/tpu-ci/kueue-controller-resources.yaml` | Lowers the Kueue controller's CPU request to fit `system-pool`; re-apply after every Kueue upgrade |
| `.github/tpu-ci/reconcile_quota.py` | Creates and updates every CI flavor, queue and quota from `topology.yaml` and live pools |
| `.github/tpu-ci/render_jobset.py` | Renders a JobSet for suite, launch, exec or P/D mode |
| `.github/tpu-ci/jobset-tpu.yaml`, `jobset-tpu-pd.yaml` | JobSet templates |
| `.github/tpu-ci/discover_tests.py`, `discover_multihost.py` | Nightly single-host and multi-host matrices |
| `.github/tpu-ci/weights.yaml` | Weights key to PVC name |
| `.github/tpu-ci/pvc-model-weights-filestore.yaml` | The weights PV and PVC (Filestore IP substituted when applied) |
| `.github/tpu-ci/dashboard_aliases.json` | Dashboard series-id renames |
| `docker/tpu-base.Dockerfile`, `docker/tpu.Dockerfile` | Base image (dependencies) and final image (SGLang) |

## Deployment configuration

The cluster, project, region, service account, WIF provider and image repositories are
written into `_tpu-build.yml` and `_tpu-test.yml`. The one repository variable read is
`TPU_CI_POD_KSA`, an optional Kubernetes service account for test pods. The external
cluster's identities are repository secrets (`EXTERNAL_*`; see
[`external/README.md`](external/README.md)).

`tpu-ci@` needs `container.developer`, `artifactregistry.writer` and `logging.logWriter`
in the project, a cluster-scoped grant to manage Kueue objects, and a Workload Identity
binding from the `github` WIF pool scoped to `sgl-project/sglang-torchtpu`. Test pods run in namespace `tpu-ci`.

New CI node pools: `--machine-type=tpu7x-standard-4t`, `--reservation-affinity=specific`
on the reservation, `--node-labels=ci-tpu-pool=<unique>`,
`--workload-metadata=GKE_METADATA`. Single-host pools take no `--tpu-topology`; multi-host
pools need `--tpu-topology` and a `HIGH_THROUGHPUT` workload policy for that topology.
`system-pool` (2 × e2-standard-2) runs the JobSet operator (v0.12.0) and the Kueue
controller.

## Dashboard — making a new test show up

The nightly dashboard is generated **outside this repo**, reusing the same parsers as
the upstream sglang CI dashboard. Nothing here renders it, and nothing here needs to change to add
a test to it — its **only** input is the job log this repo already prints. What follows
is the contract that log has to satisfy.

**Correctness (overview + history tabs) is free.** Register the file the usual way and
the parser picks it up from `run_suite.py`'s `Test Summary` / `TIMINGS` block, or from
the multi-host loop's per-file records:

```python
register_tpu_ci(est_time=3000, suite="nightly-tpu-4-v7", nightly=True)
```

**Scheduled** nightlies are always ingested. A `workflow_dispatch` run is ingested only
when dispatched with `update-dashboard: true` — that stamps `[publish]` into the run
name, and the ingester honors it only on `main`, so a branch experiment can never write
the main trend. Rerun jobs freely: records are keyed `(run_id, attempt)` and the latest
attempt replaces the datapoint.

**Accuracy and performance tabs are driven by `dump_metric` labels.** A metric lands in
a tab because the test said so, never because the dashboard pattern-matched its name.
For a `MODEL_CONFIG` test (nightly-accuracy / nightly-perf) the labels come from the
config — declare `id`, `display_name`, `model` and the machinery stamps every record;
a pr-gate lint (`test/registered/unit/test_tpu_dashboard_contract.py`) fails a PR whose
config misses one:

```python
from sglang.test.test_utils import dump_metric

# -> accuracy tab
dump_metric("mmlu_score", 0.82,
            labels={"kind": "accuracy", "model": MODEL,
                    "id": "accuracy-mymodel", "display_name": "MyModel 120B"})

# -> performance tab
dump_metric(f"{test_name}.{workload}.total_throughput", tok_s,
            labels={"kind": "performance", "model": MODEL, "workload": workload,
                    "id": "perf-mymodel", "display_name": "MyModel 120B", "c": 16})

# -> no chart; stored forever, shown in the point's details panel
dump_metric("tuning_notes", json.dumps({"kv_dtype": "fp8"}),
            labels={"kind": "info", "model": MODEL})
```

The contract kinds are exactly `accuracy` / `performance` / `info` (pr-gate lint); the
legacy inference rows (`workload` ⇒ performance, `eval` ⇒ accuracy) are kept for
migration and will be removed. Labels are open: any extra label
(`variant`, `quant`, `ep_size`, …) is stored, shown on hover, and offered as a
filter/group-by — never an error. Keep label values low-cardinality; bulky one-off
payloads go in a `kind: info` record instead (as a `json.dumps` string value — `info`
is a label on the ordinary `dump_metric` record, not a separate API). The reserved
label keys are `kind`, `id`, `display_name`, `model`, `workload`, `eval`, `reason`;
a future contract key joins that list (and the lint) first, so it can never collide
with a label some test already uses.

**Test-list and chart filters come from `register_tpu_ci(labels=[...])`** — free-form
topology / model / feature tags (`multihost`, `gpt-oss-120b`, `pd`, `spec`, …),
AST-scanned from source by the dashboard and rendered as filter dropdowns on every tab. These
are *test* properties, one declaration per file; `dump_metric` labels are
*measurement* properties — don't restate one in the other.

**Series identity.** `id` is the join key across nights: never renamed, never shown.
`display_name` is cosmetic and free to change. When an `id` truly must change, record
it once in [`dashboard_aliases.json`](dashboard_aliases.json) (old → new; the dashboard merges
at render time, so history needs no rewrite). A pr-gate lint keeps the map acyclic,
its targets live, and every live `id` unique. `labels["model"]` groups a model's
charts, so keep it stable too — the canonical HF id, not a mount path.

**A failed perf run is explained, not hidden.** A perf suite may declare smoke legs
(`"smoke": True` gates, listed before the sweep); if one fails — or the measurement
itself is incomplete, which `bench_serving_runner` checks unconditionally — the sweep
dumps no numbers. It dumps one `smoke_failed` record per point instead, with the same
routing labels plus a free-text `reason`, and the dashboard renders that night as a ✕
with the reason on hover, config panels still attached.

**Every perf measurement also ends with one framed payload.** After the digest
lines and the per-number `dump_metric` calls (clean legs only — a failed leg
complains and dumps a ✕ instead), `sglang.test.tpu.emit_result_payload` prints the
whole summary between `=== TPU_BENCH_RESULT_JSON BEGIN/END ===` markers, which
`_tpu-test.yml` lifts into a run artifact + the job summary. The payload is
pretty-printed on purpose: containerd caps a log record at 16 KiB, and an indent
means a split only ever falls BETWEEN JSON tokens, so the rejoined block parses
identically — a one-line dump has twice corrupted the rest of a job log. The
digest lines come first because the payload is the fragile half: triage must never
depend on it surviving.

**Server / bench config panels come free** for a perf test that launches through
`RecipeLauncher` and measures with `benchmark/tpu/bench_serving.sh`. The launcher
prints its exact boot env, the script echoes the settings it resolved, and the
dashboard reassembles both into the two collapsed commands under each chart. A test
that boots a server by hand gets a chart with no reproduce commands.

**Multi-host tests must let the job name its files.** `jobset-tpu.yaml` prints
`== test files: ... ==` before the server boots. A job that dies during bring-up has
no test to blame otherwise, and it would land on the dashboard as a row named after
the job. Keep that line ahead of anything that can fail.

**P/D tests get the same two markers from `jobset-tpu-pd.yaml`,** driven by
`_tpu-test.yml`'s `files` input (`render_jobset.py --files`, which P/D mode
*requires* — an empty one would emit a record with no `file`). Every pod of both
slices announces the files; only the prefill leader emits the TIMINGS record,
because the other three run the same command but do not carry the verdict, and
four records would be four duplicate rows for one test.

**Exec mode emits both markers too, and needs the same `files` input.** `--exec`
without `--files` runs, but never reaches the dashboard.

**A dispatch-only workflow earns no history row.** Ingestion covers scheduled runs plus
`update-dashboard` dispatches on main, so a test that only runs from a dispatch-only
workflow such as `combo-test-tpu.yml` never trends, however well-formed its log. It needs a
leg in the nightly (or an accuracy slot) as well.

## Enabling Kueue (new cluster)

Once per cluster, in order. `_tpu-test.yml` has no hardware mutex of its own, so CI must
not run before step 3 reports its queues Active.

**1. Install** (needs cluster-admin). JobSet v0.12.0 is already present.

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/kueue/releases/download/v0.18.5/manifests.yaml
kubectl -n kueue-system rollout status deploy/kueue-controller-manager --timeout=180s
```

⚠️ **Steps 1 and 2 must be completed in one sitting.** The install registers webhooks on
ALL JobSets with `failurePolicy: Fail`, served by the controller pod. From the moment the
manifests apply until that pod is Ready, **every JobSet creation in the cluster is
rejected** — including main's PR gate and the nightly. Two consequences:

- **Lower the controller's CPU request immediately after applying.** It ships asking for
  500m, which does not fit `system-pool`, so the pod sits `Pending` and the cluster-wide
  outage persists (observed 2026-08-14: 50 minutes). The release manifests reset this, so
  re-apply after every install or upgrade:

  ```bash
  kubectl -n kueue-system patch deploy kueue-controller-manager \
    --patch-file .github/tpu-ci/kueue-controller-resources.yaml
  kubectl -n kueue-system rollout status deploy/kueue-controller-manager --timeout=240s
  ```

- **Growing `system-pool` does not fix this.** GKE's DNS autoscaler scales `kube-dns` with
  node count: a node added to relieve the pressure took five replicas at 270m each (1350m
  of 1930m allocatable) within four minutes, and Kueue still would not fit. Lower the
  request instead.

- If you need to back out entirely: `kubectl delete -f <same manifests URL>` removes the
  webhooks with the rest. Do this while the controller is *healthy* — deleting during an
  outage works but leaves the cluster rejecting JobSets until the webhooks are gone.

**2. Configure** — set `kueue-manager-config` to the following, then
`kubectl -n kueue-system rollout restart deploy/kueue-controller-manager`. The v0.18.5
default enables ~17 frameworks and comments out everything else, so this is a replacement
of `integrations`, not an addition to it.

```yaml
integrations:
  frameworks:
    - "jobset.x-k8s.io/jobset"      # only jobset -- nothing else is ever managed
waitForPodsReady:                   # evict + requeue a gang that cannot fully place
  enable: true
  timeout: 5m
  blockAdmission: true
managedJobsNamespaceSelector:       # inert today; scopes the setting below if ever enabled
  matchLabels:
    kubernetes.io/metadata.name: tpu-ci
```

**`manageJobsWithoutQueueName` is deliberately left unset (`false`).** Kueue therefore
manages only JobSets that carry a `kueue.x-k8s.io/queue-name` label, and ignores everything
else. The obvious alternative — set it `true` so an unlabelled JobSet fails closed — is not
used, because `_tpu-test.yml` already asserts after every create that Kueue produced a
`Workload` for its JobSet and fails the job otherwise. That covers every JobSet CI submits,
which is every JobSet that matters for automation, and it fails *loudly* rather than by
silently queueing. Setting it `true` would add coverage only for hand-launched JobSets,
while suspending any debug JobSet in `tpu-ci` that lacks a label.

**So: a hand-launched JobSet in `tpu-ci` that needs a slice must carry the queue label**
(`tpu-<lane>-<hosts>h`). Without one it bypasses admission and can
deadlock a multi-host slice against a CI job. That is the one gap this design accepts.

Trimming `integrations.frameworks` to `jobset` is defence in depth for that decision: if
anyone later sets `manageJobsWithoutQueueName: true` against the stock list, it would
suspend the `hydrate-deepseek-r1.yaml` weights Job (a plain `batch/Job` in this namespace)
and any unlabelled pod, deployment or statefulset — with no queue to admit them and no
error anywhere. With only `jobset` enabled, that mistake cannot reach them.

Then confirm nothing unexpected was caught: `kubectl get workloads -A`.

**3. Create the queues.**

```bash
python3 .github/tpu-ci/reconcile_quota.py --dry-run     # server-validated plan
python3 .github/tpu-ci/reconcile_quota.py               # every CI job also runs this
kubectl get clusterqueues -o wide                       # ACTIVE=True on each
kubectl -n tpu-ci get localqueues
```

`ACTIVE=True` is necessary but not sufficient. A queue is Active as soon as its flavors
resolve, even if it can never admit anything — so also check that a real workload reaches
`ADMITTED=True` (step 5). Two ways a live-looking queue admits nothing:

- **A resource the pods request is not in `coveredResources`.** Kueue needs *every*
  requested resource declared, so TPU alone is not enough — the pods also ask for `cpu` and
  `memory`, and omitting them fails with `couldn't assign flavors to pod set slice:
  resource memory unavailable in ClusterQueue`. The quotas here derive cpu and memory from
  the TPU quota (1 cpu and 8Gi per chip) so TPU stays the only binding resource.
- **The flavor's pool does not exist** — see step 4.

**4. Every flavor's pool must exist**, or workloads queue until the admission timeout. A
quota whose pool does not exist still admits workloads, and their pods then find no
matching node. See "Capacity" for the pools that exist today.

```bash
kubectl get nodes -L ci-tpu-pool,cloud.google.com/gke-tpu-topology
# compare with the Capacity table
```

**5. Check gang admission.** Slice isolation relies on Kueue assigning each PodSet to a
single ResourceFlavor. Submit two multi-host JobSets at the same time: the first is
admitted and the second stays queued with no pods scheduled, and both pods of the
admitted JobSet run in the same pool:

```bash
kubectl -n tpu-ci get workloads
kubectl -n tpu-ci get pods -o wide
```

If the pods of one JobSet are spread across two pools, one flavor per slice is not
enough, and the queues need Topology-Aware Scheduling.

## Local validation helpers

Render and dry-run exactly as the workflow does (no build needed):

```bash
IMG=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu:manual
# single-host suite:
python3 .github/tpu-ci/render_jobset.py --image "$IMG" --suite stage-b-test-tpu-4-v7 \
  | kubectl -n tpu-ci create --dry-run=server -f -
# multi-host matrix, as the nightly builds it:
python3 .github/tpu-ci/discover_multihost.py --suite nightly-tpu-multihost
# Kueue changes the next job would make, validated by the API server but not written:
python3 .github/tpu-ci/reconcile_quota.py --dry-run
```

**Weights claim.** `pvc-model-weights-filestore.yaml` keeps the Filestore address as a
placeholder; substitute it when applying:

```bash
FILESTORE_IP=...   # gcloud filestore instances list --project sglang-torchtpu
sed "s/<FILESTORE_IP>/${FILESTORE_IP}/" .github/tpu-ci/pvc-model-weights-filestore.yaml \
  | kubectl apply -f -
```

**Qwen3-0.6B is single-host only:** it runs out of HBM at tp>1 on tpu7x, so multi-host
tests use Qwen3-30B-A3B-FP8.
