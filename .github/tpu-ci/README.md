# TPU CI (GKE, tpu7x)

PR-gate and nightly SGLang tests on Google Cloud TPU, dispatched from GitHub
Actions into a reserved GKE cluster. Auth is keyless (Workload Identity
Federation); no long-lived keys.

> **External cluster:** the `external-nightly-*` suites run the same tests on
> the external tpu7x cluster, serving weights from the CI bucket over gcsfuse. That path
> has its own renderer/workflow/config — see [`external/README.md`](external/README.md).

## Architecture

```
PR / nightly ─▶ GitHub-hosted runner
                 │  WIF ⇒ impersonate tpu-ci  (keyless OIDC)
                 │
                 │  _tpu-build.yml   (ONCE per run)
                 │    gcloud builds submit ─▶ Cloud Build (as tpu-ci)
                 │    → docker/tpu.Dockerfile → Artifact Registry
                 │    → outputs the immutable digest  (repo@sha256:…)
                 │
                 │  _tpu-test.yml    (per run, reusing that digest)
                 │    render_jobset.py → ONE JobSet (slice sized from topology.yaml)
                 │    labelled for Kueue → created SUSPENDED
                 ▼
            GKE tpu-cluster (us-central1-c)
                 │
                 │  Kueue gang-admits when a whole slice's quota is free
                 │    LocalQueue → ClusterQueue → ResourceFlavor (≡ one slice)
                 ▼
                 ├─ pr-gate lane
                 │    ├─ ci-tpu7x-2x2x1          (pr-gate,    2 nodes = 2 slices)
                 │    └─ ci-tpu7x-2x2x2-prgate   (pr-gate-2h, 1 slice = 2 hosts)
                 └─ nightly lane
                      ├─ ci-tpu7x-2x2x1-nightly  (nightly-1h, 3 nodes = 3 slices)
                      ├─ ci-tpu7x-2x2x2          (nightly,    1 slice = 2 hosts)
                      └─ ci-tpu7x-2x2x4-nightly  (nightly-4h, 1 slice = 4 hosts)
```

Everything runs on the `ghostfish-u2zcmybntv70q` reservation.

**Slice arithmetic.** `tpu7x-standard-4t` is 4 chips per host, so `2x2x1` = 4 chips = 1
node, `2x2x2` = 8 chips = 2 nodes, `2x2x4` = 16 chips = 4 nodes. Each `2x2x1` node is
therefore a *complete* single-host slice, which is why those pools may hold several nodes;
a multi-host pool holds exactly one slice. Totals: **pr-gate 16 chips, nightly 36 chips.**

**Lane, not just host count.** Both lanes own 1- and 2-host slices, so a host count alone no
longer identifies a pool. `topology.yaml` is keyed `(lane, hosts)`, `_tpu-test` takes a
`lane` input defaulting to `pr-gate`, and `nightly-test-tpu.yml` passes `lane: nightly`. That
is also what finally isolates the two: before this, `hosts=1` mapped to `pr-gate` for every
caller, so the nightly's single-host jobs competed with the PR gate.

**One pool per slice is an invariant, not a preference.** GKE `tpu7x` nodes carry no
per-slice label — only accelerator, topology, nodepool, hostname and zone, and none of the
`cloud.google.com/gke-topology-*` labels Kueue's Topology-Aware Scheduling needs. So
`ci-tpu-pool` is the finest slice selector available. A pool holding two slices cannot be
split, and a JobSet's pods could straddle both and hang at ICI rendezvous. Add multi-host
capacity as a **new pool** (`ci-tpu-pool=nightly-2h-b`), never as extra nodes on an existing
slice pool. The `2x2x1` pools are the exception — each node is already its own slice, so
they are safe to resize.

### Build-once

The image is built **once** (`_tpu-build`) and every run reuses the same digest
(`_tpu-test`), so a nightly with N runs does **one** build, not N.

### Detection = (lane, host count) → slice (topology.yaml)

`topology.yaml` maps a **lane and host count** to a slice `{topology, pool, queue}`.
`render_jobset.py`
renders **one** JobSet template (`jobset-tpu.yaml`) with `completions = hosts`; the
container branches on `NNODES` (from GKE's `TPU_WORKER_HOSTNAMES`):

- **single-host** (`hosts=1`) → a 1-pod JobSet; each test self-launches its own server
  (`popen_launch_server`); `run_suite` runs the suite directly.
- **multi-host** (`hosts≥2`) → an N-pod JobSet, gang-admitted by Kueue; every host runs the
  launch **group's recipe** (which owns model + config), and rank 0 runs each of the
  group's test files in **live-server mode** (`SGLANG_TPU_USE_LIVE_SERVER=1` + `BASE_URL`),
  so tests assert on real results. Workers serve until the leader tears down, exit 0.

### Multi-host = test-owned launch, grouped by setup

The model+config is **not** bound to a suite — each multi-host test file declares its
own launch in a module-level `MULTIHOST_LAUNCH` dict:

```python
MULTIHOST_LAUNCH = {"hosts": 2, "launch_script": "recipe/qwen3_30b_a3b_fp8_mh.sh",
                    "weights_pvc": "model-weights"}   # weights_pvc optional
```

The shared PVC name is defined in `.github/tpu-ci/weights.yaml`:
`model-weights` selects the shared model weights. Use this literal key in
`register_tpu_ci(weights_pvc="model-weights")`, `MULTIHOST_LAUNCH`, standalone
launcher defaults, and workflow inputs. CI discovery reads ordinary literals;
the JobSet renderer resolves the key for both ordinary and P/D jobs.

`render_jobset.py --weights-pvc` also accepts a literal PVC name for custom
claims; an empty value mounts no weights. The container mount remains `/weights`
(test-side lookup can be overridden with `SGLANG_TPU_WEIGHTS_ROOT`). Provisioning
manifests must still name the actual cluster resources; changing `weights.yaml`
selects a claim, it does not create one.

`launch_script` is relative to `benchmark/tpu/`: a shell recipe, or a `.py`
recipe config the JobSet hands to `benchmark/tpu/launch_server.py`, which
resolves each host's rank from the JobSet topology. `discover_multihost.py`
emits one JobSet per registered test file, even when several name the same
recipe -- which is how the two Kimi MMMU-Pro shards get independent slices with
identical serving settings.

To add a **single-host** suite: `register_*_ci(..., suite="…")`, add it to `run_suite.py`,
and call `_tpu-test` with that `suite`. To add a **multi-host** test: drop a file with a
`MULTIHOST_LAUNCH` dict (+ a recipe) registered to `nightly-tpu-multihost` — discovery
picks it up and groups it automatically; identical setup joins an existing group's server.

A test with **no server that every host must run** (a collectives probe spawning its
own ranks) fits neither, since a launch group runs the files on rank 0 alone. That
shape is exec mode. Exec and P/D jobs are hand-wired: in-project in
`feature-validation-tpu.yml` with `files:`, and on the external cluster, where most
nightly feature validation runs, in `external/external-topology.yaml`'s `commands`
(a single-host test there registers to `external-nightly-1host`; see `external/README.md`).

**Every TPU suite must be declared in `run_suite.py`**, including hand-wired and
external-cluster ones. TPU is in `_SUITE_CHECKED_BACKENDS`, and `run_suite.main()`
validates every registration before it filters by `--hw`, so one TPU test registered to
an undeclared suite fails *every* backend's run_suite job at startup.
`test/registered/unit/tools/test_run_suite_registrations.py` catches this on the CPU
pr-gate.

## Pipelines

### Nightly — `nightly-test-tpu.yml`

`on: schedule` (daily) plus `workflow_dispatch` with an `only` filter (a stage, or a
single feature-validation leg) and an `update-dashboard` publish opt-in (see
[Dashboard](#dashboard--making-a-new-test-show-up) below).

```
build (once)
  ├─▶ feature validation   feature-validation-tpu.yml
  └─▶ perf                 perf-test-tpu.yml
      both in parallel: slice contention is Kueue's job (the nightly
      lane's quota queues the jobs), and a red stage costs the others nothing
```

The accuracy stage (`accuracy-test-tpu.yml`) runs on its own 10 AM / 5 PM PT schedule
instead: with it the nightly needed about twice the slice time the 6 h runner cap allows.
Each slot fills the lane its MMMU-Pro legs leave idle:

| Slot | 1-host lane | 2-host lane |
|---|---|---|
| `day-10am` | single-host accuracy (gpt-oss, Gemma 4) | Kimi-K2.6 MMMU shards |
| `day-5pm` | Gemma 4 / Qwen3.5 MMMU-Pro | DeepSeek R1 / V3.1 accuracy |

Each stage is a reusable workflow (the nightly calls feature validation and perf with the
shared image), and ALSO its own `workflow_dispatch` trigger (with its own build) for running one stage — or one
test — without a full nightly. Inside feature validation:

```
  ├─▶ single-host           (hosts=1)  one job per nightly-tpu-4-v7 test
  ├─▶ P/D feature gates     (hosts=2)  │  dedicated registered tests
  └─▶ discover ─▶ multi-host (hosts=2) ┘  matrix over discovered launch groups
```

`discover` AST-scans `test/registered/**` for `MULTIHOST_LAUNCH` on files registered to
`nightly-tpu-multihost`, groups those with an identical `(launch_script, weights_pvc,
hosts)`, and emits one matrix entry per group — so files sharing a spec share one server
and the expensive bring-up happens once per group, not once per file.

**Admission.** Every job calls `_tpu-test.yml`, which renders a JobSet labelled
`kueue.x-k8s.io/queue-name` and creates it suspended. Kueue releases it only once a whole
slice's quota is free:

| leg | LocalQueue | flavor | quota | concurrent |
|------|-----------|--------|-------|-----------|
| `hosts=1` | `tpu-nightly-1h` | `tpu7x-2x2x1-nightly` | 12 chips | 3 |
| `hosts=2` | `tpu-nightly-2h` | `tpu7x-2x2x2-nightly` | 8 chips | 1 per slice |
| `hosts=4` | `tpu-nightly-4h` | `tpu7x-2x2x4-nightly` | 16 chips | 1 per slice |

One ClusterQueue per `(lane, hosts)`. The shapes must not share a queue: a 2-host job needs
8 chips, which a single-host flavor would satisfy on quota while being unschedulable.

A PodSet is assigned exactly one ResourceFlavor, which is what keeps a multi-host JobSet's
two pods inside a single slice.

Multihost P/D (2p+2d) is one JobSet with two PodSets, one 2x2x2 slice each — the whole
`tpu-nightly-2h` quota. Kueue admits a Workload's PodSets atomically, so a P/D run takes
both slices or waits; there is no GitHub-side P/D lock.

Before creating the JobSet, `_tpu-test.yml` runs `reconcile_quota.py --pool=<pool>`, which
sets every ClusterQueue's `nominalQuota` from live allocatable chips and refuses to submit
if the target pool holds more than one slice. After creating it, the workflow asserts Kueue
produced a `Workload` for that JobSet and fails the job otherwise — so a JobSet running with
no admission control turns the run red instead of going unnoticed.

GitHub concurrency groups are **supersession only** — "don't stack two runs of me." They no
longer ration hardware. That was the old `tpu-slice-<hosts>` mutex, and it was the wrong
primitive: the group holds one in-progress plus one pending, so a third claimant silently
cancelled the pending one. Adding a slice now means a new pool, a new ResourceFlavor and one
quota entry — no workflow change.

**Budgets.** Three phases, kept separate so queue time is never charged as test time:

| phase | budget | covers |
|-------|--------|--------|
| admission | the runner's remaining lifetime | suspended in Kueue; no pods exist yet |
| running | 900s | placed, but binding + image pull |
| runtime | 9600 (launch) / 7800 (exec) / `est_time`×1.5+600 (single-host shard) | the test itself |

One slice runs one multi-host job at a time, so a job's wait is however long everything
queued ahead of it takes — there is no useful number smaller than "as much of the runner as
is left". `admit_s` is therefore derived: `timeout-minutes` (360, GitHub's hard cap for
hosted runners) minus this job's own runtime, the bind/pull window and setup. The runtime
figure must OUTLAST the JobSet's `activeDeadlineSeconds` so the JobSet — not the poll loop —
reports the outcome.

The nightly single-host jobs are ONE JOB PER TEST FILE (`discover_tests.py` renders the
matrix), so a red job names its test directly — the dashboard contract. Kueue queues
them against the nightly lane's single-host quota, and they all reuse one image build.

**The leaf job is named after the test.** The nightly nests two reusable workflows deep
(`nightly-test-tpu.yml` → stage → `_tpu-test.yml`), and the Actions sidebar of a nightly
run shows only the innermost job's name per leg — the stage job's name is elided. So
`_tpu-test.yml` names its job from `display_name` (falling back to the suite), and every
caller passes the test it runs. The stage job keeps the same name, taken from the matrix:
a static name on a matrix job makes GitHub append every matrix value in parentheses.

**`run_name` has a budget.** It prefixes every Kubernetes object a leg creates, and JobSet
validates the POD name — `tpu-ci-<run_name>-<5 rand>-<role>-<job idx>-<pod idx>-<5 rand>` —
against the 63-char DNS-label cap at admission. That leaves `run_name` 34 chars on a slice
and 32 on a P/D role; the discovery scripts hand every leg a 12-hex compact name for this,
and a hand-written one (`accuracy-<model>-<eval>`, `pd-mh-gpt-120b-spec`) stays short. The
suite name is NOT a safe default — `nightly-tpu-accuracy-gpt-oss-120b-mmlupro` (41) is
rejected. Neither this nor the job name is a dashboard key: series identity is the metric
`id`.

**Adding a nightly test.** Single-host: `register_tpu_ci(est_time=…, suite=…, nightly=True)`
plus the suite in `run_suite.py`. Multi-host: drop a file with a `MULTIHOST_LAUNCH` dict
registered to `nightly-tpu-multihost` — discovery groups it automatically, and an identical
spec joins an existing group's server for free.

`est_time` is not decoration: it sizes both the JobSet's `activeDeadlineSeconds` and the CI
poll budget, so an inflated value pushes a serialized run toward the next day's cron.

**Known gaps.**

- Worst-case serialized multi-host work (~11.5h at full per-job allowances) exceeds
  GitHub's 6h hosted-runner cap. Real runtimes are well inside it, but the fix is measured
  deadlines or a second 2x2x2 slice.
- No `est_time` refit loop — values are hand-written and have drifted by ~8x.
- Results land only in the run summary; no digest, no notification.

### PR gate — `pr-test-tpu.yml`

_TODO._

### Combo — `combo-test-tpu.yml`

_TODO._

## Files

| Path | Purpose |
|------|---------|
| `docker/tpu-base.Dockerfile` | TPU base image: installs `python/pyproject_tpu.toml` from the private torch-tpu registry (token via BuildKit secret) |
| `docker/cloudbuild.tpu-base.yaml` | Cloud Build config for base image; runs **as `tpu-ci`**, pushes `sglang-tpu-base` to AR |
| `docker/tpu.Dockerfile` | Final TPU image: builds on `sglang-tpu-base` and installs editable SGLang without reinstalling deps |
| `docker/cloudbuild.tpu.yaml` | Cloud Build config for final image; builds on base and pushes `sglang-tpu` to AR |
| `.github/tpu-ci/topology.yaml` | `(lane, host count)` → slice `{topology, pool, queue}` (detection source of truth) |
| `.github/tpu-ci/kueue-controller-resources.yaml` | Strategic-merge patch dropping the Kueue controller's CPU request to fit `system-pool`; re-apply after every Kueue upgrade |
| `.github/tpu-ci/kueue-tpu.yaml` | Kueue ResourceFlavors (one per slice), one ClusterQueue + LocalQueue per `(lane, hosts)` |
| `.github/tpu-ci/jobset-tpu.yaml` | **Unified** JobSet template (any host count); rendered by `render_jobset.py` |
| `.github/tpu-ci/render_jobset.py` | Renders the JobSet — single-host suite mode or multi-host group mode |
| `.github/tpu-ci/reconcile_quota.py` | Sets each ClusterQueue's `nominalQuota` from live allocatable chips; refuses to run if a multi-host pool holds more than one slice. Run by `_tpu-test.yml` before every submission; `--dry-run` locally |
| `.github/tpu-ci/discover_multihost.py` | Scans `MULTIHOST_LAUNCH` decls, groups files by launch spec → matrix JSON |
| `benchmark/tpu/recipe/*_mh.sh` | Per-model multi-host launch recipes (own `MODEL`/`EP_SIZE`/…, exec `launch_server.sh`) |
| `.github/workflows/_tpu-build.yml` | Reusable: build once → output digest |
| `.github/workflows/_tpu-test.yml` | Reusable: render JobSet (suite or group) → wait for Kueue admission → stream logs → gate on JobSet condition |
| `.github/workflows/pr-test-tpu.yml` | `on: pull_request` (path-or-`tpu`-label gated, on demand: runs only on a `/tag-and-rerun-ci` re-run); decide → build → single-host test |
| `.github/workflows/nightly-test-tpu.yml` | `on: schedule` + dispatch; build once → feature validation and perf stages in parallel |
| `.github/workflows/feature-validation-tpu.yml` | Feature-validation stage, also dispatchable alone: single-host matrix (one job per test) + static P/D gates + discover → multi-host matrix over groups |
| `.github/workflows/accuracy-test-tpu.yml` | Accuracy stage, on its own 10 AM / 5 PM schedule and dispatchable alone: one job per benchmark via run_suite (suites `nightly-tpu-accuracy-<model>-<benchmark>`); ondemand multihost accuracy models (dispatch-only) |
| `.github/workflows/perf-test-tpu.yml` | Perf stage, also dispatchable alone: one job per model via run_suite |
| `.github/tpu-ci/dashboard_aliases.json` | Series-id rename registry: old-id → new-id, applied by keel at render time (pr-gate lint) |

## Cluster infra

CI node pools, all on the `ghostfish-u2zcmybntv70q` reservation. Create a new one with
`--machine-type=tpu7x-standard-4t --reservation-affinity=specific
--node-labels=ci-tpu-pool=<unique>,ci-tpu-lane=<lane> --workload-metadata=GKE_METADATA`;
multi-host also needs `--tpu-topology` plus a `workload-policy` resource policy
(`--type=HIGH_THROUGHPUT --accelerator-topology=<topo>`).
Single-host pools take **no** `--tpu-topology` — GKE derives `2x2x1` from the machine type,
and passing it takes the multi-host path, which is rejected for this machine type.

**Both labels matter.** `ci-tpu-pool` must be unique per slice (it is the only selector
that can pin a gang to one slice — tpu7x exposes no per-slice label). `ci-tpu-lane` is what
`discover_pools.py` reads to place the pool in a lane without a code change: it unions the
discovered pools into `topology.yaml`'s list, mints the Kueue `ResourceFlavor`, and attaches
it to that `(lane, hosts)`'s ClusterQueue. Omit `ci-tpu-lane` and the pool is simply never
used — no error, so check the `pools for lane=… hosts=…` line in the job log.

| pool | `ci-tpu-pool` | `ci-tpu-lane` | topology | nodes | chips |
|------|---------------|---------------|----------|-------|-------|
| `ci-tpu7x-2x2x1` | `pr-gate` | `pr-gate` | 2x2x1 | 2 | 8 |
| `ci-tpu7x-2x2x2-prgate` | `pr-gate-2h` | `pr-gate` | 2x2x2 | 2 | 8 |
| `ci-tpu7x-2x2x1-nightly` | `nightly-1h` | `nightly` | 2x2x1 | 3 | 12 |
| `ci-tpu7x-2x2x2` | `nightly` | `nightly` | 2x2x2 | 2 | 8 |
| `ci-tpu7x-2x2x2-b` | `nightly-b` | `nightly` | 2x2x2 | 2 | 8 |
| `ci-tpu7x-2x2x4-nightly` | `nightly-4h` | `nightly` | 2x2x4 | 4 | 16 |

The pools above predate `ci-tpu-lane` and are still carried by `topology.yaml`'s static
list, so they work whether or not the label is applied; the column records what to set when
backfilling. Discovery unions rather than replaces precisely so a partial backfill cannot
shrink the pool set.

Only the `2x2x1` pools are safe to resize — each of their nodes is already a whole slice,
so quota follows automatically via `reconcile_quota.py` and no manifest edit is needed.
Resizing a multi-host pool puts two slices behind one `ci-tpu-pool` label and a JobSet's
pods can straddle them: no error, just an ICI rendezvous hang. `reconcile_quota.py` refuses
to submit when it sees that, but the fix is to add capacity as a **new pool** — which, with
both labels set, needs no edit to `topology.yaml` or `kueue-tpu.yaml`.

- `tpu7x-2x2x2-haolin-dev` and `tpu7x-2x2x2-mimo` also exist — not CI, do not repurpose.
- `system-pool` (e2-standard-2 ×2) hosts the **JobSet operator v0.12.0** and the **Kueue
  controller** (the e2-medium default-pool is CPU-saturated by GKE daemons).
- Artifact Registry: `us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci`.
- Service account `tpu-ci@sglang-torchtpu.iam.gserviceaccount.com` with
  `container.developer`, `artifactregistry.writer`, `logging.logWriter`.
- WIF pool `github` + provider `github-oidc`, scoped to `sgl-project/sglang-torchtpu`,
  bound to `tpu-ci` via `workloadIdentityUser`.
- Namespace `tpu-ci`; KSA `weights-loader` granted `storage.objectViewer` on
  `gs://sglang-torchtpu-gcs-shared` via Workload Identity (no GSA key).

## Enabling Kueue

Once per cluster, in order. `_tpu-test.yml` has no hardware mutex of its own, so it must
not merge before step 3 reports all five queues Active.

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
(see `topology.yaml` for the lane's queue). Without one it bypasses admission and can
deadlock a multi-host slice against a CI job. That is the one gap this design accepts.

Trimming `integrations.frameworks` to `jobset` is defence in depth for that decision: if
anyone later sets `manageJobsWithoutQueueName: true` against the stock list, it would
suspend the `hydrate-deepseek-r1.yaml` weights Job (a plain `batch/Job` in this namespace)
and any unlabelled pod, deployment or statefulset — with no queue to admit them and no
error anywhere. With only `jobset` enabled, that mistake cannot reach them.

Then confirm nothing unexpected was caught: `kubectl get workloads -A`.

**3. Apply the quotas.**

```bash
kubectl apply -f .github/tpu-ci/kueue-tpu.yaml
kubectl get clusterqueues -o wide        # ACTIVE=True on all five
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
quota whose pool is absent is fiction — Kueue admits against the number and the pods then
find no matching node. All five exist today; see "Cluster infra" to add or check one.

```bash
kubectl get nodes -L ci-tpu-pool,cloud.google.com/gke-tpu-topology
# expect: 4 pr-gate, 2 pr-gate-2h, 2 nightly-1h, 2 nightly, 4 nightly-4h
```

**5. Verify gang admission** before merging. The design assumes Kueue assigns a whole
PodSet to one ResourceFlavor; everything about slice isolation rests on it. Submit two
multi-host JobSets at once — the first should admit, the second stay queued (not
half-scheduled), and the admitted one's two pods share a pool:

```bash
kubectl -n tpu-ci get workloads
kubectl -n tpu-ci get pods -o wide
```

If pods from one JobSet ever straddle two pools, flavor-per-slice is wrong and this needs
Topology-Aware Scheduling instead.

## ⚠️ Required before CI can run (owner actions)

1. **Allowlist `tpu-ci` on the private torch-tpu registry.** The build pulls the
   pinned `torch-tpu` wheel from `ml-oss-artifacts-transient` (Google-owned). Ask
   Google / the TPU program (duckie or Jetski) to grant:
   > `roles/artifactregistry.reader` to
   > `serviceAccount:tpu-ci@sglang-torchtpu.iam.gserviceaccount.com` on the
   > `torch-tpu-virtual-registry` Python repo in project `ml-oss-artifacts-transient`.

   Until this lands, `gcloud builds submit` (and therefore every CI run) fails at
   the `pip install` step. Validation to date builds once under the owner's own
   identity (which already has read access) and pushes to AR (`:manual`).

2. **Rotate the Docker Hub PAT** that was shared in chat, and store the new token
   as repo secrets `RADIXARK_DOCKERHUB_USERNAME` / `RADIXARK_DOCKERHUB_TOKEN`
   (only needed for publishing release images, not for TPU CI).

3. **Pin action versions to commit SHAs** to match this repo's security convention
   (currently `@v4`/`@v2`).

## Status

- **PR gate (single-host, run_suite)** — validated GREEN on the 2x2x1 slice.
- **Multi-host (2x2x2, tp=16)** — validated GREEN: Qwen3-30B-A3B-FP8 MoE (EP=8) serving
  from the original **ReadOnlyMany** Hyperdisk (both hosts mount concurrently),
  rank-0 asserting on real inference.
- **Filestore weights** — `model-weights-filestore` mounts the existing 10Ti
  NFSv3 share `10.93.193.130:/weights` read-only at `/weights`.

Apply the claim before running CI from this revision:

```bash
gcloud container clusters get-credentials tpu-cluster \
  --project sglang-torchtpu --zone us-central1-c
kubectl apply -f .github/tpu-ci/pvc-model-weights-filestore.yaml
kubectl -n tpu-ci wait pvc/model-weights-filestore \
  --for=jsonpath='{.status.phase}'=Bound --timeout=60s
```

Keep `model-weights-rox` until existing jobs finish and the new CI runs pass.
Resize through Filestore, then update the PV capacity; this static NFS claim
cannot resize the instance through PVC expansion.

### ⚠️ Model note: Qwen3-0.6B is single-host only

Qwen3-0.6B **OOMs at tp>1** on tpu7x (a bug: ~151G HLO temporaries > 94G HBM) — do not
use it as a multi-host launch model. Multi-host tests use Qwen3-30B-A3B-FP8.

## Dashboard — making a new test show up

The nightly dashboard is generated **outside this repo** (`ci-monitor/src/tpu_ci.py`
and `tpu_dashboard.py` in `radixark/keel`, reusing the same parsers as the upstream
sglang CI dashboard). Nothing here renders it, and nothing here needs to change to add
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
AST-scanned from source by keel and rendered as filter dropdowns on every tab. These
are *test* properties, one declaration per file; `dump_metric` labels are
*measurement* properties — don't restate one in the other.

**Series identity.** `id` is the join key across nights: never renamed, never shown.
`display_name` is cosmetic and free to change. When an `id` truly must change, record
it once in [`dashboard_aliases.json`](dashboard_aliases.json) (old → new; keel merges
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

**A dispatch-only workflow earns no history row.** Ingestion covers scheduled runs
plus `update-dashboard` dispatches on main, so a test that lives solely in
`integration-test-tpu.yml` (`workflow_dispatch` only, no publish opt-in) never
trends no matter how well-formed its log is. It needs a leg in the nightly's
feature-validation stage too — that is what the `pd-mh-gpt-120b-pc-offload` and
`pd-mh-gpt-120b-tp-reshard` jobs in `feature-validation-tpu.yml` are.

Sanity-check a new test's log against the parser before waiting a night for it:

```bash
# from a keel checkout, against nightlies that already ran (needs repo read scope)
export GITHUB_TOKEN=$(gh auth token)
python3 ci-monitor/src/tpu_ci.py sync --limit 3 --db-dir /tmp/tpu-ci
python3 ci-monitor/src/tpu_dashboard.py --db-dir /tmp/tpu-ci --out /tmp/tpu-ci.html
```

`sync` is idempotent — records are keyed by run, so re-running over the same window
converges instead of duplicating.

## Local validation helpers

Render + dry-run exactly as the workflow does (no live build needed):

```bash
IMG=us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci/sglang-tpu:manual
# single-host suite:
python3 .github/tpu-ci/render_jobset.py --image "$IMG" --suite stage-b-test-tpu-4-v7 \
  | kubectl -n tpu-ci create --dry-run=server -f -
# multi-host groups (as the nightly builds them):
python3 .github/tpu-ci/discover_multihost.py --suite nightly-tpu-multihost   # -> groups JSON
```

The old per-manifest files (`job-integration.yaml`, `job-nightly.yaml`,
`jobset-nightly.yaml`) and `_tpu-test-stage.yml` were replaced by the unified
template + `_tpu-build`/`_tpu-test`.
