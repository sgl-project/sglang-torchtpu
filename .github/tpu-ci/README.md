# TPU CI (GKE, tpu7x)

PR-gate and nightly SGLang tests on Google Cloud TPU, dispatched from GitHub
Actions into a reserved GKE cluster. Auth is keyless (Workload Identity
Federation); no long-lived keys.

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
                 ├─ ci-tpu7x-2x2x1  (ci-tpu-pool=pr-gate, 4 nodes) ── single-host (run_suite)
                 └─ ci-tpu7x-2x2x2  (ci-tpu-pool=nightly, 1 slice = 2 hosts) ── multi-host groups
```

Everything runs on the `ghostfish-u2zcmybntv70q` reservation.

**Slice arithmetic.** `tpu7x-standard-4t` is 4 chips per host, so `2x2x1` = 4 chips = 1
node and `2x2x2` = 8 chips = 2 nodes. Each `ci-tpu7x-2x2x1` node is therefore a *complete*
single-host slice — the pool's 4 nodes are 4 independent slices — while `ci-tpu7x-2x2x2`
is one slice spanning two nodes.

**The pools do not fully isolate PR from nightly.** `hosts=1` maps to `pr-gate` for every
caller, so the nightly's single-host jobs land on the same pool as the PR gate. Only
`hosts=2` work is nightly-exclusive.

**One pool per slice is an invariant, not a preference.** GKE `tpu7x` nodes carry no
per-slice label — only accelerator, topology, nodepool, hostname and zone, and none of the
`cloud.google.com/gke-topology-*` labels Kueue's Topology-Aware Scheduling needs. So
`ci-tpu-pool` is the finest slice selector available. A pool holding two slices cannot be
split, and a JobSet's pods could straddle both and hang at ICI rendezvous. Add capacity as
a **new pool** (`ci-tpu-pool=nightly-b`), never as extra nodes on an existing slice pool.

### Build-once

The image is built **once** (`_tpu-build`) and every run reuses the same digest
(`_tpu-test`), so a nightly with N runs does **one** build, not N.

### Detection = host count → slice (topology.yaml)

`topology.yaml` maps a **host count** to a slice `{topology, pool}`. `render_jobset.py`
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
                    "weights_pvc": "model-weights-rox"}   # weights_pvc optional
```

`launch_script` is a recipe under `benchmark/tpu/recipe/` that sets `MODEL`/`EP_SIZE`/…
and execs `launch_server.sh`. `discover_multihost.py` scans the files registered to a
suite, **groups** those with an identical `(launch_script, weights_pvc, hosts)`, and
emits one group per unique setup. Each group → **one** JobSet: launch the server once,
rank 0 runs every file in the group against it. So files with the same setup **share a
launched server** (no redundant bring-up); different setups get their own server.

To add a **single-host** suite: `register_*_ci(..., suite="…")`, add it to `run_suite.py`,
and call `_tpu-test` with that `suite`. To add a **multi-host** test: drop a file with a
`MULTIHOST_LAUNCH` dict (+ a recipe) registered to `nightly-tpu-multihost` — discovery
picks it up and groups it automatically; identical setup joins an existing group's server.

## Pipelines

### Nightly — `nightly-test-tpu.yml`

`on: schedule` (daily) plus `workflow_dispatch` with an `only` filter to run a single leg.

```
build ─────────────────────────────────┐
  ├─▶ single-host           (hosts=1)  │  suite: nightly-tpu-4-v7
  ├─▶ singlehost-combo      (hosts=1)  │  exec: feature_combo singlehost
  ├─▶ multihost-combo       (hosts=2)  │  exec: feature_combo pd | spec
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
| `hosts=1` | `tpu-singlehost` | `tpu7x-2x2x1-pr-gate` | 12 of 16 chips | 3 |
| `hosts=2` | `tpu-multihost` | `tpu7x-2x2x2-nightly` | 8 chips | 1 per slice |

A PodSet is assigned exactly one ResourceFlavor, which is what keeps a multi-host JobSet's
two pods inside a single slice.

GitHub concurrency groups are **supersession only** — "don't stack two runs of me." They no
longer ration hardware. That was the old `tpu-slice-<hosts>` mutex, and it was the wrong
primitive: the group holds one in-progress plus one pending, so a third claimant silently
cancelled the pending one. Adding a slice now means a new pool, a new ResourceFlavor and one
quota entry — no workflow change.

**Budgets.** Three phases, kept separate so queue time is never charged as test time:

| phase | budget | covers |
|-------|--------|--------|
| admission | 3600s | suspended in Kueue; no pods exist yet |
| running | 900s | placed, but binding + image pull |
| runtime | `est_time`×1.5 / 5400 (launch) / 7800 (exec) | the test itself |

**Adding a nightly test.** Single-host: `register_tpu_ci(est_time=…, suite=…, nightly=True)`
plus the suite in `run_suite.py`. Multi-host: drop a file with a `MULTIHOST_LAUNCH` dict
registered to `nightly-tpu-multihost` — discovery groups it automatically, and an identical
spec joins an existing group's server for free.

`est_time` is not decoration: it sizes both the JobSet's `activeDeadlineSeconds` and the CI
poll budget, so an inflated value pushes a serialized run toward the next day's cron.

**Known gaps.**

- `nominalQuota` is static and never reconciled against real node count. A node going
  NotReady over-states it; `waitForPodsReady` evicts and requeues instead of deadlocking,
  but the number is still maintained by hand.
- No `est_time` refit loop — values are hand-written and have drifted by ~8x.
- Results land only in the run summary; no digest, no notification.

### PR gate — `pr-test-tpu.yml`

_TODO._

### Combo — `combo-test-tpu.yml`

_TODO._

## Files

| Path | Purpose |
|------|---------|
| `docker/tpu.Dockerfile` | TPU image: installs `python/pyproject_tpu.toml` from the private torch-tpu registry (token via BuildKit secret) |
| `docker/cloudbuild.tpu.yaml` | Cloud Build config; runs **as `tpu-ci`**, token from the metadata server, pushes to AR |
| `.github/tpu-ci/topology.yaml` | Host count → slice `{topology, pool, queue}` (detection source of truth) |
| `.github/tpu-ci/kueue-tpu.yaml` | Kueue ResourceFlavors (one per slice), ClusterQueues and LocalQueues; header carries the cluster apply runbook |
| `.github/tpu-ci/jobset-tpu.yaml` | **Unified** JobSet template (any host count); rendered by `render_jobset.py` |
| `.github/tpu-ci/render_jobset.py` | Renders the JobSet — single-host suite mode or multi-host group mode |
| `.github/tpu-ci/discover_multihost.py` | Scans `MULTIHOST_LAUNCH` decls, groups files by launch spec → matrix JSON |
| `benchmark/tpu/recipe/*_mh.sh` | Per-model multi-host launch recipes (own `MODEL`/`EP_SIZE`/…, exec `launch_server.sh`) |
| `.github/workflows/_tpu-build.yml` | Reusable: build once → output digest |
| `.github/workflows/_tpu-test.yml` | Reusable: render JobSet (suite or group) → wait for Kueue admission → stream logs → gate on JobSet condition |
| `.github/workflows/pr-test-tpu.yml` | `on: pull_request` (path-or-`tpu`-label gated); decide → build → single-host test |
| `.github/workflows/nightly-test-tpu.yml` | `on: schedule` + dispatch; build → single-host + discover → multi-host **matrix** over groups |

## Cluster infra (and what is currently missing)

- Nodepool `ci-tpu7x-2x2x1` (4 nodes, `ci-tpu-pool=pr-gate`).
- ⚠️ **`ci-tpu7x-2x2x2` does not currently exist.** It was present as of 2026-08-09 and is
  gone as of 2026-08-12, so multi-host jobs sit Pending with `0/11 nodes are available:
  11 node(s) didn't match Pod's node affinity/selector` until their deadline. The recreate
  command (workload policy `--type=HIGH_THROUGHPUT --accelerator-topology=2x2x2`, then the
  node pool) is in the header of `kueue-tpu.yaml`. The only other 2x2x2 pools on the cluster
  are `tpu7x-2x2x2-haolin-dev` and `tpu7x-2x2x2-mimo` — do not repurpose either.
- ⚠️ **Kueue is not installed.** Multi-host admission depends on it; the install and
  verification steps are in the header of `kueue-tpu.yaml`. JobSet v0.12.0 is present.
- `system-pool` (e2-standard-2) hosting the **JobSet operator v0.12.0** (the e2-medium
  default-pool is CPU-saturated by GKE daemons).
- Artifact Registry: `us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci`.
- Service account `tpu-ci@sglang-torchtpu.iam.gserviceaccount.com` with
  `container.developer`, `artifactregistry.writer`, `logging.logWriter`.
- WIF pool `github` + provider `github-oidc`, scoped to `sgl-project/sglang-torchtpu`,
  bound to `tpu-ci` via `workloadIdentityUser`.
- Namespace `tpu-ci`; KSA `weights-loader` granted `storage.objectViewer` on
  `gs://sglang-torchtpu-gcs-shared` via Workload Identity (no GSA key).

## Enabling Kueue

Once per cluster, in order. `_tpu-test.yml` has no hardware mutex of its own, so it must
not merge before step 3 reports both queues Active.

**1. Install** (needs cluster-admin). JobSet v0.12.0 is already present.

```bash
kubectl apply --server-side -f \
  https://github.com/kubernetes-sigs/kueue/releases/download/v0.18.5/manifests.yaml
kubectl -n kueue-system rollout status deploy/kueue-controller-manager --timeout=180s
```

**2. Configure** — three settings in `kueue-manager-config`, then
`kubectl -n kueue-system rollout restart deploy/kueue-controller-manager`.

```yaml
integrations:
  frameworks:
    - "jobset.x-k8s.io/jobset"      # without this the queue-name label is IGNORED
manageJobsWithoutQueueName: true    # unlabelled JobSet queues instead of running unmanaged
managedJobsNamespaceSelector:
  matchLabels:
    kubernetes.io/metadata.name: tpu-ci   # NOT optional -- otherwise cluster-wide
waitForPodsReady:                   # evict + requeue a gang that can't fully place
  enable: true
  timeout: 5m
  blockAdmission: true
```

Each guards a distinct silent failure. Without the JobSet framework, Kueue ignores the
label and every JobSet runs with no admission control, logging nothing. Without
`manageJobsWithoutQueueName`, a typo'd or missing label does the same. Without the
namespace selector, that setting goes cluster-wide and Kueue starts suspending workloads in
`haolin-dev` and `mimo`. Without `waitForPodsReady`, an over-stated quota admits a gang the
cluster can't hold and it hangs to `activeDeadlineSeconds` instead of requeueing.

Then confirm nothing outside `tpu-ci` was caught: `kubectl get workloads -A`.

**3. Apply the quotas.**

```bash
kubectl apply -f .github/tpu-ci/kueue-tpu.yaml
kubectl get clusterqueues -o wide        # ACTIVE=True on both
kubectl -n tpu-ci get localqueues
```

**4. The multi-host pool must exist**, or `tpu-multihost` workloads queue until the
admission timeout. Flags verified against `ci-tpu7x-2x2x1` and `tpu7x-2x2x2-haolin-dev`:

```bash
gcloud compute resource-policies create workload-policy ci-tpu7x-2x2x2-wp \
  --type=HIGH_THROUGHPUT --accelerator-topology=2x2x2 --region=us-central1 \
  --project=sglang-torchtpu

gcloud container node-pools create ci-tpu7x-2x2x2 \
  --cluster=tpu-cluster --zone=us-central1-c --project=sglang-torchtpu \
  --machine-type=tpu7x-standard-4t --tpu-topology=2x2x2 \
  --placement-policy=ci-tpu7x-2x2x2-wp \
  --reservation-affinity=specific \
  --reservation=projects/sglang-torchtpu/reservations/ghostfish-u2zcmybntv70q \
  --node-labels=ci-tpu-pool=nightly
```

`--type` must be `HIGH_THROUGHPUT` — `COMPACT` is a different API and `HIGH_AVAILABILITY`
rejects `--accelerator-topology`. `--num-nodes` is inferred from `--tpu-topology`.

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
  from the **ReadOnlyMany** Hyperdisk (now the shared `model-weights-rox`; both hosts mount concurrently),
  rank-0 asserting on real inference.
- **Hyperdisk ML weights** — `model-weights-rox` is a hydrated RWO disk (all models on one 5Ti disk) converted to
  `READ_ONLY_MANY` (GKE ≥1.30.2 auto-converts on first ROX mount; cluster is 1.35.x).
  Lifecycle (hydrate → convert → append-later) is documented in
  `pvc-model-weights-rox.yaml.tmpl`; note ROX conversion is one-way, so adding a model
  goes through a snapshot → new RWO disk → re-convert cycle.

### ⚠️ Model note: Qwen3-0.6B is single-host only

Qwen3-0.6B **OOMs at tp>1** on tpu7x (a bug: ~151G HLO temporaries > 94G HBM) — do not
use it as a multi-host launch model. Multi-host tests use Qwen3-30B-A3B-FP8.

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
