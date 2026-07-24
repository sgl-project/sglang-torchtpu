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
                 ▼
            GKE tpu-cluster (us-central1-c)
                 ├─ ci-tpu7x-2x2x1  (ci-tpu-pool=pr-gate, 2 nodes) ── single-host (run_suite)
                 └─ ci-tpu7x-2x2x2  (ci-tpu-pool=nightly, 1 slice = 2 hosts) ── multi-host groups
```

Everything runs on the `ghostfish-7vg0iyvvgks52` reservation. The PR and nightly
pools are separate, so PRs and the nightly never contend for hardware.

### Build-once

The image is built **once** (`_tpu-build`) and every run reuses the same digest
(`_tpu-test`), so a nightly with N runs does **one** build, not N.

### Detection = host count → slice (topology.yaml)

`topology.yaml` maps a **host count** to a slice `{topology, pool}`. `render_jobset.py`
renders **one** JobSet template (`jobset-tpu.yaml`) with `completions = hosts`; the
container branches on `NNODES` (from GKE's `TPU_WORKER_HOSTNAMES`):

- **single-host** (`hosts=1`) → a 1-pod JobSet; each test self-launches its own server
  (`popen_launch_server`); `run_suite` runs the suite directly.
- **multi-host** (`hosts≥2`) → an N-pod gang-scheduled JobSet; every host runs the
  launch **group's recipe** (which owns model + config), and rank 0 runs each of the
  group's test files in **live-server mode** (`SGLANG_TPU_USE_LIVE_SERVER=1` + `BASE_URL`),
  so tests assert on real results. Workers serve until the leader tears down, exit 0.

### Multi-host = test-owned launch, grouped by setup

The model+config is **not** bound to a suite — each multi-host test file declares its
own launch in a module-level `MULTIHOST_LAUNCH` dict:

```python
MULTIHOST_LAUNCH = {"hosts": 2, "launch_script": "recipe/qwen3_30b_a3b_fp8_mh.sh",
                    "weights_pvc": "qwen30b-rox"}   # weights_pvc optional
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

## Files

| Path | Purpose |
|------|---------|
| `docker/tpu.Dockerfile` | TPU image: installs `python/pyproject_tpu.toml` from the private torch-tpu registry (token via BuildKit secret) |
| `docker/cloudbuild.tpu.yaml` | Cloud Build config; runs **as `tpu-ci`**, token from the metadata server, pushes to AR |
| `.github/tpu-ci/topology.yaml` | Host count → slice `{topology, pool}` (detection source of truth) |
| `.github/tpu-ci/jobset-tpu.yaml` | **Unified** JobSet template (any host count); rendered by `render_jobset.py` |
| `.github/tpu-ci/render_jobset.py` | Renders the JobSet — single-host suite mode or multi-host group mode |
| `.github/tpu-ci/discover_multihost.py` | Scans `MULTIHOST_LAUNCH` decls, groups files by launch spec → matrix JSON |
| `benchmark/tpu/recipe/*_mh.sh` | Per-model multi-host launch recipes (own `MODEL`/`EP_SIZE`/…, exec `launch_server.sh`) |
| `.github/workflows/_tpu-build.yml` | Reusable: build once → output digest |
| `.github/workflows/_tpu-test.yml` | Reusable: render JobSet (suite or group) → stream logs → gate on JobSet condition |
| `.github/workflows/pr-test-tpu.yml` | `on: pull_request` (path-or-`tpu`-label gated); decide → build → single-host test |
| `.github/workflows/nightly-test-tpu.yml` | `on: schedule` + dispatch; build → single-host + discover → multi-host **matrix** over groups |

## Provisioned infra (already created)

- Nodepools `ci-tpu7x-2x2x1`, `ci-tpu7x-2x2x2` (+ workload policy for the multi-host one).
- `system-pool` (e2-standard-2) hosting the **JobSet operator v0.12.0** (the e2-medium
  default-pool is CPU-saturated by GKE daemons).
- Artifact Registry: `us-central1-docker.pkg.dev/sglang-torchtpu/sglang-torchtpu-ci`.
- Service account `tpu-ci@sglang-torchtpu.iam.gserviceaccount.com` with
  `container.developer`, `artifactregistry.writer`, `logging.logWriter`.
- WIF pool `github` + provider `github-oidc`, scoped to `sgl-project/sglang-torchtpu`,
  bound to `tpu-ci` via `workloadIdentityUser`.
- Namespace `tpu-ci`; KSA `weights-loader` granted `storage.objectViewer` on
  `gs://sglang-torchtpu-gcs-shared` via Workload Identity (no GSA key).

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
  from the **ReadOnlyMany** Hyperdisk (`qwen30b-rox`, both hosts mounting concurrently),
  rank-0 asserting on real inference.
- **Hyperdisk ML weights** — `qwen30b-rox` is a hydrated RWO disk converted to
  `READ_ONLY_MANY` (GKE ≥1.30.2 auto-converts on first ROX mount; cluster is 1.35.x).
  Flow: `hydrate-qwen30b.yaml` (RWO + HF download) → delete RWO PVC (Retain keeps the
  disk) → `pvc-qwen30b-rox.yaml` (static ROX PV/PVC, fill the disk handle).

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
