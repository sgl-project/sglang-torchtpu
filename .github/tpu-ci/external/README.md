# External TPU test suite — `external-nightly-*`

Runs sglang-torchtpu TPU e2e tests on **the external Ironwood (tpu7x) cluster**,
serving model weights from the in-project CI GCS bucket over **gcsfuse**.

> **Status: active.** Runs the `test_tpu_bench_*` benchmark matrix (GPT-OSS-120B and
> Qwen3-Coder-480B x {1-host prefix-cache, 1-host +EAGLE3+KV-offload, 4-host 2P+2D P/D} x
> {agentic, 8k1k} — see `benchmark/tpu/recipe/bench_matrix/`) and the TPU feature
> validation tests: single-host, multi-host, P/D and exec. Groups that share a slice
> serialize within that slice's job.

## Architecture

```
nightly-test-external-tpu.yml
  ├─ build      _tpu-build.yml → image digest (Project A Artifact Registry; external nodes pull it)
  ├─ discover   render_external_jobset.py --list-groups → matrix of one entry PER SLICE
  └─ run (matrix, one job PER SLICE — slices run in parallel)
        _external-tpu-test.yml   [concurrency: external-slice-<nodepool> → cross-run guard]
          auth as CI SA (WIF) → get-credentials on the external cluster over the DNS endpoint
          for each group on this slice (serially, cross-group concurrency = 1):
            render_external_jobset.py → ONE JobSet pinned to the slice's nodepool (no Kueue)
            weights: gcsfuse mount of gs://…/weights/<model>  (tuned file cache)
            apply → stream logs → gate on JobSet Completed/Failed → tear down
```

**No Kueue, so scheduling is done by construction:** discovery round-robins the groups across
their slices and emits **one matrix job per slice** (keyed by nodepool: several suites share
the 2x2x2 slices). A P/D group whose roles each fill a slice gets a job per slice PAIR. Slices run in parallel
(full 4+2 utilization); groups that land on the same slice run **serially inside that one job**,
so two JobSets never race for a slice (no half-placement, no drops). A slice-keyed
`concurrency` group guards the cross-run case. When Kueue is installed on the cluster this all
collapses to a queue-label + gang admission (see the repo's in-project `reconcile_quota.py`).

It is separate from the in-project CI because the external cluster differs structurally:

| | In-project (Project A) | External |
|---|---|---|
| Scheduling | Kueue gang-admits onto `ci-tpu-pool` flavors | **No Kueue** — one job per slice, round-robin |
| Weights | Hyperdisk-ROX PVC at `/weights` | **gcsfuse** mount of the GCS bucket |
| Cluster auth | same-project `get-credentials` | **DNS endpoint**, cross-project as the CI SA |

**Shared with in-project (reused, not copied):** the registration scan
(`discover_multihost._scan`, incl. `disabled` handling) and the path-derived test names
(`test_names.names_from_path`, from #552) are imported by `render_external_jobset.py`, so their
improvements land here automatically. `register_tpu_ci(...)`, `MULTIHOST_LAUNCH`, the
launch/rank-0 run model, and the image are the same too. `tp = hosts × 8` (tpu7x = 4 chips/host
× 2 TensorCores).

## Add an `external-nightly-*` test

1. **Stage the weights** once (you need `objectAdmin` on the bucket). Set
   `TPU_WEIGHTS_BUCKET` to your deployment's bucket name before running:

   ```sh
   gcloud storage rsync -r ./Qwen3-Coder-30B-A3B-Instruct-FP8 \
     "gs://${TPU_WEIGHTS_BUCKET}/weights/Qwen3-Coder-30B-A3B-Instruct-FP8"
   ```

2. **Write the test** under `test/registered/tpu/...` like any in-project multi-host test —
   a `MULTIHOST_LAUNCH` dict + a `register_tpu_ci(...)` marker + a `__main__` entry. The only
   differences: the suite name (`external-nightly-*`) and a `weights` bucket subdir instead of a
   `weights_pvc`:

   ```python
   MULTIHOST_LAUNCH = {
       "launch_script": "recipe/qwen3_coder_30b_a3b_fp8.sh",  # relative to benchmark/tpu/
       "weights": "Qwen3-Coder-30B-A3B-Instruct-FP8",         # subdir under weights/
       # "extra_args": "...",   # optional; overrides weights.default_extra_args
   }
   register_tpu_ci(est_time=1200, suite="external-nightly-2x2x2", nightly=True)

   if __name__ == "__main__":
       sys.exit(pytest.main([__file__, "-v"]))
   ```

   The suite name fixes the slice: `external-nightly-2x2x2` ⇒ 2 hosts / `tp=16`,
   `external-nightly-2x2x4` ⇒ 4 hosts / `tp=32` (see [`external-topology.yaml`](external-topology.yaml)).
   The runner injects `MODEL`, `TP_SIZE`, `EP_SIZE`, `EXTRA_ARGS`; the recipe owns the rest.

### Other launch models

A test declares at most one of these next to its registration; the suite still fixes the
slice. Every pod sees the bucket's `weights/` at `/weights`, the in-project path, so tests
that hard-code `/weights/<model>` run unchanged once the model is staged.

The cluster has **no internet egress**. Pods copy `weights/datasets/gsm8k/test.jsonl` to
`/tmp/test.jsonl` (the GSM8K evals' download cache) and run HF offline against
`weights/hf-hub`, a flat hub cache (`models--<org>--<name>/refs/main` +
`snapshots/<rev>/`), with gpt-oss's harmony vocab read from `weights/datasets/tiktoken`.
A test that pulls a new HF repo id or dataset needs it staged there first.

| Declared in | Suite | What runs |
|---|---|---|
| the test: registration only | `external-nightly-1host` | Single-host tests that launch their own server. Rank 0 of a 2x2x2 slice drops the slice-wide TPU env and runs each file through `run_suite`; rank 1 idles. Files are dealt across the slices, one bundle per slice. |
| `external-topology.yaml`: the suite's `commands`, entry with `pd: {prefill_hosts, decode_hosts}` | `external-nightly-2x2x2` | P/D: `prefill` and `decode` replicated jobs run `cmd` with `CI_PD_ROLE` set (`jobset-external-tpu-pd.yaml`). 1+1 shares one slice; 2+2 takes a slice pair. |
| `external-topology.yaml`: the suite's `commands`, entry without `pd` | `external-nightly-2x2x2` | Every host runs `cmd`; rank 0's exit code gates. |

`commands` entries are wired by hand, as in-project's `feature-validation-tpu.yml` wires
its static jobs: the test file keeps its registration and serving config, and CI owns
how it is run.

## Run it

```sh
gh workflow run "Nightly Test (External TPU)" -f ref=<branch>      # build + discover + run
gh workflow run "Nightly Test (External TPU)" --ref <branch> -f ref=<branch> \
  -f test_filter=<path substring>                                   # only the matching groups

python3 .github/tpu-ci/external/render_external_jobset.py --list-groups  # what would run
```

## Configuration

Non-sensitive structure lives in [`external-topology.yaml`](external-topology.yaml). External-cluster
identifiers are **repo secrets** (masked in logs, out of source) — set once:

| Secret | Value |
|---|---|
| `EXTERNAL_CLUSTER` | external GKE cluster name |
| `EXTERNAL_PROJECT` | external GCP project id |
| `EXTERNAL_NODEPOOL_2X2X2_0..3` | the **4** nodepools backing the `2x2x2` slices |
| `EXTERNAL_NODEPOOL_2X2X4_0..1` | the **2** nodepools backing the `2x2x4` slices |

The cluster provides **4× `2x2x2`** and **2× `2x2x4`** slices. There is no Kueue, so discovery
**round-robins** the launch groups across a shape's slices (`nodepool_vars` in
`external-topology.yaml`) and buckets them into **one job per slice**: up to 4 `2x2x2` + 2
`2x2x4` jobs run in **parallel**, and groups beyond the slice count run **serially inside their
slice's job** (no Pending race, no dropped work). Cross-run overlap on a slice is held by the
slice-keyed `concurrency` group (depth-1 — fine until Kueue).

Discover nodepools with
`kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology`. To add
a slice, append a secret name to that shape's `nodepool_vars`, set the secret, and add a line
to `_external-tpu-test.yml`'s render-step env. A new shape also needs a `suites:` entry.

## Public log boundary

The workflow publishes pod output to Actions logs, log artifacts, and the
configured live-log bucket. Use only release-approved models and public or
synthetic datasets in these suites; do not use production requests or customer
data. Log filtering is best effort and does not make arbitrary test output safe.

Pod status output includes names and phases only. Event summaries omit free-form
messages, and the workflow does not print full JobSet descriptions or node selectors.
Pod logs pass through `sanitize_logs.py` **before** display and before writing the
file uploaded as an Actions artifact or copied to GCS. The filter removes configured
cluster/project/nodepool values, GKE node hostnames, and IPv4/IPv6 addresses while
preserving benchmark metrics. GitHub's console masking alone does not sanitize
uploaded files. Inspect detailed infrastructure diagnostics through authorized
cluster access instead.

This is a targeted infrastructure filter, not a general secret scanner: tests must
still avoid printing credentials or other confidential content. These changes do
not redact existing artifacts, GCS logs, commits, branch names, PRs, or reports.
Review previously published material separately; use secret **names**, not values
or partner-identifying details, in shareable reports.
