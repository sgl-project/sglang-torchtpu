# External TPU test suite — `external-nightly-*`

Runs sglang-torchtpu TPU e2e tests on **the external Ironwood (tpu7x) cluster**,
serving model weights from the in-project CI GCS bucket over **gcsfuse**.

> **Status: scaffold.** No `external-nightly-*` test is registered yet, so the nightly workflow
> discovers zero groups and skips its run job (stays green). Add the first test as below.

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

**No Kueue, so scheduling is done by construction:** discovery round-robins each shape's
launch groups across its slices and emits **one matrix job per slice**. Slices run in parallel
(full 4+2 utilization); groups that land on the same slice run **serially inside that one job**,
so two JobSets never race for a slice (no half-placement, no drops). A slice-keyed
`concurrency` group guards the cross-run case. When Kueue is installed on the cluster this all
collapses to a queue-label + gang admission (see the repo's in-project `kueue-tpu.yaml`).

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

1. **Stage the weights** once (you need `objectAdmin` on the bucket):

   ```sh
   gcloud storage rsync -r ./Qwen3-Coder-30B-A3B-Instruct-FP8 \
     gs://sglang-torchtpu-ci-artifacts/weights/Qwen3-Coder-30B-A3B-Instruct-FP8
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

## Run it

```sh
gh workflow run "Nightly Test (External TPU)" -f ref=<branch>      # build + discover + run

python3 .github/tpu-ci/external/render_external_jobset.py --list-groups  # what would run
```

## Configuration

Non-sensitive structure lives in [`external-topology.yaml`](external-topology.yaml). Partner-internal
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
