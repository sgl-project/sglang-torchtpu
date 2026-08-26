# Meta (Project B) TPU test suite — `meta-nightly-*`

Runs sglang-torchtpu TPU e2e tests on **Meta's external Ironwood (tpu7x) cluster** (Project B),
serving model weights from the in-project CI GCS bucket over **gcsfuse**.

> **Status: scaffold.** No `meta-nightly-*` test is registered yet, so the nightly workflow
> discovers zero groups and skips its run job (stays green). Add the first test as below.

## Architecture

```
nightly-test-meta-tpu.yml
  ├─ build      _tpu-build.yml → image digest (Project A Artifact Registry; Meta nodes pull it)
  ├─ discover   render_meta_jobset.py --list-groups → JSON matrix of meta-nightly-* groups
  └─ run (matrix, one per group)
        _meta-tpu-test.yml
          auth as CI SA (WIF) → get-credentials on Meta's cluster over the DNS endpoint
          render_meta_jobset.py → ONE JobSet, pinned to the slice's nodepool (no Kueue)
          weights: gcsfuse mount of gs://…/weights/<model>  (tuned file cache)
          apply → stream logs → gate on JobSet Completed/Failed
```

It is separate from the in-project CI because Meta's cluster differs structurally:

| | In-project (Project A) | Meta (Project B) |
|---|---|---|
| Scheduling | Kueue admits onto `ci-tpu-pool` flavors | **No Kueue** — JobSet pins to one nodepool |
| Weights | Hyperdisk-ROX PVC at `/weights` | **gcsfuse** mount of the GCS bucket |
| Cluster auth | same-project `get-credentials` | **DNS endpoint**, cross-project as the CI SA |

Everything else — `register_tpu_ci(...)`, `MULTIHOST_LAUNCH`, the launch/rank-0 run model,
the image — matches the in-project multi-host tests. `tp = hosts × 8` (tpu7x = 4 chips/host ×
2 TensorCores).

## Add a `meta-nightly-*` test

1. **Stage the weights** once (you need `objectAdmin` on the bucket):

   ```sh
   gcloud storage rsync -r ./Qwen3-Coder-30B-A3B-Instruct-FP8 \
     gs://sglang-torchtpu-ci-artifacts/weights/Qwen3-Coder-30B-A3B-Instruct-FP8
   ```

2. **Write the test** under `test/registered/tpu/...` like any in-project multi-host test —
   a `MULTIHOST_LAUNCH` dict + a `register_tpu_ci(...)` marker + a `__main__` entry. The only
   differences: the suite name (`meta-nightly-*`) and a `weights` bucket subdir instead of a
   `weights_pvc`:

   ```python
   MULTIHOST_LAUNCH = {
       "launch_script": "recipe/qwen3_coder_30b_a3b_fp8.sh",  # relative to benchmark/tpu/
       "weights": "Qwen3-Coder-30B-A3B-Instruct-FP8",         # subdir under weights/
       # "extra_args": "...",   # optional; overrides weights.default_extra_args
   }
   register_tpu_ci(est_time=1200, suite="meta-nightly-2x2x2", nightly=True)

   if __name__ == "__main__":
       sys.exit(pytest.main([__file__, "-v"]))
   ```

   The suite name fixes the slice: `meta-nightly-2x2x2` ⇒ 2 hosts / `tp=16`,
   `meta-nightly-2x2x4` ⇒ 4 hosts / `tp=32` (see [`meta-topology.yaml`](meta-topology.yaml)).
   The runner injects `MODEL`, `TP_SIZE`, `EP_SIZE`, `EXTRA_ARGS`; the recipe owns the rest.

## Run it

```sh
gh workflow run "Nightly Test (Meta TPU)" -f ref=<branch>      # build + discover + run

python3 .github/tpu-ci/meta/render_meta_jobset.py --list-groups  # what would run
```

## Configuration

Non-sensitive structure lives in [`meta-topology.yaml`](meta-topology.yaml). Partner-internal
identifiers are **repo secrets** (masked in logs, out of source) — set once:

| Secret | Value |
|---|---|
| `META_CLUSTER` | Meta GKE cluster name |
| `META_PROJECT` | Meta GCP project id |
| `META_NODEPOOL_2X2X2` | nodepool backing the `2x2x2` slice |
| `META_NODEPOOL_2X2X4` | nodepool backing the `2x2x4` slice |

Discover nodepools with
`kubectl get nodes -L cloud.google.com/gke-nodepool,cloud.google.com/gke-tpu-topology`. To add
a new slice shape, add a `suites:` entry (with its `nodepool_var`) + the matching secret + a
line in `_meta-tpu-test.yml`'s render-step env.

**Access (one-time, Meta-side):** grant the CI SA cluster RBAC / `container.developer` + DNS
endpoint access on the Meta cluster, and grant the pod KSA (`default/default`)
`storage.objectViewer` on the weights bucket (read is over Private Google Access — no egress
rule needed).

## Notes

- **gcsfuse cache** (`meta-topology.yaml` `mount_options`) is load-bearing:
  `cache-file-for-range-read` turns safetensors' many small seeky reads into one whole-file
  sequential download into a bounded, node-local cache — don't remove it.
- **Known limitation:** MoE models currently hit a TPU Mosaic kernel halt (`gmm_v2_fused_rs`)
  during compile — a serving/kernel issue independent of this harness; resolve with the kernel
  owners before gating on a MoE model.
