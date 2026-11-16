# GLM5.2 channelwise FP8 — minimal draft validation

This draft is for work decomposition, **not merge-ready model support**.
No observer, dump capture, scheduler monkeypatch, custom admission, or historical
failure archive is included. Full experiment history remains on
`epic/glm52-channelwise-port-20261113` at `ca8dbe8e62`.

## Review entry points

- `KERNEL_ROUTE_COMPARISON.md`: exact private-JAX versus Torch source routes.
- `WORK_BREAKDOWN.md`: retained production scope and independent follow-up tasks.
- `torch_staged_loading_candidate.json`: proven compiled candidate configuration.

## Minimal execution path

`peer_bootstrap_8chip.sh` prepares the pinned runtime and two-host discovery;
`torch_8chip_runner.sh` validates the source/runtime/topology, performs once-per-host
staged reads, launches the native server, and verifies a 12288-input/16-output request.
The only admitted mode is `GLM52_BASELINE_MODE=compiled-long-e2e`.

Host preparation uses `prepare_host_checkpoint.py`, `plan_expert_reads.py`, and
`verify_expert_shard.py` for exact header/range interpretation. Eight workers use
4MiB buffers; preparation does not guarantee page-cache residency. Physical TPU
coordinates, not Falcon pod order, determine the host expert union.
`validate_staged_cli.py` checks the native argument contract before launch.

Optional checks on the same native server:

- `GLM52_TEXT_E2E=1 GLM52_TEXT_CHAT_TEMPLATE=1`: `text_e2e_client.py` loads the
  checkpoint-authored nonthinking chat template, checks four question cases and
  sixteen native routed requests, including returned `dp_rank` receipts.
- `GLM52_NATIVE_LONG_PERF=1`: `native_long_client.py` performs one warmup, three
  sequential repeats, and sixteen routed cold-prefix 12288-input/16-output requests.
  It records actual cache hits and host HTTP timing, not kernel timing or matched
  JAX cadence. This is a synthetic workload, not an accuracy test.

Both scripts are benchmark tooling, not automatic CI or a supported API. Caller
must provide the pinned source/wheel/bootstrap identities and existing Falcon
mounts; bootstrap validates these inputs rather than inventing image provenance.
No new experiment manifest is included; consult the original pinned experiment
when preparing a fresh allocation. Run identically on both hosts and bound the
whole allocation; existing timeout and failure cleanup remain mandatory.

## Original pinned hardware results

Topology: 8 physical chips, 2 hosts, TP16/attention-DP16/EP16.

- `exp-k0nnadpb6q`, source `552e93ab66`: both hosts SUCCEEDED, 20/20 chat checks,
  all sixteen native DP receipts verified. Replies include Paris, 巴黎, 42,
  and CEDAR-7284. Raw completion without chat formatting was not equivalent.
- `exp-r188nvdf96`, source `2160012cc6`: sixteen cold routed long requests
  succeeded in 19.3465 host seconds; all cache hits zero. Native logs show six
  2048-token chunks per DP, 196608 input tokens in total.
- Three hot repeats took 1.995/1.891/1.880 seconds with 12224 cached tokens.
  Hot output sequences agreed with each other but differed from the cold
  synthetic-token output: numerical consistency remains unresolved.

These original results do not validate the rebased/cleaned head. No new TPU run
was performed for this cleanup. Not a matched reference C32/active16/49152-prefix/
12-round benchmark, not full accuracy certification, and no JAX speedup claimed.
