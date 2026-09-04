## The workload

A multi-turn agentic coding session: each round feeds back a tool observation and
the whole history is re-sent, so context grows monotonically and almost every round
is a near-total prefix-cache hit.

| | default | knob |
|---|---|---|
| new input per round | 4096 tokens, vocabulary-sampled | `TURN_INPUT_TOKENS` |
| output per round | 1024 tokens (`ignore_eos`, so exactly) | `TURN_OUTPUT_TOKENS` |
| context growth | 5130 tok/round (4096 + 1024 + measured template overhead) | — |
| normal session | 3 rounds → ~14.4k final prompt | `TARGET_CONTEXT` (16384) |
| long session | 51 rounds → ~262k context | `LONG_TARGET_CONTEXT` (262144) |
| sweep | concurrency `{1,16,32}` × long sessions `{0,1,2,4}` | `SWEEP_BS` / `SWEEP_LONG` |

16384 is the Mooncake corpus's **p90** total prompt (16.8k, below): a normal session
is the heaviest ordinary agentic request rather than an outlier. Round counts are
`target / growth` rounded to the nearest whole round, so 16384 is 3 rounds ending
6% under the target; the generator prints what a session actually reaches and
records it in the trace metadata. Only the long sessions carry long-context work.

Legs with more long sessions than concurrency slots are skipped and recorded as
such. The two sweep axes are independent: long sessions are emitted first in the
trace, so exactly `NUM_LONG` of them are in flight for the whole leg.

For calibration, the same shape measured over the public Mooncake FAST'25
`toolagent` corpus (23,608 requests): total prompt p50 6.3k / p90 16.8k / max 126k;
new tokens per round p50 689 / p90 9,990; output p50 30 / mean 182; prefix-hit p50
82%. So 4k/round sits between that corpus's p50 and p90, and 1k output/round is ~5x
its mean.

## Choices that affect the numbers

Read these before comparing two runs. The full *why* for each sits next to the code.

- **`pallas_tpu_rpa_v3` is forced, not chosen.** It is the only TPU backend with the
  HiCache reload-commit hook, so `--hicache-io-backend torch_tpu` rejects anything
  else. The faster `pallas_tpu_rpa_batched` is unavailable while KV offload is in
  the contract.
- **Server context length is always 1M**, independent of the workload target, so no
  two results can differ by a server setting. This keeps YaRN on.
- **`--chunked-prefill-size` is pre-multiplied by `dp_size`**, because sglang divides
  it by `dp_size` under DP attention. The contract value is per rank.
- **`--warmup-requests 0` on every leg.** bench_serving warms up with
  `input_requests[0]`, and long sessions come first — it would replay an entire long
  conversation before the timer started. Compile-bucket warmup happens once per
  server instead.
- **One server serves every leg**, sized for the largest concurrency in the sweep;
  the client's semaphore is what varies. Reloading 480 GB per leg would dominate the
  run.
- **The EAGLE3 multi-host CI launch** uses the matching Qwen3-Coder draft, with
  3 draft steps, top-k 1, and 4 draft tokens. It is a separate test and launch
  group from the original non-spec agentic benchmark. The generic recipe stays
  non-speculative unless `DRAFT_MODEL` is set, so existing hand-run and P/D
  comparisons do not silently change configuration. Its dashboard series id is
  `perf-qwen3-coder-480b-agentic-eagle3`, separate from historical non-spec runs.
- **Profiles time-box the long sessions only** — the normal sessions keep their round
  granularity in every profile:

  | `PROFILE` | long sessions ramp at | use |
  |---|---|---|
  | `full` (default) | `TURN_INPUT_TOKENS` (~51 rounds) | the letter of the contract |
  | `nightly` | 16384 tok/round (~15 rounds) | manual overnight run |
  | `ci` | 16384, sweep `{1,16}`×`{0,1}`, 64k long target | what the GKE nightly runs |
  | `smoke` | 64k long target, few legs | plumbing check |

  A time-boxed ramp is flagged in the trace metadata, in the leg meta, and in
  `summary.md` — it can never be mistaken for a contract-shaped run.
- **`MEM_FRACTION_STATIC=0.85` and `HICACHE_RATIO=3` are estimates**, never run
  against the real weights. Lower the first if the server OOMs at launch.

## Reading a result

`legs[].metrics` is `sglang.benchmark.serving`'s own record with **keys unchanged**,
so an existing renderer works on it as-is. Values this harness computes live in
`legs[].derived`. Two traps:

- `metrics.total_input_tokens` / `input_throughput` / `total_throughput` are **0** —
  bench_serving cannot account input tokens for a multi-turn replay. Use
  `derived.prompt_throughput`, taken from the server's `prompt_tokens_total` counter.
- `metrics.cache_report.total_prompt_tokens` is rounds × the first round's estimate,
  not a prompt volume. Its device/host/storage split is meaningful.

Check these signals before trusting a number; `metrics_probe.py` raises them as
`warnings`, and the multi-host CI test gates the prefix-cache and speculative
signals:

- `feature_derived.prefix_cache_hit_rate` should be close to what the round count
  and the DP count together allow. Each DP rank owns its own radix tree and dispatch
  is round_robin, so round *r* can only reuse round *r−dp*: a session of R rounds
  reaches `(R−dp)(R−dp+1)/(R(R−1))` of the single-cache ideal, and **nothing at all**
  when R ≤ dp. At the `ci` profile's dp=2 that is 51% for a 6-round session, against
  the 76% one shared cache would give. Much lower than the reachable figure means the
  cache is not being reused, which invalidates every number in the leg. The CI test
  derives its floor the same way (60% of reachable), reading `dp_size` back off the
  run — see `_reachable_prefix_hit_rate` in the test.
- `feature_derived.kv_offload_written_tokens` / `..._reloaded_tokens` should both be
  non-zero on the legs with long sessions. Written-but-not-reloaded means prefixes are evicted and
  recomputed; zero written means the working set fit in HBM and the leg did not
  exercise offload at all.
- `feature_derived.spec_verify_calls` must increase in every leg, proving the
  requests traversed speculative verification. The paired
  `spec_accept_length_by_scheduler` reads every DP scheduler rather than only rank
  zero. The per-leg cache flush also resets these scheduler metrics, so the values
  apply to that leg rather than the whole server run. Each must be strictly greater
  than `MIN_SPEC_ACCEPT_LENGTH` (1.0 by default), proving that at least one draft
  token was accepted. This intentionally starts at the non-fallback boundary
  because the trace uses vocabulary-sampled inputs; the 1.90 GSM8K floor is not
  transferable to this workload.
