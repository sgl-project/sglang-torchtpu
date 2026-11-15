# Native-only E2E and performance acceptance

No dump, new scheduler observer, admission monkeypatch, or device-timer instrumentation.

## Preconditions

Finish current real-text acceptance before scheduling another hardware allocation.
Proven anchor exp-19jk2vguvj: compiled 12288 input /16 output synthetic request passed.
That is functional evidence, not accuracy or C16 throughput.

## Existing interfaces

- `GET /server_info`: resolved configuration and per-DP internal states, including
  `effective_max_running_requests_per_dp`. Preserve response before measurement.
- Sixteen independent `/generate` requests with `routed_dp_rank=0..15`, unique RID.
  Verify returned `meta_info.dp_rank`, request identity, token accounting and no abort.
  Destinations alone do not establish simultaneous active C16.
- `--log-level info --decode-log-interval 1`: native DP-prefixed batch logs.
  Prefill logs each processed prefill batch. `#new-token=2048` is evidence of
  full chunk work; configured chunk2048 alone is only a ceiling.
  `#new-seq:1,#running-req:0` is valid first prefill: running count excludes new arrivals.
  Decode `#running-req:1` establishes that rank's occupancy at that iteration.
- `--enable-metrics --enable-request-time-stats-logging`: native request timing.
  Optional request metadata logging: `--log-requests --log-requests-level 0
  --log-requests-format json`; avoid default payload logging.

## Timing and scope

Completed responses always include `e2e_latency`; metrics enables
`first_token_latency`, `decode_throughput`, `queue_time`, `forward_entry_time`,
`prefill_finished_time`. Host prefill interval is prefill_finished_time minus
forward_entry_time; host decode interval is e2e_latency minus first_token_latency.
TPOT is decode interval/(completion_tokens-1), for completion_tokens>1.
These include host/scheduling/collective effects, not accelerator-only time.
Never subtract perf_counter timestamps from different hosts. Batch log throughput
is inter-log wall-time throughput, not kernel timing; first log includes idle time.
Do not enable CUDA-event-backed device timers on TPU.

## Incremental execution

1. Once real-text task is resolved, one warm request then three sequential repeats
   of an already-proven tokenized shape, fixed outputs16, bounded timeout600s/request.
2. Separate compile/warmup from measured repeats; save raw responses and server_info.
3. Native routed16 functional cohort; then long routed16 throughput cohort only
   after checking KV capacity (16 unique49152 prefixes need distributed perDP caches,
   not a single rank's cap) and output correctness. Stop on first execution failure.
4. For matched reference, preserve frozen data/cache warm sequence and 32 requests
   with actual active16. Native logs must support local2048/global32768 and the
   required12 rounds/393216 tokens; if they cannot, report unmatched throughput,
   not a matched JAX ratio. Do not add custom synchronized admission just to force it.

Original JAX FP8-Q/PV settings differ from audited Torch fallback semantics;
label native-mode comparison and do not claim equal numerical precision without evidence.
