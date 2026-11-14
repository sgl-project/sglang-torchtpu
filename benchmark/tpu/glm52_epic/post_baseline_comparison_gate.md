# Post-baseline comparison gate (planning only)

Current model smoke exp-48q9eb5nxn is pending. Do not expand the running job
or submit a performance experiment before baseline request/decode acceptance.

## Existing reference and mismatches

The durable JAX result records experiment exp-266i34jcm1, reference source
3695225808f0293cae422ea7adf55eddacadd43d, prefix49152 + extend12288,
DP16 and local chunk2048. Its metric is consecutive host-result completion
intervals, excluding first/final chunks: eight selected samples,
median1242.394121ms, derived aggregate32768/interval=26374.8833tokens/s.
The result itself labels acceptance parser-only, not independent workload or
hardware validation; keep this provenance limitation visible. Reference uses
FP8 KV, fp8_q=true and fp8_pv=true, not an asserted Torch numerical match.

Current Torch compile smoke has context4096, local chunk256, cache8192,
512 input IDs and two output IDs. It cannot hold the reference prefix, does
not measure the same chunk cadence and is NOT a performance denominator.

## Ordered next gates after smoke acceptance

1. Inspect exact full-model stage evidence, output IDs and failure-cleanup
   boundaries; retain memory samples and native compiler diagnostics.
2. Review long-context/cache capacity and compile shapes as a separate change.
   Merely adding frozen long requests to the small server cannot work.
3. Reuse frozen input hashes, request order and tokenizer-independent IDs.
   Audit whether equivalent DP assignment/cache-hit admission is observable.
4. Current torch_cache_hit_client.py fails closed for matched admission.
   Its explicit override produces HTTPmetrics_NOT_JAXcadence and must remain
   UNMATCHED: pause/queue/resume is not a proved scheduler admission barrier.
5. Capture rank-tagged completion events under the same timing definition,
   exclude compile/warmup, first/final chunks, and select the same request IDs.
6. Audit precision and output consistency before any performance ratio.

No scheduler instrumentation/core changes, long-context submission or timing
claim is authorized by this planning note. Historical exit137 remains a
separate risk when restoring larger cache/context/compile shapes.
