# DP routing seam audit after baseline acceptance

No runtime/core change or accelerator submission performed.

## Existing direct routing is a useful seam, not a matched-admission proof

Current head provides GenerateReqInput.routed_dp_rank (legacy
 data_parallel_rank normalizes into it). TokenizerManager validates against
elastic_worker_count (initialized from get_parallel().num_dp_ranks) and
forwards it into tokenized generation requests. Parallel field definition
states num_dp_ranks=dp_size*attn_dp_size, hence16 for accepted TP16/attnDP16/DP1.
DataParallelController.maybe_external_dp_rank_routing validates active rank,
sends directly to that worker and bypasses the round-robin counter.
round_robin_scheduler checks this seam before normal selection.

Source anchors: io_struct.py:300–304,415–425;
tokenizer_manager.py:539,972–981,1671;
arg_groups/fields/parallel.py:308–313;
data_parallel_controller.py:732–761.

This could make a *new* warm/measured workload's placement deterministic by
pinning corresponding requests to the same rank, rather than assuming two
round-robin passes align. It is not proof that the frozen JAX reference used
that same placement or simultaneous admission. Batched input expands a single
scalar route field; per-request routing would need individual payloads grouped
by target rank, which can change timing/admission semantics.

## Remaining gates before touching the client

- Read frozen reference request-to-rank/cache admission evidence; do not invent
  mapping from request order alone.
- Verify direct-route behavior on actual multi-host serving and observe rank
  identity/cache-hit lengths, not merely successful HTTP return.
- Need scheduler waiting RID inventory and a real dispatch barrier for cadence
  equivalence. Tokenizer pause waits before dispatch, not after scheduler queue.
- Native completion timing metric and precision still require independent work.
- Keep existing client's fail-closed matched gate unchanged until these hold.

Direct routing alone is therefore a concrete possible tool for a later scoped
diagnostic, NOT permission to relabel HTTP timing as matched JAX cadence.
