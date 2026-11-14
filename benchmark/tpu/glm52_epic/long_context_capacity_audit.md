# Frozen workload capacity audit (no submission)

Frozen client constants: prefix49152, extend12288, output16, concurrency32.
Measured input length61440; required single-request context including output
is at least61456 (round upward for scheduler/page safety). Smoke context4096
and pool8192 cannot serve it.

For a balanced16-attention-DP assignment, two unique prefixes/requests reside
on each worker. Two live measured sequences require at least122912 token
slots/worker before page rounding (2*(49152+12288+16)). At page128, a simple
per-sequence ceiling gives2*61568=123136 slots. This is a necessary workload
bound, not an allocator-fit guarantee. Warm branch/radix leftovers, reserved
pages, scheduler accounting and actual rank skew may require more.

Historical prefill candidate max_total_tokens147456 exceeds this elementary
bound, but actual allocation is capped by profiled device capacity:
kv_cache_configurator._apply_token_constraints lines2675–2690 takes
min(profiled_capacity,user_limit); it does not promise to allocate the user
limit. max_total_tokens is applied to the worker token-capacity path, NOT a
pool automatically divided by attentionDP16. Never divide147456 by16 here.
max_running_requests32 is a separate global concurrency constraint and
request-pool paths do divide by attn_dp_size (e.g. lines2866 onward).

A rank can only reuse its locally held prefix. A deterministic two-per-rank
assignment is not yet proven for the reference admission. If warm and measured
routing skew or differ, the above balanced bound understates requirements and
cache-hit assumptions fail. Existing direct routed_dp_rank may support a new
controlled diagnostic, but does not retroactively prove reference equivalence.

Local chunk2048 requires global chunk32768 under this configuration, unlike
the accepted global4096/local256 smoke. Restoring it plus context/pool changes
alters compile metadata, request-table width and sparse visits. Historical
page64 candidate also differs from validated page128 selector32/pps2. A future
candidate must explicitly resolve these differences, inspect real profiled
capacity and kernel dispatch limits, and retain a fail-closed admission gate.

No new config or remote experiment was created by this arithmetic audit.
