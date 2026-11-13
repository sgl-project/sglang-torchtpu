# Workload-sized cache diagnostic

The actual eight-chip run exp-t8m2fuaixl reached startup precompile and failed:
96.05 GB HLO memory required versus 94.74 GB HBM. Its cache was 586112 tokens
per attention-DP worker, consuming 30.19 GB. This is not evidence of a loader
deadlock or full-model FP8 dequantization.

The next candidate caps `max_total_tokens` at **147456 per worker**, leaving
context length135168, C32, TP16/attention-DP16/EP16, global chunk32768/local2048,
FP8KV and mem_fraction_static0.93 unchanged. The native configurator applies
this user limit directly to local token capacity; parallel arg normalization
divides chunk size, not this cache limit.

For the intended two requests per DP worker, measured input+output is at most
2*(49152+12288+16)=122912 tokens, plus page rounding and warm-branch residue.
147456 is page64 aligned, exceeds configured single-request context135168,
and leaves over24k tokens above that two-request bound. DP assignment and
actual cache hits still require validation; this is not a general C32-full-
context capacity claim.

Expected cache allocation scales to about7.6GB, rather than30.19GB. This may
reduce compiler cache-buffer pressure, but **has not yet been compiled or
measured**. Do not call it a confirmed fix. Log actual capacity/allocation
and startup acceptance in the next run, preserving the fixed benchmark input.
