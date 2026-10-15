# Shared TPU startup compilation

With `--allow-spmd-safe`, an installed policy whose `spmd_safe` declaration is
true shares each queued specialization across the full TP group. One
`SharedCompiler` belongs to the outer `collect_warmups` queue. Each new full
`callsite:specialization_key` gets owner `len(submissions) % world_size`, continuing
across policy boundaries. Duplicate keys reuse their owner. There are no generated
ownership plans.

The owner traces and compiles; peers enqueue local input bindings without tracing.
After all policies enqueue, ranks agree on the ordered keys and owners, validate
the owner's executable interface against their local tensor bindings and physical
layouts, then broadcast serialized executables in the same order. Owners prepare
artifacts concurrently. Every rank executes using its own weights, KV buffers,
request tensors and rank input. Pending calls snapshot reused input containers and
retain their original forward/inference context.

Flags are exchanged before branching into verification, sharing or local paths.
Different runtime/device mappings cause a coordinated local fallback. Tracing,
binding, compilation and loading failures are propagated to peers. Callback-bearing
and implicit-RNG graphs are unsupported by sharing and fail explicitly.

Unmarked policies and specializations outside the startup queue compile locally.
This change introduces no serving-time collectives or persistent executable cache.
`VERIFY_SPMD_SAFE=1` bypasses sharing and independently checks every rank's graphs.

The native `test/manual/tpu/test_shared_async_compile.py` workload runs under
`torchrun` on a full mesh. It checks round-robin ownership across policy boundaries
and wraparound, duplicate specializations, owner-only native compilation, reused
input containers, rank-local weights/KV state and collective results.
