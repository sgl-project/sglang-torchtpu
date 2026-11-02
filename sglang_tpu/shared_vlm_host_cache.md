# Shared VLM CPU cache on one host

TP1/PP1 engines can share completed vision embeddings through one host-memory directory. Their TPU embedding caches and language-model KV caches remain private. This stores encoder outputs; it does not share KV or route requests.

Set the same values on every participating engine:

```bash
export SGLANG_VLM_SHARED_HOST_TIER_DIR=/dev/shm/wm-vlm-cache
export SGLANG_VLM_SHARED_HOST_TIER_NAMESPACE=checkpoint-preprocess-encoder-version
export SGLANG_VLM_HOST_TIER_MB=65536
export SGLANG_VLM_HOST_TIER_STATS_S=60
```

`HOST_TIER_MB` is the **total shared payload budget**, including active writes and leased entries. Eight private 8192 MiB caches can be replaced with one shared 65536 MiB cache. Leave host memory for weights, preprocessing and other serving allocations. Metadata adds a small amount beyond the payload budget.

Use a namespace that identifies the checkpoint, pixel preprocessing and encoder code/configuration. Change it whenever those change. Keys combine this namespace, the existing media content hash and expected token count. All participants must see the same directory and support local `flock`, SQLite and mmap semantics. A private mount in each container will not share entries; this implementation supplies no network transport.

On a device-cache miss, the engine maps a completed entry and uploads its physical buffers. A reader lease protects the source until a recorded device event completes. Misses encode normally and publish only after device-to-host copying and shared CPU copying finish. Concurrent first misses can still encode the same image more than once; completed entries are deduplicated. Full physical buffers, BF16 bits and valid-span metadata are preserved.

Eviction uses one shared LRU and skips active readers/writers. When all space is leased, a write is declined and the engine serves its freshly encoded result. Dead writers are recovered without exposing partial bytes. Unexpected storage/upload errors are reported to the scheduler.

For focused correctness checks, set `SGLANG_VLM_HOST_TIER_VERIFY=1`. A host hit also encodes the image fresh and checks valid rows, full physical buffers and uploaded bytes. Verification adds encoder work and device synchronization; disable it for performance measurements. Counters include `shared_put_*`, `shared_lookup_ns`, `shared_h2d_complete_ns` and `shared_upload_verify_*`. Completion latency is host wall time from upload submission through the completion worker, including queueing; it is not isolated device DMA duration.

Stop every participating engine before clearing the directory. Do not unlink active payloads. Restarting engines retains entries unless the directory is cleared. Without `SHARED_HOST_TIER_DIR`, the existing private host tier remains in use.
