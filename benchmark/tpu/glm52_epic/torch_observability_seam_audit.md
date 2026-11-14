# Torch matched-admission/completion seams (read-only audit)

Reference-side evidence now confirms measured mapping index%16 and exact32
waiting RIDs before scheduler resume (jax_reference_admission_verified.json).
Torch direct routing may reproduce mapping, but existing public pause cannot
reproduce that admission state:

- TokenizerManager.pause_generation sets is_pause=True before scheduler dispatch
  (2318–2322); generate waits on is_pause_cond before scheduler submission
  (995–996). New requests stay in tokenizer, not scheduler waiting_queue.
- continue clears tokenizer gate and sends scheduler continue (2333–2337).
  There is no observed acknowledgment that all32 requests reached paused
  schedulers before resume. HTTP response order is not that acknowledgment.
- Scheduler.get_internal_state delegates to SchedulerInternalState (5653–5654).
  Its contributor list (internal_state.py:48–67) reports config/memory/etc,
  no exact waiting RID list or paused-state queue snapshot.
- Existing load_snapshot counters are not an exact request inventory; identical
  counts can conceal a missing or duplicate RID.

## Timing seam

Scheduler.process_batch_result (5179 onward) delegates to batch result processor.
That processor synchronizes result.copy_done (prefill path near270–272), then
consumes host results. An optional benchmark-only AROUND hook after the
prefill processor could record host completion with rank, stable batch ID,
RID, pre-completion prefix and per-request extend counts. Must snapshot
metadata BEFORE processing mutates requests, timestamp AFTER existing sync,
and never add another device fence. Scheduler method-entry timing is too early;
whole process_batch_result exit includes unrelated publisher/cache bookkeeping.

## Minimal mechanism decision, not implementation

Do not change generic pause semantics or monkeypatch scheduler core merely to
force this comparison. Benchmark opt-in process-scoped wrappers installed only
by the experiment entrypoint could expose exact paused queue readback and
completion records, but still need a scheduler-only pause transport and an
acknowledged multi-rank barrier. Any TPU plugin permanent hook would be tracked
migration debt and needs the standard Why/How/upstream/removal header. No new
core branches/imports or permanent hooks were added by this audit.

A local 16-rank simulation now exercises the opt-in file barrier plus the
exact32-RID inventory validator: all ranks pause, publish snapshots, an
adversarial running-RID mutation is rejected, and all ranks resume. This is
only a state-machine test; it does not prove shared-filesystem visibility,
atomic multi-rank snapshot timing, scheduler construction hooks, or TPU
behavior.

Until the actual transport/barrier and rank completion aggregation are proven,
keep torch_cache_hit_client.py matched gate fail-closed. A long-context HTTP
diagnostic can be labeled UNMATCHED, but cannot close the cadence objective.
