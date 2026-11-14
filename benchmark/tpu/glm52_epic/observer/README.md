# Rank-local Torch completion observer

`observer.py` is opt-in only (`GLM52_EPIC_OBSERVER_DIR` absolute path). It
records the existing `copy_done.synchronize()` boundary for ordinary overlap
EXTEND batches and writes one JSONL file per host/PID. It snapshots RID, local
attention-DP rank, prefix and extend tokens before `run_batch` mutates state.

The observer is intentionally **not** a matched JAX metric: it does not prove
DP cache assignment, global admission, batch completeness, or cross-rank clock
alignment. Missing/failed existing waits are emitted as unavailable rather than
substituted with method-entry/exit time. It imports no accelerator at startup.

Validation: `python3 benchmark/tpu/glm52_epic/observer/selftest.py` exercises
metadata rejection, existing-wait timing, unavailable paths, import-hook
patching and sink naming. TPU/server integration remains unrun.
