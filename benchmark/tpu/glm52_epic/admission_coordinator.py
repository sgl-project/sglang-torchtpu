"""Bounded benchmark admission waits; never auto-resume on error.

Caller must catch AdmissionAbort and tear down BOTH server hosts via the
existing runner cleanup. This module does not kill processes or claim recovery.
"""
import json
from pathlib import Path
import time


class AdmissionAbort(RuntimeError):
    pass


def wait_inventory(directory, *, expected_routes, run_token, epoch, sequence,
                   timeout_s=60, poll_s=0.1):
    from benchmark.tpu.glm52_epic.admission_inventory import validate_inventory
    if timeout_s <= 0 or poll_s <= 0:
        raise ValueError('positive bounded timeout/poll required')
    root=Path(directory)
    deadline=time.monotonic()+timeout_s
    last={'reasons':['no complete snapshot set']}
    while True:
        errors=[root/'ordered-error.json']+[root/f'ordered-rank-{r}-error.json' for r in range(16)]
        if any(path.exists() for path in errors):
            raise AdmissionAbort('control/ack terminal error; whole-run cleanup required')
        snapshots=[]
        try:
            snapshots=[json.loads((root/f'ordered-rank-{r}.json').read_text()) for r in range(16)]
        except (OSError,json.JSONDecodeError):
            pass
        if len(snapshots)==16:
            last=validate_inventory(expected_routes,snapshots,run_token=run_token,
                                    epoch=epoch,sequence=sequence)
            if last['ready']:
                return snapshots
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise AdmissionAbort(f'admission timeout; whole-run cleanup required: {last["reasons"]}')
        time.sleep(min(remaining,poll_s))
