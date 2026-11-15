"""Dormant benchmark scheduler-thread acknowledgment wrapper.

Observe already processed typed broadcast controls; NEVER mutate pause state.
No automatic installation or per-rank command polling.
"""
import functools
import json
import os
from pathlib import Path

from benchmark.tpu.glm52_epic.ordered_receiver_adapter import validate_mesh
from benchmark.tpu.glm52_epic.scheduler_barrier import snapshot


def install(scheduler_class, *, directory, run_token, topology):
    original = scheduler_class.ingest_requests
    if getattr(original, '_glm52_ordered_ack', False):
        raise RuntimeError('ordered acknowledgment already installed')
    states = {}
    prefix = f'glm52-control:{run_token}:'

    @functools.wraps(original)
    def ingest(scheduler, *args, **kwargs):
        rank = validate_mesh(topology(scheduler))
        received = original(scheduler, *args, **kwargs)
        key = id(scheduler)
        if key not in states:
            states[key] = (scheduler, dict(sequence=-1, epoch=None, action=None))
        state = states[key][1]
        changed = False
        for request in received:
            rid = getattr(request, 'rid', None)
            if not isinstance(rid, str) or not rid.startswith(prefix):
                continue
            fields = rid[len(prefix):].split(':')
            if len(fields) != 3:
                raise ValueError('invalid broadcast control identity')
            epoch, raw_sequence, action = fields
            sequence = int(raw_sequence)
            if sequence != state['sequence']+1 or not epoch:
                raise ValueError('broadcast control sequence mismatch')
            from sglang.srt.managers.io_struct import ContinueGenerationReqInput, PauseGenerationReqInput
            if action == 'pause':
                if not isinstance(request, PauseGenerationReqInput) or request.mode != 'in_place':
                    raise ValueError('pause identity/type mismatch')
                if state['action'] == 'pause' or scheduler._engine_paused is not True:
                    raise ValueError('pause handler state mismatch')
            elif action == 'resume':
                if not isinstance(request, ContinueGenerationReqInput) or request.torch_empty_cache:
                    raise ValueError('resume identity/type mismatch')
                if state['action'] != 'pause' or epoch != state['epoch'] or scheduler._engine_paused:
                    raise ValueError('resume handler ownership mismatch')
            else:
                raise ValueError('unsupported broadcast action')
            state.update(sequence=sequence, epoch=epoch, action=action)
            changed = True
        if state['action'] == 'pause' or changed:
            # Publish changing queue inventory while paused and initial resume ack.
            # Never perform D2H reads: snapshot accesses Python scheduler metadata.
            record = dict(snapshot(scheduler, rank), run_token=run_token,
                          epoch=state['epoch'], sequence=state['sequence'],
                          action=state['action'], pid=os.getpid())
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            target = root/f'ordered-rank-{rank}.json'
            temp = root/f'.ordered-rank-{rank}-{os.getpid()}.tmp'
            temp.write_text(json.dumps(record)+'\n')
            os.replace(temp, target)
        return received

    ingest._glm52_ordered_ack = True
    scheduler_class.ingest_requests = ingest

    def uninstall():
        if any(state['action'] == 'pause' for _, state in states.values()):
            raise RuntimeError('cannot uninstall while pause owned')
        if scheduler_class.ingest_requests is not ingest:
            raise RuntimeError('scheduler ingest wrapper replaced')
        scheduler_class.ingest_requests = original
        states.clear()

    return uninstall
