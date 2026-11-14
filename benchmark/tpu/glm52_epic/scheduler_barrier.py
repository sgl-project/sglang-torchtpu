"""Explicit benchmark-only scheduler-thread file barrier; never auto-installed.

Call tick AFTER scheduler.ingest_requests on every participating scheduler.
Requires PP1/attentionTP1; caller owns installation, shared directory, unique
run token, idle/exclusive server and waiting for all rank acknowledgments.
No tokenizer gate, threads, device imports, collectives or device fences.
"""
import json
from pathlib import Path


def publish_command(directory, *, run_token, sequence, epoch, action):
    """Atomically publish one immutable command for all scheduler processes."""
    if (type(run_token) is not str or not run_token or type(sequence) is not int
            or sequence < 0 or type(epoch) is not str or not epoch
            or action not in ('pause', 'resume')):
        raise ValueError('invalid barrier command')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    target = root / 'command.json'
    temporary = root / 'command.json.tmp'
    temporary.write_text(json.dumps({'run_token': run_token, 'sequence': sequence,
                                     'epoch': epoch, 'action': action},
                                    sort_keys=True) + '\n')
    temporary.replace(target)


def load_acknowledgments(directory, *, run_token, epoch, sequence, ranks):
    """Read exact per-rank acks; return None until every rank is valid."""
    if type(ranks) is not list or not ranks or len(set(ranks)) != len(ranks):
        raise ValueError('explicit unique rank list required')
    result = []
    for rank in ranks:
        path = Path(directory) / f'rank-{rank}.json'
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if (state.get('run_token') != run_token or state.get('epoch') != epoch
                or state.get('sequence') != sequence or state.get('dp_rank') != rank):
            return None
        result.append(state)
    return result


def snapshot(scheduler, rank):
    running = getattr(scheduler, 'running_batch', None)
    chunked = getattr(scheduler, 'chunked_req', None)
    last = getattr(scheduler, 'last_batch', None)
    return {
        'dp_rank': rank,
        'paused': scheduler._engine_paused,
        'waiting_rids': [r.rid for r in scheduler.waiting_queue],
        'running_rids': [] if running is None else [r.rid for r in running.reqs],
        'chunked_rids': [] if chunked is None else [chunked.rid],
        'last_batch_rids': [] if last is None else [r.rid for r in last.reqs],
        'pending_results': len(getattr(scheduler, 'result_queue', ())),
        'deferred_inputs': len(scheduler._deferred_input_requests),
    }


class SchedulerBarrier:
    def __init__(self, directory, run_token, rank):
        if not run_token or type(rank) is not int or rank < 0:
            raise ValueError('unique run token and nonnegative rank required')
        self.directory = Path(directory)
        self.run_token = run_token
        self.rank = rank
        self.epoch = None
        self.last_sequence = -1
        self.owned_pause = False
        self.last_command = None

    def tick(self, scheduler):
        """Scheduler thread only; malformed commands fail closed, not silently."""
        command_path = self.directory / 'command.json'
        if not command_path.exists():
            return
        command = json.loads(command_path.read_text())
        if command.get('run_token') != self.run_token:
            raise ValueError('barrier run token mismatch')
        sequence = command.get('sequence')
        if type(sequence) is not int or sequence < 0:
            raise ValueError('nonnegative command sequence required')
        if sequence < self.last_sequence:
            raise ValueError('stale barrier command')
        if sequence == self.last_sequence and command != self.last_command:
            raise ValueError('same sequence command mutated')
        if sequence > self.last_sequence:
            epoch = command.get('epoch')
            if type(epoch) is not str or not epoch:
                raise ValueError('explicit epoch required')
            state = snapshot(scheduler, self.rank)
            action = command.get('action')
            if action == 'pause':
                if (self.owned_pause or state['paused'] or state['waiting_rids']
                        or state['running_rids'] or state['chunked_rids']
                        or state['last_batch_rids']
                        or state['pending_results'] or state['deferred_inputs']):
                    raise RuntimeError('barrier pause requires exclusive fully idle scheduler')
                scheduler._engine_paused = True
                self.owned_pause = True
                self.epoch = epoch
            elif action == 'resume':
                if not self.owned_pause or epoch != self.epoch or not state['paused']:
                    raise RuntimeError('resume requires matching owned pause epoch')
                scheduler._engine_paused = False
                self.owned_pause = False
            else:
                raise ValueError('unsupported barrier action')
            self.last_sequence = sequence
            self.last_command = command
        state = snapshot(scheduler, self.rank)
        state.update(run_token=self.run_token, epoch=self.epoch,
                     sequence=self.last_sequence)
        # One writer per rank; command publisher must also use atomic replace.
        target = self.directory / f'rank-{self.rank}.json'
        temporary = self.directory / f'rank-{self.rank}.json.tmp'
        temporary.write_text(json.dumps(state) + '\n')
        temporary.replace(target)
