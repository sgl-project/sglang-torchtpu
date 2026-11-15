"""Benchmark-only ordered command source; no automatic runtime patching.

Only TP0 polls this source, before existing request receiver full-TP broadcast.
Other ranks must never poll it to decide pause/resume. This module does not
prove collective safety or perform queue snapshots; integration is separate.
"""
import hashlib
import json
from pathlib import Path


class OrderedCommandSource:
    def __init__(self, directory, run_token):
        if not isinstance(run_token, str) or not run_token or ':' in run_token:
            raise ValueError('nonempty unambiguous run token required')
        self.directory = Path(directory)
        self.run_token = run_token
        self.sequence = -1
        self.last = None
        self.paused_epoch = None

    def poll(self):
        path = self.directory/'ordered-command.json'
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        # Atomic publication required; malformed/mutated commands fail closed.
        command = json.loads(raw)
        expected = {'run_token','sequence','epoch','action'}
        if not isinstance(command, dict) or set(command) != expected:
            raise ValueError('invalid command schema')
        if command['run_token'] != self.run_token:
            raise ValueError('foreign run command')
        seq, epoch, action = command['sequence'],command['epoch'],command['action']
        if type(seq) is not int or seq < 0 or not isinstance(epoch,str) or not epoch or ':' in epoch:
            raise ValueError('invalid sequence/epoch')
        if seq == self.sequence:
            if command != self.last:
                raise ValueError('mutated consumed command')
            return None
        if seq != self.sequence+1:
            raise ValueError('stale/skipped command sequence')
        if action == 'pause':
            if self.paused_epoch is not None:
                raise ValueError('pause already owned')
        elif action == 'resume':
            if self.paused_epoch != epoch:
                raise ValueError('resume does not own paused epoch')
            # Coordinator immutable admission artifact: runtime integration must
            # write only after exact all-rank snapshots have been validated.
            receipt = json.loads((self.directory/f'accepted-{seq}.json').read_text())
            identity = {k:command[k] for k in ('run_token','sequence','epoch')}
            if any(receipt.get(k) != v for k,v in identity.items()):
                raise ValueError('resume acceptance identity mismatch')
            if receipt.get('status') != 'ALL_RANK_INVENTORY_ACCEPTED':
                raise ValueError('resume missing validated inventory acceptance')
            inventory = (self.directory/receipt['inventory_file']).resolve()
            if inventory.parent != self.directory.resolve():
                raise ValueError('inventory must be directly inside run directory')
            if hashlib.sha256(inventory.read_bytes()).hexdigest() != receipt.get('inventory_sha256'):
                raise ValueError('accepted inventory digest mismatch')
        else:
            raise ValueError('invalid action')
        self.sequence, self.last = seq, dict(command)
        self.paused_epoch = epoch if action == 'pause' else None
        return dict(command, rid=f'glm52-control:{self.run_token}:{epoch}:{seq}:{action}')


def accept_inventory(directory, *, expected_routes, snapshots, run_token, epoch, pause_sequence):
    """Coordinator-only: validate exact rank inventory before issuing resume receipt."""
    from benchmark.tpu.glm52_epic.admission_inventory import validate_inventory

    if set(expected_routes.values()) != set(range(16)) or len(expected_routes) != 32:
        raise ValueError('exact32 requests across16ranks required')
    if any(sum(v == rank for v in expected_routes.values()) != 2 for rank in range(16)):
        raise ValueError('exact2 queued requests perrank required')
    verdict = validate_inventory(expected_routes, snapshots, run_token=run_token,
                                 epoch=epoch, sequence=pause_sequence)
    if not verdict['ready']:
        raise ValueError(f'inventory rejected: {verdict["reasons"]}')
    root = Path(directory)
    sequence = pause_sequence+1
    inventory = root/f'inventory-{sequence}.json'
    receipt = root/f'accepted-{sequence}.json'
    if inventory.exists() or receipt.exists():
        raise ValueError('accepted inventory cannot be overwritten')
    from benchmark.tpu.glm52_epic.admission_run import publish_immutable
    publish_immutable(inventory, dict(expected_routes=expected_routes, snapshots=snapshots))
    raw = inventory.read_bytes()
    acceptance = dict(run_token=run_token, sequence=sequence, epoch=epoch,
                      status='ALL_RANK_INVENTORY_ACCEPTED', inventory_file=inventory.name,
                      inventory_sha256=hashlib.sha256(raw).hexdigest())
    publish_immutable(receipt, acceptance)
    return acceptance


def typed_control(command):
    """Lazy real IPC import, called only at the receiver boundary."""
    from sglang.srt.managers.io_struct import (
        ContinueGenerationReqInput, PauseGenerationReqInput,
    )
    if command['action'] == 'pause':
        return PauseGenerationReqInput(mode='in_place', rid=command['rid'])
    return ContinueGenerationReqInput(torch_empty_cache=False, rid=command['rid'])
