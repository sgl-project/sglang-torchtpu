"""Exclusive benchmark run namespace and immutable atomic artifact publication.

No model/device imports. Coordinator owns publication. Not runtime installation.
"""
import json
import os
from pathlib import Path
import secrets


def publish_immutable(path, value):
    """Publish complete JSON via hardlink; never overwrite an existing artifact."""
    path = Path(path)
    temporary = path.parent/f'.publish-{secrets.token_hex(16)}'
    try:
        with temporary.open('x') as stream:
            json.dump(value, stream, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def create_run(parent):
    """Fresh random token AND exclusive directory: never reopen an old run."""
    token = secrets.token_hex(24)
    root = Path(parent)/token
    root.mkdir(parents=False, exist_ok=False)
    publish_immutable(root/'run.json', dict(run_token=token, owner_pid=os.getpid(),
                                          schema=1, status='NEW_EXCLUSIVE_RUN'))
    return root, token


def verify_run(directory, run_token):
    root = Path(directory)
    marker = json.loads((root/'run.json').read_text())
    if root.name != run_token or marker.get('run_token') != run_token or marker.get('schema') != 1:
        raise ValueError('run directory/nonce marker mismatch')
    return marker


def publish_command(directory, *, run_token, sequence, epoch, action):
    root = Path(directory)
    verify_run(root, run_token)
    ready = json.loads((root/'root-consumer.json').read_text())
    if ready.get('run_token') != run_token or ready.get('status') != 'ROOT_READY':
        raise ValueError('root consumer not ready for this run')
    if (root/'ordered-error.json').exists():
        raise ValueError('run already has terminal control error')
    if type(sequence) is not int or sequence < 0 or not isinstance(epoch,str) or not epoch or ':' in epoch:
        raise ValueError('invalid sequence/epoch')
    if action not in ('pause','resume'):
        raise ValueError('invalid action')
    if sequence and not (root/f'command-{sequence-1}.json').is_file():
        raise ValueError('command sequence gap')
    command=dict(run_token=run_token, sequence=sequence, epoch=epoch, action=action)
    # Immutable journal first. Publication failure cannot overwrite any command.
    publish_immutable(root/f'command-{sequence}.json',command)
    temporary=root/f'.command-pointer-{secrets.token_hex(16)}'
    try:
        temporary.write_text(json.dumps(command)+'\n')
        os.replace(temporary,root/'ordered-command.json')
    finally:
        temporary.unlink(missing_ok=True)
    return command
