"""Opt-in, signal-free startup stacks; enable before spawning workers only.

Uses Python's own watchdog, not ptrace. Files are process-local. This is a
startup diagnostic, never enable during performance measurements.
"""
import faulthandler
import os
from pathlib import Path
import socket

_stream = None


def install():
    global _stream
    directory = os.environ.get('GLM52_STARTUP_STACK_DIR')
    if not directory or _stream is not None:
        return
    path = Path(directory)
    if not path.is_absolute():
        raise ValueError('GLM52_STARTUP_STACK_DIR must be absolute')
    seconds = int(os.environ.get('GLM52_STARTUP_STACK_SECONDS', '300'))
    if seconds < 10:
        raise ValueError('startup stack interval must be at least ten seconds')
    path.mkdir(parents=True, exist_ok=True)
    _stream = (path / f'{socket.gethostname()}.pid{os.getpid()}.stacks.txt').open('a')
    faulthandler.dump_traceback_later(seconds, repeat=True, file=_stream, exit=False)
