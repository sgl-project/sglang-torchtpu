"""Explicit benchmark receiver adapter; NOT automatically installed.

Root injects typed controls BEFORE the existing fullTP control broadcast.
No independent per-rank pause mutation, new collective, or device fence.
"""
import functools

from benchmark.tpu.glm52_epic.ordered_admission import OrderedCommandSource, typed_control


def validate_mesh(config):
    exact = dict(dp_size=1, tp_size=16, attn_dp_size=16, pp_size=1,
                 attn_tp_size=1, attn_cp_size=1, attn_dcp_size=1)
    for name, value in exact.items():
        if type(config.get(name)) is not int or config[name] != value:
            raise ValueError(f'unsupported ordered admission mesh: {name}')
    for name in ('local_control', 'elastic_join', 'speculative', 'disaggregation'):
        if config.get(name) is not False:
            raise ValueError(f'unsupported ordered admission mode: {name}')
    rank = config.get('tp_rank')
    if type(rank) is not int or not 0 <= rank < 16:
        raise ValueError('invalid TP rank')
    if config.get('attn_dp_rank') != rank:
        raise ValueError('exact mesh requires attentionDP rank equal TP rank')
    return rank


def install(receiver_class, *, directory, run_token, topology, factory=typed_control):
    """Explicit topology callback must inspect process runtime; no defaults.

    Does NOT install scheduler acknowledgments or a coordinator. Keep dormant
    until both exist and full receive/broadcast integration is CPU tested.
    """
    from benchmark.tpu.glm52_epic.admission_run import verify_run
    verify_run(directory, run_token)
    original = receiver_class._pull_raw_reqs
    if getattr(original, '_glm52_ordered_receiver', False):
        raise RuntimeError('ordered receiver already installed')
    sources = {}

    @functools.wraps(original)
    def pull(receiver, *args, **kwargs):
        rank = validate_mesh(topology(receiver))
        if receiver.recv_skipper is not None or receiver.input_blocker is not None:
            raise ValueError('receiver skipper/blocker invalidates ordered admission')
        if receiver.scripted_scheduler_hook is not None:
            raise ValueError('scripted receiver unsupported')
        # Validate before consuming anything, but preserve actual ingress first.
        reqs = original(receiver, *args, **kwargs)
        if rank != 0:
            return reqs
        if not isinstance(reqs, list):
            raise ValueError('root receiver must return request list')
        key = id(receiver)
        if key not in sources:
            sources[key] = (receiver, OrderedCommandSource(directory, run_token))
        source = sources[key][1]
        if getattr(source, 'terminal_error', None) is not None:
            # Do not strand peers in their next native control broadcast.
            # External coordinator must observe failure and terminate the run;
            # never resume or retry a failed command implicitly.
            return reqs
        try:
            command = source.poll()
            if command is not None:
                control = factory(command)
                reqs.append(control)
        except Exception as error:
            # Root-only file/validation failures must NOT escape before the
            # collective that peers are already entering. Keep ingress alive,
            # latch failure, reject all later commands and emit best-effort ack.
            import json
            import os
            from pathlib import Path

            source.terminal_error = f'{type(error).__name__}: {error}'
            try:
                root = Path(directory)
                temp = root / f'.ordered-error-{os.getpid()}.tmp'
                temp.write_text(json.dumps(dict(run_token=run_token,
                    status='CONTROL_TERMINAL_ERROR', error=source.terminal_error))+'\n')
                os.replace(temp, root / 'ordered-error.json')
            except OSError:
                import warnings
                warnings.warn('ordered admission failed; error artifact unavailable', RuntimeWarning)
        return reqs

    pull._glm52_ordered_receiver = True
    receiver_class._pull_raw_reqs = pull

    def uninstall():
        if any(source.paused_epoch is not None for _, source in sources.values()):
            raise RuntimeError('cannot uninstall while pause owned')
        if receiver_class._pull_raw_reqs is not pull:
            raise RuntimeError('receiver wrapper replaced')
        receiver_class._pull_raw_reqs = original
        sources.clear()

    return uninstall
