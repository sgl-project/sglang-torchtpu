"""Explicit benchmark entrypoint adapter; no imports/patches on module import.

Install in EACH spawned scheduler process before constructing its event loop.
Do not install from tokenizer-only parent and assume spawn inherits patches.
This is diagnostic machinery, not a supported serving feature or timing proof.
"""
import functools

from benchmark.tpu.glm52_epic.scheduler_barrier import SchedulerBarrier


def install(scheduler_class, *, directory, run_token, topology):
    """Wrap ingest; topology(scheduler) must return verified process-local values.

    Required keys: pp_size, attn_tp_size, attn_cp_size, attn_dcp_size,
    num_dp_ranks, dp_rank, speculative_enabled. No inferred topology defaults.
    Returns an explicit uninstall callable. Existing instances aren't supported.
    """
    original = scheduler_class.ingest_requests
    if getattr(original, '_glm52_barrier_adapter', False):
        raise RuntimeError('barrier adapter already installed')
    controllers = {}

    @functools.wraps(original)
    def ingest(scheduler, *args, **kwargs):
        # Validate before receiving any work. Keep one strong reference per
        # scheduler to avoid object-id reuse and preserve pause ownership.
        key = id(scheduler)
        if key not in controllers:
            config = topology(scheduler)
            for name in ('pp_size', 'attn_tp_size', 'attn_cp_size', 'attn_dcp_size'):
                if type(config.get(name)) is not int or config[name] != 1:
                    raise ValueError(f'unsupported barrier topology: {name}')
            if config.get('speculative_enabled') is not False:
                raise ValueError('non-speculative scheduler required')
            rank, size = config.get('dp_rank'), config.get('num_dp_ranks')
            if (type(size) is not int or size < 1 or type(rank) is not int
                    or not 0 <= rank < size):
                raise ValueError('explicit valid scheduler DP identity required')
            controllers[key] = (scheduler, SchedulerBarrier(directory, run_token, rank))
        result = original(scheduler, *args, **kwargs)
        controllers[key][1].tick(scheduler)
        return result

    ingest._glm52_barrier_adapter = True
    scheduler_class.ingest_requests = ingest

    def uninstall():
        if any(controller.owned_pause for _, controller in controllers.values()):
            raise RuntimeError('cannot uninstall while adapter owns paused scheduler')
        if scheduler_class.ingest_requests is not ingest:
            raise RuntimeError('ingest wrapper changed; refusing to overwrite it')
        scheduler_class.ingest_requests = original
        controllers.clear()

    return uninstall
