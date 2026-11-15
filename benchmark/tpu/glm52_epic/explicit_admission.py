"""Dormant benchmark-only ordered admission installation.

Importing this module does not install anything or read runtime configuration.
The coordinator must create an exclusive directory with admission_run.create_run,
then explicitly call install(directory=..., run_token=...) in EVERY scheduler
process (including spawned children), before its first ingest_requests call.
No sitecustomize hook, environment switch, core change, or device operation.

Installation/mesh acknowledgments are NOT admission or runtime proof. The current
runtime gate still applies: a bounded external watchdog must abort the whole run
on missing acknowledgments or terminal errors; never auto-resume. In particular,
startup guard/ack failures can stop a rank before a collective. This installer
neither implements distributed recovery nor establishes end-to-end readiness.
"""

import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys


_TARGETS = {
    "sglang.srt.managers.scheduler_components.request_receiver": "receiver",
    "sglang.srt.managers.scheduler": "scheduler",
}


def guard_runtime(scheduler):
    """Fail closed on actual owners and resolved topology; called once per PID.

    Scheduler supplies parsed runtime enums. Receiver supplies its handed config
    record, read through the resolution overlay rather than its pristine inputs.
    No guessed false defaults, tensor reads, new collectives, or pause writes.
    """
    from sglang.srt.arg_groups.model_override_base import resolved_view
    from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel, get_spec
    from benchmark.tpu.glm52_epic.ordered_receiver_adapter import validate_mesh

    parallel = get_parallel()
    moe = get_exec().moe
    receiver = scheduler.request_receiver
    cfg = resolved_view(receiver.server_args)
    # Validate the fields that choose the native control broadcast on THIS rank.
    if parallel.attn_dp_enabled is not True:
        raise ValueError("ordered admission requires attention-DP runtime")
    if moe.elastic_ep_backend is not None:
        raise ValueError("elastic EP is unsupported")
    if moe.is_ep_joiner is not False or moe.is_ep_scale_joiner is not False:
        raise ValueError("elastic EP joiners are unsupported")
    if cfg.enable_dp_attention_local_control_broadcast is not False:
        raise ValueError("receiver local control is unsupported")
    if cfg.speculative_algorithm is not None or get_spec().speculative_algorithm is not None:
        raise ValueError("receiver/runtime speculative configuration unsupported")
    if cfg.disaggregation_mode != "null" or get_disagg().disaggregation_mode != "null":
        raise ValueError("receiver/runtime disaggregation unsupported")
    if scheduler.spec_algorithm.is_none() is not True:
        raise ValueError("scheduler speculative algorithm unsupported")
    if scheduler.disaggregation_mode.value != "null":
        raise ValueError("scheduler disaggregation unsupported")
    if receiver.recv_skipper is not None or receiver.input_blocker is not None:
        raise ValueError("receiver skipper/blocker unsupported")
    if receiver.scripted_scheduler_hook is not None:
        raise ValueError("scripted receiver unsupported")

    names = ("dp_size", "tp_size", "attn_dp_size", "pp_size", "attn_tp_size",
             "attn_cp_size", "attn_dcp_size", "tp_rank", "attn_dp_rank")
    topology = {name: getattr(parallel, name) for name in names}
    topology.update(local_control=parallel.enable_dp_attention_local_control_broadcast,
                    elastic_join=moe.is_ep_joiner,
                    speculative=not scheduler.spec_algorithm.is_none(),
                    disaggregation=scheduler.disaggregation_mode.value != "null")
    rank = validate_mesh(topology)
    for name in ("dp_size", "tp_size", "attn_dp_size", "pp_size", "attn_cp_size"):
        if type(getattr(cfg, name)) is not int or getattr(cfg, name) != topology[name]:
            raise ValueError(f"receiver/runtime topology mismatch: {name}")
    if parallel.dcp_size != 1 or cfg.dcp_size != 1:
        raise ValueError("decode context parallelism unsupported")
    if parallel.pp_rank != 0 or parallel.attn_tp_rank != 0 or parallel.attn_cp_rank != 0:
        raise ValueError("runtime rank incompatible with exact attention-DP mesh")
    group = parallel.tp_group
    if (type(group.rank) is not int or group.rank != rank
            or len(group.ranks) != 16 or len(set(group.ranks)) != 16
            or group.cpu_group is None):
        raise ValueError("full TP CPU control group missing or inconsistent")
    return topology


def export_install_ack(directory, run_token, topology):
    """Immutable per-rank mesh/install evidence, NOT an admission-ready receipt."""
    from benchmark.tpu.glm52_epic.admission_run import publish_immutable, verify_run
    from benchmark.tpu.glm52_epic.ordered_receiver_adapter import validate_mesh

    verify_run(directory, run_token)
    rank = validate_mesh(topology)
    record = dict(run_token=run_token, pid=os.getpid(), tp_rank=rank,
                  status="WRAPPERS_INSTALLED_MESH_VALIDATED_NOT_RUNTIME_PROOF",
                  topology=dict(topology), runtime_proof=False)
    publish_immutable(Path(directory) / f"ordered-install-rank-{rank}.json", record)
    return record


class _Loader(importlib.abc.Loader):
    def __init__(self, loader, finder):
        self.loader, self.finder = loader, finder

    def create_module(self, spec):
        return self.loader.create_module(spec)

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.finder.patch(module)

    def __getattr__(self, name):
        return getattr(self.loader, name)


class AdmissionFinder(importlib.abc.MetaPathFinder):
    def __init__(self, directory, run_token):
        self.directory, self.run_token = directory, run_token
        self.patched = {}
        self.checked = None

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, self)
        return spec

    def scheduler_topology(self, scheduler):
        pid = os.getpid()
        if self.checked is None or self.checked[0] != pid:
            if set(self.patched) != {"receiver", "scheduler"}:
                raise RuntimeError("both ordered wrappers must be installed before ingress")
            topology = guard_runtime(scheduler)
            export_install_ack(self.directory, self.run_token, topology)
            self.checked = (pid, scheduler, scheduler.request_receiver, topology)
        if self.checked[1] is not scheduler:
            raise RuntimeError("one scheduler per process/run required")
        return dict(self.checked[3])

    def receiver_topology(self, receiver):
        # ingest wrapper runs first, before native receive/broadcast. Receiver has
        # no scheduler enums: never pretend it is the scheduler callback owner.
        if (self.checked is None or self.checked[0] != os.getpid()
                or self.checked[2] is not receiver):
            raise RuntimeError("receiver has no validated scheduler owner in this process")
        return dict(self.checked[3])

    def patch(self, module):
        kind = _TARGETS[module.__name__]
        if kind in self.patched:
            raise RuntimeError("ordered admission target reloaded; new run required")
        if kind == "receiver":
            from benchmark.tpu.glm52_epic.ordered_receiver_adapter import install
            undo = install(module.SchedulerRequestReceiver, directory=self.directory,
                           run_token=self.run_token, topology=self.receiver_topology)
        else:
            from benchmark.tpu.glm52_epic.ordered_scheduler_ack import install
            undo = install(module.Scheduler, directory=self.directory,
                           run_token=self.run_token, topology=self.scheduler_topology)
        self.patched[kind] = undo


def install(*, directory, run_token):
    """Explicit process-local opt-in; return finder, not a ready verdict.

    No directory creation here: only the coordinator owns the fresh run root.
    Do not issue commands until the separate root-consumer handshake succeeds.
    Topology is intentionally read only at first scheduler ingress after init;
    changing topology/owners afterwards is outside this fixed-mesh experiment.
    """
    from benchmark.tpu.glm52_epic.admission_run import verify_run

    root = Path(directory)
    if not root.is_absolute():
        raise ValueError("explicit absolute coordinator-created run directory required")
    verify_run(root, run_token)
    if any(isinstance(finder, AdmissionFinder) for finder in sys.meta_path):
        raise RuntimeError("ordered admission already explicitly installed")
    finder = AdmissionFinder(root, run_token)
    sys.meta_path.insert(0, finder)
    try:
        for name in _TARGETS:
            if name in sys.modules:
                finder.patch(sys.modules[name])
    except Exception:
        sys.meta_path.remove(finder)
        for undo in reversed(list(finder.patched.values())):
            undo()
        raise
    return finder
