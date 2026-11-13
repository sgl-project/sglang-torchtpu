"""Standalone stdlib contract selftest, intentionally NOT a registered SGLang test.

Fake external events/owners exercise the actual benchmark wrapper functions.
No torch, SGLang import, accelerator, sleep, or new synchronization is needed.
"""

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS

from observer import (
    ENV,
    MARKER,
    SNAPSHOT_ATTR,
    JsonlSink,
    wrap_prefill,
    wrap_run_batch,
)


def fixture(mode="EXTEND"):
    owner = NS(
        is_generation=True, enable_overlap=True, enable_overlap_mlx=False,
        disaggregation_mode=NS(value="null"), forward_ct=11,
    )
    parallel = NS(pp_size=1, tp_rank=7, pp_rank=0, attn_dp_rank=3)
    batch = NS(
        reqs=[NS(rid="measured-0")], prefix_lens=[49152], extend_lens=[2048],
        spec_algorithm=NS(is_none=lambda: True), forward_mode=NS(name=mode),
    )
    return owner, parallel, batch


class ExternalEvent:
    def __init__(self, trace, fail=False):
        self.trace = trace
        self.fail = fail
        self.handle = object()

    def query(self):
        self.trace.append("query")
        return False

    def synchronize(self, token=None):
        self.trace.append(("wait", token))
        if self.fail:
            raise RuntimeError("external wait failed")
        return "external-return"


def launched(owner, parallel, batch, event, mismatch=False):
    def forward(self, current):
        self.forward_ct += 1
        current.forward_iter = self.forward_ct + int(mismatch)
        # Real scheduling may mutate/reuse request and batch CPU metadata.
        current.reqs[0].rid = "mutated"
        current.prefix_lens[0] = 99999
        current.extend_lens[0] = 1
        return NS(copy_done=event)

    return wrap_run_batch(forward, lambda: parallel)(owner, batch)


def check_wait_and_snapshot():
    trace, records = [], []
    owner, parallel, batch = fixture()
    event = ExternalEvent(trace)
    result = launched(owner, parallel, batch, event)
    assert trace == []  # Launch wrapper introduced no event wait or query.
    snapshot = getattr(result, SNAPSHOT_ATTR)
    assert snapshot.forward_step == 12
    assert snapshot.requests[0].rid == "measured-0"
    assert snapshot.requests[0].prefix == 49152
    assert snapshot.requests[0].tokens == 2048
    assert snapshot.requests[0].dp == 3  # NOT tp_rank=7 / generic dp_rank.
    try:
        snapshot.requests[0].prefix = 8
        raise AssertionError("snapshot is mutable")
    except dataclasses.FrozenInstanceError:
        pass

    def clock():
        trace.append("timestamp")
        return 123456

    def processor(self, current, output):
        # Entry proxy must already be present before any processor work.
        assert output.copy_done is not event
        assert output.copy_done.handle is event.handle
        assert output.copy_done.query() is False  # Existing drain query delegates.
        assert output.copy_done.synchronize("first") == "external-return"
        assert output.copy_done.synchronize("second") == "external-return"
        trace.append("host_processing")
        return 97

    def emit(record):
        assert result.copy_done is event  # finally restores BEFORE JSONL work.
        trace.append("emit")
        records.append(record)

    actual = wrap_prefill(processor, emit, lambda: parallel, clock)(owner, batch, result)
    assert actual == 97 and result.copy_done is event
    assert trace == ["query", ("wait", "first"), "timestamp",
                     ("wait", "second"), "host_processing", "emit"]
    record = records[0]
    assert record["marker"] == MARKER and record["status"] == "observed"
    assert record["completed_ns"] == 123456
    assert record["requests"] == [dict(rid="measured-0", dp=3, prefix=49152,
                                       tokens=2048)] or record["requests"] == (
        dict(rid="measured-0", dp=3, prefix=49152, tokens=2048),
    )


def check_unavailable_and_exceptions():
    for scenario, reason in [
        ("missing", "missing_copy_done"),
        ("no_wait", "existing_wait_not_observed"),
        ("failed_wait", "copy_done_synchronize_failed"),
        ("host_failure", None),
        ("ordinal", "forward_iter_mismatch"),
        ("metadata", "invalid_cpu_metadata"),
    ]:
        trace, records = [], []
        owner, parallel, batch = fixture()
        if scenario == "metadata":
            # Coercion would be an accidental device-to-host read.
            class NotAnInt:
                def __int__(self):
                    raise AssertionError("must not coerce device metadata")
            batch.prefix_lens[0] = NotAnInt()
        event = None if scenario == "missing" else ExternalEvent(
            trace, fail=scenario == "failed_wait"
        )
        result = launched(owner, parallel, batch, event, scenario == "ordinal")

        def processor(self, current, output):
            if output.copy_done is not None and scenario != "no_wait":
                output.copy_done.synchronize()
            if scenario == "host_failure":
                raise RuntimeError("host processing failed")
            return "preserved"

        def clock():
            trace.append("timestamp")
            return 42

        wrapper = wrap_prefill(processor, records.append, lambda: parallel, clock)
        try:
            assert wrapper(owner, batch, result) == "preserved"
            assert scenario not in ("failed_wait", "host_failure")
        except RuntimeError as exc:
            assert str(exc) == ("external wait failed" if scenario == "failed_wait"
                                else "host processing failed")
        assert result.copy_done is event
        assert len(records) == 1
        assert records[0]["unavailable_reason"] == reason
        assert records[0]["completed_ns"] == (42 if reason is None else None)
        assert trace.count("timestamp") == int(
            scenario not in ("missing", "no_wait", "failed_wait")
        )

    # Emitter failure must not mask original exception or prevent restoration.
    owner, parallel, batch = fixture()
    event = ExternalEvent([])
    result = launched(owner, parallel, batch, event)

    def broken_processor(*args):
        raise LookupError("original")

    def broken_sink(record):
        raise OSError("disk full")

    import warnings
    with warnings.catch_warnings(record=True) as caught:
        try:
            wrap_prefill(broken_processor, broken_sink, lambda: parallel)(
                owner, batch, result
            )
            raise AssertionError("exception swallowed")
        except LookupError as exc:
            assert str(exc) == "original"
        assert len(caught) == 1
    assert result.copy_done is event


def check_path_gates():
    for mode in ("DECODE", "MIXED", "IDLE", "SPLIT_PREFILL", "TARGET_VERIFY",
                 "DRAFT_EXTEND_V2", "DLLM_EXTEND", "PREBUILT"):
        owner, parallel, batch = fixture(mode)
        _check_bypass(owner, parallel, batch)
    for field, value in (("is_generation", False), ("enable_overlap", False),
                         ("enable_overlap_mlx", True),
                         ("disaggregation_mode", NS(value="prefill"))):
        owner, parallel, batch = fixture()
        setattr(owner, field, value)
        _check_bypass(owner, parallel, batch)
    owner, parallel, batch = fixture()
    parallel.pp_size = 2
    _check_bypass(owner, parallel, batch)
    owner, parallel, batch = fixture()
    batch.spec_algorithm = NS(is_none=lambda: False)
    _check_bypass(owner, parallel, batch)


def _check_bypass(owner, parallel, batch):
    event = ExternalEvent([])
    result = NS(copy_done=event)
    # Use a real wrapped launch and result-processor pair: excluded paths never
    # carry metadata or emit observations even if original code itself waits.
    launch = wrap_run_batch(lambda *a: result, lambda: parallel)
    assert launch(owner, batch) is result
    assert not hasattr(result, SNAPSHOT_ATTR)

    def processor(self, current, output):
        assert output.copy_done is event
        return "unmodified"

    records = []
    assert wrap_prefill(processor, records.append, lambda: parallel)(
        owner, batch, result
    ) == "unmodified"
    assert records == []


def check_sink_and_import_hook():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sink = JsonlSink(root / "events")
        sink({"marker": MARKER})
        sink({"marker": MARKER})
        paths = list((root / "events").glob("*.jsonl"))
        assert len(paths) == 1
        rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        assert len(rows) == 2 and rows[0]["pid"] == os.getpid()
        # Exercise Python startup + actual meta-path import interception without
        # importing production SGLang (fake only the external module boundary).
        package = root / "sglang" / "srt" / "managers"
        component = package / "scheduler_components"
        component.mkdir(parents=True)
        for directory in (root / "sglang", root / "sglang" / "srt", package,
                          component):
            (directory / "__init__.py").write_text("")
        (package / "scheduler.py").write_text(
            "class Scheduler:\n    def run_batch(self, batch): return 1\n"
        )
        (component / "batch_result_processor.py").write_text(
            "class SchedulerBatchResultProcessor:\n"
            "    def process_batch_result_prefill(self, batch, result): return 2\n"
        )
        code = """
import sys
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
from observer import install
active = sys.argv[1] == 'active'
assert bool(getattr(Scheduler.run_batch, '_glm52_epic_wrapped', False)) == active
assert bool(getattr(SchedulerBatchResultProcessor.process_batch_result_prefill, '_glm52_epic_wrapped', False)) == active
assert install() == active
from observer import ObserverFinder
assert sum(isinstance(f, ObserverFinder) for f in sys.meta_path) == int(active)
assert 'torch' not in sys.modules
"""
        for active in (False, True):
            env = dict(os.environ)
            env.pop(ENV, None)
            env["PYTHONPATH"] = os.pathsep.join((str(Path(__file__).parent.resolve()),
                                               str(root)))
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            if active:
                env[ENV] = str(root / "import-events")
            subprocess.run([sys.executable, "-c", code,
                            "active" if active else "inactive"],
                           env=env, cwd=root, check=True)


def main():
    checks = [check_wait_and_snapshot, check_unavailable_and_exceptions,
              check_path_gates, check_sink_and_import_hook]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")
    print("PASS stdlib benchmark observer selftest (no accelerator validation)")


if __name__ == "__main__":
    main()
