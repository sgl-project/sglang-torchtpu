# SPDX-License-Identifier: Apache-2.0
"""Resident anchor worker: load the weights once, then run anchor jobs from a queue.

Every host runs one worker (``run4h.sh`` with ``MAIN=serve``, or ``serve1h.sh`` for a
single host). Workers take jobs ``<queue>/jobs/<n>.json`` in order n = 1, 2, ... so all
hosts run the same sequence without extra coordination. Each job re-imports the
megakernel package (code edits apply without a restart; ``tree`` switches the code tree)
and runs ``anchor.run`` on the resident weights. The job's anchor args are appended to
the worker's own, so weight-shaping args (ckpt / layers / embed / expert load) stay fixed;
``reload_dense`` reloads every family except the routed experts (a dense layout change).

Submit by file path (stdlib only; ``-m`` would import sglang_tpu, which grabs this host's chips):
  python3 .../bench/serve.py --submit <queue> [--tree T] [--reload-dense] [--wait] \\
      -- --warmups 1 --requests 3 --debug-repeat 2
  python3 .../bench/serve.py --submit <queue> --stop
Results: records and SUMMARY lines (tag ``job<n>`` unless given) in ``<queue>/results.jsonl``;
worker status per process in ``<queue>/status.p<k>``.
"""

import argparse
import importlib
import json
import os
import sys
import time
import traceback

_PKG = "sglang_tpu.k3_megakernel"
_SELF = "bench.serve"


def _job_path(queue: str, n: int) -> str:
    return os.path.join(queue, "jobs", f"{n:06d}.json")


def submit(queue: str, job: dict) -> int:
    """Appends ``job`` as the next numbered file (exclusive create; safe for concurrent submitters)."""
    os.makedirs(os.path.join(queue, "jobs"), exist_ok=True)
    n = 1 + max(
        [
            int(f[:6])
            for f in os.listdir(os.path.join(queue, "jobs"))
            if f.endswith(".json")
        ]
        or [0]
    )
    while True:
        tmp = _job_path(queue, n) + ".tmp"
        try:
            fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            n += 1
            continue
        with os.fdopen(fd, "w") as f:
            json.dump(dict(job, submitted=time.time()), f)
        if os.path.exists(_job_path(queue, n)):
            os.unlink(tmp)
            n += 1
            continue
        os.rename(tmp, _job_path(queue, n))
        return n


def wait_summary(queue: str, tag: str, timeout_s: float) -> dict | None:
    path = os.path.join(queue, "results.jsonl")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("tag") == tag and (
                        rec.get("summary") or rec.get("error")
                    ):
                        return rec
        time.sleep(3)
    return None


def _wait_job(queue: str, n: int, poll_s: float) -> dict:
    path = _job_path(queue, n)
    while not os.path.exists(path):
        time.sleep(poll_s)
    with open(path) as f:
        return json.load(f)


_HOME_PATH: list = []


def _fresh_modules(tree: str):
    """Drops the megakernel package and the bench modules (except this one) so the next import
    reads the code again, from ``tree`` (default: the worker's own tree); returns the fresh
    ``anchor`` module."""
    for name in [
        m
        for m in sys.modules
        if (m.startswith(_PKG) or m.startswith("bench.")) and m != _SELF
    ]:
        del sys.modules[name]
    package = importlib.import_module(_PKG)
    bench = importlib.import_module("bench")
    if not _HOME_PATH:
        _HOME_PATH.extend([list(package.__path__), list(bench.__path__)])
    if tree:
        package.__path__[:] = [os.path.join(tree, "sglang_tpu", "k3_megakernel")]
        bench.__path__[:] = [
            os.path.join(tree, "test", "manual", "k3_megakernel", "bench")
        ]
    else:
        package.__path__[:], bench.__path__[:] = _HOME_PATH
    importlib.invalidate_caches()
    return importlib.import_module("bench.anchor")


def _status(queue: str, msg: str) -> None:
    import jax

    with open(os.path.join(queue, f"status.p{jax.process_index()}"), "w") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _run_job(
    queue: str, n: int, job: dict, base_argv: list, mesh, weights, loaded_relax: list
):
    """Runs one job; returns the (possibly reloaded) weights. The dense families are reloaded
    when asked or when the job's ``--relax`` differs from the loaded one (it shapes them)."""
    anchor = _fresh_modules(job.get("tree", ""))
    extra = list(job.get("args", []))
    if "--tag" not in extra:
        extra += ["--tag", f"job{n}"]
    if "--out" not in extra:
        extra += ["--out", os.path.join(queue, "results.jsonl")]
    args = anchor._parse_args(base_argv + extra)
    from sglang_tpu.envs import envs as tpu_envs

    tpu_envs.SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX.set(args.relax)
    if job.get("reload_dense") or args.relax != loaded_relax[0]:
        stack_w, _, _ = weights
        families = importlib.import_module(_PKG + ".model_io").EXPERT_FAMILIES
        experts = {k: stack_w[k] for k in families}
        for k, a in stack_w.items():
            if k not in experts:
                a.delete()
        weights = anchor.load_weights(args, mesh, experts=experts)
        loaded_relax[0] = args.relax
    bench = anchor.run(args, mesh, weights)
    for spec in args.variant:
        anchor.run(
            anchor._variant_args(args, base_argv + extra, spec), mesh, weights=weights
        )
    del bench
    return weights


def _serve(ns, base_argv: list) -> int:
    import jax
    from jax.experimental import multihost_utils

    anchor = _fresh_modules("")
    base = anchor._parse_args(base_argv)
    from sglang_tpu.k3_megakernel import mesh as mesh_mod

    mesh_mod.init_distributed(
        coordinator=base.coordinator,
        num_processes=base.num_processes,
        process_id=base.process_id,
    )
    mesh = mesh_mod.make_mesh()
    os.makedirs(ns.queue, exist_ok=True)
    _status(ns.queue, "loading")
    weights = anchor.load_weights(base, mesh)
    loaded_relax = [base.relax]
    anchor._log(
        f"resident: weights loaded in {weights[2]:.0f}s; waiting for {_job_path(ns.queue, ns.first_job)}"
    )
    n = ns.first_job
    while True:
        _status(ns.queue, f"idle, next job {n}")
        job = _wait_job(ns.queue, n, ns.poll)
        if job.get("stop"):
            anchor._log(f"job {n}: stop")
            _status(ns.queue, "stopped")
            return 0
        _status(ns.queue, f"running job {n}")
        anchor._log(f"job {n}: {json.dumps(job)}")
        t0 = time.time()
        try:
            weights = _run_job(ns.queue, n, job, base_argv, mesh, weights, loaded_relax)
        except (
            Exception,
            SystemExit,
        ) as e:  # a bad job must not take the loaded weights down
            anchor._log(f"job {n} failed:\n{traceback.format_exc()}")
            if jax.process_index() == 0:
                with open(os.path.join(ns.queue, "results.jsonl"), "a") as f:
                    f.write(
                        json.dumps(
                            dict(tag=f"job{n}", error=f"{type(e).__name__}: {e}"[:2000])
                        )
                        + "\n"
                    )
        anchor._log(f"job {n} done in {time.time() - t0:.0f}s")
        if jax.process_count() > 1:
            multihost_utils.sync_global_devices(f"k3mk_job_{n}")
        n += 1


def _submit_main(ns, rest: list) -> int:
    if ns.stop:
        job = {"stop": True}
    else:
        job = {"args": rest, "tree": ns.tree, "reload_dense": ns.reload_dense}
    n = submit(ns.submit, job)
    tag = rest[rest.index("--tag") + 1] if "--tag" in rest else f"job{n}"
    print(f"submitted job {n} (tag {tag}) to {ns.submit}", flush=True)
    if ns.wait and not ns.stop:
        rec = wait_summary(ns.submit, tag, ns.wait_timeout)
        print(
            json.dumps(rec)
            if rec
            else f"no summary for {tag} after {ns.wait_timeout:.0f}s",
            flush=True,
        )
        return 0 if rec and not rec.get("error") else 1
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument(
        "--queue", default="", help="worker mode: job queue directory (on /work)"
    )
    p.add_argument(
        "--first-job",
        type=int,
        default=1,
        help="worker mode: number of the first job to run",
    )
    p.add_argument("--poll", type=float, default=2.0)
    p.add_argument(
        "--submit", default="", metavar="QUEUE", help="submit mode: queue directory"
    )
    p.add_argument(
        "--tree",
        default="",
        help="submit: code tree for this job (default: the worker's)",
    )
    p.add_argument(
        "--reload-dense",
        action="store_true",
        help="submit: reload all but the routed experts",
    )
    p.add_argument("--stop", action="store_true", help="submit: stop the workers")
    p.add_argument(
        "--wait",
        action="store_true",
        help="submit: wait for the job's SUMMARY and print it",
    )
    p.add_argument("--wait-timeout", type=float, default=3600)
    argv = sys.argv[1:] if argv is None else list(argv)
    ours, rest = (
        (argv[: argv.index("--")], argv[argv.index("--") + 1 :])
        if "--" in argv
        else (argv, [])
    )
    ns, unknown = p.parse_known_args(ours)
    if ns.submit:
        return _submit_main(ns, unknown + rest)
    if not ns.queue:
        p.error("--queue (worker) or --submit (client) is required")
    return _serve(ns, unknown + rest)


if __name__ == "__main__":
    sys.exit(main())
