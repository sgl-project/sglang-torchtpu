"""Repro: a subgroup collective hangs unless every chip in the slice launches.

Pipeline parallelism needs a stage to run collectives over its own devices
while the other stages run nothing (or something else). This script checks
that on one 8-chip host (e.g. TPU v7x 2x2x1), modelling PP4 x TP2: stage s is
ranks (2s, 2s+1).

Every subgroup collective runs on its own: torch_tpu splits collectives out
of graphs by default, and each one is followed by a host read, so it is its
own executable. Where the case says so, the ranks finally meet at a CPU (gloo)
barrier, not at a device collective.

Cases:
  pair0_only            only ranks 0,1 launch one pair all_reduce; the rest
                        launch nothing; then a CPU barrier.
  pair0_only_notier2    same, TORCH_TPU_TIER2_COMPILATION_CACHE=disabled.
  pair0_only_blocking   same, TORCH_TPU_LAUNCH_BLOCKING=1.
  pair0_only_spmd_safe  same, the pair all_reduce inside an eager @spmd_safe
                        region.
  pair0_only_subgroups  same, groups made by dist.new_subgroups_by_enumeration.
  pair0_only_devmesh    same, group = init_device_mesh((4,2),("pp","tp"))["tp"].
  pair0_late            ranks 0,1 launch a pair all_reduce at t=0; the others
                        launch their own pair all_reduce 20 s later. Do ranks
                        0,1 finish before the others launch?
  pair0_late_other      same, but the late ranks launch a different program
                        (another shape): is launch matching by count only?
  uneven_notier2        stage s does s+1 pair all_reduces, then a world
                        all_reduce; tier-2 cache disabled.
  uneven_materialize0   same, TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS=0.
  uneven_spmd_safe      same, each pair all_reduce in an eager @spmd_safe region.
  mixed_groups          same launch slot: ranks 0-3 all_reduce over a quad group
                        while ranks 4-7 all_reduce over pair groups.

Observed (TPU v7x 2x2x1, libtpu 0.0.47, 2026-10-01):
  torch_tpu #4472 (0.1.1.dev20260930190326): all pair0_only* and uneven*
    cases TIMEOUT; in pair0_late / pair0_late_other ranks 0,1 complete only
    at the moment the other ranks launch; mixed_groups PASS.
  torch_tpu #4387 (0.1.1.dev20260930221824): every case PASS, and ranks 0,1
    complete at their own launch.

Usage: python3 repro_subgroup_launch_hang.py [case ...]
  SUB_TIMEOUT (default 120) is the per-case timeout in seconds; a hung
  case's process group is killed so the next case gets the chips.
"""

import os
import signal
import subprocess
import sys
import time

CASES = [
    "pair0_only",
    "pair0_only_notier2",
    "pair0_only_blocking",
    "pair0_only_spmd_safe",
    "pair0_only_subgroups",
    "pair0_only_devmesh",
    "pair0_late",
    "pair0_late_other",
    "uneven_notier2",
    "uneven_materialize0",
    "uneven_spmd_safe",
    "mixed_groups",
]
CASE_ENV = {
    "pair0_only_notier2": {"TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled"},
    "uneven_notier2": {"TORCH_TPU_TIER2_COMPILATION_CACHE": "disabled"},
    "pair0_only_blocking": {"TORCH_TPU_LAUNCH_BLOCKING": "1"},
    "uneven_materialize0": {"TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "0"},
}
TIMEOUT = int(os.environ.get("SUB_TIMEOUT", "120"))
STAGES, PAIR = 4, 2
LATE_S = 20


def _log(rank, msg):
    print(f"[r{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_case(name):
    import torch
    import torch.distributed as dist
    import torch_tpu  # noqa: F401
    from torch_tpu._internal.distributed import spmd_util

    dist.init_process_group(backend="tpu_dist")
    cpu_pg = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    stage = rank // PAIR
    peer = rank ^ 1
    dev = torch.device("tpu")
    bad = []

    if name == "pair0_only_subgroups":
        pg, _ = dist.new_subgroups_by_enumeration(
            [[2 * s, 2 * s + 1] for s in range(STAGES)]
        )
    elif name == "pair0_only_devmesh":
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("tpu", (STAGES, PAIR), mesh_dim_names=("pp", "tp"))
        pg = mesh.get_group("tp")
    else:
        pairs = [dist.new_group(ranks=[2 * s, 2 * s + 1]) for s in range(STAGES)]
        pg = pairs[stage]
    quads = [dist.new_group(ranks=list(range(4 * q, 4 * q + 4))) for q in (0, 1)]
    if hasattr(spmd_util, "make_standalone"):
        spmd_util.make_standalone(pg)
        spmd_util.make_standalone(quads[rank // 4])
    _log(rank, f"groups ready, pair members {dist.get_process_group_ranks(pg)}")

    def _ar(x, group):
        dist.all_reduce(x, group=group)
        return x

    spmd_ar = spmd_util.spmd_safe(_ar)

    def pair_all_reduce(i, tag, shape=(8, 128), safe=False):
        x = torch.full(shape, float(rank + i), device=dev)
        _log(rank, f"{tag} launch")
        x = spmd_ar(x, pg) if safe else _ar(x, pg)
        # The host read makes this collective its own executable and blocks
        # until it has run.
        got, exp = float(x.reshape(-1)[0].cpu()), float(rank + peer + 2 * i)
        _log(rank, f"{tag} done got={got} exp={exp}")
        if got != exp:
            bad.append(tag)

    def world_all_reduce():
        x = torch.full((8, 128), float(rank), device=dev)
        dist.all_reduce(x)
        got = float(x[0, 0].cpu())
        _log(rank, f"world got={got}")
        if got != 28.0:
            bad.append("world")

    if name.startswith("pair0_only"):
        if stage == 0:
            pair_all_reduce(0, "pair0", safe=name.endswith("spmd_safe"))
        else:
            _log(rank, "idle (no launch)")
    elif name in ("pair0_late", "pair0_late_other"):
        if stage == 0:
            pair_all_reduce(0, "pair0_t0")
        else:
            time.sleep(LATE_S)
            shape = (1, 128) if name.endswith("other") else (8, 128)
            pair_all_reduce(0, f"late_{shape[0]}", shape=shape)
    elif name.startswith("uneven"):
        for i in range(stage + 1):
            pair_all_reduce(i, f"ar{i}", safe=name.endswith("spmd_safe"))
        world_all_reduce()
    elif name == "mixed_groups":
        if rank < 4:
            x = torch.full((8, 128), float(rank), device=dev)
            _log(rank, "quad launch")
            dist.all_reduce(x, group=quads[0])
            got = float(x[0, 0].cpu())
            _log(rank, f"quad done got={got} exp=6.0")
            if got != 6.0:
                bad.append("quad")
        else:
            pair_all_reduce(0, "pair")
        world_all_reduce()
    else:
        raise ValueError(name)

    torch.accelerator.synchronize()
    _log(rank, "device sync done, entering cpu barrier")
    dist.barrier(group=cpu_pg)
    _log(rank, f"{name} {'PASS' if not bad else f'MISMATCH {bad}'}")


def _worker(rank, name):
    import faulthandler

    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(TIMEOUT - 30, exit=False, file=sys.stderr)
    os.environ.update(
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="8",
        LOCAL_WORLD_SIZE="8",
        GROUP_RANK="0",
    )
    run_case(name)


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--one":
        import portpicker
        import torch.multiprocessing as mp

        try:  # The module moved between nightlies.
            from torch_tpu._internal.distributed import distributed as environment
        except ImportError:
            try:
                from torch_tpu._internal.distributed import environment
            except ImportError:
                from torch_tpu._internal.distributed.launchers import environment

        os.environ.pop("JOB_COMPLETION_INDEX", None)
        for k in (
            "TPU_WORKER_HOSTNAMES",
            "TPU_PROCESS_ADDRESSES",
            "MEGASCALE_NUM_SLICES",
            "MEGASCALE_SLICE_ID",
            "MEGASCALE_COORDINATOR_ADDRESS",
        ):
            os.environ.pop(k, None)
        environment.set_tpu_launch_env(nproc_per_node=8)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(portpicker.pick_unused_port())
        mp.set_start_method("spawn", force=True)
        mp.spawn(_worker, args=(sys.argv[2],), nprocs=8, join=True)
        return

    results = {}
    for name in sys.argv[1:] or CASES:
        print(f"\n>>> {name} env={CASE_ENV.get(name, {})}", flush=True)
        t0 = time.time()
        env = dict(os.environ, **CASE_ENV.get(name, {}))
        # Own session, so a hung case's ranks die with it and free the chips.
        p = subprocess.Popen(
            [sys.executable, __file__, "--one", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
        try:
            out, _ = p.communicate(timeout=TIMEOUT)
            st = (
                "PASS"
                if p.returncode == 0 and "MISMATCH" not in out
                else f"FAIL rc={p.returncode}"
            )
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            out, _ = p.communicate()
            st = "TIMEOUT"
        keep = (
            "[r",
            "Error",
            "Check failed",
            "Traceback",
            "Thread 0x",
            "Current thread",
            "  File ",
            "Timeout (",
        )
        lines = [
            l
            for l in out.splitlines()
            if (
                l.startswith(keep)
                or "Error" in l
                or (l[:1] in "EWF" and l[1:5].isdigit())
            )
            and "]     @" not in l
        ]
        print("\n".join(lines[-300:]), flush=True)
        if st.startswith("FAIL"):
            print(out[-5000:], flush=True)
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        time.sleep(10)
        results[name] = (st, time.time() - t0)
        print(f"<<< {name}: {st} ({results[name][1]:.0f}s)", flush=True)

    print("\n==== SUMMARY ====")
    for n, (st, dt) in results.items():
        print(f"RESULT {n:22s} {st} ({dt:.0f}s)")


if __name__ == "__main__":
    main()
