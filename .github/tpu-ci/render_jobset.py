#!/usr/bin/env python3
"""Render the unified JobSet (jobset-tpu.yaml). Single source of truth for rendering --
used by _tpu-test.yml and runnable locally for dry-runs.

Two modes:
  single-host suite:  --suite <name>
      hosts=1; run_suite runs the suite (each test self-launches its own server).
  multi-host group:   --launch-script <rel-to-benchmark/tpu> --files "<f1> <f2> ..."
                      --hosts N [--weights-pvc <pvc>] [--name <run-name>]
      every host runs the launch script (it owns model+config); rank-0 runs each file.

The slice shape and Kueue queue are looked up by host count in topology.yaml. Only the render
vars are substituted; the container's own $spid / ${TPU_WORKER_ID} stay literal.
${WEIGHTS_VOLUME}/${WEIGHTS_MOUNT} expand to indented YAML blocks (empty when no PVC),
as does ${SERVICE_ACCOUNT} (empty -> namespace default KSA).

Also computes the CI poll budget for a suite (--print-wait-seconds): a static AST scan
of the registered tree sums est_time over the suite's enabled tests -- no sglang import
needed (same technique as discover_multihost.py). The workflow uses it so the poll
timeout scales with the suite instead of a fixed ceiling.

A third mode renders jobset-tpu-pd.yaml instead: P/D disaggregation across TWO
slices (one per role). Every pod of both slices runs the --exec command; the
test file dispatches roles from CI_PD_ROLE + the slice-local TPU worker env.

Usage: render_jobset.py --image <ref> --suite <name> [--nightly] > job.yaml
       render_jobset.py --image <ref> --launch-script <rel> --files "<f>..." --hosts N
       render_jobset.py --image <ref> --exec "<cmd>" --pd-prefill-hosts N \
           --pd-decode-hosts N [--weights-pvc <pvc>] [--name <run-name>]
       render_jobset.py --suite <name> --print-wait-seconds
"""

import argparse
import glob
import os
import re
import sys

import ci_topology
import yaml
from ci_register_loader import ci_register as _ci_register

_WAIT_MARGIN = 1.5
_WAIT_FLOOR_SECONDS = 360

# Fraction of perfect scaling the chip-pinned phase actually achieves. Four
# workers measured 2.8x, the loss being the 3s launch spacing plus the tail
# where fewer than four files remain. Used only to size the runtime budget:
# over-stating it would cut a slow run short, understating it hands the
# runtime clock seconds that the admission budget needs more.
_CHIP_PARALLEL_EFFICIENCY = 0.7

# P/D mode: two engines to bring up (weight load + compile warmup on both
# slices) before the probe + accuracy eval even start. Longest seen: 100 min;
# kept under 3h so _tpu-test.yml's 6h job cap leaves a 3h admission window.
_PD_ACTIVE_DEADLINE = 8700

# Indentation matches jobset-tpu.yaml: volume items under `volumes:` at 16 spaces,
# volumeMount items under `volumeMounts:` at 20 spaces.
WEIGHTS_VOLUME_TMPL = (
    "                - name: weights\n"
    "                  persistentVolumeClaim:\n"
    "                    claimName: {pvc}\n"
    "                    readOnly: true"
)
# Pod-level nodeAffinity restricting the pod to CI pools. Indented to sit under the pod
# spec, same level as nodeSelector. Kueue's ResourceFlavor injects the exact ci-tpu-pool via
# nodeSelector on admission; this is the outer bound, and it is what keeps a JobSet Kueue
# is NOT managing off the dev pools (manageJobsWithoutQueueName is false).
POOL_AFFINITY = (
    "              affinity:\n"
    "                nodeAffinity:\n"
    "                  requiredDuringSchedulingIgnoredDuringExecution:\n"
    "                    nodeSelectorTerms:\n"
    "                      - matchExpressions:\n"
    "                          - key: ci-tpu-pool\n"
    "                            operator: Exists"
)


WEIGHTS_MOUNT_TMPL = (
    "                    - name: weights\n"
    "                      mountPath: /weights\n"
    "                      readOnly: true"
)
# serviceAccountName sits under `spec:` of the pod template, at 14 spaces. Empty unless
# --service-account is given, so the pod keeps using the namespace default.
SERVICE_ACCOUNT_TMPL = "              serviceAccountName: {sa}"


def dns_safe(name):
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    return (s[:40].strip("-")) or "run"


def single_test_name(files):
    """Return a test filename stem when files contains exactly one path."""
    paths = files.split()
    if len(paths) != 1:
        return ""
    return os.path.basename(paths[0]).removesuffix(".py")


def suite_wait_seconds(
    suite,
    tests_dir="test/registered",
    *,
    nightly=False,
    test_file="",
    partition_id=None,
    partition_size=None,
    chip_workers=0,
):
    files = [
        path
        for path in glob.glob(os.path.join(tests_dir, "**", "*.py"), recursive=True)
        if os.path.basename(path) not in ("conftest.py", "__init__.py")
    ]
    tests = [
        test
        for test in _ci_register.collect_tests(files, sanity_check=False)
        if test.backend == _ci_register.HWBackend.TPU
        and test.effective_suite == suite
        and test.nightly == nightly
        and test.disabled is None
    ]
    if test_file:
        requested = os.path.normpath(os.path.abspath(test_file))
        tests = [
            test
            for test in tests
            if os.path.normpath(os.path.abspath(test.filename)) == requested
        ]
    if partition_size is not None:
        tests = _ci_register.auto_partition(tests, partition_id, partition_size)
    total = sum(test.est_time for test in tests)
    if chip_workers > 1:
        # `chips=1` files run several at a time, so their est_time no longer
        # adds up along the clock. Whole-node files still do. Getting this
        # wrong is not symmetric: _tpu-test.yml spends what is left of the
        # runner's 6h on queueing, so an over-booked runtime is why a job
        # sits behind a busy queue and then reports NOT ADMITTED.
        pinned = sum(test.est_time for test in tests if test.chips == 1)
        total -= pinned - pinned / (chip_workers * _CHIP_PARALLEL_EFFICIENCY)
    return max(int(total * _WAIT_MARGIN), _WAIT_FLOOR_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--suite", default="")  # single-host mode
    ap.add_argument("--test-file", default="")  # one file from a single-host suite
    ap.add_argument("--launch-script", default="")  # multi-host mode
    ap.add_argument("--files", default="")  # multi-host: space-separated
    # Exec mode: every host runs this command; pod exit codes gate the JobSet.
    # Plain words only -- the command is inlined into the pod script unquoted.
    ap.add_argument("--exec", dest="exec_cmd", default="")
    ap.add_argument("--hosts", type=int, default=1)
    # Which lane's Kueue queue to submit to; every lane shares every CI pool of a shape.
    ap.add_argument("--lane", default="pr-gate")
    # P/D mode: hosts per role, each role on its own slice. Both must be set
    # (with --exec); --hosts is ignored in this mode.
    ap.add_argument("--pd-prefill-hosts", type=int, default=0)
    ap.add_argument("--pd-decode-hosts", type=int, default=0)
    ap.add_argument(
        "--weights-pvc",
        default="",
        help="Key from weights.yaml or literal PVC name; empty disables the mount.",
    )
    ap.add_argument("--timeout-per-file", type=int, default=3600)
    ap.add_argument(
        "--service-account",
        default="",
        help="KSA for the pod; Workload Identity uses it to read weights from GCS.",
    )
    ap.add_argument("--name", default="")  # multi-host run label
    # SGLANG_PLATFORM for the pods: tpu (mode A, in-tree) or tpu_oot (mode B, plugin).
    ap.add_argument("--platform", choices=("tpu", "tpu_oot"), default="tpu")
    ap.add_argument("--dir", default=".github/tpu-ci")
    ap.add_argument("--tests-dir", default="test/registered")
    ap.add_argument(
        "--tpu-chip-workers",
        type=int,
        default=0,
        help=(
            "Run this many `chips=1` tests at once, each pinned to one chip of "
            "the slice the job already holds. 0 or 1 keeps the whole node per test."
        ),
    )
    ap.add_argument("--partition-id", type=int, default=-1)
    ap.add_argument("--partition-size", type=int, default=0)
    ap.add_argument(
        "--print-wait-seconds",
        action="store_true",
        help="Print the CI poll budget (s) for --suite and exit (no rendering).",
    )
    a = ap.parse_args()

    partitioned = a.partition_id != -1 or a.partition_size != 0
    if partitioned and not (
        a.partition_size > 0 and 0 <= a.partition_id < a.partition_size
    ):
        ap.error("--partition-id must be in [0, --partition-size)")

    partition_args = {
        "partition_id": a.partition_id if partitioned else None,
        "partition_size": a.partition_size if partitioned else None,
    }
    if a.print_wait_seconds:
        print(
            suite_wait_seconds(
                a.suite,
                a.tests_dir,
                nightly=a.nightly,
                test_file=a.test_file,
                chip_workers=a.tpu_chip_workers,
                **partition_args,
            )
        )
        return

    if not a.image:
        ap.error("--image is required when rendering")

    # Resolve shared storage names once, before either JobSet rendering mode.
    with open(f"{a.dir}/weights.yaml") as f:
        weights_pvcs = yaml.safe_load(f)
    a.weights_pvc = weights_pvcs.get(a.weights_pvc, a.weights_pvc)

    topo = ci_topology.load(f"{a.dir}/topology.yaml")
    if a.lane not in topo["lanes"]:
        ap.error(f"unknown --lane {a.lane!r}; topology.yaml has {topo['lanes']}")

    def slice_for(hosts):
        shape = ci_topology.shape_for_hosts(topo, hosts)
        if shape is None:
            sys.exit(f"no slice shape with {hosts} hosts in topology.yaml")
        return {"topology": shape, "queue": ci_topology.queue_name(a.lane, hosts)}

    if a.pd_prefill_hosts or a.pd_decode_hosts:
        if partitioned:
            ap.error("partitioning is only supported in single-host suite mode")
        if not (a.pd_prefill_hosts >= 1 and a.pd_decode_hosts >= 1):
            ap.error("P/D mode needs both --pd-prefill-hosts and --pd-decode-hosts")
        if not a.exec_cmd:
            ap.error("P/D mode needs --exec")
        if a.platform != "tpu":
            ap.error(
                "P/D mode runs mode A only; jobset-tpu-pd.yaml pins SGLANG_PLATFORM=tpu"
            )
        # The pod script announces --files and reports the run's verdict against
        # them in TIMINGS format, which is the only thing that gives a P/D run a
        # test name on the dashboard. Without it the run lands there named after
        # the job, which reads as a test that does not exist -- so require it
        # rather than emitting a record with an empty `file`.
        if not a.files:
            ap.error("P/D mode needs --files (the test file(s) --exec runs)")
        run_name = dns_safe(a.name or single_test_name(a.files) or "pd")
        # Pods locate their peer slice by swapping the replicated-job segment
        # ("-prefill-"/"-decode-") of their own hostname; a run name containing
        # either word would make that swap ambiguous.
        for role in ("prefill", "decode"):
            if role in run_name:
                sys.exit(f"P/D run name must not contain {role!r}: {run_name}")
        prefill_slice = slice_for(a.pd_prefill_hosts)
        decode_slice = slice_for(a.pd_decode_hosts)
        # One JobSet carries one queue-name label, so both roles must resolve to
        # the same LocalQueue; Kueue then assigns each role's PodSet its own
        # ResourceFlavor (one slice each) out of that queue's quota.
        if prefill_slice["queue"] != decode_slice["queue"]:
            sys.exit(
                "P/D roles must share a Kueue LocalQueue; lane="
                f"{a.lane} gives {prefill_slice['queue']} (prefill, "
                f"{a.pd_prefill_hosts}h) vs {decode_slice['queue']} (decode, "
                f"{a.pd_decode_hosts}h)"
            )
        # Exclusive placement gives each child Job its own nodepool, which is
        # what pins a multihost group onto one slice (a GKE multihost nodepool
        # is one slice). Two single-host groups share ONE nodepool, so the
        # same annotation would strand the second group Pending forever.
        multihost = a.pd_prefill_hosts > 1 or a.pd_decode_hosts > 1
        exclusive = (
            "  annotations:\n"
            "    alpha.jobset.sigs.k8s.io/exclusive-topology: "
            "cloud.google.com/gke-nodepool"
        )
        pvc = a.weights_pvc
        vals = {
            "IMAGE": a.image,
            "ACTIVE_DEADLINE": str(_PD_ACTIVE_DEADLINE),
            "EXEC_CMD": a.exec_cmd,
            "TEST_FILES": a.files,
            "RUN_NAME": run_name,
            "EXCLUSIVE_PLACEMENT": exclusive if multihost else "",
            "PREFILL_HOSTS": str(a.pd_prefill_hosts),
            "DECODE_HOSTS": str(a.pd_decode_hosts),
            "QUEUE": prefill_slice["queue"],
            "PREFILL_TOPOLOGY": prefill_slice["topology"],
            # Kueue's per-PodSet flavor puts the two roles on different slices.
            "PREFILL_POOL_AFFINITY": POOL_AFFINITY,
            "DECODE_TOPOLOGY": decode_slice["topology"],
            "DECODE_POOL_AFFINITY": POOL_AFFINITY,
            "WEIGHTS_VOLUME": WEIGHTS_VOLUME_TMPL.format(pvc=pvc) if pvc else "",
            "WEIGHTS_MOUNT": WEIGHTS_MOUNT_TMPL if pvc else "",
            "SERVICE_ACCOUNT": (
                SERVICE_ACCOUNT_TMPL.format(sa=a.service_account)
                if a.service_account
                else ""
            ),
        }
        tpl = open(f"{a.dir}/jobset-tpu-pd.yaml").read()
        out = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), tpl)
        sys.stdout.write(out)
        return

    if a.launch_script:
        if partitioned:
            ap.error("partitioning is only supported in single-host suite mode")
        if a.hosts < 1:
            ap.error("--hosts must be a positive integer for multi-host mode")
        hosts = a.hosts
        run_name = dns_safe(a.name or single_test_name(a.files) or a.launch_script)
        suite = ""
        if not a.files:
            sys.exit("multi-host mode needs --files")
    elif a.exec_cmd:
        if partitioned:
            ap.error("partitioning is only supported in single-host suite mode")
        if a.hosts < 1:
            ap.error("--hosts must be a positive integer for exec mode")
        hosts = a.hosts
        run_name = dns_safe(a.name or "exec")
        suite = ""
    else:
        if not a.suite:
            sys.exit(
                "need --suite (single-host), --launch-script (multi-host) or --exec"
            )
        hosts = 1
        default_name = (
            f"{a.suite}-{a.partition_id + 1}-of-{a.partition_size}"
            if partitioned
            else a.suite
        )
        run_name = dns_safe(a.name or single_test_name(a.test_file) or default_name)
        suite = a.suite
        if a.timeout_per_file <= 0:
            ap.error("--timeout-per-file must be a positive integer")

    sl = slice_for(hosts)

    # Suite mode's deadline scales with the suite (same est_time formula as
    # _tpu-test's poll budget, keeping the two in agreement), discounted for
    # chip sharing when it is on. Exec gets 7200: the combo rows budget WAIT_SECS=2400 for
    # engine bring-up alone.
    if suite:
        active_deadline = suite_wait_seconds(
            suite,
            a.tests_dir,
            nightly=a.nightly,
            test_file=a.test_file,
            chip_workers=a.tpu_chip_workers,
            **partition_args,
        )
    elif a.exec_cmd:
        active_deadline = 7200
    else:
        # Launch mode: up to ~50m server bring-up (R1 weight load + cold compile;
        # the health poll in jobset-tpu.yaml allows 300x10s) + each file's
        # `timeout` budget. _tpu-test's mh wait_s mirrors this formula.
        active_deadline = 3600 + len(a.files.split()) * a.timeout_per_file

    pvc = a.weights_pvc
    vals = {
        "IMAGE": a.image,
        "ACTIVE_DEADLINE": str(active_deadline),
        "EXEC_CMD": a.exec_cmd,
        "HOSTS": str(hosts),
        "TOPOLOGY": sl["topology"],
        "POOL_AFFINITY": POOL_AFFINITY,
        "QUEUE": sl["queue"],
        "RUN_NAME": run_name,
        "SUITE": suite,
        "PLATFORM": a.platform,
        # --continue-on-error only on the nightly, matching the CUDA nightly: a
        # scheduled run should report every file's verdict, where the PR gate still
        # fast-fails to give the slice back. Without it the suite stops at the first
        # failure and the files behind it never report at all.
        "NIGHTLY_FLAG": "--nightly --continue-on-error" if a.nightly else "",
        "PARTITION_FLAGS": (
            f"--auto-partition-id {a.partition_id} "
            f"--auto-partition-size {a.partition_size}"
            if partitioned
            else ""
        ),
        "TEST_FILE_FLAG": f"--test-file {a.test_file}" if a.test_file else "",
        "CHIP_WORKERS_FLAG": (
            f"--tpu-chip-workers {a.tpu_chip_workers}" if a.tpu_chip_workers > 1 else ""
        ),
        # Sweeps trimmed for the gate run whole on the nightly; see
        # python/sglang/test/ci/gate_matrix.py.
        "FULL_MATRIX": "1" if a.nightly else "0",
        "LAUNCH_SCRIPT": a.launch_script,
        "TEST_FILES": a.files,
        "WEIGHTS_VOLUME": WEIGHTS_VOLUME_TMPL.format(pvc=pvc) if pvc else "",
        "WEIGHTS_MOUNT": WEIGHTS_MOUNT_TMPL if pvc else "",
        "TIMEOUT_PER_FILE": str(a.timeout_per_file),
        "SERVICE_ACCOUNT": (
            SERVICE_ACCOUNT_TMPL.format(sa=a.service_account)
            if a.service_account
            else ""
        ),
    }

    tpl = open(f"{a.dir}/jobset-tpu.yaml").read()
    out = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), tpl)
    sys.stdout.write(out)


if __name__ == "__main__":
    main()
