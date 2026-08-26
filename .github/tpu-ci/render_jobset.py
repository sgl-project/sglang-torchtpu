#!/usr/bin/env python3
"""Render the unified JobSet (jobset-tpu.yaml). Single source of truth for rendering --
used by _tpu-test.yml and runnable locally for dry-runs.

Two modes:
  single-host suite:  --suite <name>
      hosts=1; run_suite runs the suite (each test self-launches its own server).
  multi-host group:   --launch-script <rel-to-benchmark/tpu> --files "<f1> <f2> ..."
                      --hosts N [--weights-pvc <pvc>] [--name <run-name>]
      every host runs the launch script (it owns model+config); rank-0 runs each file.

The SLICE (topology + pool) is looked up by host count in topology.yaml. Only the render
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
import ast
import os
import re
import sys

import yaml

# Poll budget = max(sum(est_time) * MARGIN, FLOOR). The margin gives headroom over the
# (approximate) est_time estimates; the floor covers tiny/near-empty suites.
_WAIT_MARGIN = 1.5
_WAIT_FLOOR_SECONDS = 360

# P/D mode: two engines to bring up (weight load + compile warmup on both
# slices) before the probe + accuracy eval even start.
_PD_ACTIVE_DEADLINE = 10800

# Positional order of register_*_ci(...) args (mirrors ci_register._PARAM_ORDER) so the
# scan handles both register_tpu_ci(120, "suite") and register_tpu_ci(est_time=120, ...).
_REGISTER_PARAMS = ("est_time", "suite", "nightly", "disabled")

# Indentation matches jobset-tpu.yaml: volume items under `volumes:` at 16 spaces,
# volumeMount items under `volumeMounts:` at 20 spaces.
WEIGHTS_VOLUME_TMPL = (
    "                - name: weights\n"
    "                  persistentVolumeClaim:\n"
    "                    claimName: {pvc}\n"
    "                    readOnly: true"
)
# Pod-level nodeAffinity restricting the pod to this (lane, hosts)'s pools. Indented to sit
# under the pod spec, same level as nodeSelector. Kueue's ResourceFlavor injects the exact
# ci-tpu-pool via nodeSelector on admission; this is the outer bound, and it is what keeps a
# JobSet Kueue is NOT managing off the dev pools (manageJobsWithoutQueueName is false).
POOL_AFFINITY_TMPL = (
    "              affinity:\n"
    "                nodeAffinity:\n"
    "                  requiredDuringSchedulingIgnoredDuringExecution:\n"
    "                    nodeSelectorTerms:\n"
    "                      - matchExpressions:\n"
    "                          - key: ci-tpu-pool\n"
    "                            operator: In\n"
    "                            values: [{pools}]"
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


def suite_wait_seconds(suite, tests_dir="test/registered"):
    """CI poll budget (seconds) for `suite`: max(sum(est_time) * MARGIN, FLOOR).

    Sums est_time over the enabled tests registered to `suite` by statically parsing
    the registered tree (no sglang import). A test counts if it has a register_*_ci(...)
    call with suite=<suite> and no truthy `disabled=` reason.
    """
    total = 0.0
    for root, _dirs, files in os.walk(tests_dir):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            try:
                tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fname = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                if not (fname.startswith("register_") and fname.endswith("_ci")):
                    continue
                # Merge positional + keyword args by name so both call styles work.
                argv = {
                    _REGISTER_PARAMS[i]: a
                    for i, a in enumerate(node.args)
                    if i < len(_REGISTER_PARAMS)
                }
                argv.update({k.arg: k.value for k in node.keywords if k.arg})
                suite_val = argv.get("suite")
                if not (
                    isinstance(suite_val, ast.Constant) and suite_val.value == suite
                ):
                    continue
                disabled = argv.get("disabled")
                if disabled is not None and not (
                    isinstance(disabled, ast.Constant) and not disabled.value
                ):
                    continue  # disabled registration -- not run, so excluded
                est = argv.get("est_time")
                if isinstance(est, ast.Constant) and isinstance(
                    est.value, (int, float)
                ):
                    total += float(est.value)
    return max(int(total * _WAIT_MARGIN), _WAIT_FLOOR_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--nightly", action="store_true")
    ap.add_argument("--suite", default="")  # single-host mode
    ap.add_argument("--launch-script", default="")  # multi-host mode
    ap.add_argument("--files", default="")  # multi-host: space-separated
    # Exec mode: every host runs this command; pod exit codes gate the JobSet.
    # Plain words only -- the command is inlined into the pod script unquoted.
    ap.add_argument("--exec", dest="exec_cmd", default="")
    ap.add_argument("--hosts", type=int, default=1)
    # Which lane's pools to use. pr-gate and nightly both own 1- and 2-host slices, so a
    # host count alone no longer identifies a pool (see topology.yaml).
    ap.add_argument("--lane", default="pr-gate")
    # Overrides topology.yaml's `pools` for this (lane, hosts). discover_pools.py supplies
    # it so a new slice needs no edit there; empty falls back to the static list.
    ap.add_argument("--pools", default="")
    # P/D mode: hosts per role, each role on its own slice. Both must be set
    # (with --exec); --hosts is ignored in this mode.
    ap.add_argument("--pd-prefill-hosts", type=int, default=0)
    ap.add_argument("--pd-decode-hosts", type=int, default=0)
    ap.add_argument("--weights-pvc", default="")
    ap.add_argument("--timeout-per-file", type=int, default=3600)
    ap.add_argument(
        "--service-account",
        default="",
        help="KSA for the pod; Workload Identity uses it to read weights from GCS.",
    )
    ap.add_argument("--name", default="")  # multi-host run label
    ap.add_argument("--dir", default=".github/tpu-ci")
    ap.add_argument("--tests-dir", default="test/registered")
    ap.add_argument(
        "--print-wait-seconds",
        action="store_true",
        help="Print the CI poll budget (s) for --suite and exit (no rendering).",
    )
    a = ap.parse_args()

    if a.print_wait_seconds:
        print(suite_wait_seconds(a.suite, a.tests_dir))
        return

    if not a.image:
        ap.error("--image is required when rendering")

    lanes = yaml.safe_load(open(f"{a.dir}/topology.yaml"))["lanes"]
    if a.lane not in lanes:
        ap.error(f"unknown --lane {a.lane!r}; topology.yaml has {sorted(lanes)}")
    slices = lanes[a.lane]

    def slice_for(hosts):
        sl = slices.get(hosts, slices.get(str(hosts)))
        if not sl:
            sys.exit(
                f"no slice for lane={a.lane} hosts={hosts} in topology.yaml "
                f"(lane has {sorted(slices)})"
            )
        return sl

    if a.pd_prefill_hosts or a.pd_decode_hosts:
        if not (a.pd_prefill_hosts >= 1 and a.pd_decode_hosts >= 1):
            ap.error("P/D mode needs both --pd-prefill-hosts and --pd-decode-hosts")
        if not a.exec_cmd:
            ap.error("P/D mode needs --exec")
        run_name = dns_safe(a.name or "pd")
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
        # Both roles draw from the same discovered pool set (--pools already unions
        # topology.yaml's entry); Kueue's per-PodSet flavor is what separates the
        # two roles onto different slices within it.
        pd_pools = a.pools.split() or list(
            dict.fromkeys(prefill_slice["pools"] + decode_slice["pools"])
        )
        pvc = a.weights_pvc
        vals = {
            "IMAGE": a.image,
            "ACTIVE_DEADLINE": str(_PD_ACTIVE_DEADLINE),
            "EXEC_CMD": a.exec_cmd,
            "RUN_NAME": run_name,
            "EXCLUSIVE_PLACEMENT": exclusive if multihost else "",
            "PREFILL_HOSTS": str(a.pd_prefill_hosts),
            "DECODE_HOSTS": str(a.pd_decode_hosts),
            "QUEUE": prefill_slice["queue"],
            "PREFILL_TOPOLOGY": prefill_slice["topology"],
            "PREFILL_POOL_AFFINITY": POOL_AFFINITY_TMPL.format(
                pools=", ".join(pd_pools)
            ),
            "DECODE_TOPOLOGY": decode_slice["topology"],
            "DECODE_POOL_AFFINITY": POOL_AFFINITY_TMPL.format(
                pools=", ".join(pd_pools)
            ),
            "POOL_LIST": " ".join(pd_pools),
            "WEIGHTS_VOLUME": WEIGHTS_VOLUME_TMPL.format(pvc=pvc) if pvc else "",
            "WEIGHTS_MOUNT": WEIGHTS_MOUNT_TMPL if pvc else "",
        }
        tpl = open(f"{a.dir}/jobset-tpu-pd.yaml").read()
        out = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), tpl)
        sys.stdout.write(out)
        return

    if a.launch_script:
        if a.hosts < 1:
            ap.error("--hosts must be a positive integer for multi-host mode")
        hosts = a.hosts
        run_name = dns_safe(a.name or a.launch_script)
        suite = ""
        if not a.files:
            sys.exit("multi-host mode needs --files")
    elif a.exec_cmd:
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
        run_name = dns_safe(a.suite)
        suite = a.suite
        if a.timeout_per_file <= 0:
            ap.error("--timeout-per-file must be a positive integer")

    sl = slice_for(hosts)

    # Discovered set when --pools is passed (it already unions in topology.yaml's entry).
    pools = a.pools.split() or sl["pools"]

    # Suite mode runs every registered file sequentially, so its deadline scales
    # with the suite (same est_time formula as _tpu-test's poll budget, keeping the
    # two in agreement). Exec gets 7200: the combo rows budget WAIT_SECS=2400 for
    # engine bring-up alone.
    if suite:
        active_deadline = suite_wait_seconds(suite, a.tests_dir)
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
        "POOL_AFFINITY": POOL_AFFINITY_TMPL.format(pools=", ".join(pools)),
        "POOL_LIST": " ".join(pools),
        "QUEUE": sl["queue"],
        "RUN_NAME": run_name,
        "SUITE": suite,
        # --continue-on-error only on the nightly, matching the CUDA nightly: a
        # scheduled run should report every file's verdict, where the PR gate still
        # fast-fails to give the slice back. Without it the suite stops at the first
        # failure and the files behind it never report at all.
        "NIGHTLY_FLAG": "--nightly --continue-on-error" if a.nightly else "",
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
