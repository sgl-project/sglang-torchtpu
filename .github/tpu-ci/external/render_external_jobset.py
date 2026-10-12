"""Render the external TPU JobSets, and discover the `external-nightly-*` work for the
nightly matrix. Single source of truth for rendering onto the external TPU cluster -- used by
_external-tpu-test.yml and runnable locally for dry-runs. See external-topology.yaml +
README.md.

REUSE: the registration scan (`discover_multihost._scan`, incl. its `disabled` handling) and
the path-derived naming (`test_names.names_from_path`, from #552) are imported from the
in-project CI helpers, so improvements to those land here too. What stays unique to the
external path lives in this file: slice bucketing (no Kueue), gcsfuse weights, the
nodepool-pin JobSets, and the DNS-endpoint gating (in _external-tpu-test.yml).

Three kinds of external work, by where they are declared (the suite fixes the slice shape):

    launch  -- a test registered to an external suite with a module-level
               MULTIHOST_LAUNCH = {"launch_script": "recipe/x.sh", "weights": "<bucket subdir>"}:
               every host runs the recipe; rank 0 runs the file against the live server.
               Files sharing (suite, launch_script, weights) share one server.
    self    -- a test registered to a suite with `self_launch: true`: rank 0 runs the file
               through run_suite; the test starts its own server.
    pd/exec -- a suite's `commands` in external-topology.yaml, wired there as in-project's
               feature-validation-tpu.yml wires its static jobs: every pod runs `cmd`. A `pd`
               entry gets prefill and decode replicated jobs with CI_PD_ROLE set; 1+1 shares
               one slice, N+N with N == the slice's hosts takes a PAIR of slices.

Every pod sees the weights bucket at /weights (as in-project), so tests that hard-code
/weights/<model> paths run unchanged.

Modes:
  --list-groups [--nightly]
      Scan test/registered/**.py (AST only) and bucket the work ONE matrix row PER SLICE
      (keyed by nodepool, since several suites share the same slices): each row carries its
      ordered `groups`, run back-to-back on that slice. P/D groups that need two slices get
      rows of their own, one per slice PAIR:
        [{"name","suite","hosts","topology","nodepool_var",["nodepool_var_decode"],"groups"}]
      `nodepool_var` is the NAME of the repo secret holding that slice's nodepool; the
      workflow resolves it so the value never appears in source or the matrix.
  render (default)
      Render ONE group's JobSet (the workflow calls this once per group in a row):
      --image <ref> --group '<group json>' --nodepool <value> [--nodepool-decode <value>]

Usage: render_external_jobset.py --list-groups
       render_external_jobset.py --image <ref> --group "$(jq -c '.[0].groups[0]' <<<"$rows")" \
           --nodepool <value> > job.yaml
"""

import argparse
import json
import os
import re
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Reuse the in-project CI helpers (external/ is a subdir, so put the parent tpu-ci dir on the
# path). _scan gives us registration + `disabled` handling; names_from_path gives the #552
# path-derived display/compact names. Importing here means future changes to them flow through.
sys.path.insert(0, os.path.dirname(HERE))
from discover_multihost import _scan as _scan_suite  # noqa: E402
from test_names import names_from_path  # noqa: E402

# Bring-up allowance before any test file's own budget counts: gcsfuse weight
# load + engine compile warmup, measured 15-30 min on the benchmark groups.
_BRINGUP_ALLOWANCE = 3600
# Floor for the JobSet deadline (the pre-benchmark fixed value): ~30m bring-up
# + rank-0's test files + margin. Short groups keep at least this much.
_ACTIVE_DEADLINE = 9000
# Same budgets as the in-project render_jobset.py for the modes it shares.
_PD_TIMEOUT = 8700
_EXEC_TIMEOUT = 7200
_SELF_TIMEOUT_PER_FILE = 3600
# DNS-1035 label budget for the run name (children append suffixes).
_RUN_NAME_MAX = 40


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def dns_safe(name):
    s = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")
    s = (s[:_RUN_NAME_MAX].strip("-")) or "run"
    # The JobSet webhook enforces DNS-1035 on the derived child Job names,
    # which must START WITH A LETTER -- and the compact run_name is a sha256
    # prefix that can start with a digit (bit us with '24afde9fbae2...').
    if not s[0].isalpha():
        s = "g" + s[: _RUN_NAME_MAX - 1]
    return s


def _scan(cfg, root, nightly_only):
    """[(path, suite, mode, spec)] for every external test, sorted by path."""
    found = []
    for suite, sl in cfg["suites"].items():
        if sl.get("self_launch"):
            for path in _scan_suite(root, suite, nightly_only, launch_var=None):
                found.append((path, suite, "self", {}))
            continue
        for path, spec in _scan_suite(root, suite, nightly_only).items():
            found.append((path, suite, "launch", spec))
        for spec in sl.get("commands", []):
            found.append((spec["files"], suite, "pd" if "pd" in spec else "exec", spec))
    return sorted(found)


def _launch_groups(entries):
    """Launch-mode files sharing (suite, launch_script, weights) share one server."""
    shared = {}
    for path, suite, _, spec in entries:
        ls, weights = spec.get("launch_script"), spec.get("weights")
        if not ls or not weights:
            print(
                f"skip {path}: MULTIHOST_LAUNCH needs launch_script + weights, got {spec!r}",
                file=sys.stderr,
            )
            continue
        shared.setdefault((suite, ls, weights), []).append(
            (path, int(spec.get("timeout_per_file", 3600)))
        )
    groups = []
    for (suite, ls, weights), files in sorted(shared.items()):
        paths = sorted(path for path, _ in files)
        groups.append(
            {
                **names_from_path(paths[0]),  # #552: path-derived name + run_name
                "mode": "launch",
                "suite": suite,
                "launch_script": ls,
                "weights": weights,
                "files": " ".join(paths),
                # The group's per-file budget is the largest any of its files
                # declares; the workflow passes it back to render so the
                # JobSet deadline is sized from what actually runs.
                "timeout_per_file": max(tpf for _, tpf in files),
            }
        )
    return groups


def _cmd_groups(entries, mode, slice_hosts):
    """One group per P/D or exec file; P/D groups say whether they need a slice pair."""
    groups = []
    for path, suite, _, spec in entries:
        cmd = spec.get("cmd")
        if not cmd:
            print(
                f"skip {path}: {mode} launch needs `cmd`, got {spec!r}", file=sys.stderr
            )
            continue
        group = {
            **names_from_path(path),
            "mode": mode,
            "suite": suite,
            "cmd": cmd,
            "files": path,
            "timeout_per_file": int(
                spec.get("timeout", _PD_TIMEOUT if mode == "pd" else _EXEC_TIMEOUT)
            ),
        }
        if mode == "pd":
            p, d = int(spec["pd"]["prefill_hosts"]), int(spec["pd"]["decode_hosts"])
            hosts = slice_hosts[suite]
            # A role either shares the slice as one single-host engine per pod, or fills it.
            if p + d > hosts and not p == d == hosts:
                print(
                    f"skip {path}: P/D {p}+{d} fits no {hosts}-host slice",
                    file=sys.stderr,
                )
                continue
            group.update(prefill_hosts=p, decode_hosts=d, pair=p + d > hosts)
        groups.append(group)
    return groups


def _self_groups(entries, cfg):
    """Self-launch files, dealt across their suite's slices: one bundle per slice."""
    by_suite = {}
    for path, suite, _, _ in entries:
        by_suite.setdefault(suite, []).append(path)
    groups = []
    for suite, paths in sorted(by_suite.items()):
        n = len(cfg["suites"][suite]["nodepool_vars"])
        for k in range(min(n, len(paths))):
            bundle = paths[k::n]
            groups.append(
                {
                    "name": f"{suite}-bundle-{k}",
                    "run_name": dns_safe(f"selfhost-{k}"),
                    "mode": "self",
                    "suite": suite,
                    "files": " ".join(bundle),
                    "timeout_per_file": _SELF_TIMEOUT_PER_FILE,
                }
            )
    return groups


def list_groups(cfg, root, nightly_only):
    suites = cfg["suites"]
    entries = _scan(cfg, root, nightly_only)

    def of(mode):
        return [e for e in entries if e[2] == mode]

    hosts = {s: int(sl["hosts"]) for s, sl in suites.items()}
    groups = (
        _launch_groups(of("launch"))
        + _cmd_groups(of("pd"), "pd", hosts)
        + _cmd_groups(of("exec"), "exec", hosts)
        + _self_groups(of("self"), cfg)
    )

    # Bucket by SLICE, not by suite: suites share the same nodepools, and two matrix jobs on
    # one slice would race for it (the slice-keyed concurrency group keeps only one pending).
    # Groups are dealt round-robin across their suite's slices with one counter per slice
    # set, so suites sharing slices share the rotation. There is no Kueue to spread work.
    rows, counters = {}, {}
    # A row is named for the slice shape: the first suite (config order) on its slices.
    owner = {}
    for name, sl in suites.items():
        owner.setdefault(tuple(sl["nodepool_vars"]), name)

    def row(key, suite, **extra):
        suite = owner[tuple(suites[suite]["nodepool_vars"])]
        sl = suites[suite]
        return rows.setdefault(
            key,
            {
                "name": key,
                "suite": suite,
                "hosts": sl["hosts"],
                "topology": sl["topology"],
                **extra,
                "groups": [],
            },
        )

    for g in groups:
        sl = suites[g["suite"]]
        pool = tuple(sl["nodepool_vars"])
        prefix = f"external-nightly-{sl['topology']}"
        if g.get("pair"):
            # Prefill and decode each fill a slice: deal the group across slice PAIRS.
            pairs = [(i, i + 1) for i in range(0, len(pool) - 1, 2)]
            i = counters.get(("pair", pool), 0)
            counters[("pair", pool)] = i + 1
            a, b = pairs[i % len(pairs)]
            r = row(
                f"{prefix}-p{a}{b}",
                g["suite"],
                nodepool_var=pool[a],
                nodepool_var_decode=pool[b],
            )
        else:
            i = counters.get(pool, 0)
            counters[pool] = i + 1
            slot = i % len(pool)
            r = row(f"{prefix}-s{slot}", g["suite"], nodepool_var=pool[slot])
        r["groups"].append(g)
    return [rows[k] for k in sorted(rows)]


def _active_deadline(group):
    n_files = len(group["files"].split())
    tpf = int(group["timeout_per_file"])
    if group["mode"] in ("pd", "exec"):
        return tpf
    # Size the JobSet deadline from what actually runs: ~1h of bring-up
    # (gcsfuse weight load + compile warmup) plus every file's own budget,
    # floored at the old fixed value so short groups keep their margin.
    return max(_ACTIVE_DEADLINE, _BRINGUP_ALLOWANCE + n_files * tpf)


def render(cfg, a):
    g = json.loads(a.group)
    suites = cfg["suites"]
    if g.get("suite") not in suites:
        sys.exit(
            f"unknown suite {g.get('suite')!r}; external-topology.yaml has {sorted(suites)}"
        )
    if not a.nodepool:
        sys.exit(
            "render needs --nodepool (the resolved slice nodepool, from a repo secret)"
        )

    sl = suites[g["suite"]]
    cl = cfg["cluster"]
    w = cfg["weights"]
    hosts = int(sl["hosts"])
    mode = g["mode"]
    vals = {
        "IMAGE": a.image,
        # The group's compact run_name (names_from_path) -> collision-resistant.
        "RUN_NAME": dns_safe(g.get("run_name") or g["suite"]),
        "NAMESPACE": cl["namespace"],
        "TOPOLOGY": sl["topology"],
        "ACCELERATOR": cl["accelerator"],
        "ACTIVE_DEADLINE": str(a.active_deadline or _active_deadline(g)),
        "TEST_FILES": g["files"],
        "TIMEOUT_PER_FILE": str(g["timeout_per_file"]),
        "WEIGHTS_BUCKET": w["bucket"],
        "WEIGHTS_MOUNT_OPTIONS": w["mount_options"],
        "WEIGHTS_MOUNT_PATH": w["mount_path"],
        "WEIGHTS_DIR": f"{w['mount_path']}/{w['prefix']}",
        "EPHEMERAL_LIMIT": w["ephemeral_storage_limit"],
        "EXEC_CMD": g.get("cmd", ""),
    }

    if mode == "pd":
        p, d = int(g["prefill_hosts"]), int(g["decode_hosts"])
        if g.get("pair") and not a.nodepool_decode:
            sys.exit("a slice-pair P/D group needs --nodepool-decode")
        vals.update(
            PREFILL_HOSTS=str(p),
            DECODE_HOSTS=str(d),
            PREFILL_NODEPOOL=a.nodepool,
            DECODE_NODEPOOL=a.nodepool_decode if g.get("pair") else a.nodepool,
            # A role smaller than its slice runs single-host engines: drop the slice env.
            PREFILL_SINGLE_HOST="1" if p < hosts else "",
            DECODE_SINGLE_HOST="1" if d < hosts else "",
        )
        template = "jobset-external-tpu-pd.yaml"
    else:
        tp = hosts * int(cl["devices_per_host"])
        weights = g.get("weights", "")
        vals.update(
            MODE=mode,
            HOSTS=str(hosts),
            NODEPOOL=a.nodepool,
            SUITE=g["suite"],
            LAUNCH_SCRIPT=g.get("launch_script", ""),
            MODEL_PATH=f"{vals['WEIGHTS_DIR']}/{weights}" if weights else "",
            TP_SIZE=str(tp),
            EP_SIZE=str(tp),
            EXTRA_ARGS=a.extra_args or w.get("default_extra_args", ""),
        )
        template = "jobset-external-tpu.yaml"

    tpl = open(f"{HERE}/{template}").read()
    out = re.sub(r"\$\{(\w+)\}", lambda m: vals.get(m.group(1), m.group(0)), tpl)
    sys.stdout.write(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{HERE}/external-topology.yaml")
    ap.add_argument("--tests-dir", default="test/registered")
    ap.add_argument(
        "--list-groups",
        action="store_true",
        help="Print the JSON per-slice matrix for the external suites and exit.",
    )
    ap.add_argument("--nightly", action="store_true", default=True)
    # render mode (one group)
    ap.add_argument("--image")
    ap.add_argument("--group", default="", help="One group object from --list-groups.")
    ap.add_argument(
        "--nodepool",
        default="",
        help="Resolved slice nodepool (the workflow reads it from the row's repo secret).",
    )
    ap.add_argument(
        "--nodepool-decode",
        default="",
        help="Decode slice's nodepool, for a P/D group on a slice pair.",
    )
    ap.add_argument("--extra-args", default="")
    ap.add_argument(
        "--active-deadline",
        type=int,
        default=None,
        help="JobSet activeDeadlineSeconds; default sized per mode from the group.",
    )
    a = ap.parse_args()

    cfg = load_config(a.config)

    if a.list_groups:
        print(json.dumps(list_groups(cfg, a.tests_dir, a.nightly)))
        return

    if not a.image or not a.group:
        ap.error("--image and --group are required when rendering")
    render(cfg, a)


if __name__ == "__main__":
    main()
