#!/usr/bin/env python3
"""Keep TPU CI Kubernetes object names inside the 63-char cap.

`run_name` prefixes every object a leg creates:

    tpu-ci-<run_name>-<5 random>-<replicated job>-<index>

which is a DNS label, capped at 63 chars. Overrun it and the apiserver rejects
the JobSet at admission -- after the queue wait, on a nightly nobody is
watching. So measure it here, statically, instead of at 3am.

A `run_name` this script cannot measure is an error too: the discovery scripts
hand every leg a compact `run_name` (12 hex chars) alongside the file stem, and
passing the stem is exactly how a leg overruns the cap.
"""

import glob
import itertools
import re
import sys

import yaml

CALLEE = "_tpu-test.yml"

# The shape render_jobset.py + jobset-tpu{,-pd}.yaml give the objects.
NAME_CAP = 63
RANDOM_SUFFIX = "x" * 5
SLICE_ROLES = ("slice",)
PD_ROLES = ("prefill", "decode")

# An expression ending in `.run_name` is discovery's compact name
# (test_names.names_from_path): bounded, though only known at run time.
COMPACT_RUN_NAME = "0" * 12

MATRIX_REF = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_.]+)\s*\}\}")
# A matrix key can be a plain list, or an expression whose values are spelled
# out inline (perf picks its suite with a `fromJSON('["..."]')` branch). Read
# only the fromJSON payloads: the rest of such an expression holds literals that
# are conditions, not values (`inputs.model == 'gpt-oss-120b'`).
FROM_JSON = re.compile(r"fromJSON\(\s*'([^']*)'\s*\)")
INLINE_LITERAL = re.compile(r"""["']([a-z0-9][a-z0-9._-]*)["']""")


def object_name(run_name: str, role: str) -> str:
    """The longest Kubernetes object name this run_name produces."""
    return f"tpu-ci-{run_name}-{RANDOM_SUFFIX}-{role}-0"


def inline_literals(text: str) -> list:
    """The values an expression spells out inline, in order, deduplicated."""
    payloads = FROM_JSON.findall(text)
    found = INLINE_LITERAL.findall(" ".join(payloads) if payloads else text)
    return list(dict.fromkeys(found))


def matrix_legs(job: dict) -> list:
    """The assignments a job's matrix can take, as {key: value} dicts.

    `include:` legs are whole assignments; a plain matrix is the product of its
    keys. A key whose values are not spelled out in the YAML (a matrix built by
    `fromJSON` of a discovery script) is left out, so `expand` reports the
    reference it could not resolve.
    """
    matrix = ((job.get("strategy") or {}).get("matrix")) or {}
    if not isinstance(matrix, dict):
        return []
    legs = [leg for leg in (matrix.get("include") or []) if isinstance(leg, dict)]
    if legs:
        return legs
    keys, values = [], []
    for key, value in matrix.items():
        if isinstance(value, list):
            candidates = [str(v) for v in value if isinstance(v, (str, int, float))]
        else:
            candidates = inline_literals(str(value))
        if not candidates:
            continue
        keys.append(key)
        values.append(candidates)
    if not keys:
        return []
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def expand(value, legs: list):
    """The literal values `value` can take, or None if that is not knowable."""
    value = str(value)
    if "${{" not in value:
        return [value]
    resolved = []
    for leg in legs:
        candidate = MATRIX_REF.sub(
            lambda m: str(leg.get(m.group(1), m.group(0))), value
        )
        if "${{" not in candidate:
            resolved.append(candidate)
    return resolved or None


def check_job(path: str, job_id: str, job: dict) -> list:
    """Errors for one job that calls _tpu-test.yml."""
    with_ = job.get("with") or {}
    # render_jobset.py falls back to the suite when no run_name is passed.
    source = "run_name" if with_.get("run_name") else "suite"
    value = str(with_.get(source) or "")
    if not value:
        return []  # a launch/exec leg named after its --files; not measured here
    where = f"{path}: job `{job_id}`"

    candidates = expand(value, matrix_legs(job))
    if candidates is None:
        if not value.endswith(".run_name }}"):
            return [
                f"{where} takes `{source}` from {value}, whose length is not "
                f"knowable here, so nothing stops it from overrunning the "
                f"{NAME_CAP}-char object-name cap. Pass a literal, or the "
                f"compact `.run_name` the discovery scripts emit."
            ]
        candidates = [COMPACT_RUN_NAME]

    errors = []
    roles = PD_ROLES if with_.get("pd_prefill_hosts") else SLICE_ROLES
    for candidate, role in itertools.product(candidates, roles):
        rendered = object_name(candidate, role)
        if len(rendered) > NAME_CAP:
            errors.append(
                f"{where} would name objects {rendered} ({len(rendered)} chars "
                f"> {NAME_CAP}); shorten `{source}` by {len(rendered) - NAME_CAP} "
                f"chars or pass a compact `run_name`."
            )
    return errors


def check_workflows(workflows: dict) -> list:
    """Errors across {path: parsed workflow}."""
    errors = []
    for path, data in sorted(workflows.items()):
        for job_id, job in ((data or {}).get("jobs") or {}).items():
            if str(job.get("uses") or "").endswith(CALLEE):
                errors.extend(check_job(path, job_id, job))
    return errors


def main() -> int:
    workflows = {}
    for path in sorted(glob.glob(".github/workflows/*.yml")):
        with open(path, encoding="utf-8") as f:
            workflows[path] = yaml.safe_load(f)

    errors = check_workflows(workflows)
    if not errors:
        return 0
    print("ERROR: TPU CI object names (see .github/tpu-ci/README.md):\n")
    for error in errors:
        print(f"  - {error}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
