#!/usr/bin/env python3
"""Check the two names every TPU CI leg carries (.github/tpu-ci/README.md).

GitHub renders a called job as "<caller job> / <called job>", so a leg of a TPU
stage workflow is named twice, and the two names are not interchangeable:

  * `display_name` is the LAST segment of the check name -- the half a reader's
    eye lands on and the half that survives a truncated checks list. A caller
    that forgets it leaves `_tpu-test.yml`'s fallback showing, so a red leg ends
    with the literal word `test`, naming nothing.

  * `run_name` prefixes every Kubernetes object the leg creates:

        tpu-ci-<run_name>-<5 random>-<replicated job>-<index>

    against the 63-char DNS label cap. An over-long run_name is rejected at
    admission -- after the queue wait, on a nightly nobody is watching -- so the
    budget is checked here, statically, instead of at 3am.

Both regress silently and neither is covered by the dashboard contract lint
(series identity is the metric `id`, not a job or object name), hence this file.
"""

import glob
import itertools
import re
import sys

import yaml

CALLEE = "_tpu-test.yml"

# k8s DNS label cap, and the shape render_jobset.py + jobset-tpu{,-pd}.yaml give
# the objects: generateName `tpu-ci-<run_name>-`, 5 random chars, then the
# replicated job and its index.
NAME_CAP = 63
RANDOM_SUFFIX = "x" * 5
SLICE_ROLES = ("slice",)
PD_ROLES = ("prefill", "decode")

# discover_tests.py / discover_multihost.py hand a leg a compact `run_name`
# (test_names.names_from_path: 12 hex chars), so an expression ending in
# `.run_name` is bounded even though its value is only known at run time.
COMPACT_RUN_NAME = "0" * 12


def object_name(run_name: str, role: str) -> str:
    """The longest Kubernetes object name this run_name produces."""
    return f"tpu-ci-{run_name}-{RANDOM_SUFFIX}-{role}-0"


MATRIX_REF = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_.]+)\s*\}\}")
# A matrix key can be a plain list, or an expression whose values are spelled
# out inline (perf picks its suite with a `fromJSON('["..."]')` branch). Read
# only the fromJSON payloads: the rest of such an expression holds literals that
# are conditions, not values (`inputs.model == 'gpt-oss-120b'`).
FROM_JSON = re.compile(r"fromJSON\(\s*'([^']*)'\s*\)")
INLINE_LITERAL = re.compile(r"""["']([a-z0-9][a-z0-9._-]*)["']""")


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
    return (
        [dict(zip(keys, combo)) for combo in itertools.product(*values)] if keys else []
    )


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
    errors = []
    with_ = job.get("with") or {}
    display_name = str(with_.get("display_name") or "")
    suite = str(with_.get("suite") or "")
    run_name = str(with_.get("run_name") or "")
    where = f"{path}: job `{job_id}`"

    if not display_name and not suite:
        errors.append(
            f"{where} passes neither `display_name` nor `suite`, so its check "
            f"name ends with `test` and names no test. Pass the test this leg "
            f"runs as `display_name`."
        )

    name = str(job.get("name") or "")
    if display_name and name == display_name:
        errors.append(
            f"{where} has the same `name` and `display_name` ({name!r}), so the "
            f"check reads `x / x`. The job name holds the compact stage tag; "
            f"`display_name` holds the test."
        )
    if name and "${{" in name:
        errors.append(
            f"{where} computes its `name` ({name}). The caller's name is the "
            f"stage tag and must be a short literal; per-leg detail belongs in "
            f"`display_name`."
        )

    # `run_name` names the k8s objects; render_jobset.py falls back to the suite.
    source, value = ("run_name", run_name) if run_name else ("suite", suite)
    if not value:
        return errors  # a launch/exec leg naming itself after --files; not ours
    legs = matrix_legs(job)
    candidates = expand(value, legs)
    if candidates is None:
        if value.endswith(".run_name }}"):
            candidates = [COMPACT_RUN_NAME]  # discovery's compact name
        else:
            errors.append(
                f"{where} takes `{source}` from {value}, whose length is not "
                f"knowable here, so nothing stops it from overrunning the "
                f"{NAME_CAP}-char object-name cap. Pass a literal, or the "
                f"compact `.run_name` the discovery scripts emit."
            )
            return errors

    roles = PD_ROLES if with_.get("pd_prefill_hosts") else SLICE_ROLES
    for candidate in candidates:
        for role in roles:
            rendered = object_name(candidate, role)
            if len(rendered) > NAME_CAP:
                errors.append(
                    f"{where} would name objects {rendered} "
                    f"({len(rendered)} chars > {NAME_CAP}); shorten `{source}` "
                    f"by {len(rendered) - NAME_CAP} chars or pass a compact "
                    f"`run_name`."
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
    print("ERROR: TPU CI job naming (see .github/tpu-ci/README.md):\n")
    for error in errors:
        print(f"  - {error}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
