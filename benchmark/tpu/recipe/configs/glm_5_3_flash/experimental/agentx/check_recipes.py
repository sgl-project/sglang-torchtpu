"""Render every AgentX recipe and diff it against the launch JSON of its point.

usage, from the repo root:
    python3 benchmark/tpu/recipe/configs/glm_5_3_flash/experimental/agentx/check_recipes.py

glm_5_3_flash/agentx.py: each profile must render its MEASURED run JSON exactly.
agentx_cc<N>.py (the earlier dummy-weight points): AGENTX_DUMMY=1 must render the run
JSON exactly, and the submission render (real weights) may differ only by
SUBMISSION_ONLY. Exits 1 on any other difference.
"""

import importlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
PACKAGE = "benchmark.tpu.recipe.configs.glm_5_3_flash.experimental.agentx"
# Recipe -> launch JSON of the run behind its point.
POINTS = {
    "agentx_cc1": "R1b_c1",
    "agentx_cc4": "C4u_c4",
    "agentx_cc8": "K2ts_c8",
    "agentx_cc16": "C16_S2ts",
    "agentx_cc16_kda1400": "C16_S2ts_kda1400",
    "agentx_cc32": "R5_c32_ts",
}
ABSENT = "<absent>"
# (section, key) -> (run JSON value, submission value): real weights, not load_format dummy.
SUBMISSION_ONLY = {
    ("server_args", "load_format"): ("dummy", ABSENT),
    ("server_args", "weight_loader_prefetch_checkpoints"): (ABSENT, True),
    ("server_args", "weight_loader_drop_cache_after_load"): (ABSENT, True),
    ("runtime_env", "SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S"): (ABSENT, "2400"),
}


def render(recipe: str, *, dummy: bool) -> dict:
    saved = os.environ.get("AGENTX_DUMMY")
    os.environ["AGENTX_DUMMY"] = "1" if dummy else "0"
    try:
        module = importlib.reload(importlib.import_module(f"{PACKAGE}.{recipe}"))
        return json.loads(json.dumps(module.CONFIG))
    finally:
        if saved is None:
            os.environ.pop("AGENTX_DUMMY")
        else:
            os.environ["AGENTX_DUMMY"] = saved


def diff(run: dict, rendered: dict) -> dict:
    out = {}
    for section in sorted(set(run) | set(rendered)):
        a, b = run.get(section, {}), rendered.get(section, {})
        for key in sorted(set(a) | set(b)):
            if a.get(key, ABSENT) != b.get(key, ABSENT):
                out[(section, key)] = (a.get(key, ABSENT), b.get(key, ABSENT))
    return out


def check(recipe: str, run_name: str) -> tuple[list[str], bool]:
    """Report lines and whether the recipe passes."""
    run = json.loads((HERE / "runs" / f"{run_name}.json").read_text())
    lines = [f"{recipe}.py vs runs/{run_name}.json"]
    ok = run["server_args"]["model_path"] == "/weights/GLM-5.3-Flash"
    for mode, dummy in (("AGENTX_DUMMY=1", True), ("submission", False)):
        rendered = render(recipe, dummy=dummy)
        found = diff(run, rendered)
        unexpected = (
            found
            if dummy
            else {k: v for k, v in found.items() if SUBMISSION_ONLY.get(k) != v}
        )
        missing = [] if dummy else [k for k in SUBMISSION_ONLY if k not in found]
        ok &= not unexpected and not missing
        sizes = f"{len(rendered['server_args'])} server args, {len(rendered['runtime_env'])} env"
        lines.append(
            f"  {mode}: {'identical' if not found else f'{len(found)} differences'} ({sizes})"
        )
        for (section, key), (a, b) in found.items():
            tag = "UNEXPECTED" if (section, key) in unexpected else "submission-only"
            lines.append(f"    {section}.{key}: run {a!r} -> recipe {b!r} [{tag}]")
        for section, key in missing:
            lines.append(
                f"    {section}.{key}: missing submission-only difference [UNEXPECTED]"
            )
    return lines, ok


def _as_json(config: dict) -> dict:
    """Values as JSON text: 2 and 2.0, or True and 1, launch different flags."""
    return {s: {k: json.dumps(v) for k, v in d.items()} for s, d in config.items()}


def check_profile(conc: int) -> tuple[list[str], bool]:
    """agentx.py --conc <conc> against the run JSON it was measured with."""
    from benchmark.tpu.recipe.configs.glm_5_3_flash.agentx import MEASURED, config_for

    run_name = MEASURED[conc]
    run = json.loads((HERE / "runs" / f"{run_name}.json").read_text())
    rendered = config_for(conc)
    found = diff(_as_json(run), _as_json(rendered))
    sizes = f"{len(rendered['server_args'])} server args, {len(rendered['runtime_env'])} env"
    lines = [
        f"agentx.py --conc {conc} vs runs/{run_name}.json: "
        + ("identical" if not found else f"{len(found)} differences")
        + f" ({sizes})"
    ]
    for (section, key), (a, b) in found.items():
        lines.append(f"    {section}.{key}: run {a!r} -> agentx.py {b!r} [UNEXPECTED]")
    return lines, not found


def main() -> int:
    sys.path.insert(0, str(HERE.parents[6]))
    from benchmark.tpu.recipe.configs.glm_5_3_flash.agentx import MEASURED

    failed = []
    for conc in MEASURED:
        lines, ok = check_profile(conc)
        print("\n".join(lines))
        if not ok:
            failed.append(f"agentx.py --conc {conc}")
    for recipe, run_name in POINTS.items():
        lines, ok = check(recipe, run_name)
        print("\n".join(lines))
        if not ok:
            failed.append(recipe)
    total = len(MEASURED) + len(POINTS)
    print(f"FAIL: {', '.join(failed)}" if failed else f"OK: {total} configs")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
