"""Generate monitor and visualization artifacts from LTL properties."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent

try:
    import buddy
    import spot
except ModuleNotFoundError as error:
    if error.name not in {"buddy", "spot"}:
        raise
    raise SystemExit(
        "Spot Python bindings are required to generate monitors. Install them with:\n"
        "  python3 -m pip install "
        "http://www.lre.epita.fr/dload/spot/spot-2.16.tar.gz"
    ) from error

TOOL_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOL_DIR.parent.parent
RUNTIME_DIR = REPO_ROOT / "python" / "sglang" / "srt" / "raiden_runtime_verification"
DEFAULT_LTL_PATH = TOOL_DIR / "formulas.ltl"
DEFAULT_HOA_PATH = TOOL_DIR / "generated_runtime_monitor.hoa"
DEFAULT_MONITOR_PY_PATH = RUNTIME_DIR / "generated_runtime_monitor.py"
DEFAULT_MONITOR_DOT_PATH = TOOL_DIR / "generated_runtime_monitor.dot"
DEFAULT_MONITOR_SVG_PATH = TOOL_DIR / "generated_runtime_monitor.svg"


def _preprocess_formulas(path: Path) -> str:
    cpp = shutil.which("cpp")
    if cpp is None:
        raise RuntimeError("C preprocessor 'cpp' was not found")

    result = subprocess.run(
        [cpp, "-P", "-undef", "-nostdinc", str(path)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or f"cpp exited with status {result.returncode}"
        raise RuntimeError(f"failed to preprocess {path}: {message}")
    return result.stdout


def _read_formulas(path: Path) -> list[spot.formula]:
    formulas = []
    for raw_line in _preprocess_formulas(path).splitlines():
        line = raw_line.strip()
        if line:
            formulas.append(spot.formula(line))
    if not formulas:
        raise ValueError(f"formula file {path} contains no formulas")
    return formulas


def _optional_path(value: str) -> Path | None:
    return Path(value) if value else None


def _repo_relative_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def _format_atomic_propositions(values: tuple[str, ...]) -> str:
    lines = ["("]
    lines.extend(f"    {json.dumps(value)}," for value in values)
    lines.append(")")
    return "\n".join(lines)


def _format_condition_nodes(nodes: dict[int, tuple[int, int, int]]) -> str:
    lines = ["{"]
    lines.extend(
        f"    {node}: ({proposition}, {low}, {high}),"
        for node, (proposition, low, high) in sorted(nodes.items())
    )
    lines.append("}")
    return "\n".join(lines)


def _format_transitions(transitions: dict[int, list[tuple[int, int]]]) -> str:
    lines = ["{"]
    for state, edges in sorted(transitions.items()):
        if len(edges) == 1:
            condition, destination = edges[0]
            lines.append(f"    {state}: (({condition}, {destination}),),")
            continue
        lines.append(f"    {state}: (")
        lines.extend(
            f"        ({condition}, {destination})," for condition, destination in edges
        )
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines)


def _generate_python_monitor(automaton: spot.twa_graph, ltl_path: Path) -> str:
    if not automaton.is_deterministic():
        raise RuntimeError("cannot generate code for a nondeterministic monitor")

    dictionary = automaton.get_dict()
    propositions = list(automaton.ap())
    proposition_names = tuple(str(ap) for ap in propositions)
    proposition_indices = {
        dictionary.varnum(ap): index for index, ap in enumerate(propositions)
    }

    condition_ids: dict[buddy.bdd, int] = {}
    condition_nodes: dict[int, tuple[int, int, int]] = {}

    def encode_condition(condition: buddy.bdd) -> int:
        if condition == buddy.bddfalse:
            return 0
        if condition == buddy.bddtrue:
            return 1
        if condition in condition_ids:
            return condition_ids[condition]

        condition_id = len(condition_ids) + 2
        condition_ids[condition] = condition_id
        condition_nodes[condition_id] = (
            proposition_indices[buddy.bdd_var(condition)],
            encode_condition(buddy.bdd_low(condition)),
            encode_condition(buddy.bdd_high(condition)),
        )
        return condition_id

    transitions = {
        state: [
            (encode_condition(edge.cond), edge.dst) for edge in automaton.out(state)
        ]
        for state in range(automaton.num_states())
    }

    return dedent('''\
        """Generated from {ltl_name}. Do not edit by hand."""

        import logging

        logger = logging.getLogger(__name__)

        _ATOMIC_PROPOSITIONS: tuple[str, ...] = {atomic_propositions}
        _INITIAL_STATE: int = {initial_state}
        _CONDITION_NODES: dict[int, tuple[int, int, int]] = {condition_nodes}
        _TRANSITIONS: dict[int, tuple[tuple[int, int], ...]] = {transitions}


        def _condition_holds(condition: int, valuation: int) -> bool:
            while condition > 1:
                proposition, low, high = _CONDITION_NODES[condition]
                condition = high if valuation & (1 << proposition) else low
            return condition == 1


        class GeneratedRuntimeMonitor:
            def __init__(self, *, trace_id: object | None = None) -> None:
                self.trace_id = trace_id
                self.state = _INITIAL_STATE
                self.violated = False

            @property
            def atomic_propositions(self) -> tuple[str, ...]:
                return _ATOMIC_PROPOSITIONS

            def step(self, true_propositions: set[str]) -> bool:
                if self.violated:
                    logger.info(
                        "GeneratedRuntimeMonitor trace_id=%s state=%s "
                        "atomic_propositions=%s accepted=False already_violated=True",
                        self.trace_id,
                        self.state,
                        sorted(true_propositions),
                    )
                    return False

                unknown = true_propositions - set(_ATOMIC_PROPOSITIONS)
                if unknown:
                    raise ValueError(f"unknown proposition(s): {{', '.join(sorted(unknown))}}")

                valuation = sum(
                    1 << index
                    for index, name in enumerate(_ATOMIC_PROPOSITIONS)
                    if name in true_propositions
                )
                source = self.state
                for condition, destination in _TRANSITIONS[self.state]:
                    if _condition_holds(condition, valuation):
                        self.state = destination
                        logger.info(
                            "GeneratedRuntimeMonitor trace_id=%s transition=%s->%s "
                            "atomic_propositions=%s accepted=True",
                            self.trace_id,
                            source,
                            destination,
                            sorted(true_propositions),
                        )
                        return True

                self.violated = True
                logger.error(
                    "GeneratedRuntimeMonitor trace_id=%s state=%s "
                    "atomic_propositions=%s accepted=False",
                    self.trace_id,
                    source,
                    sorted(true_propositions),
                )
                return False
        ''').format(
        atomic_propositions=_format_atomic_propositions(proposition_names),
        condition_nodes=_format_condition_nodes(condition_nodes),
        initial_state=automaton.get_init_state_number(),
        ltl_name=_repo_relative_path(ltl_path),
        transitions=_format_transitions(transitions),
    )


def _generate_dot(automaton: spot.twa_graph) -> str:
    return automaton.to_str("dot", "v").replace(" & ", r" &\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ltl-path",
        type=Path,
        default=DEFAULT_LTL_PATH,
        metavar="PATH",
        help=f"LTL input path (default: {DEFAULT_LTL_PATH})",
    )
    parser.add_argument(
        "--hoa-path",
        type=_optional_path,
        default=DEFAULT_HOA_PATH,
        metavar="PATH",
        help=f"HOA output path (default: {DEFAULT_HOA_PATH})",
    )
    parser.add_argument(
        "--monitor-py-path",
        type=_optional_path,
        default=DEFAULT_MONITOR_PY_PATH,
        metavar="PATH",
        help=f"Python monitor output path (default: {DEFAULT_MONITOR_PY_PATH})",
    )
    parser.add_argument(
        "--monitor-dot-path",
        type=_optional_path,
        default=DEFAULT_MONITOR_DOT_PATH,
        metavar="PATH",
        help=f"DOT output path (default: {DEFAULT_MONITOR_DOT_PATH})",
    )
    parser.add_argument(
        "--monitor-svg-path",
        type=_optional_path,
        default=DEFAULT_MONITOR_SVG_PATH,
        metavar="PATH",
        help=f"SVG output path (default: {DEFAULT_MONITOR_SVG_PATH})",
    )
    args = parser.parse_args()

    try:
        conjunction = spot.formula.And(_read_formulas(args.ltl_path))
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    automaton = spot.translate(conjunction, "monitor")
    automaton.set_name(args.ltl_path.name)
    dot = (
        _generate_dot(automaton)
        if args.monitor_dot_path is not None or args.monitor_svg_path is not None
        else None
    )

    if args.hoa_path is not None:
        args.hoa_path.write_text(automaton.to_str("hoa") + "\n")
        print(f"Generated {args.hoa_path} from {args.ltl_path}")
    else:
        print("Skipped HOA output: --hoa-path is empty")

    if args.monitor_py_path is not None:
        args.monitor_py_path.write_text(
            _generate_python_monitor(automaton, args.ltl_path)
        )
        print(f"Generated {args.monitor_py_path}")
    else:
        print("Skipped Python monitor output: --monitor-py-path is empty")

    if args.monitor_dot_path is not None:
        assert dot is not None
        args.monitor_dot_path.write_text(dot)
        print(f"Generated {args.monitor_dot_path}")
    else:
        print("Skipped DOT output: --monitor-dot-path is empty")

    if args.monitor_svg_path is not None:
        assert dot is not None
        if shutil.which("dot") is None:
            raise SystemExit("Graphviz 'dot' was not found")
        subprocess.run(
            ["dot", "-Tsvg", "-o", str(args.monitor_svg_path)],
            input=dot,
            check=True,
            text=True,
        )
        print(f"Generated {args.monitor_svg_path}")
    else:
        print("Skipped SVG output: --monitor-svg-path is empty")


if __name__ == "__main__":
    main()
