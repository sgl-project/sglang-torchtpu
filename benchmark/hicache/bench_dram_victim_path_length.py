"""CPU metadata benchmark for the already-pushed victim-path guard.

No model, weights, Torch runtime or native transfer is used. Unchanged source
scan/helper bodies are AST-extracted over synthetic metadata doubles. This
measures host scan cost, not serving latency or real tree/pool integration.

After committing the guard, run from the repository root:
  uv run --no-project python benchmark/hicache/bench_dram_victim_path_length.py \
      --baseline-ref HEAD^ --out /tmp/victim-path-length
"""

import argparse
import ast
import csv
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from weakref import WeakKeyDictionary

ROOT = Path(__file__).resolve().parents[2]
HELPERS = "sglang_tpu/disaggregation/decode_dram_offload/radix_host_ops.py"
PUSHER = "sglang_tpu/disaggregation/decode_dram_offload/prefill_dram_victim_pusher.py"


def functions(pusher_source):
    namespace = {"_FULL": "full", "_SWA": "swa"}
    helpers = ast.parse((ROOT / HELPERS).read_text())
    names = {
        "_path_to_root",
        "path_tokens",
        "path_token_count",
        "host_page_ids",
        "host_pages",
        "window_host_pages",
        "is_live_host_backed_leaf",
    }
    selected = [
        n for n in helpers.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    pusher = next(
        n
        for n in ast.parse(pusher_source).body
        if isinstance(n, ast.ClassDef) and n.name == "PrefillDramVictimPusher"
    )
    selected.append(
        next(
            n
            for n in pusher.body
            if isinstance(n, ast.FunctionDef) and n.name == "_pick_coldest_host_leaf"
        )
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *selected,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), "source-scan", "exec"), namespace)
    return namespace


@dataclass
class Key:
    token_ids: array

    def __len__(self):
        return len(self.token_ids)


@dataclass(eq=False)
class Node:
    id: int
    key: Key
    parent: object
    last_access_time: float
    children: dict = field(default_factory=dict)
    component_data: dict = field(default_factory=dict)
    write_through_pending_id: object = None


class HostPool:
    def _to_page_ids(self, indices):
        return [int(i) // 64 for i in indices[::64]]


def fixture(routes, tokens, order, state):
    assert routes > 0 and tokens > 0 and tokens % 256 == 0
    root = Node(-1, Key(array("q")), None, 0)
    nodes, leaves = {-1: root}, []
    for route in range(routes):
        parent = root
        for depth in range(4):
            node_id = 4 * route + depth
            raw = array("q", range(tokens // 4))
            node = Node(
                node_id,
                Key(raw),
                parent,
                float(route),
                component_data={"full": SimpleNamespace(host_value=raw)},
            )
            parent.children[node_id] = node
            nodes[node_id] = node
            parent = node
        leaves.append(parent)
    core = SimpleNamespace(root_node=root, node_by_id=nodes.__getitem__)
    ordered = list(leaves)
    if order == "newest":
        ordered.reverse()
    elif order == "shuffled":
        random.Random(7).shuffle(ordered)
    pushed = WeakKeyDictionary()
    if state == "already_pushed":
        for leaf in leaves:
            pushed[leaf] = tokens
    elif state == "pending":
        for leaf in leaves:
            leaf.write_through_pending_id = 1
    return SimpleNamespace(
        _cache=SimpleNamespace(tree_core=core),
        _host_pool=HostPool(),
        _swa_host_pool=None,
        _window_tokens=0,
        _pushes={},
        _pushed_len=pushed,
        _routes=WeakKeyDictionary(
            (leaf, (("peer", leaf.id), (leaf.id, 0))) for leaf in ordered
        ),
    ), leaves


def timed(function, iterations):
    start = time.perf_counter_ns()
    for _ in range(iterations):
        function()
    return (time.perf_counter_ns() - start) / iterations / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--routes", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--blocks", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    assert args.blocks > 0 and args.iterations > 0
    baseline_source = subprocess.check_output(
        ["git", "show", f"{args.baseline_ref}:{PUSHER}"], cwd=ROOT, text=True
    )
    current_source = (ROOT / PUSHER).read_text()
    baseline, candidate = functions(baseline_source), functions(current_source)
    raw_blocks, results = [], []
    for state in ("already_pushed", "eligible", "pending"):
        for order in ("oldest", "shuffled", "newest"):
            f, leaves = fixture(args.routes, args.tokens, order, state)
            arms = {
                "baseline": lambda: baseline["_pick_coldest_host_leaf"](f),
                "candidate": lambda: candidate["_pick_coldest_host_leaf"](f),
            }
            assert arms["baseline"]() == arms["candidate"]()
            paths = {}
            for arm, namespace in (("baseline", baseline), ("candidate", candidate)):
                calls = []
                original = namespace["path_tokens"]

                def counted(*a, **kw):
                    calls.append(1)
                    return original(*a, **kw)

                namespace["path_tokens"] = counted
                try:
                    arms[arm]()
                finally:
                    namespace["path_tokens"] = original
                paths[arm] = len(calls)
            for function in arms.values():
                for _ in range(5):
                    function()
            paired = []
            for block in range(args.blocks):
                ordering = (
                    ("baseline", "candidate")
                    if block % 2 == 0
                    else ("candidate", "baseline")
                )
                values = {arm: timed(arms[arm], args.iterations) for arm in ordering}
                paired.append(values)
                raw_blocks.append(
                    {
                        "state": state,
                        "order": order,
                        "block": block,
                        "baseline_ms": values["baseline"],
                        "candidate_ms": values["candidate"],
                    }
                )
            saved = [v["baseline"] - v["candidate"] for v in paired]
            rng = random.Random(7)
            bootstrap = sorted(
                statistics.mean(rng.choices(saved, k=len(saved))) for _ in range(2000)
            )
            results.append(
                {
                    "state": state,
                    "order": order,
                    "baseline_median_ms": statistics.median(
                        v["baseline"] for v in paired
                    ),
                    "candidate_median_ms": statistics.median(
                        v["candidate"] for v in paired
                    ),
                    "saved_mean_ms": statistics.mean(saved),
                    "paired_bootstrap_95_ms": [bootstrap[49], bootstrap[1949]],
                    "paths": paths,
                }
            )
            # Retain strong references throughout the timed weak-route fixture.
            assert len(leaves) == args.routes
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "blocks.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=raw_blocks[0])
        writer.writeheader()
        writer.writerows(raw_blocks)
    report = {
        "python": platform.python_version(),
        "cpu_architecture": platform.machine(),
        "routes": args.routes,
        "tokens": args.tokens,
        "depth": 4,
        "page_size": 64,
        "blocks": args.blocks,
        "iterations_per_arm": args.iterations,
        "warmups_per_arm": 5,
        "baseline_source_sha256": hashlib.sha256(baseline_source.encode()).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(current_source.encode()).hexdigest(),
        "results": results,
        "limitations": [
            "Synthetic CPU metadata; no real tree/pool, native or serving validation.",
            "Bootstrap intervals describe within-process paired blocks only.",
        ],
    }
    (args.out / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
