"""SGLANG_OPT_RADIX_INSERT_KEY_OFFSET: the unified-cache insert walk tracks an offset
into the key instead of slicing the remaining key at every node. The tree, every
insert's result and the allocator must come out identical to the slicing walk."""

import random
import sys
import unittest
from array import array
from pathlib import Path

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

sys.path.insert(0, str(Path(__file__).parent))
import test_unified_radix_cache_bench as bench  # noqa: E402

PAGE = 4


def _tree_snapshot(tree):
    """Every node as (path tokens, own key, value, evicted, per-component value / locks),
    in a canonical order, plus the root's lock refs."""
    out = []
    stack = [(tree.tree_core.root_node, ())]
    while stack:
        node, path = stack.pop()
        for child_key in sorted(node.children, key=repr):
            child = node.children[child_key]
            tokens = tuple(child.key.token_ids[: child.key._raw_len()])
            comps = tuple(
                (
                    None if cd.value is None else tuple(cd.value.tolist()),
                    cd.lock_ref,
                )
                for cd in child.component_data
            )
            out.append(
                (
                    path,
                    tokens,
                    child.priority,
                    child.output_offset,
                    child.hit_count,
                    comps,
                )
            )
            stack.append((child, path + tokens))
    return sorted(out, key=repr)


def _run(flag: bool, seed: int, components):
    """A fuzz workload: requests sharing prefixes, cached chunk by chunk (as
    chunked prefill does) with prev_prefix_len, re-inserted on finish, with
    evictions in between."""
    rng = random.Random(seed)
    with envs.SGLANG_OPT_RADIX_INSERT_KEY_OFFSET.override(flag):
        with bench._suppress_logs():
            tree, alloc, rtp, make_req = bench.create_bench_cache(
                kv_size=40_000,
                max_num_reqs=64,
                max_context_len=8_000,
                components=components,
                page_size=PAGE,
            )
        env = bench._Env(
            tree,
            alloc,
            rtp,
            make_req,
            [],
            False,
            ComponentType.SWA in components,
            PAGE,
            0,
        )
        bases = [
            [rng.randrange(1, 50) for _ in range(rng.randrange(8, 400))]
            for _ in range(6)
        ]
        results = []
        for _ in range(40):
            base = rng.choice(bases)
            seq = base[: rng.randrange(1, len(base) + 1)] + [
                rng.randrange(1, 50) for _ in range(rng.randrange(0, 600))
            ]
            seq = seq[: len(seq) // PAGE * PAGE]
            if not seq:
                continue
            v = bench._alloc_with_evict(env, len(seq))
            if v is None:
                continue
            prev = 0
            chunk = rng.choice([8, 32, 128])
            for n in list(range(chunk, len(seq), chunk)) + [len(seq)]:
                n = n // PAGE * PAGE
                if n <= prev:
                    continue
                r = tree.insert(
                    InsertParams(
                        key=RadixKey(array("q", seq[:n])),
                        value=v[:n],
                        prev_prefix_len=prev,
                    )
                )
                results.append((r.prefix_len, r.last_device_node))
                prev = n
            if rng.random() < 0.2:
                tree.evict(EvictParams(num_tokens=rng.randrange(16, 400)))
        return _tree_snapshot(tree), results, alloc.available_size()


class TestRadixInsertKeyOffset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._orig = bench.get_device
        bench.get_device = lambda *a, **k: "cpu"

    @classmethod
    def tearDownClass(cls):
        bench.get_device = cls._orig

    def _check(self, components):
        for seed in range(6):
            with self.subTest(seed=seed):
                # The node id counter is global: compare structure, not ids.
                off = _run(False, seed, components)
                on = _run(True, seed, components)
                self.assertEqual(off[0], on[0])
                self.assertEqual([p for p, _ in off[1]], [p for p, _ in on[1]])
                self.assertEqual(off[2], on[2])

    def test_full(self):
        self._check((ComponentType.FULL,))

    def test_full_swa(self):
        self._check((ComponentType.FULL, ComponentType.SWA))

    def test_match_at_equals_sliced_match(self):
        """The offset helpers the walk uses agree with matching the sliced key,
        in plain and bigram keys, at page sizes 1 and 4."""
        rng = random.Random(0)
        for bigram in (False, True):
            for page in (1, 4):
                for _ in range(300):
                    a = array(
                        "q", [rng.randrange(1, 4) for _ in range(rng.randrange(1, 40))]
                    )
                    b = array(
                        "q", [rng.randrange(1, 4) for _ in range(rng.randrange(1, 60))]
                    )
                    ka, kb = (
                        RadixKey(a, is_bigram=bigram),
                        RadixKey(b, is_bigram=bigram),
                    )
                    off = rng.randrange(0, len(kb) + 1)
                    self.assertEqual(
                        ka.match_at(kb, off, page_size=page),
                        ka.match(kb[off:], page_size=page),
                    )
                    if off + page <= len(kb):
                        self.assertEqual(
                            kb.child_key_at(off, page), kb[off:].child_key(page)
                        )


if __name__ == "__main__":
    unittest.main()
