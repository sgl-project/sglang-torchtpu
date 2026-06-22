"""Prefix-cache (radix) correctness tests for the TPU backend — single engine.

Validates radix prefix caching on the Pallas RPA v3 backend across realistic
scenarios using ONE engine instance (engine init is ~90 s on TPU, so the whole
suite shares a single radix-ON engine created in setUpClass):

* long multi-page prompt reuse (cold vs warm),
* multiple prompts sharing a long system prefix (sequential),
* primer-invariance — the strongest KV gate (a cached prefix must give the same
  result regardless of which request wrote it),
* concurrent batch sharing the prefix,
* cache eviction under a bounded KV pool.

Design notes
------------
* The engine runs radix ON with a small ``max_total_tokens`` so the *same*
  engine can both serve the normal scenarios and be driven into eviction.
* No separate radix-OFF engine is needed: a ``flush_cache()`` + run is the
  "cold" (no-reuse) reference, and on wide-margin counting prompts the cached
  (partial-prefill) result is bit-identical to that cold (full-prefill) result.
* Eager / greedy / tp=1 so greedy decoding is deterministic and exact-match
  assertions are valid (torch.compile would use different XLA programs for
  cached-partial vs full prefill, whose ~ULP rounding can flip greedy on
  low-margin tokens — a precision artifact, not a cache bug).
* Each test flushes the cache in setUp so scenarios are order-independent.

Requires TPU hardware + torch_tpu. Single chip (tp=1).

Usage:
    SGLANG_PLATFORM=tpu TPU_VISIBLE_CHIPS=0 \
        python test/manual/tpu/test_prefix_cache.py
"""

import os
import unittest

MODEL = "Qwen/Qwen3-0.6B"
PAGE_SIZE = 64
MAX_NEW = 12
POOL = 4096  # small KV pool so the same engine can be driven into eviction


def _skip_if_no_tpu():
    try:
        import importlib.util

        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _count_prompt(n):
    # Wide greedy margin: counting continues deterministically, so cached
    # (partial-prefill) == recompute (full-prefill) exactly.
    return (
        "Continue this integer sequence, comma separated:\n"
        + ", ".join(str(i) for i in range(1, n + 1))
        + ","
    )


COUNT_LONG = _count_prompt(120)  # ~300 tokens -> ~5 pages
SYSTEM = (
    "The following is reference material. The Sun is the star at the center of "
    "the solar system. Mercury, Venus, Earth, and Mars are the four terrestrial "
    "planets. Jupiter and Saturn are gas giants; Uranus and Neptune are ice "
    "giants. Earth has one moon. Mars has two moons, Phobos and Deimos. Answer "
    "using only the reference material above."
)  # ~120 tokens -> 2 pages of shared prefix
QUESTIONS = [
    " Q: How many terrestrial planets are there? A:",
    " Q: Name Earth's moon. A:",
    " Q: Which planets are ice giants? A:",
    " Q: What are the moons of Mars? A:",
]
# Distinct, non-prefix-sharing docs whose cumulative KV (~4800 tok) exceeds POOL
# so the first-cached prompt is provably evicted (LRU).
CHURN = [
    f"Article {i}. " + ", ".join(str(i * 1000 + j) for j in range(150)) + "."
    for i in range(12)
]


def _gen(engine, prompt, n=MAX_NEW):
    out = engine.generate(prompt, {"temperature": 0.0, "max_new_tokens": n})
    if isinstance(out, list):
        out = out[0]
    return out["text"], out["meta_info"].get("cached_tokens", 0)


def _gen_batch(engine, prompts, n=MAX_NEW):
    outs = engine.generate(prompts, {"temperature": 0.0, "max_new_tokens": n})
    return [(o["text"], o["meta_info"].get("cached_tokens", 0)) for o in outs]


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuPrefixCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_PLATFORM"] = "tpu"
        os.environ.setdefault("TPU_VISIBLE_CHIPS", "0")

        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL)
        assert len(tok.encode(COUNT_LONG)) >= 3 * PAGE_SIZE, (
            f"COUNT_LONG tokenizes to fewer than {3 * PAGE_SIZE} tokens; "
            "test_1 and test_5 assertions will always fail"
        )
        assert len(tok.encode(SYSTEM)) >= PAGE_SIZE, (
            f"SYSTEM tokenizes to fewer than {PAGE_SIZE} tokens; "
            "test_2 and test_3 assertions will always fail"
        )

        import sglang as sgl

        cls.engine = sgl.Engine(
            model_path=MODEL,
            device="tpu",
            tp_size=1,
            dtype="bfloat16",
            page_size=PAGE_SIZE,
            disable_radix_cache=False,
            mem_fraction_static=0.4,
            max_total_tokens=POOL,
            skip_server_warmup=True,
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()

    def setUp(self):
        # Isolate each scenario: start from an empty cache.
        self.engine.flush_cache()

    # (1) Long, multi-page prompt: a cache hit must not change the output. --- #
    def test_1_long_prompt_multipage_reuse(self):
        cold, cold_c = _gen(self.engine, COUNT_LONG)
        warm, warm_c = _gen(self.engine, COUNT_LONG)
        self.assertEqual(cold_c, 0, "cold run after flush must not hit cache")
        self.assertGreaterEqual(warm_c, 3 * PAGE_SIZE, "expected multi-page reuse")
        self.assertEqual(warm, cold, "cache reuse changed the long-prompt output")

    # (2) Many prompts sharing a long system prefix reuse it (sequential). --- #
    def test_2_multiple_prompts_shared_prefix(self):
        first = [_gen(self.engine, SYSTEM + q) for q in QUESTIONS]
        for i, (_, c) in enumerate(first[1:], start=1):
            self.assertGreaterEqual(
                c, PAGE_SIZE, f"Q{i} did not reuse the shared system prefix"
            )
        second = [_gen(self.engine, SYSTEM + q) for q in QUESTIONS]
        self.assertEqual(
            [t for t, _ in first],
            [t for t, _ in second],
            "warm re-run not deterministic",
        )

    # (3) Primer-invariance: cached prefix must be independent of its writer. - #
    def test_3_primer_invariance(self):
        target = SYSTEM + QUESTIONS[-1]
        _gen(self.engine, SYSTEM + QUESTIONS[0])  # primer A
        a, _ = _gen(self.engine, target)
        self.engine.flush_cache()
        _gen(self.engine, SYSTEM + QUESTIONS[1])  # primer B
        c, cc = _gen(self.engine, target)
        self.assertGreaterEqual(cc, PAGE_SIZE, "target did not reuse the shared prefix")
        self.assertEqual(
            a, c, "cached prefix KV depends on primer -> cross-request bug"
        )

    # (4) Concurrent batch sharing the prefix: completes, hits, deterministic. #
    def test_4_concurrent_batch(self):
        prompts = [SYSTEM + q for q in QUESTIONS]
        b1 = _gen_batch(self.engine, prompts)
        b2 = _gen_batch(self.engine, prompts)
        self.assertTrue(all(t for t, _ in b1), "a concurrent request returned empty")
        self.assertEqual(
            [t for t, _ in b1], [t for t, _ in b2], "concurrent batch not deterministic"
        )
        # b1 ran against an empty cache (setUp flushed); b2 is the warm rerun that hits.
        self.assertGreater(
            sum(c for _, c in b2), 0, "no cache hits on warm batch rerun"
        )

    # (5) Eviction: overflow the pool, prove the first prompt was evicted and  #
    #     recomputes correctly (no pool corruption).                           #
    def test_5_eviction_path(self):
        # Starting cache state is irrelevant: counting is deterministic, so
        # out1 (warm or cold) must equal out2 (cold recompute after eviction).
        out1, _ = _gen(self.engine, COUNT_LONG)
        for p in CHURN:
            txt, _ = _gen(self.engine, p)
            self.assertTrue(txt, "a churn request returned empty (alloc/evict failed)")
        # COUNT_LONG was cached first and is LRU; cumulative churn KV > POOL, so
        # it must have been evicted -> this requery is a miss (recompute).
        out2, c2 = _gen(self.engine, COUNT_LONG)
        self.assertLess(c2, PAGE_SIZE, "expected COUNT_LONG to be evicted (cache miss)")
        self.assertEqual(out1, out2, "output changed after eviction (pool corruption)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
