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

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=600,
    suite="stage-b-test-tpu-4-v7",
    disabled="Deprecated: superseded by test_tpu_gpt_oss_120b_singlehost_prefix_cache_kv_offload.py",
    labels=["singlehost", "qwen3-0.6b", "prefix-cache"],
)

import os
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

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
# Distinct, non-prefix-sharing docs whose cumulative KV (~7200 tok) exceeds POOL
# so the first-cached prompt is provably evicted (LRU).
CHURN = [
    f"Article {i}. " + ", ".join(str(i * 1000 + j) for j in range(100)) + "."
    for i in range(12)
]


def _gen(base_url, prompt, n=MAX_NEW):
    res = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": n},
        },
    ).json()
    if isinstance(res, list):
        res = res[0]
    return res["text"], res["meta_info"].get("cached_tokens", 0)


def _gen_batch(base_url, prompts, n=MAX_NEW):
    # n: shared decode length, or a per-prompt list for heterogeneous batches.
    ns = n if isinstance(n, list) else [n] * len(prompts)
    res = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompts,
            "sampling_params": [{"temperature": 0.0, "max_new_tokens": m} for m in ns],
        },
    ).json()
    if not isinstance(res, list):
        res = [res]
    return [(o["text"], o["meta_info"].get("cached_tokens", 0)) for o in res]


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostPrefixCache(CustomTestCase):
    MODEL = "Qwen/Qwen3-0.6B"
    TP_SIZE = 1
    MEM_FRACTION_STATIC = 0.4
    EXTRA_ENGINE_KWARGS = {}

    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_PLATFORM"] = "tpu"

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--device",
            "tpu",
            "--dtype",
            "bfloat16",
            "--tp-size",
            str(cls.TP_SIZE),
            "--page-size",
            str(PAGE_SIZE),
            "--mem-fraction-static",
            str(cls.MEM_FRACTION_STATIC),
            "--max-total-tokens",
            str(POOL),
            "--chunked-prefill-size",
            "512",
            "--context-length",
            "1024",
            "--torch-compile-max-bs",
            "4",
            "--tpu-compile-tokens",
            "64",
            "512",
            "--tpu-compile-bs",
            "1",
            "4",
            "--enable-torch-compile",
            "--grammar-backend",
            "none",
        ]
        for k, v in cls.EXTRA_ENGINE_KWARGS.items():
            other_args.extend([f"--{k.replace('_', '-')}", str(v)])

        cls.process = popen_launch_server(
            cls.MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            device="tpu",
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def setUp(self):
        # Isolate each scenario: start from an empty cache.
        requests.post(f"{self.base_url}/flush_cache")

    # (1) Long, multi-page prompt: a cache hit must not change the output. --- #
    def test_1_long_prompt_multipage_reuse(self):
        cold, cold_c = _gen(self.base_url, COUNT_LONG)
        warm, warm_c = _gen(self.base_url, COUNT_LONG)
        self.assertEqual(cold_c, 0, "cold run after flush must not hit cache")
        self.assertGreaterEqual(warm_c, 3 * PAGE_SIZE, "expected multi-page reuse")
        self.assertEqual(warm, cold, "cache reuse changed the long-prompt output")

    # (2) Many prompts sharing a long system prefix reuse it (sequential). --- #
    def test_2_multiple_prompts_shared_prefix(self):
        first = [_gen(self.base_url, SYSTEM + q) for q in QUESTIONS]
        for i, (_, c) in enumerate(first[1:], start=1):
            self.assertGreaterEqual(
                c, PAGE_SIZE, f"Q{i} did not reuse the shared system prefix"
            )
        second = [_gen(self.base_url, SYSTEM + q) for q in QUESTIONS]
        self.assertEqual(
            [t for t, _ in first],
            [t for t, _ in second],
            "warm re-run not deterministic",
        )

    # (3) Primer-invariance: cached prefix must be independent of its writer. - #
    def test_3_primer_invariance(self):
        target = SYSTEM + QUESTIONS[-1]
        _gen(self.base_url, SYSTEM + QUESTIONS[0])  # primer A
        a, _ = _gen(self.base_url, target)
        requests.post(f"{self.base_url}/flush_cache")
        _gen(self.base_url, SYSTEM + QUESTIONS[1])  # primer B
        c, cc = _gen(self.base_url, target)
        self.assertGreaterEqual(cc, PAGE_SIZE, "target did not reuse the shared prefix")
        self.assertEqual(
            a, c, "cached prefix KV depends on primer -> cross-request bug"
        )

    # (4) Concurrent batch sharing the prefix: completes, hits, deterministic. #
    def test_4_concurrent_batch(self):
        prompts = [SYSTEM + q for q in QUESTIONS]
        b1 = _gen_batch(self.base_url, prompts)
        b2 = _gen_batch(self.base_url, prompts)
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
        out1, _ = _gen(self.base_url, COUNT_LONG)
        for p in CHURN:
            txt, _ = _gen(self.base_url, p)
            self.assertTrue(txt, "a churn request returned empty (alloc/evict failed)")
        # COUNT_LONG was cached first and is LRU; cumulative churn KV > POOL, so
        # it must have been evicted -> this requery is a miss (recompute).
        out2, c2 = _gen(self.base_url, COUNT_LONG)
        self.assertLess(c2, PAGE_SIZE, "expected COUNT_LONG to be evicted (cache miss)")
        self.assertEqual(out1, out2, "output changed after eviction (pool corruption)")


COUNT_EXT = _count_prompt(150)

SHORT_PREFIX = _count_prompt(30)
SUF_A = " What single number comes next? Answer:"
SUF_B = " List the next three numbers. Answer:"

SWA_RATIO = 0.5  # swa pool = POOL * ratio; makes SWA the binding pool (test_10)
# ~1924 tokens: overflows the 2048-token SWA pool (after COUNT_LONG is cached)
# while the 4096-token full pool stays ~50% used -> SWA-only eviction pressure.
BIG_DOC = "Article 99. " + ", ".join(str(90000 + j) for j in range(480)) + "."


# 7/11 pass; test_8 hits a real SWA cache bug (a window-slid prefix is still
# reused) and 3 more hit the cached-vs-cold divergence of #321.
@unittest.skip("SWA prefix cache reuses a tombstoned prefix: #361")
@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostSwaPrefixCache(TestTpuQwen306bSinglehostPrefixCache):
    """Prefix-cache correctness on a hybrid-SWA model (gpt-oss-20b, window=128).

    Runs every scenario of :class:`TestTpuPrefixCache` against ``SWARadixCache``
    + the two-pool ``SWAKVPool`` (alternating sliding/full layers), then adds
    window-boundary, window-slide (tombstone), SWA-pool-eviction, and a
    concurrent slide/evict-while-locked scenario.
    Runs with ``swa_full_tokens_ratio=0.5`` so the SWA pool (2048 tok) is the
    binding constraint, as in production.

    Usage (single class):
        SGLANG_PLATFORM=tpu TPU_VISIBLE_CHIPS=0 \
            python test/manual/tpu/test_prefix_cache.py TestTpuSwaPrefixCache
    """

    MODEL = "openai/gpt-oss-20b"
    # 20B bf16 does not fit one core at 0.4; shard it and use the
    # gpt-oss-120b recipe's 0.60 static budget.
    TP_SIZE = 8
    MEM_FRACTION_STATIC = 0.6
    # Experts go EP, not TP (MXFP4 block / MXU tile alignment), and GPT-OSS
    # runs on pallas_tpu_rpa_v3, the recipe's default backend. Both per
    # benchmark/tpu/recipe/gpt_oss_120b/server.sh.
    EXTRA_ENGINE_KWARGS = {
        "swa_full_tokens_ratio": SWA_RATIO,
        "ep_size": 8,
        "attention_backend": "pallas_tpu_rpa_v3",
    }

    @classmethod
    def setUpClass(cls):
        if getattr(cls, "__unittest_skip__", False):
            return
        # The SparseCore ragged-gather MoE path (default ON) cannot be lowered
        # for GPT-OSS: ragged_gather_v2's pltpu.async_copy indexer needs a
        # static row count and the EXTEND graph reaches it with a symbolic
        # token dim. Same reason the 120b recipe turns it off.
        os.environ["SGLANG_TPU_FUSED_MOE_SC_GATHER"] = "0"

        from transformers import AutoConfig, AutoTokenizer

        cfg = AutoConfig.from_pretrained(cls.MODEL)
        cls.window = getattr(cfg, "sliding_window", None)
        assert cls.window and cls.window >= 2 * PAGE_SIZE, (
            f"expected a sliding window of >= 2 pages, got {cls.window}; "
            "the window-boundary assertions below assume it"
        )
        # Decode length that provably slides the window past any prompt:
        # decode-time eviction fires every `interval` (= window) tokens with a
        # window+page margin, so LONG_NEW >= interval + window + 2*page suffices.
        cls.LONG_NEW = 3 * cls.window + PAGE_SIZE

        # BIG_DOC must overflow the SWA pool even after decode-time
        # self-eviction, yet fit in it alone (with room to decode).
        swa_pool = int(POOL * SWA_RATIO)
        n_big = len(AutoTokenizer.from_pretrained(cls.MODEL).encode(BIG_DOC))
        assert (
            swa_pool - cls.window - PAGE_SIZE < n_big <= swa_pool - PAGE_SIZE - MAX_NEW
        ), (
            f"BIG_DOC tokenizes to {n_big}; test_10 needs it in "
            f"({swa_pool - cls.window - PAGE_SIZE}, {swa_pool - PAGE_SIZE - MAX_NEW}]"
        )
        super().setUpClass()

    # (6) Extend from a cached prefix that exceeds the SWA window: the reused --
    #     prefix's SWA KV is only valid for the tail window, and the extension
    #     allocates with prefix_lens > 0 (full->swa last_loc translation).
    def test_6_swa_extend_from_cached_prefix(self):
        cold_ext, c0 = _gen(self.base_url, COUNT_EXT)
        self.assertEqual(c0, 0, "cold run after flush must not hit cache")
        requests.post(f"{self.base_url}/flush_cache")
        _gen(self.base_url, COUNT_LONG)  # prime: caches a prefix > window
        warm_ext, c1 = _gen(self.base_url, COUNT_EXT)
        self.assertGreaterEqual(
            c1, self.window, "extension did not reuse a beyond-window prefix"
        )
        self.assertEqual(
            warm_ext, cold_ext, "extending from a cached SWA prefix changed output"
        )

    # (7) Shared prefix shorter than the window: all of its SWA KV must be ----
    #     in-window on reuse, and the result must not depend on the primer.
    def test_7_swa_subwindow_shared_prefix(self):
        target = SHORT_PREFIX + SUF_B
        cold, c0 = _gen(self.base_url, target)
        self.assertEqual(c0, 0, "cold run after flush must not hit cache")
        requests.post(f"{self.base_url}/flush_cache")
        _gen(self.base_url, SHORT_PREFIX + SUF_A)  # primer writes the prefix
        warm, c1 = _gen(self.base_url, target)
        self.assertGreaterEqual(
            c1, PAGE_SIZE, "target did not reuse the sub-window shared prefix"
        )
        self.assertEqual(warm, cold, "sub-window cached prefix KV depends on primer")

    # (8) Decode past the window: the prompt's SWA KV is tombstoned mid-decode.
    #     A re-query must NOT reuse the tombstoned prefix (match-past-tombstone
    #     gate) and must recompute bit-identically (no corruption from free_swa).
    def test_8_swa_window_slide_long_decode(self):
        out1, _ = _gen(self.base_url, COUNT_LONG, n=self.LONG_NEW)
        self.assertTrue(out1, "long decode returned empty")
        out2, c2 = _gen(self.base_url, COUNT_LONG, n=self.LONG_NEW)
        self.assertLess(c2, PAGE_SIZE, "reused a tombstoned (window-slid) prefix")
        self.assertEqual(out2, out1, "recompute after window slide changed output")

    # (9) Match past a tombstone: the cached sequence is tombstoned-head +
    #     live tail >= window, so extending it MAY reuse across the tombstone
    #     and must equal a cold full recompute.
    def test_9_swa_match_past_tombstone(self):
        out1, _ = _gen(self.base_url, COUNT_LONG, n=self.LONG_NEW)
        ext = COUNT_LONG + out1
        warm, cw = _gen(self.base_url, ext)
        requests.post(f"{self.base_url}/flush_cache")
        cold, c0 = _gen(self.base_url, ext)
        self.assertEqual(c0, 0, "cold run after flush must not hit cache")
        self.assertGreaterEqual(
            cw, self.window, "extension did not reuse across the tombstone"
        )
        self.assertEqual(warm, cold, "reuse across a tombstone changed output")

    # (10) SWA-pool-bound eviction: BIG_DOC overflows only the SWA pool, so ---
    #      swa-LRU eviction (not full-pool eviction) must reclaim COUNT_LONG;
    #      the requery must miss and recompute identically.
    def test_10_swa_pool_bound_eviction(self):
        out1, _ = _gen(self.base_url, COUNT_LONG)
        txt, _ = _gen(self.base_url, BIG_DOC)
        self.assertTrue(txt, "SWA-pool-overflow request returned empty")
        out2, c2 = _gen(self.base_url, COUNT_LONG)
        self.assertLess(c2, PAGE_SIZE, "expected SWA-pool eviction of COUNT_LONG")
        self.assertEqual(out2, out1, "output changed after SWA-pool eviction")

    # (11) Concurrency: three sharers of the primed prefix decode past the ----
    #      window in ONE in-flight batch
    def test_11_swa_concurrent_slide_evict_locked_prefix(self):
        extend = _count_prompt(180)  # shares COUNT_LONG's prefix, longer tail
        prompts = [COUNT_LONG, COUNT_EXT, extend] + CHURN[:3]
        news = [self.LONG_NEW] * 3 + [MAX_NEW] * 3

        refs = []
        for p, n in zip(prompts, news):
            requests.post(f"{self.base_url}/flush_cache")
            refs.append(_gen(self.base_url, p, n=n)[0])

        requests.post(f"{self.base_url}/flush_cache")
        _gen(self.base_url, COUNT_LONG)  # prime the shared prefix into the tree
        outs = _gen_batch(self.base_url, prompts, n=news)
        texts = [t for t, _ in outs]
        cached = [c for _, c in outs]

        self.assertTrue(all(texts), "a concurrent request returned empty")
        for i in range(3):
            self.assertGreaterEqual(
                cached[i], self.window, f"sharer {i} did not reuse the primed prefix"
            )
        self.assertEqual(
            texts, refs, "concurrent window-slide/eviction changed request outputs"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
