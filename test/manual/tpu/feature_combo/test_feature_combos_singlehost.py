"""Single-host feature-combination tests: the milestone combo gate.

A combo earns a row only when its features can AFFECT each other (shared KV
pools/indices, threads, or collectives); equivalent combos are deduced, not
run, and mechanism depth lives in unit tests next to the mechanism. Full
principle, pass/fail matrix, and failure signatures: COMBO_STATUS.md (same
dir). Multihost P/D: multihost/test_feature_combos.py. Real-model nightly
quad: test/registered/tpu/test_qwen3_moe_feature_combo_nightly.py.

All rows run under torch.compile (the serving recipes ship compiled) at the
full v7x-host shape; dp is carved out of tp, so every combo uses 8 chips:
  1     prefix caching + EAGLE3 spec + KV offloading        (tp=8)
        interaction: draft KV piggybacks target slots; spec multiplies
        forwards racing the reload worker.
  2     prefix caching + DP attention + KV offloading       (tp=8 x dp=2)
        interaction: per-rank pools + idle-rank dummy batches vs reload/evict.
  3     prefix caching + DP attention + KV offloading + spec (tp=8 x dp=2)

ONE COMBO PER PROCESS: TPU runtime re-init in-process is fragile (PJRT
teardown/reinit), so each combo builds its engine in a fresh interpreter:

    SGLANG_PLATFORM=tpu HF_HUB_OFFLINE=1 \
        python test/manual/tpu/feature_combo/test_feature_combos_singlehost.py 1
    # arg = 1 | 2 | 3 | all
"""

import os
import subprocess
import sys
import unittest

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-1.7B")
DRAFT = os.environ.get("DRAFT_MODEL", "AngelSlim/Qwen3-1.7B_eagle3")
PAGE_SIZE = 64
MAX_NEW = 12
POOL = 4096  # small device pool so churn evicts to host
HICACHE_RATIO = 3

# ---- the gradual feature stack -------------------------------------------
_BASE = dict(  # always on: prefix caching + KV offloading
    model_path=MODEL,
    device="tpu",
    dtype="bfloat16",
    page_size=PAGE_SIZE,
    disable_radix_cache=False,
    enable_hierarchical_cache=True,
    hicache_ratio=HICACHE_RATIO,
    hicache_io_backend="torch_tpu",
    hicache_mem_layout="page_first_fused",
    max_total_tokens=POOL,
    mem_fraction_static=0.4,
    skip_server_warmup=True,
    enable_torch_compile=True,
)
_SPEC = dict(  # + EAGLE3 (spec-on-TPU recipe: overlap schedule off)
    speculative_algorithm="EAGLE3",
    speculative_draft_model_path=DRAFT,
    speculative_num_steps=3,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
    disable_overlap_schedule=True,
)

_DP = dict(tp_size=8, dp_size=2, enable_dp_attention=True)

COMBOS = {
    # Default = the full-host serving shape (v7x host = 8 devices): real TP
    # sharding (per-rank KV pools, sharded draft), the shape that ships.
    "1": dict(name="prefix+spec+kvoffload", engine={**_BASE, "tp_size": 8, **_SPEC}),
    "2": dict(name="prefix+dp+kvoffload", engine={**_BASE, **_DP}),
    "3": dict(name="prefix+dp+kvoffload+spec (quad)", engine={**_BASE, **_DP, **_SPEC}),
}
COMBO = os.environ.get("COMBO", "1")  # set by __main__ dispatch


def _skip_if_no_tpu():
    try:
        import importlib.util

        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


# ---- shared workload ------------------------------------------------------
def _count_prompt(n):
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
)
QUESTIONS = [f" Q: question number {i}, name any planet. A:" for i in range(4)]
CHURN = [
    f"Article {i}. " + ", ".join(str(i * 1000 + j) for j in range(150)) + "."
    for i in range(12)
]


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestFeatureCombo(unittest.TestCase):
    """Runs the scenario suite under COMBOS[$COMBO] (one combo per process)."""

    @classmethod
    def setUpClass(cls):
        cfg = COMBOS[COMBO]
        cls.combo_name = cfg["name"]
        kwargs = cfg["engine"]
        cls.spec_on = "speculative_algorithm" in kwargs
        cls.dp_size = kwargs.get("dp_size", 1)
        print(f"\n=== combo {COMBO}: {cls.combo_name} ===")

        os.environ["SGLANG_PLATFORM"] = "tpu"
        # These are SINGLE-HOST engines, but the multihost dev pods bake
        # TPU_WORKER_ID / TPU_WORKER_HOSTNAMES into the pod env — torch_tpu
        # then expects a 2-host slice and hangs ~30 min waiting for the peer
        # before aborting with "PjRtClient is null". Scrub them (same as the
        # P/D test's _no_multihost_env for its per-host engines).
        for v in (
            "TPU_WORKER_ID",
            "TPU_WORKER_HOSTNAMES",
            "TPU_PROCESS_ADDRESSES",
            "TPU_PROCESS_PORT",
        ):
            os.environ.pop(v, None)
        if kwargs.get("tp_size", 1) == 1:
            # tp=1 needs the FULL single-chip pinning trio (matches
            # launch_server.sh): visibility alone is not enough — on a wired
            # multihost slice libtpu otherwise keeps the multi-host topology,
            # waits for peer hosts and aborts with "PjRtClient is null".
            os.environ.setdefault("TPU_VISIBLE_CHIPS", "0")
            os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "1,1,1")
            os.environ.setdefault("TPU_HOST_BOUNDS", "1,1,1")
        else:
            for v in (
                "TPU_VISIBLE_CHIPS",
                "TPU_CHIPS_PER_HOST_BOUNDS",
                "TPU_HOST_BOUNDS",
            ):
                os.environ.pop(v, None)

        import sglang as sgl

        cls.engine = sgl.Engine(**kwargs)

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "engine", None) is not None:
            cls.engine.shutdown()

    def setUp(self):
        self.engine.flush_cache()

    # ---- helpers ----
    def _gen(self, prompt, max_new=MAX_NEW):
        out = self.engine.generate(
            prompt, {"temperature": 0.0, "max_new_tokens": max_new}
        )
        meta = out["meta_info"]
        verify_ct = meta.get("spec_verify_ct") or 0
        accept_length = (
            (meta.get("completion_tokens") or 0) / verify_ct if verify_ct else 0.0
        )
        return out["text"], meta.get("cached_tokens", 0), verify_ct, accept_length

    def _assert_spec(self, verify_ct, accept_length, where):
        if self.spec_on:
            self.assertGreater(verify_ct, 0, f"{where}: spec never verified")
            self.assertGreater(accept_length, 1.0, f"{where}: no draft accepted")

    def _prime_all_ranks(self, prompt):
        """DP ranks have independent radix trees: cache on every rank first."""
        ref = None
        for _ in range(2 * self.dp_size):
            text, _, v, al = self._gen(prompt)
            self._assert_spec(v, al, "priming")
            ref = ref or text
            self.assertEqual(text, ref, "primer outputs diverged across ranks")
        return ref

    # ---- scenarios ----
    def test_1_warm_hit(self):
        ref = self._prime_all_ranks(COUNT_LONG)
        for k in range(2 * self.dp_size):
            text, c, v, al = self._gen(COUNT_LONG)
            self.assertGreaterEqual(c, 3 * PAGE_SIZE, f"repeat {k}: prefix miss")
            self.assertEqual(text, ref, f"repeat {k}: cache hit changed output")
            self._assert_spec(v, al, f"repeat {k} (cache hit)")

    def test_2_shared_prefix_questions(self):
        self._prime_all_ranks(SYSTEM + QUESTIONS[0])
        for q in QUESTIONS[1:]:
            _, c, v, al = self._gen(SYSTEM + q)
            # partial hit: with dp>1 routing is arbitrary, but every rank was
            # primed with the SYSTEM prefix above, so all must hit >= 1 page.
            self.assertGreaterEqual(c, PAGE_SIZE, f"prefix miss on {q!r}")
            self._assert_spec(v, al, f"shared-prefix {q!r}")

    def test_3_eviction_reloads_from_host(self):
        ref = self._prime_all_ranks(COUNT_LONG)
        for doc in CHURN:  # push every rank's device pool past capacity
            self._gen(doc, max_new=1)
        for k in range(2 * self.dp_size):
            text, c, v, _ = self._gen(COUNT_LONG)
            # THE hicache assertion: plain radix would miss after eviction;
            # with KV offloading the prefix must reload from the host pool.
            self.assertGreaterEqual(
                c, 3 * PAGE_SIZE, f"repeat {k}: no host reload after eviction"
            )
            self.assertEqual(text, ref, f"repeat {k}: host reload changed output")
            if self.spec_on:
                self.assertGreater(v, 0, f"repeat {k}: spec inactive after reload")

    def test_4_flush_then_repopulate(self):
        self._gen(COUNT_LONG)
        self.engine.flush_cache()
        _, c, _, _ = self._gen(COUNT_LONG)
        self.assertEqual(c, 0, "flush did not clear the cache")
        self._prime_all_ranks(COUNT_LONG)
        _, c2, _, _ = self._gen(COUNT_LONG)
        self.assertGreaterEqual(c2, 3 * PAGE_SIZE)

    def test_5_concurrent_churn(self):
        # Correct-under-load: every scenario above sends ONE request at a
        # time, so the engine never sees mixed prefill+decode batches, a
        # reload racing another request's decode, competing reloads of the
        # same evicted prefix, or retract/preemption when the pool runs dry.
        # That last path is a proven gap: #152's decode-retract crash was
        # found by a concurrency bench, not by this suite. One batched wave
        # against the tight pool covers all of it with the same oracle:
        # every evicted-prefix hit must still equal the solo-run baseline.
        ref = self._prime_all_ranks(COUNT_LONG)
        for doc in CHURN:  # evict the primed prefix out of every device pool
            self._gen(doc, max_new=1)

        prompts, hot_idx = [], []
        for _ in range(2 * self.dp_size):  # >= 2 hits per DP rank on average
            prompts.extend(CHURN[:3])
            hot_idx.append(len(prompts))
            prompts.append(COUNT_LONG)
        outs = self.engine.generate(
            prompts,
            [{"temperature": 0.0, "max_new_tokens": MAX_NEW}] * len(prompts),
        )
        self.assertEqual(len(outs), len(prompts))
        # Only the correctness oracle: routing/timing decide whether a given
        # hit reloads from host or recomputes, but greedy output must be
        # bit-identical either way. Divergence = lost/garbled reload or a
        # retract that corrupted KV.
        for k, i in enumerate(hot_idx):
            self.assertEqual(
                outs[i]["text"],
                ref,
                f"concurrent wave: evicted-prefix hit #{k} diverged from baseline",
            )

    def test_6_host_overflow_recompute(self):
        # Overflow BOTH tiers: churn far past the host pool (hicache_ratio x
        # device pool) so the primed prefix can be evicted from device AND
        # host. When no reload is possible the engine must recompute —
        # correctly. Certifies the "offload gave the data back up" path.
        # Deliberately no cached_tokens assertion: depending on write policy
        # and LRU timing the prefix may still partially reload; the only
        # invariant is that the output equals the baseline either way.
        ref = self._prime_all_ranks(COUNT_LONG)
        for _ in range(3):  # ~3x the churn that fills the device pool
            for doc in CHURN:
                self._gen(doc, max_new=1)
        for k in range(2 * self.dp_size):
            text, _, v, al = self._gen(COUNT_LONG)
            self.assertEqual(
                text, ref, f"repeat {k}: output after full-tier eviction diverged"
            )
            self._assert_spec(v, al, f"repeat {k} (post-overflow)")


def _run_all():
    """Run every combo sequentially, each in a FRESH interpreter (TPU runtime
    re-init in-process is fragile). Fail fast so the first red row identifies
    which added feature broke."""
    for combo in ("1", "2", "3"):
        print(f"\n{'#' * 60}\n# combo {combo}: {COMBOS[combo]['name']}\n{'#' * 60}")
        env = {**os.environ, "COMBO": combo}
        rc = subprocess.call([sys.executable, os.path.abspath(__file__)], env=env)
        if rc != 0:
            print(f"COMBO {combo} FAILED (rc={rc}) — stopping the ladder here")
            return rc
    print("\nALL COMBOS PASSED")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and (sys.argv[1] in COMBOS or sys.argv[1] == "all"):
        arg = sys.argv.pop(1)
        if arg == "all":
            sys.exit(_run_all())
        os.environ["COMBO"] = arg
        COMBO = arg
    unittest.main(verbosity=2)
