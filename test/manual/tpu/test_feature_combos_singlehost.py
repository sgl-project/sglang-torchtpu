"""Single-host feature-combination tests — one file, gradually stacking
features (milestone combo track; multihost P/D lives in
multihost/test_feature_combos.py, real-model nightly quad in
test/registered/tpu/test_qwen3_moe_feature_combo_nightly.py).

Combos (each = the SAME max-coverage scenario suite under a richer config):
  1  prefix caching + EAGLE3 spec + KV offloading          (tp=1)
  2  prefix caching + DP attention + KV offloading         (tp=2 x dp=2)
  3  prefix caching + DP attention + KV offloading + spec  (tp=2 x dp=2, quad)

Scenario suite (runs under every combo):
  * cold vs warm multi-page reuse (+ spec verify/accept on the cache-hit
    request when spec is on — the interaction case),
  * shared system-prefix across distinct questions,
  * eviction -> HiCache host reload (churn past the device pool, then the
    prefix must come BACK: cached >= pages — plain radix would miss),
  * flush -> repopulate,
  * greedy determinism: cache-hit output bit-identical to cold output.
  DP configs prime every rank first (each DP rank has its own radix tree —
  a prefix cached on rank 0 is a miss on rank 1 until primed).

ONE COMBO PER PROCESS: TPU runtime re-init in-process is fragile (PJRT
teardown/reinit), so each combo builds its engine in a fresh interpreter.
Run one combo directly, or all three sequentially via subprocesses:

    SGLANG_PLATFORM=tpu HF_HUB_OFFLINE=1 \
        python test/manual/tpu/test_feature_combos_singlehost.py 1
    python test/manual/tpu/test_feature_combos_singlehost.py all   # 1 -> 2 -> 3

Known landmine scenario 3 is designed to step on if present: the plain
torch_tpu HiCacheController load path writes device tensors into the
CPU-resident radix tree on host->HBM reload (raiden .cpu()s them; see
cache_controller.move_indices notes) — a crash there is a real bug.
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
)
_SPEC = dict(  # + EAGLE3 (spec-on-TPU recipe: overlap schedule off)
    speculative_algorithm="EAGLE3",
    speculative_draft_model_path=DRAFT,
    speculative_num_steps=3,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
    disable_overlap_schedule=True,
)
_DP = dict(tp_size=2, dp_size=2, enable_dp_attention=True)  # + DP attention

COMBOS = {
    "1": dict(name="prefix+spec+kvoffload", engine={**_BASE, "tp_size": 1, **_SPEC}),
    # Same combo at the full-host serving shape (v7x host = 8 devices):
    # exercises spec + hicache under real TP sharding (per-rank KV pools,
    # sharded draft) rather than the single-chip smoke config.
    "1tp8": dict(
        name="prefix+spec+kvoffload (tp=8)",
        engine={**_BASE, "tp_size": 8, **_SPEC},
    ),
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


def _run_all():
    """Run each combo in a FRESH interpreter (TPU runtime re-init is fragile
    in-process), sequentially: 1 -> 2 -> 3. Fail fast per user preference for
    isolating which added feature broke."""
    for combo in ("1", "1tp8", "2", "3"):
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
