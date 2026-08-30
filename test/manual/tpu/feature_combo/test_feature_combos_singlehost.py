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
    # combo = 1 | 2 | 3 | all

Select the large-model profile explicitly (the small model remains the CI
default):

    python test/manual/tpu/feature_combo/test_feature_combos_singlehost.py \
        --model-profile qwen3-coder-480b all
"""

import argparse
import os
import subprocess
import sys
import unittest
import warnings

MAX_NEW = 12
HICACHE_RATIO = 3

# Workload sizes scale with page/cache geometry so every profile exercises the
# same invariants: multi-page hits, device eviction, and host-tier overflow.
MODEL_PROFILES = {
    "qwen3-1.7b": {
        "model": "Qwen/Qwen3-1.7B",
        "draft_model": "AngelSlim/Qwen3-1.7B_eagle3",
        "page_size": 64,
        "mem_fraction_static": 0.4,
        "workload": {
            "max_total_tokens": 4096,
            "long_prompt_num_integers": 120,
            "min_cached_pages": 3,
            "system_repeats": 1,
            "churn_count": 12,
        },
        "base_engine": {},
        # Native BF16 avoids ULP shifts between DP ranks in the exact-output
        # oracle. This preserves the original combo-2 configuration.
        "dp_engine": {"quantization": None, "kv_cache_dtype": "bf16"},
        "accept_length_ref": "1.275",
    },
    "qwen3-coder-480b": {
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
        "draft_model": (
            "lmsys/SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-" "SpecForge-EigenAI"
        ),
        "page_size": 256,
        "mem_fraction_static": 0.8,
        "workload": {
            # The common concurrent wave needs 24 pages at dp=1. Keep enough
            # headroom for it while retaining deterministic device eviction.
            "max_total_tokens": 8192,
            "long_prompt_num_integers": 400,
            "min_cached_pages": 3,
            "system_repeats": 5,
            # One wave evicts the primer from device cache without exhausting
            # the larger host cache. Tests route this many prompts to each DP
            # rank so cache pressure does not depend on scheduler routing.
            "churn_count": 16,
        },
        "base_engine": {
            "quantization": "fp8",
            "kv_cache_dtype": "fp8_e4m3",
            "ep_size": 8,
            "tpu_fp8_runtime": "channelwise",
            # Bound compilation to the focused test's request envelope.
            "max_prefill_tokens": 2048,
            "max_running_requests": 16,
            "tpu_compile_bs": [1, 2, 4],
            "tpu_compile_tokens": [512, 2048],
        },
        # The BF16 target does not fit this single-host shape. Retain FP8
        # weights, but use BF16 KV to isolate cache precision from weight
        # quantization. Page size 256 avoids the RPA shared-memory overflow
        # seen with page size 64.
        "dp_engine": {"quantization": "fp8", "kv_cache_dtype": "bf16"},
        # Establish this on hardware before turning it into a regression gate.
        "accept_length_ref": None,
    },
}


def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Run single-host TPU feature-combination tests.",
        epilog="combo: 1, 2, 3, or all (default: COMBO environment variable or 1)",
    )
    parser.add_argument(
        "--model-profile",
        choices=MODEL_PROFILES,
        default=os.environ.get("MODEL_PROFILE", "qwen3-1.7b"),
        help="model-specific engine settings (default: %(default)s)",
    )
    args, remaining = parser.parse_known_args(argv)
    combo = None
    if remaining and remaining[0] in ("1", "2", "3", "all"):
        combo = remaining.pop(0)
    # Keep unittest selectors/options working as they did before this CLI was
    # added; only a recognized leading combo selector belongs to us.
    args.combo = combo
    return args, remaining


ARGS, UNITTEST_ARGS = _parse_args(sys.argv[1:])
MODEL_PROFILE = ARGS.model_profile
COMBO = ARGS.combo if ARGS.combo in ("1", "2", "3") else os.environ.get("COMBO", "1")
PROFILE = MODEL_PROFILES[MODEL_PROFILE]
MODEL = os.environ.get("MODEL", PROFILE["model"])
DRAFT = os.environ.get("DRAFT_MODEL", PROFILE["draft_model"])
PAGE_SIZE = PROFILE["page_size"]
WORKLOAD = PROFILE["workload"]
POOL = WORKLOAD["max_total_tokens"]

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
    mem_fraction_static=PROFILE["mem_fraction_static"],
    skip_server_warmup=True,
    enable_torch_compile=True,
    **PROFILE["base_engine"],
)
_SPEC = dict(  # + EAGLE3
    speculative_algorithm="EAGLE3",
    speculative_draft_model_path=DRAFT,
    speculative_num_steps=3,
    speculative_eagle_topk=1,
    speculative_num_draft_tokens=4,
)

_DP = dict(
    tp_size=8,
    dp_size=2,
    enable_dp_attention=True,
    enable_dp_lm_head=True,
    **PROFILE["dp_engine"],
)

COMBOS = {
    # Default = the full-host serving shape (v7x host = 8 devices): real TP
    # sharding (per-rank KV pools, sharded draft), the shape that ships.
    "1": dict(name="prefix+spec+kvoffload", engine={**_BASE, "tp_size": 8, **_SPEC}),
    # Combo 2: DP attention with prefix caching and KV offloading. Its profile
    # chooses KV precision for the exact-output oracle.
    "2": dict(
        name="prefix+dp+kvoffload",
        engine={**_BASE, **_DP},
    ),
    "3": dict(name="prefix+dp+kvoffload+spec (quad)", engine={**_BASE, **_DP, **_SPEC}),
}


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


COUNT_LONG = _count_prompt(WORKLOAD["long_prompt_num_integers"])
MIN_CACHED_TOKENS = WORKLOAD["min_cached_pages"] * PAGE_SIZE
SYSTEM = (
    "The following is reference material. The Sun is the star at the center of "
    "the solar system. Mercury, Venus, Earth, and Mars are the four terrestrial "
    "planets. Jupiter and Saturn are gas giants; Uranus and Neptune are ice "
    "giants. Earth has one moon. Mars has two moons, Phobos and Deimos. Answer "
    "using only the reference material above."
)
SYSTEM = " ".join([SYSTEM] * WORKLOAD["system_repeats"])
QUESTIONS = [f" Q: question number {i}, name any planet. A:" for i in range(4)]
CHURN = [
    f"Article {i}. " + ", ".join(str(i * 1000 + j) for j in range(150)) + "."
    for i in range(WORKLOAD["churn_count"])
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
        cls.fp8_weights = kwargs.get("quantization") == "fp8"
        print(
            f"\n=== profile {MODEL_PROFILE}: {MODEL} ==="
            f"\n=== combo {COMBO}: {cls.combo_name} ==="
        )

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

    def _flush_cache(self):
        result = self.engine.flush_cache(timeout_s=60.0)
        self.assertTrue(result.success, result.message)

    def setUp(self):
        self._flush_cache()

    # ---- helpers ----
    def _gen(self, prompt, max_new=MAX_NEW, *, routed_dp_rank=None):
        out = self.engine.generate(
            prompt,
            {"temperature": 0.0, "max_new_tokens": max_new},
            routed_dp_rank=routed_dp_rank,
        )
        meta = out["meta_info"]
        if routed_dp_rank is not None:
            self.assertEqual(meta.get("dp_rank"), routed_dp_rank)
        verify_ct = meta.get("spec_verify_ct") or 0
        accept_length = (
            (meta.get("completion_tokens") or 0) / verify_ct if verify_ct else 0.0
        )
        return out["text"], meta.get("cached_tokens", 0), verify_ct, accept_length

    def _assert_spec(self, verify_ct, accept_length, where):
        if self.spec_on:
            self.assertGreater(verify_ct, 0, f"{where}: spec never verified")
            self.assertGreater(accept_length, 1.0, f"{where}: no draft accepted")

    def _assert_output(self, actual, expected, message):
        # TODO(#321): FP8 weights can shift close greedy token choices between a
        # cached and recomputed prefix. Keep the cache/offload invariants, but
        # reserve exact text equality for non-FP8 weights.
        if not self.fp8_weights:
            self.assertEqual(actual, expected, message)

    def _prime_all_ranks(self, prompt):
        """Send exactly one cold priming request to every DP radix tree."""
        observations = []
        for dp_rank in range(self.dp_size):
            routed_dp_rank = dp_rank if self.dp_size > 1 else None
            text, cached, v, al = self._gen(
                prompt,
                routed_dp_rank=routed_dp_rank,
            )
            self.assertEqual(
                cached, 0, f"dp_rank={dp_rank} was not cold before priming"
            )
            self._assert_spec(v, al, "priming")
            observations.append((dp_rank, text))

        ref = observations[0][1]
        for dp_rank, text in observations[1:]:
            if text != ref:
                message = (
                    f"cold primer output diverged at dp_rank={dp_rank}: "
                    f"rank 0={ref!r}, rank {dp_rank}={text!r}"
                )
                if self.fp8_weights:
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                else:
                    self.assertEqual(text, ref, message)
        return ref

    def _churn_all_ranks(self):
        """Apply one device-eviction churn wave to every DP radix tree."""
        for dp_rank in range(self.dp_size):
            routed_dp_rank = dp_rank if self.dp_size > 1 else None
            for doc in CHURN:
                self._gen(doc, max_new=1, routed_dp_rank=routed_dp_rank)

    # ---- scenarios ----
    def test_1_warm_hit(self):
        ref = self._prime_all_ranks(COUNT_LONG)
        for k in range(2 * self.dp_size):
            text, c, v, al = self._gen(COUNT_LONG)
            self.assertGreaterEqual(c, MIN_CACHED_TOKENS, f"repeat {k}: prefix miss")
            self._assert_output(text, ref, f"repeat {k}: cache hit changed output")
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
        self._churn_all_ranks()
        for k in range(2 * self.dp_size):
            text, c, v, _ = self._gen(COUNT_LONG)
            # THE hicache assertion: plain radix would miss after eviction;
            # with KV offloading the prefix must reload from the host pool.
            self.assertGreaterEqual(
                c, MIN_CACHED_TOKENS, f"repeat {k}: no host reload after eviction"
            )
            self._assert_output(text, ref, f"repeat {k}: host reload changed output")
            if self.spec_on:
                self.assertGreater(v, 0, f"repeat {k}: spec inactive after reload")

    def test_4_flush_then_repopulate(self):
        self._gen(COUNT_LONG)
        self._flush_cache()
        self._prime_all_ranks(COUNT_LONG)
        _, c2, _, _ = self._gen(COUNT_LONG)
        self.assertGreaterEqual(c2, MIN_CACHED_TOKENS)

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
        self._churn_all_ranks()

        prompts, hot_idx = [], []
        for _ in range(2 * self.dp_size):  # >= 2 hits per DP rank on average
            prompts.extend(CHURN[:3])
            hot_idx.append(len(prompts))
            prompts.append(COUNT_LONG)
        outs = self.engine.generate(
            prompts,
            [{"temperature": 0.6, "max_new_tokens": MAX_NEW}] * len(prompts),
        )
        self.assertEqual(len(outs), len(prompts))
        # Only the correctness oracle: routing/timing decide whether a given
        # hit reloads from host or recomputes, but greedy output must be
        # bit-identical either way. Divergence = lost/garbled reload or a
        # retract that corrupted KV.
        for k, i in enumerate(hot_idx):
            self._assert_output(
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
        for _ in range(3):
            self._churn_all_ranks()
        for k in range(2 * self.dp_size):
            text, _, v, al = self._gen(COUNT_LONG)
            self._assert_output(
                text, ref, f"repeat {k}: output after full-tier eviction diverged"
            )
            self._assert_spec(v, al, f"repeat {k} (post-overflow)")

    def test_7_accept_length_on_par(self):
        # Regression gate: mean accept_length must stay >= 90% of the profile's
        # pinned reference.
        if not self.spec_on:
            self.skipTest("combo has no spec decoding")
        lengths = []
        for q in QUESTIONS:
            _, _, v, al = self._gen(SYSTEM + q)
            if v:
                lengths.append(al)
        self.assertTrue(lengths, "no spec verifications recorded")
        mean_al = sum(lengths) / len(lengths)
        print(f"  mean accept_length={mean_al:.3f} over {len(lengths)} prompts")
        # Reference measured on v7x, combo 1 (Qwen3-1.7B + AngelSlim EAGLE3,
        # greedy, this exact prompt set), 2026-07-29.
        ref = PROFILE["accept_length_ref"]
        if ref is None:
            self.skipTest(
                f"no accept-length reference established for {MODEL_PROFILE}; "
                f"observed mean={mean_al:.3f}"
            )
        self.assertGreaterEqual(
            mean_al,
            0.9 * float(ref),
            f"mean accept_length {mean_al:.3f} < 90% of reference {ref}",
        )


def _run_all():
    failures = []
    for combo, cfg in COMBOS.items():
        print(f"\n{'#' * 60}\n# combo {combo}: {cfg['name']}\n{'#' * 60}")
        env = {**os.environ, "COMBO": combo, "MODEL_PROFILE": MODEL_PROFILE}
        rc = subprocess.call(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--model-profile",
                MODEL_PROFILE,
                combo,
            ],
            env=env,
        )
        if rc != 0:
            failures.append((combo, rc))

    if failures:
        print("\nFAILED COMBOS:")
        for combo, rc in failures:
            print(f"  combo {combo}: rc={rc}")
        return 1

    print("\nALL COMBOS PASSED")
    return 0


if __name__ == "__main__":
    if ARGS.combo == "all":
        sys.exit(_run_all())
    unittest.main(argv=[sys.argv[0], *UNITTEST_ARGS], verbosity=2)
