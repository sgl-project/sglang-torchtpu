"""L3 file-storage HiCache tests for the TPU backend -- single engine, tp=1.

Mirrors ``test/registered/hicache/test_hicache_storage_file_backend.py`` on the
native ``torch_tpu`` KV-offload path (``TpuFusedMHATokenToKVPoolHost`` +
``HiCacheFile``) and adds exact-match correctness: greedy decoding on
wide-margin counting prompts, so a page reloaded from disk must give the same
continuation as a cold prefill.

What each case drives:

* backup writes one ``.bin`` per page, all the same size (host row = one page);
* ``/flush_cache`` drops L1 and L2 but not the files, so the rerun is served by
  storage (``#storage-hit-token`` / ``cached_tokens``) and is bit-identical;
* a prompt sharing only the first half of the pages gets a partial storage hit
  and still matches its own cold run;
* a server restarted on the same directory serves its first request from
  storage (the point of L3);
* missing files fall back to recompute without error.

Requires TPU hardware + torch_tpu. Single chip (tp=1).

Usage:
    SGLANG_PLATFORM=tpu python -m pytest -s \\
        test/registered/tpu/e2e/test_tpu_qwen3_0_6b_singlehost_hicache_file_storage.py
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=360,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "kv-offload", "l3-file"],
)

import os
import shutil
import tempfile
import time
import unittest

import requests

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

PAGE_SIZE = 64
MAX_NEW = 12
DEVICE_POOL = 4096  # tokens; small so the host and storage tiers do real work
STORAGE_SETTLE_S = 30


def _skip_if_no_tpu():
    try:
        import importlib.util

        return importlib.util.find_spec("torch_tpu") is None
    except Exception:
        return True


def _count_prompt(start, n):
    # Wide greedy margin: counting continues deterministically, so a cached
    # (partial-prefill) result equals the recomputed (full-prefill) result.
    return (
        "Continue this integer sequence, comma separated:\n"
        + ", ".join(str(i) for i in range(start, start + n))
        + ","
    )


LONG = _count_prompt(1, 400)  # ~1.9k tokens -> ~29 pages


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostHiCacheFileStorage(CustomTestCase):
    MODEL = "Qwen/Qwen3-0.6B"
    process = None

    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.storage_dir = tempfile.mkdtemp(prefix="hicache-file-")
        cls.process = cls._launch(cls.storage_dir)

    @classmethod
    def tearDownClass(cls):
        cls._stop()
        shutil.rmtree(cls.storage_dir, ignore_errors=True)

    @classmethod
    def _launch(cls, storage_dir):
        other_args = [
            "--device", "tpu", "--dtype", "bfloat16", "--tp-size", "1",
            "--page-size", str(PAGE_SIZE),
            "--mem-fraction-static", "0.4",
            "--max-total-tokens", str(DEVICE_POOL),
            "--chunked-prefill-size", "1024",
            "--context-length", "2048",
            "--max-running-requests", "4",
            "--torch-compile-max-bs", "4",
            "--tpu-compile-tokens", "64", "1024",
            "--tpu-compile-bs", "1", "4",
            "--enable-torch-compile",
            "--grammar-backend", "none",
            # L2: native torch_tpu host pool (the only layout it supports).
            "--enable-hierarchical-cache",
            "--hicache-io-backend", "torch_tpu",
            "--hicache-mem-layout", "page_first_fused",
            "--hicache-ratio", "2",
            "--hicache-write-policy", "write_through",
            # L3: file backend; wait_complete makes storage hits deterministic.
            "--hicache-storage-backend", "file",
            "--hicache-storage-prefetch-policy", "wait_complete",
            "--enable-metrics",
        ]  # fmt: skip
        env = {**os.environ, "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": storage_dir}
        return popen_launch_server(
            cls.MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            device="tpu",
            env=env,
        )

    @classmethod
    def _stop(cls):
        if cls.process is not None:
            terminate_and_kill_process_tree(cls.process)
            cls.process = None

    # -- helpers ---------------------------------------------------------- #

    def _gen(self, text):
        r = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {"max_new_tokens": MAX_NEW, "temperature": 0},
            },
            timeout=300,
        )
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        return d["output_ids"], d["meta_info"]

    def _flush(self):
        # Flush refuses while HiCache backups (host or storage) are in flight;
        # timeout= defers it until the scheduler is fully idle, as upstream's
        # test_hicache_storage_file_backend.py does.
        r = requests.post(
            f"{self.base_url}/flush_cache", params={"timeout": 30}, timeout=60
        )
        self.assertEqual(r.status_code, 200, r.text)

    def _page_files(self):
        return sorted(f for f in os.listdir(self.storage_dir) if f.endswith(".bin"))

    def _wait_for_files(self, at_least, timeout=STORAGE_SETTLE_S):
        deadline = time.time() + timeout
        while time.time() < deadline:
            files = self._page_files()
            if len(files) >= at_least:
                return files
            time.sleep(0.5)
        self.fail(f"expected >= {at_least} page files, found {len(self._page_files())}")

    def _storage_hit_tokens(self):
        """Sum of sglang:cached_tokens_total{cache_source="storage"}."""
        txt = requests.get(f"{self.base_url}/metrics", timeout=30).text
        total = 0.0
        for line in txt.splitlines():
            if (
                line.startswith("sglang:cached_tokens_total")
                and 'cache_source="storage"' in line
            ):
                total += float(line.rsplit(" ", 1)[1])
        return total

    @staticmethod
    def _pages(prompt_tokens):
        return prompt_tokens // PAGE_SIZE

    def setUp(self):
        # Each case starts from empty device/host tiers AND an empty directory.
        self._flush()
        for f in self._page_files():
            os.remove(os.path.join(self.storage_dir, f))

    # -- cases ------------------------------------------------------------ #

    def test_backup_writes_one_file_per_page(self):
        _, meta = self._gen(LONG)
        pages = self._pages(meta["prompt_tokens"])
        self.assertGreater(pages, 4)
        files = self._wait_for_files(pages)
        sizes = {os.path.getsize(os.path.join(self.storage_dir, f)) for f in files}
        self.assertEqual(len(sizes), 1, f"page files differ in size: {sizes}")
        self.assertGreater(sizes.pop(), 0)

    def test_storage_hit_after_flush_matches_cold(self):
        cold_ids, cold_meta = self._gen(LONG)
        pages = self._pages(cold_meta["prompt_tokens"])
        self._wait_for_files(pages)

        hbm_ids, hbm_meta = self._gen(LONG)
        self.assertEqual(hbm_ids, cold_ids)
        self.assertGreater(hbm_meta.get("cached_tokens", 0), 0)

        before = self._storage_hit_tokens()
        self._flush()  # drops L1 + L2; the page files stay
        storage_ids, storage_meta = self._gen(LONG)
        self.assertEqual(storage_ids, cold_ids, "storage reload changed the output")
        # wait_complete: the whole on-disk prefix is loaded before scheduling.
        self.assertGreaterEqual(
            storage_meta.get("cached_tokens", 0), (pages - 1) * PAGE_SIZE
        )
        self.assertGreater(self._storage_hit_tokens(), before)

    def test_partial_prefix_storage_hit(self):
        # Shared ~200-number prefix (~14 pages) followed by a multi-page
        # diverging tail (~9 pages) that LONG never backs up.
        half = (
            _count_prompt(1, 200)
            + " "
            + ", ".join(str(i) for i in range(9000, 9100))
            + ","
        )
        half_cold_ids, half_cold_meta = self._gen(half)
        self._wait_for_files(self._pages(half_cold_meta["prompt_tokens"]))
        self._flush()
        for f in self._page_files():
            os.remove(os.path.join(self.storage_dir, f))

        _, long_meta = self._gen(LONG)
        self._wait_for_files(self._pages(long_meta["prompt_tokens"]))
        self._flush()

        half_storage_ids, half_meta = self._gen(half)
        self.assertEqual(half_storage_ids, half_cold_ids)
        hit = half_meta.get("cached_tokens", 0)
        self.assertGreaterEqual(hit, 10 * PAGE_SIZE)
        self.assertLessEqual(hit, half_cold_meta["prompt_tokens"] - 4 * PAGE_SIZE)

    def test_host_eviction_falls_through_to_storage(self):
        cold_ids, cold_meta = self._gen(LONG)
        pages = self._pages(cold_meta["prompt_tokens"])
        self._wait_for_files(pages)
        # Host pool is 2x the 4096-token device pool: ~9k tokens of distinct
        # traffic evicts LONG from HBM and then from host RAM, with no flush.
        for i in range(8):
            self._gen(_count_prompt(10_000 * (i + 1), 200))
        time.sleep(2)
        before = self._storage_hit_tokens()
        ids, meta = self._gen(LONG)
        self.assertEqual(ids, cold_ids)
        self.assertGreaterEqual(meta.get("cached_tokens", 0), (pages - 1) * PAGE_SIZE)
        self.assertGreater(self._storage_hit_tokens(), before)

    def test_missing_files_fall_back_to_recompute(self):
        cold_ids, cold_meta = self._gen(LONG)
        self._wait_for_files(self._pages(cold_meta["prompt_tokens"]))
        for f in self._page_files():
            os.remove(os.path.join(self.storage_dir, f))
        self._flush()
        ids, meta = self._gen(LONG)
        self.assertEqual(ids, cold_ids)
        self.assertEqual(meta.get("cached_tokens", 0), 0)

    def test_persists_across_server_restart(self):
        cold_ids, cold_meta = self._gen(LONG)
        pages = self._pages(cold_meta["prompt_tokens"])
        self._wait_for_files(pages)

        type(self)._stop()
        type(self).process = self._launch(self.storage_dir)

        ids, meta = self._gen(LONG)
        self.assertEqual(ids, cold_ids)
        self.assertGreaterEqual(meta.get("cached_tokens", 0), (pages - 1) * PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()
