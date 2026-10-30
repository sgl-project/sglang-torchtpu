# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the MLA v2 tuned block-size table and its lookup.

Ported from sglang-jax ``python/sgl_jax/test/test_tuned_block_sizes_mla.py``
(pytest fixtures -> unittest/``CustomTestCase``, import paths rewritten to the
vendored copy) and extended with two checks on the *shipped* table, which the
upstream test does not have:

  * every entry is well-formed for the schema the kernel expects, and
  * every entry is actually *reachable* through ``get_tuned_block_sizes_mla``
    — a key whose ``num_q_heads`` / ``page_size`` / ``max_num_tokens`` is not
    already a power of two can never be hit (the lookup normalizes those three
    fields via ``next_power_of_2``), so it would silently do nothing.

Pure lookup logic: no TPU and no pallas compile. ``get_tpu_version`` /
``get_device_name`` both read ``jax.devices()[0]``, so they are patched out and
the whole file runs on a CPU-only host as long as JAX imports.

``_DECODE_ARGS`` / ``_MIXED_ARGS`` are deliberately a shape that is *not* in the
shipped table for any device, so the "miss" assertions stay valid as the table
grows: 8 q heads exists only at page_size 256 on v7, and mnt=128 is a decode
bucket there, so this key is only ever a hit when a test inserts it.
"""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from unittest import mock

import jax.numpy as jnp

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention.pallas import tuned_block_sizes_mla as tbs
from sglang_tpu.attention.pallas.tuned_block_sizes_mla import (
    get_tuned_block_sizes_mla,
)
from sglang_tpu.attention.pallas.util import next_power_of_2

_DECODE_ARGS = ("decode", "bfloat16", "bfloat16", 8, 512, 64, 256, 128)
_MIXED_ARGS = ("mixed", "bfloat16", "bfloat16", 8, 512, 64, 256, 512)


class _TunedTableTestBase(CustomTestCase):
    """Snapshot/restore the module-level table and warned-misses set.

    ``TPU_VERSION`` / ``DEVICE_NAME`` pretend we are on a v7 host, since both
    lookup helpers would otherwise touch a device. The warned-misses set starts
    clean because a miss only logs once per key per process — otherwise the log
    assertions would depend on test execution order.
    """

    TPU_VERSION = 7
    DEVICE_NAME = "TPU v7"

    def setUp(self):
        super().setUp()
        table_snapshot = {k: dict(v) for k, v in tbs.TUNED_BLOCK_SIZES_MLA.items()}
        warned_snapshot = set(tbs._WARNED_MISSES)

        def _restore():
            tbs.TUNED_BLOCK_SIZES_MLA.clear()
            tbs.TUNED_BLOCK_SIZES_MLA.update(table_snapshot)
            tbs._WARNED_MISSES.clear()
            tbs._WARNED_MISSES.update(warned_snapshot)

        self.addCleanup(_restore)
        tbs._WARNED_MISSES.clear()
        self.patch_env()

    def patch_env(self, *, tpu_version=None, device=None):
        for target, value in (
            (
                "get_tpu_version",
                self.TPU_VERSION if tpu_version is None else tpu_version,
            ),
            ("get_device_name", self.DEVICE_NAME if device is None else device),
        ):
            patcher = mock.patch.object(tbs, target, lambda _v=value: _v)
            patcher.start()
            self.addCleanup(patcher.stop)


class TestTunedBlockSizesMlaLookup(_TunedTableTestBase):
    """Lookup semantics, against a table emptied of shipped entries.

    Hermetic by construction: every assertion runs against entries this class
    inserts, never against whatever happens to be tuned for v7 today.
    """

    def setUp(self):
        super().setUp()
        tbs.TUNED_BLOCK_SIZES_MLA[self.DEVICE_NAME] = {}

    def _put(self, key, value):
        tbs.TUNED_BLOCK_SIZES_MLA[self.DEVICE_NAME][key] = value

    def test_empty_table_returns_none(self):
        self.assertIsNone(get_tuned_block_sizes_mla(*_DECODE_ARGS))
        self.assertIsNone(get_tuned_block_sizes_mla(*_MIXED_ARGS))

    def test_decode_hit_returns_3tuple(self):
        self._put(_DECODE_ARGS, (3, 2, 8))
        self.assertEqual(get_tuned_block_sizes_mla(*_DECODE_ARGS), (3, 2, 8))

    def test_mixed_hit_returns_2tuple(self):
        self._put(_MIXED_ARGS, (4, 32))
        self.assertEqual(get_tuned_block_sizes_mla(*_MIXED_ARGS), (4, 32))

    def test_decode_and_mixed_buckets_are_separate(self):
        """Same shape under a different case_label must not collide.

        The kernel uses slot[0] for both decode passes and slot[2] for MIXED;
        if the two buckets aliased, a decode-tuned block size would be applied
        to a prefill-sized block and OOM VMEM (or vice versa, silently slow).
        """
        self._put(_DECODE_ARGS, (3, 2, 8))
        self.assertEqual(get_tuned_block_sizes_mla(*_DECODE_ARGS), (3, 2, 8))
        self.assertIsNone(get_tuned_block_sizes_mla("mixed", *_DECODE_ARGS[1:]))

    def test_key_normalization_pow2_buckets(self):
        """num_q_heads, page_size and max_num_tokens bucket by next_power_of_2.

        mnt=100 rounds up to the stored 128 bucket and hits; mnt=200 rounds to
        256, a different bucket, and misses. num_q_heads=5 -> 8 together with
        page_size=200 -> 256 land back on the stored key.
        """
        self._put(_DECODE_ARGS, (3, 2, 8))
        self.assertEqual(get_tuned_block_sizes_mla(*_DECODE_ARGS[:-1], 100), (3, 2, 8))
        self.assertIsNone(get_tuned_block_sizes_mla(*_DECODE_ARGS[:-1], 200))
        self.assertEqual(
            get_tuned_block_sizes_mla(
                "decode", "bfloat16", "bfloat16", 5, 512, 64, 200, 128
            ),
            (3, 2, 8),
        )

    def test_dtype_objects_and_names_are_equivalent_keys(self):
        """Callers pass jnp dtypes; the table stores dtype *names*.

        A different dtype is a different key — fp8 KV would need its own tune.
        """
        self._put(_DECODE_ARGS, (3, 2, 8))
        self.assertEqual(
            get_tuned_block_sizes_mla(
                "decode", jnp.bfloat16, jnp.bfloat16, 8, 512, 64, 256, 128
            ),
            (3, 2, 8),
        )
        self.assertIsNone(
            get_tuned_block_sizes_mla(
                "decode", jnp.float32, jnp.bfloat16, 8, 512, 64, 256, 128
            )
        )

    def test_invalid_case_label_raises(self):
        """``prefill`` is the kernel's dead slot[1].

        Accepting it would hand the caller a config that is never applied.
        """
        with self.assertRaises(ValueError):
            get_tuned_block_sizes_mla("prefill", *_DECODE_ARGS[1:])

    def test_unknown_device_returns_none(self):
        self.patch_env(device="TPU vUNKNOWN")
        self.assertIsNone(get_tuned_block_sizes_mla(*_DECODE_ARGS))

    def test_unknown_device_does_not_log_a_miss(self):
        """Unsupported device != supported-but-untuned; only the latter warns."""
        self.patch_env(device="TPU vUNKNOWN")
        with self.assertNoLogs(tbs.logger.name, level="INFO"):
            get_tuned_block_sizes_mla(*_DECODE_ARGS)

    def test_pre_v5_returns_none(self):
        self.patch_env(tpu_version=4)
        self.assertIsNone(get_tuned_block_sizes_mla(*_DECODE_ARGS))

    def test_warned_misses_is_one_shot_per_key(self):
        """Each unique miss-key logs INFO at most once (per process)."""
        with self.assertLogs(tbs.logger.name, level="INFO") as ctx:
            for _ in range(3):
                get_tuned_block_sizes_mla(*_DECODE_ARGS)
        self.assertEqual(len(self._miss_records(ctx)), 1)

        with self.assertLogs(tbs.logger.name, level="INFO") as ctx:
            get_tuned_block_sizes_mla(*_MIXED_ARGS)
        self.assertEqual(len(self._miss_records(ctx)), 1)

    @staticmethod
    def _miss_records(ctx):
        return [
            r
            for r in ctx.records
            if "MLA tuned-block-size LOOKUP MISS" in r.getMessage()
        ]


class TestShippedTunedTable(_TunedTableTestBase):
    """The entries actually shipped in this file must be usable.

    A malformed or non-normalized key is invisible at runtime — the lookup just
    misses and the kernel silently falls back to the hardcoded defaults — so it
    can only be caught here.
    """

    _POW2_FIELDS = ("num_q_heads", "page_size", "max_num_tokens")

    def test_entries_are_wellformed(self):
        for device, table in tbs.TUNED_BLOCK_SIZES_MLA.items():
            for key, cfg in table.items():
                with self.subTest(device=device, key=key):
                    self.assertEqual(len(key), 8, "key arity must match the schema")
                    case_label, q_dtype, kv_dtype, heads, lkv, rope, page, mnt = key

                    self.assertIn(case_label, ("decode", "mixed"))
                    for dtype_name in (q_dtype, kv_dtype):
                        self.assertEqual(
                            jnp.dtype(dtype_name).name,
                            dtype_name,
                            f"{dtype_name!r} is not the canonical dtype name, so "
                            "the lookup's jnp.dtype(...).name can never match it",
                        )

                    for field, value in zip(self._POW2_FIELDS, (heads, page, mnt)):
                        self.assertEqual(
                            next_power_of_2(value),
                            value,
                            f"{field}={value} is not a power of two; the lookup "
                            "normalizes it, so this entry can never be hit",
                        )
                    self.assertGreater(lkv, 0, "kv_lora_rank is a model constant")
                    self.assertGreater(rope, 0, "qk_rope_head_dim is a model constant")

                    self.assertEqual(
                        len(cfg),
                        3 if case_label == "decode" else 2,
                        "decode carries decode_batch_size, mixed does not",
                    )
                    for value in cfg:
                        self.assertIsInstance(value, int)
                        self.assertGreater(value, 0)

    def test_entries_are_reachable_through_the_public_lookup(self):
        """The key tuple is exactly the lookup's argument order."""
        for device, table in tbs.TUNED_BLOCK_SIZES_MLA.items():
            with mock.patch.object(tbs, "get_device_name", lambda _d=device: _d):
                for key, cfg in table.items():
                    with self.subTest(device=device, key=key):
                        self.assertEqual(get_tuned_block_sizes_mla(*key), cfg)

    def test_kimi_k3_fp8_kv_serving_buckets_are_tuned(self):
        """Kimi-K3 TP32 (3 q heads per shard, padded to 4) with an fp8 KV
        cache: every decode bucket of its recipes and every prefill chunk size
        has a tuned row. Without them the kernel falls back to blocks that ran
        2x slower at decode and up to 20x slower at prefill."""
        with mock.patch.object(tbs, "get_device_name", lambda: "TPU v7"):
            for bs in (1, 2, 4, 8):
                with self.subTest(case="decode", mnt=bs):
                    self.assertIsNotNone(
                        get_tuned_block_sizes_mla(
                            "decode",
                            jnp.bfloat16,
                            jnp.float8_e4m3fn,
                            3,
                            512,
                            64,
                            128,
                            bs,
                        )
                    )
            for tokens in (512, 1024, 4096, 8192):
                with self.subTest(case="mixed", mnt=tokens):
                    self.assertIsNotNone(
                        get_tuned_block_sizes_mla(
                            "mixed",
                            jnp.bfloat16,
                            jnp.float8_e4m3fn,
                            3,
                            512,
                            64,
                            128,
                            tokens,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
