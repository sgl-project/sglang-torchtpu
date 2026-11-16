"""Tests for SGLANG_TPU_MUSE_MOE_MODE parsing (host-only)."""

import os
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.muse_megakernel.moe_mode import parse_moe_mode, resolve_moe_mode

register_tpu_ci(est_time=1, suite="stage-a-test-tpu-unit", chips=1)


class TestMuseMoeMode(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("MUSE_PF_MOE_EP", None)
        os.environ.pop("SGLANG_TPU_MUSE_MOE_MODE", None)

    def test_auto_follows_ep_size(self):
        self.assertEqual(parse_moe_mode("auto", 1), "tp")
        self.assertEqual(parse_moe_mode("auto", 8), "ep")
        self.assertEqual(resolve_moe_mode(1), "tp")
        self.assertEqual(resolve_moe_mode(8), "ep")

    def test_explicit_modes(self):
        for ep_size in (1, 8):
            self.assertEqual(parse_moe_mode("tp", ep_size), "tp")
            self.assertEqual(parse_moe_mode("ep", ep_size), "ep")
        os.environ["SGLANG_TPU_MUSE_MOE_MODE"] = "ep"
        self.assertEqual(resolve_moe_mode(1), "ep")

    def test_invalid_mode_raises(self):
        for value in ("", "EP", "1", "gather"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_moe_mode(value, 1)
        os.environ["SGLANG_TPU_MUSE_MOE_MODE"] = "tp8"
        with self.assertRaises(ValueError):
            resolve_moe_mode(1)

    def test_legacy_env_raises(self):
        for value in ("1", "0"):
            os.environ["MUSE_PF_MOE_EP"] = value
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "SGLANG_TPU_MUSE_MOE_MODE"):
                    parse_moe_mode("tp", 1)


if __name__ == "__main__":
    unittest.main()
