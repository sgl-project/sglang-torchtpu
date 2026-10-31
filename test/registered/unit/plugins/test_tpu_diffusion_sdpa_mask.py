# SPDX-License-Identifier: Apache-2.0
"""CPU tests for sglang_tpu's diffusion SDPA mask patch (sglang_tpu.multimodal_gen.sdpa_mask)."""

import os
import sys
import unittest
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.multimodal_gen import sdpa_mask  # noqa: E402


def _tpu_query(*shape):
    query = MagicMock()
    query.device.type = "tpu"
    query.shape = torch.Size(shape)
    return query


class TestSdpaQueryBroadcastMask(CustomTestCase):
    """torch_tpu SDPA returns NaN for masks that broadcast over queries."""

    def test_key_padding_mask_is_expanded_over_queries(self):
        mask = torch.zeros(2, 1, 1, 8)
        out = sdpa_mask.expand_query_broadcast_mask(_tpu_query(2, 4, 5, 64), mask)
        self.assertEqual(out.shape, (2, 1, 5, 8))
        # expand() is a view: no copy of the mask.
        self.assertEqual(out.data_ptr(), mask.data_ptr())

    def test_other_masks_are_untouched(self):
        expand = sdpa_mask.expand_query_broadcast_mask
        full = torch.zeros(2, 1, 5, 8)
        self.assertIs(expand(_tpu_query(2, 4, 5, 64), full), full)
        # Single-query attention needs no expansion.
        row = torch.zeros(2, 1, 1, 8)
        self.assertIs(expand(_tpu_query(2, 4, 1, 64), row), row)
        self.assertIsNone(expand(_tpu_query(2, 4, 5, 64), None))
        # Non-TPU tensors are never touched.
        cpu_query = torch.zeros(2, 4, 5, 64)
        self.assertIs(expand(cpu_query, row), row)

    def test_patch_is_idempotent_and_preserves_results(self):
        current = torch.nn.functional.scaled_dot_product_attention
        # A TPU worker in this process may already have installed the patch.
        original = getattr(current, "__wrapped__", current)
        try:
            torch.nn.functional.scaled_dot_product_attention = original
            sdpa_mask.install()
            patched = torch.nn.functional.scaled_dot_product_attention
            sdpa_mask.install()
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, patched)
            self.assertIs(patched.__wrapped__, original)

            q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
            mask = torch.zeros(1, 1, 1, 5)
            mask[..., :2] = float("-inf")
            torch.testing.assert_close(
                patched(q, k, v, attn_mask=mask), original(q, k, v, attn_mask=mask)
            )
            torch.testing.assert_close(
                patched(q, k, v, mask, 0.0, False), original(q, k, v, mask, 0.0, False)
            )
        finally:
            torch.nn.functional.scaled_dot_product_attention = current


if __name__ == "__main__":
    unittest.main()
