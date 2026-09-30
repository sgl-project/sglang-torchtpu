"""Patch-count bucketing for eager TPU vision towers."""

import unittest

import torch

from sglang.srt.hardware_backend.tpu.vision_shapes import (
    pad_patches_to_bucket,
    patch_count_buckets,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Gemma4: vision_soft_tokens_per_image 280 x pooling_kernel_size 3^2.
POOLING_WINDOW = 9
MAX_PATCHES = 280 * POOLING_WINDOW


class TestPatchCountBuckets(unittest.TestCase):
    def test_ladder_is_aligned_ascending_and_ends_at_max(self):
        buckets = patch_count_buckets(MAX_PATCHES, align=POOLING_WINDOW)
        self.assertEqual(buckets[-1], MAX_PATCHES)
        self.assertEqual(buckets, sorted(set(buckets)))
        # Every bucket must be a whole number of pooled tokens, or the pooler's
        # reshape fails on the padded length.
        for bucket in buckets:
            self.assertEqual(bucket % POOLING_WINDOW, 0, bucket)

    def test_num_buckets_bounds_the_ladder(self):
        for num_buckets in (1, 2, 4, 8):
            buckets = patch_count_buckets(
                MAX_PATCHES, align=POOLING_WINDOW, num_buckets=num_buckets
            )
            self.assertLessEqual(len(buckets), num_buckets + 1)
            self.assertEqual(buckets[-1], MAX_PATCHES)

    def test_rejects_a_max_that_is_not_a_whole_number_of_windows(self):
        with self.assertRaises(ValueError):
            patch_count_buckets(MAX_PATCHES + 1, align=POOLING_WINDOW)


class TestPadPatchesToBucket(unittest.TestCase):
    def setUp(self):
        self.buckets = patch_count_buckets(MAX_PATCHES, align=POOLING_WINDOW)

    def _image(self, patches):
        pixel_values = torch.arange(patches * 4, dtype=torch.float32).reshape(
            1, patches, 4
        )
        position_ids = torch.arange(patches * 2, dtype=torch.int32).reshape(
            1, patches, 2
        )
        return pixel_values, position_ids

    def test_pads_up_to_the_smallest_fitting_bucket(self):
        patches = self.buckets[0] + 1
        pv, pp = pad_patches_to_bucket(*self._image(patches), self.buckets)
        self.assertEqual(pv.shape[-2], self.buckets[1])
        self.assertEqual(pp.shape[-2], self.buckets[1])

    def test_padding_carries_the_mask_sentinel_and_keeps_real_rows(self):
        patches = self.buckets[0] - 3
        pixel_values, position_ids = self._image(patches)
        pv, pp = pad_patches_to_bucket(pixel_values, position_ids, self.buckets)
        torch.testing.assert_close(pv[:, :patches], pixel_values)
        torch.testing.assert_close(pp[:, :patches], position_ids)
        # The tower masks a patch when every position component is -1.
        self.assertTrue(bool((pp[:, patches:] == -1).all()))
        self.assertTrue(bool((pv[:, patches:] == 0).all()))

    def test_exact_bucket_and_oversize_are_untouched(self):
        for patches in (self.buckets[0], MAX_PATCHES, MAX_PATCHES + 1):
            pixel_values, position_ids = self._image(patches)
            pv, pp = pad_patches_to_bucket(pixel_values, position_ids, self.buckets)
            self.assertIs(pv, pixel_values)
            self.assertIs(pp, position_ids)


if __name__ == "__main__":
    unittest.main()
