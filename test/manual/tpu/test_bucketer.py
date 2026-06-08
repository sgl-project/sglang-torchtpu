"""Unit tests for the TPU shape bucketer."""

import unittest

from sglang.srt.hardware_backend.tpu.compilation.bucketer import (
    Bucketer,
    PowerBucketer,
)
from sglang.test.test_utils import CustomTestCase


class TestBucketer(CustomTestCase):
    """Test Bucketer boundaries and error handling."""

    def test_sorts_and_dedupes_input_list(self):
        b = Bucketer(name="t", buckets=[8, 1, 4, 8, 2])
        self.assertEqual(b.buckets, [1, 2, 4, 8])
        self.assertEqual(b.min_val, 1)
        self.assertEqual(b.max_val, 8)

    def test_empty_list_raises(self):
        with self.assertRaises(ValueError):
            Bucketer(name="t", buckets=[])

    def test_non_positive_bucket_raises(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            Bucketer(name="t", buckets=[0, 4])

        with self.assertRaisesRegex(ValueError, "must be positive"):
            Bucketer(name="t", buckets=[-1, 4])

    def test_next_bucket_exact_hit(self):
        b = Bucketer(name="t", buckets=[1, 4, 16])
        self.assertEqual(b.next_bucket(4), 4)
        self.assertEqual(b.next_bucket(1), 1)
        self.assertEqual(b.next_bucket(16), 16)

    def test_next_bucket_rounds_up_to_next(self):
        b = Bucketer(name="t", buckets=[1, 4, 16])
        self.assertEqual(b.next_bucket(2), 4)
        self.assertEqual(b.next_bucket(3), 4)
        self.assertEqual(b.next_bucket(5), 16)

    def test_next_bucket_above_max_raises(self):
        b = Bucketer(name="t", buckets=[1, 4, 16])
        with self.assertRaisesRegex(ValueError, "exceeds the maximum"):
            b.next_bucket(17)

    def test_repr_includes_buckets(self):
        b = Bucketer(name="t", buckets=[1, 4, 16])
        self.assertIn("buckets=[1, 4, 16]", repr(b))


class TestPowerBucketer(CustomTestCase):
    """Test PowerBucketer geometric progressions."""

    def test_power_of_two_basic(self):
        pb = PowerBucketer(name="t", min_val=1, max_val=16, power=2)
        self.assertEqual(pb.buckets, [1, 2, 4, 8, 16])

    def test_max_not_a_clean_power_is_still_appended(self):
        pb = PowerBucketer(name="t", min_val=64, max_val=5000, power=2)
        self.assertEqual(pb.buckets, [64, 128, 256, 512, 1024, 2048, 4096, 5000])

    def test_min_equals_max(self):
        pb = PowerBucketer(name="b", min_val=4, max_val=4, power=2)
        self.assertEqual(pb.buckets, [4])

    def test_power_three_progression(self):
        pb = PowerBucketer(name="t", min_val=1, max_val=27, power=3)
        self.assertEqual(pb.buckets, [1, 3, 9, 27])

    def test_next_bucket_above_max_raises(self):
        pb = PowerBucketer(name="t", min_val=64, max_val=2048, power=2)
        self.assertIsInstance(pb, Bucketer)
        self.assertEqual(pb.next_bucket(100), 128)
        self.assertEqual(pb.next_bucket(2048), 2048)
        with self.assertRaisesRegex(ValueError, "exceeds the maximum"):
            pb.next_bucket(2049)

    def test_invalid_min_val_raises(self):
        with self.assertRaisesRegex(ValueError, "min_val must be >= 1"):
            PowerBucketer(name="t", min_val=0, max_val=8, power=2)

    def test_max_less_than_min_raises(self):
        with self.assertRaisesRegex(ValueError, "must be >="):
            PowerBucketer(name="t", min_val=8, max_val=4, power=2)

    def test_invalid_power_raises(self):
        with self.assertRaisesRegex(ValueError, "power must be >= 2"):
            PowerBucketer(name="t", min_val=1, max_val=8, power=1)


if __name__ == "__main__":
    unittest.main()
