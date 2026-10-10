"""Unit tests for raiden host staging admission.

Pure bookkeeping, so these run anywhere -- no TPU / torch_tpu / tpu_raiden
required.

Run: python -m pytest test/manual/kv_transfer/test_raiden_staging_admission.py
"""

import unittest

from sglang_tpu.disaggregation.raiden.host_pool import StagingAdmission


class TestStagingAdmission(unittest.TestCase):
    def test_admits_reads_while_their_pages_fit(self):
        gate = StagingAdmission(8)
        self.assertTrue(gate.admit("a", 3))
        self.assertTrue(gate.admit("b", 3))
        self.assertTrue(gate.admit("c", 2))
        self.assertEqual(len(gate), 3)
        self.assertEqual(gate.pages(), 8)

    def test_defers_a_read_past_capacity_instead_of_failing(self):
        gate = StagingAdmission(8)
        gate.admit("a", 6)
        self.assertFalse(gate.admit("b", 3))
        self.assertEqual(len(gate), 1)

    def test_small_reads_outnumber_max_length_slots(self):
        # A pool sized for two 16-page reads seats eight 4-page reads.
        gate = StagingAdmission(2 * 16)
        self.assertTrue(all(gate.admit(room, 4) for room in range(8)))
        self.assertFalse(gate.admit(8, 4))

    def test_a_deferred_read_is_admitted_once_pages_free(self):
        gate = StagingAdmission(4)
        gate.admit(1, 2)
        gate.admit(2, 2)
        self.assertFalse(gate.admit(3, 2))
        gate.release(1)
        self.assertTrue(gate.admit(3, 2))
        self.assertEqual(sorted(gate.rooms()), [2, 3])

    def test_a_waiting_read_holds_the_head_of_the_line(self):
        gate = StagingAdmission(8)
        gate.admit("a", 6)
        self.assertFalse(gate.admit("big", 4))
        # Two pages are free, but the smaller read waits behind the larger.
        self.assertFalse(gate.admit("small", 2))
        self.assertEqual(gate.waiting, "big")
        gate.release("a")
        self.assertTrue(gate.admit("big", 4))
        self.assertIsNone(gate.waiting)
        self.assertTrue(gate.admit("small", 2))

    def test_a_forgotten_read_gives_up_its_place(self):
        gate = StagingAdmission(8)
        gate.admit("a", 6)
        self.assertFalse(gate.admit("big", 4))
        gate.forget("big")
        self.assertTrue(gate.admit("small", 2))

    def test_sliding_window_pool_binds_on_its_own(self):
        gate = StagingAdmission(64, swa_capacity=2)
        self.assertTrue(gate.admit("a", 4, swa_pages=1))
        self.assertTrue(gate.admit("b", 4, swa_pages=1))
        # Refused on the sliding-window pool alone, and nothing is reserved
        # on either pool for the refused read.
        self.assertFalse(gate.admit("c", 4, swa_pages=1))
        self.assertEqual(sorted(gate.rooms()), ["a", "b"])
        self.assertEqual(gate.pages(), 8)
        gate.release("a")
        self.assertTrue(gate.admit("c", 4, swa_pages=1))
        self.assertEqual(gate.swa_pages(), 2)

    def test_readmitting_an_in_flight_read_is_idempotent(self):
        gate = StagingAdmission(4)
        self.assertTrue(gate.admit(7, 3))
        self.assertTrue(gate.admit(7, 3))
        self.assertEqual(len(gate), 1)
        self.assertEqual(gate.pages(), 3)

    def test_releasing_an_unknown_read_is_a_noop(self):
        gate = StagingAdmission(2)
        gate.admit(1, 1)
        gate.release(42)
        self.assertEqual(gate.rooms(), [1])

    def test_capacity_is_at_least_one_page(self):
        self.assertTrue(StagingAdmission(0).admit(1, 1))

    def test_a_read_the_pool_can_never_seat_is_an_error(self):
        with self.assertRaises(ValueError):
            StagingAdmission(4).admit(1, 5)
        with self.assertRaises(ValueError):
            StagingAdmission(4).admit(1, 1, swa_pages=1)

    def test_concurrency_above_capacity_drains_in_waves(self):
        gate = StagingAdmission(4)
        pending = list(range(16))
        waves = 0
        while pending:
            admitted = [room for room in pending if gate.admit(room, 1)]
            self.assertLessEqual(gate.pages(), 4)
            for room in admitted:
                gate.release(room)
                pending.remove(room)
            waves += 1
        self.assertEqual(waves, 4)


if __name__ == "__main__":
    unittest.main()
