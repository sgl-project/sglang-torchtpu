import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.disaggregation.tpu.transfer import PrefillTransferInfo, ReshardDirection

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _info(total_num_head: int) -> PrefillTransferInfo:
    return PrefillTransferInfo(
        room=1,
        decode_endpoint="tcp://127.0.0.1:1",
        num_pages=1,
        kv_packing=1,
        src_tp_rank=0,
        dst_tp_rank=0,
        src_attn_tp_size=4,
        dst_attn_tp_size=8,
        total_num_head=total_num_head,
        n_layers=1,
        page_size=64,
    )


class TestPrefillTransferInfoZeroHeads(CustomTestCase):
    def test_zero_heads_records_error_and_stays_readable(self):
        # The sender thread reads these before check_status reports the error.
        info = _info(0)
        self.assertIn("total_num_head", info._reshard_error)
        self.assertEqual(info.resharding_direction, ReshardDirection.UNSPECIFIED)
        self.assertFalse(info.if_skip_sending)
        self.assertTrue(info.if_send_aux_data)
        self.assertEqual(info.n_lane_to_send, 0)
        self.assertEqual(info.gathered_lanes, 0)

    def test_positive_heads_unchanged(self):
        info = _info(8)
        self.assertIsNone(info._reshard_error)
        self.assertEqual(info.resharding_direction, ReshardDirection.FAN_OUT)


if __name__ == "__main__":
    unittest.main()
