"""StateStorage.acquire writes each table column on that column's own device: the
ping-pong mapping follows the slot ids (host on TPU), not the table."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.kv.state_storage import StateStorage
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Slots:
    def __init__(self):
        self._next = 10

    def alloc(self, count):
        out = torch.arange(self._next, self._next + count, dtype=torch.int64)
        self._next += count
        return out

    def available_size(self):
        return 100


def _req():
    return SimpleNamespace(
        kv=SimpleNamespace(
            holds_mamba=False,
            mamba_pool_idx=None,
            mamba_needs_clear=False,
            mamba_ping_pong_track_buffer=None,
        )
    )


class TestStateStorageDevices(CustomTestCase):
    def test_host_ping_pong_mapping_beside_an_off_host_table(self):
        num_rows, track_slots = 4, 2
        # "meta" stands in for the TPU table; indexing a host tensor with a
        # meta index silently writes nothing, which the values below catch.
        table = SimpleNamespace(
            device=torch.device("meta"),
            enable_mamba_extra_buffer=True,
            enable_mamba_extra_buffer_lazy=False,
            mamba_initial_tracking_slots=track_slots,
            mamba_ping_pong_track_buffer_size=track_slots,
            get_mamba_ping_pong_other_idx=lambda idx: 1 - idx,
            req_index_to_mamba_index_mapping=torch.zeros(
                num_rows, dtype=torch.int32, device="meta"
            ),
            req_index_to_mamba_ping_pong_track_buffer_mapping=torch.zeros(
                (num_rows, track_slots), dtype=torch.int64
            ),
        )
        storage = StateStorage(
            sub_pool="state",
            pool=SimpleNamespace(replayssm_write_pos=None),
            slots=_Slots(),
            layer_map={},
            short_conv_pool=None,
            ngram_pool=None,
        )
        reqs = [_req(), _req()]

        storage.acquire(table, reqs, [1, 3])

        mapping = table.req_index_to_mamba_ping_pong_track_buffer_mapping
        self.assertEqual(mapping.device.type, "cpu")
        for row, req in zip([1, 3], reqs):
            self.assertTrue(
                torch.equal(mapping[row], req.kv.mamba_ping_pong_track_buffer)
            )
        self.assertEqual(mapping[[0, 2]].abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
