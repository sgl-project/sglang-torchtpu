import os
import tempfile
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.utils.common import (
    _PYOBJ_TENSOR_BUFFER_MIN_BYTES,
    _dumps_with_tensor_buffers,
    _loads_with_tensor_buffers,
    broadcast_pyobj,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_WORLD_SIZE = 3


def _payload():
    big = torch.arange(_PYOBJ_TENSOR_BUFFER_MIN_BYTES, dtype=torch.bfloat16).view(4, -1)
    tagged = torch.zeros(_PYOBJ_TENSOR_BUFFER_MIN_BYTES, dtype=torch.uint8)
    tagged.tag = "kept"
    return [
        {"rid": "a", "feature": big, "again": big, "small": torch.arange(5)},
        {
            "rid": "b",
            "feature": torch.ones(3, 512, 512, dtype=torch.float32).transpose(1, 2),
        },
        {
            "rid": "c",
            "mask": torch.ones(_PYOBJ_TENSOR_BUFFER_MIN_BYTES, dtype=torch.bool),
            "tagged": tagged,
            "np_empty": np.zeros((0,), dtype=np.float32),
            "np_small": np.arange(4, dtype=np.float32),
        },
    ]


def _assert_payload_equal(test, got, expected):
    test.assertEqual(len(got), len(expected))
    for got_req, expected_req in zip(got, expected):
        test.assertEqual(got_req.keys(), expected_req.keys())
        for key, value in expected_req.items():
            if isinstance(value, torch.Tensor):
                test.assertEqual(got_req[key].dtype, value.dtype)
                test.assertTrue(torch.equal(got_req[key], value), key)
                test.assertEqual(got_req[key].__dict__, value.__dict__, key)
            elif isinstance(value, np.ndarray):
                test.assertEqual(got_req[key].dtype, value.dtype)
                test.assertTrue(np.array_equal(got_req[key], value), key)
            else:
                test.assertEqual(got_req[key], value)


def _broadcast_worker(rank, init_file, results):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=_WORLD_SIZE
    )
    try:
        # A non-zero src on an explicit gloo group, as in the scheduler's TP broadcast.
        group = dist.new_group([0, 1, 2], backend="gloo")
        data = _payload() if rank == 1 else None
        out = broadcast_pyobj(data, rank, group, src=1)
        _assert_payload_equal(unittest.TestCase(), out, _payload())
        empty = broadcast_pyobj([] if rank == 1 else None, rank, group, src=1)
        results[rank] = empty == []
    finally:
        dist.destroy_process_group()


class TestBroadcastPyobj(CustomTestCase):
    def test_large_tensors_travel_out_of_band(self):
        data = _payload()
        stream, buffers = _dumps_with_tensor_buffers(data)
        # The bf16 feature (once, despite two references) and the bool mask; small,
        # non-contiguous and attribute-carrying tensors and numpy arrays stay in band.
        self.assertEqual(len(buffers), 2)
        self.assertEqual(
            [b.numel() for b in buffers],
            [2 * _PYOBJ_TENSOR_BUFFER_MIN_BYTES, _PYOBJ_TENSOR_BUFFER_MIN_BYTES],
        )

        received = [buffer.clone() for buffer in buffers]
        out = _loads_with_tensor_buffers(bytes(stream), received)
        _assert_payload_equal(self, out, data)
        self.assertEqual(out[0]["feature"].data_ptr(), received[0].data_ptr())
        self.assertIs(out[0]["feature"], out[0]["again"])
        out[0]["feature"].add_(1)  # rebuilt tensors are writable

    def test_broadcast_across_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = mp.Manager().dict()
            mp.spawn(
                _broadcast_worker,
                args=(os.path.join(tmp, "init"), results),
                nprocs=_WORLD_SIZE,
            )
            self.assertEqual(dict(results), {rank: True for rank in range(_WORLD_SIZE)})


if __name__ == "__main__":
    unittest.main()
