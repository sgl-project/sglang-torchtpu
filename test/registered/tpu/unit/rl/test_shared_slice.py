"""A shared-slice layout the join could only hang on must fail before the runtime starts;
only a worker in a shared slice serves the ICI receiver."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=5, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase
from sglang_tpu.rl.ici_receiver import ICI_RECEIVE_HANDLER
from sglang_tpu.rl.setup import RlSetup, setup_worker, weight_update_handlers
from sglang_tpu.rl.shared_slice import SharedSliceSpec, check_layout

_SLICE = SharedSliceSpec(rank_offset=4, size=8, store_host="127.0.0.1", store_port=1)
_SLICE_ENV = {
    "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(
        f"localhost:{8070 + i}" for i in range(8)
    ),
    "TORCH_TPU_TOPOLOGY": "2,2,1,2",
}


class TestCheckLayout(CustomTestCase):
    def test_a_job_that_fits_the_slice_passes(self):
        with patch.dict("os.environ", _SLICE_ENV):
            check_layout(_SLICE, job_size=4)

    def test_a_job_past_the_end_of_the_slice_is_rejected(self):
        with patch.dict("os.environ", _SLICE_ENV):
            with self.assertRaisesRegex(ValueError, "outside a slice of 8"):
                check_layout(_SLICE, job_size=6)

    def test_addresses_of_the_job_alone_are_rejected(self):
        env = {
            **_SLICE_ENV,
            "TORCH_TPU_SLICEBUILDER_ADDRESSES": "localhost:1,localhost:2",
        }
        with patch.dict("os.environ", env):
            with self.assertRaisesRegex(ValueError, "slice-wide addresses"):
                check_layout(_SLICE, job_size=2)

    def test_a_topology_of_the_job_alone_is_rejected(self):
        with patch.dict("os.environ", {**_SLICE_ENV, "TORCH_TPU_TOPOLOGY": "1,2,1,2"}):
            with self.assertRaisesRegex(ValueError, "slice's topology"):
                check_layout(_SLICE, job_size=4)


class TestWeightUpdateHandlers(CustomTestCase):
    def test_only_a_worker_in_a_shared_slice_serves_the_ici_receiver(self):
        with patch.dict("os.environ", {"SGLANG_TPU_SHARED_SLICE": ""}):
            outside = setup_worker(job_size=1)
        inside = RlSetup(shared_slice=SimpleNamespace())

        self.assertEqual(weight_update_handlers(outside), {})
        self.assertEqual(list(weight_update_handlers(inside)), [ICI_RECEIVE_HANDLER])


if __name__ == "__main__":
    unittest.main()
