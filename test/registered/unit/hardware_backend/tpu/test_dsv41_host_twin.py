# SPDX-License-Identifier: Apache-2.0
"""CPU tests: an extend whose req_pool_indices reaches the compile policy already on
the device gets its padded req_pool_indices_cpu twin recorded as the host copy, so
the DSV4.1 attention metadata does not read the slots back from the device."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.compilation import host_input_mirror
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.models.deepseek_v41 import attention_backend as ab

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FB(SimpleNamespace):
    """A ForwardBatch stand-in; like the dataclass, weak-referenceable."""


@pytest.fixture(autouse=True)
def _fresh_record():
    host_input_mirror.clear()
    yield
    host_input_mirror.clear()


def _policy():
    policy = object.__new__(DecoderTpuCompilePolicy)
    policy.model_runner = SimpleNamespace(device=torch.device("meta"))
    return policy


def _extend(slots, twin):
    # A meta tensor stands in for the device: reading it back raises.
    return FB(
        spec_info=None,
        req_pool_indices=torch.tensor(slots, dtype=torch.int64, device="meta"),
        req_pool_indices_cpu=twin,
    )


def test_device_slots_take_the_twin():
    fb = _extend([3, 5, 0, 0], torch.tensor([3, 5, 0, 0], dtype=torch.int64))
    _policy()._upload_host_inputs(fb)
    assert ab._host_or_read(fb, "req_pool_indices", 4).tolist() == [3, 5, 0, 0]


@pytest.mark.parametrize(
    "twin",
    [None, torch.tensor([3, 5, 0], dtype=torch.int64)],
    ids=["missing", "other_length"],
)
def test_no_usable_twin_records_nothing(twin):
    fb = _extend([3, 5, 0, 0], twin)
    _policy()._upload_host_inputs(fb)
    assert host_input_mirror.host_copy(fb, "req_pool_indices") is None


def test_rebind_after_the_record_falls_back():
    fb = _extend([3, 5], torch.tensor([3, 5], dtype=torch.int64))
    _policy()._upload_host_inputs(fb)
    fb.req_pool_indices = torch.tensor([3, 6], dtype=torch.int64)
    assert host_input_mirror.host_copy(fb, "req_pool_indices") is None
    assert ab._host_or_read(fb, "req_pool_indices", 2).tolist() == [3, 6]


def test_host_slots_keep_the_upload_record():
    fb = FB(
        spec_info=None,
        req_pool_indices=torch.tensor([4, 1], dtype=torch.int64),
        req_pool_indices_cpu=torch.tensor([8, 8], dtype=torch.int64),
    )
    _policy()._upload_host_inputs(fb)
    assert fb.req_pool_indices.device.type == "meta"
    assert ab._host_or_read(fb, "req_pool_indices", 2).tolist() == [4, 1]


def test_spec_batches_are_left_alone():
    fb = _extend([3, 5], torch.tensor([3, 5], dtype=torch.int64))
    fb.spec_info = object()
    _policy()._upload_host_inputs(fb)
    assert host_input_mirror.host_copy(fb, "req_pool_indices") is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
