"""EAGLE page reserves must skip empty allocation without skipping bookkeeping."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

import sglang.srt.speculative.eagle_utils as eagle_utils
import sglang.srt.speculative.spec_utils as spec_utils
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.mark.parametrize("committed,expected_new", [(55, 0), (56, 0), (57, 64)])
def test_page_reserve_boundary(committed, expected_new):
    reqs = [
        SimpleNamespace(
            kv=SimpleNamespace(kv_allocated_len=64, kv_committed_len=n),
            decode_batch_idx=7,
        )
        for n in (1, committed)
    ]
    batch = SimpleNamespace(
        maybe_evict_swa=Mock(),
        batch_size=lambda: 2,
        sampling_info=SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=True)
        ),
        cumulate_penalty_output_tokens=Mock(),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=64),
        reqs=reqs,
        tree_cache=object(),
        req_to_token_pool=SimpleNamespace(host_resident=True),
        enable_overlap=False,
        req_pool_indices=torch.tensor([1, 2]),
        req_pool_indices_cpu=torch.tensor([1, 2]),
        device=torch.device("cpu"),
    )
    # prepare_device_accept_tpu is imported under `if _is_tpu`, so it is absent
    # on a CPU host; create=True installs the stub the forced flag will reach.
    with (
        patch.object(eagle_utils, "_is_tpu", True),
        patch.object(
            eagle_utils, "prepare_device_accept_tpu", Mock(), create=True
        ) as prepare_device_accept,
        patch.object(eagle_utils, "get_alloc_reserve_per_decode", return_value=8),
        patch.object(
            eagle_utils,
            "get_spec",
            return_value=SimpleNamespace(speculative_eagle_topk=1),
        ),
        patch.object(spec_utils, "allocate_and_assign_decode_cache") as allocate,
    ):
        eagle_utils.eagle_prepare_for_decode(batch)
    batch.maybe_evict_swa.assert_called_once()
    batch.cumulate_penalty_output_tokens.assert_called_once()
    assert [r.decode_batch_idx for r in reqs] == [8, 8]
    assert [r.kv.kv_allocated_len for r in reqs] == [64, 64 + expected_new]
    # Skipping the allocator must not skip device-accept routing.
    prepare_device_accept.assert_called_once_with(batch, bs=2, double_alloc=8)
    if expected_new:
        assert allocate.call_args.kwargs["num_needed_tokens"] == expected_new
    else:
        allocate.assert_not_called()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
