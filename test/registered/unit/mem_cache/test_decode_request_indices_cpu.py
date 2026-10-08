"""Host-owned request indices must not materialize TPU control-flow graphs."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import schedule_batch  # noqa: E402
from sglang.srt.mem_cache import allocation  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _UnmaterializedDeviceIndices:
    def cpu(self):
        raise AssertionError("request indices must use the existing CPU mirror")

    def __getitem__(self, index):
        raise AssertionError("request filtering must happen on the CPU mirror")


def _request(length):
    return SimpleNamespace(
        kv=SimpleNamespace(kv_allocated_len=length, kv_committed_len=length),
        return_logprob=False,
        grammar=None,
        return_hidden_states=False,
        return_hidden_states_mode=CaptureHiddenMode.NULL,
    )


def _batch(indices):
    seq_lens = torch.arange(2, 2 + len(indices), dtype=torch.int64)
    return schedule_batch.ScheduleBatch(
        reqs=[_request(int(length)) for length in seq_lens],
        device="cpu",
        model_config=SimpleNamespace(is_encoder_decoder=False),
        req_pool_indices=_UnmaterializedDeviceIndices(),
        req_pool_indices_cpu=torch.tensor(indices),
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.clone(),
        orig_seq_lens=seq_lens.clone(),
        sampling_info=Mock(),
        return_hidden_states_mode=CaptureHiddenMode.NULL,
    )


@pytest.mark.parametrize("indices", [[4], [3, 1], [4, 0, 2], [1, 4, 0, 3]])
def test_host_decode_allocation_uses_request_index_mirror(monkeypatch, indices):
    monkeypatch.setattr(allocation, "_is_tpu", True)
    monkeypatch.setattr(allocation, "_is_npu", False)
    monkeypatch.setattr(allocation, "_alloc_page_size", lambda batch: 16)
    table = torch.arange(5 * 16, dtype=torch.int32).reshape(5, 16)
    batch = _batch(indices)
    batch.req_to_token_pool = SimpleNamespace(
        req_to_token=table,
        write=Mock(side_effect=lambda index, values: table.__setitem__(index, values)),
        alloc_aux_to_lengths=Mock(),
    )
    batch.tree_cache = Mock()
    monkeypatch.setattr(batch, "maybe_evict_swa", Mock())
    expected_last = table[batch.req_pool_indices_cpu, batch.seq_lens_cpu - 1].clone()
    output = torch.arange(101, 101 + len(indices), dtype=torch.int64)
    allocate = Mock(return_value=output)
    monkeypatch.setattr(allocation, "alloc_paged_token_slots_decode", allocate)

    result = allocation.alloc_for_decode(batch, token_per_req=1)

    assert result is output
    assert torch.equal(allocate.call_args.kwargs["last_loc"], expected_last)
    assert torch.equal(
        table[batch.req_pool_indices_cpu, batch.seq_lens_cpu], output.to(torch.int32)
    )
    assert [req.kv.kv_committed_len for req in batch.reqs] == (
        batch.seq_lens_cpu + 1
    ).tolist()
    batch.req_to_token_pool.alloc_aux_to_lengths.assert_called_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
