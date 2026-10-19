"""Which DFLASH draft linears get the two-shot all-reduce, and which calls keep the
linear's own reduction."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import sys

import pytest
import torch

from sglang.srt.layers.linear import RowParallelLinear
from sglang_tpu import twoshot_all_reduce


class _Row(RowParallelLinear):
    """A RowParallelLinear shell: the partial output is ``2 * x``; its own all-reduce adds 2."""

    def __init__(
        self,
        tp_size=8,
        bias=None,
        reduce_results=True,
        dp_attention=False,
        output_size=8192,
    ):
        torch.nn.Module.__init__(self)
        self.tp_size = tp_size
        self.bias = bias
        self.reduce_results = reduce_results
        self.use_dp_attention_reduce = dp_attention
        self.use_decode_attn_tp = False
        self.output_size = output_size

    def forward(self, x, skip_all_reduce=False, **kwargs):
        return 2 * x + (0 if skip_all_reduce else 2), None


def _fake_twoshot(monkeypatch):
    calls = []
    monkeypatch.setattr(
        twoshot_all_reduce,
        "get_twoshot_all_reduce_op",
        lambda world_size: lambda x: calls.append(world_size) or x + 1,
    )
    return calls


def test_install_wraps_only_reducing_bias_free_tp_row_linears(monkeypatch):
    _fake_twoshot(monkeypatch)
    model = torch.nn.Module()
    model.o_proj = _Row()
    model.down_proj = _Row(bias=torch.nn.Parameter(torch.zeros(4)))
    model.gate_proj = _Row()
    model.inner = torch.nn.Module()
    model.inner.down_proj = _Row(reduce_results=False)
    model.inner.o_proj = _Row(dp_attention=True)
    model.tp1 = torch.nn.Module()
    model.tp1.o_proj = _Row(tp_size=1)
    # A draft TP group smaller than the world (DP attention): the kernel's mesh
    # spans the world, so building it for 4 of 8 devices fails at startup.
    model.draft_tp = torch.nn.Module()
    model.draft_tp.o_proj = _Row(tp_size=4)
    model.ragged = torch.nn.Module()
    model.ragged.o_proj = _Row(output_size=8192 + 128)

    count = twoshot_all_reduce.install_twoshot_all_reduce(
        model, ("o_proj", "down_proj"), world_size=8
    )

    assert count == 1
    assert isinstance(model.o_proj, twoshot_all_reduce.TwoShotAllReduceLinear)
    for kept in (
        model.down_proj,
        model.gate_proj,
        model.inner.down_proj,
        model.inner.o_proj,
        model.tp1.o_proj,
        model.draft_tp.o_proj,
        model.ragged.o_proj,
    ):
        assert isinstance(kept, _Row)


def test_forward_goes_two_shot_for_decode_payloads_only(monkeypatch):
    calls = _fake_twoshot(monkeypatch)
    model = torch.nn.Module()
    model.o_proj = _Row()
    twoshot_all_reduce.install_twoshot_all_reduce(model, ("o_proj",), world_size=8)
    decode = torch.ones(8, 8192, dtype=torch.bfloat16)  # 128 KiB
    prefill = torch.ones(64, 8192, dtype=torch.bfloat16)  # 1 MiB

    out, _ = model.o_proj(decode)
    assert calls == [8] and torch.equal(out, 2 * decode + 1)
    out, _ = model.o_proj(prefill)
    assert torch.equal(out, 2 * prefill + 2)
    out, _ = model.o_proj(decode, skip_all_reduce=True)
    assert torch.equal(out, 2 * decode)
    assert calls == [8]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
