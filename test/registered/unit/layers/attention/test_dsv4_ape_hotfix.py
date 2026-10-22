"""The DSV4 compressor's APE re-layout must follow its forward dispatch."""

import sys

import pytest
import torch
from torch import nn

import sglang.kernels.fused_op as fo
from sglang.kernels.fused_op import BaseFusedOp
from sglang.kernels.spec import PlatformInfo
from sglang.srt.layers.attention.dsv4.compressor import Compressor
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _overlap_compressor() -> Compressor:
    compressor = Compressor.__new__(Compressor)
    BaseFusedOp.__init__(compressor)
    compressor.ratio = 4
    compressor.overlap = True
    compressor.ape = nn.Parameter(torch.arange(16.0).view(4, 4))
    compressor.ape_converted = False
    return compressor


@pytest.mark.parametrize(
    ("platform_key", "oot_key", "relayout"),
    [
        ("cuda", None, True),
        ("hip", None, True),
        ("", None, True),
        ("tpu", None, False),
        ("npu", None, False),
        ("", "tpu", False),
        ("", "other_oot", True),
    ],
)
def test_ape_is_relaid_out_only_for_the_native_forward(
    monkeypatch, platform_key, oot_key, relayout
):
    """forward_native's kernels read the re-laid-out APE; forward_npu and
    forward_tpu (in-tree or out-of-tree) read the checkpoint layout."""
    monkeypatch.setattr(fo, "_platform_key", lambda: platform_key)
    monkeypatch.setattr(fo, "_platform", lambda: PlatformInfo())
    monkeypatch.setattr(fo, "_oot_dispatch_key", lambda: oot_key)
    compressor = _overlap_compressor()
    original = compressor.ape.data.clone()

    compressor.apply_ape_hotfix()

    halves = torch.chunk(original, 2, dim=-1)
    expected = torch.cat(halves, dim=0).view(4, -1) if relayout else original
    torch.testing.assert_close(compressor.ape.data, expected)
    assert compressor.ape_converted


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
