import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
import pytest

from sglang.srt.speculative.eagle_utils import fused_draft_extend_select_argmax_func


@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("with_hidden", [False, True])
def test_refresh_counts_include_inert_padding(width, with_hidden):
    counts = torch.tensor([0, 1, width, max(1, width - 1)], dtype=torch.int32)
    logits = torch.arange(4 * width * 11, dtype=torch.float32).reshape(4 * width, 11)
    # Different winning columns, including a tie resolved to the first index.
    logits[1::2] *= -1
    hidden = (
        torch.arange(4 * width * 7, dtype=torch.float32).reshape(4 * width, 7)
        if with_hidden
        else None
    )
    fn = fused_draft_extend_select_argmax_func.__wrapped__
    actual = fn(logits, hidden, counts, width)
    expected_rows = torch.tensor(
        [i * width + max(int(n), 1) - 1 for i, n in enumerate(counts)]
    )
    expected = fn(logits, hidden, expected_rows)
    for a, b in zip(actual, expected):
        assert (a is b is None) or torch.equal(a, b)
    assert torch.equal(actual[3].flatten(), logits[expected_rows].argmax(-1))


@pytest.mark.parametrize("enabled", [False, True])
def test_warmup_matches_refresh_selection_signature(enabled):
    from unittest.mock import patch

    from sglang.srt.environ import envs
    from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
        SpecOpWarmupContext,
        make_draft_extend_select_args,
    )

    context = SpecOpWarmupContext(
        device=torch.device("cpu"),
        physical_bs=3,
        num_verify_tokens=4,
        vocab_size=11,
        draft_vocab_size=11,
        hidden_size=7,
        hidden_dtype=torch.float32,
    )
    with patch.object(
        envs.SGLANG_TPU_EAGLE_FUSED_REFRESH_INDICES, "get", return_value=enabled
    ):
        args = make_draft_extend_select_args(context)
    if enabled:
        assert args[2].dtype == torch.int32
        assert args[2].tolist() == [4, 4, 4]
        assert args[3] == 4
    else:
        assert args[2].dtype == torch.int64
        assert args[2].tolist() == [3, 7, 11]
        assert len(args) == 3
    fused_draft_extend_select_argmax_func.__wrapped__(*args)
