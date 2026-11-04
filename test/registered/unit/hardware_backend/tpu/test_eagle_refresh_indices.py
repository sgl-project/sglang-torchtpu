import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.speculative.eagle_utils_tpu import (
    fused_draft_extend_select_argmax_func,
    last_accepted_row_indices,
)
from sglang_tpu.speculative.spec_op_warmup import (
    SpecOpWarmupContext,
    make_draft_extend_select_args,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _select_argmax(*args):
    # Bypass the compiled_op wrapper so the body runs unchanged off-TPU.
    return fused_draft_extend_select_argmax_func.__wrapped__(*args)


@pytest.mark.parametrize("width", [1, 4, 8])
def test_row_indices_land_on_last_accepted_rows(width):
    # Includes the zero count the compile policy pads with, a partial accept,
    # and a request that filled the whole window.
    counts = torch.tensor([0, 1, width, max(1, width - 1)], dtype=torch.int32)
    rows = last_accepted_row_indices(counts, width)
    # Hand-derived: request i starts at i * width; its last accepted row is
    # count - 1, except the zero-count padding row, which clamps to row 0.
    expected = [
        0 * width + 0,
        1 * width + 0,
        2 * width + (width - 1),
        3 * width + (max(1, width - 1) - 1),
    ]
    assert rows.tolist() == expected
    assert rows.dtype == torch.int64
    # Every index stays inside the request's own window.
    starts = torch.arange(len(counts), dtype=torch.int64) * width
    assert torch.all((rows >= starts) & (rows < starts + width))


def test_row_indices_reject_unset_width():
    # EagleDraftExtendInput defaults num_tokens_per_req to -1; a sentinel width
    # must trip instead of producing negative, wrapping rows.
    counts = torch.tensor([1, 2], dtype=torch.int32)
    for width in (-1, 0):
        with pytest.raises(AssertionError):
            last_accepted_row_indices(counts, width)


@pytest.mark.parametrize("width", [1, 4, 8])
@pytest.mark.parametrize("with_hidden", [False, True])
def test_select_argmax_uses_last_accepted_rows(width, with_hidden):
    counts = torch.tensor([0, 1, width, max(1, width - 1)], dtype=torch.int32)
    logits = torch.arange(4 * width * 11, dtype=torch.float32).reshape(4 * width, 11)
    # Different winning columns, including a tie resolved to the first index.
    logits[1::2] *= -1
    hidden = (
        torch.arange(4 * width * 7, dtype=torch.float32).reshape(4 * width, 7)
        if with_hidden
        else None
    )
    expected_rows = last_accepted_row_indices(counts, width)

    selected_logits, selected_hidden, topk_p, topk_index = _select_argmax(
        logits, hidden, counts
    )

    assert torch.equal(selected_logits, logits[expected_rows])
    if with_hidden:
        assert torch.equal(selected_hidden, hidden[expected_rows])
    else:
        assert selected_hidden is None
    assert torch.equal(topk_index.flatten(), logits[expected_rows].argmax(-1))
    assert torch.equal(topk_p, torch.ones_like(topk_index, dtype=torch.float32))


def test_warmup_args_match_execution_signature():
    # The warmup recipe must build exactly what the worker passes at execution:
    # acceptance counts as int32, with the window width left implicit in the
    # logits row count. A mismatch compiles a graph serving never replays.
    context = SpecOpWarmupContext(
        device=torch.device("cpu"),
        physical_bs=3,
        num_verify_tokens=4,
        vocab_size=11,
        draft_vocab_size=11,
        hidden_size=7,
        hidden_dtype=torch.float32,
    )
    logits, hidden, num_accept_tokens = make_draft_extend_select_args(context)

    assert num_accept_tokens.dtype == torch.int32
    assert num_accept_tokens.shape == (3,)
    assert logits.shape[0] == hidden.shape[0] == 3 * 4
    # Fully-accepted warmup counts land on each request's last window row.
    assert _select_argmax(logits, hidden, num_accept_tokens)[0].shape == (3, 11)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
