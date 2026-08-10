import sys

import pytest
import torch

from sglang.srt.hardware_backend.tpu.spec_op_warmup import (
    SpecOpWarmupContext,
    make_draft_extend_select_args,
    make_verify_bookkeeping_args,
    make_verify_tree_greedy_args,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_warmup_inputs_match_physical_helper_shapes():
    context = SpecOpWarmupContext(
        device=torch.device("cpu"),
        physical_bs=4,
        num_verify_tokens=3,
        vocab_size=8,
        draft_vocab_size=5,
        hidden_size=6,
        hidden_dtype=torch.bfloat16,
    )

    verify_logits, candidates, retrieve_index = make_verify_tree_greedy_args(context)
    assert verify_logits.shape == (12, 8)
    assert verify_logits.dtype == torch.float32
    assert candidates.shape == retrieve_index.shape == (4, 3)

    seq_lens, accept_lens, predict, accept_index = make_verify_bookkeeping_args(context)
    assert seq_lens.shape == accept_lens.shape == (4,)
    assert predict.shape == (12,)
    assert accept_index.shape == (4, 3)

    draft_logits, hidden_states, select_index = make_draft_extend_select_args(context)
    assert draft_logits.shape == (12, 5)
    assert draft_logits.dtype == torch.float32
    assert hidden_states.shape == (12, 6)
    assert hidden_states.dtype == torch.bfloat16
    assert select_index.tolist() == [2, 5, 8, 11]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
