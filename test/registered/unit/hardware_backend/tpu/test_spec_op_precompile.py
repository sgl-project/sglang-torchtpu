import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.dflash_utils_tpu import (
    dflash_verify_bookkeeping_torch,
)
from sglang_tpu.spec_op_warmup import (
    SpecOpWarmupContext,
    make_dflash_verify_bookkeeping_args,
    make_draft_extend_select_args,
    make_verify_bookkeeping_args,
    make_verify_tree_greedy_args,
)

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

    draft_logits, hidden_states, num_accept_tokens = make_draft_extend_select_args(
        context
    )
    assert draft_logits.shape == (12, 5)
    assert draft_logits.dtype == torch.float32
    assert hidden_states.shape == (12, 6)
    assert hidden_states.dtype == torch.bfloat16
    assert num_accept_tokens.tolist() == [3, 3, 3, 3]
    assert num_accept_tokens.dtype == torch.int32

    dflash_cands, dflash_target_top1, dflash_prefix_lens = (
        make_dflash_verify_bookkeeping_args(context)
    )
    assert dflash_cands.shape == (4, 3)
    assert dflash_target_top1.shape == (4, 3)
    assert dflash_prefix_lens.shape == (4,)


def test_dflash_verify_bookkeeping_torch_correctness():
    # Candidates: [bs=2, block_size=4]
    # Row 0: bonus=10, draft=[1, 2, 3] -> targets=[1, 2, 99, 100] -> match prefix: [1, 2], accept_len=2, bonus=99
    # Row 1: bonus=20, draft=[4, 5, 6] -> targets=[4, 5, 6, 200] -> all match: [4, 5, 6], accept_len=3, bonus=200
    candidates = torch.tensor([[10, 1, 2, 3], [20, 4, 5, 6]], dtype=torch.int64)
    target_predict = torch.tensor([[1, 2, 99, 100], [4, 5, 6, 200]], dtype=torch.int64)
    prefix_lens = torch.tensor([10, 20], dtype=torch.int64)

    accept_len, commit_lens, bonus, out_tokens, new_seq_lens = (
        dflash_verify_bookkeeping_torch(candidates, target_predict, prefix_lens)
    )

    assert accept_len.tolist() == [2, 3]
    assert commit_lens.tolist() == [3, 4]
    assert bonus.tolist() == [99, 200]
    assert out_tokens.tolist() == [[1, 2, 99, 0], [4, 5, 6, 200]]
    assert new_seq_lens.tolist() == [13, 24]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
