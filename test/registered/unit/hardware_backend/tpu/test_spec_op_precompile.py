import torch

from sglang.srt.compilation.compiled_op import get_compiled_ops
from sglang.srt.hardware_backend.tpu.spec_op_precompile import precompile_spec_v2_ops
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def test_fused_spec_ops_are_discovered_from_decorator_registry():
    names = (
        "_verify_tree_greedy_torch",
        "fused_verify_post_func",
        "fused_draft_extend_select_argmax_func",
    )

    assert {registered_op.op.__name__ for registered_op in get_compiled_ops()} == set(
        names
    )


def test_precompile_spec_v2_ops_covers_physical_token_shapes():
    shapes = precompile_spec_v2_ops(
        device=torch.device("cpu"),
        physical_batch_buckets=(1, 4, 16),
        num_verify_tokens=4,
        vocab_size=8,
        verify_logits_dtype=torch.float32,
        draft_logits_dtype=torch.bfloat16,
        hidden_size=4,
        hidden_dtype=torch.bfloat16,
    )

    assert shapes == ((1, 4), (4, 16), (16, 64))
