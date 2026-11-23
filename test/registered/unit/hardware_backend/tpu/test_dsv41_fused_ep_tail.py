# SPDX-License-Identifier: Apache-2.0
"""CPU checks: the decoder tail hands fused EP its own real-row count.

DSV41TPUAttnBackend.enter_late_layer_tail switches the DP row counts to the
decoder tail (D rows per rank). Fused EP masks MoE padding rows by
forward_batch.num_token_non_padded, so with SGLANG_TPU_DSV41_FUSED_EP=1 the
tail swaps it to its valid-row count (its real rows come first) and
exit_late_layer_tail restores it. With fused EP off it is left alone.

    PYTHONPATH=<tree>/python:<tree> python -m pytest -q <this file>
"""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _backend_and_batch(valid):
    from sglang_tpu.models.deepseek_v41.attention_backend import DSV41TPUAttnBackend

    rows = len(valid)
    tail = SimpleNamespace(valid=torch.tensor(valid), selected={}, candidates=None)
    full = SimpleNamespace(
        tail=tail,
        tail_gather=torch.arange(rows, dtype=torch.int32),
        selected={},
        candidates=None,
    )
    be = SimpleNamespace(
        forward_metadata=full, decoder_low_ratios=(), _decoder_tail=None
    )
    fb = SimpleNamespace(
        num_token_non_padded=torch.tensor([1000], dtype=torch.int32),
        global_num_tokens_cpu=[1024] * 8,
        global_num_tokens_gpu=torch.full((8,), 1024),
        global_dp_buffer_len=8 * 1024,
        dp_local_start_pos=None,
        dp_local_num_tokens=None,
    )
    return DSV41TPUAttnBackend, be, fb


def _run(valid, fused_ep):
    cls, be, fb = _backend_and_batch(valid)
    published = []
    with (
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled", return_value=True
        ),
        patch(
            "sglang.srt.layers.dp_attention.get_global_dp_buffer_len", return_value=8192
        ),
        patch(
            "sglang.srt.layers.dp_attention.get_local_dp_buffer_len", return_value=1024
        ),
        patch(
            "sglang.srt.layers.dp_attention.get_dp_global_num_tokens",
            return_value=[1024] * 8,
        ),
        patch("sglang.srt.layers.dp_attention.is_dp_max_padding", return_value=True),
        patch(
            "sglang.srt.layers.dp_attention._DpGatheredBufferWrapper."
            "get_dp_global_num_tokens_gpu",
            return_value=None,
        ),
        patch(
            "sglang.srt.layers.dp_attention.set_dp_buffer_len",
            side_effect=lambda *a: published.append(a),
        ),
        patch(
            "sglang.srt.runtime_context.get_parallel",
            return_value=SimpleNamespace(attn_dp_size=8),
        ),
        patch(
            "sglang_tpu.models.deepseek_v41.fused_ep_op.is_dsv41_fused_ep_enabled",
            return_value=fused_ep,
        ),
    ):
        saved = cls.enter_late_layer_tail(be, fb)
        inside = fb.num_token_non_padded.clone()
        inside_counts = list(fb.global_num_tokens_cpu)
        cls.exit_late_layer_tail(be, saved, fb)
    return inside, inside_counts, fb


def test_tail_swaps_and_restores_real_rows_with_fused_ep():
    valid = [True] * 5 + [False] * 3
    inside, counts, fb = _run(valid, fused_ep=True)
    assert inside.tolist() == [5] and inside.dtype == torch.int32
    assert counts == [8] * 8
    assert fb.num_token_non_padded.tolist() == [1000]
    assert fb.global_num_tokens_cpu == [1024] * 8


def test_all_padding_tail_has_zero_real_rows():
    inside, _, fb = _run([False] * 8, fused_ep=True)
    assert inside.tolist() == [0]
    assert fb.num_token_non_padded.tolist() == [1000]


def test_tail_leaves_the_count_alone_without_fused_ep():
    inside, _, fb = _run([True] * 2 + [False] * 6, fused_ep=False)
    assert inside.tolist() == [1000]
    assert fb.num_token_non_padded.tolist() == [1000]
