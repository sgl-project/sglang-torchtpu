"""TPU lowering of the real-draft-token accept and the vocab-shard argmax, under direct compile."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch_tpu._internal import execution_mode

from sglang.srt.compilation import compiled_op as compiled_op_module
from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_real_draft_step,
    accept_real_draft_top_k_step,
    finalize_accept,
)
from sglang_tpu.top_k import local_argmax_pairs, merge_argmax_pairs

register_tpu_ci(
    est_time=60,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "spec", "compile-tracer"],
)

_TP = 8
_SHARD = 19_360


@compiled_op()
def _shard_argmax(logits: torch.Tensor, tp: int):
    shards = logits.chunk(tp, dim=-1)
    pairs = torch.stack([local_argmax_pairs(shard) for shard in shards])
    return merge_argmax_pairs(pairs, shards[0].shape[-1])


def _eager_reference(logits, drafts, forced, num_verify, seq_lens, live):
    """The eager ops the compiled accept replaced: argmax, clamp, gather, finalize."""
    bs, n = drafts.shape
    target = torch.argmax(logits, dim=-1).reshape(bs, n + 1)
    num_correct = torch.minimum(forced, num_verify.to(forced.dtype))
    bonus = torch.gather(target.to(drafts.dtype), 1, num_correct.reshape(bs, 1))
    return finalize_accept(drafts, num_correct, bonus.reshape(bs), seq_lens, live)


def _assert_same(actual, expected):
    for field in expected._fields:
        a, e = getattr(actual, field).cpu(), getattr(expected, field).cpu()
        assert a.dtype == e.dtype and torch.equal(a, e), field


@pytest.mark.parametrize("bs", [1, 4, 16])
def test_compiled_accept_and_shard_argmax_match_eager_argmax(bs):
    gen = torch.Generator().manual_seed(bs)
    n, vocab = 3, _TP * _SHARD
    # Four bf16 values: every row's maximum ties within and across shards.
    logits = torch.randint(0, 4, (bs * (n + 1), vocab), generator=gen)
    logits = logits.to(torch.bfloat16).float()
    logits[0] = 1.0
    drafts = torch.randint(0, vocab, (bs, n), generator=gen)
    forced = torch.full((bs,), 2, dtype=torch.int64)
    num_verify = torch.tensor([3, 2, 1, 0] * 4, dtype=torch.int32)[:bs]
    seq_lens = torch.randint(1, 10_000, (bs,), generator=gen)
    live = torch.arange(bs) < max(bs - 1, 1)
    inputs = [t.to("tpu") for t in (logits, drafts, forced, num_verify, seq_lens, live)]
    with (
        execution_mode.set_eager_mode(execution_mode.EagerMode.DEFER_NEVER),
        envs.SGLANG_ENABLE_TORCH_COMPILE.override(True),
        patch.object(
            compiled_op_module,
            "get_exec",
            return_value=SimpleNamespace(
                graph=SimpleNamespace(tpu_compile_tracer="direct")
            ),
        ),
    ):
        expected = _eager_reference(*inputs)
        _assert_same(accept_real_draft_step(*inputs), expected)
        _, ids = _shard_argmax(inputs[0], _TP)
        assert torch.equal(ids[:, 0].cpu().long(), logits.argmax(-1))
        _assert_same(accept_real_draft_top_k_step(ids, *inputs[1:]), expected)
