# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 TPU post-experts MoE sum in FP32 (tpu_dsv41_fp32_moe_sum): the BF16
partials reach the all-reduce as FP32 and come back BF16 once, only for V4.1,
and the result no longer depends on the order the ranks' partials are added."""

import threading
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.models.deepseek_v41 import hooks

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

TARGET = "sglang.srt.layers.moe.utils._post_experts_sum"
COLLECTIVES = {
    "ep": "moe_expert_parallel_all_reduce",
    "tp": "moe_tensor_model_parallel_all_reduce",
    "merged": "tensor_model_parallel_all_reduce",
}


@pytest.fixture
def moe(monkeypatch):
    """moe.utils with the V4.1 hook installed through HookRegistry, a fake
    parallel context, and all-reduces that record the dtype they receive.

    Tests set state.model_type, the group sizes, and state.reduce (the fake
    all-reduce, the same for all three collectives)."""
    from sglang.srt import runtime_context
    from sglang.srt.distributed import communication_op
    from sglang.srt.layers.moe import utils as moe_utils
    from sglang.srt.plugins.hook_registry import HookRegistry

    state = SimpleNamespace(
        model_type="deepseek_v41",
        config_reads=0,
        parallel=SimpleNamespace(moe_ep_size=8, moe_tp_size=1, moe_dp_size=1),
        reduce=lambda t: t * 3,
        seen=[],
    )

    def model_config():
        state.config_reads += 1
        return SimpleNamespace(hf_config=SimpleNamespace(model_type=state.model_type))

    monkeypatch.setattr(runtime_context, "process_model_config", model_config)
    monkeypatch.setattr(moe_utils, "get_parallel", lambda: state.parallel)
    # The skip rules read flags this test does not publish; every sum is owed.
    monkeypatch.setattr(
        moe_utils, "should_skip_post_experts_all_reduce", lambda **_: False
    )
    monkeypatch.setattr(moe_utils, "post_experts_output_is_complete", lambda **_: False)
    for name, fn in COLLECTIVES.items():

        def fake(t, name=name):
            state.seen.append((name, t.dtype))
            return state.reduce(t)

        monkeypatch.setattr(communication_op, fn, fake)

    spec = [s for s in hooks.hook_specs() if s[0] == TARGET]
    assert spec == [(TARGET, hooks.tpu_dsv41_fp32_moe_sum, spec[0][2])]
    assert spec[0][2].name == "AROUND"
    original = moe_utils._post_experts_sum
    monkeypatch.setattr(moe_utils, "_post_experts_sum", original)  # restored after
    monkeypatch.setattr(HookRegistry, "_hooks", defaultdict(list))
    monkeypatch.setattr(HookRegistry, "_patched", set())
    HookRegistry.register(*spec[0])
    HookRegistry.apply_hooks()
    assert moe_utils._post_experts_sum is not original
    state.utils = moe_utils
    return state


def _bf16(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(torch.bfloat16)


@pytest.mark.parametrize(
    "ep, tp, collective",
    [(8, 1, "ep"), (1, 8, "tp"), (4, 2, "merged")],
)
@pytest.mark.parametrize(
    "caller", ["post_experts_all_reduce", "sum_post_experts_output"]
)
def test_bf16_reaches_the_all_reduce_as_fp32(moe, ep, tp, collective, caller):
    moe.parallel.moe_ep_size, moe.parallel.moe_tp_size = ep, tp
    x = _bf16(5, 16)
    out = getattr(moe.utils, caller)(x)
    assert moe.seen == [(collective, torch.float32)]
    assert out.dtype == torch.bfloat16
    assert torch.equal(out, (x.float() * 3).to(torch.bfloat16))


def test_two_step_sum_runs_both_reductions_in_fp32(moe):
    """moe_dp_size > 1: EP and MoE-TP sums do not merge; both see FP32 and the
    result is rounded to BF16 once, after the second."""
    moe.parallel.moe_ep_size, moe.parallel.moe_tp_size = 2, 2
    moe.parallel.moe_dp_size = 2
    x = _bf16(3, 8)
    out = moe.utils.post_experts_all_reduce(x)
    assert moe.seen == [("ep", torch.float32), ("tp", torch.float32)]
    assert torch.equal(out, (x.float() * 9).to(torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_other_dtypes_pass_through(moe, dtype):
    x = _bf16(4, 8).to(dtype)
    out = moe.utils.post_experts_all_reduce(x)
    assert moe.seen == [("ep", dtype)]
    assert out.dtype == dtype and torch.equal(out, x * 3)
    assert moe.config_reads == 0


def test_no_reduction_passes_through(moe):
    moe.parallel.moe_ep_size, moe.parallel.moe_tp_size = 1, 1
    x = _bf16(4, 8)
    assert moe.utils.post_experts_all_reduce(x) is x
    assert moe.utils.sum_post_experts_output(x) is x
    assert moe.seen == [] and moe.config_reads == 0


@pytest.mark.parametrize("model_type", ["deepseek_v4", "qwen3_moe", None])
def test_other_models_keep_the_bf16_sum(moe, model_type):
    moe.model_type = model_type
    x = _bf16(4, 8)
    out = moe.utils.post_experts_all_reduce(x)
    assert moe.seen == [("ep", torch.bfloat16)]
    assert torch.equal(out, x * 3)
    assert moe.config_reads == 1


def _partials(world, tokens, hidden):
    """[world, tokens, hidden] BF16 partials.

    Column 0 is crafted: rank 0 holds 1.0, the others 2**-8 (half a BF16 ulp at
    1.0). Added to 1.0 one at a time, each rounds away (ties to even); added
    to each other first, they survive. The other columns are random with
    magnitudes in [0.5, 4), so their FP32 sum is exact in any order."""
    g = torch.Generator().manual_seed(11)
    mag = torch.rand(world, tokens, hidden, generator=g) * 3.5 + 0.5
    sign = torch.randint(0, 2, (world, tokens, hidden), generator=g) * 2 - 1
    p = (mag * sign).to(torch.bfloat16)
    p[:, :, 0] = 2.0**-8
    p[0, :, 0] = 1.0
    return p


def _run_ranks(moe, partials):
    """Each of the world threads calls post_experts_all_reduce on its partial.

    The fake EP all-reduce adds the partials in the input dtype, starting at the
    calling rank (rank 0 in rank order, the others rotated), as a ring
    all-reduce adds each chunk starting at a different rank."""
    world = partials.shape[0]
    tls = threading.local()
    barrier = threading.Barrier(world)
    slots = [None] * world
    results = [None] * world
    errors = []

    def rotated_sum(t):
        slots[tls.rank] = t
        barrier.wait()
        total = slots[tls.rank]
        for k in range(1, world):
            total = total + slots[(tls.rank + k) % world]
        barrier.wait()
        return total

    moe.reduce = rotated_sum

    def run(rank):
        try:
            tls.rank = rank
            results[rank] = moe.utils.post_experts_all_reduce(partials[rank])
        except Exception as exc:  # surface thread failures in the test
            errors.append(exc)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    return results


def test_fp32_sum_does_not_depend_on_the_summation_order(moe):
    world = 8
    partials = _partials(world, tokens=4, hidden=64)
    exact = partials.double().sum(0).to(torch.bfloat16)

    moe.model_type = "deepseek_v4"  # control: the BF16 sum
    bf16 = _run_ranks(moe, partials)
    assert all(r.dtype == torch.bfloat16 for r in bf16)
    assert not torch.equal(bf16[0], bf16[1])
    assert bf16[0][:, 0].float().unique().tolist() == [1.0]
    assert bf16[1][:, 0].float().unique().tolist() == [1.03125]

    moe.model_type = "deepseek_v41"
    moe.seen.clear()
    fp32 = _run_ranks(moe, partials)
    assert {dtype for _, dtype in moe.seen} == {torch.float32}
    for rank in range(world):
        assert fp32[rank].dtype == torch.bfloat16
        assert torch.equal(fp32[rank], exact), rank


def test_gate_is_a_trace_time_constant(moe, monkeypatch):
    """Dynamo traces post_experts_all_reduce through the hook in one graph; the
    model check runs once at trace time, outside the graph."""
    from sglang.srt.distributed import communication_op

    # A collective without the recording side effect, which would add a guard
    # on the record and recompile on the second call.
    monkeypatch.setattr(communication_op, COLLECTIVES["ep"], lambda t: moe.reduce(t))
    graphs = []

    def capture(gm, example_inputs):
        graphs.append(gm.code)
        return gm.forward

    x = _bf16(4, 8)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(
            moe.utils.post_experts_all_reduce, backend=capture, fullgraph=True
        )
        first = compiled(x)
        second = compiled(x)
    finally:
        torch._dynamo.reset()
    assert len(graphs) == 1
    assert moe.config_reads == 1
    assert ".float()" in graphs[0] and "torch.bfloat16" in graphs[0]
    want = (x.float() * 3).to(torch.bfloat16)
    assert torch.equal(first, want) and torch.equal(second, want)
