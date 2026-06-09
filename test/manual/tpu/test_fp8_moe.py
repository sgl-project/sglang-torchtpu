"""Manual TPU tests for FP8 fused MoE experts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

import sglang.srt.layers.quantization.fp8 as fp8_mod
from sglang.srt.layers.moe.pallas.fused_moe_op import (
    _build_fused_moe_op,
    fused_moe_call,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.test.test_utils import CustomTestCase


def _quantize_per_expert(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = weight.abs().amax(dim=(1, 2)).clamp(min=1e-12) / fp8_max
    qweight = (
        (weight / scale[:, None, None])
        .clamp(min=-fp8_max, max=fp8_max)
        .to(torch.float8_e4m3fn)
    )
    return qweight, scale


def _expand_scale(scale: torch.Tensor, out_size: int) -> torch.Tensor:
    return scale[:, None, None, None].expand(-1, 1, 1, out_size).contiguous()


def _quantize_block_rhs(
    weight: torch.Tensor, block_k: int = 128, block_n: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    num_experts, k_size, n_size = weight.shape
    assert k_size % block_k == 0
    assert n_size % block_n == 0
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    scale = torch.empty(
        num_experts, k_size // block_k, 1, n_size // block_n, dtype=torch.float32
    )
    for k_start in range(0, k_size, block_k):
        k_idx = k_start // block_k
        for n_start in range(0, n_size, block_n):
            n_end = n_start + block_n
            block = weight[:, k_start : k_start + block_k, n_start:n_end]
            block_scale = block.abs().amax(dim=(1, 2)).clamp(min=1e-12) / fp8_max
            qweight[:, k_start : k_start + block_k, n_start:n_end] = (
                (block / block_scale[:, None, None])
                .clamp(min=-fp8_max, max=fp8_max)
                .to(torch.float8_e4m3fn)
            )
            scale[:, k_idx, 0, n_start // block_n] = block_scale
    return qweight, scale


def _dequant_rhs(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    k_size = weight.shape[1]
    n_size = weight.shape[2]
    block_k = k_size // scale.shape[1]
    block_n = n_size // scale.shape[3]
    expanded_scale = scale[:, :, 0, :].repeat_interleave(block_k, dim=1)
    expanded_scale = expanded_scale.repeat_interleave(block_n, dim=2)
    return weight.float().cpu() * expanded_scale[:, :k_size, :n_size].float().cpu()


def _reference_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    x = hidden_states.float().cpu()
    w1_deq = _dequant_rhs(w1, w1_scale)
    w2_deq = _dequant_rhs(w2, w2_scale)
    weights = topk_weights.float().cpu()
    ids = topk_ids.cpu()

    output = torch.zeros_like(x)
    intermediate = w2.shape[1]
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            gate_up = x[token] @ w1_deq[expert]
            gate = gate_up[:intermediate]
            up = gate_up[intermediate:]
            activated = F.silu(gate) * up
            output[token] += weights[token, slot] * (activated @ w2_deq[expert])
    return output.to(hidden_states.dtype)


class TestTpuFp8MoE(CustomTestCase):
    def test_fp8_moe_method_routes_to_tpu_fused_moe_call(self):
        method = object.__new__(Fp8MoEMethod)
        method.moe_runner_config = SimpleNamespace(
            activation="silu",
            apply_router_weight_on_input=False,
        )
        method.block_quant = False
        hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
        topk_weights = torch.tensor([[0.75, 0.25], [1.0, 0.5]], dtype=torch.bfloat16)
        topk_ids = torch.tensor([[0, -1], [1, 0]], dtype=torch.int64)
        layer = SimpleNamespace(
            w13_weight=torch.empty(2, 8, 16, dtype=torch.float8_e4m3fn),
            w2_weight=torch.empty(2, 8, 8, dtype=torch.float8_e4m3fn),
            w13_weight_scale=torch.ones(2, 1, 1, 16, dtype=torch.float32),
            w2_weight_scale=torch.ones(2, 1, 1, 8, dtype=torch.float32),
        )
        dispatch_output = SimpleNamespace(
            hidden_states=hidden_states,
            topk_output=(topk_weights, topk_ids, None),
        )
        captured = {}

        def fake_fused_moe_call(**kwargs):
            captured.update(kwargs)
            return torch.zeros_like(kwargs["hidden_states"])

        with (
            patch.object(fp8_mod, "_is_tpu", True),
            patch(
                "sglang.srt.layers.moe.pallas.fused_moe_op.fused_moe_call",
                fake_fused_moe_call,
            ),
        ):
            output = method.apply(layer, dispatch_output)

        self.assertEqual(output.hidden_states.shape, hidden_states.shape)
        self.assertIs(captured["w1"], layer.w13_weight)
        self.assertIs(captured["w2"], layer.w2_weight)
        self.assertIs(captured["w1_scale"], layer.w13_weight_scale)
        self.assertIs(captured["w2_scale"], layer.w2_weight_scale)
        self.assertEqual(captured["topk"], 2)
        self.assertEqual(captured["activation"], "silu")
        torch.testing.assert_close(
            captured["topk_weights"],
            torch.tensor([[0.75, 0.0], [1.0, 0.5]], dtype=torch.bfloat16),
        )
        torch.testing.assert_close(
            captured["topk_ids"], torch.tensor([[0, 0], [1, 0]], dtype=torch.int32)
        )

    def test_fp8_moe_runs_eager_and_compiled(self):
        _build_fused_moe_op()

        device = torch.device("tpu")
        num_tokens = 8
        hidden_size = 128
        intermediate_size = 128
        num_experts = 4
        topk = 2

        torch.manual_seed(0)
        hidden_states = (
            torch.randn(num_tokens, hidden_size, dtype=torch.float32) * 0.5
        ).to(torch.bfloat16)
        w1_fp32 = torch.randn(num_experts, hidden_size, 2 * intermediate_size) * 0.05
        w2_fp32 = torch.randn(num_experts, intermediate_size, hidden_size) * 0.05
        w1, w1_scale = _quantize_per_expert(w1_fp32)
        w2, w2_scale = _quantize_per_expert(w2_fp32)
        w1_scale = _expand_scale(w1_scale, 2 * intermediate_size)
        w2_scale = _expand_scale(w2_scale, hidden_size)

        topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]],
            dtype=torch.int32,
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, topk, dtype=torch.float32), dim=-1
        ).to(torch.bfloat16)

        ref = _reference_moe(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
        )

        hidden_states_tpu = hidden_states.to(device)
        w1_tpu = w1.to(device)
        w2_tpu = w2.to(device)
        w1_scale_tpu = w1_scale.to(device)
        w2_scale_tpu = w2_scale.to(device)
        topk_weights_tpu = topk_weights.to(device)
        topk_ids_tpu = topk_ids.to(device)

        eager_out = fused_moe_call(
            hidden_states=hidden_states_tpu,
            w1=w1_tpu,
            w2=w2_tpu,
            topk_weights=topk_weights_tpu,
            topk_ids=topk_ids_tpu,
            topk=topk,
            activation="silu",
            w1_scale=w1_scale_tpu,
            w2_scale=w2_scale_tpu,
        )

        def run_fp8_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        ):
            return fused_moe_call(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                topk=topk,
                activation="silu",
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )

        compiled = torch.compile(
            run_fp8_moe,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        compiled_out = compiled(
            hidden_states_tpu,
            w1_tpu,
            w2_tpu,
            topk_weights_tpu,
            topk_ids_tpu,
            w1_scale_tpu,
            w2_scale_tpu,
        )

        self.assertEqual(eager_out.shape, (num_tokens, hidden_size))
        self.assertEqual(eager_out.dtype, torch.bfloat16)
        torch.testing.assert_close(eager_out.cpu(), ref, atol=0.1, rtol=0.1)
        torch.testing.assert_close(compiled_out.cpu(), eager_out.cpu())

    def test_block_fp8_moe_runs_eager_and_compiled(self):
        _build_fused_moe_op()

        device = torch.device("tpu")
        num_tokens = 8
        hidden_size = 256
        intermediate_size = 128
        num_experts = 4
        topk = 2

        torch.manual_seed(1)
        hidden_states = (
            torch.randn(num_tokens, hidden_size, dtype=torch.float32) * 0.5
        ).to(torch.bfloat16)
        w1_fp32 = torch.randn(num_experts, hidden_size, 2 * intermediate_size) * 0.05
        w2_fp32 = torch.randn(num_experts, intermediate_size, hidden_size) * 0.05
        w1, w1_scale = _quantize_block_rhs(w1_fp32)
        w2, w2_scale = _quantize_block_rhs(w2_fp32)

        topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]],
            dtype=torch.int32,
        )
        topk_weights = torch.softmax(
            torch.randn(num_tokens, topk, dtype=torch.float32), dim=-1
        ).to(torch.bfloat16)

        ref = _reference_moe(
            hidden_states,
            w1,
            w2,
            w1_scale,
            w2_scale,
            topk_weights,
            topk_ids,
        )

        def run_fp8_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            w1_scale,
            w2_scale,
        ):
            return fused_moe_call(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                topk=topk,
                activation="silu",
                w1_scale=w1_scale,
                w2_scale=w2_scale,
            )

        hidden_states_tpu = hidden_states.to(device)
        w1_tpu = w1.to(device)
        w2_tpu = w2.to(device)
        topk_weights_tpu = topk_weights.to(device)
        topk_ids_tpu = topk_ids.to(device)
        w1_scale_tpu = w1_scale.to(device)
        w2_scale_tpu = w2_scale.to(device)

        eager_out = run_fp8_moe(
            hidden_states_tpu,
            w1_tpu,
            w2_tpu,
            topk_weights_tpu,
            topk_ids_tpu,
            w1_scale_tpu,
            w2_scale_tpu,
        )

        compiled = torch.compile(
            run_fp8_moe,
            backend="tpu",
            fullgraph=True,
            dynamic=False,
        )
        compiled_out = compiled(
            hidden_states_tpu,
            w1_tpu,
            w2_tpu,
            topk_weights_tpu,
            topk_ids_tpu,
            w1_scale_tpu,
            w2_scale_tpu,
        )

        self.assertEqual(eager_out.shape, (num_tokens, hidden_size))
        self.assertEqual(eager_out.dtype, torch.bfloat16)
        torch.testing.assert_close(eager_out.cpu(), ref, atol=0.1, rtol=0.1)
        torch.testing.assert_close(compiled_out.cpu(), eager_out.cpu())


if __name__ == "__main__":
    unittest.main()
