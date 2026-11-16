"""Device smoke for block-FP8 conversion and offline channelwise MoE loading.

This exercises post-load transfers, TPU-owned scale buffers and compiled MoE;
the CPU loader tests cover the checkpoint names, TP shards and NextN mapping.
"""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.runtime_context import get_context, get_flags
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.fp8 import TpuFp8Config
from sglang_tpu.moe.pallas.fused_moe_op import _build_fused_moe_op, fused_moe_call

register_tpu_ci(
    est_time=20,
    suite="stage-a-test-tpu-unit",
    chips=1,
    labels=["singlehost", "glm-5.2", "smoke", "compile"],
)


class TestTpuChannelwiseFp8Loading(CustomTestCase):
    def test_block_conversion_and_offline_run_the_same_compiled_moe(self):
        _build_fused_moe_op()
        torch.manual_seed(17)
        experts, hidden, intermediate = 4, 128, 128
        weights = {}
        for stem, shape in (
            ("w13", (experts, 2 * intermediate, hidden)),
            ("w2", (experts, hidden, intermediate)),
        ):
            block_q = (torch.randn(shape) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
            # Keep dequantized weights in the original smoke's ~0.05 range.
            # The final FP32 sanity oracle does not model W8A8 activation
            # quantization; exact source/layout/output parity is checked above.
            block_scales = 2.0 ** torch.randint(
                -13, -10, (experts, shape[1] // 128, shape[2] // 128)
            )
            source = block_q.float() * block_scales.repeat_interleave(
                128, 1
            ).repeat_interleave(128, 2)
            channel_scales = (source.abs().amax(-1) / 448).clamp_min(1e-12)
            channel_q = (
                (source / channel_scales[:, :, None])
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
            weights[stem] = (block_q, block_scales, channel_q, channel_scales)

        layers = []
        with (
            get_context().override_server_args(
                moe_runner_backend="pallas_tpu", tpu_fp8_runtime="channelwise"
            ),
            get_flags().moe.override(runner_backend=MoeRunnerBackend.PALLAS_TPU),
        ):
            for blockwise in (True, False):
                metadata = {
                    "quant_method": "fp8",
                    "activation_scheme": "dynamic",
                    "fmt": "e4m3",
                    "hf_config": SimpleNamespace(
                        architectures=["GlmMoeDsaForCausalLM"]
                    ),
                }
                if blockwise:
                    metadata["weight_block_size"] = [128, 128]
                else:
                    metadata["weight_scale_granularity"] = "channel"
                config = TpuFp8Config.from_config(metadata)
                layer = FusedMoE.__new__(FusedMoE)
                torch.nn.Module.__init__(layer)
                layer.moe_runner_config = SimpleNamespace(is_gated=True)
                method = config.get_quant_method(layer, "model.layers.0.mlp.experts")
                layer.quant_method = method
                with torch.device("tpu"):
                    method.create_weights(
                        layer, experts, hidden, intermediate, torch.bfloat16
                    )
                for stem, values in weights.items():
                    weight, scale = values[:2] if blockwise else values[2:]
                    getattr(layer, f"{stem}_weight").data.copy_(weight.to("tpu"))
                    getattr(layer, f"{stem}_weight_scale_inv").data.copy_(
                        scale.to("tpu")
                    )
                method.process_weights_after_loading(layer)
                self.assertFalse(method.handles_moe_reduction)
                layers.append(layer)

        for stem in weights:
            left, right = (getattr(layer, f"{stem}_weight") for layer in layers)
            self.assertTrue(
                torch.equal(left.cpu().view(torch.uint8), right.cpu().view(torch.uint8))
            )
            torch.testing.assert_close(
                getattr(layers[0], f"{stem}_weight_scale").cpu(),
                getattr(layers[1], f"{stem}_weight_scale").cpu(),
                rtol=0,
                atol=0,
            )

        x = (torch.randn(8, hidden) * 0.5).to(torch.bfloat16)
        ids = torch.tensor(
            [[i % experts, (i + 1) % experts] for i in range(8)], dtype=torch.int32
        )
        routing = torch.full((8, 2), 0.5, dtype=torch.bfloat16)

        def run(x, w1, w2, s1, s2, routing, ids):
            return fused_moe_call(
                hidden_states=x,
                w1=w1,
                w2=w2,
                w1_scale=s1,
                w2_scale=s2,
                topk_weights=routing,
                topk_ids=ids,
                topk=2,
                activation="silu",
            )

        compiled = torch.compile(run, backend="tpu", fullgraph=True, dynamic=False)
        outputs = []
        for layer in layers:
            args = (
                x.to("tpu"),
                layer.w13_weight,
                layer.w2_weight,
                layer.w13_weight_scale,
                layer.w2_weight_scale,
                routing.to("tpu"),
                ids.to("tpu"),
            )
            eager = run(*args).cpu()
            output = compiled(*args).cpu()
            torch.testing.assert_close(output, eager, atol=2e-4, rtol=0.016)
            outputs.append(output)
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

        w1 = (
            weights["w13"][2].float()
            * weights["w13"][3].to(torch.bfloat16)[:, :, None].float()
        )
        w2 = (
            weights["w2"][2].float()
            * weights["w2"][3].to(torch.bfloat16)[:, :, None].float()
        )
        reference = torch.zeros_like(x, dtype=torch.float32)
        for token in range(8):
            for expert in ids[token].tolist():
                gate, up = F.linear(x[token].float(), w1[expert]).chunk(2)
                reference[token] += 0.5 * F.linear(F.silu(gate) * up, w2[expert])
        torch.testing.assert_close(outputs[0].float(), reference, atol=0.03, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
