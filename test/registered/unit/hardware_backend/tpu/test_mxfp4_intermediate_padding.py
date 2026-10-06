"""CPU validation of intermediate-only BF16 MXFP4 weight padding."""

import sys

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import pytest
import torch

from sglang.srt.layers.quantization.mxfp4_tpu_util import process_mxfp4_moe_weights_bf16


def test_intermediate_padding_preserves_weights_biases_and_hidden_layout():
    torch.random.default_generator.manual_seed(29)
    experts, hidden, intermediate = 2, 160, 2880
    w13 = torch.randint(
        0, 256, (experts, intermediate * 2, hidden // 2), dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (experts, hidden, intermediate // 2), dtype=torch.uint8)
    s13 = torch.full((experts, intermediate * 2, hidden // 32), 127, dtype=torch.uint8)
    s2 = torch.full((experts, hidden, intermediate // 32), 127, dtype=torch.uint8)
    b13 = torch.randn(experts, intermediate * 2).bfloat16()
    b2 = torch.randn(experts, hidden).bfloat16()
    for interleaved in (False, True):
        args = (w13, s13, b13, w2, s2, b2)
        old = process_mxfp4_moe_weights_bf16(
            *args, w13_interleaved=interleaved, align_to=128
        )
        new = process_mxfp4_moe_weights_bf16(
            *args, w13_interleaved=interleaved, align_to=256
        )
        assert old[0].shape == (experts, 256, 5888)
        assert new[0].shape == (experts, 256, 6144)
        assert old[2].shape == (experts, 2944, 256)
        assert new[2].shape == (experts, 3072, 256)
        for k in (0, 1):
            for half in (0, 1):
                assert torch.equal(
                    old[k][..., half * 2944 : half * 2944 + intermediate],
                    new[k][..., half * 3072 : half * 3072 + intermediate],
                )
                assert (
                    torch.count_nonzero(
                        new[k][..., half * 3072 + intermediate : (half + 1) * 3072]
                    )
                    == 0
                )
        assert torch.equal(old[2][:, :intermediate], new[2][:, :intermediate])
        assert torch.count_nonzero(new[2][:, intermediate:]) == 0
        assert torch.equal(old[3], new[3])


def test_loader_forwards_alignment_and_rejects_invalid_values():
    from types import SimpleNamespace
    from unittest.mock import patch

    import sglang.srt.layers.quantization.mxfp4_tpu_util as module

    layer = SimpleNamespace(moe_runner_config=SimpleNamespace(activation="swigluoai"))
    for name in (
        "w13_weight",
        "w13_weight_scale",
        "w13_weight_bias",
        "w2_weight",
        "w2_weight_scale",
        "w2_weight_bias",
    ):
        setattr(layer, name, torch.zeros(1))
    with (
        patch.object(
            module.envs.SGLANG_TPU_MXFP4_MOE_INTERMEDIATE_ALIGNMENT,
            "get",
            return_value=256,
        ),
        patch.object(
            module,
            "process_mxfp4_moe_weights_bf16",
            return_value=tuple(torch.zeros(1) for _ in range(4)),
        ) as process,
    ):
        module.prepare_tpu_mxfp4_moe_weights(layer)
        assert process.call_args.kwargs["align_to"] == 256
        assert process.call_args.kwargs["w13_interleaved"] is True
        assert layer.w13_weight_scale is None
        assert layer.w2_weight_scale is None
    for invalid in (0, -128, 129):
        with (
            patch.object(
                module.envs.SGLANG_TPU_MXFP4_MOE_INTERMEDIATE_ALIGNMENT,
                "get",
                return_value=invalid,
            ),
            pytest.raises(ValueError),
        ):
            module.prepare_tpu_mxfp4_moe_weights(layer)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
