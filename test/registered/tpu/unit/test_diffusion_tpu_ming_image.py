# SPDX-License-Identifier: Apache-2.0
"""On-TPU tests, and the test models, for sglang_tpu's Ming-Image patches."""

import contextlib
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.encoders.ming_image import (
    MingImageEncoderArchConfig,
    MingImageEncoderConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import USPAttention
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import ming_image as dit
from sglang.multimodal_gen.runtime.models.encoders import ming_image as encoder
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import ming_image, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=50, suite="stage-a-test-tpu-unit")

IMAGE_TOKEN = 500


def _gmm_reference(
    *, hidden_states, w1, w2, topk_weights, topk_ids, topk, activation, no_combine
):
    """The fused MoE's contract in (experts, in, out) layout, per-expert outputs."""
    outputs = []
    for slot in range(topk):
        expert = topk_ids[:, slot].long()
        gate, up = torch.einsum("th,thn->tn", hidden_states, w1[expert]).chunk(2, -1)
        outputs.append(torch.einsum("ti,tih->th", F.silu(gate) * up, w2[expert]))
    return torch.stack(outputs, 1)


def _stock_reference(hidden_states, w13, w2, topk_output, config):
    """The Triton fused_experts contract in the stock (experts, out, in) layout."""
    return _gmm_reference(
        hidden_states=hidden_states,
        w1=w13.transpose(1, 2),
        w2=w2.transpose(1, 2),
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        topk=config.top_k,
        activation="silu",
        no_combine=True,
    )


@contextlib.contextmanager
def _installed(backend):
    """The patches installed, compiling with ``backend``, restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (encoder, "fused_experts"),
            (encoder.MingImageEncoder, "forward"),
            (encoder.MingEncoderBlock, "forward"),
            (dit.MingImageTransformer2DModel, "forward"),
            (dit.MingImageTransformer2DModel, "__call__"),
            (USPAttention, "forward"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        compile_kwargs = {"backend": backend}
        stack.enter_context(
            patch.object(
                ming_image, "build_torch_compile_kwargs", lambda mode: compile_kwargs
            )
        )
        ming_image._compiled_block.cache_clear()
        stack.callback(ming_image._compiled_block.cache_clear)
        ming_image.install()
        ming_image.install()  # idempotent
        yield


def _encoder(layers=3, connector_layers=2, moe_width=32):
    gen = torch.Generator().manual_seed(0)
    llm = dict(
        hidden_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        partial_rotary_factor=0.5,
        rope_theta=600000,
        rms_norm_eps=1e-6,
        intermediate_size=64,
        moe_intermediate_size=moe_width,
        num_experts=16,
        num_experts_per_tok=4,
        n_group=4,
        topk_group=2,
        routed_scaling_factor=2.5,
        num_shared_experts=1,
        first_k_dense_replace=1,
        num_hidden_layers=layers,
        vocab_size=512,
        image_patch_token=IMAGE_TOKEN,
    )
    connector = dict(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=128,
        num_hidden_layers=connector_layers,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
    )
    vision = dict(
        depth=1,
        hidden_size=32,
        num_heads=2,
        intermediate_size=64,
        out_hidden_size=64,
        hidden_act="silu",
        patch_size=14,
        spatial_merge_size=2,
        temporal_patch_size=2,
        in_channels=3,
        window_size=112,
        fullatt_block_indexes=[0],
        tokens_per_second=2,
    )
    projection = dict(
        diffusion_c_input_dim=48,
        diffusion_inner_dim=64,
        img_gen_scales=[2],
        selected_hidden_states_layers=[1, layers],
        connector_norm=False,
    )
    config = MingImageEncoderConfig(
        arch_config=MingImageEncoderArchConfig(
            llm_config=llm,
            connector_config=connector,
            vision_config=vision,
            projection_config=projection,
        )
    )
    with patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_tpu):
        model = encoder.MingImageEncoder(config)
    for param in model.parameters():
        param.data.normal_(0, 0.05, generator=gen)
    return model.float().eval()


def _encoder_inputs(prompt_length):
    """A prompt, then <image>, the 4 learned queries and </image>, as the stage builds them."""
    ids = [(7 * i + 3) % 400 for i in range(prompt_length)]
    ids += [401] + [IMAGE_TOKEN] * 4 + [402]
    positions = encoder.ming_position_ids(ids, [(1, 2, 8)], IMAGE_TOKEN)
    return dict(input_ids=torch.tensor([ids]), position_ids=positions)


def _run(model, kwargs):
    with set_forward_context(current_timestep=0, attn_metadata=None), torch.no_grad():
        return model(**kwargs)


class _TestBase(CustomTestCase):
    def setUp(self):
        self._previous_args = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            with patch.object(worker, "install_runtime_patches"):
                maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1, sp_size=1
                )

    def tearDown(self):
        set_global_server_args(self._previous_args)


@unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires torch_tpu")
class TestMingImageOnTpu(_TestBase):
    def test_compiled_moe_encoder_runs_on_tpu(self):
        model = _encoder(layers=2, moe_width=128)
        inputs = _encoder_inputs(40)
        with patch.object(encoder, "fused_experts", _stock_reference):
            expected = _run(model, inputs).last_hidden_state
        # torch_tpu's backend, as on a TPU worker.
        with _installed("tpu"):
            model = model.bfloat16().to("tpu")
            actual = _run(model, {k: v.to("tpu") for k, v in inputs.items()})
        actual = actual.last_hidden_state.float().cpu()
        self.assertTrue(torch.isfinite(actual).all())
        error = (actual - expected).norm() / expected.norm()
        self.assertLess(error.item(), 5e-2)

    def test_vision_tower_runs_on_tpu(self):
        model = _encoder()
        vision = model.vision
        gen = torch.Generator().manual_seed(2)
        grid = torch.tensor([[1, 8, 8]])
        pixels = torch.randn(64, 3 * 2 * 14 * 14, generator=gen)
        with (
            torch.no_grad(),
            set_forward_context(current_timestep=0, attn_metadata=None),
        ):
            expected = vision(pixels, grid)
            actual = vision.to("tpu")(pixels.to("tpu"), grid).cpu()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    def test_segment_kernel_steps_match_reference(self):
        from sglang_tpu.multimodal_gen import tpu_splash_attn

        impl = tpu_splash_attn.TpuSplashAttentionImpl(
            num_heads=30, head_size=128, causal=False, softmax_scale=128**-0.5
        )
        usp = SimpleNamespace(attn_impl=impl, sp_attention_mode="ulysses")
        gen = torch.Generator().manual_seed(0)
        image, caption, real_caption = 1024, 512, 300
        q, k, v = (
            torch.randn(1, image + caption, 30, 128, generator=gen) for _ in range(3)
        )
        to_tpu = lambda t: t.bfloat16().to("tpu")  # noqa: E731
        segments = [
            (n, torch.tensor([0, real], dtype=torch.int32).to("tpu"))
            for n, real in ((image, image), (caption, real_caption))
        ]
        out = (
            ming_image._segment_attention(
                usp, *map(to_tpu, (q, k, v)), segments, local=True
            )
            .float()
            .cpu()
        )
        keys = torch.cat((k[:, :image], k[:, image : image + real_caption]), 1)
        values = torch.cat((v[:, :image], v[:, image : image + real_caption]), 1)
        rounded = lambda t: t.bfloat16().float().transpose(1, 2)  # noqa: E731
        expected = F.scaled_dot_product_attention(
            rounded(q), rounded(keys), rounded(values), scale=128**-0.5
        ).transpose(1, 2)
        torch.testing.assert_close(out, expected, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
