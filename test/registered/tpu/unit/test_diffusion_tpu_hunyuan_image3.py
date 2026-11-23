# SPDX-License-Identifier: Apache-2.0
"""HunyuanImage-3 on TPU: tensor parallelism opt-in, and the srt fused MoE inside
the diffusion runtime (sglang_tpu.multimodal_gen.hunyuan_image3)."""

import os

# As the diffusion launcher: JAX on the CPU, torch_tpu owns the chips.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.multimodal_gen.configs.pipeline_configs.base import PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan_image3 import (
    HunyuanImage3PipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.utils import set_default_torch_dtype
from sglang.multimodal_gen.runtime.models.dits import hunyuan_image3 as dit
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import hunyuan_image3, worker
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

register_tpu_ci(est_time=120, suite="nightly-tpu-4-v7", nightly=True)

HIDDEN, EXPERTS, TOP_K, MOE_WIDTH = 256, 8, 2, 128


def _has_tpu():
    try:
        return torch.tpu.device_count() > 0
    except Exception:
        return False


class TestTensorParallelOptIn(CustomTestCase):
    def validate(self, args):
        server_args_module.ServerArgs._validate_tpu_tensor_parallel(args)

    def _args(self, config, tp_size):
        return SimpleNamespace(pipeline_config=config, tp_size=tp_size)

    def test_tp_needs_the_pipeline_opt_in_on_tpu(self):
        with patch.object(server_args_module, "current_platform", TpuOOTPlatform()):
            with self.assertRaisesRegex(
                ValueError, "no tensor-parallel path for PipelineConfig"
            ):
                self.validate(self._args(PipelineConfig(), 2))
            self.validate(self._args(HunyuanImage3PipelineConfig(), 2))
            self.validate(self._args(HunyuanImage3PipelineConfig(), 8))
            self.validate(self._args(PipelineConfig(), 1))
            self.validate(self._args(PipelineConfig(), None))

    def test_other_platforms_are_unaffected(self):
        cpu = SimpleNamespace(is_tpu=lambda: False)
        with patch.object(server_args_module, "current_platform", cpu):
            self.validate(self._args(PipelineConfig(), 2))

    def test_distributed_layout_no_longer_rejects_tp(self):
        args = SimpleNamespace(num_gpus=1, tp_size=8, is_arg_explicitly_set=bool)
        with patch.dict(os.environ, {"TPU_WORKER_HOSTNAMES": ""}):
            server_args_module.ServerArgs._adjust_tpu_distributed(args)


def _init_parallel_state():
    """Distributed init without the TPU worker's process-wide runtime patches,
    plus the dummy srt configuration the diffusion GPU worker publishes."""
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        with patch.object(worker, "install_runtime_patches"):
            maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)
    from sglang.srt.runtime_context import get_context, publish
    from sglang.srt.server_args import ServerArgs as SrtServerArgs

    if get_context()._server_args is None:
        publish(SrtServerArgs(model_path="dummy", tp_size=1), role="test")


def _moe_config():
    return PretrainedConfig(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        moe_topk=[TOP_K],
        moe_intermediate_size=[MOE_WIDTH],
        intermediate_size=MOE_WIDTH,
        use_mixed_mlp_moe=1,
        num_shared_expert=[1],
        hidden_act="silu",
    )


def _weights(seed=0):
    gen = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=gen) * shape[-1] ** -0.5

    return dict(
        router=rand(EXPERTS, HIDDEN),
        gate=rand(EXPERTS, MOE_WIDTH, HIDDEN),
        up=rand(EXPERTS, MOE_WIDTH, HIDDEN),
        down=rand(EXPERTS, HIDDEN, MOE_WIDTH),
        shared_gate=rand(MOE_WIDTH, HIDDEN),
        shared_up=rand(MOE_WIDTH, HIDDEN),
        shared_down=rand(HIDDEN, MOE_WIDTH),
    )


def _reference(x, w):
    """Softmax top-k routing (renormalized) over SwiGLU experts plus the shared
    expert, in float32."""
    probs = (x @ w["router"].T).softmax(-1)
    weights, ids = probs.topk(TOP_K, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    out = F.silu(x @ w["shared_gate"].T) * (x @ w["shared_up"].T) @ w["shared_down"].T
    for slot in range(TOP_K):
        e = ids[:, slot]
        gate = torch.einsum("th,tih->ti", x, w["gate"][e])
        up = torch.einsum("th,tih->ti", x, w["up"][e])
        expert = torch.einsum("ti,thi->th", F.silu(gate) * up, w["down"][e])
        out = out + weights[:, slot, None] * expert
    return out


class _Holder(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block


def _load(block, w):
    """Load through the checkpoint loaders (FusedMoE, merged/row linears)."""
    with torch.no_grad():
        block.gate.weight.copy_(w["router"])
    experts = block.experts
    for e in range(EXPERTS):
        for shard, name, tensor in (
            ("w1", "w13_weight", w["gate"][e]),
            ("w3", "w13_weight", w["up"][e]),
            ("w2", "w2_weight", w["down"][e]),
        ):
            param = getattr(experts, name)
            experts.weight_loader(
                param, tensor, f"experts.{name}", shard_id=shard, expert_id=e
            )
    gate_up = block.shared_mlp.gate_up_proj.weight
    gate_up.weight_loader(gate_up, w["shared_gate"], 0)
    gate_up.weight_loader(gate_up, w["shared_up"], 1)
    down = block.shared_mlp.down_proj.weight
    down.weight_loader(down, w["shared_down"])


def _block(device, dtype, w):
    """A loaded sparse MoE block, built as the pipeline builds the backbone."""
    with set_default_torch_dtype(dtype), torch.device(device):
        block = dit.HunYuanSparseMoeBlock(_moe_config(), layer_id=0)
    _load(block, w)
    dit.HunyuanImage3ForCausalMM.post_load_weights(_Holder(block))
    return block


class TestFinalLayerInput(CustomTestCase):
    def test_group_norms_get_contiguous_inputs(self):
        # torch_tpu's group_norm views its input; the token-to-image rearrange
        # yields a strided tensor it cannot view.
        final = dit.UNetUp(
            patch_size=1,
            in_channels=64,
            emb_channels=64,
            hidden_channels=64,
            out_channels=32,
            out_norm=True,
        )
        seen = []
        for module in final.modules():
            if isinstance(module, nn.GroupNorm):
                module.register_forward_pre_hook(
                    lambda _, args: seen.append(args[0].is_contiguous())
                )
        out = final(torch.randn(2, 16, 64), torch.randn(2, 64), 4, 4)
        self.assertEqual(out.shape, (2, 32, 4, 4))
        self.assertTrue(seen and all(seen), seen)


class _FakeVae(nn.Module):
    """A posterior whose sample needs random draws, as the real VAE's."""

    shift_factor = None
    scaling_factor = 0.5

    def encode(self, x):
        from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

        mean = x.mean(dim=1, keepdim=True).repeat(1, 4, 1, 1, 1)
        params = torch.cat([mean, torch.full_like(mean, -2.0)], dim=1)
        return SimpleNamespace(latent_dist=DiagonalGaussianDistribution(params))


def _edit_stage():
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.ar_stage import (
        HunyuanImage3AR,
    )

    stage = object.__new__(HunyuanImage3AR)
    stage._vae = _FakeVae()
    stage.ar_model = SimpleNamespace(
        time_embed=lambda t: t[:, None].float(),
        patch_embed=lambda latents, t_emb: (latents.flatten(1)[:, None], 1, 1),
    )
    stage._vision_model = lambda pixels, attention_mask, spatial_shapes: pixels
    stage._vision_aligner = lambda x: x
    return stage


def _edit_request_infos():
    image = torch.linspace(-1, 1, 3 * 8 * 8).reshape(3, 8, 8)
    info = SimpleNamespace(
        vae_image_info=SimpleNamespace(image_tensor=image),
        vision_image_info=SimpleNamespace(image_tensor=torch.ones(4, 6)),
        vision_encoder_kwargs={
            "spatial_shapes": torch.tensor([2, 2]),
            "pixel_attention_mask": torch.ones(4, dtype=torch.bool),
        },
    )
    return [[info]]


class TestEditSampling(CustomTestCase):
    def _edit(self, stage, seed):
        request = SimpleNamespace(seed=seed)
        generators = stage._request_generators([request], torch.device("cpu"))
        vae_embeds, _, _ = stage._encode_conditions(
            _edit_request_infos(), torch.device("cpu"), generators
        )
        noise = stage._prepare_noise(
            [request], 4, 2, 2, torch.device("cpu"), generators
        )
        return vae_embeds[0][0], noise

    def test_identical_edit_requests_sample_identical_latents(self):
        stage = _edit_stage()
        first = self._edit(stage, seed=42)
        torch.manual_seed(123)  # the global generator plays no part
        second = self._edit(stage, seed=42)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(first[0], self._edit(stage, seed=7)[0]))

    def test_condition_latents_then_noise_from_one_generator(self):
        # The reference implementation's order: the request's generator samples
        # the condition latents, then the starting noise continues from it.
        stage = _edit_stage()
        _, noise = self._edit(stage, seed=42)
        generator = torch.Generator().manual_seed(42)
        torch.randn(1, 4, 1, 8, 8, generator=generator, dtype=torch.float16)
        expected = torch.randn(1, 4, 2, 2, generator=generator, dtype=torch.bfloat16)
        self.assertTrue(torch.equal(noise, expected))


def _tokenized(prompt_rows, image_side=4, cond_side=2):
    """A CFG batch as the tokenizer emits it: one condition image after the
    system prompt, the prompt (dropped in the uncond row, which is right-padded),
    the timestep row, the generated image and an end row."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.tokenizer import (
        TokenizerEncodeOutput,
    )

    rows, cond, image = [], slice(8, 8 + cond_side * cond_side), image_side**2
    for text in (prompt_rows, 3):
        start = cond.stop + text
        rows.append((start, slice(start + 1, start + 1 + image), start + 2 + image))
    length = rows[0][2]
    tokens = torch.stack([torch.arange(length) + 10 * b for b in range(2)])
    gen_mask = torch.zeros(2, length, dtype=torch.bool)
    for b, (_, gen, _) in enumerate(rows):
        gen_mask[b, gen] = True
    return TokenizerEncodeOutput(
        tokens=tokens,
        gen_image_mask=gen_mask,
        gen_image_slices=[[gen] for _, gen, _ in rows],
        joint_image_slices=[[cond]] * 2,
        cond_vae_image_slices=[[cond]] * 2,
        cond_vit_image_slices=[[]] * 2,
        gen_timestep_scatter_index=torch.tensor([[t] for t, _, _ in rows]),
        cond_timestep_scatter_index=torch.tensor([[cond.start - 1]] * 2),
        real_pos=torch.tensor([[end] for _, _, end in rows]),
    )


def _bucket_stage():
    from sglang.multimodal_gen.runtime.models.dits.hunyuan_image3_utils import (
        CachedRoPE,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.ar_stage import (
        HunyuanImage3AR,
    )

    stage = object.__new__(HunyuanImage3AR)
    stage._custom_tokenizer = SimpleNamespace(
        special_token_map={"<pad>": 0}, pad_token_id=0
    )
    stage.ar_model = SimpleNamespace(cached_rope=CachedRoPE(10000.0, 128))
    return stage


class TestSequenceBuckets(CustomTestCase):
    info = SimpleNamespace(token_height=4, token_width=4)

    def _bucketed(self, prompt_rows):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
            ar_stage,
        )

        stage, out = _bucket_stage(), _tokenized(prompt_rows)
        with _installed(), patch.object(hunyuan_image3, "_DEVICE_TYPES", ("cpu",)):
            tok = stage._parse_tokenizer_output(out, torch.device("cpu"))
            mask, cos, sin, shared = stage._build_attention_and_rope(
                out, None, 2, tok["seq_len"], 4, 4, self.info, torch.device("cpu"), True
            )
        prefix = ar_stage._step_invariant_prefix_len(
            tok["image_mask"], tok["timestep_index"], out
        )
        return out, tok, (mask, cos, sin, shared), prefix

    def test_prompts_of_one_bucket_share_every_shape(self):
        a, b = self._bucketed(20), self._bucketed(150)
        for x, y in ((a, b),):
            self.assertEqual(x[1]["input_ids"].shape, y[1]["input_ids"].shape)
            self.assertEqual(x[3], hunyuan_image3.PREFIX_ALIGNMENT)
            self.assertEqual(x[3], y[3])
            self.assertTrue(torch.equal(x[1]["image_mask"], y[1]["image_mask"]))
            self.assertTrue(torch.equal(x[1]["timestep_index"], y[1]["timestep_index"]))
            for t, u in zip(x[2][:3], y[2][:3]):
                self.assertEqual(t.shape, u.shape)
        self.assertFalse(a[2][3])  # the uncond row has its own pad rows

    def test_real_rows_keep_the_stock_mask_rope_and_tokens(self):
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3 import (
            ar_stage,
        )

        stock, (out, tok, (mask, cos, sin, _), _) = _tokenized(150), self._bucketed(150)
        orig, rows = out._tpu_bucket
        stock_mask = ar_stage._build_causal_attention_mask(
            2,
            stock.tokens.shape[1],
            [j + g for j, g in zip(stock.joint_image_slices, stock.gen_image_slices)],
            "cpu",
        )[:, 0]
        info = ar_stage._build_rope_image_info(stock, 2, 4, 4, self.info, None)
        stock_cos, stock_sin = _bucket_stage().ar_model.cached_rope(
            stock.tokens.shape[1], "cpu", info
        )
        length = stock.tokens.shape[1]
        for b in range(2):
            real = (rows[b] < length).nonzero().flatten()
            src = rows[b, real]
            self.assertTrue(
                torch.equal(tok["input_ids"][b, real], stock.tokens[b, src])
            )
            self.assertTrue(
                torch.equal(
                    mask[b, 0][real[:, None], real[None, :]],
                    stock_mask[b][src[:, None], src[None, :]],
                )
            )
            self.assertTrue(torch.equal(cos[b, real], stock_cos[b, src]))
            self.assertTrue(torch.equal(sin[b, real], stock_sin[b, src]))
            pads = (rows[b] == length).nonzero().flatten()
            self.assertTrue(
                torch.equal(
                    mask[b, 0][pads], torch.eye(rows.shape[1], dtype=torch.bool)[pads]
                )
            )
            self.assertFalse(mask[b, 0][:, pads][real].any())


class TestPostLoad(CustomTestCase):
    def test_post_load_runs_every_fused_moe_quant_method(self):
        class FakeMoE(nn.Module):
            def __init__(self):
                super().__init__()
                self.quant_method = SimpleNamespace(
                    process_weights_after_loading=lambda layer: calls.append(layer)
                )

        calls = []
        holder = nn.ModuleList([FakeMoE(), nn.Linear(2, 2), FakeMoE()])
        with patch.object(dit, "FusedMoE", FakeMoE):
            dit.HunyuanImage3ForCausalMM.post_load_weights(holder)
        self.assertEqual(calls, [holder[0], holder[2]])


class TestExpertLayout(CustomTestCase):
    def test_post_load_moves_experts_to_the_fused_moe_layout(self):
        # On a TPU build srt's unquantized MoE method keeps the experts in the
        # Pallas kernel's (experts, in, out) layout after its post-load step.
        _init_parallel_state()
        w = _weights()
        experts = _block("cpu", torch.float32, w).experts
        self.assertEqual(experts.w13_weight.shape, (EXPERTS, HIDDEN, 2 * MOE_WIDTH))
        self.assertEqual(experts.w2_weight.shape, (EXPERTS, MOE_WIDTH, HIDDEN))
        for e in range(EXPERTS):
            gate_up = torch.cat([w["gate"][e], w["up"][e]]).T
            torch.testing.assert_close(experts.w13_weight[e], gate_up)
            torch.testing.assert_close(experts.w2_weight[e], w["down"][e].T)


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    from sglang.multimodal_gen.runtime.models.vaes import (
        autoencoder_kl_conv3d_hunyuan_image3 as vae,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan_image3.ar_stage import (
        HunyuanImage3AR,
    )

    with contextlib.ExitStack() as stack:
        for owner, name in (
            (dit.HunYuanSparseMoeBlock, "forward"),
            (HunyuanImage3AR, "_parse_tokenizer_output"),
            (HunyuanImage3AR, "_build_attention_and_rope"),
            (vae.ResnetBlock, "forward"),
            (vae.AttnBlock, "forward"),
            (vae.UpsampleDCAE, "forward"),
            (vae.DownsampleDCAE, "forward"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        hunyuan_image3.install()
        hunyuan_image3.install()  # idempotent
        yield


class TestExportMetadata(CustomTestCase):
    def test_moe_forward_runs_inside_the_tpu_metadata_context(self):
        from sglang_tpu import pallas_export

        events = []

        @contextlib.contextmanager
        def metadata():
            events.append("enter")
            yield
            events.append("exit")

        def stock(self, hidden_states):
            events.append("forward")
            return hidden_states

        x = torch.zeros(2, 4)
        with (
            patch.object(dit.HunYuanSparseMoeBlock, "forward", stock),
            patch.object(pallas_export, "tpu_info_for_cpu_export", metadata),
            _installed(),
        ):
            forward = dit.HunYuanSparseMoeBlock.forward
            self.assertIs(forward(None, x), x)  # CPU tensors: stock, no context
            self.assertEqual(events, ["forward"])
            with patch.object(hunyuan_image3, "_DEVICE_TYPES", ("cpu",)):
                forward(None, x)
        self.assertEqual(events, ["forward", "enter", "forward", "exit"])


class TestVaeBlockSync(CustomTestCase):
    def test_vae_waits_for_the_device_after_each_block_and_keeps_its_output(self):
        from sglang.multimodal_gen.runtime.models.vaes import (
            autoencoder_kl_conv3d_hunyuan_image3 as vae,
        )

        torch.manual_seed(0)
        model = vae.AutoencoderKLConv3D(
            in_channels=3,
            out_channels=3,
            latent_channels=32,
            block_out_channels=[32, 32, 64, 64, 64],
            layers_per_block=1,
            ffactor_spatial=16,
            ffactor_temporal=4,
            sample_size=64,
            sample_tsize=8,
        ).eval()
        z = torch.randn(1, 32, 1, 2, 2)
        image = torch.rand(1, 3, 1, 32, 32) * 2 - 1
        with torch.no_grad():
            expected = model.decode(z).sample, model.encode(image).latent_dist.mode()
        syncs = []
        with (
            _installed(),
            patch.object(hunyuan_image3, "_synchronize", syncs.append),
            torch.no_grad(),
        ):
            stock_calls = model.decode(z).sample
            self.assertEqual(syncs, [])  # CPU tensors: no device waits
            with patch.object(hunyuan_image3, "_DEVICE_TYPES", ("cpu",)):
                actual = model.decode(z).sample, model.encode(image).latent_dist.mode()
        blocks = sum(
            isinstance(
                m,
                (vae.ResnetBlock, vae.AttnBlock, vae.UpsampleDCAE, vae.DownsampleDCAE),
            )
            for m in model.modules()
        )
        self.assertEqual(len(syncs), blocks)
        self.assertTrue(torch.equal(stock_calls, expected[0]))
        for a, e in zip(actual, expected):
            self.assertTrue(torch.equal(a, e))


@unittest.skipUnless(_has_tpu(), "requires a TPU")
class TestFusedMoEOnTpu(CustomTestCase):
    def test_sparse_moe_block_matches_float32_math(self):
        _init_parallel_state()
        w = _weights()
        x = torch.randn(300, HIDDEN, generator=torch.Generator().manual_seed(1))
        expected = _reference(x, w)
        block = _block("tpu", torch.bfloat16, w)
        with _installed(), torch.no_grad():
            actual = block(x.to("tpu", torch.bfloat16)).float().cpu()
        self.assertTrue(torch.isfinite(actual).all())
        error = (actual - expected).norm() / expected.norm()
        self.assertLess(error.item(), 3e-2)


if __name__ == "__main__":
    unittest.main()
