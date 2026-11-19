# SPDX-License-Identifier: Apache-2.0
"""Tests for SANA-Video 720p and sglang_tpu's SANA-Video RoPE (sglang_tpu.multimodal_gen.sana_video)."""

import contextlib
import re
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen import registry
from sglang.multimodal_gen.configs.models.dits.sana_video import (
    SanaVideoArchConfig,
    SanaVideoConfig,
)
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs import sana_video as config
from sglang.multimodal_gen.configs.sample.sana_video import (
    SanaVideo720pSamplingParams,
    SanaVideoSamplingParams,
)
from sglang.multimodal_gen.runtime.layers import linear
from sglang.multimodal_gen.runtime.models.dits import sana_video as dit
from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import AutoencoderKLLTX2Video
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
    scale_and_shift_latents,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.latent_preparation import (
    LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.utils.common import parse_size
from sglang.multimodal_gen.runtime.utils.precision import resolve_decode_precision
from sglang.multimodal_gen.runtime.warmup_request_builder import (
    _resolve_warmup_num_frames,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.utils import is_in_ci
from sglang_tpu.multimodal_gen import sana_video

register_tpu_ci(est_time=30, suite="nightly-tpu-4-v7", nightly=True)

HF_ID_720P = "Efficient-Large-Model/SANA-Video_2B_720p_diffusers"
# The latent layout keys of the 720p checkpoint's vae/config.json.
VAE_720P = {
    "latent_channels": 128,
    "patch_size": 4,
    "scaling_factor": 1.0,
    "spatial_compression_ratio": 32,
    "temporal_compression_ratio": 8,
    "decoder_upsample_type": None,
    "decoder_causal": False,
}
CPU = torch.device("cpu")
RECIPE = Path(__file__).parents[4] / "benchmark/tpu/diffusion/sana_video.sh"
# A one-rank tensor-parallel group: the DiT builds without a process group.
TP1 = SimpleNamespace(world_size=1, rank_in_group=0, all_gather=lambda x, dim=0: x)


@contextlib.contextmanager
def _installed(device=CPU):
    """The TPU paths on ``device``, with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (dit.SanaVideoTransformer3DModel, "post_load_weights"),
            (dit.SanaVideoRotaryPosEmbed, "forward"),
            (dit, "apply_interleaved_rotary_emb_pair"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(sana_video, "_DEVICE_TYPES", (device.type,)))
        stack.enter_context(
            patch.object(sana_video, "get_local_torch_device", lambda: device)
        )
        sana_video.install()
        sana_video.install()  # idempotent
        yield


def _split(rope):
    return (
        *sana_video.split_table(rope.freqs_cos),
        *sana_video.split_table(rope.freqs_sin),
    )


def _rope_case(seed, grid, patch_size, heads=4):
    """Stock float64 tables, the patched float32 tables and q/k on a latent grid."""
    rope = dit.SanaVideoRotaryPosEmbed(112, patch_size, 1024)
    shape = (1, 1, *(n * p for n, p in zip(grid, patch_size)))
    stock = sana_video._STOCK_ROPE_FORWARD(rope, torch.zeros(shape))
    rope.tpu_tables = _split(rope)
    tables = sana_video.rope_forward(rope, torch.zeros(shape))
    gen = torch.Generator().manual_seed(seed)
    size = (1, stock[0].shape[1], heads, 112)

    def qk():
        # Six decades of magnitude, both signs, and zeros.
        scale = 10 ** (torch.rand(size, generator=gen) * 6 - 3)
        x = (torch.randn(size, generator=gen) * scale).bfloat16()
        x[..., ::7] = 0
        return x

    return stock, tables, qk(), qk()


def _tiny_dit():
    torch.manual_seed(0)
    arch = SanaVideoArchConfig(
        patch_size=(1, 2, 2),
        in_channels=4,
        out_channels=4,
        num_layers=2,
        attention_head_dim=24,
        num_attention_heads=2,
        num_cross_attention_heads=2,
        cross_attention_head_dim=24,
        cross_attention_dim=48,
        caption_channels=32,
        mlp_ratio=2.0,
        rope_max_seq_len=32,
    )
    with patch.object(linear, "get_tp_group", lambda: TP1):
        model = dit.SanaVideoTransformer3DModel(SanaVideoConfig(arch_config=arch))
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    model = model.to(torch.bfloat16).eval()
    # The loader keeps the RoPE tables float64, as post_load_weights builds them.
    model.rope._init_freqs_buffers()
    return model


def _branch(seed, real_tokens):
    gen = torch.Generator().manual_seed(seed)
    mask = torch.zeros(1, 7, dtype=torch.long)
    mask[:, :real_tokens] = 1
    return dict(
        hidden_states=torch.randn(1, 4, 3, 8, 12, generator=gen).bfloat16(),
        encoder_hidden_states=torch.randn(1, 7, 32, generator=gen).bfloat16(),
        encoder_attention_mask=mask,
    )


class TestSanaVideoRope(CustomTestCase):
    def test_rotation_equals_the_stock_float64_rotation(self):
        # 720p, and 480p (every table row it reads, one head).
        cases = ((0, (11, 22, 40), (1, 1, 1), 4), (1, (21, 30, 52), (1, 2, 2), 1))
        for seed, grid, patch_size, heads in cases:
            with self.subTest(grid=grid, seed=seed):
                (cos, sin), tables, q, k = _rope_case(seed, grid, patch_size, heads)
                q_out, k_out = sana_video.apply_interleaved_rotary_emb_pair(
                    q, k, *tables
                )
                self.assertTrue(
                    torch.equal(q_out, dit.apply_interleaved_rotary_emb(q, cos, sin))
                )
                self.assertTrue(
                    torch.equal(k_out, dit.apply_interleaved_rotary_emb(k, cos, sin))
                )

    def test_tables_split_float64_exactly_in_stock_token_order(self):
        (cos, sin), tables, _, _ = _rope_case(0, (3, 5, 7), (1, 2, 2))
        cos_hi, cos_lo, sin_hi, sin_lo = tables
        for hi, lo, want in ((cos_hi, cos_lo, cos), (sin_hi, sin_lo, sin)):
            self.assertEqual(hi.dtype, torch.float32)
            self.assertFalse(bool((hi.view(torch.int32) & 0xFF).any()))
            err = (hi.double() + lo.double() - want).abs().max().item()
            self.assertLessEqual(err, 2**-40)
        rope = dit.SanaVideoRotaryPosEmbed(24, (1, 2, 2), 32)
        x = torch.zeros(1, 1, 3, 4, 6)
        for got, want in zip(
            sana_video.rope_forward(rope, x), sana_video._STOCK_ROPE_FORWARD(rope, x)
        ):
            self.assertTrue(torch.equal(got, want))

    def test_float32_queries_take_the_stock_rotation_of_hi_plus_lo(self):
        (cos, sin), tables, _, _ = _rope_case(0, (2, 3, 4), (1, 1, 1))
        gen = torch.Generator().manual_seed(1)
        q, k = torch.randn(2, 1, cos.shape[1], 4, 112, generator=gen).unbind(0)
        hi_lo = [tables[i].double() + tables[i + 1].double() for i in (0, 2)]
        q_out, k_out = sana_video.apply_interleaved_rotary_emb_pair(q, k, *tables)
        for got, x in ((q_out, q), (k_out, k)):
            self.assertTrue(
                torch.equal(got, dit.apply_interleaved_rotary_emb(x, *hi_lo))
            )
            want = dit.apply_interleaved_rotary_emb(x, cos, sin)
            torch.testing.assert_close(got, want, rtol=2.4e-7, atol=1e-10)

    def test_install_is_idempotent_and_restorable(self):
        with _installed():
            self.assertIs(
                dit.SanaVideoTransformer3DModel.post_load_weights,
                sana_video.post_load_weights,
            )
            self.assertIs(dit.SanaVideoRotaryPosEmbed.forward, sana_video.rope_forward)
            self.assertIs(
                dit.apply_interleaved_rotary_emb_pair,
                sana_video.apply_interleaved_rotary_emb_pair,
            )
        self.assertIs(
            dit.apply_interleaved_rotary_emb_pair, sana_video._STOCK_ROTATE_PAIR
        )


class TestSanaVideoDiT(CustomTestCase):
    def test_post_load_keeps_float32_tables_on_the_device(self):
        want = _split(dit.SanaVideoRotaryPosEmbed(24, (1, 2, 2), 32))
        # The worker's device stands in as "meta"; the values are checked on CPU.
        for device in (torch.device("meta"), CPU):
            model = _tiny_dit()
            # The loader leaves the non-persistent RoPE buffers on the meta device.
            model.rope.freqs_cos = model.rope.freqs_cos.to("meta")
            with _installed(device):
                model.post_load_weights()
            tables = model.rope.tpu_tables
            self.assertEqual(len(tables), 4)
            for table, expected in zip(tables, want, strict=True):
                self.assertEqual(table.device, device)
                self.assertEqual(table.dtype, torch.float32)
                self.assertEqual(table.shape, expected.shape)
                self.assertEqual(table.storage_offset(), 0)
                self.assertTrue(table.is_contiguous())
                if device == CPU:
                    self.assertTrue(torch.equal(table, expected))
            held = {id(t) for t in (*model.parameters(), *model.buffers())}
            self.assertFalse(any(id(t) in held for t in tables))
            model.to(torch.bfloat16)
            self.assertTrue(
                all(t.dtype == torch.float32 for t in model.rope.tpu_tables)
            )

        # A device outside _DEVICE_TYPES keeps the stock tables only.
        other = _tiny_dit()
        with patch.object(sana_video, "get_local_torch_device", lambda: CPU):
            sana_video.post_load_weights(other)
        self.assertFalse(hasattr(other.rope, "tpu_tables"))

    def test_dit_output_equals_stock_and_compiles_once(self):
        model = _tiny_dit()
        branches = (_branch(1, 5), _branch(2, 7))
        steps = (999.0, 500.0, 1.0)
        with torch.no_grad():
            want = [
                model(**branch, timestep=torch.tensor([t]))
                for t in steps
                for branch in branches
            ]
            with _installed():
                model.post_load_weights()
                graphs, inputs = [], []

                def backend(gm, example_inputs):
                    graphs.append(gm)
                    inputs.extend(example_inputs)
                    return gm.forward

                torch._dynamo.reset()
                compiled = torch.compile(model, backend=backend)
                got = [
                    compiled(**branch, timestep=torch.tensor([t]))
                    for t in steps
                    for branch in branches
                ]
                eager = model(**branches[0], timestep=torch.tensor([steps[0]]))
            torch._dynamo.reset()
        self.assertTrue(torch.equal(eager, want[0]))
        for actual, expected in zip(got, want, strict=True):
            self.assertTrue(torch.equal(actual, expected))
        # One program for every step and both CFG branches, whose inputs all
        # start at storage offset 0 and hold no float64 table.
        self.assertEqual(len(graphs), 1)
        tensors = [t for t in inputs if isinstance(t, torch.Tensor)]
        self.assertTrue(all(t.storage_offset() == 0 for t in tensors))
        self.assertFalse(any(t.dtype == torch.float64 for t in tensors))


class TestSanaVideo720pConfig(CustomTestCase):
    def _config(self):
        cfg = config.SanaVideo720pPipelineConfig()
        # What the VAE and DiT loaders read from the checkpoint.
        cfg.vae_config.update_model_arch(dict(VAE_720P))
        cfg.dit_config.update_model_arch(
            {"in_channels": 128, "out_channels": 128, "patch_size": [1, 1, 1]}
        )
        return cfg

    def test_registry_resolves_the_720p_checkpoint(self):
        cache = f"/hf/hub/models--{HF_ID_720P.replace('/', '--')}/snapshots/0a1b2c"
        cases = (
            (
                HF_ID_720P,
                config.SanaVideo720pPipelineConfig,
                SanaVideo720pSamplingParams,
            ),
            (cache, config.SanaVideo720pPipelineConfig, SanaVideo720pSamplingParams),
            (
                "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
                config.SanaVideoPipelineConfig,
                SanaVideoSamplingParams,
            ),
        )
        index = {"_class_name": "SanaVideoPipeline"}
        with patch.object(registry, "maybe_download_model_index", lambda _: index):
            for path, pipeline_config_cls, sampling_cls in cases:
                registry.get_model_info.cache_clear()
                info = registry.get_model_info(path)
                self.assertIs(info.pipeline_config_cls, pipeline_config_cls, path)
                self.assertIs(info.sampling_param_cls, sampling_cls, path)
        registry.get_model_info.cache_clear()

    def test_latents_follow_the_ltx2_vae(self):
        cfg = self._config()
        sampling = SanaVideo720pSamplingParams()
        batch = SimpleNamespace(
            height=sampling.height, width=sampling.width, num_frames=81
        )
        stage = LatentPreparationStage.__new__(LatentPreparationStage)
        frames = stage.adjust_video_length(batch, SimpleNamespace(pipeline_config=cfg))
        self.assertEqual(cfg.adjust_num_frames(81), 81)
        self.assertEqual(
            cfg.prepare_latent_shape(batch, 1, frames), (1, 128, 11, 22, 40)
        )
        # The checkpoint's VAE config builds the LTX-2 decoder.
        with torch.device("meta"):
            vae = AutoencoderKLLTX2Video(cfg.vae_config)
        self.assertEqual(tuple(vae.latents_std.shape), (128,))

    def test_fp32_decode_denormalizes_as_diffusers(self):
        cfg = self._config()  # with --vae-precision fp32
        gen = torch.Generator().manual_seed(0)
        vae = SimpleNamespace(
            latents_mean=torch.randn(128, generator=gen),
            latents_std=torch.rand(128, generator=gen) + 0.5,
        )
        latents = torch.randn(1, 128, 2, 3, 4, generator=gen)
        got = scale_and_shift_latents(
            latents, SimpleNamespace(pipeline_config=cfg), vae
        )
        # Diffusers' SanaVideoPipeline for AutoencoderKLLTX2Video.
        std = 1.0 / vae.latents_std.view(1, -1, 1, 1, 1)
        want = latents / std + vae.latents_mean.view(1, -1, 1, 1, 1)
        self.assertTrue(torch.equal(got, want))

    def test_bf16_decode_rounds_the_float32_denormalization_once(self):
        cfg = self._config()
        gen = torch.Generator().manual_seed(0)
        # A bf16-loaded VAE holds its latent statistics in bf16.
        mean = torch.randn(128, generator=gen).bfloat16()
        std = (torch.rand(128, generator=gen) + 0.5).bfloat16()
        vae = SimpleNamespace(latents_mean=mean, latents_std=std)
        latents = torch.randn(1, 128, 4, 6, 8, generator=gen)
        # DecodingStage.decode: de-normalize the float32 latents, then cast.
        got = scale_and_shift_latents(
            latents, SimpleNamespace(pipeline_config=cfg), vae
        ).bfloat16()
        mean, std = (t.float().view(1, -1, 1, 1, 1) for t in (mean, std))
        self.assertTrue(torch.equal(got, (latents / (1.0 / std) + mean).bfloat16()))
        # Diffusers casts first and de-normalizes in bf16, so its VAE input
        # differs by bf16 roundings and lies farther from the exact value.
        bf16 = latents.bfloat16() / (1.0 / std.bfloat16()) + mean.bfloat16()
        exact = latents.double() * std.double() + mean.double()
        err, bf16_err = ((t.double() - exact).abs().max() for t in (got, bf16))
        self.assertFalse(torch.equal(got, bf16))
        self.assertLess(err, bf16_err)

    def test_requests_keep_their_frames_and_warmup_runs_the_served_shape(self):
        recipe = RECIPE.read_text()
        size = re.search(r"--warmup-resolutions (\S+)", recipe).group(1)
        frames = int(re.search(r"--warmup-num-frames (\d+)", recipe).group(1))
        for cfg, sampling_cls in (
            (self._config(), SanaVideo720pSamplingParams),
            (config.SanaVideoPipelineConfig(), SanaVideoSamplingParams),
        ):
            for num_gpus in (1, 2):  # 2: CFG parallel
                args = SimpleNamespace(
                    pipeline_config=cfg,
                    num_gpus=num_gpus,
                    output_path=None,
                    comfyui_mode=True,
                    warmup_num_frames=frames,
                )
                served = sampling_cls(num_frames=81)
                served._adjust(args)
                self.assertEqual(served.num_frames, 81, (sampling_cls, num_gpus))
                if sampling_cls is SanaVideo720pSamplingParams:
                    warmup = _resolve_warmup_num_frames(
                        args, sampling_cls(), server_based_warmup=True
                    )
                    self.assertEqual(
                        (parse_size(size), warmup),
                        ((served.width, served.height), served.num_frames),
                    )

    def test_sizes_off_the_32_pixel_latent_grid_are_rejected(self):
        cfg = self._config()
        SanaVideo720pSamplingParams()._validate_with_pipeline_config(cfg)
        for width, height in ((1280, 720), (1270, 704)):
            params = SanaVideo720pSamplingParams(width=width, height=height)
            with self.assertRaisesRegex(ValueError, "multiples of 32"):
                params._validate_with_pipeline_config(cfg)

    def test_vae_runs_bf16_and_fp32_is_one_flag_away(self):
        cfg = self._config()
        args = SimpleNamespace(component_precisions={}, pipeline_config=cfg)
        self.assertEqual(cfg.vae_precision, "bf16")
        self.assertIs(resolve_decode_precision(args, "vae"), torch.bfloat16)
        cfg.vae_precision = "fp32"  # --vae-precision fp32
        self.assertIs(resolve_decode_precision(args, "vae"), torch.float32)

    def test_480p_keeps_the_wan_vae(self):
        cfg = config.SanaVideoPipelineConfig()
        self.assertIsInstance(cfg.vae_config, WanVAEConfig)
        batch = SimpleNamespace(height=480, width=832, num_frames=81)
        stage = LatentPreparationStage.__new__(LatentPreparationStage)
        frames = stage.adjust_video_length(batch, SimpleNamespace(pipeline_config=cfg))
        self.assertEqual(
            cfg.prepare_latent_shape(batch, 1, frames), (1, 16, 21, 60, 104)
        )
        self.assertEqual(cfg.vae_decode_precision, "fp32")


class TestSanaVideoRopeTpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from sglang_tpu.env_override import apply_tpu_env_overrides

            apply_tpu_env_overrides()
            import torch_tpu  # noqa: F401

            torch.zeros(1, device="tpu").cpu()
        except (ImportError, RuntimeError) as error:
            # TPU CI fails here rather than skipping.
            if is_in_ci():
                raise
            raise unittest.SkipTest(f"no TPU device: {error}")

    def test_rotation_on_device_equals_the_stock_float64_rotation(self):
        # XLA must keep the TwoSum: a rewrite of (s - a) or a fused lo term
        # shows up as mismatches.
        import torch_tpu._internal.compile  # noqa: F401

        (cos, sin), tables, q, _ = _rope_case(0, (11, 22, 40), (1, 1, 1))
        want = dit.apply_interleaved_rotary_emb(q, cos, sin)
        args = [t.to("tpu") for t in (q, *tables)]
        compiled = torch.compile(sana_video._rotate, backend="tpu", dynamic=False)
        for rotate in (sana_video._rotate, compiled):
            got = rotate(*args)
            self.assertEqual(got.device.type, "tpu")
            self.assertEqual(int((got.cpu() != want).sum()), 0)


if __name__ == "__main__":
    unittest.main()
