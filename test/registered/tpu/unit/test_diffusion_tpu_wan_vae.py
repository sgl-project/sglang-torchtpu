# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the TPU Wan VAE decode patches (sglang_tpu.multimodal_gen.wan_vae)."""

import contextlib
import os
import sys
import unittest
from unittest.mock import patch

# Host-CPU tests: in the TPU image the diffusion platform and JAX would
# otherwise select the TPU.
os.environ.setdefault("SGLANG_DIFFUSION_PLATFORM_OVERRIDE", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.runtime.models.vaes import wanvae
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=60,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "diffusion"],
)

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# sglang_tpu from the checkout where it is not installed.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.envs import envs  # noqa: E402
from sglang_tpu.multimodal_gen import wan_vae  # noqa: E402


def _tiny_vae():
    config = WanVAEConfig()
    arch = config.arch_config
    arch.base_dim, arch.z_dim, arch.dim_mult = 8, 4, (1, 2, 2)
    arch.num_res_blocks, arch.temperal_downsample = 1, (False, True)
    config.load_encoder = False
    torch.manual_seed(0)
    return wanvae.AutoencoderKLWan(config).eval()


def _latents(frames=6):
    return torch.randn(1, 4, frames, 4, 4, generator=torch.Generator().manual_seed(4))


class _WanVAECase(CustomTestCase):
    """Restores AutoencoderKLWan.decode and WanAttentionBlock.forward."""

    def setUp(self):
        cls = wanvae.AutoencoderKLWan
        self._decode = cls.__dict__["decode"]
        self._attention_forward = wanvae.WanAttentionBlock.__dict__["forward"]

    def tearDown(self):
        cls = wanvae.AutoencoderKLWan
        cls.decode = self._decode
        wanvae.WanAttentionBlock.forward = self._attention_forward
        if wan_vae._INSTALLED in cls.__dict__:
            delattr(cls, wan_vae._INSTALLED)
        if wan_vae._FULL_FRAME_INSTALLED in wanvae.WanAttentionBlock.__dict__:
            delattr(wanvae.WanAttentionBlock, wan_vae._FULL_FRAME_INSTALLED)


class TestFullFrameAttention(_WanVAECase):
    def test_each_rank_keeps_its_rows_of_the_whole_frame_attention(self):
        from sglang.multimodal_gen.runtime.layers import parallel_conv

        torch.manual_seed(0)
        block = wanvae.WanAttentionBlock(8)
        x = torch.randn(1, 8, 2, 7, 5)
        with torch.no_grad():
            whole = wanvae.attention_block_forward(block, x)
        forward = wan_vae.full_frame_attention_forward(wanvae)
        world = 3
        for rank in range(world):
            block.world_size = world
            gathered = []

            def gather(t, parallel_group=None):
                gathered.append(t)
                return x

            def chunk(t, parallel_group=None, rank=rank):
                return torch.tensor_split(t, world, dim=-2)[rank]

            with (
                patch.object(parallel_conv, "gather_height_for_global_op", gather),
                patch.object(parallel_conv, "chunk_height_for_parallel_decode", chunk),
                torch.no_grad(),
            ):
                got = forward(block, chunk(x))
            self.assertEqual(len(gathered), 1)
            torch.testing.assert_close(got, chunk(whole), rtol=0, atol=0)

    def test_one_rank_attends_its_frame_without_collectives(self):
        torch.manual_seed(0)
        block = wanvae.WanAttentionBlock(8)
        x = torch.randn(1, 8, 2, 7, 5)
        forward = wan_vae.full_frame_attention_forward(wanvae)
        with torch.no_grad():
            torch.testing.assert_close(
                forward(block, x),
                wanvae.attention_block_forward(block, x),
                rtol=0,
                atol=0,
            )

    def test_install_wraps_the_attention_once(self):
        wan_vae.install_full_frame_attention()
        forward = wanvae.WanAttentionBlock.__dict__["forward"]
        wan_vae.install_full_frame_attention()
        self.assertIs(wanvae.WanAttentionBlock.__dict__["forward"], forward)


class TestInstall(_WanVAECase):
    def test_unset_leaves_the_decode_alone(self):
        with (
            envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(None),
            envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.override(None),
        ):
            wan_vae.install()
        self.assertIs(wanvae.AutoencoderKLWan.__dict__["decode"], self._decode)
        # Only the full-frame attention, which is installed in every case.
        self.assertIsNot(
            wanvae.WanAttentionBlock.__dict__["forward"], self._attention_forward
        )
        self.assertTrue(
            wanvae.WanAttentionBlock.__dict__[wan_vae._FULL_FRAME_INSTALLED]
        )

    def test_decode_is_wrapped_once_and_passes_on_frames_through(self):
        vae = _tiny_vae()
        z = _latents()
        with envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(128):
            wan_vae.install()
            wan_vae.install()  # idempotent: one wrapper
        self.assertEqual(wanvae.AutoencoderKLWan.decode.__name__, "decode")
        self.assertIs(wanvae.AutoencoderKLWan.decode.__wrapped__, self._decode)
        frames = []
        with torch.no_grad():
            out = vae.decode(z, on_frames=frames.append)
        # Each decoder call's final frames, once and in order: frame 0, then
        # two for each later latent frame (one temporal 2x upsample).
        self.assertEqual([f.shape[2] for f in frames], [1, 2, 2, 2, 2, 2])
        self.assertTrue(torch.equal(torch.cat(frames, dim=2), out))


class TestPaddedConvOut(_WanVAECase):
    def test_values(self):
        with envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(None):
            self.assertIsNone(wan_vae.conv_out_channels())
        with envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(128):
            self.assertEqual(wan_vae.conv_out_channels(), 128)
        with (
            envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(0),
            self.assertRaises(ValueError),
        ):
            wan_vae.conv_out_channels()

    def _install(self, channels=128):
        with envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(channels):
            wan_vae.install()

    def test_padded_conv_out_honors_a_skipped_bias(self):
        vae = _tiny_vae()
        conv = vae.decoder.conv_out
        wan_vae.pad_conv_out(vae.decoder, 128)
        x = torch.randn(1, conv.in_channels, 3, 4, 4)
        conv3d = torch.nn.functional.conv3d
        args = (conv.stride, conv.padding, conv.dilation, conv.groups)
        with torch.no_grad():
            for bias in (None, conv.bias):
                torch.testing.assert_close(
                    conv._conv_forward(x, conv.weight, bias),
                    conv3d(x, conv.weight, bias, *args),
                    rtol=0,
                    atol=1e-5,
                )

    def test_padded_conv_out_decodes_the_same_video(self):
        vae = _tiny_vae()
        z = _latents()
        with torch.no_grad():
            expected = vae.decode(z)
        widths = []
        conv3d = torch.nn.functional.conv3d

        def recording_conv3d(input, weight, *args, **kwargs):
            if weight.shape[0] in (3, 128):  # conv_out's; the others are wider
                widths.append(weight.shape[0])
            return conv3d(input, weight, *args, **kwargs)

        self._install()
        conv_out = vae.decoder.conv_out
        with (
            patch.object(wan_vae.F, "conv3d", recording_conv3d),
            torch.no_grad(),
        ):
            actual = vae.decode(z)
        # Every latent frame's call ran conv_out at 128 output channels.
        self.assertEqual(widths, [128] * z.shape[2])
        # The same products and sums, possibly in another order: CPU oneDNN
        # reorders this 8-channel input (the A14B conv_out, 96 channels, is
        # bitwise on CPU).
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)
        self.assertEqual(conv_out._tpu_padded_weight.shape[0], 128)
        self.assertEqual(int(conv_out._tpu_padded_weight[3:].count_nonzero()), 0)
        self.assertEqual(int(conv_out._tpu_padded_bias[3:].count_nonzero()), 0)

    def test_padded_weights_follow_the_module_weights(self):
        vae = _tiny_vae()
        z = _latents(frames=3)
        self._install()
        with torch.no_grad():
            vae.decode(z)
            vae.decoder.conv_out.weight.mul_(2)
            vae.decoder.conv_out.bias.add_(1)
            actual = vae.decode(z)
        reference = _tiny_vae()
        with torch.no_grad():
            reference.decoder.conv_out.weight.mul_(2)
            reference.decoder.conv_out.bias.add_(1)
            # The decode wrapper is installed on the class: drop the override
            # so the reference runs the stock 3-channel convolution.
            expected = vae.decode.__wrapped__(reference, z)
        # A stale padded copy would decode with the old weights (off by ~1).
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-5)

    def test_wide_output_passes_a_fusion_barrier_before_the_cut(self):
        # A graph compiler could otherwise fold the cut into the convolution and
        # run it at 3 channels again.
        vae = _tiny_vae()
        z = _latents(frames=2)
        self._install()
        seen = []

        def recording_barrier(x):
            seen.append(x.shape[1])
            return x

        with (
            patch.object(wan_vae.current_platform, "fusion_barrier", recording_barrier),
            torch.no_grad(),
        ):
            out = vae.decode(z)
        self.assertEqual(seen, [128] * z.shape[2])
        self.assertEqual(out.shape[1], 3)

    def test_narrower_target_leaves_conv_out_alone(self):
        vae = _tiny_vae()
        self._install(channels=3)
        with torch.no_grad():
            vae.decode(_latents(frames=2))
        self.assertNotIn("_conv_forward", vae.decoder.conv_out.__dict__)


class TestXlaOptions(_WanVAECase):
    def test_values(self):
        with envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.override(" a = b ,c=d"):
            self.assertEqual(wan_vae.xla_options(), {"a": "b", "c": "d"})
        with envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.override(None):
            self.assertIsNone(wan_vae.xla_options())
        for value in ("a", "=b", "a=", "a=b,a=c", "a=b,"):
            with (
                self.subTest(value=value),
                envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.override(value),
                self.assertRaises(ValueError),
            ):
                wan_vae.xla_options()

    def test_options_scope_the_decode_only(self):
        vae = _tiny_vae()
        z = _latents(frames=2)
        scopes = []

        @contextlib.contextmanager
        def recording(options):
            scopes.append(dict(options))
            yield

        with envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.override("x=1"):
            wan_vae.install()
        with (
            patch.object(wan_vae, "_custom_compiler_options", recording),
            torch.no_grad(),
        ):
            vae.decode(z)
        self.assertEqual(scopes, [{"x": "1"}])


if __name__ == "__main__":
    unittest.main()
