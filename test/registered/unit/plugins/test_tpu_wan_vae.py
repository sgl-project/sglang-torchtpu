# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Wan VAE decode precision scope (sglang_tpu.multimodal_gen.wan_vae)."""

import contextlib
import logging
import math
import os
import sys
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.runtime.models.vaes import wanvae
from sglang.multimodal_gen.runtime.utils.decode_stream import stream_decoded_chunks
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir)
)

# The CPU gate installs python/ only; an installed sglang_tpu still wins.
if _REPO_ROOT not in sys.path:
    sys.path.append(_REPO_ROOT)

from sglang_tpu.envs import envs  # noqa: E402
from sglang_tpu.multimodal_gen import wan_vae  # noqa: E402

_CHUNK_ENV = "SGLANG_DIFFUSION_WAN_VAE_DECODE_CHUNK_FRAMES"
_COMPILE_ENV = "SGLANG_DIFFUSION_ENABLE_WAN_VAE_DECODER_COMPILE"


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
    """Restores AutoencoderKLWan.decode, the chunk env and the precision."""

    def setUp(self):
        cls = wanvae.AutoencoderKLWan
        self._decode = cls.__dict__["decode"]
        self._attention_forward = wanvae.WanAttentionBlock.__dict__["forward"]
        self._chunk_env = os.environ.pop(_CHUNK_ENV, None)
        self._compile_env = os.environ.pop(_COMPILE_ENV, None)
        self._precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

    def tearDown(self):
        cls = wanvae.AutoencoderKLWan
        cls.decode = self._decode
        wanvae.WanAttentionBlock.forward = self._attention_forward
        if wan_vae._INSTALLED in cls.__dict__:
            delattr(cls, wan_vae._INSTALLED)
        if wan_vae._FULL_FRAME_INSTALLED in wanvae.WanAttentionBlock.__dict__:
            delattr(wanvae.WanAttentionBlock, wan_vae._FULL_FRAME_INSTALLED)
        for name, value in (
            (_CHUNK_ENV, self._chunk_env),
            (_COMPILE_ENV, self._compile_env),
        ):
            os.environ.pop(name, None)
            if value is not None:
                os.environ[name] = value
        torch.set_float32_matmul_precision(self._precision)


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
                forward(block, x), wanvae.attention_block_forward(block, x), rtol=0, atol=0
            )

    def test_install_wraps_the_attention_once(self):
        wan_vae.install_full_frame_attention()
        forward = wanvae.WanAttentionBlock.__dict__["forward"]
        wan_vae.install_full_frame_attention()
        self.assertIs(wanvae.WanAttentionBlock.__dict__["forward"], forward)


class TestDecodePrecision(_WanVAECase):
    def test_unset_leaves_the_decode_alone(self):
        with (
            envs.SGLANG_TPU_WAN_VAE_DECODE_FP32_MATMUL_PRECISION.override(None),
            envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.override(None),
            envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(None),
        ):
            self.assertFalse(wan_vae.requested())
            wan_vae.install()
        self.assertIs(wanvae.AutoencoderKLWan.__dict__["decode"], self._decode)
        # Only the full-frame attention, which is installed in every case, and
        # no precision wrapper (functools.wraps would set __wrapped__).
        forward = wanvae.WanAttentionBlock.__dict__["forward"]
        self.assertIsNot(forward, self._attention_forward)
        self.assertFalse(hasattr(forward, "__wrapped__"))
        self.assertTrue(
            wanvae.WanAttentionBlock.__dict__[wan_vae._FULL_FRAME_INSTALLED]
        )

    def test_values(self):
        cases = {"high": "high", " HIGH ": "high", "highest": "highest", "": None}
        for value, expected in cases.items():
            with (
                self.subTest(value=value),
                envs.SGLANG_TPU_WAN_VAE_DECODE_FP32_MATMUL_PRECISION.override(value),
            ):
                self.assertEqual(wan_vae.decode_precision(), expected)
        # "medium" (one bf16 pass) is below TF32; it is not offered.
        for value in ("medium", "fp32", "bf16_3x"):
            with (
                self.subTest(value=value),
                envs.SGLANG_TPU_WAN_VAE_DECODE_FP32_MATMUL_PRECISION.override(value),
                self.assertRaises(ValueError),
            ):
                wan_vae.decode_precision()

    def test_the_decode_and_only_the_decode_runs_at_the_precision(self):
        vae = _tiny_vae()  # its mid block has the attention block
        z = _latents()
        with torch.no_grad():
            expected = vae.decode(z)
        seen = []
        conv_in = vae.decoder.conv_in
        conv_in_forward = conv_in.forward
        attention_block_forward = wanvae.attention_block_forward

        # WanCausalConv3d.forward also takes the residual/skip_bias epilogue
        # keywords, which the cached decode passes on every call.
        def recording_conv_in(*args, **kwargs):
            seen.append(("conv", torch.get_float32_matmul_precision()))
            return conv_in_forward(*args, **kwargs)

        def recording_attention(block, x):
            seen.append(("attention", torch.get_float32_matmul_precision()))
            return attention_block_forward(block, x)

        with envs.SGLANG_TPU_WAN_VAE_DECODE_FP32_MATMUL_PRECISION.override("high"):
            wan_vae.install()
            wan_vae.install()  # idempotent: one wrapper
        with (
            patch.object(conv_in, "forward", recording_conv_in),
            patch.object(wanvae, "attention_block_forward", recording_attention),
            torch.no_grad(),
        ):
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
            actual = vae.decode(z)
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        # Per latent frame: the convolutions at "high", the attention block
        # (float32 matmuls, IEEE float32 on CUDA by default) at "highest".
        self.assertEqual(
            seen, [("conv", "high"), ("attention", "highest")] * z.shape[2]
        )
        # CPU convolutions ignore the setting: the wrapper returns the decode.
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(wanvae.AutoencoderKLWan.decode.__name__, "decode")
        self.assertIs(
            wanvae.AutoencoderKLWan.decode.__wrapped__, self._decode
        )  # wrapped once

    def test_precision_is_restored_when_the_decode_raises(self):
        vae = _tiny_vae()
        with envs.SGLANG_TPU_WAN_VAE_DECODE_FP32_MATMUL_PRECISION.override("high"):
            wan_vae.install()
        with (
            patch.object(vae.decoder, "forward", side_effect=RuntimeError("boom")),
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            vae.decode(_latents())
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")

    def test_scope_keeps_an_equal_global_and_restores_others(self):
        with wan_vae.fp32_matmul_precision("highest"):
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        with wan_vae.fp32_matmul_precision(None):
            self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        torch.set_float32_matmul_precision("medium")
        with wan_vae.fp32_matmul_precision("high"):
            self.assertEqual(torch.get_float32_matmul_precision(), "high")
        self.assertEqual(torch.get_float32_matmul_precision(), "medium")


def _fake_decode(vae, z, on_frames=None):
    """Encodes the call's settings in its output: chunk + 0.25 at "high"."""
    chunk = int(os.environ.get(_CHUNK_ENV, "1"))
    high = torch.get_float32_matmul_precision() == "high"
    vae.calls.append(
        (
            chunk,
            torch.get_float32_matmul_precision(),
            os.environ.get(_COMPILE_ENV) == "1",
        )
    )
    out = torch.full_like(z, (chunk + (0.25 if high else 0.0)) / 100)
    if on_frames is not None:
        on_frames(out)
    return out


class _FakeVAE:
    def __init__(self):
        self.calls = []


class TestDecodeAB(_WanVAECase):
    def test_only_the_first_variant_reaches_on_frames(self):
        frames = []
        variants = [(5, "high", False, None), (1, "highest", False, None)]
        out = wan_vae.decode_ab(
            _FakeVAE(), _fake_decode, torch.zeros(1, 4, 3, 2, 2), variants, frames.append
        )
        self.assertEqual(len(frames), 1)
        self.assertTrue(torch.equal(frames[0], out))

    def test_variants(self):
        value = (
            " 1:highest, 5:HIGH ,20:high, 5:highest:compile, 5:highest:convout128,"
            "5:high:convout8:compile, 5:highest:xla, 5:high:xla:convout128"
        )
        with (
            envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.override(value),
            envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.override("a=b"),
        ):
            self.assertEqual(
                wan_vae.ab_variants(),
                [
                    (1, "highest", False, None, False),
                    (5, "high", False, None, False),
                    (20, "high", False, None, False),
                    (5, "highest", True, None, False),
                    (5, "highest", False, 128, False),
                    (5, "high", True, 8, False),
                    (5, "highest", False, None, True),
                    (5, "high", False, 128, True),
                ],
            )
            self.assertTrue(wan_vae.requested())
        # ':xla' takes SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS's options.
        with (
            envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.override("5:high:xla"),
            self.assertRaises(ValueError),
        ):
            wan_vae.ab_variants()
        for value in (
            "5:high:xla:xla",
            "5:high:compile:xla",
            "5",
            "0:high",
            "x:high",
            "5:medium",
            "5:high,",
            "5:high:eager",
            "5:high:compile:1",
            "5:high:compile:compile",
            "5:high:convout0",
            "5:high:convout",
            "5:high:convout8:convout16",
        ):
            with (
                self.subTest(value=value),
                envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.override(value),
                self.assertRaises(ValueError),
            ):
                wan_vae.ab_variants()

    def test_runs_every_variant_and_returns_the_first(self):
        vae = _FakeVAE()
        z = torch.zeros(1, 4, 3, 2, 2)
        variants = [
            (5, "high", False, None),
            (1, "highest", True, None),
            (20, "high", False, None),
        ]
        os.environ[_CHUNK_ENV] = "3"
        os.environ[_COMPILE_ENV] = "1"
        with self.assertLogs(wan_vae.logger, logging.INFO) as logs:
            out = wan_vae.decode_ab(vae, _fake_decode, z, variants)
        # Each variant pins the decoder compile on or off for its decode.
        self.assertEqual(vae.calls, [v[:3] for v in variants])
        self.assertTrue(torch.equal(out, torch.full_like(z, 5.25 / 100)))
        self.assertEqual(os.environ[_CHUNK_ENV], "3")
        self.assertEqual(os.environ[_COMPILE_ENV], "1")
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertEqual(len(logs.records), 3)
        self.assertIn("chunk 5 high:", logs.output[0])
        self.assertIn("reference", logs.output[0])
        self.assertIn("chunk 1 highest compiled:", logs.output[1])
        # Variant 2 differs from the reference by (5.25 - 1) / 100 in [-1, 1].
        self.assertIn(f"max_abs {4.25 / 200:.3e}", logs.output[1])

    def test_installed_decode_returns_the_per_frame_video(self):
        vae = _tiny_vae()
        z = _latents()
        with torch.no_grad():
            expected = vae.decode(z)
        widths = []
        decoder_forward = vae.decoder.forward

        def counting_forward(x):
            widths.append(x.shape[2])
            return decoder_forward(x)

        with envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.override("1:highest,5:high"):
            wan_vae.install()
        with (
            patch.object(vae.decoder, "forward", counting_forward),
            torch.no_grad(),
            self.assertLogs(wan_vae.logger, logging.INFO) as logs,
        ):
            actual = vae.decode(z)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(widths, [1] * 6 + [1, 5])
        self.assertNotIn(_CHUNK_ENV, os.environ)
        self.assertEqual(len(logs.records), 2)

    def test_compiled_variant_decodes_the_eager_video(self):
        vae = _tiny_vae()
        z = _latents()
        compile_kwargs = {"backend": "eager", "fullgraph": True, "dynamic": False}
        with envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.override("1:highest,1:highest:compile"):
            wan_vae.install()
        torch._dynamo.reset()
        with (
            patch.object(
                wanvae,
                "resolve_torch_compile_kwargs",
                return_value=(compile_kwargs, None),
            ),
            torch.no_grad(),
            self.assertLogs(wan_vae.logger, logging.INFO) as logs,
        ):
            vae.decode(z)
        torch._dynamo.reset()
        # Calls 3 to 6 of the second variant ran the compiled decoder.
        self.assertIs(vae._compiled_decoder["decoder"], vae.decoder)
        self.assertIsNotNone(vae._compiled_decoder["module"])
        self.assertIn("chunk 1 highest compiled:", logs.output[1])
        self.assertIn("psnr inf dB", logs.output[1])
        self.assertNotIn(_COMPILE_ENV, os.environ)

    def test_conv_out_variant_runs_the_padded_conv_then_restores_it(self):
        vae = _tiny_vae()
        z = _latents(frames=3)
        widths = []
        conv3d = torch.nn.functional.conv3d

        def recording_conv3d(input, weight, *args, **kwargs):
            if weight.shape[0] in (3, 128):  # conv_out's; the others are wider
                widths.append(weight.shape[0])
            return conv3d(input, weight, *args, **kwargs)

        with envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.override(
            "1:highest,1:highest:convout128"
        ):
            wan_vae.install()
        with (
            patch.object(wan_vae.F, "conv3d", recording_conv3d),
            torch.no_grad(),
            self.assertLogs(wan_vae.logger, logging.INFO) as logs,
        ):
            vae.decode(z)
            vae.decode(z)
        # Per decode: the reference's stock calls, then the variant's padded ones.
        self.assertEqual(widths, ([3] * 3 + [128] * 3) * 2)
        self.assertIn("chunk 1 highest conv_out 128:", logs.output[1])
        self.assertIsNone(vae.decoder.conv_out._tpu_conv_out_channels)

    def test_only_the_returned_variant_reaches_a_streaming_save(self):
        vae = _tiny_vae()
        z = _latents()
        with envs.SGLANG_TPU_DEBUG_WAN_VAE_AB.override("1:highest,5:high"):
            wan_vae.install()
        streamed = []
        with (
            torch.no_grad(),
            stream_decoded_chunks(lambda c: streamed.append(c.shape[2])),
            self.assertLogs(wan_vae.logger, logging.INFO),
        ):
            vae.decode(z)
        # One temporal 2x upsample: frame 0, then 2 frames a latent frame.
        self.assertEqual(streamed, [1] + [2] * 5)


class TestPaddedConvOut(_WanVAECase):
    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

    def test_values(self):
        with envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(None):
            self.assertIsNone(wan_vae.conv_out_channels())
        with envs.SGLANG_TPU_OPT_WAN_VAE_CONV_OUT_CHANNELS.override(128):
            self.assertEqual(wan_vae.conv_out_channels(), 128)
            self.assertTrue(wan_vae.requested())
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
        # The attention block has the full-frame forward and no precision
        # wrapper (functools.wraps would set __wrapped__): none is requested.
        forward = wanvae.WanAttentionBlock.__dict__["forward"]
        self.assertTrue(wanvae.WanAttentionBlock.__dict__[wan_vae._FULL_FRAME_INSTALLED])
        self.assertFalse(hasattr(forward, "__wrapped__"))
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

    def test_compiled_calls_keep_the_padded_conv_out(self):
        vae = _tiny_vae()
        z = _latents()
        self._install()
        with torch.no_grad():
            expected = vae.decode(z)
        compile_kwargs = {"backend": "eager", "fullgraph": True, "dynamic": False}
        torch._dynamo.reset()
        with (
            patch.object(
                wanvae,
                "resolve_torch_compile_kwargs",
                return_value=(compile_kwargs, None),
            ),
            patch.dict(os.environ, {_COMPILE_ENV: "1"}),
            torch.no_grad(),
        ):
            actual = vae.decode(z)
        self.assertIsNotNone(vae._compiled_decoder["module"])
        self.assertTrue(torch.equal(actual, expected))


class TestXlaOptions(_WanVAECase):
    def test_values(self):
        with envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.override(" a = b ,c=d"):
            self.assertEqual(wan_vae.xla_options(), {"a": "b", "c": "d"})
            self.assertTrue(wan_vae.requested())
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

    def test_ab_scopes_only_the_xla_variant(self):
        scopes = []

        @contextlib.contextmanager
        def recording(options):
            scopes.append(dict(options))
            yield

        variants = [(5, "high", False, None, False), (5, "high", False, None, True)]
        with (
            envs.SGLANG_TPU_OPT_WAN_VAE_XLA_OPTIONS.override("x=1"),
            patch.object(wan_vae, "_custom_compiler_options", recording),
            self.assertLogs(wan_vae.logger, logging.INFO) as logs,
        ):
            wan_vae.decode_ab(
                _FakeVAE(), _fake_decode, torch.zeros(1, 4, 3, 2, 2), variants
            )
        self.assertEqual(scopes, [{"x": "1"}])
        self.assertIn("chunk 5 high xla:", logs.output[1])
        self.assertIn("max_abs 0.000e+00", logs.output[1])


class TestCompare(CustomTestCase):
    def test_identical_decodes(self):
        x = torch.rand(3, 2, 4, 4) * 2 - 1
        d = wan_vae.compare(x, x.clone())
        self.assertEqual(d["max_abs"], 0.0)
        self.assertEqual(d["psnr_db"], math.inf)
        self.assertEqual(d["u8_mismatched"], 0)
        self.assertEqual(d["u8_max_diff"], 0)

    def test_frame_scale_and_uint8_counts(self):
        ref = torch.zeros(3, 1, 2, 2)  # 0.5 on the frame scale, uint8 127
        out = ref.clone()
        out[0, 0, 0, 0] = 0.02  # +0.01 on the frame scale: 0.51 * 255 = 130.05
        d = wan_vae.compare(out, ref)
        self.assertAlmostEqual(d["max_abs"], 0.01, places=6)
        self.assertAlmostEqual(d["psnr_db"], 10 * math.log10(12 / 0.01**2), delta=1e-3)
        self.assertEqual(d["u8_mismatched"], 1)
        self.assertEqual(d["u8_max_diff"], 3)
        self.assertAlmostEqual(d["u8_mismatched_frac"], 1 / 12)


if __name__ == "__main__":
    unittest.main()
