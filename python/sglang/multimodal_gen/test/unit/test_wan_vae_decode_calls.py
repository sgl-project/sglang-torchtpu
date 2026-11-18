"""Wan VAE feature-cached decode: the compiled decoder calls decode the same
video as one eager call per latent frame."""

import os
import threading
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
from torch._dynamo.utils import counters

from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.runtime.models.vaes import wanvae

_COMPILE_ENV = "SGLANG_DIFFUSION_ENABLE_WAN_VAE_DECODER_COMPILE"


def _tiny_vae(is_residual: bool) -> wanvae.AutoencoderKLWan:
    """Wan2.2-style (residual, patchified) or Wan2.1-style decoder with both
    temporal upsamples."""
    config = WanVAEConfig(
        arch_config=WanVAEArchConfig(
            base_dim=8,
            z_dim=4,
            dim_mult=(1, 2, 2, 2),
            num_res_blocks=1,
            is_residual=is_residual,
            out_channels=12 if is_residual else 3,
            patch_size=2 if is_residual else None,
            latents_mean=(0.0,) * 4,
            latents_std=(1.0,) * 4,
        )
    )
    config.load_encoder = False
    torch.manual_seed(0)
    return wanvae.AutoencoderKLWan(config).eval()


def _decode(vae, z, compile_decoder=False):
    with patch.dict(os.environ, {_COMPILE_ENV: str(int(compile_decoder))}):
        with torch.no_grad():
            return vae.decode(z)


class TestWanVaeDecodeCalls(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        counters.clear()

    def tearDown(self):
        torch._dynamo.reset()

    def test_compiled_decoder_calls_trace_one_graph(self):
        compile_kwargs = {"backend": "eager", "fullgraph": True, "dynamic": False}
        with patch.object(
            wanvae,
            "resolve_torch_compile_kwargs",
            return_value=(compile_kwargs, None),
        ):
            for is_residual in (True, False):
                with self.subTest(is_residual=is_residual):
                    vae = _tiny_vae(is_residual)
                    torch._dynamo.reset()
                    counters.clear()
                    for frames in (8, 10):
                        z = torch.randn(1, 4, frames, 4, 6)
                        traced = counters["stats"]["unique_graphs"]
                        expected = _decode(vae, z)
                        # The eager decode traces nothing.
                        self.assertEqual(counters["stats"]["unique_graphs"], traced)
                        for _ in range(2):
                            # fullgraph=True: a graph break would raise.
                            actual = _decode(vae, z, compile_decoder=True)
                            self.assertTrue(torch.equal(actual, expected))
                    # The calls with full caches share one graph, across decodes
                    # and clip lengths.
                    self.assertEqual(counters["stats"]["unique_graphs"], 1)
                    self.assertIs(vae._compiled_decoder["decoder"], vae.decoder)
                    self.assertNotIn("_compiled_decoder", dict(vae.named_modules()))

    def test_compiled_calls_take_offset_zero_contiguous_frames(self):
        # torch_tpu guards input offsets and strides: a view of the latent
        # would compile one graph per frame and per clip length.
        vae = _tiny_vae(True)
        seen = []

        def recording_compile(module, **kwargs):
            def compiled(x):
                seen.append((x.storage_offset(), x.is_contiguous()))
                return module(x)

            return compiled

        with (
            patch.object(
                wanvae, "resolve_torch_compile_kwargs", return_value=({}, None)
            ),
            patch.object(wanvae.torch, "compile", recording_compile),
        ):
            for frames in (8, 10):
                z = torch.randn(1, 4, frames, 4, 6)
                expected = _decode(vae, z)
                actual = _decode(vae, z, compile_decoder=True)
                self.assertTrue(torch.equal(actual, expected))
        # Calls 3 to 8 and 3 to 10 run compiled.
        self.assertEqual(len(seen), 6 + 8)
        self.assertEqual(set(seen), {(0, True)})

    def test_failed_compiled_call_decodes_eagerly_from_then_on(self):
        vae = _tiny_vae(True)
        z = torch.randn(1, 4, 8, 4, 6)
        expected = _decode(vae, z)
        calls = []

        def failing_compile(module, **kwargs):
            def compiled(x):
                calls.append(x.shape)
                raise RuntimeError("the backend cannot lower this graph")

            return compiled

        with (
            patch.object(
                wanvae, "resolve_torch_compile_kwargs", return_value=({}, None)
            ),
            patch.object(wanvae.torch, "compile", failing_compile),
        ):
            for _ in range(2):
                actual = _decode(vae, z, compile_decoder=True)
                self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(vae._compiled_decoder["module"])
        self.assertFalse(wanvae.feat_idx._plain)

    def test_decode_vars_stay_context_local_outside_compiled_calls(self):
        seen = []
        with wanvae.forward_context(feat_cache_arg=[None], feat_idx_arg=3):
            worker = threading.Thread(target=lambda: seen.append(wanvae.feat_idx.get()))
            worker.start()
            worker.join()
            with wanvae.traceable_decode_vars():
                self.assertEqual(wanvae.feat_idx.get(), 3)
                self.assertIsNone(wanvae.feat_idx.set(5))
                self.assertEqual(wanvae.feat_idx.get(), 5)
            # Written back to the context var on exit.
            self.assertEqual(wanvae.feat_idx.get(), 5)
            self.assertFalse(wanvae.feat_idx._plain)
        self.assertEqual(seen, [0])
        self.assertEqual(wanvae.feat_idx.get(), 0)
        self.assertIsNone(wanvae.feat_cache.get())


if __name__ == "__main__":
    unittest.main()
