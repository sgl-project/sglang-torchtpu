# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's Kandinsky 6 SR patches (sglang_tpu.multimodal_gen.kandinsky6_sr)."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.vaes.kandinsky6_sr import (
    Kandinsky6SRVAEArchConfig,
    Kandinsky6SRVAEConfig,
)
from sglang.multimodal_gen.runtime.models.dits import kandinsky6_sr as sr
from sglang.multimodal_gen.runtime.models.vaes import kandinsky6_sr_vae as kvae
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr import (
    stages,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import kandinsky6_sr, tpu_splash_attn

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=30, suite="nightly-tpu-4-v7", nightly=True)


@contextlib.contextmanager
def _installed():
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (sr.Kandinsky6SRTransformer3DModel, "__init__"),
            (sr.Kandinsky6SRTransformer3DModel, "_compile_regions_spmd_safe"),
            (sr, "shard_seq"),
            (sr, "shard_like"),
            (kvae.Kandinsky6SRVAE, "encode"),
            (kvae.Kandinsky6SRVAE, "_decode"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(kandinsky6_sr, "_DEVICE_TYPES", ("cpu",)))
        kandinsky6_sr.install()
        kandinsky6_sr.install()  # idempotent
        yield


class _Impl:
    softmax_scale = 64**-0.5
    causal = False

    def forward(self, query, key, value, attn_metadata, return_softmax_lse=False):
        raise AssertionError("the stock forward ran")


class TestPaddedHeads(CustomTestCase):
    def test_padded_heads_attend_as_the_narrow_ones(self):
        gen = torch.Generator().manual_seed(0)
        # [B, S, H, D] with 64-wide heads, as the SR DiT hands them over.
        q, k, v = (torch.randn(1, 600, 3, 64, generator=gen) for _ in range(3))
        impl = _Impl()
        with patch.object(kandinsky6_sr, "_DEVICE_TYPES", ("cpu",)):
            got = kandinsky6_sr.padded_head_forward(impl)(q, k, v, None)
        want = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=64**-0.5
        ).transpose(1, 2)
        self.assertEqual(got.shape, q.shape)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    def test_kernel_sees_128_lanes_and_the_narrow_scale(self):
        q = k = v = torch.randn(1, 8, 2, 64)
        calls = []

        def attention(query, key, value, scale=None, is_causal=False):
            calls.append((query.shape, key.shape, value.shape, scale, is_causal))
            return torch.zeros_like(query)

        with (
            patch.object(kandinsky6_sr, "_DEVICE_TYPES", ("cpu",)),
            patch.object(tpu_splash_attn, "tpu_attention", attention),
        ):
            out = kandinsky6_sr.padded_head_forward(_Impl())(q, k, v, None)
        self.assertEqual(out.shape, q.shape)
        shape = torch.Size([1, 2, 8, 128])
        self.assertEqual(calls, [(shape, shape, shape, 64**-0.5, False)])

    def test_other_devices_and_wide_heads_keep_stock(self):
        class Impl(_Impl):
            def forward(self, *args, **kwargs):
                return "stock"

        impl = Impl()
        forward = kandinsky6_sr.padded_head_forward(impl)
        self.assertEqual(forward(*(torch.zeros(1, 4, 2, 64),) * 3, None), "stock")
        with patch.object(kandinsky6_sr, "_DEVICE_TYPES", ("cpu",)):
            wide = torch.zeros(1, 4, 2, 128)
            self.assertEqual(forward(wide, wide, wide, None), "stock")


class TestRingPaddedHeads(CustomTestCase):
    def test_ring_steps_see_128_lanes(self):
        q = k = v = torch.randn(1, 2, 8, 64)  # [B, H, S, D] in the ring path
        calls = []

        class Impl(_Impl):
            def forward_ring_resume(self, query, key, value, kv_info, carry=None):
                calls.append((query.shape[-1], key.shape[-1], value.shape[-1], carry))
                return query + 1, "carry"

        with patch.object(kandinsky6_sr, "_DEVICE_TYPES", ("cpu",)):
            resume = kandinsky6_sr.padded_head_ring_resume(Impl())
            out, carry = resume(q, k, v, torch.tensor([0, 8]))
            resume(q, k, v, torch.tensor([0, 8]), carry)
        self.assertEqual(out.shape, q.shape)
        torch.testing.assert_close(out, q + 1)
        self.assertEqual(calls, [(128, 128, 128, None), (128, 128, 128, "carry")])


def _tiny_vae():
    arch = Kandinsky6SRVAEArchConfig(
        in_channels=3,
        out_channels=3,
        z_channels=8,
        ch=16,
        ch_mult=[0.5, 1, 2, 2, 2],
        decoder_ch=16,
        decoder_ch_mult=[0.5, 1, 2, 2, 2],
        num_res_blocks=1,
        padding_mode="zeros",
        temporal_compress_times=4,
        temporal_compress_start_level=1,
        norm_type="rms_norm",
        double_z=True,
        downsample_version=2,
        fix_pxs=True,
    )
    vae = kvae.Kandinsky6SRVAE(Kandinsky6SRVAEConfig(arch_config=arch)).eval()
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for param in vae.parameters():
            param.copy_(torch.randn(param.shape, generator=gen) * 0.1)
    return vae


class TestCompiledSegments(CustomTestCase):
    def test_segments_compile_once_per_shape_and_match_stock(self):
        vae = _tiny_vae()
        gen = torch.Generator().manual_seed(1)
        z = torch.randn(1, 8, 11, 4, 6, generator=gen)  # segments of 5, 4 and 2
        x = torch.rand(1, 3, 41, 64, 96, generator=gen) * 2 - 1
        with torch.no_grad():
            want_z = kvae.Kandinsky6SRVAE._decode(vae, z)
            want_x, want_split = kvae.Kandinsky6SRVAE.encode(vae, x)
        graphs = {"encoder": [], "decoder": []}

        def backend_for(name):
            def backend(gm, example_inputs):
                graphs[name].append(
                    [
                        (tuple(t.shape), t.storage_offset())
                        for t in example_inputs
                        if isinstance(t, torch.Tensor) and t.dim() == 5
                    ]
                )
                return gm.forward

            return backend

        def compile_kwargs(mode=None):
            return {"backend": backend_for(order.pop(0)), "dynamic": False}

        order = ["decoder", "encoder"]
        args = SimpleNamespace(enable_torch_compile=True)
        with (
            torch.no_grad(),
            _installed(),
            patch.object(kandinsky6_sr, "get_global_server_args", lambda: args),
            patch.object(kandinsky6_sr, "build_torch_compile_kwargs", compile_kwargs),
        ):
            for _ in range(2):  # the second run reuses the programs
                got_z = vae._decode(z)
                got_x, split = vae.encode(x)
        torch.testing.assert_close(got_z, want_z, atol=0, rtol=0)
        torch.testing.assert_close(got_x, want_x, atol=0, rtol=0)
        self.assertEqual(split, want_split)
        self.assertEqual(len(graphs["decoder"]), 3)  # first, middle and last segment
        self.assertEqual(len(graphs["encoder"]), 3)
        for name, calls in graphs.items():
            for inputs in calls:
                self.assertTrue(all(offset == 0 for _, offset in inputs), name)


class TestOutputStage(CustomTestCase):
    def test_only_the_output_rank_stitches(self):
        calls = []

        def stitch(tiles, plan):
            calls.append(len(tiles))
            return torch.zeros(3, 2, 4, 4, dtype=torch.uint8)

        args = SimpleNamespace(num_gpus=8, dp_size=1)
        outputs = {}
        for rank in (0, 3):
            batch = SimpleNamespace(
                extra={
                    stages.SR_TILES_KEY: ["tile"],
                    stages.SR_PLAN_KEY: SimpleNamespace(tiling_scale=2),
                },
                audio=None,
                audio_sample_rate=None,
                fps=24,
                metrics=None,
                usage=None,
                sr_target_resolution=None,
                sr_target_resize_mode="fit",
            )
            with (
                patch.object(stages, "world_group_is_initialized", lambda: True),
                patch.object(stages, "get_world_rank", lambda: rank),
                patch.object(stages, "stitch_tiles", stitch),
            ):
                stage = stages.Kandinsky6SROutputStage.__new__(
                    stages.Kandinsky6SROutputStage
                )
                outputs[rank] = stage.forward(batch, args)
        self.assertEqual(calls, [1])
        self.assertIsNotNone(outputs[0].output)
        self.assertIsNone(outputs[3].output)
        self.assertEqual(outputs[3].fps, 24)


class TestShards(CustomTestCase):
    def test_shards_are_fresh_at_offset_zero(self):
        x = torch.arange(24.0).view(1, 6, 4)
        view = torch.cat([x[:, :1], x], dim=1)[:, 1:]
        self.assertNotEqual(view.storage_offset(), 0)
        with (
            _installed(),
            patch.object(
                kandinsky6_sr, "_STOCK_SHARD_SEQ", lambda t, *a, **k: (view, "plan")
            ),
            patch.object(kandinsky6_sr, "_STOCK_SHARD_LIKE", lambda t, *a, **k: view),
        ):
            out, plan = sr.shard_seq(x)
            like = sr.shard_like(x, plan)
            spmd_safe = sr.Kandinsky6SRTransformer3DModel._compile_regions_spmd_safe
        self.assertEqual(plan, "plan")
        for t in (out, like):
            self.assertEqual(t.storage_offset(), 0)
            self.assertTrue(t.is_contiguous())
            torch.testing.assert_close(t, x, atol=0, rtol=0)
        self.assertTrue(spmd_safe)


if __name__ == "__main__":
    import unittest

    unittest.main()
