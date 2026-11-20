# SPDX-License-Identifier: Apache-2.0
"""HY-Nomas on TPU workers (sglang_tpu.multimodal_gen.nomas): the segment
attention on the flash kernel, PixelUMM's grouped tile attention, host-side
tables and the compiled DiT and decoder regions, against the stock CPU paths.
CPU tensors take the TPU paths where a test routes them (``_DEVICE_TYPES``);
the kernel test runs on a TPU7x core."""

import contextlib
import importlib.util
import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.dits.nomas import NomasArchConfig, NomasConfig
from sglang.multimodal_gen.configs.models.vaes.nomas import (
    NomasVAEArchConfig,
    NomasVAEConfig,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.kvcache.nomas_segment_cache import (
    NomasAttentionPlan,
)
from sglang.multimodal_gen.runtime.models.dits import nomas as dit
from sglang.multimodal_gen.runtime.models.dits import nomas_layers
from sglang.multimodal_gen.runtime.models.dits.nomas_layers import ParallelInfo
from sglang.multimodal_gen.runtime.models.vaes import nomas as vae
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas import (
    stages,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.denoise_loop import (
    NomasLatentSession,
    NomasSegmentCondition,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import nomas

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=60, suite="nightly-tpu-4-v7", nightly=True)


def setUpModule():
    ensure_distributed_env_defaults()
    maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)


@contextlib.contextmanager
def _installed(device_types=("cpu",), compiled=False):
    """The TPU paths on ``device_types``; ``compiled`` compiles the regions
    with aot_eager (eager numerics). Every patched name is restored."""
    with contextlib.ExitStack() as stack:
        for owner, name, _ in nomas.patches():
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(nomas, "_DEVICE_TYPES", device_types))
        stack.enter_context(patch.dict(nomas._REGIONS, clear=True))
        stack.enter_context(patch.object(nomas, "_compile_enabled", lambda: compiled))
        stack.enter_context(
            patch.object(nomas, "_compile_backend", lambda: "aot_eager")
        )
        nomas._cached_local_layout.cache_clear()
        nomas.install()
        nomas.install()  # idempotent
        yield
    nomas._cached_local_layout.cache_clear()


def _tiny_model(heads=2):
    torch.manual_seed(123)
    model = dit.NomasTransformer3DModel(
        NomasConfig(
            arch_config=NomasArchConfig(
                hidden_size=128 * heads, num_attention_heads=heads, num_layers=2
            )
        ),
        attention_backend="sdpa",
    )
    model = model.eval().requires_grad_(False)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim == 1 and "norm" in name and name.endswith("weight"):
                p.fill_(1)
            else:
                p.normal_(0, 0.02)
    return model


def _condition(valid, offsets=None, windows=None, camera=False):
    gen = torch.Generator().manual_seed(valid)
    return NomasSegmentCondition(
        text_states=torch.randn(1, 8, 3584, generator=gen),
        text_mask=torch.arange(8)[None] < valid,
        camera_embedding=torch.randn(1, 2, 64, generator=gen) if camera else None,
        text_offsets=offsets,
        video_windows=windows,
        action_start=-1 if offsets else 3,
    )


# Plain, two timed actions, camera on, camera on again, plain: the history
# grows, then FIFO eviction rewrites it.
CONDITIONS = (
    lambda: _condition(6),
    lambda: _condition(7, (2, 4, 7), ((0, 1, 0), (1, 2, 1))),
    lambda: _condition(8, camera=True),
    lambda: _condition(5, (3, 5), ((0, 1, -1), (1, 2, 0)), camera=True),
    lambda: _condition(4),
)


def _session(model, conditions, capacity=2 * (2 * 12 + 8)):
    session = NomasLatentSession(model, cache_capacity=capacity, seed=7, steps=2)
    with torch.inference_mode():
        return [
            session.generate_segment((2, 3, 4), c(), cache_segment=True, evict=True)
            for c in conditions
        ]


def _layout(history_segments=2, offsets=(6, 10, 10, 18), windows=None, valid=18):
    """A segment of 6 frames x 16 tokens and 24 text rows after
    ``history_segments`` cached segments whose padding and action tails are
    hidden; frames 2-3 see no action text."""
    frames, spatial, text = 6, 16, 24
    if windows is None and offsets is not None:
        windows = ((0, 2, 0), (2, 4, -1), (4, 6, 2))
    segment = frames * spatial + text
    history = torch.zeros(1, history_segments * segment, dtype=torch.bool)
    for start in range(0, history.shape[1], segment):
        history[:, start : start + frames * spatial + 5] = True
    text_mask = (torch.arange(text) < valid)[None]
    return (text_mask, history, frames, spatial, offsets, windows)


def _qkv(layout, heads=4, dim=32, dtype=torch.float32):
    gen = torch.Generator().manual_seed(11)
    n = layout[2] * layout[3] + layout[0].shape[1]
    h = layout[1].shape[1]
    q = torch.randn(1, n, heads, dim, generator=gen).to(dtype)
    k, v = (torch.randn(1, h + n, heads, dim, generator=gen).to(dtype) for _ in "kv")
    return q, k[:, h:], v[:, h:], {"k": k[:, :h], "v": v[:, :h]}, h


class TestSegmentAttention(CustomTestCase):
    def test_matches_the_stock_mask(self):
        cases = {
            "history, actions": _layout(),
            "history, no actions": _layout(offsets=None),
            "no history, actions": _layout(history_segments=0),
            "action text rows only": _layout(offsets=(10, 18), windows=((0, 6, -1),)),
        }
        for name, layout in cases.items():
            with self.subTest(name):
                q, k, v, cache, h = _qkv(layout)
                want = NomasAttentionPlan(*layout, "sdpa")(q, k, v, cache, h)
                with _installed():
                    plan = dit.NomasAttentionPlan(*layout, "flex")
                    self.assertIsInstance(plan, nomas.TpuNomasAttentionPlan)
                    got = plan(q, k, v, cache, h)
                self.assertTrue(got.is_contiguous())
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
                # Padded text queries return zeros.
                valid = layout[0].sum().item()
                self.assertFalse(got[:, layout[2] * layout[3] + valid :].any())

    def test_shapes_do_not_depend_on_the_prompt(self):
        """Valid text length, action offsets and windows change values only."""
        with _installed():
            plans = [
                dit.NomasAttentionPlan(*_layout(valid=18), "flex"),
                dit.NomasAttentionPlan(
                    *_layout(offsets=(3, 9, 12, 20), valid=20), "flex"
                ),
            ]
        shapes = [
            {k: v.shape for k, v in vars(p).items() if isinstance(v, torch.Tensor)}
            for p in plans
        ]
        self.assertEqual(shapes[0], shapes[1])
        self.assertEqual(shapes[0]["history_index"], (2 * (6 * 16 + 24),))


@unittest.skipIf(importlib.util.find_spec("torch_tpu") is None, "no torch_tpu")
class TestSegmentAttentionKernel(CustomTestCase):
    def test_flash_kernel_matches_the_stock_mask(self):
        from torch_tpu._internal.utils.hardware import get_tpu_device_name

        from sglang_tpu.multimodal_gen import tpu_splash_attn as tsa

        if "7" not in get_tpu_device_name():
            self.skipTest("Pallas kernels are TPU7x-only")
        self.assertTrue(tsa._check_splash_available())
        tsa._register_ops()
        # 2 frames x 256 tokens + 128 text rows (above the SDPA-only 512 x 512);
        # two cached segments.
        frames, spatial, text, valid = 2, 256, 128, 90
        segment = frames * spatial + text
        history = torch.zeros(1, 2 * segment, dtype=torch.bool)
        for start in (0, segment):
            history[:, start : start + frames * spatial + 40] = True
        layout = (
            (torch.arange(text) < valid)[None],
            history,
            frames,
            spatial,
            (30, 60, 90),
            ((0, 1, 0), (1, 2, 1)),
        )
        q, k, v, cache, h = _qkv(layout, heads=2, dim=128, dtype=torch.bfloat16)
        want = NomasAttentionPlan(*layout, "sdpa")(
            q.float(), k.float(), v.float(), {x: y.float() for x, y in cache.items()}, h
        )
        tpu = torch.device("tpu")
        with (
            _installed(device_types=("tpu",)),
            patch.object(tsa, "_FLASH_RING_OP", wraps=tsa._FLASH_RING_OP) as flash,
        ):
            plan = dit.NomasAttentionPlan(
                layout[0].to(tpu), layout[1].to(tpu), *layout[2:], "flex"
            )
            cache = {x: y.to(tpu) for x, y in cache.items()}
            got = plan(q.to(tpu), k.to(tpu), v.to(tpu), cache, h).cpu()
        self.assertEqual(flash.call_count, 2)  # current segment, then history
        self.assertLess(((got.float() - want).norm() / want.norm()).item(), 1e-2)
        self.assertFalse(got[:, frames * spatial + valid :].any())


class TestPixelUMMTiles(CustomTestCase):
    def _decoder(self, tile):
        torch.manual_seed(5)
        arch = NomasVAEArchConfig(
            hidden_size=64,
            intermediate_size=128,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            head_dim=32,
            local_attention_tile=tile,
        )
        decoder = vae.NomasPixelUMMDecoder(NomasVAEConfig(arch_config=arch)).eval()
        with torch.no_grad():
            for p in decoder.parameters():
                p.normal_(0, 0.05)
        return decoder

    def test_groups_hold_the_block_mask_key_sets(self):
        # Ragged edge tiles (10 = 3 + 3 + 3 + 1 rows) and a 5 x 9 tile plane.
        for grid, tile in (((2, 10, 14), 3), ((2, 10, 18), 2)):
            frames, height, width = grid
            with self.subTest(grid=grid, tile=tile), _installed():
                perm, inverse, groups = vae._cached_local_layout(grid, tile, "cpu")
            torch.testing.assert_close(perm[inverse], torch.arange(perm.numel()))
            tiles_w = -(-width // tile)
            t, h, w = torch.meshgrid(*(torch.arange(n) for n in grid), indexing="ij")
            tile_h, tile_w = (h // tile).flatten()[perm], (w // tile).flatten()[perm]
            center_h = tile_h.clamp(1, max(-(-height // tile) - 2, 1))
            center_w = tile_w.clamp(1, max(tiles_w - 2, 1))
            covered = 0
            for (start, end), keys in zip(groups.slices, groups.keys):
                self.assertEqual(start, covered)
                covered = end
                for q in (start, end - 1):
                    want = (
                        (
                            ((center_h[q] - tile_h).abs() <= 1)
                            & ((center_w[q] - tile_w).abs() <= 1)
                        )
                        .nonzero()
                        .squeeze(-1)
                    )
                    torch.testing.assert_close(keys, want)
            self.assertEqual(covered, perm.numel())
            # One dense call per window centre.
            centres = set(zip(center_h.tolist(), center_w.tolist()))
            self.assertEqual(len(groups.slices), len(centres))

    def test_decoder_matches_the_flex_block_mask(self):
        for latent_hw, tile in (((5, 9), 2), ((5, 7), 3)):
            decoder = self._decoder(tile)
            latent = torch.randn(1, 32, 2, *latent_hw)
            xt = torch.randn(1, 3, 8, latent_hw[0] * 32, latent_hw[1] * 32)
            with torch.inference_mode():
                want = decoder.decoder(latent, xt)
                with _installed():
                    got = decoder.decoder(latent, xt)
            torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)

    def test_decode_output_takes_in_place_normalization(self):
        with torch.no_grad():  # TPU stages run under no_grad, not inference mode
            self._decoder(2).decode(torch.randn(1, 32, 2, 3, 3)).sample.div_(2)

    def test_compiled_decode_matches_eager(self):
        torch._dynamo.reset()
        decoder = self._decoder(2)
        latent = torch.randn(1, 32, 12, 5, 9)
        with torch.inference_mode(), _installed():
            want = decoder.decode(latent).sample
        counters.clear()
        with torch.inference_mode(), _installed(compiled=True):
            got = decoder.decode(latent).sample
            self.assertIsNotNone(nomas._REGIONS["GenLayer.forward"])
        torch.testing.assert_close(got, want, atol=0, rtol=0)
        # Two layers, two windows: one graph.
        self.assertEqual(counters["stats"]["unique_graphs"], 1)


class TestHostTables(CustomTestCase):
    def test_rope_tables_match_and_take_the_rank_slice_contiguous(self):
        model = _tiny_model()
        latent = torch.zeros(1, 32, 2, 3, 4)
        cache = model.allocate_cache(64)
        cache.frame_offset = 5
        mask = torch.arange(8)[None] < 6
        want = model.prepare_segment_context(latent, mask, cache).rope
        with _installed():
            got = model.prepare_segment_context(latent, mask, cache).rope
            model.parallel = ParallelInfo(1, 2, 1)
            shard = model.prepare_segment_context(latent, mask, cache).rope
        for a, b, s in zip(got, want, shard):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            torch.testing.assert_close(s, b[12:], atol=0, rtol=0)
            self.assertEqual(s.storage_offset(), 0)

    def test_pair_swap_rotation_is_bitwise(self):
        x = torch.randn(1, 24, 2, 128).to(torch.bfloat16) * 3
        tables = dit.get_nd_rotary_pos_embed([16, 56, 56], (4, 0, 0), (6, 3, 4), 256.0)
        want = nomas_layers.apply_rotary_emb_single(x, tables)
        with _installed():
            got = nomas_layers.apply_rotary_emb_single(x, tables)
        torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_text_positions_come_from_a_host_mask(self):
        from sglang.multimodal_gen.runtime.models.encoders.qwen_vl import (
            QwenVLModelBase,
        )

        mask = (torch.arange(12) < 7).long()[None]
        want = QwenVLModelBase._get_text_rope_index(None, None, mask)
        seen = []
        stock = nomas._STOCK_TEXT_ROPE
        with (
            _installed(),
            patch.object(
                nomas,
                "_STOCK_TEXT_ROPE",
                lambda *a: seen.append(a[2].device.type) or stock(*a),
            ),
        ):
            got = QwenVLModelBase._get_text_rope_index(None, None, mask)
        self.assertEqual(seen, ["cpu"])
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_action_metadata_reads_a_host_mask(self):
        seen = []
        with (
            _installed(),
            patch.object(
                nomas,
                "_STOCK_ACTION_METADATA",
                lambda *args: seen.append(args[3].device.type),
            ),
        ):
            stages.action_metadata("{}", None, None, torch.ones(1, 4).bool(), 1, 24)
        self.assertEqual(seen, ["cpu"])


class TestAutocastPolicy(CustomTestCase):
    def test_norms_and_pooling_run_in_float32_under_autocast(self):
        model = _tiny_model().to(torch.bfloat16)
        h = torch.randn(1, 24, 256).to(torch.bfloat16)
        cond = _condition(6)
        x, mask = cond.text_states.to(torch.bfloat16), cond.text_mask
        pooled = []
        model.txt_in.c_embedder.register_forward_pre_hook(
            lambda m, a: pooled.append(a[0])
        )
        with _installed(), torch.inference_mode():
            latent = torch.zeros(1, 32, 2, 3, 4, dtype=torch.bfloat16)
            model.prepare_segment_context(latent, mask, model.allocate_cache(64))
            norm = model.double_blocks[0].img_norm1
            self.assertIs(type(norm), nomas.Fp32AutocastLayerNorm)
            torch.testing.assert_close(norm(h), F.layer_norm(h, (256,), eps=1e-6))
            with torch.autocast("cpu", dtype=torch.bfloat16):
                want = F.layer_norm(h.float(), (256,), eps=1e-6)
                torch.testing.assert_close(norm(h), want, atol=0, rtol=0)
                model.txt_in(x, torch.tensor([500.0]), mask)
        w = mask.to(x.dtype)[..., None]
        want = (x * w).float().sum(1) / w.float().sum(1)
        torch.testing.assert_close(pooled[0], want, atol=0, rtol=0)


class TestSessions(CustomTestCase):
    def test_eager_session_matches_the_stock_plan(self):
        model = _tiny_model()
        want = _session(model, CONDITIONS)
        with _installed():
            got = _session(model, CONDITIONS)
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_compiled_regions_match_eager_and_compile_once(self):
        model = _tiny_model()
        with _installed():
            want = _session(model, CONDITIONS)
        torch._dynamo.reset()
        counters.clear()
        graphs = []
        with _installed(compiled=True):
            session = NomasLatentSession(model, cache_capacity=64, seed=7, steps=2)
            got = []
            with torch.inference_mode():
                for c in CONDITIONS:
                    got.append(
                        session.generate_segment(
                            (2, 3, 4), c(), cache_segment=True, evict=True
                        )
                    )
                    graphs.append(counters["stats"]["unique_graphs"])
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        # Block halves and the text refiner, then the camera module; a growing
        # and evicted history compiles nothing.
        self.assertEqual(graphs, [3, 3, 4, 4, 4])
        self.assertEqual(dict(counters["graph_break"]), {})


if __name__ == "__main__":
    unittest.main()
