# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's SANA-WM patches (sglang_tpu.multimodal_gen.sana_wm) and
the host-side schedule reads of the SANA-WM streaming stages."""

import contextlib
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch._dynamo
import torch.fx
import torch.utils._pytree as pytree
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.utils import counters
from torch.overrides import TorchFunctionMode

from sglang.multimodal_gen.configs.models.dits.sana_wm import (
    SanaWMArchConfig,
    SanaWMConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.models.dits import sana_wm_components as components
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    SanaWMBlock,
    SanaWMTransformer3DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    base,
    streaming,
    streaming_refiner,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.self_forcing import (
    SanaWMSelfForcingSampler,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import sana_wm

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=120, suite="nightly-tpu-4-v7", nightly=True)

_ROPE = components.WanRotaryPosEmbed
_CPU = torch.device("cpu")
_PREVIOUS_SERVER_ARGS = None


def setUpModule():
    # The DiT blocks build their cross-attention backend from the server args.
    global _PREVIOUS_SERVER_ARGS
    _PREVIOUS_SERVER_ARGS = server_args_module._global_server_args
    server_args_module.set_global_server_args(_make_unit_server_args())


def tearDownModule():
    server_args_module.set_global_server_args(_PREVIOUS_SERVER_ARGS)


@contextlib.contextmanager
def _installed(device_types=("tpu",)):
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (_ROPE, "_init_freqs_buffer"),
            (_ROPE, "forward"),
            (components, "_apply_rotary_emb_dn"),
            (components, "_apply_rotary_emb_bhnd"),
            (components, "_apply_complex_rope"),
            (components, "_sana_wm_reverse_scan"),
            (components.BidirectionalGDNUCPESinglePathLiteLA, "_get_cam_qkv_params"),
            (SanaWMBlock, "forward"),
            (SanaWMBlock, "forward_long"),
            *sana_wm.NOISE_METHODS,
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(sana_wm, "_DEVICE_TYPES", device_types))
        stack.enter_context(patch.object(sana_wm, "_EAGER_PATHS", set()))
        sana_wm.install()
        sana_wm.install()  # idempotent
        yield


def _tables(fhw, frame_index=None, head_dim=112):
    """One grid's per-token tables: stock (complex) and installed (real)."""
    want = _ROPE(head_dim, (1, 1, 1))(fhw, _CPU, frame_index)
    with _installed():
        got = _ROPE(head_dim, (1, 1, 1))(fhw, _CPU, frame_index)
    return want, got


class _HostReads(TorchFunctionMode):
    """Counts tensor reads into Python values (device syncs on TPU)."""

    READS = {"item", "tolist", "__float__", "__int__", "__bool__"}

    def __init__(self):
        super().__init__()
        self.count = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.count += getattr(func, "__name__", "") in self.READS
        return func(*args, **(kwargs or {}))


class _ZeroCross(torch.nn.Module):
    def forward(self, x, y, mask=None):
        return torch.zeros_like(x)


def _tiny_dit(num_layers, **arch_overrides):
    arch = SanaWMArchConfig(
        in_channels=8,
        out_channels=8,
        num_layers=num_layers,  # block 3 is a softmax block
        num_attention_heads=2,
        attention_head_dim=16,
        linear_head_dim=16,
        num_cross_attention_heads=2,
        cross_attention_head_dim=16,
        cross_attention_dim=32,
        caption_channels=32,
        model_max_length=8,
        chunk_size=None,
        **arch_overrides,
    )
    dit = SanaWMTransformer3DModel(SanaWMConfig(arch_config=arch)).eval()
    for block in dit.blocks:
        block.cross_attn = _ZeroCross()
    return dit


def _dit_inputs(frames, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return dict(
        hidden_states=torch.randn(1, 8, frames, 2, 3, generator=gen),
        encoder_hidden_states=torch.randn(1, 4, 32, generator=gen),
        camera_conditions=torch.randn(1, frames, 20, generator=gen),
        chunk_plucker=torch.randn(1, 48, frames, 2, 3, generator=gen),
    )


class TestSanaWMRopeTables(CustomTestCase):
    def test_real_tables_are_the_complex_ones_on_the_720p_grid(self):
        # 704x1280 is a 22x40 latent grid; windows and frame indices as streaming
        # chunks ask for them.
        for fhw, frame_index in (
            ((7, 22, 40), None),
            (((4, 7), 22, 40), None),
            ((3, 22, 40), torch.arange(38, 41)),
        ):
            want, got = _tables(fhw, frame_index)
            self.assertEqual(got.dtype, torch.float32)
            self.assertTrue(
                torch.equal(got, torch.view_as_real(want).float().movedim(-1, 0))
            )

    def test_table_is_a_float32_attribute_rebuilt_after_a_meta_init(self):
        with _installed():
            rope = _ROPE(112, (1, 1, 1)).to(torch.bfloat16)
            self.assertEqual(rope._freqs.dtype, torch.float32)
            self.assertNotIn("_freqs", rope.state_dict())
            with torch.device("meta"):
                meta_rope = _ROPE(112, (1, 1, 1))
            self.assertTrue(meta_rope._freqs.is_meta)
            dit = SimpleNamespace(modules=lambda: [meta_rope])
            SanaWMTransformer3DModel.post_load_weights(dit)
            self.assertTrue(torch.equal(meta_rope._freqs, rope._freqs))


class TestSanaWMRotations(CustomTestCase):
    def test_rotations_round_once_from_float32(self):
        complex_t, real_t = _tables((2, 22, 40))
        cam = [components._slice_rope_for_cam(t, 112, 56) for t in (complex_t, real_t)]
        gen = torch.Generator().manual_seed(0)
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            x = (torch.randn(1, 2, complex_t.shape[-2], 112, generator=gen) * 3).to(
                dtype
            )
            cases = {
                "bhnd": (sana_wm._STOCK_ROTARY_EMB_BHND, "_apply_rotary_emb_bhnd", x),
                "dn": (sana_wm._STOCK_ROTARY_EMB_DN, "_apply_rotary_emb_dn", x.mT),
            }
            for name, (stock, rebound, inp) in cases.items():
                want = stock(inp, complex_t)
                with _installed():
                    got = getattr(components, rebound)(inp, real_t)
                self._assert_rounded_once(got, want, dtype, name)
            for inverse in (False, True):
                want = sana_wm._STOCK_COMPLEX_ROPE(x[..., 56:], cam[0], inverse)
                with _installed():
                    got = components._apply_complex_rope(x[..., 56:], cam[1], inverse)
                self._assert_rounded_once(got, want, dtype, f"camera {inverse=}")

    def _assert_rounded_once(self, got, want, dtype, name):
        with self.subTest(name=name, dtype=dtype):
            self.assertEqual((got.dtype, got.shape), (want.dtype, want.shape))
            if dtype == torch.float32:
                torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
            else:
                # One rounding from float32 against the stock's from float64.
                torch.testing.assert_close(
                    got.float(), want.float(), atol=2e-2, rtol=1e-2
                )
                self.assertLess((got != want).float().mean().item(), 1e-3)

    def test_complex_tables_keep_the_stock_path(self):
        complex_t, _ = _tables((1, 2, 3), head_dim=16)
        x = torch.randn(1, 2, 6, 16)
        with _installed():
            got = components._apply_rotary_emb_bhnd(x, complex_t)
        self.assertTrue(torch.equal(got, sana_wm._STOCK_ROTARY_EMB_BHND(x, complex_t)))

    def test_streaming_chunks_match_the_complex_rope(self):
        # Three chunks of forward_long: windowed tables, the camera branch, the
        # softmax concat-window and (chunks 1 and 2) the sink path's frame index.
        def stream(dit):
            inputs, outs = _dit_inputs(7), []
            segments = [0, 3, 5, 7]
            kv = [[[None] * 10 for _ in dit.blocks] for _ in range(3)]
            with torch.no_grad():
                for i in range(3):
                    chunk_kv, sink = SanaWMSelfForcingSampler.accumulate_kv_cache(
                        kv, i, segments, 1, True, len(dit.blocks)
                    )
                    start, end = segments[i], segments[i + 1]
                    kwargs = dict(
                        inputs,
                        hidden_states=inputs["hidden_states"][:, :, start:end],
                        kv_cache=chunk_kv,
                        start_f=start,
                        end_f=end,
                        frame_index=torch.arange(start, end) if sink else None,
                    )
                    out, _ = dit.forward_long(
                        **kwargs, timestep=torch.full((1, end - start), 700.0)
                    )
                    _, kv[i] = dit.forward_long(
                        **kwargs,
                        timestep=torch.zeros(1, end - start),
                        save_kv_cache=True,
                    )
                    outs.append(out)
            return torch.cat(outs, dim=2)

        torch.manual_seed(0)
        stock_dit = _tiny_dit(4)
        want = stream(stock_dit)
        with _installed():
            dit = _tiny_dit(4)
            dit.load_state_dict(stock_dit.state_dict())
            got = stream(dit)
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)


class TestSanaWMCompiledBlocks(CustomTestCase):
    def test_compiled_blocks_see_one_layout_per_input(self):
        # torch_tpu guards every compiled input's storage offset and dynamo its
        # strides: each graph must see one layout across steps and requests.
        layouts = {}

        def backend(graph_module, example_inputs):
            seen = layouts.setdefault(len(layouts), set())

            def run(*args):
                seen.add(
                    tuple(
                        (a.shape, a.stride(), a.storage_offset())
                        for a in args
                        if isinstance(a, torch.Tensor)
                    )
                )
                return graph_module.forward(*args)

            return run

        with _installed():
            torch.manual_seed(0)
            dit = _tiny_dit(2)
            for block in dit.blocks:
                block.compile(backend=backend, dynamic=False)
            graphs = []
            with torch.no_grad():
                for seed in (0, 1):
                    for level in (900.0, 500.0):
                        timestep = torch.full((1, 1, 5), level)
                        timestep[..., 0] = 0.0
                        dit(timestep=timestep, **_dit_inputs(5, seed))
                    graphs.append(len(layouts))
        self.assertEqual(graphs[0], graphs[1])
        self.assertTrue(all(len(seen) == 1 for seen in layouts.values()), layouts)


_VIEW_OPS = frozenset(
    "view _unsafe_view permute transpose t expand slice select unsqueeze squeeze "
    "alias as_strided split split_with_sizes unbind narrow detach unflatten "
    "flatten getitem".split()
)


class _RowMajorOutputs(torch.fx.Interpreter):
    """torch_tpu exports a graph by running its ATen ops: compute ops return
    row-major tensors (a flip of a permuted view too), so a view the fake trace
    found legal can fail (wm_pr_bidi, j-a59c2834). Attention keeps its layout,
    as TPU-compiled Wan blocks transpose and flatten SDPA outputs."""

    def run_node(self, n):
        out = super().run_node(n)
        name = getattr(n.target, "__name__", "").split(".")[0]
        if n.op != "call_function" or name in _VIEW_OPS or "dot_product" in name:
            return out
        return pytree.tree_map_only(torch.Tensor, torch.Tensor.contiguous, out)


def _tpu_like_backend(graphs):
    def compile_graph(gm, example_inputs):
        graphs.append(gm)
        return lambda *args: _RowMajorOutputs(gm).run(*args)

    return aot_autograd(fw_compiler=compile_graph)


def _random_dit():
    # Two GDN blocks and a softmax block; random weights reach every branch.
    torch.manual_seed(0)
    dit = _tiny_dit(3, softmax_every_n=3)
    with torch.no_grad():
        for param in dit.parameters():
            param.copy_(torch.randn_like(param) * 0.05)
    return dit


def _stream(dit, segments=(0, 4, 7, 10)):
    # Chunk 0, chunk 1 over its cache, chunk 2 over the sink and the window.
    inputs, outs = _dit_inputs(segments[-1]), []
    kv = [[[None] * 10 for _ in dit.blocks] for _ in segments[1:]]
    with torch.no_grad():
        for i, (start, end) in enumerate(zip(segments, segments[1:])):
            chunk_kv, sink = SanaWMSelfForcingSampler.accumulate_kv_cache(
                kv, i, list(segments), 2, True, len(dit.blocks)
            )
            kwargs = dict(
                inputs,
                hidden_states=inputs["hidden_states"][:, :, start:end],
                kv_cache=chunk_kv,
                start_f=start,
                end_f=end,
                frame_index=torch.arange(start, end) if sink else None,
            )
            timestep = torch.full((1, end - start), 900.0)
            outs.append(dit.forward_long(**kwargs, timestep=timestep)[0])
            zeros = torch.zeros(1, end - start)
            _, kv[i] = dit.forward_long(**kwargs, timestep=zeros, save_kv_cache=True)
    return torch.cat(outs, dim=2)


class TestSanaWMCompiledStreaming(CustomTestCase):
    def setUp(self):
        torch._dynamo.reset()
        counters.clear()

    def test_streaming_blocks_run_compiled_and_match_stock(self):
        want = _stream(_random_dit())  # stock: complex RoPE, flipped scans, eager
        with _installed(("cpu",)):
            dit, graphs = _random_dit(), []
            backend = _tpu_like_backend(graphs)
            for block in dit.blocks:
                block.compile(backend=backend, dynamic=False)
            got, again = _stream(dit), _stream(dit)
            self.assertEqual(sana_wm._EAGER_PATHS, set())  # no eager fallback
        torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-4)
        self.assertTrue(torch.equal(again, got))
        # Per block kind: chunk 0, chunk 1 and the full window, step and save;
        # the repeat compiles nothing.
        self.assertEqual(len(graphs), 8)
        self.assertEqual(dict(counters["graph_break"]), {})
        # The cache type flags stay host tensors outside the graphs.
        targets = {str(n.target) for g in graphs for n in g.graph.nodes}
        self.assertFalse([t for t in targets if "lift_fresh" in t])

    def test_a_failing_compiled_block_runs_eagerly(self):
        calls = []

        def failing_backend(gm, example_inputs):
            calls.append(gm)
            raise RuntimeError("view(): cannot create a view")

        with _installed(("cpu",)):
            want = _stream(_random_dit())
            dit = _random_dit()
            for block in dit.blocks:
                block.compile(backend=failing_backend, dynamic=False)
            with self.assertLogs(sana_wm.logger.name, "WARNING"):
                got = _stream(dit)
            self.assertEqual(sana_wm._EAGER_PATHS, {"stage-1 block"})
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(got, want))


class TestSanaWMNoise(CustomTestCase):
    SHAPE = (1, 4, 3, 2, 2)

    def _draw(self, dtype, generator, shape=SHAPE):
        stage = base.SanaWMBeforeDenoisingStage.__new__(base.SanaWMBeforeDenoisingStage)
        return stage._prepare_noise_latents(shape, dtype, _CPU, generator)

    def test_device_draws_come_from_float32_samples(self):
        stock_randn = torch.randn
        with _installed(device_types=("cpu",)):
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                got = self._draw(dtype, torch.Generator().manual_seed(3))
                gen = torch.Generator().manual_seed(3)
                self.assertTrue(
                    torch.equal(got, torch.randn(self.SHAPE, generator=gen).to(dtype))
                )
            gens = [torch.Generator().manual_seed(s) for s in (1, 2)]
            got = self._draw(torch.bfloat16, gens, (2, *self.SHAPE[1:]))
            want = [
                torch.randn(self.SHAPE, generator=torch.Generator().manual_seed(s))
                for s in (1, 2)
            ]
            self.assertTrue(torch.equal(got, torch.cat(want).bfloat16()))
        self.assertIs(torch.randn, stock_randn)

    def test_host_draws_and_raising_methods_keep_the_stock_randn(self):
        stock_randn = torch.randn
        with _installed():
            got = self._draw(torch.bfloat16, torch.Generator().manual_seed(3))
            gen = torch.Generator().manual_seed(3)
            self.assertTrue(
                torch.equal(
                    got, stock_randn(self.SHAPE, generator=gen, dtype=torch.bfloat16)
                )
            )
            for cls, name in sana_wm.NOISE_METHODS:
                self.assertTrue(getattr(cls, name).__wrapped_for_tpu__, name)

            def fails():
                raise RuntimeError("draw failed")

            with self.assertRaises(RuntimeError):
                sana_wm._draws_in_float32(fails)()
            self.assertIs(torch.randn, stock_randn)


class _DeviceLookupScheduler(streaming.FlowMatchEulerDiscreteScheduler):
    """Main's chunk loop: the first step() looks its index up on the device."""

    def set_begin_index(self, begin_index=0):
        pass


class TestSanaWMStreamingHostReads(CustomTestCase):
    def _denoise_chunk(self, scheduler):
        latents = {
            "lat": torch.randn(
                1, 4, 3, 2, 2, generator=torch.Generator().manual_seed(0)
            )
        }
        cond_mask = torch.zeros(1, 4, 3, 2, 2)
        cond_mask[:, :, 0] = 1.0
        seen = []

        def forward_long(hidden_states, timestep, **kwargs):
            return hidden_states * 0.5 + timestep.mean() * 1e-3, ["cache"]

        with _HostReads() as reads:
            cache = streaming.self_forcing_denoise_chunk(
                transformer=SimpleNamespace(forward_long=forward_long),
                scheduler=scheduler,
                sigmas=SanaWMSelfForcingSampler.build_per_chunk_sigmas(
                    (1000, 960, 889, 727, 0)
                ),
                get_chunk=lambda: latents["lat"],
                set_chunk=lambda lat: latents.update(lat=lat.float()),
                cond_mask=cond_mask,
                embeds=torch.zeros(1, 4, 32),
                mask=None,
                camera_conditions=None,
                chunk_plucker=None,
                chunk_kv=[],
                start_f=0,
                end_f=3,
                frame_index=None,
                do_cfg=False,
                cfg_scale=1.0,
                target_dtype=torch.float32,
                device=_CPU,
                forward_ctx=lambda ts: seen.append(ts) or contextlib.nullcontext(),
                on_noise_pred=lambda ts, pred: seen.append(("pred", ts)),
            )
        self.assertEqual(cache, ["cache"])
        return latents["lat"], seen, reads.count

    def test_a_chunk_reads_its_schedule_once_and_steps_as_the_device_lookup(self):
        got, seen, reads = self._denoise_chunk(
            streaming.FlowMatchEulerDiscreteScheduler(shift=1.0)
        )
        stock = _DeviceLookupScheduler(shift=1.0)
        want, want_seen, _ = self._denoise_chunk(stock)
        self.assertTrue(torch.equal(got, want))
        self.assertEqual(reads, 1)
        timesteps = [int(t.item()) for t in stock.timesteps]
        self.assertEqual(timesteps, [1000, 960, 889, 727])
        self.assertEqual(seen, want_seen)
        self.assertEqual(seen, [x for t in timesteps for x in (t, ("pred", t))] + [0])

    def test_the_refiner_runner_steps_on_host_sigmas(self):
        sigmas = torch.tensor(streaming_refiner.STAGE_2_DISTILLED_SIGMA_VALUES)
        core = SimpleNamespace(
            device=_CPU,
            dtype=torch.bfloat16,
            transformer=SimpleNamespace(
                config=SimpleNamespace(patch_size=1, patch_size_t=1),
                transformer_blocks=[None],
            ),
            _capture_block_kv=lambda clean_block, **kw: [
                (clean_block.flatten(2).mT,) * 2
            ],
            _predict_x0_active_block=lambda active, sigma_cur, **kw: (
                active.float() * sigma_cur
            ),
        )

        def refine(blocks):
            runner = streaming_refiner.RefinerChunkRunner(
                core,
                prompt_embeds=None,
                prompt_attention_mask=None,
                fps=16,
                sigmas=sigmas,
                source_sink_frames=0,
                block_size=3,
                kv_max_frames=11,
                seed=42,
                spatial_shape=(2, 2),
            )
            # Main read float(self.sigmas[i].item()) at every level of every block.
            self.assertEqual(runner.sigma_values, [float(s.item()) for s in sigmas])
            with _HostReads() as reads:
                out = [
                    runner.refine_block(
                        block_idx=i,
                        clean_block=b,
                        block_start=3 * i,
                        block_end=3 * i + 3,
                    )
                    for i, b in enumerate(blocks)
                ]
            self.assertEqual(reads.count, 0)
            return out

        blocks = torch.randn(2, 1, 4, 3, 2, 2).bfloat16()
        first, second = refine(blocks), refine(blocks)
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a, b))
            self.assertTrue(torch.isfinite(a.float()).all())

    def test_cache_type_flags_stay_on_the_host(self):
        # Activations on another device (meta here, TPU in serving): the chunk
        # accumulator reads each block's flag without a device read.
        with torch.device("meta"):
            attn = components.BidirectionalGDNUCPESinglePathLiteLA(
                in_dim=32, heads=2, head_dim=16
            )
            x = torch.empty(1, 6, 32)
            freqs = _ROPE(16, (1, 1, 1))(((0, 1), 2, 3), torch.device("meta"))
        gdn, concat = [None] * 10, [None] * 10
        attn._main_branch_gdn_cached(x, (1, 2, 3), freqs, gdn, True)
        attn._main_branch_softmax_cached(x, (1, 2, 3), freqs, concat, True)
        for cache, flag in ((gdn, 1.0), (concat, 0.0)):
            self.assertEqual(cache[6].device, _CPU)
            self.assertEqual(cache[6].tolist(), [flag])
        kv = [[gdn, concat], [[None] * 10, [None] * 10]]
        cur, _ = SanaWMSelfForcingSampler.accumulate_kv_cache(
            kv, 1, [0, 1, 2], 2, True, 2
        )
        self.assertIs(cur[0][0], gdn[0])
        self.assertEqual(cur[1][0].shape, concat[0].shape)


if __name__ == "__main__":
    import unittest

    unittest.main()
