"""Wan VAE feature-cached decode: the compiled decoder calls decode the same
video as one eager call per latent frame."""

import collections
import os
import queue
import socket
import threading
import traceback
import unittest
from unittest.mock import patch

import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
from torch._dynamo.utils import counters

from sglang.multimodal_gen.configs.models.vaes.wanvae import (
    WanVAEArchConfig,
    WanVAEConfig,
)
from sglang.multimodal_gen.runtime.layers import parallel_conv
from sglang.multimodal_gen.runtime.models.vaes import wanvae

_COMPILE_ENV = "SGLANG_DIFFUSION_ENABLE_WAN_VAE_DECODER_COMPILE"
_ALL_CALLS_ENV = "SGLANG_DIFFUSION_WAN_VAE_DECODER_COMPILE_ALL_CALLS"
_CHUNK_ENV = "SGLANG_DIFFUSION_WAN_VAE_DECODE_CHUNK_FRAMES"


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

    def test_all_calls_compile_the_first_two_as_graphs_of_their_own(self):
        compile_kwargs = {"backend": "eager", "fullgraph": True, "dynamic": False}
        with patch.object(
            wanvae,
            "resolve_torch_compile_kwargs",
            return_value=(compile_kwargs, None),
        ):
            for is_residual, chunk, frames in ((True, 1, 6), (False, 5, 11)):
                with self.subTest(is_residual=is_residual, chunk=chunk):
                    vae = _tiny_vae(is_residual)
                    torch._dynamo.reset()
                    counters.clear()
                    z = torch.randn(1, 4, frames, 4, 6)
                    env = {_CHUNK_ENV: str(chunk), _ALL_CALLS_ENV: "1"}
                    with patch.dict(os.environ, env):
                        expected = _decode(vae, z)
                        # Without the compile flag nothing compiles.
                        self.assertEqual(vae._compiled_decoder, {})
                        for _ in range(2):
                            actual = _decode(vae, z, compile_decoder=True)
                            self.assertTrue(torch.equal(actual, expected))
                    # Call 0 (empty caches), call 1 (caches filling), the rest.
                    self.assertEqual(counters["stats"]["unique_graphs"], 3)

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


def _sharded_vae(is_residual: bool, world_size: int) -> wanvae.AutoencoderKLWan:
    """_tiny_vae's model built for a spatially sharded decode over the decode
    group of ``world_size`` ranks, whatever the latent size."""
    config = _tiny_vae(is_residual).config
    config.parallel_decode_mode = "spatial_shard"
    torch.manual_seed(0)
    with patch.object(wanvae.dist, "is_initialized", return_value=True):
        return wanvae.AutoencoderKLWan(config).eval()


def _expected_collectives(decoder) -> int:
    """All-gathers in one sharded decoder call: one per conv with a height
    halo (no p2p), the attention block's height gather and the output's."""
    halo_convs = sum(
        isinstance(
            m,
            (
                parallel_conv.SpatialParallelCausalConv3d,
                parallel_conv.SpatialParallelConv2d,
            ),
        )
        and m.height_halo_size > 0
        for m in decoder.modules()
    )
    attentions = sum(
        isinstance(m, wanvae.WanAttentionBlock) and m.world_size > 1
        for m in decoder.modules()
    )
    return halo_convs + attentions + 1


def _collectives(graph_module) -> collections.Counter:
    counts = collections.Counter()
    for node in graph_module.graph.nodes:
        packet = getattr(node.target, "overloadpacket", None)
        name = getattr(packet, "_qualified_op_name", "")
        if name.startswith("_c10d_functional::") and not name.endswith("wait_tensor"):
            counts[name.split("::", 1)[1]] += 1
    return counts


# (is_residual, latent frames, latent rows, latent frames per call, all calls
# compiled): the rows split unevenly over 2 and 3 ranks, and the calls after
# the second are as long as the chunk.
_SHARDED_CASES = (
    (False, 11, 7, 5, False),
    (True, 7, 7, 2, False),
    (False, 11, 7, 5, True),
)


def _sharded_decodes_on_rank(rank: int, world_size: int) -> list[dict]:
    """This rank's eager and compiled decodes of every case over a gloo decode
    group with no p2p (halos by all-gather, as on TPU)."""
    from torch._dynamo.backends.common import aot_autograd
    from torch._functorch.aot_autograd import make_boxed_func

    from sglang.multimodal_gen.runtime.distributed import group_coordinator
    from sglang.multimodal_gen.runtime.models.vaes import common

    group = group_coordinator.GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=rank,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
    )
    aot_graphs = []

    def record_aot_graph(graph_module, example_inputs):
        aot_graphs.append(graph_module)
        return make_boxed_func(graph_module.forward)

    # Through AOT autograd, where torch_tpu's split compiler sees the graph;
    # the compiled kernels are the eager ones, so the output stays bitwise.
    compile_kwargs = {
        "backend": aot_autograd(fw_compiler=record_aot_graph),
        "dynamic": False,
    }
    results = []
    # Plain callables, not mocks: dynamo traces the compiled calls through them.
    with (
        patch.object(
            parallel_conv, "get_decode_parallel_group_coordinator", lambda: group
        ),
        patch.object(parallel_conv, "get_decode_parallel_rank", lambda: rank),
        patch.object(
            parallel_conv, "get_decode_parallel_world_size", lambda: world_size
        ),
        patch.object(wanvae, "get_decode_parallel_rank", lambda: rank),
        patch.object(wanvae, "get_decode_parallel_world_size", lambda: world_size),
        patch.object(common, "_decode_parallel_world_size", lambda: world_size),
        patch.object(
            parallel_conv.current_platform, "supports_distributed_p2p", lambda: False
        ),
        patch.object(group_coordinator, "is_shm_available", lambda *args: False),
        patch.object(
            wanvae,
            "resolve_torch_compile_kwargs",
            return_value=(compile_kwargs, None),
        ),
    ):
        for is_residual, frames, rows, chunk, all_calls in _SHARDED_CASES:
            vae = _sharded_vae(is_residual, world_size)
            z = torch.randn(
                1, 4, frames, rows, 6, generator=torch.Generator().manual_seed(7)
            )
            env = {_CHUNK_ENV: str(chunk), _ALL_CALLS_ENV: str(int(all_calls))}
            torch._dynamo.reset()
            counters.clear()
            aot_graphs.clear()
            with patch.dict(os.environ, env):
                expected = _decode(vae, z)
                actual = [_decode(vae, z, compile_decoder=True) for _ in range(2)]
            results.append(
                {
                    "equal": [torch.equal(a, expected) for a in actual],
                    "fell_back": vae._compiled_decoder["module"] is None,
                    "unique_graphs": counters["stats"]["unique_graphs"],
                    "collectives": [dict(_collectives(g)) for g in aot_graphs],
                    "expected_collectives": _expected_collectives(vae.decoder),
                    "pinned_after": parallel_conv._pinned_decode,
                }
            )
    return results


def _sharded_decode_worker(rank, world_size, port, results):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        results.put((rank, _sharded_decodes_on_rank(rank, world_size)))
    except BaseException:
        results.put((rank, traceback.format_exc()))
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestWanVaeShardedDecodeCalls(unittest.TestCase):
    """The compiled calls of a spatially sharded decode, one process per rank."""

    def _run_ranks(self, world_size: int) -> dict:
        ctx = mp.get_context("spawn")
        results = ctx.Queue()
        port = _free_port()
        procs = [
            ctx.Process(
                target=_sharded_decode_worker, args=(rank, world_size, port, results)
            )
            for rank in range(world_size)
        ]
        for proc in procs:
            proc.start()
        by_rank = {}
        try:
            for _ in range(world_size):
                rank, result = results.get(timeout=600)
                by_rank[rank] = result
        except queue.Empty:
            self.fail(f"ranks {sorted(set(range(world_size)) - set(by_rank))} hung")
        finally:
            for proc in procs:
                proc.join(timeout=60)
                if proc.is_alive():
                    proc.kill()
        for rank, result in by_rank.items():
            self.assertIsInstance(result, list, f"rank {rank}:\n{result}")
        return by_rank

    def test_decoder_compiles_with_fullgraph(self):
        # A call that does not trace whole then fails before it runs a
        # collective, so its eager rerun stays in step with the other ranks.
        seen = []

        def recording_compile(module, **kwargs):
            seen.append(kwargs)
            return module

        with (
            patch.object(
                wanvae,
                "resolve_torch_compile_kwargs",
                return_value=({"fullgraph": False, "dynamic": None}, None),
            ),
            patch.object(wanvae.torch, "compile", recording_compile),
        ):
            _decode(_tiny_vae(False), torch.randn(1, 4, 4, 4, 6), compile_decoder=True)
        self.assertEqual(seen, [{"fullgraph": True, "dynamic": None}])

    def test_sharded_compiled_calls_trace_one_graph_and_match_eager(self):
        for world_size in (2, 3):
            for rank, results in self._run_ranks(world_size).items():
                for case, result in zip(_SHARDED_CASES, results, strict=True):
                    with self.subTest(world_size=world_size, rank=rank, case=case):
                        all_calls = case[-1]
                        self.assertFalse(result["fell_back"])
                        self.assertEqual(result["equal"], [True, True])
                        # fullgraph: one graph for the calls with full caches,
                        # and one each for the first two calls when they compile.
                        graphs = 3 if all_calls else 1
                        self.assertEqual(result["unique_graphs"], graphs)
                        # The halo and output all-gathers, and no height
                        # exchange: torch_tpu cuts a partition per collective.
                        self.assertEqual(
                            result["collectives"],
                            [{"all_gather_into_tensor": result["expected_collectives"]}]
                            * graphs,
                        )
                        self.assertIsNone(result["pinned_after"])


class TestPinnedSpatialParallelDecode(unittest.TestCase):
    def test_tensor_split_sizes_match_tensor_split(self):
        for length in range(1, 20):
            for parts in range(1, 9):
                with self.subTest(length=length, parts=parts):
                    self.assertEqual(
                        parallel_conv.tensor_split_sizes(length, parts),
                        tuple(
                            len(t)
                            for t in torch.tensor_split(torch.arange(length), parts)
                        ),
                    )

    def test_pinned_heights_scale_the_split_without_an_exchange(self):
        x = torch.empty(1, 2, 1, 24, 3)
        with (
            patch.object(parallel_conv, "get_decode_parallel_world_size", lambda: 3),
            patch.object(parallel_conv, "get_decode_parallel_rank", lambda: 1),
            patch.object(parallel_conv.dist, "all_gather") as all_gather,
        ):
            with parallel_conv.pin_spatial_parallel_decode((4, 3, 3)):
                self.assertEqual(parallel_conv.gather_height_sizes(x), [32, 24, 24])
                with self.assertRaisesRegex(ValueError, "not a multiple"):
                    parallel_conv.gather_height_sizes(x[..., :20, :])
                with parallel_conv.disable_spatial_parallel_decode():
                    # Pinned: the value at entry, not the context's.
                    self.assertFalse(parallel_conv.spatial_parallel_decode_disabled())
            all_gather.assert_not_called()
            self.assertIsNone(parallel_conv._pinned_decode)

    def test_pin_holds_the_entry_value_then_releases(self):
        with parallel_conv.disable_spatial_parallel_decode():
            with parallel_conv.pin_spatial_parallel_decode(None):
                self.assertTrue(parallel_conv.spatial_parallel_decode_disabled())
                self.assertEqual(parallel_conv._pinned_decode, (True, None))
            self.assertIsNone(parallel_conv._pinned_decode)
        self.assertFalse(parallel_conv.spatial_parallel_decode_disabled())
        with self.assertRaises(RuntimeError):
            with parallel_conv.pin_spatial_parallel_decode((1,)):
                raise RuntimeError
        self.assertIsNone(parallel_conv._pinned_decode)


if __name__ == "__main__":
    unittest.main()
