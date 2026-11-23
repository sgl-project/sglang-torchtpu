# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's FLUX.3 Image Klein patches (sglang_tpu.multimodal_gen.flux3_image)."""

import contextlib
import os
import socket
import tempfile
import unittest
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from transformers import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

from sglang.multimodal_gen.configs.pipeline_configs.flux3_image import (
    flux3_image_dit_config,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import flux3
from sglang.multimodal_gen.runtime.models.dits import flux3_image as dit
from sglang.multimodal_gen.runtime.models.encoders import flux3_text_encoder
from sglang.multimodal_gen.runtime.pipelines import flux3_image as pipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    flux3_image as stages,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import flux3_image, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=60, suite="nightly-tpu-4-v7", nightly=True)

_TINY = dict(
    hidden_size=256,
    num_attention_heads=2,
    depth=1,
    depth_single_blocks=2,
    context_in_dim=64,
)
_CHANNELS = {"image": 128, "image_cond": 128, dit.TEXT: 64}
_TOL = dict(rtol=1e-5, atol=1e-5)
# Text tokens per sample: one sample (18 joint rows, even under SP 2), one odd
# (17 rows: the SP 2 tail is padding), and a CFG pair of two lengths.
_CASES = [(4,), (3,), (3, 5)]


@contextlib.contextmanager
def _installed(region=None):
    """install() on CPU tensors (streams of 4+ tokens shard under SP); patches
    restored after. ``region`` replaces the compiled block program."""
    with contextlib.ExitStack() as stack:
        for owner, name, _ in flux3_image.patches():
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(flux3_image, "_DEVICE_TYPES", ("tpu", "cpu")))
        stack.enter_context(patch.object(dit, "SP_MIN_STREAM_TOKENS", 4))
        if region is not None:
            stack.enter_context(patch.object(flux3_image, "_region", lambda: region))
        flux3_image._REGION.clear()
        flux3_image.install()
        yield


def _init_parallel_state(sp=1):
    set_global_server_args(_make_unit_server_args())
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        with patch.object(worker, "install_runtime_patches"):
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=1, sp_size=sp, ulysses_degree=sp
            )


def _model(cls=dit.Flux3ImageDiT):
    torch.manual_seed(0)
    model = cls(flux3_image_dit_config(**_TINY)).float()
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.05)
    return model.eval()


def _samples(text_lens):
    gen = torch.Generator().manual_seed(sum(text_lens))

    def tokens(count, width):
        return torch.randn(count, width, generator=gen), torch.rand(
            count, 4, generator=gen
        ) * 8

    return [(tokens(8, 128), tokens(6, 128), tokens(n, 64)) for n in text_lens]


def _run(model, text_lens):
    packed = dit.pack(
        [{"image": a, "image_cond": b, dit.TEXT: c} for a, b, c in _samples(text_lens)],
        _CHANNELS,
    )
    packed.timesteps["image"].fill_(0.7)
    with torch.no_grad(), set_forward_context(current_timestep=0, attn_metadata=None):
        return model(packed)["image"]


def _sp_worker(rank, port, out_dir):
    """One Ulysses rank of an SP 2 run over gloo; rank 0 saves the outputs."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
    )
    _init_parallel_state(sp=2)
    model, layouts, kept = _model(), [], []

    def region(self, *args):
        tensors = [a for a in args if isinstance(a, torch.Tensor)]
        layouts.append([(t.shape, t.stride(), t.storage_offset()) for t in tensors])
        kept.append(tensors[1:5])  # alive, so distinct tensors keep distinct storage
        return flux3_image._block(self, *args)

    with _installed(region):
        for index, text_lens in enumerate(_CASES):
            out = _run(model, text_lens)
            if rank == 0:
                torch.save(out, os.path.join(out_dir, f"{index}.pt"))
    torch.save(layouts, os.path.join(out_dir, f"layouts.{rank}.pt"))
    pointers = [[t.data_ptr() for t in call] for call in kept]
    torch.save(pointers, os.path.join(out_dir, f"pointers.{rank}.pt"))


class _Tokenizer:
    def __call__(self, text, max_length, padding=None, **kwargs):
        if padding != "max_length":
            return {"input_ids": [0] * (5 + len(text) % 3)}
        ids = [[3 + ord(c) % 90 for c in t][:max_length] for t in text]
        mask = torch.tensor([[1] * len(r) + [0] * (max_length - len(r)) for r in ids])
        ids = torch.tensor([r + [0] * (max_length - len(r)) for r in ids])
        return {"input_ids": ids, "attention_mask": mask}


def _text_encoder(model, output_layers):
    cls = flux3_text_encoder.Flux3TextEncoder
    encoder = cls.__new__(cls)
    torch.nn.Module.__init__(encoder)
    encoder.processor = SimpleNamespace(
        tokenizer=_Tokenizer(), apply_chat_template=lambda m, **_: m[0]["content"]
    )
    encoder.model, encoder.output_layers = model, output_layers
    encoder.pad_multiple, encoder.max_length = 8, 64
    encoder.embed_dtype = torch.float32
    return encoder


class _TestBase(CustomTestCase):
    def setUp(self):
        self._previous_args = server_args_module._global_server_args
        _init_parallel_state()

    def tearDown(self):
        set_global_server_args(self._previous_args)


class TestFlux3ImageBlocks(_TestBase):
    def test_packed_samples_match_the_reference_transformer(self):
        # Each packed sample (a CFG pair of two text lengths, with references)
        # equals the shared FLUX 3 transformer on that sample alone.
        native, reference = _model(), _model(flux3.Flux3Transformer)
        state = native.state_dict()
        for name, tensor in list(state.items()):
            if ".to_qkv." in name:
                mlp_in = state.pop(name.replace(".to_qkv.", ".mlp_in."))
                state[name.replace(".to_qkv.", ".qkv_mlp.")] = torch.cat(
                    [tensor, mlp_in]
                )
                del state[name]
        state = {k.replace("_norm.weight", "_norm.scale"): v for k, v in state.items()}
        reference.load_state_dict(state, strict=True)
        with _installed():
            got = _run(native, (3, 5))
        want, zero = [], torch.zeros(1)
        with torch.no_grad(), set_forward_context(0, None):
            for image, cond, text in _samples((3, 5)):
                out = reference(
                    ctx=text[0][None],
                    ctx_ids=text[1][None],
                    timesteps_ctx=zero,
                    x_image=image[0][None],
                    x_image_ids=image[1][None],
                    x_image_timesteps=zero + 0.7,
                    x_image_cond=cond[0][None],
                    x_image_cond_ids=cond[1][None],
                    x_image_cond_timesteps=zero,
                )
                want.append(out["x_image"][0])
        torch.testing.assert_close(got, torch.cat(want), **_TOL)

    def test_sequence_parallel_shards_match_one_rank(self):
        model = _model()
        with _installed():
            wants = [_run(model, text_lens) for text_lens in _CASES]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (
            tempfile.TemporaryDirectory() as out_dir,
            patch.dict(os.environ, SGLANG_DIFFUSION_PLATFORM_OVERRIDE="cpu"),
        ):
            mp.spawn(_sp_worker, args=(port, out_dir), nprocs=2)
            for index, want in enumerate(wants):
                got = torch.load(os.path.join(out_dir, f"{index}.pt"))
                torch.testing.assert_close(got, want, **_TOL)
            layouts = [
                torch.load(os.path.join(out_dir, f"layouts.{rank}.pt"))
                for rank in range(2)
            ]
            pointers = torch.load(os.path.join(out_dir, "pointers.1.pt"))
        # torch_tpu bakes an input's strides and storage offset into the block
        # program: every rank feeds it one layout at offset 0.
        self.assertEqual(layouts[0], layouts[1])
        self.assertEqual({o for call in layouts[0] for *_, o in call}, {0})
        # Calls 3-4 are case 0's two joint blocks: rank 1 shards its RoPE rows
        # and modulation once per stream, not once per block.
        self.assertEqual(pointers[3], pointers[4])


class TestFlux3ImageCompile(_TestBase):
    def setUp(self):
        super().setUp()
        # The TPU worker turns automatic dynamic shapes off.
        self.enterContext(torch._dynamo.config.patch(automatic_dynamic_shapes=False))
        self.addCleanup(torch._dynamo.reset)

    def test_blocks_share_one_graph_per_shape(self):
        torch._dynamo.reset()
        graphs = []
        region = torch.compile(
            flux3_image._block,
            backend=lambda gm, _: graphs.append(gm) or gm.forward,
            dynamic=False,
        )
        model = _model()
        with _installed(region):
            want = _run(model, (4,))
            count = len(graphs)
            # A new request of the same shapes: no new graph.
            torch.testing.assert_close(_run(model, (4,)), want, rtol=0, atol=0)
        # image, image_cond, text mode blocks and the joint blocks: four shapes.
        self.assertEqual((count, len(graphs)), (4, 4))


class TestFlux3ImageHostSide(_TestBase):
    def test_output_head_keeps_the_modulated_rows_float32(self):
        torch.manual_seed(0)
        head = flux3.Flux3LastLayer(64, 16).bfloat16()
        x, vec = torch.randn(1, 5, 64).bfloat16(), torch.randn(1, 64).bfloat16()
        act = F.silu(vec[:, None])
        shift_w, scale_w = head.adaLN_modulation[1].weight.chunk(2)
        rows = F.layer_norm(x.float(), (64,), eps=1e-6)
        rows = rows * (F.linear(act, scale_w) + 1).float() + F.linear(act, shift_w)
        with _installed():
            got = head(x, vec)
        want = head.linear(rows.bfloat16())
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_latent_ids_and_rope_tables_build_on_the_host(self):
        # The DiT on another device (meta): the table is built from host ids
        # and moved there.
        model = _model().to("meta")
        with _installed(), patch.object(flux3_image, "_DEVICE_TYPES", ("meta",)):
            ids = stages.scaled_ids(stages.latent_ids(3, 4, device="meta"), 0.5)
            table = model.rope(ids)
        self.assertEqual((ids.device.type, table.device.type), ("cpu", "meta"))
        self.assertEqual(tuple(table.shape), (12, 128))

    def test_vae_decodes_one_image_at_a_time(self):
        decoded = []
        vae = torch.nn.Conv2d(1, 1, 1)
        vae.decode = lambda z: decoded.append(z.shape[0]) or z[:, :3]
        ids = stages.latent_ids(2, 3)
        with _installed():
            out = stages.vae_decode(vae, torch.randn(3 * 6, 16), ids, batched=True)
        self.assertEqual((decoded, tuple(out.shape)), ([1, 1, 1], (3, 3, 2, 3)))

    def test_vae_weights_load_on_the_host_where_safetensors_cannot(self):
        vae_config = SimpleNamespace(dtype="float32")
        package = SimpleNamespace(config=SimpleNamespace(vae=vae_config), vae="ae")
        decoder = pipeline.RoleType.DECODER
        args = SimpleNamespace(
            quantization=None,
            disagg_role=SimpleNamespace(includes_stage=lambda role: role == decoder),
            should_start_component_on_cpu=lambda name: False,
            pipeline_config=SimpleNamespace(flux3_package=package),
        )
        host = {"bn.running_mean": torch.zeros(128)}
        platform = pipeline.current_platform
        with (
            patch.object(platform, "supports_safetensors_device_load", lambda: False),
            patch.object(
                pipeline, "get_local_torch_device", lambda: torch.device("meta")
            ),
            patch.object(pipeline, "load_file", return_value=host) as load_file,
            patch.object(pipeline, "load_model_state_dict") as load_state,
            patch.object(pipeline, "get_memory_usage_of_component", return_value=0),
        ):
            stub = SimpleNamespace(memory_usages={})
            pipeline.Flux3ImagePipeline.load_modules(stub, args)
        self.assertEqual(load_file.call_args.kwargs["device"], "cpu")
        moved = load_state.call_args.args[1]
        self.assertEqual({t.device.type for t in moved.values()}, {"meta"})


class TestFlux3ImageTextEncoder(_TestBase):
    def test_text_positions_come_from_the_host_mask(self):
        # torch_tpu aborts on an int64 cumsum on the device (meta here).
        devices, seen = [], {}
        stock_cumsum = torch.Tensor.cumsum

        def cumsum(self, *args, **kwargs):
            devices.append(self.device.type)
            return stock_cumsum(self, *args, **kwargs)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(1, device="meta"))

            def forward(self, input_ids, attention_mask, position_ids, **kwargs):
                seen["positions"] = position_ids
                hidden = torch.zeros(*input_ids.shape, 2, device="meta")
                return SimpleNamespace(hidden_states=[hidden] * 2)

        with patch.object(torch.Tensor, "cumsum", cumsum):
            _text_encoder(Model(), (0, 1)).encode_batch(["a prompt"])
        self.assertEqual(devices, ["cpu"])
        self.assertEqual(seen["positions"].device.type, "meta")

    def test_decoder_layers_share_one_graph_per_bucket(self):
        torch.manual_seed(0)
        rope = {"rope_type": "default", "mrope_section": [6, 5, 5]}
        config = Qwen3VLConfig(
            text_config=dict(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=32,
                vocab_size=128,
                rope_scaling={**rope, "mrope_interleaved": True},
            ),
            vision_config=dict(
                depth=1,
                hidden_size=32,
                intermediate_size=64,
                num_heads=2,
                out_hidden_size=64,
                num_position_embeddings=16,
                deepstack_visual_indexes=[0],
            ),
        )
        model = Qwen3VLModel(config).eval()
        encoder = _text_encoder(model, (1, 2, 3))
        prompts = [["a fox", "a red fox"], ["a cat", "a red fox"]]
        want = [encoder.encode_batch(p) for p in prompts]
        graphs = []
        region = partial(
            torch.compile,
            backend=lambda gm, _: graphs.append(gm) or gm.forward,
            dynamic=False,
        )
        with (
            _installed(),
            patch.object(flux3_image, "_compile_enabled", lambda: True),
            patch.object(flux3_image, "_compile", lambda fn, spmd=False: region(fn)),
        ):
            got = [encoder.encode_batch(p) for p in prompts]
        # Four layers, two requests of the 8-token bucket: one graph.
        self.assertEqual(len(graphs), 1)
        for g, w in zip(sum(got, []), sum(want, [])):
            torch.testing.assert_close(g, w, rtol=0, atol=0)


class TestFlux3ImageStage(unittest.TestCase):
    def test_grouped_decode_runs_in_inference_mode(self):
        # A spatial-parallel decoder's halo buffers are inference tensors that
        # the single-request decode created.
        seen = []

        def decode(vae, tokens, ids, batched=False):
            seen.append(torch.is_inference_mode_enabled())
            return torch.zeros(len(tokens) // len(ids), 3, 2, 2)

        stage = stages.Flux3ImageDecodeStage.__new__(stages.Flux3ImageDecodeStage)
        stage.vae = torch.nn.Module()
        stage.use_declared_component = lambda **_: contextlib.nullcontext()
        batch = SimpleNamespace(
            width=2,
            height=2,
            quality="lossless",
            latents=torch.zeros(4, 128),
            latent_ids=stages.latent_ids(2, 2),
            num_outputs_per_prompt=1,
            extra={"flux3_crop": None},
            metrics=None,
        )
        with patch.object(stages, "vae_decode", decode):
            outputs = stage.run_grouped_requests([batch, batch], None)
        self.assertEqual((seen, len(outputs)), ([True], 2))

    def test_a_tpu_dit_is_not_compiled_whole(self):
        stage = SimpleNamespace()
        with (
            _installed(),
            patch.object(flux3_image, "_STOCK_MAYBE_TORCH_COMPILE") as stock,
        ):
            stages.Flux3ImageDenoiseStage._maybe_torch_compile(
                stage, torch.nn.Linear(1, 1)
            )
            stages.Flux3ImageDenoiseStage._maybe_torch_compile(
                stage, torch.nn.Linear(1, 1, device="meta")
            )
        self.assertEqual(stock.call_count, 1)  # only the meta (non-TPU) module


if __name__ == "__main__":
    unittest.main()
