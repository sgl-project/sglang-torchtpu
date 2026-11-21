# SPDX-License-Identifier: Apache-2.0
"""Tests for sglang_tpu's HiDream-O1-Image patches (sglang_tpu.multimodal_gen.hidream_o1_image)."""

import contextlib
import os
import socket
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.multiprocessing as mp

from sglang.multimodal_gen.configs.models.dits.hidream_o1_image import (
    HiDreamO1ImageArchConfig,
    HiDreamO1ImageDitConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.hidream_o1_image import (
    HiDreamO1ImagePipelineConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import hidream_o1_image as dit
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    hidream_o1_image as stages,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import hidream_o1_image, platform, worker

# Whole node, not chips=1: the parallel-state setup runs the TPU worker
# bootstrap, which opens a chip outside the chip-parallel TPU_VISIBLE_CHIPS pin.
register_tpu_ci(est_time=120, suite="nightly-tpu-4-v7", nightly=True)

_GRID = (2, 3)  # image patches (rows, cols)
_ARCH = dict(
    vocab_size=64,
    hidden_size=512,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    tms_token_id=63,
)


@contextlib.contextmanager
def _installed():
    """Run install() with every patched name restored afterwards."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (HiDreamO1ImagePipelineConfig, "validate_server_args"),
            (HiDreamO1ImagePipelineConfig, "shard_latents_for_sp"),
            (HiDreamO1ImagePipelineConfig, "get_model_deployment_config"),
            (stages.HiDreamO1ImageBeforeDenoisingStage, "_build_conditioning"),
            (stages.HiDreamO1ImageBeforeDenoisingStage, "forward"),
            (dit.HiDreamO1ImageTransformer, "forward"),
            (dit.HiDreamO1Attention, "forward"),
            (dit.HiDreamO1ImageTransformer, "_compile_regions_spmd_safe"),
            (stages.HiDreamO1ImageDenoisingStage, "_predict_noise_with_cfg"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        hidream_o1_image.install()
        hidream_o1_image.install()  # idempotent
        yield


def _server_args():
    args = _make_unit_server_args()
    args.pipeline_config.dit_config = HiDreamO1ImageDitConfig(
        arch_config=HiDreamO1ImageArchConfig(**_ARCH)
    )
    return args


def _model():
    torch.manual_seed(0)
    config = HiDreamO1ImageDitConfig(arch_config=HiDreamO1ImageArchConfig(**_ARCH))
    model = dit.HiDreamO1ImageTransformer(config=config)
    for name, param in model.named_parameters():
        # Unit norm scales: per-head q/k of unit RMS give attention scores
        # that depend on the RoPE positions and the mask.
        param.data.fill_(1.0) if "norm" in name else param.data.normal_(0, 0.05)
    return model.to(torch.bfloat16).eval()


def _stage(text_len):
    """The before-denoising stage on a prompt of ``text_len`` tokens."""
    stage = stages.HiDreamO1ImageBeforeDenoisingStage.__new__(
        stages.HiDreamO1ImageBeforeDenoisingStage
    )
    gen = torch.Generator().manual_seed(text_len)
    ids = torch.randint(1, _ARCH["tms_token_id"], (1, text_len), generator=gen)
    ids[0, -1] = _ARCH["tms_token_id"]  # the stage appends it last
    stage._encode_prompt = lambda prompt, device: ids.to(device)
    return stage


def _inputs(text_len, dtype=torch.bfloat16, batch_size=1, stock=False):
    """The stage's DiT kwargs and the request's latents and timestep."""
    gen = torch.Generator().manual_seed(1000 + text_len)
    image_len = _GRID[0] * _GRID[1]
    build = (
        hidream_o1_image._STOCK_BUILD_CONDITIONING
        if stock
        else stages.HiDreamO1ImageBeforeDenoisingStage._build_conditioning
    )
    cond = build(
        _stage(text_len),
        "p",
        batch_size=batch_size,
        h_patches=_GRID[0],
        w_patches=_GRID[1],
        mask_dtype=dtype,
        device=torch.device("cpu"),
    )
    return dict(
        hidden_states=torch.randn(batch_size, image_len, 3072, generator=gen).to(dtype),
        timestep=torch.full((batch_size,), 0.37),
        **cond,
    )


def _run(model, kwargs):
    with (
        set_forward_context(current_timestep=0, attn_metadata=None),
        torch.no_grad(),
    ):
        return model(**kwargs)


def _init_parallel_state(sp=1):
    set_global_server_args(_server_args())
    if not model_parallel_is_initialized():
        ensure_distributed_env_defaults()
        with patch.object(worker, "install_runtime_patches"):
            maybe_init_distributed_environment_and_model_parallel(
                tp_size=1, sp_size=sp, ulysses_degree=sp
            )


def _layer_layouts(model, calls):
    for layer in model.model.language_model.layers:
        layer.register_forward_pre_hook(
            lambda module, args: calls.append(
                [
                    (t.stride(), t.storage_offset())
                    for t in args
                    if isinstance(t, torch.Tensor)
                ]
            )
        )


def _sp_worker(rank, port, out_dir, text_lens):
    """One Ulysses rank of an SP 2 run over gloo; rank 0 saves the outputs."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
    )
    _init_parallel_state(sp=2)
    with patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_native):
        model = _model()
        layouts = []
        _layer_layouts(model, layouts)
        with _installed():
            for text_len in text_lens:
                out = _run(model, _inputs(text_len))
                if rank == 0:
                    torch.save(out, os.path.join(out_dir, f"{text_len}.pt"))
            # Both CFG branches as one batch over a real exchange.
            pos, neg = _inputs(40), _inputs(7)
            out = _run(
                model,
                {
                    **pos,
                    "hidden_states": torch.cat(
                        [pos["hidden_states"], neg["hidden_states"]]
                    ),
                    "timestep": pos["timestep"].repeat(2),
                    "input_ids": torch.cat([pos["input_ids"], neg["input_ids"]]),
                    "tpu_layout": hidream_o1_image._stack(
                        (pos["tpu_layout"], neg["tpu_layout"]), 1
                    ),
                },
            )
            serial = torch.cat([_run(model, kwargs) for kwargs in (pos, neg)])
            if rank == 0:
                torch.save((out, serial), os.path.join(out_dir, "cfg.pt"))
        torch.save(layouts, os.path.join(out_dir, f"layouts.{rank}.pt"))


class _TestBase(CustomTestCase):
    def setUp(self):
        self.enterContext(
            patch.object(CustomOp, "dispatch_forward", lambda op: op.forward_native)
        )
        self._previous_args = server_args_module._global_server_args
        _init_parallel_state()

    def tearDown(self):
        set_global_server_args(self._previous_args)


class TestHiDreamO1Layout(_TestBase):
    def test_rows_positions_rope_and_key_counts(self):
        image_len, rotary = (
            _GRID[0] * _GRID[1],
            _model().model.language_model.rotary_emb,
        )
        with _installed():
            for text_len, rows in ((7, 128), (122, 128), (123, 256)):
                kwargs = _inputs(text_len, batch_size=2)
                layout = kwargs["tpu_layout"]
                self.assertEqual(kwargs["input_ids"].shape, (2, rows - image_len - 1))
                self.assertEqual(layout.attention.query_rows, image_len + 1)
                self.assertEqual(
                    layout.attention.keys.tolist(), [[0, image_len + text_len]]
                )
                # The stock positions as [image | timestep | prompt] rows.
                got = hidream_o1_image.row_positions(text_len, *_GRID, rows)
                stock = stages.build_hidream_o1_position_ids(
                    text_len=text_len,
                    h_patches=_GRID[0],
                    w_patches=_GRID[1],
                    device="cpu",
                )
                order = [
                    *range(text_len, text_len + image_len),
                    text_len - 1,
                    *range(text_len - 1),
                ]
                self.assertTrue(
                    torch.equal(got[..., : image_len + text_len], stock[..., order])
                )
                cos, sin = rotary(got, torch.bfloat16)
                self.assertTrue(
                    torch.equal(layout.cos, cos) and torch.equal(layout.sin, sin)
                )


class TestHiDreamO1Denoise(_TestBase):
    def test_bucketed_rows_match_the_stock_mask(self):
        # 7 and 122 tokens share the 128-row bucket, 123 takes 256. In float32
        # the stock dense-mask SDPA and the split attention agree to rounding.
        # In bfloat16 the split attention runs a float32 softmax like the TPU
        # kernel and stock CPU SDPA rounds otherwise: within a bfloat16 step of
        # the outputs, and no further from float32 than stock.
        model, float32_model = _model(), _model().float()
        for text_len in (7, 122, 123):
            stock = _run(model, _inputs(text_len))
            exact = _run(float32_model, _inputs(text_len, torch.float32))
            with _installed():
                got = _run(model, _inputs(text_len))
                got32 = _run(float32_model, _inputs(text_len, torch.float32))
            torch.testing.assert_close(got32, exact, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(got, stock, rtol=0, atol=2**-5)
            self.assertLessEqual(
                (got - exact).abs().max(), (stock - exact).abs().max() + 2**-6
            )

    def test_padded_prompt_rows_never_reach_the_image(self):
        model = _model()
        with _installed():
            kwargs = _inputs(7)
            want = _run(model, kwargs)
            ids = kwargs["input_ids"].clone()
            ids[:, 6:] = 13  # other padding tokens
            got = _run(model, {**kwargs, "input_ids": ids})
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_cfg_branches_in_one_call_match_serial_calls(self):
        model = _model()
        calls = []

        def predict_noise(current_model, latent_model_input, **kwargs):
            calls.append(latent_model_input.shape[0])
            del kwargs["target_dtype"], kwargs["guidance"]
            return current_model(hidden_states=latent_model_input, **kwargs)

        stage = SimpleNamespace(_predict_noise=predict_noise)
        batch = SimpleNamespace(
            do_classifier_free_guidance=True,
            extra={},
            sampling_params=SimpleNamespace(quality=None),
            cfg_normalization=0.0,
            guidance_rescale=0.0,
        )
        config = HiDreamO1ImagePipelineConfig()
        with _installed():
            # 40 and 7 prompt tokens share a bucket; two images per prompt.
            pos, neg = _inputs(40, batch_size=2), _inputs(7, batch_size=2)
            for kwargs in (neg, pos):
                latents, timestep = kwargs.pop("hidden_states"), kwargs.pop("timestep")
            policy = CFGPolicy().build(batch, {}, pos, neg)
            call = dict(
                current_model=model,
                latent_model_input=latents,
                timestep=timestep,
                batch=batch,
                timestep_index=0,
                attn_metadata=None,
                target_dtype=torch.bfloat16,
                current_guidance_scale=5.0,
                cfg_policy=policy,
                cfg_gate_state=None,
                server_args=SimpleNamespace(
                    enable_cfg_parallel=False, pipeline_config=config
                ),
                guidance=None,
                latents=latents,
            )
            with torch.no_grad():
                want = hidream_o1_image._STOCK_PREDICT_WITH_CFG(stage, **call)
                # One core: the serial calls.
                hidream_o1_image.predict_noise_with_cfg(stage, **call)
                # Under Ulysses (the exchanges are no-ops at one rank): one call.
                with patch.object(hidream_o1_image, "get_sp_world_size", lambda: 2):
                    got = hidream_o1_image.predict_noise_with_cfg(stage, **call)
        self.assertEqual(calls, [2, 2, 2, 2, 4])
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_cfg_branches_share_the_longer_bucket_under_ulysses(self):
        # 130 and 7 prompt tokens: 256 and 128 rows.
        stage, keys, (h, w) = _stage(7), hidream_o1_image._COND_KEYS, _GRID
        ids = {
            "pos": _stage(130)._encode_prompt(None, "cpu"),
            "neg": stage._encode_prompt(None, "cpu"),
        }
        stage._encode_prompt = lambda prompt, device: ids[prompt].to(device)
        batch = SimpleNamespace(
            prompt="pos",
            negative_prompt="neg",
            seeds=[42],
            height=32 * h,
            width=32 * w,
            extra={},
        )
        build = dict(
            batch_size=1,
            h_patches=h,
            w_patches=w,
            mask_dtype=torch.bfloat16,
            device="cpu",
        )

        def stock(self, batch, server_args):
            for key, prompt in zip(keys, ("pos", "neg")):
                batch.extra[key] = hidream_o1_image.build_conditioning(
                    self, prompt, **build
                )
            return batch

        with (
            _installed(),
            patch.object(hidream_o1_image, "_STOCK_STAGE_FORWARD", stock),
        ):
            hidream_o1_image.stage_forward(stage, batch, _server_args())
            self.assertEqual(
                [batch.extra[k]["input_ids"].shape[1] for k in keys], [249, 121]
            )
            with patch.object(hidream_o1_image, "get_sp_world_size", lambda: 2):
                hidream_o1_image.stage_forward(stage, batch, _server_args())
        pos, neg = (batch.extra[k] for k in keys)
        self.assertEqual(neg["tpu_layout"].cos.shape, pos["tpu_layout"].cos.shape)
        self.assertEqual(neg["tpu_layout"].attention.keys.tolist(), [[0, h * w + 7]])
        self.assertIs(batch.negative_prompt_embeds[0], neg["input_ids"])

    def test_sequence_parallel_rows_match_one_rank(self):
        # SP 2 over gloo: 7 tokens leave rank 1 only prompt padding; 123 take
        # the 256-row bucket.
        text_lens = (7, 123)
        model = _model()
        with _installed():
            wants = [_run(model, _inputs(n)) for n in text_lens]
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (
            tempfile.TemporaryDirectory() as out_dir,
            patch.dict(os.environ, SGLANG_DIFFUSION_PLATFORM_OVERRIDE="cpu"),
        ):
            mp.spawn(_sp_worker, args=(port, out_dir, text_lens), nprocs=2)
            for text_len, want in zip(text_lens, wants, strict=True):
                got = torch.load(os.path.join(out_dir, f"{text_len}.pt"))
                torch.testing.assert_close(got, want, rtol=0, atol=2**-6)
            layouts = [
                torch.load(os.path.join(out_dir, f"layouts.{rank}.pt"))
                for rank in range(2)
            ]
            batched, serial = torch.load(os.path.join(out_dir, "cfg.pt"))
        torch.testing.assert_close(batched, serial, rtol=0, atol=0)
        # torch_tpu bakes input strides and storage offsets into each program:
        # every layer on every rank must see fresh tensors of one layout.
        self.assertEqual(layouts[0], layouts[1])
        self.assertEqual({offset for call in layouts[0] for _, offset in call}, {0})


class _CompileCounter:
    """Regional compile of the decoder layers, as --regional-compile on TPU."""

    def __init__(self, model):
        self.graphs = []
        torch._dynamo.reset()
        for layer in model.model.language_model.layers:
            layer.compile(backend=self._backend, dynamic=False)

    def _backend(self, graph_module, example_inputs):
        self.graphs.append(graph_module)
        return graph_module.forward


class TestHiDreamO1Compile(_TestBase):
    def setUp(self):
        super().setUp()
        # The TPU worker turns automatic dynamic shapes off.
        self.enterContext(torch._dynamo.config.patch(automatic_dynamic_shapes=False))
        self.addCleanup(torch._dynamo.reset)

    def test_layers_compile_once_per_prompt_bucket(self):
        model = _model()
        counter, counts = _CompileCounter(model), []
        with _installed():
            for text_len in (7, 40, 122, 123):
                _run(model, _inputs(text_len))
                counts.append(len(counter.graphs))
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[2], counts[0])  # 40 and 122 tokens: 7's bucket
        self.assertGreater(counts[3], counts[2])  # 123 tokens: the next bucket


class TestHiDreamO1Config(_TestBase):
    def _args(self, **parallel):
        none = (
            "flow_shift",
            "scheduler_class_override",
            "transformer_weights_path",
            "quantization",
        )
        empty = (
            "component_paths",
            "component_weights_paths",
            "component_quantizations",
        )
        return SimpleNamespace(
            **dict.fromkeys(none),
            **{k: {} for k in empty},
            enable_breakable_cuda_graph=False,
            quantization_ignored_layers=None,
            **{
                "sp_degree": 1,
                "ulysses_degree": 1,
                "ring_degree": 1,
                "kv_gather_degree": 1,
                **parallel,
            },
        )

    def test_ulysses_is_accepted_and_other_sp_layouts_refused(self):
        config = HiDreamO1ImagePipelineConfig()
        args = self._args(sp_degree=8, ulysses_degree=8)
        with self.assertRaisesRegex(ValueError, "cannot be sharded"):
            config.validate_server_args(args)
        with _installed():
            config.validate_server_args(args)
            self.assertEqual(args.sp_degree, 8)
            # Ring, K/V gather (which aliases ulysses_degree), more ranks than
            # the 8 K/V heads.
            for refused in (
                dict(sp_degree=8, ulysses_degree=4, ring_degree=2),
                dict(sp_degree=8, ulysses_degree=8, kv_gather_degree=8),
                dict(sp_degree=16, ulysses_degree=16),
            ):
                with self.assertRaisesRegex(ValueError, "Ulysses only"):
                    config.validate_server_args(self._args(**refused))
            # The stock checks still run.
            with self.assertRaisesRegex(ValueError, "--quantization"):
                config.validate_server_args(
                    SimpleNamespace(**{**vars(args), "quantization": "fp8"})
                )
            latents = torch.zeros(1, 6, 3072)
            self.assertIs(config.shard_latents_for_sp(None, latents)[0], latents)
            deployment = config.get_model_deployment_config()
            self.assertFalse(deployment.auto_enable_cfg_parallel)
            self.assertFalse(deployment.supports_cfg_parallel)
        self.assertTrue(config.get_model_deployment_config().supports_cfg_parallel)

    def test_the_launcher_installs_the_config_patches(self):
        with _installed():
            HiDreamO1ImagePipelineConfig.validate_server_args = (
                hidream_o1_image._STOCK_VALIDATE
            )
            platform.TpuOOTPlatform().apply_server_args_defaults(
                SimpleNamespace(
                    is_arg_explicitly_set=lambda name: False,
                    offload_during_compile=False,
                )
            )
            self.assertIs(
                HiDreamO1ImagePipelineConfig.validate_server_args,
                hidream_o1_image.validate_server_args,
            )


if __name__ == "__main__":
    unittest.main()
