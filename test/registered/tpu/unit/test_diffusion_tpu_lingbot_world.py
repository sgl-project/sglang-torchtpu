# SPDX-License-Identifier: Apache-2.0
"""LingBot-World realtime chunks through the real stages: the chunk start the
DiT leaves on its KV caches, the host-side schedule lookups, and sglang_tpu's
LingBot-World patches (sglang_tpu.multimodal_gen.lingbot_world) under regional
compile. The meta device stands in for a TPU where a test shows that no device
op reads back to the host."""

import contextlib
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    fuse_scale_shift_kernel_native,
)
from sglang.multimodal_gen.configs.models.dits.lingbot_world import (
    LingBotWorldArchConfig,
    LingBotWorldVideoConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.lingbot_world import (
    LingBotWorldCausalDMDConfig,
    LingBotWorldI2VConfig,
    LingBotWorldV2CausalDMDConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits import lingbot_world as dit
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    pred_noise_to_pred_video,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    causal_denoising,
    timestep_preparation,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.condition_encoding import (
    AuxiliaryConditionEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_world import (
    lingbot_world_causal_denoising as stage_module,
)
from sglang.multimodal_gen.runtime.realtime.session import RealtimeSession
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.utils.torch_compile import (
    compile_matching_submodules,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import lingbot_world, worker

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=40, suite="nightly-tpu-4-v7", nightly=True)

Stage = stage_module.LingBotWorldCausalDMDDenoisingStage
Block = dit.CausalLingBotWorldTransformerBlock
Model = dit.CausalLingBotWorldTransformer3DModel
CPU, META = torch.device("cpu"), torch.device("meta")
PATCHED = (
    (dit.LingBotWorldCausalSelfAttention, "_attend_with_kv_cache"),
    (Model, "_compute_rope_for_sequence_shard_with_offset"),
    (Stage, "_prepare_causal_dmd_timesteps"),
    (Stage, "_add_noise_for_next_timestep"),
    (LingBotWorldI2VConfig, "prepare_world_condition"),
)


@contextlib.contextmanager
def _installed(device_types=("cpu",), stock_draw=False):
    """The TPU paths on ``device_types``, with every patched name restored;
    ``stock_draw`` keeps the stock bfloat16 re-noise draw."""
    with contextlib.ExitStack() as stack:
        for owner, name in PATCHED:
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(lingbot_world, "_DEVICE_TYPES", device_types))
        lingbot_world.install()
        lingbot_world.install()  # idempotent
        if stock_draw:
            draw = lingbot_world._STOCK_ADD_NOISE
            stack.enter_context(patch.object(Stage, PATCHED[3][1], draw))
        yield


@contextlib.contextmanager
def _ulysses(world, rank):
    """One rank of a Ulysses group of ``world`` ranks that hold the same tokens."""

    def a2a_in(x, head_dim):
        h = x.shape[2] // world
        return torch.cat([x[:, :, rank * h : (rank + 1) * h]] * world, dim=1)

    def a2a_out(x, head_dim):
        s = x.shape[1] // world
        return torch.cat([x[:, rank * s : (rank + 1) * s]] * world, dim=2)

    sizes = dict(get_ulysses_parallel_world_size=lambda: world)
    sizes.update(get_ring_parallel_world_size=lambda: 1)
    names = {stage_module: sizes, dit: dict(sizes, get_sp_world_size=lambda: world)}
    names[dit].update(get_sp_parallel_rank=lambda: rank, get_tp_rank=lambda: 0)
    names[dit].update(_usp_input_all_to_all=a2a_in, _usp_output_all_to_all=a2a_out)
    names[dit].update(get_sp_group=lambda: SimpleNamespace(device_group=None))
    gather = lambda x, dim: torch.cat([x] * world, dim)  # noqa: E731
    names[dit].update(sequence_model_parallel_all_gather=gather)
    with contextlib.ExitStack() as stack:
        for module, attrs in names.items() if world > 1 else ():
            for name, value in attrs.items():
                stack.enter_context(patch.object(module, name, value))
        yield


def _model(layers=2):
    torch.manual_seed(0)
    # The checkpoints' 9-frame sink and 18-frame window over 3-frame chunks:
    # the visible K/V grows for six chunks, the seventh rolls the cache.
    arch = dict(num_attention_heads=2, attention_head_dim=32, in_channels=8)
    arch.update(out_channels=4, text_dim=24, text_len=6, freq_dim=32, ffn_dim=48)
    arch.update(num_layers=layers, sink_size=9, sliding_window_num_frames=18)
    config = LingBotWorldVideoConfig(arch_config=LingBotWorldArchConfig(**arch))
    model = Model(config, hf_config={})
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    model = model.to(torch.bfloat16).eval()
    model.post_load_weights()
    return model


def _session(
    model,
    chunks,
    camera=True,
    camera_stop=None,
    sp=True,
    interactive=False,
    after_chunk=None,
):
    """A realtime session through the DMD schedule, camera condition and
    LingBot-World denoising stages, with camera actions up to chunk
    ``camera_stop``; returns each chunk's latents."""
    scheduler = SelfForcingFlowMatchScheduler(
        num_inference_steps=1000, shift=5.0, sigma_min=0.0, extra_one_step=True
    )
    stages = (
        timestep_preparation.DMDTimestepPreparationStage(scheduler),
        AuxiliaryConditionEncodingStage(),
        Stage(transformer=model, scheduler=scheduler),
    )
    config = LingBotWorldV2CausalDMDConfig(interactive_kv_window_enable=interactive)
    server_args = SimpleNamespace(
        pipeline_config=config,
        disable_autocast=False,
        kv_cache_quant_config=SimpleNamespace(enabled=False),
    )
    gen = torch.Generator().manual_seed(1)
    image_latent = torch.randn(1, 4, 6, 4, 4, generator=gen)
    fields = dict(image_latent=image_latent, session=RealtimeSession())
    fields.update(prompt_embeds=[torch.randn(1, 6, 24, generator=gen).bfloat16()])
    fields.update(realtime_session_id="s", realtime_chunk_size=3, width=32, height=32)
    fields.update(enable_sequence_shard=sp, is_warmup=False, extra={})
    fields.update(dict.fromkeys(("timesteps", "scheduler", "c2ws_plucker_emb")))
    out = []
    with torch.no_grad():
        for index in range(chunks):
            actions = [["w"], ["w", "l"], []] if index % 2 else [["a"], [], ["i"]]
            batch = SimpleNamespace(
                **fields,
                latents=torch.randn(1, 4, 3, 4, 4, generator=gen).bfloat16(),
                generator=torch.Generator().manual_seed(100 + index),
                block_idx=index,
                condition_inputs=(
                    {"camera_actions": actions}
                    if camera and index < (camera_stop or chunks)
                    else {}
                ),
                record_stage_iterations=lambda *args: None,
            )
            for stage in stages:
                stage.forward(batch, server_args)
            out.append(batch.latents)
            if after_chunk is not None:
                after_chunk()
    return out


# (Ulysses world, rank, session options); 7 chunks reach the first roll.
SCENARIOS = {
    "ulysses_rank0": (2, 0, {}),
    "ulysses_rank1": (2, 1, {}),
    "one_core": (1, 0, {"sp": False}),
    "no_camera": (2, 1, {"camera": False}),
    "interactive_window": (2, 0, {"interactive": True}),
    "camera_stops": (2, 1, {"camera_stop": 2}),
}


def _int_chunk_start(stack, model):
    forward = Block.forward

    def stock(self, *args, **kwargs):
        # The stock DiT passed its blocks the chunk start.
        kwargs["current_start"] = kwargs["kv_cache"].chunk_start
        return forward(self, *args, **kwargs)

    stack.enter_context(patch.object(Block, "forward", stock))


def _context_pass_cam_shifts(stack, model):
    prepare = Model._prepare_cam_conditioner_scale_shifts

    def stock(self, c2ws, forward_batch):
        if get_forward_context().current_timestep < 0:
            return None
        return prepare(self, c2ws, forward_batch)

    name = "_prepare_cam_conditioner_scale_shifts"
    stack.enter_context(patch.object(Model, name, stock))


def _channel_major(module, args, kwargs):
    # The stock one-core DiT handed the blocks the patch embedding's view.
    hidden_states = args[0].transpose(1, 2).contiguous().transpose(1, 2)
    return (hidden_states, *args[1:]), kwargs


def _token_major(stack, model):
    hook = model.blocks[0].register_forward_pre_hook(_channel_major, with_kwargs=True)
    stack.callback(hook.remove)


# Each restores one core change to its stock form.
STOCK = {
    "int_chunk_start": _int_chunk_start,
    "context_pass_cam_shifts": _context_pass_cam_shifts,
    "token_major": _token_major,
}


def _heads_major(model):
    """Attention results laid out as the TPU backends return them, a
    ``[B, H, S, D]`` result transposed (CPU SDPA returns a contiguous one)."""
    stack = contextlib.ExitStack()
    for block in model.blocks:
        for attn in (block.attn1.attn, block.attn1.ulysses_attn):

            def forward(*args, _forward=attn.forward):
                return _forward(*args).transpose(1, 2).contiguous().transpose(1, 2)

            stack.enter_context(patch.object(attn, "forward", forward))
    return stack


class _DeviceOps(TorchDispatchMode):
    """The ops that took a meta ("device") tensor."""

    def __init__(self):
        super().__init__()
        self.ops = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if any(isinstance(a, torch.Tensor) and a.is_meta for a in args):
            self.ops.add(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def _schedule(config, device=META):
    """``DMDTimestepPreparationStage`` on a worker whose device is ``device``."""
    scheduler = SelfForcingFlowMatchScheduler(
        num_inference_steps=1000,
        shift=config.flow_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    batch = SimpleNamespace(scheduler=None, timesteps=None, is_warmup=False)
    stage = timestep_preparation.DMDTimestepPreparationStage(scheduler)
    with patch.object(timestep_preparation, "get_local_torch_device", lambda: device):
        stage.forward(batch, SimpleNamespace(pipeline_config=config))
    return batch


class _TestBase(CustomTestCase):
    def setUp(self):
        # The forms TPU runs: on the CPU platform the fused norms call
        # sgl_kernel's CPU build.
        def native(op):
            return inspect.unwrap(type(op).forward_native).__get__(op)

        for patcher in (
            # Stages on the CPU with the latents, not the platform's TPU.
            patch.object(causal_denoising, "get_local_torch_device", lambda: CPU),
            patch.object(PipelineStage, "device", property(lambda self: CPU)),
            patch.object(CustomOp, "dispatch_forward", native),
            patch.object(
                layernorm, "fuse_scale_shift_kernel", fuse_scale_shift_kernel_native
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self._previous_args = server_args_module._global_server_args
        set_global_server_args(_make_unit_server_args())
        if not model_parallel_is_initialized():
            ensure_distributed_env_defaults()
            with patch.object(worker, "install_runtime_patches"):
                maybe_init_distributed_environment_and_model_parallel(
                    tp_size=1, sp_size=1
                )

    def tearDown(self):
        set_global_server_args(self._previous_args)


class TestStockEquivalence(_TestBase):
    def test_core_changes_and_patches_match_stock_bitwise(self):
        """Each STOCK form, and the patches, against the branch. The block's
        modulation split and the RoPE and resume casts compute the stock values
        (chunk views, same-dtype casts); TestHostSchedule checks the host
        lookups against the stock formula."""

        def patches(stack, model):
            stack.enter_context(_installed(stock_draw=True))

        variants = {**STOCK, "patches": patches}
        cases = ((), *((name,) for name in variants), tuple(STOCK))
        for scenario, (world, rank, options) in SCENARIOS.items():
            expected = None
            for case in cases:
                with self.subTest(scenario=scenario, stock=case):
                    with _ulysses(world, rank), contextlib.ExitStack() as stack:
                        model = _model()
                        for name in case:
                            variants[name](stack, model)
                        latents = _session(model, 7, **options)
                    self.assertTrue(all(bool(x.isfinite().all()) for x in latents))
                    expected = expected or latents
                    self.assertTrue(all(map(torch.equal, latents, expected)))


class TestRegionalCompile(_TestBase):
    def _compile(self, model):
        self.graphs, self.layouts = 0, set()

        def record(gm, example_inputs):
            self.graphs += 1

            def run(*args):
                # torch_tpu guards each input's storage offset, dynamo every
                # stride: (offset, standard strides) on every call.
                for t in args:
                    if isinstance(t, torch.Tensor):
                        standard = torch.empty(t.shape).stride()
                        self.layouts.add((t.storage_offset(), t.stride() == standard))
                return gm.forward(*args)

            return run

        torch._dynamo.reset()
        compile_kwargs = {"backend": record, "fullgraph": False, "dynamic": None}
        compile_matching_submodules(model, compile_kwargs=compile_kwargs)

    def test_chunks_after_the_first_compile_nothing(self):
        names = ("ulysses_rank1", "one_core", "interactive_window", "camera_stops")
        for name in names:
            world, rank, options = SCENARIOS[name]
            with self.subTest(scenario=name), _ulysses(world, rank):
                expected = _session(_model(), 9, **options)
                model, counts = _model(), []
                with (
                    _installed(stock_draw=True),
                    _heads_major(model),
                    # TpuOOTPlatform's static shapes; every worker imports
                    # sglang.multimodal_gen.runtime.layers.lora.linear, which
                    # sets the recompile limit to 64.
                    torch._dynamo.config.patch(
                        automatic_dynamic_shapes=False, recompile_limit=64
                    ),
                ):
                    self._compile(model)
                    count = lambda: counts.append(self.graphs)  # noqa: E731
                    latents = _session(model, 9, after_chunk=count, **options)
                    _session(model, 9, **options)  # a second session
                    count()
                self.assertTrue(all(map(torch.equal, latents, expected)))
                # The first chunk compiles 7 graphs (4 DMD steps and the context
                # pass, whose last block only writes its K/V), and the chunk the
                # camera condition stops at one more, for blocks without one.
                stop = options.get("camera_stop", 10)
                self.assertEqual(counts, [7] * stop + [8] * (10 - stop))
                self.assertEqual(self.layouts, {(0, True)})


class TestHostRope(_TestBase):
    def test_shard_tables_built_on_the_host(self):
        model, stock = _model(layers=1), lingbot_world._STOCK_SHARD_ROPE

        def rope(*args):
            return model._compute_rope_for_sequence_shard_with_offset(*args)

        # Ulysses 8 shards of 832x480 chunks (30x52 latent tokens a frame).
        for chunk, rank in ((0, 0), (0, 5), (7, 3)):
            args = (585, chunk * 4680 + rank * 585, 1560, 52)
            expected = stock(model, *args, CPU)
            stock.cache_clear()
            with _installed():
                actual = rope(*args, CPU)
            for a, e in zip(actual, expected):
                self.assertEqual((a.dtype, a.storage_offset()), (torch.float32, 0))
                self.assertTrue(a.is_contiguous() and torch.equal(a, e))
        # The stock tables read each axis back with tolist(), which a meta
        # tensor cannot; the host tables move to the device.
        with self.assertRaisesRegex(NotImplementedError, "copy out of meta"):
            rope(*args, META)
        with _installed(("meta",)):
            cos, sin = rope(*args, META)
        self.assertEqual(
            (cos.device, sin.dtype, sin.shape), (META, torch.float32, e.shape)
        )


class TestHostSchedule(_TestBase):
    def test_lookups_run_on_the_host(self):
        # v1 and v2 schedules (shift 10 and 5): the device runs only the
        # elementwise x0 and re-noise updates, with the stock values.
        gen = torch.Generator().manual_seed(0)
        pred, x, noise = torch.randn(3, 3, 4, 2, 2, generator=gen).bfloat16()
        for config in (LingBotWorldCausalDMDConfig(), LingBotWorldV2CausalDMDConfig()):
            batch = _schedule(config)
            scheduler = batch.scheduler
            self.assertEqual(batch.timesteps.device, CPU)
            for t in batch.timesteps:
                step = t.reshape(1)
                x0 = pred_noise_to_pred_video(pred, x, step, scheduler)
                renoised = scheduler.add_noise(x0, noise, step)
                ids = torch.argmin((scheduler.timesteps.double() - t.double()).abs())
                sigma = scheduler.sigmas[ids].reshape(1, 1, 1, 1)
                expected = (x.double() - sigma.double() * pred.double()).bfloat16()
                self.assertTrue(torch.equal(x0, expected))
                expected = ((1 - sigma) * x0 + sigma * noise).bfloat16()
                self.assertTrue(torch.equal(renoised, expected))
                ops = _DeviceOps()
                with ops:
                    p, n, z = (a.to(META) for a in (pred, x, noise))
                    scheduler.add_noise(
                        pred_noise_to_pred_video(p, n, step, scheduler), z, step
                    )
                self.assertNotIn("argmin", ops.ops)
            self.assertEqual(scheduler.timesteps.device, CPU)

    def test_stage_steps_on_the_host_schedule(self):
        batch, stage = _schedule(LingBotWorldV2CausalDMDConfig()), Stage.__new__(Stage)
        for devices, expected in ((("tpu",), META), (("meta",), CPU)):
            with _installed(devices):
                args = (batch, None, batch.scheduler, META)
                self.assertEqual(
                    stage._prepare_causal_dmd_timesteps(*args).device, expected
                )
        # The step's x0 takes the host timestep, not its device copy.
        ops, latents = _DeviceOps(), torch.zeros(1, 3, 4, 2, 2, device=META)
        pred = lambda *args, **kw: torch.zeros(1, 4, 3, 2, 2, device=META)  # noqa: E731
        unused = ("prompt_embeds", "kv_cache", "crossattn_cache", "image_kwargs")
        unused += ("pos_cond_kwargs", "attn_raw_latent_shape", "target_dtype")
        kwargs = dict.fromkeys(unused) | dict(current_start_tokens=0, start_frame=0)
        kwargs.update(current_timestep=1, autocast_enabled=False, device=META)
        kwargs.update(timestep=batch.timesteps[1], scheduler=batch.scheduler)
        with (
            patch.object(Stage, "_forward_causal_transformer", pred),
            patch.object(Stage, "_build_causal_attn_metadata", lambda *a, **k: None),
            ops,
        ):
            stage._predict_x0_btchw(
                None,
                None,
                latent_model_input=latents.permute(0, 2, 1, 3, 4),
                noise_latents_btchw=latents,
                **kwargs,
            )
        self.assertNotIn("argmin", ops.ops)


class TestRenoise(_TestBase):
    def _renoise(self, generator, device):
        x0 = torch.zeros(1, 3, 4, 8, 8, dtype=torch.bfloat16, device=device)
        drawn = []
        scheduler = SimpleNamespace(add_noise=lambda x, n, t: drawn.append(n) or x)
        Stage._add_noise_for_next_timestep(
            Stage.__new__(Stage),
            SimpleNamespace(generator=generator),
            x0_btchw=x0,
            raw_latent_shape=x0.shape,
            next_timestep=torch.ones(1),
            scheduler=scheduler,
            device=device,
        )
        return drawn[0].reshape(x0.shape)

    def test_float32_draw_on_the_generator_device(self):
        gen = torch.Generator().manual_seed(5)
        expected = torch.randn(1, 3, 4, 8, 8, generator=gen).bfloat16()
        with _installed():
            noise = self._renoise(torch.Generator().manual_seed(5), CPU)
        self.assertTrue(torch.equal(noise, expected))
        # A CPU generator with device latents: drawn on the generator, then
        # moved (the stock draw passes the generator to the device's randn).
        cpu_gen = torch.Generator().manual_seed(5)
        with _installed(("meta",)):
            noise = self._renoise(cpu_gen, META)
        self.assertEqual((noise.device, noise.dtype), (META, torch.bfloat16))
        next_draws = (torch.randn(1, generator=g) for g in (cpu_gen, gen))
        self.assertTrue(torch.equal(*next_draws))


class TestCameraCondition(_TestBase):
    def _conditions(self, device, chunks=4):
        config, session, out = LingBotWorldV2CausalDMDConfig(), RealtimeSession(), []
        scripts = (
            [["w"], ["w", "j"], ["i"]],
            [["d", "k"]],
            [[], ["s"], ["l"]],
            [["a"]],
        )
        for index in range(chunks):
            batch = SimpleNamespace(
                c2ws_plucker_emb=None,
                condition_inputs={"camera_actions": scripts[index]},
                realtime_chunk_size=3,
                session=session,
                block_idx=index,
                width=64,
                height=48,
                realtime_session_id="s",
            )
            out.append(config.prepare_world_condition(batch, device, torch.bfloat16))
        return [c["c2ws_plucker_emb"] for c in out]

    def test_host_condition_matches_stock(self):
        expected = self._conditions(CPU)
        with _installed():
            actual = self._conditions(CPU)
        for a, e in zip(actual, expected):
            self.assertEqual(tuple(a.shape), (1, 384, 3, 6, 8))
            self.assertTrue(torch.equal(a, e))
        # The stock condition reads max_norm > 0 back from the device.
        with self.assertRaisesRegex(RuntimeError, r"item\(\) cannot be called"):
            self._conditions(META, chunks=1)
        with _installed(("meta",)):
            self.assertEqual(self._conditions(META, chunks=1)[0].device, META)


class TestPatch(_TestBase):
    def test_install_rebinds_and_restores(self):
        stock = [getattr(owner, name) for owner, name in PATCHED]
        with _installed():
            for owner, name in PATCHED:
                self.assertIs(getattr(owner, name), getattr(lingbot_world, name))
        self.assertEqual([getattr(owner, name) for owner, name in PATCHED], stock)

    def test_attend_takes_the_base_signature(self):
        # A caller that passes (q, k, v, kv_cache, current_start) writes and attends.
        for attend in (
            lingbot_world._STOCK_ATTEND,
            lingbot_world._attend_with_kv_cache,
        ):
            parameter = inspect.signature(attend).parameters["update_cache_only"]
            self.assertIs(parameter.default, False)

    def test_cpu_tensors_and_int_starts_run_stock(self):
        expected = _session(_model(), 2, sp=False)
        with (
            _installed(("tpu",)),
            patch.object(lingbot_world, "_attend_eagerly") as eager,
        ):
            self.assertTrue(
                all(map(torch.equal, _session(_model(), 2, sp=False), expected))
            )
        eager.assert_not_called()
        # A chunk start passed as an int takes the stock attention.
        attn, q = _model().blocks[0].attn1, torch.zeros(1, 4, 2, 32)
        with (
            _installed(),
            patch.object(lingbot_world, "_STOCK_ATTEND") as stock,
            patch.object(lingbot_world, "_attend_eagerly") as eager,
        ):
            attn._attend_with_kv_cache(q, q, q, None, 0, False)
            attn._attend_with_kv_cache(q, q, q, None, None, False)
        stock.assert_called_once()
        eager.assert_called_once()


if __name__ == "__main__":
    unittest.main()
