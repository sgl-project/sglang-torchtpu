# SPDX-License-Identifier: Apache-2.0
"""Helios on TPU: the denoising stage's compile hook and timestep, the
scheduler's device-side UniPC solve, SP sharding, and sglang_tpu's Helios
patches (sglang_tpu.multimodal_gen.helios)."""

import contextlib
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from sglang.kernels.ops.diffusion.common.fallback_torch import (
    fuse_scale_shift_kernel_native,
)
from sglang.multimodal_gen.configs.models.dits.helios import (
    HeliosArchConfig,
    HeliosConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs import helios as helios_config
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers import layernorm
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits import helios as dit
from sglang.multimodal_gen.runtime.models.schedulers import scheduling_helios
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
    helios_denoising as stage_module,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    ensure_distributed_env_defaults,
)
from sglang.multimodal_gen.test.unit.conftest import _make_unit_server_args
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import helios, worker

# Nightly: the stage-a unit suite is at its time budget.
register_tpu_ci(est_time=40, suite="nightly-tpu-4-v7", nightly=True)

Scheduler = scheduling_helios.HeliosScheduler
# The checkpoints' scheduler_config.json.
_UNIPC = dict(scheduler_type="unipc", solver_order=2, time_shift_type="exponential")
BASE = dict(_UNIPC, stages=1, stage_range=[0, 1], use_dynamic_shifting=True)
MID = dict(_UNIPC, stages=3, use_dynamic_shifting=True)
DISTILLED = dict(
    scheduler_type="dmd", stages=3, time_shift_type="linear", use_dynamic_shifting=True
)
# Current tokens per pyramid level of a 384x640 chunk.
PYRAMID_TOKENS = (540, 2160, 8640)


@contextlib.contextmanager
def _installed(device_types=("cpu",)):
    """The TPU paths on ``device_types``, with every patched name restored."""
    with contextlib.ExitStack() as stack:
        for owner, name in (
            (dit, "apply_rotary_emb_transposed"),
            (Scheduler, "set_timesteps"),
            (Scheduler, "step_dmd"),
        ):
            stack.enter_context(patch.object(owner, name, getattr(owner, name)))
        stack.enter_context(patch.object(helios, "_DEVICE_TYPES", device_types))
        helios.install()
        helios.install()  # idempotent
        yield


def _model_output(sample, step, dtype):
    """A deterministic stand-in for the DiT that depends on the sample."""
    if sample.is_meta:
        return torch.empty_like(sample, dtype=dtype)
    gen = torch.Generator().manual_seed(step)
    noise = torch.randn(sample.shape, generator=gen, dtype=torch.float64)
    return (torch.tanh(sample.cpu().double()) + noise).to(sample.device, dtype)


def _run_unipc(scheduler, levels, steps, device="cpu", schedule_device="cpu"):
    """The stage's scheduler calls (one stage-1 level, or the pyramid levels) at
    its dtypes: float32 latents, a bfloat16 DiT output."""
    sample = torch.randn(1, 4, 3, 4, 6, generator=torch.Generator().manual_seed(1))
    sample = sample.to(device)
    outputs = []
    for level, tokens in enumerate(levels):
        mu = stage_module.calculate_shift(tokens)
        if scheduler.config.stages == 1:
            sigmas = np.linspace(0.999, 0.0, steps + 1)[:-1]
            scheduler.set_timesteps(steps, device=schedule_device, sigmas=sigmas, mu=mu)
        else:
            scheduler.set_timesteps(steps, level, device=schedule_device, mu=mu)
        for t in scheduler.timesteps:
            noise_pred = _model_output(sample, len(outputs), torch.bfloat16)
            sample = scheduler.step(noise_pred, t, sample, return_dict=False)[0]
            outputs.append(sample)
    return outputs


class _StockSolve(Scheduler):
    """The stock order-2 corrector (bh2, predict_x0) in its operation order:
    torch.linalg.solve over R and b built on the host."""

    def multistep_uni_c_bh_update(
        self, this_model_output, last_sample=None, this_sample=None, order=None
    ):
        if order != 2:
            return super().multistep_uni_c_bh_update(
                this_model_output, last_sample, this_sample, order
            )
        m0, m1, x = self.model_outputs[-1], self.model_outputs[-2], last_sample

        def lam(sigma):
            alpha, sigma = self._sigma_to_alpha_sigma_t(sigma)
            return torch.log(alpha) - torch.log(sigma), alpha, sigma

        lambda_t, alpha_t, sigma_t = lam(self.sigmas[self.step_index])
        lambda_s0, _, sigma_s0 = lam(self.sigmas[self.step_index - 1])
        h = lambda_t - lambda_s0
        rk = (lam(self.sigmas[self.step_index - 2])[0] - lambda_s0) / h
        hh = -h
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1
        b = [h_phi_k * 1 / h_phi_1, (h_phi_k / hh - 1 / 2) * 2 / h_phi_1]
        R = torch.tensor([[1.0, 1.0], [rk.item(), 1.0]], dtype=h.dtype)
        b = torch.tensor([v.item() for v in b], dtype=h.dtype)
        rhos_c = torch.linalg.solve(R, b).to(x.dtype)
        x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
        D1s = torch.stack([(m1 - m0) / rk], dim=1)
        corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
        D1_t = this_model_output - m0
        x_t = x_t_ - alpha_t * h_phi_1 * (corr_res + rhos_c[-1] * D1_t)
        return x_t.to(x.dtype)


class TestUniPCScheduler(CustomTestCase):
    def test_base_and_mid_schedules_match_the_stock_solve(self):
        for config, levels, steps in (
            (BASE, PYRAMID_TOKENS[-1:], 50),
            (MID, PYRAMID_TOKENS, 20),
        ):
            with self.subTest(stages=config["stages"]):
                expected = _run_unipc(_StockSolve(**config), levels, steps)
                actual = _run_unipc(Scheduler(**config), levels, steps)
                self.assertEqual(len(actual), len(levels) * steps)
                for a, e in zip(actual, expected):
                    self.assertEqual(a.dtype, torch.float32)
                    self.assertTrue(torch.equal(a, e))

    def test_steps_read_nothing_back_and_solve_nothing_on_the_device(self):
        real_tensor = torch.tensor

        def tensor(data, *args, **kwargs):
            if isinstance(data, (list, tuple)) and any(
                isinstance(v, torch.Tensor) for v in data
            ):
                raise AssertionError("torch.tensor over tensors reads them back")
            return real_tensor(data, *args, **kwargs)

        def solve(*args, **kwargs):
            raise AssertionError("order-2 steps solve in closed form")

        with patch.object(torch, "tensor", tensor):
            with patch.object(torch.linalg, "solve", solve):
                outputs = _run_unipc(Scheduler(**MID), PYRAMID_TOKENS, 20)
        self.assertEqual(len(outputs), 60)

    def test_schedule_on_another_device(self):
        # set_timesteps without a device keeps the schedule on the host.
        scheduler = Scheduler(**BASE)
        outputs = _run_unipc(scheduler, PYRAMID_TOKENS[-1:], 4, "meta", None)
        self.assertTrue(all(o.is_meta for o in outputs))

    def test_schedule_built_on_the_host(self):
        # Meta tensors stand in for the TPU: only the copies touch them.
        for config in (BASE, MID):
            with self.subTest(stages=config["stages"]):
                ops = _MetaOps()
                with _installed(("meta",)):
                    scheduler = Scheduler(**config)
                    with ops:
                        scheduler.set_timesteps(20, 0, device="meta", mu=1.0)
                self.assertTrue(scheduler.timesteps.is_meta)
                self.assertTrue(scheduler.sigmas.is_meta)
                stock = Scheduler(**config)
                stock.set_timesteps(20, 0, device="cpu", mu=1.0)
                self.assertEqual(scheduler.sigmas.dtype, stock.sigmas.dtype)
                self.assertEqual(scheduler.timesteps.dtype, stock.timesteps.dtype)
                self.assertLessEqual(ops.ops, {"_to_copy", "copy_", "empty_strided"})
                self.assertIsNotNone(scheduler._tpu_host_schedule)

    def test_host_schedule_matches_stock_bitwise(self):
        for config, levels, steps in (
            (BASE, PYRAMID_TOKENS[-1:], 50),
            (MID, PYRAMID_TOKENS, 20),
        ):
            with self.subTest(stages=config["stages"]):
                expected = _run_unipc(Scheduler(**config), levels, steps)
                with _installed():
                    actual = _run_unipc(Scheduler(**config), levels, steps)
                self.assertEqual(len(actual), len(expected))
                for a, e in zip(actual, expected):
                    self.assertEqual(a.dtype, e.dtype)
                    self.assertTrue(torch.equal(a, e))


class _MetaOps(TorchDispatchMode):
    """Records the ops that touch a meta tensor (the stand-in device)."""

    def __init__(self):
        super().__init__()
        self.ops = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        leaves = tree_leaves((args, kwargs))
        if any(isinstance(v, torch.Tensor) and v.is_meta for v in leaves):
            self.ops.add(func.overloadpacket.__name__)
        return func(*args, **(kwargs or {}))


def _run_dmd(scheduler, device="cpu"):
    """Helios-Distilled's first chunk: the stage's DMD calls on three levels."""
    gen = torch.Generator().manual_seed(2)
    latents = torch.randn(1, 4, 3, 4, 6, generator=gen).to(device)
    outputs, schedules = [], []
    for level, tokens in enumerate(PYRAMID_TOKENS):
        mu = stage_module.calculate_shift(tokens)
        scheduler.set_timesteps(
            10, level, device=device, mu=mu, is_amplify_first_chunk=True
        )
        schedules.append((scheduler.timesteps, scheduler.sigmas))
        start = latents
        timesteps = scheduler.timesteps
        for idx, t in enumerate(timesteps):
            noise_pred = _model_output(latents, len(outputs), torch.bfloat16)
            out = scheduler.step(
                noise_pred,
                t,
                latents,
                return_dict=bool(idx % 2),
                cur_sampling_step=idx,
                dmd_noisy_tensor=start,
                dmd_sigmas=scheduler.sigmas,
                dmd_timesteps=scheduler.timesteps,
                all_timesteps=timesteps,
            )
            latents = out.prev_sample if idx % 2 else out[0]
            outputs.append(latents)
    return outputs, schedules


class TestDmdScheduler(CustomTestCase):
    def test_host_index_matches_stock_bitwise(self):
        expected, expected_schedules = _run_dmd(Scheduler(**DISTILLED))
        with _installed():
            scheduler = Scheduler(**DISTILLED)
            actual, schedules = _run_dmd(scheduler)
            self.assertIsNotNone(scheduler._tpu_host_schedule)
        # The first chunk runs 2 x 10 steps on every level.
        self.assertEqual(len(actual), 60)
        for a, e in zip(actual, expected):
            self.assertEqual(a.dtype, e.dtype)
            self.assertTrue(torch.equal(a, e))
        for got, want in zip(schedules, expected_schedules):
            for a, e in zip(got, want):
                self.assertEqual(a.dtype, torch.float64)
                self.assertTrue(torch.equal(a, e))

    def test_device_runs_no_argmin_and_reads_nothing_back(self):
        # Meta tensors stand in for the TPU: reading one back raises.
        ops = _MetaOps()
        with _installed(("meta",)), ops:
            outputs, schedules = _run_dmd(Scheduler(**DISTILLED), device="meta")
        self.assertEqual(len(outputs), 60)
        self.assertTrue(all(o.is_meta for o in outputs))
        self.assertTrue(all(t.is_meta and s.is_meta for t, s in schedules))
        self.assertNotIn("argmin", ops.ops)
        self.assertNotIn("_local_scalar_dense", ops.ops)

    def test_other_devices_run_stock(self):
        with _installed(("tpu",)):
            scheduler = Scheduler(**DISTILLED)
            with patch.object(
                helios, "_STOCK_STEP_DMD", wraps=helios._STOCK_STEP_DMD
            ) as stock:
                _run_dmd(scheduler)
            self.assertIsNone(scheduler._tpu_host_schedule)
            self.assertEqual(stock.call_count, 60)

    def test_other_schedules_run_stock(self):
        sample = torch.zeros(1, 4, 3, 4, 6)
        with _installed():
            scheduler = Scheduler(**DISTILLED)
            scheduler.set_timesteps(10, 0, device="cpu", mu=1.0)
            own = dict(
                dmd_sigmas=scheduler.sigmas,
                dmd_timesteps=scheduler.timesteps,
                all_timesteps=scheduler.timesteps,
            )
            for name in own:
                with (
                    self.subTest(name),
                    patch.object(helios, "_STOCK_STEP_DMD") as stock,
                ):
                    scheduler.step(
                        sample,
                        scheduler.timesteps[0],
                        sample,
                        dmd_noisy_tensor=sample,
                        **dict(own, **{name: own[name].clone()}),
                    )
                stock.assert_called_once()


def _native_forward(op):
    return inspect.unwrap(type(op).forward_native).__get__(op)


class _TestBase(CustomTestCase):
    def setUp(self):
        # The forms TPU runs: on the CPU platform the fused norms call
        # sgl_kernel's CPU build, and forward_native compiles itself.
        for patcher in (
            patch.object(CustomOp, "dispatch_forward", _native_forward),
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


def _model(dtype=torch.float32):
    torch.manual_seed(0)
    arch = HeliosArchConfig(
        num_attention_heads=2,
        attention_head_dim=32,
        rope_dim=(12, 10, 10),
        text_dim=24,
        freq_dim=32,
        ffn_dim=48,
        num_layers=2,
    )
    model = dit.HeliosTransformer3DModel(HeliosConfig(arch_config=arch), hf_config={})
    for param in model.parameters():
        param.data.normal_(0, 0.2)
    return model.to(dtype).eval()


def _dit_inputs(height=8, width=8, dtype=torch.float32):
    """Stage-1 inputs for one latent chunk with its three history terms."""
    gen = torch.Generator().manual_seed(3)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(dtype)

    return dict(
        hidden_states=randn(1, 16, 9, height, width),
        timestep=torch.tensor([700.0], dtype=torch.float64),
        encoder_hidden_states=randn(1, 7, 24),
        indices_hidden_states=torch.arange(20, 29)[None],
        indices_latents_history_short=torch.tensor([[0, 19]]),
        indices_latents_history_mid=torch.tensor([[17, 18]]),
        indices_latents_history_long=torch.arange(1, 17)[None],
        latents_history_short=randn(1, 16, 2, height, width),
        latents_history_mid=randn(1, 16, 2, height, width),
        latents_history_long=randn(1, 16, 16, height, width),
    )


class _BlockInputs(Exception):
    pass


class TestSequenceParallelShard(_TestBase):
    def _forward(self, model, inputs, sp):
        """The DiT as SP rank 1 of ``sp``; no collective runs unsharded."""
        batch = SimpleNamespace(enable_sequence_shard=True, extra={})
        with (
            patch.object(model, "sp_size", sp),
            patch.object(dit, "get_sp_group", lambda: SimpleNamespace(rank_in_group=1)),
            set_forward_context(
                current_timestep=0, attn_metadata=None, forward_batch=batch
            ),
            torch.no_grad(),
        ):
            return model(**inputs)

    def test_divisible_current_tokens_are_sharded(self):
        model = _model()

        def stop(module, args, kwargs):
            raise _BlockInputs(args[4], kwargs["skip_sequence_parallel"])

        model.blocks[0].register_forward_pre_hook(stop, with_kwargs=True)
        with self.assertRaises(_BlockInputs) as caught:
            self._forward(model, _dit_inputs(), 2)  # 9 x 4 x 4 = 144 current tokens
        self.assertEqual(caught.exception.args, (72, False))

    def test_output_norm_takes_the_step_timestep_under_sp(self):
        model, inputs, temb = _model(), _dit_inputs(), []
        model.norm_out.register_forward_pre_hook(lambda m, a: temb.append(a[1]))
        gather = lambda x, dim: x.repeat(1, 2, 1)  # noqa: E731
        with (
            patch.object(model, "blocks", torch.nn.ModuleList()),
            patch.object(dit, "sequence_model_parallel_all_gather", gather),
        ):
            self._forward(model, inputs, 2)  # history rows hold timestep 0
            _run_dit(model, inputs)
        self.assertTrue(torch.equal(temb[0][:, -1], temb[1][:, -1]))

    def test_indivisible_current_tokens_run_unsharded(self):
        model = _model()
        inputs = _dit_inputs(height=6, width=6)  # 9 x 3 x 3 = 81 current tokens
        expected = _run_dit(model, inputs)
        with contextlib.ExitStack() as stack:
            attns = [
                stack.enter_context(
                    patch.object(b.attn1.attn, "forward", wraps=b.attn1.attn.forward)
                )
                for b in model.blocks
            ]
            actual = self._forward(model, inputs, 2)
        self.assertTrue(torch.equal(actual, expected))
        for attn in attns:
            self.assertIs(
                attn.call_args.kwargs["skip_sequence_parallel_override"], True
            )


def _run_dit(model, inputs):
    batch = SimpleNamespace(enable_sequence_shard=False, extra={})
    with set_forward_context(
        current_timestep=0, attn_metadata=None, forward_batch=batch
    ):
        with torch.no_grad():
            return model(**inputs)


class _LaneStrides(TorchDispatchMode):
    """Records the last-dim stride of every op output wider than one lane."""

    def __init__(self):
        super().__init__()
        self.strides = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in tree_leaves(out):
            if isinstance(t, torch.Tensor) and t.dim() and t.shape[-1] > 1:
                self.strides.add(t.stride(-1))
        return out


class TestRope(_TestBase):
    def test_matches_stock_bitwise_without_stride_2_lanes(self):
        x = torch.randn(1, 37, 3, 128, generator=torch.Generator().manual_seed(4)) * 4
        x = x.bfloat16()
        table = dit.HeliosRotaryPosEmbed((44, 42, 42), 10000.0)(
            torch.arange(5, 6)[None], 6, 7, torch.device("cpu")
        )
        # Contiguous, as the DiT hands it to the blocks.
        table = table.flatten(2).transpose(1, 2)[:, :37].contiguous()
        expected = helios._STOCK_ROTARY(x, table)
        lanes = _LaneStrides()
        with _installed():
            self.assertIs(
                dit.apply_rotary_emb_transposed, helios.apply_rotary_emb_transposed
            )
            with lanes:
                actual = dit.apply_rotary_emb_transposed(x, table)
        self.assertEqual(lanes.strides, {1})
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual, expected))

    def test_bf16_dit_output_matches_stock_bitwise(self):
        # The history tables are padded and pooled from the frequency tables.
        model = _model(torch.bfloat16)
        inputs = _dit_inputs(dtype=torch.bfloat16)
        expected = _run_dit(model, inputs)
        with (
            _installed(),
            patch.object(
                helios, "_pair_swap_matrix", wraps=helios._pair_swap_matrix
            ) as pair_swap,
        ):
            actual = _run_dit(model, inputs)
        self.assertEqual(pair_swap.call_count, 4)  # q and k in two blocks
        self.assertTrue(torch.equal(actual, expected))

    def test_other_dtypes_and_devices_run_stock(self):
        x = torch.randn(1, 5, 2, 32)
        table = torch.randn(1, 5, 64)
        for device_types, x_in, table_in in (
            (("cpu",), x, table),
            (("tpu",), x.bfloat16(), table),
            (("cpu",), x.bfloat16(), table.bfloat16()),
        ):
            with (
                _installed(device_types),
                patch.object(helios, "_pair_swap_matrix") as pair_swap,
            ):
                actual = dit.apply_rotary_emb_transposed(x_in, table_in)
            pair_swap.assert_not_called()
            expected = helios._STOCK_ROTARY(x_in, table_in)
            self.assertTrue(torch.equal(actual, expected))


def _stage_run(stage_config, transformer, server_args, steps=(2, 2, 2), seeds=None):
    """One request through ``HeliosChunkedDenoisingStage.forward``: one chunk."""
    config = stage_config(
        precision="fp32", pyramid_num_inference_steps_list=list(steps)
    )
    scheduler_config = BASE if not config.is_enable_stage2 else MID
    server_args = SimpleNamespace(**vars(server_args), pipeline_config=config)
    stage = stage_module.HeliosChunkedDenoisingStage(
        transformer, Scheduler(**scheduler_config)
    )
    batch = SimpleNamespace(
        sampling_params=SimpleNamespace(),
        scheduler=None,
        latents=torch.zeros(1),
        guidance_scale=1.0,
        num_inference_steps=steps[0],
        prompt_embeds=[
            torch.randn(1, 7, 24, generator=torch.Generator().manual_seed(3))
        ],
        negative_prompt_embeds=None,
        height=64,
        width=64,
        num_frames=33,
        generator=torch.Generator().manual_seed(0),
        metrics=None,
        perf_dump_path=None,
        enable_sequence_shard=False,
        extra={},
        record_stage_iterations=lambda *args: None,
        seeds=seeds,
    )
    with torch.no_grad():
        stage.forward(batch, server_args)
    return batch


class TestDenoisingStage(_TestBase):
    def test_dit_timestep_is_a_fresh_tensor(self):
        seen = []

        def record(module, args, kwargs):
            t = kwargs["timestep"]
            own_storage = t.untyped_storage().nbytes() == t.numel() * t.element_size()
            seen.append((t.storage_offset(), t.stride(), own_storage))

        model = _model()
        model.register_forward_pre_hook(record, with_kwargs=True)
        compile_off = SimpleNamespace(enable_torch_compile=False)
        for stage_config, calls in (
            (helios_config.HeliosT2VConfig, 2),
            (helios_config.HeliosMidConfig, 6),
        ):
            seen.clear()
            _stage_run(stage_config, model, compile_off)
            self.assertEqual(len(seen), calls)
            self.assertEqual(set(seen), {(0, (1,), True)})

    def test_pyramid_renoise_follows_the_request_seed(self):
        model = _model()
        compile_off = SimpleNamespace(enable_torch_compile=False)

        def run(seed, global_seed):
            torch.manual_seed(global_seed)  # the global CPU RNG plays no part
            batch = _stage_run(
                helios_config.HeliosMidConfig, model, compile_off, seeds=[seed]
            )
            return batch.latents

        first = run(5, global_seed=1)
        self.assertTrue(torch.equal(first, run(5, global_seed=2)))
        self.assertFalse(torch.equal(first, run(6, global_seed=1)))

    def test_compile_flags(self):
        graphs = []

        def count(gm, example_inputs):
            graphs.append(gm)
            return gm.forward

        cases = (
            (dict(enable_torch_compile=False, regional_compile=True), set()),
            (
                dict(enable_torch_compile=True, regional_compile=True),
                {"blocks.0", "blocks.1"},
            ),
            (dict(enable_torch_compile=True, regional_compile=False), {""}),
        )
        for flags, compiled in cases:
            with (
                self.subTest(**flags),
                patch.object(
                    stage_module,
                    "resolve_torch_compile_kwargs",
                    return_value=({"backend": count}, None),
                ),
            ):
                torch._dynamo.reset()
                graphs.clear()
                model = _model()
                _stage_run(
                    helios_config.HeliosT2VConfig, model, SimpleNamespace(**flags)
                )
                names = {
                    n
                    for n, m in model.named_modules()
                    if m._compiled_call_impl is not None
                }
                self.assertEqual(names, compiled)
                self.assertEqual(bool(graphs), bool(compiled))


if __name__ == "__main__":
    unittest.main()
