"""Wan2.2-T2V-A14B request overrides: ``boundary_ratio`` reaches the sampling params
(the denoiser's expert switch reads it) and ``flow_shift`` the scheduler's timesteps;
without them the server's values stay in effect."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.wan import Wan2_2_T2V_A14B_Config
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.wan import Wan2_2_T2V_A14B_SamplingParam
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    VideoGenerationsRequest,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    request_model_kwargs,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines import wan_pipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    base,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    timestep_preparation as module,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")

_SERVER_SHIFT = Wan2_2_T2V_A14B_Config().flow_shift  # 12.0


def _stage_timesteps(flow_shift, steps=40):
    args = SimpleNamespace(pipeline_config=Wan2_2_T2V_A14B_Config())
    with patch.object(base, "get_global_server_args", return_value=args):
        stage = module.TimestepPreparationStage(
            FlowUniPCMultistepScheduler(shift=_SERVER_SHIFT),
            prepare_extra_set_timesteps_kwargs=[wan_pipeline.request_flow_shift],
        )
    batch = Req(
        sampling_params=Wan2_2_T2V_A14B_SamplingParam(
            num_inference_steps=steps, flow_shift=flow_shift
        )
    )
    with patch.object(
        module, "get_local_torch_device", return_value=torch.device("cpu")
    ):
        stage.forward(batch, args)
    return batch.timesteps


def _scheduler_timesteps(shift, steps=40):
    scheduler = FlowUniPCMultistepScheduler(shift=_SERVER_SHIFT)
    scheduler.set_timesteps(steps, device="cpu", shift=shift)
    return scheduler.timesteps


class TestWan22RequestOverrides(CustomTestCase):
    def test_boundary_ratio_is_a_wan22_a14b_request_field(self):
        request = VideoGenerationsRequest(prompt="a fox in snow", boundary_ratio=0.9)
        kwargs = request_model_kwargs(request, Wan2_2_T2V_A14B_SamplingParam, "video")
        self.assertEqual(kwargs, {"boundary_ratio": 0.9})
        params = Wan2_2_T2V_A14B_SamplingParam(**kwargs)
        self.assertEqual(Req(sampling_params=params).boundary_ratio, 0.9)
        self.assertEqual(request_model_kwargs(request, SamplingParams, "video"), {})
        # Not sent: the checkpoint's boundary stays in effect.
        request = VideoGenerationsRequest(prompt="a fox in snow")
        self.assertEqual(
            request_model_kwargs(request, Wan2_2_T2V_A14B_SamplingParam, "video"), {}
        )

    def test_timesteps_follow_the_request_flow_shift(self):
        default = _stage_timesteps(None)
        self.assertTrue(torch.equal(default, _scheduler_timesteps(None)))
        # The spec's 12.0 is the server's default: the same schedule, bit for bit.
        self.assertTrue(torch.equal(_stage_timesteps(12.0), default))
        shifted = _stage_timesteps(5.0)
        self.assertTrue(torch.equal(shifted, _scheduler_timesteps(5.0)))
        self.assertFalse(torch.equal(shifted, default))

    def test_pipeline_passes_the_request_flow_shift(self):
        pipeline = wan_pipeline.WanPipeline.__new__(wan_pipeline.WanPipeline)
        for name in (
            "add_stage",
            "add_standard_text_encoding_stage",
            "add_standard_latent_preparation_stage",
            "add_standard_timestep_preparation_stage",
            "add_progressive_denoising_stage",
            "add_standard_decoding_stage",
        ):
            setattr(pipeline, name, MagicMock())
        args = SimpleNamespace(pipeline_config=Wan2_2_T2V_A14B_Config())
        with patch.object(base, "get_global_server_args", return_value=args):
            wan_pipeline.WanPipeline.create_pipeline_stages(pipeline, args)
        pipeline.add_standard_timestep_preparation_stage.assert_called_once_with(
            prepare_extra_kwargs=[wan_pipeline.request_flow_shift]
        )


if __name__ == "__main__":
    unittest.main()
