"""Wan2.2-TI2V-5B text-to-video on one TPU chip (nightly).

Mirrors ``test/registered/xpu/test_xpu_zimage_turbo.py``: the diffusion server
harness is device-agnostic and picks the TPU platform up through
``current_platform`` at server launch. No TPU perf or consistency baseline
exists yet, so the guard is a decodable video with the requested frame count.
"""

from __future__ import annotations

import io
import logging

import pytest
import torch

from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerContext,
    get_generate_fn,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

logger = logging.getLogger(__name__)

# est_time covers a cold Hugging Face pull; /weights is used when hydrated.
register_tpu_ci(
    est_time=2700,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "wan2.2-ti2v-5b", "diffusion"],
)

NUM_FRAMES = 9

TPU_WAN22_CASES = [
    DiffusionTestCase(
        "wan2_2_ti2v_5b_t2v",
        DiffusionServerArgs(
            model_path=resolve_tpu_model_path("Wan-AI/Wan2.2-TI2V-5B-Diffusers"),
            modality="video",
            num_gpus=1,
        ),
        DiffusionSamplingParams(
            prompt="A curious raccoon peers through a field of sunflowers",
            output_size="832x480",
            num_frames=NUM_FRAMES,
            # Workaround: seeding a TPU torch.Generator can deadlock on the GIL in
            # torch_tpu's blocking host-to-device copy; drop once TorchTPU fixes it.
            extras={"num_inference_steps": 4, "seed": 0, "generator_device": "cpu"},
        ),
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
]


def _decoded_frame_count(video_bytes: bytes) -> int:
    import av

    with av.open(io.BytesIO(video_bytes)) as container:
        return sum(1 for _ in container.decode(video=0))


@pytest.mark.skipif(not hasattr(torch, "tpu"), reason="torch_tpu found no TPU")
class TestWan22TI2V5BTPU(DiffusionServerBase):
    """Nightly single-chip TPU text-to-video generation."""

    @pytest.fixture(params=TPU_WAN22_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        return request.param

    def test_diffusion_generation(
        self,
        case: DiffusionTestCase,
        diffusion_server: ServerContext,
    ):
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )

        perf_record, content = self.run_and_collect(
            diffusion_server, case.id, generate_fn
        )

        self._validate_and_record(case, perf_record)
        self._test_v1_models_endpoint(diffusion_server, case)

        frames = _decoded_frame_count(content)
        logger.info("decoded %d frames, %d bytes", frames, len(content))
        assert frames == NUM_FRAMES, f"expected {NUM_FRAMES} frames, decoded {frames}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
