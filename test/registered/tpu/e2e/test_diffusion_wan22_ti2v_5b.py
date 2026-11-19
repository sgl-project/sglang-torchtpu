"""Wan2.2-TI2V-5B text-to-video on one TPU chip (nightly).

The diffusion server harness is device-agnostic and picks the TPU platform up
through ``current_platform`` at server launch. With no TPU perf or consistency
baseline, the guard is the decoded video itself: the requested frame count and
size, and pixel statistics that blank, NaN, frozen and noise videos fail, noise
including a random latent decoded by the VAE.
"""

from __future__ import annotations

import io

import numpy as np
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

# est_time covers a cold Hugging Face pull; /weights is used when hydrated.
register_tpu_ci(
    est_time=2700,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "wan2.2-ti2v-5b", "diffusion"],
)

NUM_FRAMES = 9
FRAME_SHAPE = (480, 832, 3)
# Bounds on the decoded uint8 frames after the server's libx264 save. This
# request measures std 86, temporal diff 4.2 and neighbour ratio 0.037 (CPU run,
# same seed); real Wan2.2 TPU videos cut to 9x480x832 measure std 39-79,
# temporal diff 1.0-33 and neighbour ratio 0.012-0.056.
# Pixel std of every frame: black, constant and NaN output (cast to black)
# measure 0, so a clip that turns blank partway through fails too.
MIN_PIXEL_STD = 10.0
# Mean |frame[t+1] - frame[t]|: frozen copies of those videos measure 0.02 at most.
MIN_TEMPORAL_DIFF = 0.1
# Mean |pixel - right neighbour| / std, per frame: a random latent decoded by the
# VAE, what a DiT or scheduler that never denoises yields, measures 0.13.
MAX_NEIGHBOUR_DIFF_RATIO = 0.08

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
            # torch_tpu's blocking host-to-device copy; drop it together with
            # TpuOOTPlatform.drain_device_queue_after_request once TorchTPU fixes it.
            extras={"num_inference_steps": 4, "seed": 0, "generator_device": "cpu"},
        ),
        run_perf_check=False,
        run_consistency_check=False,
        run_component_accuracy_check=False,
    ),
]


def _decode_frames(video_bytes: bytes) -> np.ndarray:
    """Every frame as RGB uint8, ``[frames, height, width, 3]``."""
    import av

    with av.open(io.BytesIO(video_bytes)) as container:
        return np.stack(
            [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        )


def _assert_plausible_video(frames: np.ndarray) -> None:
    expected_shape = (NUM_FRAMES, *FRAME_SHAPE)
    assert frames.shape == expected_shape, (
        f"decoded {frames.shape}, expected {expected_shape}"
    )
    pixels = frames.astype(np.float32)
    frame_std = pixels.std(axis=(1, 2, 3))
    temporal = float(np.abs(np.diff(pixels, axis=0)).mean())
    frame_ratio = np.abs(np.diff(pixels, axis=2)).mean(axis=(1, 2, 3)) / np.maximum(
        frame_std, 1e-6
    )
    worst_std, worst_ratio = int(frame_std.argmin()), int(frame_ratio.argmax())
    print(
        f"video {frames.shape}: mean {pixels.mean():.2f}, "
        f"min frame std {frame_std[worst_std]:.2f}, temporal diff {temporal:.3f}, "
        f"max frame neighbour diff {frame_ratio[worst_ratio]:.3f} x std"
    )
    assert frame_std[worst_std] > MIN_PIXEL_STD, (
        f"frame {worst_std} pixel std {frame_std[worst_std]:.2f}: blank or constant"
    )
    assert temporal > MIN_TEMPORAL_DIFF, f"temporal diff {temporal:.3f}: frozen video"
    assert frame_ratio[worst_ratio] < MAX_NEIGHBOUR_DIFF_RATIO, (
        f"frame {worst_ratio} neighbour diff {frame_ratio[worst_ratio]:.3f} x std: noise"
    )


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
        capsys: pytest.CaptureFixture[str],
    ):
        generate_fn = get_generate_fn(
            model_path=case.server_args.model_path,
            modality=case.server_args.modality,
            sampling_params=case.sampling_params,
        )

        # This smoke test has no TPU latency baseline or load-time measurement.
        # Performance validation requires both even with run_perf_check=False.
        _, content = self.run_and_collect(
            diffusion_server, case.id, generate_fn, collect_perf=False
        )

        self._test_v1_models_endpoint(diffusion_server, case)

        frames = _decode_frames(content)
        # Uncaptured, so the stats of passing runs reach the nightly log too.
        with capsys.disabled():
            _assert_plausible_video(frames)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
