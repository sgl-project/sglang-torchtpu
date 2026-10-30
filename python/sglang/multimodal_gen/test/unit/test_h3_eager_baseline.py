"""Request and media acceptance contracts for the standard eager baseline."""

import importlib.util
import itertools
import shutil
import subprocess
from pathlib import Path

import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.request_validation import (
    minimax_h3_validate_canonical_request,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.resolved_plan import (
    minimax_h3_resolve_plan,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_time_shift_sigmas,
)


def _script(name):
    path = (
        Path(__file__).resolve().parents[5]
        / "scripts/torchtpu/h3_eager_baseline"
        / f"{name}.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standard_request_preserves_released_shape_and_schedule():
    request = _script("client").REQUEST
    canonical = minimax_h3_validate_canonical_request(
        **{
            key: request[key]
            for key in (
                "task",
                "prompt",
                "conditions",
                "target",
                "seed",
                "flow_shift",
                "audio_flow_shift",
            )
        }
    )
    shape = minimax_h3_resolve_plan(canonical).shape
    assert (shape["width"], shape["height"], shape["frame_count"], shape["fps"]) == (
        1344,
        768,
        107,
        24,
    )
    assert (shape["video_latent_t"], shape["audio_latent_t"]) == (32, 178)
    for shift in (request["flow_shift"], request["audio_flow_shift"]):
        sigmas = minimax_h3_time_shift_sigmas(
            num_steps=request["num_inference_steps"],
            shift_scale=shift,
        )
        assert len(sigmas) == 50
        assert len(list(itertools.pairwise(sigmas))) == 49
        assert sigmas[0] == 1 and sigmas[-1] == 0


@pytest.fixture
def encoded_fixture(tmp_path):
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("media validator tests require ffmpeg and ffprobe")

    def encode(frames=107, sample_rate=32000, channels=2):
        path = tmp_path / f"fixture-{frames}-{sample_rate}-{channels}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc2=size=128x64:rate=24",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:sample_rate={sample_rate}",
                "-frames:v",
                str(frames),
                "-t",
                str(frames / 24),
                "-c:v",
                "libx264",
                "-threads",
                "1",
                "-c:a",
                "aac",
                "-ac",
                str(channels),
                str(path),
            ],
            capture_output=True,
            check=True,
        )
        return path

    return encode


def test_media_validator_decodes_valid_fixture(encoded_fixture):
    result = _script("validate_media").validate_media(
        encoded_fixture(),
        width=128,
        height=64,
    )
    assert result["status"] == "PASS_MEDIA_ONLY"
    assert result["complete_decode"] == "PASS"


@pytest.mark.parametrize(
    "frames,sample_rate,channels,error",
    [
        (96, 32000, 2, "nb_read_frames"),
        (107, 48000, 2, "32 kHz"),
        (107, 32000, 1, "stereo"),
    ],
)
def test_media_validator_rejects_wrong_contract(
    encoded_fixture,
    frames,
    sample_rate,
    channels,
    error,
):
    with pytest.raises(ValueError, match=error):
        _script("validate_media").validate_media(
            encoded_fixture(frames, sample_rate, channels),
            width=128,
            height=64,
        )
