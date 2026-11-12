# SPDX-License-Identifier: Apache-2.0
"""Standard Full/base H3 T2VA in compiled and eager modes on eight devices."""

from __future__ import annotations

import array
import importlib.metadata
import importlib.util
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "scripts/torchtpu/h3_eager_baseline"


def _port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait_ready(process, url: str, log: Path, timeout: int = 2400) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"H3 service exited with {process.returncode}; log: {log}"
            )
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(2)
    raise TimeoutError(f"H3 service readiness timed out; log: {log}")


@pytest.mark.skipif(not hasattr(torch, "tpu"), reason="requires eight TPU devices")
@pytest.mark.parametrize("compile_mode", ["compiled", "eager"])
def test_standard_full_base_t2va(compile_mode):
    model = Path(os.environ.get("MINIMAX_H3_MODEL_PATH", "/weights/MiniMax-H3"))
    if not (model / "FL2VA/model_index.json").is_file():
        raise FileNotFoundError(
            "Hydrate the Full/base MiniMax-H3 assets under the weights root, "
            "or set MINIMAX_H3_MODEL_PATH; online model downloads are disabled."
        )
    artifacts = Path(tempfile.mkdtemp(prefix=f"minimax-h3-{compile_mode}-"))
    print(f"H3 artifacts: {artifacts}", flush=True)
    recipe = SCRIPTS.parent / f"h3_{compile_mode}_baseline/server.json"
    config = json.loads(recipe.read_text())
    ports = set()
    while len(ports) < 4:
        ports.add(_port())
    http_port, scheduler_port, master_port, rendezvous_port = sorted(ports)
    config.update(
        model_path=str(model),
        host="127.0.0.1",
        port=http_port,
        scheduler_port=scheduler_port,
        master_port=master_port,
        dist_init_addr=f"127.0.0.1:{rendezvous_port}",
        output_path=str(artifacts / "outputs"),
        input_save_path=str(artifacts / "inputs"),
    )
    config_path = artifacts / "server.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    environment = os.environ.copy()
    environment.update(
        JAX_PLATFORMS="cpu",
        TORCH_TPU_DEVICES_PER_HOST="8",
        SGLANG_DIFFUSION_PLATFORM_OVERRIDE="tpu",
        MINIMAX_H3_VAE_DECODER_FUSED_NORM="0",
        MINIMAX_H3_USE_FUSED_NORM="false",
        MINIMAX_H3_VAE_DECODER_VIT_FF_TORCH_COMPILE="0",
        MINIMAX_H3_VAE_DECODER_VIT_ROPE_TORCH_COMPILE="0",
        SGLANG_DIFFUSION_DISABLE_VAE_DECODER_STORE="1",
    )
    console = Path(sys.executable).with_name("sglang")
    entry = (
        [str(console)]
        if console.is_file()
        else [
            sys.executable,
            "-c",
            "from sglang.cli.main import main; main()",
        ]
    )
    command = [
        *entry,
        "serve",
        "--model-type",
        "diffusion",
        "--model-path",
        str(model),
        "--config",
        str(config_path),
        "--node-rank",
        "0",
    ]
    dependencies = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torch-tpu",
            "libtpu",
            "transformers",
            "diffusers",
            "sglang-tpu",
        )
    }
    (artifacts / "launch.json").write_text(
        json.dumps(
            {
                "command": command,
                "dependencies": dependencies,
                "config": config,
            },
            indent=2,
        )
        + "\n"
    )
    url = f"http://127.0.0.1:{http_port}"
    log = artifacts / "server.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            _wait_ready(process, url, log)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "client.py"),
                    "--url",
                    url,
                    "--receipts",
                    str(artifacts),
                    "--timeout",
                    "3600",
                ],
                check=True,
                env=environment,
                timeout=3720,
            )
            spec = importlib.util.spec_from_file_location(
                "h3_media", SCRIPTS / "validate_media.py"
            )
            media = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(media)
            video = artifacts / "h3-standard-4s.mp4"
            result = media.validate_media(video)
            if compile_mode == "compiled":
                assert (
                    "Enabled regional torch.compile for 52 submodules in MiniMaxH3DiTModel"
                    in log.read_text()
                ), "H3 transformer blocks were not registered for regional compilation"
            audio = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-xerror",
                    "-i",
                    str(video),
                    "-map",
                    "0:a:0",
                    "-f",
                    "f32le",
                    "-",
                ],
                capture_output=True,
                check=True,
            )
            samples = array.array("f", audio.stdout)
            assert samples and all(math.isfinite(value) for value in samples)
            rms = math.sqrt(sum(value * value for value in samples) / len(samples))
            assert rms > 1e-6, "Output audio is silent"
            result["audio_rms"] = rms
            (artifacts / "media-validation.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {key: value for key, value in result.items() if key != "ffprobe"}
                ),
                flush=True,
            )
        finally:
            # Only terminate this test's service group and its worker children.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=30)


if __name__ == "__main__":
    extra = ["-x" if arg == "-f" else arg for arg in sys.argv[1:]]
    sys.exit(pytest.main([__file__, "-v", "-s", *extra]))
