"""Qwen3-VL-8B EPD MMMU-Pro (vision) gate on one host.

The vision tower runs in a standalone encode server (--encoder-only) on chips
0-1 and the language model (--language-only) on chips 2-3. A language-only
server rejects images it cannot dispatch (503), so every scored answer went
through the encode server: a broken encoder bootstrap, compile policy or
embedding transfer fails the launch or drops the score.
"""

import json
import os
import subprocess
import time
import unittest
import urllib.request

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.evals.model_config import run_model_config

register_tpu_ci(
    est_time=3600,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "qwen3-vl-8b", "multimodal", "epd"],
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 4)))
LAUNCH_SERVER = os.path.join(REPO, "benchmark", "tpu", "launch_server.sh")
LOG_DIR = os.environ.get("EPD_LOG_DIR", "/tmp/epd")

MODEL = os.environ.get("MODEL", "/weights/Qwen3-VL-8B-Instruct")
SERVED_MODEL_NAME = "qwen3-vl-8b"
ENCODER_PORT = 47904
LM_PORT = 47903
ENCODER_URL = f"http://127.0.0.1:{ENCODER_PORT}"
LM_URL = f"http://127.0.0.1:{LM_PORT}"
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "3000"))

# Each engine takes two chips (four v7x chiplets) of the host.
TP_SIZE = 4
IMAGE_LIMITS = {"min_pixels": 65536, "max_pixels": 313600}

# The qwen3_vl_8b recipe's serving shape, cut to the gate's concurrency.
COMMON_ENV = {
    "MODEL": MODEL,
    "TP_SIZE": str(TP_SIZE),
    "QUANTIZATION": "unquant",
    "KV_CACHE_DTYPE": "bfloat16",
    "PAGE_SIZE": "128",
    "CONTEXT_LEN": "9216",
    "CHUNKED_PREFILL_SIZE": "8192",
    "MAX_PREFILL_TOKENS": "8192",
    "MEM_FRACTION_STATIC": "0.30",
    "CONC": "8",
    "TORCH_COMPILE_MAX_BS": "8",
    "TPU_COMPILE_TOKENS": "512 1024 2048 8192",
    "HOST": "127.0.0.1",
    "FORCE_SINGLE_HOST": "1",
    # Pins each rank to base_gpu_id + local_rank, so the engines split the chips.
    "SGLANG_TPU_PD_COLOCATED": "1",
    # A fresh pod has nothing to reuse, and two engines would share one cache.
    "ENABLE_COMPILE_CACHE": "0",
    "TORCH_LOGS": "-dynamo",
}
COMMON_ARGS = (
    f"--served-model-name {SERVED_MODEL_NAME} --trust-remote-code "
    "--watchdog-timeout 3600 --dist-timeout 3600"
)

ENCODER_ENV = {
    **COMMON_ENV,
    "PORT": str(ENCODER_PORT),
    "DIST_PORT": "10011",
    "MASTER_PORT": "29511",
    "SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR": "/dev/shm/inductor_encoder",
    "EXTRA_ARGS": (
        f"--encoder-only --base-gpu-id 0 {COMMON_ARGS} --mm-process-config "
        + json.dumps({"vision_config": {"image": IMAGE_LIMITS}}, separators=(",", ":"))
    ),
}
LM_ENV = {
    **COMMON_ENV,
    "PORT": str(LM_PORT),
    "DIST_PORT": "10021",
    "MASTER_PORT": "29521",
    "SGLANG_BASE_TORCHINDUCTOR_CACHE_DIR": "/dev/shm/inductor_lm",
    # Every image goes to the encoder; the local ViT is never run.
    "SGLANG_TPU_VIT_PRECOMPILE_SIZES": "",
    # The encoder is up before the LM starts; never evict it from the bootstrap.
    "SGLANG_ENCODER_BOOTSTRAP_HEALTH_CHECK_TIMEOUT": "30",
    "SGLANG_ENCODER_BOOTSTRAP_EVICTED_TTL": "0",
    "EXTRA_ARGS": (
        f"--language-only --encoder-urls {ENCODER_URL} --base-gpu-id {TP_SIZE} "
        f"{COMMON_ARGS} --mm-process-config "
        + json.dumps({"image": IMAGE_LIMITS}, separators=(",", ":"))
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "qwen3-vl-8b-epd-mmmu-pro-vision",
    "variant": "EPD encoder TP4 + LM TP4",
    "gates": [
        {
            "dataset": "mmmu_pro_vision",
            "label": "mmmu_pro_vision",
            "report_only": True,
            "num_examples": 200,
            "num_threads": 8,
            # Greedy CoT answers are long; 512 truncates about half of them.
            "max_tokens": 4096,
            "temperature": 0.0,
        }
    ],
}


def _launch(name: str, env: dict[str, str]) -> subprocess.Popen:
    os.makedirs(LOG_DIR, exist_ok=True)
    log = open(os.path.join(LOG_DIR, f"{name}.log"), "w")
    print(f"[epd] launching {name}: {env['EXTRA_ARGS']}", flush=True)
    return subprocess.Popen(
        ["bash", LAUNCH_SERVER],
        cwd=REPO,
        env={**os.environ, **env},
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def _wait_healthy(name: str, url: str, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + HEALTH_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited with {proc.returncode}; {_tail(name)}")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as resp:
                if resp.status == 200:
                    print(f"[epd] {name} healthy", flush=True)
                    return
        except OSError:
            pass
        time.sleep(10)
    raise TimeoutError(f"{name} not healthy in {HEALTH_TIMEOUT}s; {_tail(name)}")


def _tail(name: str, lines: int = 80) -> str:
    with open(os.path.join(LOG_DIR, f"{name}.log"), errors="replace") as f:
        return "log tail:\n" + "".join(f.readlines()[-lines:])


def _get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


class TestQwen3VL8BSingleHostEPD(unittest.TestCase):
    procs: list[subprocess.Popen] = []

    @classmethod
    def setUpClass(cls) -> None:
        # The LM registers the encoder at boot, so the encoder must be up first.
        encoder = _launch("encoder", ENCODER_ENV)
        cls.procs.append(encoder)
        try:
            _wait_healthy("encoder", ENCODER_URL, encoder)
            lm = _launch("lm", LM_ENV)
            cls.procs.append(lm)
            _wait_healthy("lm", LM_URL, lm)
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for proc in reversed(cls.procs):
            kill_process_tree(proc.pid)
        cls.procs = []

    def test_lm_is_language_only_tp4(self) -> None:
        info = _get_json(f"{LM_URL}/server_info")
        self.assertTrue(info.get("language_only"))
        self.assertEqual(info.get("encoder_urls"), [ENCODER_URL])
        self.assertEqual(info.get("tp_size"), TP_SIZE)

    def test_mmmu_pro_vision_through_encoder(self) -> None:
        run_model_config(MODEL_CONFIG, base_url=LM_URL)


if __name__ == "__main__":
    unittest.main()
