"""P/D disaggregation with a pipeline-parallel prefill on one TPU host.

A two-stage Qwen3-0.6B prefill, one device per stage, hands its KV over Raiden
to a two-device tensor-parallel decode on the same host. Only the first stage
polls a bootstrapping room, so a room starts once every stage has the decode's
request, and the decode then reads both stages' layers, split by KV head across
its ranks. Each request recalls a code word from the middle of its prompt: the
prefill samples only the first token of the answer, and the decode can finish
the word only from the transferred KV.
"""

import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlparse

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    popen_with_error_check,
    terminate_and_kill_process_tree,
)
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path
from sglang.utils import wait_for_http_ready

register_tpu_ci(
    est_time=240,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "qwen3-0.6b", "pd", "pp"],
)

MODEL = resolve_tpu_model_path("Qwen/Qwen3-0.6B")
PP_SIZE = 2
DECODE_TP_SIZE = 2
CHUNK_TOKENS = 512
MAX_BATCH = 8

_HOST = urlparse(DEFAULT_URL_FOR_TEST).hostname
_BASE_PORT = urlparse(DEFAULT_URL_FOR_TEST).port
LB_URL = f"http://{_HOST}:{_BASE_PORT}"
PREFILL_URL = f"http://{_HOST}:{_BASE_PORT + 100}"
DECODE_URL = f"http://{_HOST}:{_BASE_PORT + 200}"
BOOTSTRAP_PORT = _BASE_PORT + 500

# One extend bucket (the prefill chunk) and one batch bucket keep startup
# compilation short; the batch bucket also caps the running requests.
COMMON_ARGS = [
    "--device",
    "tpu",
    "--dtype",
    "bfloat16",
    "--grammar-backend",
    "none",
    "--disaggregation-transfer-backend",
    "raiden",
    # A disaggregated engine's warmup request needs its peer and the router.
    "--skip-server-warmup",
    "--enable-torch-compile",
    "--page-size",
    "64",
    "--context-length",
    "2048",
    "--chunked-prefill-size",
    str(CHUNK_TOKENS),
    "--max-total-tokens",
    "16384",
    "--torch-compile-max-bs",
    str(MAX_BATCH),
    "--tpu-compile-bs",
    str(MAX_BATCH),
    "--tpu-compile-tokens",
    str(CHUNK_TOKENS),
]

PREFILL_ARGS = COMMON_ARGS + [
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-bootstrap-port",
    str(BOOTSTRAP_PORT),
    "--pp-size",
    str(PP_SIZE),
    "--tp-size",
    "1",
    "--pp-async-batch-depth",
    "1",
    "--base-gpu-id",
    "0",
    "--nccl-port",
    str(_BASE_PORT + 300),
]

DECODE_ARGS = COMMON_ARGS + [
    "--disaggregation-mode",
    "decode",
    "--tp-size",
    str(DECODE_TP_SIZE),
    "--base-gpu-id",
    str(PP_SIZE),
    "--nccl-port",
    str(_BASE_PORT + 400),
    # No HiCache host pool: these few short requests never retract.
    "--disaggregation-decode-retraction-backup",
    "cpu_tensor",
]


def _engine_env(*, master_port: int, raiden_port: int) -> dict[str, str]:
    return {
        # Each rank binds device --base-gpu-id + rank, so the engines share the host.
        "SGLANG_TPU_PD_COLOCATED": "1",
        # Each engine rendezvouses its own device world and Raiden control ports.
        "MASTER_PORT": str(master_port),
        "SGLANG_RAIDEN_KV_PORT": str(raiden_port),
        # Each engine sizes its host staging as if it had the host to itself.
        "SGLANG_RAIDEN_HOST_POOL_GB": "4",
    }


PREFILL_ENV = _engine_env(master_port=_BASE_PORT + 600, raiden_port=31000)
DECODE_ENV = {
    **_engine_env(master_port=_BASE_PORT + 700, raiden_port=31100),
    # The decode is not itself a pipeline.
    "SGLANG_RAIDEN_PIPELINE_TRANSFER": "1",
}

FILLER = "The quick brown fox jumps over the lazy dog. "
WORDS = (
    "TANGERINE",
    "PERIWINKLE",
    "ZEPPELIN",
    "MARMALADE",
    "OBSIDIAN",
    "KALEIDOSCOPE",
    "SASSAFRAS",
    "QUICKSILVER",
)


def _needle_prompt(word: str, repeats: int) -> str:
    side = FILLER * repeats
    return (
        f"{side}The secret code word is {word}. Remember it. {side}"
        "Question: what is the secret code word? Answer: The secret code word is"
    )


def _generate(text: str) -> dict:
    response = requests.post(
        f"{LB_URL}/generate",
        json={
            "text": text,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 16},
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def _skip_if_no_tpu():
    try:
        import importlib.util

        if importlib.util.find_spec("torch_tpu") is None:
            return True
    except Exception:
        return True
    return False


@unittest.skipIf(_skip_if_no_tpu(), "torch_tpu not available")
class TestTpuQwen306bSinglehostPdPrefillPp(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # setdefault: mode B runs these suites with SGLANG_PLATFORM=tpu_oot.
        os.environ.setdefault("SGLANG_PLATFORM", "tpu")
        cls.prefill = cls.decode = cls.lb = None
        try:
            cls.prefill = popen_launch_pd_server(
                MODEL,
                PREFILL_URL,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=PREFILL_ARGS,
                env=PREFILL_ENV,
            )
            # TPU backend initialisations that start at the same moment stall.
            time.sleep(3)
            cls.decode = popen_launch_pd_server(
                MODEL,
                DECODE_URL,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=DECODE_ARGS,
                env=DECODE_ENV,
            )
            for url, process in ((PREFILL_URL, cls.prefill), (DECODE_URL, cls.decode)):
                wait_for_http_ready(
                    url=f"{url}/health",
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    process=process,
                )
            cls.lb = popen_with_error_check(
                [
                    "python3",
                    "-m",
                    "sglang_router.launch_router",
                    "--pd-disaggregation",
                    "--mini-lb",
                    "--prefill",
                    PREFILL_URL,
                    str(BOOTSTRAP_PORT),
                    "--decode",
                    DECODE_URL,
                    "--host",
                    _HOST,
                    "--port",
                    str(_BASE_PORT),
                ]
            )
            wait_for_http_ready(url=f"{LB_URL}/health", timeout=120, process=cls.lb)
        except BaseException:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls):
        if cls.lb is not None:
            kill_process_tree(cls.lb.pid, wait_timeout=60)
        for process in (cls.decode, cls.prefill):
            if process is not None:
                terminate_and_kill_process_tree(process, wait_timeout=60)

    def test_engines_form_a_pipeline_prefill_pair(self):
        prefill = requests.get(f"{PREFILL_URL}/server_info", timeout=30).json()
        decode = requests.get(f"{DECODE_URL}/server_info", timeout=30).json()
        self.assertEqual(prefill["disaggregation_mode"], "prefill")
        self.assertEqual(prefill["pp_size"], PP_SIZE)
        self.assertEqual(decode["disaggregation_mode"], "decode")
        self.assertEqual(decode["tp_size"], DECODE_TP_SIZE)
        for info in (prefill, decode):
            self.assertEqual(info["disaggregation_transfer_backend"], "raiden")

    def test_prompt_over_several_chunks_is_recalled(self):
        out = _generate(_needle_prompt(WORDS[0], repeats=60))
        prompt_tokens = out["meta_info"]["prompt_tokens"]
        self.assertGreater(prompt_tokens, 2 * CHUNK_TOKENS)
        self.assertIn(WORDS[0], out["text"], f"{prompt_tokens} tokens")

    def test_concurrent_rooms_each_recall_their_own_word(self):
        with ThreadPoolExecutor(max_workers=MAX_BATCH) as pool:
            outs = list(
                pool.map(lambda word: _generate(_needle_prompt(word, 4)), WORDS)
            )
        for word, out in zip(WORDS, outs):
            self.assertIn(word, out["text"])


if __name__ == "__main__":
    unittest.main()
