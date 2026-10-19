"""GPT-OSS-120B P/D decode-side radix cache gate, one host per engine.

The decode engine runs with `--disaggregation-decode-enable-radix-cache`
over the Raiden transfer backend so a returning turn reuses KV cached in
the decode engine, including its self-generated response, while Raiden
transfers only the new suffix from prefill. The prefill pod launches its
engine and mini router, verifies the decode engine is in radix mode,
executes the multi-turn probe, and evaluates GSM8K through the router.

The probe gate crosses the generated-token boundary: turn 2 of a conversation
resends turn 1's prompt plus turn 1's response. The prefill engine never saw the
response tokens, so turn-2 `cached_tokens` exceeding turn 1's prompt length can
only come from decode-side radix cache reuse.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import unittest
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.pd_singlehost_base import PDSingleHostTestCase

# Expected runtime: the single-host P/D GSM8K gate plus a few minutes of probe.
register_tpu_ci(
    est_time=3000,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd", "decode-radix"],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_decode_radix.py"

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"

# Each engine uses one eight-chiplet TPU host.
HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = HOSTS_PER_ENGINE * CHIPLETS_PER_HOST
EP_SIZE = TP_SIZE

# Separate pods may reuse the same service and Raiden ports.
PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LOCAL_HOST = "127.0.0.1"

PYTHON = E("PYTHON", "python3")

# The runtime image excludes the JobSet renderer.
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

# Pod name prefix within Kubernetes' 63-character limit;
# role words are omitted because peer mapping swaps the role segment.
RUN_NAME = "gpt-oss-120b-pd-sh-radix"

PREFILL_LOG = "/tmp/pd_decode_radix_prefill.log"
DECODE_LOG = "/tmp/pd_decode_radix_decode.log"

PAGE_SIZE = 64
DISAGG_TRANSFER_BACKEND = "raiden"

# Unique filler ensures turn 1 is cold (~1900 tokens);
# exceeds the 128-token sliding window to verify reuse beyond SWA eviction.
NUM_CONVERSATIONS = 8
FILLER_ENTRIES = 160
# Pinned via ignore_eos to four pages (256 tokens);
# ensures decode-held KV clears turn-1 prompt length after page-floor rounding.
TURN1_NEW_TOKENS = 256
TURN2_NEW_TOKENS = 32
# Cold threshold: conversations share only a short system prefix,
# and prior health probes may leave up to one residual page in cache.
COLD_SLACK_TOKENS = 2 * PAGE_SIZE
# Tolerates retracted decode requests that re-bootstrap via pop_preallocated,
# which legitimately stops turn-2 matching at the prefill prefix boundary.
CROSSING_TOLERANCE = 2
GENERATE_TIMEOUT = 600

# Measured 0.97 accuracy on this topology; 0.80 serves as the CI passing floor.
GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.80")))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    """Resolve the checkpoint from the environment, else the mounted default."""
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Fail clearly when CI points at an unprovisioned local checkpoint."""
    if not os.path.isabs(model_path):
        return  # HF repo ids and bucket URIs resolve remotely
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"GPT-OSS-120B checkpoint incomplete or missing at {model_path}, "
            f"provision it on the weights volume or set {MODEL_ENV}"
        )


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    """This pod's own DNS name."""
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def engine_configuration_failures(
    *, prefill_info: Mapping[str, Any], decode_info: Mapping[str, Any]
) -> list[str]:
    """Reject a run where the engines are not a Raiden P/D pair in radix mode."""
    failures = []
    for label, info, expected_mode in (
        (PREFILL_ROLE, prefill_info, PREFILL_ROLE),
        (DECODE_ROLE, decode_info, DECODE_ROLE),
    ):
        mode = info.get("disaggregation_mode")
        if mode != expected_mode:
            failures.append(
                f"{label} engine reports disaggregation_mode={mode!r}, expected "
                f"{expected_mode!r}: it is not serving a P/D role"
            )
        backend = info.get("disaggregation_transfer_backend")
        if backend != DISAGG_TRANSFER_BACKEND:
            failures.append(
                f"{label} engine reports disaggregation_transfer_backend="
                f"{backend!r}, expected {DISAGG_TRANSFER_BACKEND!r}"
            )
    # The probe below is void on a decode engine that fell back to chunk cache.
    if decode_info.get("disaggregation_decode_enable_radix_cache") is not True:
        failures.append(
            "decode engine reports disaggregation_decode_enable_radix_cache="
            f"{decode_info.get('disaggregation_decode_enable_radix_cache')!r}, "
            "expected True"
        )
    return failures


def engine_configuration_runner(*, prefill_url: str, decode_url: str):
    """Verify both servers report the expected P/D roles, backend and cache mode."""

    def run(base_url: str) -> dict[str, Any]:
        prefill_info = _get_json(f"{prefill_url}/server_info")
        decode_info = _get_json(f"{decode_url}/server_info")
        failures = engine_configuration_failures(
            prefill_info=prefill_info, decode_info=decode_info
        )
        print(
            f"[pd-decode-radix] lb={base_url} prefill="
            f"{prefill_info.get('disaggregation_mode')}/"
            f"{prefill_info.get('disaggregation_transfer_backend')} decode="
            f"{decode_info.get('disaggregation_mode')}/"
            f"{decode_info.get('disaggregation_transfer_backend')} decode_radix="
            f"{decode_info.get('disaggregation_decode_enable_radix_cache')}",
            flush=True,
        )
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


class TurnRecord(NamedTuple):
    prompt_tokens: int
    cached_tokens: int


class ConversationRecord(NamedTuple):
    """What the gate reasons over for one two-turn conversation."""

    turn1: TurnRecord
    turn2: TurnRecord
    # Turn 2 asked for the secret number stated at the start of turn 1.
    answered: bool


def conversation(index: int) -> tuple[int, str, str]:
    """Return the secret, the turn-1 prompt and the turn-2 question."""
    secret = 40007 + index * 37
    filler = "".join(
        f"Archive {index}: entry {j} records event number {index * 1000 + j}. "
        for j in range(FILLER_ENTRIES)
    )
    turn1 = (
        f"The secret number is {secret}. Remember it. "
        + filler
        + "Now tell a short story about a lighthouse."
    )
    question = (
        "\n\nQuestion: what was the secret number stated at the beginning? "
        "Answer with just the number.\nAnswer:"
    )
    return secret, turn1, question


def decode_radix_failures(records: Sequence[ConversationRecord]) -> list[str]:
    """Gate the probe on decode-side reuse plus a known-answer sanity check."""
    total = len(records)
    failures = []

    # Turn 1 must be cold, or the boundary argument below is void.
    warm = sum(1 for r in records if r.turn1.cached_tokens >= COLD_SLACK_TOKENS)
    if warm:
        failures.append(f"turn 1 not cold ({warm}/{total} warm)")

    # Turn-2 reuse must extend past turn 1's prompt into decode-generated tokens;
    # hits from the prefill radix cache alone cannot exceed the prompt boundary.
    crossed = sum(1 for r in records if r.turn2.cached_tokens > r.turn1.prompt_tokens)
    if crossed < total - CROSSING_TOLERANCE:
        failures.append(
            f"decode radix reuse missing ({crossed}/{total} crossed the turn-1 "
            "prompt boundary)"
        )

    # Corrupt or mismatched cached KV garbles the answer; occasional misses
    # are tolerated because non-deterministic TPU drift cascades during decode.
    misses = sum(1 for r in records if not r.answered)
    if misses > total // 2:
        failures.append(f"secret number lost in {misses}/{total} conversations")
    return failures


def _tokenize(tokenize_url: str, text: str) -> list[int]:
    reply = _post_json(
        f"{tokenize_url}/tokenize",
        {"model": SERVED_MODEL_NAME, "prompt": text, "add_special_tokens": False},
        timeout=60,
    )
    return list(reply["tokens"])


def _generate(
    base_url: str, input_ids: list[int], max_new_tokens: int, *, ignore_eos: bool
) -> tuple[str, list[int], dict[str, Any]]:
    reply = _post_json(
        f"{base_url}/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": ignore_eos,
            },
        },
        timeout=GENERATE_TIMEOUT,
    )
    return reply["text"], list(reply["output_ids"]), reply["meta_info"]


def run_conversation(
    *, base_url: str, tokenize_url: str, index: int
) -> ConversationRecord:
    """Run one two-turn conversation through the router, carrying token IDs.

    Token IDs are chained directly because GPT-OSS completions include Harmony
    special tokens stripped by `skip_special_tokens`; re-tokenizing decoded text
    would diverge from the served ID stream and invalidate prefix matching.
    """
    secret, turn1_text, question = conversation(index)
    prompt1 = _tokenize(tokenize_url, turn1_text)
    _, response1, meta1 = _generate(
        base_url, prompt1, TURN1_NEW_TOKENS, ignore_eos=True
    )
    prompt2 = prompt1 + response1 + _tokenize(tokenize_url, question)
    text2, _, meta2 = _generate(base_url, prompt2, TURN2_NEW_TOKENS, ignore_eos=False)
    record = ConversationRecord(
        turn1=TurnRecord(meta1["prompt_tokens"], meta1["cached_tokens"]),
        turn2=TurnRecord(meta2["prompt_tokens"], meta2["cached_tokens"]),
        answered=str(secret) in text2,
    )
    print(
        f"[pd-decode-radix] conv {index}: t1_prompt={record.turn1.prompt_tokens} "
        f"t1_cached={record.turn1.cached_tokens} "
        f"t2_prompt={record.turn2.prompt_tokens} "
        f"t2_cached={record.turn2.cached_tokens} "
        f"answer={'OK' if record.answered else 'MISS'}",
        flush=True,
    )
    return record


def decode_radix_runner(*, tokenize_url: str):
    """Score the fraction of conversations whose turn 2 reused decode-held KV."""

    def run(base_url: str) -> dict[str, Any]:
        records = [
            run_conversation(base_url=base_url, tokenize_url=tokenize_url, index=i)
            for i in range(NUM_CONVERSATIONS)
        ]
        failures = decode_radix_failures(records)
        crossed = sum(
            1 for r in records if r.turn2.cached_tokens > r.turn1.prompt_tokens
        )
        print(
            f"[pd-decode-radix] crossed={crossed}/{len(records)} "
            f"answer_misses={sum(1 for r in records if not r.answered)}/{len(records)}",
            flush=True,
        )
        return {
            "score": crossed / len(records),
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def build_model_config(*, decode_host: str) -> dict[str, Any]:
    """Assemble the gate list, which needs the decode pod's resolved name."""
    prefill_url = f"http://{LOCAL_HOST}:{PREFILL_PORT}"
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-decode-radix",
        "variant": f"PD-singlehost-TP{TP_SIZE}+EP{EP_SIZE}-raiden-decode-radix",
        # Keep this specialized P/D result off the standard accuracy dashboard.
        "gates": [
            {
                "dataset": "pd_engine_configuration",
                "label": "pd-engine-configuration",
                "runner": engine_configuration_runner(
                    prefill_url=prefill_url,
                    decode_url=f"http://{decode_host}:{DECODE_PORT}",
                ),
            },
            {
                "dataset": "decode_radix_reuse",
                "label": "decode-radix-reuse",
                "runner": decode_radix_runner(tokenize_url=prefill_url),
            },
            {
                "dataset": "gsm8k",
                "label": "gsm8k",
                "baseline": GSM8K_BASELINE,
                "atol": ATOL,
                "num_examples": NUM_EXAMPLES,
                "num_threads": NUM_THREADS,
                "max_tokens": MAX_TOKENS,
                "temperature": 0.0,
            },
        ],
    }


class TestPdSingleHostDecodeRadixGptOss120bIntegration(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and gates."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    decode_host_env = DECODE_HOST_ENV
    tag = "pd-singlehost-decode-radix"

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config(decode_host=cls.decode_hostname())

    @classmethod
    def setUpClass(cls) -> None:
        validate_model_source(resolve_model_path())
        super().setUpClass()

    def test_accuracy(self) -> None:
        self.run_evaluations()


def render_jobset_yaml(
    image: str = "sglang-tpu:latest",
    lane: str = "nightly",
    weights_pvc: str = "model-weights",
    run_name: str = RUN_NAME,
) -> str:
    """Render the JobSet for this lane: one host of prefill, one of decode."""
    rel_path = os.path.relpath(__file__, REPO)
    cmd = [
        PYTHON,
        RENDER_JOBSET,
        "--pd-prefill-hosts",
        str(HOSTS_PER_ENGINE),
        "--pd-decode-hosts",
        str(HOSTS_PER_ENGINE),
        "--lane",
        lane,
        "--image",
        image,
        "--weights-pvc",
        weights_pvc,
        "--name",
        run_name,
        # Names the dashboard row and the TIMINGS record.
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


def _record(
    *, t1_prompt: int, t1_cached: int, t2_cached: int, answered: bool = True
) -> ConversationRecord:
    return ConversationRecord(
        turn1=TurnRecord(t1_prompt, t1_cached),
        turn2=TurnRecord(t1_prompt + TURN1_NEW_TOKENS + 24, t2_cached),
        answered=answered,
    )


class TestPdSingleHostDecodeRadixGptOss120bConfiguration(unittest.TestCase):
    """Host-side gate tests, runnable without TPU hardware."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.plans = {}
        for role in (PREFILL_ROLE, DECODE_ROLE):
            output = subprocess.check_output(
                [
                    sys.executable,
                    "-m",
                    "benchmark.tpu.launch_server",
                    "--config",
                    LAUNCH_CONFIG_MODULE + ":" + role.upper() + "_CONFIG",
                    "--runtime-envs",
                    json.dumps(
                        {"FORCE_SINGLE_HOST": "1", "SGLANG_HOST_IP": "127.0.0.1"}
                    ),
                    "--dry-run",
                ],
                cwd=REPO,
                text=True,
            )
            cls.plans[role] = json.loads(output)

    # A 1900-token turn-1 prompt, as the probe's filler tokenizes to.
    T1_PROMPT = 1900

    def _crossed(self) -> ConversationRecord:
        # Decode-held response KV: the match runs to the page floor of turn 2.
        t2_prompt = self.T1_PROMPT + TURN1_NEW_TOKENS + 24
        return _record(
            t1_prompt=self.T1_PROMPT,
            t1_cached=0,
            t2_cached=t2_prompt // PAGE_SIZE * PAGE_SIZE,
        )

    def _prefill_only(self) -> ConversationRecord:
        # Chunk-cache decode hits come solely from the prefill radix cache;
        # because prefill never saw response tokens, matching stops at prompt 1.
        return _record(
            t1_prompt=self.T1_PROMPT,
            t1_cached=0,
            t2_cached=self.T1_PROMPT // PAGE_SIZE * PAGE_SIZE,
        )

    def test_decode_enables_the_radix_cache_and_prefill_does_not(self) -> None:
        decode = self.plans[DECODE_ROLE]
        self.assertTrue(
            decode["server_args"]["disaggregation_decode_enable_radix_cache"]
        )
        self.assertIn("SGLANG_ENABLE_UNIFIED_RADIX_TREE", decode["runtime_env_keys"])
        prefill = self.plans[PREFILL_ROLE]
        self.assertFalse(
            prefill["server_args"].get(
                "disaggregation_decode_enable_radix_cache", False
            )
        )

    def test_configuration_gate_requires_decode_radix_mode(self) -> None:
        prefill = {
            "disaggregation_mode": PREFILL_ROLE,
            "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
        }
        decode = {**prefill, "disaggregation_mode": DECODE_ROLE}
        # A decode engine that fell back to chunk cache reports the flag off.
        failures = engine_configuration_failures(
            prefill_info=prefill,
            decode_info={**decode, "disaggregation_decode_enable_radix_cache": False},
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("disaggregation_decode_enable_radix_cache", failures[0])
        self.assertEqual(
            engine_configuration_failures(
                prefill_info=prefill,
                decode_info={
                    **decode,
                    "disaggregation_decode_enable_radix_cache": True,
                },
            ),
            [],
        )

    def test_probe_gate_separates_decode_reuse_from_prefill_reuse(self) -> None:
        self.assertEqual(
            decode_radix_failures([self._crossed()] * NUM_CONVERSATIONS), []
        )
        # Chunk-cache decode hits the prefill radix cache on turn 2;
        # gating purely on non-zero cached_tokens would falsely pass.
        failures = decode_radix_failures([self._prefill_only()] * NUM_CONVERSATIONS)
        self.assertEqual(len(failures), 1)
        self.assertIn("crossed", failures[0])

    def test_probe_gate_tolerates_a_bounded_number_of_misses(self) -> None:
        good, bad = self._crossed(), self._prefill_only()
        within = [bad] * CROSSING_TOLERANCE + [good] * (
            NUM_CONVERSATIONS - CROSSING_TOLERANCE
        )
        self.assertEqual(decode_radix_failures(within), [])
        beyond = [bad] + within[:-1]
        self.assertTrue(decode_radix_failures(beyond))

    def test_probe_gate_requires_cold_first_turns(self) -> None:
        # A warm turn 1 invalidates the boundary proof; hits can no longer
        # be attributed specifically to decode-side radix cache reuse.
        warm = _record(
            t1_prompt=self.T1_PROMPT,
            t1_cached=self.T1_PROMPT // PAGE_SIZE * PAGE_SIZE,
            t2_cached=self.T1_PROMPT + TURN1_NEW_TOKENS,
        )
        records = [warm] + [self._crossed()] * (NUM_CONVERSATIONS - 1)
        failures = decode_radix_failures(records)
        self.assertEqual(len(failures), 1)
        self.assertIn("not cold", failures[0])

    def test_probe_gate_fails_when_most_answers_are_lost(self) -> None:
        good = self._crossed()
        lost = good._replace(answered=False)
        half = NUM_CONVERSATIONS // 2
        self.assertEqual(
            decode_radix_failures([lost] * half + [good] * (NUM_CONVERSATIONS - half)),
            [],
        )
        failures = decode_radix_failures(
            [lost] * (half + 1) + [good] * (NUM_CONVERSATIONS - half - 1)
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("secret number", failures[0])

    # Skip in runtime images, which do not contain the renderer.
    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_render_jobset_yaml(self) -> None:
        # Verify the workflow name, roles, and command remain aligned.
        yaml_content = render_jobset_yaml()
        self.assertIn(f"generateName: tpu-ci-{RUN_NAME}-", yaml_content)
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn("CI_PD_ROLE", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/"
            "test_tpu_gpt_oss_120b_pd_singlehost_decode_radix.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-host-per-engine P/D decode radix cache gate"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
        help="Execution mode (default: unittest)",
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(render_jobset_yaml())
        sys.exit(0)

    test_case = (
        TestPdSingleHostDecodeRadixGptOss120bIntegration
        if args.mode == "integration"
        else TestPdSingleHostDecodeRadixGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
