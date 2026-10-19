"""GPT-OSS-120B P/D gate for RAM KV offload on both engines.

One TP8/EP8 host per engine, Raiden transfer. Both engines keep a radix prefix
cache backed by host RAM, including the decode engine
(`--disaggregation-decode-enable-radix-cache`), whose RAM pool holds
full-attention KV while the sliding window comes from prefill.

The probe pushes a two-turn conversation out of HBM on both engines and
resends turn 2. That turn must be restored from RAM on both engines (the
decode engine's load-back counter grows and the prefill reports host-cached
tokens), still reuse the decode-held response KV and reproduce the output of
a device-resident resend. GSM8K then runs through the same pair.
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
import unittest
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.pd_singlehost_base import (
    PDSingleHostTestCase,
    peer_hostname,
)

register_tpu_ci(
    est_time=3600,
    suite="tpu-integration",
    nightly=True,
    labels=[
        "singlehost",
        "gpt-oss-120b",
        "pd",
        "prefix-cache",
        "kv-offload",
        "decode-radix",
    ],
)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, *[".."] * 4))
BENCH = os.path.join(REPO, "benchmark", "tpu")

LAUNCH_CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_kv_offload.py"

E = os.environ.get

SERVED_MODEL_NAME = "gpt-oss-120b"
DEFAULT_MODEL_PATH = "/weights/gpt-oss-120b"
MODEL_ENV = "GPT_OSS_MODEL"

PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
SELF_HOST_ENV = "PD_SELF_HOST"
DECODE_HOST_ENV = "PD_DECODE_HOST"

HOSTS_PER_ENGINE = 1
CHIPLETS_PER_HOST = 8
TP_SIZE = CHIPLETS_PER_HOST
EP_SIZE = CHIPLETS_PER_HOST

PREFILL_PORT = int(E("PREFILL_PORT", "30000"))
DECODE_PORT = int(E("DECODE_PORT", "30000"))
LOCAL_HOST = "127.0.0.1"

WAIT_SECS = int(E("WAIT_SECS", "3600"))
HEALTH_TIMEOUT = float(E("HEALTH_TIMEOUT", "3600"))
PYTHON = E("PYTHON", "python3")
RENDER_JOBSET = os.path.join(REPO, ".github", "tpu-ci", "render_jobset.py")

# Short enough for the generated Job and Pod names, without role words.
RUN_NAME = "pd-sh-gpt-120b-offload"

PREFILL_LOG = "/tmp/pd_offload_prefill.log"
DECODE_LOG = "/tmp/pd_offload_decode.log"

PAGE_SIZE = 64
DISAGG_TRANSFER_BACKEND = "raiden"
RETRACTION_BACKUP = "cpu_tensor"
# The recipe's device KV pool on each engine, in tokens.
MAX_TOTAL_TOKENS = 8192

# A ~1900-token turn 1 with a 256-token response (four pages, ignore_eos) puts
# turn 2's decode reuse past turn 1's prompt, so it cannot come from prefill.
NUM_CONVERSATIONS = 2
FILLER_ENTRIES = 160
TURN1_NEW_TOKENS = 256
TURN2_NEW_TOKENS = 32
COLD_SLACK_TOKENS = 2 * PAGE_SIZE
# Churn just past one device pool: enough to evict both conversations from
# HBM while the 2x host pool still holds them.
CHURN_TOKENS = MAX_TOTAL_TOKENS + 2 * PAGE_SIZE
GENERATE_TIMEOUT = 600

GSM8K_BASELINE = float(E("GSM8K_THRESHOLD_GPT_OSS_120B", E("GSM8K_THRESHOLD", "0.80")))
ATOL = float(E("GSM8K_ATOL", "0.03"))
NUM_EXAMPLES = int(E("GSM8K_NUM_EXAMPLES", "100"))
NUM_THREADS = int(E("GSM8K_NUM_THREADS", "8"))
MAX_TOKENS = int(E("GSM8K_MAX_TOKENS", "1024"))


def resolve_model_path(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    return values.get(MODEL_ENV) or values.get("MODEL") or DEFAULT_MODEL_PATH


def validate_model_source(model_path: str) -> None:
    """Fail clearly for a missing mounted checkpoint, HF ids resolve remotely."""
    if not os.path.isabs(model_path):
        return
    if not os.path.isfile(os.path.join(model_path, "config.json")):
        raise FileNotFoundError(
            f"checkpoint incomplete or missing at {model_path}, "
            f"provision it or set {MODEL_ENV}"
        )


def self_hostname(env: Mapping[str, str] | None = None) -> str:
    values = os.environ if env is None else env
    if values.get(SELF_HOST_ENV):
        return values[SELF_HOST_ENV]
    hostnames = [h for h in values.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]
    return hostnames[0] if hostnames else socket.gethostname()


def _get_json(url: str, timeout: int = 30) -> Any:
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
    """Reject a run that is not a Raiden P/D pair offloading on both engines."""
    failures = []
    for label, info, expected_mode in (
        (PREFILL_ROLE, prefill_info, PREFILL_ROLE),
        (DECODE_ROLE, decode_info, DECODE_ROLE),
    ):
        expected = {
            "disaggregation_mode": expected_mode,
            "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
            "enable_hierarchical_cache": True,
            "hicache_io_backend": "raiden",
            "tp_size": TP_SIZE,
            "ep_size": EP_SIZE,
        }
        for key, value in expected.items():
            if info.get(key) != value:
                failures.append(
                    f"{label} reports {key}={info.get(key)!r}, expected {value!r}"
                )
    # Without decode radix cache the decode engine never offloads, and the
    # probe would only measure prefill.
    if decode_info.get("disaggregation_decode_enable_radix_cache") is not True:
        failures.append(
            "decode reports disaggregation_decode_enable_radix_cache="
            f"{decode_info.get('disaggregation_decode_enable_radix_cache')!r}, "
            "expected True"
        )
    if decode_info.get("disaggregation_decode_retraction_backup") != RETRACTION_BACKUP:
        failures.append(
            "decode reports disaggregation_decode_retraction_backup="
            f"{decode_info.get('disaggregation_decode_retraction_backup')!r}, "
            f"expected {RETRACTION_BACKUP!r}"
        )
    return failures


class Turn(NamedTuple):
    """One /generate reply, as the gate reasons over it."""

    prompt_tokens: int
    cached_tokens: int
    host_cached_tokens: int
    text: str
    output_ids: tuple[int, ...] = ()


class ConversationRecord(NamedTuple):
    secret: int
    turn1: Turn
    # Turn 2 while its prefix is in HBM on both engines.
    device: Turn
    # Turn 2 again after churn, and decode's load-back tokens during it.
    restored: Turn
    decode_load_back_tokens: float
    # Turn 2 a third time, back in HBM after the restore.
    resend: Turn


def offload_failures(records: Sequence[ConversationRecord]) -> list[str]:
    """Gate on RAM restores on both engines that reproduce the device output.

    Every turn must answer with the secret, a reference no restored KV can
    influence. The restored turn is not compared token by token with the turn
    before the eviction: greedy decoding drifts after the answer when prefill
    computes a different number of new tokens (13 and 27 shared tokens of 32
    in measurements), so that comparison is only logged.
    """
    failures = []
    for i, r in enumerate(records):
        tag = f"conversation {i}"
        if r.turn1.cached_tokens >= COLD_SLACK_TOKENS:
            failures.append(f"{tag}: turn 1 was not cold")
        # Reuse past turn 1's prompt covers the decode-generated response,
        # which only the decode engine's own cache holds.
        for label, turn in (
            ("device turn 2", r.device),
            ("restored turn 2", r.restored),
        ):
            if turn.cached_tokens <= r.turn1.prompt_tokens:
                failures.append(
                    f"{tag}: {label} reused {turn.cached_tokens} tokens, not "
                    f"past turn 1's {r.turn1.prompt_tokens}-token prompt"
                )
        if r.decode_load_back_tokens <= 0:
            failures.append(f"{tag}: decode did not restore the prefix from RAM")
        if r.restored.host_cached_tokens <= 0:
            failures.append(f"{tag}: prefill did not restore the prefix from RAM")
        for label, turn in (
            ("device turn 2", r.device),
            ("restored turn 2", r.restored),
            ("resent turn 2", r.resend),
        ):
            answer = _answer(turn.text)
            if answer != str(r.secret):
                failures.append(
                    f"{tag}: {label} answered {answer!r}, the secret is {r.secret}"
                )
        # The same restored KV must decode the same way twice.
        if r.restored.output_ids != r.resend.output_ids:
            failures.append(f"{tag}: RAM-restored turn 2 differs from the resend")
    return failures


def _answer(text: str) -> str | None:
    """The first five-digit number in the reply. Secrets are five digits, the
    ledger filler numbers never are."""
    match = re.search(r"\b\d{5}\b", text)
    return match.group(0) if match else None


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def conversation(index: int) -> tuple[int, str, str]:
    """Return the secret, the turn-1 prompt and the turn-2 question."""
    secret = 50021 + index * 41
    filler = "".join(
        f"Ledger {index}: line {j} books transfer number {index * 1000 + j}. "
        for j in range(FILLER_ENTRIES)
    )
    turn1 = (
        f"The secret number is {secret}. Remember it. "
        + filler
        + "Now tell a short story about a harbor."
    )
    question = (
        "\n\nQuestion: what was the secret number stated at the beginning? "
        "Answer with just the number.\nAnswer:"
    )
    return secret, turn1, question


def _tokenize(tokenize_url: str, text: str) -> list[int]:
    reply = _post_json(
        f"{tokenize_url}/tokenize",
        {"model": SERVED_MODEL_NAME, "prompt": text, "add_special_tokens": False},
        timeout=60,
    )
    return list(reply["tokens"])


def _generate(
    base_url: str, input_ids: list[int], max_new_tokens: int, *, ignore_eos: bool
) -> tuple[Turn, list[int]]:
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
    meta = reply["meta_info"]
    details = meta.get("cached_tokens_details") or {}
    turn = Turn(
        prompt_tokens=int(meta["prompt_tokens"]),
        cached_tokens=int(meta["cached_tokens"]),
        host_cached_tokens=int(details.get("host") or 0),
        text=reply["text"],
        output_ids=tuple(reply["output_ids"]),
    )
    return turn, list(reply["output_ids"])


def _decode_load_back_tokens(decode_url: str) -> float:
    """Tokens the decode engine has restored from host RAM, summed over ranks."""
    with urllib.request.urlopen(f"{decode_url}/metrics", timeout=30) as r:
        lines = r.read().decode("utf-8").splitlines()
    return sum(
        float(line.rsplit(" ", 1)[1])
        for line in lines
        if line.startswith("sglang:load_back_tokens_total{")
    )


def _churn(base_url: str, tokenize_url: str) -> None:
    """Fill both engines' HBM pools with unrelated prompts."""
    sent = 0
    index = 0
    while sent < CHURN_TOKENS:
        _, prompt, _ = conversation(1000 + index)
        ids = _tokenize(tokenize_url, prompt)
        _generate(base_url, ids, 1, ignore_eos=True)
        sent += len(ids)
        index += 1


def offload_runner(*, tokenize_url: str, decode_url: str):
    """Score the conversations restored from RAM on both engines."""

    def run(base_url: str) -> dict[str, Any]:
        prompts = []
        devices = []
        firsts = []
        for i in range(NUM_CONVERSATIONS):
            secret, turn1_text, question = conversation(i)
            prompt1 = _tokenize(tokenize_url, turn1_text)
            turn1, response1 = _generate(
                base_url, prompt1, TURN1_NEW_TOKENS, ignore_eos=True
            )
            prompt2 = prompt1 + response1 + _tokenize(tokenize_url, question)
            device, _ = _generate(base_url, prompt2, TURN2_NEW_TOKENS, ignore_eos=False)
            prompts.append((secret, prompt2))
            firsts.append(turn1)
            devices.append(device)
        _churn(base_url, tokenize_url)
        records = []
        for (secret, prompt2), turn1, device in zip(prompts, firsts, devices):
            before = _decode_load_back_tokens(decode_url)
            restored, _ = _generate(
                base_url, prompt2, TURN2_NEW_TOKENS, ignore_eos=False
            )
            load_back = _decode_load_back_tokens(decode_url) - before
            resend, _ = _generate(base_url, prompt2, TURN2_NEW_TOKENS, ignore_eos=False)
            record = ConversationRecord(
                secret, turn1, device, restored, load_back, resend
            )
            records.append(record)
            print(
                "[pd-offload] "
                + json.dumps(
                    {
                        "secret": secret,
                        "turn1_prompt": turn1.prompt_tokens,
                        "device_cached": device.cached_tokens,
                        "restored_cached": restored.cached_tokens,
                        "restored_prefill_host": restored.host_cached_tokens,
                        "decode_load_back_tokens": load_back,
                        "answers": [
                            str(secret) in t.text for t in (device, restored, resend)
                        ],
                        "parsed": [_answer(t.text) for t in (device, restored, resend)],
                        "device_restored_shared_tokens": _common_prefix_len(
                            device.output_ids, restored.output_ids
                        ),
                        "texts": [t.text[:60] for t in (device, restored, resend)],
                    }
                ),
                flush=True,
            )
        failures = offload_failures(records)
        return {
            "score": 0.0 if failures else 1.0,
            "_passed": not failures,
            "_failures": failures,
        }

    return run


def build_model_config(*, decode_host: str) -> dict[str, Any]:
    prefill_url = f"http://{LOCAL_HOST}:{PREFILL_PORT}"
    return {
        "model": SERVED_MODEL_NAME,
        "test_name": "gpt-oss-120b-pd-singlehost-kv-offload",
        "variant": "PD-1P1D-TP8+EP8+Raiden+offload-both",
        "gates": [
            {
                "dataset": "kv_offload",
                "label": "kv-offload",
                "runner": offload_runner(
                    tokenize_url=prefill_url,
                    decode_url=f"http://{decode_host}:{DECODE_PORT}",
                ),
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


class TestPdSingleHostKvOffloadGptOss120bIntegration(PDSingleHostTestCase):
    """Configure the shared P/D fixture for this test's recipe and gates."""

    config_module = LAUNCH_CONFIG_MODULE
    repo_root = REPO
    cwd = BENCH
    log_path = PREFILL_LOG
    decode_log_path = DECODE_LOG
    self_hostname = staticmethod(self_hostname)
    decode_host_env = DECODE_HOST_ENV
    tag = "pd-singlehost-kv-offload"
    health_timeout = 2 * WAIT_SECS + 300
    health_generate_timeout = HEALTH_TIMEOUT

    @classmethod
    def build_model_config(cls) -> dict[str, Any]:
        return build_model_config(decode_host=cls.decode_hostname())

    @classmethod
    def check_server_configuration(cls) -> None:
        failures = engine_configuration_failures(
            prefill_info=_get_json(f"http://{LOCAL_HOST}:{PREFILL_PORT}/server_info"),
            decode_info=_get_json(
                f"http://{cls.decode_hostname()}:{DECODE_PORT}/server_info"
            ),
        )
        if failures:
            raise RuntimeError("; ".join(failures))

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
    """Render one full prefill host plus one full decode host."""
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
        "--files",
        rel_path,
        "--exec",
        f"python3 {rel_path} integration",
    ]
    return subprocess.check_output(cmd, text=True)


def _turn(
    *, prompt=2300, cached=2240, host=0, text="50021", ids=(11, 12, 13, 14, 15, 16)
):
    return Turn(prompt, cached, host, text, tuple(ids))


def _record(**overrides) -> ConversationRecord:
    answer = "50021"
    fields = dict(
        secret=50021,
        turn1=_turn(prompt=1900, cached=0),
        device=_turn(text=answer),
        restored=_turn(host=2176, text=answer),
        decode_load_back_tokens=2048.0,
        resend=_turn(text=answer),
    )
    fields.update(overrides)
    return ConversationRecord(**fields)


class TestPdSingleHostKvOffloadGptOss120bConfiguration(unittest.TestCase):
    """Launch-contract and gate tests that do not initialize TPU hardware."""

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

    def test_both_engines_run_raiden_ram_offload(self) -> None:
        for plan in self.plans.values():
            args = plan["server_args"]
            self.assertIsNone(args.get("speculative_algorithm"))
            self.assertFalse(args["disable_radix_cache"])
            self.assertTrue(args["enable_hierarchical_cache"])
            self.assertEqual(args["hicache_io_backend"], "raiden")
            self.assertEqual(args["hicache_mem_layout"], "page_first_fused")
            self.assertEqual(args["max_total_tokens"], MAX_TOTAL_TOKENS)
            self.assertTrue(args["enable_metrics"])

    def test_decode_reuses_its_own_cache(self) -> None:
        decode = self.plans[DECODE_ROLE]
        self.assertTrue(
            decode["server_args"]["disaggregation_decode_enable_radix_cache"]
        )
        self.assertIn("SGLANG_ENABLE_UNIFIED_RADIX_TREE", decode["runtime_env_keys"])
        self.assertEqual(
            decode["server_args"]["disaggregation_decode_retraction_backup"],
            RETRACTION_BACKUP,
        )

    def test_configuration_gate_requires_decode_offload(self) -> None:
        prefill = {
            "disaggregation_mode": PREFILL_ROLE,
            "disaggregation_transfer_backend": DISAGG_TRANSFER_BACKEND,
            "enable_hierarchical_cache": True,
            "hicache_io_backend": "raiden",
            "tp_size": TP_SIZE,
            "ep_size": EP_SIZE,
        }
        decode = {
            **prefill,
            "disaggregation_mode": DECODE_ROLE,
            "disaggregation_decode_enable_radix_cache": True,
            "disaggregation_decode_retraction_backup": RETRACTION_BACKUP,
        }
        self.assertEqual(
            engine_configuration_failures(prefill_info=prefill, decode_info=decode),
            [],
        )
        for broken in (
            {"enable_hierarchical_cache": False},
            {"disaggregation_decode_enable_radix_cache": False},
        ):
            with self.subTest(broken=broken):
                self.assertTrue(
                    engine_configuration_failures(
                        prefill_info=prefill, decode_info={**decode, **broken}
                    )
                )

    def test_offload_gate_passes_restores_on_both_engines(self) -> None:
        self.assertEqual(offload_failures([_record()] * NUM_CONVERSATIONS), [])

    def test_offload_gate_requires_each_engine_to_restore(self) -> None:
        for overrides, reason in (
            ({"decode_load_back_tokens": 0.0}, "decode did not restore"),
            ({"restored": _turn(host=0, text="50021")}, "prefill did not restore"),
            # A restore that stops at turn 1's prompt lost the decode response.
            ({"restored": _turn(cached=1856, host=2176, text="50021")}, "not past"),
        ):
            with self.subTest(reason=reason):
                failures = offload_failures([_record(**overrides)])
                self.assertTrue(any(reason in f for f in failures), failures)

    def test_offload_gate_requires_the_same_output(self) -> None:
        diverged = _record(resend=_turn(ids=(11, 12, 13, 14, 15, 99)))
        self.assertTrue(any("differs" in f for f in offload_failures([diverged])))

    def test_offload_gate_checks_restored_output_against_the_device_turn(self):
        # The answer must match the secret on every turn.
        wrong = _record(restored=_turn(host=2176, text="50022"))
        self.assertTrue(any("answered '50022'" in f for f in offload_failures([wrong])))
        # Drift from the device turn is tolerated, the resend must match.
        drift = (11, 12, 13, 77, 88, 99)
        drifted = _record(restored=_turn(host=2176, ids=drift), resend=_turn(ids=drift))
        self.assertEqual(offload_failures([drifted]), [])

    def test_offload_gate_requires_cold_first_turns_and_answers(self) -> None:
        warm = _record(turn1=_turn(prompt=1900, cached=1856))
        self.assertTrue(any("not cold" in f for f in offload_failures([warm])))
        lost = _record(
            device=_turn(text="no"),
            restored=_turn(host=2176, text="no"),
            resend=_turn(text="no"),
        )
        self.assertTrue(any("secret" in f for f in offload_failures([lost])))

    def test_peer_hostname_resolution(self) -> None:
        prefill = f"tpu-ci-{RUN_NAME}-prefill-0-0"
        self.assertEqual(
            peer_hostname(prefill, from_role=PREFILL_ROLE, to_role=DECODE_ROLE),
            f"tpu-ci-{RUN_NAME}-decode-0-0",
        )

    @unittest.skipUnless(
        os.path.exists(RENDER_JOBSET), "renderer is not shipped in the runtime image"
    )
    def test_dry_run_renders_one_host_per_role(self) -> None:
        yaml_content = render_jobset_yaml()
        self.assertIn("name: prefill", yaml_content)
        self.assertIn("name: decode", yaml_content)
        self.assertNotIn("name: router", yaml_content)
        self.assertIn(
            "python3 test/registered/tpu/e2e/"
            "test_tpu_gpt_oss_120b_pd_singlehost_kv_offload.py integration",
            yaml_content,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-host-per-engine GPT-OSS P/D RAM KV offload test"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="unittest",
        choices=["unittest", "integration", "integration_dry_run"],
    )
    args, extra_args = parser.parse_known_args()

    if args.mode == "integration_dry_run":
        print(render_jobset_yaml())
        sys.exit(0)
    test_case = (
        TestPdSingleHostKvOffloadGptOss120bIntegration
        if args.mode == "integration"
        else TestPdSingleHostKvOffloadGptOss120bConfiguration
    )
    unittest.main(
        defaultTest=test_case.__name__,
        argv=[sys.argv[0], *extra_args],
        verbosity=2,
    )
