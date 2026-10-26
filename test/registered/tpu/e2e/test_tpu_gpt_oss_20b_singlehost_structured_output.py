"""GPT-OSS-20B single-host structured output and function calling gate.

Every other TPU server launches with --grammar-backend none, so this is the only
test that runs xgrammar's vocab mask on TPU: JSON schema through the reasoning
parser, a mixed regex/EBNF/JSON batch on /generate, and tool calls -- auto, and
`required`, which xgrammar enforces with the harmony structural tag.
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path

# Expected runtime includes model download, compilation, and the three gates.
register_tpu_ci(
    est_time=1800,
    suite="stage-b-test-tpu-4-v7",
    labels=["singlehost", "gpt-oss-20b", "structured-output", "function-calling"],
)

SERVED_MODEL_NAME = "gpt-oss-20b"
MODEL_PATH = resolve_tpu_model_path("openai/gpt-oss-20b")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "1800"))
TIMEOUT = 600

CITY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "population": {"type": "integer"},
    },
    "required": ["name", "population"],
    "additionalProperties": False,
}
DATE_REGEX = r"(19|20)\d{2}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])"
YES_NO_EBNF = 'root ::= "yes" | "no"'
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city", "unit"],
        },
    },
}
# Without strict the grammar only guarantees a JSON object ({} included), as in
# OpenAI's semantics; a forced call must be strict for its arguments to hold.
WEATHER_TOOL_STRICT = {
    "type": "function",
    "function": {**WEATHER_TOOL["function"], "strict": True},
}


def _chat(base_url: str, **body) -> dict:
    body = {"model": SERVED_MODEL_NAME, "temperature": 0, **body}
    r = requests.post(f"{base_url}/v1/chat/completions", json=body, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["choices"][0]


def _verdict(checks: list[tuple[str, str | None]]) -> dict:
    """checks: (case, failure or None) -> a self-gating runner result."""
    failures = [f"{case}: {why}" for case, why in checks if why]
    for f in failures:
        print(f"✗ {f}")
    return {
        "score": 1.0 - len(failures) / len(checks),
        "_passed": not failures,
        "_failures": failures,
    }


def json_schema_runner(base_url: str) -> dict:
    # Concurrent so the mask is applied to a multi-row batch, after reasoning.
    def one(city: str) -> tuple[str, str | None]:
        choice = _chat(
            base_url,
            messages=[{"role": "user", "content": f"Describe {city} as JSON."}],
            max_tokens=2048,
            reasoning_effort="low",
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "city", "schema": CITY_SCHEMA},
            },
        )
        content = choice["message"].get("content")
        try:
            obj = json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return city, f"not JSON: {content!r}"
        if set(obj) != {"name", "population"}:
            return city, f"keys {sorted(obj)}"
        if not isinstance(obj["name"], str) or not isinstance(obj["population"], int):
            return city, f"types {obj!r}"
        return city, None

    with ThreadPoolExecutor(4) as pool:
        return _verdict(list(pool.map(one, ["Paris", "Tokyo", "Rome", "Berlin"])))


def mixed_grammar_runner(base_url: str) -> dict:
    # One /generate batch whose rows carry different grammar types.
    cases = [
        ("regex", "Today's date in ISO format is ", {"regex": DATE_REGEX}),
        ("ebnf", "Is the sky blue? Answer: ", {"ebnf": YES_NO_EBNF}),
        ("json", "A city as JSON: ", {"json_schema": json.dumps(CITY_SCHEMA)}),
    ]
    r = requests.post(
        f"{base_url}/generate",
        json={
            "text": [prompt for _, prompt, _ in cases],
            "sampling_params": [
                {"temperature": 0, "max_new_tokens": 128, **grammar}
                for _, _, grammar in cases
            ],
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    outputs = [o["text"] for o in r.json()]

    def check(kind: str, text: str) -> str | None:
        if kind == "regex" and not re.fullmatch(DATE_REGEX, text):
            return f"{text!r} does not match {DATE_REGEX}"
        if kind == "ebnf" and text not in ("yes", "no"):
            return f"{text!r} is not yes/no"
        if kind == "json":
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                return f"not JSON: {text!r}"
            if set(obj) != {"name", "population"}:
                return f"keys {sorted(obj)}"
        return None

    return _verdict([(k, check(k, t)) for (k, _, _), t in zip(cases, outputs)])


def tool_call_runner(base_url: str) -> dict:
    def call(
        case: str, prompt: str, tool_choice: str, tool: dict
    ) -> tuple[str, str | None]:
        choice = _chat(
            base_url,
            messages=[{"role": "user", "content": prompt}],
            tools=[tool],
            tool_choice=tool_choice,
            max_tokens=2048,
            reasoning_effort="low",
        )
        calls = choice["message"].get("tool_calls") or []
        if not calls:
            return case, f"no tool call; content={choice['message'].get('content')!r}"
        fn = calls[0]["function"]
        if fn["name"] != "get_weather":
            return case, f"called {fn['name']!r}"
        try:
            args = json.loads(fn["arguments"])
        except (TypeError, json.JSONDecodeError):
            return case, f"arguments not JSON: {fn['arguments']!r}"
        if not isinstance(args.get("city"), str):
            return case, f"arguments {args!r}"
        if choice.get("finish_reason") != "tool_calls":
            return case, f"finish_reason {choice.get('finish_reason')!r}"
        return case, None

    return _verdict(
        [
            # The model chooses the tool; the parser must extract it.
            call(
                "auto",
                "What is the weather in Paris, in celsius?",
                "auto",
                WEATHER_TOOL,
            ),
            # Nothing asks for weather: only the grammar can force the call.
            call("required", "Say hello.", "required", WEATHER_TOOL_STRICT),
        ]
    )


LAUNCH_ENV = {
    "MODEL": MODEL_PATH,
    "ISL": "2048",
    "OSL": "512",
    "CONCUR": "4",
    "CONTEXT_LEN": "4096",
    "MAX_TOTAL_TOKENS": "16384",
    "TPU_COMPILE_TOKENS": "512 1024 2048",
    "TPU_COMPILE_BS": "1 2 4",
    "ENABLE_PROFILER_ENDPOINT": "0",
    "GRAMMAR_BACKEND": "xgrammar",
    "EXTRA_ARGS": (
        "--reasoning-parser gpt-oss --tool-call-parser gpt-oss "
        f"--served-model-name {SERVED_MODEL_NAME}"
    ),
}

MODEL_CONFIG = {
    "model": SERVED_MODEL_NAME,
    "test_name": "gpt-oss-20b-structured-output",
    "variant": "TP8+EP8+xgrammar",
    "launch": {
        "recipe": "recipe/gpt_oss_120b/server.sh",
        "env": LAUNCH_ENV,
        "health_timeout": HEALTH_TIMEOUT,
        "tag": "gpt-oss-20b-structured-output-e2e",
    },
    "gates": [
        {
            "dataset": "json_schema",
            "label": "json-schema",
            "runner": json_schema_runner,
        },
        {
            "dataset": "mixed_grammar",
            "label": "mixed-grammar-batch",
            "runner": mixed_grammar_runner,
        },
        {"dataset": "tool_call", "label": "tool-call", "runner": tool_call_runner},
    ],
}


if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
