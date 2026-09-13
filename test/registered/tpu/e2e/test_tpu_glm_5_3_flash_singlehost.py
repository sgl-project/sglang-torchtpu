# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash text generation on one TPU host, using the mounted FP8 weights.

This is a functional gate; benchmark accuracy baselines have not been measured.
Run with Bazel or directly like the other MODEL_CONFIG TPU e2e tests.
"""

import math
import os

import requests

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu import model_config_main

register_tpu_ci(
    est_time=3600,
    suite="stage-b-test-tpu-4-v7",
    disabled="TPU PR runners are not provisioned yet",
    labels=["singlehost", "glm-5.3-flash", "smoke"],
)

MODEL_PATH = os.environ.get("MODEL", "/weights/GLM-5.3-Flash")
BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:30000")


def generation_gate(base_url):
    def generate(prompts):
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": prompts,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                "return_logprob": True,
            },
            timeout=600,
        )
        response.raise_for_status()
        items = response.json()
        if not isinstance(items, list):
            items = [items]
        for item in items:
            assert item["text"].strip(), item
            assert item["meta_info"]["completion_tokens"] > 0, item
            probs = item["meta_info"]["output_token_logprobs"]
            assert probs and all(math.isfinite(p[0]) for p in probs), item
        return [item["text"] for item in items]

    prompts = ["The capital of France is", "The capital of Japan is"]
    single = [generate(p)[0] for p in prompts]
    assert "Paris" in single[0], single
    assert "Tokyo" in single[1], single
    batched = generate(prompts)
    # BF16/FP8 batched matmuls can change later greedy wording. Check the
    # requested answers in both batch positions, rather than bytewise prose.
    assert "Paris" in batched[0] and "Tokyo" in batched[1], batched
    # More than one prefill chunk must carry the same request's KDA state.
    long_prompt = "The following is a geography exercise. " * 40 + prompts[0]
    long_answer = generate(long_prompt)[0]
    assert "Paris" in long_answer, long_answer
    # Reusing freed recurrent-state slots must reproduce the original answers.
    repeated = [generate(p)[0] for p in prompts]
    assert repeated == single, (repeated, single)
    return {"_passed": True, "score": 1.0}


MODEL_CONFIG = {
    "model": MODEL_PATH,
    "test_name": "GLM-5.3-Flash",
    "variant": "TP8+EP8 text",
    "launch": {
        "recipe": "recipe/glm_5_3_flash.sh",
        "health_timeout": float(os.environ.get("HEALTH_TIMEOUT", "3600")),
        "tag": "glm-5.3-flash-e2e",
    },
    "gates": [{"dataset": "generation", "label": "smoke", "runner": generation_gate}],
}

if __name__ == "__main__":
    model_config_main(MODEL_CONFIG, base_url=BASE_URL)
