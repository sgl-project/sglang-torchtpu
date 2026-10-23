"""Kimi-K3's multimodal model under DP attention with TPU compilation: TP4, two
attention-DP groups of attention TP 2.

Under DP attention every rank runs the MoE layers together, so every rank must
run the same compiled program each step. When one group prefills an image, the
other groups take the input_embeds path too (``is_tpu_mm_extend_in_batch``),
and the vision tower runs collective-free on the group that owns the image.
A decode batch that runs as one-token extends while the other group extends
keeps that group's program even when it holds an image request, whose image
is already embedded. Requests go in four at a time, an image request after
every second text prefix, so one group prefills an image while the other
prefills or decodes text, and image requests decode beside the other group's
text prefills.

The reference is the same checkpoint (kimi_k3_tiny_harness.py) at TP4 without
DP attention. Greedy continuations of this random model part ways within a
few tokens at any other attention TP, so the comparison is teacher-forced:
each server continues the same prefixes (cut points through every prompt,
including a three-chunk one) by two tokens. The first token checks the
prefill, the second, where the first agrees, one decode step. Calibration on
tpu7x over these 96 prefixes:

    layout                     token 1   token 2 | token 1
    TP4 again                  100.0%    100.0%
    TP2 / EP2 (no DP)           70.8%     83.8%
    DP attention                70.8%     77.9%
    DP attention, concurrent    71.9%     76.8%
    another prompt's prefix      0.0%      --

so DP attention must stay well clear of the wrong-context floor. The
concurrent row predates the image check: it had one image request in flight
beside the text.

The image requests, each with its own image, are compared the same way, by
their first two tokens, and held to the same ``MIN_AGREEMENT``.
"""

import base64
import functools
import io
import tempfile
import threading
import unittest
from collections.abc import Sequence

import requests
from kimi_k3_tiny_harness import (
    PROMPTS,
    assert_no_serving_recompile,
    build_model,
    launch,
    read_log,
)

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_tpu_ci(
    est_time=900,
    suite="nightly-tpu-4-v7",
    nightly=True,
    labels=["singlehost", "kimi-k3", "dpa", "multimodal", "compile"],
)

PORT = 21000
BASE_URL = f"http://127.0.0.1:{PORT}"
DP_ATTENTION_ARGS = ["--enable-dp-attention", "--dp-size", "2", "--enable-dp-lm-head"]
MIN_AGREEMENT = 0.55
WIDTH = 4
# The image requests take these texts in turn.
IMAGE_PROMPTS = [
    "Describe the image.",
    "What colors are in this image?",
    "Is there a pattern?",
    "Which part of the image is the brightest, and which is the darkest?",
    "Give the image a title.",
    "What might this image be used for?",
    "Describe how the colors change from the top left to the bottom right.",
    "Name the dominant color.",
    "Is this a photograph or a rendering? Explain how you can tell.",
    "What shapes do you see?",
    "Describe the image to someone who cannot see it, in as much detail as you can.",
    "Does the pattern repeat? If it does, how often?",
    "Compare the left half of the image with the right half.",
    "List three adjectives for this image.",
    "Write a short poem about this picture.",
    "What would you change about this image to make it look calmer?",
]


def _prefixes() -> list[list[int]]:
    prefixes = []
    for prompt in PROMPTS:
        n = len(prompt)
        for cut in sorted({max(1, (n * i) // 24) for i in range(1, 25)}):
            prefixes.append(prompt[:cut])
    return prefixes


PREFIXES = _prefixes()
# An image request after every second prefix, so 48. If image requests agree
# with the reference independently and as often as the prefixes do (assumed
# from the text calibration until measured), their agreement falls below
# MIN_AGREEMENT on a correct build about one run in 100.
IMAGE_EVERY = 2
NUM_IMAGES = len(PREFIXES) // IMAGE_EVERY


def _image_url(seed: int) -> str:
    """The harness image's pattern, shifted per channel by ``seed`` (0 is the
    harness image). Each seed gives another 224x224 image, so the same image
    token count and compile shapes, but its own embedding: every image request
    runs the vision tower instead of reusing a cached one."""
    from PIL import Image

    size = 224
    image = Image.new("RGB", (size, size))
    image.putdata(
        [
            (
                (x * 7 + seed * 37) % 256,
                (y * 5 + seed * 59) % 256,
                ((x + y) * 3 + seed * 83) % 256,
            )
            for y in range(size)
            for x in range(size)
        ]
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def _image_requests() -> list[tuple[str, str]]:
    """The image and the text of each image request."""
    return [
        (_image_url(seed), IMAGE_PROMPTS[seed % len(IMAGE_PROMPTS)])
        for seed in range(NUM_IMAGES)
    ]


def _next_two(prefix: list[int]) -> tuple[int, int]:
    response = requests.post(
        f"{BASE_URL}/generate",
        json={
            "input_ids": prefix,
            "sampling_params": {
                "max_new_tokens": 2,
                "temperature": 0,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    return tuple(response.json()["output_ids"])


def _image_next_two(model: str, image: str, prompt: str) -> tuple[tuple[int, int], int]:
    """``image`` with ``prompt``, continued by two tokens, and the prompt's
    image token count."""
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 2,
            "temperature": 0,
            "ignore_eos": True,
            "return_token_ids": True,
        },
        timeout=600,
    )
    response.raise_for_status()
    body = response.json()
    image_tokens = body["usage"]["prompt_tokens_details"]["image_tokens"]
    return tuple(body["choices"][0]["response_token_ids"]), image_tokens


def _serve_all(model: str, image_requests: list[tuple[str, str]]) -> tuple[list, list]:
    """Every prefix and every image request, ``WIDTH`` in flight at once, an
    image request after every ``IMAGE_EVERY`` prefixes."""
    texts: list = [None] * len(PREFIXES)
    images: list = [None] * len(image_requests)
    tasks = []
    for i, prefix in enumerate(PREFIXES):
        tasks.append((texts, i, functools.partial(_next_two, prefix)))
        if i % IMAGE_EVERY == IMAGE_EVERY - 1:
            j = i // IMAGE_EVERY
            call = functools.partial(_image_next_two, model, *image_requests[j])
            tasks.append((images, j, call))
    errors: list = []
    lock = threading.Lock()
    next_index = [0]

    def worker():
        try:
            while True:
                with lock:
                    k = next_index[0]
                    next_index[0] += 1
                if k >= len(tasks):
                    return
                results, i, call = tasks[k]
                results[i] = call()
        except Exception as e:  # surfaced on the main thread
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(WIDTH)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return texts, images


def _agreement(name: str, served: Sequence, reference: Sequence) -> tuple[float, float]:
    """Token 1 agreement with the reference, and token 2 agreement where token
    1 agrees."""
    first = [got[0] == want[0] for got, want in zip(served, reference)]
    second = [
        got[1] == want[1] for got, want in zip(served, reference) if got[0] == want[0]
    ]
    first_rate = sum(first) / len(first)
    second_rate = sum(second) / max(1, len(second))
    print(
        f"{name}: token 1 agreement {sum(first)}/{len(first)} = {first_rate:.3f}; "
        f"token 2 given token 1 {sum(second)}/{len(second)} = {second_rate:.3f}"
    )
    return first_rate, second_rate


class TestKimiK3TinyDPAttention(CustomTestCase):
    def test_dp_attention_agrees_with_the_reference(self):
        model_dir = tempfile.TemporaryDirectory(prefix="kimi_k3_tiny_")
        self.addCleanup(model_dir.cleanup)
        build_model(model_dir.name)
        model = model_dir.name
        logs = tempfile.mkdtemp(prefix="kimi_k3_dpa_logs_")
        image_requests = _image_requests()

        dpa_log = f"{logs}/dp_attention.log"
        process, log = launch(model, PORT, DP_ATTENTION_ARGS, dpa_log)
        try:
            served, served_images = _serve_all(model, image_requests)
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()
        assert_no_serving_recompile(self, read_log(dpa_log), "DP attention server")

        process, log = launch(model, PORT, [], f"{logs}/reference.log")
        try:
            reference = [_next_two(prefix) for prefix in PREFIXES]
            reference_images = [
                _image_next_two(model, *request) for request in image_requests
            ]
        finally:
            kill_process_tree(process.pid, wait_timeout=60)
            log.close()

        served_ids, served_image_tokens = zip(*served_images)
        reference_ids, reference_image_tokens = zip(*reference_images)
        text_rates = _agreement("text prefixes", served, reference)
        image_rates = _agreement("image requests", served_ids, reference_ids)
        for rate in text_rates + image_rates:
            self.assertGreaterEqual(rate, MIN_AGREEMENT)
        self.assertGreater(min(served_image_tokens), 0)
        self.assertEqual(served_image_tokens, reference_image_tokens)


if __name__ == "__main__":
    unittest.main()
