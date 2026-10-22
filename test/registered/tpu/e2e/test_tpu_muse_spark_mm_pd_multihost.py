# SPDX-License-Identifier: Apache-2.0
"""Live Muse Spark NVFP4 2P/2D multimodal and long-prefix transfer canaries.

Each engine uses four TPU hosts, TP32/EP32. BASE_URL points to this experiment's
router. SGLANG_TPU_USE_LIVE_SERVER=1 opts into the existing deployment.
"""

import base64
import io
import json
import os
import unittest
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from urllib.parse import urlsplit

import requests
from PIL import Image, ImageDraw

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(
    est_time=3600,
    suite="tpu-integration",
    nightly=True,
    disabled="manual: attaches to a live 2P2D Muse deployment (four 2x2x4 slices); no CI lane provides one",
    labels=[
        "multihost",
        "muse-spark-open-816b-a42b",
        "smoke",
        "tp",
        "pd",
        "compile",
        "multimodal",
        "video",
    ],
)


def video_part(colors, *, fps=8, frames_per_color=8, size=(280, 224), variant=None):
    """Encode an actual MP4, with one centered square color per temporal phase."""
    import av

    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "ultrafast"}
        stream.codec_context.thread_count = 1
        index = 0
        for color in colors:
            frame_image = Image.new("RGB", size, "white")
            width, height = size
            side = int(min(size) * 0.72)
            left, top = (width - side) // 2, (height - side) // 2
            ImageDraw.Draw(frame_image).rectangle(
                (left, top, left + side, top + side), fill=color
            )
            if variant is not None:
                # Neutral border marks vary encoded content without changing
                # the square or revealing its answer in the text prompt.
                draw = ImageDraw.Draw(frame_image)
                for bit in range(16):
                    gray = 32 + ((variant >> (bit * 4)) & 15) * 8
                    draw.rectangle((bit * 8, 0, bit * 8 + 7, 7), fill=(gray,) * 3)
            for _ in range(frames_per_color):
                frame = av.VideoFrame.from_image(frame_image)
                frame.pts = index
                frame.time_base = Fraction(1, fps)
                index += 1
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return {
        "type": "video_url",
        "video_url": {
            "url": "data:video/mp4;base64,"
            + base64.b64encode(output.getvalue()).decode()
        },
    }


def square(color, size=280):
    image = Image.new("RGB", (size, size), "white")
    inset = size // 7
    ImageDraw.Draw(image).rectangle(
        (inset, inset, size - inset, size - inset), fill=color
    )
    out = io.BytesIO()
    image.save(out, format="PNG")
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()
        },
    }


@unittest.skipUnless(
    os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
    "requires the live four-engine deployment",
)
class TestMuseSparkMultimodalPd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = os.environ.get("BASE_URL", "http://127.0.0.1:8000").rstrip("/")

    def ask(self, content):
        response = requests.post(
            self.url + "/v1/chat/completions",
            headers={"x-routing-key": "muse-vision-cache-canary"},
            json={
                "model": "Muse-Spark-Open-816B-A42B",
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 2048,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
            timeout=1800,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop", choice)
        text = choice["message"].get("content") or ""
        self.assertTrue(text, choice)
        return text.lower()

    def test_video_frame_rates_resolutions_and_longer_clips(self):
        cases = [
            (4, 16, (560, 420), ("red", "green", "blue")),
            (12, 48, (1120, 840), ("blue", "green", "red")),
        ]
        for fps, frames_per_color, size, colors in cases:
            with self.subTest(fps=fps, frames_per_color=frames_per_color, size=size):
                answer = self.ask(
                    [
                        video_part(
                            colors,
                            fps=fps,
                            frames_per_color=frames_per_color,
                            size=size,
                            variant=fps * 1017,
                        ),
                        {
                            "type": "text",
                            "text": "Name the square colors at the beginning, middle, and end of the video. Reply only with those three colors in temporal order.",
                        },
                    ]
                )
                self.assertRegex(
                    answer, colors[0] + r"\W+.*" + colors[1] + r"\W+.*" + colors[2]
                )

    def test_two_videos_with_unequal_clip_counts(self):
        answer = self.ask(
            [
                video_part(("red", "blue")),
                video_part(
                    ("green", "yellow"), fps=12, frames_per_color=48, size=(560, 420)
                ),
                {
                    "type": "text",
                    "text": "Give the square colors at the beginning and end of the first video, then the beginning and end of the second video. Reply only with those four colors in that order.",
                },
            ]
        )
        self.assertRegex(answer, r"red\W+.*blue\W+.*green\W+.*yellow")

    def test_video_temporal_order_survives_cache_reuse(self):
        for colors in (("red", "green", "blue"), ("blue", "green", "red")):
            with self.subTest(colors=colors):
                answer = self.ask(
                    [
                        video_part(colors),
                        {
                            "type": "text",
                            "text": "Name the square colors at the beginning, middle, and end of this video, in temporal order. Reply with only those three colors.",
                        },
                    ]
                )
                self.assertRegex(
                    answer, colors[0] + r"\W+.*" + colors[1] + r"\W+.*" + colors[2]
                )

    def test_mixed_image_and_video_keep_modalities_in_order(self):
        answer = self.ask(
            [
                square("yellow"),
                video_part(("red", "blue")),
                {
                    "type": "text",
                    "text": "Give the square color in the still image, followed by the square colors at the beginning and end of the video. Reply with only the three colors in that order.",
                },
            ]
        )
        self.assertRegex(answer, r"yellow\W+.*red\W+.*blue")

    def test_video_with_long_prefix_survives_chunked_prefill(self):
        answer = self.ask(
            [
                {
                    "type": "text",
                    "text": "The private archive pin is 72931. Remember it exactly.\n"
                    + "The catalog stores routine folders and inventory records.\n"
                    * 2200,
                },
                video_part(("red", "green", "blue")),
                {
                    "type": "text",
                    "text": "What is the private archive pin, and what are the square colors at the beginning and end of the video? Reply with only the pin and two colors in that order.",
                },
            ]
        )
        self.assertRegex(answer, r"72931\W+.*red\W+.*blue")

    def test_concurrent_image_requests_reuse_warmed_decode_shapes(self):
        # One routing key keeps this cohort on an engine pair while concurrent
        # completions exercise decoder buckets beyond the single-request path.
        def request(index):
            color = ("red", "green", "blue", "yellow")[index % 4]
            answer = self.ask(
                [
                    square(color),
                    {
                        "type": "text",
                        "text": "What color is the square? Reply with its color only.",
                    },
                ]
            )
            self.assertIn(color, answer)

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(request, range(16)))

    def test_two_prefill_and_two_decode_engines(self):
        response = requests.get(self.url + "/workers", timeout=30)
        response.raise_for_status()
        workers = response.json()["workers"]
        for role in ("prefill", "decode"):
            urls = {
                worker["url"] for worker in workers if worker["worker_type"] == role
            }
            self.assertEqual(len(urls), 2, workers)
            self.assertEqual(
                {urlsplit(url).hostname.split("-server-0-")[0] for url in urls},
                {f"wyzhang-muse-{role[0]}0", f"wyzhang-muse-{role[0]}1"},
            )
            for url in urls:
                info = requests.get(url + "/server_info", timeout=30)
                info.raise_for_status()
                state = info.json()
                self.assertEqual(state["disaggregation_mode"], role)
                self.assertEqual(state["tp_size"], 32)
                self.assertEqual(state["ep_size"], 32)
                self.assertEqual(state["nnodes"], 4)
                self.assertEqual(state["dtype"], "bfloat16")
                self.assertTrue(state["enable_multimodal"])
                self.assertTrue(state["internal_states"])

    def stream_ask(self, content, key):
        text, finish, done = "", None, False
        with requests.post(
            self.url + "/v1/chat/completions",
            headers={"x-routing-key": key},
            json={
                "model": "Muse-Spark-Open-816B-A42B",
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 2048,
                "stream": True,
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
            timeout=1800,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                self.assertNotIn("error", event)
                for choice in event.get("choices", []):
                    text += choice.get("delta", {}).get("content") or ""
                    finish = choice.get("finish_reason") or finish
        self.assertTrue(done)
        self.assertEqual(finish, "stop")
        return text.lower()

    def test_streaming_image_completion_finishes_cleanly(self):
        answer = self.stream_ask(
            [
                square("yellow"),
                {
                    "type": "text",
                    "text": "What color is the square? Reply with its color only.",
                },
            ],
            "muse-stream-canary",
        )
        self.assertIn("yellow", answer)

    def test_streaming_video_completion_finishes_cleanly(self):
        answer = self.stream_ask(
            [
                video_part(("red", "green", "blue")),
                {
                    "type": "text",
                    "text": "Name the square colors at the beginning, middle, and end of the video. Reply only with the three colors in temporal order.",
                },
            ],
            "muse-video-stream-canary",
        )
        self.assertRegex(answer, r"red\W+.*green\W+.*blue")

    def test_image_inputs_change_with_same_routing_key(self):
        for color in ("blue", "red", "green"):
            with self.subTest(color=color):
                answer = self.ask(
                    [
                        square(color),
                        {
                            "type": "text",
                            "text": "What color is the square? Answer with its color only.",
                        },
                    ]
                )
                self.assertIn(color, answer)

    def test_multi_image_order_survives_embedding_cache(self):
        for colors in (("red", "blue"), ("blue", "red")):
            with self.subTest(colors=colors):
                answer = self.ask(
                    [
                        square(colors[0]),
                        square(colors[1]),
                        {
                            "type": "text",
                            "text": "Name the square colors in image order. Reply with the first color, then the second color.",
                        },
                    ]
                )
                self.assertRegex(answer, colors[0] + r"\W+.*" + colors[1])

    def test_long_prefix_needle_survives_pd_transfer(self):
        prefix = "The private archive pin is 52947. Remember this pin exactly.\n"
        filler = (
            "The catalog contains ordinary folders and routine inventory records.\n"
            * 2200
        )
        answer = self.ask(
            [
                {"type": "text", "text": prefix + filler},
                square("yellow"),
                {
                    "type": "text",
                    "text": "What is the private archive pin and what color is the square? Reply with the pin and color only.",
                },
            ]
        )
        self.assertIn("52947", answer)
        self.assertIn("yellow", answer)

    def test_uneven_image_lengths_survive_encoder_dp_padding(self):
        for colors in (("red", "green", "blue"), ("blue", "green", "red")):
            with self.subTest(colors=colors):
                answer = self.ask(
                    [
                        *(
                            square(color, size)
                            for color, size in zip(colors, (224, 280, 392))
                        ),
                        {
                            "type": "text",
                            "text": "Name the square colors in image order. Reply only with the three colors in order.",
                        },
                    ]
                )
                self.assertRegex(
                    answer, colors[0] + r"\W+.*" + colors[1] + r"\W+.*" + colors[2]
                )


if __name__ == "__main__":
    unittest.main()
