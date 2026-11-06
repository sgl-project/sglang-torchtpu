#!/usr/bin/env python3
"""Burst of N mixed requests sent at once through the P/D router (needs: pip install aiohttp).

python3 pd_burst.py --url http://LB:8000 --model gpt-oss-120b --n 512 --prio-mode mixed
python3 pd_burst.py --url http://LB:8000 --model qwen3-vl-8b --vl --n 256 --prio-mode mixed
"""

import argparse
import asyncio
import base64
import random
import struct
import time
import zlib

import aiohttp

WORDS = (
    "time year people way day man thing woman life child world school state family student group country "
    "problem hand part place case week company system program question work government number night point home "
    "water room mother area money story fact month lot right study book eye job word business issue side kind head "
    "house service friend father power hour game line end member law car city community name president team minute "
    "idea kid body information back parent face others level office door health person art war history party result "
    "change morning reason research girl guy moment air teacher force education"
).split()
PROMPTS = [
    "Explain how a refrigerator works in three short paragraphs.",
    "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring.",
    "List five countries in South America and their capitals.",
    "What is 17 * 23? Show the multiplication step by step.",
    "Summarize the plot of Romeo and Juliet in four sentences.",
    "Translate to French: 'The weather is nice today and I would like to go for a walk.'",
    "Give three tips for writing clear technical documentation.",
    "Describe the water cycle for a ten-year-old.",
    "Write a haiku about distributed systems.",
    "Compare TCP and UDP in a short table.",
    "What causes the seasons on Earth?",
    "Write a SQL query that finds the second highest salary in an employees table.",
]
A = None


def filler(n, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(n))


def png(seed, w=640, h=480):  # white canvas with 5 random bars, as a data URI
    r, px = random.Random(seed), bytearray(b"\xff" * (w * h * 3))
    for i in range(5):
        x0, x1, top = (
            w * (6 + 11 * i) // 64,
            w * (13 + 11 * i) // 64,
            int(h * r.uniform(0.1, 0.83)),
        )
        color = bytes((r.randint(0, 255), 80, 160)) * (x1 - x0)
        for y in range(top, h * 15 // 16):
            px[(y * w + x0) * 3 : (y * w + x1) * 3] = color
    raw = b"".join(b"\x00" + px[y * w * 3 : (y + 1) * w * 3] for y in range(h))
    ck = lambda t, d: (
        struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))
    )
    data = (
        b"\x89PNG\r\n\x1a\n"
        + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + ck(b"IDAT", zlib.compress(bytes(raw)))
        + ck(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


def chat(
    content,
    max_tokens,
    *,
    system=None,
    images=None,
    temperature=0.0,
    stream=False,
    priority=None,
):
    user = (
        [{"type": "image_url", "image_url": {"url": u}} for u in images]
        + [{"type": "text", "text": content}]
        if images
        else content
    )
    p = {
        "model": A.model,
        "messages": ([{"role": "system", "content": system}] if system else [])
        + [{"role": "user", "content": user}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if A.model.startswith("gpt-oss"):
        p["reasoning_effort"] = "low"
    if stream:
        p["stream_options"] = {"include_usage": True}
    if priority is not None:
        p["priority"] = priority
    return "/v1/chat/completions", p


def mixed_request(rnd, i, prio_mode):
    """One request of the mixed workload; the order of rnd calls matters for seed reproducibility."""
    kind = rnd.random()
    pr = rnd.choice([None, 0, 1, 10, 100, -10]) if prio_mode == "mixed" else None
    stream = rnd.random() < 0.35
    if (
        A.vl and kind < 0.3
    ):  # image chat; we sent PNG or JPEG of a drawn chart, this sends PNG only
        seed, n_img, _fmt = (
            rnd.randint(0, 50),
            rnd.choice([1, 1, 1, 2, 3]),
            rnd.choice(["PNG", "JPEG"]),
        )
        return chat(
            "What do you see? Answer briefly.",
            rnd.choice([16, 64, 200]),
            images=[png(seed)] * n_img,
            stream=stream,
            priority=pr,
        )
    if kind < 0.55:
        mt, prompt = rnd.choice([1, 8, 64, 256, 512]), rnd.choice(PROMPTS)
        return chat(
            prompt,
            mt,
            temperature=rnd.choice([0.0, 0.0, 0.7, 1.0]),
            stream=stream,
            priority=pr,
        )
    if kind < 0.8:  # 9 distinct long system prompts -> prefix-cache / HiCache hits
        n_words = rnd.choice([50, 400, 1500, 3000])
        system = filler(n_words, rnd.randint(0, 8))
        return chat(
            "Summarize the text above in one sentence.",
            rnd.choice([32, 128]),
            system=system,
            stream=stream,
            priority=pr,
        )
    mt = rnd.choice([1, 32, 300])
    p = {
        "text": filler(rnd.choice([10, 200, 2000]), i),
        "stream": stream,
        "sampling_params": {
            "max_new_tokens": mt,
            "temperature": rnd.choice([0.0, 0.8]),
        },
    }
    if pr is not None:
        p["priority"] = pr
    return "/generate", p


async def send(s, tag, path, payload):
    t, status = time.time(), None
    try:
        async with s.post(
            A.url + path, json=payload, timeout=aiohttp.ClientTimeout(total=A.timeout)
        ) as r:
            status = r.status
            await r.read()
    except Exception as e:  # client timeout, cut stream, connection reset
        status = type(e).__name__
    print(
        f"{time.strftime('%H:%M:%S')} {tag:>10} prio={payload.get('priority')} {status} {time.time() - t:6.1f}s",
        flush=True,
    )
    return status


async def burst(s):
    rnd = random.Random(A.seed)
    reqs = [mixed_request(rnd, i, A.prio_mode) for i in range(A.n)]
    res = await asyncio.gather(
        *(send(s, f"b{i}", u, p) for i, (u, p) in enumerate(reqs))
    )
    print("OK", sum(st == 200 for st in res), "/", len(res))


async def main():
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        await burst(s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument(
        "--vl", action="store_true", help="include image requests in the mixed workload"
    )
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prio-mode", choices=["mixed", "none"], default="mixed")
    ap.add_argument("--timeout", type=float, default=1800)
    A = ap.parse_args()
    asyncio.run(main())
