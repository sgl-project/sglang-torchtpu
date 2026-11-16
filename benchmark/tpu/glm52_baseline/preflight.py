"""Exercise the real driver/client against CPU fixtures, including retained recovery.

No model correctness, TPU compilation, or performance claim follows from this test.
Run with a Python environment containing aiohttp and numpy, without TorchTPU.
"""

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]


def checkpoint(path):
    path.mkdir()
    config = dict(
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        first_k_dense_replace=1,
        n_routed_experts=256,
    )
    (path / "config.json").write_text(json.dumps(config))
    (path / "tokenizer_config.json").write_text("{}")
    tensors = {}
    for layer in range(3):
        for expert in (0, 255):
            stem = f"model.layers.{layer}.mlp.experts.{expert}.down_proj"
            tensors[stem + ".weight"] = ("F8_E4M3", [2, 2], 4)
            tensors[stem + ".weight_scale_inv"] = ("F32", [2], 8)
    tensors["model.layers.2.eh_proj.weight"] = ("BF16", [2, 4], 16)
    header, offset = {}, 0
    for name, (dtype, shape, size) in tensors.items():
        header[name] = dict(
            dtype=dtype, shape=shape, data_offsets=[offset, offset + size]
        )
        offset += size
    raw = json.dumps(header).encode()
    (path / "fixture.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + bytes(offset)
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(
                metadata=dict(total_size=offset),
                weight_map={k: "fixture.safetensors" for k in tensors},
            )
        )
    )


class Tokenizer:
    chat_template = "fixture"

    def apply_chat_template(self, *args, **kwargs):
        return "What is 2 + 2?"

    def encode(self, text, **kwargs):
        return list(text.encode())


def client_mode(args):
    from benchmark.tpu.glm52_baseline import client

    def load(path):
        # Deliberately exceed the fixture server's keep-alive timeout.
        time.sleep(1.7)
        return Tokenizer()

    with patch.object(client, "load_tokenizer", load):
        asyncio.run(client.main(client.parse_args(args)))


def rank_mode(rank):
    from benchmark.tpu.glm52_baseline import session

    original_popen = subprocess.Popen
    marks = Path(os.environ["GLM52_FIXTURE_MARKS"])
    real_run = subprocess.run

    class Engine:
        pid = 900000 + rank
        returncode = None

        def __init__(self, command, **kwargs):
            assert "benchmark.tpu.launch_server" in command
            (marks / f"engine-start-{rank}").write_text("1")

        def poll(self):
            return self.returncode

        def send_signal(self, sig):
            (marks / f"engine-stop-{rank}").write_text(str(sig))
            self.returncode = 0

        def wait(self, **kwargs):
            return self.returncode

    def execute(command, **kwargs):
        if "benchmark.tpu.glm52_baseline.validate" in command:
            recipe = json.loads(Path(command[-2]).read_text())["server_args"]
            assert (
                recipe["tp_size"] == recipe["attn_dp_size"] == recipe["ep_size"] == 16
            )
            assert recipe["chunked_prefill_size"] == 32768
            Path(command[-1]).write_text(json.dumps(recipe))
            return subprocess.CompletedProcess(command, 0)
        if "-c" in command:
            assert "query_node_rank" in command[-1]
            return subprocess.CompletedProcess(
                command, 0, stdout=f"PHYSICAL_RANK={rank}\n", stderr=""
            )
        if "benchmark.tpu.launch_server" in command:
            assert "--dry-run" in command
            return subprocess.CompletedProcess(command, 0)
        if "benchmark.tpu.glm52_baseline.client" in command:
            failed = marks / "injected-client-failure"
            if not failed.exists():
                failed.write_text("deliberate transport failure")
                raise subprocess.CalledProcessError(9, command)
            i = command.index("benchmark.tpu.glm52_baseline.client")
            command = [
                sys.executable,
                "-m",
                "benchmark.tpu.glm52_baseline.preflight",
                "--client",
                *command[i + 1 :],
            ]
        # Run real host checkpoint preparation and the real full benchmark client.
        with patch.object(subprocess, "Popen", original_popen):
            return real_run(command, **kwargs)

    with (
        patch.object(session.importlib.metadata, "version", return_value="fixture"),
        patch.object(session.subprocess, "Popen", Engine),
        patch.object(session.subprocess, "run", execute),
        patch.object(
            session.socket, "gethostbyname", return_value=f"10.0.0.{rank + 1}"
        ),
    ):
        return session.run()


async def verify(directory):
    from aiohttp import web
    from benchmark.tpu.glm52_baseline.prepare import prepare

    marks = directory / "marks"
    marks.mkdir()
    model = directory / "model"
    checkpoint(model)
    resume = asyncio.Event()
    resume.set()
    active = [0] * 16
    cached = {}
    counters = dict(requests=0, paused=0)
    configs = []

    async def info(request):
        return web.json_response(
            {
                **configs[0],
                "chunked_prefill_size": 2048,
                "speculative_algorithm": "EAGLE",
                "preflight_mock": True,
            }
        )

    async def pause(request):
        counters["paused"] += 1
        resume.clear()
        return web.json_response({"status": "ok"})

    async def cont(request):
        resume.set()
        return web.json_response({"ok": True})

    async def flush(request):
        assert not any(active)
        cached.clear()
        return web.Response(text="Cache flushed.\n")

    async def loads(request):
        return web.json_response(
            {
                "loads": [
                    dict(
                        dp_rank=r,
                        num_running_reqs=active[r],
                        max_total_num_tokens=219008,
                    )
                    for r in range(16)
                ]
            }
        )

    async def generate(request):
        payload = await request.json()
        rank = payload["routed_dp_rank"]
        ids = payload.get("input_ids", [1, 2, 3])
        n = payload["sampling_params"]["max_new_tokens"]
        hit = 0
        for length in (89984, 49152):
            if (
                len(ids) >= length
                and (rank, length, hashlib.sha256(bytes(ids[:length])).hexdigest())
                in cached
            ):
                hit = length
                break
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        active[rank] += 1
        counters["requests"] += 1
        try:
            await resume.wait()
            steps = list(range(32, 1025, 32)) if n == 1024 else sorted({1, n // 2, n})
            for count in steps:
                await asyncio.sleep(0.03 if n == 1024 else 0.09)
                meta = dict(
                    dp_rank=rank,
                    prompt_tokens=len(ids),
                    completion_tokens=count,
                    cached_tokens=hit,
                    spec_verify_ct=math.ceil(count / 3),
                    spec_num_correct_drafts=count // 2,
                    finish_reason={"type": "length"} if count == n else None,
                )
                data = (
                    "data: " + json.dumps({"text": "4", "meta_info": meta}) + "\n\n"
                ).encode()
                await response.write(data[:7])
                await response.write(data[7:])
            if len(ids) in (49152, 89984):
                cached[(rank, len(ids), hashlib.sha256(bytes(ids)).hexdigest())] = True
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        finally:
            active[rank] -= 1
        return response

    app = web.Application(client_max_size=8 * 1024 * 1024)
    for path, fn in [
        ("/server_info", info),
        ("/health", lambda r: web.Response(status=200)),
        ("/v1/loads", loads),
        ("/flush_cache", flush),
    ]:
        app.router.add_get(path, fn)
    for path, fn in [
        ("/generate", generate),
        ("/pause_generation", pause),
        ("/continue_generation", cont),
        ("/flush_cache", flush),
    ]:
        app.router.add_post(path, fn)
    runner = web.AppRunner(app, keepalive_timeout=1.5, shutdown_timeout=0.1)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    artifact = directory / "artifact"
    artifact.mkdir()
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    envs = []
    for rank in range(2):
        runtime = directory / f"host-{rank}"
        runtime.mkdir()
        (runtime / "SETUP_READY").touch()
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(model, runtime / "recipe")
        config_path = runtime / "recipe/server.json"
        config = json.loads(config_path.read_text())
        config["server_args"]["port"] = port
        config_path.write_text(json.dumps(config))
        configs.append(config["server_args"])
        env = {
            **os.environ,
            "FALCON_RANK": str(rank),
            "SOURCE_COMMIT": source,
            "ARTIFACT_LOCAL_DIR": str(artifact),
            "GLM52_RUN_ROOT": str(runtime),
            "GLM52_RUN_ID": "preflight-no-overlay",
            "GLM52_SESSION_SECONDS": "90",
            "GLM52_FIXTURE_MARKS": str(marks),
            "PYTHONPATH": str(ROOT),
        }
        envs.append(env)
    logs = []
    children = []
    try:
        for rank, env in enumerate(envs):
            log = (directory / f"rank-{rank}.log").open("wb")
            logs.append(log)
            children.append(
                await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "benchmark.tpu.glm52_baseline.preflight",
                    "--rank",
                    str(rank),
                    env=env,
                    cwd=ROOT,
                    stdout=log,
                    stderr=asyncio.subprocess.STDOUT,
                )
            )
        shared = artifact / "glm52-e2e/session-preflight-no-overlay"
        deadline = time.monotonic() + 30
        while not (shared / "client-attention.json").exists():
            assert time.monotonic() < deadline, "client attention not recorded"
            assert all(p.returncode is None for p in children)
            await asyncio.sleep(0.05)
        assert (marks / "engine-start-0").exists() and not (
            marks / "engine-stop-0"
        ).exists()
        assert not (shared / "result-0.json").exists(), (
            "client failure must not end the session"
        )
        (shared / "retry-client.json").write_text(
            json.dumps({"id": "verified-client-fix"})
        )
        finish_deadline = time.monotonic() + 65
        while any(p.returncode is None for p in children):
            attention = json.loads((shared / "client-attention.json").read_text())
            if attention["status"] == "ATTENTION" and attention["attempt"] > 1:
                failure = (
                    artifact
                    / "glm52-e2e/rank-0"
                    / f"client-attempt-{attention['attempt']}.log"
                )
                raise AssertionError(failure.read_text()[-5000:])
            assert time.monotonic() < finish_deadline, "driver did not finish"
            await asyncio.sleep(0.05)
        codes = await asyncio.gather(*(p.wait() for p in children))
        assert codes == [0, 0], codes
        result = json.loads((shared / "result-0.json").read_text())
        summary = json.loads(
            (Path(result["measurements"]) / "summary.json").read_text()
        )
        assert summary["decode_prefix"] == "shared" and len(summary["rounds"]) == 6
        assert counters["paused"] == 8 and counters["requests"] == 368, counters
        assert all((marks / f"engine-stop-{r}").exists() for r in range(2))
        print(
            json.dumps(
                dict(
                    status="PASS",
                    scope="CPU fixtures only; no TPU or performance evidence",
                    checks=[
                        "native source identity without tar",
                        "both rank rendezvous",
                        "real checkpoint metadata and host reads",
                        "client failure retains server",
                        "explicit client retry without engine restart",
                        "16 DP smoke",
                        "C16 49K+4K warmup and 3 rounds",
                        "C64 shared 90K/1024 warmup and 3 rounds",
                        "actual HTTP streaming, cache receipts, sustained batch validation",
                        "both rank cleanup",
                    ],
                    requests=counters["requests"],
                    output=str(directory),
                ),
                indent=2,
            )
        )
    finally:
        for child in children:
            if child.returncode is None:
                child.terminate()
                await child.wait()
        for log in logs:
            log.close()
        await runner.cleanup()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int)
    p.add_argument("--client", action="store_true")
    p.add_argument("--output", type=Path)
    args, remaining = p.parse_known_args()
    if args.client:
        if args.output is not None:
            remaining += ["--output", str(args.output)]
        client_mode(remaining)
        return 0
    if args.rank is not None:
        return rank_mode(args.rank)
    directory = args.output or Path(
        tempfile.mkdtemp(prefix="glm52-full-runner-preflight-")
    )
    directory.mkdir(parents=True, exist_ok=True)
    asyncio.run(verify(directory))
    return 0


if __name__ == "__main__":
    sys.exit(main())
