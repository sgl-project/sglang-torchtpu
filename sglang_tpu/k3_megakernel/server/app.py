# SPDX-License-Identifier: Apache-2.0
"""Standalone OpenAI-compatible HTTP server for the K3 decode megakernel (no sglang scheduler).

Every host runs this program (SPMD, like the anchor harness). Process 0 serves HTTP
(``/v1/completions``, ``/v1/chat/completions``, sglang-style ``/generate``, ``/health``,
``/v1/models``), queues requests (one at a time) and publishes each one in the JAX
distributed key-value store; the other processes read the same sequence of jobs and run
the same engine calls. Stop decisions (stop ids, stop strings, max_tokens) are a
deterministic function of the generated ids, so all processes agree without talking.

Single host:
  python -m sglang_tpu.k3_megakernel.server.app --port 30000 --decode-steps 64 -- \\
      --layers 13 --expert-load stream --relax dense_fp8,pair_oproj --kernel-opt wire_relax=bf16_transport
Multi-host: ``server/run4h.sh`` on every host (adds --coordinator/--num-processes/--process-id).
Speculative decoding: ``--spec-draft dspark --draft-path <DSpark draft checkpoint>`` (greedy DSpark
draft + 8-row verify on device, ``--spec-rounds`` rounds per device call; see engine.generate_spec).
Manual checks and benchmarks: test/manual/k3_megakernel/README.md.

Limits: one request at a time, greedy decoding (sampling parameters such as temperature are
ignored; n > 1, logprobs, echo, tools and several prompts per request are rejected with 400).
A client that disconnects does not abort its request: every process finishes it in lockstep
(an abort would have to be broadcast to the followers like a job). The HTTP server binds
127.0.0.1 unless ``--host`` says otherwise.
"""

import argparse
import asyncio
import json
import os
import queue
import sys
import threading
import time
import traceback
import uuid

# [EOS], <|end_of_msg|> (generation_config eos_token_id).
STOP_IDS = (163585, 163586)
_KV_PREFIX = "k3srv/job/"
# Job keys older than this many jobs are deleted from the KV store; a follower is never that far behind
# (every job that reaches the device is a collective all processes take part in).
_KV_KEEP = 64
_MAX_STOPS, _MAX_STOP_LEN = 16, 256
# Per-request benchmark / test hooks: rejected with 400 unless the server runs with --bench-hooks.
_BENCH_HOOKS = ("k3_force_acc", "k3_draft", "k3_stop_ids")
# OpenAI / sglang request fields this server does not implement: a request using them gets a 400.
_UNSUPPORTED = (
    "logprobs",
    "top_logprobs",
    "echo",
    "tools",
    "functions",
    "tool_choice",
    "suffix",
    "best_of",
)


def _parse(argv):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument(
        "--decode-steps",
        type=int,
        default=64,
        help="greedy B1 steps per device call (K); the tail of a request uses K/4, K/16, .., 1",
    )
    p.add_argument(
        "--lookahead",
        type=int,
        default=1,
        help="decode calls kept in flight ahead of the host (0 or 1)",
    )
    p.add_argument(
        "--capacity", type=int, default=16384, help="max prompt + generated tokens"
    )
    p.add_argument(
        "--spec-draft",
        choices=("none", "dspark"),
        default="none",
        help="speculative decoding: 'dspark' = server.draft.DraftModel from --draft-path",
    )
    p.add_argument(
        "--draft-path", default="", help="DSpark draft checkpoint (--spec-draft dspark)"
    )
    p.add_argument(
        "--draft-tp",
        type=int,
        default=0,
        help="DSpark draft tensor-parallel group (0 = the whole mesh; 8 = one host: every host runs its own copy, "
        "so no draft collective crosses hosts; at 4 hosts the whole mesh measured ~40 us/round faster)",
    )
    p.add_argument(
        "--spec-rounds", type=int, default=8, help="speculative rounds per device call"
    )
    p.add_argument(
        "--bench-hooks",
        action="store_true",
        help="accept the per-request benchmark / test hooks (output meaningless): k3_force_acc=N "
        "(spec: every round emits N tokens whatever the target says), k3_draft=zero (spec: drafts "
        "token 0 without running the draft), k3_stop_ids=[ids] (extra ids treated like EOS). Off: "
        "a request using them gets a 400",
    )
    p.add_argument("--served-model-name", default="")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument(
        "--anchor-check",
        action="store_true",
        help="before serving: the anchor prompt through the engine (loop and rows prefill, fresh and "
        "reused state, K steps with and without lookahead); logs the token shas (the loop sha equals "
        "the manual harness's, test/manual/k3_megakernel/bench/anchor.py)",
    )
    argv = list(sys.argv[1:] if argv is None else argv)
    ours, rest = (
        (argv[: argv.index("--")], argv[argv.index("--") + 1 :])
        if "--" in argv
        else (argv, [])
    )
    ns, unknown = p.parse_known_args(ours)
    if ns.spec_draft == "dspark" and not ns.draft_path:
        p.error("--spec-draft dspark needs --draft-path")
    return ns, unknown + rest


def _log(msg):
    import jax

    print(
        f"[k3srv p{jax.process_index()} {time.strftime('%H:%M:%S')}] {msg}", flush=True
    )


class Job:
    """One generation request; ``spec`` is what every process needs to run it."""

    def __init__(self, spec: dict, loop=None):
        self.spec, self.loop = spec, loop
        self.events = asyncio.Queue() if loop is not None else None

    def push(self, item):
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self.events.put_nowait, item)


def _stop_fn(tok, stop_strs: list[str]):
    """Deterministic ``(ids, n_new) -> bool``: a stop string appears in text that includes the ``n_new``
    tokens of the last call (the whole new span plus enough earlier tokens for a stop string that
    straddles calls; a token decodes to at least one character or a byte of one)."""
    if not stop_strs:
        return None
    overlap = max(len(s) for s in stop_strs) + 8

    def check(ids, n_new):
        text = tok.decode(ids[-(n_new + overlap) :], skip_special_tokens=True)
        return any(s in text for s in stop_strs)

    return check


def _stop_prefix_len(text: str, stops: list[str]) -> int:
    """Length of the longest tail of ``text`` that is a proper prefix of a stop string."""
    best = 0
    for s in stops:
        for k in range(min(len(s) - 1, len(text)), best, -1):
            if text.endswith(s[:k]):
                best = k
                break
    return best


def run_job(engine, tok, job: Job):
    spec = job.spec

    def on_tokens(new, finished, reason):
        job.push(("tokens", new, finished, reason))

    try:
        stop_ids = (() if spec.get("ignore_eos") else STOP_IDS) + tuple(
            spec.get("stop_ids") or ()
        )
        stop_check = _stop_fn(tok, spec.get("stop") or [])
        spec_engine = getattr(engine, "draft", None) is not None
        if spec_engine and spec.get("prefill_mode", "rows") == "rows":
            ids, reason, timing = engine.generate_spec(
                spec["prompt"],
                spec["max_tokens"],
                stop_ids,
                on_tokens=on_tokens,
                stop_check=stop_check,
                force=spec.get("force_acc", 0),
                zero_drafts=spec.get("zero_drafts", False),
            )
        else:
            ids, reason, timing = engine.generate(
                spec["prompt"],
                spec["max_tokens"],
                stop_ids,
                on_tokens=on_tokens,
                stop_check=stop_check,
                prefill_mode=spec.get("prefill_mode", "rows"),
                steps=int(spec.get("steps", 0)),
            )
        _log(f"job {spec['n']}: {json.dumps(timing)} reason {reason}")
        job.push(("done", ids, reason, timing))
    except Exception as e:  # a bad request must not take the server down
        _log(f"job {spec['n']} failed:\n{traceback.format_exc()}")
        job.push(("error", f"{type(e).__name__}: {e}"))


# ----- process 0: HTTP -----
def _int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer (got {value!r})")
    return value


def _stops(stop) -> list[str]:
    """The request's stop strings, validated (every process runs the same stop check on them)."""
    if stop is None:
        return []
    stops = [stop] if isinstance(stop, str) else stop
    if not isinstance(stops, list) or not all(isinstance(s, str) and s for s in stops):
        raise ValueError(
            "stop must be a non-empty string or a list of non-empty strings"
        )
    if len(stops) > _MAX_STOPS or any(len(s) > _MAX_STOP_LEN for s in stops):
        raise ValueError(
            f"at most {_MAX_STOPS} stop strings of at most {_MAX_STOP_LEN} characters"
        )
    return stops


def job_spec(
    prompt_ids,
    body: dict,
    *,
    max_tokens_default: int,
    max_prompt: int,
    vocab: int,
    steps_allowed,
    bench_hooks: bool = False,
) -> dict:
    """Validated job spec (no job number yet: the leader numbers jobs as it publishes them, so a
    rejected request never leaves a gap the followers would wait on). Raises ValueError -> 400."""
    for key in _UNSUPPORTED:
        if body.get(key):
            raise ValueError(f"{key!r} is not supported by this server")
    if body.get("n") not in (None, 1):
        raise ValueError("only n=1 is supported")
    if not isinstance(prompt_ids, list) or not all(
        isinstance(t, int) and not isinstance(t, bool) for t in prompt_ids
    ):
        raise ValueError("prompt must be text or a list of token ids")
    if not 0 < len(prompt_ids) <= max_prompt:
        raise ValueError(
            f"prompt has {len(prompt_ids)} tokens; this server takes 1..{max_prompt}"
        )
    if not all(0 <= t < vocab for t in prompt_ids):
        raise ValueError(f"token ids must be in [0, {vocab})")
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    max_tokens = (
        max_tokens_default if max_tokens is None else _int(max_tokens, "max_tokens")
    )
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    steps = _int(body.get("k3_steps") or 0, "k3_steps")
    if steps and steps not in steps_allowed:
        raise ValueError(f"k3_steps must be one of {sorted(steps_allowed)}")
    prefill = body.get("k3_prefill") or "rows"
    if prefill not in ("rows", "loop"):
        raise ValueError("k3_prefill must be rows or loop")
    spec = dict(
        prompt=list(prompt_ids),
        max_tokens=max_tokens,
        ignore_eos=bool(body.get("ignore_eos", False)),
        stop=_stops(body.get("stop")),
        prefill_mode=prefill,
        steps=steps,
    )
    used = [k for k in _BENCH_HOOKS if body.get(k) is not None]
    if used and not bench_hooks:
        raise ValueError(
            f"{used}: benchmark / test hooks, off in serving (server flag --bench-hooks)"
        )
    if used:
        spec.update(_bench_hooks(body))
    return spec


def _bench_hooks(body: dict) -> dict:
    """Job fields of the --bench-hooks request fields (see ``_parse``)."""
    stop_ids = body.get("k3_stop_ids") or []
    if not isinstance(stop_ids, list):
        raise ValueError("k3_stop_ids must be a list of token ids")
    force = _int(body.get("k3_force_acc") or 0, "k3_force_acc")
    if not 0 <= force <= 8:
        raise ValueError("k3_force_acc must be in [0, 8]")
    draft = body.get("k3_draft") or "dspark"
    if draft not in ("dspark", "zero"):
        raise ValueError("k3_draft must be dspark or zero")
    return dict(
        stop_ids=[_int(t, "k3_stop_ids") for t in stop_ids],
        force_acc=force,
        zero_drafts=draft == "zero",
    )


def _prompt_ids(tok, prompt):
    """``prompt`` of /v1/completions: text, a one-element list of text, or token ids."""
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], (str, list)):
        if len(prompt) != 1:
            raise ValueError("only one prompt per request is supported")
        prompt = prompt[0]
    if isinstance(prompt, str):
        return tok.encode(prompt)
    return prompt


def make_app(
    *,
    jobs: "queue.Queue",
    tok,
    model_name: str,
    max_prompt: int,
    vocab: int,
    steps_allowed=(),
    bench_hooks: bool = False,
):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()

    def new_job(prompt_ids, body, max_tokens_default):
        spec = job_spec(
            prompt_ids,
            body,
            max_tokens_default=max_tokens_default,
            max_prompt=max_prompt,
            vocab=vocab,
            steps_allowed=set(steps_allowed),
            bench_hooks=bench_hooks,
        )
        job = Job(spec, asyncio.get_running_loop())
        jobs.put(job)
        return job

    class Detok:
        """Incremental detokenizer (prefix / read offsets, as sglang's): text already sent never changes, an
        incomplete UTF-8 tail is held back until the next tokens (or the end), stop strings cut the text."""

        def __init__(self, stop):
            self.ids, self.sent, self.stop, self.stopped = [], "", stop, False
            self.text = ""  # decoded so far (sent + held back)
            self.prefix, self.read = 0, 0

        def _delta(self, final):
            prefix_text = tok.decode(
                self.ids[self.prefix : self.read], skip_special_tokens=True
            )
            new_text = tok.decode(self.ids[self.prefix :], skip_special_tokens=True)
            # A UTF-8 character spans at most 4 byte tokens; a longer U+FFFD tail is invalid bytes, sent as is.
            # The first token is never held back (clients time TTFT by the first non-empty text).
            pending_utf8 = (
                new_text.endswith("\ufffd")
                and len(self.ids) - self.read < 4
                and self.read > 0
            )
            if len(new_text) <= len(prefix_text) or (pending_utf8 and not final):
                return ""
            self.prefix, self.read = self.read, len(self.ids)
            return new_text[len(prefix_text) :]

        def add(self, new, final):
            if self.stopped:
                return ""
            self.ids += new
            self.text += self._delta(final)
            text = self.text
            cut = None
            for s in self.stop:
                i = text.find(s, max(0, len(self.sent) - len(s)))
                if i >= 0 and (cut is None or i < cut):
                    cut = i
            if cut is not None:
                text, self.stopped = text[: max(cut, len(self.sent))], True
            elif not final:
                # Hold back a tail that could still grow into a stop string with the next tokens.
                text = text[: len(text) - _stop_prefix_len(text, self.stop)]
            delta = text[len(self.sent) :] if len(text) > len(self.sent) else ""
            self.sent += delta
            return delta

    async def events(job):
        while True:
            ev = await job.events.get()
            yield ev
            if ev[0] in ("done", "error"):
                return

    def usage(job, n):
        p = len(job.spec["prompt"])
        return dict(prompt_tokens=p, completion_tokens=n, total_tokens=p + n)

    def finish(reason, det):
        return "stop" if det.stopped or reason == "stop" else (reason or "length")

    async def run(job, stream, chunk_fn, final_fn):
        """``chunk_fn(delta, finish_reason, n_tokens)`` -> dict; ``final_fn(text, reason, n)`` -> dict."""
        det = Detok(job.spec["stop"])
        if stream:

            async def gen():
                n = 0
                async for ev in events(job):
                    if ev[0] == "tokens":
                        n += len(ev[1])
                        delta = det.add(ev[1], ev[2])
                        if delta or ev[2]:
                            yield (
                                "data: "
                                + json.dumps(
                                    chunk_fn(
                                        delta, finish(ev[3], det) if ev[2] else None, n
                                    )
                                )
                                + "\n\n"
                            )
                    elif ev[0] == "error":
                        yield (
                            "data: "
                            + json.dumps({"error": {"message": ev[1]}})
                            + "\n\n"
                        )
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        n, reason = 0, None
        async for ev in events(job):
            if ev[0] == "tokens":
                n += len(ev[1])
                det.add(ev[1], ev[2])
            elif ev[0] == "done":
                reason = ev[2]
            else:
                return JSONResponse({"error": {"message": ev[1]}}, status_code=500)
        return JSONResponse(final_fn(det.sent, finish(reason, det), n))

    def bad(e):
        return JSONResponse({"error": {"message": str(e)}}, status_code=400)

    @app.get("/health")
    @app.get("/health_generate")
    async def health():
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def models():
        return {
            "object": "list",
            "data": [{"id": model_name, "object": "model", "owned_by": "k3mk"}],
        }

    @app.get("/get_model_info")
    @app.get("/model_info")
    async def model_info():
        return {
            "model_path": model_name,
            "tokenizer_path": model_name,
            "is_generation": True,
        }

    @app.get("/server_info")
    @app.get("/get_server_info")
    async def server_info():
        return {
            "model_path": model_name,
            "backend": "k3_megakernel standalone",
            "internal_states": [{}],
        }

    @app.api_route("/flush_cache", methods=["GET", "POST"])
    async def flush_cache():
        return JSONResponse({"ok": True})

    @app.post("/v1/completions")
    async def completions(req: Request):
        body = await req.json()
        try:
            ids = _prompt_ids(tok, body.get("prompt", ""))
            job = new_job(ids, body, 16)
        except Exception as e:
            return bad(e)
        rid, created = f"cmpl-{uuid.uuid4().hex}", int(time.time())
        model = body.get("model", model_name)

        def chunk(delta, reason, n):
            out = dict(
                id=rid,
                object="text_completion",
                created=created,
                model=model,
                choices=[
                    dict(index=0, text=delta, logprobs=None, finish_reason=reason)
                ],
            )
            if reason is not None:
                out["usage"] = usage(job, n)
            return out

        def final(text, reason, n):
            return dict(
                id=rid,
                object="text_completion",
                created=created,
                model=model,
                choices=[dict(index=0, text=text, logprobs=None, finish_reason=reason)],
                usage=usage(job, n),
            )

        return await run(job, bool(body.get("stream")), chunk, final)

    @app.post("/v1/chat/completions")
    async def chat(req: Request):
        body = await req.json()
        try:
            kwargs = dict(body.get("chat_template_kwargs") or {})
            ids = tok.apply_chat_template(
                body["messages"], tokenize=True, add_generation_prompt=True, **kwargs
            )
            ids = ids["input_ids"] if isinstance(ids, dict) else ids
            job = new_job(ids, body, 2048)
        except Exception as e:
            return bad(e)
        rid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        model = body.get("model", model_name)
        first = [True]

        def chunk(delta, reason, n):
            d = {"content": delta}
            if first[0]:
                d["role"], first[0] = "assistant", False
            out = dict(
                id=rid,
                object="chat.completion.chunk",
                created=created,
                model=model,
                choices=[dict(index=0, delta=d, logprobs=None, finish_reason=reason)],
            )
            if reason is not None:
                out["usage"] = usage(job, n)
            return out

        def final(text, reason, n):
            return dict(
                id=rid,
                object="chat.completion",
                created=created,
                model=model,
                choices=[
                    dict(
                        index=0,
                        message={"role": "assistant", "content": text},
                        logprobs=None,
                        finish_reason=reason,
                    )
                ],
                usage=usage(job, n),
            )

        return await run(job, bool(body.get("stream")), chunk, final)

    @app.post("/generate")
    async def generate(req: Request):
        body = await req.json()
        sp = dict(body.get("sampling_params") or {})
        try:
            ids = (
                body["input_ids"]
                if body.get("input_ids") is not None
                else tok.encode(body["text"])
            )
            for key in ("n", "return_logprob", "top_logprobs_num"):
                if (sp if key == "n" else body).get(key) not in (None, False, 0, 1):
                    raise ValueError(f"{key!r} is not supported by this server")
            job = new_job(
                ids,
                dict(
                    max_tokens=sp.get("max_new_tokens"),
                    ignore_eos=sp.get("ignore_eos", False),
                    stop=sp.get("stop"),
                    k3_prefill=body.get("k3_prefill"),
                    k3_steps=body.get("k3_steps"),
                    **{k: body.get(k) for k in _BENCH_HOOKS},
                ),
                128,
            )
        except Exception as e:
            return bad(e)
        rid = uuid.uuid4().hex

        def meta(reason, n):
            return dict(
                id=rid,
                prompt_tokens=len(job.spec["prompt"]),
                completion_tokens=n,
                finish_reason=None if reason is None else {"type": reason},
            )

        det = Detok(job.spec["stop"])
        if body.get("stream"):

            async def gen():
                n = 0
                async for ev in events(job):
                    if ev[0] == "tokens":
                        n += len(ev[1])
                        det.add(ev[1], ev[2])
                        out = dict(
                            text=det.sent,
                            meta_info=meta(finish(ev[3], det) if ev[2] else None, n),
                        )
                        if body.get("return_ids"):
                            out["output_ids"] = det.ids
                        yield "data: " + json.dumps(out) + "\n\n"
                    elif ev[0] == "error":
                        yield (
                            "data: "
                            + json.dumps({"error": {"message": ev[1]}})
                            + "\n\n"
                        )
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")
        n, reason = 0, None
        async for ev in events(job):
            if ev[0] == "tokens":
                n += len(ev[1])
                det.add(ev[1], ev[2])
            elif ev[0] == "done":
                reason = ev[2]
            else:
                return JSONResponse({"error": {"message": ev[1]}}, status_code=500)
        return JSONResponse(dict(text=det.sent, meta_info=meta(finish(reason, det), n)))

    return app


def _serve_http(app, host, port, timeout_s: float = 60.0):
    """Starts uvicorn in a thread; raises if it does not come up (e.g. the port is taken)."""
    import uvicorn

    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", timeout_keep_alive=600
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="http")
    thread.start()
    deadline = time.time() + timeout_s
    while not server.started:
        if not thread.is_alive() or time.time() > deadline:
            raise RuntimeError(f"HTTP server failed to start on {host}:{port}")
        time.sleep(0.1)


# ----- job distribution -----
def _kv_client():
    from jax._src import distributed

    return distributed.global_state.client


def _publish(client, n: int, message: dict) -> None:
    client.key_value_set(_KV_PREFIX + str(n), json.dumps(message))
    if n > _KV_KEEP:
        try:
            client.key_value_delete(_KV_PREFIX + str(n - _KV_KEEP))
        except Exception as e:  # best effort: a leftover key only costs memory
            _log(f"kv delete failed: {e}")


def leader_loop(engine, tok, jobs: "queue.Queue"):
    """Numbers jobs as it publishes them: job n is published right before the leader runs it, so
    the followers see n = 1, 2, ... without gaps. A ``None`` job shuts every process down."""
    import jax

    client = _kv_client() if jax.process_count() > 1 else None
    n = 0
    while True:
        job = jobs.get()
        n += 1
        if job is None:
            if client is not None:
                _publish(client, n, {"shutdown": True})
            return
        job.spec["n"] = n
        if client is not None:
            _publish(client, n, {"job": job.spec})
        run_job(engine, tok, job)


def follower_loop(engine, tok):
    client = _kv_client()
    n, backoff = 1, 0.5
    while True:
        try:
            raw = client.blocking_key_value_get(_KV_PREFIX + str(n), 600_000)
        except Exception as e:
            if "DEADLINE" not in str(e).upper() and "TIMEOUT" not in str(e).upper():
                _log(f"kv get failed: {e}; retrying in {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(2 * backoff, 10.0)
            continue
        backoff = 0.5
        message = json.loads(raw)
        if message.get("shutdown"):
            return
        run_job(engine, tok, Job(message["job"]))
        n += 1


def _anchor_check(engine, max_tokens: int = 1000):
    """Engine runs on the anchor prompt; the loop-prefill sha equals the harness's
    (test/manual/k3_megakernel/bench/anchor.py with the same model options)."""
    import hashlib

    import numpy as np

    from sglang_tpu.k3_megakernel.server.loader import ANCHOR

    prompt = list(ANCHOR)
    for mode, fresh in (
        ("loop", True),
        ("loop", False),
        ("rows", True),
        ("rows", False),
    ):
        if fresh:
            engine.reset_states()
        ids, _, timing = engine.generate(prompt, max_tokens, (), prefill_mode=mode)
        sha = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:16]
        _log(
            f"ENGINE_CHECK prefill={mode} fresh={fresh} sha={sha} first={ids[:12]} {json.dumps(timing)}"
        )
    lookahead = engine.lookahead
    for steps, engine.lookahead in (
        (engine.K, 0),
        (engine.K, lookahead),
    ):
        ids, _, timing = engine.generate(
            prompt, max_tokens, (), prefill_mode="rows", steps=steps
        )
        sha = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:16]
        _log(
            f"ENGINE_CHECK steps={steps} lookahead={engine.lookahead} sha={sha} {json.dumps(timing)}"
        )
    engine.lookahead = lookahead


def main(argv=None):
    ns, model_argv = _parse(argv)
    from sglang_tpu.k3_megakernel import mesh as mesh_mod
    from sglang_tpu.k3_megakernel.server import loader

    mp = argparse.ArgumentParser(description="model options (after --)")
    loader.add_model_args(mp)
    args = mp.parse_args(model_argv)
    if args.relax:
        from sglang_tpu.envs import envs as tpu_envs

        tpu_envs.SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX.set(args.relax)
    mesh_mod.init_distributed(
        coordinator=args.coordinator,
        num_processes=args.num_processes,
        process_id=args.process_id,
    )
    import jax
    from transformers import AutoTokenizer

    from sglang_tpu.k3_megakernel.server.engine import Engine

    _log(
        f"processes {jax.process_count()} devices {jax.device_count()}; server {vars(ns)}; model args {model_argv}"
    )
    tok = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    mesh = mesh_mod.make_mesh()
    draft = None
    if ns.spec_draft == "dspark":
        from sglang_tpu.k3_megakernel.server.draft import DraftModel

        tp = min(ns.draft_tp, mesh.devices.size) if ns.draft_tp else None
        draft = DraftModel.load(ns.draft_path, mesh, embed_mode=args.embed, tp=tp)
    engine = Engine(
        args,
        mesh,
        decode_steps=ns.decode_steps,
        capacity=ns.capacity,
        lookahead=ns.lookahead,
        draft=draft,
        spec_rounds=ns.spec_rounds,
    )
    _log(f"weights loaded in {engine.load_s:.0f}s")
    if not ns.no_warmup:
        engine.warmup()
    if ns.anchor_check:
        _anchor_check(engine)
    if jax.process_index() != 0:
        follower_loop(engine, tok)
        return 0
    jobs = queue.Queue()
    app = make_app(
        jobs=jobs,
        tok=tok,
        model_name=ns.served_model_name or args.ckpt,
        max_prompt=engine.max_prompt,
        vocab=engine.vocab,
        steps_allowed=engine.compiled_steps(),
        bench_hooks=ns.bench_hooks,
    )
    _serve_http(app, ns.host, ns.port)
    import socket

    _log(
        f"serving on {ns.host}:{ns.port} host {socket.gethostname()} TPU_WORKER_ID {os.environ.get('TPU_WORKER_ID')} "
        f"(K={ns.decode_steps}, capacity {ns.capacity})"
    )
    leader_loop(engine, tok, jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
