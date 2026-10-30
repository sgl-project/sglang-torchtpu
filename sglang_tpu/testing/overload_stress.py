"""Overload stress harness for P/D serving tests: mixed load, 429s, drain.

A disaggregated engine does not shed load itself (max_queued_requests applies
only without disaggregation), so a P/D stress test caps admission at the router
and expects the overflow back as HTTP 429. The load here is closed-loop: each
client waits for its answer before it sends again, so the backlog an overload
leaves behind is bounded by the client count and the drain can have a deadline.
A token-bucket router refills however many requests are in flight, so steady
closed-loop load stops drawing 429s once every client is admitted; periodic
bursts larger than the bucket keep it shedding.

Every text request plants a random code at a random depth of a filler document
and checks the answer for it, so a wrong answer under load reads as corrupted KV
rather than noise. An answer cut off at max_tokens before naming the code proves
nothing either way: it counts as truncated, which fails only a long-context
probe, since a probe must come back answered. Image and video requests are load
only: any HTTP status other than 200 or 429 fails them, their answers are not
graded.
"""

import base64
import io
import math
import random
import re
import threading
import time
import uuid
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import msgspec
import requests
from PIL import Image, ImageDraw

# Outcome of one request, as the stress gates count it.
OK = "ok"
SHED = "shed"  # HTTP 429: the router turned the request away, as intended.
TIMEOUT = "timeout"  # HTTP 408 (router queue timeout) or no answer in time.
SERVER_ERROR = "server_error"  # HTTP 5xx.
CLIENT_ERROR = "client_error"  # Any other 4xx.
CONNECTION_ERROR = "connection_error"
WRONG_ANSWER = "wrong_answer"  # Finished, but the planted code is missing.
TRUNCATED = "truncated"  # Hit max_tokens before naming the code: inconclusive.
HARNESS_ERROR = "harness_error"  # The client raised: a stopped client hides load.
FAILURES = frozenset(
    {TIMEOUT, SERVER_ERROR, CLIENT_ERROR, CONNECTION_ERROR, WRONG_ANSWER, HARNESS_ERROR}
)

# Engine gauges. A request waiting anywhere in an engine sits in one of these
# queues, so a drained engine reports zero in all of them.
RUNNING_GAUGE = "sglang:num_running_reqs"
USAGE_GAUGE = "sglang:token_usage"
QUEUE_GAUGES = (
    "sglang:num_queue_reqs",
    "sglang:num_grammar_queue_reqs",
    "sglang:num_prefill_bootstrap_queue_reqs",
    "sglang:num_prefill_inflight_queue_reqs",
    "sglang:num_decode_prealloc_queue_reqs",
    "sglang:num_decode_transfer_queue_reqs",
    "sglang:num_decode_host_receive_queue_reqs",
)

FILLER_LINE = "The catalog contains ordinary folders and routine inventory records.\n"
CODE_QUESTION = (
    "What is the access code stated in the document above? Reply with the code only."
)
# No 0/O or 1/I, so a code read back correctly cannot differ by a lookalike.
_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
_COLORS = ("red", "green", "blue", "yellow", "purple", "orange")
_METRIC_SAMPLE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+(\S+)")


def classify_status(status: int) -> str:
    if 200 <= status < 300:
        return OK
    if status == HTTPStatus.TOO_MANY_REQUESTS:
        return SHED
    if status == HTTPStatus.REQUEST_TIMEOUT:
        return TIMEOUT
    if status >= 500:
        return SERVER_ERROR
    return CLIENT_ERROR


def jittered_backoff(
    attempt: int, rng: random.Random, *, base: float = 0.5, cap: float = 8.0
) -> float:
    """Full-jitter exponential backoff, so shed clients do not retry in lockstep."""
    return rng.uniform(0.0, min(cap, base * 2**attempt))


# --- engine load -------------------------------------------------------------


class EngineLoad(msgspec.Struct, frozen=True):
    url: str
    running: float
    queued: float
    token_usage: float

    def is_idle(self, usage_ceiling: float) -> bool:
        return (
            self.running == 0 and self.queued == 0 and self.token_usage <= usage_ceiling
        )


def engine_load_from_metrics(url: str, text: str) -> EngineLoad:
    """Sum the request gauges over label sets; report the busiest pool's usage."""
    samples: dict[str, list[float]] = {
        name: [] for name in (RUNNING_GAUGE, USAGE_GAUGE, *QUEUE_GAUGES)
    }
    for line in text.splitlines():
        match = _METRIC_SAMPLE.match(line)
        if match and match.group(1) in samples:
            samples[match.group(1)].append(float(match.group(2)))
    # The P/D queue gauges exist only on the engine role that has the queue, but
    # an engine without these two exports no load at all: reading them as zero
    # would pass the drain check against a server with metrics turned off.
    missing = [name for name in (RUNNING_GAUGE, USAGE_GAUGE) if not samples[name]]
    if missing:
        raise ValueError(f"{url}/metrics has no {missing}; is --enable-metrics set?")
    return EngineLoad(
        url=url,
        running=sum(samples[RUNNING_GAUGE]),
        queued=float(sum(sum(samples[name]) for name in QUEUE_GAUGES)),
        token_usage=max(samples[USAGE_GAUGE]),
    )


def scrape_engine(url: str, timeout: float = 10.0) -> EngineLoad:
    response = requests.get(url.rstrip("/") + "/metrics", timeout=timeout)
    response.raise_for_status()
    return engine_load_from_metrics(url, response.text)


class DrainResult(msgspec.Struct, frozen=True):
    idle: bool
    waited_s: float
    loads: list[EngineLoad]
    error: str | None = None


def wait_until_idle(
    urls: Sequence[str],
    *,
    timeout: float,
    usage_ceiling: float,
    interval: float = 5.0,
) -> DrainResult:
    """Poll every engine until all are idle at once, or the timeout passes."""
    start = time.monotonic()
    while True:
        error = None
        try:
            loads = [scrape_engine(url) for url in urls]
        except (requests.RequestException, ValueError) as exc:
            loads, error = [], str(exc)
        waited = time.monotonic() - start
        if loads and all(load.is_idle(usage_ceiling) for load in loads):
            return DrainResult(idle=True, waited_s=waited, loads=loads)
        if waited >= timeout:
            return DrainResult(idle=False, waited_s=waited, loads=loads, error=error)
        time.sleep(interval)


class MetricsMonitor:
    """Sample engine load in the background and keep each engine's peaks."""

    def __init__(self, urls: Iterable[str], *, interval: float = 5.0):
        self.urls = list(urls)
        self.interval = interval
        self.peak_queued = dict.fromkeys(self.urls, 0.0)
        self.peak_running = dict.fromkeys(self.urls, 0.0)
        self.rounds = 0
        self.errors = 0
        self.last_error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="metrics-monitor", daemon=True
        )

    def start(self) -> "MetricsMonitor":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def reset_peaks(self) -> None:
        """Start a new window: peaks from here on describe only what follows."""
        with self._lock:
            self.peak_queued = dict.fromkeys(self.urls, 0.0)
            self.peak_running = dict.fromkeys(self.urls, 0.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            for url in self.urls:
                try:
                    load = scrape_engine(url)
                except (requests.RequestException, ValueError) as exc:
                    self.errors += 1
                    self.last_error = f"{url}: {exc}"
                    continue
                with self._lock:
                    self.peak_queued[url] = max(self.peak_queued[url], load.queued)
                    self.peak_running[url] = max(self.peak_running[url], load.running)
            self.rounds += 1
            self._stop.wait(self.interval)


# --- request content -----------------------------------------------------------


def new_code(rng: random.Random) -> str:
    def part() -> str:
        return "".join(rng.choice(_CODE_ALPHABET) for _ in range(4))

    return f"{part()}-{part()}"


def planted_document(
    code: str, *, filler_lines: int, depth: float, salt: str | None = None
) -> str:
    """Filler lines with one sentence stating `code` at `depth` (0..1) of them.

    `salt` opens the document with a line of its own, so its prefix shares no
    cache entry with any other request.
    """
    before = round(filler_lines * depth)
    return (
        (f"{salt}\n" if salt else "")
        + FILLER_LINE * before
        + f"The access code is {code}.\n"
        + FILLER_LINE * (filler_lines - before)
        + CODE_QUESTION
    )


def answer_has_code(answer: str, code: str) -> bool:
    return code.upper() in answer.upper()


class PromptSizer(msgspec.Struct, frozen=True):
    """Filler lines for a target prompt length, from two measured prompts."""

    tokens_per_line: float
    overhead_tokens: int
    # Codes tokenize to slightly different lengths than the measured one.
    margin_tokens: int = 64

    @classmethod
    def from_measurements(
        cls, lines: tuple[int, int], prompt_tokens: tuple[int, int]
    ) -> "PromptSizer":
        tokens_per_line = (prompt_tokens[1] - prompt_tokens[0]) / (lines[1] - lines[0])
        if tokens_per_line <= 0:
            raise ValueError(f"filler measured {tokens_per_line} tokens per line")
        overhead = math.ceil(prompt_tokens[0] - lines[0] * tokens_per_line)
        return cls(tokens_per_line=tokens_per_line, overhead_tokens=overhead)

    def lines_for(self, prompt_tokens: int) -> int:
        budget = prompt_tokens - self.overhead_tokens - self.margin_tokens
        lines = math.floor(budget / self.tokens_per_line)
        if lines < 1:
            raise ValueError(f"{prompt_tokens} tokens leave no room for filler")
        return lines


def image_part(color: str, size: int = 280) -> dict[str, Any]:
    """A PNG data URL: one centered square of `color` on white."""
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


def video_part(
    colors: Sequence[str],
    *,
    fps: int = 8,
    frames_per_color: int = 8,
    size: tuple[int, int] = (280, 224),
    variant: int | None = None,
) -> dict[str, Any]:
    """An MP4 data URL with one centered square color per temporal phase.

    `variant` adds gray border marks, so two clips of the same colors encode to
    different content without the text prompt revealing anything.
    """
    import av

    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "ultrafast"}
        stream.codec_context.thread_count = 1
        index = 0
        width, height = size
        side = int(min(size) * 0.72)
        left, top = (width - side) // 2, (height - side) // 2
        for color in colors:
            frame_image = Image.new("RGB", size, "white")
            draw = ImageDraw.Draw(frame_image)
            draw.rectangle((left, top, left + side, top + side), fill=color)
            if variant is not None:
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


# --- chat client -------------------------------------------------------------


class ChatResult(msgspec.Struct, frozen=True):
    category: str
    status: int
    latency_s: float
    text: str = ""
    finish_reason: str = ""
    prompt_tokens: int = 0
    detail: str = ""


def post_chat(
    base_url: str,
    *,
    model: str,
    content: list[dict[str, Any]],
    max_tokens: int,
    timeout: float,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> ChatResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = dict(chat_template_kwargs)
    start = time.monotonic()
    try:
        response = requests.post(
            base_url.rstrip("/") + "/v1/chat/completions", json=payload, timeout=timeout
        )
    except requests.Timeout as exc:
        return ChatResult(TIMEOUT, 0, time.monotonic() - start, detail=str(exc))
    except requests.RequestException as exc:
        return ChatResult(
            CONNECTION_ERROR, 0, time.monotonic() - start, detail=str(exc)
        )
    latency = time.monotonic() - start
    category = classify_status(response.status_code)
    if category != OK:
        return ChatResult(
            category, response.status_code, latency, detail=response.text[:300]
        )
    try:
        body = response.json()
        choice = body["choices"][0]
        text = choice["message"].get("content") or ""
        finish_reason = choice.get("finish_reason") or ""
        prompt_tokens = int(body.get("usage", {}).get("prompt_tokens", 0))
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return ChatResult(
            SERVER_ERROR,
            response.status_code,
            latency,
            detail=f"malformed completion ({exc}): {response.text[:200]}",
        )
    return ChatResult(
        OK,
        response.status_code,
        latency,
        text=text,
        finish_reason=finish_reason,
        prompt_tokens=prompt_tokens,
    )


def post_chat_until_admitted(
    base_url: str, *, deadline: float, rng: random.Random, **kwargs: Any
) -> ChatResult:
    """Repeat a request while the router sheds it, until `deadline` (monotonic)."""
    attempt = 0
    while True:
        result = post_chat(base_url, **kwargs)
        delay = jittered_backoff(attempt, rng)
        if result.category != SHED or time.monotonic() + delay >= deadline:
            return result
        time.sleep(delay)
        attempt += 1


def measure_prompt_sizer(
    base_url: str,
    *,
    model: str,
    timeout: float,
    lines: tuple[int, int] = (256, 2304),
) -> PromptSizer:
    """Fit filler tokens per line from the server's own prompt token counts."""
    rng = random.Random(0)
    counts = []
    for count in lines:
        result = post_chat_until_admitted(
            base_url,
            deadline=time.monotonic() + timeout,
            rng=rng,
            model=model,
            content=[
                {
                    "type": "text",
                    "text": planted_document(
                        new_code(rng), filler_lines=count, depth=0.5
                    ),
                }
            ],
            max_tokens=1,
            timeout=timeout,
        )
        if result.category != OK or result.prompt_tokens <= 0:
            raise RuntimeError(f"sizing a {count}-line prompt failed: {result}")
        counts.append(result.prompt_tokens)
    return PromptSizer.from_measurements(lines, (counts[0], counts[1]))


# --- eval relay --------------------------------------------------------------


class EvalRetryRelay:
    """Local HTTP relay for eval clients: retries 429s, relays everything else.

    A shed eval request is not an answer, so it must not reach the eval client as
    one: the relay sends it again with backoff until it is admitted or its retry
    budget runs out. Any other error goes back to the client unchanged and is
    recorded here, so a 5xx or timeout behind the router fails the stress gate
    even when the eval client's own retries would hide it from the score.
    """

    _FORWARDED_HEADERS = ("content-type", "authorization", "accept")

    def __init__(
        self,
        upstream: str,
        *,
        retry_budget_s: float,
        request_timeout_s: float,
        host: str = "127.0.0.1",
    ):
        self.upstream = upstream.rstrip("/")
        self.retry_budget_s = retry_budget_s
        self.request_timeout_s = request_timeout_s
        self.answered = 0
        self.retried_429 = 0
        self.failures: list[str] = []
        self._lock = threading.Lock()
        self._rng = random.Random(0)
        relay = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                relay._handle(self)

            def do_POST(self) -> None:
                relay._handle(self)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer((host, 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="eval-relay", daemon=True
        )

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "EvalRetryRelay":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        body = handler.rfile.read(length) if length else None
        headers = {
            name: value
            for name, value in handler.headers.items()
            if name.lower() in self._FORWARDED_HEADERS
        }
        status, payload, content_type = self._forward(
            handler.command, handler.path, body, headers
        )
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            handler.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass  # The client gave up; the outcome is already recorded.

    def _forward(
        self, method: str, path: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[int, bytes, str]:
        deadline = time.monotonic() + self.retry_budget_s
        attempt = 0
        while True:
            try:
                response = requests.request(
                    method,
                    self.upstream + path,
                    data=body,
                    headers=headers,
                    timeout=self.request_timeout_s,
                )
            except requests.Timeout as exc:
                return self._fail(TIMEOUT, path, str(exc), HTTPStatus.GATEWAY_TIMEOUT)
            except requests.RequestException as exc:
                return self._fail(
                    CONNECTION_ERROR, path, str(exc), HTTPStatus.BAD_GATEWAY
                )
            category = classify_status(response.status_code)
            if category == SHED:
                with self._lock:
                    delay = jittered_backoff(attempt, self._rng)
                if time.monotonic() + delay < deadline:
                    with self._lock:
                        self.retried_429 += 1
                    time.sleep(delay)
                    attempt += 1
                    continue
            with self._lock:
                if category == OK:
                    self.answered += 1
                else:
                    self.failures.append(
                        f"{category} HTTP {response.status_code} on {path} "
                        f"after {attempt} retries: {response.text[:200]}"
                    )
            return (
                response.status_code,
                response.content,
                response.headers.get("Content-Type", "application/json"),
            )

    def _fail(
        self, category: str, path: str, detail: str, status: HTTPStatus
    ) -> tuple[int, bytes, str]:
        with self._lock:
            self.failures.append(f"{category} on {path}: {detail[:200]}")
        payload = msgspec.json.encode({"error": {"message": detail, "type": category}})
        return int(status), payload, "application/json"


# --- load driver -------------------------------------------------------------


class LoadMix(msgspec.Struct, frozen=True):
    """Relative weights of the request kinds, and the text prompt lengths."""

    text: float
    image: float
    video: float
    text_prompt_tokens: tuple[int, ...]
    text_prompt_weights: tuple[float, ...]


class OverloadDriver:
    """Closed-loop mixed load, bursts, and long-context probes against a router.

    Background clients send text, image and video requests back to back; a shed
    request is dropped and the client sends a new one after a jittered backoff.
    The router's token bucket refills at a fixed rate however many requests are
    in flight, so closed-loop clients alone end up all admitted and stop drawing
    429s. Bursts keep the shedding going: every `burst_interval_s` the driver
    sends `burst_size` short text requests at once, which a bucket smaller than
    the burst must partly turn away. The next burst waits until the last one's
    admitted requests finish, so bursts add a bounded backlog.

    Probes are long planted-code documents that must complete: each is sent
    again while it is shed, until the probe deadline.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str,
        sizer: PromptSizer,
        mix: LoadMix,
        clients: int,
        probe_prompt_tokens: Sequence[int],
        max_tokens: int,
        probe_max_tokens: int,
        request_timeout_s: float,
        probe_deadline_s: float,
        burst_size: int = 0,
        burst_interval_s: float = 0.0,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        seed: int = 0,
    ):
        self.base_url = base_url
        self.model = model
        self.sizer = sizer
        self.mix = mix
        self.clients = clients
        self.probe_prompt_tokens = tuple(probe_prompt_tokens)
        self.max_tokens = max_tokens
        self.probe_max_tokens = probe_max_tokens
        self.request_timeout_s = request_timeout_s
        self.probe_deadline_s = probe_deadline_s
        self.burst_size = burst_size
        self.burst_interval_s = burst_interval_s
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.seed = seed
        self.counts: Counter[tuple[str, str]] = Counter()
        self.probe_results: dict[int, ChatResult] = {}
        self.failure_details: list[str] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def start(self) -> "OverloadDriver":
        # Daemon threads: if the flow raises, a probe still waiting on its
        # deadline must not keep the test process, and the slices, alive.
        deadline = time.monotonic() + self.probe_deadline_s
        targets = [
            (self._client, (index,), f"load-{index}") for index in range(self.clients)
        ]
        targets += [
            (self._probe, (index, tokens, deadline), f"probe-{tokens}")
            for index, tokens in enumerate(self.probe_prompt_tokens)
        ]
        if self.burst_size > 0:
            targets.append((self._bursts, (), "bursts"))
        for target, args, name in targets:
            self._threads.append(
                threading.Thread(target=target, args=args, name=name, daemon=True)
            )
        for thread in self._threads:
            thread.start()
        return self

    def stop_new_traffic(self) -> None:
        """Clients finish the request in flight and exit; probes run to the end."""
        self._stop.set()

    def join(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0.0, deadline - time.monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    def shed_count(self) -> int:
        with self._lock:
            return sum(
                n for (_, category), n in self.counts.items() if category == SHED
            )

    def failures(self) -> list[str]:
        with self._lock:
            failed = {
                f"{kind}/{category}": n
                for (kind, category), n in sorted(self.counts.items())
                if category in FAILURES
            }
            details = list(self.failure_details)
            probes = dict(self.probe_results)
        messages = []
        if failed:
            messages.append(f"load requests failed {failed}; first: {details[:5]}")
        for tokens in self.probe_prompt_tokens:
            result = probes.get(tokens)
            if result is None:
                messages.append(f"{tokens}-token probe never finished")
            elif result.category != OK:
                messages.append(
                    f"{tokens}-token probe ended {result.category}: "
                    f"{result.detail[:200]}"
                )
        return messages

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts = {f"{k}/{c}": n for (k, c), n in sorted(self.counts.items())}
            probes = {
                tokens: {
                    "category": result.category,
                    "prompt_tokens": result.prompt_tokens,
                    "latency_s": round(result.latency_s, 1),
                }
                for tokens, result in self.probe_results.items()
            }
        return {"counts": counts, "probes": probes}

    def _record(self, kind: str, result: ChatResult) -> None:
        with self._lock:
            self.counts[(kind, result.category)] += 1
            if result.category in FAILURES and len(self.failure_details) < 20:
                self.failure_details.append(
                    f"{kind} {result.category} HTTP {result.status}: "
                    f"{(result.detail or result.text)[:200]}"
                )

    def _post(self, content: list[dict[str, Any]], max_tokens: int) -> ChatResult:
        return post_chat(
            self.base_url,
            model=self.model,
            content=content,
            max_tokens=max_tokens,
            timeout=self.request_timeout_s,
            chat_template_kwargs=self.chat_template_kwargs,
        )

    def _graded(self, result: ChatResult, code: str) -> ChatResult:
        if result.category != OK or answer_has_code(result.text, code):
            return result
        category = TRUNCATED if result.finish_reason == "length" else WRONG_ANSWER
        return msgspec.structs.replace(
            result, category=category, detail=f"want {code}: {result.text[-200:]}"
        )

    def _client(self, index: int) -> None:
        rng = random.Random(self.seed * 10007 + index)
        shed_streak = 0
        while not self._stop.is_set():
            kind = rng.choices(
                ("text", "image", "video"),
                weights=(self.mix.text, self.mix.image, self.mix.video),
            )[0]
            result = self._safe_request(kind, rng)
            self._record(kind, result)
            if result.category == SHED:
                self._stop.wait(jittered_backoff(shed_streak, rng))
                shed_streak += 1
            else:
                shed_streak = 0

    def _bursts(self) -> None:
        rng = random.Random(self.seed * 10007 + 104729)
        while not self._stop.wait(self.burst_interval_s):
            threads = [
                threading.Thread(
                    target=self._burst_request,
                    args=(random.Random(rng.getrandbits(64)),),
                    name="burst",
                    daemon=True,
                )
                for _ in range(self.burst_size)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def _burst_request(self, rng: random.Random) -> None:
        # The shortest text prompt: a burst exists to empty the router's bucket,
        # and the requests it gets admitted should clear the engines quickly.
        result = self._safe(
            lambda: self._text_request(rng, min(self.mix.text_prompt_tokens))
        )
        self._record("burst", result)

    def _safe_request(self, kind: str, rng: random.Random) -> ChatResult:
        return self._safe(lambda: self._request(kind, rng))

    @staticmethod
    def _safe(send: Callable[[], ChatResult]) -> ChatResult:
        try:
            return send()
        except Exception as exc:
            return ChatResult(HARNESS_ERROR, 0, 0.0, detail=repr(exc))

    def _text_request(self, rng: random.Random, prompt_tokens: int) -> ChatResult:
        code = new_code(rng)
        document = planted_document(
            code, filler_lines=self.sizer.lines_for(prompt_tokens), depth=rng.random()
        )
        result = self._post([{"type": "text", "text": document}], self.max_tokens)
        return self._graded(result, code)

    def _request(self, kind: str, rng: random.Random) -> ChatResult:
        if kind == "text":
            tokens = rng.choices(
                self.mix.text_prompt_tokens, weights=self.mix.text_prompt_weights
            )[0]
            return self._text_request(rng, tokens)
        if kind == "image":
            content = [
                image_part(rng.choice(_COLORS), size=rng.choice((224, 280, 392))),
                {
                    "type": "text",
                    "text": "What color is the square? Reply with its color only.",
                },
            ]
        else:
            content = [
                video_part(rng.sample(_COLORS, 3), variant=rng.getrandbits(64)),
                {
                    "type": "text",
                    "text": "Name the square colors at the beginning, middle, "
                    "and end of the video. Reply only with the three colors.",
                },
            ]
        return self._post(content, self.max_tokens)

    def _probe(self, index: int, prompt_tokens: int, deadline: float) -> None:
        rng = random.Random(self.seed * 10007 + 7919 * (index + 1))
        code = new_code(rng)
        try:
            document = planted_document(
                code,
                filler_lines=self.sizer.lines_for(prompt_tokens),
                depth=rng.random(),
                salt=f"Document {uuid.UUID(int=rng.getrandbits(128))}",
            )
            result = post_chat_until_admitted(
                self.base_url,
                deadline=deadline,
                rng=rng,
                model=self.model,
                content=[{"type": "text", "text": document}],
                max_tokens=self.probe_max_tokens,
                timeout=self.request_timeout_s,
                chat_template_kwargs=self.chat_template_kwargs,
            )
            result = self._graded(result, code)
        except Exception as exc:
            result = ChatResult(HARNESS_ERROR, 0, 0.0, detail=repr(exc))
        self._record("probe", result)
        with self._lock:
            self.probe_results[prompt_tokens] = result
