"""Overload, drain and quality gates for a P/D deployment behind the mini-LB.

The stress flow has three parts that run against one mini-LB URL:

* ``RetryProxy`` sits between quality evals and the mini-LB. It retries every
  429 the mini-LB sheds until the request is answered or a deadline passes, so
  an eval under overload scores every item. Anything else (5xx, timeout,
  connection error) goes back to the eval unchanged and is recorded.
* ``LoadDriver`` floods the mini-LB with text, image, video and planted-code
  long-context requests, and classifies every response.
* ``EngineLoadMonitor`` polls each engine's ``/v1/loads`` to prove the queues
  filled and to wait until both engines are idle again.

Every response lands in a ``StressLedger``; the calling test turns the ledger
and monitor readings into pass or fail reasons.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import enum
import io
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import aiohttp
import requests
from aiohttp import web

logger = logging.getLogger(__name__)


class Outcome(enum.Enum):
    """How one request ended, from the client's point of view."""

    OK = "ok"
    SHED = "shed_429"
    WRONG_ANSWER = "wrong_answer"
    SERVER_ERROR = "server_error"
    CLIENT_ERROR = "client_error"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"


# Outcomes that fail the run. A 429 is the expected way to shed overload.
FAILING_OUTCOMES = frozenset(
    {
        Outcome.WRONG_ANSWER,
        Outcome.SERVER_ERROR,
        Outcome.CLIENT_ERROR,
        Outcome.TIMEOUT,
        Outcome.CONNECTION_ERROR,
    }
)


def classify_status(status: int) -> Outcome:
    """Map a final HTTP status to an outcome; 200 still needs answer checks."""
    if status == 200:
        return Outcome.OK
    if status == 429:
        return Outcome.SHED
    if status >= 500:
        return Outcome.SERVER_ERROR
    return Outcome.CLIENT_ERROR


@dataclass
class RequestRecord:
    kind: str
    outcome: Outcome
    status: Optional[int]
    latency_s: float
    detail: str = ""
    prompt_tokens: Optional[int] = None


class StressLedger:
    """Thread-safe record of every request outcome, grouped by request kind."""

    def __init__(self, max_failure_examples: int = 20):
        self._lock = threading.Lock()
        self._counts: dict[str, collections.Counter] = collections.defaultdict(
            collections.Counter
        )
        self._failures: list[RequestRecord] = []
        self._max_failure_examples = max_failure_examples
        self._max_prompt_tokens: dict[str, int] = {}
        self.retried_429 = 0

    def add(self, record: RequestRecord) -> None:
        with self._lock:
            self._counts[record.kind][record.outcome] += 1
            if record.outcome is Outcome.OK and record.prompt_tokens is not None:
                previous = self._max_prompt_tokens.get(record.kind, 0)
                self._max_prompt_tokens[record.kind] = max(
                    previous, record.prompt_tokens
                )
            if (
                record.outcome in FAILING_OUTCOMES
                and len(self._failures) < self._max_failure_examples
            ):
                self._failures.append(record)

    def note_retry(self) -> None:
        with self._lock:
            self.retried_429 += 1

    def count(self, outcome: Outcome, kind: Optional[str] = None) -> int:
        with self._lock:
            kinds = [kind] if kind is not None else list(self._counts)
            return sum(self._counts[k][outcome] for k in kinds)

    def failures(self) -> list[RequestRecord]:
        with self._lock:
            return list(self._failures)

    def max_prompt_tokens(self, kind: str) -> int:
        with self._lock:
            return self._max_prompt_tokens.get(kind, 0)

    def summary(self) -> dict:
        with self._lock:
            return {
                "by_kind": {
                    kind: {o.value: n for o, n in counts.items()}
                    for kind, counts in sorted(self._counts.items())
                },
                "retried_429": self.retried_429,
                "max_prompt_tokens": dict(self._max_prompt_tokens),
                "failure_examples": [
                    {
                        "kind": r.kind,
                        "outcome": r.outcome.value,
                        "status": r.status,
                        "detail": r.detail[:300],
                    }
                    for r in self._failures
                ],
            }


class _LoopThread:
    """An asyncio event loop on a daemon thread."""

    def __init__(self, name: str):
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, name=name)
        self._thread.daemon = True
        self._thread.start()

    def run(self, coro, timeout: Optional[float] = None):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def stop(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=10)


class RetryProxy:
    """Forward HTTP requests to ``target_url``, retrying 429s until answered.

    Every final response is recorded in the ledger under ``kind``. Retries
    back off from ``initial_backoff_s`` to ``max_backoff_s``; a request still
    shed at ``retry_deadline_s`` is returned as its last 429 and counts as a
    failure (the eval never got its answer).
    """

    def __init__(
        self,
        target_url: str,
        ledger: StressLedger,
        *,
        kind: str = "eval",
        port: int = 0,
        request_timeout_s: float = 1800.0,
        retry_deadline_s: float = 3600.0,
        initial_backoff_s: float = 0.5,
        max_backoff_s: float = 5.0,
    ):
        self.target_url = target_url.rstrip("/")
        self.ledger = ledger
        self.kind = kind
        self._requested_port = port
        self.request_timeout_s = request_timeout_s
        self.retry_deadline_s = retry_deadline_s
        self.initial_backoff_s = initial_backoff_s
        self.max_backoff_s = max_backoff_s
        self._loop: Optional[_LoopThread] = None
        self._runner: Optional[web.AppRunner] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self.port: Optional[int] = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> RetryProxy:
        self._loop = _LoopThread("retry-proxy")
        self._loop.run(self._start())
        return self

    def stop(self) -> None:
        if self._loop is None:
            return
        self._loop.run(self._stop(), timeout=30)
        self._loop.stop()
        self._loop = None

    async def _start(self) -> None:
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.request_timeout_s),
            # A fresh connection per request: at eval rates a pooled keep-alive
            # connection can sit idle past the mini-LB's keep-alive timeout, and
            # reusing it fails with ServerDisconnectedError before any response.
            connector=aiohttp.TCPConnector(force_close=True, limit=0),
        )
        app = web.Application(client_max_size=1 << 30)
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self._requested_port)
        await site.start()
        self.port = self._runner.addresses[0][1]

    async def _stop(self) -> None:
        await self._runner.cleanup()
        await self._session.close()

    async def _handle(self, request: web.Request) -> web.Response:
        body = await request.read()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        url = self.target_url + request.rel_url.path_qs
        started = time.monotonic()
        backoff = self.initial_backoff_s
        while True:
            try:
                async with self._session.request(
                    request.method, url, data=body, headers=headers
                ) as upstream:
                    payload = await upstream.read()
                    status = upstream.status
                    content_type = upstream.content_type
            except asyncio.TimeoutError:
                return self._finish(Outcome.TIMEOUT, None, started, "proxy timeout")
            except aiohttp.ClientError as error:
                return self._finish(
                    Outcome.CONNECTION_ERROR, None, started, repr(error)
                )
            elapsed = time.monotonic() - started
            if status == 429 and elapsed < self.retry_deadline_s:
                self.ledger.note_retry()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                continue
            outcome = classify_status(status)
            if status == 429:
                # Shed past the deadline: the eval item went unanswered.
                outcome = Outcome.CLIENT_ERROR
            self.ledger.add(
                RequestRecord(
                    kind=self.kind,
                    outcome=outcome,
                    status=status,
                    latency_s=elapsed,
                    detail=""
                    if status == 200
                    else payload[:300].decode(errors="replace"),
                )
            )
            return web.Response(body=payload, status=status, content_type=content_type)

    def _finish(
        self, outcome: Outcome, status: Optional[int], started: float, detail: str
    ) -> web.Response:
        self.ledger.add(
            RequestRecord(
                kind=self.kind,
                outcome=outcome,
                status=status,
                latency_s=time.monotonic() - started,
                detail=detail,
            )
        )
        return web.json_response({"error": {"message": detail}}, status=502)


# --------------------------------------------------------------------------
# Request builders
# --------------------------------------------------------------------------

_COLORS = ("red", "green", "blue", "yellow")
_RGB = {
    "red": (220, 20, 20),
    "green": (20, 170, 40),
    "blue": (20, 40, 220),
    "yellow": (240, 220, 20),
}


def square_image_part(color: str, size: int = 280) -> dict:
    """A PNG data-URI chat part with one centered square of ``color``."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), "white")
    inset = size // 7
    ImageDraw.Draw(image).rectangle(
        (inset, inset, size - inset, size - inset), fill=_RGB[color]
    )
    out = io.BytesIO()
    image.save(out, format="PNG")
    data = base64.b64encode(out.getvalue()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def color_video_part(
    colors: tuple[str, ...], *, fps: int = 4, frames_per_color: int = 8
) -> dict:
    """An MP4 data-URI chat part showing one square color per temporal phase."""
    from fractions import Fraction

    import av
    from PIL import Image, ImageDraw

    size = (280, 224)
    output = io.BytesIO()
    with av.open(output, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "ultrafast"}
        index = 0
        for color in colors:
            frame_image = Image.new("RGB", size, "white")
            side = int(min(size) * 0.72)
            left, top = (size[0] - side) // 2, (size[1] - side) // 2
            ImageDraw.Draw(frame_image).rectangle(
                (left, top, left + side, top + side), fill=_RGB[color]
            )
            for _ in range(frames_per_color):
                frame = av.VideoFrame.from_image(frame_image)
                frame.pts = index
                frame.time_base = Fraction(1, fps)
                index += 1
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    data = base64.b64encode(output.getvalue()).decode()
    return {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + data}}


def filler_line(index: int) -> str:
    """One distinct ledger line; distinct lines keep the prefix cache honest."""
    return (
        f"Ledger entry {index:07d}: crate {index * 7919 % 100000:05d} cleared "
        f"inspection at dock {index % 37} and moved to aisle {index % 211}.\n"
    )


@dataclass(frozen=True)
class PlantedPrompt:
    text: str
    code: str
    vault: str


def planted_code_prompt(num_lines: int, rng: random.Random) -> PlantedPrompt:
    """A long filler document with one access code planted at a random depth."""
    code = f"{rng.randrange(10**7, 10**8)}"
    vault = f"V{rng.randrange(1000, 10000)}"
    offset = rng.randrange(1 << 30)
    depth = rng.randrange(num_lines + 1)
    lines = [filler_line(offset + i) for i in range(num_lines)]
    lines.insert(depth, f"IMPORTANT: the access code for vault {vault} is {code}.\n")
    question = (
        f"\nWhat is the access code for vault {vault}? Reply with only the "
        "eight-digit code."
    )
    return PlantedPrompt(text="".join(lines) + question, code=code, vault=vault)


# --------------------------------------------------------------------------
# Load driver
# --------------------------------------------------------------------------


@dataclass
class TrafficMix:
    """Relative weights of each request kind in the overload traffic."""

    text_short: float = 0.55
    image: float = 0.2
    video: float = 0.1
    text_long: float = 0.15

    def pick(self, rng: random.Random) -> str:
        kinds = ("text_short", "image", "video", "text_long")
        weights = [getattr(self, k) for k in kinds]
        return rng.choices(kinds, weights=weights)[0]


@dataclass
class LoadDriverConfig:
    model: str
    concurrency: int = 64
    # Prompt lengths, in tokens, the long-context requests cycle through.
    long_context_tokens: tuple[int, ...] = (32768, 131072, 253952)
    max_concurrent_long: int = 3
    request_timeout_s: float = 1800.0
    # Long requests must complete, so a 429 is retried until this deadline.
    long_retry_deadline_s: float = 2400.0
    mix: TrafficMix = field(default_factory=TrafficMix)
    # Seconds a worker waits after its request is shed, drawn uniformly.
    shed_backoff_s: tuple[float, float] = (1.0, 3.0)
    seed: int = 1867
    extra_body: dict = field(
        default_factory=lambda: {"chat_template_kwargs": {"reasoning_effort": "low"}}
    )


class LoadDriver:
    """Concurrent mixed traffic against the mini-LB until ``stop()``."""

    def __init__(self, base_url: str, config: LoadDriverConfig, ledger: StressLedger):
        self.base_url = base_url.rstrip("/")
        self.config = config
        self.ledger = ledger
        self._rng = random.Random(config.seed)
        self._loop: Optional[_LoopThread] = None
        self._stop: Optional[asyncio.Event] = None
        self._workers: list[asyncio.Task] = []
        self._long_slots: Optional[asyncio.Semaphore] = None
        self._long_index = 0
        self._tokens_per_line: Optional[float] = None
        self.started_at: Optional[float] = None

    def calibrate(self) -> float:
        """Measure prompt tokens per filler line with one short request."""
        lines = 400
        prompt = planted_code_prompt(lines, random.Random(0))
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json=self._chat_body([{"type": "text", "text": prompt.text}], 1),
            timeout=600,
        )
        response.raise_for_status()
        prompt_tokens = response.json()["usage"]["prompt_tokens"]
        self._tokens_per_line = prompt_tokens / lines
        return self._tokens_per_line

    def start(self) -> None:
        if self._tokens_per_line is None:
            self.calibrate()
        self._loop = _LoopThread("load-driver")
        self._loop.run(self._start())
        self.started_at = time.monotonic()

    def stop(self, drain_timeout_s: float) -> bool:
        """Stop issuing requests; True if in-flight requests ended in time."""
        if self._loop is None:
            return True
        done = self._loop.run(self._stop_and_wait(drain_timeout_s))
        self._loop.stop()
        self._loop = None
        return done

    async def _start(self) -> None:
        self._stop = asyncio.Event()
        self._long_slots = asyncio.Semaphore(self.config.max_concurrent_long)
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_s),
            # See RetryProxy: never reuse a keep-alive connection the mini-LB
            # may already have closed.
            connector=aiohttp.TCPConnector(limit=0, force_close=True),
        )
        self._workers = [
            asyncio.create_task(self._worker(i)) for i in range(self.config.concurrency)
        ]

    async def _stop_and_wait(self, timeout: float) -> bool:
        self._stop.set()
        done, pending = await asyncio.wait(self._workers, timeout=timeout)
        for task in pending:
            task.cancel()
        await self._session.close()
        return not pending

    def _chat_body(self, content: list, max_tokens: int) -> dict:
        return {
            "model": self.config.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
            **self.config.extra_body,
        }

    async def _worker(self, index: int) -> None:
        rng = random.Random(self.config.seed * 1000 + index)
        while not self._stop.is_set():
            kind = self.config.mix.pick(rng)
            if kind == "text_long":
                if self._long_slots.locked():
                    kind = "text_short"
                else:
                    async with self._long_slots:
                        await self._long_request(rng)
                    continue
            shed = await self._one_request(kind, rng)
            if shed:
                # A shed client backs off like a real one. Retrying at once
                # turns the mini-LB into a 429 loop that crowds out the evals.
                await asyncio.sleep(rng.uniform(*self.config.shed_backoff_s))

    async def _one_request(self, kind: str, rng: random.Random) -> bool:
        if kind == "image":
            color = rng.choice(_COLORS)
            content = [
                square_image_part(color),
                {
                    "type": "text",
                    "text": "What color is the square? Reply with the color only.",
                },
            ]
            expect = (color,)
            max_tokens = 512
        elif kind == "video":
            colors = tuple(rng.sample(_COLORS, 2))
            content = [
                color_video_part(colors),
                {
                    "type": "text",
                    "text": "Name the square colors at the beginning and end of "
                    "the video, in order. Reply with only the two colors.",
                },
            ]
            expect = colors
            max_tokens = 512
        else:
            topic = rng.randrange(1, 10**6)
            content = [
                {
                    "type": "text",
                    "text": f"Write two sentences about warehouse shipment {topic}.",
                }
            ]
            expect = ()
            max_tokens = rng.choice((64, 128, 256))
        return await self._send(
            kind, self._chat_body(content, max_tokens), expect, retry=False
        )

    async def _long_request(self, rng: random.Random) -> None:
        target = self.config.long_context_tokens[
            self._long_index % len(self.config.long_context_tokens)
        ]
        self._long_index += 1
        # Leave room for the question, chat template and answer.
        lines = max(1, int((target - 1024) / self._tokens_per_line))
        prompt = planted_code_prompt(lines, rng)
        body = self._chat_body([{"type": "text", "text": prompt.text}], 1024)
        kind = f"text_long_{target // 1024}k"
        await self._send(kind, body, (prompt.code,), retry=True)

    async def _send(self, kind: str, body: dict, expect: tuple, *, retry: bool) -> bool:
        """Send one request and record it; True if it ended shed with a 429."""
        started = time.monotonic()
        backoff = 0.5
        while True:
            try:
                async with self._session.post(
                    self.base_url + "/v1/chat/completions", json=body
                ) as response:
                    status = response.status
                    payload = await response.read()
            except asyncio.TimeoutError:
                self.ledger.add(
                    RequestRecord(
                        kind, Outcome.TIMEOUT, None, time.monotonic() - started
                    )
                )
                return False
            except aiohttp.ClientError as error:
                self.ledger.add(
                    RequestRecord(
                        kind,
                        Outcome.CONNECTION_ERROR,
                        None,
                        time.monotonic() - started,
                        repr(error),
                    )
                )
                return False
            if (
                status == 429
                and retry
                and time.monotonic() - started < self.config.long_retry_deadline_s
            ):
                self.ledger.note_retry()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
                continue
            break
        latency = time.monotonic() - started
        outcome = classify_status(status)
        detail = ""
        prompt_tokens = None
        if outcome is Outcome.OK:
            data = json.loads(payload)
            prompt_tokens = data.get("usage", {}).get("prompt_tokens")
            choice = data["choices"][0]
            message = choice["message"]
            text = (message.get("content") or "").lower()
            if expect:
                if not _answer_matches(text, expect):
                    outcome = Outcome.WRONG_ANSWER
                    detail = (
                        f"expected {expect}, got {text[:200]!r} "
                        f"(finish_reason={choice.get('finish_reason')})"
                    )
            elif (
                not text.strip()
                and not (message.get("reasoning_content") or "").strip()
            ):
                # Free text has no reference answer; a reasoning model may spend
                # a short budget thinking, so either field shows it generated.
                outcome = Outcome.WRONG_ANSWER
                detail = (
                    f"empty completion (finish_reason={choice.get('finish_reason')})"
                )
        elif outcome is not Outcome.SHED:
            detail = payload[:300].decode(errors="replace")
        self.ledger.add(
            RequestRecord(kind, outcome, status, latency, detail, prompt_tokens)
        )
        return outcome is Outcome.SHED


def _answer_matches(text: str, expect: tuple) -> bool:
    """True if every expected token appears in order in ``text``."""
    position = 0
    for token in expect:
        found = text.find(str(token).lower(), position)
        if found < 0:
            return False
        position = found + len(str(token))
    return True


# --------------------------------------------------------------------------
# Engine load monitor
# --------------------------------------------------------------------------

_QUEUE_FIELDS = (
    "prefill_bootstrap_queue_reqs",
    "prefill_inflight_queue_reqs",
    "decode_prealloc_queue_reqs",
    "decode_transfer_queue_reqs",
    "decode_retracted_queue_reqs",
)


@dataclass
class EngineLoad:
    running: int
    waiting: int
    pd_queued: int
    token_usage: float

    @property
    def busy(self) -> bool:
        return bool(self.running or self.waiting or self.pd_queued)


def parse_loads(payload: dict) -> EngineLoad:
    """Sum one engine's /v1/loads response over its DP ranks."""
    loads = payload["loads"]
    return EngineLoad(
        running=sum(load.get("num_running_reqs", 0) for load in loads),
        waiting=sum(load.get("num_waiting_reqs", 0) for load in loads),
        pd_queued=sum(
            (load.get("disaggregation") or {}).get(name, 0)
            for load in loads
            for name in _QUEUE_FIELDS
        ),
        token_usage=max((load.get("token_usage", 0.0) for load in loads), default=0.0),
    )


class EngineLoadMonitor:
    """Poll each engine's /v1/loads in the background and keep the peaks."""

    def __init__(
        self,
        engines: dict[str, str],
        *,
        interval_s: float = 2.0,
        fetch: Optional[Callable[[str], dict]] = None,
    ):
        self.engines = {name: url.rstrip("/") for name, url in engines.items()}
        self.interval_s = interval_s
        self._fetch = fetch or _fetch_loads
        self._lock = threading.Lock()
        self.peak: dict[str, EngineLoad] = {
            name: EngineLoad(0, 0, 0, 0.0) for name in self.engines
        }
        self.latest: dict[str, Optional[EngineLoad]] = dict.fromkeys(self.engines)
        self.poll_errors = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def read(self) -> dict[str, EngineLoad]:
        loads = {
            name: parse_loads(self._fetch(url)) for name, url in self.engines.items()
        }
        with self._lock:
            for name, load in loads.items():
                peak = self.peak[name]
                self.peak[name] = EngineLoad(
                    max(peak.running, load.running),
                    max(peak.waiting, load.waiting),
                    max(peak.pd_queued, load.pd_queued),
                    max(peak.token_usage, load.token_usage),
                )
                self.latest[name] = load
        return loads

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, name="load-monitor", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.read()
            except Exception as error:  # noqa: BLE001 - keep polling through blips
                self.poll_errors += 1
                logger.warning("load poll failed: %r", error)

    def wait_idle(
        self,
        *,
        timeout_s: float,
        token_usage_ceiling: dict[str, float],
        settle_polls: int = 3,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> tuple[bool, dict[str, EngineLoad]]:
        """Wait until every engine is idle for ``settle_polls`` reads in a row.

        Idle means no running, waiting or P/D-queued requests and KV usage at
        or under the engine's ceiling (its pre-load baseline plus slack).
        """
        deadline = clock() + timeout_s
        streak = 0
        loads: dict[str, EngineLoad] = {}
        while True:
            try:
                loads = self.read()
                idle = all(
                    not load.busy and load.token_usage <= token_usage_ceiling[name]
                    for name, load in loads.items()
                )
            except Exception as error:  # noqa: BLE001
                logger.warning("idle poll failed: %r", error)
                idle = False
            streak = streak + 1 if idle else 0
            if streak >= settle_polls:
                return True, loads
            if clock() >= deadline:
                return False, loads
            sleep(self.interval_s)


def _fetch_loads(url: str) -> dict:
    response = requests.get(
        url + "/v1/loads", params={"include": "core,disagg"}, timeout=30
    )
    response.raise_for_status()
    return response.json()


def scrape_counters(url: str, names: tuple[str, ...]) -> dict[str, float]:
    """Sum Prometheus counters named ``sglang:<name>`` from an engine's /metrics."""
    response = requests.get(url.rstrip("/") + "/metrics", timeout=30)
    response.raise_for_status()
    totals = dict.fromkeys(names, 0.0)
    for line in response.text.splitlines():
        if line.startswith("#"):
            continue
        key, _, value = line.rpartition(" ")
        metric = key.split("{", 1)[0]
        if metric.startswith("sglang:") and metric[len("sglang:") :] in totals:
            totals[metric[len("sglang:") :]] += float(value)
    return totals
