# SPDX-License-Identifier: Apache-2.0
"""Render the server's Prometheus /metrics outside the serving event loop.

Why: core mounts prometheus_client's make_asgi_app with a MultiProcessCollector. Its handler merges every
     process's .db file and encodes the exposition synchronously inside the async handler, so the uvicorn
     loop that streams tokens stalls for the whole render: 56-96 ms per scrape on DSV4.1 AgentX at c4,
     which delays the final chunks of a stream and costs up to ~6% p90 interactivity.
How: AROUND server_utils.add_prometheus_middleware mounts an ASGI app that hands each scrape to one
     standalone helper process (sglang_tpu/metrics_render_helper.py: stdlib + prometheus_client only,
     rendering with prometheus_client's own _bake_output) and
     awaits the answer, so the loop keeps serving. Scrapes are coalesced: every request is answered by a
     render that starts after it arrived (fresh, as stock), and requests waiting for the same
     representation share one render (the key is the representation stock would produce). Negotiation
     (Accept, Accept-Encoding, name[]) and the response bytes are the same as stock. A render that takes longer than SGLANG_TPU_METRICS_RENDER_TIMEOUT_S is
     answered 503 and the helper is restarted. Not applied with several tokenizer workers (each spawned
     worker mounts its own app without plugins); those keep the stock app.
Upstream PR or issue: none filed; the same stall exists on every platform (core server_utils.py).
Remove when: core renders /metrics off the event loop.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import re
import sys
import time

logger = logging.getLogger(__name__)

_HELPER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "metrics_render_helper.py",
)
_RESTARTS_PER_MINUTE = 5


class OffloopMetricsRenderer:
    """One helper process; renders are serialized there and coalesced per representation here."""

    def __init__(self, multiproc_dir: str, timeout_s: float):
        self.multiproc_dir = multiproc_dir
        self.timeout_s = timeout_s
        self.proc = None
        self.pending: dict = {}  # key -> [future, [waiters]] of a render that has not started yet
        self.draining = False
        self.stopping = False
        self.renders = 0
        self.starts: list = []  # monotonic times of helper starts, to bound a restart loop
        atexit.register(self.stop)

    def stop(self):
        self.stopping = True
        p, self.proc = self.proc, None
        if p is not None and p.returncode is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass

    async def aclose(self):
        """stop() and reap the helper (app shutdown, tests)."""
        p = self.proc
        self.stop()
        if p is not None:
            await p.wait()
            if p.stdin is not None:
                p.stdin.close()

    async def _ensure_proc(self):
        if self.proc is not None and self.proc.returncode is None:
            return self.proc
        if self.stopping:
            raise RuntimeError("metrics renderer is stopping")
        now = time.monotonic()
        self.starts = [t for t in self.starts if now - t < 60.0]
        if (
            len(self.starts) >= _RESTARTS_PER_MINUTE
        ):  # it keeps dying, or is being killed at teardown
            raise RuntimeError(
                f"metrics helper started {len(self.starts)} times in 60 s"
            )
        self.starts.append(now)
        # -P keeps this package's directory off sys.path (sglang_tpu/platform.py would shadow the
        # stdlib) while honouring PYTHONPATH and user site-packages for prometheus_client.
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-P",
            _HELPER,
            self.multiproc_dir,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        if self.stopping:  # stop() ran while the helper was starting
            proc.kill()
            await proc.wait()
            raise RuntimeError("metrics renderer is stopping")
        self.proc = proc
        return proc

    async def _render_once(self, key):
        accept, accept_encoding, query = key  # content type, "gzip" or "", name[] query
        proc = await self._ensure_proc()
        req = json.dumps(
            {"accept": accept, "accept_encoding": accept_encoding, "query": query}
        )
        proc.stdin.write(req.encode() + b"\n")
        await proc.stdin.drain()
        head = json.loads(await proc.stdout.readline())
        body = await proc.stdout.readexactly(head["len"])
        self.renders += 1
        return (
            head["status"],
            [(k.encode(), v.encode()) for k, v in head["headers"]],
            body,
        )

    async def _drain(self):
        try:
            while self.pending:
                key = next(iter(self.pending))
                fut, waiters = self.pending.pop(
                    key
                )  # from here on, new requests queue a later render
                if waiters[0] <= 0:  # every caller left before the render started
                    fut.cancel()
                    continue
                try:
                    result = await asyncio.wait_for(
                        self._render_once(key), self.timeout_s
                    )
                except Exception as e:
                    if not self.stopping:
                        logger.warning(
                            "off-loop /metrics render failed (%r); restarting the helper",
                            e,
                        )
                    p, self.proc = self.proc, None
                    if p is not None and p.returncode is None:
                        try:
                            p.kill()
                        except ProcessLookupError:
                            pass
                    result = (
                        503,
                        [(b"content-type", b"text/plain; charset=utf-8")],
                        b"metrics render failed\n",
                    )
                if not fut.done():
                    fut.set_result(result)
        finally:
            self.draining = False

    async def render(self, key):
        entry = self.pending.get(key)
        if entry is None:
            entry = self.pending[key] = [
                asyncio.get_running_loop().create_future(),
                [0],
            ]
        fut, waiters = entry
        waiters[0] += 1
        if not self.draining:
            self.draining = True
            asyncio.ensure_future(self._drain())
        try:
            return await asyncio.shield(fut)
        finally:
            waiters[0] -= 1


def render_key(raw_headers, query_string):
    """The representation stock would produce: (content type, "gzip" or "", name[] query)."""
    from urllib.parse import parse_qs, urlencode

    from prometheus_client import exposition

    def joined(name):  # as prometheus_client's ASGI app: repeated fields comma-joined
        return ",".join(
            v.decode("utf8") for k, v in raw_headers if k.decode("utf8").lower() == name
        )

    _, content_type = exposition.choose_encoder(joined("accept"))
    gz = exposition.gzip_accepted(joined("accept-encoding"))
    names = sorted(set(parse_qs(query_string.decode("utf8")).get("name[]", [])))
    return content_type, "gzip" if gz else "", urlencode([("name[]", n) for n in names])


def make_offloop_metrics_app(renderer: OffloopMetricsRenderer):
    async def app(scope, receive, send):
        assert scope["type"] == "http"
        key = render_key(scope.get("headers") or [], scope.get("query_string") or b"")
        status, out_headers, body = await renderer.render(key)
        await send(
            {"type": "http.response.start", "status": status, "headers": out_headers}
        )
        await send({"type": "http.response.body", "body": body})

    return app


def offloop_add_prometheus_middleware(original, app, *args, **kwargs):
    """AROUND server_utils.add_prometheus_middleware."""
    from sglang_tpu.envs import envs as tpu_envs

    try:
        from sglang.srt.runtime_context import get_serving

        workers = int(get_serving().tokenizer_worker_num or 1)
    except Exception:
        workers = 1
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if workers > 1 or not multiproc_dir:
        return original(app, *args, **kwargs)
    from starlette.routing import Mount

    renderer = OffloopMetricsRenderer(
        multiproc_dir, tpu_envs.SGLANG_TPU_METRICS_RENDER_TIMEOUT_S.get()
    )
    # Lifecycle owner: no helper (re)start once the app shuts down; atexit is the fallback, and the
    # helper exits by itself on pipe EOF or parent death.
    try:
        app.router.on_shutdown.append(renderer.aclose)
    except AttributeError:
        pass
    route = Mount("/metrics", make_offloop_metrics_app(renderer))
    route.path_regex = re.compile("^/metrics(?P<path>.*)$")  # as core: no 307 redirect
    app.routes.append(route)
    logger.info("/metrics renders in a helper process")


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    return [
        (
            "sglang.srt.entrypoints.server_utils.add_prometheus_middleware",
            offloop_add_prometheus_middleware,
            HookType.AROUND,
        )
    ]
