# SPDX-License-Identifier: Apache-2.0
"""Standalone renderer process for the server's Prometheus /metrics.

Run by ``sglang_tpu.hooks.metrics_offloop`` as ``python -P <this file> <multiproc_dir>``: it imports only
the standard library and prometheus_client (never torch, sglang or sglang_tpu), so start-up is cheap and
no accelerator runtime is touched. It renders on request, one request at a time, exactly as
prometheus_client's ASGI app does (MultiProcessCollector over the directory, Accept / Accept-Encoding /
``name[]`` negotiation, gzip), but outside the server's event loop.

Protocol on stdin/stdout (binary): a request is one JSON line ``{"accept", "accept_encoding", "query"}``;
the response is one JSON line ``{"status", "headers", "len"}`` followed by ``len`` body bytes. The helper
exits on stdin EOF (the server closed the pipe or died) and, as a backstop, when its parent changes.
"""

import gzip
import json
import os
import sys
import threading
import time
from urllib.parse import parse_qs


def _render(registry, req):
    from prometheus_client import exposition

    params = parse_qs(req.get("query") or "")
    accept = req.get("accept") or ""
    accept_encoding = req.get("accept_encoding") or ""
    bake = getattr(exposition, "_bake_output", None)
    if (
        bake is not None
    ):  # exactly stock's negotiation (prometheus_client's ASGI app calls this)
        status, headers, body = bake(registry, accept, accept_encoding, params, False)
        return int(str(status).split(" ")[0]), [list(h) for h in headers], body
    reg = (
        registry.restricted_registry(params["name[]"])
        if "name[]" in params
        else registry
    )
    encoder, content_type = exposition.choose_encoder(accept)
    body = encoder(reg)
    headers = [["Content-Type", content_type]]
    if exposition.gzip_accepted(accept_encoding):
        body = gzip.compress(body)
        headers.append(["Content-Encoding", "gzip"])
    return 200, headers, body


def main():
    multiproc_dir = sys.argv[1]
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = multiproc_dir
    from prometheus_client import CollectorRegistry, multiprocess

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=multiproc_dir)
    parent = os.getppid()

    def watch_parent():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent:
                os._exit(0)

    threading.Thread(target=watch_parent, daemon=True).start()
    out = sys.stdout.buffer
    sys.stdout = sys.stderr  # stray prints must not corrupt the framing
    for line in sys.stdin.buffer:
        try:
            status, headers, body = _render(registry, json.loads(line))
        except (
            Exception
        ) as e:  # e.g. a torn .db read: log here, answer a static 500, keep serving
            print(
                f"metrics_render_helper: render failed: {e!r}",
                file=sys.stderr,
                flush=True,
            )
            status, headers = 500, [["Content-Type", "text/plain; charset=utf-8"]]
            body = b"metrics render failed\n"
        try:
            out.write(
                json.dumps(
                    {"status": status, "headers": headers, "len": len(body)}
                ).encode()
                + b"\n"
            )
            out.write(body)
            out.flush()
        except (
            BrokenPipeError,
            OSError,
        ):  # the server closed the pipe (timeout, shutdown)
            break


if __name__ == "__main__":
    main()
