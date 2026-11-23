"""Session-affinity front end for a DeepSeek-V4.1 server with attention DP (stdlib asyncio).

Under attention DP each rank has its own scheduler and radix cache, and the DP
dispatcher has no cache-aware policy, so an agentic session's turns would scatter
over the ranks' caches. This proxy pins every conversation to one DP rank and sends
its subagents to their parent's rank by adding ``X-Data-Parallel-Rank``, which the
OpenAI server honours. The conversation comes from the Dynamo session headers
AIPerf sends with ``AIPERF_HTTP_X_DYNAMO_SESSION_ID_FROM_CORRELATION_ID=true``:
``X-Dynamo-Session-ID`` per conversation and ``X-Dynamo-Parent-Session-ID`` for a
subagent (``X-Correlation-ID`` is accepted too). Nothing else in the request changes.

A new conversation goes to the rank with the fewest in-flight requests, then the
fewest conversations active in the last --active-s seconds, then round robin.
Requests without a session header pass through unpinned.

One request per connection: the request (Content-Length or chunked body) is
forwarded with ``Connection: close`` and the response is piped back until the
upstream closes. --log writes one JSON line per request; --summary writes the
per-rank counts on SIGTERM / SIGINT. See README.md.
"""

import argparse
import asyncio
import json
import signal
import time

HOP = {
    b"connection",
    b"keep-alive",
    b"proxy-connection",
    b"te",
    b"trailer",
    b"upgrade",
    b"x-data-parallel-rank",
}


class Router:
    def __init__(self, dp, active_s):
        self.dp = dp
        self.active_s = active_s
        self.rank_of = {}  # session id -> rank (roots and children)
        self.root_of = {}  # session id -> root session id
        self.last_seen = {}  # root -> unix time of its latest request start
        self.inflight = [0] * dp
        self.requests = [0] * dp
        self.roots = [0] * dp
        self.children = [0] * dp
        self.rr = 0
        self.unpinned = 0

    def active_roots(self, now):
        n = [0] * self.dp
        for root, t in self.last_seen.items():
            if now - t <= self.active_s:
                n[self.rank_of[root]] += 1
        return n

    def pick(self, sid, parent):
        if sid in self.rank_of:
            return self.rank_of[sid]
        if parent is not None and parent in self.rank_of:
            r = self.rank_of[parent]
            self.rank_of[sid] = r
            self.root_of[sid] = self.root_of.get(parent, parent)
            self.children[r] += 1
            return r
        act = self.active_roots(time.time())
        order = [(self.rr + i) % self.dp for i in range(self.dp)]
        r = min(order, key=lambda k: (self.inflight[k], act[k]))
        self.rr = (r + 1) % self.dp
        self.rank_of[sid] = r
        self.root_of[sid] = sid
        self.roots[r] += 1
        return r

    def summary(self):
        return {
            "dp": self.dp,
            "requests": self.requests,
            "roots": self.roots,
            "children": self.children,
            "unpinned": self.unpinned,
            "sessions": len(self.rank_of),
        }


async def read_chunked(reader):
    body = []
    while True:
        size = int((await reader.readuntil(b"\r\n")).split(b";")[0].strip(), 16)
        if size == 0:
            while (await reader.readuntil(b"\r\n")) != b"\r\n":  # trailers
                pass
            return b"".join(body)
        body.append(await reader.readexactly(size))
        await reader.readexactly(2)


async def read_head(reader):
    head = await reader.readuntil(b"\r\n\r\n")
    lines = head[:-4].split(b"\r\n")
    hdrs = []
    for ln in lines[1:]:
        k, _, v = ln.partition(b":")
        hdrs.append((k.strip(), v.strip()))
    return lines[0], hdrs


class Proxy:
    def __init__(self, a):
        self.a = a
        self.router = Router(a.dp, a.active_s)
        self.log = open(a.log, "a", buffering=1) if a.log else None

    async def handle(self, creader, cwriter):
        rank = None
        t0 = time.time()
        status = b""
        nbytes = 0
        sid = parent = None
        path = ""
        try:
            try:
                line, hdrs = await asyncio.wait_for(read_head(creader), self.a.idle_s)
            except (
                asyncio.IncompleteReadError,
                asyncio.TimeoutError,
                ConnectionError,
                asyncio.LimitOverrunError,
            ):
                return
            low = {k.lower(): v for k, v in hdrs}
            path = (
                line.split(b" ")[1][:120].decode(errors="replace")
                if line.count(b" ") >= 2
                else ""
            )
            if b"chunked" in low.get(b"transfer-encoding", b"").lower():
                body = await read_chunked(creader)  # re-sent with Content-Length
                hdrs = [
                    (k, v)
                    for k, v in hdrs
                    if k.lower() not in (b"transfer-encoding", b"content-length")
                ]
                hdrs.append((b"Content-Length", str(len(body)).encode()))
            else:
                n = int(low.get(b"content-length", b"0") or 0)
                body = await creader.readexactly(n) if n else b""
            sid = (
                low.get(b"x-dynamo-session-id") or low.get(b"x-correlation-id") or b""
            ).decode() or None
            parent = (low.get(b"x-dynamo-parent-session-id") or b"").decode() or None
            if sid is not None:
                rank = self.router.pick(sid, parent)
            else:
                self.router.unpinned += 1
            out = [line]
            for k, v in hdrs:
                if k.lower() in HOP:
                    continue
                out.append(k + b": " + v)
            if rank is not None:
                out.append(b"X-Data-Parallel-Rank: " + str(rank).encode())
            out.append(b"Connection: close")
            ureader, uwriter = await asyncio.open_connection(
                self.a.upstream_host, self.a.upstream_port, limit=1 << 22
            )
            if rank is not None:
                self.router.inflight[rank] += 1
                self.router.requests[rank] += 1
                self.router.last_seen[self.router.root_of[sid]] = t0
            try:
                uwriter.write(b"\r\n".join(out) + b"\r\n\r\n" + body)
                await uwriter.drain()
                first = True
                while True:
                    chunk = await ureader.read(1 << 16)
                    if not chunk:
                        break
                    if first:
                        status = chunk.split(b"\r\n", 1)[0][9:12]
                        first = False
                    nbytes += len(chunk)
                    cwriter.write(chunk)
                    await cwriter.drain()
            finally:
                if rank is not None:
                    self.router.inflight[rank] -= 1
                uwriter.close()
        except (ConnectionError, asyncio.IncompleteReadError) as e:
            status = status or ("err " + type(e).__name__).encode()
        finally:
            try:
                cwriter.close()
            except Exception:  # noqa: BLE001 - client already gone
                pass
            if self.log is not None and (sid is not None or status):
                self.log.write(
                    json.dumps(
                        {
                            "t": round(t0, 3),
                            "dt": round(time.time() - t0, 3),
                            "path": path,
                            "sid": sid,
                            "parent": parent,
                            "rank": rank,
                            "status": status.decode(errors="replace"),
                            "bytes": nbytes,
                        }
                    )
                    + "\n"
                )

    def dump(self):
        if self.a.summary:
            with open(self.a.summary, "w") as f:
                json.dump(self.router.summary(), f, indent=1)


async def main(a):
    p = Proxy(a)
    srv = await asyncio.start_server(
        p.handle, a.host, a.port, limit=1 << 24, backlog=1024
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for s in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(s, stop.set)
    print(
        f"affinity proxy :{a.port} -> {a.upstream_host}:{a.upstream_port} dp={a.dp}",
        flush=True,
    )
    async with srv:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), 60)
            except asyncio.TimeoutError:
                p.dump()
    p.dump()
    print("affinity proxy summary", json.dumps(p.router.summary()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--upstream-host", default="127.0.0.1")
    ap.add_argument("--upstream-port", type=int, default=30000)
    ap.add_argument("--dp", type=int, default=8)
    ap.add_argument("--active-s", type=float, default=300.0)
    ap.add_argument(
        "--idle-s",
        type=float,
        default=900.0,
        help="close a client connection idle this long",
    )
    ap.add_argument("--log", default="")
    ap.add_argument("--summary", default="")
    asyncio.run(main(ap.parse_args()))
