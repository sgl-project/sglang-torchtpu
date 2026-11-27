"""/metrics rendered in a helper process, off the serving event loop (sglang_tpu/hooks/metrics_offloop.py)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

import ast
import asyncio
import gzip
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from sglang_tpu.hooks import metrics_offloop as mo

WRITER = """
import os, sys
os.environ["PROMETHEUS_MULTIPROC_DIR"] = sys.argv[1]
from prometheus_client import Counter, Gauge, Histogram
lab = ["tp_rank"]
i = sys.argv[2]
hs = [Histogram(f"t:h{j}", "h", lab, buckets=[0.001 * 1.5 ** k for k in range(40)]) for j in range(int(sys.argv[3]))]
for h in hs: h.labels(tp_rank=i).observe(0.01)
c = Counter("t:requests", "c", lab); c.labels(tp_rank=i).inc(int(sys.argv[4]))
Gauge("t:g", "g", lab, multiprocess_mode="mostrecent").labels(tp_rank=i).set(1)
"""


def _populate(d, procs=3, hists=25, inc=1):
    for i in range(procs):
        subprocess.run(
            [sys.executable, "-c", WRITER, d, str(i), str(hists), str(inc)],
            check=True,
            stdin=subprocess.DEVNULL,
        )


def _stock(d, accept="", enc="", query=""):
    code = f"""
import os, sys
os.environ["PROMETHEUS_MULTIPROC_DIR"] = {d!r}
from urllib.parse import parse_qs
from prometheus_client import CollectorRegistry, multiprocess
from prometheus_client.exposition import _bake_output
r = CollectorRegistry(); multiprocess.MultiProcessCollector(r, path={d!r})
s, h, b = _bake_output(r, {accept!r}, {enc!r}, parse_qs({query!r}), False)
sys.stdout.buffer.write(b)
"""
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    ).stdout


class TestMetricsOffloop(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        _populate(self.d)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_bytes_match_stock_for_each_representation(self):
        r = mo.OffloopMetricsRenderer(self.d, 10.0)

        async def go():
            out = {}
            for key in [
                ("", "", ""),
                ("", "gzip, deflate", ""),
                ("application/openmetrics-text; version=1.0.0", "", ""),
                ("", "", "name[]=t:requests_total"),
            ]:
                out[key] = await r.render(key)
            await r.aclose()
            return out

        out = self._run(go())
        for (accept, enc, query), (status, headers, body) in out.items():
            self.assertEqual(status, 200)
            stock = _stock(self.d, accept, enc, query)
            if "gzip" in enc:
                self.assertIn((b"Content-Encoding", b"gzip"), headers)
                body, stock = gzip.decompress(body), gzip.decompress(stock)
            self.assertEqual(body, stock, (accept, enc, query))

    def test_fresh_after_increment_and_coalesced(self):
        r = mo.OffloopMetricsRenderer(self.d, 10.0)

        async def go():
            before = (await r.render(("", "", "")))[2]
            _populate(self.d, procs=1, hists=1, inc=5)  # tp_rank 0 counter grows
            n0 = r.renders
            res = await asyncio.gather(*[r.render(("", "", "")) for _ in range(10)])
            renders = r.renders - n0
            await r.aclose()
            return before, res, renders

        before, res, renders = self._run(go())
        self.assertNotEqual(before, res[0][2])
        self.assertTrue(all(x[2] == res[0][2] for x in res))
        self.assertLessEqual(renders, 2)

    def test_loop_keeps_serving_during_a_slow_render(self):
        _populate(self.d, procs=8, hists=60)  # a render of a few hundred ms
        r = mo.OffloopMetricsRenderer(self.d, 30.0)

        async def go():
            await r.render(("", "", ""))  # start the helper
            ticks, stop = [], False

            async def ticker():
                last = time.perf_counter()
                while not stop:
                    await asyncio.sleep(0.01)
                    now = time.perf_counter()
                    ticks.append(now - last)
                    last = now

            t = asyncio.ensure_future(ticker())
            t0 = time.perf_counter()
            for _ in range(5):
                await r.render(("", "gzip", ""))
            render_s = (time.perf_counter() - t0) / 5
            stop = True
            await t
            await r.aclose()
            return max(ticks), render_s

        worst_tick, render_s = self._run(go())
        self.assertGreater(render_s, 0.05)  # the render itself is slow...
        self.assertLess(worst_tick, 0.05)  # ...but the loop never stalled 50 ms

    def test_timeout_answers_503_and_restarts(self):
        r = mo.OffloopMetricsRenderer(self.d, 10.0)

        async def go():
            await r.render(("", "", ""))
            first = r.proc.pid
            r.proc.send_signal(signal.SIGSTOP)  # a hung helper
            r.timeout_s = 0.5
            status = (await r.render(("", "", "")))[0]
            r.timeout_s = 10.0
            again = await r.render(("", "", ""))
            pid2 = r.proc.pid
            await r.aclose()
            return first, status, again[0], pid2

        first, status, status2, pid2 = self._run(go())
        self.assertEqual(status, 503)
        self.assertEqual(status2, 200)
        self.assertNotEqual(first, pid2)

    def test_helper_exits_when_its_parent_dies(self):
        code = f"""
import subprocess, sys, time
p = subprocess.Popen([sys.executable, "-I", {mo._HELPER!r}, {self.d!r}], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
print(p.pid, flush=True); time.sleep(60)
"""
        parent = subprocess.Popen(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )
        helper = int(parent.stdout.readline())
        parent.kill()
        parent.wait()
        for _ in range(50):
            try:
                os.kill(helper, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            os.kill(helper, signal.SIGKILL)
            self.fail("helper outlived its parent")

    def test_helper_imports_only_stdlib_and_prometheus(self):
        tree = ast.parse(open(mo._HELPER).read())
        mods = {
            n.names[0].name.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.Import)
        }
        mods |= {
            n.module.split(".")[0]
            for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom) and n.module
        }
        self.assertTrue(
            mods
            <= {
                "gzip",
                "json",
                "os",
                "sys",
                "threading",
                "time",
                "urllib",
                "prometheus_client",
            },
            mods,
        )

    def test_no_multiproc_dir_and_multi_worker_keep_stock(self):
        calls = []
        original = lambda app, *a, **k: calls.append(app)
        app = type("A", (), {"routes": []})()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            mo.offloop_add_prometheus_middleware(original, app)
        self.assertEqual(len(calls), 1)
        with (
            patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": self.d}),
            patch(
                "sglang.srt.runtime_context.get_serving",
                return_value=type("S", (), {"tokenizer_worker_num": 2})(),
            ),
        ):
            mo.offloop_add_prometheus_middleware(original, app)
        self.assertEqual(len(calls), 2)
        self.assertEqual(app.routes, [])


async def _asgi(app, headers, query=b""):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(m):
        sent.append(m)

    await app(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": headers,
            "query_string": query,
        },
        receive,
        send,
    )
    start = next(m for m in sent if m["type"] == "http.response.start")
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return start["status"], sorted((k.lower(), v) for k, v in start["headers"]), body


class TestMountedAppParity(unittest.TestCase):
    """Complete ASGI responses vs prometheus_client's make_asgi_app (review round 2)."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        _populate(self.d)

    def tearDown(self):
        import shutil

        shutil.rmtree(self.d, ignore_errors=True)

    def test_responses_equal_stock(self):
        from prometheus_client import CollectorRegistry, make_asgi_app, multiprocess

        reg = CollectorRegistry()
        multiprocess.MultiProcessCollector(reg, path=self.d)
        stock = make_asgi_app(reg)
        cases = [
            [],
            [(b"accept-encoding", b"gzip, deflate")],
            [(b"accept-encoding", b"GZIP")],
            [(b"accept-encoding", b"xgzip")],
            [(b"accept", b"application/openmetrics-text; version=1.0.0")],
            [
                (b"accept", b"text/plain"),
                (b"accept", b"application/openmetrics-text; version=1.0.0"),
            ],
            [(b"accept-encoding", b"identity"), (b"accept-encoding", b"gzip")],
        ]

        async def go():
            r = mo.OffloopMetricsRenderer(self.d, 10.0)
            app = mo.make_offloop_metrics_app(r)
            out = []
            for h in cases:
                out.append((h, await _asgi(stock, h), await _asgi(app, h)))
            await r.aclose()
            return out

        for h, (s1, h1, b1), (s2, h2, b2) in asyncio.run(go()):
            self.assertEqual(s1, s2, h)
            self.assertEqual([k for k, _ in h1], [k for k, _ in h2], h)
            self.assertEqual(
                dict(h1).get(b"content-type"), dict(h2).get(b"content-type"), h
            )
            if dict(h1).get(b"content-encoding") == b"gzip":
                b1, b2 = gzip.decompress(b1), gzip.decompress(b2)
            self.assertEqual(b1, b2, h)

    def test_cache_busting_queries_share_one_render(self):
        async def go():
            r = mo.OffloopMetricsRenderer(self.d, 10.0)
            app = mo.make_offloop_metrics_app(r)
            await _asgi(app, [])
            n0 = r.renders
            await asyncio.gather(
                *[_asgi(app, [], f"x={i}".encode()) for i in range(10)]
            )
            n = r.renders - n0
            await r.aclose()
            return n

        self.assertLessEqual(asyncio.run(go()), 2)


class TestRendererLifecycle(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        _populate(self.d, procs=6, hists=60)  # renders of ~100+ ms

    def test_arrivals_mid_render_get_a_later_render(self):
        async def go():
            r = mo.OffloopMetricsRenderer(self.d, 30.0)
            await r.render(("", "", ""))
            first = asyncio.ensure_future(r.render(("", "", "")))
            while not (r.draining and ("", "", "") not in r.pending):
                await asyncio.sleep(0.001)  # render 1 is in flight
            _populate(self.d, procs=1, hists=1, inc=7)
            n0 = r.renders
            late = await asyncio.gather(*[r.render(("", "", "")) for _ in range(10)])
            early = await first
            n = r.renders - n0
            await r.aclose()
            return early, late, n

        early, late, n = asyncio.run(go())
        self.assertTrue(all(x[2] == late[0][2] for x in late))
        self.assertNotEqual(early[2], late[0][2])  # the late callers saw the increment
        self.assertEqual(n, 2)  # render 1 plus exactly one follow-up

    def test_cancelled_caller_does_not_cancel_shared_render(self):
        async def go():
            r = mo.OffloopMetricsRenderer(self.d, 30.0)
            a = asyncio.ensure_future(r.render(("", "", "")))
            b = asyncio.ensure_future(r.render(("", "", "")))
            await asyncio.sleep(0)
            a.cancel()
            res = await b
            await r.aclose()
            return res[0]

        self.assertEqual(asyncio.run(go()), 200)

    def test_render_error_keeps_the_helper(self):
        async def go():
            r = mo.OffloopMetricsRenderer(self.d, 10.0)
            await r.render(("", "", ""))
            pid = r.proc.pid
            bad = os.path.join(self.d, "counter_999999.db")
            open(bad, "wb").write(b"\x00\x00\x00\x10corrupt-garbage-bytes")
            s1 = (await r.render(("", "", "")))[0]
            os.remove(bad)
            s2 = (await r.render(("", "", "")))[0]
            same = r.proc.pid == pid
            await r.aclose()
            return s1, s2, same

        s1, s2, same = asyncio.run(go())
        self.assertIn(
            s1, (200, 500)
        )  # prometheus_client may tolerate it; never a crash
        self.assertEqual(s2, 200)
        self.assertTrue(same)

    def test_no_restart_after_stop_and_restart_limit(self):
        async def go():
            r = mo.OffloopMetricsRenderer(self.d, 10.0)
            await r.render(("", "", ""))
            r.stop()
            after_stop = (await r.render(("", "", "")))[0]
            r2 = mo.OffloopMetricsRenderer(self.d, 10.0)
            statuses = []
            for _ in range(mo._RESTARTS_PER_MINUTE + 2):
                statuses.append((await r2.render(("", "", "")))[0])
                if r2.proc is not None:
                    r2.proc.kill()
                    await r2.proc.wait()
            await r2.aclose()
            return after_stop, statuses

        after_stop, statuses = asyncio.run(go())
        self.assertEqual(after_stop, 503)
        self.assertEqual(
            statuses[: mo._RESTARTS_PER_MINUTE], [200] * mo._RESTARTS_PER_MINUTE
        )
        self.assertEqual(statuses[-1], 503)


if __name__ == "__main__":
    unittest.main()
