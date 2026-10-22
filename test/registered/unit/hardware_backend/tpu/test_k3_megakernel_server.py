# SPDX-License-Identifier: Apache-2.0
"""K3 megakernel standalone server (k3_megakernel/server/app.py): stop strings, request\nvalidation and leader/follower job hand-off, with a scripted engine and a fake KV store."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import json
import queue
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.test.test_utils import CustomTestCase

try:
    from fastapi.testclient import TestClient

    from sglang_tpu.k3_megakernel.server import app as srv
except ImportError:  # pragma: no cover
    srv = None


# app.py reads the process count / index from jax; the CPU CI has no jax, so stand in a
# two-process one (this process is the leader, index 0).
_JAX = SimpleNamespace(process_count=lambda: 2, process_index=lambda: 0)


class _Tok:
    """One character per token id (ord), so text and ids line up."""

    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kw):
        return self.encode("".join(m["content"] for m in messages))


class _Engine:
    """Scripted greedy engine: emits ``script`` in calls of ``k`` tokens, honours stop_check."""

    def __init__(self, script: str, k: int):
        self.script, self.k, self.calls = [ord(c) for c in script], k, 0

    def generate(
        self,
        prompt,
        max_tokens,
        stop_ids=(),
        on_tokens=None,
        stop_check=None,
        prefill_mode="rows",
        steps=0,
    ):
        ids, reason, pos = [], None, 0
        while reason is None:
            new = self.script[pos : pos + self.k][: max_tokens - len(ids)]
            pos += len(new)
            self.calls += 1
            ids += new
            if len(ids) >= max_tokens or pos >= len(self.script):
                reason = "length"
            if stop_check is not None and stop_check(ids, len(new)):
                reason = "stop"
            on_tokens(new, reason is not None, reason)
        return ids, reason, {}


class _KV:
    def __init__(self):
        self.data, self.deleted, self.cv = {}, [], threading.Condition()

    def key_value_set(self, key, value):
        with self.cv:
            self.data[key] = value
            self.cv.notify_all()

    def key_value_delete(self, key):
        self.deleted.append(key)

    def blocking_key_value_get(self, key, timeout_ms):
        with self.cv:
            if not self.cv.wait_for(lambda: key in self.data, timeout_ms / 1e3):
                raise TimeoutError("DEADLINE_EXCEEDED")
            return self.data[key]


@unittest.skipIf(srv is None, "fastapi not installed")
class TestK3MegakernelServer(CustomTestCase):
    def _server(self, script, k=4, bench_hooks=False):
        engine, tok, jobs, kv = _Engine(script, k), _Tok(), queue.Queue(), _KV()
        app = srv.make_app(
            jobs=jobs,
            tok=tok,
            model_name="m",
            max_prompt=64,
            vocab=0x110000,
            steps_allowed=(4, 1),
            bench_hooks=bench_hooks,
        )
        patches = [
            patch.object(srv, "_kv_client", return_value=kv),
            patch.dict(sys.modules, {"jax": _JAX}),
        ]
        for p in patches:
            p.start()
        leader = threading.Thread(
            target=srv.leader_loop, args=(engine, tok, jobs), daemon=True
        )
        leader.start()

        def stop():
            jobs.put(None)
            leader.join(10)
            for p in patches:
                p.stop()

        self.addCleanup(stop)
        return TestClient(app), engine, kv

    def _stream(self, client, body):
        with client.stream(
            "POST", "/v1/completions", json=dict(body, stream=True)
        ) as r:
            events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
        chunks = [json.loads(e) for e in events if e != "[DONE]"]
        return [c["choices"][0]["text"] for c in chunks], chunks[-1]["choices"][0][
            "finish_reason"
        ]

    def test_stop_string_across_calls_is_never_streamed(self):
        # "XY" straddles the first two calls ("abcX" | "Yd.."): the "X" must be held back, then cut.
        client, _, _ = self._server("abcXYdefgh", k=4)
        deltas, reason = self._stream(
            client, dict(prompt="p", max_tokens=10, stop="XY")
        )
        self.assertEqual("".join(deltas), "abc")
        self.assertTrue(all("X" not in d for d in deltas))
        self.assertEqual(reason, "stop")

    def test_rejected_requests_leave_no_gap_for_followers(self):
        client, _, kv = self._server("hello", k=4)
        for bad in (
            dict(prompt="p", stop=7),
            dict(prompt="p", max_tokens=0),
            dict(prompt="p", n=2),
            dict(prompt=[1, -5]),
        ):
            with self.subTest(bad=bad):
                self.assertEqual(
                    client.post("/v1/completions", json=bad).status_code, 400
                )
        out = client.post("/v1/completions", json=dict(prompt="p", max_tokens=5)).json()
        self.assertEqual(out["choices"][0]["text"], "hello")
        # The only published job is number 1: a follower waiting on key 1 runs it.
        self.assertEqual(sorted(kv.data), [srv._KV_PREFIX + "1"])
        self.assertEqual(json.loads(kv.data[srv._KV_PREFIX + "1"])["job"]["n"], 1)

    def test_bench_hooks_need_the_server_flag(self):
        body = dict(prompt="p", max_tokens=5, k3_force_acc=6, k3_stop_ids=[7])
        client, _, _ = self._server("hello", k=4)
        self.assertEqual(client.post("/v1/completions", json=body).status_code, 400)
        client, _, kv = self._server("hello", k=4, bench_hooks=True)
        self.assertEqual(client.post("/v1/completions", json=body).status_code, 200)
        job = json.loads(kv.data[srv._KV_PREFIX + "1"])["job"]
        self.assertEqual((job["force_acc"], job["stop_ids"]), (6, [7]))

    def test_follower_runs_jobs_with_stop_strings_until_shutdown(self):
        kv, engine = _KV(), _Engine("a\nb", k=4)
        spec = dict(
            n=1,
            prompt=[1],
            max_tokens=4,
            ignore_eos=False,
            stop=["\n"],
            prefill_mode="rows",
            steps=0,
        )
        kv.key_value_set(srv._KV_PREFIX + "1", json.dumps({"job": spec}))
        kv.key_value_set(srv._KV_PREFIX + "2", json.dumps({"shutdown": True}))
        with (
            patch.object(srv, "_kv_client", return_value=kv),
            patch.dict(sys.modules, {"jax": _JAX}),
        ):
            done = threading.Thread(
                target=srv.follower_loop, args=(engine, _Tok()), daemon=True
            )
            done.start()
            done.join(10)
        self.assertFalse(done.is_alive())
        self.assertEqual(engine.calls, 1)


if __name__ == "__main__":
    unittest.main()
