#!/usr/bin/env python3
"""Stdlib-only native Torch HTTP workload; NOT the JAX completion-cadence metric.

prepare --data frozen.npy --manifest frozen.manifest.json --output requests.json
run --requests requests.json --base-url http://HOST:30000 --output result.json

Strict run refuses this checkout's unsupported admission/DP-cache proof. For an
explicitly UNMATCHED HTTP diagnostic, add --allow-unmatched-admission and
--exclusive-server (operator assertion: no other traffic). Never launches a model
server. `selftest` uses only a local HTTP boundary fake, not inference.
"""

import argparse
import ast
import hashlib
import io
import json
import struct
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

DATA_SHA256 = "9a393227fb7104feef12440dba1d104dc1c2c6a5a54fc956d9225000b1e25c53"
PREFIX, EXTEND, OUTPUT, CONCURRENCY = 49152, 12288, 16, 32
UNSUPPORTED = [
    "server_info.internal_states has no live running/waiting counts or waiting RID inventory",
    "TokenizerManager waits on is_pause before dispatch: pause/queue/resume is not a scheduler admission barrier",
    "ordered native batches preserve frontend submission order, not verified attention-DP cache assignment",
]


class ProtocolError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_npy(raw, expected_shape=(64, 61440)):
    """NPY v1/v2 C-order little-endian signed integer arrays, without NumPy."""
    stream = io.BytesIO(raw)
    if stream.read(6) != b"\x93NUMPY":
        raise ProtocolError("not an NPY file")
    version = stream.read(2)
    fmt = {b"\x01\x00": "<H", b"\x02\x00": "<I"}.get(version)
    if fmt is None:
        raise ProtocolError("only NPY versions 1.0 and 2.0 are supported")
    size = struct.calcsize(fmt)
    length_bytes = stream.read(size)
    if len(length_bytes) != size:
        raise ProtocolError("truncated NPY header length")
    header_len = struct.unpack(fmt, length_bytes)[0]
    if header_len > 1024 * 1024:
        raise ProtocolError("NPY header too large")
    try:
        header_raw = stream.read(header_len)
        if len(header_raw) != header_len:
            raise ProtocolError("truncated NPY header")
        header = ast.literal_eval(header_raw.decode("latin1"))
    except (SyntaxError, UnicodeError, ValueError) as exc:
        raise ProtocolError("invalid NPY header") from exc
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        raise ProtocolError("invalid NPY header fields")
    if header["fortran_order"] is not False or header["shape"] != expected_shape:
        raise ProtocolError("expected C-order shape %r" % (expected_shape,))
    code = {"<i4": "i", "<i8": "q"}.get(header["descr"])
    if code is None:
        raise ProtocolError("expected little-endian int32 or int64")
    width = struct.calcsize("<" + code)
    rows, columns = expected_shape
    payload = stream.read()
    if len(payload) != rows * columns * width:
        raise ProtocolError("NPY payload size mismatch (including trailing bytes)")
    selected = []
    for row in range(min(CONCURRENCY, rows)):
        tokens = list(struct.unpack_from("<" + str(columns) + code, payload, row * columns * width))
        if any(token < 0 for token in tokens):
            raise ProtocolError("negative token ID")
        selected.append(tokens)
    return selected


def load_dataset(path):
    raw = Path(path).read_bytes()
    if digest(raw) != DATA_SHA256:
        raise ProtocolError("frozen NPY SHA256 mismatch")
    return read_npy(raw)


def token_digest(tokens):
    # Portable canonical ID hash, independent of original NPY integer width.
    return digest(struct.pack("<" + str(len(tokens)) + "q", *tokens))


def make_requests(rows):
    namespace = str(uuid.uuid4())
    records = []
    for index, row in enumerate(rows):
        # 0 and 1 are valid in the model vocabulary; branch must differ from E[0].
        branch = 0 if row[PREFIX] != 0 else 1
        records.append({"row": index, "warm_rid": str(uuid.uuid4()),
                        "measured_rid": str(uuid.uuid4()), "branch_token": branch,
                        "input_sha256_i64le": token_digest(row)})
    return {"format": "torch-native-cache-hit-v1", "namespace": namespace,
            "data_sha256": DATA_SHA256, "prefix_len": PREFIX,
            "extend_len": EXTEND, "output_len": OUTPUT, "concurrency": CONCURRENCY,
            "requests": records}


def validate_requests(manifest, rows):
    if manifest.get("format") != "torch-native-cache-hit-v1":
        raise ProtocolError("unknown request manifest format")
    for key, value in (("data_sha256", DATA_SHA256), ("prefix_len", PREFIX),
                       ("extend_len", EXTEND), ("output_len", OUTPUT), ("concurrency", CONCURRENCY)):
        if manifest.get(key) != value:
            raise ProtocolError("request manifest contract mismatch: " + key)
    records = manifest.get("requests", [])
    if len(records) != CONCURRENCY or len(rows) != CONCURRENCY:
        raise ProtocolError("expected C32")
    if len({token_digest(row[:PREFIX]) for row in rows}) != CONCURRENCY:
        raise ProtocolError("expected 32 unique prefixes")
    ids = set()
    for index, (record, row) in enumerate(zip(records, rows)):
        if record.get("row") != index or record.get("input_sha256_i64le") != token_digest(row):
            raise ProtocolError("rows must be 0..31 in stored order with unchanged IDs")
        branch = record.get("branch_token")
        if type(branch) is not int or branch not in (0, 1) or branch == row[PREFIX]:
            raise ProtocolError("warm branch must differ from measured extension")
        for key in ("warm_rid", "measured_rid"):
            rid = record.get(key)
            try:
                if str(uuid.UUID(rid)) != rid or rid in ids:
                    raise ValueError()
            except (ValueError, TypeError, AttributeError) as exc:
                raise ProtocolError("request IDs must be distinct explicit UUIDs") from exc
            ids.add(rid)


def payload_for(rows, manifest, phase):
    warm = phase == "warm"
    if phase not in ("warm", "measured"):
        raise ProtocolError("unknown phase")
    records = manifest["requests"]
    inputs = [row[:PREFIX] + [record["branch_token"]] if warm else row
              for row, record in zip(rows, records)]
    return {"input_ids": inputs,
            "rid": [record["warm_rid" if warm else "measured_rid"] for record in records],
            "sampling_params": {"temperature": 0, "ignore_eos": True,
                                "max_new_tokens": 1 if warm else OUTPUT}, "stream": False}


class NativeHTTPClient:
    def __init__(self, base_url, timeout=3600, api_key=None):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.query or parsed.fragment:
            raise ProtocolError("expected HTTP(S) base URL without query or fragment")
        self.base_url, self.timeout, self.api_key = base_url.rstrip("/"), timeout, api_key

    def request(self, route, payload=None, raw=False):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(self.base_url + route, headers=headers,
                                         data=None if payload is None else json.dumps(payload).encode())
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                content = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ProtocolError("HTTP request failed at " + route + ": " + str(exc)) from exc
        if raw:
            return content.decode()
        try:
            return json.loads(content)
        except (ValueError, UnicodeError) as exc:
            raise ProtocolError("invalid HTTP JSON at " + route) from exc

    def generate(self, payload, phase):
        start = time.perf_counter_ns()
        response = self.request("/generate", payload)
        end = time.perf_counter_ns()
        # Retain rejected wire evidence too (e.g. a cold miss), not just successes.
        self.last_exchange = {"phase": phase, "http_start_ns": start, "http_end_ns": end,
                              "http_batch_e2e_s": (end - start) / 1e9,
                              "ttft_s": None, "ttft_status": "not_observed_nonstreaming",
                              "responses": response}
        validate_response(response, payload, phase)
        return self.last_exchange


def validate_response(response, payload, phase):
    if not isinstance(response, list) or len(response) != len(payload["rid"]):
        raise ProtocolError("native batch response count mismatch")
    expected = dict(zip(payload["rid"], payload["input_ids"]))
    seen = set()
    for item in response:
        if not isinstance(item, dict) or not isinstance(item.get("meta_info"), dict):
            raise ProtocolError("missing native meta_info")
        meta = item["meta_info"]
        rid = meta.get("id")
        if not isinstance(rid, str) or rid not in expected or rid in seen:
            raise ProtocolError("unknown or duplicate response RID")
        seen.add(rid)
        for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
            if type(meta.get(key)) is not int or meta[key] < 0:
                raise ProtocolError("missing/invalid native counter " + key)
        count = payload["sampling_params"]["max_new_tokens"]
        if meta["prompt_tokens"] != len(expected[rid]) or meta["completion_tokens"] != count:
            raise ProtocolError("prompt/output token count mismatch")
        if meta["cached_tokens"] > len(expected[rid]):
            raise ProtocolError("impossible cache counter")
        finish = meta.get("finish_reason")
        if not isinstance(finish, dict) or finish.get("type") != "length":
            raise ProtocolError("request did not finish at requested output length")
        output = item.get("output_ids")
        if not isinstance(output, list) or len(output) != count or any(type(x) is not int or x < 0 for x in output):
            raise ProtocolError("missing/invalid output token IDs")
        if phase == "measured" and not PREFIX - 64 <= meta["cached_tokens"] <= PREFIX:
            raise ProtocolError("measured cold miss or extension branch was already cached")


def save_new(path, value):
    # Never silently overwrite earlier evidence.
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run(args):
    manifest = json.loads(Path(args.requests).read_text())
    source = Path(manifest["source_manifest_path"]).read_bytes()
    if (digest(source) != manifest["source_manifest_sha256"]
            or source.decode() != manifest["source_manifest_utf8"]):
        raise ProtocolError("adjacent frozen source manifest changed")
    rows = load_dataset(manifest["data_path"])
    validate_requests(manifest, rows)
    client = NativeHTTPClient(args.base_url, args.timeout)
    report = {"metric_family": "HTTPmetrics_NOT_JAXcadence", "matched_jax_cadence": False,
              "steady_chunk_extend_ms": None, "dp_cache_assignment": "unsupportedDPmatch",
              "round_robin": "roundrobinunverified", "unsupported_reasons": UNSUPPORTED,
              "request_manifest": manifest, "status": "unsupported_matched_admission"}
    try:
        report["server_info_before"] = client.request("/get_server_info")
        # No heuristic key probing: audited Torch internal states do not expose
        # the exact waiting RID inventory and rank mapping the matched gate needs.
        if not args.allow_unmatched_admission:
            raise ProtocolError("matched admission unsupported; use explicit --allow-unmatched-admission for HTTP-only diagnosis")
        if not args.exclusive_server:
            raise ProtocolError("unmatched diagnosis requires --exclusive-server operator assertion")
        report["exclusive_server"] = "operator_asserted_not_observed"
        report["flush_response"] = client.request("/flush_cache", {}, raw=True)
        if not report["flush_response"].startswith("Cache flushed."):
            raise ProtocolError("unrecognized successful flush response")
        report["warm"] = client.generate(payload_for(rows, manifest, "warm"), "warm")
        report["measured"] = client.generate(payload_for(rows, manifest, "measured"), "measured")
        report["server_info_after"] = client.request("/get_server_info")
        report["status"] = "http_diagnostic_validated_UNMATCHED"
        code = 0
    except ProtocolError as exc:
        report["error"] = str(exc)
        if hasattr(client, "last_exchange"):
            report["last_http_exchange"] = client.last_exchange
        if args.allow_unmatched_admission:
            report["status"] = "failed_http_diagnostic"
        code = 2
    save_new(args.output, report)
    return code


def selftest():
    """Manual stdlib self-test: real urllib client against an HTTP boundary fake."""
    import http.server
    import threading

    def must_reject(function):
        try:
            function()
        except ProtocolError:
            return
        raise AssertionError("invalid evidence was accepted")

    for version, dtype in ((1, "<i4"), (2, "<i8")):
        header = repr({"descr": dtype, "fortran_order": False, "shape": (2, 3)}).encode() + b"\n"
        raw = b"\x93NUMPY" + bytes([version, 0]) + struct.pack("<H" if version == 1 else "<I", len(header)) + header
        raw += struct.pack("<6" + ("i" if dtype == "<i4" else "q"), 1, 2, 3, 4, 5, 6)
        assert read_npy(raw, (2, 3)) == [[1, 2, 3], [4, 5, 6]]
        must_reject(lambda: read_npy(raw[:-1], (2, 3)))
        must_reject(lambda: read_npy(raw + b"x", (2, 3)))
        must_reject(lambda: read_npy(raw.replace(dtype.encode(), b">i4"), (2, 3)))
    rows = [[index + 1] * (PREFIX + EXTEND) for index in range(CONCURRENCY)]
    manifest = make_requests(rows)
    validate_requests(manifest, rows)
    swapped = dict(manifest, requests=list(reversed(manifest["requests"])))
    must_reject(lambda: validate_requests(swapped, rows))
    state = {"mode": "valid", "seen": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *unused):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["seen"].append(body)
            count = body["sampling_params"]["max_new_tokens"]
            result = [{"output_ids": [7] * count, "meta_info": {
                "id": rid, "prompt_tokens": len(tokens), "completion_tokens": count,
                "cached_tokens": 0 if state["mode"] == "cold" else PREFIX,
                "finish_reason": {"type": "length", "length": count}}}
                for rid, tokens in zip(body["rid"], body["input_ids"])]
            if state["mode"] == "duplicate":
                result[1] = result[0]
            if state["mode"] == "short":
                result[0]["output_ids"] = []
            raw = json.dumps(result).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = NativeHTTPClient("http://127.0.0.1:%d" % server.server_port, 10)
        warm = payload_for(rows, manifest, "warm")
        measured = payload_for(rows, manifest, "measured")
        assert warm["input_ids"][0][-1] != measured["input_ids"][0][PREFIX]
        client.generate(warm, "warm")
        result = client.generate(measured, "measured")
        assert result["http_batch_e2e_s"] > 0
        assert state["seen"][1]["rid"] == [record["measured_rid"] for record in manifest["requests"]]
        for mode in ("cold", "duplicate", "short"):
            state["mode"] = mode
            must_reject(lambda: client.generate(measured, "measured"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    print("PASS: stdlib NPY v1/v2 + ordered manifests + real HTTP client; cold/duplicate/short responses rejected")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data", required=True)
    prepare.add_argument("--manifest", required=True, help="adjacent frozen source manifest, preserved verbatim")
    prepare.add_argument("--output", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--requests", required=True)
    execute.add_argument("--base-url", required=True)
    execute.add_argument("--output", required=True)
    execute.add_argument("--timeout", type=float, default=3600)
    execute.add_argument("--allow-unmatched-admission", action="store_true")
    execute.add_argument("--exclusive-server", action="store_true")
    commands.add_parser("selftest")
    args = parser.parse_args()
    try:
        if args.command == "selftest":
            return selftest()
        if args.command == "run":
            return run(args)
        rows = load_dataset(args.data)
        source = Path(args.manifest).read_bytes()
        json.loads(source)  # Require actual JSON, preserve source bytes/hash too.
        manifest = make_requests(rows)
        manifest.update(data_path=str(Path(args.data).resolve()),
                        source_manifest_path=str(Path(args.manifest).resolve()),
                        source_manifest_sha256=digest(source), source_manifest_utf8=source.decode())
        validate_requests(manifest, rows)
        save_new(args.output, manifest)
        return 0
    except (ProtocolError, OSError, ValueError, KeyError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
