# SPDX-License-Identifier: Apache-2.0
"""Child process that writes this rank's DeepSeek-V4.1 Engram rows straight into
the shared host tables (``engram_row_reader.start_row_child``).

Run as ``python -I engram_row_child.py SPEC_JSON``. It imports only Run:AI,
NumPy and torch: never ``sglang``/``sglang_tpu`` or a TPU runtime. Run:AI 0.17's
GCS backend loses responses when two streamers run at once in one process, so
these reads live in their own process, next to the rank's main stream.

The parent's tables are memfds (``_HostTable``, replication per_node). The child
opens each one through ``/proc/<parent>/fd/<fd>`` and ``pwrite``s the checkpoint
bytes of rows ``[lo, hi)`` at the parent-given offset: no staging copy, so the
tables stay one resident copy per host. On success it writes ``done.json`` with
the bytes written per table; on failure ``error`` with the traceback. It dies
with its parent (PR_SET_PDEATHSIG) and stops at the spec's deadline.
"""

import ctypes
import json
import os
import shutil
import signal
import sys
import time
import traceback

PR_SET_PDEATHSIG = 1


def _die_with_parent(parent_pid: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG)")
    # The parent may have died before prctl took effect.
    if os.getppid() != parent_pid:
        sys.exit(3)


def _read_header(path: str) -> tuple[int, dict]:
    """(header length, header) of a local or object-storage safetensors shard."""
    if "://" not in path:
        with open(path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            return n, json.loads(f.read(n))
    from runai_model_streamer.file_streamer import FileChunks, FileStreamer

    def read(offset: int, size: int) -> bytes:
        out = bytearray()
        with FileStreamer() as fs:
            fs.stream_files([FileChunks(0, path, [offset], [size])], memory_limit=size)
            for _, _, chunk in fs.get_chunks():
                out += chunk.numpy().tobytes()
        return bytes(out)

    n = int.from_bytes(read(0, 8), "little")
    return n, json.loads(read(8, n))


def _pwrite_all(fd: int, data: memoryview, offset: int) -> int:
    written = 0
    while data:
        n = os.pwrite(fd, data, offset)
        data, offset, written = data[n:], offset + n, written + n
    return written


def _copy_local(plan: dict, path: str, src: int, deadline: float) -> None:
    """Local or mounted shard: plain preads, no Run:AI."""
    with open(path, "rb") as f:
        for start in plan["starts"]:
            if time.monotonic() > deadline:
                raise TimeoutError("Engram rows not read within the deadline")
            size = min(plan["chunk_bytes"], plan["nbytes"] - start)
            data = os.pread(f.fileno(), size, src + start)
            if len(data) != size:
                raise EOFError(f"{path}: short read at {src + start}")
            plan["written"] += _pwrite_all(
                plan["fd"], memoryview(data), plan["dest"] + start
            )
    plan["seconds"] = time.monotonic() - plan["t0"]


def main(spec_path: str) -> None:
    spec = json.load(open(spec_path))
    out_dir = os.path.dirname(spec_path)
    _die_with_parent(spec["parent_pid"])
    # Parent death (PDEATHSIG) or a kill: drop the status dir; the tables die with the parent.
    signal.signal(
        signal.SIGTERM,
        lambda *_: (shutil.rmtree(out_dir, ignore_errors=True), os._exit(4)),
    )
    deadline = time.monotonic() + spec["deadline_s"]
    chunk_bytes, in_flight = spec["chunk_bytes"], spec["in_flight_chunks"]

    headers = {}
    for t in spec["tables"]:
        if t["path"] not in headers:
            headers[t["path"]] = _read_header(t["path"])

    if any("://" in t["path"] for t in spec["tables"]):
        from runai_model_streamer.file_streamer import FileChunks, FileStreamer

    requests, plans = [], []
    for i, t in enumerate(spec["tables"]):
        header_len, header = headers[t["path"]]
        meta = header[t["name"]]
        num_rows = meta["shape"][0]
        start, end = meta["data_offsets"]
        row_bytes = (end - start) // num_rows
        if row_bytes != t["dest_row_bytes"] or t["hi"] > num_rows:
            raise ValueError(
                f"{t['name']}: checkpoint rows are {row_bytes} B x {num_rows}, the "
                f"table expects {t['dest_row_bytes']} B and rows [{t['lo']}, {t['hi']})"
            )
        nbytes = (t["hi"] - t["lo"]) * row_bytes
        src = 8 + header_len + start + t["lo"] * row_bytes
        starts = list(range(0, nbytes, chunk_bytes))
        sizes = [min(chunk_bytes, nbytes - s) for s in starts]
        fd = os.open(f"/proc/{spec['parent_pid']}/fd/{t['fd']}", os.O_RDWR)
        plan = {
            "fd": fd,
            "dest": t["dest_offset"],
            "starts": starts,
            "chunk_bytes": chunk_bytes,
            "nbytes": nbytes,
            "written": 0,
            "t0": time.monotonic(),
            "seconds": None,
        }
        plans.append(plan)
        if "://" not in t["path"]:
            _copy_local(plan, t["path"], src, deadline)
        elif sizes:
            requests.append(FileChunks(i, t["path"], [src + s for s in starts], sizes))

    if requests:
        biggest = max(r.max_chunk_size() for r in requests)
        total = sum(len(r.sizes) for r in requests)
        with FileStreamer() as fs:
            fs.stream_files(requests, memory_limit=biggest * min(total, in_flight))
            for index, chunk_index, chunk in fs.get_chunks():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Engram rows not read within {spec['deadline_s']} s"
                    )
                p = plans[index]
                data = memoryview(chunk.numpy()).cast("B")
                p["written"] += _pwrite_all(
                    p["fd"], data, p["dest"] + p["starts"][chunk_index]
                )
                if p["written"] == p["nbytes"]:
                    p["seconds"] = time.monotonic() - p["t0"]
    for p in plans:
        os.close(p["fd"])
    done = [
        {
            "name": t["name"],
            "written": p["written"],
            "nbytes": p["nbytes"],
            "seconds": p["seconds"] if p["seconds"] is not None else 0.0,
        }
        for t, p in zip(spec["tables"], plans)
    ]
    tmp = os.path.join(out_dir, "done.json.tmp")
    with open(tmp, "w") as f:
        json.dump(done, f)
    os.replace(tmp, os.path.join(out_dir, "done.json"))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except BaseException:
        err = traceback.format_exc()
        try:
            with open(os.path.join(os.path.dirname(sys.argv[1]), "error"), "w") as f:
                f.write(err)
        finally:
            sys.stderr.write(err)
        sys.exit(1)
