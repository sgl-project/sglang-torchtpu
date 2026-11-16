"""On-device probe: can a host-only store in a foreign block layout hold and
return another engine's host pages?

The KV offload to decode DRAM moves whole host blocks between two raiden
managers with the same block layout. A tensor-parallel decode that keeps the
branches of a pipeline-parallel prefill has no pool in a prefill stage's layout,
so it would build one manager per source stage over placeholder device tensors
shaped like one page of that stage, and use only its host blocks. This probe
checks that such a manager works, on one host, with no engine:

    source: a real device buffer with a known pattern, plus host blocks
    store:  placeholder device tensors of the same per-page shape, plus host blocks

    source device -> source host -> store host -> source host -> source device

and compares the bytes that come back. Each variant runs in its own process,
since a native failure can abort the interpreter. Requires a TPU, torch_tpu and
tpu_sync. NOT a CI unit test. Run with:

    python3 test/manual/tpu/probe_stage_layout_host_store.py
"""

import itertools
import json
import os
import subprocess
import sys
import time

PAGE_SIZE = 64
PORTS = itertools.count(21000)
# (name, layers, device of the placeholders, placeholder pages, same layout)
VARIANTS = (
    ("tpu-placeholder-1-page", 1, "tpu:1", 1, True),
    ("tpu-placeholder-3-layers", 3, "tpu:1", 1, True),
    ("tpu-placeholder-same-device", 3, "tpu:0", 1, True),
    ("cpu-placeholder", 1, "cpu", 1, True),
    ("other-layout-must-fail", 1, "tpu:1", 1, False),
)


class _BufferPool:
    """The device-pool surface RaidenKVTransfer needs."""

    def __init__(self, buffers):
        self._buffers = buffers
        self.layer_num = len(buffers)
        self.start_layer = 0
        self.page_size = PAGE_SIZE

    def page_buffers(self):
        return self._buffers


def _wait(poll, what, timeout_s=60.0):
    start = time.monotonic()
    while True:
        status = poll()
        if status:
            return status
        if time.monotonic() - start > timeout_s:
            raise TimeoutError(f"{what} did not settle in {timeout_s:.0f}s")
        time.sleep(0.002)


def _host_read(src, dst, name, uuid, src_pages, dst_pages):
    """Read ``src``'s host pages into ``dst``'s host pages; both ends must settle."""
    src.register_read(name, uuid, src_pages)
    dst.start_read(
        name, uuid, src.host_control_address("127.0.0.1"), src_pages, dst_pages
    )
    got = _wait(lambda: dst.poll_transfer(name), f"{name} destination")
    sent = _wait(lambda: src.poll_transfer(name), f"{name} source")
    src.forget_transfer(name)
    dst.forget_transfer(name)
    return sent, got


def run_variant(layers, placeholder_device, placeholder_pages, same_layout):
    import torch

    from sglang_tpu import raiden_transfer
    from sglang_tpu.raiden_transfer import RaidenKVTransfer

    raiden_transfer.host_control_port_candidates = lambda: iter([next(PORTS)])
    page_shape = (PAGE_SIZE, 2, 2, 128)
    values = {}
    buffers = []
    for layer in range(layers):
        pattern = torch.zeros(8, *page_shape, dtype=torch.bfloat16)
        for page in (1, 2):
            values[(layer, page)] = float(100 * (layer + 1) + page)
            pattern[page] = values[(layer, page)]
        buffers.append(pattern.to("tpu:0"))
    source = RaidenKVTransfer(_BufferPool(buffers), num_host_pages=8)

    store_shape = page_shape if same_layout else (PAGE_SIZE, 2, 1, 128)
    placeholders = [
        torch.zeros(placeholder_pages, *store_shape, dtype=torch.bfloat16).to(
            placeholder_device
        )
        for _ in range(layers)
    ]
    store = RaidenKVTransfer(_BufferPool(placeholders), num_host_pages=8)

    def done(event):
        return lambda: event.query()

    _wait(done(source.backup([1, 2], [0, 1])), "D2H")
    out = _host_read(source, store, "push", 11, [0, 1], [3, 4])
    back = _host_read(store, source, "recall", 12, [3, 4], [5, 6])
    _wait(done(source.load([5, 6], [4, 5])), "H2D")
    torch.accelerator.synchronize()
    result = {"push": out, "recall": back, "pages": {}}
    ok = out == ("sent", "received") and back == ("sent", "received")
    for layer in range(layers):
        for src_page, dst_page in ((1, 4), (2, 5)):
            got = buffers[layer][dst_page].float().cpu()
            want = values[(layer, src_page)]
            same = bool((got == want).all())
            result["pages"][f"l{layer}p{dst_page}"] = [
                want,
                float(got.flatten()[0]),
                same,
            ]
            ok = ok and same
    result["ok"] = ok
    return result


def main():
    if len(sys.argv) > 1:
        _, layers, device, pages, same = VARIANTS[int(sys.argv[1])]
        print("PROBE_RESULT " + json.dumps(run_variant(layers, device, pages, same)))
        return
    summary = {}
    for index, variant in enumerate(VARIANTS):
        env = dict(os.environ, SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM="1")
        try:
            proc = subprocess.run(
                [sys.executable, __file__, str(index)],
                env=env,
                capture_output=True,
                text=True,
                timeout=600,
            )
            lines = [
                l for l in proc.stdout.splitlines() if l.startswith("PROBE_RESULT ")
            ]
            outcome = json.loads(lines[-1][13:]) if lines else None
            tail = (proc.stdout + proc.stderr)[-3000:]
            code = proc.returncode
        except subprocess.TimeoutExpired as exc:
            outcome, code, tail = None, "timeout", str(exc.stderr)[-3000:]
        summary[variant[0]] = {"exit": code, "result": outcome}
        print(f"=== {variant[0]}: exit={code} result={outcome}")
        if not outcome or not outcome.get("ok"):
            print(tail)
    print("PROBE_SUMMARY " + json.dumps(summary))
    expected_pass = [v[0] for v in VARIANTS if v[4] and v[2] != "cpu"]
    passed = all((summary[n]["result"] or {}).get("ok") for n in expected_pass)
    print("PROBE_VERDICT " + ("stage-layout host store works" if passed else "FAILED"))


if __name__ == "__main__":
    main()
