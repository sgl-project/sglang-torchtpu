"""Manual TPU check of the reverse Raiden transfer of the P/D prefix pull.

A TP2 "decode" (two processes, one chip each) fans its pages into a TP1 "prefill
stage" (a third process) through tpu-sync's pool reshard, with the span builder
and arguments of pp_reverse.py: each decode rank registers its kv-head slice of
the stage's layers on the decode's controller, and the stage coordinates the
transfer there. The decode holds two layers, the stage only the second, as a
pipeline stage holds a layer subset. The stage's pages must equal the decode's,
head for head.

Run on one tpu7x host with three free chips, from the repo root:
    PYTHONPATH=python:. python3 test/manual/kv_transfer/tpu_prefix_pull_reverse.py
"""

from __future__ import annotations

import concurrent.futures
import math
import multiprocessing as mp
import os
import socket
import sys
import threading
import time
import traceback
from types import SimpleNamespace

HEADS = 4
TP = 2
HD_WORDS = int(os.environ.get("PULL_TEST_HD_WORDS", "2"))
PACKING = 4
PAGE = 128
BLOCKS = 64
# Pages one transfer may stage, below the pool size as in serving, so host
# staging runs on bounded leases instead of full mirrors.
STAGING_BLOCKS = 8
DECODE_TAGS = ["fa.l0", "fa.l1"]
STAGE_TAGS = ["fa.l1"]
# Decode pages of the prefix the stage pulls, and the stage's pages for them.
DECODE_PAGES = [9, 2, 7, 4, 11]
ROOM = 4242
# Concurrent rooms, and the staging leases of each side: fewer leases than
# rooms makes the stage's coordinates meet the lease-wait refusal.
ROOMS = int(os.environ.get("PULL_TEST_ROOMS", "1"))
DECODE_LEASES = int(os.environ.get("PULL_TEST_DECODE_LEASES", "2"))
PREFILL_LEASES = int(os.environ.get("PULL_TEST_PREFILL_LEASES", "2"))


def stage_pages(index):
    return [6 * index + offset for offset in (3, 0, 4, 1, 2)]


FINGERPRINT = "prefix-pull-reverse-test"
TIMEOUT_S = 120.0


def _truth(tag):
    import torch

    gen = torch.Generator().manual_seed(DECODE_TAGS.index(tag) + 1)
    shape = (BLOCKS, 2 * HEADS, HD_WORDS, PACKING, PAGE)
    return torch.randint(-1000, 1000, shape, generator=gen).to(torch.bfloat16)


def _pool(tag, index, tensor):
    size = math.prod(int(n) for n in tensor.shape[1:]) * tensor.element_size()
    return dict(
        tag=tag,
        storage_index=index,
        base_offset_bytes=0,
        block_stride_bytes=size,
        num_blocks=int(tensor.shape[0]),
        dtype_tag="bfloat16",
        regions=[
            dict(
                name="kv",
                offset_bytes=0,
                stride_bytes=size,
                unit_bytes=size,
                num_units=1,
            )
        ],
    )


def _unit(controller, data_name, rank):
    return dict(
        job_name="sglang",
        job_replica_id=controller,
        data_name=data_name,
        data_replica_idx=rank,
    )


def _worker(role, rank, chip, inbox, outbox):
    os.environ.update(
        TPU_CHIPS_PER_PROCESS_BOUNDS="1,1,1",
        TPU_PROCESS_BOUNDS="1,1,1",
        TPU_VISIBLE_CHIPS=str(chip),
    )
    try:
        _run(role, rank, inbox, outbox)
    except Exception:
        outbox.put(("error", role, rank, traceback.format_exc()))


def _store(ip, port, name):
    from tpu_sync.api.torch.reshard_store import RaidenId, ReshardStore

    store = ReshardStore(
        RaidenId(
            job_name="sglang",
            job_replica_id=f"{ip}:{port}",
            data_name=name,
            data_replica_idx=0,
        ),
        store_server_ip=ip,
    )
    return store, (
        f"{ip}:{store.reshard_service_port}",
        store.raiden_controller_address,
    )


def _run(role, rank, inbox, outbox):
    import torch
    import torch_tpu  # noqa: F401
    from tpu_sync.api.torch.kv_cache_manager import KVCacheManager
    from tpu_sync.api.torch.reshard_client import ReshardClient
    from tpu_sync.rpc import raiden_service_pb2

    from sglang_tpu.disaggregation.raiden.pp_reverse import (
        destination_args,
        fan_in_spans,
        pull_req_id,
        pull_uuid,
    )

    ip = socket.gethostbyname(socket.gethostname())
    control_port = 41000 + 10 * (rank if role == "decode" else TP)
    decode = role == "decode"
    store = None
    if decode and rank == 0 or not decode:
        store, addresses = _store(ip, control_port, f"{role}-store")
        if decode:
            outbox.put(("decode_store", addresses))
    if decode and rank != 0:
        addresses = inbox.get(timeout=TIMEOUT_S)
    controller, dispatch = addresses

    if decode:
        rows = 2 * HEADS // TP
        tags = DECODE_TAGS
        tensors = [
            _truth(tag)[:, rank * rows : (rank + 1) * rows].contiguous().to("tpu")
            for tag in tags
        ]
        workers = TP
    else:
        tags = STAGE_TAGS
        tensors = [torch.zeros_like(_truth(tag)).to("tpu") for tag in tags]
        workers = 1
    pools = [_pool(tag, i, t) for i, (tag, t) in enumerate(zip(tags, tensors))]
    engine = KVCacheManager(
        kv_caches=tensors,
        node_id=rank if decode else 0,
        local_control_port=control_port,
        max_blocks=STAGING_BLOCKS,
        num_slots=DECODE_LEASES if decode else PREFILL_LEASES,
        timeout_s=60.0,
        listener_port=0,
        parallelism=workers,
        raiden_controller_address=dispatch,
        worker_id=f"worker_{rank if decode else 0}",
    )
    leases = DECODE_LEASES if decode else PREFILL_LEASES
    engine.register_pools(
        pools,
        staging_leases=leases,
        staging_blocks_per_pool=[STAGING_BLOCKS] * len(pools),
    )
    data_name = "decode" if decode else "prefill"
    unit_fields = _unit(controller, data_name, rank if decode else 0)
    unit = SimpleNamespace(**unit_fields)
    client = ReshardClient(controller)
    client.register_work_unit(
        unit=unit,
        shards=[engine.transfer_address],
        control_plane_rpc_address=engine.listener_address,
        pool_manifest=pools,
        layout_fingerprint=FINGERPRINT,
        page_tokens=PAGE,
        transfer_parallelism=workers,
        transfer_rank=rank if decode else 0,
    )
    ids = [(pull_req_id(ROOM + i, 0), pull_uuid(ROOM + i, 0)) for i in range(ROOMS)]

    if decode:
        stage_pools = [p for p in pools if p["tag"] in STAGE_TAGS]
        stage_tensors = [t for p, t in zip(pools, tensors) if p["tag"] in STAGE_TAGS]
        spans = fan_in_spans(
            stage_pools,
            stage_tensors,
            DECODE_PAGES,
            [],
            src_tp=TP,
            src_rank=rank,
            total_kv_heads=HEADS,
            seq_along_lane=True,
        )
        for req_id, uuid in ids:
            client.register_request_blocks(
                req_id=req_id,
                uuid=uuid,
                unit=unit,
                block_ids=DECODE_PAGES,
                pool_spans=spans,
            )
        outbox.put(("registered", rank, unit_fields))
        kinds = _wait_all(engine, [req_id for req_id, _ in ids])
        for req_id, uuid in ids:
            client.complete_request_blocks(req_id, uuid, unit)
        outbox.put(("done", role, rank, _summary(kinds, "sent")))
        inbox.get(timeout=TIMEOUT_S)  # stay up until the stage checked
        return

    decode_units = inbox.get(timeout=TIMEOUT_S)
    decode_controller = decode_units[0]["job_replica_id"]
    refusals = {}
    lock = threading.Lock()
    deadline = time.monotonic() + TIMEOUT_S
    started = time.monotonic()

    def coordinate(index):
        req_id, uuid = ids[index]
        while not _coordinate_once(index, req_id, uuid):
            with lock:
                refusals[req_id] = refusals.get(req_id, 0) + 1
            if time.monotonic() > deadline:
                raise TimeoutError(f"coordinate_transfer never accepted: {refusals}")
            time.sleep(0.02)

    def _coordinate_once(index, req_id, uuid):
        # One client per thread: a ReshardClient owns one socket.
        return ReshardClient(decode_controller).coordinate_transfer(
            src_units=[SimpleNamespace(**u) for u in decode_units],
            dst_units=[unit],
            req_id=req_id,
            uuid=uuid,
            use_block_chunks=True,
            is_sender=True,
            dst_mem_type=raiden_service_pb2.MEMORY_TYPE_HBM,
            src_controller_address=decode_controller,
            dst_controller_address=controller,
            **destination_args(STAGE_TAGS, stage_pages(index), []),
        )

    # Every room coordinates at once, as concurrent pulls would.
    with concurrent.futures.ThreadPoolExecutor(max_workers=ROOMS) as pool:
        for future in [pool.submit(coordinate, i) for i in range(ROOMS)]:
            future.result()
    kinds = _wait_all(engine, [req_id for req_id, _ in ids])
    elapsed = time.monotonic() - started
    got = tensors[0].cpu()
    want = _truth(STAGE_TAGS[0])
    mismatched = [
        (i, src, dst)
        for i in range(ROOMS)
        for src, dst in zip(DECODE_PAGES, stage_pages(i))
        if not torch.equal(got[dst], want[src])
    ]
    head_bytes = HD_WORDS * PACKING * PAGE * got.element_size() * 2
    summary = _summary(kinds, "received")
    if mismatched:
        summary = f"mismatch {mismatched}"
    outbox.put(
        (
            "done",
            role,
            0,
            f"{summary}; pages checked={ROOMS * len(DECODE_PAGES)} x {HEADS} heads "
            f"({head_bytes} B/head); refusals={refusals}; {elapsed:.2f}s",
        )
    )


def _summary(kinds, want):
    bad = {k: v for k, v in kinds.items() if v != want}
    return want if not bad else f"unexpected {bad}"


def _wait_all(engine, req_ids):
    kinds = {}
    deadline = time.monotonic() + TIMEOUT_S
    while len(kinds) < len(req_ids) and time.monotonic() < deadline:
        stats = engine.poll_stats_with_details()
        for kind in ("sent", "received", "failed"):
            for t in getattr(stats, kind):
                if t.req_id in req_ids:
                    kinds[t.req_id] = kind
        time.sleep(0.01)
    for req_id in req_ids:
        kinds.setdefault(req_id, "timeout")
    return kinds


def main() -> int:
    ctx = mp.get_context("spawn")
    to_d1, to_stage, to_d0, results = (ctx.Queue() for _ in range(4))
    d0 = ctx.Process(target=_worker, args=("decode", 0, 0, to_d0, results))
    d0.start()
    msg = results.get(timeout=TIMEOUT_S)
    if msg[0] != "decode_store":
        print(msg)
        return 1
    to_d1.put(msg[1])
    d1 = ctx.Process(target=_worker, args=("decode", 1, 1, to_d1, results))
    stage = ctx.Process(target=_worker, args=("prefill", 0, 2, to_stage, results))
    d1.start()
    stage.start()
    units, done, ok = {}, {}, True
    deadline = time.monotonic() + 3 * TIMEOUT_S
    while len(done) < 3 and time.monotonic() < deadline:
        msg = results.get(timeout=3 * TIMEOUT_S)
        if msg[0] == "error":
            print(f"{msg[1]} rank {msg[2]} failed:\n{msg[3]}")
            ok = False
            break
        if msg[0] == "registered":
            units[msg[1]] = msg[2]
            if len(units) == TP:
                to_stage.put([units[r] for r in range(TP)])
        elif msg[0] == "done":
            done[(msg[1], msg[2])] = msg[3]
            print(f"{msg[1]} rank {msg[2]}: {msg[3]}")
    ok = (
        ok
        and done.get(("decode", 0)) == "sent"
        and done.get(("decode", 1)) == "sent"
        and str(done.get(("prefill", 0), "")).startswith("received;")
    )
    for queue in (to_d0, to_d1):
        queue.put("exit")
    for proc in (d0, d1, stage):
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
