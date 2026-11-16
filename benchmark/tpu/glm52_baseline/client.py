"""Native HTTP cache-hit workload with DP receipts, MTP and streaming evidence.

Natural source-code inputs are frozen before timing. Shared prefixes change KV
occupancy, and are explicitly labelled; they are not an independent-prefix run.
"""

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import time
import uuid
from pathlib import Path

import aiohttp


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def persist_record(path, record, ids):
    record["input_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    save(path, record)
    return record


async def api(session, base, endpoint, payload=None):
    method = session.get if payload is None else session.post
    async with method(
        base + endpoint, **({} if payload is None else {"json": payload})
    ) as r:
        body = await r.text()
        if r.status != 200:
            raise RuntimeError(f"{endpoint}: HTTP {r.status}: {body[:2000]}")
        # The native cache-control endpoint returns plain text on success.
        return body if endpoint == "/flush_cache" else json.loads(body)


async def generate(
    session, base, ids, rank, output_len, output, label, text=None, accepted=None
):
    payload = dict(
        input_ids=ids,
        routed_dp_rank=rank,
        rid=str(uuid.uuid4()),
        stream=True,
        sampling_params=dict(temperature=0, max_new_tokens=output_len, ignore_eos=True),
    )
    if text is not None:
        payload.pop("input_ids")
        payload["text"] = text
    started = time.perf_counter()
    events, pending = [], b""
    async with session.post(base + "/generate", json=payload) as r:
        if r.status != 200:
            raise RuntimeError(f"generate: HTTP {r.status}: {(await r.text())[:2000]}")
        if accepted is not None:
            accepted.set()
        async for raw in r.content.iter_any():
            pending += raw
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    continue
                obj = json.loads(data)
                events.append(dict(time=time.perf_counter(), response=obj))
    assert events, "no streaming responses"
    final = events[-1]["response"]
    meta = final["meta_info"]
    result = dict(
        label=label,
        rank=rank,
        rid=payload["rid"],
        started=started,
        ended=time.perf_counter(),
        events=events,
        meta=meta,
        text=final.get("text"),
    )
    await asyncio.to_thread(persist_record, output / f"{label}.json", result, ids)
    assert meta.get("dp_rank") == rank, meta
    assert meta["completion_tokens"] == output_len, meta
    assert meta.get("finish_reason", {}).get("type") != "abort", meta
    if output_len > 6:
        assert meta.get("spec_verify_ct", 0) > 0, f"MTP verify missing: {meta}"
    return result


def load_tokenizer(tokenizer_path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


def freeze_inputs(tokenizer_path, source, out, decode_prefix_mode, frozen=None):
    import numpy as np

    tokenizer = load_tokenizer(tokenizer_path)
    decode_prefix_count = 1 if decode_prefix_mode == "shared" else 64
    need = 16 * (49152 + 4 * 4096) + decode_prefix_count * 89984 + 64 * 4 * 16
    if frozen is not None:
        frozen = Path(frozen)
        manifest = json.loads((frozen / "input-manifest.json").read_text())
        arr = np.load(frozen / "natural-code.npy", allow_pickle=False)
        assert arr.dtype == np.int32 and arr.shape == (need,), (arr.dtype, arr.shape, need)
        assert manifest["token_count"] == need
        assert hashlib.sha256(arr.tobytes()).hexdigest() == manifest["token_ids_sha256"]
        np.save(out / "natural-code.npy", arr)
        save(out / "input-manifest.json", dict(manifest, reused_from=str(frozen)))
        return tokenizer, arr.tolist()
    ids, files = [], []
    for path in sorted((source / "python/sglang").rglob("*.py")):
        raw = path.read_bytes()
        text = raw.decode("utf-8", errors="strict")
        ids.extend(tokenizer.encode(text, add_special_tokens=False))
        files.append(
            dict(
                path=str(path.relative_to(source)),
                sha256=hashlib.sha256(raw).hexdigest(),
            )
        )
        if len(ids) >= need:
            break
    assert len(ids) >= need, (len(ids), need)
    arr = np.asarray(ids[:need], dtype=np.int32)
    np.save(out / "natural-code.npy", arr)
    save(
        out / "input-manifest.json",
        dict(
            input_mode="natural-code",
            files=files,
            token_count=need,
            token_ids_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
            construction="sorted UTF-8 python/sglang/*.py; concatenate encoded source; no repeated filling",
        ),
    )
    return tokenizer, ids


async def batch(session, base, inputs, output_len, out, label, synchronized=False):
    snapshots = []
    stop = asyncio.Event()

    async def monitor():
        while not stop.is_set():
            try:
                state = await api(session, base, "/v1/loads")
                snapshots.append(dict(time=time.perf_counter(), **state))
            except Exception as e:
                snapshots.append(dict(time=time.perf_counter(), error=str(e)))
            await asyncio.sleep(0.02)

    if synchronized:
        await api(session, base, "/pause_generation", {"mode": "in_place"})
    accepted = [asyncio.Event() for _ in inputs]
    tasks = [
        asyncio.create_task(
            generate(
                session,
                base,
                ids,
                rank,
                output_len,
                out,
                f"{label}-{i:02d}",
                accepted=accepted[i],
            )
        )
        for i, (rank, ids) in enumerate(inputs)
    ]
    monitoring = None
    try:
        if synchronized:
            # Paused Torch schedulers do not publish fresh /v1/loads snapshots.
            # HTTP headers acknowledge ingress only. Sustained per-DP batch size
            # is checked independently after inference resumes.
            ingress = asyncio.ensure_future(
                asyncio.wait_for(
                    asyncio.gather(*(event.wait() for event in accepted)), timeout=120
                )
            )
            try:
                done, _ = await asyncio.wait(
                    [ingress, *tasks], return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
                if not ingress.done():
                    raise RuntimeError(
                        "a request finished before synchronized ingress completed"
                    )
            finally:
                if not ingress.done():
                    ingress.cancel()
                await asyncio.gather(ingress, return_exceptions=True)
            for task in tasks:
                if task.done():
                    task.result()
            await asyncio.to_thread(
                save,
                out / f"{label}-ingress.json",
                dict(
                    http_accepted=len(accepted),
                    scope="HTTP SSE headers accepted while paused; not scheduler admission proof",
                ),
            )
            await api(
                session, base, "/continue_generation", {"torch_empty_cache": False}
            )
        monitoring = asyncio.create_task(monitor())
        return await asyncio.gather(*tasks), snapshots
    finally:
        stop.set()
        if monitoring:
            await monitoring
        await asyncio.to_thread(save, out / f"{label}-loads.json", snapshots)
        if synchronized:
            await api(
                session, base, "/continue_generation", {"torch_empty_cache": False}
            )
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def summarize(records, loads, cached, per_dp, out, label):
    for record in records:
        assert record["meta"]["prompt_tokens"] == cached + (4096 if per_dp == 1 else 16)
        # The final cached page can require recomputing one page for the logits.
        hit = record["meta"].get("cached_tokens", -1)
        assert cached - 64 <= hit <= cached, (record["label"], hit, cached)
    tpot, ttft = [], []
    for r in records:
        events = [
            e
            for e in r["events"]
            if e["response"]["meta_info"]["completion_tokens"] > 0
        ]
        first, last = events[0]["time"], events[-1]["time"]
        ttft.append(1000 * (first - r["started"]))
        tpot.append(1000 * (last - first) / (r["meta"]["completion_tokens"] - 1))
    full = [
        s
        for s in loads
        if len(s.get("loads", [])) == 16
        and all(r["num_running_reqs"] == per_dp for r in s["loads"])
    ]
    assert full, "no load snapshot proves the requested per-DP active batch"
    result = dict(
        label=label,
        physical_chips=8,
        requests=len(records),
        requests_per_dp=per_dp,
        expected_cached_tokens=cached,
        observed_cached_tokens=[r["meta"]["cached_tokens"] for r in records],
        ttft_ms_median=statistics.median(ttft),
        client_tpot_ms_median=statistics.median(tpot),
        mtp_verify_count=sum(r["meta"].get("spec_verify_ct", 0) for r in records),
        mtp_correct_drafts=sum(
            r["meta"].get("spec_num_correct_drafts", 0) for r in records
        ),
        full_concurrency_load_samples=len(full),
        timing_scope="client SSE; TTFT includes HTTP and scheduling; TPOT = first-to-last / (tokens-1)",
    )
    if per_dp == 4:
        # Common window after the first 64 outputs and before any final 64 outputs.
        starts, ends = [], []
        for r in records:
            events = r["events"]
            starts.append(
                next(
                    e["time"]
                    for e in events
                    if e["response"]["meta_info"]["completion_tokens"] >= 64
                )
            )
            ends.append(
                next(
                    e["time"]
                    for e in events
                    if e["response"]["meta_info"]["completion_tokens"] >= 960
                )
            )
        start, end = max(starts), min(ends)
        assert end > start, "no common steady-state decode window"
        samples = [s for s in loads if start <= s["time"] <= end]
        assert len(samples) >= 2 and all(
            len(s.get("loads", [])) == 16
            and all(r["num_running_reqs"] == 4 for r in s["loads"])
            for s in samples
        ), "C64 not sustained"

        def count_at(record, timestamp):
            return max(
                (
                    e["response"]["meta_info"]["completion_tokens"]
                    for e in record["events"]
                    if e["time"] <= timestamp
                ),
                default=0,
            )

        tokens = sum(count_at(r, end) - count_at(r, start) for r in records)
        result.update(
            steady_start=start,
            steady_end=end,
            steady_output_tokens=tokens,
            steady_output_tokens_per_second=tokens / (end - start),
            steady_output_tokens_per_second_per_chip=tokens / (end - start) / 8,
            tail_seconds=max(r["ended"] for r in records) - end,
        )
    save(out / f"{label}-summary.json", result)
    return result


async def main(a):
    a.output.mkdir(parents=True, exist_ok=False)
    await asyncio.to_thread(
        save,
        a.output / "client-identity.json",
        dict(
            client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            implementation_commit=__import__("os").environ.get("SOURCE_COMMIT"),
            benchmark_source_sha256=__import__("os").environ.get(
                "GLM52_BENCHMARK_SHA256"
            ),
        ),
    )
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=aiohttp.TCPConnector(limit=160, keepalive_timeout=1)
    ) as session:
        info = await api(session, a.base_url, "/server_info")
        await asyncio.to_thread(save, a.output / "server-info.json", info)
        expected = dict(
            tp_size=16,
            attn_dp_size=16,
            ep_size=16,
            speculative_algorithm="EAGLE",
            speculative_num_steps=5,
            speculative_num_draft_tokens=6,
            kv_cache_dtype="fp8_e4m3",
            chunked_prefill_size=2048,
            disable_radix_cache=False,
            enable_dp_lm_head=True,
        )
        assert all(info.get(k) == v for k, v in expected.items()), {
            k: info.get(k) for k in expected
        }
        # Keep the loop alive while imports/tokenization run: otherwise an idle
        # HTTP connection can expire on the server while aiohttp still sees it
        # as reusable, making the next POST fail with ServerDisconnectedError.
        tokenizer = await asyncio.to_thread(load_tokenizer, a.model)
        assert tokenizer.chat_template
        checks = []
        for rank in range(16):
            text = tokenizer.apply_chat_template(
                [
                    dict(
                        role="user",
                        content="What is 2 + 2? Answer with the number only.",
                    )
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            r = await generate(
                session,
                a.base_url,
                [],
                rank,
                16,
                a.output,
                f"smoke-{rank:02d}",
                text=text,
            )
            assert "4" in r["text"], r["text"]
            checks.append(r["meta"])
        await asyncio.to_thread(
            save, a.output / "smoke-summary.json", dict(status="PASS", checks=checks)
        )
        if a.smoke_only:
            return
        capacities = await api(session, a.base_url, "/v1/loads")
        await asyncio.to_thread(
            save, a.output / "capacity-before-workload.json", capacities
        )
        assert len(capacities["loads"]) == 16
        minimum_capacity = min(
            row["max_total_num_tokens"] for row in capacities["loads"]
        )
        unique_slots = 4 * ((90000 + 1024 + 6 + 63) // 64) * 64
        decode_mode = a.decode_prefix
        if decode_mode == "auto":
            decode_mode = "unique" if minimum_capacity >= unique_slots else "shared"
        await asyncio.to_thread(
            save,
            a.output / "prefix-decision.json",
            dict(
                mode=decode_mode,
                minimum_capacity=minimum_capacity,
                unique_slots_lower_bound=unique_slots,
                shared_prefix_tokens=89984,
                reason="independent 90K/C64 requests fit"
                if decode_mode == "unique"
                else "share prefix within each DP to reduce KV slots",
            ),
        )
        _, ids = await asyncio.to_thread(
            freeze_inputs, a.model, a.source, a.output, decode_mode, a.frozen_inputs
        )
        all_summaries = []
        for scenario in ("prefill", "decode"):
            prefix_len, extend, count = (
                (49152, 4096, 16) if scenario == "prefill" else (89984, 16, 64)
            )
            await api(session, a.base_url, "/flush_cache")
            stride = 49152 + 4 * 4096
            shared = decode_mode == "shared" and scenario == "decode"
            decode_start = 16 * stride
            if scenario == "prefill":
                prefixes = [
                    ids[r * stride : r * stride + prefix_len] for r in range(16)
                ]
            elif shared:
                prefixes = [ids[decode_start : decode_start + prefix_len]] * 64
            else:
                prefixes = [
                    ids[
                        decode_start + i * prefix_len : decode_start
                        + (i + 1) * prefix_len
                    ]
                    for i in range(64)
                ]
            priming_count = 16 if shared else count
            await batch(
                session,
                a.base_url,
                [(i % 16, prefixes[i]) for i in range(priming_count)],
                1,
                a.output,
                f"{scenario}-prefix",
            )
            for iteration in range(4):
                inputs = []
                for i in range(count):
                    rank = i % 16
                    offset = (
                        rank * stride + prefix_len + iteration * extend
                        if scenario == "prefill"
                        else decode_start
                        + (1 if shared else 64) * 89984
                        + (iteration * count + i) * 16
                    )
                    inputs.append((rank, prefixes[i] + ids[offset : offset + extend]))
                nout = (16 if scenario == "prefill" else 1024) if iteration else 64
                label = (
                    f"{scenario}-{'shared' if shared else 'unique'}-round{iteration}"
                )
                records, loads = await batch(
                    session,
                    a.base_url,
                    inputs,
                    nout,
                    a.output,
                    label,
                    synchronized=True,
                )
                if iteration:
                    all_summaries.append(
                        await asyncio.to_thread(
                            summarize,
                            records,
                            loads,
                            prefix_len,
                            count // 16,
                            a.output,
                            label,
                        )
                    )
            # Raw metrics and logs remain authoritative for retraction/eviction checks.
        aggregated = {}
        for scenario, metric in (
            ("prefill", "ttft_ms_median"),
            ("decode", "client_tpot_ms_median"),
        ):
            values = [
                r[metric] for r in all_summaries if r["label"].startswith(scenario)
            ]
            assert len(values) == 3
            aggregated[scenario] = dict(
                metric=metric,
                median=statistics.median(values),
                minimum=min(values),
                maximum=max(values),
            )
        await asyncio.to_thread(
            save,
            a.output / "summary.json",
            dict(rounds=all_summaries, aggregate=aggregated, decode_prefix=decode_mode),
        )


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", default="/models/GLM5.2-fp8-channel-wise")
    p.add_argument("--source", type=Path, default=Path.cwd())
    p.add_argument("--frozen-inputs", type=Path, default=os.environ.get("GLM52_FROZEN_INPUTS"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument(
        "--decode-prefix", choices=["auto", "shared", "unique"], default="auto"
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
