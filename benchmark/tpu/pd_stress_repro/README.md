# P/D stress repro kit (TPU, 1P1D)

Scripts referenced by the P/D stress-test issues. Each engine runs on one TPU v7x
2x2x1 host (TP8), with raiden transfer, EAGLE3, priority scheduling and the
overlap scheduler on (default).

| File | What it does |
|---|---|
| `launch_pd.sh` | Starts the prefill engine, decode engine or router for one of three profiles: `gptoss`, `qwen3vl`, `qwen3vl-tight`. |
| `pd_burst.py` | Sends N mixed requests at once (short and long-prefix chats, raw `/generate`, 35% streaming, optional images); `--prio-mode mixed` draws priorities from {None, 0, 1, 10, 100, -10}. |
| `pd_load.py` | Tight-pool workloads: `prefix` (multi-turn sessions on long system prompts), `mmmu-bg` (image questions plus background load), `burst`. |
| `priority_probe.py` | Checks whether a request's `priority` reaches the prefill engine through a router. |
| `qwen3vl_moe_flags_workaround.patch` | One-line workaround that lets dense Qwen3-VL start with `--enable-torch-compile` on current main. Not a proposed fix. |

## Quick start

```bash
# prefill host
bash launch_pd.sh gptoss prefill
# decode host
bash launch_pd.sh gptoss decode
# router host: mini-LB on :8000 (use "rust" for the Rust PD gateway)
bash launch_pd.sh gptoss router <prefill-host> <decode-host> mini 8000

pip install aiohttp
python3 pd_burst.py --url http://<router>:8000 --model gpt-oss-120b --n 512 --seed 0 --prio-mode mixed
```

For Qwen3-VL, apply the patch first (`git apply qwen3vl_moe_flags_workaround.patch` from the repo root).

Overrides: `EXTRA_ARGS` appends server flags. `MAX_TOTAL_TOKENS=auto` drops
`--max-total-tokens` for `gptoss`. `HICACHE_RATIO` sets the prefill `--hicache-ratio`.
`SGLANG_HOST_IP` defaults to this host's first address.

## Watching for a P/D stall

- Engine logs: the last `Prefill batch` / `Decode batch` lines show the queue counters
  (`#queue-req`, `#bootstrap-req`, `#inflight-req`, `#prealloc-req`, `#transfer-req`),
  and no new batch lines appear afterwards.
- `/metrics`: `sglang:num_prefill_inflight_queue_reqs`, `sglang:num_prefill_bootstrap_queue_reqs`,
  `sglang:num_decode_transfer_queue_reqs`, `sglang:num_decode_prealloc_queue_reqs`.
- Recovery: `curl -X POST http://<engine>:30000/abort_request -H 'Content-Type: application/json' -d '{"abort_all": true}'` on both engines.

The load scripts generate the same request payloads as the tool used in the original
runs (image bytes excepted: they send a generated bar-chart PNG).
