# Raiden TPU disaggregation backend

`--disaggregation-transfer-backend raiden` moves KV pages between TPU prefill
and decode hosts using the [tpu-raiden](../../../../../../tpu-raiden) DMA engine,
the TPU analogue of the CUDA mooncake/nixl backends.

## How it works

Raiden is **consumer-pull**: the decode side reads KV from the prefill side
(the reverse of the push-based RDMA backends). SGLang only does rendezvous over
the existing ZMQ rank sockets + HTTP bootstrap server; raiden owns the actual
network transport, host staging pool, D2H/H2D DMA, and completion tracking.

Per request (keyed by `bootstrap_room`):

1. **Prefill** computes KV, then `RaidenKVSender.send()` accumulates the page
   block-ids and, on the last chunk, calls raiden `register_read(room, blocks)`
   and reads the aux/first-token blob from its (host) metadata buffer.
2. **Decode** `RaidenKVReceiver.send_metadata()` pushes a `RAIDEN_PULL_REQ`
   (room + its rank endpoint) to the target prefill rank.
3. **Prefill** replies `RAIDEN_BLOCKS` (its page block-ids + raiden control
   endpoint + aux blob) once both the blocks and the pull-request are known.
4. **Decode** writes the aux blob into its metadata buffer, then issues raiden
   `start_read(endpoint, remote_blocks, local_blocks)`.
5. Both sides drain `poll_stats()` in `poll()`; the prefill request frees on
   `done_sending`, the decode request runs on `done_recving`. Cross-rank gating
   is the framework's existing `poll_and_all_reduce` (min over ranks) — no
   worker-side barrier.

KV page mapping is exact: `kv_to_page_indices` already hands `send()` /
`send_metadata()` page ids in token order, which are raiden block ids directly
(the TPU fused KV pool's major dim is `num_blocks`).

## Requirements / scope (v1)

- `tpu_raiden` importable in the serving env (wheel built against the same
  `torch_tpu` as this fork; see tpu-raiden `build.sh`).
- Fused TPU KV pool (`pallas_tpu_rpa_v3` → `TpuFusedMHATokenToKVPool`).
- MHA models, **equal prefill/decode attn TP size** (heterogeneous TP raises
  `NotImplementedError` rather than transferring a subset).
- Decode runs in chunk-cache mode (decode radix cache stays gated to
  nixl/mooncake upstream).
- Not yet covered: MLA, PP, Mamba/SWA/NSA state, speculative draft KV, decode
  prefix-cache reuse.

## Tunables (env)

| Env | Default | Meaning |
|-----|---------|---------|
| `SGLANG_RAIDEN_KV_PORT` | `30000` | base control port; rank *r* uses `base + 2r` (data = `+1`) |
| `SGLANG_RAIDEN_NUM_SLOTS` | `16` | raiden host-pool transfer slots |
| `SGLANG_RAIDEN_MAX_BLOCKS` | `0` (auto) | max pages/request in raiden host pool; auto = `ceil(context_len/page_size)` |
| `SGLANG_RAIDEN_MAX_MODEL_LEN` | `8192` | fallback when `--context-length` is unset |
| `SGLANG_RAIDEN_TIMEOUT_S` | `120` | raiden transfer timeout |

## Launch sketch (one host each, equal TP)

```bash
# prefill
python -m sglang.launch_server --model-path <model> \
  --attention-backend pallas_tpu_rpa_v3 --page-size 64 \
  --disaggregation-mode prefill --disaggregation-transfer-backend raiden \
  --host 0.0.0.0 --port 30001 --disaggregation-bootstrap-port 8998

# decode (points at the prefill bootstrap server via the PD router)
python -m sglang.launch_server --model-path <model> \
  --attention-backend pallas_tpu_rpa_v3 --page-size 64 \
  --disaggregation-mode decode --disaggregation-transfer-backend raiden \
  --host 0.0.0.0 --port 30002
```

Drive both through the standard SGLang PD router (`sglang.srt.disaggregation`
mini-lb / router), same as mooncake.

## Testing

- Wire format / page mapping (host-only):
  `python -m pytest test/manual/kv_transfer/test_raiden_serde.py`
- End-to-end correctness + throughput: **TPU VM only** (this dev host is not a
  TPU VM and has no `tpu_raiden`). Run via the team cdk flow against a known
  baseline; compare decode output token-for-token against a single-instance
  (non-disaggregated) run of the same model/prompt.
