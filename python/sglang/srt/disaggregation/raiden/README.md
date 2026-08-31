# Raiden TPU disaggregation backend

`--disaggregation-transfer-backend raiden` moves KV pages between TPU prefill
and decode hosts with the [tpu-raiden](../../../../../../tpu-raiden) DMA engine —
the TPU analogue of the CUDA mooncake/nixl backends.

## How it works

Raiden is **consumer-pull**: decode reads KV from prefill (reverse of the
push-based RDMA backends). SGLang only does rendezvous over the existing ZMQ
rank sockets + HTTP bootstrap server; raiden owns the network transport, host
staging, D2H/H2D DMA, and completion tracking.

Per request (keyed by `bootstrap_room`): prefill `register_read`s its page
block-ids; decode pushes a pull request with its rank endpoint; prefill replies
with its block-ids + raiden endpoint + aux blob; decode issues `start_read`.
Both sides drain `poll_stats()` in `poll()` — prefill frees on `done_sending`,
decode runs on `done_recving`. Cross-rank status first reduces to the minimum
over the complete attention-TP/CP group. If any rank fails, every Raiden poller
tombstones the room before a second reduction confirms that accepted native
I/O has drained on every rank; only then can the framework release KV pages.
Page ids come from `kv_to_page_indices` in token order and are raiden block ids
directly.

KV moves chunk by chunk: as each chunked-prefill step commits its pages the
prefill announces them, and the decode pulls each piece as soon as a staging
slot is free, so only the last chunk's transfer is left after the prefill. The
closing message carries the aux blob and the remaining pages. A request keeps
one chunk in flight at a time, and pages the decode already holds from its own
prefix cache are never announced. A hybrid model's sliding-window pages ride on
the closing read, since the window is only settled when the prefill ends.

## Asymmetric attention TP (strided push-resharding)

When decode TP < prefill TP (e.g. 8TP (P) -> 4TP2DP (D)), a decode rank
holds a wider kv-head slice than any single prefill rank, so the whole-block pull
cannot place it. That case uses raiden's push-resharding, engaged per request
when a decode rank is fed by several prefill ranks: decode rank `r` is fed by prefill ranks
`r*ratio`(`ratio = src_tp // dst_tp`). The strided interleave lands on the decode
receiver, which then auto-H2Ds the assembled host staging into HBM.

Handshake (decode-initiated, data flows prefill→decode): `RESHARD_REQ` (decode
endpoint + blocks + tp) → `RESHARD_BLOCKS` (prefill blocks) → decode registers its
receiver plan → `RESHARD_READY` → prefill `d2h`-stages and `push_registered_plan`.
Schedules are built in `reshard_schedule.py` (unit-tested in
`test/manual/kv_transfer/test_reshard_schedule.py`). Requires `ratio` to divide
the total kv-head count evenly; the byte granule is one kv-head's contiguous K+V.

DP-attention replicas that co-locate on a decode host get distinct raiden control
ports via `base + 4*node_worker_rank`, where the node worker rank counts the
attention-TP ranks of every context-parallel, attention-DP, and system-DP
replica on the node (reduces to `base + 4*attn_tp_rank` for a single replica);
each rank owns 4 ports —
full-engine control/data, then SWA-engine control/data.

## Hybrid sliding-window models

Hybrid-SWA models (e.g. gpt-oss) keep two KV pools with distinct page-id
spaces: full-attention layers in a full-size pool, sliding-window layers in a
smaller pool that only retains the attention window. One raiden transfer
applies a single block-id list to every tensor it wraps, so each rank runs a
second engine over the SWA sub-pool's layers. The full engine pulls the whole
sequence's pages; the SWA engine pulls just the window's pages, using the
per-side SWA page ids both schedulers already derive for `StateType.SWA`
(`_swa_payload` in prefill.py/decode.py). The extra block list and endpoint
ride in the existing `BLOCKS` reply.

Push-resharding runs the same two-engine split. `RESHARD_REQ` carries the
decode's SWA pages and its SWA transfer address alongside the full ones, and
`RESHARD_BLOCKS` carries the prefill's; each side then builds a second plan
over the SWA sub-pool with `_make_geometry(..., swa=True)`, which takes the
head-pair granule from the SWA tensor so the two pools may differ in dtype or
head_dim. The window is **never chunked** — it is settled only when the prefill
ends — so it always travels on the closing push, even when chunks already cover
every full-attention page. Completion is unchanged: `is_send_done` /
`is_recv_done` already require `swa:<room>` whenever a second engine exists.

## Scope (v1)

Fused TPU KV pool (`pallas_tpu_rpa_v3`), MHA models (incl. hybrid
sliding-window at symmetric TP and under push-resharding), decode in
chunk-cache mode.
Symmetric attn TP uses the pull path; `dst_tp < src_tp` uses the
push-resharding path above. Needs `tpu_raiden` built against the same `torch_tpu`
as this fork (tpu-raiden `build.sh`). Not covered: MLA, PP, Mamba/NSA,
speculative draft KV, decode prefix-cache reuse, `dst_tp > src_tp`.

## Tunables (env)

| Env | Default | Meaning |
|-----|---------|---------|
| `SGLANG_RAIDEN_KV_PORT` | `30000` | base control port; local rank uses `base + 4*node_worker_rank` over every CP/DP replica's TP ranks (data = `+1`, SWA engine control/data = `+2`/`+3`) |
| `SGLANG_RAIDEN_HOST_POOL_GB` | `128` | host staging ceiling per node, split across local ranks and engines; the pool holds as many in-flight transfers as fit it, at most the running batch. Startup fails with the required and available sizes when not even one transfer fits |
| `SGLANG_RAIDEN_MAX_BLOCKS` | `0` (auto) | full-attention max pages/request; auto = `ceil(context_len/page_size)` |
| `SGLANG_RAIDEN_SWA_MAX_BLOCKS` | `0` (auto) | SWA max pages/request; auto includes one page for an unaligned window start |
| `SGLANG_RAIDEN_MAX_MODEL_LEN` | `8192` | fallback when `--context-length` is unset |
| `SGLANG_RAIDEN_TIMEOUT_S` | `120` | raiden transfer timeout |
| `TPU_RAIDEN_DYNAMIC_HOST_STAGING` | `1` | raiden allocates host staging per transfer from the pool; resharding stages each plan in its own host blocks instead of shadowing the device KV pool. Setting `0` keeps fixed slots and rejects asymmetric resharding, which needs per-transfer staging |

A decode rank issues at most one pull per host staging slot; requests past that stay in the transfer queue and are retried as slots free, so the batch size does not have to fit the slot count.

## Launch (one host each, equal TP)

```bash
# prefill
python -m sglang.launch_server --model-path <model> \
  --attention-backend pallas_tpu_rpa_v3 --page-size 64 \
  --disaggregation-mode prefill --disaggregation-transfer-backend raiden \
  --host 0.0.0.0 --port 30001 --disaggregation-bootstrap-port 8998
# decode (points at prefill's bootstrap via the PD router)
python -m sglang.launch_server --model-path <model> \
  --attention-backend pallas_tpu_rpa_v3 --page-size 64 \
  --disaggregation-mode decode --disaggregation-transfer-backend raiden \
  --host 0.0.0.0 --port 30002
```

Drive both through the standard SGLang PD router, same as mooncake.

## Testing

- Wire format / page mapping (host-only):
  `python -m pytest test/manual/kv_transfer/test_raiden_serde.py`
- Resharding schedule byte-math (host-only, proto mocked):
  `python -m pytest test/manual/kv_transfer/test_reshard_schedule.py`
- End-to-end correctness + throughput: TPU VM only (via the team cdk flow);
  compare decode output token-for-token against a single-instance run.
