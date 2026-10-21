# Raiden TPU disaggregation backend

`--disaggregation-transfer-backend raiden` moves KV pages between TPU prefill
and decode hosts with the [tpu-sync](https://github.com/google/tpu-sync) DMA engine —
the TPU analogue of the CUDA mooncake/nixl backends.

## How it works

Raiden is **consumer-pull**: decode reads KV from prefill (reverse of the
push-based RDMA backends). SGLang only does rendezvous over the existing ZMQ
rank sockets + HTTP bootstrap server; raiden owns the network transport, host
staging, D2H/H2D DMA, and completion tracking.

Per request (keyed by `bootstrap_room`): prefill `register_read`s its page
block-ids; decode pushes a pull request with its rank endpoint; prefill replies
with its block-ids + raiden endpoint + aux blob; decode issues `start_read`.
The decode sends the pull request only after it reserved the request's KV
pages, and the prefill holds the request in its bootstrap queue until that
request (or a reshard or pipeline request) arrives, so prefill KV is never
computed for a decode that has no pages for it.
Both sides bound that wait with `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`, each
counted from when it received the request: the prefill fails a room the decode
has not asked for, and the decode fails a room still waiting for the handshake
or for KV pages. Set it the same on both. A prefill that receives a pull,
reshard, or pipeline request for a room it has already failed answers with
`RAIDEN_ROOM_FAILED`, and the decode fails the room at once instead of holding
its pages until `SGLANG_DISAGGREGATION_WAITING_TIMEOUT`.
Both sides drain `poll_stats_with_details()` in `poll()` — prefill frees on
`sent`, decode runs on `received`. Each settled transfer carries raiden's finish
time; on TPU the prefill's `prefill_transfer_kv_cache` stage ends there, not at
the later poll (`hardware_backend/tpu/patches.py`).
Cross-rank status first reduces to the minimum
over the complete attention-TP/CP group. If any rank fails, every Raiden poller
tombstones the room before a second reduction confirms that accepted native
I/O has drained on every rank; only then can the framework release KV pages.
Page ids come from `kv_to_page_indices` in token order and are raiden block ids
directly.

KV moves chunk by chunk: as each chunked-prefill step commits its pages the
prefill announces them, and the decode pulls each piece as soon as the host
staging pool can seat it, so only the last chunk's transfer is left after the prefill. The
closing message carries the aux blob and the remaining pages. A request keeps
one chunk in flight at a time, and pages the decode already holds from its own
prefix cache are never announced; a decode that already holds every page settles
the request on the closing message alone. A hybrid model's sliding-window pages
ride on the closing read, since the window is only settled when the prefill ends.

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
ports via `base + 6*node_worker_rank`, where the node worker rank counts the
attention-TP ranks of every context-parallel, attention-DP, and system-DP
replica on the node (reduces to `base + 6*attn_tp_rank` for a single replica);
each rank owns 6 ports —
full-engine control/data, SWA-engine control/data, then GDN-state-engine
control/data.

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

With `--enable-unified-memory` both kinds share one pool's buffers
(`sglang_tpu/unified_kv_pool.py`): full-attention layer i uses buffer i, and
sliding-window layers beyond the full count form G groups whose k-th layers
share buffer k, so a sliding-window page is one page id per group. The SWA
engine registers each buffer the sliding-window layers use once and moves a
window page as its G page ids, page by page
(`TpuUnifiedSWAKVPool.swa_transfer_pages`), so one block-id list fills every
layer and a tail of the list is a tail of the window. Its max blocks and
staging are G per window page. Both peers must run the unified pool with the
same layer split: a unified decode names its layout (`swa_layout`: buffers,
blocks per page) in `PULL_REQ` and `RESHARD_REQ`, and other decodes leave the
field out, so their messages are unchanged. A prefill with another layout fails
the room and answers `ROOM_FAILED`; a prefill that predates the field rejects
the message. A unified decode on the pipeline transfer
(`SGLANG_RAIDEN_PIPELINE_TRANSFER`) is refused at startup.

Push-resharding runs the same two-engine split. `RESHARD_REQ` carries the
decode's SWA pages and its SWA transfer address alongside the full ones, and
`RESHARD_BLOCKS` carries the prefill's; each side then builds a second plan
over the SWA sub-pool with `_make_geometry(..., swa=True)`, which takes the
head-pair granule from the SWA tensor so the two pools may differ in dtype or
head_dim. The window is **never chunked** — it is settled only when the prefill
ends — so it always travels on the closing push, even when chunks already cover
every full-attention page. Completion is unchanged: `is_send_done` /
`is_recv_done` already require `swa:<room>` whenever a second engine exists.

## Hybrid linear-attention (GDN) models

`HybridLinearKVPool` registers only its full-attention layers on the full
engine; the `TpuMambaPool` conv and recurrent tensors (one row per state slot)
go on a third engine with `max_blocks=1`. Each side maps its request's
`StateType.MAMBA` slot through `TpuMambaPool.committed_row`, so prefill and
decode may use different checkpoint widths. The state row rides the `BLOCKS`
reply and is read as `state:<room>` alongside the whole read (or the closing
read of a chunked transfer); a full prefix hit still moves it, since state
cannot be rebuilt from cached KV. Push-resharding and pipeline transfer raise
`NotImplementedError` for these models.

## Scope (v1)

Fused TPU KV pool (`pallas_tpu_rpa_v3`), MHA models (incl. hybrid
sliding-window at symmetric TP and under push-resharding, and hybrid GDN at
symmetric TP), decode in
chunk-cache mode.
Symmetric attn TP uses the pull path; `dst_tp < src_tp` uses the
push-resharding path above. Needs `tpu_sync` built against the same `torch_tpu`
as this fork (tpu-sync `build.sh`). Not covered: MLA,
NSA, Mamba/GDN under push-resharding or pipeline transfer, speculative draft
KV, decode prefix-cache reuse, `dst_tp > src_tp`.

## Tunables (env)

| Env | Default | Meaning |
|-----|---------|---------|
| `SGLANG_RAIDEN_PIPELINE_TRANSFER` | `0` | named layer pools for a PP producer and a DP, TP, or matching-PP consumer; set to `1` on a decode that is not itself a pipeline |
| `SGLANG_RAIDEN_KV_PORT` | `31000` | base control port, kept clear of the default serving ports; local rank uses `base + 6*node_worker_rank` over every PP/CP/DP replica's TP ranks (data = `+1`, SWA engine control/data = `+2`/`+3`, GDN state engine control/data = `+4`/`+5`) |
| `SGLANG_RAIDEN_HOST_POOL_GB` | `128` | host staging ceiling per node, split across local ranks and engines; each rank's pool is sized for at most one max-length transfer per running request and seats in-flight transfers page by page. Startup fails with the required and available sizes when not even one max-length transfer fits |
| `SGLANG_RAIDEN_MAX_BLOCKS` | `0` (auto) | full-attention max pages/request; auto = `ceil(context_len/page_size)` |
| `SGLANG_RAIDEN_SWA_MAX_BLOCKS` | `0` (auto) | SWA max pages/request; auto includes one page for an unaligned window start. A unified pool with G sliding-window groups stages G blocks per page |
| `SGLANG_RAIDEN_MAX_MODEL_LEN` | `8192` | fallback when `--context-length` is unset |
| `SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S` | `60` | raiden deadline of each KV chunk transfer (the closing read counts as a chunk; not the whole request; `SGLANG_RAIDEN_TIMEOUT_S` is a deprecated alias), set the same on prefill and decode: the prefill keeps a read registered for this long after announcing it and then releases its pages; a decode read that waited longer than this for a staging slot is failed instead of pulled |
| `SGLANG_RAIDEN_LISTENER_START_READ` | `1` | decode only: the listener thread starts a request's next read as soon as the prefill announces it, instead of waiting for the next scheduler poll; `0` starts reads only on the poll. Admission is unchanged and completion is still detected by the poll. Pipeline and resharded transfers are unaffected |
| `SGLANG_RAIDEN_DYNAMIC_HOST_STAGING` | `1` | raiden seats each transfer in its own pages of the pool pinned at startup; resharding stages each plan in its own host blocks instead of shadowing the device KV pool. Setting `0` keeps fixed slots, where every transfer takes a whole max-length slot, and rejects asymmetric resharding, which needs per-transfer staging. Passed to raiden as `TPU_RAIDEN_DYNAMIC_HOST_STAGING`; that old name still works but is deprecated |
| `TPU_PREMAPPED_BUFFER_SIZE` | `4 GiB` (set by the TPU platform) | libtpu's premapped DMA buffer. Raiden builds that stage through libtpu's host allocator (full-mirror staging; tpu-sync before [49d29c0f](https://github.com/google/tpu-sync/commit/49d29c0fc07046bcda8640c3841c579665e93e24), including builds with PP pool resharding) carve each rank's host staging out of it, and staging past the buffer is DMA-mapped as each copy lands, which stalls the decode steps. Set it per rank above the rank's staging (`mirror_peak_bytes` in the `Raiden pipeline ready` line, about the rank's share of `SGLANG_RAIDEN_HOST_POOL_GB`; the pipeline warns when the buffer is smaller). Builds from 49d29c0f on DMA-map their staging when allocating it and do not draw on this buffer |

A decode rank issues a pull only while its host staging pool has the pages for it; requests past that stay in the transfer queue, in order, and are retried as pages free, so the pool does not have to hold a max-length transfer per running request. A pull that does not fit waits at the head of the queue, so smaller pulls cannot starve it. A request that waits in that queue for longer than `SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S` fails with that reason: its KV is gone from the prefill by then, so the pull is never sent.

The decode's `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` restarts at each chunk or closing announcement of a request, so it bounds how long the prefill goes quiet, not how long the whole transfer takes. A request with nothing announced yet still fails that long after the decode asked the prefill for its KV, and resharded and pipeline transfers always count it from then.

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
- End-to-end correctness + throughput: TPU VM only;
  compare decode output token-for-token against a single-instance run.

## Pipeline-parallel prefill

A PP prefill transfers KV to an attention-DP decode, a tensor-parallel decode,
or a decode pipeline with the same stage count; set
`SGLANG_RAIDEN_PIPELINE_TRANSFER=1` on the decode when it is not itself a
pipeline. The producer uses attention TP=1 and
CP=1 (`--pp-size 8 --tp-size 1`); an eight-way attention-DP consumer uses
`--tp-size 8 --dp-size 8 --enable-dp-attention`, a tensor-parallel consumer
uses `--tp-size 8`, and a matching pipeline consumer uses
`--pp-size 8 --tp-size 1`.

A tensor-parallel consumer receives every source page split by kv-head:
each rank's engine lands its own head slice of every token, and rank 0
plans the one transfer that fills all ranks. A slice must be whole 32-bit
packs of the fused page, so BF16 KV takes any head count per rank and FP8
KV needs an even head count per rank. The rank-4 RPA-seq pool
(`[blocks, 2 * kv_heads, head_dim, page]`) keeps a head's K and V rows
adjacent, so each rank's slice is one contiguous run per page.

This path requires a tpu-sync with PP pool resharding: per-layer pool
pairing, sender-local pool indices on the wire, registered transfer ranks,
shared claims across destination pipeline stages, and per-destination strided
byte spans (one strided entry per page and destination rank). The strided
spans a tensor-parallel consumer needs are on tpu-sync main from
[120bab96](https://github.com/google/tpu-sync/commit/120bab966c3a6c360abc752cf55357ebeddb5b41);
with an older build each such transfer fails at planning. For
full throughput the controller also needs its own thread pool for worker
RPCs ([tpu-sync main 6b3e8206](https://github.com/google/tpu-sync/commit/6b3e8206f5a0f4ad7bf4717790bff4c8eb7e1c1c)):
a transfer coordinates every source stage, and on asyncio's default
executor concurrent transfers queue behind each other.
Each worker registers the full-attention and sliding-window layers it owns
as `fa.lN` and `swa.lN`. tpu-sync requires every layer of one engine to hold
the same number of blocks, so a worker with hybrid SWA memory wraps its
sliding-window layers in a second engine (control port `+2`, as on the TP
path) registered with a second controller on stage zero. The window pages of
the closing transfer and of each stage group move as a twin native transfer
`swa:<id>` on that engine, announced in the same READY, and a transfer settles
only once its twin has. By default the closing transfer and its twin move after
the prefill finishes. Full-attention and window page lists remain independent,
and a cached decode prefix is clipped from each layer's transfer.

Three opt-in modes overlap the hand-off with the prefill; all are set on the
prefill (`SGLANG_RAIDEN_PIPELINE_COORD_WORKERS` on the decode):

| Env var | Default | Meaning |
|---|---|---|
| `SGLANG_RAIDEN_PIPELINE_EARLY_PARTS` | `0` | full-attention pages final before the prefill ends (a cached prefix, finished chunks) move as their own transfers (`<room>.p<i>`) while the pipeline computes |
| `SGLANG_RAIDEN_PIPELINE_PART_PAGES` | `16` | minimum pages per early part; fewer ride the closing transfer |
| `SGLANG_RAIDEN_PIPELINE_STAGE_PARTS` | `0` | each group of stages moves its layers (`<room>.s<g>`) once its forwards of the final chunk complete, instead of after the last stage samples; the first token follows from the last stage |
| `SGLANG_RAIDEN_PIPELINE_STAGE_GROUP` | `1` | consecutive stages per group transfer |
| `SGLANG_RAIDEN_PIPELINE_EARLY_CLOSE` | `0` | each stage registers its share of the closing transfer when it launches the final chunk, the last stage once the PP loop has copied that forward's sampled tokens, so no stage waits for its result; the first token comes with the last stage's share; not with stage groups |
| `SGLANG_RAIDEN_PIPELINE_COORD_WORKERS` | `1` | decode threads coordinating transfers, one controller client each |

A stage group is fenced on a host copy of one element of its forward's
output: torch_tpu reports a tensor materialized once its buffer exists, before
the execution that writes the KV completes. Stage groups other than the first
name non-contiguous source ranks, which needs a tpu-sync planner that accepts
any subset of a controller's registered transfer ranks. Each early part and
stage group holds its own decode staging reservation.

With early close, the decode starts the transfer only once the last stage has
announced its share, after its forward, which needs every earlier stage's. A
request returning logprobs or a sampling mask, using a grammar or a KV
checksum, or still bootstrapping gets its first token at the result point.

Each worker owns one Raiden engine, and with hybrid SWA memory a second one for
its sliding-window layers. Pipeline stages share stage zero's controller for
each kind of engine; independent attention-DP workers each host their own.
Pool geometry and materialized TPU layouts must match between peers, and
prefill and decode must both use hybrid SWA memory or both run with
`--disable-hybrid-swa-memory`. Every pipeline stage needs at least one
full-attention layer and, with hybrid SWA memory, at least one sliding-window
layer. Speculative decoding is supported only on the decode, and only when its
draft has no separate KV pool and takes no draft hidden states from the
prefill.
Host staging uses bounded leases when the installed Raiden supports per-pool
staging, which hybrid SWA memory requires. Older PP-capable wheels use full
host mirrors; startup measures their
materialized block-major page layouts and bounds their
peak allocation, including a temporary replacement buffer, against each
rank's share of `SGLANG_RAIDEN_HOST_POOL_GB`. Over-budget mirrors are rejected;
reduce `--max-total-tokens` or increase the host budget to fit them. Shared-memory
mirrors (`RAIDEN_SHM_KEY`) are unsupported. Both modes limit concurrent
transfers before calling the coordinator.
A failed or aborted request retains its pages until native transfer completion
or atomic cancellation of an unclaimed source registration. A transfer the
controller rejects fails its request only, and the decode keeps that request's
pages until its engine settles the transfer or the native deadline passes. A stage whose
entire decode prefix is cached releases only its matching source stage after
the shared request has been claimed. If every stage hits its full prefix,
source pages remain reserved until the unclaimed registration expires
(`SGLANG_RAIDEN_CHUNK_TRANSFER_TIMEOUT_S`, 60 seconds by default).

The pipeline unit tests are in
`test/registered/unit/hardware_backend/tpu/test_raiden_pipeline.py`,
`test_raiden_pipeline_overlap.py`, `test_raiden_pipeline_rank4.py` and
`test_raiden_pipeline_swa.py`.
