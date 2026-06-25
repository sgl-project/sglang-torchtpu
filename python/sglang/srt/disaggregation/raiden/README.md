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
decode runs on `done_recving`. Cross-rank gating is the framework's existing
`poll_and_all_reduce` (min over ranks); no worker-side barrier. Page ids come
from `kv_to_page_indices` in token order and are raiden block ids directly.

## Scope (v1)

Fused TPU KV pool (`pallas_tpu_rpa_v3`), MHA models, **equal prefill/decode attn
TP**, decode in chunk-cache mode. Needs `tpu_raiden` built against the same
`torch_tpu` as this fork (tpu-raiden `build.sh`). Not covered: MLA, PP,
Mamba/SWA/NSA, speculative draft KV, decode prefix-cache reuse.

## Tunables (env)

| Env | Default | Meaning |
|-----|---------|---------|
| `SGLANG_RAIDEN_KV_PORT` | `30000` | base control port; rank *r* uses `base + 2r` (data = `+1`) |
| `SGLANG_RAIDEN_NUM_SLOTS` | `16` | raiden host-pool transfer slots |
| `SGLANG_RAIDEN_MAX_BLOCKS` | `0` (auto) | max pages/request; auto = `ceil(context_len/page_size)` |
| `SGLANG_RAIDEN_MAX_MODEL_LEN` | `8192` | fallback when `--context-length` is unset |
| `SGLANG_RAIDEN_TIMEOUT_S` | `120` | raiden transfer timeout |

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
- End-to-end correctness + throughput: TPU VM only (via the team cdk flow);
  compare decode output token-for-token against a single-instance run.
