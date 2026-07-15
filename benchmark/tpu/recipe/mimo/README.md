# MiMo-V2.5 DP-attention on TPU

Serving recipe for **MiMo-V2.5 (310B)** and **MiMo-V2.5-Pro (1T)** with DP-attention
on `tpu7x`: FP8 weights + FP8 KV cache, torch.compile, `pallas_tpu_rpa_v3` attention.

## Files
| file | what |
|---|---|
| `launch_mimo_25.sh` | MiMo-V2.5 310B — **single host** (tp8 / ep8 / dp2, attn_tp=4) |
| `launch_mimo_25_pro_multihost.sh` | MiMo-V2.5-Pro 1T — **2 hosts** (tp16 / ep16 / dp2, attn_tp=8) |

## Launch

### MiMo-V2.5 (310B) — single host
```bash
./launch_mimo_25.sh
```

### MiMo-V2.5-Pro (1T) — 2 hosts
Start **rank 0 first**; wait until it logs `Initializing TPU distributed runtime`,
then start rank 1. Open `DIST_PORT` (10011) between the hosts.
```bash
# host 0 (serves HTTP :30000):
./launch_mimo_25_pro_multihost.sh 0
# host 1:
RANK0_HOST=<host0-internal-ip> ./launch_mimo_25_pro_multihost.sh 1
```


## Notes / gotchas
- **fp8 runtime = channelwise** (the default). Do **not** pass `--tpu-fp8-runtime blockwise`
  for Pro — its attention qkv shard (3392) isn't divisible by 128 and the block kernel rejects it.
- **Pro `mem_fraction_static=0.7`**: the 1T fp8 weights are ~⅓ of each chip's HBM; lower
  values (e.g. 0.3) leave a negative KV budget ("Not enough memory").
- **DP-attention sizing**: `attention_tp = tp_size / dp_size` must equal `num_key_value_heads`
  (4 for 310B, 8 for Pro). `torch_compile_max_bs ≥ max_running × dp_size`.
