# Inkling2 inference

Launch an NVFP4 checkpoint on eight NVLink-connected GB300 GPUs:

```bash
python -m sglang.launch_server --model-path /path/to/checkpoint \
    --attention-backend fa4 --bf16-gemm-backend torch --dtype bfloat16 \
    --tp 8 --ep 8 --page-size 128 --grammar-backend xgrammar
```

NVFP4 requires equal `--tp` and `--ep` values. Checkpoint precision, chat
rendering, and reasoning/tool parsers are selected automatically.

Optional flags:

- **Images:** `--enable-multimodal`. Requires vision weights and
  `image_preprocessing.json` in the checkpoint.
- **Speculative decoding:** `--speculative-algorithm DSPARK --context-length 131072`.
  Requires `mtp_config` and `mtp.safetensors`; the context length must not exceed
  `mtp_config.max_position_embeddings`.
- **Responses storage and continuation:** `--enable-response-store` enables
  retrieval and `previous_response_id`.

See [ARCHITECTURE.md](ARCHITECTURE.md) and [PROMPTING.md](PROMPTING.md) for model
and prompting details.
