"""DeepSeek-V4.1-Flash + its bundled DSpark draft on one tpu7x-8 host.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark.py

The deepseek_v41_flash.py deployment with DSpark on the
experimental TPU spec worker: the target and draft graphs are traced by the
direct tracer and each compiled once on an owner rank (--use-spmd-compile).
The draft's MoE shards over the target's EP8 group (~1.6 GB/device).
VERIFY_SPMD_SAFE=1 passed on all 8 ranks for the target graphs; the four
draft block graphs share the compile too, and every rank's accept length
matched the per-rank compile (4.2-4.6).

Measured at 103k on one tpu7x-8 with MODEL=gs://... (weights and Engram
streamed from object storage, tokenizer from a local copy), cold compile
caches: ready in 10.5 min; 1024 in / 1024 out random-ids c64 1782 tok/s and
c128 2999 tok/s at accept length 4.3-4.4.

At 1M with the radix cache and SWA bounded replay: ready in 12 min, 18.5M-token
KV pool per worker, ~15.8 GB HBM left after precompile. GSM8K 1319 (5-shot,
c128) 0.906 at accept length 3.5 (0.911 with 2048 prefill tokens per worker,
the same as at 103k without the radix cache). AgentX traces with 128 running
requests, accept length simulated at 3.51: c8 6179 and c32 12348 output
tok/s/chip.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        # The draft weights are the mtp.* tensors of the target checkpoint.
        "speculative_algorithm": "DSPARK",
        # 5 draft tokens per step: the verify runs 6 rows per request.
        "speculative_dspark_block_size": 5,
        # At 0.85 the larger KV pool leaves the 2048-token extend program
        # ~280 MB short of HBM at load; 0.83 loads every graph.
        "mem_fraction_static": 0.83,
        # Under attention DP any rank's prefill chunk stalls all 8 ranks' spec
        # decode. The prefill delayer batched lone prefills (random 1k/1k c64:
        # 1004 -> 1900 tok/s) but on AgentX it only adds TTFT: it holds a lone
        # request up to 30 passes, re-matching its prefix each pass. Instead,
        # run 4 decode rounds after each prefill step. AgentX FAST, same pod
        # (j-1b901cb8-32f5-4331-ba04), vs delayer on: c32 +30% tok/s, +160%
        # p90 interactivity, p90 TTFT +10%; c8 +4% tok/s, -6% p90, p90 TTFT
        # -30%.
        "enable_prefill_delayer": False,
        "prefill_decode_interval": 4,
        # Keeps only request-owned SWA windows, so prefix hits replay the
        # bounded SWA tail instead of reserving an SWA pool.
        "disable_radix_cache": False,
        "enable_encoder_swa_bounded_replay": True,
    }
)
CONFIG["runtime_env"].update(
    {
        "SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1",
        # Each rank holds 1/8 of the LM head; the target verify and the draft
        # score their rows through one all-to-all pair; GSM8K 0.914 vs 0.911.
        "SGLANG_TPU_ENABLE_DSV41_VOCAB_PARALLEL_HEAD": "1",
    }
)
