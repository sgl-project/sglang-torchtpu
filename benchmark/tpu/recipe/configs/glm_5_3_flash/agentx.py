"""GLM-5.3-Flash AgentX on one tpu7x host (tp8, 4 chips) at the model's 1M context.

The server configs of the AgentX points at concurrency 1, 4, 8, 16, 32 and 48, as one
profile table on top of mtp.py. Each profile renders the launch JSON its point was
measured with (MEASURED, in experimental/agentx/runs/; check_recipes.py there diffs
them). experimental/agentx/launch_agentx.sh boots one point with its caches on a PVC.

    python3 -m benchmark.tpu.recipe.configs.glm_5_3_flash.agentx --conc 16
    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/glm_5_3_flash/agentx.py --concurrency 16

Every point runs fp8 weights and KV with 128-token pages and 4096-token prefill
chunks, MTP (topk 1) at InferenceX's golden acceptance length, the MoE ring kernel
with decode on the model path (megakernel off), and a raiden host tier that also holds
KDA states, with a KDA checkpoint at every prefill chunk. c32 and c48 also split the
index-key cache, with libtpu's SparseCore all-gather offload off (SPARSE_CORE_FIX);
c32 also drops the KDA checkpoints taken during decode.
"""

import argparse
import json
from copy import deepcopy
from typing import NamedTuple

from benchmark.tpu.recipe.configs.glm_5_3_flash.mtp import CONFIG as MTP_CONFIG

CONTEXT_LENGTH = 1048576

# InferenceX golden acceptance length (bonus token included) per MTP draft length K
# (speculative_num_steps; K + 1 draft tokens), SPEED-Bench coding, thinking on.
GOLDEN_AL = {1: 1.86, 2: 2.52, 3: 3.00, 4: 3.35, 5: 3.60, 6: 3.79, 7: 3.93, 8: 3.99}

# The SparseCore-offloaded all-gathers return wrong data in the index-split decode graph.
SPARSE_CORE_FIX = " --xla_tpu_enable_sparse_core_collective_offload_all_gather=false"


class Profile(NamedTuple):
    running: int  # max running requests
    steps: int | str  # MTP draft steps K, or K by running requests ("1:7,2:3")
    pool_tokens: int  # device KV pool cap; HBM may bound it lower
    kda_slots: int  # device KDA checkpoints, for the conversations AgentX keeps cached
    host_ratio: float  # host KV tier, x device pool
    interval: int  # decode steps between prefill batches
    mk_rows: int  # megakernel row cap: inert while it is off, as the runs set it
    shards: int = 1  # ranks splitting the MLA latent cache by page (1: unsplit)
    index_split: bool = False  # split the index-key cache too (needs shards=8)
    bf16_merge: bool = False  # the split cache's prefill merge in bf16
    state_ratio: float | None = None  # host KDA states, x KDA slots (None: host_ratio)
    track_interval: int | None = None  # decode tokens per KDA checkpoint (None: 256)
    compile_bs: tuple[int, ...] | None = None  # decode buckets (None: 1..running)
    spmd_compile: bool = False  # unsplit cache only: compile once, broadcast the result


PROFILES = {
    1: Profile(
        running=2,
        steps="1:7,2:3",
        pool_tokens=3407872,
        kda_slots=74,
        host_ratio=2,
        interval=16,
        mk_rows=16,
        spmd_compile=True,
    ),
    4: Profile(
        running=4,
        steps=3,
        pool_tokens=3407872,
        kda_slots=384,
        host_ratio=2,
        interval=16,
        mk_rows=32,
    ),
    # Split 2 without the index-key split, as measured.
    8: Profile(
        running=8,
        steps="1:7,2:3",
        pool_tokens=9437184,
        kda_slots=800,
        host_ratio=1.5,
        interval=16,
        mk_rows=32,
        shards=2,
    ),
    # Unsplit index keys. Split (runs/C16_S2ts_kda1400_bf16_real_l3_ring_full_ixsplit.json:
    # 8 shards, pool 25M, host 1.71x): same tok/s, lower E2E-normalized p90, more host RAM.
    16: Profile(
        running=16,
        steps=3,
        pool_tokens=9437184,
        kda_slots=1400,
        host_ratio=1.7,
        interval=16,
        mk_rows=64,
        shards=2,
        bf16_merge=True,
        state_ratio=1.429,
    ),
    # A fixed interval starves admission from c32: prefill runs whenever it waits.
    32: Profile(
        running=32,
        steps=3,
        pool_tokens=14155776,
        kda_slots=1200,
        host_ratio=1.2,
        interval=0,
        mk_rows=64,
        shards=8,
        index_split=True,
        bf16_merge=True,
        state_ratio=2.5,
        # Decode-time KDA checkpoints off: unused, they forced KDA and KV evictions.
        track_interval=CONTEXT_LENGTH,
        # Host RAM grows with every decode program: 1..32 does not fit.
        compile_bs=(1, 2, 4, 8, 12, 16, 20, 24, 28, 32),
    ),
    48: Profile(
        running=48,
        steps=3,
        pool_tokens=28400000,
        kda_slots=800,
        host_ratio=1.127,
        interval=0,
        mk_rows=64,
        shards=8,
        index_split=True,
        bf16_merge=True,
        state_ratio=2.5,
        compile_bs=(*range(1, 17), *range(20, 49, 4)),
    ),
}

# The launch JSON each point was measured with, in experimental/agentx/runs/.
MEASURED = {
    1: "R1b_c1_real_l3_ring_full",
    4: "C4u_c4_real_l3_ring_full",
    8: "K2ts_c8_real_l3_ring_full",
    16: "C16_S2ts_kda1400_bf16_real_l3_ring_full",
    32: "R10_c32_mr32_combo_real_l3_ring_full_ixsplit",
    48: "R9_c48_mr48_real_l3_ring_full_ixsplit",
}


def config_for(conc, *, simulate_acceptance=True, spmd_compile=True):
    """The server config for one AgentX concurrency.

    simulate_acceptance=False verifies drafts for real (evals); spmd_compile=False
    compiles on every rank (torch_tpu 0.2.0, as c1 was measured).
    """
    if conc not in PROFILES:
        raise ValueError(
            f"no AgentX profile for concurrency {conc}: {sorted(PROFILES)}"
        )
    p = PROFILES[conc]
    schedule = p.steps if isinstance(p.steps, str) else None
    tiers = [p.steps]
    if schedule:  # "1:7,2:3": 7 draft steps at 1 running request, 3 from 2
        tiers = sorted({int(tier.split(":")[1]) for tier in schedule.split(",")})
    steps = tiers[-1]  # launch with the most any decode step drafts
    config = deepcopy(MTP_CONFIG)
    args, env = config["server_args"], config["runtime_env"]
    args.update(
        {
            "served_model_name": "zai-org/GLM-5.3-Flash",
            "reasoning_parser": "glm45",
            "tool_call_parser": "glm47",
            "context_length": CONTEXT_LENGTH,
            "max_running_requests": p.running,
            "torch_compile_max_bs": p.running,
            "tpu_compile_bs": list(p.compile_bs or range(1, p.running + 1)),
            "tpu_compile_tokens": [1024, 2048, 3072, 4096],  # prefill programs
            "chunked_prefill_size": 4096,
            "max_prefill_tokens": 4096,
            "tpu_compile_tracer": "direct",  # trace graphs from the module, not Dynamo
            "max_total_tokens": p.pool_tokens,
            "max_mamba_cache_size": p.kda_slots,
            "hicache_ratio": p.host_ratio,
            "hybrid_cache_checkpoints_level": 3,  # a KDA checkpoint every prefill chunk
            "speculative_num_steps": steps,
            "speculative_num_draft_tokens": steps + 1,
            "prefill_decode_interval": p.interval,
            "scheduler_recv_interval": 8,
            "enable_metrics": True,
            "enable_cache_report": True,  # cached_tokens in usage: the prefix hit rate
            "enable_request_time_stats_logging": True,
            "weight_loader_prefetch_checkpoints": True,  # NFS load 454 s, ~19 min without
            "weight_loader_drop_cache_after_load": True,  # else page cache evicts the pod
        }
    )
    if p.track_interval is not None:
        args["mamba_track_interval"] = p.track_interval
    if p.shards == 1:
        # Unsplit cache: scattered prefill input off (SPMD slices rows at a rank offset).
        args["enable_attn_tp_input_scattered"] = False
        args["use_spmd_compile"] = p.spmd_compile and spmd_compile
    env.update(
        {
            "SGLANG_TIMEOUT_KEEP_ALIVE": "900",  # outlast the client's pooled connections
            "SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S": "2400",  # skewed NFS loads
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL": "0",  # decode on the model path
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_LEN": str(CONTEXT_LENGTH),
            "SGLANG_TPU_GLM_FLASH_MEGAKERNEL_MAX_ROWS": str(p.mk_rows),
            "SGLANG_TPU_GLM_FLASH_PAGE_TABLE_HBM": "1",  # 1M context: page table in HBM
            "SGLANG_TPU_ENABLE_SCHEDULER_THREAD_PINNING": "1",
            "SGLANG_TPU_DSA_KPOOL_RAW_RING": "1",  # pooled indexer rows in a ring
            "SGLANG_TPU_ENABLE_MOE_RING_KERNEL": "1",  # experts stream once via VMEM
            "SGLANG_TPU_ENABLE_MOE_PREQUANT_DISPATCH": "1",  # fp8 rows into the MoE
            "SGLANG_TPU_DSA_OWNED_ROWS_KERNEL": "1",  # token-split rows in one pass
            "SGLANG_TPU_FUSED_MHC_INPUT_SCATTERED": "1",  # fused mHC on scattered prefill
            "SGLANG_TPU_ROUTER_TOPK_KERNEL_PREFILL": "1",
            "SGLANG_TPU_DSA_KV_SHARD_RAGGED_BLOCK": "128",  # decode: own candidates only
        }
    )
    if p.shards > 1:
        env["SGLANG_TPU_DSA_MLA_KV_SHARDS"] = str(p.shards)
        # Each selected cache row is gathered by the rank that holds it.
        env["SGLANG_TPU_ENABLE_DSA_ATTENTION_TOKEN_SPLIT"] = "1"
    if p.bf16_merge:
        env["SGLANG_TPU_DSA_MLA_KV_SHARD_BF16_MERGE"] = "1"
    if p.index_split:
        assert p.shards == 8, "the index split's exact prefill exchange needs 8 shards"
        env["SGLANG_TPU_DSA_INDEX_KV_SHARD"] = "1"
        env["LIBTPU_INIT_ARGS"] += SPARSE_CORE_FIX
    if p.state_ratio is not None:
        env["SGLANG_TPU_HICACHE_STATE_RATIO"] = str(p.state_ratio)
    if schedule:
        env["SGLANG_TPU_SPEC_STEPS_BY_RUNNING"] = schedule
    if simulate_acceptance:
        env.update(
            {
                "SGLANG_SIMULATE_ACC_LEN": str(GOLDEN_AL[steps]),
                "SGLANG_SIMULATE_ACC_METHOD": "match-expected",
                "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token",
            }
        )
        if schedule:
            env["SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS"] = ",".join(
                f"{k}:{GOLDEN_AL[k]:.2f}" for k in tiers
            )
    return config


def get_config(concurrency):
    """launch_server --concurrency builds the config here."""
    return config_for(concurrency)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conc", type=int, required=True, choices=sorted(PROFILES))
    parser.add_argument(
        "--eval", action="store_true", help="verify for real (no simulated acceptance)"
    )
    parser.add_argument(
        "--no-spmd-compile",
        action="store_true",
        help="compile on every rank (torch_tpu 0.2.0, as c1 was measured)",
    )
    args = parser.parse_args(argv)
    config = config_for(
        args.conc,
        simulate_acceptance=not args.eval,
        spmd_compile=not args.no_spmd_compile,
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
