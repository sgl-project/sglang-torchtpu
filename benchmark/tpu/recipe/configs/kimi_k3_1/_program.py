"""What Kimi-K3.1-Pro changes in the Kimi-K3 program recipes (K3_PROGRAM):
its DSpark draft, its golden acceptance and its pre-shard."""

import os
import re

# Inferact/Kimi-K3.1-Pro-DSpark-preview is the draft of InferenceX's K3.1
# DSpark golden curve (probabilistic draft sampling, block verification);
# AgentX accepts at the curve's mean length per draft width
# (infx/golden_al_distribution/kimi-k3.1-pro__dspark__...): 1.82 at one
# draft, 2.88 at three, 3.69 at seven.
MODEL = "moonshotai/Kimi-K3.1-Pro"
DRAFT = "Inferact/Kimi-K3.1-Pro-DSpark-preview"
ACC_LENGTHS = "7:3.69,3:2.88,1:1.82"
# The decode indexer selects every key at or above the top-2048 threshold's
# 20-bit bin (five 4-bit rounds over the DCP group): the top-2048 and the bin's
# other keys, three exchange rounds fewer than the exact threshold.
SELECT_ROUNDS = "dsa_select_rounds=5"

# The colocated c1 AgentX settings (colocated_defaults): 497.3 tok/s/user at
# 1289 tok/s/chip on one 2x2x4 slice (mean ITL 2.011 ms), needles at 6k-548k
# and GSM8K 0.990 with real acceptance. Every kernel setting keeps the decode's
# outputs bit for bit except kda_fp8_projections and decode_router_fp8 (FP8 KDA
# projection and router weights). K3_PROGRAM_KERNEL_EXTRA overrides any of them.
DECODE_SETTINGS = (
    "dsa_compact_tokens=128",
    "dsa_compact_window=512",
    "b8_singleton_select=true",
    "b16_one_hot_staging=true",
    "draft_whole_weights=true",
    "decode_router_both_halves=true",
    "expert_schedule=true",
    "dsa_fused_commit=true",
    "decode_tc_all_reduce=true",
    "decode_mla_prefix_region=true",
    "shared_expert_passes=true",
    "dcp_pair_combine=true",
    "attention_host_bf16=true",
    "dsa_stage_slots=4",
    "dsa_page_slots=8",
    "dsa_resident_keys=true",
    "dsa_patch_loop=true",
    "dsa_early_plan=true",
    "decode_late_shared_down=true",
    "draft_fused_epilogue=true",
    "draft_lm_prefetch=true",
    "expert_steal=true",
    "attention_rank_rows=true",
    "kda_fp8_projections=true",
    "dsa_live_activity=true",
    "dsa_odd_stride=true",
    "draft_epilogue_unroll=8",
    "dsa_seeded_window=true",
    "b8_compact_experts=true",
    "decode_presorted_routing=true",
    "verify_greedy_kernel=true",
    "decode_embedding_broadcast=true",
    "draft_native_pages=true",
    "draft_step_trims=true",
    "moe_token_parallel_up=true",
    "attention_host_round_after_join=true",
    "decode_copy_free_weights=true",
    "verify_fused_tail=true",
    "crossed_pair_gathers=true",
    "dsa_prologue_in_select=true",
    "dsa_row_block_masks=true",
    "dcp_shift_scalars=true",
    "dcp_query_tail_unpack=true",
    "dsa_exchange_fill=true",
    "dsa_vector_patch=true",
    "decode_mla_early_handout=true",
    "dsa_gather_overlap=true",
    "draft_with_verify=true",
    "draft_prologue_overlap=true",
    "draft_markov_rows=true",
    "decode_router_fp8=true",
    "decode_latent_down_prefetch=true",
    "dsa_native_pages=true",
    "dcp_pair_lse_rows=true",
    "moe_routed_send_in_expert_region=true",
    "attention_input_gather_in_branches=true",
)
# The TensorCore's highest clock (11-15% on the MXU, VPU and scalar work; the
# interconnect rounds are unchanged).
LIBTPU_EXTRA = "--xla_tpu_dvfs_p_state=7"


def apply(config: dict) -> dict:
    """``config`` (a Kimi-K3 program recipe's CONFIG) for Kimi-K3.1-Pro."""
    config["server_args"]["model_path"] = MODEL
    env = config["runtime_env"]
    if env.get("SGLANG_TPU_K3_PROGRAM_ATTENTION_DP", "1") != "1":
        # The DSA indexer gathers its 32 query heads over a DCP32 group.
        raise ValueError(
            "Kimi-K3.1 on the K3 programs decodes with attention DP1 (DCP32)"
        )
    env["SGLANG_TPU_K3_PROGRAM_DRAFT"] = DRAFT
    env["SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS"] = os.environ.get(
        "K3_PROGRAM_SIMULATE_ACC_LENGTHS", ACC_LENGTHS
    )
    env["SGLANG_TPU_K3_PROGRAM_PRESHARDED"] = os.environ.get(
        "SGLANG_TPU_K3_PROGRAM_PRESHARDED", "/presharded-k31"
    )
    add_libtpu_flags(env, os.environ.get("K3_PROGRAM_LIBTPU_EXTRA", ""))
    kernel = env.get("SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG") or ""
    if "dsa_select_rounds=" not in kernel:
        env["SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG"] = ";".join(
            filter(None, [kernel, SELECT_ROUNDS])
        )
    if "prefill_sparse=1" in (env.get("SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG") or ""):
        sparse_prefill_memory(config)
    return config


def add_libtpu_flags(env: dict, flags: str) -> None:
    """Set libtpu ``flags`` (space- or comma-separated) on top of ``env``'s
    LIBTPU_INIT_ARGS, replacing a flag of the same name. runtime_env replaces
    the inherited environment, so an exported LIBTPU_INIT_ARGS has no effect:
    K3_PROGRAM_LIBTPU_EXTRA carries extra flags instead."""
    extra = flags.replace(",", " ").split()
    if not extra:
        return
    names = {flag.split("=", 1)[0] for flag in extra}
    kept = [
        flag
        for flag in (env.get("LIBTPU_INIT_ARGS") or "").split()
        if flag.split("=", 1)[0] not in names
    ]
    env["LIBTPU_INIT_ARGS"] = " ".join(kept + extra)


def colocated_defaults(config: dict) -> dict:
    """The colocated c1 AgentX settings on top of :func:`apply`: both programs at
    expert parallelism 8, DECODE_SETTINGS ahead of K3_PROGRAM_KERNEL_EXTRA, the
    host-side bookkeeping and first-step drafts, a prefill wait of 150 rounds
    and LIBTPU_EXTRA. Each follows its environment variable when set
    (K3_PROGRAM_LIBTPU_EXTRA="" drops the clock flag)."""
    env = config["runtime_env"]
    parts = [
        p
        for p in (env.get("SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG") or "").split(";")
        if p
    ]
    user = [
        p for p in (os.environ.get("K3_PROGRAM_KERNEL_EXTRA") or "").split(";") if p
    ]
    for item in user:  # moved behind DECODE_SETTINGS, so they still win
        parts.remove(item)
    parallel = os.environ.get("K3_PROGRAM_EXPERT_PARALLEL", "8")
    parts = [
        re.sub(
            r"^(prefill|decode)_expert_parallel=\d+$",
            rf"\1_expert_parallel={parallel}",
            p,
        )
        for p in parts
    ]
    env["SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG"] = ";".join(
        parts + list(DECODE_SETTINGS) + user
    )
    for name, default in (
        ("SGLANG_TPU_K3_PROGRAM_OPT_HOST_BOOKKEEPING", "1"),
        ("SGLANG_TPU_K3_PROGRAM_OPT_FIRST_STEP_DRAFTS", "1"),
    ):
        env[name] = os.environ.get(name, default)
    env["SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_ROUNDS"] = os.environ.get(
        "K3_PROGRAM_PREFILL_WAIT", "150"
    )
    if "K3_PROGRAM_LIBTPU_EXTRA" not in os.environ:
        add_libtpu_flags(env, LIBTPU_EXTRA)
    return config


def sparse_prefill_memory(config: dict) -> dict:
    """The sparse prefill's largest window program reserves ~9.1 GiB of HBM
    scratch (dense: ~1.3 GiB less): a smaller KV budget leaves it room."""
    args = config["server_args"]
    args["mem_fraction_static"] = min(args.get("mem_fraction_static", 0.92), 0.88)
    return config
