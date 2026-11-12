"""What Kimi-K3.1-Pro changes in the Kimi-K3 program recipes (K3_PROGRAM):
its DSpark draft, its golden acceptance and its pre-shard."""

import os

# Inferact/Kimi-K3.1-Pro-DSpark-preview is the draft of InferenceX's K3.1
# DSpark golden curve (probabilistic draft sampling, block verification);
# AgentX accepts at the curve's mean length per draft width
# (infx/golden_al_distribution/kimi-k3.1-pro__dspark__...): 1.82 at one
# draft, 2.88 at three, 3.69 at seven.
MODEL = "moonshotai/Kimi-K3.1-Pro"
DRAFT = "Inferact/Kimi-K3.1-Pro-DSpark-preview"
ACC_LENGTHS = "7:3.69,3:2.88,1:1.82"


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
    if "prefill_sparse=1" in (env.get("SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG") or ""):
        sparse_prefill_memory(config)
    return config


def sparse_prefill_memory(config: dict) -> dict:
    """The sparse prefill's largest window program reserves ~9.1 GiB of HBM
    scratch (dense: ~1.3 GiB less): a smaller KV budget leaves it room."""
    args = config["server_args"]
    args["mem_fraction_static"] = min(args.get("mem_fraction_static", 0.92), 0.88)
    return config
