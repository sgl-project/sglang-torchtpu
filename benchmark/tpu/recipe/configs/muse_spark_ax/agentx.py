"""Muse Spark AgentX on one tpu7x 2x2x1 host with the EP megakernel (SGLANG_TPU_MUSE_MOE_MODE=ep).

agentic.py's mixed engine (decode megakernel, 8-token DFlash on the experimental spec worker,
HiCache, fp8 SEQ_ALONG_LANE KV) with EP8 experts: extend runs prefill/spep, decode and verify
run decode/spep. EP lifts the TP megakernel's 48-row cap, so a step carries up to
running x 8 rows; SGLANG_TPU_MUSE_MK_EP_ARM picks gather / a2a / mixed.

    python3 -m benchmark.tpu.recipe.configs.muse_spark_ax.agentx --conc 32
"""

import argparse
import json
import os

from benchmark.tpu.recipe.configs.muse_spark.agentic import DRAFT_TOKENS, get_config
from benchmark.tpu.recipe.configs.muse_spark_ax.windowed_draft import default_dir

SERVED_MODEL = "meta-models/Muse-Spark-Open-816B-A42B-NVFP4"

# conc: (running requests, decode steps between prefill batches, compile batch sizes)
PROFILES = {
    1: (2, 128, [1, 2]),
    8: (8, 64, [1, 2, 4, 8]),
    16: (16, 64, [1, 2, 4, 8, 12, 16]),
    # the best single-host TP line peaks here (junyanxu q4mk sweeps)
    22: (22, 48, [1, 2, 4, 8, 12, 16, 22]),
    24: (24, 48, [1, 2, 4, 8, 12, 16, 24]),
    24: (24, 48, [1, 2, 4, 8, 12, 16, 24]),
    32: (32, 40, [1, 2, 4, 8, 12, 16, 24, 32]),
    # VR200 nov03 single-host points. Past 256 rows the row-tiled megakernel needs a multiple
    # of mv_rows (256): no 33-63 request buckets, so both cap admission at 64 (512 rows)
    48: (64, 32, [1, 2, 4, 8, 16, 24, 32, 64]),
    96: (64, 32, [1, 2, 4, 8, 16, 24, 32, 64]),
}

EP_ENV = {
    "SGLANG_TPU_MUSE_MOE_MODE": "ep",
    "SGLANG_TPU_MUSE_MK_QBLOCK": "256",
    "SGLANG_TPU_MUSE_MK_EXPERT_GROUP": "0",
    "SGLANG_TPU_MUSE_MK_PREFILL": "1",
    "SGLANG_TPU_MUSE_MK_PREFILL_SP": "1",
    "SGLANG_TPU_MUSE_MK_SP_EP_LAYOUT": "1",
    "SGLANG_TPU_MUSE_MK_SP_O_RS": "1",
}


def config_for(conc: int, *, draft_model_path: str | None, moe_mode: str, ep_arm: str) -> dict:
    running, interval, compile_bs = PROFILES[conc]
    # PDI: decode steps between prefill batches, overriding the profile's
    interval = int(os.environ.get("PDI") or interval)
    # agentic's c16 patch carries the fp8 SEQ_ALONG_LANE pool --mk-prefill needs
    config = get_config(16, mode="mixed", mk_prefill=True)
    args, env = config["server_args"], config["runtime_env"]
    args.update(
        served_model_name=SERVED_MODEL,
        speculative_draft_model_path=draft_model_path or default_dir(),
        max_running_requests=running,
        torch_compile_max_bs=running,
        tpu_compile_bs=compile_bs,
        prefill_decode_interval=interval,
        host="0.0.0.0",
    )
    env.pop("SGLANG_TPU_MUSE_MK_PREFILL_FP8_GATHER", None)
    rows = running * DRAFT_TOKENS
    env["SGLANG_TPU_MUSE_MK_MAX_ROWS"] = str(max(32, -(-rows // 16) * 16))
    env["SGLANG_TIMEOUT_KEEP_ALIVE"] = "900"
    if os.environ.get("NO_CKPT_PREFETCH") == "1":
        # many hosts reading one bucket: a prefetch read can outlast the loader's 60 s stop wait
        args["weight_loader_prefetch_checkpoints"] = False
    if moe_mode == "ep":
        args["ep_size"] = 8
        env.update(EP_ENV)
        env["SGLANG_TPU_MUSE_MK_EP_ARM"] = ep_arm
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--conc", type=int, required=True)
    parser.add_argument("--draft-model-path")
    # run_agentx.sh passes these for the TP harness; the EP profile ignores them
    parser.add_argument("--mk-prefill", action="store_true")
    parser.add_argument("--mk-decode", action="store_true")
    args = parser.parse_args(argv)
    config = config_for(
        args.conc,
        draft_model_path=args.draft_model_path,
        moe_mode=os.environ.get("MOE_MODE", "ep"),
        ep_arm=os.environ.get("EP_ARM", "mixed"),
    )
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
