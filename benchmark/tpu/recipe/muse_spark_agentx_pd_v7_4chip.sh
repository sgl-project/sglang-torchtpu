#!/usr/bin/env bash
# Muse Spark AgentX Raiden P/D on two tpu7x 2x2x1 hosts: PP8 prefill on worker 0, TP8 decode on worker 1.
#
#   bash benchmark/tpu/recipe/muse_spark_agentx_pd_v7_4chip.sh             # on both hosts
#   bash benchmark/tpu/recipe/muse_spark_agentx_pd_v7_4chip.sh --dry-run   # print config only
#
# Run on every host of the group. TPU_WORKER_ID picks the role and TPU_WORKER_HOSTNAMES
# lists both hosts (set by the JobSet); worker 0 also serves the router on LB_PORT.
# Engine knobs live in configs/muse_spark/agentic_pd.py; this only points it at local weights.
# MODEL_ROOT holds meta-models/Muse-Spark-Open-816B-A42B-NVFP4 and RadixArk/Muse-Spark-Open-DFlash2.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # recipe/ -> repo root

DRY_RUN="${1:-}"
MODEL_ROOT="${MODEL_ROOT:-/models}"
export LB_PORT="${LB_PORT:-8000}"
MODEL=meta-models/Muse-Spark-Open-816B-A42B-NVFP4
DRAFT=RadixArk/Muse-Spark-Open-DFlash2
SWA_DRAFT="${MODEL_ROOT}/${DRAFT}-swa"

# Windowed draft: declaring the DFlash2 layers SWA keeps its KV in the target's SWA slots.
write_swa_draft() {
  mkdir -p "${SWA_DRAFT}"
  for f in "${MODEL_ROOT}/${DRAFT}"/*; do
    [ "$(basename "$f")" = config.json ] || ln -sfn "$f" "${SWA_DRAFT}/$(basename "$f")"
  done
  python3 - "${MODEL_ROOT}/${DRAFT}/config.json" "${SWA_DRAFT}/config.json" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["is_hybrid_swa"] = True
cfg["hybrid_layer_pattern"] = [1] * cfg["num_hidden_layers"]
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
PY
}

print_config() {
  python3 - "$MODEL_ROOT" "$MODEL" "$SWA_DRAFT" <<'PY'
import json, sys
from benchmark.tpu.launch_server import load_recipe
root, model, draft = sys.argv[1:]
cfg = load_recipe("recipe/configs/muse_spark/agentic_pd.py")
for role in ("prefill", "decode"):
    cfg[role]["server_args"].update(
        model_path=f"{root}/{model}",
        served_model_name=model,
        speculative_draft_model_path=draft,
    )
print(json.dumps(cfg, indent=1))
PY
}

if [ "${DRY_RUN}" = --dry-run ]; then
  print_config
  exit 0
fi
write_swa_draft
CONFIG=/tmp/muse_agentx_pd.json
print_config > "$CONFIG"
cat "$CONFIG"
exec python3 -m benchmark.tpu.launch_server --config "@${CONFIG}"
