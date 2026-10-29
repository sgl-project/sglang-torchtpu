#!/usr/bin/env bash
# Muse Spark AgentX server on one tpu7x 2x2x1 host (4 chips, TP8) for one client concurrency.
#
#   bash benchmark/tpu/recipe/muse_spark_agentx_v7_4chip.sh 16
#   bash benchmark/tpu/recipe/muse_spark_agentx_v7_4chip.sh 16 --dry-run   # print config only
#
# Every concurrency-dependent knob (batch shape, KV layout, envs) lives in
# configs/muse_spark/agentic.py; this only points it at local weights.
# MODEL_ROOT holds meta-models/Muse-Spark-Open-816B-A42B-NVFP4 and RadixArk/Muse-Spark-Open-DFlash2.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # recipe/ -> repo root

CONCURRENCY="${1:?usage: $0 <concurrency> [--dry-run]}"
DRY_RUN="${2:-}"
MODEL_ROOT="${MODEL_ROOT:-/models}"
PORT="${PORT:-8000}"
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
  python3 - "$CONCURRENCY" "$MODEL_ROOT" "$MODEL" "$SWA_DRAFT" "$PORT" <<'PY'
import json, sys
from benchmark.tpu.launch_server import load_recipe
conc, root, model, draft, port = sys.argv[1:]
cfg = load_recipe("recipe/configs/muse_spark/agentic.py", int(conc))
cfg["server_args"].update(
    model_path=f"{root}/{model}",
    served_model_name=model,
    speculative_draft_model_path=draft,
    host="0.0.0.0",
    port=int(port),
)
print(json.dumps(cfg, indent=1))
PY
}

if [ "${DRY_RUN}" = --dry-run ]; then
  print_config
  exit 0
fi
write_swa_draft
CONFIG=/tmp/muse_agentx_c${CONCURRENCY}.json
print_config > "$CONFIG"
cat "$CONFIG"
exec python3 -m benchmark.tpu.launch_server --config "@${CONFIG}"
