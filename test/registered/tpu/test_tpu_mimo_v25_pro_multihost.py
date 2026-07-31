"""MiMo-V2.5-Pro (1T MoE, blockwise-fp8) multi-host e2e lm-eval (live-server mode).

The JobSet -- not this file -- brings the server up, so the model config supplies an
ExternalServerLauncher: the shared runner waits for /health_generate and never
launches or kills anything.

Its MULTIHOST_LAUNCH forms its own launch group: recipe/mimo_v25_pro_mh.sh on 2 hosts.
The recipe mirrors the E2E-validated DP-attention config from
recipe/mimo/launch_mimo_25_pro_multihost.sh (PR #155): tp=16, ep=16, dp=2 ->
attention_tp=8, mem_fraction 0.7, max_running 8 -- with prefill raised to 8192.
Weights (~1.0TB fp8) come from the shared ReadOnlyMany Hyperdisk PVC
model-weights-rox (see .github/tpu-ci/pvc-model-weights-rox.yaml.tmpl).
The CI runs this file on rank-0 against the server the JobSet brought up; skips
unless a live server is provided.

First-run watch items: the ~1TB weight load + tp=16 compile vs the JobSet's rank-0
health wait (3000s) / activeDeadlineSeconds (5400s), and DP-attention decode
precompile (a known attn_tp=4 all-gather deadlock exists on main; this config is
attn_tp=8, validated in PR #155).

Asserts real accuracy: lm-eval mmlu_llama 5-shot (limit=100) against
test/lm_eval_configs/MiMo-V2.5-Pro-TPU.yaml -- ground truth is the model card's
official 5-shot MMLU (89.4) with the agreed 3-point absolute tolerance, and the YAML
stays the only place those numbers live.
"""

import os
import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import DEFAULT_URL_FOR_TEST
from sglang.test.tpu import (
    ExternalServerLauncher,
    TpuModelLaunchSettings,
    lm_eval_runner,
)

register_tpu_ci(est_time=5400, suite="nightly-tpu-multihost", nightly=True)

# Own launch group: MiMo-V2.5-Pro DP-attention recipe + the MiMo ROX weights volume.
MULTIHOST_LAUNCH = {
    "hosts": 2,
    "launch_script": "recipe/mimo_v25_pro_mh.sh",
    "weights_pvc": "model-weights-rox",
}

MODEL_PATH = "XiaomiMiMo/MiMo-V2.5-Pro"
LM_EVAL_CONFIG = "test/lm_eval_configs/MiMo-V2.5-Pro-TPU.yaml"
BASE_URL = os.environ.get("BASE_URL", DEFAULT_URL_FOR_TEST)
# The JobSet's rank-0 already gated on /health_generate; this recheck keeps the file
# runnable by hand against any live server (BASE_URL=... HEALTH_TIMEOUT=...).
HEALTH_TIMEOUT = float(os.environ.get("HEALTH_TIMEOUT", "600"))


@unittest.skipUnless(
    os.environ.get("SGLANG_TPU_USE_LIVE_SERVER") == "1",
    "multi-host live-server not provided (the JobSet launches the server)",
)
class TestTpuMiMoV25ProLMEval(unittest.TestCase):
    """lm-eval accuracy against the 2-host DP-attention MiMo-V2.5-Pro server."""

    def test_mimo_v25_pro_accuracy(self):
        model = TpuModelLaunchSettings(
            MODEL_PATH,
            launcher=ExternalServerLauncher(
                health_timeout=HEALTH_TIMEOUT, tag="mimo-e2e"
            ),
            # tp=16 across 2 hosts, but the JobSet owns the launch, so this is
            # recorded for reporting only -- no flags are derived from it here.
            tp_size=16,
            variant="2host-TP16+EP16+DP2",
        )

        run_combined_tests(
            models=[model],
            test_name="MiMo-V2.5-Pro-multihost",
            base_url=BASE_URL,
            accuracy_params=AccuracyTestParams(
                dataset="mmlu_llama",
                label="lm_eval",
                runner=lm_eval_runner(LM_EVAL_CONFIG),
            ),
        )


if __name__ == "__main__":
    unittest.main()
