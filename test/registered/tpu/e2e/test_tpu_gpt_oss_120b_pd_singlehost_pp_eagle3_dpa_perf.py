"""GPT-OSS-120B P/D throughput: a PP8 EAGLE3 prefill feeding a TP8/EP8 decode with
DP8 attention, on the experimental spec worker (sglang_tpu/testing/pd_pp_spec_perf.py)."""

import os
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path
from sglang_tpu.testing import pd_pp_spec_perf as base
from sglang_tpu.testing.pd_spec_recompile import parse_mode

register_tpu_ci(
    est_time=7200,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "gpt-oss-120b", "pd", "pp", "spec", "eagle3", "dpa", "perf"],
)

CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_pp_eagle3_dpa.py"
# Read by the recipe module, which the launcher imports in this process; resolved
# only when unset, since resolving raises for a checkpoint missing from the root.
if "GPT_OSS_DRAFT_MODEL" not in os.environ:
    os.environ["GPT_OSS_DRAFT_MODEL"] = resolve_tpu_model_path(
        "lmsys/EAGLE3-gpt-oss-120b-bf16"
    )


class TestPdSingleHostPpEagle3DpaGptOss120bPerf(base.PDPipelineSpecPerfTestCase):
    config_module = CONFIG_MODULE
    algorithm = "EAGLE3"
    log_path = "/tmp/pd_pp_eagle3_dpa_prefill.log"
    decode_log_path = "/tmp/pd_pp_eagle3_dpa_decode.log"
    tag = "pd-singlehost-pp-eagle3-dpa"
    test_name = "gpt-oss-120b-pd-singlehost-pp-eagle3-dpa8-c64"
    metric_id = "perf-gpt-oss-120b-pd-singlehost-pp8-eagle3-dpa8-c64"
    display_name = (
        "GPT-OSS 120B P/D PP8 prefill + TP8 DP8-attention decode EAGLE3 (c64)"
    )
    variant = "PD-1P1D-PP8+TP8EP8DP8+Raiden+EAGLE3-c64"


class TestPdSingleHostPpEagle3DpaGptOss120bConfiguration(
    base.PDPipelineSpecConfigurationTestCase
):
    config_module = CONFIG_MODULE
    algorithm = "EAGLE3"
    decode_dp = 8
    perf_case = TestPdSingleHostPpEagle3DpaGptOss120bPerf


if __name__ == "__main__":
    mode, argv = parse_mode()
    test_case = (
        TestPdSingleHostPpEagle3DpaGptOss120bPerf
        if mode == "integration"
        else TestPdSingleHostPpEagle3DpaGptOss120bConfiguration
    )
    unittest.main(defaultTest=test_case.__name__, argv=argv, verbosity=2)
