"""GPT-OSS-120B P/D throughput: a PP8 DFlash prefill feeding a TP8/EP8 decode on
the experimental spec worker (sglang_tpu/testing/pd_pp_spec_perf.py)."""

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
    labels=["singlehost", "gpt-oss-120b", "pd", "pp", "spec", "dflash", "perf"],
)

CONFIG_MODULE = "recipe/configs/gpt_oss_120b/pd_singlehost_pp_dflash.py"
# Read by the recipe module, which the launcher imports in this process; resolved
# only when unset, since resolving raises for a checkpoint missing from the root.
if "GPT_OSS_DRAFT_MODEL" not in os.environ:
    os.environ["GPT_OSS_DRAFT_MODEL"] = resolve_tpu_model_path(
        "z-lab/gpt-oss-120b-DFlash"
    )


class TestPdSingleHostPpDflashGptOss120bPerf(base.PDPipelineSpecPerfTestCase):
    config_module = CONFIG_MODULE
    algorithm = "DFLASH"
    log_path = "/tmp/pd_pp_dflash_prefill.log"
    decode_log_path = "/tmp/pd_pp_dflash_decode.log"
    tag = "pd-singlehost-pp-dflash"
    test_name = "gpt-oss-120b-pd-singlehost-pp-dflash-c64"
    metric_id = "perf-gpt-oss-120b-pd-singlehost-pp8-dflash-c64"
    display_name = "GPT-OSS 120B P/D PP8 prefill + TP8 decode DFlash (c64)"
    variant = "PD-1P1D-PP8+TP8EP8+Raiden+DFLASH-c64"


class TestPdSingleHostPpDflashGptOss120bConfiguration(
    base.PDPipelineSpecConfigurationTestCase
):
    config_module = CONFIG_MODULE
    algorithm = "DFLASH"
    perf_case = TestPdSingleHostPpDflashGptOss120bPerf


if __name__ == "__main__":
    mode, argv = parse_mode()
    test_case = (
        TestPdSingleHostPpDflashGptOss120bPerf
        if mode == "integration"
        else TestPdSingleHostPpDflashGptOss120bConfiguration
    )
    unittest.main(defaultTest=test_case.__name__, argv=argv, verbosity=2)
