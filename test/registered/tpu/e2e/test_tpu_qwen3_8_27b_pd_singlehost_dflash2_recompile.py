"""Qwen3.8-27B DFlash2 P/D gate: the experimental spec worker serves through
the Raiden handoff with no XLA compile on prefill or decode.

Each engine owns one v7x host; prefill runs the router and drives the traffic.
"""

import os
import unittest

from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.utils.server_test_suite import resolve_tpu_model_path
from sglang_tpu.testing import pd_spec_recompile as base

register_tpu_ci(
    est_time=3600,
    suite="tpu-integration",
    nightly=True,
    labels=["singlehost", "qwen3.8-27b", "pd", "spec", "compile"],
)

CONFIG_MODULE = "recipe/configs/qwen3_8_27b/pd_singlehost_dflash2.py"
MODEL_PATH = resolve_tpu_model_path("Qwen/Qwen3.8-27B")
# Read by the recipe module, which the launcher imports in this process.
os.environ.setdefault(
    "DRAFT_MODEL", resolve_tpu_model_path("z-lab/Qwen3.8-27B-DFlash2")
)

# The run name cannot contain the role names: peer discovery swaps them.
RUN_NAME = "pd-sh-q38-dflash2-rc"


class TestQwen38Dflash2PdServesWithoutRecompile(base.PDSpecRecompileTestCase):
    config_module = CONFIG_MODULE
    served_algorithm = "DFLASH"
    tag = "pd-sh-q38-dflash2-rc"
    log_path = "/tmp/pd_qwen3_8_27b_dflash2_prefill.log"
    decode_log_path = "/tmp/pd_qwen3_8_27b_dflash2_decode.log"
    runtime_envs = {**base.PDSpecRecompileTestCase.runtime_envs, "MODEL": MODEL_PATH}


class TestQwen38Dflash2PdGateConfiguration(base.PDSpecGateConfigurationTestCase):
    config_module = CONFIG_MODULE
    expected_algorithm = "DFLASH"


if __name__ == "__main__":
    mode, argv = base.parse_mode()
    if mode == "integration_dry_run":
        print(base.render_pd_jobset(__file__, RUN_NAME))
    else:
        test_case = (
            TestQwen38Dflash2PdServesWithoutRecompile
            if mode == "integration"
            else TestQwen38Dflash2PdGateConfiguration
        )
        unittest.main(defaultTest=test_case.__name__, argv=argv, verbosity=2)
