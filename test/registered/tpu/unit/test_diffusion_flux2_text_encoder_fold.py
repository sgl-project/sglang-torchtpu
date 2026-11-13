"""TPU CI entry for the FLUX.2 text encoder fold test: the Mistral3 sizes come up
from ``text_config`` so the encoder folds across the ranks instead of loading
whole on each.

Host-only checks, registered here because the CPU CI image does not install the
diffusion runtime.
"""

import sys

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=10, suite="stage-a-test-tpu-unit", chips=1)

import pytest

from sglang.multimodal_gen.test.unit.test_flux2_text_encoder_fold import (  # noqa: F401
    test_configs_without_text_config_are_unchanged,
    test_fold_policy_keeps_folding_across_eight_ranks,
    test_sizes_come_up_from_text_config,
)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
