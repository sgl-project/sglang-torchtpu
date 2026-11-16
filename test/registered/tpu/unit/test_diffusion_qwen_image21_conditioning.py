"""TPU CI entry for Qwen-Image-2.1's host-side prompt conditioning tests."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import sys

import pytest

# The core directory's autouse fixture that sets default server args.
from sglang.multimodal_gen.test.unit.conftest import (  # noqa: F401
    default_global_server_args,
)
from sglang.multimodal_gen.test.unit.test_qwen_image21 import (  # noqa: F401
    test_prompt_cache_skips_processor_and_weight_preparation,
    test_prompt_conditioning_uses_training_template_and_pre_norm,
)

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-p", "no:cacheprovider"]))
