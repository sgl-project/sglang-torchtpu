"""Source-tree entry point for the packaged checkpoint prefetch helper."""

import runpy
from pathlib import Path

# Launch planning runs before SGLang's runtime imports or environment setup.
_helper = runpy.run_path(
    str(
        Path(__file__).absolute().parents[2]
        / "python/sglang/test/ci/checkpoint_prefetch.py"
    ),
    run_name=__name__,
)
configure_checkpoint_prefetch = _helper["configure_checkpoint_prefetch"]
