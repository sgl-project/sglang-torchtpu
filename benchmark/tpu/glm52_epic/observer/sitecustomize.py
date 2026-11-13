"""Load only when this benchmark directory is explicitly on PYTHONPATH."""

import os

if os.environ.get("GLM52_EPIC_OBSERVER_DIR"):
    from observer import install

    install()
