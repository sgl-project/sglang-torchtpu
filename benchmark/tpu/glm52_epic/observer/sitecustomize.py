"""Load only when this benchmark directory is explicitly on PYTHONPATH."""

import os

if os.environ.get("GLM52_STARTUP_STACK_DIR"):
    from startup_stacks import install as install_startup_stacks

    install_startup_stacks()

if os.environ.get("GLM52_EPIC_OBSERVER_DIR"):
    from observer import install

    install()
