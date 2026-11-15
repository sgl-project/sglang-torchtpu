"""Load only when this benchmark directory is explicitly on PYTHONPATH."""

import os

if os.environ.get("GLM52_EAGER_IR_DIR"):
    # Must precede JAX/TorchTPU initialization in EACH spawned interpreter.
    # Python normally ignores sitecustomize exceptions: force explicit failure
    # instead of silently running a diagnostic without compiler artifacts.
    try:
        # Scoped mode avoids loading-time dump flood; initialization/weight
        # loading are intentionally outside capture. No silent fallback global.
        from benchmark.tpu.glm52_epic.scoped_eager_ir import install as install_ir

        install_ir(os.environ["GLM52_EAGER_IR_DIR"])
    except Exception as error:
        import sys

        print(f"GLM52_IR_SETUP_FAILED: {error!r}", file=sys.stderr, flush=True)
        raise SystemExit(78)

if os.environ.get("GLM52_STARTUP_STACK_DIR"):
    from startup_stacks import install as install_startup_stacks

    install_startup_stacks()

if os.environ.get("GLM52_EPIC_OBSERVER_DIR"):
    from observer import install

    install()
