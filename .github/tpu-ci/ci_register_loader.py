"""Load CI registration metadata without importing the SGLang runtime."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_ci_register() -> ModuleType:
    path = Path(__file__).resolve().parents[2] / "python/sglang/test/ci/ci_register.py"
    # NOTE: importing through sglang runs its runtime initialization; discovery
    # runs on CI hosts with only the standard library available.
    spec = importlib.util.spec_from_file_location("tpu_ci_register", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create a module loader for {path}")

    module = importlib.util.module_from_spec(spec)
    # NOTE: dataclasses with postponed annotations look up their defining module
    # during execution, so it must already be registered in sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[spec.name]
        raise
    return module


ci_register = _load_ci_register()
