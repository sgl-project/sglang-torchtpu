"""Apply a Python accuracy recipe to main's existing shell TPU launcher.

Host discovery, compile caches, and process setup remain in launch_server.sh.
This adapter only selects the model's launch environment.
"""

import argparse
import importlib
import json
import os
import re
from pathlib import Path


def recipe_env(name, inherited):
    if not re.fullmatch(r"[a-z0-9_]+", name):
        raise ValueError(f"Invalid accuracy recipe name: {name}")
    settings = importlib.import_module(f"benchmark.tpu.recipe.configs.{name}").ENV
    if any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in settings.items()
    ):
        raise ValueError(
            "Accuracy recipe environment must contain string keys and values"
        )
    env = {**inherited, **settings}
    # CI may resolve a model ID to a local weights mount before launching.
    if inherited.get("MODEL"):
        env["MODEL"] = inherited["MODEL"]
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    env = recipe_env(args.config, os.environ)
    if args.dry_run:
        settings = importlib.import_module(
            f"benchmark.tpu.recipe.configs.{args.config}"
        ).ENV
        print(json.dumps({key: env[key] for key in settings}, indent=2))
        return
    launcher = Path(__file__).resolve().parents[2] / "launch_server.sh"
    os.execvpe("bash", ["bash", str(launcher)], env)


if __name__ == "__main__":
    main()
