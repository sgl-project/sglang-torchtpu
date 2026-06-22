#!/usr/bin/env python3
"""Run lm-eval against a live SGLang server and check configured baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _endpoint(base_url: str, api: str) -> str:
    suffix = "/v1/chat/completions" if api == "chat" else "/v1/completions"
    return base_url.rstrip("/") + suffix


def _gen_kwargs(config: dict[str, Any]) -> dict[str, Any] | None:
    gen_kwargs = config.get("gen_kwargs")
    if isinstance(gen_kwargs, str):
        return json.loads(gen_kwargs)
    return gen_kwargs


def _check_baselines(results: dict[str, Any], config: dict[str, Any]) -> bool:
    atol = float(config.get("atol", 0.0))
    rtol = float(config.get("rtol", 0.0))

    ok = True
    print("| task | metric | expected | measured | status |")
    print("| ---- | ------ | -------- | -------- | ------ |")
    for task in config["tasks"]:
        task_name = task["name"]
        for metric in task["metrics"]:
            metric_name = metric["name"]
            expected = float(metric["value"])
            measured = float(results["results"][task_name][metric_name])
            floor = expected * (1 - rtol) - atol
            passed = measured >= floor
            ok = ok and passed
            status = "PASS" if passed else "FAIL"
            print(
                f"| {task_name} | {metric_name} | {expected:.4f} | "
                f"{measured:.4f} | {status} |"
            )

    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-baseline-check", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    model_name = args.model or config["model_name"]
    task_names = [task["name"] for task in config["tasks"]]
    api = config.get("api", "completion")

    import lm_eval

    model_args = {
        "model": model_name,
        "base_url": _endpoint(args.base_url, api),
        "num_concurrent": config.get("num_concurrent", 1),
    }
    model_args.update(config.get("model_args", {}))

    eval_kwargs = {
        "model": config.get("backend", "local-completions"),
        "model_args": model_args,
        "tasks": task_names,
        "limit": config.get("limit"),
        "apply_chat_template": config.get("apply_chat_template", False),
        "fewshot_as_multiturn": config.get("fewshot_as_multiturn", False),
        "gen_kwargs": _gen_kwargs(config),
        "batch_size": config.get("batch_size", "auto"),
    }
    if "num_fewshot" in config:
        eval_kwargs["num_fewshot"] = config["num_fewshot"]

    results = lm_eval.simple_evaluate(**eval_kwargs)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = args.output_dir / "lm_eval_results.json"
        with result_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"wrote {result_path}")

    if args.skip_baseline_check:
        return 0

    return 0 if _check_baselines(results, config) else 1


if __name__ == "__main__":
    raise SystemExit(main())
