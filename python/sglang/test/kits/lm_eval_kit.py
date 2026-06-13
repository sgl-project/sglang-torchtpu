"""
This module provides a mixin class for running lm-eval harness evaluations
against SGLang servers
"""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests
import yaml

from sglang.test.test_utils import dump_metric


@contextmanager
def scoped_env_vars(new_env: dict[str, str] | None):
    """Context manager to temporarily set environment variables."""
    if not new_env:
        yield
        return

    old_values = {}
    new_keys = []

    try:
        for key, value in new_env.items():
            if key in os.environ:
                old_values[key] = os.environ[key]
            else:
                new_keys.append(key)
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in old_values.items():
            os.environ[key] = value
        for key in new_keys:
            os.environ.pop(key, None)


class LMEvalMixin:
    """
    Mixin class for running lm-eval harness evaluations.
    """

    other_args: list[str] = []
    model_config_name: str = ""
    default_rtol: float = 0.08

    @staticmethod
    def _lm_eval_endpoint(base_url: str, api: str) -> str:
        suffix = "/v1/chat/completions" if api == "chat" else "/v1/completions"
        return base_url.rstrip("/") + suffix

    def test_lm_eval(self):
        """Run lm-eval evaluation and validate results."""
        # Flush cache before evaluation
        try:
            requests.get(self.base_url + "/flush_cache", timeout=30)
        except requests.RequestException:
            pass

        eval_config = yaml.safe_load(
            Path(self.model_config_name).read_text(encoding="utf-8")
        )
        results = self.launch_lm_eval(eval_config)

        rtol = eval_config.get("rtol", self.default_rtol)
        atol = eval_config.get("atol", 0.0)

        success = True
        for task in eval_config["tasks"]:
            for metric in task["metrics"]:
                ground_truth = metric["value"]
                measured_value = results["results"][task["name"]][metric["name"]]
                floor = ground_truth * (1 - rtol) - atol
                print(
                    f"{task['name']} | {metric['name']}: "
                    f"ground_truth={ground_truth:.3f} | "
                    f"measured={measured_value:.3f} | floor={floor:.3f} | "
                    f"rtol={rtol} | atol={atol}"
                )
                dump_metric(
                    f"{task['name']}_{metric['name']}",
                    measured_value,
                    labels={
                        "model": eval_config.get("model_name", ""),
                        "eval": "lm-eval",
                        "task": task["name"],
                    },
                )
                success = success and measured_value >= floor

        self.assertTrue(success, f"lm-eval validation failed")

    def launch_lm_eval(self, eval_config: dict[str, Any]) -> dict:
        """
        Args:
            eval_config: Configuration dictionary with model and task settings
        """
        import lm_eval

        batch_size = eval_config.get("batch_size", "auto")
        backend = eval_config.get("backend", "local-completions")
        num_concurrent = eval_config.get("num_concurrent", 1)
        api = eval_config.get(
            "api", "chat" if "chat" in backend else "completion"
        )

        model_args = {
            "model": eval_config["model_name"],
            "base_url": self._lm_eval_endpoint(self.base_url, api),
            "num_concurrent": num_concurrent,
        }
        model_args.update(eval_config.get("model_args", {}))

        env_vars = eval_config.get("env_vars", None)
        with scoped_env_vars(env_vars):
            results = lm_eval.simple_evaluate(
                model=backend,
                model_args=model_args,
                tasks=[task["name"] for task in eval_config["tasks"]],
                num_fewshot=eval_config.get("num_fewshot", 0),
                limit=eval_config.get("limit", None),
                apply_chat_template=eval_config.get("apply_chat_template", False),
                fewshot_as_multiturn=eval_config.get("fewshot_as_multiturn", False),
                gen_kwargs=eval_config.get("gen_kwargs"),
                batch_size=batch_size,
            )

        return results
