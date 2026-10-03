"""Reusable single-server fixture for TPU functional integration tests.

Each feature test subclasses :class:`SGLangTPUServerTestSuite`, supplies its
complete ``server_args`` dictionary, and writes ordinary unittest methods. One
``sgl.Engine`` is created for the subclass and reused by all of its tests::

    class TestSpecDecoding(SGLangTPUServerTestSuite):
        server_args = {
            "model_path": "Qwen/Qwen3-1.7B",
            "speculative_algorithm": "EAGLE3",
            "speculative_draft_model_path": "draft-model",
        }

        def test_generates_with_speculation(self):
            output = self.engine.generate("Hello", {"max_new_tokens": 8})
            self.assertGreater(output["meta_info"]["spec_verify_ct"], 0)
"""

import os
import statistics
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar


def resolve_tpu_model_path(model_id: str) -> str:
    """Prefer an explicit root or the standard CI mount, then Hugging Face."""

    configured_root = os.environ.get("SGLANG_TPU_WEIGHTS_ROOT")
    root = Path(configured_root or "/weights")
    if not root.exists():
        if configured_root:
            raise FileNotFoundError(
                "configured TPU weights root does not exist: "
                f"SGLANG_TPU_WEIGHTS_ROOT={configured_root!r}"
            )
        return model_id

    candidates = (root / model_id, root / model_id.rsplit("/", 1)[-1])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if not configured_root:
        return model_id
    raise FileNotFoundError(
        f"{model_id!r} was not found under TPU weights root {str(root)!r}"
    )


class SGLangTPUServerTestSuite(unittest.TestCase):
    """Bring up and sanity-check one local TPU engine per test subclass.

    Distributed and prefill/decode tests use separate fixtures instead of adding
    orchestration responsibilities to this single-server base.
    """

    server_args: ClassVar[Mapping[str, Any]] = {}
    server_env: ClassVar[Mapping[str, str]] = {}
    engine: ClassVar[Any | None] = None
    _original_server_env: ClassVar[dict[str, str | None]] = {}

    @classmethod
    def get_server_args(cls) -> dict[str, Any]:
        """Return a copy so engine setup cannot mutate the class declaration."""

        return dict(cls.server_args)

    @classmethod
    def _apply_server_env(cls) -> None:
        cls._original_server_env = {
            name: os.environ.get(name) for name in cls.server_env
        }
        os.environ.update(cls.server_env)

    @classmethod
    def _restore_server_env(cls) -> None:
        for name, value in cls._original_server_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        cls._original_server_env = {}

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls._apply_server_env()
        os.environ["SGLANG_PLATFORM"] = "tpu"

        # A single-host test may run in a pod carrying multihost TPU variables.
        # Remove them before libtpu initializes so it does not wait for peers.
        for name in (
            "TPU_WORKER_ID",
            "TPU_WORKER_HOSTNAMES",
            "TPU_PROCESS_ADDRESSES",
            "TPU_PROCESS_PORT",
        ):
            os.environ.pop(name, None)

        try:
            import sglang as sgl

            args = {
                "device": "tpu",
                "disable_radix_cache": False,
                **cls.get_server_args(),
            }
            cls.engine = sgl.Engine(**args)
            cls.verify_prefix_cache_reuse()
        except Exception:
            # unittest does not call tearDownClass when setUpClass fails.
            try:
                if cls.engine is not None:
                    cls.engine.shutdown()
            finally:
                cls.engine = None
                cls._restore_server_env()
            raise

    @classmethod
    def verify_prefix_cache_reuse(cls) -> None:
        """Require every integration server to reuse a basic cached prefix."""

        prompt = (
            "Continue this integer sequence, comma separated:\n"
            + ", ".join(str(i) for i in range(1, 401))
            + ","
        )
        try:
            cls.assert_prefix_cache_reuse(
                prompt,
                sampling_params={"temperature": 0.0, "max_new_tokens": 1},
            )
        finally:
            # Feature tests must not inherit state from the bring-up check.
            cls.engine.flush_cache()

    @classmethod
    def generate(cls, prompt, sampling_params=None):
        """Generate with deterministic integration-test defaults."""

        params = {"temperature": 0.0, "max_new_tokens": 64}
        if sampling_params:
            params.update(sampling_params)
        return cls.engine.generate(prompt, params)

    @classmethod
    def assert_prefix_cache_reuse(
        cls, prompt, *, sampling_params=None, min_cached_tokens=1
    ):
        """Populate a prefix, repeat it, and return the cold and warm outputs."""

        cls.engine.flush_cache()
        cold = cls.generate(prompt, sampling_params)
        warm = cls.generate(prompt, sampling_params)
        cached_tokens = warm["meta_info"].get("cached_tokens", 0)
        if cached_tokens < min_cached_tokens:
            raise AssertionError(
                "prefix-cache reuse failed: the repeated request reported "
                f"cached_tokens={cached_tokens}, expected >= {min_cached_tokens}"
            )
        return cold, warm

    @classmethod
    def assert_spec_accept_length(cls, outputs, *, min_accept_length=1.0):
        """Verify a series of spec responses and return mean acceptance length."""

        accept_lengths = []
        for index, output in enumerate(outputs):
            meta = output["meta_info"]
            verify_count = meta.get("spec_verify_ct", 0)
            if verify_count <= 0:
                continue

            completion_tokens = meta.get("completion_tokens")
            if completion_tokens is None:
                raise AssertionError(
                    f"response {index} did not report completion_tokens"
                )
            accept_lengths.append(completion_tokens / verify_count)

        if not accept_lengths:
            raise AssertionError("no spec verifications were recorded")

        mean_accept_length = statistics.fmean(accept_lengths)
        if mean_accept_length <= min_accept_length:
            raise AssertionError(
                f"mean accept_length={mean_accept_length:.4f}, "
                f"expected > {min_accept_length:.4f} "
                f"over {len(accept_lengths)} responses"
            )
        return mean_accept_length

    @classmethod
    def verify_spec_decoding(
        cls, prompt, *, sampling_params=None, min_accept_length=1.0
    ):
        """Generate once and return its speculative acceptance length."""

        output = cls.generate(prompt, sampling_params)
        accept_length = cls.assert_spec_accept_length(
            [output],
            min_accept_length=min_accept_length,
        )
        return output, accept_length

    @classmethod
    def verify_spec_decoding_prompts(
        cls,
        prompts,
        *,
        batch_size,
        sampling_params=None,
        min_accept_length=1.0,
    ):
        """Generate prompts in batches and verify mean acceptance length."""

        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        params = {"temperature": 0.0, "max_new_tokens": 64}
        if sampling_params:
            params.update(sampling_params)

        # Match the feature-combo concurrent-wave test: a prompt list and one
        # sampling-parameter dictionary per request form a real engine batch.
        outputs = []
        prompts = list(prompts)
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            batch_outputs = cls.engine.generate(
                batch,
                [dict(params) for _ in batch],
            )
            if len(batch_outputs) != len(batch):
                raise AssertionError(
                    f"engine returned {len(batch_outputs)} responses for "
                    f"a batch of {len(batch)} prompts"
                )
            outputs.extend(batch_outputs)
        mean_accept_length = cls.assert_spec_accept_length(
            outputs,
            min_accept_length=min_accept_length,
        )
        return outputs, mean_accept_length

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            if cls.engine is not None:
                cls.engine.shutdown()
        finally:
            cls.engine = None
            cls._restore_server_env()
            super().tearDownClass()
