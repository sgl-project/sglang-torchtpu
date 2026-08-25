"""Basic EAGLE3 speculative-decoding tests on one TPU host.

Run all model configurations with:

    SGLANG_PLATFORM=tpu SPEED_BENCH_NUM_PROMPTS=64 \
        python test/registered/tpu/integration/test_spec_decoding.py

Set ``SPEED_BENCH_NUM_PROMPTS=-1`` to run the entire qualitative split.

The SPEED-Bench dataset is downloaded from Hugging Face unless it is already
cached. Set ``HF_HUB_OFFLINE=1`` when running with a populated local cache.
"""

import os
import unittest
from typing import ClassVar

from sglang.srt.utils import is_tpu
from sglang.test.ci.ci_register import register_tpu_ci
from sglang.test.tpu.server_test_suite import (
    SGLangTPUServerTestSuite,
    resolve_tpu_model_path,
)

register_tpu_ci(
    est_time=21600,
    suite="nightly-tpu-integration",
    nightly=True,
)

SPEC_TEST_PROMPT = (
    "Continue this integer sequence, comma separated:\n"
    + ", ".join(str(i) for i in range(1, 121))
    + ","
)
SPEED_BENCH_NUM_PROMPTS = int(os.environ.get("SPEED_BENCH_NUM_PROMPTS", "64"))
SPEED_BENCH_BATCH_SIZE = 16


def load_speed_bench_prompts(*, category=None):
    """Load a deterministic qualitative subset, optionally by task category."""

    if SPEED_BENCH_NUM_PROMPTS == 0 or SPEED_BENCH_NUM_PROMPTS < -1:
        raise ValueError(
            "SPEED_BENCH_NUM_PROMPTS must be positive or -1 for all prompts, "
            f"got {SPEED_BENCH_NUM_PROMPTS}"
        )

    from datasets import load_dataset

    dataset = load_dataset("nvidia/SPEED-Bench", "qualitative", split="test")
    if category is not None:
        dataset = dataset.filter(lambda row: row["category"] == category)
        if not len(dataset):
            raise ValueError(f"no SPEED-Bench prompts found for category={category!r}")

    if SPEED_BENCH_NUM_PROMPTS != -1 and SPEED_BENCH_NUM_PROMPTS > len(dataset):
        raise ValueError(
            f"requested {SPEED_BENCH_NUM_PROMPTS} SPEED-Bench prompts, but only "
            f"{len(dataset)} are available"
            + (f" for category={category!r}" if category is not None else "")
        )

    prompts = []
    for row in dataset.shuffle(seed=0):
        turns = row.get("turns") or []
        if turns:
            prompts.append(turns[0])
        if SPEED_BENCH_NUM_PROMPTS != -1 and len(prompts) == SPEED_BENCH_NUM_PROMPTS:
            break

    if SPEED_BENCH_NUM_PROMPTS != -1 and len(prompts) != SPEED_BENCH_NUM_PROMPTS:
        raise ValueError(
            f"requested {SPEED_BENCH_NUM_PROMPTS} SPEED-Bench prompts, but only "
            f"{len(prompts)} rows contain prompts"
        )
    return prompts


class HybridDPPrefixCacheMixin:
    @classmethod
    def verify_prefix_cache_reuse(cls) -> None:
        """Keep the cold and warm probes on the same DP-local radix tree."""
        prompt = (
            "Continue this integer sequence, comma separated:\n"
            + ", ".join(str(i) for i in range(1, 401))
            + ","
        )
        params = {"temperature": 0.0, "max_new_tokens": 1}
        try:
            cls.engine.flush_cache()
            cls.engine.generate(prompt, params, routed_dp_rank=0)
            warm = cls.engine.generate(prompt, params, routed_dp_rank=0)
            if warm["meta_info"].get("cached_tokens", 0) < 1:
                raise AssertionError("DP rank 0 did not reuse the fixture prefix")
        finally:
            cls.engine.flush_cache()


@unittest.skipUnless(is_tpu(), "TPU required")
class TestGPTOss120BSpecDecoding(SGLangTPUServerTestSuite):
    expected_accept_length = 1.0
    # v7x reference: 1.5808 over 64 prompts at bs=16; gate at 90%.
    expected_non_greedy_accept_length = 1.42
    # v7x reference: 1.7898 over 64 prompts at bs=16; gate at 90%.
    expected_speed_bench_accept_length = 1.61

    server_env: ClassVar[dict[str, str]] = {
        # GPT-OSS EXTEND has a symbolic token dimension that the SparseCore
        # ragged-gather path cannot lower.
        "SGLANG_TPU_FUSED_MOE_SC_GATHER": "0",
    }
    server_args: ClassVar[dict[str, object]] = {
        "model_path": resolve_tpu_model_path("openai/gpt-oss-120b"),
        "dtype": "bfloat16",
        "tp_size": 8,
        "ep_size": 8,
        "page_size": 64,
        # The checkpoint declares MXFP4 itself; do not force quantization.
        "kv_cache_dtype": "fp8_e4m3",
        # GPT-OSS needs attention sinks and hybrid-SWA index mapping.
        "attention_backend": "pallas_tpu_rpa_v3",
        "tpu_compile_tracer": "direct",
        "max_total_tokens": 8192,
        "mem_fraction_static": 0.6,
        "max_prefill_tokens": 4096,
        "max_running_requests": 16,
        "tpu_compile_bs": [1, 2, 4],
        "tpu_compile_tokens": [512, 2048, 4096],
        "enable_torch_compile": True,
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": resolve_tpu_model_path(
            "lmsys/EAGLE3-gpt-oss-120b-bf16"
        ),
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }

    def test_accept_length_single_prompt(self):
        output, accept_length = self.verify_spec_decoding(
            SPEC_TEST_PROMPT,
            min_accept_length=self.expected_accept_length,
        )
        meta = output["meta_info"]
        print(
            "single-prompt spec decoding: "
            f"model={self.server_args['model_path']}, "
            f"prompt_tokens={meta.get('prompt_tokens')}, "
            f"completion_tokens={meta.get('completion_tokens')}, "
            f"spec_verify_ct={meta.get('spec_verify_ct')}, "
            f"accept_length={accept_length:.4f}, "
            f"expected>{self.expected_accept_length:.4f}"
        )

    def test_speed_bench_accept_length(self):
        prompts = load_speed_bench_prompts()
        outputs, accept_length = self.verify_spec_decoding_prompts(
            prompts,
            batch_size=SPEED_BENCH_BATCH_SIZE,
            sampling_params={"max_new_tokens": 512},
            min_accept_length=self.expected_speed_bench_accept_length,
        )
        verified_responses = sum(
            output["meta_info"].get("spec_verify_ct", 0) > 0 for output in outputs
        )
        print(
            f"SPEED-Bench mean accept_length={accept_length:.4f} "
            f"over {verified_responses}/{len(prompts)} verified prompts "
            f"at batch_size={SPEED_BENCH_BATCH_SIZE}"
        )

    def test_speed_bench_non_greedy_accept_length(self):
        prompts = load_speed_bench_prompts()
        outputs, accept_length = self.verify_spec_decoding_prompts(
            prompts,
            batch_size=SPEED_BENCH_BATCH_SIZE,
            sampling_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "sampling_seed": 0,
                "max_new_tokens": 512,
            },
            min_accept_length=self.expected_non_greedy_accept_length,
        )
        verified_responses = sum(
            output["meta_info"].get("spec_verify_ct", 0) > 0 for output in outputs
        )
        print(
            "SPEED-Bench non-greedy "
            f"temperature=0.7 top_p=0.9 mean accept_length={accept_length:.4f} "
            f"over {verified_responses}/{len(prompts)} verified prompts "
            f"at batch_size={SPEED_BENCH_BATCH_SIZE}"
        )


@unittest.skipUnless(is_tpu(), "TPU required")
class TestGPTOss120BHybridDPSpecDecoding(
    HybridDPPrefixCacheMixin, TestGPTOss120BSpecDecoding
):
    server_args: ClassVar[dict[str, object]] = {
        **TestGPTOss120BSpecDecoding.server_args,
        "dp_size": 2,
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
        "tpu_compile_bs": [1, 2, 4, 8],
    }


@unittest.skipUnless(is_tpu(), "TPU required")
class TestQwen3Coder480BSpecDecoding(SGLangTPUServerTestSuite):
    expected_accept_length = 1.0
    # v7x coding reference: 2.2075 over 64 prompts at bs=16; gate at 90%.
    expected_non_greedy_accept_length = 1.98
    # v7x coding reference: 2.2730 over 64 prompts at bs=16; gate at 90%.
    expected_speed_bench_accept_length = 2.04

    server_args: ClassVar[dict[str, object]] = {
        "model_path": resolve_tpu_model_path("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"),
        "dtype": "bfloat16",
        "tp_size": 8,
        "ep_size": 8,
        "page_size": 256,
        "quantization": "fp8",
        "kv_cache_dtype": "fp8_e4m3",
        "tpu_fp8_runtime": "channelwise",
        "max_total_tokens": 8192,
        "mem_fraction_static": 0.8,
        "max_prefill_tokens": 4096,
        "max_running_requests": 16,
        "tpu_compile_bs": [1, 2, 4],
        "tpu_compile_tokens": [512, 2048, 4096],
        "enable_torch_compile": True,
        "speculative_algorithm": "EAGLE3",
        "speculative_draft_model_path": resolve_tpu_model_path(
            "lmsys/" "SGLang-EAGLE3-Qwen3-Coder-480B-A35B-Instruct-SpecForge-EigenAI"
        ),
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    }

    def test_accept_length_single_prompt(self):
        output, accept_length = self.verify_spec_decoding(
            SPEC_TEST_PROMPT,
            min_accept_length=self.expected_accept_length,
        )
        meta = output["meta_info"]
        print(
            "single-prompt spec decoding: "
            f"model={self.server_args['model_path']}, "
            f"prompt_tokens={meta.get('prompt_tokens')}, "
            f"completion_tokens={meta.get('completion_tokens')}, "
            f"spec_verify_ct={meta.get('spec_verify_ct')}, "
            f"accept_length={accept_length:.4f}, "
            f"expected>{self.expected_accept_length:.4f}"
        )

    def test_speed_bench_accept_length(self):
        prompts = load_speed_bench_prompts(category="coding")
        outputs, accept_length = self.verify_spec_decoding_prompts(
            prompts,
            batch_size=SPEED_BENCH_BATCH_SIZE,
            sampling_params={"max_new_tokens": 512},
            min_accept_length=self.expected_speed_bench_accept_length,
        )
        verified_responses = sum(
            output["meta_info"].get("spec_verify_ct", 0) > 0 for output in outputs
        )
        print(
            f"SPEED-Bench category=coding mean accept_length={accept_length:.4f} "
            f"over {verified_responses}/{len(prompts)} verified prompts "
            f"at batch_size={SPEED_BENCH_BATCH_SIZE}"
        )

    def test_speed_bench_non_greedy_accept_length(self):
        prompts = load_speed_bench_prompts(category="coding")
        outputs, accept_length = self.verify_spec_decoding_prompts(
            prompts,
            batch_size=SPEED_BENCH_BATCH_SIZE,
            sampling_params={
                "temperature": 0.7,
                "top_p": 0.9,
                "sampling_seed": 0,
                "max_new_tokens": 512,
            },
            min_accept_length=self.expected_non_greedy_accept_length,
        )
        verified_responses = sum(
            output["meta_info"].get("spec_verify_ct", 0) > 0 for output in outputs
        )
        print(
            "SPEED-Bench category=coding non-greedy "
            f"temperature=0.7 top_p=0.9 mean accept_length={accept_length:.4f} "
            f"over {verified_responses}/{len(prompts)} verified prompts "
            f"at batch_size={SPEED_BENCH_BATCH_SIZE}"
        )


@unittest.skipUnless(is_tpu(), "TPU required")
class TestQwen3Coder480BHybridDPSpecDecoding(
    HybridDPPrefixCacheMixin, TestQwen3Coder480BSpecDecoding
):
    server_args: ClassVar[dict[str, object]] = {
        **TestQwen3Coder480BSpecDecoding.server_args,
        "dp_size": 2,
        "enable_dp_attention": True,
        "enable_dp_lm_head": True,
        "tpu_compile_bs": [1, 2, 4, 8],
    }


if __name__ == "__main__":
    unittest.main(verbosity=2)
