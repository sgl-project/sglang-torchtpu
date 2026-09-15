"""Qwen3-Coder-480B under multihost P/D with TP resharding (issue #378).

Registers the Qwen3-Coder-480B row for the multihost P/D TP-resharding combo. Topology,
engine launch, probes and gsm8k live in pd_reshard_harness; this file exists
because the dashboard contract is one job to one test file, and the labels below
render its filter chips.

Run on hardware:

  CI_PD_ROLE=prefill python3 test/registered/tpu/e2e/test_tpu_qwen3_coder_480b_pd_multihost_kv_offload_resharding.py

Without CI_PD_ROLE it runs only the host-side configuration cases.
"""

from pd_reshard_harness import main

try:
    from sglang.test.ci.ci_register import register_tpu_ci
except ModuleNotFoundError:
    # Keep the file runnable in a source-only checkout; the harness applies the
    # same guard, and registration is a build-time marker parsed from source.
    def register_tpu_ci(*_args, **_kwargs):
        return None


# One invocation runs ONE model profile: two engines (weight load + compile),
# the resharding probes, and gsm8k.
register_tpu_ci(
    est_time=10800,
    suite="tpu-integration",
    labels=["multihost", "qwen3-coder-480b", "pd", "tp-reshard", "kv-offload"],
)


if __name__ == "__main__":
    main("qwen3_coder_480b")
