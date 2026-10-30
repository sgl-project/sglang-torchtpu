"""TPU KV offloading on the token-major RPA v3 backend.

The sequence, prompts and assertions live in hicache_pallas_tpu_harness; CI
runs one job per registered test file, so each profile gets its own file.

Run on hardware:

  python3 test/registered/hicache/test_hicache_pallas_tpu.py
"""

from hicache_pallas_tpu_harness import main

from sglang.test.ci.ci_register import register_tpu_ci

# One TPU server launch + torch.compile warmup, then ~11 short generations.
# feature-validation discovers single-host rows by this suite and nightly=True.
register_tpu_ci(
    est_time=2400,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "qwen3-0.6b", "kv-offload", "rpa-v3"],
)


if __name__ == "__main__":
    main("rpa_v3")
