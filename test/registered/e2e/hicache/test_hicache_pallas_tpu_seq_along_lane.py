"""TPU KV offloading on the head-major SEQ_ALONG_LANE KV layout.

Same host round-trip as the rpa_v3 row, served by the batched RPA backend with
SGLANG_TPU_USE_SEQ_ALONG_LANE_KV=1 -- the only backend that reads that layout.
Its page puts the token axis on the lanes, so a host pool that assumes
token-major page geometry fails the round-trip.

Run on hardware:

  python3 test/registered/hicache/test_hicache_pallas_tpu_seq_along_lane.py
"""

from hicache_pallas_tpu_harness import main

from sglang.test.ci.ci_register import register_tpu_ci

# One TPU server launch + torch.compile warmup, then ~11 short generations.
register_tpu_ci(
    est_time=2400,
    suite="external-nightly-1host",
    nightly=True,
    labels=["singlehost", "qwen3-0.6b", "kv-offload", "seq-along-lane"],
)


if __name__ == "__main__":
    main("seq_along_lane")
