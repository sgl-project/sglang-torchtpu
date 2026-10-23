# SPDX-License-Identifier: Apache-2.0
"""Explicit Q/B coverage; Q is tokens per request and T is the padded total."""

BATCHES = (1, 2, 4, 8, 16, 32, 64)
DECODE_QUERIES = tuple(range(1, 33))
PREFILL_QUERIES = tuple(2**i for i in range(14))
PREFILL_TOKEN_LIMIT = 8192


def cases(section="all"):
    result = []
    for mode, queries in (("decode", DECODE_QUERIES), ("prefill", PREFILL_QUERIES)):
        if section not in ("all", mode):
            continue
        for q in queries:
            for b in BATCHES:
                if mode == "prefill" and b * q > PREFILL_TOKEN_LIMIT:
                    continue
                result.append(
                    {
                        "mode": mode,
                        "batch": b,
                        "query_length": q,
                        # Fixed-width decode/verify buckets whole requests:
                        # compiled_forward_shim uses T = padded B * Q.
                        "tokens": b * q if mode == "decode" else max(64, b * q),
                    }
                )
    return result
