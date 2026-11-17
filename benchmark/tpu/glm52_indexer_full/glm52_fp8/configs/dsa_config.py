"""Static serving policy for exact DSA, resolved before JIT tracing.

Kernel callers remain free to select their own static parameters. Serving
configuration and automatic execution policy live here.
"""

from dataclasses import dataclass


def is_dsa_prefill(mode):
    return (
        mode.is_extend()
        and not mode.is_draft_extend()
        and not mode.is_target_verify()
        and not mode.is_mixed()
    )


def supports_fp8_attention(mode):
    return mode.is_decode() or mode.is_target_verify() or is_dsa_prefill(mode)


def indexer_query_block_size(mode, draft_tokens):
    return draft_tokens if mode.is_target_verify() or mode.is_draft_extend() else None


def pipeline_short_indexer_requests(*, query_block_size, max_queries_per_seq, max_kv, physical):
    # Keep the measured Q6 MXU/TopK layout; batch size is not a kernel constraint.
    return (
        query_block_size == 6
        and max_kv <= 131072
        and not physical
        and max_queries_per_seq is not None
        and max_queries_per_seq <= query_block_size
    )


def parse_tiles(value):
    if value is None:
        return None
    try:
        tiles = tuple(int(v) for v in (value.split(",") if isinstance(value, str) else value))
    except (TypeError, ValueError) as exc:
        raise ValueError("DSA tiles must be bq_sparse,bq,b_topk integers") from exc
    if (
        len(tiles) != 3
        or min(tiles) <= 0
        or tiles[0] % tiles[1]
        or tiles[2] % 128
        or 2048 % tiles[2]
    ):
        raise ValueError("DSA tiles must be compatible bq_sparse,bq,b_topk")
    return tiles


def exact_dsa_tiles(num_queries, *, fp8_q=False, override=None, partial=False):
    # Pair attention doubles Q24 to Q48. Native measurements favor two
    # gathers for FP8 verify and three for the longer BF16 draft kernel.
    if num_queries == 48 and partial:
        return (24, 8, 256) if fp8_q else (16, 16, 256)
    if num_queries == 48 and fp8_q:
        return (12, 12, 256)
    if num_queries >= 128 and override is not None:
        tiles = parse_tiles(override)
    else:
        tiles = {8: (4, 4, 256), 12: (6, 6, 256), 16: (8, 8, 256), 24: (12, 12, 256)}.get(
            num_queries
        )
        if tiles is None:
            tiles = (2, 1, 256) if num_queries < 128 else (256, 16, 256)
    if fp8_q and tiles[1] > 16:
        raise ValueError("FP8 attention requires 1 <= bq <= 16")
    return tiles


@dataclass(frozen=True)
class DsaExecutionPlan:
    fp8_q: bool
    fp8_pv: bool
    cache_read_mode: str
    pipeline: str
    tiles: tuple[int, int, int]


@dataclass(frozen=True)
class DsaConfig:
    precision: str = "bf16"
    extend_tiles: tuple[int, int, int] | None = None
    hlo_overlap: bool = True

    def __post_init__(self):
        if self.precision not in ("bf16", "fp8-qk", "fp8-qk-pv"):
            raise ValueError(f"Invalid DSA precision: {self.precision!r}")
        object.__setattr__(self, "extend_tiles", parse_tiles(self.extend_tiles))
        if self.extend_tiles is not None and self.precision != "bf16":
            exact_dsa_tiles(128, fp8_q=True, override=self.extend_tiles)

    def validate_cache_dtype(self, dtype):
        if self.precision != "bf16" and dtype != "fp8_e4m3":
            raise ValueError("FP8 attention requires an FP8 KV cache in E4M3 format")

    def resolve(self, num_queries, *, eligible, fp8_kv, is_prefill, partial=False):
        fp8_q = eligible and self.precision != "bf16"
        if fp8_q and not fp8_kv:
            raise ValueError("FP8 QK requires an FP8 KV cache")
        read_mode = (
            "expanded-bounded"
            if fp8_kv and not fp8_q and is_prefill and num_queries >= 128 and not partial
            else "packed"
        )
        # Disabling HLO overlap selects the established non-pass paths.
        # Partial attention needs FP32 output/LSE from the bounded packed path.
        if fp8_q or partial:
            pipeline = "split"
        elif self.hlo_overlap:
            pipeline = "overlap"
        else:
            pipeline = "split" if num_queries >= 128 else "pool"
        return DsaExecutionPlan(
            fp8_q,
            fp8_q and self.precision == "fp8-qk-pv",
            read_mode,
            pipeline,
            exact_dsa_tiles(num_queries, fp8_q=fp8_q, override=self.extend_tiles, partial=partial),
        )
