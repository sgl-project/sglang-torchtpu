# SPDX-License-Identifier: Apache-2.0
"""Dedicated Ghostfish Pallas fast path for RPA v3 single-token decode (RPAd).

Ghostfish LLO-optimized architecture (Wave 1-4):
1. Whole-batch VMEM ``BlockSpec`` for ``queries`` ``[B, 16, 128]``, ``new_kv``
   ``[B_pad32, 1, 4, 128]``, and ``out`` ``[B, 16, 128]`` + ``N``-slot static
   whole-page async DMA ring + epilogue-waited non-blocking 1-row KV-cache writes
   (``46.86 us`` non-compute at ``B=128`` vs ``121.62 us`` stock, ``83%`` of the
   ``38.7 us`` physical HBM floor).
2. VMEM boundary page sanitization + streaming ``VLD -> VPACK -> MATPREP`` MXU
   engines (``0`` elementwise K/V ``jnp.where`` barriers, ``0x7F`` FP8-NaN safe on
   free, tail, AND evicted SWA prefix slots):
   - ``stream0b`` (Default): Zero-branch 2-page VMEM sanitizer + monolithic ``[16, W]``
     ``QK`` -> 1-shot ``exp2`` softmax -> ``PV`` (only 2 MXU matmuls/request, 0
     ``@pl.when`` basic-block splits inside the sequence loop).
   - ``stream1p`` / ``stream2ld``: Conditional-tail-sanitize variants for A/B comparison.
   - ``clean1p`` / ``clean2k``: Chunked ``256``-key K + ``128``-key V fallback
     (used for multi-block ``> 16K`` Full Attention).
"""

import functools
import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.util import (
    align_to,
    cdiv,
    get_tpu_version,
)

_LOG2_E = math.log2(math.e)
_MASK_LOGIT = -1.0e30


@dataclass(frozen=True)
class DecodeFastConfig:
    num_slots: int
    chunk_sz: int
    loop_kind: str  # "stream0b" | "streamdyn" | "ring4k" | "stream2h" | "stream1p" | "stream2ld" | "clean1p" | "clean2k" | "swafast2" | "shared16" | "shared32" | "shared64"
    max_ring_pages: int
    coalesce_dma: bool = False
    in_vmem_pack2: bool = False
    lazy_slot_zero: bool = False
    pv_bf16: bool = False
    page_reduce: bool = False
    fp8_dots: bool = False
    hoist_new_kv: bool = False
    fine_gears: bool = False
    shared_group_size: int = 0
    shared_q_tile: int = 256
    shared_split_dense2: bool = False
    shared_fp8_qkpv: bool = False
    shared_fp8_pv_only: bool = False
    batch_full: bool = False
    dyn2k_unroll: bool = False
    u4_flat: bool = False
    gear_4k: bool = False
    fold_128: bool = False
    fold_col: bool = False
    no_mask0: bool = False
    swabatch_group_size: int = 1
    dense2_group: bool = False


def _uses_swabatch_group(
    cfg: DecodeFastConfig,
    *,
    one_block_per_seq: bool,
    decode_query_size: int,
    kv_pack_dense2: bool,
    probe: str | None,
    b_tokens: int,
    max_num_seqs: int,
) -> bool:
    return (
        cfg.swabatch_group_size > 1
        and cfg.loop_kind == "swabatch"
        and one_block_per_seq
        and decode_query_size == 1
        and (not kv_pack_dense2 or cfg.dense2_group)
        and not cfg.fp8_dots
        and probe is None
        and b_tokens >= 32
        and b_tokens % cfg.swabatch_group_size == 0
        and max_num_seqs == b_tokens
    )


def can_use_swabatch_group_runtime(
    cfg: DecodeFastConfig,
    *,
    one_block_per_seq: bool,
    decode_query_size: int,
    kv_pack_dense2: bool,
    probe: str | None,
    b_tokens: int,
    max_num_seqs: int,
    bkv_slot_pages: int,
    has_init_sv: bool,
) -> bool:
    """Single source of truth for group-vectorized ``swabatch`` across pre-DMA and compute."""
    return (
        _uses_swabatch_group(
            cfg,
            one_block_per_seq=one_block_per_seq,
            decode_query_size=decode_query_size,
            kv_pack_dense2=kv_pack_dense2,
            probe=probe,
            b_tokens=b_tokens,
            max_num_seqs=max_num_seqs,
        )
        and bkv_slot_pages >= cfg.swabatch_group_size * cfg.max_ring_pages
        and has_init_sv
    )


def _resolve_swabatch_group_flag(
    use_grp_vec: bool | None,
    *,
    cfg: DecodeFastConfig,
    one_block_per_seq: bool,
    decode_query_size: int,
    kv_pack_dense2: bool,
    probe: str | None,
    b_tokens: int,
    max_num_seqs: int,
    bkv_slot_pages: int,
    has_init_sv: bool,
) -> bool:
    """Validates explicit ``use_grp_vec`` against runtime invariants or computes it."""
    can_grp_runtime = can_use_swabatch_group_runtime(
        cfg,
        one_block_per_seq=one_block_per_seq,
        decode_query_size=decode_query_size,
        kv_pack_dense2=kv_pack_dense2,
        probe=probe,
        b_tokens=b_tokens,
        max_num_seqs=max_num_seqs,
        bkv_slot_pages=bkv_slot_pages,
        has_init_sv=has_init_sv,
    )
    if use_grp_vec is None:
        return can_grp_runtime
    if bool(use_grp_vec) and not can_grp_runtime:
        raise ValueError(
            "use_grp_vec=True requires sized bkv, init_sv, and valid swabatch_group_size"
        )
    return bool(use_grp_vec)


def get_shared_group_size(mode_str: str) -> int:
    raw = mode_str.strip().lower() if mode_str else ""
    if not raw:
        return 0
    if raw == "shared16_host":
        return 16
    toks = set(raw.replace(",", "_").replace("-", "_").split("_"))
    if "shared64" in toks:
        return 64
    if "shared32" in toks:
        return 32
    if "shared16" in toks or "shared" in toks:
        return 16
    if "shared8" in toks:
        return 8
    return 0


def _is_w8_enabled(mode_str: str) -> bool:
    from sglang_tpu.envs import envs

    w8_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_W8", None)
    if w8_env is not None and w8_env.get():
        return True
    raw = mode_str.strip().lower() if mode_str else ""
    toks = set(raw.replace(",", "_").replace("-", "_").split("_"))
    return bool(toks & {"w8", "swafast2", "w7auto2", "hoistkv", "g512"})


def _is_w7_enabled(mode_str: str) -> bool:
    from sglang_tpu.envs import envs

    if _is_w8_enabled(mode_str):
        return True
    w7_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_W7", None)
    if w7_env is not None and w7_env.get():
        return True
    raw = mode_str.strip().lower() if mode_str else ""
    toks = set(raw.replace(",", "_").replace("-", "_").split("_"))
    return bool(
        toks
        & {
            "w7",
            "w7auto",
            "swabatch",
            "swafast",
            "noz",
            "ring4k",
            "ring8k",
            "pps256",
            "pbf16",
            "pbf16swa",
            "pgred",
            "fp8dp",
            "c4096",
            "c5120",
            "c8192",
            "adapt",
            "verify",
        }
    )


def _is_v2_enabled(mode_str: str) -> bool:
    from sglang_tpu.envs import envs

    if _is_w7_enabled(mode_str):
        return True
    v2_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_V2", None)
    if v2_env is not None and v2_env.get():
        return True
    raw = mode_str.strip().lower() if mode_str else ""
    if raw in ("v2", "2", "streamdyn"):
        return True
    toks = set(raw.replace(",", "_").replace("-", "_").split("_"))
    return bool(toks & {"v2", "streamdyn", "stream2h", "coal", "ivpack2", "dyn2ku"})


def can_use_decode_fast(
    *,
    mode_str: str,
    q: jax.Array,
    kv_cache_fused: jax.Array,
    out_dtype,
    decode_query_size: int,
    custom_mask,
    attention_sink,
    soft_cap,
    xai_temperature_len,
    q_scale,
    actual_num_q_heads: int,
    actual_num_kv_heads: int,
    actual_head_dim: int,
    sliding_window: int | None = None,
    pages_per_seq: int = 0,
    pure_unpacked_decode: bool = False,
    kv_pack_dense2: bool = False,
    need_fused_post: bool = False,
    kv_pack_bd4: bool = False,
) -> bool:
    from sglang_tpu.envs import envs

    raw = mode_str.strip().lower() if mode_str else ""
    if not raw or raw in ("0", "false", "off", "none"):
        return False
    use_v2 = _is_v2_enabled(raw)
    use_w7 = _is_w7_enabled(raw)
    raw_toks = set(raw.replace(",", "_").replace("-", "_").split("_"))
    explicit_streamdyn = "streamdyn" in raw_toks
    explicit_dyn = "dyn2k" in raw_toks or "dyn4k" in raw_toks
    explicit_ring4k = "ring4k" in raw_toks or "ring8k" in raw_toks
    explicit_multiq = "multiq" in raw_toks
    explicit_verify = "verify" in raw_toks
    shared_grp = get_shared_group_size(raw)
    verify_env = getattr(envs, "SGLANG_TPU_RPA_V3_DECODE_FAST_VERIFY", None)
    use_verify_fast = (
        explicit_verify
        or need_fused_post
        or bool(verify_env is not None and verify_env.get())
    )
    max_full_pps = (
        4096
        if (explicit_ring4k or use_w7 or shared_grp > 0)
        else (160 if explicit_dyn else (144 if explicit_streamdyn else 128))
    )
    if (
        (
            raw in ("1", "true", "auto", "on", "v2", "2")
            or explicit_streamdyn
            or explicit_ring4k
            or use_w7
            or shared_grp > 0
        )
        and sliding_window is None
        and pages_per_seq > max_full_pps
    ):
        return False
    native_head = (actual_num_q_heads, actual_head_dim) == (8, 64) or (
        kv_pack_bd4 and (actual_num_q_heads, actual_head_dim) == (8, 128)
    )
    has_tp16_tok = "tp16" in raw_toks
    if (raw == "tp16" or has_tp16_tok or raw == "shared16_host" or kv_pack_bd4) and native_head:
        allow_tp16_full = (raw == "shared16_host") or (raw != "tp16") or kv_pack_bd4
        if (not pure_unpacked_decode and not kv_pack_bd4) or not (
            sliding_window in (2048, 2049)
            or (allow_tp16_full and sliding_window is None)
        ):
            return False
        head_geometry = (actual_num_q_heads, actual_head_dim)
    else:
        head_geometry = (16, 128)
    phys_page_rows = kv_cache_fused.shape[1]
    expected_phys_rows = (
        (32, 64, 128)
        if kv_pack_bd4
        else ((64, 128, 256) if kv_pack_dense2 else (128, 256, 512))
    )
    # Keep dq > 1 routed to VERIFY_RPA_FAST=1 unless multiq, verify, DECODE_FAST_VERIFY=1,
    # or fused post-RPA norm/gate (which requires RPAd_fast across dq in 2..8).
    dq_ok = decode_query_size == 1 or (
        (
            (explicit_multiq and decode_query_size in (2, 4))
            or (use_verify_fast and 2 <= decode_query_size <= 8)
        )
        and (
            head_geometry == (16, 128)
            or ((head_geometry == (8, 64) or kv_pack_bd4) and use_verify_fast)
        )
        and raw != "shared16_host"
    )
    return (
        dq_ok
        and custom_mask is None
        and attention_sink is None
        and soft_cap is None
        and xai_temperature_len is None
        and q_scale is None
        and q.dtype == jnp.bfloat16
        and kv_cache_fused.dtype == jnp.float8_e4m3fn
        and out_dtype == jnp.bfloat16
        and actual_num_kv_heads == 1
        and (actual_num_q_heads, actual_head_dim) == head_geometry
        and kv_cache_fused.shape[2] == 1
        and kv_cache_fused.shape[3] == 4
        and kv_cache_fused.shape[4] == 128
        and phys_page_rows in expected_phys_rows
    )


def parse_decode_fast_config(
    mode_str: str,
    *,
    sliding_window: int | None,
    pages_per_seq: int,
    page_size: int = 128,
    decode_query_size: int = 1,
    shared_group: int = 0,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
) -> DecodeFastConfig:
    if kv_pack_bd4:
        kv_pack_dense2 = False
    from sglang_tpu.envs import envs

    raw = mode_str.strip().lower()
    if raw == "shared16_host" and sliding_window is None and shared_group > 0:
        # Two 2048-token DMA buffers independent of serving context capacity.
        return DecodeFastConfig(
            2,
            2048,
            "shared16",
            max(1, 2048 // page_size),
            shared_group_size=shared_group,
        )
    use_v2 = _is_v2_enabled(raw)
    use_w7 = _is_w7_enabled(raw)
    use_w8 = _is_w8_enabled(raw)
    toks_cfg = set(raw.replace(",", "_").replace("-", "_").split("_"))
    explicit_streamdyn = "streamdyn" in toks_cfg
    explicit_ring8k = "ring8k" in toks_cfg
    explicit_ring4k = "ring4k" in toks_cfg or explicit_ring8k
    explicit_multiq = "multiq" in toks_cfg
    explicit_dyn_chunk = (
        (2048 if "dyn2k" in toks_cfg else (4096 if "dyn4k" in toks_cfg else 0))
        if decode_query_size == 1
        else 0
    )
    max_full_pps = (
        4096
        if (explicit_ring4k or use_w7 or shared_group > 0)
        else (160 if explicit_dyn_chunk else (144 if explicit_streamdyn else 128))
    )
    coalesce_dma = False
    in_vmem_pack2 = use_v2
    lazy_slot_zero = (use_w7 or ("noz" in toks_cfg)) and ("nonoz" not in toks_cfg)
    pv_bf16 = ("pbf16" in toks_cfg) or (
        sliding_window is not None and "pbf16swa" in toks_cfg
    )
    page_reduce = "pgred" in toks_cfg
    fp8_dots = "fp8dp" in toks_cfg
    hoist_new_kv = decode_query_size == 1 and sliding_window is None and ("hoistkv" in toks_cfg)
    shared_q_tile = (
        128
        if "q128" in toks_cfg
        else (1024 if "q1024" in toks_cfg else (512 if "q512" in toks_cfg else 256))
    )
    shared_split_dense2 = "shsplit" in toks_cfg
    shared_fp8_qkpv = "shfp8" in toks_cfg
    shared_fp8_pv_only = "shfp8pv" in toks_cfg
    batch_full = "batchfull" in toks_cfg
    dyn2k_unroll = "dyn2ku" in toks_cfg
    u4_flat = "u4flat" in toks_cfg
    gear_4k = "g4k" in toks_cfg
    no_fold = "nofold" in toks_cfg
    fold_col = ("foldcol" in toks_cfg or gear_4k) and not no_fold
    fold_128 = ("fold128" in toks_cfg or fold_col) and not no_fold
    no_mask0 = "nomask0" in toks_cfg or gear_4k
    swabatch_group_size = (
        1
        if "nobatch4" in toks_cfg
        else (4 if ("batch4" in toks_cfg or "b4" in toks_cfg) else (2 if ("batch2" in toks_cfg or "b2" in toks_cfg) else 1))
    )
    fine_gears = (
        decode_query_size == 1
        and sliding_window is None
        and (("g512" in toks_cfg) or ("w7auto2" in toks_cfg))
    )
    # Keep chunk_sz an exact multiple of 2 * page_size (256) so [16*dq, chunk_sz] stays
    # spill-free (<= 24..32 VREGs) across dq in 1..8 (including DFlash dq=5/6).
    if decode_query_size <= 1:
        default_v2_chunk = 2048
    elif decode_query_size == 2:
        default_v2_chunk = 1024
    elif decode_query_size <= 4:
        default_v2_chunk = 512
    else:
        default_v2_chunk = 256
    kv_cap_tokens = pages_per_seq * page_size
    if sliding_window is not None:
        win_pages = cdiv(sliding_window + page_size - 1, page_size)
        num_slots = 4
        chunk_sz = max(default_v2_chunk if use_v2 else 256, page_size)
        if win_pages * page_size <= 4096:
            if decode_query_size > 1:
                loop_kind = (
                    "stream0b"
                    if (explicit_multiq and decode_query_size in (2, 4) and not use_w7)
                    else "streamdyn"
                )
            else:
                loop_kind = "swabatch" if (use_w7 or use_w8) else "stream0b"
        else:
            loop_kind = "streamdyn" if decode_query_size > 1 else "clean2k"
        if loop_kind in ("stream0b", "swabatch") and "noz" not in toks_cfg:
            lazy_slot_zero = False
    else:
        win_pages = pages_per_seq
        num_slots = 4 if kv_cap_tokens <= 10752 else 3
        chunk_sz = max(default_v2_chunk if use_v2 else 256, page_size)
        if batch_full and shared_group == 0 and decode_query_size == 1 and kv_cap_tokens <= 4096:
            loop_kind = "swabatch"
        elif shared_group > 0 and decode_query_size == 1:
            loop_kind = f"shared{shared_group}"
            chunk_sz = 2048
            num_slots = 3 if kv_pack_dense2 else 4
            coalesce_dma = True
        elif explicit_ring4k or (raw == "shared16_host" and kv_cap_tokens > 16384) or (use_w7 and kv_cap_tokens > 20480 and decode_query_size == 1):
            loop_kind = "ring4k"
            num_slots = 3 if explicit_ring8k else 4
            chunk_sz = 8192 if explicit_ring8k else 4096
            if use_w7 or explicit_ring4k or raw == "shared16_host":
                coalesce_dma = True
        elif explicit_dyn_chunk and decode_query_size == 1 and 4096 < kv_cap_tokens <= 20480:
            loop_kind = "streamdyn"
            chunk_sz = explicit_dyn_chunk
        elif use_w7 and decode_query_size == 1 and kv_cap_tokens <= 20480 and not explicit_streamdyn:
            loop_kind = "w7auto"
            chunk_sz = 5120 if kv_cap_tokens > 8192 else 4096
        elif (
            explicit_streamdyn
            or (use_w7 and kv_cap_tokens > 16384)
            or (decode_query_size > 1 and (kv_cap_tokens > 4096 or use_w7 or not explicit_multiq))
        ):
            loop_kind = "streamdyn"
            if use_w7 and kv_cap_tokens > 16384 and decode_query_size == 1:
                chunk_sz = 4096
        else:
            loop_kind = "stream0b" if pages_per_seq <= max_full_pps else "clean2k"

    if raw not in ("1", "true", "auto", "on", "v2", "2"):
        explicit_chunk_sz = None
        for tok in raw.replace(",", "_").replace("-", "_").split("_"):
            if not tok or tok in (
                "v2",
                "w7",
                "w8",
                "pps256",
                "adapt",
                "noadapt",
                "verify",
                "multiq",
                "shared",
                "shared16",
                "shared8",
                "shared32",
                "shared64",
                "shsplit",
                "shfp8",
                "shfp8pv",
                "q128",
                "dyn2k",
                "dyn4k",
                "batchfull",
                "dyn2ku",
                "u4flat",
                "ring8k",
                "batch4",
                "b4",
                "batch2",
                "b2",
                "nobatch4",
            ):
                continue
            if tok.startswith("s") and tok[1:].isdigit():
                num_slots = int(tok[1:])
            elif tok in ("c128", "c256", "c512", "c1024", "c2048", "c4096", "c5120", "c8192"):
                parsed_c = int(tok[1:])
                # Keep w7auto/ring4k/ring8k/shared*/dyn* chunk sizes intact when paired with "w8,c1024".
                if (
                    (
                        loop_kind in ("w7auto", "ring4k")
                        or (loop_kind == "streamdyn" and explicit_dyn_chunk > 0)
                    )
                    and parsed_c == 1024
                    and ("w8" in toks_cfg or "w7" in toks_cfg)
                ) or (
                    loop_kind.startswith("shared")
                    and parsed_c == 1024
                    and ("w8" in toks_cfg or "w7" in toks_cfg or loop_kind == "shared16")
                ):
                    continue
                chunk_sz = parsed_c
                explicit_chunk_sz = chunk_sz
            elif tok == "shc1024" and loop_kind.startswith("shared"):
                chunk_sz = 1024
                explicit_chunk_sz = chunk_sz
            elif tok == "coal":
                coalesce_dma = True
            elif tok == "nocoal":
                coalesce_dma = False
            elif tok == "noz":
                lazy_slot_zero = True
            elif tok == "nonoz":
                lazy_slot_zero = False
            elif tok == "pbf16":
                pv_bf16 = True
            elif tok == "pf32":
                pv_bf16 = False
            elif tok == "pgred":
                page_reduce = True
            elif tok == "nopgred":
                page_reduce = False
            elif tok == "fp8dp":
                fp8_dots = True
            elif tok == "hoistkv":
                hoist_new_kv = decode_query_size == 1 and sliding_window is None
            elif tok == "nohoistkv":
                hoist_new_kv = False
            elif tok == "g512":
                fine_gears = decode_query_size == 1
            elif tok == "nog512":
                fine_gears = False
            elif tok == "ivpack2":
                in_vmem_pack2 = True
            elif tok == "noivpack2":
                in_vmem_pack2 = False
            elif tok in (
                "w7auto",
                "w7auto2",
                "swabatch",
                "swafast",
                "swafast2",
                "stream0b",
                "streamdyn",
                "ring4k",
                "stream2h",
                "stream1p",
                "stream2ld",
                "full1p",
                "clean1p",
                "clean2k",
            ):
                if not (
                    tok in ("w7auto", "w7auto2", "streamdyn", "ring4k")
                    and sliding_window is not None
                    and win_pages <= 32
                ):
                    if tok == "w7auto2":
                        loop_kind = "w7auto"
                        fine_gears = decode_query_size == 1
                    elif tok == "full1p":
                        loop_kind = "stream0b"
                    else:
                        loop_kind = tok
                    if loop_kind == "ring4k" and explicit_chunk_sz is None:
                        chunk_sz = 4096
                    elif loop_kind == "w7auto" and explicit_chunk_sz is None:
                        chunk_sz = 5120 if kv_cap_tokens > 8192 else 4096

    if sliding_window != 2049 and loop_kind in ("clean1p", "stream2h"):
        loop_kind = "streamdyn" if use_v2 else ("stream0b" if win_pages * page_size <= 4096 else "clean2k")
    if sliding_window is None and pages_per_seq > max_full_pps and loop_kind != "ring4k" and not loop_kind.startswith("shared"):
        num_slots = max(2, min(num_slots, 3))
        loop_kind = "clean2k"

    if loop_kind == "swabatch" and decode_query_size == 1 and "swanoz" not in toks_cfg:
        lazy_slot_zero = False

    num_slots = max(2, num_slots)
    d2_grp_env = getattr(envs, "SGLANG_TPU_RPA_V3_DENSE2_GROUP", None)
    dense2_group = ("d2grp" in toks_cfg) or bool(d2_grp_env is not None and d2_grp_env.get())
    sh_hkv_env = getattr(envs, "SGLANG_TPU_RPA_V3_SHARED_HOISTKV", None)
    shared_hoistkv = bool(sh_hkv_env is not None and sh_hkv_env.get())
    eff_shared_group = shared_group if (sliding_window is None and loop_kind.startswith("shared")) else 0
    if eff_shared_group > 0 and decode_query_size == 1 and shared_hoistkv and "nohoistkv" not in toks_cfg:
        hoist_new_kv = True
    eff_swabatch_grp = (
        swabatch_group_size
        if (loop_kind == "swabatch" and decode_query_size == 1 and (not kv_pack_dense2 or dense2_group) and not fp8_dots)
        else 1
    )

    if sliding_window is not None:
        if loop_kind in ("stream0b", "swabatch", "swafast", "swafast2", "stream2h", "stream1p", "stream2ld", "clean1p"):
            max_ring_pages = win_pages
        elif loop_kind in ("streamdyn", "w7auto"):
            chunk_pages = max(1, chunk_sz // page_size)
            max_ring_pages = align_to(win_pages, chunk_pages)
        else:
            max_ring_pages = align_to(win_pages, 16)
    else:
        if batch_full and loop_kind == "swabatch":
            max_ring_pages = win_pages
        elif eff_shared_group > 0:
            max_ring_pages = max(1, chunk_sz // page_size)
        elif loop_kind == "ring4k":
            max_ring_pages = max(1, chunk_sz // page_size)
            lazy_slot_zero = True
        elif loop_kind in ("streamdyn", "w7auto"):
            if loop_kind == "streamdyn" and chunk_sz < page_size:
                chunk_sz = page_size
            chunk_pages = max(1, chunk_sz // page_size)
            eff_cap = max(max_full_pps, cdiv(18432, page_size)) if use_w7 else max_full_pps
            min_w7_pgs = (
                (max(1, 8192 // page_size) if kv_cap_tokens > 8192 else max(1, 1024 // page_size))
                if loop_kind == "w7auto"
                else 1
            )
            max_ring_pages = min(
                align_to(max(pages_per_seq, min_w7_pgs), chunk_pages),
                align_to(eff_cap, chunk_pages),
            )
        elif loop_kind in ("stream0b", "stream1p", "stream2ld"):
            max_ring_pages = min(align_to(max(pages_per_seq, 1), 2), max_full_pps)
        else:
            max_ring_pages = min(align_to(max(pages_per_seq, 1), 16), max_full_pps)

    return DecodeFastConfig(
        num_slots=num_slots,
        chunk_sz=chunk_sz,
        loop_kind=loop_kind,
        max_ring_pages=max_ring_pages,
        coalesce_dma=coalesce_dma,
        in_vmem_pack2=in_vmem_pack2,
        lazy_slot_zero=lazy_slot_zero,
        pv_bf16=pv_bf16,
        page_reduce=page_reduce,
        fp8_dots=fp8_dots,
        hoist_new_kv=hoist_new_kv,
        fine_gears=fine_gears,
        shared_group_size=eff_shared_group,
        shared_q_tile=shared_q_tile,
        shared_split_dense2=shared_split_dense2,
        shared_fp8_qkpv=shared_fp8_qkpv,
        shared_fp8_pv_only=shared_fp8_pv_only,
        batch_full=batch_full,
        dyn2k_unroll=dyn2k_unroll,
        u4_flat=u4_flat,
        gear_4k=gear_4k,
        fold_128=fold_128,
        fold_col=fold_col,
        no_mask0=no_mask0,
        swabatch_group_size=eff_swabatch_grp,
        dense2_group=dense2_group,
    )


def _broadcast_minor_128(src: jax.Array) -> jax.Array:
    if src.shape[-1] == 128:
        return src
    return jnp.broadcast_to(src, (*src.shape[:-1], 128))


def _fold128_op(x: jax.Array, op) -> jax.Array:
    """Folds 2D ``[q_rows, k*128]`` down to ``[q_rows, 128]`` via pure-2D slicing."""
    cur = x
    while cur.shape[1] > 128:
        n_b = cur.shape[1] // 128
        half_b = n_b // 2
        hw = half_b * 128
        folded = op(cur[:, :hw], cur[:, hw : 2 * hw])
        if (n_b & 1) == 0:
            cur = folded
        else:
            cur = (
                op(folded[:, :128], cur[:, 2 * hw :])
                if half_b == 1
                else jnp.concatenate([folded, cur[:, 2 * hw :]], axis=1)
            )
    return cur


def _exp2_fold_col(
    scores: jax.Array, ref_m_128: jax.Array, exp2_sc
) -> tuple[jax.Array, jax.Array]:
    """Computes ``exp2((scores - ref_m) * exp2_sc)`` & ``[q, 128]`` sum without wide concats."""
    n_cols = scores.shape[1] // 128
    p_cols = [
        jnp.exp2((scores[:, j * 128 : (j + 1) * 128] - ref_m_128) * exp2_sc)
        for j in range(n_cols)
    ]
    p_full = jnp.concatenate(p_cols, axis=1)
    while len(p_cols) > 1:
        nxt = [p_cols[i] + p_cols[i + 1] for i in range(0, len(p_cols) - 1, 2)]
        if len(p_cols) & 1:
            nxt.append(p_cols[-1])
        p_cols = nxt
    return p_full, p_cols[0]


def _broadcast_minor_128_any(src_128: jax.Array, width: int) -> jax.Array:
    if width < 128:
        return src_128[:, :width]
    if width == 128:
        return src_128
    parts = [src_128] * (width // 128)
    rem = width % 128
    if rem > 0:
        parts.append(src_128[:, :rem])
    return jnp.concatenate(parts, axis=1)


def _u8_to_fp8(u8: jax.Array, *, interpret_mode: bool) -> jax.Array:
    if interpret_mode:
        return lax.bitcast_convert_type(u8, jnp.float8_e4m3fn)
    return pltpu.bitcast(u8, jnp.float8_e4m3fn)


def _load_bkv_fp8_split(
    bkv_vmem_ref,
    slot,
    row_start: int,
    chunk_sz: int,
    *,
    kind: str,
    page_size: int = 128,
    interpret_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(even_fp8, odd_fp8)`` ``[chunk_sz//2, 128]`` rows from a dense2 buffer."""
    phys_rows = page_size // 2
    half_sz = chunk_sz // 2
    if interpret_mode:
        pg_start = row_start // page_size
        n_pg = chunk_sz // page_size
        blk = bkv_vmem_ref[slot, pl.ds(pg_start, n_pg)].reshape(half_sz, 4, 128)
        e_idx, o_idx = (0, 2) if kind == "k" else (1, 3)
        return blk[:, e_idx, :], blk[:, o_idx, :]
    max_phys_rows = bkv_vmem_ref.shape[1] * phys_rows
    u32_ref = bkv_vmem_ref.bitcast(jnp.uint32).at[slot].reshape(max_phys_rows, 128)
    pair_u32 = u32_ref[pl.ds(pl.multiple_of(row_start // 2, phys_rows), half_sz)]
    if kind == "k":
        even_u8 = pair_u32.astype(jnp.uint8)
        odd_u8 = lax.shift_right_logical(pair_u32, jnp.uint32(16)).astype(jnp.uint8)
    else:
        even_u8 = lax.shift_right_logical(pair_u32, jnp.uint32(8)).astype(jnp.uint8)
        odd_u8 = lax.shift_right_logical(pair_u32, jnp.uint32(24)).astype(jnp.uint8)
    return (
        _u8_to_fp8(even_u8, interpret_mode=False),
        _u8_to_fp8(odd_u8, interpret_mode=False),
    )


def _load_bkv_fp8_split_half(
    bkv_vmem_ref,
    slot,
    row_start: int,
    chunk_sz: int,
    *,
    kind: str,
    odd: bool,
    page_size: int = 128,
    interpret_mode: bool = False,
) -> jax.Array:
    """Return even or odd ``[chunk_sz//2, 128]`` rows from a dense2 buffer."""
    phys_rows = page_size // 2
    half_sz = chunk_sz // 2
    idx = (0 if kind == "k" else 1) + (2 if odd else 0)
    if interpret_mode:
        pg_start = row_start // page_size
        n_pg = chunk_sz // page_size
        blk = bkv_vmem_ref[slot, pl.ds(pg_start, n_pg)].reshape(half_sz, 4, 128)
        return blk[:, idx, :]
    max_phys_rows = bkv_vmem_ref.shape[1] * phys_rows
    u32_ref = bkv_vmem_ref.bitcast(jnp.uint32).at[slot].reshape(max_phys_rows, 128)
    pair_u32 = u32_ref[pl.ds(pl.multiple_of(row_start // 2, phys_rows), half_sz)]
    shift = idx * 8
    raw_u8 = (
        pair_u32 if shift == 0 else lax.shift_right_logical(pair_u32, jnp.uint32(shift))
    ).astype(jnp.uint8)
    return _u8_to_fp8(raw_u8, interpret_mode=False)


def _load_bkv_bd4_pair_parts(
    bkv_vmem_ref,
    slot,
    row_start: int,
    width: int,
    *,
    kind: str,
    page_size: int = 128,
    interpret_mode: bool = False,
    valid_tokens: jax.Array | int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return ``(pair01_fp8, pair23_fp8)`` ``[width//4, 128]`` rows from a bd4 buffer."""
    phys_rows = page_size // 4
    n_u32 = width // 4
    if interpret_mode:
        pg_start = row_start // page_size
        n_pg = width // page_size
        blk_u8 = lax.bitcast_convert_type(
            bkv_vmem_ref[slot, pl.ds(pg_start, n_pg)].reshape(n_u32, 4, 128),
            jnp.uint8,
        )
        i0, i1 = (0, 2) if kind == "k" else (1, 3)
        u8_0, u8_1 = blk_u8[:, i0, :], blk_u8[:, i1, :]
    else:
        max_phys_rows = bkv_vmem_ref.shape[1] * phys_rows
        u32_ref = bkv_vmem_ref.bitcast(jnp.uint32).at[slot].reshape(max_phys_rows, 128)
        phys_align = min(phys_rows, max(1, n_u32))
        pair_u32 = u32_ref[pl.ds(pl.multiple_of(row_start // 4, phys_align), n_u32)]
        if kind == "k":
            u8_0 = pair_u32.astype(jnp.uint8)
            u8_1 = lax.shift_right_logical(pair_u32, jnp.uint32(16)).astype(jnp.uint8)
        else:
            u8_0 = lax.shift_right_logical(pair_u32, jnp.uint32(8)).astype(jnp.uint8)
            u8_1 = lax.shift_right_logical(pair_u32, jnp.uint32(24)).astype(jnp.uint8)
    if valid_tokens is not None:
        r_id = (row_start // 4) + lax.broadcasted_iota(jnp.int32, (n_u32, 128), 0)
        lo_h = lax.broadcasted_iota(jnp.int32, (n_u32, 128), 1) < 64
        t0 = 4 * r_id + jnp.where(lo_h, 0, 1)
        t1 = 4 * r_id + jnp.where(lo_h, 2, 3)
        u8_0 = jnp.where(t0 < valid_tokens, u8_0, jnp.uint8(0))
        u8_1 = jnp.where(t1 < valid_tokens, u8_1, jnp.uint8(0))
    return (
        _u8_to_fp8(u8_0, interpret_mode=interpret_mode),
        _u8_to_fp8(u8_1, interpret_mode=interpret_mode),
    )


def _load_bkv_bd4_kv(
    bkv_vmem_ref,
    slot,
    row_start: int,
    width: int,
    *,
    kind: str,
    page_size: int = 128,
    interpret_mode: bool = False,
    valid_tokens: jax.Array | int | None = None,
) -> jax.Array:
    p01, p23 = _load_bkv_bd4_pair_parts(
        bkv_vmem_ref,
        slot,
        row_start,
        width,
        kind=kind,
        page_size=page_size,
        interpret_mode=interpret_mode,
        valid_tokens=valid_tokens,
    )
    return jnp.concatenate([p01, p23], axis=0)


def _bd4_c_even_iota(
    row_start, half_w: int, rows: int, *, eff_hw: int | None = None
) -> jax.Array:
    n_u32 = half_w // 2
    col_w = half_w if eff_hw is None else eff_hw
    h_io = lax.broadcasted_iota(jnp.int32, (rows, col_w), 1)
    return row_start + jnp.where(h_io < n_u32, 4 * h_io, 4 * (h_io - n_u32) + 2)


def _load_bkv_fp8_chunk(
    bkv_vmem_ref,
    slot,
    row_start: int,
    chunk_sz: int,
    *,
    kind: str,
    page_size: int = 128,
    kv_pack_dense2: bool = False,
    kv_pack_bd4: bool = False,
    interpret_mode: bool = False,
    valid_tokens: jax.Array | int | None = None,
) -> jax.Array:
    """Load K or V FP8 rows from ``bkv_vmem_ref`` with 8-bit dense2 unpacking.

    Unpacking into ``uint8`` before concatenation keeps peak register footprint at
    ``[chunk_sz, 128]`` 8-bit (~32 VREGs at 1024 keys) instead of 32-bit (~128 VREGs).
    When ``valid_tokens`` is provided, rows at or beyond ``valid_tokens`` are
    zeroed in ``uint8`` before bitcasting to FP8 so poisoned ``0x7F`` NaNs cannot
    enter arithmetic.
    """
    if kv_pack_bd4:
        return _load_bkv_bd4_kv(
            bkv_vmem_ref,
            slot,
            row_start,
            chunk_sz,
            kind=kind,
            page_size=page_size,
            interpret_mode=interpret_mode,
            valid_tokens=valid_tokens,
        )
    if kv_pack_dense2:
        even_f8, odd_f8 = _load_bkv_fp8_split(
            bkv_vmem_ref,
            slot,
            row_start,
            chunk_sz,
            kind=kind,
            page_size=page_size,
            interpret_mode=interpret_mode,
        )
        if interpret_mode:
            if valid_tokens is None:
                return jnp.concatenate([even_f8, odd_f8], axis=0)
            even_u8 = lax.bitcast_convert_type(even_f8, jnp.uint8)
            odd_u8 = lax.bitcast_convert_type(odd_f8, jnp.uint8)
            raw_u8 = jnp.concatenate([even_u8, odd_u8], axis=0)
            pos_row = _bkv_col_iota(row_start, chunk_sz, 1, kv_pack_dense2=True).T
            raw_u8 = jnp.where(pos_row < valid_tokens, raw_u8, jnp.uint8(0))
            return _u8_to_fp8(raw_u8, interpret_mode=True)
        even_u8 = pltpu.bitcast(even_f8, jnp.uint8)
        odd_u8 = pltpu.bitcast(odd_f8, jnp.uint8)
        raw_u8 = jnp.concatenate([even_u8, odd_u8], axis=0)
        if valid_tokens is not None:
            pos_row = _bkv_col_iota(row_start, chunk_sz, 1, kv_pack_dense2=True).T
            raw_u8 = jnp.where(pos_row < valid_tokens, raw_u8, jnp.uint8(0))
        return _u8_to_fp8(raw_u8, interpret_mode=False)
    if interpret_mode:
        pg_start = row_start // page_size
        n_pg = chunk_sz // page_size
        blk = bkv_vmem_ref[slot, pl.ds(pg_start, n_pg)].reshape(chunk_sz, 4, 128)
        col = 0 if kind == "k" else 1
        if valid_tokens is None:
            return blk[:, col, :]
        raw_u8 = lax.bitcast_convert_type(blk[:, col, :], jnp.uint8)
        pos_row = _bkv_col_iota(row_start, chunk_sz, 1, kv_pack_dense2=False).T
        raw_u8 = jnp.where(pos_row < valid_tokens, raw_u8, jnp.uint8(0))
        return _u8_to_fp8(raw_u8, interpret_mode=True)
    max_rows = bkv_vmem_ref.shape[1] * page_size
    u32_ref = bkv_vmem_ref.bitcast(jnp.uint32).at[slot].reshape(max_rows, 128)
    u32 = u32_ref[pl.ds(pl.multiple_of(row_start, page_size), chunk_sz)]
    raw_u8 = (
        u32.astype(jnp.uint8)
        if kind == "k"
        else lax.shift_right_logical(u32, jnp.uint32(8)).astype(jnp.uint8)
    )
    if valid_tokens is not None:
        pos_row = _bkv_col_iota(row_start, chunk_sz, 1, kv_pack_dense2=False).T
        raw_u8 = jnp.where(pos_row < valid_tokens, raw_u8, jnp.uint8(0))
    return _u8_to_fp8(raw_u8, interpret_mode=interpret_mode)


def _bkv_col_iota(
    row_start,
    width: int,
    q_rows: int,
    *,
    kv_pack_dense2: bool = False,
) -> jax.Array:
    if not kv_pack_dense2:
        return row_start + lax.broadcasted_iota(jnp.int32, (q_rows, width), 1)
    hw = width // 2
    h_io = lax.broadcasted_iota(jnp.int32, (q_rows, hw), 1)
    return row_start + jnp.concatenate([2 * h_io, 2 * h_io + 1], axis=1)


def _load_new_kv_1row_u16(
    kv_merged_vmem_ref,
    q_row,
    *,
    interpret_mode: bool = False,
) -> jax.Array:
    aligned8 = (q_row // 8) * 8
    sub8 = q_row - aligned8
    if interpret_mode:
        blk = kv_merged_vmem_ref[pl.ds(aligned8, 8)].reshape(8, 4, 128)
        u8 = lax.bitcast_convert_type(blk, jnp.uint8)
        u32 = u8[:, 0, :].astype(jnp.uint32) | (u8[:, 1, :].astype(jnp.uint32) << 8)
    else:
        u32_ref = kv_merged_vmem_ref.bitcast(jnp.uint32).reshape(
            kv_merged_vmem_ref.shape[0], 128
        )
        u32 = u32_ref[pl.ds(pl.multiple_of(aligned8, 8), 8)]
    row_iota = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
    u32_m = jnp.where(row_iota == sub8, u32 & jnp.uint32(0xFFFF), jnp.uint32(0))
    return jnp.sum(u32_m.astype(jnp.int32), axis=0, keepdims=True).astype(jnp.uint32)


def _load_new_kv_1row_f32(
    kv_merged_vmem_ref,
    q_row,
    *,
    interpret_mode: bool = False,
    mask_lo64: bool = False,
) -> tuple[jax.Array, jax.Array]:
    n_rows = kv_merged_vmem_ref.shape[0]
    step_r = min(32, n_rows)
    aligned_r = (q_row // step_r) * step_r
    sub_r = q_row - aligned_r
    if interpret_mode:
        blk = kv_merged_vmem_ref[pl.ds(aligned_r, step_r)].reshape(step_r, 4, 128)
        u8 = lax.bitcast_convert_type(blk, jnp.uint8)
        u32 = u8[:, 0, :].astype(jnp.uint32) | (u8[:, 1, :].astype(jnp.uint32) << 8)
    else:
        u32_ref = kv_merged_vmem_ref.bitcast(jnp.uint32).reshape(n_rows, 128)
        u32 = u32_ref[pl.ds(pl.multiple_of(aligned_r, step_r), step_r)]
    row_iota = lax.broadcasted_iota(jnp.int32, (step_r, 128), 0)
    row_mask = row_iota == sub_r
    if mask_lo64:
        # bd4 zero-pads d=64..127 in Q; scrub padded KV lanes before FP8->F32 to avoid 0.0*NaN.
        row_mask = row_mask & (lax.broadcasted_iota(jnp.int32, (step_r, 128), 1) < 64)
    u32_m = jnp.where(row_mask, u32, jnp.uint32(0))
    k_f8 = _u8_to_fp8(u32_m.astype(jnp.uint8), interpret_mode=interpret_mode)
    v_f8 = _u8_to_fp8((u32_m >> 8).astype(jnp.uint8), interpret_mode=interpret_mode)
    return (
        jnp.sum(k_f8.astype(jnp.float32), axis=0, keepdims=True),
        jnp.sum(v_f8.astype(jnp.float32), axis=0, keepdims=True),
    )


def _roll_lanes_fast(x: jax.Array, shift: int, *, interpret_mode: bool) -> jax.Array:
    if interpret_mode:
        return jnp.roll(x, shift=shift, axis=1)
    return pltpu.roll(x, shift=shift, axis=1)


def _rpad_seq_info(
    seq,
    *,
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    num_page_indices: int,
    page_size: int,
    sliding_window: int | None,
    dq: int,
    probe: str | None = None,
):
    q_start = cu_q_lens_ref[seq]
    q_end = cu_q_lens_ref[seq + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq]
    active = (q_len > 0) & (kv_len > 0) if dq == 1 else ((q_len == dq) & (kv_len >= dq))
    cache_kv = jnp.maximum(kv_len - dq, 0)
    if sliding_window is not None:
        first_key = jnp.maximum(cache_kv - (sliding_window - 1), 0)
        base = (first_key // page_size) * page_size
    else:
        first_key = jnp.int32(0)
        base = jnp.int32(0)
    rel_first = first_key - base
    rel_end = jnp.where(active, jnp.maximum(cache_kv - base, 0), 0)
    n_pages = (rel_end + page_size - 1) // page_size
    if probe == "dense_kv":
        n_pages = (n_pages + 1) // 2
        rel_end = jnp.minimum(rel_end, n_pages * page_size)
        rel_first = jnp.minimum(rel_first, rel_end)
    seq_pg_base = cdiv(cu_kv_lens_ref[seq], page_size) + (base // page_size)
    return q_start, active, kv_len, rel_first, rel_end, n_pages, seq_pg_base


def _start_fetch_bkv_pages(
    seq_pg_base,
    n_pages,
    slot,
    *,
    kv_cache_hbm_ref,
    bkv_vmem_ref,
    bkv_sems,
    page_indices_ref,
    num_page_indices: int,
    cfg: DecodeFastConfig,
    interpret_mode: bool,
    probe: str | None = None,
):
    if probe == "nodma":
        return

    def _run_coal_dma():
        def _cond_dma(i):
            return i < n_pages

        def _body_dma(i):
            pg0 = page_indices_ref[jnp.minimum(seq_pg_base + i, num_page_indices - 1)]

            def _ext_cond(run):
                nxt_idx = jnp.minimum(seq_pg_base + i + run, num_page_indices - 1)
                return (i + run < n_pages) & (page_indices_ref[nxt_idx] == pg0 + run)

            run = lax.while_loop(_ext_cond, lambda r: r + jnp.int32(1), jnp.int32(1))
            pltpu.make_async_copy(
                kv_cache_hbm_ref.at[pl.ds(pg0, run)],
                bkv_vmem_ref.at[slot, pl.ds(i, run)],
                bkv_sems.at[slot],
            ).start()
            return i + run

        lax.while_loop(_cond_dma, _body_dma, jnp.int32(0))

    def _dma_page(i, _):
        pg_idx = jnp.minimum(seq_pg_base + i, num_page_indices - 1)
        pg = page_indices_ref[pg_idx]
        pltpu.make_async_copy(
            kv_cache_hbm_ref.at[pl.ds(pg, 1)],
            bkv_vmem_ref.at[slot, pl.ds(i, 1)],
            bkv_sems.at[slot],
        ).start()

    if (
        cfg.loop_kind in ("w7auto", "swafast", "swafast2")
        or (cfg.loop_kind == "swabatch" and cfg.coalesce_dma)
    ) and not interpret_mode:
        @pl.when(n_pages <= 8)
        def _short_coal():
            _run_coal_dma()

        @pl.when(n_pages > 8)
        def _long_direct():
            lax.fori_loop(0, n_pages, _dma_page, None)

        return

    if cfg.coalesce_dma and not interpret_mode:
        _run_coal_dma()
        return

    lax.fori_loop(0, n_pages, _dma_page, None)


def _zero_bkv_pages_u32(
    bkv_vmem_ref,
    slot,
    start_pg,
    end_pg,
    *,
    max_ring_pages: int,
    phys_page_rows: int,
    kv_pack_dense2: bool,
    interpret_mode: bool,
):
    if interpret_mode:
        z_f8 = jnp.zeros(bkv_vmem_ref.shape[2:], bkv_vmem_ref.dtype)

        def _z_interp(p_i, _):
            bkv_vmem_ref[slot, p_i] = z_f8

        lax.fori_loop(start_pg, end_pg, _z_interp, None)
        return
    if phys_page_rows <= 128:
        z_pg = jnp.zeros((phys_page_rows, 128), jnp.uint32)
        u32_ref = (
            bkv_vmem_ref.bitcast(jnp.uint32)
            .at[slot]
            .reshape(max_ring_pages, phys_page_rows, 128)
        )
        lax.fori_loop(
            start_pg,
            end_pg,
            lambda p_i, _: u32_ref.__setitem__(p_i, z_pg),
            None,
        )
        return
    u32_flat = (
        bkv_vmem_ref.bitcast(jnp.uint32)
        .at[slot]
        .reshape(max_ring_pages * phys_page_rows, 128)
    )
    n_subs = phys_page_rows // 128
    z_128 = jnp.zeros((128, 128), jnp.uint32)
    lax.fori_loop(
        start_pg * n_subs,
        end_pg * n_subs,
        lambda s_i, _: u32_flat.__setitem__(
            pl.ds(pl.multiple_of(s_i * 128, 128), 128), z_128
        ),
        None,
    )


def _zero_all_bkv_slots_u32(
    bkv_vmem_ref,
    *,
    num_slots: int,
    max_ring_pages: int,
    phys_page_rows: int,
    interpret_mode: bool,
):
    """Zeroes ``bkv_vmem_ref`` once in 8-sublane ``uint32`` before any DMAs start."""
    if interpret_mode:
        z_f8 = jnp.zeros_like(bkv_vmem_ref[0, 0])
        for s_i in range(num_slots):
            for p_i in range(max_ring_pages):
                bkv_vmem_ref[s_i, p_i] = z_f8
        return
    if phys_page_rows <= 128:
        z_pg = jnp.zeros((phys_page_rows, 128), jnp.uint32)
        flat_u32 = bkv_vmem_ref.bitcast(jnp.uint32).reshape(
            num_slots * max_ring_pages, phys_page_rows, 128
        )
        lax.fori_loop(
            0,
            num_slots * max_ring_pages,
            lambda idx, _: flat_u32.__setitem__(idx, z_pg),
            None,
        )
        return
    n_subs = phys_page_rows // 128
    z_128 = jnp.zeros((128, 128), jnp.uint32)
    flat_u32 = bkv_vmem_ref.bitcast(jnp.uint32).reshape(
        num_slots * max_ring_pages * phys_page_rows, 128
    )
    lax.fori_loop(
        0,
        num_slots * max_ring_pages * n_subs,
        lambda s_i, _: flat_u32.__setitem__(
            pl.ds(pl.multiple_of(s_i * 128, 128), 128), z_128
        ),
        None,
    )


def _dispatch_swabatch_w_pg(
    max_re_b,
    *,
    max_ring_pages: int,
    page_size: int,
    run_fn,
):
    """Dispatches active-window page count gears consistently across pre-DMA & compute."""
    gears: list[tuple[int, int]] = []
    seen_pg: set[int] = set()
    for thresh in (256, 512, 1024, 1536, 2048):
        pg = cdiv(thresh, page_size)
        if pg >= max_ring_pages:
            break
        if gears and gears[-1][1] == pg:
            gears[-1] = (thresh, pg)
        elif pg not in seen_pg:
            gears.append((thresh, pg))
            seen_pg.add(pg)
    if not gears:
        run_fn(max_ring_pages)
        return
    prev = 0
    for idx, (thresh, pg) in enumerate(gears):
        cond = (
            (max_re_b <= thresh)
            if idx == 0
            else ((max_re_b > prev) & (max_re_b <= thresh))
        )

        @pl.when(cond)
        def _b_gear(p=pg):
            run_fn(p)

        prev = thresh

    @pl.when(max_re_b > prev)
    def _b_full():
        run_fn(max_ring_pages)


def _start_swabatch_group_dmas(
    g_idx,
    slot,
    w_pg: int,
    *,
    group_sz: int,
    n_active,
    kv_cache_hbm_ref,
    bkv_vmem_ref,
    bkv_sems,
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    num_page_indices: int,
    page_size: int,
    sliding_window: int | None,
):
    """Starts fixed-byte (group_sz * w_pg) HBM->VMEM page DMAs for one swabatch group."""
    sem = bkv_sems.at[slot]
    n_phys_pages = kv_cache_hbm_ref.shape[0]
    for sub in range(group_sz):
        seq = group_sz * g_idx + sub
        _, act, _, _, _, npg, pb = _rpad_seq_info(
            seq,
            kv_lens_ref=kv_lens_ref,
            page_indices_ref=page_indices_ref,
            cu_q_lens_ref=cu_q_lens_ref,
            cu_kv_lens_ref=cu_kv_lens_ref,
            num_page_indices=num_page_indices,
            page_size=page_size,
            sliding_window=sliding_window,
            dq=1,
        )
        act_seq = (seq < n_active) & act
        pb_safe = jnp.where(act_seq, jnp.clip(pb, 0, num_page_indices - 1), jnp.int32(0))
        pg0 = jnp.clip(page_indices_ref[pb_safe], 0, n_phys_pages - 1)
        for p_i in range(w_pg):
            pg_idx = jnp.clip(pb_safe + p_i, 0, num_page_indices - 1)
            pg = jnp.where(
                act_seq & (p_i < npg),
                jnp.clip(page_indices_ref[pg_idx], 0, n_phys_pages - 1),
                pg0,
            )
            pltpu.make_async_copy(
                kv_cache_hbm_ref.at[pl.ds(pg, 1)],
                bkv_vmem_ref.at[slot, pl.ds(sub * w_pg + p_i, 1)],
                sem,
            ).start()


def prepare_and_start_rpad_dmas(
    *,
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    kv_cache_hbm_ref,
    bkv_vmem_ref,
    bkv_sems,
    sliding_window: int | None,
    cfg: DecodeFastConfig,
    one_block_per_seq: bool,
    decode_query_size: int = 1,
    kv_pack_dense2: bool = False,
    probe: str | None = None,
    interpret_mode: bool = False,
    b_tokens: int | None = None,
    init_sv_vmem_ref=None,
    use_grp_vec: bool | None = None,
    kv_pack_bd4: bool = False,
) -> tuple[jax.Array, ...] | None:
    """Optionally pre-zeroes ``bkv_vmem_ref`` and fires initial RPAd page DMAs.

    Safe to call from a preceding fused VMEM stage (e.g. after ``b2`` weight DMAs
    drain and before ``w_qkv``/RoPE math) because it depends only on SMEM metadata,
    ``kv_cache_hbm_ref``, ``bkv_vmem_ref``, and ``bkv_sems``. Pass the returned
    state tuple directly as ``prestarted_bkv_state`` to ``_run_rpad_vmem_core``.
    Returns ``None`` when pre-starting is inapplicable (leaving ``_run_rpad_vmem_core``
    to run its normal startup path).
    """
    if (
        probe == "nocompute"
        or decode_query_size != 1
        or not one_block_per_seq
        or cfg.shared_group_size > 0
        or cfg.loop_kind == "shared16"
    ):
        return None
    if cfg.loop_kind == "swabatch" and cfg.u4_flat:
        return None

    max_num_seqs = kv_lens_ref.shape[0]
    phys_page_rows = kv_cache_hbm_ref.shape[1]
    if kv_pack_bd4:
        kv_pack_dense2 = False
    page_size = (
        phys_page_rows * 4
        if kv_pack_bd4
        else (phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows)
    )
    num_page_indices = page_indices_ref.shape[0]
    n_active = jnp.clip(distribution_ref[0], 0, max_num_seqs)

    def _seq_info(seq):
        return _rpad_seq_info(
            seq,
            kv_lens_ref=kv_lens_ref,
            page_indices_ref=page_indices_ref,
            cu_q_lens_ref=cu_q_lens_ref,
            cu_kv_lens_ref=cu_kv_lens_ref,
            num_page_indices=num_page_indices,
            page_size=page_size,
            sliding_window=sliding_window,
            dq=1,
            probe=probe,
        )

    def _start_fetch(seq_pg_base, n_pages, slot):
        _start_fetch_bkv_pages(
            seq_pg_base,
            n_pages,
            slot,
            kv_cache_hbm_ref=kv_cache_hbm_ref,
            bkv_vmem_ref=bkv_vmem_ref,
            bkv_sems=bkv_sems,
            page_indices_ref=page_indices_ref,
            num_page_indices=num_page_indices,
            cfg=cfg,
            interpret_mode=interpret_mode,
            probe=probe,
        )

    if cfg.loop_kind == "swabatch":
        def _scan_swa_b(s_i, acc):
            mx_re, mn_pg, all_a = acc
            _, act_i, _, _, re_i, npg_i, _ = _seq_info(s_i)
            return (
                jnp.where(act_i, jnp.maximum(mx_re, re_i), mx_re),
                jnp.where(act_i, jnp.minimum(mn_pg, npg_i), mn_pg),
                all_a & act_i,
            )

        max_re_b, min_pg_b, all_act_b = lax.fori_loop(
            0,
            n_active,
            _scan_swa_b,
            (jnp.int32(0), jnp.int32(cfg.max_ring_pages), jnp.bool_(True)),
        )

        def _prep_swa(w_pg: int):
            grp_sz = cfg.swabatch_group_size
            eff_b_tokens = max_num_seqs if b_tokens is None else int(b_tokens)
            can_grp_pre = _resolve_swabatch_group_flag(
                use_grp_vec,
                cfg=cfg,
                one_block_per_seq=one_block_per_seq,
                decode_query_size=decode_query_size,
                kv_pack_dense2=kv_pack_dense2,
                probe=probe,
                b_tokens=eff_b_tokens,
                max_num_seqs=max_num_seqs,
                bkv_slot_pages=bkv_vmem_ref.shape[1],
                has_init_sv=(
                    True
                    if (use_grp_vec is None and init_sv_vmem_ref is None)
                    else (init_sv_vmem_ref is not None)
                ),
            )

            def _prep_single():
                if not cfg.lazy_slot_zero:
                    @pl.when(min_pg_b < w_pg)
                    def _zero_prefix():
                        bkv_slot_pages = bkv_vmem_ref.shape[1]
                        for s_z in range(cfg.num_slots):
                            _zero_bkv_pages_u32(
                                bkv_vmem_ref,
                                s_z,
                                0,
                                w_pg,
                                max_ring_pages=bkv_slot_pages,
                                phys_page_rows=phys_page_rows,
                                kv_pack_dense2=kv_pack_dense2,
                                interpret_mode=interpret_mode,
                            )

                num_ahead_b = cfg.num_slots - 1
                for s_init in range(num_ahead_b):
                    @pl.when(s_init < n_active)
                    def _(si=s_init):
                        _, _, _, _, _, n_pg, pg_base = _seq_info(si)
                        _start_fetch(pg_base, n_pg, si)

            if can_grp_pre:
                n_grp = (n_active + grp_sz - 1) // grp_sz
                for g_init in range(cfg.num_slots - 1):
                    @pl.when(g_init < n_grp)
                    def _(gi=g_init):
                        _start_swabatch_group_dmas(
                            gi,
                            gi,
                            w_pg,
                            group_sz=grp_sz,
                            n_active=n_active,
                            kv_cache_hbm_ref=kv_cache_hbm_ref,
                            bkv_vmem_ref=bkv_vmem_ref,
                            bkv_sems=bkv_sems,
                            kv_lens_ref=kv_lens_ref,
                            page_indices_ref=page_indices_ref,
                            cu_q_lens_ref=cu_q_lens_ref,
                            cu_kv_lens_ref=cu_kv_lens_ref,
                            num_page_indices=num_page_indices,
                            page_size=page_size,
                            sliding_window=sliding_window,
                        )
            else:
                _prep_single()

        _dispatch_swabatch_w_pg(
            max_re_b,
            max_ring_pages=cfg.max_ring_pages,
            page_size=page_size,
            run_fn=_prep_swa,
        )
        return (max_re_b, min_pg_b, all_act_b)

    use_branchless_sanitize = cfg.loop_kind in ("stream0b", "swabatch", "swafast")
    if use_branchless_sanitize and not cfg.lazy_slot_zero:
        _zero_all_bkv_slots_u32(
            bkv_vmem_ref,
            num_slots=cfg.num_slots,
            max_ring_pages=cfg.max_ring_pages,
            phys_page_rows=phys_page_rows,
            interpret_mode=interpret_mode,
        )

    num_ahead = cfg.num_slots - 1
    for s_init in range(num_ahead):
        @pl.when(s_init < n_active)
        def _(si=s_init):
            _, _, _, _, _, n_pg, pg_base = _seq_info(si)
            _start_fetch(pg_base, n_pg, si)
    return ()


def _run_rpad_vmem_core(
    *,
    # Scalar prefetches
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    # Inputs / Outputs (VMEM & HBM refs).
    # For native TP=16 q8, pass q_vmem_ref shaped (B, 8, 128) (e.g. ref.at[:, pl.ds(0, 8), :]).
    q_vmem_ref,
    kv_merged_vmem_ref,
    gate_vmem_ref,
    kv_cache_hbm_ref,
    o_out_vmem_ref,
    updated_kv_cache_hbm_ref,
    # Scratches
    bkv_vmem_ref,
    state_vmem_ref,
    o_vmem_ref,
    kv_write_vmem_ref,
    init_sv_vmem_ref,
    written_cnt_smem_ref,
    bkv_sems,
    cache_write_sem,
    # Kernel parameters
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float | None,
    v_scale: float | None,
    cfg: DecodeFastConfig,
    one_block_per_seq: bool,
    decode_query_size: int = 1,
    in_kernel_pack2: bool = False,
    has_gate: bool = False,
    out_norm_eps: float | None = None,
    kv_pack_dense2: bool = False,
    probe: str | None = None,
    interpret_mode: bool = False,
    prestarted_bkv_state: tuple[jax.Array, ...] | None = None,
    use_grp_vec: bool | None = None,
    skip_kv_write: bool = False,
    kv_pack_bd4: bool = False,
):
    """Executes VMEM-resident decode RPA core (reusable in fused superkernels)."""
    max_num_tokens = q_vmem_ref.shape[0]
    max_num_seqs = kv_lens_ref.shape[0]
    phys_page_rows = kv_cache_hbm_ref.shape[1]
    if kv_pack_bd4:
        kv_pack_dense2 = False
    page_size = (
        phys_page_rows * 4
        if kv_pack_bd4
        else (phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows)
    )
    num_page_indices = page_indices_ref.shape[0]
    num_slots = cfg.num_slots
    max_ring_pages = cfg.max_ring_pages
    bkv_slot_pages = bkv_vmem_ref.shape[1]
    max_ring_rows = max_ring_pages * page_size
    use_branchless_sanitize = cfg.loop_kind in ("stream0b", "swabatch", "swafast")
    can_grp_vec = _resolve_swabatch_group_flag(
        use_grp_vec,
        cfg=cfg,
        one_block_per_seq=one_block_per_seq,
        decode_query_size=decode_query_size,
        kv_pack_dense2=kv_pack_dense2,
        probe=probe,
        b_tokens=max_num_tokens,
        max_num_seqs=max_num_seqs,
        bkv_slot_pages=bkv_slot_pages,
        has_init_sv=(init_sv_vmem_ref is not None),
    )
    use_hoist_kv = (cfg.hoist_new_kv or can_grp_vec) and decode_query_size == 1

    s_scale = float(sm_scale * (k_scale if k_scale is not None else 1.0))
    exp2_scale = float(s_scale * _LOG2_E)
    v_scale_val = float(v_scale) if v_scale is not None else 1.0

    dq = decode_query_size
    q_head_rows = q_vmem_ref.shape[1]
    q_out_head_rows = o_vmem_ref.shape[1]
    q_rows = q_head_rows * dq
    q_out_rows = q_out_head_rows * dq
    n_active = jnp.clip(distribution_ref[0], 0, max_num_seqs)

    def seq_info(seq):
        return _rpad_seq_info(
            seq,
            kv_lens_ref=kv_lens_ref,
            page_indices_ref=page_indices_ref,
            cu_q_lens_ref=cu_q_lens_ref,
            cu_kv_lens_ref=cu_kv_lens_ref,
            num_page_indices=num_page_indices,
            page_size=page_size,
            sliding_window=sliding_window,
            dq=dq,
            probe=probe,
        )

    def start_fetch_pages(seq_pg_base, n_pages, slot):
        _start_fetch_bkv_pages(
            seq_pg_base,
            n_pages,
            slot,
            kv_cache_hbm_ref=kv_cache_hbm_ref,
            bkv_vmem_ref=bkv_vmem_ref,
            bkv_sems=bkv_sems,
            page_indices_ref=page_indices_ref,
            num_page_indices=num_page_indices,
            cfg=cfg,
            interpret_mode=interpret_mode,
            probe=probe,
        )

    def wait_fetch_pages(n_pages, slot):
        if probe == "nodma":
            return

        @pl.when(n_pages > 0)
        def _():
            dst = bkv_vmem_ref.at[slot, pl.ds(0, n_pages)]
            pltpu.make_async_copy(dst, dst, bkv_sems.at[slot]).wait()

    def _store_kv_write_pair(w_idx, dst_pg, pair_r, packed_1x128):
        if interpret_mode:
            b0 = packed_1x128.astype(jnp.uint8)
            b1 = (packed_1x128 >> 8).astype(jnp.uint8)
            b2 = (packed_1x128 >> 16).astype(jnp.uint8)
            b3 = (packed_1x128 >> 24).astype(jnp.uint8)
            row_f8 = lax.bitcast_convert_type(
                jnp.stack([b0, b1, b2, b3], axis=1)[:, None, :, :],
                jnp.float8_e4m3fn,
            )
            kv_write_vmem_ref[w_idx] = jnp.broadcast_to(row_f8, (8, 1, 4, 128))
        else:
            w_u32_ref = kv_write_vmem_ref.bitcast(jnp.uint32).reshape(-1, 8, 128)
            w_u32_ref[w_idx] = jnp.broadcast_to(packed_1x128, (8, 128))
        pltpu.make_async_copy(
            kv_write_vmem_ref.at[w_idx, pl.ds(0, 1)],
            updated_kv_cache_hbm_ref.at[dst_pg, pl.ds(pair_r, 1)],
            cache_write_sem.at[0],
        ).start()

    def _read_bkv_row_u32(slot, slot_pg, row_idx):
        if probe == "nodma":
            return jnp.zeros((1, 128), jnp.uint32)
        if interpret_mode:
            pg_u32 = _get_page_u32(slot, slot_pg)
            r_io = lax.broadcasted_iota(jnp.int32, (phys_page_rows, 128), 0)
            return jnp.sum(
                jnp.where(
                    r_io == row_idx,
                    pg_u32.astype(jnp.int32),
                    jnp.int32(0),
                ),
                axis=0,
                keepdims=True,
            ).astype(jnp.uint32)
        sub_base = (row_idx // 8) * 8
        sub_off = row_idx & 7
        u32_ref = (
            bkv_vmem_ref.bitcast(jnp.uint32)
            .at[slot]
            .reshape(bkv_slot_pages, phys_page_rows, 128)
        )
        sub_u32 = u32_ref[slot_pg, pl.ds(pl.multiple_of(sub_base, 8), 8)]
        r_sub = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
        return jnp.sum(
            jnp.where(
                r_sub == sub_off,
                sub_u32.astype(jnp.int32),
                jnp.int32(0),
            ),
            axis=0,
            keepdims=True,
        ).astype(jnp.uint32)

    def enqueue_kv_write(
        seq,
        q_row,
        active,
        kv_len,
        slot=jnp.int32(0),
        prev_loc_tok=jnp.int32(0),
        slot_pg_off=jnp.int32(0),
    ):
        if skip_kv_write or probe == "nokvwrite":
            return

        @pl.when(active)
        def _():
            seq_base = cdiv(cu_kv_lens_ref[seq], page_size)
            if kv_pack_bd4:
                lane_lt_64 = lax.broadcasted_iota(jnp.int32, (1, 128), 1) < 64
                u16_ev = [
                    jnp.where(
                        lane_lt_64,
                        _load_new_kv_1row_u16(
                            kv_merged_vmem_ref, q_row + j, interpret_mode=interpret_mode
                        ),
                        jnp.uint32(0),
                    )
                    for j in range(dq)
                ]
                u16_od = [
                    jnp.where(
                        lane_lt_64,
                        jnp.uint32(0),
                        _roll_lanes_fast(
                            u16_ev[j], 64, interpret_mode=interpret_mode
                        ),
                    )
                    for j in range(dq)
                ]
                if dq == 1:
                    pos = kv_len - 1
                    pg_off = seq_base + (pos // page_size)
                    dst_pg = page_indices_ref[
                        jnp.minimum(pg_off, num_page_indices - 1)
                    ]
                    phys_r = (pos % page_size) // 4
                    slot4 = pos & 3
                    w_cur = written_cnt_smem_ref[0]

                    @pl.when(slot4 == 0)
                    def _w_bd4_s0():
                        _store_kv_write_pair(w_cur, dst_pg, phys_r, u16_ev[0])

                    @pl.when(slot4 != 0)
                    def _w_bd4_rmw():
                        loc_clamped = jnp.maximum(prev_loc_tok, 0)
                        slot_pg = slot_pg_off + jnp.clip(
                            loc_clamped // page_size, 0, max_ring_pages - 1
                        )
                        old_u32 = _read_bkv_row_u32(slot, slot_pg, phys_r)
                        keep_lo = jnp.where(
                            (slot4 >= 2) | ((slot4 == 1) & lane_lt_64),
                            jnp.uint32(0x0000FFFF),
                            jnp.uint32(0),
                        )
                        keep_hi = jnp.where(
                            (slot4 == 3) & lane_lt_64,
                            jnp.uint32(0xFFFF0000),
                            jnp.uint32(0),
                        )
                        new_v = jnp.where(
                            (slot4 & 1) != 0, u16_od[0], u16_ev[0]
                        )
                        new_sh = jnp.where(
                            (slot4 & 2) != 0, new_v << jnp.uint32(16), new_v
                        )
                        _store_kv_write_pair(
                            w_cur, dst_pg, phys_r, (old_u32 & (keep_lo | keep_hi)) | new_sh
                        )

                    written_cnt_smem_ref[0] = w_cur + 1
                    return

                cache_kv = kv_len - dq
                s0 = cache_kv & 3
                loc_clamped = jnp.maximum(prev_loc_tok, 0)
                slot_pg = slot_pg_off + jnp.clip(
                    loc_clamped // page_size, 0, max_ring_pages - 1
                )
                phys_r0 = ((jnp.maximum(cache_kv - 1, 0)) % page_size) // 4
                old_u32_0 = jnp.where(
                    s0 != 0,
                    _read_bkv_row_u32(slot, slot_pg, phys_r0),
                    jnp.uint32(0),
                )
                n_row_iters = (3 + dq + 3) // 4
                for r_i in range(n_row_iters):
                    row_tok_base = (cache_kv // 4 + r_i) * 4
                    row_act = row_tok_base < kv_len

                    @pl.when(row_act)
                    def _emit_bd4_row(r_i=r_i, row_tok_base=row_tok_base):
                        if r_i == 0:
                            k_lo = jnp.where(
                                (s0 >= 2) | ((s0 == 1) & lane_lt_64),
                                jnp.uint32(0x0000FFFF),
                                jnp.uint32(0),
                            )
                            k_hi = jnp.where(
                                (s0 == 3) & lane_lt_64,
                                jnp.uint32(0xFFFF0000),
                                jnp.uint32(0),
                            )
                            acc_u32 = old_u32_0 & (k_lo | k_hi)
                        else:
                            acc_u32 = jnp.zeros((1, 128), jnp.uint32)
                        for j in range(dq):
                            tok_j = cache_kv + j
                            acc_u32 = jnp.where(
                                tok_j == row_tok_base, acc_u32 | u16_ev[j], acc_u32
                            )
                            acc_u32 = jnp.where(
                                tok_j == row_tok_base + 1, acc_u32 | u16_od[j], acc_u32
                            )
                            acc_u32 = jnp.where(
                                tok_j == row_tok_base + 2,
                                acc_u32 | (u16_ev[j] << jnp.uint32(16)),
                                acc_u32,
                            )
                            acc_u32 = jnp.where(
                                tok_j == row_tok_base + 3,
                                acc_u32 | (u16_od[j] << jnp.uint32(16)),
                                acc_u32,
                            )
                        pg_off = seq_base + (row_tok_base // page_size)
                        dst_pg = page_indices_ref[
                            jnp.minimum(pg_off, num_page_indices - 1)
                        ]
                        phys_r = (row_tok_base % page_size) // 4
                        w_cur = written_cnt_smem_ref[0]
                        _store_kv_write_pair(w_cur, dst_pg, phys_r, acc_u32)
                        written_cnt_smem_ref[0] = w_cur + 1

                return

            if not kv_pack_dense2:
                for j in range(dq):
                    pos = kv_len - dq + j
                    pg_off = seq_base + (pos // page_size)
                    dst_pg = page_indices_ref[
                        jnp.minimum(pg_off, num_page_indices - 1)
                    ]
                    pltpu.make_async_copy(
                        kv_merged_vmem_ref.at[pl.ds(q_row + j, 1)],
                        updated_kv_cache_hbm_ref.at[dst_pg, pl.ds(pos % page_size, 1)],
                        cache_write_sem.at[0],
                    ).start()
                written_cnt_smem_ref[0] = written_cnt_smem_ref[0] + dq
                return

            cache_kv = kv_len - dq
            u16_list = [
                _load_new_kv_1row_u16(
                    kv_merged_vmem_ref, q_row + j, interpret_mode=interpret_mode
                )
                for j in range(dq)
            ]
            is_odd_start = (cache_kv & 1) == 1

            @pl.when(~is_odd_start)
            def _write_even_start():
                w_cur = written_cnt_smem_ref[0]
                n_pairs = (dq + 1) // 2
                for p_i in range(n_pairs):
                    j_e = 2 * p_i
                    pos_e = cache_kv + j_e
                    pg_off = seq_base + (pos_e // page_size)
                    dst_pg = page_indices_ref[
                        jnp.minimum(pg_off, num_page_indices - 1)
                    ]
                    pair_r = (pos_e % page_size) // 2
                    packed = (
                        u16_list[j_e] | (u16_list[j_e + 1] << jnp.uint32(16))
                        if (j_e + 1 < dq)
                        else u16_list[j_e]
                    )
                    _store_kv_write_pair(w_cur + p_i, dst_pg, pair_r, packed)
                written_cnt_smem_ref[0] = w_cur + n_pairs

            @pl.when(is_odd_start)
            def _write_odd_start():
                w_cur = written_cnt_smem_ref[0]
                pos_prev = cache_kv - 1
                pg_off0 = seq_base + (pos_prev // page_size)
                dst_pg0 = page_indices_ref[jnp.minimum(pg_off0, num_page_indices - 1)]
                pair_r0 = (pos_prev % page_size) // 2
                if probe == "nodma":
                    prev_u16 = jnp.zeros((1, 128), jnp.uint32)
                elif interpret_mode:
                    loc_clamped = jnp.maximum(prev_loc_tok, 0)
                    slot_pg = slot_pg_off + jnp.clip(loc_clamped // page_size, 0, max_ring_pages - 1)
                    pg_u32 = _get_page_u32(slot, slot_pg)
                    r_io = lax.broadcasted_iota(jnp.int32, (phys_page_rows, 128), 0)
                    prev_u16 = jnp.sum(
                        jnp.where(
                            r_io == pair_r0,
                            (pg_u32 & jnp.uint32(0xFFFF)).astype(jnp.int32),
                            jnp.int32(0),
                        ),
                        axis=0,
                        keepdims=True,
                    ).astype(jnp.uint32)
                else:
                    loc_clamped = jnp.maximum(prev_loc_tok, 0)
                    slot_pg = slot_pg_off + jnp.clip(loc_clamped // page_size, 0, max_ring_pages - 1)
                    sub_base = (pair_r0 // 8) * 8
                    sub_off = pair_r0 & 7
                    u32_ref = (
                        bkv_vmem_ref.bitcast(jnp.uint32)
                        .at[slot]
                        .reshape(bkv_slot_pages, phys_page_rows, 128)
                    )
                    sub_u32 = u32_ref[slot_pg, pl.ds(pl.multiple_of(sub_base, 8), 8)]
                    r_sub = lax.broadcasted_iota(jnp.int32, (8, 128), 0)
                    prev_u16 = jnp.sum(
                        jnp.where(
                            r_sub == sub_off,
                            (sub_u32 & jnp.uint32(0xFFFF)).astype(jnp.int32),
                            jnp.int32(0),
                        ),
                        axis=0,
                        keepdims=True,
                    ).astype(jnp.uint32)
                _store_kv_write_pair(
                    w_cur, dst_pg0, pair_r0, prev_u16 | (u16_list[0] << jnp.uint32(16))
                )
                rem = dq - 1
                n_tail = (rem + 1) // 2
                for t_i in range(n_tail):
                    j_e = 1 + 2 * t_i
                    pos_e = cache_kv + j_e
                    pg_off = seq_base + (pos_e // page_size)
                    dst_pg = page_indices_ref[
                        jnp.minimum(pg_off, num_page_indices - 1)
                    ]
                    pair_r = (pos_e % page_size) // 2
                    packed = (
                        u16_list[j_e] | (u16_list[j_e + 1] << jnp.uint32(16))
                        if (j_e + 1 < dq)
                        else u16_list[j_e]
                    )
                    _store_kv_write_pair(w_cur + 1 + t_i, dst_pg, pair_r, packed)
                written_cnt_smem_ref[0] = w_cur + 1 + n_tail

    def precompute_hoisted_new_kv():
        """Vectorized whole-batch (s_new, acc0) precompute overlapped under initial page DMAs."""
        b_pad = kv_merged_vmem_ref.shape[0]
        step_b = min(32, b_pad)
        for b0 in range(0, b_pad, step_b):
            cur_b = min(step_b, max_num_tokens - b0)
            read_b = min(step_b, b_pad - b0)
            if cur_b <= 0:
                break
            if interpret_mode:
                blk = kv_merged_vmem_ref[pl.ds(b0, read_b)].reshape(read_b, 4, 128)
                u8 = lax.bitcast_convert_type(blk, jnp.uint8)
                u32 = u8[:, 0, :].astype(jnp.uint32) | (
                    u8[:, 1, :].astype(jnp.uint32) << 8
                )
            else:
                u32_ref = kv_merged_vmem_ref.bitcast(jnp.uint32).reshape(b_pad, 128)
                u32 = u32_ref[pl.ds(b0, read_b)]
            if kv_pack_bd4:
                # Mask padded d=64..127 lanes before FP8->F32 so caller NaNs cannot yield 0.0*NaN.
                u32 = jnp.where(
                    lax.broadcasted_iota(jnp.int32, u32.shape, 1) < 64,
                    u32,
                    jnp.uint32(0),
                )
            k_f32 = _u8_to_fp8(
                u32.astype(jnp.uint8), interpret_mode=interpret_mode
            ).astype(jnp.float32)
            v_f32 = (
                _u8_to_fp8(
                    (u32 >> 8).astype(jnp.uint8), interpret_mode=interpret_mode
                ).astype(jnp.float32)
                * v_scale_val
            )
            q_blk = q_vmem_ref[pl.ds(b0, cur_b), :q_out_head_rows, :].astype(
                jnp.float32
            )
            s_blk = _broadcast_minor_128(
                jnp.sum(q_blk * k_f32[:cur_b, None, :], axis=2, keepdims=True)
            )
            v_blk = jnp.broadcast_to(
                v_f32[:cur_b, None, :], (cur_b, q_out_head_rows, 128)
            )
            init_sv_vmem_ref[0, pl.ds(b0, cur_b)] = s_blk
            init_sv_vmem_ref[1, pl.ds(b0, cur_b)] = v_blk

    def init_new_token_state(q_bf16, q_row):
        if dq == 1:
            if use_hoist_kv:
                return (
                    init_sv_vmem_ref[0, q_row],
                    jnp.full((q_out_rows, 128), 1.0 / 128.0, jnp.float32),
                    init_sv_vmem_ref[1, q_row],
                )
            k_new_f32, v_new_f32 = _load_new_kv_1row_f32(
                kv_merged_vmem_ref,
                q_row,
                interpret_mode=interpret_mode,
                mask_lo64=kv_pack_bd4,
            )
            q_f32 = q_bf16[:q_out_rows].astype(jnp.float32)
            s_new = _broadcast_minor_128(
                jnp.sum(
                    q_f32 * jnp.broadcast_to(k_new_f32, (q_out_rows, 128)),
                    axis=1,
                    keepdims=True,
                )
            )
            l_vec0 = jnp.full((q_out_rows, 128), 1.0 / 128.0, jnp.float32)
            acc0 = jnp.broadcast_to(v_new_f32, (q_out_rows, 128)) * v_scale_val
            return s_new, l_vec0, acc0

        kv_list = [
            _load_new_kv_1row_f32(
                kv_merged_vmem_ref,
                q_row + j,
                interpret_mode=interpret_mode,
                mask_lo64=kv_pack_bd4,
            )
            for j in range(dq)
        ]
        m_list, l_list, a_list = [], [], []
        for i in range(dq):
            q_i = q_bf16[
                i * q_head_rows : i * q_head_rows + q_out_head_rows
            ].astype(jnp.float32)
            s_js = []
            m_i = None
            for j in range(i + 1):
                kj, _ = kv_list[j]
                s_ij = _broadcast_minor_128(
                    jnp.sum(
                        q_i * jnp.broadcast_to(kj, (q_out_head_rows, 128)),
                        axis=1,
                        keepdims=True,
                    )
                )
                s_js.append(s_ij)
                m_i = s_ij if m_i is None else jnp.maximum(m_i, s_ij)
            p_sum_i = jnp.zeros((q_out_head_rows, 128), jnp.float32)
            acc_i = jnp.zeros((q_out_head_rows, 128), jnp.float32)
            for j in range(i + 1):
                _, vj = kv_list[j]
                p_ij = jnp.exp2((s_js[j] - m_i) * exp2_scale)
                p_sum_i = p_sum_i + p_ij
                acc_i = acc_i + p_ij * (
                    jnp.broadcast_to(vj, (q_out_head_rows, 128)) * v_scale_val
                )
            m_list.append(m_i)
            l_list.append(p_sum_i * jnp.float32(1.0 / 128.0))
            a_list.append(acc_i)
        return (
            jnp.concatenate(m_list, axis=0),
            jnp.concatenate(l_list, axis=0),
            jnp.concatenate(a_list, axis=0),
        )

    def finalize_out(l_vec: jax.Array, acc: jax.Array) -> jax.Array:
        l_sum = _broadcast_minor_128(jnp.sum(l_vec, axis=1, keepdims=True))
        if interpret_mode:
            return (acc / l_sum).astype(jnp.bfloat16)
        return (acc * pltpu.reciprocal(l_sum, approx=True)).astype(jnp.bfloat16)

    def _get_page_u32(slot, pg_idx):
        if interpret_mode:
            blk = bkv_vmem_ref[slot, pg_idx].reshape(phys_page_rows, 4, 128)
            u8 = lax.bitcast_convert_type(blk, jnp.uint8)
            if kv_pack_dense2 or kv_pack_bd4:
                return (
                    u8[:, 0, :].astype(jnp.uint32)
                    | (u8[:, 1, :].astype(jnp.uint32) << 8)
                    | (u8[:, 2, :].astype(jnp.uint32) << 16)
                    | (u8[:, 3, :].astype(jnp.uint32) << 24)
                )
            return u8[:, 0, :].astype(jnp.uint32) | (
                u8[:, 1, :].astype(jnp.uint32) << 8
            )
        u32_ref = (
            bkv_vmem_ref.bitcast(jnp.uint32)
            .at[slot]
            .reshape(bkv_slot_pages, phys_page_rows, 128)
        )
        return u32_ref[pg_idx]

    def _set_page_u32(slot, pg_idx, val_u32):
        if interpret_mode:
            if kv_pack_dense2 or kv_pack_bd4:
                b0 = val_u32.astype(jnp.uint8)
                b1 = (val_u32 >> 8).astype(jnp.uint8)
                b2 = (val_u32 >> 16).astype(jnp.uint8)
                b3 = (val_u32 >> 24).astype(jnp.uint8)
                packed = jnp.stack([b0, b1, b2, b3], axis=1)[:, None, :, :]
            else:
                k_u8 = val_u32.astype(jnp.uint8)
                v_u8 = (val_u32 >> 8).astype(jnp.uint8)
                z_u8 = jnp.zeros((phys_page_rows, 128), jnp.uint8)
                packed = jnp.stack([k_u8, v_u8, z_u8, z_u8], axis=1)[:, None, :, :]
            bkv_vmem_ref[slot, pg_idx] = lax.bitcast_convert_type(
                packed, jnp.float8_e4m3fn
            )
            return
        u32_ref = (
            bkv_vmem_ref.bitcast(jnp.uint32)
            .at[slot]
            .reshape(bkv_slot_pages, phys_page_rows, 128)
        )
        u32_ref[pg_idx] = val_u32

    def _mask_slice_u32(blk_u32, off_r: int, n_rows: int, lo, hi):
        r_pg = off_r + lax.broadcasted_iota(jnp.int32, (n_rows, 128), 0)
        if kv_pack_bd4:
            lo_h = lax.broadcasted_iota(jnp.int32, (n_rows, 128), 1) < 64
            t0 = 4 * r_pg + jnp.where(lo_h, 0, 1)
            t1 = 4 * r_pg + jnp.where(lo_h, 2, 3)
            m0 = jnp.where(
                (t0 >= lo) & (t0 < hi), jnp.uint32(0x0000FFFF), jnp.uint32(0)
            )
            m1 = jnp.where(
                (t1 >= lo) & (t1 < hi), jnp.uint32(0xFFFF0000), jnp.uint32(0)
            )
            return blk_u32 & (m0 | m1)
        if not kv_pack_dense2:
            return jnp.where(
                (r_pg >= lo) & (r_pg < hi),
                blk_u32,
                jnp.zeros((n_rows, 128), jnp.uint32),
            )
        e_idx = 2 * r_pg
        o_idx = e_idx + 1
        e_m = jnp.where(
            (e_idx >= lo) & (e_idx < hi), jnp.uint32(0x0000FFFF), jnp.uint32(0)
        )
        o_m = jnp.where(
            (o_idx >= lo) & (o_idx < hi), jnp.uint32(0xFFFF0000), jnp.uint32(0)
        )
        return blk_u32 & (e_m | o_m)

    def _sanitize_page_u32(slot, pg_idx, lo, hi):
        # Chunk >128-row physical pages (ps=256 nod2, ps=512 d2/nod2) into 128-row
        # VREG slices so iota/mask temporaries never blow register pressure.
        if interpret_mode or phys_page_rows <= 128:
            _set_page_u32(
                slot,
                pg_idx,
                _mask_slice_u32(_get_page_u32(slot, pg_idx), 0, phys_page_rows, lo, hi),
            )
            return
        u32_flat = (
            bkv_vmem_ref.bitcast(jnp.uint32)
            .at[slot]
            .reshape(bkv_slot_pages * phys_page_rows, 128)
        )
        base_r = pg_idx * phys_page_rows
        n_subs = phys_page_rows // 128
        iota_128 = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
        if not kv_pack_dense2:
            z_128 = jnp.zeros((128, 128), jnp.uint32)
            for sub_i in range(n_subs):
                row_off = sub_i * 128
                src_off = pl.multiple_of(base_r + row_off, 128)
                blk = u32_flat[pl.ds(src_off, 128)]
                cond = (iota_128 >= (lo - row_off)) & (iota_128 < (hi - row_off))
                u32_flat[pl.ds(src_off, 128)] = jnp.where(cond, blk, z_128)
            return

        e_base = 2 * iota_128
        o_base = e_base + 1
        for sub_i in range(n_subs):
            row_off = sub_i * 128
            tok_off = 2 * row_off
            src_off = pl.multiple_of(base_r + row_off, 128)
            blk = u32_flat[pl.ds(src_off, 128)]
            lo_loc = lo - tok_off
            hi_loc = hi - tok_off
            e_m = jnp.where(
                (e_base >= lo_loc) & (e_base < hi_loc), jnp.uint32(0x0000FFFF), jnp.uint32(0)
            )
            o_m = jnp.where(
                (o_base >= lo_loc) & (o_base < hi_loc), jnp.uint32(0xFFFF0000), jnp.uint32(0)
            )
            u32_flat[pl.ds(src_off, 128)] = blk & (e_m | o_m)

    def _zero_pages_u32(slot, start_pg, end_pg):
        _zero_bkv_pages_u32(
            bkv_vmem_ref,
            slot,
            start_pg,
            end_pg,
            max_ring_pages=bkv_slot_pages,
            phys_page_rows=phys_page_rows,
            kv_pack_dense2=kv_pack_dense2,
            interpret_mode=interpret_mode,
        )

    def zero_all_slots_once():
        _zero_all_bkv_slots_u32(
            bkv_vmem_ref,
            num_slots=num_slots,
            max_ring_pages=bkv_slot_pages,
            phys_page_rows=phys_page_rows,
            interpret_mode=interpret_mode,
        )

    def sanitize_slot_boundaries_branchless(
        slot, rel_first, rel_end, active_pages: int
    ):
        """2-page VMEM sanitizer skipping Page 0 when ``rel_first == 0`` (all 1/1k SWA).

        With ``zero_all_slots_once()`` run before the loop, every slot starts finite
        (``0.0``), and every DMA'd sequence sanitizes Page 0 (only when ``rel_first > 0``
        and ``last_p > 0``) and Page ``last_p`` in place so no ``0x7F``/``0xFF`` FP8 NaN
        can ever enter or persist in ``slot``.
        """
        n_pg = (rel_end + page_size - 1) // page_size
        last_p = jnp.clip(n_pg - 1, 0, active_pages - 1)
        if sliding_window is not None and active_pages * page_size >= sliding_window:
            _sanitize_page_u32(slot, 0, rel_first, rel_end)
        if cfg.lazy_slot_zero:
            hwm_b0 = written_cnt_smem_ref[2 + slot]
            start_b0 = jnp.maximum(n_pg, hwm_b0)

            @pl.when(start_b0 < active_pages)
            def _lazy_init_b0():
                _zero_pages_u32(slot, start_b0, active_pages)

            written_cnt_smem_ref[2 + slot] = jnp.maximum(
                hwm_b0, jnp.maximum(n_pg, active_pages)
            )

        last_lo = jnp.where(last_p == 0, rel_first, 0)
        last_hi = jnp.clip(rel_end - last_p * page_size, 0, page_size)
        _sanitize_page_u32(slot, last_p, last_lo, last_hi)

    def sanitize_slot_boundaries(slot, rel_first, rel_end, active_pages: int):
        """Zeroes <=2 boundary pages + any unfetched tail pages in VMEM (0x7F NaN-safe)."""
        if use_branchless_sanitize:
            sanitize_slot_boundaries_branchless(slot, rel_first, rel_end, active_pages)
            return
        _sanitize_page_u32(slot, 0, rel_first, rel_end)
        n_pg = (rel_end + page_size - 1) // page_size
        rem_last = rel_end % page_size

        @pl.when((n_pg > 1) & (rem_last != 0))
        def _clean_last():
            last_p = jnp.minimum(n_pg - 1, active_pages - 1)
            _sanitize_page_u32(slot, last_p, 0, rem_last)

        start_zero_pg = jnp.maximum(n_pg, 1)

        @pl.when(start_zero_pg < active_pages)
        def _clean_tail_pages():
            _zero_pages_u32(slot, start_zero_pg, active_pages)

    def _bkv_fp8_chunk(
        slot,
        row_start: int,
        chunk_sz: int,
        *,
        kind: str,
        valid_tokens: jax.Array | int | None = None,
    ) -> jax.Array:
        return _load_bkv_fp8_chunk(
            bkv_vmem_ref,
            slot,
            row_start,
            chunk_sz,
            kind=kind,
            page_size=page_size,
            kv_pack_dense2=kv_pack_dense2,
            kv_pack_bd4=kv_pack_bd4,
            interpret_mode=interpret_mode,
            valid_tokens=valid_tokens,
        )

    def _to_bd4_q(q_raw):
        if (not kv_pack_bd4) or (q_raw.shape[0] % 16 == 0 and q_head_rows == 16):
            return q_raw
        # Step (c) fused caller may pass (8*dq, 128); tile into [q8, 0; 0, q8].
        lane_lt = lax.broadcasted_iota(jnp.int32, (8, 128), 1) < 64
        z8 = jnp.zeros((8, 128), q_raw.dtype)
        parts = []
        n_q = q_raw.shape[0] // 8
        for qi in range(n_q):
            q8 = q_raw[qi * 8 : (qi + 1) * 8]
            q_top = jnp.where(lane_lt, q8, z8)
            q_bot = jnp.where(
                lane_lt, z8, _roll_lanes_fast(q8, 64, interpret_mode=interpret_mode)
            )
            parts.append(q_top)
            parts.append(q_bot)
        return jnp.concatenate(parts, axis=0)

    def _bd4_logits_split(q_bd_dot, k_pair, n_seq: int, half_w: int):
        s_2 = jnp.matmul(q_bd_dot, k_pair.T, preferred_element_type=jnp.float32)
        if n_seq == 1:
            return s_2[:8, :], s_2[8:, :]
        s_4d = s_2.reshape(n_seq, 2, 8, half_w)
        return (
            s_4d[:, 0, :, :].reshape(n_seq * 8, half_w),
            s_4d[:, 1, :, :].reshape(n_seq * 8, half_w),
        )

    def _bd4_softmax_reduce(
        s_e,
        s_o,
        m_prev,
        eff_sc,
        half_w: int,
        rows: int,
        *,
        use_page_reduce: bool = True,
    ):
        s_max = jnp.maximum(s_e, s_o)
        use_f128 = cfg.fold_128 and (half_w >= 128) and (half_w % 128 == 0)
        use_fcol = cfg.fold_col and use_f128
        if use_f128:
            m_new = jnp.maximum(
                m_prev,
                _broadcast_minor_128(
                    jnp.max(_fold128_op(s_max, jnp.maximum), axis=1, keepdims=True)
                ),
            )
            alpha = jnp.exp2((m_prev - m_new) * eff_sc)
            if use_fcol:
                p_e, l_e = _exp2_fold_col(s_e, m_new, eff_sc)
                p_o, l_o = _exp2_fold_col(s_o, m_new, eff_sc)
                l_add = l_e + l_o
            else:
                m_1 = m_new[:, :1]
                p_e = jnp.exp2((s_e - m_1) * eff_sc)
                p_o = jnp.exp2((s_o - m_1) * eff_sc)
                l_add = _fold128_op(p_e + p_o, jnp.add)
        elif use_page_reduce and (half_w >= 128) and (half_w % 128 == 0):
            n_hb = half_w // 128
            s_pgs = jnp.max(s_max.reshape(rows, n_hb, 128), axis=1)
            m_new = jnp.maximum(
                m_prev, _broadcast_minor_128(jnp.max(s_pgs, axis=1, keepdims=True))
            )
            alpha = jnp.exp2((m_prev - m_new) * eff_sc)
            m_hw = _broadcast_minor_128_any(m_new, half_w)
            p_e = jnp.exp2((s_e - m_hw) * eff_sc)
            p_o = jnp.exp2((s_o - m_hw) * eff_sc)
            l_add = jnp.sum((p_e + p_o).reshape(rows, n_hb, 128), axis=1)
        else:
            m_new = jnp.maximum(
                m_prev, _broadcast_minor_128(jnp.max(s_max, axis=1, keepdims=True))
            )
            alpha = jnp.exp2((m_prev - m_new) * eff_sc)
            m_hw = _broadcast_minor_128_any(m_new, half_w)
            p_e = jnp.exp2((s_e - m_hw) * eff_sc)
            p_o = jnp.exp2((s_o - m_hw) * eff_sc)
            l_add = (
                _broadcast_minor_128(jnp.sum(p_e + p_o, axis=1, keepdims=True))
                * jnp.float32(1.0 / 128.0)
            )
        return m_new, alpha, p_e, p_o, l_add

    def _bd4_pv_reduce(p_e, p_o, v_pair, *, n_seq: int = 1, p_bf16: bool = False):
        if n_seq == 1:
            p_bd = jnp.concatenate([p_e, p_o], axis=0)
            p_dot = p_bd.astype(jnp.bfloat16) if p_bf16 else p_bd
            o_bd = jnp.matmul(p_dot, v_pair, preferred_element_type=jnp.float32)
            return o_bd[:8] + _roll_lanes_fast(
                o_bd[8:], 64, interpret_mode=interpret_mode
            )
        half_w = p_e.shape[1]
        p_bd = jnp.concatenate(
            [p_e.reshape(n_seq, 8, half_w), p_o.reshape(n_seq, 8, half_w)], axis=1
        ).reshape(n_seq * 16, half_w)
        p_dot = p_bd.astype(jnp.bfloat16) if p_bf16 else p_bd
        o_bd = jnp.matmul(p_dot, v_pair, preferred_element_type=jnp.float32).reshape(
            n_seq, 2, 8, 128
        )
        o_odd = _roll_lanes_fast(
            o_bd[:, 1].reshape(n_seq * 8, 128), 64, interpret_mode=interpret_mode
        ).reshape(n_seq, 8, 128)
        return (o_bd[:, 0] + o_odd).reshape(n_seq * 8, 128)

    def _bd4_step_blk(
        slot,
        q_bd_dot,
        rf_rows,
        rel_end,
        r0,
        w_k: int,
        carry,
        *,
        eff_sc=None,
        mask_bounds: bool = True,
        allow_fp8_dots: bool = False,
        use_page_reduce: bool = True,
        use_pbf16: bool = False,
    ):
        m_prev, l_prev, acc_prev = carry
        half_w = w_k // 2
        pad_rem = half_w % 128
        pad_rows = (128 - pad_rem) if pad_rem else 0
        eff_hw = half_w + pad_rows
        q_bd = _to_bd4_q(q_bd_dot)
        k_pair = _load_bkv_bd4_kv(
            bkv_vmem_ref,
            slot,
            r0,
            w_k,
            kind="k",
            page_size=page_size,
            interpret_mode=interpret_mode,
        )
        if pad_rows:
            k_pair = jnp.concatenate(
                [k_pair, jnp.zeros((pad_rows, 128), k_pair.dtype)], axis=0
            )
        s_e, s_o = _bd4_logits_split(q_bd, k_pair, dq, eff_hw)
        valid_pad = (
            lax.broadcasted_iota(jnp.int32, (q_out_rows, eff_hw), 1) < half_w
            if pad_rows
            else True
        )
        if mask_bounds:
            c_e = _bd4_c_even_iota(r0, half_w, q_out_rows, eff_hw=eff_hw)
            c_o = c_e + 1
            if sliding_window is not None:
                cond_e = (c_e >= rf_rows) & (c_e < rel_end) & valid_pad
                cond_o = (c_o >= rf_rows) & (c_o < rel_end) & valid_pad
            else:
                cond_e = (c_e < rel_end) & valid_pad
                cond_o = (c_o < rel_end) & valid_pad
            s_e = jnp.where(cond_e, s_e, _MASK_LOGIT)
            s_o = jnp.where(cond_o, s_o, _MASK_LOGIT)
        elif pad_rows:
            s_e = jnp.where(valid_pad, s_e, _MASK_LOGIT)
            s_o = jnp.where(valid_pad, s_o, _MASK_LOGIT)
        sc_val = exp2_scale if eff_sc is None else eff_sc
        m_new, alpha, p_e, p_o, l_add = _bd4_softmax_reduce(
            s_e, s_o, m_prev, sc_val, eff_hw, q_out_rows, use_page_reduce=use_page_reduce
        )
        v_pair = _load_bkv_bd4_kv(
            bkv_vmem_ref,
            slot,
            r0,
            w_k,
            kind="v",
            page_size=page_size,
            interpret_mode=interpret_mode,
        )
        if pad_rows:
            v_pair = jnp.concatenate(
                [v_pair, jnp.zeros((pad_rows, 128), v_pair.dtype)], axis=0
            )
        p_bf16 = use_pbf16 or cfg.pv_bf16 or (allow_fp8_dots and cfg.fp8_dots)
        pv = _bd4_pv_reduce(p_e, p_o, v_pair, n_seq=dq, p_bf16=p_bf16)
        if v_scale is not None and v_scale_val != 1.0:
            pv = pv * v_scale_val
        return (m_new, l_prev * alpha + l_add, acc_prev * alpha + pv)

    def _bkv_fp8_split(
        slot, row_start: int, chunk_sz: int, *, kind: str
    ) -> tuple[jax.Array, jax.Array]:
        return _load_bkv_fp8_split(
            bkv_vmem_ref,
            slot,
            row_start,
            chunk_sz,
            kind=kind,
            page_size=page_size,
            interpret_mode=interpret_mode,
        )

    def _col_iota(row_start, width: int, rows: int) -> jax.Array:
        return _bkv_col_iota(
            row_start, width, rows, kv_pack_dense2=kv_pack_dense2
        )

    def _dense2_pv_half(
        slot,
        row_start: int,
        width: int,
        s_x: jax.Array,
        m_ref: jax.Array,
        eff_sc: jax.Array | float,
        odd: bool,
        *,
        use_page_reduce: bool = False,
        allow_fp8_dots: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        half_w = width // 2
        n_hb = half_w // 128
        if cfg.fold_col:
            p_x, l_x = _exp2_fold_col(s_x, m_ref, eff_sc)
        elif cfg.fold_128:
            p_x = jnp.exp2((s_x - m_ref[:, :1]) * eff_sc)
            l_x = _fold128_op(p_x, jnp.add)
        elif use_page_reduce:
            m_hw = _broadcast_minor_128_any(m_ref, half_w)
            p_x = jnp.exp2((s_x - m_hw) * eff_sc)
            l_x = jnp.sum(p_x.reshape(q_rows, n_hb, 128), axis=1)
        else:
            m_hw = _broadcast_minor_128_any(m_ref, half_w)
            p_x = jnp.exp2((s_x - m_hw) * eff_sc)
            l_x = jnp.sum(p_x, axis=1, keepdims=True)
        p_dot = p_x.astype(jnp.bfloat16) if (cfg.pv_bf16 or (allow_fp8_dots and cfg.fp8_dots)) else p_x
        v_x = _load_bkv_fp8_split_half(
            bkv_vmem_ref,
            slot,
            row_start,
            width,
            kind="v",
            odd=odd,
            page_size=page_size,
            interpret_mode=interpret_mode,
        )
        return l_x, jnp.matmul(p_dot, v_x, preferred_element_type=jnp.float32)

    def compute_swa_clean1p_2176(slot, q_bf16, q_row, rel_first, rel_end):
        """Zero-spill 1-shot SWA softmax over 17x128 (or 8x256 + 1x128) K/V chunks."""
        s_new, l0, acc0 = init_new_token_state(q_bf16, q_row)
        if probe == "nocompute":
            return acc0.astype(jnp.bfloat16)

        sanitize_slot_boundaries(slot, rel_first, rel_end, 17)
        if kv_pack_bd4:
            if cfg.fp8_dots:
                q_f32_d = q_bf16.astype(jnp.float32)
                q_mx = jnp.maximum(jnp.max(jnp.abs(q_f32_d)), jnp.float32(1e-6))
                q_gain = jnp.float32(256.0) / q_mx
                q_dot = (q_f32_d * q_gain).astype(jnp.float8_e4m3fn)
                s_new_eff = s_new * q_gain
                exp2_sc_eff = jnp.float32(exp2_scale) / q_gain
            else:
                q_dot = q_bf16
                s_new_eff = s_new
                exp2_sc_eff = exp2_scale
            st = (s_new_eff, l0, acc0)
            for c_start, c_w in ((0, 1024), (1024, 1024), (2048, 128)):
                st = _bd4_step_blk(
                    slot,
                    q_dot,
                    rel_first,
                    rel_end,
                    c_start,
                    c_w,
                    st,
                    eff_sc=exp2_sc_eff,
                    allow_fp8_dots=True,
                    use_page_reduce=cfg.page_reduce,
                )
            return finalize_out(st[1], st[2])
        s_list = []
        chunk_specs = (
            [(pg_i * 128, 128) for pg_i in range(17)]
            if kv_pack_dense2
            else ([(c_i * 256, 256) for c_i in range(8)] + [(2048, 128)])
        )
        for c_start, c_w in chunk_specs:
            k_c = _bkv_fp8_chunk(slot, c_start, c_w, kind="k")
            s_c = jnp.matmul(q_bf16, k_c.T, preferred_element_type=jnp.float32)
            k_col = _col_iota(c_start, c_w, q_rows)
            valid_s = (
                (k_col >= rel_first) & (k_col < rel_end)
                if c_start == 0
                else (k_col < rel_end)
            )
            s_list.append(jnp.where(valid_s, s_c, _MASK_LOGIT))

        m = s_new
        for s_c in s_list:
            m = jnp.maximum(
                m, _broadcast_minor_128(jnp.max(s_c, axis=1, keepdims=True))
            )

        p_new = jnp.exp2((s_new - m) * exp2_scale)
        p_pages = []
        l_vec = jnp.zeros((q_rows, 128), jnp.float32)
        for s_c in s_list:
            m_b = m if s_c.shape[1] == 128 else jnp.concatenate([m, m], axis=1)
            p_c = jnp.exp2((s_c - m_b) * exp2_scale)
            if s_c.shape[1] == 256:
                p0, p1 = p_c[:, :128], p_c[:, 128:]
                l_vec = l_vec + p0 + p1
                p_pages.append(p0)
                p_pages.append(p1)
            else:
                l_vec = l_vec + p_c
                p_pages.append(p_c)

        acc = p_new * acc0
        for pg_i in range(17):
            v_p = _bkv_fp8_chunk(slot, pg_i * 128, 128, kind="v")
            pv = jnp.matmul(p_pages[pg_i], v_p, preferred_element_type=jnp.float32)
            if v_scale is not None and v_scale_val != 1.0:
                pv = pv * v_scale_val
            acc = acc + pv

        l_sum = _broadcast_minor_128(jnp.sum(l_vec, axis=1, keepdims=True)) + p_new
        if interpret_mode:
            return (acc / l_sum).astype(jnp.bfloat16)
        return (acc * pltpu.reciprocal(l_sum, approx=True)).astype(jnp.bfloat16)

    def compute_stream1p(
        slot,
        q_bf16,
        q_row,
        rel_first,
        rel_end,
        n_slot_pages: int,
        *,
        skip_sanitize: bool = False,
    ):
        """Boundary-sanitized monolithic streaming QK -> 1-shot softmax -> PV."""
        s_new, _, acc0 = init_new_token_state(q_bf16, q_row)
        if probe == "nocompute":
            return acc0.astype(jnp.bfloat16)

        w_tot = n_slot_pages * page_size
        if not skip_sanitize:
            sanitize_slot_boundaries(slot, rel_first, rel_end, n_slot_pages)
        if cfg.fp8_dots:
            q_f32_d = q_bf16.astype(jnp.float32)
            q_mx = jnp.maximum(jnp.max(jnp.abs(q_f32_d)), jnp.float32(1e-6))
            q_gain = jnp.float32(256.0) / q_mx
            q_dot = (q_f32_d * q_gain).astype(jnp.float8_e4m3fn)
            s_new_eff = s_new * q_gain
            exp2_sc_eff = jnp.float32(exp2_scale) / q_gain
        else:
            q_dot = q_bf16
            s_new_eff = s_new
            exp2_sc_eff = exp2_scale

        if kv_pack_bd4:
            st_1p = _bd4_step_blk(
                slot,
                q_dot,
                rel_first,
                rel_end,
                0,
                w_tot,
                (
                    s_new_eff,
                    jnp.full((q_out_rows, 128), 1.0 / 128.0, jnp.float32),
                    acc0,
                ),
                eff_sc=exp2_sc_eff,
                allow_fp8_dots=True,
                use_page_reduce=cfg.page_reduce,
            )
            return finalize_out(st_1p[1], st_1p[2])

        if kv_pack_dense2 and w_tot >= 512 and (w_tot % 256 == 0):
            # Zero-concat dense2 path evaluated sequentially (even then odd) so only
            # one [q_rows, half_w] probability tensor and one V half live at once.
            half_w = w_tot // 2
            n_half_blocks = half_w // 128
            k_e, k_o = _bkv_fp8_split(slot, 0, w_tot, kind="k")
            s_e = jnp.matmul(q_dot, k_e.T, preferred_element_type=jnp.float32)
            s_o = jnp.matmul(q_dot, k_o.T, preferred_element_type=jnp.float32)
            h_io = lax.broadcasted_iota(jnp.int32, (q_rows, half_w), 1)
            c_e = 2 * h_io
            c_o = c_e + 1
            if sliding_window is not None:
                s_e = jnp.where((c_e >= rel_first) & (c_e < rel_end), s_e, _MASK_LOGIT)
                s_o = jnp.where((c_o >= rel_first) & (c_o < rel_end), s_o, _MASK_LOGIT)
            else:
                s_e = jnp.where(c_e < rel_end, s_e, _MASK_LOGIT)
                s_o = jnp.where(c_o < rel_end, s_o, _MASK_LOGIT)

            s_eo_max = jnp.maximum(s_e, s_o)
            if cfg.fold_128:
                m = jnp.maximum(
                    s_new_eff,
                    _broadcast_minor_128(
                        jnp.max(_fold128_op(s_eo_max, jnp.maximum), axis=1, keepdims=True)
                    ),
                )
            elif cfg.page_reduce:
                s_pgs = jnp.max(s_eo_max.reshape(q_rows, n_half_blocks, 128), axis=1)
                m = jnp.maximum(
                    s_new_eff, _broadcast_minor_128(jnp.max(s_pgs, axis=1, keepdims=True))
                )
            else:
                m = jnp.maximum(
                    s_new_eff, _broadcast_minor_128(jnp.max(s_eo_max, axis=1, keepdims=True))
                )
            l_e, pv_e = _dense2_pv_half(
                slot, 0, w_tot, s_e, m, exp2_sc_eff, False, use_page_reduce=cfg.page_reduce, allow_fp8_dots=True
            )
            l_o, pv_o = _dense2_pv_half(
                slot, 0, w_tot, s_o, m, exp2_sc_eff, True, use_page_reduce=cfg.page_reduce, allow_fp8_dots=True
            )
            p_new = jnp.exp2((s_new_eff - m) * exp2_sc_eff)
            l_sum = _broadcast_minor_128(jnp.sum(l_e + l_o, axis=1, keepdims=True)) + p_new
            pv = pv_e + pv_o
            if v_scale is not None and v_scale_val != 1.0:
                pv = pv * v_scale_val
            acc = pv + p_new * acc0
            if interpret_mode:
                return (acc / l_sum).astype(jnp.bfloat16)
            return (acc * pltpu.reciprocal(l_sum, approx=True)).astype(jnp.bfloat16)

        k_all = _bkv_fp8_chunk(slot, 0, w_tot, kind="k")
        if cfg.loop_kind not in (
            "stream2ld",
            "stream0b",
            "swafast",
            "swafast2",
            "swabatch",
        ):
            v_early = _bkv_fp8_chunk(slot, 0, w_tot, kind="v")
        else:
            v_early = None
        s_all = jnp.matmul(q_dot, k_all.T, preferred_element_type=jnp.float32)
        k_col = _col_iota(0, w_tot, q_rows)
        if sliding_window is not None:
            valid_s = (k_col >= rel_first) & (k_col < rel_end)
        else:
            valid_s = k_col < rel_end
        s_all = jnp.where(valid_s, s_all, _MASK_LOGIT)

        if cfg.page_reduce:
            s_pgs = jnp.max(s_all.reshape(q_rows, n_slot_pages, page_size), axis=1)
            m = jnp.maximum(
                s_new_eff, _broadcast_minor_128(jnp.max(s_pgs, axis=1, keepdims=True))
            )
            p_all = jnp.exp2((s_all - _broadcast_minor_128_any(m, w_tot)) * exp2_sc_eff)
            p_new = jnp.exp2((s_new_eff - m) * exp2_sc_eff)
            p_pgs = jnp.sum(p_all.reshape(q_rows, n_slot_pages, page_size), axis=1)
            l_sum = _broadcast_minor_128(jnp.sum(p_pgs, axis=1, keepdims=True)) + p_new
        elif cfg.fold_128:
            m = jnp.maximum(
                s_new_eff,
                _broadcast_minor_128(
                    jnp.max(_fold128_op(s_all, jnp.maximum), axis=1, keepdims=True)
                ),
            )
            p_all = jnp.exp2((s_all - m[:, :1]) * exp2_sc_eff)
            p_new = jnp.exp2((s_new_eff - m) * exp2_sc_eff)
            l_sum = (
                _broadcast_minor_128(
                    jnp.sum(_fold128_op(p_all, jnp.add), axis=1, keepdims=True)
                )
                + p_new
            )
        else:
            m = jnp.maximum(
                s_new_eff, _broadcast_minor_128(jnp.max(s_all, axis=1, keepdims=True))
            )
            p_all = jnp.exp2((s_all - _broadcast_minor_128_any(m, w_tot)) * exp2_sc_eff)
            p_new = jnp.exp2((s_new_eff - m) * exp2_sc_eff)
            l_sum = _broadcast_minor_128(jnp.sum(p_all, axis=1, keepdims=True)) + p_new
        p_dot = p_all.astype(jnp.bfloat16) if (cfg.pv_bf16 or cfg.fp8_dots) else p_all
        if v_early is None:
            v_all = _bkv_fp8_chunk(slot, 0, w_tot, kind="v")
        else:
            v_all = v_early
        pv = jnp.matmul(p_dot, v_all, preferred_element_type=jnp.float32)
        if v_scale is not None and v_scale_val != 1.0:
            pv = pv * v_scale_val
        acc = pv + p_new * acc0
        if interpret_mode:
            return (acc / l_sum).astype(jnp.bfloat16)
        return (acc * pltpu.reciprocal(l_sum, approx=True)).astype(jnp.bfloat16)

    def store_stream1p_multiq(slot, q_row, rel_first, rel_end, n_slot_pages: int):
        """Zero-spill branchless ``@pl.loop(0, dq)`` streaming SWA/Full for ``dq in {2, 4}``.

        Sanitizes ``slot`` once branchlessly, then loops ``q_i in [0, dq)`` with zero
        cross-iteration VREG carry, keeping peak VREG pressure ``<= 34`` VREGs (identical
        to ``dq == 1`` ``stream0b``, 0 spills, 1 HBM DMA/seq) and writing ``o_vmem_ref`` directly.
        """
        w_tot = n_slot_pages * page_size
        if probe != "nocompute":
            sanitize_slot_boundaries(slot, rel_first, rel_end, n_slot_pages)
        kv_list = [
            _load_new_kv_1row_f32(
                kv_merged_vmem_ref,
                q_row + j,
                interpret_mode=interpret_mode,
                mask_lo64=kv_pack_bd4,
            )
            for j in range(dq)
        ]

        @pl.loop(0, dq)
        def _q_step(q_i):
            q_sub_full = q_vmem_ref[q_row + q_i]
            q_sub = q_sub_full[:q_out_head_rows]
            q_f = q_sub.astype(jnp.float32)
            m_tok = None
            s_js = []
            for j in range(dq):
                kj, _ = kv_list[j]
                s_ij = _broadcast_minor_128(
                    jnp.sum(
                        q_f * jnp.broadcast_to(kj, (q_out_head_rows, 128)),
                        axis=1,
                        keepdims=True,
                    )
                )
                s_ij = jnp.where(jnp.int32(j) <= q_i, s_ij, _MASK_LOGIT)
                s_js.append(s_ij)
                m_tok = s_ij if m_tok is None else jnp.maximum(m_tok, s_ij)
            p_sum_tok = jnp.zeros((q_out_head_rows, 128), jnp.float32)
            a_tok = jnp.zeros((q_out_head_rows, 128), jnp.float32)
            for j in range(dq):
                _, vj = kv_list[j]
                p_ij = jnp.where(
                    jnp.int32(j) <= q_i,
                    jnp.exp2((s_js[j] - m_tok) * exp2_scale),
                    0.0,
                )
                p_sum_tok = p_sum_tok + p_ij
                a_tok = a_tok + p_ij * (
                    jnp.broadcast_to(vj, (q_out_head_rows, 128)) * v_scale_val
                )
            if probe == "nocompute":
                o_vmem_ref[q_row + q_i] = (a_tok / p_sum_tok).astype(jnp.bfloat16)
                return

            if kv_pack_bd4:
                half_w = w_tot // 2
                q_bd_i = _to_bd4_q(q_sub_full)
                k_pair = _bkv_fp8_chunk(slot, 0, w_tot, kind="k")
                s_e, s_o = _bd4_logits_split(q_bd_i, k_pair, 1, half_w)
                c_e = _bd4_c_even_iota(0, half_w, q_out_head_rows)
                c_o = c_e + 1
                if sliding_window is not None:
                    rf_i = jnp.maximum(rel_end + q_i - (sliding_window - 1), 0)
                    s_e = jnp.where((c_e >= rf_i) & (c_e < rel_end), s_e, _MASK_LOGIT)
                    s_o = jnp.where((c_o >= rf_i) & (c_o < rel_end), s_o, _MASK_LOGIT)
                else:
                    s_e = jnp.where(c_e < rel_end, s_e, _MASK_LOGIT)
                    s_o = jnp.where(c_o < rel_end, s_o, _MASK_LOGIT)
                m, alpha_tok, p_e, p_o, l_add_mq = _bd4_softmax_reduce(
                    s_e, s_o, m_tok, exp2_scale, half_w, q_out_head_rows, use_page_reduce=False
                )
                l_sum = p_sum_tok * alpha_tok + l_add_mq
                v_pair = _bkv_fp8_chunk(slot, 0, w_tot, kind="v")
                pv = _bd4_pv_reduce(p_e, p_o, v_pair, n_seq=1, p_bf16=cfg.pv_bf16)
                if v_scale is not None and v_scale_val != 1.0:
                    pv = pv * v_scale_val
                acc = pv + alpha_tok * a_tok
                o_vmem_ref[q_row + q_i] = (
                    (acc / l_sum).astype(jnp.bfloat16)
                    if interpret_mode
                    else (acc * pltpu.reciprocal(l_sum, approx=True)).astype(
                        jnp.bfloat16
                    )
                )
                return

            k_all = _bkv_fp8_chunk(slot, 0, w_tot, kind="k")
            s_all = jnp.matmul(q_sub, k_all.T, preferred_element_type=jnp.float32)
            k_col = _col_iota(0, w_tot, q_head_rows)
            if sliding_window is not None:
                rf_i = jnp.maximum(rel_end + q_i - (sliding_window - 1), 0)
                valid_s = (k_col >= rf_i) & (k_col < rel_end)
            else:
                valid_s = k_col < rel_end
            s_all = jnp.where(valid_s, s_all, _MASK_LOGIT)
            if cfg.fold_col:
                m = jnp.maximum(
                    m_tok,
                    _broadcast_minor_128(
                        jnp.max(_fold128_op(s_all, jnp.maximum), axis=1, keepdims=True)
                    ),
                )
                alpha_tok = jnp.exp2((m_tok - m) * exp2_scale)
                p_all, l_add_mq = _exp2_fold_col(s_all, m, exp2_scale)
                l_sum = (
                    p_sum_tok * alpha_tok
                    + _broadcast_minor_128(jnp.sum(l_add_mq, axis=1, keepdims=True))
                )
            elif cfg.fold_128:
                m = jnp.maximum(
                    m_tok,
                    _broadcast_minor_128(
                        jnp.max(_fold128_op(s_all, jnp.maximum), axis=1, keepdims=True)
                    ),
                )
                alpha_tok = jnp.exp2((m_tok - m) * exp2_scale)
                p_all = jnp.exp2((s_all - m[:, :1]) * exp2_scale)
                l_sum = (
                    p_sum_tok * alpha_tok
                    + _broadcast_minor_128(
                        jnp.sum(_fold128_op(p_all, jnp.add), axis=1, keepdims=True)
                    )
                )
            else:
                m = jnp.maximum(
                    m_tok, _broadcast_minor_128(jnp.max(s_all, axis=1, keepdims=True))
                )
                alpha_tok = jnp.exp2((m_tok - m) * exp2_scale)
                p_all = jnp.exp2((s_all - _broadcast_minor_128_any(m, w_tot)) * exp2_scale)
                l_sum = (
                    p_sum_tok * alpha_tok
                    + _broadcast_minor_128(jnp.sum(p_all, axis=1, keepdims=True))
                )
            p_dot = p_all.astype(jnp.bfloat16) if cfg.pv_bf16 else p_all
            v_all = _bkv_fp8_chunk(slot, 0, w_tot, kind="v")
            pv = jnp.matmul(p_dot, v_all, preferred_element_type=jnp.float32)
            if v_scale is not None and v_scale_val != 1.0:
                pv = pv * v_scale_val
            acc = pv + alpha_tok * a_tok
            o_vmem_ref[q_row + q_i] = (
                (acc / l_sum).astype(jnp.bfloat16)
                if interpret_mode
                else (acc * pltpu.reciprocal(l_sum, approx=True)).astype(jnp.bfloat16)
            )

    def compute_subblock2k_loop(slot, q_bf16, rel_first, local_end):
        """Zero-vector-carry ``@pl.loop`` over active 2048-key blocks (>16K fallback)."""
        if probe == "nocompute":
            return
        n_sub2k = (local_end + 2047) // 2048

        @pl.loop(0, n_sub2k)
        def _sub2k_body(sb_idx):
            row0 = sb_idx * 2048
            m = state_vmem_ref[0]
            l_vec = state_vmem_ref[1]
            acc = state_vmem_ref[2]
            if kv_pack_bd4:
                m_new, l_new, acc_new = _bd4_step_blk(
                    slot, q_bf16, rel_first, local_end, row0, 2048, (m, l_vec, acc)
                )
                state_vmem_ref[0] = m_new
                state_vmem_ref[1] = l_new
                state_vmem_ref[2] = acc_new
                return

            s_sub_list = []
            for c_i in range(8):
                c_st = row0 + c_i * 256
                k_c = _bkv_fp8_chunk(slot, c_st, 256, kind="k")
                s_c = jnp.matmul(q_bf16, k_c.T, preferred_element_type=jnp.float32)
                k_col = _col_iota(c_st, 256, q_rows)
                s_c = jnp.where(
                    (k_col >= rel_first) & (k_col < local_end),
                    s_c,
                    _MASK_LOGIT,
                )
                s_sub_list.append(s_c)
            m_new = m
            for s_c in s_sub_list:
                m_new = jnp.maximum(
                    m_new,
                    _broadcast_minor_128(jnp.max(s_c, axis=1, keepdims=True)),
                )
            alpha = jnp.exp2((m - m_new) * exp2_scale)
            m_256 = jnp.concatenate([m_new, m_new], axis=1)
            p_sub_chunks = []
            l_add = jnp.zeros((q_rows, 128), jnp.float32)
            for s_c in s_sub_list:
                p_c = jnp.exp2((s_c - m_256) * exp2_scale)
                l_add = l_add + p_c[:, :128] + p_c[:, 128:]
                p_sub_chunks.append(p_c)
            l_new = l_vec * alpha + l_add
            acc_new = acc * alpha
            for c_i in range(8):
                v_p = _bkv_fp8_chunk(slot, row0 + c_i * 256, 256, kind="v")
                pv = jnp.matmul(
                    p_sub_chunks[c_i], v_p, preferred_element_type=jnp.float32
                )
                if v_scale is not None and v_scale_val != 1.0:
                    pv = pv * v_scale_val
                acc_new = acc_new + pv

            state_vmem_ref[0] = m_new
            state_vmem_ref[1] = l_new
            state_vmem_ref[2] = acc_new

    def sanitize_slot_v2(slot, rel_first, rel_end, n_pages, target_pages):
        """Zero-upfront-cost VMEM sanitizer for ``stream2h`` and ``streamdyn``.

        Zeroes only unfetched pages ``[n_pages, target_pages)`` (0 pages in the
        steady-state common case) plus the <=2 active boundary pages ``0`` and
        ``n_pages - 1``, eliminating ``zero_all_slots_once()`` while staying
        100% ``0x7F`` FP8-NaN safe on free, tail, evicted-SWA, and reused slots.
        """
        if cfg.lazy_slot_zero:
            hwm_v2 = written_cnt_smem_ref[2 + slot]
            start_v2 = jnp.maximum(n_pages, hwm_v2)

            @pl.when(start_v2 < target_pages)
            def _zero_unfetched_once():
                _zero_pages_u32(slot, start_v2, target_pages)

            written_cnt_smem_ref[2 + slot] = jnp.maximum(
                hwm_v2, jnp.maximum(n_pages, target_pages)
            )
        else:

            @pl.when(n_pages < target_pages)
            def _zero_unfetched():
                _zero_pages_u32(slot, n_pages, target_pages)

        @pl.when(target_pages > 0)
        def _clean_bounds():
            last_p = jnp.clip(n_pages - 1, 0, jnp.maximum(target_pages - 1, 0))
            if sliding_window is not None:
                @pl.when((rel_first > 0) & (last_p > 0))
                def _clean_pg0_v2():
                    _sanitize_page_u32(slot, 0, rel_first, rel_end)
            last_lo = jnp.where(last_p == 0, rel_first, 0)
            last_hi = jnp.clip(rel_end - last_p * page_size, 0, page_size)
            _sanitize_page_u32(slot, last_p, last_lo, last_hi)

    def compute_swa_stream2h(slot, q_bf16, q_row, rel_first, rel_end, n_pages):
        """Zero-spill 2-half (1152 + 1024 = 2176 keys) online-softmax SWA engine (dq in {1, 2})."""
        m_init, l_init, a_init = init_new_token_state(q_bf16, q_row)
        if probe == "nocompute":
            return finalize_out(l_init, a_init)

        sanitize_slot_v2(slot, rel_first, rel_end, n_pages, 17)
        if dq > 1:
            rf_rows = jnp.concatenate(
                [
                    jnp.full(
                        (q_out_head_rows, 1),
                        jnp.maximum(rel_end + i - (sliding_window - 1), 0),
                        jnp.int32,
                    )
                    for i in range(dq)
                ],
                axis=0,
            )
        else:
            rf_rows = rel_first
        w0, pg0_cnt = 1152, 9
        w1, pg1_cnt = 1024, 8
        if kv_pack_bd4:
            st = (m_init, l_init, a_init)
            st = _bd4_step_blk(slot, q_bf16, rf_rows, rel_end, 0, w0, st)
            st = _bd4_step_blk(slot, q_bf16, rf_rows, rel_end, w0, w1, st)
            return finalize_out(st[1], st[2])

        k0 = _bkv_fp8_chunk(slot, 0, w0, kind="k")
        s0 = jnp.matmul(q_bf16, k0.T, preferred_element_type=jnp.float32)
        c0 = _col_iota(0, w0, q_rows)
        s0 = jnp.where((c0 >= rf_rows) & (c0 < rel_end), s0, _MASK_LOGIT)
        m0_pages = jnp.max(s0.reshape(q_rows, pg0_cnt, 128), axis=1)
        m0 = jnp.maximum(
            m_init, _broadcast_minor_128(jnp.max(m0_pages, axis=1, keepdims=True))
        )
        alpha0 = jnp.exp2((m_init - m0) * exp2_scale)
        p0 = jnp.exp2((s0 - _broadcast_minor_128_any(m0, w0)) * exp2_scale)
        l0_vec = l_init * alpha0 + jnp.sum(p0.reshape(q_rows, pg0_cnt, 128), axis=1)
        v0 = _bkv_fp8_chunk(slot, 0, w0, kind="v")
        pv0 = jnp.matmul(p0, v0, preferred_element_type=jnp.float32)
        if v_scale is not None and v_scale_val != 1.0:
            pv0 = pv0 * v_scale_val
        acc0_h = a_init * alpha0 + pv0

        k1 = _bkv_fp8_chunk(slot, w0, w1, kind="k")
        s1 = jnp.matmul(q_bf16, k1.T, preferred_element_type=jnp.float32)
        c1 = _col_iota(w0, w1, q_rows)
        s1 = jnp.where((c1 >= rf_rows) & (c1 < rel_end), s1, _MASK_LOGIT)
        m1_pages = jnp.max(s1.reshape(q_rows, pg1_cnt, 128), axis=1)
        m = jnp.maximum(
            m0, _broadcast_minor_128(jnp.max(m1_pages, axis=1, keepdims=True))
        )
        alpha1 = jnp.exp2((m0 - m) * exp2_scale)
        p1 = jnp.exp2((s1 - _broadcast_minor_128_any(m, w1)) * exp2_scale)
        l1_vec = jnp.sum(p1.reshape(q_rows, pg1_cnt, 128), axis=1)
        v1 = _bkv_fp8_chunk(slot, w0, w1, kind="v")
        pv1 = jnp.matmul(p1, v1, preferred_element_type=jnp.float32)
        if v_scale is not None and v_scale_val != 1.0:
            pv1 = pv1 * v_scale_val

        l_tot = l0_vec * alpha1 + l1_vec
        acc_tot = acc0_h * alpha1 + pv1
        return finalize_out(l_tot, acc_tot)

    def compute_streamdyn(slot, q_bf16, q_row, rel_first, rel_end, n_pages):
        """Zero-spill VREG-carried dynamic-window online-softmax loop over ``cfg.chunk_sz`` keys."""
        m0, l0, a0 = init_new_token_state(q_bf16, q_row)
        if probe == "nocompute":
            return finalize_out(l0, a0)

        chunk_sz = cfg.chunk_sz
        chunk_pages = chunk_sz // page_size
        n_chunks = (rel_end + chunk_sz - 1) // chunk_sz
        target_pages = n_chunks * chunk_pages
        sanitize_slot_v2(slot, rel_first, rel_end, n_pages, target_pages)

        if sliding_window is not None and dq > 1:
            rf_rows = jnp.concatenate(
                [
                    jnp.full(
                        (q_out_head_rows, 1),
                        jnp.maximum(rel_end + i - (sliding_window - 1), 0),
                        jnp.int32,
                    )
                    for i in range(dq)
                ],
                axis=0,
            )
        else:
            rf_rows = rel_first

        if cfg.fp8_dots:
            q_f32_d = q_bf16.astype(jnp.float32)
            q_mx = jnp.maximum(jnp.max(jnp.abs(q_f32_d)), jnp.float32(1e-6))
            q_gain = jnp.float32(256.0) / q_mx
            q_dot_dyn = (q_f32_d * q_gain).astype(jnp.float8_e4m3fn)
            m0 = m0 * q_gain
            exp2_sc_dyn = jnp.float32(exp2_scale) / q_gain
        else:
            q_dot_dyn = q_bf16
            exp2_sc_dyn = exp2_scale

        def _dyn_chunk(c_idx, carry):
            m_prev, l_prev, acc_prev = carry
            row0 = c_idx * chunk_sz

            if kv_pack_bd4:
                return _bd4_step_blk(
                    slot,
                    q_dot_dyn,
                    rf_rows,
                    rel_end,
                    row0,
                    chunk_sz,
                    carry,
                    eff_sc=exp2_sc_dyn,
                    allow_fp8_dots=True,
                    use_page_reduce=True,
                )

            if kv_pack_dense2 and chunk_sz >= 512 and (chunk_sz % 256 == 0) and not cfg.fp8_dots:
                half_w = chunk_sz // 2
                n_hb = half_w // 128
                k_e, k_o = _bkv_fp8_split(slot, row0, chunk_sz, kind="k")
                s_e = jnp.matmul(q_dot_dyn, k_e.T, preferred_element_type=jnp.float32)
                s_o = jnp.matmul(q_dot_dyn, k_o.T, preferred_element_type=jnp.float32)
                h_io = lax.broadcasted_iota(jnp.int32, (q_rows, half_w), 1)
                c_e = row0 + 2 * h_io
                c_o = c_e + 1
                if sliding_window is not None:
                    s_e = jnp.where((c_e >= rf_rows) & (c_e < rel_end), s_e, _MASK_LOGIT)
                    s_o = jnp.where((c_o >= rf_rows) & (c_o < rel_end), s_o, _MASK_LOGIT)
                else:
                    s_e = jnp.where(c_e < rel_end, s_e, _MASK_LOGIT)
                    s_o = jnp.where(c_o < rel_end, s_o, _MASK_LOGIT)
                s_max = jnp.maximum(s_e, s_o)
                if cfg.fold_128:
                    m_new = jnp.maximum(
                        m_prev,
                        _broadcast_minor_128(
                            jnp.max(_fold128_op(s_max, jnp.maximum), axis=1, keepdims=True)
                        ),
                    )
                else:
                    s_pgs = jnp.max(s_max.reshape(q_rows, n_hb, 128), axis=1)
                    m_new = jnp.maximum(
                        m_prev, _broadcast_minor_128(jnp.max(s_pgs, axis=1, keepdims=True))
                    )
                alpha = jnp.exp2((m_prev - m_new) * exp2_sc_dyn)
                l_e, pv_e = _dense2_pv_half(
                    slot, row0, chunk_sz, s_e, m_new, exp2_sc_dyn, False, use_page_reduce=True
                )
                l_o, pv_o = _dense2_pv_half(
                    slot, row0, chunk_sz, s_o, m_new, exp2_sc_dyn, True, use_page_reduce=True
                )
                l_add = l_e + l_o
                pv = pv_e + pv_o
                if v_scale is not None and v_scale_val != 1.0:
                    pv = pv * v_scale_val
                return (
                    m_new,
                    l_prev * alpha + l_add,
                    acc_prev * alpha + pv,
                )

            k_c = _bkv_fp8_chunk(slot, row0, chunk_sz, kind="k")
            s_c = jnp.matmul(q_dot_dyn, k_c.T, preferred_element_type=jnp.float32)
            k_col = _col_iota(row0, chunk_sz, q_rows)
            if sliding_window is not None:
                valid_s = (k_col >= rf_rows) & (k_col < rel_end)
            else:
                valid_s = k_col < rel_end
            s_c = jnp.where(valid_s, s_c, _MASK_LOGIT)

            if cfg.fold_col:
                m_blk = _broadcast_minor_128(
                    jnp.max(_fold128_op(s_c, jnp.maximum), axis=1, keepdims=True)
                )
                m_new = jnp.maximum(m_prev, m_blk)
                alpha = jnp.exp2((m_prev - m_new) * exp2_sc_dyn)
                p_c, l_add = _exp2_fold_col(s_c, m_new, exp2_sc_dyn)
            elif cfg.fold_128:
                m_blk = _broadcast_minor_128(
                    jnp.max(_fold128_op(s_c, jnp.maximum), axis=1, keepdims=True)
                )
                m_new = jnp.maximum(m_prev, m_blk)
                alpha = jnp.exp2((m_prev - m_new) * exp2_sc_dyn)
                p_c = jnp.exp2((s_c - m_new[:, :1]) * exp2_sc_dyn)
                l_add = _fold128_op(p_c, jnp.add)
            else:
                s_pages = jnp.max(s_c.reshape(q_rows, chunk_sz // 128, 128), axis=1)
                m_new = jnp.maximum(
                    m_prev,
                    _broadcast_minor_128(jnp.max(s_pages, axis=1, keepdims=True)),
                )
                alpha = jnp.exp2((m_prev - m_new) * exp2_sc_dyn)
                p_c = jnp.exp2(
                    (s_c - _broadcast_minor_128_any(m_new, chunk_sz)) * exp2_sc_dyn
                )
                l_add = jnp.sum(p_c.reshape(q_rows, chunk_sz // 128, 128), axis=1)

            v_c = _bkv_fp8_chunk(slot, row0, chunk_sz, kind="v")
            p_dot = p_c.astype(jnp.bfloat16) if (cfg.pv_bf16 or cfg.fp8_dots) else p_c
            pv = jnp.matmul(p_dot, v_c, preferred_element_type=jnp.float32)
            if v_scale is not None and v_scale_val != 1.0:
                pv = pv * v_scale_val

            return (
                m_new,
                l_prev * alpha + l_add,
                acc_prev * alpha + pv,
            )

        _, l_fin, acc_fin = lax.fori_loop(0, n_chunks, _dyn_chunk, (m0, l0, a0))
        return finalize_out(l_fin, acc_fin)

    def _step_2k_blk(
        slot,
        q_bf16,
        rel_first,
        rel_end,
        r0,
        w_k: int,
        carry,
        *,
        mask_bounds: bool = True,
    ):
        if kv_pack_bd4:
            return _bd4_step_blk(
                slot,
                q_bf16,
                rel_first,
                rel_end,
                r0,
                w_k,
                carry,
                mask_bounds=mask_bounds,
            )
        m_prev, l_prev, acc_prev = carry
        if kv_pack_dense2:
            half_w = w_k // 2
            n_hb = half_w // 128
            k_e, k_o = _bkv_fp8_split(slot, r0, w_k, kind="k")
            s_e = jnp.matmul(q_bf16, k_e.T, preferred_element_type=jnp.float32)
            s_o = jnp.matmul(q_bf16, k_o.T, preferred_element_type=jnp.float32)
            if mask_bounds:
                h_io = lax.broadcasted_iota(jnp.int32, (q_rows, half_w), 1)
                c_e = r0 + 2 * h_io
                c_o = c_e + 1
                if sliding_window is not None:
                    s_e = jnp.where((c_e >= rel_first) & (c_e < rel_end), s_e, _MASK_LOGIT)
                    s_o = jnp.where((c_o >= rel_first) & (c_o < rel_end), s_o, _MASK_LOGIT)
                else:
                    s_e = jnp.where(c_e < rel_end, s_e, _MASK_LOGIT)
                    s_o = jnp.where(c_o < rel_end, s_o, _MASK_LOGIT)
            s_max = jnp.maximum(s_e, s_o)
            if cfg.fold_128:
                m_new = jnp.maximum(
                    m_prev,
                    _broadcast_minor_128(
                        jnp.max(_fold128_op(s_max, jnp.maximum), axis=1, keepdims=True)
                    ),
                )
                alpha = jnp.exp2((m_prev - m_new) * exp2_scale)
                if cfg.fold_col:
                    p_e, l_e = _exp2_fold_col(s_e, m_new, exp2_scale)
                    p_o, l_o = _exp2_fold_col(s_o, m_new, exp2_scale)
                    l_add = l_e + l_o
                else:
                    m_1 = m_new[:, :1]
                    p_e = jnp.exp2((s_e - m_1) * exp2_scale)
                    p_o = jnp.exp2((s_o - m_1) * exp2_scale)
                    l_add = _fold128_op(p_e + p_o, jnp.add)
            else:
                s_pgs = jnp.max(s_max.reshape(q_rows, n_hb, 128), axis=1)
                m_new = jnp.maximum(
                    m_prev, _broadcast_minor_128(jnp.max(s_pgs, axis=1, keepdims=True))
                )
                alpha = jnp.exp2((m_prev - m_new) * exp2_scale)
                m_hw = _broadcast_minor_128_any(m_new, half_w)
                p_e = jnp.exp2((s_e - m_hw) * exp2_scale)
                p_o = jnp.exp2((s_o - m_hw) * exp2_scale)
                l_add = jnp.sum((p_e + p_o).reshape(q_rows, n_hb, 128), axis=1)
            v_e, v_o = _bkv_fp8_split(slot, r0, w_k, kind="v")
            pe_dot = p_e.astype(jnp.bfloat16) if cfg.pv_bf16 else p_e
            po_dot = p_o.astype(jnp.bfloat16) if cfg.pv_bf16 else p_o
            pv = jnp.matmul(pe_dot, v_e, preferred_element_type=jnp.float32) + jnp.matmul(
                po_dot, v_o, preferred_element_type=jnp.float32
            )
            if v_scale is not None and v_scale_val != 1.0:
                pv = pv * v_scale_val
            return (m_new, l_prev * alpha + l_add, acc_prev * alpha + pv)

        k_c = _bkv_fp8_chunk(slot, r0, w_k, kind="k")
        s_c = jnp.matmul(q_bf16, k_c.T, preferred_element_type=jnp.float32)
        if mask_bounds:
            k_col = _col_iota(r0, w_k, q_rows)
            valid_s = (
                ((k_col >= rel_first) & (k_col < rel_end))
                if sliding_window is not None
                else (k_col < rel_end)
            )
            s_c = jnp.where(valid_s, s_c, _MASK_LOGIT)
        if cfg.fold_col:
            m_blk = _broadcast_minor_128(
                jnp.max(_fold128_op(s_c, jnp.maximum), axis=1, keepdims=True)
            )
            m_new = jnp.maximum(m_prev, m_blk)
            alpha = jnp.exp2((m_prev - m_new) * exp2_scale)
            p_c, l_add = _exp2_fold_col(s_c, m_new, exp2_scale)
        elif cfg.fold_128:
            m_blk = _broadcast_minor_128(
                jnp.max(_fold128_op(s_c, jnp.maximum), axis=1, keepdims=True)
            )
            m_new = jnp.maximum(m_prev, m_blk)
            alpha = jnp.exp2((m_prev - m_new) * exp2_scale)
            p_c = jnp.exp2((s_c - m_new[:, :1]) * exp2_scale)
            l_add = _fold128_op(p_c, jnp.add)
        else:
            s_pgs = jnp.max(s_c.reshape(q_rows, w_k // 128, 128), axis=1)
            m_new = jnp.maximum(
                m_prev, _broadcast_minor_128(jnp.max(s_pgs, axis=1, keepdims=True))
            )
            alpha = jnp.exp2((m_prev - m_new) * exp2_scale)
            p_c = jnp.exp2((s_c - _broadcast_minor_128_any(m_new, w_k)) * exp2_scale)
            l_add = jnp.sum(p_c.reshape(q_rows, w_k // 128, 128), axis=1)
        v_c = _bkv_fp8_chunk(slot, r0, w_k, kind="v")
        p_dot = p_c.astype(jnp.bfloat16) if cfg.pv_bf16 else p_c
        pv = jnp.matmul(p_dot, v_c, preferred_element_type=jnp.float32)
        if v_scale is not None and v_scale_val != 1.0:
            pv = pv * v_scale_val
        return (m_new, l_prev * alpha + l_add, acc_prev * alpha + pv)

    def compute_stream2k_unrolled(slot, q_bf16, q_row, rel_first, rel_end, n_pages):
        """Unrolled 0-fori_loop 2K-chunk streaming engine with zero-concat even/odd d2."""
        m0, l0, a0 = init_new_token_state(q_bf16, q_row)
        if probe == "nocompute":
            return finalize_out(l0, a0)

        c_sz = 2048
        c_pgs = c_sz // page_size
        n_chunks_d = (rel_end + c_sz - 1) // c_sz
        # Match target_pgs to the exact gear read width so 0x7F padding is sanitized without extra work.
        if max_ring_pages * page_size >= 8192:
            rem_w = max_ring_pages * page_size - 8192
            fb_pgs = (
                max_ring_pages
                if (rem_w >= 1024 and (rem_w <= 2048) and (rem_w % 1024 == 0))
                else jnp.minimum(n_chunks_d * c_pgs, max_ring_pages)
            )
            target_pgs = jnp.where(
                rel_end <= 4096,
                n_chunks_d * c_pgs,
                jnp.where(
                    rel_end <= 5120,
                    5120 // page_size,
                    jnp.where(rel_end <= 8192, n_chunks_d * c_pgs, fb_pgs),
                ),
            )
        else:
            target_pgs = n_chunks_d * c_pgs
        target_pgs = jnp.minimum(target_pgs, max_ring_pages)
        sanitize_slot_v2(slot, rel_first, rel_end, n_pages, target_pgs)

        def _step_blk(r0, w_k: int, carry):
            return _step_2k_blk(slot, q_bf16, rel_first, rel_end, r0, w_k, carry)

        def _step_2k(c_idx, carry):
            return _step_blk(c_idx * c_sz, c_sz, carry)

        def _run_k(k_n: int, tail_1k: bool = False):
            st = (m0, l0, a0)
            for ci in range(k_n):
                st = _step_2k(ci, st)
            if tail_1k:
                st = _step_blk(k_n * c_sz, 1024, st)
            return finalize_out(st[1], st[2])

        if max_ring_pages * page_size >= 8192:
            @pl.when(rel_end <= 2048)
            def _u1():
                o_vmem_ref[q_row] = _run_k(1)

            @pl.when((rel_end > 2048) & (rel_end <= 4096))
            def _u2():
                o_vmem_ref[q_row] = _run_k(2)

            @pl.when((rel_end > 4096) & (rel_end <= 5120))
            def _u2_1k():
                o_vmem_ref[q_row] = _run_k(2, tail_1k=True)

            @pl.when((rel_end > 5120) & (rel_end <= 6144))
            def _u3():
                o_vmem_ref[q_row] = _run_k(3)

            @pl.when((rel_end > 6144) & (rel_end <= 8192))
            def _u4():
                o_vmem_ref[q_row] = _run_k(4)

            @pl.when(rel_end > 8192)
            def _ufb():
                rem_w = max_ring_pages * page_size - 8192
                st = (m0, l0, a0)
                for ci in range(4):
                    st = _step_2k(ci, st)
                if rem_w >= 1024 and (rem_w <= 2048) and (rem_w % 1024 == 0):
                    st = _step_blk(8192, rem_w, st)
                else:
                    _, l_f, a_f = lax.fori_loop(4, n_chunks_d, _step_2k, st)
                    st = (st[0], l_f, a_f)
                o_vmem_ref[q_row] = finalize_out(st[1], st[2])

            return None
        sanitize_slot_v2(slot, rel_first, rel_end, n_pages, jnp.minimum(n_chunks_d * c_pgs, max_ring_pages))
        _, l_f, a_f = lax.fori_loop(0, n_chunks_d, _step_2k, (m0, l0, a0))
        return finalize_out(l_f, a_f)

    def _one_shot_attn(slot, q_bf16, q_row, rel_first, rel_end, n_pages, w_sz: int, use_pbf16: bool, is_swa: bool):
        w_pg = w_sz // page_size
        m0, l0, a0 = init_new_token_state(q_bf16, q_row)
        sanitize_slot_v2(slot, rel_first, rel_end, n_pages, w_pg)
        if cfg.fp8_dots:
            q_f32_d = q_bf16.astype(jnp.float32)
            q_mx = jnp.maximum(jnp.max(jnp.abs(q_f32_d)), jnp.float32(1e-6))
            q_gain = jnp.float32(256.0) / q_mx
            q_dot = (q_f32_d * q_gain).astype(jnp.float8_e4m3fn)
            m0_eff = m0 * q_gain
            exp2_sc = jnp.float32(exp2_scale) / q_gain
        else:
            q_dot = q_bf16
            m0_eff = m0
            exp2_sc = exp2_scale
        if kv_pack_bd4:
            if is_swa and dq > 1:
                rf_one = jnp.concatenate(
                    [
                        jnp.full(
                            (q_out_head_rows, 1),
                            jnp.maximum(rel_end + i - (sliding_window - 1), 0),
                            jnp.int32,
                        )
                        for i in range(dq)
                    ],
                    axis=0,
                )
            else:
                rf_one = rel_first
            st_os = _bd4_step_blk(
                slot,
                q_dot,
                rf_one,
                rel_end,
                0,
                w_sz,
                (m0_eff, l0, a0),
                eff_sc=exp2_sc,
                mask_bounds=True,
                allow_fp8_dots=True,
                use_pbf16=use_pbf16,
            )
            return finalize_out(st_os[1], st_os[2])
        k_c = _bkv_fp8_chunk(slot, 0, w_sz, kind="k")
        s_c = jnp.matmul(q_dot, k_c.T, preferred_element_type=jnp.float32)
        k_col = _col_iota(0, w_sz, q_rows)
        if is_swa and dq > 1:
            rf_one = jnp.concatenate(
                [
                    jnp.full(
                        (q_head_rows, 1),
                        jnp.maximum(rel_end + i - (sliding_window - 1), 0),
                        jnp.int32,
                    )
                    for i in range(dq)
                ],
                axis=0,
            )
            valid_s = (k_col >= rf_one) & (k_col < rel_end)
        else:
            valid_s = ((k_col >= rel_first) & (k_col < rel_end)) if is_swa else (k_col < rel_end)
        s_c = jnp.where(valid_s, s_c, _MASK_LOGIT)
        if ((w_sz // 128) % 4) == 0 or cfg.page_reduce:
            s_pages = jnp.max(s_c.reshape(q_rows, w_sz // 128, 128), axis=1)
            m_new = jnp.maximum(
                m0_eff, _broadcast_minor_128(jnp.max(s_pages, axis=1, keepdims=True))
            )
            alpha = jnp.exp2((m0_eff - m_new) * exp2_sc)
            p_c = jnp.exp2((s_c - _broadcast_minor_128_any(m_new, w_sz)) * exp2_sc)
            l_fin = l0 * alpha + jnp.sum(p_c.reshape(q_rows, w_sz // 128, 128), axis=1)
        else:
            m_new = jnp.maximum(
                m0_eff, _broadcast_minor_128(jnp.max(s_c, axis=1, keepdims=True))
            )
            alpha = jnp.exp2((m0_eff - m_new) * exp2_sc)
            p_c = jnp.exp2((s_c - _broadcast_minor_128_any(m_new, w_sz)) * exp2_sc)
            l_fin = l0 * alpha + _broadcast_minor_128(jnp.sum(p_c, axis=1, keepdims=True)) * jnp.float32(1.0 / 128.0)
        v_c = _bkv_fp8_chunk(slot, 0, w_sz, kind="v")
        p_dot = p_c.astype(jnp.bfloat16) if (use_pbf16 or cfg.pv_bf16 or cfg.fp8_dots) else p_c
        pv = jnp.matmul(p_dot, v_c, preferred_element_type=jnp.float32)
        if v_scale is not None and v_scale_val != 1.0:
            pv = pv * v_scale_val
        return finalize_out(l_fin, a0 * alpha + pv)

    def store_w7auto(slot, q_bf16, q_row, rel_first, rel_end, n_pages):
        """Per-sequence 3/4-gear decode engine: [c512], c1024 (<=1K), c8192 (1K..8K), c5120 (>8K)."""
        if probe == "nocompute":
            _, l0, a0 = init_new_token_state(q_bf16, q_row)
            o_vmem_ref[q_row] = finalize_out(l0, a0)
            return

        if cfg.fine_gears:
            @pl.when(rel_end <= 512)
            def _short_512():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, 512, cfg.pv_bf16, False
                )

            @pl.when((rel_end > 512) & (rel_end <= 1024))
            def _short_1k_f():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, 1024, cfg.pv_bf16, False
                )
        else:
            @pl.when(rel_end <= 1024)
            def _short_1k():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, 1024, cfg.pv_bf16, False
                )

        if max_ring_pages >= 64:
            @pl.when((rel_end > 1024) & (rel_end <= 8192))
            def _mid_8k():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, 8192, cfg.pv_bf16, False
                )

            @pl.when(rel_end > 8192)
            def _long_gt8k():
                o_vmem_ref[q_row] = compute_streamdyn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages
                )
        elif (cfg.dyn2k_unroll or cfg.gear_4k) and max_ring_pages * page_size >= 8192 and cfg.chunk_sz == 5120:
            @pl.when((rel_end > 1024) & (rel_end <= 5120))
            def _mid_5k_u():
                o_vmem_ref[q_row] = compute_streamdyn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages
                )

            @pl.when((rel_end > 5120) & (rel_end <= 8192))
            def _mid_8k_2ku():
                st = init_new_token_state(q_bf16, q_row)
                sanitize_slot_v2(slot, rel_first, rel_end, n_pages, 8192 // page_size)
                if cfg.gear_4k:
                    for ci in range(2):
                        st = _step_2k_blk(
                            slot,
                            q_bf16,
                            rel_first,
                            rel_end,
                            ci * 4096,
                            4096,
                            st,
                            mask_bounds=(ci > 0 or not cfg.no_mask0),
                        )
                else:
                    for ci in range(4):
                        st = _step_2k_blk(slot, q_bf16, rel_first, rel_end, ci * 2048, 2048, st)
                o_vmem_ref[q_row] = finalize_out(st[1], st[2])

            if cfg.gear_4k and max_ring_pages * page_size >= 9216:
                @pl.when((rel_end > 8192) & (rel_end <= 9216))
                def _mid_9k_g4k():
                    st = init_new_token_state(q_bf16, q_row)
                    sanitize_slot_v2(slot, rel_first, rel_end, n_pages, cdiv(9216, page_size))
                    for ci in range(2):
                        st = _step_2k_blk(
                            slot,
                            q_bf16,
                            rel_first,
                            rel_end,
                            ci * 4096,
                            4096,
                            st,
                            mask_bounds=(not cfg.no_mask0),
                        )
                    st = _step_2k_blk(
                        slot, q_bf16, rel_first, rel_end, 8192, 1024, st, mask_bounds=True
                    )
                    o_vmem_ref[q_row] = finalize_out(st[1], st[2])

            if cfg.gear_4k and max_ring_pages * page_size >= 10240:
                @pl.when((rel_end > 9216) & (rel_end <= 10240))
                def _mid_10k_g4k():
                    st = init_new_token_state(q_bf16, q_row)
                    sanitize_slot_v2(slot, rel_first, rel_end, n_pages, cdiv(10240, page_size))
                    for ci in range(2):
                        st = _step_2k_blk(
                            slot,
                            q_bf16,
                            rel_first,
                            rel_end,
                            ci * 4096,
                            4096,
                            st,
                            mask_bounds=(not cfg.no_mask0),
                        )
                    st = _step_2k_blk(
                        slot, q_bf16, rel_first, rel_end, 8192, 2048, st, mask_bounds=True
                    )
                    o_vmem_ref[q_row] = finalize_out(st[1], st[2])

            gt_thresh = (
                10240
                if (cfg.gear_4k and max_ring_pages * page_size >= 10240)
                else (9216 if (cfg.gear_4k and max_ring_pages * page_size >= 9216) else 8192)
            )

            @pl.when(rel_end > gt_thresh)
            def _long_gt8k_u():
                o_vmem_ref[q_row] = compute_streamdyn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages
                )
        else:
            @pl.when(rel_end > 1024)
            def _long_small_cap():
                o_vmem_ref[q_row] = compute_streamdyn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages
                )

    def store_swafast2(slot, q_bf16, q_row, rel_first, rel_end, n_pages):
        """Lazy-HWM SWA decode engine (0 upfront zeroing) with per-seq 512/1024/full gears."""
        if probe == "nocompute":
            _, l0, a0 = init_new_token_state(q_bf16, q_row)
            o_vmem_ref[q_row] = finalize_out(l0, a0)
            return

        if max_ring_pages > 8:
            @pl.when(rel_end <= 512)
            def _swa2_512():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, 512, cfg.pv_bf16, True
                )

            @pl.when((rel_end > 512) & (rel_end <= 1024))
            def _swa2_1024():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, 1024, cfg.pv_bf16, True
                )

            @pl.when(rel_end > 1024)
            def _swa2_full():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, max_ring_pages * page_size, cfg.pv_bf16, True
                )
        elif max_ring_pages > 4:
            @pl.when(rel_end <= 512)
            def _swa2_s_512():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, 512, cfg.pv_bf16, True
                )

            @pl.when(rel_end > 512)
            def _swa2_s_rem():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, max_ring_pages * page_size, cfg.pv_bf16, True
                )
        else:
            o_vmem_ref[q_row] = _one_shot_attn(
                slot, q_bf16, q_row, rel_first, rel_end, n_pages, max_ring_pages * page_size, cfg.pv_bf16, True
            )

    def store_swafast(slot, q_bf16, q_row, rel_first, rel_end, n_pages):
        """Per-sequence 2-gear SWA decode engine: c1024 (<=8p, 1/1k) + stream1p (9..17p, 8k/1k)."""
        if probe == "nocompute":
            _, l0, a0 = init_new_token_state(q_bf16, q_row)
            o_vmem_ref[q_row] = finalize_out(l0, a0)
            return

        if max_ring_pages > 8:
            @pl.when(rel_end <= 1024)
            def _swa_short():
                o_vmem_ref[q_row] = _one_shot_attn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages, 1024, cfg.pv_bf16, True
                )

            @pl.when(rel_end > 1024)
            def _swa_full_win():
                o_vmem_ref[q_row] = compute_stream1p(
                    slot, q_bf16, q_row, rel_first, rel_end, max_ring_pages
                )
        else:
            o_vmem_ref[q_row] = compute_stream1p(
                slot, q_bf16, q_row, rel_first, rel_end, max_ring_pages
            )

    def run_single_seq_compute(slot, q_row, active, rel_first, rel_end, n_pages):
        @pl.when(active)
        def _():
            if (
                dq > 1
                and cfg.loop_kind in ("stream0b", "stream1p", "stream2ld")
                and one_block_per_seq
            ):
                store_stream1p_multiq(
                    slot, q_row, rel_first, rel_end, max_ring_pages
                )
                return
            q_bf16 = (
                q_vmem_ref[q_row]
                if dq == 1
                else q_vmem_ref[pl.ds(q_row, dq)].reshape(q_rows, 128)
            )
            if dq == 1 and cfg.loop_kind == "w7auto" and one_block_per_seq:
                store_w7auto(slot, q_bf16, q_row, rel_first, rel_end, n_pages)
                return
            if dq == 1 and cfg.loop_kind == "swafast2" and one_block_per_seq:
                store_swafast2(slot, q_bf16, q_row, rel_first, rel_end, n_pages)
                return
            if dq == 1 and cfg.loop_kind == "swafast" and one_block_per_seq:
                store_swafast(slot, q_bf16, q_row, rel_first, rel_end, n_pages)
                return
            if dq in (1, 2) and cfg.loop_kind == "stream2h" and sliding_window == 2049:
                out = compute_swa_stream2h(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages
                )
            elif (
                cfg.loop_kind == "streamdyn"
                and one_block_per_seq
                and cfg.dyn2k_unroll
                and max_ring_pages * page_size >= 8192
                and dq == 1
                and not cfg.fp8_dots
            ):
                compute_stream2k_unrolled(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages
                )
                return
            elif cfg.loop_kind == "streamdyn" and one_block_per_seq:
                out = compute_streamdyn(
                    slot, q_bf16, q_row, rel_first, rel_end, n_pages
                )
            elif dq == 1 and cfg.loop_kind == "clean1p" and sliding_window == 2049:
                out = compute_swa_clean1p_2176(slot, q_bf16, q_row, rel_first, rel_end)
            elif (
                dq == 1
                and cfg.loop_kind in ("stream0b", "stream1p", "stream2ld")
                and one_block_per_seq
            ):
                out = compute_stream1p(
                    slot,
                    q_bf16,
                    q_row,
                    rel_first,
                    rel_end,
                    max_ring_pages,
                )
            else:
                sanitize_slot_boundaries(slot, rel_first, rel_end, max_ring_pages)
                m0, l0, a0 = init_new_token_state(q_bf16, q_row)
                state_vmem_ref[0] = m0
                state_vmem_ref[1] = l0
                state_vmem_ref[2] = a0
                compute_subblock2k_loop(slot, q_bf16, rel_first, rel_end)
                out = finalize_out(state_vmem_ref[1], state_vmem_ref[2])
            if dq == 1:
                o_vmem_ref[q_row] = out
            else:
                o_vmem_ref[pl.ds(q_row, dq)] = out.reshape(dq, q_out_head_rows, 128)

        @pl.when(~active)
        def _():
            if dq == 1:
                o_vmem_ref[q_row] = jnp.zeros((q_out_rows, 128), jnp.bfloat16)
            else:
                o_vmem_ref[pl.ds(q_row, dq)] = jnp.zeros(
                    (dq, q_out_head_rows, 128), jnp.bfloat16
                )

    if (
        prestarted_bkv_state is None
        and use_branchless_sanitize
        and cfg.loop_kind != "swabatch"
        and not cfg.lazy_slot_zero
        and probe != "nocompute"
    ):
        zero_all_slots_once()

    written_cnt_smem_ref[0] = 0
    if cfg.lazy_slot_zero:
        for s_z in range(num_slots):
            written_cnt_smem_ref[2 + s_z] = 0
    first_inactive_row = cu_q_lens_ref[n_active]

    @pl.when(first_inactive_row < max_num_tokens)
    def _zero_inactive_o():
        z = jnp.zeros((q_out_head_rows, 128), jnp.bfloat16)
        lax.fori_loop(
            first_inactive_row,
            max_num_tokens,
            lambda r, _: o_vmem_ref.__setitem__(r, z),
            None,
        )

    if cfg.shared_group_size > 0 or cfg.loop_kind == "shared16":
        group = cfg.shared_group_size or 16
        ahead_sh = num_slots - 1
        meta_base_sh = 2 + num_slots
        priv_smem_cap = written_cnt_smem_ref.shape[0] - meta_base_sh
        pps_in_kern = num_page_indices // max_num_seqs
        use_flat_priv = (
            probe != "nocompute"
            and priv_smem_cap >= 4 * group * cdiv(max(pps_in_kern, 1), max_ring_pages)
        )

        if use_hoist_kv:
            precompute_hoisted_new_kv()

        # Every row keeps independent FP32 online-softmax state. Only equal
        # complete physical pages can feed the group's QK and PV operations.
        # Unpack K before QK and V after exp2 in 8-bit VREGs (4x smaller than
        # uint32) and slice large shared Q batches (shared32/64) into 256-row
        # chunks without an inner K/state RMW loop.
        def _accum_shared_or_priv(
            q,
            head_offset,
            slot,
            mask_valid,
            valid,
            *,
            active_heads=None,
            q_exp2_sc=None,
            blk_width: int | None = None,
        ):
            eff_width = max_ring_rows if blk_width is None else blk_width
            heads = q.shape[0]
            if kv_pack_bd4 and q.shape[0] >= 16:
                tgt_q_tile = cfg.shared_q_tile * 2
                q_tile = tgt_q_tile if heads > tgt_q_tile else heads
            else:
                q_tile = cfg.shared_q_tile if heads > cfg.shared_q_tile else heads
            valid_arg = valid if mask_valid else None
            # Direct non-split matmul/softmax beats 2-half shsplit across DENSE2 and BD4
            # shared blocks (-4% to -11% at B=8/16/64) by halving matmul/exp2/VMEM reads.
            use_split_bd4 = False
            use_split = False
            use_fp8_pv = cfg.shared_fp8_qkpv or cfg.shared_fp8_pv_only

            def _accum_q_tile(q0: int):
                q_w = min(q_tile, heads - q0)
                q_s = q if q_w == heads else q[q0 : q0 + q_w]
                if kv_pack_bd4:
                    q_s = _to_bd4_q(q_s)
                    n_seq_t = q_s.shape[0] // 16
                    out_w = n_seq_t * 8
                    out_off = q0 // 2 if q_head_rows == 16 else q0
                    lanes = pl.ds(head_offset + out_off, out_w)
                    eff_sc = (
                        q_exp2_sc[out_off : out_off + out_w]
                        if q_exp2_sc is not None
                        else jnp.float32(exp2_scale)
                    )
                    if use_split_bd4:
                        width_s = max_ring_rows // 4
                        k01, k23 = _load_bkv_bd4_pair_parts(
                            bkv_vmem_ref,
                            slot,
                            0,
                            max_ring_rows,
                            kind="k",
                            page_size=page_size,
                            interpret_mode=interpret_mode,
                        )
                        s_e0, s_o0 = _bd4_logits_split(q_s, k01, n_seq_t, width_s)
                        s_e1, s_o1 = _bd4_logits_split(q_s, k23, n_seq_t, width_s)
                        m01 = jnp.maximum(s_e0, s_o0)
                        m23 = jnp.maximum(s_e1, s_o1)
                        blk_max = jnp.maximum(
                            jnp.max(
                                _fold128_op(m01, jnp.maximum)
                                if (cfg.fold_128 and width_s >= 128 and width_s % 128 == 0)
                                else m01,
                                axis=1,
                                keepdims=True,
                            ),
                            jnp.max(
                                _fold128_op(m23, jnp.maximum)
                                if (cfg.fold_128 and width_s >= 128 and width_s % 128 == 0)
                                else m23,
                                axis=1,
                                keepdims=True,
                            ),
                        )
                    else:
                        width_s = eff_width // 2
                        k_pair = _bkv_fp8_chunk(
                            slot, 0, eff_width, kind="k", valid_tokens=valid_arg
                        )
                        s_e, s_o = _bd4_logits_split(q_s, k_pair, n_seq_t, width_s)
                        if mask_valid:
                            c_e = _bd4_c_even_iota(0, width_s, out_w)
                            c_o = c_e + 1
                            s_e = jnp.where(c_e < valid, s_e, _MASK_LOGIT)
                            s_o = jnp.where(c_o < valid, s_o, _MASK_LOGIT)
                        s_mx = jnp.maximum(s_e, s_o)
                        blk_max = jnp.max(
                            _fold128_op(s_mx, jnp.maximum)
                            if (cfg.fold_128 and width_s >= 128 and width_s % 128 == 0)
                            else s_mx,
                            axis=1,
                            keepdims=True,
                        )
                    blk_m_128 = _broadcast_minor_128(blk_max)
                    old_m = state_vmem_ref[0, lanes, :]
                    m = jnp.maximum(old_m, blk_m_128)
                    alpha = jnp.exp2((old_m - m) * eff_sc)
                    eff_sc_w = (
                        _broadcast_minor_128_any(eff_sc, width_s)
                        if (q_exp2_sc is not None and not cfg.fold_col)
                        else eff_sc
                    )
                    if use_fp8_pv:
                        ref_m_128 = blk_m_128
                        beta = jnp.exp2((blk_m_128 - m) * eff_sc)
                        l_scale = beta
                        pv_scale = beta * jnp.float32(v_scale_val / 448.0)
                    else:
                        ref_m_128 = m
                        l_scale = jnp.float32(1.0)
                        pv_scale = jnp.float32(v_scale_val)
                    use_f128_sh = cfg.fold_128 and (width_s >= 128) and (width_s % 128 == 0)
                    use_fcol_sh = cfg.fold_col and use_f128_sh
                    ref_m_w = (
                        None
                        if use_fcol_sh
                        else _broadcast_minor_128_any(ref_m_128, width_s)
                    )

                    def _pv_bd4_half(s_xe, s_xo, v_blk):
                        if use_fcol_sh:
                            p_xe, l_xe = _exp2_fold_col(s_xe, ref_m_128, eff_sc)
                            p_xo, l_xo = _exp2_fold_col(s_xo, ref_m_128, eff_sc)
                            l_tot = l_xe + l_xo
                        elif use_f128_sh:
                            m_1 = ref_m_128[:, :1]
                            p_xe = jnp.exp2((s_xe - m_1) * eff_sc)
                            p_xo = jnp.exp2((s_xo - m_1) * eff_sc)
                            l_tot = _fold128_op(p_xe + p_xo, jnp.add)
                        else:
                            p_xe = jnp.exp2((s_xe - ref_m_w) * eff_sc_w)
                            p_xo = jnp.exp2((s_xo - ref_m_w) * eff_sc_w)
                            if width_s >= 128 and width_s % 128 == 0:
                                l_tot = jnp.sum(
                                    (p_xe + p_xo).reshape(out_w, width_s // 128, 128),
                                    axis=1,
                                )
                            else:
                                l_tot = _broadcast_minor_128(
                                    jnp.sum(p_xe + p_xo, axis=1, keepdims=True)
                                ) * jnp.float32(1.0 / 128.0)
                        p_bd = jnp.concatenate(
                            [
                                p_xe.reshape(n_seq_t, 8, width_s),
                                p_xo.reshape(n_seq_t, 8, width_s),
                            ],
                            axis=1,
                        ).reshape(n_seq_t * 16, width_s)
                        if use_fp8_pv:
                            p_dot = (p_bd * jnp.float32(448.0)).astype(
                                jnp.float8_e4m3fn
                            )
                        elif cfg.pv_bf16:
                            p_dot = p_bd.astype(jnp.bfloat16)
                        else:
                            p_dot = p_bd
                        o_bd = jnp.matmul(
                            p_dot, v_blk, preferred_element_type=jnp.float32
                        ).reshape(n_seq_t, 2, 8, 128)
                        o_odd = _roll_lanes_fast(
                            o_bd[:, 1].reshape(out_w, 128),
                            64,
                            interpret_mode=interpret_mode,
                        ).reshape(n_seq_t, 8, 128)
                        return l_tot, (o_bd[:, 0] + o_odd).reshape(out_w, 128)

                    if use_split_bd4:
                        v01, v23 = _load_bkv_bd4_pair_parts(
                            bkv_vmem_ref,
                            slot,
                            0,
                            max_ring_rows,
                            kind="v",
                            page_size=page_size,
                            interpret_mode=interpret_mode,
                        )
                        l01, pv01 = _pv_bd4_half(s_e0, s_o0, v01)
                        l23, pv23 = _pv_bd4_half(s_e1, s_o1, v23)
                        l_add = (l01 + l23) * l_scale
                        pv = pv01 + pv23
                    else:
                        v_pair = _bkv_fp8_chunk(
                            slot, 0, eff_width, kind="v", valid_tokens=valid_arg
                        )
                        l_sum, pv = _pv_bd4_half(s_e, s_o, v_pair)
                        l_add = l_sum * l_scale
                    state_vmem_ref[0, lanes, :] = m
                    state_vmem_ref[1, lanes, :] = (
                        state_vmem_ref[1, lanes, :] * alpha + l_add
                    )
                    state_vmem_ref[2, lanes, :] = (
                        state_vmem_ref[2, lanes, :] * alpha + pv * pv_scale
                    )
                    return

                lanes = pl.ds(head_offset + q0, q_w)
                eff_sc = (
                    (q_exp2_sc if q_w == heads else q_exp2_sc[q0 : q0 + q_w])
                    if q_exp2_sc is not None
                    else jnp.float32(exp2_scale)
                )

                if use_split:
                    width_s = max_ring_rows // 2
                    k_e = _load_bkv_fp8_split_half(
                        bkv_vmem_ref,
                        slot,
                        0,
                        max_ring_rows,
                        kind="k",
                        odd=False,
                        page_size=page_size,
                        interpret_mode=interpret_mode,
                    )
                    s_e = jnp.matmul(q_s, k_e.T, preferred_element_type=jnp.float32)
                    k_o = _load_bkv_fp8_split_half(
                        bkv_vmem_ref,
                        slot,
                        0,
                        max_ring_rows,
                        kind="k",
                        odd=True,
                        page_size=page_size,
                        interpret_mode=interpret_mode,
                    )
                    s_o = jnp.matmul(q_s, k_o.T, preferred_element_type=jnp.float32)
                    blk_max = jnp.maximum(
                        jnp.max(s_e, axis=1, keepdims=True),
                        jnp.max(s_o, axis=1, keepdims=True),
                    )
                else:
                    width_s = eff_width
                    k = _bkv_fp8_chunk(
                        slot, 0, eff_width, kind="k", valid_tokens=valid_arg
                    )
                    scores = jnp.matmul(q_s, k.T, preferred_element_type=jnp.float32)
                    if mask_valid:
                        valid_col = _col_iota(0, eff_width, q_w) < valid
                        scores = jnp.where(valid_col, scores, _MASK_LOGIT)
                    blk_max = jnp.max(
                        _fold128_op(scores, jnp.maximum) if cfg.fold_128 else scores,
                        axis=1,
                        keepdims=True,
                    )

                blk_m_128 = _broadcast_minor_128(blk_max)
                old_m = state_vmem_ref[0, lanes, :]
                m = jnp.maximum(old_m, blk_m_128)
                alpha = jnp.exp2((old_m - m) * eff_sc)
                eff_sc_w = (
                    _broadcast_minor_128_any(eff_sc, width_s)
                    if (q_exp2_sc is not None and not cfg.fold_col)
                    else eff_sc
                )
                if use_fp8_pv:
                    # Rescale exp2 by local block peak so max(p_local) == 1.0, keeping
                    # (p_local * 448) across full E4M3 range with zero extra max reduction.
                    ref_m_128 = blk_m_128
                    beta = jnp.exp2((blk_m_128 - m) * eff_sc)
                    l_scale = beta
                    pv_scale = beta * jnp.float32(v_scale_val / 448.0)
                else:
                    ref_m_128 = m
                    l_scale = jnp.float32(1.0)
                    pv_scale = jnp.float32(v_scale_val)

                ref_m_w = None if cfg.fold_col else _broadcast_minor_128_any(ref_m_128, width_s)

                def _pv_half(s_x: jax.Array, v_x: jax.Array) -> tuple[jax.Array, jax.Array]:
                    if cfg.fold_col:
                        p_x, l_x = _exp2_fold_col(s_x, ref_m_128, eff_sc)
                    elif cfg.fold_128:
                        p_x = jnp.exp2((s_x - ref_m_128[:, :1]) * eff_sc)
                        l_x = _fold128_op(p_x, jnp.add)
                    else:
                        p_x = jnp.exp2((s_x - ref_m_w) * eff_sc_w)
                        l_x = jnp.sum(p_x.reshape(q_w, width_s // 128, 128), axis=1)
                    if use_fp8_pv:
                        p_dot = (p_x * jnp.float32(448.0)).astype(jnp.float8_e4m3fn)
                    elif cfg.pv_bf16:
                        p_dot = p_x.astype(jnp.bfloat16)
                    else:
                        p_dot = p_x
                    return l_x, jnp.matmul(
                        p_dot, v_x, preferred_element_type=jnp.float32
                    )

                if use_split:
                    # Evaluate even then odd halves sequentially so only one [q_w, width_s]
                    # probability tensor and one V half are live at once (-50% VREG/VMEM peak).
                    l_e, pv_e = _pv_half(
                        s_e,
                        _load_bkv_fp8_split_half(
                            bkv_vmem_ref,
                            slot,
                            0,
                            max_ring_rows,
                            kind="v",
                            odd=False,
                            page_size=page_size,
                            interpret_mode=interpret_mode,
                        ),
                    )
                    l_o, pv_o = _pv_half(
                        s_o,
                        _load_bkv_fp8_split_half(
                            bkv_vmem_ref,
                            slot,
                            0,
                            max_ring_rows,
                            kind="v",
                            odd=True,
                            page_size=page_size,
                            interpret_mode=interpret_mode,
                        ),
                    )
                    l_add = (l_e + l_o) * l_scale
                    pv = pv_e + pv_o
                else:
                    v_full = _bkv_fp8_chunk(
                        slot, 0, eff_width, kind="v", valid_tokens=valid_arg
                    )
                    l_sum, pv = _pv_half(scores, v_full)
                    l_add = l_sum * l_scale
                state_vmem_ref[0, lanes, :] = m
                state_vmem_ref[1, lanes, :] = (
                    state_vmem_ref[1, lanes, :] * alpha + l_add
                )
                state_vmem_ref[2, lanes, :] = (
                    state_vmem_ref[2, lanes, :] * alpha + pv * pv_scale
                )

            for q0 in range(0, heads, q_tile):
                if active_heads is not None and q0 > 0:

                    @pl.when(active_heads > q0)
                    def _run_active_q_tile(q0=q0):
                        _accum_q_tile(q0)

                else:
                    _accum_q_tile(q0)

        def _accum_tail_shared_or_priv(
            q,
            head_offset,
            slot,
            valid,
            *,
            active_heads=None,
            q_exp2_sc=None,
            tight_priv: bool = False,
        ):
            cands = (512, 1024, 2048) if tight_priv else (2048,)
            tail_widths = [
                w
                for w in cands
                if w < max_ring_rows and (w % page_size == 0) and (w % 128 == 0)
            ] + [max_ring_rows]
            prev_w = 0
            for tw in tail_widths:
                if prev_w == 0:
                    cond = valid <= tw
                elif tw < max_ring_rows:
                    cond = (valid > prev_w) & (valid <= tw)
                else:
                    cond = valid > prev_w

                @pl.when(cond)
                def _run_tail_w(w=tw):
                    _accum_shared_or_priv(
                        q,
                        head_offset,
                        slot,
                        True,
                        valid,
                        active_heads=active_heads,
                        q_exp2_sc=q_exp2_sc,
                        blk_width=w,
                    )

                prev_w = tw

        def consume(
            q,
            head_offset,
            table_base,
            first_page,
            end_token,
            *,
            active_heads=None,
            q_exp2_sc=None,
        ):
            n_pages = cdiv(end_token, page_size) - first_page
            n_blocks = cdiv(n_pages, max_ring_pages)

            def issue(block):
                offset = block * max_ring_pages
                count = jnp.minimum(n_pages - offset, max_ring_pages)
                start_fetch_pages(
                    table_base + first_page + offset, count, block % num_slots
                )

            for b_init in range(ahead_sh):

                @pl.when(b_init < n_blocks)
                def _(bi=b_init):
                    issue(jnp.int32(bi))

            @pl.loop(0, n_blocks)
            def _block(block):
                slot = block % num_slots
                offset = block * max_ring_pages

                @pl.when(block + ahead_sh < n_blocks)
                def _next():
                    issue(block + ahead_sh)

                count = jnp.minimum(n_pages - offset, max_ring_pages)
                wait_fetch_pages(count, slot)
                if probe == "nocompute":
                    return
                valid = end_token - (first_page + offset) * page_size

                @pl.when(valid >= max_ring_rows)
                def _full_blk():
                    _accum_shared_or_priv(
                        q,
                        head_offset,
                        slot,
                        False,
                        valid,
                        active_heads=active_heads,
                        q_exp2_sc=q_exp2_sc,
                    )

                @pl.when(valid < max_ring_rows)
                def _tail_blk():
                    _accum_tail_shared_or_priv(
                        q,
                        head_offset,
                        slot,
                        valid,
                        active_heads=active_heads,
                        q_exp2_sc=q_exp2_sc,
                        tight_priv=False,
                    )

            last_slot = jnp.maximum(n_blocks - 1, 0) % num_slots
            last_loc = jnp.maximum(end_token - first_page * page_size - 1, 0) % max_ring_rows
            return last_slot, last_loc

        @pl.loop(0, cdiv(n_active, group))
        def _group(group_index):
            first = group * group_index
            size = jnp.minimum(group, n_active - first)
            first_row, _, _, _, _, _, first_base = seq_info(first)
            shared = distribution_ref[3 + group_index]

            # A partial group can still reuse its physical pages. Its batched
            # dot reads all group*16 rows; initialize the unused state lanes too.
            @pl.when((size < group) & (shared > 0))
            def _init_padding():
                state_vmem_ref[...] = jnp.zeros(state_vmem_ref.shape, jnp.float32)

            @pl.loop(0, size)
            def _init(i):
                row, active, *_ = seq_info(first + i)

                @pl.when(active)
                def _():
                    init = init_new_token_state(q_vmem_ref[row], row)
                    for j in range(3):
                        state_vmem_ref[j, pl.ds(i * q_out_head_rows, q_out_head_rows), :] = init[j]

            @pl.when(shared > 0)
            def _common():
                q = q_vmem_ref[pl.ds(first_row, group), :, :].reshape(group * q_head_rows, 128)
                active_h = size * q_head_rows
                if cfg.shared_fp8_qkpv:
                    q_f32 = q.astype(jnp.float32)
                    q_amax = jnp.max(jnp.abs(q_f32), axis=1, keepdims=True)
                    q_gain = jnp.where(
                        q_amax == 0.0,
                        jnp.float32(1.0),
                        jnp.float32(448.0) / jnp.maximum(q_amax, jnp.float32(1e-12)),
                    )
                    q_deq = jnp.where(
                        q_amax == 0.0,
                        jnp.float32(1.0),
                        q_amax * jnp.float32(1.0 / 448.0),
                    )
                    q_gain_128 = _broadcast_minor_128(q_gain)
                    q_deq_128 = _broadcast_minor_128(q_deq)
                    q_in = (q_f32 * q_gain_128).astype(jnp.float8_e4m3fn)
                    if kv_pack_bd4 and q_head_rows == 16:
                        q_gain_st = q_gain_128.reshape(group, 2, 8, 128)[:, 0].reshape(
                            group * 8, 128
                        )
                        q_deq_st = q_deq_128.reshape(group, 2, 8, 128)[:, 0].reshape(
                            group * 8, 128
                        )
                    else:
                        q_gain_st = q_gain_128
                        q_deq_st = q_deq_128
                    q_exp2_sc = q_deq_st * jnp.float32(exp2_scale)
                    state_vmem_ref[0] = state_vmem_ref[0] * q_gain_st
                    consume(
                        q_in,
                        0,
                        first_base,
                        0,
                        shared * page_size,
                        active_heads=active_h,
                        q_exp2_sc=q_exp2_sc,
                    )
                    state_vmem_ref[0] = state_vmem_ref[0] * q_deq_st
                else:
                    consume(
                        q,
                        0,
                        first_base,
                        0,
                        shared * page_size,
                        active_heads=active_h,
                    )

            if use_flat_priv:
                written_cnt_smem_ref[1] = 0

                @pl.loop(0, size)
                def _build_priv_steps(i):
                    seq = first + i
                    q_r, act, kvl, _, rel_end, total_pgs, seq_pg_base = seq_info(seq)

                    @pl.when(~act)
                    def _():
                        o_vmem_ref[q_r] = jnp.zeros((q_out_head_rows, 128), jnp.bfloat16)

                    @pl.when(act)
                    def _():
                        priv_pgs = jnp.maximum(total_pgs - shared, 0)
                        priv_end = jnp.maximum(rel_end - shared * page_size, 0)

                        @pl.when(priv_pgs <= 0)
                        def _only_shared():
                            lanes = pl.ds(i * q_out_head_rows, q_out_head_rows)
                            o_vmem_ref[q_r] = finalize_out(
                                state_vmem_ref[1, lanes, :], state_vmem_ref[2, lanes, :]
                            )
                            sh_n_b = cdiv(shared, max_ring_pages)
                            sh_last_slot = jnp.maximum(sh_n_b - 1, 0) % num_slots
                            sh_last_loc = jnp.maximum(rel_end - 1, 0) % max_ring_rows
                            enqueue_kv_write(seq, q_r, act, kvl, sh_last_slot, sh_last_loc)

                        @pl.when(priv_pgs > 0)
                        def _emit_priv():
                            n_b = cdiv(priv_pgs, max_ring_pages)

                            @pl.loop(0, n_b)
                            def _push_p(b_i):
                                cur_idx = written_cnt_smem_ref[1]
                                written_cnt_smem_ref[1] = cur_idx + 1
                                off_m = meta_base_sh + 4 * cur_idx
                                cur_pg = jnp.minimum(
                                    priv_pgs - b_i * max_ring_pages, max_ring_pages
                                )
                                loc_end = jnp.clip(
                                    priv_end - b_i * max_ring_rows, 0, max_ring_rows
                                )
                                fl = jnp.where(
                                    b_i == 0, jnp.int32(1 << 16), jnp.int32(0)
                                ) | jnp.where(
                                    b_i + 1 == n_b, jnp.int32(1 << 17), jnp.int32(0)
                                )
                                written_cnt_smem_ref[off_m + 0] = (seq << 16) | q_r
                                written_cnt_smem_ref[off_m + 1] = (
                                    seq_pg_base + shared + b_i * max_ring_pages
                                )
                                written_cnt_smem_ref[off_m + 2] = cur_pg
                                written_cnt_smem_ref[off_m + 3] = loc_end | fl

                total_priv_steps = written_cnt_smem_ref[1]
                for st_init in range(ahead_sh):

                    @pl.when(st_init < total_priv_steps)
                    def _(si=st_init):
                        om = meta_base_sh + 4 * si
                        start_fetch_pages(
                            written_cnt_smem_ref[om + 1],
                            written_cnt_smem_ref[om + 2],
                            si,
                        )

                @pl.loop(0, total_priv_steps)
                def _exec_priv(st_i):
                    slot = st_i % num_slots
                    fut_st = st_i + ahead_sh
                    fut_slot = (slot + ahead_sh) % num_slots

                    @pl.when(fut_st < total_priv_steps)
                    def _pf_priv():
                        fom = meta_base_sh + 4 * fut_st
                        start_fetch_pages(
                            written_cnt_smem_ref[fom + 1],
                            written_cnt_smem_ref[fom + 2],
                            fut_slot,
                        )

                    om = meta_base_sh + 4 * st_i
                    packed_qr = written_cnt_smem_ref[om + 0]
                    q_r = packed_qr & jnp.int32(0xFFFF)
                    seq_id = packed_qr >> 16
                    cur_pg = written_cnt_smem_ref[om + 2]
                    packed_le = written_cnt_smem_ref[om + 3]
                    loc_end = packed_le & jnp.int32(0xFFFF)
                    is_last = (packed_le & jnp.int32(1 << 17)) != 0
                    head_off = (seq_id - first) * q_out_head_rows
                    lanes = pl.ds(head_off, q_out_head_rows)

                    wait_fetch_pages(cur_pg, slot)
                    q_bf16 = q_vmem_ref[q_r]

                    @pl.when(loc_end >= max_ring_rows)
                    def _full_priv_blk():
                        _accum_shared_or_priv(q_bf16, head_off, slot, False, loc_end)

                    @pl.when(loc_end < max_ring_rows)
                    def _tail_priv_blk():
                        _accum_tail_shared_or_priv(
                            q_bf16, head_off, slot, loc_end, tight_priv=True
                        )

                    @pl.when(is_last)
                    def _fin_priv():
                        o_vmem_ref[q_r] = finalize_out(
                            state_vmem_ref[1, lanes, :], state_vmem_ref[2, lanes, :]
                        )
                        enqueue_kv_write(
                            seq_id,
                            q_r,
                            jnp.bool_(True),
                            kv_lens_ref[seq_id],
                            slot,
                            loc_end - 1,
                        )
            else:
                if kv_pack_bd4 or kv_pack_dense2:
                    sh_n_b = cdiv(shared, max_ring_pages)
                    sh_last_slot = jnp.maximum(sh_n_b - 1, 0) % num_slots

                    @pl.loop(0, size)
                    def _only_shared_pass(i):
                        seq = first + i
                        row, active, kv_len, _, end, total_pgs, _ = seq_info(seq)

                        @pl.when(active & (total_pgs <= shared))
                        def _():
                            lanes = pl.ds(i * q_out_head_rows, q_out_head_rows)
                            o_vmem_ref[row] = finalize_out(
                                state_vmem_ref[1, lanes, :], state_vmem_ref[2, lanes, :]
                            )
                            sh_last_loc = jnp.maximum(end - 1, 0) % max_ring_rows
                            enqueue_kv_write(seq, row, active, kv_len, sh_last_slot, sh_last_loc)

                        @pl.when(~active)
                        def _():
                            o_vmem_ref[row] = jnp.zeros((q_out_head_rows, 128), jnp.bfloat16)

                @pl.loop(0, size)
                def _private(i):
                    seq = first + i
                    row, active, kv_len, _, end, total_pgs, base = seq_info(seq)
                    run_priv = (
                        active & (total_pgs > shared)
                        if (kv_pack_bd4 or kv_pack_dense2)
                        else active
                    )

                    @pl.when(run_priv)
                    def _():
                        lanes = pl.ds(i * q_out_head_rows, q_out_head_rows)
                        last_slot, last_loc = consume(
                            q_vmem_ref[row], i * q_out_head_rows, base, shared, end
                        )
                        o_vmem_ref[row] = finalize_out(
                            state_vmem_ref[1, lanes, :], state_vmem_ref[2, lanes, :]
                        )
                        enqueue_kv_write(seq, row, active, kv_len, last_slot, last_loc)

                    if not (kv_pack_bd4 or kv_pack_dense2):
                        @pl.when(~active)
                        def _():
                            o_vmem_ref[row] = jnp.zeros((q_out_head_rows, 128), jnp.bfloat16)
    elif cfg.loop_kind == "ring4k":
        meta_base = 2 + num_slots
        written_cnt_smem_ref[1] = 0

        @pl.loop(0, n_active)
        def _build_ring4k_steps(seq):
            q_r, act, kvl, _, rel_end, total_pgs, seq_pg_base = seq_info(seq)

            @pl.when(act)
            def _():
                @pl.when(total_pgs <= 0)
                def _only_new_tok():
                    enqueue_kv_write(seq, q_r, act, kvl)
                    _, l0, a0 = init_new_token_state(q_vmem_ref[q_r], q_r)
                    o_vmem_ref[q_r] = finalize_out(l0, a0)

                @pl.when(total_pgs > 0)
                def _emit_blks():
                    n_b = cdiv(total_pgs, max_ring_pages)

                    @pl.loop(0, n_b)
                    def _push_b(b_i):
                        cur_idx = written_cnt_smem_ref[1]
                        written_cnt_smem_ref[1] = cur_idx + 1
                        off_m = meta_base + 4 * cur_idx
                        cur_pg = jnp.minimum(
                            total_pgs - b_i * max_ring_pages, max_ring_pages
                        )
                        loc_end = jnp.clip(
                            rel_end - b_i * max_ring_rows, 0, max_ring_rows
                        )
                        fl = jnp.where(b_i == 0, jnp.int32(1 << 16), jnp.int32(0)) | jnp.where(
                            b_i + 1 == n_b, jnp.int32(1 << 17), jnp.int32(0)
                        )
                        written_cnt_smem_ref[off_m + 0] = (seq << 16) | q_r
                        written_cnt_smem_ref[off_m + 1] = seq_pg_base + b_i * max_ring_pages
                        written_cnt_smem_ref[off_m + 2] = cur_pg
                        written_cnt_smem_ref[off_m + 3] = loc_end | fl

            @pl.when(~act)
            def _():
                o_vmem_ref[q_r] = jnp.zeros((q_out_rows, 128), jnp.bfloat16)

        total_4k_steps = written_cnt_smem_ref[1]
        ahead_4k = num_slots - 1
        for st_init in range(ahead_4k):

            @pl.when(st_init < total_4k_steps)
            def _(si=st_init):
                om = meta_base + 4 * si
                start_fetch_pages(
                    written_cnt_smem_ref[om + 1], written_cnt_smem_ref[om + 2], si
                )
        if use_hoist_kv:
            precompute_hoisted_new_kv()

        @pl.loop(0, total_4k_steps)
        def _exec_ring4k(st_i):
            slot = st_i % num_slots
            fut_st = st_i + ahead_4k
            fut_slot = (slot + ahead_4k) % num_slots

            @pl.when(fut_st < total_4k_steps)
            def _pf_4k():
                fom = meta_base + 4 * fut_st
                start_fetch_pages(
                    written_cnt_smem_ref[fom + 1],
                    written_cnt_smem_ref[fom + 2],
                    fut_slot,
                )

            om = meta_base + 4 * st_i
            packed_qr = written_cnt_smem_ref[om + 0]
            q_r = packed_qr & jnp.int32(0xFFFF)
            seq_id = packed_qr >> 16
            cur_pg = written_cnt_smem_ref[om + 2]
            packed_le = written_cnt_smem_ref[om + 3]
            loc_end = packed_le & jnp.int32(0xFFFF)
            is_first = (packed_le & jnp.int32(1 << 16)) != 0
            is_last = (packed_le & jnp.int32(1 << 17)) != 0

            wait_fetch_pages(cur_pg, slot)
            q_bf16 = q_vmem_ref[q_r]

            @pl.when(is_first)
            def _init_st():
                m0, l0, a0 = init_new_token_state(q_bf16, q_r)
                state_vmem_ref[0] = m0
                state_vmem_ref[1] = l0
                state_vmem_ref[2] = a0

            if probe != "nocompute":
                def _exec_ring4k_blk(tw_k: int, *, mask_bounds: bool):
                    tw_pg = tw_k // page_size
                    if mask_bounds:
                        sanitize_slot_v2(slot, 0, loc_end, cur_pg, tw_pg)
                    else:
                        written_cnt_smem_ref[2 + slot] = max_ring_pages
                    if kv_pack_bd4:
                        m_new, l_new, acc_new = _bd4_step_blk(
                            slot,
                            q_bf16,
                            0,
                            loc_end,
                            0,
                            tw_k,
                            (state_vmem_ref[0], state_vmem_ref[1], state_vmem_ref[2]),
                            mask_bounds=mask_bounds,
                        )
                        state_vmem_ref[0] = m_new
                        state_vmem_ref[1] = l_new
                        state_vmem_ref[2] = acc_new
                        return
                    k_blk = _bkv_fp8_chunk(slot, 0, tw_k, kind="k")
                    s_blk = jnp.matmul(
                        q_bf16, k_blk.T, preferred_element_type=jnp.float32
                    )
                    if mask_bounds:
                        k_col = _col_iota(0, tw_k, q_rows)
                        s_blk = jnp.where(k_col < loc_end, s_blk, _MASK_LOGIT)
                    m_prev = state_vmem_ref[0]
                    if cfg.fold_col:
                        m_new = jnp.maximum(
                            m_prev,
                            _broadcast_minor_128(
                                jnp.max(
                                    _fold128_op(s_blk, jnp.maximum),
                                    axis=1,
                                    keepdims=True,
                                )
                            ),
                        )
                        alpha = jnp.exp2((m_prev - m_new) * exp2_scale)
                        p_blk, l_add = _exp2_fold_col(s_blk, m_new, exp2_scale)
                    elif cfg.fold_128:
                        m_new = jnp.maximum(
                            m_prev,
                            _broadcast_minor_128(
                                jnp.max(
                                    _fold128_op(s_blk, jnp.maximum),
                                    axis=1,
                                    keepdims=True,
                                )
                            ),
                        )
                        alpha = jnp.exp2((m_prev - m_new) * exp2_scale)
                        p_blk = jnp.exp2((s_blk - m_new[:, :1]) * exp2_scale)
                        l_add = _fold128_op(p_blk, jnp.add)
                    else:
                        s_pgs = jnp.max(
                            s_blk.reshape(q_rows, tw_k // 128, 128), axis=1
                        )
                        m_new = jnp.maximum(
                            m_prev,
                            _broadcast_minor_128(jnp.max(s_pgs, axis=1, keepdims=True)),
                        )
                        alpha = jnp.exp2((m_prev - m_new) * exp2_scale)
                        p_blk = jnp.exp2(
                            (s_blk - _broadcast_minor_128_any(m_new, tw_k))
                            * exp2_scale
                        )
                        l_add = jnp.sum(
                            p_blk.reshape(q_rows, tw_k // 128, 128), axis=1
                        )
                    v_blk = _bkv_fp8_chunk(slot, 0, tw_k, kind="v")
                    p_dot = p_blk.astype(jnp.bfloat16) if cfg.pv_bf16 else p_blk
                    pv = jnp.matmul(
                        p_dot, v_blk, preferred_element_type=jnp.float32
                    )
                    if v_scale is not None and v_scale_val != 1.0:
                        pv = pv * v_scale_val
                    state_vmem_ref[0] = m_new
                    state_vmem_ref[1] = state_vmem_ref[1] * alpha + l_add
                    state_vmem_ref[2] = state_vmem_ref[2] * alpha + pv

                @pl.when(loc_end == max_ring_rows)
                def _full_4k():
                    _exec_ring4k_blk(max_ring_rows, mask_bounds=False)

                if max_ring_rows >= 2048:
                    @pl.when(loc_end <= 1024)
                    def _tail_1k():
                        _exec_ring4k_blk(1024, mask_bounds=True)

                    if max_ring_rows >= 4096:
                        @pl.when((loc_end > 1024) & (loc_end <= 2048))
                        def _tail_2k():
                            _exec_ring4k_blk(2048, mask_bounds=True)

                        @pl.when((loc_end > 2048) & (loc_end < max_ring_rows))
                        def _tail_4k():
                            _exec_ring4k_blk(max_ring_rows, mask_bounds=True)
                    else:
                        @pl.when((loc_end > 1024) & (loc_end < max_ring_rows))
                        def _tail_2k_rem():
                            _exec_ring4k_blk(max_ring_rows, mask_bounds=True)
                else:
                    @pl.when(loc_end < max_ring_rows)
                    def _tail_small():
                        _exec_ring4k_blk(max_ring_rows, mask_bounds=True)

            @pl.when(is_last)
            def _fin_4k():
                o_vmem_ref[q_r] = finalize_out(state_vmem_ref[1], state_vmem_ref[2])
                enqueue_kv_write(
                    seq_id, q_r, jnp.bool_(True), kv_lens_ref[seq_id], slot, loc_end - 1
                )
    elif cfg.loop_kind == "swabatch" and one_block_per_seq and dq == 1:
        def _scan_swa_b(s_i, acc):
            mx_re, mn_pg, all_a = acc
            _, act_i, _, _, re_i, npg_i, _ = seq_info(s_i)
            return (
                jnp.where(act_i, jnp.maximum(mx_re, re_i), mx_re),
                jnp.where(act_i, jnp.minimum(mn_pg, npg_i), mn_pg),
                all_a & act_i,
            )

        if prestarted_bkv_state is not None:
            if len(prestarted_bkv_state) != 3:
                raise ValueError("prestarted_bkv_state must be a 3-tuple for swabatch")
            max_re_b, min_pg_b, all_act_b = prestarted_bkv_state
            bkv_prestarted = not cfg.u4_flat
        else:
            max_re_b, min_pg_b, all_act_b = lax.fori_loop(
                0, n_active, _scan_swa_b, (jnp.int32(0), jnp.int32(max_ring_pages), jnp.bool_(True))
            )
            bkv_prestarted = False

        def _run_swa_fixed_pages(w_pg: int):
            if (
                cfg.u4_flat
                and w_pg in (1, 2, 3, 4)
                and page_size >= 512
                and num_slots >= 4
                and not kv_pack_dense2
                and not kv_pack_bd4
                and probe is None
                and not cfg.fp8_dots
            ):
                n_ring = 4
                n_phys_pages = kv_cache_hbm_ref.shape[0]

                def _start_s(pg_base, sl: int):
                    for p_i in range(w_pg):
                        pg_idx = jnp.clip(pg_base + p_i, 0, num_page_indices - 1)
                        pg = jnp.clip(page_indices_ref[pg_idx], 0, n_phys_pages - 1)
                        pltpu.make_async_copy(
                            kv_cache_hbm_ref.at[pl.ds(pg, 1)],
                            bkv_vmem_ref.at[sl, pl.ds(p_i, 1)],
                            bkv_sems.at[sl],
                        ).start()

                def _wait_s(sl: int):
                    dst = bkv_vmem_ref.at[sl, pl.ds(0, w_pg)]
                    pltpu.make_async_copy(dst, dst, bkv_sems.at[sl]).wait()

                def _run_u4_in():
                    for s_init in range(n_ring):
                        _, _, _, _, _, _, pg_base_i = seq_info(jnp.int32(s_init))
                        _start_s(pg_base_i, s_init)
                    if use_hoist_kv:
                        precompute_hoisted_new_kv()

                    def _one_u4(seq, sl: int):
                        q_r, _, kvl, rf, re, _, _ = seq_info(seq)
                        _wait_s(sl)
                        if not skip_kv_write and probe != "nokvwrite":
                            seq_base = cdiv(cu_kv_lens_ref[seq], page_size)
                            pos = kvl - 1
                            dst_pg = page_indices_ref[jnp.minimum(seq_base + (pos // page_size), num_page_indices - 1)]
                            pltpu.make_async_copy(
                                kv_merged_vmem_ref.at[pl.ds(q_r, 1)],
                                updated_kv_cache_hbm_ref.at[dst_pg, pl.ds(pos % page_size, 1)],
                                cache_write_sem.at[0],
                            ).start()
                            written_cnt_smem_ref[0] = written_cnt_smem_ref[0] + 1
                        if sliding_window is not None and w_pg > 1:
                            _sanitize_page_u32(sl, 0, rf, re)
                        _sanitize_page_u32(
                            sl,
                            w_pg - 1,
                            0 if w_pg > 1 else rf,
                            jnp.clip(re - (w_pg - 1) * page_size, 0, page_size),
                        )
                        o_vmem_ref[q_r] = compute_stream1p(
                            sl, q_vmem_ref[q_r], q_r, rf, re, w_pg, skip_sanitize=True
                        )

                    n_full_pf = (n_active - n_ring) // 4

                    @pl.when(n_full_pf > 0)
                    def _grp_loop():
                        @pl.loop(0, n_full_pf)
                        def _grp(g):
                            base_seq = 4 * g
                            for k in range(4):
                                seq = base_seq + k
                                _one_u4(seq, k)
                                _, _, _, _, _, _, pg_base_f = seq_info(seq + n_ring)
                                _start_s(pg_base_f, k)

                    @pl.loop(n_full_pf * 4, n_active)
                    def _tail_u4(seq):
                        sl = seq % n_ring
                        _one_u4(seq, sl)
                        fut_seq = seq + n_ring

                        @pl.when(fut_seq < n_active)
                        def _pf_t():
                            _, _, _, _, _, _, pg_base_f = seq_info(fut_seq)
                            _start_s(pg_base_f, sl)

                use_fast_u4 = all_act_b & (min_pg_b >= w_pg) & (n_active >= 4)

                @pl.when(use_fast_u4)
                def _fast_u4():
                    _run_u4_in()

                @pl.when(~use_fast_u4)
                def _fb_u4():
                    if not cfg.lazy_slot_zero:
                        @pl.when(min_pg_b < w_pg)
                        def _zero_fb():
                            for s_z in range(num_slots):
                                _zero_pages_u32(s_z, 0, w_pg)
                    num_ahead_fb = num_slots - 1
                    for s_init in range(num_ahead_fb):
                        @pl.when(s_init < n_active)
                        def _(si=s_init):
                            _, _, _, _, _, n_pg, pg_base = seq_info(si)
                            start_fetch_pages(pg_base, n_pg, si)
                    if use_hoist_kv:
                        precompute_hoisted_new_kv()

                    @pl.loop(0, n_active)
                    def _seq_fb(seq):
                        slot = seq % num_slots
                        fut_seq = seq + num_ahead_fb
                        fut_slot = (slot + num_ahead_fb) % num_slots

                        @pl.when(fut_seq < n_active)
                        def _():
                            _, _, _, _, _, n_pg_f, pg_base_f = seq_info(fut_seq)
                            start_fetch_pages(pg_base_f, n_pg_f, fut_slot)

                        q_r, act, kvl, rf, re, n_pg, _ = seq_info(seq)
                        wait_fetch_pages(n_pg, slot)
                        enqueue_kv_write(seq, q_r, act, kvl, slot, re - 1)

                        @pl.when(act)
                        def _():
                            o_vmem_ref[q_r] = compute_stream1p(
                                slot, q_vmem_ref[q_r], q_r, rf, re, w_pg
                            )

                        @pl.when(~act)
                        def _():
                            o_vmem_ref[q_r] = jnp.zeros((q_out_rows, 128), jnp.bfloat16)

                return
            def _run_swa_single():
                if not bkv_prestarted:
                    if probe != "nocompute" and not cfg.lazy_slot_zero:
                        @pl.when(min_pg_b < w_pg)
                        def _zero_active_prefix():
                            for s_z in range(num_slots):
                                _zero_pages_u32(s_z, 0, w_pg)

                num_ahead_b = num_slots - 1
                if not bkv_prestarted:
                    for s_init in range(num_ahead_b):

                        @pl.when(s_init < n_active)
                        def _(si=s_init):
                            _, _, _, _, _, n_pg, pg_base = seq_info(si)
                            start_fetch_pages(pg_base, n_pg, si)
                if use_hoist_kv:
                    precompute_hoisted_new_kv()

                @pl.loop(0, n_active)
                def _seq_step_b(seq):
                    slot = seq % num_slots
                    fut_seq = seq + num_ahead_b
                    fut_slot = (slot + num_ahead_b) % num_slots

                    @pl.when(fut_seq < n_active)
                    def _():
                        _, _, _, _, _, n_pg_f, pg_base_f = seq_info(fut_seq)
                        start_fetch_pages(pg_base_f, n_pg_f, fut_slot)

                    q_r, act, kvl, rf, re, n_pg, _ = seq_info(seq)
                    wait_fetch_pages(n_pg, slot)
                    enqueue_kv_write(seq, q_r, act, kvl, slot, re - 1)

                    @pl.when(act)
                    def _():
                        o_vmem_ref[q_r] = compute_stream1p(
                            slot, q_vmem_ref[q_r], q_r, rf, re, w_pg
                        )

                    @pl.when(~act)
                    def _():
                        o_vmem_ref[q_r] = jnp.zeros((q_out_rows, 128), jnp.bfloat16)

            def _run_swa_grp_vec():
                grp_sz = cfg.swabatch_group_size
                n_grp = (n_active + grp_sz - 1) // grp_sz
                w_tok = w_pg * page_size
                ahead_g = num_slots - 1
                if not bkv_prestarted:
                    for g_init in range(ahead_g):
                        @pl.when(g_init < n_grp)
                        def _(gi=g_init):
                            _start_swabatch_group_dmas(
                                gi,
                                gi,
                                w_pg,
                                group_sz=grp_sz,
                                n_active=n_active,
                                kv_cache_hbm_ref=kv_cache_hbm_ref,
                                bkv_vmem_ref=bkv_vmem_ref,
                                bkv_sems=bkv_sems,
                                kv_lens_ref=kv_lens_ref,
                                page_indices_ref=page_indices_ref,
                                cu_q_lens_ref=cu_q_lens_ref,
                                cu_kv_lens_ref=cu_kv_lens_ref,
                                num_page_indices=num_page_indices,
                                page_size=page_size,
                                sliding_window=sliding_window,
                            )
                precompute_hoisted_new_kv()

                col_sub = lax.broadcasted_iota(jnp.int32, (q_out_rows, w_tok), 1)
                z_out_sub = jnp.zeros((q_out_rows, 128), jnp.bfloat16)
                fast_u32_san = (not interpret_mode) and (phys_page_rows > 128)
                if fast_u32_san:
                    iota_128 = lax.broadcasted_iota(jnp.int32, (128, 128), 0)
                    z_128 = jnp.zeros((128, 128), jnp.uint32)
                    flat_u32 = bkv_vmem_ref.bitcast(jnp.uint32).reshape(
                        num_slots * bkv_slot_pages * phys_page_rows, 128
                    )
                    n_san_subs = phys_page_rows // 128

                    def _san_grp_pg(sl, pg_idx_sl, lo, hi):
                        br = (sl * bkv_slot_pages + pg_idx_sl) * phys_page_rows
                        if kv_pack_dense2:
                            e_base = 2 * iota_128
                            o_base = e_base + 1
                            for si_sub in range(n_san_subs):
                                ro = si_sub * 128
                                tok_off = 2 * ro
                                so = pl.multiple_of(br + ro, 128)
                                blk = flat_u32[pl.ds(so, 128)]
                                lo_loc = lo - tok_off
                                hi_loc = hi - tok_off
                                e_m = jnp.where(
                                    (e_base >= lo_loc) & (e_base < hi_loc),
                                    jnp.uint32(0x0000FFFF),
                                    jnp.uint32(0),
                                )
                                o_m = jnp.where(
                                    (o_base >= lo_loc) & (o_base < hi_loc),
                                    jnp.uint32(0xFFFF0000),
                                    jnp.uint32(0),
                                )
                                flat_u32[pl.ds(so, 128)] = blk & (e_m | o_m)
                            return
                        for si_sub in range(n_san_subs):
                            ro = si_sub * 128
                            so = pl.multiple_of(br + ro, 128)
                            blk = flat_u32[pl.ds(so, 128)]
                            cond = (iota_128 >= (lo - ro)) & (iota_128 < (hi - ro))
                            flat_u32[pl.ds(so, 128)] = jnp.where(cond, blk, z_128)

                    def _zero_grp_pg(sl, pg_idx_sl):
                        br = (sl * bkv_slot_pages + pg_idx_sl) * phys_page_rows
                        for si_sub in range(n_san_subs):
                            flat_u32[pl.ds(pl.multiple_of(br + si_sub * 128, 128), 128)] = z_128
                else:
                    def _san_grp_pg(sl, pg_idx_sl, lo, hi):
                        _sanitize_page_u32(sl, pg_idx_sl, lo, hi)

                    def _zero_grp_pg(sl, pg_idx_sl):
                        _zero_pages_u32(sl, pg_idx_sl, pg_idx_sl + 1)

                def _grp_body(g_i, wr_cnt):
                    sl = g_i % num_slots
                    fut_g = g_i + ahead_g
                    @pl.when(fut_g < n_grp)
                    def _pf_grp():
                        _start_swabatch_group_dmas(
                            fut_g,
                            fut_g % num_slots,
                            w_pg,
                            group_sz=grp_sz,
                            n_active=n_active,
                            kv_cache_hbm_ref=kv_cache_hbm_ref,
                            bkv_vmem_ref=bkv_vmem_ref,
                            bkv_sems=bkv_sems,
                            kv_lens_ref=kv_lens_ref,
                            page_indices_ref=page_indices_ref,
                            cu_q_lens_ref=cu_q_lens_ref,
                            cu_kv_lens_ref=cu_kv_lens_ref,
                            num_page_indices=num_page_indices,
                            page_size=page_size,
                            sliding_window=sliding_window,
                        )

                    dst = bkv_vmem_ref.at[sl, pl.ds(0, grp_sz * w_pg)]
                    pltpu.make_async_copy(dst, dst, bkv_sems.at[sl]).wait()

                    rf_list, re_list, act_list = [], [], []
                    for sub in range(grp_sz):
                        seq = grp_sz * g_i + sub
                        q_r, act_raw, kvl, rf, re_raw, npg_raw, _ = seq_info(seq)
                        act = (seq < n_active) & act_raw
                        re = jnp.where(act, re_raw, jnp.int32(0))
                        npg = jnp.where(act, npg_raw, jnp.int32(0))
                        rf_seq = jnp.where(act, rf, jnp.int32(0))
                        rf_list.append(rf_seq)
                        re_list.append(re)
                        act_list.append(act)
                        pg_base_sl = sub * w_pg
                        if not skip_kv_write and probe != "nokvwrite":
                            if kv_pack_bd4 or kv_pack_dense2:
                                enqueue_kv_write(
                                    seq,
                                    q_r,
                                    act,
                                    kvl,
                                    sl,
                                    re - 1,
                                    slot_pg_off=pg_base_sl,
                                )
                            else:
                                wr_cnt = wr_cnt + jnp.where(act, jnp.int32(1), jnp.int32(0))

                                @pl.when(act)
                                def _(s=seq, qr=q_r, kv=kvl):
                                    sb = cdiv(cu_kv_lens_ref[s], page_size)
                                    pos = kv - 1
                                    dp = page_indices_ref[jnp.minimum(sb + (pos // page_size), num_page_indices - 1)]
                                    pltpu.make_async_copy(
                                        kv_merged_vmem_ref.at[pl.ds(qr, 1)],
                                        updated_kv_cache_hbm_ref.at[dp, pl.ds(pos % page_size, 1)],
                                        cache_write_sem.at[0],
                                    ).start()

                        if w_pg <= 2:
                            for p_i in range(w_pg):
                                _san_grp_pg(
                                    sl,
                                    pg_base_sl + p_i,
                                    jnp.maximum(rf_seq - p_i * page_size, 0),
                                    jnp.clip(re - p_i * page_size, 0, page_size),
                                )
                        else:
                            _san_grp_pg(sl, pg_base_sl, rf_seq, re)
                            last_p = jnp.clip(npg - 1, 0, w_pg - 1)
                            @pl.when(last_p > 0)
                            def _slp(lp=last_p, re_v=re, pbs=pg_base_sl):
                                _san_grp_pg(
                                    sl, pbs + lp, 0, jnp.clip(re_v - lp * page_size, 0, page_size)
                                )
                            for p_i in range(1, w_pg):
                                @pl.when(p_i >= npg)
                                def _zpi(pi_=p_i, pbs=pg_base_sl):
                                    _zero_grp_pg(sl, pbs + pi_)

                    q_g = q_vmem_ref[pl.ds(grp_sz * g_i, grp_sz)]
                    s_new_g = jnp.concatenate(
                        [init_sv_vmem_ref[0, grp_sz * g_i + sub] for sub in range(grp_sz)],
                        axis=0,
                    )
                    acc0_g = jnp.concatenate(
                        [init_sv_vmem_ref[1, grp_sz * g_i + sub] for sub in range(grp_sz)],
                        axis=0,
                    )
                    def _finish_grp_out(pv_g, l_flat, m_flat):
                        if v_scale is not None and v_scale_val != 1.0:
                            pv_g = pv_g * v_scale_val
                        p_new = jnp.exp2((s_new_g - m_flat) * exp2_scale)
                        acc_g = pv_g + p_new * acc0_g
                        lt_g = (
                            _broadcast_minor_128(jnp.sum(l_flat, axis=1, keepdims=True))
                            + p_new
                        )
                        out_2d = (
                            (acc_g / lt_g).astype(jnp.bfloat16)
                            if interpret_mode
                            else (acc_g * pltpu.reciprocal(lt_g, approx=True)).astype(
                                jnp.bfloat16
                            )
                        )
                        for sub in range(grp_sz):
                            o_vmem_ref[grp_sz * g_i + sub] = jnp.where(
                                act_list[sub],
                                out_2d[sub * q_out_rows : (sub + 1) * q_out_rows],
                                z_out_sub,
                            )

                    if kv_pack_bd4:
                        n_u32 = w_tok // 4
                        half_w = w_tok // 2
                        k0_g, k1_g = _load_bkv_bd4_pair_parts(
                            bkv_vmem_ref,
                            sl,
                            0,
                            grp_sz * w_tok,
                            kind="k",
                            page_size=page_size,
                            interpret_mode=interpret_mode,
                        )
                        k0_g = k0_g.reshape(grp_sz, n_u32, 128)
                        k1_g = k1_g.reshape(grp_sz, n_u32, 128)
                        col_sub_e = _bd4_c_even_iota(0, half_w, q_out_rows)
                        col_sub_o = col_sub_e + 1
                        se_list, so_list = [], []
                        for sub in range(grp_sz):
                            k_sub = jnp.concatenate([k0_g[sub], k1_g[sub]], axis=0)
                            q_sub_bd = _to_bd4_q(q_g[sub])
                            se_s, so_s = _bd4_logits_split(q_sub_bd, k_sub, 1, half_w)
                            if sliding_window is not None:
                                se_s = jnp.where(
                                    (col_sub_e >= rf_list[sub]) & (col_sub_e < re_list[sub]),
                                    se_s,
                                    _MASK_LOGIT,
                                )
                                so_s = jnp.where(
                                    (col_sub_o >= rf_list[sub]) & (col_sub_o < re_list[sub]),
                                    so_s,
                                    _MASK_LOGIT,
                                )
                            else:
                                se_s = jnp.where(col_sub_e < re_list[sub], se_s, _MASK_LOGIT)
                                so_s = jnp.where(col_sub_o < re_list[sub], so_s, _MASK_LOGIT)
                            se_list.append(se_s)
                            so_list.append(so_s)
                        se_flat = jnp.concatenate(se_list, axis=0)
                        so_flat = jnp.concatenate(so_list, axis=0)
                        m_flat, _, pe_flat, po_flat, l_flat = _bd4_softmax_reduce(
                            se_flat,
                            so_flat,
                            s_new_g,
                            exp2_scale,
                            half_w,
                            grp_sz * q_out_rows,
                            use_page_reduce=True,
                        )
                        v0_g, v1_g = _load_bkv_bd4_pair_parts(
                            bkv_vmem_ref,
                            sl,
                            0,
                            grp_sz * w_tok,
                            kind="v",
                            page_size=page_size,
                            interpret_mode=interpret_mode,
                        )
                        v0_g = v0_g.reshape(grp_sz, n_u32, 128)
                        v1_g = v1_g.reshape(grp_sz, n_u32, 128)
                        pv_g = jnp.concatenate(
                            [
                                _bd4_pv_reduce(
                                    pe_flat[sub * 8 : (sub + 1) * 8],
                                    po_flat[sub * 8 : (sub + 1) * 8],
                                    jnp.concatenate([v0_g[sub], v1_g[sub]], axis=0),
                                    n_seq=1,
                                    p_bf16=cfg.pv_bf16,
                                )
                                for sub in range(grp_sz)
                            ],
                            axis=0,
                        )
                        _finish_grp_out(pv_g, l_flat, m_flat)
                        return wr_cnt
                    if kv_pack_dense2:
                        half_w = w_tok // 2
                        k_e_g, k_o_g = _load_bkv_fp8_split(
                            bkv_vmem_ref,
                            sl,
                            0,
                            grp_sz * w_tok,
                            kind="k",
                            page_size=page_size,
                            interpret_mode=interpret_mode,
                        )
                        k_e_g = k_e_g.reshape(grp_sz, half_w, 128)
                        k_o_g = k_o_g.reshape(grp_sz, half_w, 128)
                        col_sub_e = 2 * lax.broadcasted_iota(
                            jnp.int32, (q_out_rows, half_w), 1
                        )
                        col_sub_o = col_sub_e + 1
                        se_list, so_list = [], []
                        for sub in range(grp_sz):
                            se_s = jnp.matmul(
                                q_g[sub],
                                k_e_g[sub].T,
                                preferred_element_type=jnp.float32,
                            )
                            so_s = jnp.matmul(
                                q_g[sub],
                                k_o_g[sub].T,
                                preferred_element_type=jnp.float32,
                            )
                            if sliding_window is not None:
                                se_s = jnp.where(
                                    (col_sub_e >= rf_list[sub]) & (col_sub_e < re_list[sub]),
                                    se_s,
                                    _MASK_LOGIT,
                                )
                                so_s = jnp.where(
                                    (col_sub_o >= rf_list[sub]) & (col_sub_o < re_list[sub]),
                                    so_s,
                                    _MASK_LOGIT,
                                )
                            else:
                                se_s = jnp.where(col_sub_e < re_list[sub], se_s, _MASK_LOGIT)
                                so_s = jnp.where(col_sub_o < re_list[sub], so_s, _MASK_LOGIT)
                            se_list.append(se_s)
                            so_list.append(so_s)
                        m_flat, _, pe_flat, po_flat, l_flat = _bd4_softmax_reduce(
                            jnp.concatenate(se_list, axis=0),
                            jnp.concatenate(so_list, axis=0),
                            s_new_g,
                            exp2_scale,
                            half_w,
                            grp_sz * q_out_rows,
                            use_page_reduce=True,
                        )
                        v_e_g, v_o_g = _load_bkv_fp8_split(
                            bkv_vmem_ref,
                            sl,
                            0,
                            grp_sz * w_tok,
                            kind="v",
                            page_size=page_size,
                            interpret_mode=interpret_mode,
                        )
                        v_e_g = v_e_g.reshape(grp_sz, half_w, 128)
                        v_o_g = v_o_g.reshape(grp_sz, half_w, 128)
                        pe_dot = pe_flat.astype(jnp.bfloat16) if cfg.pv_bf16 else pe_flat
                        po_dot = po_flat.astype(jnp.bfloat16) if cfg.pv_bf16 else po_flat
                        pv_g = jnp.concatenate(
                            [
                                jnp.matmul(
                                    pe_dot[sub * q_out_rows : (sub + 1) * q_out_rows],
                                    v_e_g[sub],
                                    preferred_element_type=jnp.float32,
                                )
                                + jnp.matmul(
                                    po_dot[sub * q_out_rows : (sub + 1) * q_out_rows],
                                    v_o_g[sub],
                                    preferred_element_type=jnp.float32,
                                )
                                for sub in range(grp_sz)
                            ],
                            axis=0,
                        )
                        _finish_grp_out(pv_g, l_flat, m_flat)
                        return wr_cnt
                    k_all = _bkv_fp8_chunk(sl, 0, grp_sz * w_tok, kind="k")
                    s_list = []
                    for sub in range(grp_sz):
                        sc_s = jnp.matmul(
                            q_g[sub],
                            k_all[sub * w_tok : (sub + 1) * w_tok].T,
                            preferred_element_type=jnp.float32,
                        )
                        if sliding_window is not None:
                            sc_s = jnp.where(
                                (col_sub >= rf_list[sub]) & (col_sub < re_list[sub]),
                                sc_s,
                                _MASK_LOGIT,
                            )
                        else:
                            sc_s = jnp.where(col_sub < re_list[sub], sc_s, _MASK_LOGIT)
                        s_list.append(sc_s)
                    s_flat = jnp.concatenate(s_list, axis=0)
                    m_flat = jnp.maximum(
                        s_new_g,
                        _broadcast_minor_128(
                            jnp.max(_fold128_op(s_flat, jnp.maximum), axis=1, keepdims=True)
                        ),
                    )
                    if cfg.fold_col:
                        px_flat, l_flat = _exp2_fold_col(s_flat, m_flat, exp2_scale)
                    elif cfg.fold_128:
                        px_flat = jnp.exp2((s_flat - m_flat[:, :1]) * exp2_scale)
                        l_flat = _fold128_op(px_flat, jnp.add)
                    else:
                        m_w = _broadcast_minor_128_any(m_flat, w_tok)
                        px_flat = jnp.exp2((s_flat - m_w) * exp2_scale)
                        l_flat = jnp.sum(
                            px_flat.reshape(grp_sz * q_rows, w_tok // 128, 128), axis=1
                        )
                    px_dot = px_flat.astype(jnp.bfloat16) if cfg.pv_bf16 else px_flat
                    v_all = _bkv_fp8_chunk(sl, 0, grp_sz * w_tok, kind="v")
                    pv_g = jnp.concatenate(
                        [
                            jnp.matmul(
                                px_dot[sub * q_rows : (sub + 1) * q_rows],
                                v_all[sub * w_tok : (sub + 1) * w_tok],
                                preferred_element_type=jnp.float32,
                            )
                            for sub in range(grp_sz)
                        ],
                        axis=0,
                    )
                    _finish_grp_out(pv_g, l_flat, m_flat)
                    return wr_cnt

                wr_res = lax.fori_loop(0, n_grp, _grp_body, jnp.int32(0))
                if not (kv_pack_bd4 or kv_pack_dense2):
                    written_cnt_smem_ref[0] = wr_res

            if can_grp_vec:
                _run_swa_grp_vec()
            else:
                _run_swa_single()

        _dispatch_swabatch_w_pg(
            max_re_b,
            max_ring_pages=max_ring_pages,
            page_size=page_size,
            run_fn=_run_swa_fixed_pages,
        )
    elif one_block_per_seq:
        num_ahead = num_slots - 1
        if prestarted_bkv_state is None:
            for s_init in range(num_ahead):

                @pl.when(s_init < n_active)
                def _(si=s_init):
                    _, _, _, _, _, n_pg, pg_base = seq_info(si)
                    start_fetch_pages(pg_base, n_pg, si)
        if use_hoist_kv:
            precompute_hoisted_new_kv()

        @pl.loop(0, n_active)
        def _seq_step(seq):
            slot = seq % num_slots
            fut_seq = seq + num_ahead
            fut_slot = (slot + num_ahead) % num_slots

            @pl.when(fut_seq < n_active)
            def _():
                _, _, _, _, _, n_pg_f, pg_base_f = seq_info(fut_seq)
                start_fetch_pages(pg_base_f, n_pg_f, fut_slot)

            q_r, act, kvl, rf, re, n_pg, _ = seq_info(seq)
            wait_fetch_pages(n_pg, slot)
            enqueue_kv_write(seq, q_r, act, kvl, slot, re - 1)
            run_single_seq_compute(slot, q_r, act, rf, re, n_pg)
    else:
        # Multi-block Full Attention (>16K pool capacity) via state_vmem_ref
        if use_hoist_kv:
            precompute_hoisted_new_kv()

        @pl.loop(0, n_active)
        def _multi_block_seq(seq):
            q_r, act, kvl, _, rel_end, total_pgs, seq_pg_base = seq_info(seq)

            @pl.when(act)
            def _():
                q_bf16 = q_vmem_ref[q_r]
                m0, l0, a0 = init_new_token_state(q_bf16, q_r)
                state_vmem_ref[0] = m0
                state_vmem_ref[1] = l0
                state_vmem_ref[2] = a0
                n_blks = cdiv(total_pgs, max_ring_pages)

                @pl.when(n_blks > 0)
                def _prefetch_blk0():
                    pg0 = jnp.minimum(total_pgs, max_ring_pages)
                    start_fetch_pages(seq_pg_base, pg0, 0)

                @pl.loop(0, n_blks)
                def _blk_body(b_i):
                    slot = b_i % 2
                    nxt_b = b_i + 1
                    nxt_slot = (b_i + 1) % 2

                    @pl.when(nxt_b < n_blks)
                    def _pf_next():
                        rem_pg = jnp.minimum(
                            total_pgs - nxt_b * max_ring_pages, max_ring_pages
                        )
                        start_fetch_pages(
                            seq_pg_base + nxt_b * max_ring_pages, rem_pg, nxt_slot
                        )

                    cur_pg = jnp.minimum(
                        total_pgs - b_i * max_ring_pages, max_ring_pages
                    )
                    wait_fetch_pages(cur_pg, slot)
                    local_end = jnp.clip(
                        rel_end - b_i * max_ring_rows, 0, max_ring_rows
                    )
                    sanitize_slot_boundaries(slot, 0, local_end, max_ring_pages)
                    compute_subblock2k_loop(slot, q_bf16, 0, local_end)

                last_slot = jnp.maximum(n_blks - 1, 0) % 2
                last_loc = jnp.maximum(rel_end - 1, 0) % max_ring_rows
                enqueue_kv_write(seq, q_r, act, kvl, last_slot, last_loc)
                o_vmem_ref[q_r] = finalize_out(state_vmem_ref[1], state_vmem_ref[2])

            @pl.when(~act)
            def _():
                o_vmem_ref[q_r] = jnp.zeros((q_out_rows, 128), jnp.bfloat16)

    if in_kernel_pack2:
        tile_r = max_num_tokens
        for cand_r in (64, 32, 16, 8):
            if max_num_tokens >= cand_r and max_num_tokens % cand_r == 0:
                tile_r = cand_r
                break
        lane_lt_64 = lax.broadcasted_iota(jnp.int32, (tile_r, 128), 1) < 64

        def _finish_pair_tile(pair_bf16, row_sl, col_off: int):
            out_p = pair_bf16
            if out_norm_eps is not None:
                pf = out_p.astype(jnp.float32)
                sq = pf * pf
                sq_lo = jnp.sum(jnp.where(lane_lt_64, sq, 0.0), axis=1, keepdims=True)
                sq_hi = jnp.sum(jnp.where(lane_lt_64, 0.0, sq), axis=1, keepdims=True)
                inv_lo = _broadcast_minor_128(
                    lax.rsqrt(sq_lo * (1.0 / 64.0) + float(out_norm_eps))
                )
                inv_hi = _broadcast_minor_128(
                    lax.rsqrt(sq_hi * (1.0 / 64.0) + float(out_norm_eps))
                )
                out_p = (pf * jnp.where(lane_lt_64, inv_lo, inv_hi)).astype(
                    jnp.bfloat16
                )
            if has_gate:
                g_p = (
                    gate_vmem_ref[:, pl.ds(col_off, 128)]
                    if row_sl is None
                    else gate_vmem_ref[row_sl, pl.ds(col_off, 128)]
                )
                sig_g = jax.nn.sigmoid(g_p.astype(jnp.float32)).astype(jnp.bfloat16)
                out_p = sig_g * out_p
            if row_sl is None:
                o_out_vmem_ref[:, pl.ds(col_off, 128)] = out_p
            else:
                o_out_vmem_ref[row_sl, pl.ds(col_off, 128)] = out_p

        use_full_axis = tile_r == max_num_tokens
        pack2_pairs_hi = o_out_vmem_ref.shape[1] > 512
        for r0 in range(0, max_num_tokens, tile_r):
            r_sl = None if use_full_axis else pl.ds(r0, tile_r)
            for p in range(4):
                if r_sl is None:
                    e_lo = o_vmem_ref[:, 2 * p, :]
                    d_lo = o_vmem_ref[:, 2 * p + 1, :]
                else:
                    e_lo = o_vmem_ref[r_sl, 2 * p, :]
                    d_lo = o_vmem_ref[r_sl, 2 * p + 1, :]
                pair_lo = jnp.where(
                    lane_lt_64,
                    e_lo,
                    _roll_lanes_fast(d_lo, 64, interpret_mode=interpret_mode),
                )
                _finish_pair_tile(pair_lo, r_sl, p * 128)
                if pack2_pairs_hi:
                    if r_sl is None:
                        e_hi = o_vmem_ref[:, 8 + 2 * p, :]
                        d_hi = o_vmem_ref[:, 8 + 2 * p + 1, :]
                    else:
                        e_hi = o_vmem_ref[r_sl, 8 + 2 * p, :]
                        d_hi = o_vmem_ref[r_sl, 8 + 2 * p + 1, :]
                    pair_hi = jnp.where(
                        lane_lt_64,
                        _roll_lanes_fast(e_hi, 64, interpret_mode=interpret_mode),
                        d_hi,
                    )
                    _finish_pair_tile(pair_hi, r_sl, 512 + p * 128)

    if not skip_kv_write and probe != "nokvwrite":
        w_cnt = written_cnt_smem_ref[0]

        @pl.when(w_cnt > 0)
        def _wait_all_kv_writes():
            src = (
                kv_write_vmem_ref.reshape(-1, 1, 4, 128).at[pl.ds(0, w_cnt)]
                if (kv_pack_dense2 or kv_pack_bd4)
                else kv_merged_vmem_ref.at[pl.ds(0, w_cnt)]
            )
            pltpu.make_async_copy(src, src, cache_write_sem.at[0]).wait()


def _decode_fast_kernel(
    # Scalar prefetches (5)
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    cu_kv_lens_ref,
    distribution_ref,
    # Inputs + Outputs + Scratches (*refs to support optional gate_vmem_ref & o_16_scratch)
    *body_refs,
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float | None,
    v_scale: float | None,
    cfg: DecodeFastConfig,
    one_block_per_seq: bool,
    decode_query_size: int = 1,
    in_kernel_pack2: bool = False,
    has_gate: bool = False,
    out_norm_eps: float | None = None,
    kv_pack_dense2: bool = False,
    probe: str | None = None,
    interpret_mode: bool = False,
    kv_pack_bd4: bool = False,
):
    idx_r = 0
    q_vmem_ref = body_refs[idx_r]
    kv_merged_vmem_ref = body_refs[idx_r + 1]
    idx_r += 2
    if has_gate:
        gate_vmem_ref = body_refs[idx_r]
        idx_r += 1
    else:
        gate_vmem_ref = None
    kv_cache_hbm_ref = body_refs[idx_r]
    o_out_vmem_ref = body_refs[idx_r + 1]
    updated_kv_cache_hbm_ref = body_refs[idx_r + 2]
    bkv_vmem_ref = body_refs[idx_r + 3]
    state_vmem_ref = body_refs[idx_r + 4]
    idx_r += 5
    if in_kernel_pack2:
        o_vmem_ref = body_refs[idx_r]
        idx_r += 1
    else:
        o_vmem_ref = o_out_vmem_ref
    if kv_pack_dense2 or kv_pack_bd4:
        kv_write_vmem_ref = body_refs[idx_r]
        idx_r += 1
    else:
        kv_write_vmem_ref = None
    use_hoist_kv = (
        cfg.hoist_new_kv
        or _uses_swabatch_group(
            cfg,
            one_block_per_seq=one_block_per_seq,
            decode_query_size=decode_query_size,
            kv_pack_dense2=kv_pack_dense2,
            probe=probe,
            b_tokens=q_vmem_ref.shape[0],
            max_num_seqs=kv_lens_ref.shape[0],
        )
    ) and decode_query_size == 1
    if use_hoist_kv:
        init_sv_vmem_ref = body_refs[idx_r]
        idx_r += 1
    else:
        init_sv_vmem_ref = None
    written_cnt_smem_ref = body_refs[idx_r]
    bkv_sems = body_refs[idx_r + 1]
    cache_write_sem = body_refs[idx_r + 2]

    _run_rpad_vmem_core(
        kv_lens_ref=kv_lens_ref,
        page_indices_ref=page_indices_ref,
        cu_q_lens_ref=cu_q_lens_ref,
        cu_kv_lens_ref=cu_kv_lens_ref,
        distribution_ref=distribution_ref,
        q_vmem_ref=q_vmem_ref,
        kv_merged_vmem_ref=kv_merged_vmem_ref,
        gate_vmem_ref=gate_vmem_ref,
        kv_cache_hbm_ref=kv_cache_hbm_ref,
        o_out_vmem_ref=o_out_vmem_ref,
        updated_kv_cache_hbm_ref=updated_kv_cache_hbm_ref,
        bkv_vmem_ref=bkv_vmem_ref,
        state_vmem_ref=state_vmem_ref,
        o_vmem_ref=o_vmem_ref,
        kv_write_vmem_ref=kv_write_vmem_ref,
        init_sv_vmem_ref=init_sv_vmem_ref,
        written_cnt_smem_ref=written_cnt_smem_ref,
        bkv_sems=bkv_sems,
        cache_write_sem=cache_write_sem,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        k_scale=k_scale,
        v_scale=v_scale,
        cfg=cfg,
        one_block_per_seq=one_block_per_seq,
        decode_query_size=decode_query_size,
        in_kernel_pack2=in_kernel_pack2,
        has_gate=has_gate,
        out_norm_eps=out_norm_eps,
        kv_pack_dense2=kv_pack_dense2,
        probe=probe,
        interpret_mode=interpret_mode,
        prestarted_bkv_state=None,
        kv_pack_bd4=kv_pack_bd4,
    )


def run_decode_fast_rpa(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array | None,
    kv_cache_fused: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    sliding_window: int | None,
    k_scale: float | None,
    v_scale: float | None,
    mode_str: str,
    decode_query_size: int = 1,
    fused_out_pack: int = 1,
    prepared_inputs: bool = False,
    gate: jax.Array | None = None,
    out_norm_eps: float | None = None,
    kv_pack_dense2: bool = False,
    vmem_limit_bytes: int | None = None,
    disable_semaphore_checks: bool = True,
    probe: str | None = None,
    interpret: bool = False,
    shared_pages=None,
    force_native_tp16: bool = False,
    kv_pack_bd4: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Executes the dedicated Ghostfish decode RPA kernel and formats outputs."""
    if kv_pack_bd4:
        kv_pack_dense2 = False
    raw_mode = mode_str.strip().lower() if mode_str else ""
    raw_toks = set(raw_mode.replace(",", "_").replace("-", "_").split("_"))
    has_gate = gate is not None
    native_tp16 = (
        force_native_tp16
        or kv_pack_bd4
        or "tp16" in raw_toks
        or (has_gate and gate.ndim == 2 and gate.shape[-1] == 512)
        or (
            raw_mode == "shared16_host"
            and not prepared_inputs
            and queries.shape[1:] == (8, 64)
        )
    )
    if prepared_inputs:
        q_3d = queries.reshape(queries.shape[1], -1, 128)
        kv_merged = keys
    else:
        q_3d = queries
        b_tok = keys.shape[0]
        kv_pair = jnp.concatenate([keys, values], axis=1)
        kv_merged = jnp.pad(
            kv_pair, ((0, 0), (0, 2), (0, 128 - keys.shape[-1]))
        ).reshape(b_tok, 1, 4, 128)

    b_tokens = q_3d.shape[0]
    b_pad32 = align_to(max(b_tokens, 1), 32)
    if kv_merged.shape[0] < b_pad32:
        kv_merged = jnp.pad(
            kv_merged, ((0, b_pad32 - kv_merged.shape[0]), (0, 0), (0, 0), (0, 0))
        )

    max_num_seqs = kv_lens.shape[0]
    pages_per_seq = page_indices.shape[0] // max_num_seqs
    phys_page_rows = kv_cache_fused.shape[1]
    page_size = (
        phys_page_rows * 4
        if kv_pack_bd4
        else (phys_page_rows * 2 if kv_pack_dense2 else phys_page_rows)
    )

    use_w7 = _is_w7_enabled(mode_str)
    use_adaptive_c1024 = (
        decode_query_size == 1
        and pages_per_seq > (1024 // page_size)
        and (1024 % page_size == 0)
        and probe is None
        and shared_pages is None
        and raw_mode not in ("tp16", "shared16_host")
        and (sliding_window is None or sliding_window >= 1024)
        and ("noadapt" not in raw_toks)
        and ("adapt" in raw_toks or (use_w7 and "noadapt" not in raw_toks))
    )

    shared_grp = (
        (max_num_seqs // shared_pages.shape[0])
        if (shared_pages is not None and shared_pages.shape[0] > 0)
        else 0
    )
    cfg = parse_decode_fast_config(
        mode_str,
        sliding_window=sliding_window,
        pages_per_seq=pages_per_seq,
        page_size=page_size,
        decode_query_size=decode_query_size,
        shared_group=shared_grp,
        kv_pack_dense2=kv_pack_dense2,
        kv_pack_bd4=kv_pack_bd4,
    )
    use_tp16_q8 = native_tp16 and ("noq8" not in raw_toks)
    if kv_pack_bd4:
        q8 = q_3d[:, :8, :64]
        z64 = jnp.zeros((b_tokens, 8, 64), dtype=q8.dtype)
        q_top = jnp.concatenate([q8, z64], axis=2)
        q_bot = jnp.concatenate([z64, q8], axis=2)
        q_3d = jnp.concatenate([q_top, q_bot], axis=1)
    elif prepared_inputs:
        if use_tp16_q8:
            q_3d = q_3d[:, :8, :]
    elif native_tp16:
        # Eight independent heads share one KV head. Only the low 64 lanes
        # carry data in the existing 128-wide cache and MXU inputs.
        q_row_pad = 0 if use_tp16_q8 else 8
        q_3d = jnp.pad(q_3d, ((0, 0), (0, q_row_pad), (0, 64)))
    one_block_per_seq = (
        False
        if cfg.loop_kind == "ring4k"
        else (True if sliding_window is not None else (pages_per_seq <= cfg.max_ring_pages))
    )

    def _run_cfg(run_cfg: DecodeFastConfig, run_one_block: bool) -> tuple[jax.Array, jax.Array]:
        in_kernel_pack2 = (
            fused_out_pack == 2
            and (run_cfg.in_vmem_pack2 or has_gate or out_norm_eps is not None)
        )
        if (has_gate or out_norm_eps is not None) and not in_kernel_pack2:
            raise ValueError(
                "Fused post-RPA norm/gate requires fused_out_pack == 2"
            )
        pack2_cols = 512 if native_tp16 else 1024

        scope_name = (
            f"RPAd_fast-s{run_cfg.num_slots}-c{run_cfg.chunk_sz}-{run_cfg.loop_kind}"
            f"-mp{run_cfg.max_ring_pages}"
        )
        if decode_query_size > 1:
            scope_name += f"-dq{decode_query_size}"
        if run_cfg.coalesce_dma:
            scope_name += "-coal"
        if run_cfg.lazy_slot_zero:
            scope_name += "-noz"
        if run_cfg.pv_bf16:
            scope_name += "-pbf16"
        if run_cfg.page_reduce:
            scope_name += "-pgred"
        if run_cfg.fp8_dots:
            scope_name += "-fp8dp"
        if run_cfg.shared_group_size > 0 and run_cfg.shared_q_tile != 256:
            scope_name += f"-q{run_cfg.shared_q_tile}"
        if run_cfg.shared_split_dense2:
            scope_name += "-shsplit"
        if run_cfg.shared_fp8_qkpv:
            scope_name += "-shfp8"
        elif run_cfg.shared_fp8_pv_only:
            scope_name += "-shfp8pv"
        if run_cfg.batch_full:
            scope_name += "-batchfull"
        use_grp_bkv = _uses_swabatch_group(
            run_cfg,
            one_block_per_seq=run_one_block,
            decode_query_size=decode_query_size,
            kv_pack_dense2=kv_pack_dense2,
            probe=probe,
            b_tokens=b_tokens,
            max_num_seqs=max_num_seqs,
        )
        if use_grp_bkv:
            scope_name += f"-b{run_cfg.swabatch_group_size}"
        if run_cfg.hoist_new_kv and decode_query_size == 1:
            scope_name += "-hkv"
        if run_cfg.fine_gears and decode_query_size == 1:
            scope_name += "-g512"
        if kv_pack_bd4:
            scope_name += "-bd4"
        elif kv_pack_dense2:
            scope_name += "-d2"
        if in_kernel_pack2:
            scope_name += "-pk2"
        if out_norm_eps is not None:
            scope_name += "-onorm"
        if has_gate:
            scope_name += "-gate"
        if sliding_window is not None:
            scope_name += f"-sw_{sliding_window}"
        if native_tp16:
            scope_name += "-tp16"
        if use_tp16_q8:
            scope_name += "-q8"
        if run_cfg.gear_4k:
            scope_name += "-g4k"
        if run_cfg.fold_col:
            scope_name += "-foldcol"
        elif run_cfg.fold_128:
            scope_name += "-fold128"
        if probe:
            scope_name += f"-probe_{probe}"

        dist_arg = distribution
        if run_cfg.shared_group_size > 0 or run_cfg.loop_kind == "shared16":
            grp = run_cfg.shared_group_size or 16
            if max_num_seqs % grp or b_tokens < max_num_seqs:
                raise ValueError(f"shared{grp} requires fully padded {grp}-row query groups")
            if shared_pages is None or shared_pages.shape != (max_num_seqs // grp,):
                raise ValueError(
                    f"shared{grp} requires the current forward's per-group page counts"
                )
            dist_arg = jnp.concatenate((dist_arg, shared_pages))

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            dist_arg,
        )
        in_specs = [
            pl.BlockSpec(q_3d.shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM),
            pl.BlockSpec(kv_merged.shape, lambda *_: (0, 0, 0, 0), memory_space=pltpu.VMEM),
        ]
        if has_gate:
            gate_2d = gate.reshape(b_tokens, pack2_cols)
            in_specs.append(
                pl.BlockSpec(gate_2d.shape, lambda *_: (0, 0), memory_space=pltpu.VMEM)
            )
        else:
            gate_2d = None
        in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))

        o_head_rows = 8 if kv_pack_bd4 else q_3d.shape[1]
        o_vmem_shape = (b_tokens, o_head_rows, 128)
        o_out_shape = (b_tokens, pack2_cols) if in_kernel_pack2 else o_vmem_shape
        out_specs = [
            (
                pl.BlockSpec(o_out_shape, lambda *_: (0, 0), memory_space=pltpu.VMEM)
                if in_kernel_pack2
                else pl.BlockSpec(
                    o_vmem_shape, lambda *_: (0, 0, 0), memory_space=pltpu.VMEM
                )
            ),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]
        scratch_shapes = [
            pltpu.VMEM(
                (
                    run_cfg.num_slots,
                    (run_cfg.swabatch_group_size if use_grp_bkv else 1)
                    * run_cfg.max_ring_pages,
                    phys_page_rows,
                    1,
                    4,
                    128,
                ),
                kv_cache_fused.dtype,
            ),
            pltpu.VMEM(
                (
                    3,
                    (o_head_rows * (run_cfg.shared_group_size or 16))
                    if (run_cfg.shared_group_size > 0 or run_cfg.loop_kind == "shared16")
                    else (o_head_rows * decode_query_size),
                    128,
                ),
                jnp.float32,
            ),
        ]
        if in_kernel_pack2:
            scratch_shapes.append(pltpu.VMEM(o_vmem_shape, q_3d.dtype))
        if kv_pack_dense2 or kv_pack_bd4:
            max_pair_writes = max(
                max_num_seqs * max(1, (decode_query_size + 4) // 2), 8
            )
            scratch_shapes.append(
                pltpu.VMEM((max_pair_writes, 8, 1, 4, 128), kv_cache_fused.dtype)
            )
        if (run_cfg.hoist_new_kv or use_grp_bkv) and decode_query_size == 1:
            scratch_shapes.append(
                pltpu.VMEM((2, b_pad32, o_head_rows, 128), jnp.float32)
            )
        max_blks_4k = (
            cdiv(max(pages_per_seq, 1), run_cfg.max_ring_pages)
            if run_cfg.loop_kind == "ring4k"
            else 0
        )
        sh_grp = run_cfg.shared_group_size or (16 if run_cfg.loop_kind == "shared16" else 0)
        max_blks_sh = (
            cdiv(max(pages_per_seq, 1), run_cfg.max_ring_pages)
            if sh_grp > 0
            else 0
        )
        # Keep total SMEM (page_indices + scratch) <= 960 KiB (< 1 MiB Pallas limit).
        sh_smem_words = 4 * max(sh_grp * max_blks_sh, 0)
        if (max_num_seqs * pages_per_seq + 2 + run_cfg.num_slots + sh_smem_words) * 4 > 960 * 1024:
            sh_smem_words = 0
        smem_words = (
            2
            + run_cfg.num_slots
            + max(4 * max(max_num_seqs * max_blks_4k, 0), sh_smem_words)
        )
        scratch_shapes.extend(
            [
                pltpu.SMEM((smem_words,), jnp.int32),
                pltpu.SemaphoreType.DMA((run_cfg.num_slots,)),
                pltpu.SemaphoreType.DMA((1,)),
            ]
        )

        tpu_version = 0 if interpret else get_tpu_version()
        out_shape = [
            jax.ShapeDtypeStruct(shape=o_out_shape, dtype=q_3d.dtype),
            (
                pltpu.HBM(shape=kv_cache_fused.shape, dtype=kv_cache_fused.dtype)
                if tpu_version >= 7
                else jax.ShapeDtypeStruct(
                    shape=kv_cache_fused.shape, dtype=kv_cache_fused.dtype
                )
            ),
        ]

        sem_kw = {}
        if not interpret:
            try:
                sem_kw = {"disable_semaphore_checks": disable_semaphore_checks}
                pltpu.CompilerParams(**sem_kw)
            except TypeError:
                sem_kw = {}

        kv_cache_input_idx = len(scalar_prefetches) + (3 if has_gate else 2)
        pallas_kwargs = {
            "grid_spec": pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(1,),
                scratch_shapes=scratch_shapes,
            ),
            "out_shape": out_shape,
            "input_output_aliases": {kv_cache_input_idx: 1},
            "name": scope_name,
        }
        if interpret:
            pallas_kwargs["interpret"] = pltpu.InterpretParams(
                detect_races=True, uninitialized_memory="nan"
            )
        else:
            pallas_kwargs["compiler_params"] = pltpu.CompilerParams(
                dimension_semantics=("arbitrary",),
                vmem_limit_bytes=vmem_limit_bytes,
                disable_bounds_checks=True,
                **sem_kw,
            )

        kernel = pl.pallas_call(
            functools.partial(
                _decode_fast_kernel,
                sm_scale=sm_scale,
                sliding_window=sliding_window,
                k_scale=k_scale,
                v_scale=v_scale,
                cfg=run_cfg,
                one_block_per_seq=run_one_block,
                decode_query_size=decode_query_size,
                in_kernel_pack2=in_kernel_pack2,
                has_gate=has_gate,
                out_norm_eps=out_norm_eps,
                kv_pack_dense2=kv_pack_dense2,
                probe=probe,
                interpret_mode=interpret,
                kv_pack_bd4=kv_pack_bd4,
            ),
            **pallas_kwargs,
        )

        extra_in = (gate_2d,) if has_gate else ()
        if tpu_version >= 7:
            o_out, updated_cache = kernel(
                *scalar_prefetches,
                q_3d,
                kv_merged,
                *extra_in,
                pltpu.with_memory_space_constraint(kv_cache_fused, pltpu.HBM),
            )
        else:
            o_out, updated_cache = kernel(
                *scalar_prefetches,
                q_3d,
                kv_merged,
                *extra_in,
                kv_cache_fused,
            )

        if in_kernel_pack2:
            return o_out, updated_cache
        if native_tp16:
            o8 = o_out[:, :8, :64]
            return (
                o8.reshape(b_tokens, 512) if fused_out_pack == 2 else o8
            ), updated_cache
        if fused_out_pack == 1:
            return o_out, updated_cache
        if fused_out_pack == 2:
            g0 = o_out[:, :8, :64].reshape(b_tokens, 512)
            g1 = o_out[:, 8:, 64:].reshape(b_tokens, 512)
            return jnp.concatenate([g0, g1], axis=-1), updated_cache
        raise ValueError(f"Unsupported fused_out_pack={fused_out_pack}")

    # Group-vectorized swabatch already dispatches w_pg dynamically in-kernel;
    # wrapping it in an outer lax.switch adds ~5.5-6.5 us/occ XLA overhead.
    if (
        use_adaptive_c1024
        and not _uses_swabatch_group(
            cfg,
            one_block_per_seq=one_block_per_seq,
            decode_query_size=decode_query_size,
            kv_pack_dense2=kv_pack_dense2,
            probe=probe,
            b_tokens=b_tokens,
            max_num_seqs=max_num_seqs,
        )
        and not (
            cfg.loop_kind in ("stream0b", "swafast2")
            and cfg.max_ring_pages <= (1024 // page_size)
            and one_block_per_seq
        )
    ):
        short_loop = (
            "swabatch"
            if ((sliding_window is not None or cfg.batch_full) and _is_w8_enabled(mode_str))
            else (
                "w7auto"
                if (sliding_window is None and cfg.fine_gears)
                else "stream0b"
            )
        )
        short_cfg = DecodeFastConfig(
            num_slots=4,
            chunk_sz=1024,
            loop_kind=short_loop,
            max_ring_pages=1024 // page_size,
            coalesce_dma=cfg.coalesce_dma if cfg.fine_gears else False,
            in_vmem_pack2=cfg.in_vmem_pack2,
            lazy_slot_zero=cfg.lazy_slot_zero if cfg.fine_gears else False,
            pv_bf16=cfg.pv_bf16,
            page_reduce=cfg.page_reduce,
            fp8_dots=cfg.fp8_dots,
            hoist_new_kv=cfg.hoist_new_kv,
            fine_gears=cfg.fine_gears,
            batch_full=cfg.batch_full,
            u4_flat=cfg.u4_flat,
            gear_4k=cfg.gear_4k,
            fold_128=cfg.fold_128,
            fold_col=cfg.fold_col,
            no_mask0=cfg.no_mask0,
        )
        active_mask = jnp.arange(max_num_seqs) < jnp.clip(
            distribution[0], 0, max_num_seqs
        )
        max_kv = jnp.max(jnp.where(active_mask, kv_lens, 0))
        branch_idx = (
            (max_kv - (1 if decode_query_size == 1 else 0)) > 1024
        ).astype(jnp.int32)
        return lax.switch(
            branch_idx,
            (
                lambda: _run_cfg(short_cfg, True),
                lambda: _run_cfg(cfg, one_block_per_seq),
            ),
        )
    return _run_cfg(cfg, one_block_per_seq)
