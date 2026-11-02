# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SGLang TPU layout solver, endogenous KV cache ceiling, and crossover sweep."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from .chips import TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import submesh_for_group
from .geometry import ModelGeometry
from .hf_loader import Dtypes
from .modules import Parallel, StepShape, supports_fused_rs, uniform_pp_stage_geometries
from .step import StepFloor, spec_cycle_floor, step_floor

DEFAULT_GPU_MEMORY_UTILIZATION: float = 0.90
EXAMPLE_CONCURRENCIES: tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 256)
PREFIX_HIT_SWEEP_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9, 0.95)
# 512 KiB TPU scalar memory budget minus 32 KiB runtime scalar scratch.
TPU_SMEM_PAGE_TABLE_LIMIT_BYTES: int = (512 - 32) * 1024


@dataclass(frozen=True)
class TrafficScenario:
    """Workload request geometry, prefix-sharing parameters, and optional p90 SLO bounds."""

    name: str = "custom"
    isl: int = 4096
    osl: int = 1024
    prefix_hit_rate: float = 0.0
    accept_len: float = 1.0
    think_time_s: float = 0.0
    concurrencies: tuple[int, ...] = EXAMPLE_CONCURRENCIES
    kv_working_set_turns: float = 1.0
    prefill_chunk: int = 2048
    ttft_slo_ms: float | None = None
    tpot_slo_ms: float | None = None
    spec_tokens: int | None = None
    kv_offload_gb: float = 0.0
    shared_prefix_len: int | None = None
    random_prefix_len: int | None = None
    page_size: int | None = None
    max_running_requests: int | None = None

    def __post_init__(self) -> None:
        if self.shared_prefix_len is not None or self.random_prefix_len is not None:
            sp = int(self.shared_prefix_len or 0)
            rp = int(self.random_prefix_len or 0)
            if sp < 0 or rp < 0 or (sp + rp) <= 0:
                raise ValueError(
                    f"shared_prefix_len ({sp}) and random_prefix_len ({rp}) must be >= 0 with positive sum"
                )
            object.__setattr__(self, "isl", sp + rp)
            object.__setattr__(self, "prefix_hit_rate", float(sp) / float(sp + rp))
        if self.isl <= 0 or self.osl <= 0:
            raise ValueError(f"isl and osl must be > 0, got {self.isl}, {self.osl}")
        if not 0.0 <= self.prefix_hit_rate <= 1.0:
            raise ValueError(
                f"prefix_hit_rate must be in [0, 1], got {self.prefix_hit_rate}"
            )
        if self.accept_len <= 0.0:
            raise ValueError(f"accept_len must be > 0, got {self.accept_len}")
        if self.think_time_s < 0.0:
            raise ValueError(f"think_time_s must be >= 0, got {self.think_time_s}")
        if self.kv_offload_gb < 0.0:
            raise ValueError(f"kv_offload_gb must be >= 0, got {self.kv_offload_gb}")
        if self.max_running_requests is not None and self.max_running_requests <= 0:
            raise ValueError(
                f"max_running_requests must be > 0, got {self.max_running_requests}"
            )
        if not self.concurrencies or any(c <= 0 for c in self.concurrencies):
            raise ValueError(
                f"concurrencies must be positive ints, got {self.concurrencies}"
            )


EXAMPLE_SCENARIOS: dict[str, TrafficScenario] = {
    "chat-8k1k": TrafficScenario(
        name="example:chat-8k1k",
        isl=8192,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128, 256),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "reasoning-1k8k": TrafficScenario(
        name="example:reasoning-1k8k",
        isl=1024,
        osl=8192,
        concurrencies=(1, 4, 8, 16, 32, 64, 128, 256),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
    "coding-agent": TrafficScenario(
        name="example:coding-agent",
        isl=16384,
        osl=1024,
        concurrencies=(1, 4, 8, 16, 32, 64, 128),
        prefix_hit_rate=0.90,
        accept_len=3.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
        spec_tokens=3,
    ),
    "custom": TrafficScenario(
        name="example:custom",
        isl=4096,
        osl=1024,
        concurrencies=(1, 4, 16, 64),
        prefix_hit_rate=0.0,
        accept_len=1.0,
        think_time_s=0.0,
        kv_working_set_turns=1.0,
        prefill_chunk=2048,
    ),
}

EXAMPLE_SLO_TIERS: dict[str, tuple[float | None, float | None]] = {
    "none": (None, None),
    "interactive": (5000.0, 33.33),
    "throughput": (5000.0, 66.67),
}


def build_example_scenario(
    preset: str = "chat-8k1k",
    *,
    isl: int | None = None,
    osl: int | None = None,
    concurrencies: tuple[int, ...] | None = None,
    prefix_hit_rate: float | None = None,
    shared_prefix_len: int | None = None,
    random_prefix_len: int | None = None,
    accept_len: float | None = None,
    think_time_s: float | None = None,
    prefill_chunk: int | None = None,
    spec_tokens: int | None = None,
    slo_tier: str = "none",
    ttft_slo_ms: float | None = None,
    tpot_slo_ms: float | None = None,
    kv_working_set_turns: float | None = None,
    kv_offload_gb: float | None = None,
    page_size: int | None = None,
    max_running_requests: int | None = None,
) -> TrafficScenario:
    """Resolve a TrafficScenario from a named preset with explicit CLI overrides."""
    if preset not in EXAMPLE_SCENARIOS:
        raise ValueError(
            f"Unknown example scenario preset {preset!r}; known: {sorted(EXAMPLE_SCENARIOS)}"
        )
    if slo_tier not in EXAMPLE_SLO_TIERS:
        raise ValueError(
            f"Unknown SLO tier {slo_tier!r}; known: {sorted(EXAMPLE_SLO_TIERS)}"
        )
    base = EXAMPLE_SCENARIOS[preset]
    tier_ttft, tier_tpot = EXAMPLE_SLO_TIERS[slo_tier]
    return replace(
        base,
        isl=isl if isl is not None else base.isl,
        osl=osl if osl is not None else base.osl,
        concurrencies=tuple(int(c) for c in concurrencies)
        if concurrencies is not None
        else base.concurrencies,
        prefix_hit_rate=prefix_hit_rate
        if prefix_hit_rate is not None
        else base.prefix_hit_rate,
        shared_prefix_len=shared_prefix_len,
        random_prefix_len=random_prefix_len,
        accept_len=accept_len if accept_len is not None else base.accept_len,
        think_time_s=think_time_s if think_time_s is not None else base.think_time_s,
        prefill_chunk=prefill_chunk
        if prefill_chunk is not None
        else base.prefill_chunk,
        spec_tokens=spec_tokens if spec_tokens is not None else base.spec_tokens,
        ttft_slo_ms=ttft_slo_ms if ttft_slo_ms is not None else tier_ttft,
        tpot_slo_ms=tpot_slo_ms if tpot_slo_ms is not None else tier_tpot,
        kv_working_set_turns=kv_working_set_turns
        if kv_working_set_turns is not None
        else base.kv_working_set_turns,
        kv_offload_gb=kv_offload_gb
        if kv_offload_gb is not None
        else base.kv_offload_gb,
        page_size=page_size if page_size is not None else base.page_size,
        max_running_requests=(
            max_running_requests
            if max_running_requests is not None
            else base.max_running_requests
        ),
    )


@dataclass(frozen=True)
class HbmBreakdown:
    """Per-TensorCore HBM bytes on the bottleneck uniform PP stage."""

    weights: float
    kv: float
    state: float
    total: float
    stage_index: int = 0
    stage_layers: int = 0


@dataclass(frozen=True)
class FloorTimes:
    """Step resource times (ms) behind a point's binding hardware unit."""

    hbm_ms: float
    matmul_ms: float
    ici_ms: float
    chain_ms: float
    hop_latency_ms: float
    hard_floor_ms: float
    binding: str
    serial_floor_ms: float = 0.0


@dataclass(frozen=True)
class SearchStats:
    """Search funnel counts across evaluated (layout, concurrency) tuples."""

    enumerated: int
    hbm_fit: int
    slo_met: int
    frontier: int
    legal_layouts: int = 0


@dataclass(frozen=True)
class PoolMeshPlan:
    """Topology, SGLang parallelism axes, and HBM footprint for one serving pool."""

    role: Literal["mixed", "prefill", "decode"]
    instances: int
    topology: str
    chips_per_instance: int
    tp: int
    dp: int
    ep: int
    pp: int
    pcp: int
    kv_split: int
    mesh_axes: tuple[int, int, int, int]
    hbm_used_gb: float
    hbm_limit_gb: float
    moe_dispatch: Literal["allreduce", "ag_ep", "fused_rs"] = "allreduce"
    moe_dp: int = 1
    page_size: int = 256
    kv_offload_gb: float = 0.0
    max_running_requests: int | None = None
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    hbm_weights_gb: float = 0.0
    hbm_kv_gb: float = 0.0
    hbm_state_gb: float = 0.0
    kv_cache_capacity_tokens: int | None = None
    spec_tokens: int = 0


@dataclass(frozen=True)
class ParetoPoint:
    """Feasible operating point on the p90 interactivity vs chip throughput frontier."""

    mode: Literal["mixed", "disagg"]
    concurrency: int
    total_chips: int
    prefill_pool: PoolMeshPlan | None
    decode_pool: PoolMeshPlan
    p90_ttft_ms: float
    p90_tpot_ms: float
    p90_interactivity_tok_s: float
    output_throughput_per_chip: float
    token_throughput_per_chip: float
    p_mixed: float
    binding: str
    slo_met: bool
    decode_floor: FloorTimes | None = None
    prefill_floor: FloorTimes | None = None
    served_concurrency: int = 0
    kv_cache_capacity_tokens: int | None = None
    working_set_tokens: int = 0
    effective_prefix_hit_rate: float = 0.0
    pcie_hicache_ms: float = 0.0
    dcn_transfer_ms: float = 0.0
    queue_wait_ms: float = 0.0
    frontier_role: str | None = None
    margin_vs_rank1_pct: float = 0.0


@dataclass(frozen=True)
class ServingPlan:
    """Ranked SGLang serving plan with Pareto frontier and prefix-hit crossover bounds."""

    model_layers: int
    chip: str
    scenario: TrafficScenario
    mode: str
    recommended: ParetoPoint | None
    frontier: tuple[ParetoPoint, ...]
    warnings: tuple[str, ...]
    candidates: tuple[ParetoPoint, ...] = ()
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    search: SearchStats | None = None
    prefix_hit_crossover_rate: float | None = None
    winner_prefix_hit_range: tuple[float, float] | None = None


def default_page_size(geom: ModelGeometry, requested: int | None = None) -> int:
    """Resolve SGLang TPU RPA page size (64 for sink models on RPA v3, 256 for batched RPA)."""
    floor = 64 if geom.has_attention_sink else 256
    if requested is not None:
        if requested < floor or requested % 16 != 0:
            raise ValueError(
                f"page_size={requested} must be >= kernel minimum {floor} and divisible by 16"
            )
        return int(requested)
    return floor


def layout_page_size(
    geom: ModelGeometry, par: Parallel, requested: int | None = None
) -> int:
    """Resolve SGLang TPU RPA page size with PCP/DCP batched RPA floor (>= 256)."""
    base = default_page_size(geom, requested)
    if par.pcp > 1 or par.kv_split > 1:
        if requested is not None and requested < 256:
            raise ValueError(
                f"SGLang TPU PCP/DCP forces batched RPA requiring page_size >= 256, but requested {requested}"
            )
        return max(base, 256)
    return base


def is_sglang_layout_legal(
    geom: ModelGeometry,
    dtypes: Dtypes,
    par: Parallel,
    topo: SliceTopology,
    *,
    ctx_full: int,
    b_local: int,
    page_size: int | None = None,
    prefill_chunk: int | None = None,
) -> bool:
    """Validate SGLang TPU startup invariants from sglang_tpu.tpu_platform."""
    if (
        par.tp < 1
        or par.dp < 1
        or par.pp < 1
        or par.pcp < 1
        or par.kv_split < 1
        or par.ep < 1
        or b_local < 1
        or ctx_full < 1
    ):
        return False
    try:
        eff_page = layout_page_size(geom, par, page_size)
    except ValueError:
        return False

    if par.tp * par.dp * par.pp != topo.num_cores:
        return False
    if geom.num_layers < par.pp:
        return False

    # Attention & GDN head divisibility across TP ranks.
    attn_tp = max(1, par.tp // max(1, par.pcp))
    if geom.num_heads > 0 and geom.num_heads % attn_tp != 0:
        return False
    if geom.linear_v_heads > 0 and geom.linear_v_heads % par.tp != 0:
        return False
    if geom.linear_k_heads > 0 and geom.linear_k_heads % par.tp != 0:
        return False

    # SGLang TPU PP x TP and PP x PCP startup rules (_validate_pipeline_parallel_args).
    if par.pp > 1:
        if par.tp == 1:
            if not (par.pcp == 1 and par.dp == 1 and par.ep == 1):
                return False
        else:
            stage_is_tp_pair = (
                par.tp == 2 and par.pcp == 1 and par.dp == 1 and par.ep == 1
            )
            stage_is_pcp_tp = par.pcp == par.tp and par.dp == 1
            if not (stage_is_tp_pair or stage_is_pcp_tp):
                return False
        if par.pcp > 1:
            if topo.num_chips > 4 or dtypes.kv < 2.0:
                return False

    # SGLang TPU Prefill Context Parallelism (_validate_prefill_context_parallel).
    if par.pcp > 1:
        if par.tp % par.pcp != 0 or par.dp != 1:
            return False
        if geom.linear_v_heads != 0 or geom.shared_inter != 0 or topo.num_chips > 8:
            return False
        if not any(
            a in {"Qwen3MoeForCausalLM", "GptOssForCausalLM"}
            for a in geom.architectures
        ):
            return False
        if (
            any("GptOss" in a for a in geom.architectures)
            and dtypes.kv >= 2.0
            and par.pcp > 2
        ):
            return False
        if not (
            par.moe == "tp_slice" and par.ep == 1 and par.moe_dispatch == "allreduce"
        ):
            return False

    # SGLang TPU Decode Context Parallelism (_validate_decode_context_parallel).
    if par.kv_split > 1:
        if par.pp > 1 or par.pcp > 1 or par.dp > 1:
            return False
        if par.kv_split > 2 or dtypes.kv < 2.0:
            return False
        kv_group = (
            max(1, par.tp // geom.num_kv_heads)
            if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
            else par.tp
        )
        if par.tp % par.kv_split != 0 or kv_group % par.kv_split != 0:
            return False
        if ctx_full < eff_page * par.kv_split:
            return False
        eff_chunk = prefill_chunk if prefill_chunk is not None else 512
        if eff_chunk < eff_page * par.kv_split:
            return False

    # Multi-node whole-model DP: reject whole-model dp-size on multi-host (topo.num_chips > 4) without EP
    if par.dp > 1 and par.ep == 1 and topo.num_chips > 4:
        return False

    # 512 KiB TPU SMEM int32 page-table budget.
    pages_per_seq = math.ceil(ctx_full / max(1, eff_page))
    smem_bytes = 4 * (2 * b_local + 5 + b_local * pages_per_seq)
    if smem_bytes > TPU_SMEM_PAGE_TABLE_LIMIT_BYTES:
        return False

    # MoE collective dispatch legality (SGLang omits a2a_push; fused_rs requires <= 256 experts).
    if par.moe_dispatch not in ("allreduce", "ag_ep", "fused_rs"):
        return False
    if par.moe_dispatch == "fused_rs" and not supports_fused_rs(
        geom, par, topo.num_cores
    ):
        return False
    if par.moe == "ep":
        stage_cores = topo.num_cores // par.pp
        if (
            geom.num_experts <= 0
            or stage_cores % par.ep != 0
            or geom.num_experts % par.ep != 0
        ):
            return False
    elif par.ep > 1 or par.moe_dispatch != "allreduce":
        return False

    return True


def _single_stage_hbm_breakdown(
    st_geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seqs_per_dp: int,
    context: int,
    *,
    physical_kv_layout: bool,
    head_projections: int,
    stage_index: int,
) -> HbmBreakdown:
    tp = max(1, par.tp)
    attn_tp = max(1, tp // max(1, par.pcp))
    ep = par.ep if par.moe == "ep" and par.ep > 1 else (tp if par.moe == "ep" else 1)

    n_gqa = 0
    n_swa = 0
    n_mla = 0
    n_dsa = 0
    n_gdn = 0
    n_kda = 0
    for tok in st_geom.layer_types:
        if tok in ("dsa", "dsa_attention", "sparse_mla"):
            n_dsa += 1
        elif tok in ("mla", "mla_attention"):
            n_mla += 1
        elif tok in ("sliding_attention", "swa", "sliding_window_attention"):
            n_swa += 1
        elif tok in ("kda", "kimi_linear", "kda_attention"):
            n_kda += 1
        elif tok in ("linear_attention", "gdn", "gated_deltanet", "mamba"):
            n_gdn += 1
        elif st_geom.kv_lora_rank > 0 and st_geom.index_topk > 0:
            n_dsa += 1
        elif st_geom.kv_lora_rank > 0:
            n_mla += 1
        else:
            n_gqa += 1

    q_mult = 2 if st_geom.attn_output_gate else 1
    w_attn = 0.0
    if n_gqa + n_swa > 0:
        w_q_o = (
            float(n_gqa + n_swa)
            * float(
                st_geom.hidden * st_geom.num_heads * st_geom.head_dim * q_mult
                + st_geom.num_heads * st_geom.head_dim * st_geom.hidden
            )
            * dt.w_dense
            / attn_tp
        )
        kv_div = min(attn_tp, max(1, st_geom.num_kv_heads))
        w_kv = (
            float(n_gqa + n_swa)
            * float(2 * st_geom.hidden * st_geom.num_kv_heads * st_geom.head_dim)
            * dt.w_dense
            / kv_div
        )
        w_attn += w_q_o + w_kv

    n_latent = n_mla + n_dsa
    if n_latent > 0:
        if st_geom.kv_lora_rank <= 0:
            raise ValueError("MLA/DSA layer requires config field 'kv_lora_rank'")
        if st_geom.qk_nope_head_dim <= 0:
            raise ValueError("MLA/DSA layer requires config field 'qk_nope_head_dim'")
        if st_geom.qk_rope_head_dim < 0:
            raise ValueError("MLA/DSA layer requires non-negative 'qk_rope_head_dim'")
        kv_lora = st_geom.kv_lora_rank
        nope_d = st_geom.qk_nope_head_dim
        rope_d = st_geom.qk_rope_head_dim
        v_d = st_geom.v_head_dim or nope_d
        if st_geom.q_lora_rank > 0:
            q_term_hidden = float(st_geom.hidden * st_geom.q_lora_rank)
            q_term_proj = (
                float(st_geom.q_lora_rank * st_geom.num_heads * (nope_d + rope_d))
                / attn_tp
            )
        else:
            q_term_hidden = 0.0
            q_term_proj = (
                float(st_geom.hidden * st_geom.num_heads * (nope_d + rope_d)) / attn_tp
            )
        mla_unit = (
            q_term_hidden
            + float(st_geom.hidden * (kv_lora + rope_d))
            + q_term_proj
            + float(
                kv_lora * st_geom.num_heads * (nope_d + v_d)
                + st_geom.num_heads * v_d * st_geom.hidden
            )
            / attn_tp
        )
        w_attn += float(n_latent) * mla_unit * dt.w_dense
        if n_dsa > 0 and st_geom.index_n_heads > 0 and st_geom.index_head_dim > 0:
            idx_heads_tp = max(1, st_geom.index_n_heads // attn_tp)
            w_attn += (
                float(n_dsa * st_geom.hidden * idx_heads_tp * st_geom.index_head_dim)
                * dt.w_dense
            )

    w_rec = 0.0
    if n_gdn > 0:
        kh = max(1, st_geom.linear_k_heads // tp)
        vh = max(1, st_geom.linear_v_heads // tp)
        qkv_r = 2 * kh * st_geom.linear_k_dim + vh * st_geom.linear_v_dim
        z_r = vh * st_geom.linear_v_dim
        w_rec += float(n_gdn) * (
            float(st_geom.hidden * (qkv_r + z_r) + z_r * st_geom.hidden) * dt.w_dense
            + float(st_geom.hidden * 2 * vh + qkv_r * st_geom.conv_kernel) * dt.w_aux
        )
    if n_kda > 0:
        kh = max(1, (st_geom.linear_k_heads or st_geom.num_heads) // tp)
        vh = max(1, (st_geom.linear_v_heads or st_geom.num_heads) // tp)
        kd = st_geom.linear_k_dim or st_geom.head_dim
        vd = st_geom.linear_v_dim or st_geom.head_dim
        w_rec += (
            float(n_kda)
            * float(
                st_geom.hidden * (2 * kh * kd + 2 * vh * vd) + vh * vd * st_geom.hidden
            )
            * dt.w_dense
        )

    w_ffn = 0.0
    w_dense_ffn = 0.0
    if st_geom.num_experts > 0:
        n_dense_ffn = min(st_geom.num_layers, st_geom.first_k_dense_replace)
        n_moe_ffn = st_geom.num_layers - n_dense_ffn
        if n_dense_ffn > 0 and st_geom.dense_inter > 0:
            w_dense_ffn = (
                float(n_dense_ffn * 3 * st_geom.hidden * st_geom.dense_inter)
                * dt.w_dense
                / tp
            )
            w_ffn += w_dense_ffn
        routed_div = ep if par.moe == "ep" else max(1, tp // max(1, par.pcp))
        w_ffn += (
            float(
                n_moe_ffn * st_geom.num_experts * 3 * st_geom.hidden * st_geom.moe_inter
            )
            * dt.w_expert
            / routed_div
        )
        if st_geom.shared_inter > 0:
            isr_eff = (
                st_geom.shared_inter
                if par.residual == "sp"
                else max(1, st_geom.shared_inter // tp)
            )
            w_ffn += float(n_moe_ffn * 3 * st_geom.hidden * isr_eff) * dt.w_dense
        w_ffn += float(n_moe_ffn * st_geom.hidden * st_geom.num_experts) * dt.router
    else:
        w_ffn += (
            float(st_geom.num_layers * 3 * st_geom.hidden * st_geom.dense_inter)
            * dt.w_dense
            / tp
        )

    w_head = (
        float(head_projections * st_geom.vocab * st_geom.hidden) * dt.lm_head / tp
        if st_geom.vocab > 0 and head_projections > 0
        else 0.0
    )

    w_mtp = 0.0
    if st_geom.mtp_layers > 0:
        mtp_fc = float(2 * st_geom.hidden * st_geom.hidden) * dt.w_aux / tp
        mtp_att = w_attn / float(max(1, n_gqa + n_swa + n_latent))
        if st_geom.num_experts > 0:
            n_moe_layers = max(
                1,
                st_geom.num_layers
                - min(st_geom.num_layers, st_geom.first_k_dense_replace),
            )
            mtp_ffn = (w_ffn - w_dense_ffn) / float(n_moe_layers)
        else:
            mtp_ffn = w_ffn / float(max(1, st_geom.num_layers))
        w_mtp = float(st_geom.mtp_layers) * (mtp_fc + mtp_att + mtp_ffn)

    kvh = max(1, st_geom.num_kv_heads // attn_tp) if st_geom.num_kv_heads > 0 else 0
    raw_latent_w = st_geom.kv_lora_rank + st_geom.qk_rope_head_dim
    if physical_kv_layout:
        kv_packing = max(1, int(round(4.0 / max(0.5, dt.kv))))
        padded_hd = (
            math.ceil(st_geom.head_dim / 128) * 128 if st_geom.head_dim > 0 else 0
        )
        kx2 = kvh if st_geom.head_dim == 64 else 2 * kvh
        padded_kx2 = math.ceil(kx2 / kv_packing) * kv_packing if kx2 > 0 else 0
        gqa_kv_b_per_tok = float(padded_kx2 * padded_hd) * dt.kv
        mla_latent_b = (
            float(math.ceil(raw_latent_w / 128) * 128) * dt.kv
            if raw_latent_w > 0
            else 0.0
        )
        dsa_idx_b = (
            float(st_geom.index_head_dim) * dt.kv + (4.0 if dt.kv <= 1.0 else 0.0)
            if st_geom.index_topk > 0 and st_geom.index_head_dim > 0
            else 0.0
        )
    else:
        gqa_kv_b_per_tok = float(2 * kvh * st_geom.head_dim) * dt.kv
        mla_latent_b = float(raw_latent_w) * dt.kv
        dsa_idx_b = (
            float(st_geom.index_head_dim) * dt.kv
            if st_geom.index_topk > 0 and st_geom.index_head_dim > 0
            else 0.0
        )

    seq_div = max(1, par.kv_split)
    kv_gqa = float(seqs_per_dp * n_gqa * context) * gqa_kv_b_per_tok / seq_div
    sw_ctx = (
        min(context, st_geom.sliding_window) if st_geom.sliding_window > 0 else context
    )
    kv_swa = float(seqs_per_dp * n_swa * sw_ctx) * gqa_kv_b_per_tok / seq_div
    kv_mla = float(seqs_per_dp * n_mla * context) * mla_latent_b / seq_div
    kv_dsa = float(seqs_per_dp * n_dsa * context) * (mla_latent_b + dsa_idx_b) / seq_div

    st_bytes = 0.0
    if n_gdn > 0:
        kh = max(1, st_geom.linear_k_heads // tp)
        vh = max(1, st_geom.linear_v_heads // tp)
        qkv_r = 2 * kh * st_geom.linear_k_dim + vh * st_geom.linear_v_dim
        st_bytes += float(seqs_per_dp * n_gdn) * (
            float(vh * st_geom.linear_k_dim * st_geom.linear_v_dim) * dt.state
            + float(qkv_r * max(0, st_geom.conv_kernel - 1)) * dt.conv
        )
    if n_kda > 0:
        vh = max(1, (st_geom.linear_v_heads or st_geom.num_heads) // tp)
        kd = st_geom.linear_k_dim or st_geom.head_dim
        vd = st_geom.linear_v_dim or st_geom.head_dim
        st_bytes += float(seqs_per_dp * n_kda * vh * kd * vd) * dt.state

    weights = w_attn + w_rec + w_ffn + w_head + w_mtp
    kv = kv_gqa + kv_swa + kv_mla + kv_dsa
    total = weights + kv + st_bytes
    return HbmBreakdown(
        weights=weights,
        kv=kv,
        state=st_bytes,
        total=total,
        stage_index=stage_index,
        stage_layers=st_geom.num_layers,
    )


def hbm_breakdown_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seqs_per_dp: int,
    context: int,
    *,
    physical_kv_layout: bool = True,
) -> HbmBreakdown:
    """Evaluate per-TensorCore HBM footprint across uniform SGLang PP stages and return the bottleneck stage."""
    if par.pp <= 1:
        return _single_stage_hbm_breakdown(
            geom,
            dt,
            par,
            seqs_per_dp,
            context,
            physical_kv_layout=physical_kv_layout,
            head_projections=2,
            stage_index=0,
        )
    stages = uniform_pp_stage_geometries(geom, par.pp)
    stage_par = replace(par, pp=1)
    best_bd: HbmBreakdown | None = None
    for idx, st_geom in enumerate(stages):
        head_proj = 1 if idx in (0, par.pp - 1) else 0
        bd = _single_stage_hbm_breakdown(
            st_geom,
            dt,
            stage_par,
            seqs_per_dp,
            context,
            physical_kv_layout=physical_kv_layout,
            head_projections=head_proj,
            stage_index=idx,
        )
        if best_bd is None or bd.total > best_bd.total:
            best_bd = bd
    assert best_bd is not None
    return best_bd


def hbm_footprint_per_core_bytes(
    geom: ModelGeometry,
    dt: Dtypes,
    par: Parallel,
    seqs_per_dp: int,
    context: int,
    *,
    physical_kv_layout: bool = True,
) -> float:
    """Total per-TensorCore HBM bytes on the heaviest uniform PP stage."""
    return hbm_breakdown_per_core_bytes(
        geom, dt, par, seqs_per_dp, context, physical_kv_layout=physical_kv_layout
    ).total


def _kv_capacity_and_hit_rate(
    *,
    prefix_hit_rate: float,
    hbm_limit_b: float,
    w_base_b: float,
    st_b: float,
    b_kv_phys_core: float,
    offload_b_per_core: float,
    replicas: int,
    working_set_tokens: int,
) -> tuple[int | None, float, float]:
    """Compute cluster KV token capacity, effective prefix hit rate, and host-offload hit fraction."""
    if b_kv_phys_core <= 0.0:
        return (None, prefix_hit_rate, 0.0)
    hbm_avail_b = max(0.0, hbm_limit_b - w_base_b - st_b)
    hbm_cap_cluster = (hbm_avail_b / b_kv_phys_core) * float(max(1, replicas))
    tot_cap_cluster = (
        (hbm_avail_b + max(0.0, offload_b_per_core)) / b_kv_phys_core
    ) * float(max(1, replicas))
    cap_int = None if math.isinf(tot_cap_cluster) else int(tot_cap_cluster)
    if prefix_hit_rate <= 0.0 or working_set_tokens <= 0:
        return (cap_int, prefix_hit_rate, 0.0)
    denom = float(max(1, working_set_tokens))
    h_hbm = prefix_hit_rate * min(1.0, hbm_cap_cluster / denom)
    h_eff = prefix_hit_rate * min(1.0, tot_cap_cluster / denom)
    return (cap_int, h_eff, max(0.0, h_eff - h_hbm))


def _floor_times(sf: StepFloor, cache: _SolverEvalCache | None = None) -> FloorTimes:
    if cache is not None:
        sid = id(sf)
        hit = cache.ft_map.get(sid)
        if hit is None:
            hit = FloorTimes(
                hbm_ms=sf.hbm_s * 1000.0,
                matmul_ms=sf.mm_s * 1000.0,
                ici_ms=sf.ici_s * 1000.0,
                chain_ms=sf.chain_s * 1000.0,
                hop_latency_ms=sf.lat_s * 1000.0,
                hard_floor_ms=sf.hard_floor_s * 1000.0,
                binding=sf.binding,
                serial_floor_ms=sf.serial_floor_s * 1000.0,
            )
            cache.ft_map[sid] = hit
        return hit
    return FloorTimes(
        hbm_ms=sf.hbm_s * 1000.0,
        matmul_ms=sf.mm_s * 1000.0,
        ici_ms=sf.ici_s * 1000.0,
        chain_ms=sf.chain_s * 1000.0,
        hop_latency_ms=sf.lat_s * 1000.0,
        hard_floor_ms=sf.hard_floor_s * 1000.0,
        binding=sf.binding,
        serial_floor_ms=sf.serial_floor_s * 1000.0,
    )


def _divisors(n: int) -> tuple[int, ...]:
    return tuple(d for d in range(1, n + 1) if n % d == 0)


def _candidate_decode_parallels(
    geom: ModelGeometry,
    dtypes: Dtypes,
    topo: SliceTopology,
    *,
    ctx_full: int,
    page_size: int | None = None,
    prefill_chunk: int | None = None,
) -> list[tuple[Parallel, int]]:
    """Enumerate legal SGLang decode/mixed Parallel tuples along with derived moe_dp."""
    cores = topo.num_cores
    is_gdn = geom.linear_v_heads > 0 or any(
        t in ("linear_attention", "gdn", "gated_deltanet", "mamba")
        for t in geom.layer_types
    )
    out: list[tuple[Parallel, int]] = []
    for pp in _divisors(cores):
        if geom.num_layers < pp:
            continue
        stage_cores = cores // pp
        for tp in _divisors(stage_cores):
            if geom.num_heads > 0 and geom.num_heads % tp != 0:
                continue
            dp = stage_cores // tp
            pcp_choices = _divisors(tp) if dp == 1 and tp >= 2 else (1,)
            kv_group = (
                max(1, tp // geom.num_kv_heads)
                if geom.num_kv_heads > 0 and geom.kv_lora_rank == 0
                else tp
            )
            kv_splits = (
                _divisors(kv_group)
                if (
                    pp == 1
                    and dp == 1
                    and kv_group > 1
                    and dtypes.kv >= 2.0
                    and not is_gdn
                )
                else (1,)
            )
            mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)

            modes: list[
                tuple[
                    Literal["tp_slice", "ep"],
                    int,
                    int,
                    Literal["allreduce", "ag_ep", "fused_rs"],
                ]
            ] = []
            if geom.num_experts > 0 and stage_cores > 1 and pp == 1:
                ep_val = math.gcd(stage_cores, geom.num_experts)
                if ep_val > 1:
                    moe_dp = stage_cores // ep_val
                    if dp == 1:
                        modes.append(("ep", ep_val, moe_dp, "allreduce"))
                        modes.append(("ep", ep_val, moe_dp, "ag_ep"))
                        probe_fused = Parallel(
                            tp=tp,
                            dp=dp,
                            ep=ep_val,
                            pp=pp,
                            moe="ep",
                            moe_dispatch="fused_rs",
                            residual="sp",
                            ranks_per_chip=topo.cores_per_chip,
                        )
                        if supports_fused_rs(geom, probe_fused, cores):
                            modes.append(("ep", ep_val, moe_dp, "fused_rs"))
                    else:
                        modes.append(("ep", ep_val, moe_dp, "ag_ep"))
                if dp == 1 and (geom.moe_inter // tp >= 128 or ep_val <= 1):
                    modes.append(("tp_slice", 1, 1, "allreduce"))
            else:
                if dp == 1 or geom.num_experts == 0:
                    modes.append(("tp_slice", 1, 1, "allreduce"))

            for pcp in pcp_choices:
                for kvs in kv_splits:
                    if pcp > 1 and kvs > 1:
                        continue
                    curr_modes = (
                        [("tp_slice", 1, pcp, "allreduce")] if pcp > 1 else modes
                    )
                    for m_mode, m_ep, m_dp, m_disp in curr_modes:
                        res_mode: Literal["replicated", "sp"] = (
                            "sp" if m_disp == "fused_rs" and tp > 1 else "replicated"
                        )
                        cand = Parallel(
                            tp=tp,
                            kv_split=kvs,
                            moe=m_mode,
                            moe_dispatch=m_disp,
                            residual=res_mode,
                            ranks_per_chip=topo.cores_per_chip,
                            dp=dp,
                            ep=m_ep,
                            pp=pp,
                            pcp=pcp,
                            mesh_axes=mesh_ax,
                        )
                        if is_sglang_layout_legal(
                            geom,
                            dtypes,
                            cand,
                            topo,
                            ctx_full=ctx_full,
                            b_local=1,
                            page_size=page_size,
                            prefill_chunk=prefill_chunk,
                        ):
                            out.append((cand, m_dp))
    return out


def _candidate_prefill_parallels(
    geom: ModelGeometry,
    dtypes: Dtypes,
    topo: SliceTopology,
    *,
    isl: int,
    page_size: int | None = None,
) -> list[tuple[Parallel, int]]:
    """Enumerate legal SGLang prefill Parallel tuples along with derived moe_dp."""
    cores = topo.num_cores
    out: list[tuple[Parallel, int]] = []
    for pp in _divisors(cores):
        if geom.num_layers < pp:
            continue
        stage_cores = cores // pp
        for tp in _divisors(stage_cores):
            dp = stage_cores // tp
            if dp != 1:
                continue
            for pcp in _divisors(tp):
                mesh_ax = submesh_for_group(tp, topo, topo.cores_per_chip)
                modes: list[
                    tuple[
                        Literal["tp_slice", "ep"],
                        int,
                        int,
                        Literal["allreduce", "ag_ep", "fused_rs"],
                    ]
                ] = []
                if pcp > 1:
                    modes = [("tp_slice", 1, pcp, "allreduce")]
                elif geom.num_experts > 0 and stage_cores > 1 and pp == 1:
                    ep_val = math.gcd(stage_cores, geom.num_experts)
                    if ep_val > 1:
                        moe_dp = stage_cores // ep_val
                        modes.append(("ep", ep_val, moe_dp, "allreduce"))
                        modes.append(("ep", ep_val, moe_dp, "ag_ep"))
                        probe_fused = Parallel(
                            tp=tp,
                            dp=1,
                            ep=ep_val,
                            pp=1,
                            pcp=pcp,
                            moe="ep",
                            moe_dispatch="fused_rs",
                            residual="sp",
                            ranks_per_chip=topo.cores_per_chip,
                        )
                        if supports_fused_rs(geom, probe_fused, cores):
                            modes.append(("ep", ep_val, moe_dp, "fused_rs"))
                    else:
                        modes.append(("tp_slice", 1, 1, "allreduce"))
                else:
                    modes.append(("tp_slice", 1, 1, "allreduce"))

                for m_mode, m_ep, m_dp, m_disp in modes:
                    res_mode: Literal["replicated", "sp"] = (
                        "sp" if m_disp == "fused_rs" and tp > 1 else "replicated"
                    )
                    cand = Parallel(
                        tp=tp,
                        kv_split=1,
                        moe=m_mode,
                        moe_dispatch=m_disp,
                        residual=res_mode,
                        ranks_per_chip=topo.cores_per_chip,
                        dp=1,
                        ep=m_ep,
                        pp=pp,
                        pcp=pcp,
                        mesh_axes=mesh_ax,
                    )
                    if is_sglang_layout_legal(
                        geom,
                        dtypes,
                        cand,
                        topo,
                        ctx_full=isl,
                        b_local=1,
                        page_size=page_size,
                    ):
                        out.append((cand, m_dp))
    return out


def _pareto_filter(points: list[ParetoPoint]) -> tuple[ParetoPoint, ...]:
    if not points:
        return ()
    ordered = sorted(
        points,
        key=lambda p: (
            -p.p90_interactivity_tok_s,
            -p.token_throughput_per_chip,
            p.total_chips,
        ),
    )
    kept: list[ParetoPoint] = []
    best_tput = -1.0
    for pt in ordered:
        if pt.token_throughput_per_chip > best_tput * (1.0 + 1e-9):
            kept.append(pt)
            best_tput = pt.token_throughput_per_chip
    return tuple(kept)


def _supports_tpu_spec_decode(
    geom: ModelGeometry, par: Parallel, *, is_prefill: bool = False
) -> bool:
    if geom.mtp_layers <= 0:
        return False
    if par.pcp > 1 or par.kv_split > 1:
        return False
    if par.pp > 1 and not is_prefill:
        return False
    return True


def _structural_family_key(pt: ParetoPoint) -> tuple:
    p = pt.prefill_pool
    d = pt.decode_pool
    p_key = (
        None
        if p is None
        else (
            p.topology,
            p.instances,
            p.tp,
            p.dp,
            p.ep > 1,
            p.pp,
            p.pcp,
            p.kv_split,
            p.spec_tokens,
        )
    )
    d_key = (
        d.topology,
        d.instances,
        d.tp,
        d.dp,
        d.ep > 1,
        d.pp,
        d.pcp,
        d.kv_split,
        d.spec_tokens,
    )
    return (pt.mode, p_key, d_key)


_GLOBAL_SOLVER_CACHES: dict[tuple, _SolverEvalCache] = {}


def _get_solver_cache(
    geom: ModelGeometry, dtypes: Dtypes, chip: ChipSpec
) -> _SolverEvalCache:
    key = (geom, dtypes, chip)
    c = _GLOBAL_SOLVER_CACHES.get(key)
    if c is None:
        c = _SolverEvalCache(geom, dtypes, chip)
        _GLOBAL_SOLVER_CACHES[key] = c
    return c


class _SolverEvalCache:
    """Memoizes hit-rate-independent decode floors and HBM breakdowns across crossover sweeps."""

    def __init__(self, geom: ModelGeometry, dtypes: Dtypes, chip: ChipSpec) -> None:
        self.geom = geom
        self.dtypes = dtypes
        self.chip = chip
        self.step_map: dict[tuple[str, Parallel, StepShape], StepFloor] = {}
        self.spec_map: dict[tuple[str, Parallel, int, int, int], StepFloor] = {}
        self.hbm_map: dict[tuple[Parallel, int, int], HbmBreakdown] = {}
        self.cand_p_map: dict[
            tuple[str, int, int | None], list[tuple[Parallel, int]]
        ] = {}
        self.cand_d_map: dict[
            tuple[str, int, int | None, int | None], list[tuple[Parallel, int]]
        ] = {}
        self.ft_map: dict[int, FloorTimes] = {}
        # Full-model logical KV bytes per token for disagg DCN wire transfer floor.
        par_1 = Parallel(tp=1, pp=1, dp=1, ep=1)
        self.full_model_kv_b_per_tok = max(
            0.0,
            hbm_breakdown_per_core_bytes(
                geom, dtypes, par_1, 1, 1, physical_kv_layout=False
            ).kv
            - hbm_breakdown_per_core_bytes(
                geom, dtypes, par_1, 1, 0, physical_kv_layout=False
            ).kv,
        )

    def step(self, topo: SliceTopology, par: Parallel, shape: StepShape) -> StepFloor:
        key = (topo.name, par, shape)
        hit = self.step_map.get(key)
        if hit is None:
            hit = step_floor(
                self.geom, self.chip, par, shape, self.dtypes, topology=topo
            )
            self.step_map[key] = hit
        return hit

    def spec(
        self, topo: SliceTopology, par: Parallel, batch: int, ctx: int, k: int
    ) -> StepFloor:
        key = (topo.name, par, batch, ctx, k)
        hit = self.spec_map.get(key)
        if hit is None:
            hit = spec_cycle_floor(
                self.geom, self.chip, par, batch, ctx, k, self.dtypes, topology=topo
            )
            self.spec_map[key] = hit
        return hit

    def hbm(self, par: Parallel, seqs: int, ctx: int) -> HbmBreakdown:
        key = (par, seqs, ctx)
        hit = self.hbm_map.get(key)
        if hit is None:
            hit = hbm_breakdown_per_core_bytes(self.geom, self.dtypes, par, seqs, ctx)
            self.hbm_map[key] = hit
        return hit

    def floor_times(self, sf: StepFloor) -> FloorTimes:
        sid = id(sf)
        hit = self.ft_map.get(sid)
        if hit is None:
            hit = FloorTimes(
                hbm_ms=sf.hbm_s * 1000.0,
                matmul_ms=sf.mm_s * 1000.0,
                ici_ms=sf.ici_s * 1000.0,
                chain_ms=sf.chain_s * 1000.0,
                hop_latency_ms=sf.lat_s * 1000.0,
                hard_floor_ms=sf.hard_floor_s * 1000.0,
                binding=sf.binding,
                serial_floor_ms=sf.serial_floor_s * 1000.0,
            )
            self.ft_map[sid] = hit
        return hit

    def get_cand_p(
        self,
        geom: ModelGeometry,
        dtypes: Dtypes,
        topo: SliceTopology,
        isl: int,
        page_size: int | None = None,
    ) -> list[tuple[Parallel, int]]:
        k = (topo.name, isl, page_size)
        hit = self.cand_p_map.get(k)
        if hit is None:
            hit = _candidate_prefill_parallels(
                geom, dtypes, topo, isl=isl, page_size=page_size
            )
            self.cand_p_map[k] = hit
        return hit

    def get_cand_d(
        self,
        geom: ModelGeometry,
        dtypes: Dtypes,
        topo: SliceTopology,
        ctx_full: int,
        page_size: int | None = None,
        prefill_chunk: int | None = None,
    ) -> list[tuple[Parallel, int]]:
        k = (topo.name, ctx_full, page_size, prefill_chunk)
        hit = self.cand_d_map.get(k)
        if hit is None:
            hit = _candidate_decode_parallels(
                geom,
                dtypes,
                topo,
                ctx_full=ctx_full,
                page_size=page_size,
                prefill_chunk=prefill_chunk,
            )
            self.cand_d_map[k] = hit
        return hit


def _solve_core(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    *,
    mode: Literal["auto", "mixed", "disagg"],
    max_chips: int | None,
    max_instances: int | None,
    gpu_memory_utilization: float,
    cache: _SolverEvalCache,
) -> ServingPlan:
    hbm_limit_b = float(gpu_memory_utilization) * float(chip.hbm_bytes)
    hbm_limit_gb = hbm_limit_b / 1_000_000_000.0
    eff_al = max(1.0, scenario.accept_len)
    raw_spec_k = (
        scenario.spec_tokens
        if scenario.spec_tokens is not None
        else (max(geom.mtp_layers, math.ceil(eff_al) - 1) if eff_al > 1.0 else 0)
    )
    warnings: list[str] = []
    if raw_spec_k > 0 and geom.mtp_layers <= 0:
        spec_k = 0
        eff_al = 1.0
        warnings.append(
            "MTP_UNSUPPORTED_MODEL: clamping spec_tokens to 0 because model has mtp_layers == 0"
        )
    else:
        spec_k = max(0, raw_spec_k)
    ctx_full = scenario.isl + scenario.osl
    ctx_mid = scenario.isl + scenario.osl // 2
    eff_isl_nom = max(1, math.ceil((1.0 - scenario.prefix_hit_rate) * scenario.isl))
    n_chunks_nom = max(1, math.ceil(eff_isl_nom / max(1, scenario.prefill_chunk)))
    dec_steps_per_req = max(1.0, scenario.osl / eff_al)

    is_gdn = geom.linear_v_heads > 0 or any(
        t in ("linear_attention", "gdn", "gated_deltanet", "mamba")
        for t in geom.layer_types
    )
    if scenario.kv_offload_gb > 0.0 and is_gdn:
        warnings.append(
            "HICACHE_UNSUPPORTED_GDN: clamping pool kv_offload_gb to 0.0 for TPU GDN prefix caching"
        )
    min_c1_tpot_ms = math.inf
    min_req_ttft_ms = math.inf
    topos = [
        t for t in TOPOLOGIES.values() if max_chips is None or t.num_chips <= max_chips
    ]

    enumerated = 0
    legal_layouts = 0
    hbm_fit = 0

    prefill_options: list[
        tuple[
            SliceTopology, Parallel, int, float, float, float, HbmBreakdown, StepFloor
        ]
    ] = []
    if mode in ("auto", "disagg"):
        for topo in topos:
            best_for_topo: (
                tuple[
                    SliceTopology,
                    Parallel,
                    int,
                    float,
                    float,
                    float,
                    HbmBreakdown,
                    StepFloor,
                ]
                | None
            ) = None
            for par, m_dp in cache.get_cand_p(
                geom, dtypes, topo, scenario.isl, scenario.page_size
            ):
                legal_layouts += 1
                hbm_bd = cache.hbm(par, 1, scenario.isl)
                if hbm_bd.total > hbm_limit_b:
                    continue
                sf_p = cache.step(
                    topo,
                    par,
                    StepShape(
                        batch=0,
                        rows_per_seq=0,
                        context=0,
                        state_writes=0,
                        prefill_tokens=scenario.isl,
                        prefix_hit_rate=scenario.prefix_hit_rate,
                    ),
                )
                ttft_ms = (
                    sf_p.serial_floor_s
                    * (1.0 + float(par.pp - 1) / float(max(1, n_chunks_nom)))
                    * 1000.0
                )
                min_req_ttft_ms = min(min_req_ttft_ms, ttft_ms)
                rate_p = 1.0 / max(1e-9, sf_p.serial_floor_s)
                if best_for_topo is None or rate_p > best_for_topo[4]:
                    best_for_topo = (
                        topo,
                        par,
                        m_dp,
                        ttft_ms,
                        rate_p,
                        hbm_bd.total / 1_000_000_000.0,
                        hbm_bd,
                        sf_p,
                    )
            if best_for_topo is not None:
                prefill_options.append(best_for_topo)

    candidates: list[ParetoPoint] = []
    decode_layouts_by_topo: list[tuple[SliceTopology, Parallel, int]] = []
    for topo in topos:
        for par, m_dp in cache.get_cand_d(
            geom, dtypes, topo, ctx_full, scenario.page_size, scenario.prefill_chunk
        ):
            legal_layouts += 1
            decode_layouts_by_topo.append((topo, par, m_dp))

    for topo, par, m_dp in decode_layouts_by_topo:
        l_spec_k = (
            spec_k if _supports_tpu_spec_decode(geom, par, is_prefill=False) else 0
        )
        l_eff_al = eff_al if l_spec_k > 0 else 1.0
        l_dec_steps = max(1.0, scenario.osl / l_eff_al)

        hbm_c1 = cache.hbm(par, 1, ctx_full).total
        if hbm_c1 <= hbm_limit_b:
            sf_c1 = (
                cache.spec(topo, par, 1, ctx_mid, l_spec_k)
                if l_spec_k > 0
                else cache.step(topo, par, StepShape(batch=1, context=ctx_mid))
            )
            min_c1_tpot_ms = min(
                min_c1_tpot_ms, sf_c1.serial_floor_s * 1000.0 / l_eff_al
            )

        if mode not in ("auto", "mixed"):
            continue

        w_base_b = cache.hbm(par, 0, 0).total
        b_kv_phys_core = max(
            0.0, cache.hbm(par, 1, 1).total - cache.hbm(par, 1, 0).total
        )
        is_gdn = any(
            t in ("linear_attention", "gdn", "gated_deltanet", "mamba")
            for t in geom.layer_types
        )
        eff_offload_gb = (
            0.0 if (is_gdn or par.kv_split > 1) else float(scenario.kv_offload_gb)
        )
        offload_b_per_core = eff_offload_gb * 1e9

        for c in scenario.concurrencies:
            enumerated += 1
            b_req = max(1, math.ceil(c / max(1, par.dp)))
            if scenario.max_running_requests is not None:
                b_cap = max(1, scenario.max_running_requests // max(1, par.dp))
                b_local = min(b_req, b_cap)
            else:
                b_local = b_req

            if not is_sglang_layout_legal(
                geom,
                dtypes,
                par,
                topo,
                ctx_full=ctx_full,
                b_local=b_local,
                page_size=scenario.page_size,
            ):
                continue

            if scenario.think_time_s > 0.0:
                sf_ttft_nom = cache.step(
                    topo,
                    par,
                    StepShape(
                        batch=0,
                        rows_per_seq=0,
                        context=0,
                        state_writes=0,
                        prefill_tokens=scenario.isl,
                        prefix_hit_rate=scenario.prefix_hit_rate,
                    ),
                )
                for _ in range(2):
                    sf_dec_tmp = (
                        cache.spec(topo, par, b_local, ctx_mid, l_spec_k)
                        if l_spec_k > 0
                        else cache.step(
                            topo, par, StepShape(batch=b_local, context=ctx_mid)
                        )
                    )
                    w_srv = (
                        sf_ttft_nom.serial_floor_s
                        + l_dec_steps * sf_dec_tmp.serial_floor_s
                    )
                    c_act = float(c) * w_srv / (w_srv + scenario.think_time_s)
                    b_local = max(1, math.ceil(c_act / max(1, par.dp)))
                    if scenario.max_running_requests is not None:
                        b_local = min(
                            b_local,
                            max(1, scenario.max_running_requests // max(1, par.dp)),
                        )

            c_served = min(c, b_local * par.dp)
            w_tok = int(
                round(max(1.0, scenario.kv_working_set_turns * float(c * ctx_full)))
            )
            st_b = max(0.0, cache.hbm(par, b_local, 0).total - w_base_b)
            cap_tok, h_eff, host_hit_frac = _kv_capacity_and_hit_rate(
                prefix_hit_rate=scenario.prefix_hit_rate,
                hbm_limit_b=hbm_limit_b,
                w_base_b=w_base_b,
                st_b=st_b,
                b_kv_phys_core=b_kv_phys_core,
                offload_b_per_core=offload_b_per_core,
                replicas=par.dp,
                working_set_tokens=w_tok,
            )

            hbm_bd = cache.hbm(par, b_local, ctx_full)
            if hbm_bd.total > hbm_limit_b:
                continue
            hbm_fit += 1

            h_round = round(h_eff, 4)
            sf_ttft = cache.step(
                topo,
                par,
                StepShape(
                    batch=0,
                    rows_per_seq=0,
                    context=0,
                    state_writes=0,
                    prefill_tokens=scenario.isl,
                    prefix_hit_rate=h_round,
                ),
            )
            pcie_s = (
                (host_hit_frac * float(scenario.isl) * b_kv_phys_core)
                / chip.pcie_bw_per_core
                if host_hit_frac > 0.0 and chip.pcie_bw_per_core > 0.0
                else 0.0
            )
            ttft_stage_factor = 1.0 + float(par.pp - 1) / float(
                max(
                    1,
                    math.ceil(
                        max(1.0, (1.0 - h_eff) * scenario.isl)
                        / max(1, scenario.prefill_chunk)
                    ),
                )
            )
            base_ttft_s = sf_ttft.serial_floor_s * ttft_stage_factor + pcie_s
            min_req_ttft_ms = min(min_req_ttft_ms, base_ttft_s * 1000.0)

            eff_isl_c = max(1, math.ceil((1.0 - h_eff) * scenario.isl))
            n_chunks_c = max(1, math.ceil(eff_isl_c / max(1, scenario.prefill_chunk)))
            chunk_tok_c = min(eff_isl_c, scenario.prefill_chunk)

            sf_dec = (
                cache.spec(topo, par, b_local, ctx_mid, l_spec_k)
                if l_spec_k > 0
                else cache.step(topo, par, StepShape(batch=b_local, context=ctx_mid))
            )
            sf_ver_only = (
                cache.step(
                    topo,
                    par,
                    StepShape(
                        batch=b_local,
                        rows_per_seq=l_spec_k + 1,
                        context=ctx_mid,
                        state_writes=1,
                    ),
                )
                if l_spec_k > 0
                else sf_dec
            )
            draft_serial_s = max(
                0.0, sf_dec.serial_floor_s - sf_ver_only.serial_floor_s
            )
            sf_mix = cache.step(
                topo,
                par,
                StepShape(
                    batch=b_local,
                    rows_per_seq=l_spec_k + 1,
                    context=ctx_mid,
                    state_writes=1,
                    prefill_tokens=chunk_tok_c,
                    prefix_hit_rate=0.0,
                ),
            )
            mix_serial_s = (
                sf_mix.serial_floor_s
                + draft_serial_s
                + pcie_s / float(max(1, n_chunks_c))
            )
            total_req_work_s = (
                l_dec_steps * sf_dec.serial_floor_s
                + float(b_local * n_chunks_c) * mix_serial_s
            )
            server_cap_rate = float(c_served) / max(1e-9, total_req_work_s)
            w_srv_final = base_ttft_s + l_dec_steps * sf_dec.serial_floor_s
            rate_req_s = (
                min(
                    server_cap_rate,
                    float(c) / max(1e-9, w_srv_final + scenario.think_time_s),
                )
                if scenario.think_time_s > 0.0
                else server_cap_rate
            )
            queue_wait_ms = (
                max(0.0, float(c - c_served) / float(max(1, c_served)))
                * total_req_work_s
                * 1000.0
                if c > c_served
                else 0.0
            )
            p_mix = min(
                1.0,
                float(b_local * n_chunks_c)
                / float(max(1.0, l_dec_steps + float(b_local * n_chunks_c))),
            )
            p90_step_s = (1.0 - p_mix) * sf_dec.serial_floor_s + p_mix * mix_serial_s
            p90_tpot_ms = p90_step_s * 1000.0 / l_eff_al
            p90_ttft_ms = (
                base_ttft_s * 1000.0
                + (sf_mix.serial_floor_s * 1000.0 if c_served > 1 else 0.0)
                + queue_wait_ms
            )
            interactivity = 1000.0 / max(1e-6, p90_tpot_ms)
            out_tput = rate_req_s * float(scenario.osl) / float(topo.num_chips)
            tot_tput = (
                rate_req_s * float(scenario.isl + scenario.osl) / float(topo.num_chips)
            )
            slo_ok = (
                scenario.ttft_slo_ms is None or p90_ttft_ms <= scenario.ttft_slo_ms
            ) and (scenario.tpot_slo_ms is None or p90_tpot_ms <= scenario.tpot_slo_ms)

            pool = PoolMeshPlan(
                role="mixed",
                instances=1,
                topology=topo.name,
                chips_per_instance=topo.num_chips,
                tp=par.tp,
                dp=par.dp,
                ep=par.ep,
                pp=par.pp,
                pcp=par.pcp,
                kv_split=par.kv_split,
                mesh_axes=(
                    par.mesh_axes
                    if par.mesh_axes is not None
                    else submesh_for_group(par.tp, topo, topo.cores_per_chip)
                ),
                hbm_used_gb=hbm_bd.total / 1_000_000_000.0,
                hbm_limit_gb=hbm_limit_gb,
                moe_dispatch=par.moe_dispatch,
                moe_dp=m_dp,
                page_size=layout_page_size(geom, par, scenario.page_size),
                kv_offload_gb=eff_offload_gb,
                max_running_requests=scenario.max_running_requests,
                gpu_memory_utilization=gpu_memory_utilization,
                hbm_weights_gb=hbm_bd.weights / 1_000_000_000.0,
                hbm_kv_gb=hbm_bd.kv / 1_000_000_000.0,
                hbm_state_gb=hbm_bd.state / 1_000_000_000.0,
                kv_cache_capacity_tokens=cap_tok,
                spec_tokens=l_spec_k,
            )
            candidates.append(
                ParetoPoint(
                    mode="mixed",
                    concurrency=c,
                    served_concurrency=c_served,
                    total_chips=topo.num_chips,
                    prefill_pool=None,
                    decode_pool=pool,
                    p90_ttft_ms=p90_ttft_ms,
                    p90_tpot_ms=p90_tpot_ms,
                    p90_interactivity_tok_s=interactivity,
                    output_throughput_per_chip=out_tput,
                    token_throughput_per_chip=tot_tput,
                    p_mixed=p_mix,
                    binding=f"dec:{sf_dec.binding},mix:{sf_mix.binding}"
                    if p_mix > 0.0
                    else sf_dec.binding,
                    slo_met=slo_ok,
                    decode_floor=_floor_times(sf_dec, cache),
                    prefill_floor=_floor_times(sf_ttft, cache),
                    kv_cache_capacity_tokens=cap_tok,
                    working_set_tokens=w_tok,
                    effective_prefix_hit_rate=h_eff,
                    pcie_hicache_ms=pcie_s * 1000.0,
                    dcn_transfer_ms=0.0,
                    queue_wait_ms=queue_wait_ms,
                )
            )

    if mode in ("auto", "disagg") and prefill_options:
        for topo_d, par_d, m_dp_d in decode_layouts_by_topo:
            if par_d.pp > 1 or par_d.pcp > 1 or par_d.kv_split > 1:
                continue
            w_base_d = cache.hbm(par_d, 0, 0).total
            b_kv_phys_d = max(
                0.0, cache.hbm(par_d, 1, 1).total - cache.hbm(par_d, 1, 0).total
            )
            is_gdn = any(
                t in ("linear_attention", "gdn", "gated_deltanet", "mamba")
                for t in geom.layer_types
            )
            eff_offload_gb_d = (
                0.0 if (is_gdn or par_d.kv_split > 1) else float(scenario.kv_offload_gb)
            )
            offload_b_d = eff_offload_gb_d * 1e9

            for c in scenario.concurrencies:
                b_one = max(1, math.ceil(c / max(1, par_d.dp)))
                sf_d_one_spec = (
                    cache.spec(topo_d, par_d, b_one, ctx_mid, spec_k)
                    if spec_k > 0
                    else cache.step(
                        topo_d, par_d, StepShape(batch=b_one, context=ctx_mid)
                    )
                )
                sf_d_one_nospec = cache.step(
                    topo_d, par_d, StepShape(batch=b_one, context=ctx_mid)
                )

                for (
                    topo_p,
                    par_p,
                    m_dp_p,
                    ttft_nom_ms,
                    rate_p_one,
                    hbm_p_gb,
                    hbm_p_bd,
                    sf_p_nom,
                ) in prefill_options:
                    pair_spec_k = (
                        spec_k
                        if (
                            spec_k > 0
                            and _supports_tpu_spec_decode(geom, par_d, is_prefill=False)
                            and _supports_tpu_spec_decode(geom, par_p, is_prefill=True)
                        )
                        else 0
                    )
                    pair_eff_al = eff_al if pair_spec_k > 0 else 1.0
                    pair_dec_steps = max(1.0, scenario.osl / pair_eff_al)

                    sf_d_one_use = sf_d_one_spec if pair_spec_k > 0 else sf_d_one_nospec
                    rate_d_one_use = float(c) / (
                        pair_dec_steps * max(1e-9, sf_d_one_use.serial_floor_s)
                    )

                    eff_max_inst = max_instances if max_instances is not None else 16
                    if max_chips is None:
                        ratio_xp = math.ceil(rate_d_one_use / max(1e-9, rate_p_one)) + 1
                        ratio_yd = math.ceil(rate_p_one / max(1e-9, rate_d_one_use)) + 1
                        max_xp = min(eff_max_inst, max(1, ratio_xp))
                        max_yd = min(eff_max_inst, max(1, ratio_yd))
                    else:
                        max_xp = min(
                            eff_max_inst, max(1, max_chips // topo_p.num_chips)
                        )
                        max_yd = min(
                            eff_max_inst, max(1, max_chips // topo_d.num_chips)
                        )

                    w_base_p = cache.hbm(par_p, 0, 0).total
                    b_kv_phys_p = max(
                        0.0, cache.hbm(par_p, 1, 1).total - cache.hbm(par_p, 1, 0).total
                    )
                    eff_offload_gb_p = (
                        0.0
                        if (is_gdn or par_p.kv_split > 1)
                        else float(scenario.kv_offload_gb)
                    )
                    offload_b_p = eff_offload_gb_p * 1e9

                    for x_p in range(1, max_xp + 1):
                        for y_d in range(1, max_yd + 1):
                            total_chips = (
                                x_p * topo_p.num_chips + y_d * topo_d.num_chips
                            )
                            if max_chips is not None and total_chips > max_chips:
                                continue
                            enumerated += 1
                            b_req_d = max(1, math.ceil(c / max(1, y_d * par_d.dp)))
                            if scenario.max_running_requests is not None:
                                b_cap_d = max(
                                    1, scenario.max_running_requests // max(1, par_d.dp)
                                )
                                b_local = min(b_req_d, b_cap_d)
                            else:
                                b_local = b_req_d

                            if not is_sglang_layout_legal(
                                geom,
                                dtypes,
                                par_d,
                                topo_d,
                                ctx_full=ctx_full,
                                b_local=b_local,
                                page_size=scenario.page_size,
                            ):
                                continue

                            if scenario.think_time_s > 0.0:
                                for _ in range(2):
                                    sf_d_tmp = (
                                        sf_d_one_use
                                        if b_local == b_one
                                        else (
                                            cache.spec(
                                                topo_d,
                                                par_d,
                                                b_local,
                                                ctx_mid,
                                                pair_spec_k,
                                            )
                                            if pair_spec_k > 0
                                            else cache.step(
                                                topo_d,
                                                par_d,
                                                StepShape(
                                                    batch=b_local, context=ctx_mid
                                                ),
                                            )
                                        )
                                    )
                                    w_srv = (
                                        ttft_nom_ms * 0.001
                                        + pair_dec_steps * sf_d_tmp.serial_floor_s
                                    )
                                    c_act = (
                                        float(c)
                                        * w_srv
                                        / (w_srv + scenario.think_time_s)
                                    )
                                    b_local = max(
                                        1, math.ceil(c_act / max(1, y_d * par_d.dp))
                                    )
                                    if scenario.max_running_requests is not None:
                                        b_local = min(
                                            b_local,
                                            max(
                                                1,
                                                scenario.max_running_requests
                                                // max(1, par_d.dp),
                                            ),
                                        )

                            hbm_bd_d = cache.hbm(par_d, b_local, ctx_full)
                            if hbm_bd_d.total > hbm_limit_b:
                                continue
                            hbm_fit += 1

                            c_served = min(c, b_local * y_d * par_d.dp)
                            w_tok = int(
                                round(
                                    max(
                                        1.0,
                                        scenario.kv_working_set_turns
                                        * float(c * ctx_full),
                                    )
                                )
                            )
                            st_b_d = max(
                                0.0, cache.hbm(par_d, b_local, 0).total - w_base_d
                            )
                            st_b_p = max(0.0, cache.hbm(par_p, 1, 0).total - w_base_p)

                            cap_tok_d, h_eff_d, _ = _kv_capacity_and_hit_rate(
                                prefix_hit_rate=scenario.prefix_hit_rate,
                                hbm_limit_b=hbm_limit_b,
                                w_base_b=w_base_d,
                                st_b=st_b_d,
                                b_kv_phys_core=b_kv_phys_d,
                                offload_b_per_core=offload_b_d,
                                replicas=y_d * par_d.dp,
                                working_set_tokens=w_tok,
                            )
                            cap_tok_p, h_eff_p, host_hit_p = _kv_capacity_and_hit_rate(
                                prefix_hit_rate=scenario.prefix_hit_rate,
                                hbm_limit_b=hbm_limit_b,
                                w_base_b=w_base_p,
                                st_b=st_b_p,
                                b_kv_phys_core=b_kv_phys_p,
                                offload_b_per_core=offload_b_p,
                                replicas=x_p * par_p.dp,
                                working_set_tokens=w_tok,
                            )
                            # Symmetrical clamping across prefill and decode pools.
                            h_eff = min(h_eff_p, h_eff_d)
                            cap_tok = (
                                min(cap_tok_p, cap_tok_d)
                                if cap_tok_p is not None and cap_tok_d is not None
                                else (cap_tok_d if cap_tok_d is not None else cap_tok_p)
                            )

                            if h_eff < scenario.prefix_hit_rate - 1e-6:
                                sf_p_eff = cache.step(
                                    topo_p,
                                    par_p,
                                    StepShape(
                                        batch=0,
                                        rows_per_seq=0,
                                        context=0,
                                        state_writes=0,
                                        prefill_tokens=scenario.isl,
                                        prefix_hit_rate=round(h_eff, 4),
                                    ),
                                )
                            else:
                                sf_p_eff = sf_p_nom

                            pcie_s = (
                                (host_hit_p * float(scenario.isl) * b_kv_phys_p)
                                / chip.pcie_bw_per_core
                                if host_hit_p > 0.0 and chip.pcie_bw_per_core > 0.0
                                else 0.0
                            )
                            n_chunks_p = max(
                                1,
                                math.ceil(
                                    max(1.0, (1.0 - h_eff) * scenario.isl)
                                    / max(1, scenario.prefill_chunk)
                                ),
                            )
                            prefill_s_eff = (
                                sf_p_eff.serial_floor_s
                                * (1.0 + float(par_p.pp - 1) / float(n_chunks_p))
                                + pcie_s
                            )
                            rate_p_eff = 1.0 / max(
                                1e-9, sf_p_eff.serial_floor_s + pcie_s
                            )

                            # Disaggregated xP -> yD DCN KV transfer floor (12.5 GB/s per chip).
                            xfer_tokens = max(
                                1.0, (1.0 - h_eff_d) * float(scenario.isl)
                            )
                            xfer_bytes_req = cache.full_model_kv_b_per_tok * xfer_tokens
                            chips_bottleneck = min(topo_p.num_chips, topo_d.num_chips)
                            dcn_s = (
                                (xfer_bytes_req / float(max(1, chips_bottleneck)))
                                / chip.dcn_bw
                                if xfer_bytes_req > 0.0 and chip.dcn_bw > 0.0
                                else 0.0
                            )
                            dcn_rate_cap = (
                                float(
                                    min(x_p * topo_p.num_chips, y_d * topo_d.num_chips)
                                )
                                * chip.dcn_bw
                                / max(1.0, xfer_bytes_req)
                                if xfer_bytes_req > 0.0 and chip.dcn_bw > 0.0
                                else math.inf
                            )

                            sf_d = (
                                sf_d_one_use
                                if b_local == b_one
                                else (
                                    cache.spec(
                                        topo_d, par_d, b_local, ctx_mid, pair_spec_k
                                    )
                                    if pair_spec_k > 0
                                    else cache.step(
                                        topo_d,
                                        par_d,
                                        StepShape(batch=b_local, context=ctx_mid),
                                    )
                                )
                            )
                            p90_tpot_ms = sf_d.serial_floor_s * 1000.0 / pair_eff_al
                            interactivity = 1000.0 / max(1e-6, p90_tpot_ms)
                            rate_d_cap = float(c_served) / (
                                pair_dec_steps * max(1e-9, sf_d.serial_floor_s)
                            )
                            coupled_rate = min(
                                x_p * rate_p_eff, rate_d_cap, dcn_rate_cap
                            )
                            if scenario.think_time_s > 0.0:
                                client_rate = float(c) / max(
                                    1e-9,
                                    prefill_s_eff
                                    + dcn_s
                                    + pair_dec_steps * sf_d.serial_floor_s
                                    + scenario.think_time_s,
                                )
                                coupled_rate = min(coupled_rate, client_rate)

                            queue_wait_ms = (
                                max(0.0, float(c - c_served))
                                / max(1e-9, coupled_rate)
                                * 1000.0
                                if c > c_served
                                else 0.0
                            )
                            ttft_eff_ms = (
                                prefill_s_eff + dcn_s
                            ) * 1000.0 + queue_wait_ms
                            out_tput = (
                                coupled_rate * float(scenario.osl) / float(total_chips)
                            )
                            tot_tput = (
                                coupled_rate
                                * float(scenario.isl + scenario.osl)
                                / float(total_chips)
                            )
                            slo_ok = (
                                scenario.ttft_slo_ms is None
                                or ttft_eff_ms <= scenario.ttft_slo_ms
                            ) and (
                                scenario.tpot_slo_ms is None
                                or p90_tpot_ms <= scenario.tpot_slo_ms
                            )

                            p_pool = PoolMeshPlan(
                                role="prefill",
                                instances=x_p,
                                topology=topo_p.name,
                                chips_per_instance=topo_p.num_chips,
                                tp=par_p.tp,
                                dp=par_p.dp,
                                ep=par_p.ep,
                                pp=par_p.pp,
                                pcp=par_p.pcp,
                                kv_split=par_p.kv_split,
                                mesh_axes=(
                                    par_p.mesh_axes
                                    if par_p.mesh_axes is not None
                                    else submesh_for_group(
                                        par_p.tp, topo_p, topo_p.cores_per_chip
                                    )
                                ),
                                hbm_used_gb=hbm_p_gb,
                                hbm_limit_gb=hbm_limit_gb,
                                moe_dispatch=par_p.moe_dispatch,
                                moe_dp=m_dp_p,
                                page_size=layout_page_size(
                                    geom, par_p, scenario.page_size
                                ),
                                kv_offload_gb=eff_offload_gb_p,
                                max_running_requests=scenario.max_running_requests,
                                gpu_memory_utilization=gpu_memory_utilization,
                                hbm_weights_gb=hbm_p_bd.weights / 1_000_000_000.0,
                                hbm_kv_gb=hbm_p_bd.kv / 1_000_000_000.0,
                                hbm_state_gb=hbm_p_bd.state / 1_000_000_000.0,
                                kv_cache_capacity_tokens=cap_tok_p,
                                spec_tokens=pair_spec_k,
                            )
                            d_pool = PoolMeshPlan(
                                role="decode",
                                instances=y_d,
                                topology=topo_d.name,
                                chips_per_instance=topo_d.num_chips,
                                tp=par_d.tp,
                                dp=par_d.dp,
                                ep=par_d.ep,
                                pp=par_d.pp,
                                pcp=par_d.pcp,
                                kv_split=par_d.kv_split,
                                mesh_axes=(
                                    par_d.mesh_axes
                                    if par_d.mesh_axes is not None
                                    else submesh_for_group(
                                        par_d.tp, topo_d, topo_d.cores_per_chip
                                    )
                                ),
                                hbm_used_gb=hbm_bd_d.total / 1_000_000_000.0,
                                hbm_limit_gb=hbm_limit_gb,
                                moe_dispatch=par_d.moe_dispatch,
                                moe_dp=m_dp_d,
                                page_size=layout_page_size(
                                    geom, par_d, scenario.page_size
                                ),
                                kv_offload_gb=eff_offload_gb_d,
                                max_running_requests=scenario.max_running_requests,
                                gpu_memory_utilization=gpu_memory_utilization,
                                hbm_weights_gb=hbm_bd_d.weights / 1_000_000_000.0,
                                hbm_kv_gb=hbm_bd_d.kv / 1_000_000_000.0,
                                hbm_state_gb=hbm_bd_d.state / 1_000_000_000.0,
                                kv_cache_capacity_tokens=cap_tok_d,
                                spec_tokens=pair_spec_k,
                            )
                            candidates.append(
                                ParetoPoint(
                                    mode="disagg",
                                    concurrency=c,
                                    served_concurrency=c_served,
                                    total_chips=total_chips,
                                    prefill_pool=p_pool,
                                    decode_pool=d_pool,
                                    p90_ttft_ms=ttft_eff_ms,
                                    p90_tpot_ms=p90_tpot_ms,
                                    p90_interactivity_tok_s=interactivity,
                                    output_throughput_per_chip=out_tput,
                                    token_throughput_per_chip=tot_tput,
                                    p_mixed=0.0,
                                    binding=sf_d.binding,
                                    slo_met=slo_ok,
                                    decode_floor=_floor_times(sf_d, cache),
                                    prefill_floor=_floor_times(sf_p_eff, cache),
                                    kv_cache_capacity_tokens=cap_tok,
                                    working_set_tokens=w_tok,
                                    effective_prefix_hit_rate=h_eff,
                                    pcie_hicache_ms=pcie_s * 1000.0,
                                    dcn_transfer_ms=dcn_s * 1000.0,
                                    queue_wait_ms=queue_wait_ms,
                                )
                            )

    if (
        scenario.tpot_slo_ms is not None
        and math.isfinite(min_c1_tpot_ms)
        and scenario.tpot_slo_ms < min_c1_tpot_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: tpot_slo_ms={scenario.tpot_slo_ms:.2f} < single-request floor {min_c1_tpot_ms:.2f} ms"
        )
    if (
        scenario.ttft_slo_ms is not None
        and math.isfinite(min_req_ttft_ms)
        and scenario.ttft_slo_ms < min_req_ttft_ms
    ):
        warnings.append(
            f"INFEASIBLE_SLO: ttft_slo_ms={scenario.ttft_slo_ms:.2f} < single-request prefill floor {min_req_ttft_ms:.2f} ms"
        )
    slo_candidates = [p for p in candidates if p.slo_met]
    pool_for_frontier = slo_candidates if slo_candidates else candidates
    if candidates and not slo_candidates and not warnings:
        warnings.append("SLO_UNMET: no feasible topology met both TTFT and TPOT bounds")
    frontier = _pareto_filter(pool_for_frontier)
    recommended = (
        max(frontier, key=lambda p: p.token_throughput_per_chip) if frontier else None
    )
    search = SearchStats(
        enumerated=enumerated,
        hbm_fit=hbm_fit,
        slo_met=len(slo_candidates),
        frontier=len(frontier),
        legal_layouts=legal_layouts,
    )
    return ServingPlan(
        model_layers=geom.num_layers,
        chip=chip.name,
        scenario=scenario,
        mode=mode,
        recommended=recommended,
        frontier=frontier,
        warnings=tuple(warnings),
        candidates=tuple(pool_for_frontier),
        gpu_memory_utilization=gpu_memory_utilization,
        search=search,
    )


def sweep_prefix_hit_crossover(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    *,
    mode: Literal["auto", "mixed", "disagg"] = "auto",
    max_chips: int | None = None,
    max_instances: int | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    grid: tuple[float, ...] = PREFIX_HIT_SWEEP_GRID,
    _cache: _SolverEvalCache | None = None,
    _target_family: tuple | None = None,
) -> tuple[float | None, tuple[float, float] | None]:
    """Sweep prefix hit rates on a warm cache (< 50 ms) to find structural Rank-1 crossover and winner range."""
    cache = _cache if _cache is not None else _get_solver_cache(geom, dtypes, chip)
    base_scen = replace(scenario, shared_prefix_len=None, random_prefix_len=None)
    winners: list[tuple[float, tuple]] = []
    for h in grid:
        sp = _solve_core(
            geom,
            dtypes,
            chip,
            replace(base_scen, prefix_hit_rate=float(h)),
            mode=mode,
            max_chips=max_chips,
            max_instances=max_instances,
            gpu_memory_utilization=gpu_memory_utilization,
            cache=cache,
        )
        if sp.recommended is not None:
            winners.append((float(h), _structural_family_key(sp.recommended)))

    if not winners:
        return (None, None)

    def _bisect_transition(h_lo: float, fam_at_lo: tuple, h_hi: float) -> float:
        lo = h_lo
        hi = h_hi
        for _ in range(5):
            mid = 0.5 * (lo + hi)
            sp_mid = _solve_core(
                geom,
                dtypes,
                chip,
                replace(base_scen, prefix_hit_rate=mid),
                mode=mode,
                max_chips=max_chips,
                max_instances=max_instances,
                gpu_memory_utilization=gpu_memory_utilization,
                cache=cache,
            )
            if (
                sp_mid.recommended is not None
                and _structural_family_key(sp_mid.recommended) == fam_at_lo
            ):
                lo = mid
            else:
                hi = mid
        return round(0.5 * (lo + hi), 4)

    runs: list[list[tuple[float, tuple]]] = []
    current_run: list[tuple[float, tuple]] = [winners[0]]
    for item in winners[1:]:
        if item[1] == current_run[-1][1]:
            current_run.append(item)
        else:
            runs.append(current_run)
            current_run = [item]
    runs.append(current_run)

    target_fam = _target_family if _target_family is not None else winners[0][1]
    matching_idxs = [idx for idx, run in enumerate(runs) if run[0][1] == target_fam]
    if not matching_idxs:
        first_cross = (
            _bisect_transition(runs[0][-1][0], runs[0][0][1], runs[1][0][0])
            if len(runs) > 1
            else None
        )
        return (first_cross, None)

    target_rate = float(base_scen.prefix_hit_rate)
    chosen_idx = None
    for idx in matching_idxs:
        r = runs[idx]
        if r[0][0] <= target_rate <= r[-1][0]:
            chosen_idx = idx
            break
    if chosen_idx is None:
        chosen_idx = min(
            matching_idxs,
            key=lambda idx: min(
                abs(target_rate - runs[idx][0][0]),
                abs(target_rate - runs[idx][-1][0]),
            ),
        )

    chosen_run = runs[chosen_idx]
    chosen_fam = chosen_run[0][1]
    win_lo = chosen_run[0][0]
    win_hi = chosen_run[-1][0]
    crossover: float | None = None
    left_edge: float | None = None
    right_edge: float | None = None

    if chosen_idx > 0:
        left_edge = _bisect_transition(
            runs[chosen_idx - 1][-1][0], runs[chosen_idx - 1][0][1], chosen_run[0][0]
        )
        win_lo = left_edge
    if chosen_idx < len(runs) - 1:
        right_edge = _bisect_transition(
            chosen_run[-1][0], chosen_fam, runs[chosen_idx + 1][0][0]
        )
        win_hi = right_edge

    if left_edge is not None and right_edge is not None:
        crossover = (
            left_edge
            if abs(target_rate - left_edge) <= abs(target_rate - right_edge)
            else right_edge
        )
    elif right_edge is not None:
        crossover = right_edge
    elif left_edge is not None:
        crossover = left_edge

    return (crossover, (round(win_lo, 4), round(win_hi, 4)))


def solve_serving_plan(
    geom: ModelGeometry,
    dtypes: Dtypes,
    chip: ChipSpec,
    scenario: TrafficScenario,
    mode: Literal["auto", "mixed", "disagg"] = "auto",
    max_chips: int | None = None,
    *,
    max_instances: int | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> ServingPlan:
    """Search single-pool mixed and xP:yD disaggregated SGLang TPU plans with crossover bounds."""
    if not 0.0 < gpu_memory_utilization <= 1.0:
        raise ValueError(
            f"gpu_memory_utilization must be in (0, 1], got {gpu_memory_utilization}"
        )
    cache = _get_solver_cache(geom, dtypes, chip)
    base_plan = _solve_core(
        geom,
        dtypes,
        chip,
        scenario,
        mode=mode,
        max_chips=max_chips,
        max_instances=max_instances,
        gpu_memory_utilization=gpu_memory_utilization,
        cache=cache,
    )
    target_fam = (
        _structural_family_key(base_plan.recommended)
        if base_plan.recommended is not None
        else None
    )
    crossover, win_range = sweep_prefix_hit_crossover(
        geom,
        dtypes,
        chip,
        scenario,
        mode=mode,
        max_chips=max_chips,
        max_instances=max_instances,
        gpu_memory_utilization=gpu_memory_utilization,
        _cache=cache,
        _target_family=target_fam,
    )
    return replace(
        base_plan,
        prefix_hit_crossover_rate=crossover,
        winner_prefix_hit_range=win_range,
    )
