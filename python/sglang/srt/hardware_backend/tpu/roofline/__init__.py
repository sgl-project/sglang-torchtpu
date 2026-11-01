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

"""Analytical TPU serving roofline: hardware specs, step floors, and layout solver."""

from __future__ import annotations

from .chips import CHIPS, TOPOLOGIES, ChipSpec, SliceTopology
from .collectives import (
    Collective,
    collective_ici_hops,
    submesh_for_group,
    torus_collective_s,
)
from .geometry import LAYER_FAMILIES, ModelGeometry, from_hf_config, supports
from .hf_loader import Dtypes, dtypes_from_hf_config, load_hf_config
from .modules import (
    ModuleCost,
    Parallel,
    StepShape,
    ep_busiest_experts,
    experts_touched,
    pass_costs,
    supports_fused_rs,
    uniform_pp_stage_geometries,
)
from .serving import ConcurrencyRow, plan
from .solver import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    EXAMPLE_CONCURRENCIES,
    EXAMPLE_SCENARIOS,
    EXAMPLE_SLO_TIERS,
    PREFIX_HIT_SWEEP_GRID,
    TPU_SMEM_PAGE_TABLE_LIMIT_BYTES,
    FloorTimes,
    HbmBreakdown,
    ParetoPoint,
    PoolMeshPlan,
    SearchStats,
    ServingPlan,
    TrafficScenario,
    build_example_scenario,
    default_page_size,
    hbm_breakdown_per_core_bytes,
    hbm_footprint_per_core_bytes,
    is_sglang_layout_legal,
    layout_page_size,
    solve_serving_plan,
    sweep_prefix_hit_crossover,
)
from .step import StepFloor, spec_cycle_floor, step_floor

__all__ = [
    "CHIPS",
    "DEFAULT_GPU_MEMORY_UTILIZATION",
    "EXAMPLE_CONCURRENCIES",
    "EXAMPLE_SCENARIOS",
    "EXAMPLE_SLO_TIERS",
    "LAYER_FAMILIES",
    "PREFIX_HIT_SWEEP_GRID",
    "TOPOLOGIES",
    "TPU_SMEM_PAGE_TABLE_LIMIT_BYTES",
    "ChipSpec",
    "Collective",
    "ConcurrencyRow",
    "Dtypes",
    "FloorTimes",
    "HbmBreakdown",
    "ModelGeometry",
    "ModuleCost",
    "Parallel",
    "ParetoPoint",
    "PoolMeshPlan",
    "SearchStats",
    "ServingPlan",
    "SliceTopology",
    "StepFloor",
    "StepShape",
    "TrafficScenario",
    "build_example_scenario",
    "collective_ici_hops",
    "default_page_size",
    "dtypes_from_hf_config",
    "ep_busiest_experts",
    "experts_touched",
    "from_hf_config",
    "hbm_breakdown_per_core_bytes",
    "hbm_footprint_per_core_bytes",
    "is_sglang_layout_legal",
    "layout_page_size",
    "load_hf_config",
    "pass_costs",
    "plan",
    "solve_serving_plan",
    "spec_cycle_floor",
    "step_floor",
    "submesh_for_group",
    "supports",
    "supports_fused_rs",
    "sweep_prefix_hit_crossover",
    "torus_collective_s",
    "uniform_pp_stage_geometries",
]
