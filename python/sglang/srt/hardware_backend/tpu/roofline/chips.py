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

"""Per-TensorCore TPU hardware specifications and physical 3D slice topologies."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Public Cloud TPU v7x per-TensorCore datasheet fallbacks when JAX is unavailable.
_TPU7X_FALLBACK_HBM_BYTES: int = 96 * (1024**3)
_TPU7X_FALLBACK_HBM_BW: float = 3_690_987_520_000.0
_TPU7X_FALLBACK_BF16_FLOPS: float = 1_154_872_967_168_000.0
_TPU7X_FALLBACK_FP8_FLOPS: float = 2_307_000_000_000_000.0
_TPU7X_FALLBACK_VMEM_BYTES: int = 64 * (1024**2)


@dataclass(frozen=True)
class ChipSpec:
    """Hardware ceiling parameters for one TensorCore and host/network links."""

    name: str
    hbm_bytes: int
    hbm_bw: float
    bf16_flops: float
    fp8_flops: float
    vmem_bytes: int
    ici_bw: float = math.inf
    cores_per_chip: int = 2
    ici_link_bw: float = 100_000_000_000.0
    ici_hop_latency_s: float = 1e-6
    mxu_m_tile: int = 256
    pcie_bw: float = 64_000_000_000.0
    dcn_bw: float = 12_500_000_000.0

    @property
    def pcie_bw_per_core(self) -> float:
        """Unidirectional host-to-device PCIe Gen5 x16 bandwidth per TensorCore."""
        return self.pcie_bw / float(max(1, self.cores_per_chip))


def _build_tpu7x() -> ChipSpec:
    """Construct TPU v7x per-TensorCore spec from jax tpu_info with datasheet fallback."""
    try:
        from jax.experimental.pallas import tpu as pltpu

        info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
        return ChipSpec(
            name="tpu7x",
            hbm_bytes=int(info.hbm_capacity_bytes),
            hbm_bw=float(info.mem_bw_bytes_per_second),
            bf16_flops=float(info.bf16_ops_per_second),
            fp8_flops=float(info.fp8_ops_per_second),
            vmem_bytes=int(info.vmem_capacity_bytes),
            ici_bw=300_000_000_000.0,
            cores_per_chip=2,
            ici_link_bw=100_000_000_000.0,
            ici_hop_latency_s=1e-6,
            mxu_m_tile=256,
            pcie_bw=64_000_000_000.0,
            dcn_bw=12_500_000_000.0,
        )
    except Exception:
        return ChipSpec(
            name="tpu7x",
            hbm_bytes=_TPU7X_FALLBACK_HBM_BYTES,
            hbm_bw=_TPU7X_FALLBACK_HBM_BW,
            bf16_flops=_TPU7X_FALLBACK_BF16_FLOPS,
            fp8_flops=_TPU7X_FALLBACK_FP8_FLOPS,
            vmem_bytes=_TPU7X_FALLBACK_VMEM_BYTES,
            ici_bw=300_000_000_000.0,
            cores_per_chip=2,
            ici_link_bw=100_000_000_000.0,
            ici_hop_latency_s=1e-6,
            mxu_m_tile=256,
            pcie_bw=64_000_000_000.0,
            dcn_bw=12_500_000_000.0,
        )


@dataclass(frozen=True)
class SliceTopology:
    """Physical 3D chip slice topology and torus wraparound rules."""

    name: str
    chip_axes: tuple[int, int, int]
    cores_per_chip: int = 2

    @property
    def num_chips(self) -> int:
        a, b, c = self.chip_axes
        return a * b * c

    @property
    def num_cores(self) -> int:
        return self.num_chips * self.cores_per_chip

    @property
    def has_torus_wraparound(self) -> bool:
        return all(d >= 4 and d % 4 == 0 for d in self.chip_axes)

    def axis_wrapped(self, axis: int, submesh_extent: int) -> bool:
        return (
            self.has_torus_wraparound
            and submesh_extent == self.chip_axes[axis]
            and submesh_extent >= 4
        )


CHIPS: dict[str, ChipSpec] = {
    "tpu7x": _build_tpu7x(),
}

TOPOLOGIES: dict[str, SliceTopology] = {
    "1x1x1": SliceTopology("1x1x1", (1, 1, 1)),
    "1x2x1": SliceTopology("1x2x1", (1, 2, 1)),
    "2x2x1": SliceTopology("2x2x1", (2, 2, 1)),
    "2x2x2": SliceTopology("2x2x2", (2, 2, 2)),
    "2x2x4": SliceTopology("2x2x4", (2, 2, 4)),
    "2x4x4": SliceTopology("2x4x4", (2, 4, 4)),
    "4x4x4": SliceTopology("4x4x4", (4, 4, 4)),
}
