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

"""Wire bytes, serial hops, and 4-axis torus latency/bandwidth floors for collectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .chips import ChipSpec, SliceTopology

Kind = Literal["ar", "ag", "rs", "push", "a2a", "p2p"]
Algo = Literal["one_shot", "two_shot", "ring"]


@dataclass(frozen=True)
class Collective:
    """One collective operation as observed by a single participating rank."""

    name: str
    group: int
    wire_bytes: float
    hops: int
    offchip: float
    kind: Kind
    payload: float


def offchip_share(group: int, ranks_per_chip: int) -> float:
    """Fraction of peers outside this rank's chip when adjacent ranks co-locate."""
    if group <= 1:
        return 0.0
    return max(0, group - min(group, ranks_per_chip)) / float(group - 1)


def make(
    name: str,
    kind: Kind,
    payload: float,
    group: int,
    algo: Algo,
    ranks_per_chip: int,
) -> Collective:
    """Construct a priced Collective from payload bytes, group size, and algorithm."""
    if group <= 1:
        return Collective(name, group, 0.0, 0, 0.0, kind, payload)
    n = group
    if kind in ("push", "p2p"):
        wire, hops = payload, 1
    elif kind == "ar":
        if algo == "one_shot":
            wire, hops = (n - 1) * payload, 1
        elif algo == "two_shot":
            wire, hops = 2.0 * (n - 1) / n * payload, 2
        else:
            wire, hops = 2.0 * (n - 1) / n * payload, 2 * (n - 1)
    else:
        wire = (n - 1) / n * payload
        hops = n - 1 if algo == "ring" else 1
    return Collective(
        name, n, wire, hops, offchip_share(n, ranks_per_chip), kind, payload
    )


def padded_rows(rows: int, tp: int, residual: str) -> int:
    """Pad token rows to a multiple of tp when sequence-parallel residuals are active."""
    return (
        rows if residual == "replicated" else math.ceil(rows / max(1, tp)) * max(1, tp)
    )


def submesh_for_group(
    group: int,
    topology: SliceTopology | None,
    ranks_per_chip: int = 2,
    mesh_axes: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, int, int]:
    """Pack group ranks inner-to-outer across (d2d, x, y, z) physical axes."""
    if group <= 1:
        return (1, 1, 1, 1)
    if topology is not None and group > topology.num_cores:
        raise ValueError(
            f"Collective group {group} exceeds topology {topology.name} ({topology.num_cores} cores)"
        )
    if mesh_axes is not None:
        cap = mesh_axes[0] * mesh_axes[1] * mesh_axes[2] * mesh_axes[3]
        if group <= cap and cap % group == 0:
            bounds = mesh_axes
        elif topology is not None:
            a, b, c = topology.chip_axes
            bounds = (topology.cores_per_chip, a, b, c)
        else:
            raise ValueError(
                f"Collective group {group} exceeds submesh bounds {mesh_axes}"
            )
    elif topology is not None:
        a, b, c = topology.chip_axes
        bounds = (topology.cores_per_chip, a, b, c)
    else:
        d2d = math.gcd(group, ranks_per_chip)
        return (d2d, group // d2d, 1, 1)
    rem = group
    out: list[int] = []
    for lim in bounds:
        take = math.gcd(rem, max(1, lim))
        out.append(take)
        rem //= take
    if rem > 1:
        raise ValueError(f"Collective group {group} cannot pack within bounds {bounds}")
    return (out[0], out[1], out[2], out[3])


def collective_ici_hops(
    c: Collective,
    topology: SliceTopology | None = None,
    ranks_per_chip: int = 2,
    mesh_axes: tuple[int, int, int, int] | None = None,
) -> int:
    """Count physical ICI hops traversed across one-phase or two-phase collectives."""
    if c.group <= 1 or c.offchip <= 0.0:
        return 0
    if topology is not None or mesh_axes is not None:
        _, ax, ay, az = submesh_for_group(c.group, topology, ranks_per_chip, mesh_axes)
        if ax * ay * az <= 1:
            return 0
        hops_1p = sum(
            ext // 2
            if topology is not None and topology.axis_wrapped(i, ext)
            else ext - 1
            for i, ext in enumerate((ax, ay, az))
            if ext > 1
        )
        phases = 2 if c.kind == "ar" else 1
        return phases * hops_1p
    return c.hops


def torus_collective_s(
    kind: Kind,
    payload_bytes: float,
    submesh_axes: tuple[int, int, int, int],
    chip: ChipSpec,
    topology: SliceTopology | None = None,
) -> float:
    """Compute analytical alpha-beta torus transfer floor across (d2d, x, y, z) axes."""
    d2d, ax, ay, az = submesh_axes
    group = d2d * ax * ay * az
    if group <= 1 or payload_bytes <= 0.0:
        return 0.0
    if ax * ay * az == 1:
        intra_bw = chip.ici_bw * chip.cores_per_chip
        xfer = (
            payload_bytes
            if kind in ("push", "p2p")
            else payload_bytes * (group - 1) / group
        )
        phase_s = xfer / intra_bw
        return 2.0 * phase_s if kind == "ar" else phase_s

    ici_hops = 0
    lanes_sum = 0
    for axis_idx, extent in enumerate((ax, ay, az)):
        if extent <= 1:
            continue
        wrapped = (
            topology.axis_wrapped(axis_idx, extent) if topology is not None else False
        )
        if wrapped:
            ici_hops += extent // 2
            lanes_sum += 2
        else:
            ici_hops += extent - 1
            lanes_sum += 1

    contention = float(max(1, d2d))
    cap_bw = chip.ici_bw if d2d >= 2 else chip.ici_bw * chip.cores_per_chip
    raw_bw = lanes_sum * chip.ici_link_bw / contention
    eff_bw = min(raw_bw, cap_bw) if math.isfinite(cap_bw) and cap_bw > 0.0 else raw_bw
    hop_lat = (
        chip.ici_hop_latency_s
        if chip.ici_bw > 0.0
        and math.isfinite(chip.ici_bw)
        and chip.ici_hop_latency_s > 0.0
        and math.isfinite(chip.ici_hop_latency_s)
        else 0.0
    )
    lat_1p = ici_hops * hop_lat
    if kind == "p2p":
        return max(lat_1p, payload_bytes / eff_bw)
    xfer = payload_bytes if kind == "push" else payload_bytes * (group - 1) / group
    phase_s = max(lat_1p, xfer / eff_bw)
    return 2.0 * phase_s if kind == "ar" else phase_s
