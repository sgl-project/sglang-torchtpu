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

"""Self-contained offline HTML report with inline SVG Pareto, step-floor, and HBM charts."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

MAX_EMBEDDED_CANDIDATES: int = 256


def _escape(val: Any) -> str:
    return html.escape("" if val is None else str(val), quote=True)


def _render_frontier_svg(
    points: list[dict[str, Any]], frontier: list[dict[str, Any]]
) -> str:
    width, height, pad = 680, 320, 48
    all_pts = points if points else frontier
    if not all_pts:
        return '<svg width="680" height="80" viewBox="0 0 680 80"><text x="20" y="45">No feasible points</text></svg>'

    max_x = max(float(p.get("p90_interactivity_tok_s", 0.0)) for p in all_pts) or 1.0
    max_y = max(float(p.get("token_throughput_per_chip", 0.0)) for p in all_pts) or 1.0

    def sx(x: float) -> float:
        return pad + (x / max_x) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y / max_y) * (height - 2 * pad)

    elems: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="Pareto Frontier">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" rx="8"/>',
        f'<line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" stroke="#94a3b8" stroke-width="1.5"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" stroke="#94a3b8" stroke-width="1.5"/>',
    ]
    for p in all_pts[:MAX_EMBEDDED_CANDIDATES]:
        cx = sx(float(p.get("p90_interactivity_tok_s", 0.0)))
        cy = sy(float(p.get("token_throughput_per_chip", 0.0)))
        elems.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="#cbd5e1"/>')

    if frontier:
        ordered = sorted(
            frontier, key=lambda r: float(r.get("p90_interactivity_tok_s", 0.0))
        )
        poly_coords = " ".join(
            f"{sx(float(r.get('p90_interactivity_tok_s', 0.0))):.1f},{sy(float(r.get('token_throughput_per_chip', 0.0))):.1f}"
            for r in ordered
        )
        elems.append(
            f'<polyline fill="none" stroke="#0f766e" stroke-width="2.5" points="{poly_coords}"/>'
        )
        for r in ordered:
            cx = sx(float(r.get("p90_interactivity_tok_s", 0.0)))
            cy = sy(float(r.get("token_throughput_per_chip", 0.0)))
            elems.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="#0f766e"/>')

    elems.append(
        f'<text x="{width / 2:.0f}" y="{height - 12}" text-anchor="middle" font-size="12" fill="#334155">'
        "p90 Interactivity (tok/s/user)</text>"
    )
    elems.append(
        f'<text x="16" y="{height / 2:.0f}" text-anchor="middle" font-size="12" fill="#334155" '
        f'transform="rotate(-90 16 {height / 2:.0f})">Throughput / Chip (tok/s)</text>'
    )
    elems.append("</svg>")
    return "\n".join(elems)


def _render_limits_svg(floor: dict[str, Any] | None) -> str:
    if not floor:
        return '<svg width="420" height="60" viewBox="0 0 420 60"><text x="16" y="36">No floor breakdown</text></svg>'
    bars = [
        ("HBM", float(floor.get("hbm_ms", 0.0)), "#0284c7"),
        ("Matmul", float(floor.get("matmul_ms", 0.0)), "#7c3aed"),
        ("Chain", float(floor.get("chain_ms", 0.0)), "#ea580c"),
        ("HopLat", float(floor.get("hop_latency_ms", 0.0)), "#64748b"),
    ]
    max_v = max(max(v for _, v, _ in bars), 1e-6)
    width, height = 440, 160
    elems = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="Step Resource Limits">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" rx="8"/>',
    ]
    for idx, (label, val, color) in enumerate(bars):
        y = 20 + idx * 32
        w = max(2.0, (val / max_v) * 260.0)
        elems.append(
            f'<text x="16" y="{y + 14}" font-size="12" fill="#1e293b">{_escape(label)}</text>'
        )
        elems.append(
            f'<rect x="86" y="{y}" width="{w:.1f}" height="18" fill="{color}" rx="3"/>'
        )
        elems.append(
            f'<text x="{94 + w:.1f}" y="{y + 14}" font-size="12" fill="#334155">{val:.2f} ms</text>'
        )
    elems.append("</svg>")
    return "\n".join(elems)


def _render_memory_svg(pools: list[dict[str, Any]]) -> str:
    if not pools:
        return '<svg width="420" height="60" viewBox="0 0 420 60"><text x="16" y="36">No pool HBM breakdown</text></svg>'
    width = 440
    height = 60 + 48 * len(pools)
    elems = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="Pool HBM Breakdown">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f8fafc" rx="8"/>',
    ]
    for idx, pool in enumerate(pools):
        lim = max(float(pool.get("hbm_limit_gb", 1.0)), 1e-6)
        w_gb = float(pool.get("hbm_weights_gb", 0.0))
        kv_gb = float(pool.get("hbm_kv_gb", 0.0))
        st_gb = float(pool.get("hbm_state_gb", 0.0))
        scale = 260.0 / lim
        y = 24 + idx * 48
        role = _escape(pool.get("role", "pool"))
        elems.append(
            f'<text x="16" y="{y + 15}" font-size="12" fill="#1e293b">{role}</text>'
        )
        x0 = 86.0
        for seg_gb, color in (
            (w_gb, "#0f766e"),
            (kv_gb, "#0284c7"),
            (st_gb, "#d97706"),
        ):
            sw = max(0.0, seg_gb * scale)
            if sw > 0.0:
                elems.append(
                    f'<rect x="{x0:.1f}" y="{y}" width="{sw:.1f}" height="20" fill="{color}"/>'
                )
                x0 += sw
        used = w_gb + kv_gb + st_gb
        elems.append(
            f'<text x="{x0 + 8:.1f}" y="{y + 15}" font-size="11" fill="#334155">{used:.1f} / {lim:.1f} GB</text>'
        )
    elems.append("</svg>")
    return "\n".join(elems)


def render_html(payload: dict[str, Any]) -> str:
    """Render a self-contained offline HTML report for an sglang.roofline.solve/v1 payload."""
    model = _escape(payload.get("model") or "HF config.json")
    chip = _escape(payload.get("chip", "tpu7x"))
    frontier = list(payload.get("frontier") or [])
    candidates = list(payload.get("candidates") or [])[:MAX_EMBEDDED_CANDIDATES]
    top_k = list(payload.get("top_k") or [])
    rec = payload.get("recommended")
    floor = rec.get("decode_floor") if rec else None
    pools: list[dict[str, Any]] = []
    if rec:
        if rec.get("prefill_pool"):
            pools.append(rec["prefill_pool"])
        if rec.get("decode_pool"):
            pools.append(rec["decode_pool"])

    svg_frontier = _render_frontier_svg(candidates, frontier)
    svg_limits = _render_limits_svg(floor)
    svg_mem = _render_memory_svg(pools)

    top_rows: list[str] = []
    for idx, item in enumerate(top_k, start=1):
        pt = item.get("point") or item
        launch = item.get("launch") or {}
        cmds = "<br>".join(
            f"<code>{_escape(c)}</code>" for c in (launch.get("commands") or [])
        )
        top_rows.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{_escape(pt.get('frontier_role') or '-')}</td>"
            f"<td>{_escape(pt.get('mode'))}</td>"
            f"<td>{float(pt.get('p90_interactivity_tok_s', 0.0)):.2f}</td>"
            f"<td>{float(pt.get('token_throughput_per_chip', 0.0)):.2f}</td>"
            f"<td>{float(pt.get('p90_ttft_ms', 0.0)):.1f}</td>"
            f"<td>{float(pt.get('p90_tpot_ms', 0.0)):.2f}</td>"
            f"<td>{float(pt.get('effective_prefix_hit_rate', 0.0)):.3f}</td>"
            f"<td>{cmds}</td>"
            "</tr>"
        )
    top_table_body = (
        "\n".join(top_rows) or '<tr><td colspan="9">No feasible candidates</td></tr>'
    )
    embedded = json.dumps(
        {**payload, "candidates": candidates},
        separators=(",", ":"),
    ).replace("</", "<\\/")

    cross = payload.get("prefix_hit_crossover_rate")
    win_rng = payload.get("winner_prefix_hit_range")
    cross_str = "none" if cross is None else f"{float(cross):.4f}"
    win_str = (
        "n/a" if not win_rng else f"[{float(win_rng[0]):.2f}, {float(win_rng[1]):.2f}]"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SGLang TPU Roofline Report - {model}</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; color: #0f172a; background: #ffffff; }}
  h1, h2 {{ margin-bottom: 8px; }}
  .meta {{ color: #475569; margin-bottom: 20px; font-size: 14px; }}
  .grid {{ display: grid; grid-template-columns: 1.4fr 1fr; gap: 20px; align-items: start; margin-bottom: 24px; }}
  .card {{ border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; background: #ffffff; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f1f5f9; }}
  code {{ background: #f1f5f9; padding: 2px 4px; border-radius: 4px; font-size: 12px; }}
  footer {{ margin-top: 24px; font-size: 12px; color: #64748b; }}
</style>
</head>
<body>
  <h1>SGLang TPU Serving Roofline: {model}</h1>
  <div class="meta">
    Chip: <strong>{chip}</strong> |
    Schema: <code>{_escape(payload.get("schema"))}</code> |
    Prefix-Hit Winner Range: <code>{win_str}</code> |
    Crossover Rate: <code>{cross_str}</code>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Pareto Ceiling Frontier</h2>
      {svg_frontier}
    </div>
    <div class="card">
      <h2>Decode Step Floor &amp; Pool HBM</h2>
      {svg_limits}
      {svg_mem}
    </div>
  </div>
  <div class="card">
    <h2>Top Structural Layouts &amp; SGLang Launch Commands</h2>
    <table>
      <thead>
        <tr>
          <th>#</th><th>Role</th><th>Mode</th><th>Interactivity (tok/s)</th>
          <th>Throughput/Chip</th><th>p90 TTFT (ms)</th><th>p90 TPOT (ms)</th>
          <th>Eff. Hit Rate</th><th>sglang.launch_server Command</th>
        </tr>
      </thead>
      <tbody>
        {top_table_body}
      </tbody>
    </table>
  </div>
  <script id="roofline-solve-payload" type="application/json">{embedded}</script>
  <footer>Generated offline by <code>sglang.srt.hardware_backend.tpu.roofline</code> (modeled analytical bounds, uncalibrated).</footer>
</body>
</html>
"""


def write_html(payload: dict[str, Any], path: Path | str) -> Path:
    """Write self-contained offline HTML report to disk and return the resolved Path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(payload), encoding="utf-8")
    return out_path
