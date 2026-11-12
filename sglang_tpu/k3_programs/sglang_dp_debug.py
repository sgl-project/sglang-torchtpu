# SPDX-License-Identifier: Apache-2.0
"""DP2 P/D transfer diagnostics (``K3_DP_DEBUG=1``): per-rank checksums of
MLA layer 0's pool rows on the prefill (the rows it sends) and on the decode
(in-place, landing and swapped rows around the swap), logged as ``K3DBG``."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_POOL: dict = {}


def _rank() -> int:
    try:
        return int(torch.distributed.get_rank())
    except Exception:
        return -1


def _position() -> int:
    try:
        from sglang_tpu.k3_programs import bridge

        for mesh, position in bridge._MESH.values():
            return int(position)
    except Exception:
        pass
    return -1


def row_sums(pool: torch.Tensor, rows) -> list[int]:
    rows = [int(r) for r in rows]
    if not rows:
        return []
    index = torch.tensor(rows, dtype=torch.int64).to(pool.device)
    picked = pool.index_select(0, index)
    if picked.dtype.itemsize == 1:
        picked = picked.view(torch.uint8)
    sums = picked.reshape(len(rows), -1)[:, : 128 * 4 * 576].to(torch.int64).sum(dim=1)
    return [int(s) for s in sums.cpu().tolist()]


def _mla0(caches) -> torch.Tensor:
    from sglang_tpu.k3_programs import sglang_bridge as bridge

    return caches[bridge.mla_layer_name(bridge.MLA[0])]


def _kda0_sum(slot) -> list[int]:
    from sglang_tpu.k3_programs import sglang_bridge as bridge

    caches, plan = _POOL.get("caches"), _POOL.get("plan")
    if caches is None or slot is None:
        return []
    pool = caches[bridge.kda_layer_name(bridge.KDA[0])]
    return row_sums(pool, [plan.state_row(int(slot))])


def install() -> None:
    if _POOL.get("installed"):
        return
    _POOL["installed"] = True
    from sglang_tpu.disaggregation.raiden import sender as raiden_sender
    from sglang_tpu.k3_programs import sglang_pools, sglang_worker

    bind = sglang_pools.K3StatePool.bind

    def bind_debug(self, plan, caches):
        bind(self, plan, caches)
        _POOL["caches"], _POOL["plan"] = caches, plan
        pool = _mla0(caches)
        _POOL["ptr"] = pool.data_ptr()
        logger.warning(
            "K3DBG bind rank=%d pos=%d num_blocks=%d kv_blocks=%d shape=%s ptr=%x",
            _rank(),
            _position(),
            plan.num_blocks,
            plan.kv_blocks,
            tuple(pool.shape),
            pool.data_ptr(),
        )

    sglang_pools.K3StatePool.bind = bind_debug

    send = raiden_sender.RaidenKVSender.send

    def send_debug(self, kv_indices, state_indices=None, num_kv_tokens=None):
        caches = _POOL.get("caches")
        if caches is not None and len(kv_indices):
            pool = _mla0(caches)
            logger.warning(
                "K3DBG P rank=%d pos=%d room=%s last=%s pages=%s sums=%s state=%s",
                _rank(),
                _position(),
                self.bootstrap_room,
                getattr(self, "_last_chunk", None),
                [int(p) for p in kv_indices],
                row_sums(pool, kv_indices),
                None
                if state_indices is None
                else np.asarray(state_indices).reshape(-1).tolist()[:4],
            )
            if state_indices is not None:
                slot = int(np.asarray(state_indices).reshape(-1)[0])
                logger.warning(
                    "K3DBG P-state rank=%d room=%s slot=%d kda0=%s",
                    _rank(),
                    self.bootstrap_room,
                    slot,
                    _kda0_sum(slot),
                )
        return send(self, kv_indices, state_indices, num_kv_tokens)

    raiden_sender.RaidenKVSender.send = send_debug

    swap = sglang_worker.DecodeSteps._swap

    def swap_debug(self, swaps):
        caches = _POOL.get("caches")
        pool = _mla0(caches) if caches is not None else None
        before = {}
        if pool is not None:
            for req, (pages, landing) in swaps:
                rows = [r for p in pages for r in self.dp.rows(int(p))]
                lrows = [r for p in landing for r in self.dp.rows(int(p))]
                before[req.rid] = (
                    rows,
                    row_sums(pool, rows),
                    lrows,
                    row_sums(pool, lrows),
                )
                logger.warning(
                    "K3DBG D-pre rank=%d pos=%d room=%s rid=%s group=%d pages=%s landing=%s "
                    "rows=%s sums=%s lrows=%s lsums=%s ptr=%x bound=%x slot=%s",
                    _rank(),
                    _position(),
                    getattr(req, "bootstrap_room", None),
                    req.rid,
                    self.dp.group(int(pages[0])),
                    list(pages),
                    list(landing),
                    rows,
                    before[req.rid][1],
                    lrows,
                    before[req.rid][3],
                    pool.data_ptr(),
                    _POOL.get("ptr", 0),
                    int(req.kv.mamba_pool_idx.reshape(-1)[0]),
                )
                logger.warning(
                    "K3DBG D-state rank=%d room=%s kda0=%s",
                    _rank(),
                    getattr(req, "bootstrap_room", None),
                    _kda0_sum(int(req.kv.mamba_pool_idx.reshape(-1)[0])),
                )
        swap(self, swaps)
        if pool is not None:
            for req, _ in swaps:
                rows, _, lrows, _ = before[req.rid]
                logger.warning(
                    "K3DBG D-post rank=%d pos=%d rid=%s sums=%s lsums=%s ptr=%x",
                    _rank(),
                    _position(),
                    req.rid,
                    row_sums(pool, rows),
                    row_sums(pool, lrows),
                    pool.data_ptr(),
                )

    sglang_worker.DecodeSteps._swap = swap_debug
    logger.warning("K3DBG installed rank=%d", _rank())
