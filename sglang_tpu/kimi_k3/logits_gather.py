# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 TP logits gather: all-reduce (default) and one-shot Pallas (opt-in).

The vocab-parallel lm_head output ([rows, V / tp] per rank) is gathered along
the vocab dim by XLA's all-gather. Two wrappers replace it after load; both
are bit-identical to the all-gather.

All-reduce (SGLANG_TPU_LOGITS_GATHER_ALLREDUCE, on by default)
Why: in the compiled Kimi-K3.1 prefill graph (93 layers, TP32,
     chunked_prefill_size 4096, prefill_max_requests 8) XLA's all-gather
     delivered wrong data: the first token of each request of a batched
     prefill came from garbage logits (0/8 right; 8/8 with this wrapper). The
     DSA indexer's int32 row all-gather in the same graph failed the same way,
     so the suspect is XLA's all-gather in that graph, not the logits. Cause
     not found yet; no minimal repro.
How: every 2-D gather becomes an all-reduce (sum) over the TP group of a zero
     [tp, rows, V / tp] buffer in which each rank fills only its own slot
     (slot = TP rank). x + 0 is x. SGLANG_TPU_LOGITS_GATHER_ALLREDUCE=0
     restores the all-gather.
Cost: about twice the bytes of the all-gather. On 93L TP32 no TTFT or TPOT
     change was measurable (64K / 128K, cold and 90% hit).

One-shot (SGLANG_TPU_LOGITS_ONESHOT_GATHER, opt-in)
Why: one decode token's logits ([1, V / tp] bf16, 10 KB per rank at TP32) go
     through XLA's all-gather on the SparseCore with a high fixed latency:
     47.6 us per decode step on 16 chips (TP32), 14.3 us at TP8. It sits at the
     very end of the step, after the lm_head GEMV, on the critical path. The
     one-shot kernel: 9.9 us at TP8 (13-layer proxy).
How: single-token bf16 logits go to kernels/oneshot_all_gather.py (every rank
     DMA-writes its shard into every peer's output slot, slot = TP rank).
     It wraps the all-reduce wrapper when both are on, so everything else
     (prefill, bs > 1, other dtypes) still takes the all-reduce.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# One decode token's logits shard; larger payloads keep the wrapped gatherer.
_MAX_SHARD_BYTES = 256 * 1024


class _AllReduceLogitsGatherer:
    def __init__(self, orig, holder: torch.nn.Module, world: int):
        self.orig, self.holder, self.world = orig, holder, world

    def __getattr__(self, name):
        return getattr(self.orig, name)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            return self.orig(x)
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        rows, n = x.shape
        mine = (
            torch.arange(self.world, device=x.device, dtype=torch.int32)
            == self.holder._tpu_logits_tp_rank
        ).view(self.world, 1, 1)
        buf = torch.where(mine, x.unsqueeze(0), torch.zeros_like(x).unsqueeze(0))
        buf = tensor_model_parallel_all_reduce(buf)
        # [world, rows, n] -> [rows, world * n]: rank r's shard at columns
        # [r * n, (r + 1) * n), as the all-gather along dim -1 lays it out.
        return buf.permute(1, 0, 2).reshape(rows, self.world * n)


class _OneshotLogitsGatherer:
    def __init__(self, orig, holder: torch.nn.Module, world: int):
        self.orig, self.holder, self.world = orig, holder, world

    def __getattr__(self, name):
        return getattr(self.orig, name)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not (
            x.dim() == 2
            and x.shape[0] == 1
            and x.dtype == torch.bfloat16
            and x.shape[1] % 128 == 0
            and x.numel() * x.element_size() <= _MAX_SHARD_BYTES
        ):
            return self.orig(x)
        from sglang_tpu.oneshot_collectives import oneshot_all_gather

        out = oneshot_all_gather(
            x.contiguous(),
            rank_input=self.holder._tpu_logits_tp_rank,
            world_size=self.world,
        )
        # [world, 1, n] in TP-rank order -> [1, world * n] (gather along dim -1).
        return out.reshape(1, self.world * x.shape[1])


def plain_tp_logits_gather(lp) -> bool:
    """True when the logits processor gathers the logits over the plain TP group."""
    return (
        bool(getattr(lp, "do_tensor_parallel_all_gather", False))
        and not getattr(lp, "use_attn_tp_group", False)
        and not getattr(lp, "do_tensor_parallel_all_gather_dp_attn", False)
    )


def register_tp_rank_buffer(lp: torch.nn.Module, tp_rank: int, device) -> None:
    """The TP rank as a device tensor, so every rank runs one compiled graph."""
    if "_tpu_logits_tp_rank" in lp._buffers:
        return
    lp.register_buffer(
        "_tpu_logits_tp_rank",
        torch.tensor([tp_rank], dtype=torch.int32, device=device),
        persistent=False,
    )


def bind_logits_allreduce_gather(model) -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_LOGITS_GATHER_ALLREDUCE.get():
        return False
    lp = getattr(model, "logits_processor", None)
    if lp is None or getattr(lp, "_tpu_logits_allreduce_bound", False):
        return False
    if not plain_tp_logits_gather(lp):
        logger.info("Logits all-reduce gather off: no plain TP logits gather.")
        return False
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world = parallel.tp_size
    if world <= 1:
        return False
    register_tp_rank_buffer(lp, parallel.tp_rank, next(model.parameters()).device)
    lp._logits_gatherer = _AllReduceLogitsGatherer(lp._logits_gatherer, lp, world)
    lp._tpu_logits_allreduce_bound = True
    logger.info("Logits TP gather: all-reduce of a zero-padded buffer (TP %d).", world)
    return True


def bind_logits_oneshot_gather(model) -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_LOGITS_ONESHOT_GATHER.get():
        return False
    lp = getattr(model, "logits_processor", None)
    if lp is None or getattr(lp, "_tpu_logits_oneshot_bound", False):
        return False
    if not plain_tp_logits_gather(lp):
        logger.info("Logits one-shot all-gather off: no plain TP logits gather.")
        return False
    import torch.distributed as dist

    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    world = parallel.tp_size
    if world <= 1 or world != dist.get_world_size():
        logger.info(
            "Logits one-shot all-gather off: TP %d does not span the world of %d.",
            world,
            dist.get_world_size(),
        )
        return False
    from sglang_tpu.oneshot_collectives import get_oneshot_all_gather_op

    get_oneshot_all_gather_op(world)  # build before torch.compile traces
    register_tp_rank_buffer(lp, parallel.tp_rank, next(model.parameters()).device)
    lp._logits_gatherer = _OneshotLogitsGatherer(lp._logits_gatherer, lp, world)
    lp._tpu_logits_oneshot_bound = True
    logger.info(
        "Single-token decode logits all-gather: one-shot Pallas kernel (TP %d).",
        world,
    )
    return True
