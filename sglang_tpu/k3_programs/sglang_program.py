# SPDX-License-Identifier: Apache-2.0
"""Building a Kimi-K3 program in an sglang worker and binding it to the pools.

Two moments, as in vLLM-TorchTPU's ``MegakernelModelRunner``:

* after the model loads (:func:`build_from_server`, from the
  ``post_load_weights`` hook): the program for this server's P/D role, with
  its pre-sharded weights resident, so sglang sizes the KV pools from the HBM
  the weights leave;
* once the pools exist (:func:`build_steps`, the K3_PROGRAM worker's
  ``alloc_memory_pool``): bind the unified pools and compile every op.
"""

from __future__ import annotations

import logging
from pathlib import Path

from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_config

logger = logging.getLogger(__name__)

# This worker process's program and its configuration.
_program = None
_config = None


def parse_acc_lengths(text: str) -> dict[int, float] | None:
    """``"7:3.78,3:2.87"`` as ``{7: 3.78, 3: 2.87}``; empty: None."""
    if not text.strip():
        return None
    lengths = {}
    for item in text.split(","):
        drafts, _, length = item.partition(":")
        if not length:
            raise ValueError(
                f"simulated acceptance {item!r} is not 'drafts:mean length'"
            )
        lengths[int(drafts)] = float(length)
    return lengths


def _local_checkpoint(path: str) -> Path:
    """A checkpoint argument (directory or Hugging Face id) as its local
    directory, from the cache only."""
    if Path(path).is_dir():
        return Path(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path, local_files_only=True))


def server_engine_config():
    """This server's program configuration (P/D role, limits, DSpark)."""
    from sglang.srt.runtime_context import get_disagg, get_model, get_schedule
    from sglang_tpu.envs import envs as tpu_envs

    role = str(get_disagg().disaggregation_mode)
    if role not in sglang_config.ROLES:
        raise ValueError(
            "K3_PROGRAM serves one role of a P/D pair: --disaggregation-mode "
            f"prefill or decode, not {role!r}"
        )
    max_running = get_schedule().max_running_requests
    if not max_running:
        raise ValueError("K3_PROGRAM needs --max-running-requests")
    return sglang_config.engine_config(
        role=role,
        max_running_requests=int(max_running),
        context_length=int(get_model().context_length),
        chunked_prefill_size=int(get_schedule().chunked_prefill_size),
        kv_cache_dtype=get_model().kv_cache_dtype,
        speculative_tokens=7 if role == "decode" else 0,
        simulate_acc_lengths=parse_acc_lengths(
            tpu_envs.SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS.get()
        ),
    )


def build_from_server():
    """Build this worker's program and load its weights."""
    global _program, _config
    from sglang.srt.runtime_context import get_parallel, get_spec
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.k3_programs.kimi_k3 import program as k3
    from sglang_tpu.k3_programs.kimi_k3.kernels import dspark_mla

    presharded = tpu_envs.SGLANG_TPU_K3_PROGRAM_PRESHARDED.get()
    if not presharded:
        raise ValueError(
            "K3_PROGRAM needs SGLANG_TPU_K3_PROGRAM_PRESHARDED (python -m "
            "sglang_tpu.k3_programs.tools preshard k3 --output DIR)"
        )
    draft_path = get_spec().speculative_draft_model_path
    if not draft_path:
        raise ValueError(
            "K3_PROGRAM needs --speculative-draft-model-path (the MLA DSpark draft, "
            "Inferact/Kimi-K3-DSpark): both roles hold its KV"
        )
    draft = _local_checkpoint(str(draft_path))
    if not dspark_mla.is_mla_draft(draft):
        raise ValueError(f"K3_PROGRAM: {draft} is not the MLA DSpark draft")
    layers = dspark_mla.MlaDraftConfig.from_checkpoint(draft).layers
    if layers != bridge.DSPARK_DRAFT_LAYERS:
        raise ValueError(
            f"K3_PROGRAM: the draft has {layers} layers; the unified pools hold "
            f"{bridge.DSPARK_DRAFT_LAYERS}"
        )
    config = server_engine_config()
    # The program is a handful of whole-model custom ops per step: fusing the
    # eager ops around them would compile a new device program for every
    # distinct surrounding op sequence (vLLM-TorchTPU's runner does the same).
    from torch_tpu._internal import execution_mode

    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
    program = k3.build(
        config, Path(presharded), draft, tp_rank=int(get_parallel().tp_rank)
    )
    program.load_weights()
    _program, _config = program, config
    return program


def build_steps(target_worker):
    """The role's steps: the program bound to the target's unified pools,
    every op compiled."""
    from sglang_tpu.k3_programs.sglang_pools import K3UnifiedKVPool
    from sglang_tpu.k3_programs.sglang_worker import DecodeSteps, PrefillSteps

    if _program is None or _config is None:
        raise RuntimeError("K3_PROGRAM: the program was not built at model load")
    pool = target_worker.model_runner.token_to_kv_pool.full_kv_pool
    if not isinstance(pool, K3UnifiedKVPool):
        raise RuntimeError(
            f"K3_PROGRAM: the full-attention pool is {type(pool).__name__}"
        )
    plan = pool.plan
    width = -(-_config.max_model_len // plan.logical_block_tokens)
    # The prefill reads a prefix hit's KDA state from its page's column.
    kda_block = plan.layout.tokens_per_block if _config.role == "prefill" else None
    groups = bridge.kv_groups(
        plan,
        _config.max_model_len,
        width,
        draft_layers=bridge.DSPARK_DRAFT_LAYERS,
        kda_block_size=kda_block,
    )
    _program.bind_kv_caches(pool.caches, groups)
    _program.warmup()
    if _config.role == "prefill":
        return PrefillSteps(
            _program, plan, table_width=width, kda_width=int(groups[1].table_width)
        )
    return DecodeSteps(
        _program,
        plan,
        table_width=width,
        schedule=_config.num_speculative_tokens_per_batch_size,
    )
