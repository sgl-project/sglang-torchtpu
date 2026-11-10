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


def install_kernel_config(text: str | None = None):
    """Install the programs' kernel config from ``"FIELD=VALUE;..."`` (default:
    ``SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG``). Kernel modules read it when they
    are first imported, so this runs before any of them."""
    from sglang_tpu.envs import envs as tpu_envs
    from sglang_tpu.k3_programs.kimi_k3 import config

    if text is None:
        text = tpu_envs.SGLANG_TPU_K3_PROGRAM_KERNEL_CONFIG.get()
    fields = {}
    for item in filter(None, (part.strip() for part in text.split(";"))):
        name, _, value = item.partition("=")
        if not value or name not in config.KernelConfig.__dataclass_fields__:
            raise ValueError(
                f"K3_PROGRAM kernel config {item!r}: expected FIELD=VALUE with a "
                "KernelConfig field"
            )
        fields[name] = value
    return config.install(config.KernelConfig().replace(**fields))


def parse_spec_schedule(text: str):
    """``"1:8:7,9:16:3"`` as ``((1, 8, 7), (9, 16, 3))``; empty: the default."""
    if not text.strip():
        return sglang_config.DEFAULT_DSPARK_SCHEDULE
    entries = []
    for item in text.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"DSpark schedule entry {item!r} is not 'min:max:drafts'")
        entries.append(tuple(int(p) for p in parts))
    return tuple(entries)


def _local_checkpoint(path: str) -> Path:
    """A checkpoint argument (directory or Hugging Face id) as its local
    directory, from the cache only."""
    if Path(path).is_dir():
        return Path(path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path, local_files_only=True))


COLOCATED = "colocated"


def server_role() -> str:
    """``"prefill"`` or ``"decode"`` for a P/D server, ``"colocated"`` for a
    server running both programs (no disaggregation)."""
    from sglang.srt.runtime_context import get_disagg

    role = str(get_disagg().disaggregation_mode)
    if role in sglang_config.ROLES:
        return role
    if role in ("null", "None", ""):
        return COLOCATED
    raise ValueError(f"K3_PROGRAM: no programs for --disaggregation-mode {role!r}")


def server_engine_config(role: str | None = None):
    """The program configuration for ``role`` (default: the server's P/D
    role) from the server's limits and DSpark settings."""
    from sglang.srt.runtime_context import get_model, get_schedule
    from sglang_tpu.envs import envs as tpu_envs

    role = role or server_role()
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
        schedule=parse_spec_schedule(
            tpu_envs.SGLANG_TPU_K3_PROGRAM_SPEC_SCHEDULE.get()
        ),
        simulate_acc_lengths=parse_acc_lengths(
            tpu_envs.SGLANG_TPU_K3_PROGRAM_SIMULATE_ACC_LENGTHS.get()
        ),
        # Attention DP2 on the decode only (sglang_dp).
        attention_dp=(
            tpu_envs.SGLANG_TPU_K3_PROGRAM_ATTENTION_DP.get() if role == "decode" else 1
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
    draft_path = (
        tpu_envs.SGLANG_TPU_K3_PROGRAM_DRAFT.get()
        or get_spec().speculative_draft_model_path
    )
    if not draft_path:
        raise ValueError(
            "K3_PROGRAM needs SGLANG_TPU_K3_PROGRAM_DRAFT (the MLA DSpark draft, "
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
    install_kernel_config()
    role = server_role()
    roles = sglang_config.ROLES if role == COLOCATED else (role,)
    configs = {r: server_engine_config(r) for r in roles}
    # The program is a handful of whole-model custom ops per step: fusing the
    # eager ops around them would compile a new device program for every
    # distinct surrounding op sequence (vLLM-TorchTPU's runner does the same).
    from torch_tpu._internal import execution_mode

    execution_mode.eager_mode = execution_mode.EagerMode.DEFER_NEVER
    tp_rank = int(get_parallel().tp_rank)
    programs = {
        r: k3.build(configs[r], Path(presharded), draft, tp_rank=tp_rank) for r in roles
    }
    if role == COLOCATED:
        from sglang_tpu.k3_programs.sglang_colocated import share_experts

        # The decode program's experts serve the prefill program too.
        share_experts(programs["prefill"], programs["decode"])
    for program in programs.values():
        program.load_weights()
    _program, _config = programs, configs
    return programs


def block_copier(prefiller):
    """``prefiller``'s compiled block copy, ``(pairs) -> None``, over every
    unified pool and the draft's (what the prefill runs for vLLM's
    copy-on-write copies; warmed with the program)."""
    from sglang_tpu.k3_programs.kimi_k3.prefill import COPY_PAIRS

    def copy(pairs):
        prefiller._writer.copy_blocks(pairs)
        if prefiller.draft_latents is not None:
            prefiller.draft_latents.copy_blocks(pairs, COPY_PAIRS)

    return copy


def _role_steps(role, program, config, pool):
    """``role``'s steps: its program bound to the unified pools, every op
    compiled."""
    from sglang_tpu.k3_programs.sglang_worker import DecodeSteps, PrefillSteps

    from sglang_tpu.k3_programs import sglang_dp
    from sglang_tpu.k3_programs.sglang_worker import dp_pages

    plan = pool.plan
    # Attention DP2 (decode only): table entries are DCP16 blocks.
    dp = dp_pages(plan) if role == "decode" else None
    table_tokens = sglang_dp.GROUP_BLOCK_TOKENS if dp else plan.logical_block_tokens
    width = -(-config.max_model_len // table_tokens)
    # The prefill reads a prefix hit's KDA state from its page's column.
    kda_block = plan.layout.tokens_per_block if role == "prefill" else None
    groups = bridge.kv_groups(
        plan,
        config.max_model_len,
        width,
        draft_layers=bridge.DSPARK_DRAFT_LAYERS,
        kda_block_size=kda_block,
        mla_block_tokens=table_tokens,
    )
    if role == "decode":
        # sglang's pools hold each TP rank's share (the prefill's DCP pages
        # and KDA heads, and Raiden pairs equal TP ranks), and a device's TP
        # rank need not be its kernel mesh position.
        program.kv_by_tp_rank = True
    # Kimi-K3.1: the DSA index-key pools beside the unified pools (rows aligned).
    program.index_pools = list(getattr(pool, "index_pools", ()))
    program.bind_kv_caches(pool.caches, groups)
    program.warmup()
    if role == "prefill":
        from sglang_tpu.k3_programs.sglang_pools import set_block_copier

        # The bridge's copies (prefix hits, checkpoints, tails) run the
        # program's copy op, not eager index copies into its pools.
        set_block_copier(block_copier(program))
        tails = None
        if plan.tail_slots:
            from sglang_tpu.k3_programs.sglang_tails import TailIndex

            tails = TailIndex(
                plan.tail_slots, plan.logical_block_tokens, plan.layout.tokens_per_block
            )
        return PrefillSteps(
            program,
            plan,
            table_width=width,
            kda_width=int(groups[1].table_width),
            tails=tails,
            kv_pool=pool,
            state_pool=pool.state_pool,
        )
    steps = DecodeSteps(
        program,
        plan,
        table_width=width,
        schedule=config.num_speculative_tokens_per_batch_size,
    )
    if dp is not None:
        sglang_dp.check_rank_order(program.destination_ranks)
        steps.swap = sglang_dp.make_swapper(program)
        if server_role() == COLOCATED:
            # The prefill keeps DCP32 pages; decoding requests add B rows.
            from sglang_tpu.k3_programs import sglang_worker

            steps.dp = None
            steps.colo = sglang_dp.ColoRows(plan.num_blocks, plan.dp_row_base, plan.dp_rows)
            sglang_worker._COLO_STEPS.append(steps)
    return steps


def build_steps(target_worker):
    """The server's steps over the target's unified pools: one role's, or
    both programs' on a colocated server."""
    from sglang_tpu.k3_programs.sglang_pools import K3UnifiedKVPool

    if _program is None or _config is None:
        raise RuntimeError("K3_PROGRAM: the program was not built at model load")
    pool = target_worker.model_runner.token_to_kv_pool.full_kv_pool
    if not isinstance(pool, K3UnifiedKVPool):
        raise RuntimeError(
            f"K3_PROGRAM: the full-attention pool is {type(pool).__name__}"
        )
    steps = {
        role: _role_steps(role, _program[role], _config[role], pool)
        for role in _program
    }
    if len(steps) == 1:
        return next(iter(steps.values()))
    from sglang_tpu.k3_programs.sglang_colocated import ColocatedSteps

    return ColocatedSteps(steps["prefill"], steps["decode"])
