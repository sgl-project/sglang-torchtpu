"""Kimi-K3.1-Pro on the megakernel programs, both programs in one engine per
2x2x4 tpu7x slice (tp32), for InferenceX AgentX.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/colocated_program.py

Every setting is ``kimi_k3/colocated_program.py``'s, sized for concurrency 1
by default, with Kimi-K3.1-Pro's model, DSpark draft at a pinned revision
(the released Inferact/Kimi-K3.1-Pro-DSpark unless K3_PROGRAM_K31_DRAFT=preview)
and pre-shard (``_program.apply``) and the colocated AgentX settings
(``_program.colocated_defaults``): the decode's kernel settings, expert
parallelism 8 for both programs, host-side bookkeeping, first-step drafts (with
the MLA draft), a 150-round prefill wait, the TensorCore's highest clock,
Prometheus metrics and no reasoning or tool-call parser. The decode verifies
its drafts unless K3_PROGRAM_SIMULATE_ACC_LENGTHS asks for simulated acceptance
("curve": the selected draft's InferenceX curve, ``_program.DRAFTS``); see
``pd_program.py`` for the DSA indexer on the programs and the pre-shard.

Environment: as ``kimi_k3/colocated_program.py``, with these defaults here:
    K3_PROGRAM_CONCURRENCY          1 (4 running requests, 16 KDA checkpoints
                                    and 8 prefix tails: size_for_concurrency)
    K3_PROGRAM_EXPERT_PARALLEL      8
    K3_PROGRAM_PREFILL_WAIT         150
    SGLANG_TPU_K3_PROGRAM_OPT_HOST_BOOKKEEPING   1
    SGLANG_TPU_K3_PROGRAM_OPT_FIRST_STEP_DRAFTS  1
plus
    K3_PROGRAM_K31_DRAFT            released (default) or preview
    K3_PROGRAM_SIMULATE_ACC_LENGTHS curve, or drafts:length,... (unset: the
                                    decode verifies its drafts)
    K3_PROGRAM_LIBTPU_EXTRA         libtpu flags on top of the recipe's
                                    (default --xla_tpu_dvfs_p_state=7, empty
                                    for none)
"""

import os
from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3 import colocated_program as k3
from benchmark.tpu.recipe.configs.kimi_k3_1 import _program

# This engine's share of the client concurrency: one session at a time.
CONCURRENCY = int(os.environ.get("K3_PROGRAM_CONCURRENCY", "1"))

CONFIG = _program.colocated_defaults(
    _program.apply(k3.size_for_concurrency(deepcopy(k3.CONFIG), CONCURRENCY))
)
