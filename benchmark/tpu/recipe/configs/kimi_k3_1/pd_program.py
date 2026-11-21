"""Kimi-K3.1-Pro P/D on the megakernel programs (K3_PROGRAM), one 2x2x4 tpu7x
slice (tp32) per role, for InferenceX AgentX.

    CI_PD_ROLE=prefill python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/pd_program.py
    CI_PD_ROLE=decode  python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/pd_program.py

Kimi-K3.1-Pro is Kimi-K3 with a DSA indexer on its 24 MLA layers (top-2048):
the programs keep each rank's index keys beside its latent pages (decode
context parallelism) and select each query's keys across the ranks (a superset
of the top-2048, ``_program.SELECT_ROUNDS``). Every setting is
``kimi_k3/pd_program.py``'s, with Kimi-K3.1-Pro's model, DSpark draft at a
pinned revision and pre-shard (``_program.apply``). The decode verifies its
drafts unless K3_PROGRAM_SIMULATE_ACC_LENGTHS asks for simulated acceptance
(``_program.py`` gives InferenceX's curves). Weights come pre-sharded per host.
A draft's pre-shard goes to dspark-<its revision's first 8 characters>, which
the name of its Hugging Face snapshot directory gives; the released draft shares
the target's embedding and LM head, which its pre-shard reads from the target
checkpoint:

    python3 -m sglang_tpu.k3_programs.tools preshard k3 \\
        --checkpoint /weights/Kimi-K3.1-Pro --output $SGLANG_TPU_K3_PROGRAM_PRESHARDED
    python3 -m sglang_tpu.k3_programs.tools preshard dspark \\
        --draft $HF_HOME/hub/models--Inferact--Kimi-K3.1-Pro-DSpark/snapshots/e4e6e6261e9f8b6d5f249f9627eb6b109d762d5f \\
        --checkpoint /weights/Kimi-K3.1-Pro --output $SGLANG_TPU_K3_PROGRAM_PRESHARDED

and for K3_PROGRAM_K31_DRAFT=preview

    python3 -m sglang_tpu.k3_programs.tools preshard dspark \\
        --draft $HF_HOME/hub/models--Inferact--Kimi-K3.1-Pro-DSpark-preview/snapshots/6a2ef20a05efe88aabbf9ecaa12d020c3b0edf94 \\
        --output $SGLANG_TPU_K3_PROGRAM_PRESHARDED

Environment: as ``kimi_k3/pd_program.py``.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3 import pd_program as k3
from benchmark.tpu.recipe.configs.kimi_k3_1 import _program

CONFIG = _program.apply(deepcopy(k3.CONFIG))
