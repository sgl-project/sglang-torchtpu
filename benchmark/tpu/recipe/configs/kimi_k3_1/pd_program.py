"""Kimi-K3.1-Pro P/D on the megakernel programs (K3_PROGRAM), one 2x2x4 tpu7x
slice (tp32) per role, for InferenceX AgentX.

    CI_PD_ROLE=prefill python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/pd_program.py
    CI_PD_ROLE=decode  python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/pd_program.py

Kimi-K3.1-Pro is Kimi-K3 with a DSA indexer on its 24 MLA layers (top-2048):
the programs keep each rank's index keys beside its latent pages (decode
context parallelism) and select each query's keys across the ranks. Every
setting is ``kimi_k3/pd_program.py``'s, with K3.1's DSpark draft and golden
acceptance (``_program.py``). Weights come pre-sharded per host:

    python3 -m sglang_tpu.k3_programs.tools preshard k3 \\
        --checkpoint /weights/Kimi-K3.1-Pro --output $SGLANG_TPU_K3_PROGRAM_PRESHARDED
    python3 -m sglang_tpu.k3_programs.tools preshard dspark \\
        --draft /weights/Kimi-K3.1-Pro-DSpark-preview --output $SGLANG_TPU_K3_PROGRAM_PRESHARDED

Environment: as ``kimi_k3/pd_program.py``.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3 import pd_program as k3
from benchmark.tpu.recipe.configs.kimi_k3_1 import _program

CONFIG = _program.apply(deepcopy(k3.CONFIG))
