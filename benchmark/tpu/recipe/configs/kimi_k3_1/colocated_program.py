"""Kimi-K3.1-Pro on the megakernel programs, both programs in one engine per
2x2x4 tpu7x slice (tp32), for InferenceX AgentX.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/colocated_program.py

Every setting is ``kimi_k3/colocated_program.py``'s, with Kimi-K3.1-Pro's
DSpark draft, golden acceptance and pre-shard (``_program.py``); see
``pd_program.py`` for the DSA indexer on the programs.

Environment: as ``kimi_k3/colocated_program.py``.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3 import colocated_program as k3
from benchmark.tpu.recipe.configs.kimi_k3_1 import _program

CONFIG = _program.apply(deepcopy(k3.CONFIG))
