"""Kimi-K3.1-Pro on the megakernel programs, both programs in one engine per
2x2x4 tpu7x slice (tp32), for InferenceX AgentX.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/kimi_k3_1/colocated_program.py

Every setting is ``kimi_k3/colocated_program.py``'s, with Kimi-K3.1-Pro's
DSpark draft, golden acceptance and pre-shard (``_program.py``) and its
colocated c1 AgentX settings (``_program.colocated_defaults``: the decode's
kernel settings, expert parallelism 8, host-side bookkeeping, first-step drafts,
a 150-round prefill wait and the TensorCore's highest clock); see
``pd_program.py`` for the DSA indexer on the programs.

Environment: as ``kimi_k3/colocated_program.py`` (K3_PROGRAM_EXPERT_PARALLEL
defaults to 8 and K3_PROGRAM_PREFILL_WAIT to 150 here), plus
K3_PROGRAM_LIBTPU_EXTRA (libtpu flags on top of the recipe's; default
--xla_tpu_dvfs_p_state=7, empty for none).
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.kimi_k3 import colocated_program as k3
from benchmark.tpu.recipe.configs.kimi_k3_1 import _program

CONFIG = _program.colocated_defaults(_program.apply(deepcopy(k3.CONFIG)))
