# SPDX-License-Identifier: Apache-2.0
"""`safe_open`: slower to read, and the only source whose pages stay reclaimable.

`safe_open` maps the file, so a CPU tensor it yields is a view into the
checkpoint rather than a copy. Those pages are file-backed, which is what lets
the kernel drop them under memory pressure even on a host with no swap.

Where host copies are redundant (the device shares the host pool) the mapping
is made read-only instead: safetensors maps writable, and a device copy from a
writable private mapping there turns every page it touches into anonymous
memory at a fraction of the bandwidth (see readonly_safetensors).
"""

from typing import Callable, ClassVar, Iterator

import torch
from safetensors.torch import safe_open
from tqdm.auto import tqdm

from sglang.multimodal_gen.runtime.loader.readonly_safetensors import (
    iter_safetensors_readonly,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
    host_copies_are_redundant,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

_BAR_FORMAT = "{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt}"


class SafetensorsMmapReader:
    name: ClassVar[str] = "safetensors"
    supports_key_filter: ClassVar[bool] = True
    retains_file_mapping: ClassVar[bool] = True

    @classmethod
    def is_available(cls) -> bool:
        return True

    def iter_weights(
        self,
        files: list[str],
        *,
        device: str,
        to_cpu: bool,
        key_filter: Callable[[str], bool] | None = None,
        clone_tensors: bool = True,
        show_progress: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        # A device safetensors cannot address is read on the host, then moved.
        read_on_host = (
            device != "cpu" and not current_platform.supports_safetensors_device_load()
        )
        for path in tqdm(
            files,
            desc="Loading safetensors checkpoint shards",
            disable=not show_progress,
            bar_format=_BAR_FORMAT,
        ):
            if device == "cpu" and host_copies_are_redundant():
                for name, tensor in iter_safetensors_readonly(path):
                    if key_filter is not None and not key_filter(name):
                        continue
                    yield name, tensor
                continue
            with safe_open(
                path, framework="pt", device="cpu" if read_on_host else device
            ) as handle:
                moved_to = None
                for name in handle.keys():  # noqa: SIM118
                    if key_filter is not None and not key_filter(name):
                        continue
                    tensor = handle.get_tensor(name)
                    if read_on_host:
                        tensor = tensor.to(device)
                        moved_to = tensor.device
                    yield name, tensor
                if moved_to is not None and moved_to.type != "cpu":
                    # The host tensors are views into the mapping, which closes
                    # with this block: finish the host-to-device copies first.
                    torch.accelerator.synchronize()
