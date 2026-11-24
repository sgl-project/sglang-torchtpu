# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 on TPU workers: compile the listed prompt lengths at startup.

Runtime patch, tracked debt:
- Why: torch_tpu compiles an eager op once per input signature and caches the
  executable. H3 runs its text path at the exact prompt token count: the
  Qwen3-VL encoder, the DiT's token refiner, and in every denoise step the
  text-row copy into the packed sequence and the rank-local video/audio row
  projections, whose row counts move with the prompt length on the ranks
  holding the text rows and the end of the live rows. The first request of
  each new prompt length waits for those compiles in its denoise phase (+6.8 s
  at 1344x768x124 on v7x-8: 6 of the 20 clips of a 21-prompt benchmark set,
  whose prompts have 9 token counts).
- How: with ``SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS`` set, :func:`install`
  wraps ``MiniMaxH3Pipeline.__init__``: once the pipeline is built, every rank
  runs one warmup request per listed length (and warmup resolution) through
  the pipeline's executor and its stock stages before the decoding stage, with
  one denoise step. Each is sglang's synthetic H3 warmup request (canvas, seed,
  ``is_warmup``) carrying a generated prompt of exactly that many tokens, so
  it dispatches the programs a real prompt of that length dispatches, and
  torch_tpu's compile cache keeps them, keyed on shapes. The requests' tensors
  are dropped; nothing is saved, returned or kept, and no client text is used.
- Upstream PR or issue: none.
- Remove when: torch_tpu runs these ops at a bounded dynamic text length, or
  H3's text path pads to a bucket bit-exactly.
"""

from __future__ import annotations

import functools
import gc
import re
import time
from collections.abc import Iterable, Sequence
from copy import copy
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang_tpu.envs import envs as tpu_envs

logger = init_logger(__name__)

# Words whose repeats tokenize to one token each: "the the ... the" is N tokens
# for every N up to 1,024 in H3's Qwen2 tokenizer. The rest are fallbacks.
FILLER_WORDS = ("the", "a", "of", "x")
# The warmup stops before this stage: the decode does not see the prompt.
STOP_STAGE = "MiniMaxH3DecodingStage"
_TEXT_STAGE = "MiniMaxH3TextEncodingStage"
_DENOISE_STAGE = "MiniMaxH3DenoisingStage"
# Each listed length costs one warmup request a warmup resolution.
MAX_LENGTHS = 4096

_installed = False


def requested() -> bool:
    spec = tpu_envs.SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS.get()
    return bool(spec and spec.strip())


def parse_lengths(spec: str) -> list[int]:
    """``"36-47,64"`` -> ``[36, ..., 47, 64]``: sorted, unique, positive."""
    lengths: set[int] = set()
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", part)
        if match is None:
            raise ValueError(
                f"SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS: {part!r} is not a token "
                "count or a range like 24-64"
            )
        first, last = int(match.group(1)), int(match.group(2) or match.group(1))
        if first < 1 or last < first:
            raise ValueError(
                f"SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS: {part!r} is not a range of "
                "positive token counts"
            )
        lengths.update(range(first, last + 1))
        if len(lengths) > MAX_LENGTHS:
            raise ValueError(
                f"SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS lists more than {MAX_LENGTHS} "
                "lengths"
            )
    if not lengths:
        raise ValueError("SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS lists no length")
    return sorted(lengths)


def format_lengths(lengths: Iterable[int]) -> str:
    """``[36, 37, 38, 40]`` -> ``"36-38,40"``."""
    spans: list[list[int]] = []
    for value in sorted(set(lengths)):
        if spans and value == spans[-1][1] + 1:
            spans[-1][1] = value
        else:
            spans.append([value, value])
    return ",".join(f"{a}" if a == b else f"{a}-{b}" for a, b in spans)


def filler_prompt(tokenizer, length: int) -> str:
    """A prompt of exactly ``length`` t2va text tokens, by the stage's own
    tokenization: one filler word repeated. Only the token count reaches a
    shape; no client text is involved."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.presentation import (
        minimax_h3_text_only_ids,
    )

    for word in FILLER_WORDS:
        prompt = " ".join([word] * length)
        if int(minimax_h3_text_only_ids(tokenizer, prompt).shape[0]) == length:
            return prompt
    raise ValueError(f"no filler prompt tokenizes to {length} tokens")


def warmup_requests(server_args, prompt: str) -> list[Any]:
    """sglang's synthetic H3 warmup request with ``prompt``, one a warmup
    resolution: the canvas ``prepare_synthetic_warmup_request_for_queue``
    resolves from ``--warmup-resolutions`` / ``--warmup-num-frames`` (default:
    768 short edge, 16:9, 5 s: the 1344x768 canvas), one denoise step."""
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
    from sglang.multimodal_gen.runtime.utils.common import parse_size
    from sglang.multimodal_gen.runtime.warmup_request_builder import (
        get_model_sampling_defaults,
    )

    defaults = get_model_sampling_defaults(server_args)
    task_type = server_args.pipeline_config.task_type
    sizes = (
        [parse_size(size) for size in server_args.warmup_resolutions]
        if server_args.warmup_resolutions
        else [(defaults.width, defaults.height)]
    )
    requests = []
    for width, height in sizes:
        req = Req(
            task_type=task_type,
            data_type=task_type.data_type(),
            width=width,
            height=height,
            prompt=prompt,
            sampling_params=copy(defaults),
            num_inference_steps=defaults.num_inference_steps,
        )
        req.set_as_warmup(1)
        req.sampling_params.prepare_synthetic_warmup_request_for_queue(req, server_args)
        requests.append(req)
    return requests


def _wait(*tensors) -> None:
    """Block until the device ran a warmup request: the worker's own drain
    after a request where the platform asks for one (TPU), else one element of
    each of ``tensors`` to the host (everything the request dispatched feeds
    them)."""
    from sglang.multimodal_gen.runtime.platforms import current_platform

    if current_platform.drain_device_queue_after_request():
        torch.accelerator.synchronize()
        return
    for tensor in tensors:
        if torch.is_tensor(tensor) and tensor.numel():
            tensor.reshape(-1)[:1].cpu()


def warm(pipeline, server_args, lengths: Sequence[int]) -> dict | None:
    """One warmup request a length (and warmup resolution) through the stages
    before decoding, on this rank; every rank runs the same sequence, so the
    collectives line up. Returns the timings, or None when skipped."""
    names = [type(stage).__name__ for stage in pipeline.stages]
    missing = [n for n in (_TEXT_STAGE, _DENOISE_STAGE, STOP_STAGE) if n not in names]
    metadata = getattr(pipeline, "release_metadata", None)
    tasks = tuple(getattr(metadata, "tasks", ()) or ())
    layerwise = getattr(server_args, "has_layerwise_offload_components", None)
    why = None
    if missing:
        why = f"the pipeline has no {', '.join(missing)}"
    elif "t2va" not in tasks:
        why = f"the checkpoint serves no t2va (tasks {list(tasks)})"
    elif callable(layerwise) and layerwise():
        # The worker attaches layerwise offload after the pipeline is built.
        why = "layerwise offload is configured after the pipeline is built"
    if why is None:
        # Host-only preparation first (every prompt, and one request lowered):
        # a failure here is the same on every rank, so all of them skip.
        try:
            tokenizer = pipeline.get_module("tokenizer")
            prompts = [(length, filler_prompt(tokenizer, length)) for length in lengths]
            warmup_requests(server_args, prompts[0][1])
        except Exception as error:  # noqa: BLE001 - every rank skips alike
            why = f"{type(error).__name__}: {error}"
    if why is not None:
        logger.warning("MiniMax H3 text-length warmup skipped: %s", why)
        return None

    head = pipeline.stages[: names.index(STOP_STAGE)]
    pipeline._install_component_residency_manager(server_args)
    seconds: dict[int, float] = {}
    started = time.perf_counter()
    for length, prompt in prompts:
        t0 = time.perf_counter()
        # Requests are built here and dropped after their run: none outlives it.
        for req in warmup_requests(server_args, prompt):
            req = pipeline.executor.execute_with_profiling(head, req, server_args)
            _wait(req.latents, req.audio_latents)
            del req
        seconds[length] = time.perf_counter() - t0
        logger.debug(
            "MiniMax H3 text-length warmup: %d tokens in %.2f s",
            length,
            seconds[length],
        )
    # Nothing a warmup request computed stays behind (with the conditioning
    # cache disabled, clearing it is a no-op).
    cache = getattr(pipeline.executor, "conditioning_cache", None)
    if cache is not None:
        cache.clear()
    gc.collect()
    total = time.perf_counter() - started
    # The first length also compiles the length-independent programs of these
    # stages; the others cost only their own (the per-length compile cost).
    first, *others = seconds
    per_length = (
        f", the other {len(others)} {sum(seconds[n] for n in others) / len(others):.2f} s "
        f"a length (max {max(seconds[n] for n in others):.2f} s)"
        if others
        else ""
    )
    logger.info(
        "MiniMax H3 text-length warmup: %d prompt lengths (%s) through the text "
        "encoder, token refiner and one denoise step in %.1f s: %d tokens %.2f s "
        "(with the length-independent programs)%s",
        len(seconds),
        format_lengths(seconds),
        total,
        first,
        seconds[first],
        per_length,
    )
    return {"seconds": seconds, "total": total}


def install() -> None:
    """Run :func:`warm` at the end of every MiniMax H3 pipeline construction
    (once a process); the listed lengths are parsed now, so a bad value fails
    the worker start."""
    global _installed
    if _installed:
        return
    lengths = parse_lengths(tpu_envs.SGLANG_TPU_OPT_H3_TEXT_WARMUP_LENGTHS.get())
    from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
        MiniMaxH3Pipeline,
    )

    init = MiniMaxH3Pipeline.__init__

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        warm(self, self.server_args, lengths)

    MiniMaxH3Pipeline.__init__ = __init__
    _installed = True
    logger.info(
        "MiniMax H3 text-length warmup on: %d prompt lengths (%s) at startup",
        len(lengths),
        format_lengths(lengths),
    )
