"""Backend-neutral helpers for the input-logprob unit tests.

This module registers no CI test: run_suite.py only runs files under
test/registered/.
"""

import itertools
from types import SimpleNamespace

import torch

from sglang.srt.layers.logprob_processor import InputLogprobProcessor


def coverage_cases(menu, max_seqs):
    """Every singleton, every ordered pair, and wider heterogeneous cases.

    `menu` order is load-bearing: width >= 3 walks it cyclically, so reordering
    it changes which wide combinations run.
    """
    yield from ((item,) for item in menu)
    yield from itertools.product(menu, repeat=2)
    for width in range(3, max_seqs + 1):
        for offset in range(len(menu)):
            yield tuple(menu[(offset + step) % len(menu)] for step in range(width))
            yield tuple(menu[(offset - step) % len(menu)] for step in range(width))
        # Cyclic windows never repeat an item; adjacent duplicates need their own.
        for index, item in enumerate(menu):
            other = menu[(index + 1) % len(menu)]
            yield (item,) * (width - 1) + (other,)
            yield (other,) + (item,) * (width - 1)


class CpuDefaultDeviceMixin:
    """TestCase mixin: build tensors on the CPU even if an earlier test in the
    same process left another default device (torch.set_default_device) behind."""

    def setUp(self):
        super().setUp()
        device = torch.device("cpu")
        device.__enter__()
        self.addCleanup(device.__exit__, None, None, None)


def input_logprob_metadata(extend_lens, start_lens, token_ids):
    """Duck-typed LogitsMetadata for InputLogprobProcessor over pruned rows.

    Request i has extend_lens[i] rows and asks for input logprobs from row
    start_lens[i]. Its pruned rows start there, except that a request with no
    logprob row (start == extend) keeps its last row for sampling, as
    LogitsProcessor._get_pruned_states does.

    Returns (metadata, sample_indices, input_logprob_indices, token_to_seq_idx).
    """
    sample, lp_idx, pruned_lens, t2s = [], [], [], []
    sample_pt, lp_pt = -1, 0
    for i, (extend, start) in enumerate(zip(extend_lens, start_lens)):
        eff = start - 1 if extend == start else start
        rows = extend - eff
        t2s += [i] * rows
        sample_pt += rows
        sample.append(sample_pt)
        lp_idx += [lp_pt + j for j in range(extend - start)]
        lp_pt += rows
        pruned_lens.append(extend - start)
    md = SimpleNamespace(
        sample_indices_cpu=sample,
        input_logprob_indices_cpu=lp_idx,
        extend_return_top_logprob=False,
        extend_token_ids_logprob=False,
        top_logprobs_nums=[0] * len(extend_lens),
        extend_logprob_pruned_lens_cpu=pruned_lens,
        extend_input_logprob_token_ids_gpu=token_ids[: len(lp_idx)],
        token_ids_logprobs=[None] * len(extend_lens),
    )
    return md, torch.tensor(sample), torch.tensor(lp_idx), t2s


def row_count_dependent_logits(calls):
    """A get_logits_fn stub that appends each call's row count to `calls`.

    Its output depends on that row count, like a real epilogue whose rounding
    depends on its shapes, so the same rows grouped into different calls give
    different logprobs.
    """

    def get_logits_fn(states, lm_head, logits_metadata, **kwargs):
        calls.append(states.shape[0])
        return states.float() * (1.0 + 1e-6 * states.shape[0])

    return get_logits_fn


def chunked_input_logprob_processor(vocab_size, chunk_size):
    """The real InputLogprobProcessor with chunking and fast input logprobs on."""
    proc = InputLogprobProcessor(vocab_size=vocab_size)
    proc.enable_logprobs_chunk = True
    proc.logprobs_chunk_size = chunk_size
    proc.enable_fast_input_logprobs = True
    return proc


def score_input_logprobs(
    states,
    token_ids,
    extend_lens,
    start_lens,
    cuts,
    *,
    vocab_size,
    chunk_size,
    skip_chunking_for_dp_attn=False,
):
    """Input logprobs of the pruned rows `states`, and the row count of each
    get_logits_fn call, from the chunked processor and the row-count-dependent
    stub. `cuts` sets metadata.input_logprob_chunk_cuts_cpu; None leaves it unset.
    """
    md, sample, lp_idx, t2s = input_logprob_metadata(extend_lens, start_lens, token_ids)
    if cuts is not None:
        md.input_logprob_chunk_cuts_cpu = cuts
    calls = []
    res, _ = chunked_input_logprob_processor(vocab_size, chunk_size).forward(
        pruned_states=states,
        sample_indices=sample,
        input_logprob_indices=lp_idx,
        token_to_seq_idx=t2s,
        lm_head=None,
        get_logits_fn=row_count_dependent_logits(calls),
        logits_metadata=md,
        skip_chunking_for_dp_attn=skip_chunking_for_dp_attn,
    )
    return res.token_logprobs, calls
