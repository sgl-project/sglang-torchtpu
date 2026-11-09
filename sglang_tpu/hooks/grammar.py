# SPDX-License-Identifier: Apache-2.0
"""Keep XGrammar's TPU mask compilation separate from decoder compilation.

Why: the first grammar request or a new grammar batch size compiles the mask
     after decoder warmup has enabled fail_on_recompile, killing the scheduler.
How: AROUND the two XGrammar apply_vocab_mask entry points; TPU logits use the
     plugin's static-shape mask kernel with a scoped compilation stance.
Upstream PR or issue: sgl-project/sglang-torchtpu#488 and #1130 (eager alternative).
Remove when: core exposes a platform-dispatched vocab-mask implementation and
     the TPU plugin can supply this kernel through that interface.
"""

from __future__ import annotations

from sglang_tpu.grammar import apply_vocab_mask


def apply_grammar_mask(original, grammar, logits, vocab_mask):
    if logits.device.type == "tpu":
        return apply_vocab_mask(logits, vocab_mask)
    return original(grammar, logits, vocab_mask)


def apply_backend_mask(original, logits, vocab_mask):
    if logits.device.type == "tpu":
        return apply_vocab_mask(logits, vocab_mask)
    return original(logits, vocab_mask)


def hook_specs():
    from sglang.srt.plugins.hook_registry import HookType

    prefix = "sglang.srt.constrained.xgrammar_backend."
    return [
        (
            prefix + "XGrammarGrammar.apply_vocab_mask",
            apply_grammar_mask,
            HookType.AROUND,
        ),
        (
            prefix + "XGrammarGrammarBackend.apply_vocab_mask",
            apply_backend_mask,
            HookType.AROUND,
        ),
    ]
