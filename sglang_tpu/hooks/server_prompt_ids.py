# SPDX-License-Identifier: Apache-2.0
"""Skip the vocabulary check of prompt ids the chat server rendered itself.

Why: OpenAIServingChat hands the template-rendered prompt to the tokenizer
     manager as GenerateReqInput.input_ids, and
     TokenizerManager._validate_input_ids_in_vocab then walks every id in Python
     (type check, min, max) on the event loop that also streams every running
     request: ~6.6 ms per 150k ids off-host, and AgentX turns carry 100k-400k.
     Ids from the server's own tokenizer are in range by construction.
How: pass-through unless SGLANG_TPU_SKIP_SERVER_PROMPT_VOCAB_CHECK.
  * AROUND OpenAIServingChat._process_messages remembers, by identity and
    length, the prompt_ids list it returns when the client sent no input_ids;
  * AROUND TokenizerManager._validate_input_ids_in_vocab skips a request whose
    input_ids is such a list, unchanged in length, and forgets it. Every other
    list (client-supplied, copied, cut or extended) is validated as before.
  At most 256 lists are remembered (oldest dropped); a held reference keeps an
  id from being reused while remembered.
Upstream seam issue: not filed (a "trusted ids" flag on GenerateReqInput would
     replace the identity table).
Remove when: core skips the check for server-tokenized prompts.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

from sglang.srt.plugins.hook_registry import HookType

logger = logging.getLogger(__name__)

_MAX_REMEMBERED = 256
_lock = threading.Lock()
# id(list) -> (list, len at rendering)
_rendered: OrderedDict[int, tuple[list, int]] = OrderedDict()
_skipped = 0


def _enabled() -> bool:
    from sglang_tpu.envs import envs

    return envs.SGLANG_TPU_SKIP_SERVER_PROMPT_VOCAB_CHECK.get()


def remember(ids) -> None:
    if type(ids) is not list or not ids:
        return
    with _lock:
        _rendered[id(ids)] = (ids, len(ids))
        while len(_rendered) > _MAX_REMEMBERED:
            _rendered.popitem(last=False)


def take(ids) -> bool:
    """Whether ``ids`` is a remembered rendering, unchanged in length."""
    with _lock:
        entry = _rendered.pop(id(ids), None)
    return entry is not None and entry[0] is ids and len(ids) == entry[1]


def process_messages(original, self, request, *args, **kwargs):
    result = original(self, request, *args, **kwargs)
    if _enabled() and request.input_ids is None:
        remember(result.prompt_ids)
    return result


def validate_input_ids_in_vocab(original, self, obj):
    global _skipped
    if _enabled() and take(obj.input_ids):
        _skipped += 1
        if _skipped == 1:
            logger.info(
                "kernel engaged: site=server_prompt_vocab_check used=skip "
                "(server-rendered prompt ids)"
            )
        return None
    return original(self, obj)


def hook_specs():
    return [
        (
            "sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat._process_messages",
            process_messages,
            HookType.AROUND,
        ),
        (
            "sglang.srt.managers.tokenizer_manager.TokenizerManager._validate_input_ids_in_vocab",
            validate_input_ids_in_vocab,
            HookType.AROUND,
        ),
    ]
