# SPDX-License-Identifier: Apache-2.0
"""Light GSM8K check on the anchor's device programs (``anchor --gsm8k N``).

N questions drawn with a fixed seed from a local GSM8K test jsonl (its first ``SHOTS``
lines are the few-shot examples, as in sglang's ``GSM8KEval``); each prompt is the
5-shot ``Question: / Answer:`` text as one user message through the K3 chat template
(response channel unless ``--gsm8k-think``). The prompt is fed by stepping the forced
program (64-token chunks, then single tokens: three compiled programs for any prompt
length), then greedy decode runs in 64-token chunks until a stop token or
``--gsm8k-max-new``. The answer is the last number of the response (sglang's
flexible extract). Records and the accuracy go to ``--out``.
"""

import ast
import json
import os
import random
import re
import time

SHOTS = 5
CHUNK = 64
# [EOS], <|end_of_msg|> (generation_config eos), <|close|> (ends the response channel).
STOP_IDS = (163585, 163586, 163588)
# The GSM8K test split as jsonl (sglang's GSM8KEval file); --gsm8k-data overrides it.
DEFAULT_DATA = os.environ.get("K3MK_GSM8K_DATA", "")
INVALID = -9999999


def answer_value(text: str):
    """Last number in ``text`` (commas dropped), as sglang's GSM8K scorer."""
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    if not numbers:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except (SyntaxError, ValueError):
        return INVALID


def _example(line: dict, with_answer: bool) -> str:
    s = f"Question: {line['question']}\nAnswer:"
    return s + f" {line['answer']}" if with_answer else s


def load_questions(path: str, n: int, seed: int) -> tuple[str, list[tuple[int, dict]]]:
    """(few-shot prefix, [(line index, line)]) with ``n`` lines sampled after the shots."""
    if not path:
        raise ValueError("no GSM8K data: pass --gsm8k-data or set K3MK_GSM8K_DATA")
    with open(path) as f:
        lines = [json.loads(x) for x in f if x.strip()]
    prefix = "".join(_example(lines[i], True) + "\n\n" for i in range(SHOTS))
    pool = list(range(SHOTS, len(lines)))
    picked = pool if n >= len(pool) else sorted(random.Random(seed).sample(pool, n))
    return prefix, [(i, lines[i]) for i in picked]


class Gsm8kRunner:
    """The three device programs over an anchor ``Bench``'s weights and states."""

    def __init__(self, bench, tokenizer, max_new: int):
        from bench import program

        self.bench, self.tok, self.max_new = bench, tokenizer, max_new
        build = lambda steps, forced: program.build_program(
            mesh=bench.mesh,
            kernel=bench.kernel,
            spec=bench.spec,
            shapes=bench.shapes,
            steps=steps,
            forced=forced,
            embed_mode=bench.args.embed,
            vocab=bench.vocab,
        )
        self.feed_chunk, self.feed_one, self.decode = (
            build(CHUNK, True),
            build(1, True),
            build(CHUNK, False),
        )

    def _host_row(self, out):
        import numpy as np

        return [int(t) for t in np.asarray(out.addressable_shards[0].data)[0]]

    def generate(self, prompt_ids: list[int]) -> tuple[list[int], float, float]:
        """Greedy continuation of ``prompt_ids`` (stop token excluded); prefill and decode seconds."""
        import jax
        import numpy as np

        b = self.bench
        states = b.fresh_states()
        tok, kv = b._rows(0, np.int32), b._rows(1, np.int32)
        t0 = time.perf_counter()
        pos = 0
        while pos < len(prompt_ids):
            n = CHUNK if len(prompt_ids) - pos >= CHUNK else 1
            fn = self.feed_chunk if n == CHUNK else self.feed_one
            forced = b._rows(prompt_ids[pos : pos + n], np.int32)
            states, tok, kv, out = fn(
                b.stack_w, b.vocab_w, b.consts, states, tok, kv, forced
            )
            pos += n
        gen = [self._host_row(out)[-1]]
        prefill_s = time.perf_counter() - t0
        dummy = b._rows([0], np.int32)
        t0 = time.perf_counter()
        while len(gen) < self.max_new and not any(t in STOP_IDS for t in gen):
            states, tok, kv, out = self.decode(
                b.stack_w, b.vocab_w, b.consts, states, tok, kv, dummy
            )
            gen += self._host_row(out)
        jax.block_until_ready(states)
        decode_s = time.perf_counter() - t0
        stop = next((i for i, t in enumerate(gen) if t in STOP_IDS), len(gen))
        return gen[: min(stop, self.max_new)], prefill_s, decode_s

    def prompt_ids(self, content: str, think: bool) -> list[int]:
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            thinking=think,
        )
        return list(ids["input_ids"] if isinstance(ids, dict) else ids)


def run(args, bench, log) -> dict:
    """Runs the check; writes one record per question and a summary to ``args.out`` (process 0)."""
    import jax
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
    prefix, questions = load_questions(
        args.gsm8k_data or DEFAULT_DATA, args.gsm8k, args.gsm8k_seed
    )
    runner = Gsm8kRunner(bench, tokenizer, args.gsm8k_max_new)
    records, t_start = [], time.time()

    def emit(rec):
        log(json.dumps(rec))
        if args.out and jax.process_index() == 0:
            with open(args.out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    for k, (index, line) in enumerate(questions):
        ids = runner.prompt_ids(prefix + _example(line, False), args.gsm8k_think)
        if len(ids) + args.gsm8k_max_new + 1 > args.capacity:
            raise SystemExit(
                f"--capacity {args.capacity} < prompt {len(ids)} + --gsm8k-max-new {args.gsm8k_max_new}"
            )
        gen, prefill_s, decode_s = runner.generate(ids)
        text = tokenizer.decode(gen)
        got, want = answer_value(text), answer_value(line["answer"])
        rec = dict(
            tag=args.tag,
            gsm8k=True,
            k=k,
            index=index,
            prompt_tokens=len(ids),
            new_tokens=len(gen),
            prefill_s=prefill_s,
            decode_s=decode_s,
            extracted=got,
            answer=want,
            correct=got == want,
            response=text,
        )
        records.append(rec)
        emit(rec)
    correct = sum(r["correct"] for r in records)
    summary = dict(
        tag=args.tag,
        gsm8k=True,
        summary=True,
        n=len(records),
        correct=correct,
        accuracy=correct / max(len(records), 1),
        invalid=sum(r["extracted"] == INVALID for r in records),
        seed=args.gsm8k_seed,
        data=args.gsm8k_data or DEFAULT_DATA,
        think=args.gsm8k_think,
        max_new=args.gsm8k_max_new,
        layers=args.layers,
        world=bench.mesh.devices.size,
        relax=args.relax,
        kernel_opts={k: repr(v) for k, v in bench.kernel_kw.items()},
        seconds=time.time() - t_start,
    )
    emit(summary)
    return summary
