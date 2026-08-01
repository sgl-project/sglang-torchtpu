"""Generate a synthetic multi-turn agentic-coding trace for sglang's
``--dataset-name agentic-trace`` loader.

Why a generator instead of a captured trace: the benchmark contract fixes the
*shape* of the workload (a fixed number of new tokens per round, accumulating to
a target context length, with a configurable number of sessions that run out to a
longer target) and no public trace has that shape. Real OpenHands/SWE-smith traces have
uncontrolled, short, wildly varying turns, which makes run-to-run comparison and
capacity planning impossible.

Emitted shape (exactly what ``AgenticTraceDataset`` expects — see
``python/sglang/benchmark/datasets/agentic_trace.py``)::

    {
      "metadata": {...},                 # provenance + the resolved knobs
      "conversations": [
        [ {"messages": [{"role": "system", ...}, {"role": "user", ...}],
           "prompt_tokens": 4096},       # round 1: system + task
          {"messages": [{"role": "user", ...}], "prompt_tokens": 9216},
          ... ],                         # round k: one framed body
        ...
      ]
    }

Each round's ``messages`` holds only the *new* messages for that round;
``sglang.benchmark.serving`` replays a conversation round by round and feeds the
server's real assistant reply back into the next round's history. So the
server-side prompt grows by (turn_input_tokens + turn_output_tokens) per round
and every round after the first is a ~100%-prefix-hit against the round before
it, which is the point of the workload: it exercises prefix caching and (once
the working set exceeds HBM) KV offload the way an agentic coding client does.

Context growth, with I = --turn-input-tokens and O = --turn-output-tokens:

    prompt(round k) = k*I + (k-1)*O          (+ a few chat-template tokens)
    ctx after round k = k*(I + O)

so the round count for a target context is ``target / (I + O)``, rounded to the
nearest whole round (see ``_turns_for_context``).

Round bodies are vocabulary-sampled text, the same construction every other
sglang synthetic dataset uses (``datasets/common.py :: gen_prompt``), then trimmed
in token space so each round is exactly ``--turn-input-tokens`` (+/-
``--tolerance``) -- see the comment above ``_content_token_pool`` for why the
standard helper cannot be called directly. Content is unique per (session, round)
so that prefix reuse only ever happens *within* a session — cross-session sharing
would silently inflate the cache-hit rate.

Usage:
    python3 gen_agentic_workload.py --out /tmp/trace.json \
        --tokenizer Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 \
        --num-sessions 32 --num-long 2
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple

# Round bodies are synthetic text built the way every other sglang synthetic
# dataset builds it: sample token ids from the tokenizer's vocabulary and decode
# them (datasets/common.py :: gen_prompt / get_available_tokens). Reusing that
# keeps this generator's content indistinguishable in kind from `random`,
# `random-ids` and `generated-shared-prefix`, so nothing about the token stream is
# special to this benchmark.
#
# Two things gen_prompt() does not handle, which an accumulating multi-turn chat
# workload needs:
#
#  1. EXACT token counts. gen_prompt(tok, 4096) samples 4096 ids and decodes, but
#     re-encoding that text yields 4326-4442 (+5.6..8.4% measured on the Qwen
#     tokenizer) because decode->encode is not the identity for an arbitrary id
#     sequence. A fixed-ISL benchmark absorbs that as a one-off ISL error; here it
#     compounds, so a 25-round session would overshoot its context target by ~7%
#     unevenly per round and the round-count arithmetic would be wrong. So the
#     sampled text is verified and trimmed in token space until it is exact.
#
#  2. SPECIAL TOKENS. get_available_tokens() filters only on isinstance(id, int),
#     so all 9 of the Qwen tokenizer's special ids stay in the pool -- unlike its
#     sibling get_available_multimodal_text_tokens(), which excludes them.
#     Measured over 2M draws: 57 special tokens land in content, 8 of them
#     <|im_end|>; extrapolated to a full sweep (~48M tokens) that is ~1,368
#     special tokens, ~192 of them <|im_end|>, each corrupting a chat-template
#     turn boundary. So the pool is filtered here.


def _content_token_pool(tokenizer) -> List[int]:
    """The vocabulary sglang samples synthetic text from, minus the special ids,
    in a canonical order.

    SORTED is load-bearing, not tidiness. get_available_tokens() iterates
    ``tokenizer.get_vocab().values()``, and a fast tokenizer builds that dict from a
    Rust HashMap whose iteration order differs per process -- verified: two runs on
    the same tokenizer start [197945, 202781, ...] and [5860, 203577, ...]. Seeded
    sampling over an unordered pool is therefore NOT reproducible, which would break
    the trace cache: bench_leg.sh keys a cache entry on (shape, seed) and reuses it
    across runs, so two runs with identical keys must produce identical bytes.
    (generated-shared-prefix has the same exposure and only avoids it by pickling
    its first result.)
    """
    try:
        from sglang.benchmark.datasets.common import get_available_tokens
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "gen_agentic_workload.py reuses sglang's synthetic-text vocabulary "
            f"(sglang.benchmark.datasets.common); run it with the sglang "
            f"interpreter. ({exc})"
        )
    specials = set(tokenizer.all_special_ids or [])
    pool = sorted(t for t in get_available_tokens(tokenizer) if t not in specials)
    if not pool:
        raise SystemExit("tokenizer vocabulary is empty after removing special ids")
    return pool


def _build_text_of_length(
    tokenizer,
    rng: random.Random,
    target_tokens: int,
    pool: List[int],
    tolerance: int,
) -> Tuple[str, int]:
    """Vocabulary-sampled text that encodes to exactly `target_tokens` (+/-
    `tolerance`); returns (text, actual_token_count).

    Same construction as gen_prompt() -- sample ids, decode -- then converge on the
    exact count. Trimming is done in token space (encode -> slice -> decode) since a
    character-level cut lands mid-token; because that round trip can itself drift,
    the measure/adjust step repeats, bounded by a few passes.
    """
    text = tokenizer.decode(rng.choices(pool, k=target_tokens))
    for _ in range(6):
        ids = tokenizer.encode(text, add_special_tokens=False)
        delta = len(ids) - target_tokens
        if abs(delta) <= tolerance:
            return text, len(ids)
        if delta > 0:
            text = tokenizer.decode(ids[:target_tokens])
        else:
            text += tokenizer.decode(rng.choices(pool, k=-delta))
    return text, len(tokenizer.encode(text, add_special_tokens=False))


_SYSTEM_PROMPT = """You are a coding agent working inside a large Python repository.
You have these tools: read_file(path, start, end), write_file(path, content),
run_tests(selector), grep(pattern, path), apply_patch(diff), git_status().
Work in small steps: inspect the relevant code, form a hypothesis, make one
minimal change, run the narrowest test that would catch a regression, then
report. Always cite file paths and line numbers. Never invent APIs; read the
code first. Keep each reply under 400 words unless asked to produce a patch."""


def _measure_template_overhead(tokenizer) -> Tuple[int, int]:
    """Chat-template token cost, as (first_round, per_extra_round).

    The server tokenizes the *rendered* chat prompt, not the bare message text,
    so the round-trip markers (``<|im_start|>role\\n``, ``<|im_end|>``, the
    generation prompt) are real context the request has to fit inside
    ``--context-length``. Measuring beats hardcoding: it differs per model
    family (a thinking model's generation prompt is longer). Returns (0, 0) if
    the tokenizer has no chat template.
    """

    def n(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    def render(messages: List[Dict[str, str]]) -> int:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False))

    try:
        one = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
        two = one + [
            {"role": "assistant", "content": "A"},
            {"role": "user", "content": "U"},
        ]
        first = render(one) - n("S") - n("U")
        per_round = (render(two) - n("S") - 2 * n("U") - n("A")) - first
        return first, per_round
    except Exception as exc:  # no template / non-standard template
        print(
            f"[gen] WARN: could not measure chat-template overhead ({exc}); "
            "prompt_tokens estimates will omit it",
            file=sys.stderr,
        )
        return 0, 0


def _build_session(
    tokenizer,
    seed: int,
    session: int,
    num_turns: int,
    turn_input_tokens: int,
    turn_output_tokens: int,
    template_overhead: Tuple[int, int],
    pool: List[int],
    tolerance: int,
) -> List[Dict]:
    """One conversation: round 1 carries the system prompt + task, later rounds
    carry one framed body each. Round 1's body is shortened by the system prompt
    so every round adds the same `turn_input_tokens` to the server-side prompt."""
    rng = random.Random(seed * 1_000_003 + session)
    system_tokens = len(tokenizer.encode(_SYSTEM_PROMPT, add_special_tokens=False))
    first_overhead, per_round_overhead = template_overhead

    turns: List[Dict] = []
    cumulative = first_overhead
    for turn in range(num_turns):
        if turn > 0:
            # Everything the previous round added to the server-side history:
            # the model's reply plus this round's share of template markers.
            cumulative += turn_output_tokens + per_round_overhead
        # Round 1 also carries the system prompt, and every round carries a
        # framing sentence. Both are charged against the round's token budget so
        # that each round adds exactly `turn_input_tokens` of new input.
        if turn == 0:
            framing = (
                f"Task s{session}: a regression was reported in the TPU serving "
                f"path. Diagnose it from the evidence below and propose a minimal "
                f"fix.\n\n"
            )
        else:
            framing = f"Observation for step {turn}. Continue.\n\n"
        budget = turn_input_tokens - len(
            tokenizer.encode(framing, add_special_tokens=False)
        )
        if turn == 0:
            budget -= system_tokens
        # Reseed per (session, round) so every round's body is distinct and the
        # trace stays reproducible from --seed. Content must never repeat across
        # sessions, or prefix reuse would leak between them and inflate the
        # cache-hit rate.
        text, _ = _build_text_of_length(
            tokenizer,
            random.Random(hash((seed, session, turn))),
            budget,
            pool,
            tolerance,
        )
        messages = [{"role": "user", "content": framing + text}]
        if turn == 0:
            messages.insert(0, {"role": "system", "content": _SYSTEM_PROMPT})
        # Measure the assembled messages rather than reusing `actual`: the
        # framing prose ("Observation for step N...") is a couple dozen tokens
        # the body budget does not include.
        cumulative += sum(
            len(tokenizer.encode(m["content"], add_special_tokens=False))
            for m in messages
        )
        # An *estimate* of the prompt the server will see at this round
        # (content + chat-template markers + prior replies). The replay path
        # ignores it, but it makes the trace self-describing and lets a
        # context-length overflow be spotted before the run.
        turns.append({"messages": messages, "prompt_tokens": cumulative})
    return turns


def _turns_for_context(
    target_context: int, turn_input: int, turn_output: int, overhead_per_round: int
) -> int:
    """Rounds needed to grow the server-side context to `target_context`.

    Each round adds the new user tokens, the model's reply, and the round's chat
    template markers, so k = target / (I + O + overhead).

    Rounded to the NEAREST round, not truncated: a target is rarely an exact
    multiple of the per-round growth, and truncating costs up to a full round.
    That is a rounding error at a large target (128k: 25 vs 25.6 rounds) but a
    different workload at a small one -- an 8192 target truncates to ONE round,
    i.e. a 4.1k single-turn request with no prefix reuse at all, where nearest
    gives 2 rounds at ~9.2k. (The 16384 default is 3.19 rounds, so it lands on 3
    either way.) Overshoot is bounded by half a round; config.sh keeps that much
    headroom under the server's context length.
    """
    per_round = turn_input + turn_output + overhead_per_round
    return max(1, (target_context + per_round // 2) // per_round)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output trace JSON path")
    ap.add_argument(
        "--tokenizer",
        required=True,
        help="HF repo or local dir; must match the served model",
    )
    ap.add_argument(
        "--num-sessions",
        type=int,
        required=True,
        help="conversations in the trace (= bench --num-prompts)",
    )
    ap.add_argument(
        "--num-long",
        type=int,
        default=0,
        help="how many of the sessions run out to --long-target-context. "
        "These are emitted FIRST so that, with the bench client's "
        "concurrency semaphore, exactly this many long requests are "
        "in flight for the whole leg.",
    )
    ap.add_argument(
        "--turn-input-tokens",
        type=int,
        default=4096,
        help="new user tokens per round (default 4096)",
    )
    ap.add_argument(
        "--long-turn-input-tokens",
        type=int,
        default=0,
        help="new user tokens per round for the LONG sessions only "
        "(0 = same as --turn-input-tokens). Raising this is the "
        "only way to time-box a long session: at the 4k default it "
        "takes ~51 rounds to reach 256k. Doing so is a deliberate "
        "deviation from a uniform-round workload and is recorded "
        "in the trace metadata.",
    )
    ap.add_argument(
        "--turn-output-tokens",
        type=int,
        default=1024,
        help="assistant tokens per round; must match the bench's "
        "--sharegpt-output-len, since it sets how fast the "
        "context grows (default 1024)",
    )
    ap.add_argument(
        "--target-context",
        type=int,
        default=16384,
        help="context length a normal session accumulates to (default 16384, "
        "i.e. 3 rounds at the 4k/1k default)",
    )
    ap.add_argument(
        "--long-target-context",
        type=int,
        default=262144,
        help="context length a long session accumulates to",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="hard cap on rounds per normal session (0 = no cap). "
        "Use to time-box a run.",
    )
    ap.add_argument(
        "--long-max-turns",
        type=int,
        default=0,
        help="hard cap on rounds per long session (0 = no cap)",
    )
    ap.add_argument(
        "--tolerance",
        type=int,
        default=8,
        help="allowed per-round token drift (default 8)",
    )
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if args.num_long > args.num_sessions:
        print(
            f"ERROR: --num-long {args.num_long} > --num-sessions "
            f"{args.num_sessions}",
            file=sys.stderr,
        )
        return 2

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(
            "ERROR: transformers is required (pip install transformers)",
            file=sys.stderr,
        )
        return 2

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    pool = _content_token_pool(tokenizer)
    template_overhead = _measure_template_overhead(tokenizer)
    long_turn_input = args.long_turn_input_tokens or args.turn_input_tokens
    per_round = args.turn_input_tokens + args.turn_output_tokens + template_overhead[1]
    long_per_round = long_turn_input + args.turn_output_tokens + template_overhead[1]

    normal_turns = _turns_for_context(
        args.target_context,
        args.turn_input_tokens,
        args.turn_output_tokens,
        template_overhead[1],
    )
    long_turns = _turns_for_context(
        args.long_target_context,
        long_turn_input,
        args.turn_output_tokens,
        template_overhead[1],
    )
    if args.max_turns:
        normal_turns = min(normal_turns, args.max_turns)
    if args.long_max_turns:
        long_turns = min(long_turns, args.long_max_turns)

    print(f"[gen] tokenizer={args.tokenizer}")
    print(
        f"[gen] chat-template overhead: first_round={template_overhead[0]} "
        f"per_round={template_overhead[1]} -> {per_round} tok of context per round"
    )
    print(
        f"[gen] rounds/session: normal={normal_turns} x {per_round} tok "
        f"(-> ~{normal_turns * per_round} of {args.target_context}), "
        f"long={long_turns} x {long_per_round} tok "
        f"(-> ~{long_turns * long_per_round} of {args.long_target_context})"
    )
    # A target only a round or two deep cannot be hit closely -- the residual is
    # a large fraction of the session. State what the session actually reaches so
    # the requested target never stands in for it.
    for label, turns, growth, target in (
        ("normal", normal_turns, per_round, args.target_context),
        ("long", long_turns, long_per_round, args.long_target_context),
    ):
        achieved = turns * growth
        if abs(achieved - target) > target * 0.10:
            print(
                f"[gen] NOTE: a {label} session ends at ~{achieved} tok of "
                f"context, {(achieved - target) / target:+.0%} vs the requested "
                f"{target} -- the closest whole number of {growth} tok rounds. "
                "Recorded as-is in the trace metadata."
            )
    if long_turn_input != args.turn_input_tokens:
        print(
            f"[gen] NOTE: long sessions ramp at {long_turn_input} tok/round, not "
            f"{args.turn_input_tokens} -- a time-boxing deviation, recorded in "
            "the trace metadata"
        )
    print(f"[gen] sessions={args.num_sessions} of which long={args.num_long}")

    conversations: List[List[Dict]] = []
    for session in range(args.num_sessions):
        is_long = session < args.num_long
        turns = _build_session(
            tokenizer=tokenizer,
            seed=args.seed,
            session=session,
            num_turns=long_turns if is_long else normal_turns,
            turn_input_tokens=long_turn_input if is_long else args.turn_input_tokens,
            turn_output_tokens=args.turn_output_tokens,
            template_overhead=template_overhead,
            pool=pool,
            tolerance=args.tolerance,
        )
        conversations.append(turns)
        print(
            f"[gen]   session {session:>3} {'LONG' if is_long else 'norm'} "
            f"rounds={len(turns)} final_prompt_tokens={turns[-1]['prompt_tokens']}",
            flush=True,
        )

    record = {
        "metadata": {
            "generator": os.path.basename(__file__),
            "schema": "sglang-agentic-trace-v1",
            "tokenizer": args.tokenizer,
            "seed": args.seed,
            "num_sessions": args.num_sessions,
            "num_long_sessions": args.num_long,
            "turn_input_tokens": args.turn_input_tokens,
            "long_turn_input_tokens": long_turn_input,
            "long_ramp_is_time_boxed": long_turn_input != args.turn_input_tokens,
            "turn_output_tokens": args.turn_output_tokens,
            "target_context": args.target_context,
            "long_target_context": args.long_target_context,
            "rounds_per_normal_session": normal_turns,
            "rounds_per_long_session": long_turns,
            "chat_template_overhead_first_round": template_overhead[0],
            "chat_template_overhead_per_round": template_overhead[1],
            "context_tokens_per_round": per_round,
            "context_tokens_per_long_round": long_per_round,
            # The server must be launched with --context-length >= this, or the
            # last rounds of the longest session are rejected.
            "required_context_length": max(
                normal_turns * per_round, long_turns * long_per_round
            )
            + template_overhead[0],
            # What the bench client MUST pass, or the context growth assumed
            # here will not match what the server sees.
            "required_bench_args": {
                "--dataset-name": "agentic-trace",
                "--backend": "sglang-oai-chat",
                "--sharegpt-output-len": args.turn_output_tokens,
                "--num-prompts": args.num_sessions,
            },
            "peak_prompt_tokens": max(c[-1]["prompt_tokens"] for c in conversations),
            "total_rounds": sum(len(c) for c in conversations),
        },
        "conversations": conversations,
    }

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(record, f)
        f.write("\n")
    size_mb = os.path.getsize(out) / (1 << 20)
    print(
        f"[gen] wrote {out} ({size_mb:.1f} MiB, "
        f"{record['metadata']['total_rounds']} rounds total)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
