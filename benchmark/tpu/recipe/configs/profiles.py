"""Workload profiles: the shape a server is sized for, separate from the model.

A profile answers "what traffic is this serving"; a model's server.py answers
"how is this model served". Applying one touches only the pool and bucket
sizing -- never parallelism, quantization, the backend, or a runtime env.

    from benchmark.tpu.recipe.configs.profiles import PROFILES, apply_profile

    CONFIG = apply_profile(SERVER_CONFIG, PROFILES["long_8k1k"])
"""

from copy import deepcopy

# Every key a profile is allowed to set. Anything else belongs to the model.
PROFILE_KEYS = (
    "context_length",
    "max_running_requests",
    "torch_compile_max_bs",
    "tpu_compile_tokens",
    "tpu_compile_bs",
    "max_total_tokens",
)


def _profile(
    *,
    input_tokens: int,
    output_tokens: int,
    concurrency: int,
    prefill_tokens: int,
    decode_bs: tuple[int, ...] = (),
    context_length: int | None = None,
    max_total_tokens: int | None = None,
) -> dict:
    """One profile, with the sizing derived from the shape it names.

    The context is the shape plus a chat-template allowance; the pool holds every
    in-flight request at full length plus one prefill chunk of slack, which keeps
    a late arrival from waiting on an eviction. `decode_bs` adds decode buckets
    below the concurrency, for a server whose traffic is often a single request.
    `context_length` and `max_total_tokens` replace the derived values where a
    measured configuration pinned them.
    """
    if context_length is None:
        context_length = input_tokens + output_tokens + 64
    if max_total_tokens is None:
        max_total_tokens = concurrency * context_length + prefill_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "concurrency": concurrency,
        "prefill_tokens": prefill_tokens,
        "server_args": {
            "context_length": context_length,
            "max_running_requests": concurrency,
            "torch_compile_max_bs": concurrency,
            "tpu_compile_tokens": [prefill_tokens],
            "tpu_compile_bs": [*decode_bs, concurrency],
            "max_total_tokens": max_total_tokens,
        },
    }


PROFILES = {
    # Chat: short prompts, short answers, enough concurrency to keep the MoE
    # layers busy.
    "chat_1k1k": _profile(
        input_tokens=1024, output_tokens=1024, concurrency=64, prefill_tokens=2048
    ),
    # Long context at low concurrency, the latency-shaped point; and the same
    # shape at 64, the throughput-shaped one.
    "long_8k1k": _profile(
        input_tokens=8192, output_tokens=1024, concurrency=16, prefill_tokens=8192
    ),
    "long_8k1k_c64": _profile(
        input_tokens=8192, output_tokens=1024, concurrency=64, prefill_tokens=8192
    ),
    # Few requests on a model too large for more: a 1024-token prefill chunk and
    # a bs1 bucket next to the bs4 one, so a lone request decodes unpadded.
    "chat_1k1k_c4": _profile(
        input_tokens=1024,
        output_tokens=1024,
        concurrency=4,
        prefill_tokens=1024,
        decode_bs=(1,),
    ),
    "long_8k1k_c1": _profile(
        input_tokens=8192, output_tokens=1024, concurrency=1, prefill_tokens=1024
    ),
    "long_8k1k_c4": _profile(
        input_tokens=8192,
        output_tokens=1024,
        concurrency=4,
        prefill_tokens=1024,
        decode_bs=(1,),
    ),
    # Agentic: a long shared prefix re-sent each turn, so the pool holds whole
    # conversations and the radix cache does the work. One point, not the bench
    # matrix's agentic leg, which ramps toward a per-model target context and
    # sweeps concurrency (recipe/bench_matrix/config.sh).
    "agentic": _profile(
        input_tokens=32768, output_tokens=1024, concurrency=16, prefill_tokens=8192
    ),
    # 128K context up to 64 requests, as measured on Kimi-K3.1 (16 chips): the
    # context is exactly 128K + 8K, and the pool bound is a cap far above what
    # HBM holds (mem_fraction_static sizes it, ~532K tokens), not one full
    # context per request.
    "long_128k_c64": _profile(
        input_tokens=131072,
        output_tokens=8192,
        concurrency=64,
        prefill_tokens=4096,
        decode_bs=(1, 2, 4, 8, 16, 32),
        context_length=131072 + 8192,
        max_total_tokens=4000000,
    ),
}


def apply_profile(config: dict, profile: dict) -> dict:
    """A copy of `config` sized for `profile`. Neither input is modified."""
    sized = deepcopy(config)
    sized["server_args"].update(profile["server_args"])
    # The chunk follows the profile's prefill bucket: a prefill outside a
    # precompiled bucket takes the recompile path.
    sized["server_args"]["chunked_prefill_size"] = profile["prefill_tokens"]
    sized["server_args"]["max_prefill_tokens"] = profile["prefill_tokens"]
    return sized
