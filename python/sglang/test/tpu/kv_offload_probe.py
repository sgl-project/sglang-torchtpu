"""Prefix-caching + KV-offload gate: probe mechanics and its verdict.

Plugs into ``AccuracyTestParams.runner`` like the eval harnesses in
``eval_support``, but it is a feature gate rather than a score: it drives its own
workload and asserts which cache tier served each read, so it self-gates via
``_passed``.

Split three ways on purpose. ``plan_eviction`` and ``validate_probe`` are pure and
carry every judgement the gate makes, so the reasoning is readable and reviewable
apart from the transport. ``run_probe`` owns the HTTP and nothing else.
"""

import math
import re
from typing import Any, Dict, List, Optional

import msgspec
import requests

from sglang.test.test_utils import dump_metric

# Common English words (~1 token each), so a word count is a good enough token
# estimate without pulling a tokenizer into the runner. Same trick as
# test/registered/hicache/test_hicache_pallas_tpu.py.
_PROMPT_VOCAB = (
    "the of and to in a is that for it as was with on be by at this from but not are or "
    "his they she had which were we when your can said there use each their time will how "
    "other about many then them these so some her would make like him into has two more "
    "very after our just name good through where help much before line right too mean old"
).split()

# The probe prompt ends on this, and every read has to continue the count. Greedy
# decoding on these servers is not reproducible: on Qwen3-Coder, three flushed cold
# reads of the same 2048-token filler prompt returned three different texts,
# logprobs apart by up to 1.7, with no cache involved at all. Forcing the
# continuation narrows that but does not remove it -- across six reads of this tail
# the decode still split between continuing it (" 13 14 15 16 17 ") and restarting
# it ("\n\n1 2 3 4 5 6 7 8"), and it split on the DEVICE hit as often as on the host
# reload, which no host round-trip can explain. So the gate asserts the count, not
# the text: which of those two a read picks is noise, whereas KV that came back
# wrong cannot produce either. See ``_counting_failure``.
_FORCED_TAIL = (
    "\n\nIgnore the words above. Continue this sequence.\n" "1 2 3 4 5 6 7 8 9 10 11 12"
)
# Rough, like the word-per-token estimate above, and only used to keep the prompt
# near the requested size. Nothing asserts an exact token count.
_FORCED_TAIL_TOKENS = 48

# Filler KV as a multiple of the device pool. Above 1 so the probe is provably the
# LRU victim rather than merely likely to be.
_EVICTION_MARGIN = 1.5
# Enough decode steps to land a countable run whichever way the tail is picked up,
# few enough that the probe stays cheap next to the accuracy gate sharing the boot.
_MAX_NEW_TOKENS = 16
_INTEGER = re.compile(r"\d+")
# Integers a read must land, past the first. Only a floor against a degenerate
# decode -- the ascending property in ``_counting_failure`` is what actually
# catches bad KV, and it catches it at any length.
_MIN_COUNT_RUN = 3
# Generous: the first probe is a cold prefill on a freshly compiled server.
_REQUEST_TIMEOUT = 900


class CacheObservation(msgspec.Struct, frozen=True):
    """One /generate round trip, with the tier that served its prefix."""

    text: str
    device_tokens: int
    host_tokens: int


class KvOffloadProbe(msgspec.Struct, frozen=True):
    """The three reads of the same prompt the gate reasons over.

    ``cold`` is the first prefill on an empty cache, ``warm`` the immediate re-send
    that must hit device, and ``reload`` the re-send after fillers evicted it, which
    must come back from host.
    """

    cold: CacheObservation
    warm: CacheObservation
    reload: CacheObservation


class EvictionPlan(msgspec.Struct, frozen=True):
    """How much filler it takes to push the probe off device but not off host."""

    num_fillers: int
    host_pool_tokens: int
    backed_up_tokens: int


def plan_eviction(
    *, device_pool_tokens: int, probe_tokens: int, hicache_ratio: float
) -> EvictionPlan:
    """Size the filler run, or refuse a config that cannot prove anything.

    The window is real but not wide: too little filler and the probe is still on
    device, so nothing is offloaded. Too much and it is evicted from the host pool
    as well, and the reload silently becomes a recompute that still looks like a
    pass. Both ends are arithmetic, so both are settled here rather than on the
    hardware.
    """
    num_fillers = math.ceil(_EVICTION_MARGIN * device_pool_tokens / probe_tokens)
    host_pool_tokens = int(device_pool_tokens * hicache_ratio)
    backed_up_tokens = probe_tokens * (num_fillers + 1)
    if host_pool_tokens < backed_up_tokens:
        raise ValueError(
            f"host pool holds {host_pool_tokens} tokens but the probe plus "
            f"{num_fillers} fillers back up {backed_up_tokens}, so the probe would be "
            f"evicted from host as well. Raise hicache_ratio above "
            f"{backed_up_tokens / device_pool_tokens:.1f} or shrink the pool."
        )
    return EvictionPlan(
        num_fillers=num_fillers,
        host_pool_tokens=host_pool_tokens,
        backed_up_tokens=backed_up_tokens,
    )


def _counting_failure(*, label: str, text: str) -> Optional[str]:
    """Reject a decode that did not come back as a readable ascending count.

    The first integer is dropped: it is routinely a fragment of a number the
    prompt already opened, so a read can legitimately start on "0 11 12 13 ...".
    Everything after it has to step by one, which is the part corrupted KV cannot
    fake -- a single bad token breaks the run (" 13 1<garbage> 15 16 17 ") even
    though the model recovers right after it, and KV that is wholly wrong decodes
    to no integers at all.
    """
    numbers = [int(match) for match in _INTEGER.findall(text)][1:]
    counted = len(numbers) >= _MIN_COUNT_RUN and all(
        later == earlier + 1 for earlier, later in zip(numbers, numbers[1:])
    )
    if counted:
        return None
    return (
        f"the {label} read did not continue the forced count, so the KV it "
        f"attended over is wrong: {text!r}"
    )


def validate_probe(
    *, probe: KvOffloadProbe, page_size: int, plan: Optional[EvictionPlan] = None
) -> List[str]:
    """Every judgement the gate makes. Empty list means the feature is working.

    The reload is pinned to the whole prefix, not to "some of it". A radix match is
    floored to a page multiple, so a partial split like one page from host and the
    rest still on device is a reachable state, and it means the fillers did not
    actually evict the probe. Accepting it would let the gate pass on a server that
    offloaded a few percent of the KV.

    ``plan`` is optional and only sharpens the message when the reload comes back
    short, since an undersized host pool and a broken raiden transfer look identical
    from the request side.
    """
    failures: List[str] = []

    if probe.cold.device_tokens or probe.cold.host_tokens:
        failures.append(
            f"cold probe hit the cache ({probe.cold.device_tokens} device, "
            f"{probe.cold.host_tokens} host tokens) -- /flush_cache did not empty "
            "the tree"
        )

    # A prefix match is floored to a page multiple, so one page is the smallest hit
    # that proves reuse rather than an off-by-one in the accounting.
    if probe.warm.device_tokens < page_size:
        failures.append(
            f"no device cache hit on the immediate re-send: "
            f"{probe.warm.device_tokens} cached tokens, expected at least one page "
            f"({page_size})"
        )
    if probe.warm.host_tokens:
        failures.append(
            f"the resident re-send reloaded {probe.warm.host_tokens} tokens from "
            "host, and it should have been served from device"
        )

    if probe.reload.device_tokens:
        failures.append(
            f"the probe was not fully evicted: {probe.reload.device_tokens} tokens "
            "still matched on device after the fillers, so this read did not have to "
            "come from host and the gate proves nothing about offloading"
        )
    if probe.reload.host_tokens != probe.warm.device_tokens:
        hint = ""
        if plan is not None and probe.reload.host_tokens < probe.warm.device_tokens:
            hint = (
                f" The host pool holds {plan.host_pool_tokens} tokens against "
                f"{plan.backed_up_tokens} backed up by the probe plus "
                f"{plan.num_fillers} fillers, so check that sizing before blaming "
                "the transfer."
            )
        failures.append(
            f"incomplete host reload: {probe.reload.host_tokens} host cached tokens "
            f"against the {probe.warm.device_tokens} the device hit had cached, so "
            f"the offloaded copy did not survive intact.{hint}"
        )

    # Every tier, not just the reload: the same check on the cold prefill and the
    # device hit is what makes a host-reload failure mean the host, since a bug that
    # breaks all three is not about offloading.
    for label, observation in (
        ("cold", probe.cold),
        ("device-hit", probe.warm),
        ("host-reload", probe.reload),
    ):
        counting_failure = _counting_failure(label=label, text=observation.text)
        if counting_failure is not None:
            failures.append(counting_failure)

    return failures


def synthetic_prompt(*, index: int, approx_tokens: int) -> str:
    """A deterministic prompt sharing no radix prefix with any other ``index``."""
    seed = (index + 1) * 2654435761 & 0xFFFFFFFF
    words = []
    for _ in range(approx_tokens):
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(_PROMPT_VOCAB[seed % len(_PROMPT_VOCAB)])
    return f"Document {index} header alpha{index} beta{index * 7}. " + " ".join(words)


def probe_prompt(*, index: int, approx_tokens: int) -> str:
    """The prompt whose decode is checked: filler, then a forced continuation.

    Only the probe needs this. The fillers are never decoded past one token and
    never checked, so they stay plain ``synthetic_prompt``.
    """
    return (
        synthetic_prompt(
            index=index, approx_tokens=max(1, approx_tokens - _FORCED_TAIL_TOKENS)
        )
        + _FORCED_TAIL
    )


def generate_with_cache_report(
    *, base_url: str, prompt: str, max_new_tokens: int
) -> CacheObservation:
    """Generate greedily and read back which tier served the prefix.

    /generate always reports ``cached_tokens``, and ``cached_tokens_details`` only
    once some tier is non-zero: the scheduler returns None for the all-zero case
    rather than a dict of zeros. So a missing breakdown is a genuine zero here, and
    the cold read relies on that. What must not pass silently is the contradiction
    -- a non-zero total with no breakdown to attribute it to -- so the two are told
    apart by cross-checking the total rather than by the field's presence.

    Note --enable-cache-report does not gate any of this. It gates the OpenAI usage
    block, which is the path GSM8K takes, not /generate.
    """
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
        },
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    output = response.json()
    if isinstance(output, list):
        output = output[0]
    meta_info = output["meta_info"]
    if "cached_tokens" not in meta_info:
        raise RuntimeError(
            "the /generate response carried no cached_tokens, so no cache accounting "
            f"can be read off it: {meta_info!r}"
        )
    details = meta_info.get("cached_tokens_details")
    if details is None:
        if meta_info["cached_tokens"]:
            raise RuntimeError(
                f"the server reported {meta_info['cached_tokens']} cached tokens with "
                "no cached_tokens_details to break them down, so the tier that served "
                "them cannot be identified"
            )
        details = {}
    return CacheObservation(
        text=output["text"],
        device_tokens=details.get("device", 0),
        host_tokens=details.get("host", 0),
    )


def run_probe(
    *, base_url: str, test_name: str, probe_tokens: int, plan: EvictionPlan
) -> KvOffloadProbe:
    """Flush, read the probe cold, warm, and again once the fillers evicted it."""
    prompt = probe_prompt(index=0, approx_tokens=probe_tokens)

    def generate(text: str, max_new_tokens: int = _MAX_NEW_TOKENS) -> CacheObservation:
        return generate_with_cache_report(
            base_url=base_url, prompt=text, max_new_tokens=max_new_tokens
        )

    flush = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
    flush.raise_for_status()

    cold = generate(prompt)
    warm = generate(prompt)
    print(
        f"[kv-offload] {test_name}: cold=({cold.device_tokens},{cold.host_tokens}) "
        f"warm=({warm.device_tokens},{warm.host_tokens}) device/host cached tokens",
        flush=True,
    )

    for i in range(plan.num_fillers):
        generate(
            synthetic_prompt(index=100 + i, approx_tokens=probe_tokens),
            max_new_tokens=1,
        )

    reload = generate(prompt)
    print(
        f"[kv-offload] {test_name}: after {plan.num_fillers} fillers "
        f"reload=({reload.device_tokens},{reload.host_tokens}) "
        "device/host cached tokens",
        flush=True,
    )
    return KvOffloadProbe(cold=cold, warm=warm, reload=reload)


def kv_offload_runner(
    *,
    test_name: str,
    device_pool_tokens: int,
    page_size: int,
    probe_tokens: int,
    hicache_ratio: float,
):
    """Build an ``AccuracyTestParams.runner`` gating prefix caching + KV offload.

    Sequence on one server whose device pool is small enough to overflow:

    1. flush -- cold radix tree and cold host pool (HiRadixCache.reset clears both)
    2. probe -- cold prefill, written through to host
    3. probe -- DEVICE hit, the prefix is still resident
    4. fillers -- overflow the device pool so the probe is evicted to host only
    5. probe -- HOST reload

    That covers the three things the feature owes: a cache hit (3), offloading
    observed (5), and KV that survived the host round-trip. The last is why the
    probe prompt ends on a forced count -- every read has to continue it, and KV
    that came back wrong cannot.

    It is deliberately not an equality check between reads. These servers do not
    decode reproducibly, and they diverge on the device hit as readily as on the
    host reload, so equality fails on runs where nothing was offloaded wrong at
    all. See ``_FORCED_TAIL`` and ``_counting_failure``.

    ``hicache_ratio`` is the server's, and is checked rather than used, so a config
    whose host pool cannot hold the probe plus every filler is refused here instead
    of passing vacuously on the hardware.
    """
    plan = plan_eviction(
        device_pool_tokens=device_pool_tokens,
        probe_tokens=probe_tokens,
        hicache_ratio=hicache_ratio,
    )

    def run(base_url: str) -> Dict[str, Any]:
        probe = run_probe(
            base_url=base_url,
            test_name=test_name,
            probe_tokens=probe_tokens,
            plan=plan,
        )
        failures = validate_probe(probe=probe, page_size=page_size, plan=plan)

        # Built-in backends dump their own metrics, so a runner must do its own. Left
        # unplotted (no `workload`, no kind=accuracy): the pass/fail is the gate, and
        # these counts are the detail a reader needs when it goes red.
        labels = {"model": test_name, "kind": "kv_offload"}
        dump_metric(
            f"{test_name}.device_cached_tokens", probe.warm.device_tokens, labels=labels
        )
        dump_metric(
            f"{test_name}.host_cached_tokens", probe.reload.host_tokens, labels=labels
        )

        return {
            "score": probe.reload.host_tokens,
            "_passed": not failures,
            "_failures": failures,
        }

    return run
