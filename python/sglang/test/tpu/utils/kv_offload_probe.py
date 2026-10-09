"""Feature gate for prefix caching, KV offload, and optional speculation.

The runner checks a cold read, a device-cache hit, and a host reload. Optional
speculation checks use counters returned by the same three requests. The optional
decode-tail check replays generated KV from device and from host.
"""

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import msgspec
import requests

from sglang.test.test_utils import dump_metric

# Common English words (~1 token each), so a word count is a good enough token
# estimate without pulling a tokenizer into the runner. Same trick as
# test/registered/e2e/hicache/hicache_pallas_tpu_harness.py.
_PROMPT_VOCAB = (
    "the of and to in a is that for it as was with on be by at this from but not are or "
    "his they she had which were we when your can said there use each their time will how "
    "other about many then them these so some her would make like him into has two more "
    "very after our just name good through where help much before line right too mean old"
).split()

# The probe prompt ends on this so a read has something short and definite to
# decode. Its exact text is never asserted -- the gate compares the device hit
# against the host reload (``validate_probe``), which holds whatever the model
# happens to say. Greedy decoding here sits on near-ties, and that is what makes
# the comparison sharp: a reloaded page that came back even slightly wrong flips
# a token and the two reads stop matching.
_FORCED_TAIL = (
    "\n\nIgnore the words above. Continue this sequence.\n1 2 3 4 5 6 7 8 9 10 11 12"
)
# Rough, like the word-per-token estimate above, and only used to keep the prompt
# near the requested size. Nothing asserts an exact token count.
_FORCED_TAIL_TOKENS = 48

# Filler KV as a multiple of the device pool. Above 1 so the probe is provably the
# LRU victim rather than merely likely to be.
_EVICTION_MARGIN = 1.5
# Enough decode steps that the two cached reads have real text to differ in, few
# enough that the probe stays cheap next to the accuracy gate sharing the boot.
_MAX_NEW_TOKENS = 16
# Generous: the first probe is a cold prefill on a freshly compiled server.
_REQUEST_TIMEOUT = 900

# Under overlap scheduling, a request that finishes is still in the next forward's
# batch, which writes the KV of its last token. When prompt plus output is a page
# multiple, that slot completes a page that release inserts into the radix tree and
# write-through backs up to host while the forward may still be writing it. Every
# tail request is sized that way; whether a given backup reads the page early is
# timing, so several requests each get a chance.
_TAIL_REQUESTS = 8
# A prefix match is capped one token short of the input and floored to a page, so
# a replay of only the committed tokens would recompute the tail page instead of
# reading it.
_TAIL_REPLAY_EXTRA_TOKENS = 2
# Tokens each replay generates; their ids and logprobs are what the two replays compare.
_TAIL_REPLAY_NEW_TOKENS = 4


class CacheObservation(msgspec.Struct, frozen=True):
    """Response text, cache-tier counts, and speculative decoding counters."""

    text: str
    device_tokens: int
    host_tokens: int
    completion_tokens: int
    spec_verify_ct: int
    output_ids: Tuple[int, ...] = ()
    output_logprobs: Tuple[float, ...] = ()


class TailReplay(msgspec.Struct, frozen=True):
    """One decode tail's committed tokens, replayed from device and from host."""

    committed_tokens: int
    device: CacheObservation
    host: CacheObservation


class KvOffloadProbe(msgspec.Struct, frozen=True):
    """The three reads of the same prompt the gate reasons over.

    ``cold`` is the first prefill on an empty cache, ``warm`` the immediate re-send
    that must hit device, and ``reload`` the re-send after fillers evicted it, which
    must come back from host.
    """

    cold: CacheObservation
    warm: CacheObservation
    reload: CacheObservation
    tails: Tuple[TailReplay, ...] = ()


class EvictionPlan(msgspec.Struct, frozen=True):
    """How much filler it takes to push the probe off device but not off host."""

    num_fillers: int
    host_pool_tokens: int
    backed_up_tokens: int


def plan_eviction(
    *,
    device_pool_tokens: int,
    probe_tokens: int,
    hicache_ratio: float,
    tail_tokens: int = 0,
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
    backed_up_tokens = probe_tokens * (num_fillers + 1) + tail_tokens
    if host_pool_tokens < backed_up_tokens:
        raise ValueError(
            f"host pool holds {host_pool_tokens} tokens but the probe, decode tails "
            f"and {num_fillers} fillers back up {backed_up_tokens}, so the oldest "
            f"of them would be evicted from host as well. Raise hicache_ratio above "
            f"{backed_up_tokens / device_pool_tokens:.1f} or shrink the pool."
        )
    return EvictionPlan(
        num_fillers=num_fillers,
        host_pool_tokens=host_pool_tokens,
        backed_up_tokens=backed_up_tokens,
    )


def _labelled_reads(probe: KvOffloadProbe):
    """The three reads, in the order the gate reports them."""
    return (
        ("cold", probe.cold),
        ("device-hit", probe.warm),
        ("host-reload", probe.reload),
    )


def validate_probe(
    *,
    probe: KvOffloadProbe,
    page_size: int,
    plan: Optional[EvictionPlan] = None,
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
                f"{plan.backed_up_tokens} backed up by the probe, decode tails and "
                f"{plan.num_fillers} fillers, so check that sizing before blaming "
                "the transfer."
            )
        failures.append(
            f"incomplete host reload: {probe.reload.host_tokens} host cached tokens "
            f"against the {probe.warm.device_tokens} the device hit had cached, so "
            f"the offloaded copy did not survive intact.{hint}"
        )

    # The two cached reads run the same extend shape over the same stored prefix,
    # so they must decode identically; a difference is the reloaded KV itself. The
    # cold read is excluded on purpose -- it consumes the prefix at compute dtype
    # rather than KV-cache dtype, so it differs from both at epsilon scale.
    if not probe.warm.text or not probe.reload.text:
        failures.append("the device-hit and host-reload reads must be non-empty")
    elif probe.warm.text != probe.reload.text:
        failures.append(
            "the host reload changed the device-cached output: "
            f"device={probe.warm.text!r}, host={probe.reload.text!r}"
        )

    return failures


def mean_accept_length(*, probe: KvOffloadProbe) -> float:
    """Return completion tokens per verify forward, or 0 when none ran."""
    verifies = sum(read.spec_verify_ct for _, read in _labelled_reads(probe))
    if not verifies:
        return 0.0
    return sum(read.completion_tokens for _, read in _labelled_reads(probe)) / verifies


def spec_failures(*, probe: KvOffloadProbe) -> List[str]:
    """Require verify forwards on every read and equal cost for the two cached reads."""
    failures = [
        f"speculative decoding did not run on the {label} read: spec_verify_ct=0 "
        f"over {read.completion_tokens} completion tokens"
        for label, read in _labelled_reads(probe)
        if read.spec_verify_ct <= 0
    ]
    if failures:
        return failures

    # The two cached reads decode identical text over the same stored prefix, so
    # they must cost the same verify forwards; a difference is the reloaded KV
    # perturbing the draft or verify path. The cold read is excluded for the same
    # reason the text comparison excludes it. How many forwards the pair takes is
    # a property of the decoded text, not of offloading -- the acceptance rate is
    # guarded on the GSM8K sample instead.
    if probe.reload.spec_verify_ct != probe.warm.spec_verify_ct:
        failures.append(
            f"the host reload changed speculative decoding: "
            f"{probe.reload.spec_verify_ct} verify forwards against the device "
            f"hit's {probe.warm.spec_verify_ct} for the same decoded text"
        )
    return failures


def tail_shapes(*, page_size: int) -> List[Tuple[int, int]]:
    """``(prompt_tokens, max_new_tokens)`` for each decode-tail request.

    Each pair sums to a page multiple. The output lengths differ, so the requests
    finish on different decode steps and their backups are separate exposures.
    """
    # Keeps a replay from completing a page of its own, which would add tree and
    # host state the eviction plan does not count.
    if page_size <= _TAIL_REPLAY_EXTRA_TOKENS + _TAIL_REPLAY_NEW_TOKENS:
        raise ValueError(
            f"page_size {page_size} is too small for the decode-tail replays"
        )
    shapes = []
    for i in range(_TAIL_REQUESTS):
        max_new_tokens = page_size // 2 + i
        shapes.append(((2 + i % 2) * page_size - max_new_tokens, max_new_tokens))
    return shapes


def tail_failures(*, tails: Sequence[TailReplay]) -> List[str]:
    """Require each tail's host replay to match its device replay exactly.

    Both replays run the same extend over the same committed tokens, so their
    logprobs agree bit for bit unless the host copy differs from the device KV.
    A fresh prefill is not a valid reference: it differs at epsilon scale.
    """
    failures = []
    for i, tail in enumerate(tails):
        committed = tail.committed_tokens
        device, host = tail.device, tail.host
        if device.device_tokens != committed or device.host_tokens:
            failures.append(
                f"decode tail {i}: the device replay matched {device.device_tokens} "
                f"device and {device.host_tokens} host tokens, expected all "
                f"{committed} committed tokens on device, so the tail page was not "
                "in the tree and nothing was checked (is overlap scheduling on?)"
            )
        elif host.device_tokens or host.host_tokens != committed:
            failures.append(
                f"decode tail {i}: the host replay matched {host.device_tokens} "
                f"device and {host.host_tokens} host tokens, expected all "
                f"{committed} committed tokens from host"
            )
        elif (
            not device.output_ids
            or host.output_ids != device.output_ids
            or host.output_logprobs != device.output_logprobs
        ):
            failures.append(
                f"decode tail {i}: reloading its {committed} committed tokens from "
                "host changed the replay: device "
                f"{list(zip(device.output_ids, device.output_logprobs))}, host "
                f"{list(zip(host.output_ids, host.output_logprobs))}"
            )
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
    *,
    base_url: str,
    prompt: Union[str, List[int]],
    max_new_tokens: int,
    routed_dp_rank: Optional[int] = None,
    ignore_eos: bool = False,
    return_logprob: bool = False,
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

    A list ``prompt`` is sent as token ids.
    """
    payload = {
        "input_ids" if isinstance(prompt, list) else "text": prompt,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": ignore_eos,
        },
    }
    if return_logprob:
        payload["return_logprob"] = True
    if routed_dp_rank is not None:
        payload["routed_dp_rank"] = routed_dp_rank
    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json=payload,
        timeout=_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    output = response.json()
    if isinstance(output, list):
        output = output[0]
    meta_info = output["meta_info"]
    if routed_dp_rank is not None and meta_info.get("dp_rank") != routed_dp_rank:
        raise RuntimeError(
            f"request routed to DP rank {routed_dp_rank}, but the response reports "
            f"dp_rank={meta_info.get('dp_rank')!r}"
        )
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
    # Each entry is (logprob, token_id, token_text).
    token_logprobs = meta_info.get("output_token_logprobs") or []
    return CacheObservation(
        text=output["text"],
        device_tokens=details.get("device", 0),
        host_tokens=details.get("host", 0),
        completion_tokens=meta_info.get("completion_tokens", 0),
        # Missing and zero both mean speculative verification did not run.
        spec_verify_ct=meta_info.get("spec_verify_ct") or 0,
        output_ids=tuple(entry[1] for entry in token_logprobs),
        output_logprobs=tuple(entry[0] for entry in token_logprobs),
    )


def exact_prompt_ids(*, base_url: str, index: int, num_tokens: int) -> List[int]:
    """Exactly ``num_tokens`` token ids of ``synthetic_prompt(index=index)``."""
    response = requests.post(
        base_url.rstrip("/") + "/tokenize",
        json={"prompt": synthetic_prompt(index=index, approx_tokens=2 * num_tokens)},
        timeout=60,
    )
    response.raise_for_status()
    tokens = response.json()["tokens"]
    if len(tokens) < num_tokens:
        raise RuntimeError(
            f"synthetic prompt {index} tokenized to {len(tokens)} tokens, fewer "
            f"than the {num_tokens} needed"
        )
    return tokens[:num_tokens]


def run_probe(
    *,
    base_url: str,
    test_name: str,
    probe_tokens: int,
    plan: EvictionPlan,
    routed_dp_rank: Optional[int] = None,
    flush_cache: bool = True,
    tail_sizes: Sequence[Tuple[int, int]] = (),
) -> KvOffloadProbe:
    """Flush, read the probe cold, warm, and again once the fillers evicted it.

    With ``tail_sizes``, decode-tail requests run concurrently after the warm read,
    each replayed from device before the fillers and from host after them.
    """
    prompt = probe_prompt(index=0, approx_tokens=probe_tokens)

    def generate(
        text: Union[str, List[int]],
        max_new_tokens: int = _MAX_NEW_TOKENS,
        **kwargs,
    ) -> CacheObservation:
        return generate_with_cache_report(
            base_url=base_url,
            prompt=text,
            max_new_tokens=max_new_tokens,
            routed_dp_rank=routed_dp_rank,
            **kwargs,
        )

    def first_turn(i: int, ids: List[int]) -> List[int]:
        max_new_tokens = tail_sizes[i][1]
        out = generate(ids, max_new_tokens, ignore_eos=True, return_logprob=True)
        if len(out.output_ids) != max_new_tokens:
            raise RuntimeError(
                f"decode tail {i} generated {len(out.output_ids)} tokens, expected "
                f"{max_new_tokens}"
            )
        return ids + list(out.output_ids)

    def replay(committed: List[int]) -> CacheObservation:
        return generate(
            committed + committed[:_TAIL_REPLAY_EXTRA_TOKENS],
            _TAIL_REPLAY_NEW_TOKENS,
            ignore_eos=True,
            return_logprob=True,
        )

    if flush_cache:
        flush = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
        flush.raise_for_status()

    cold = generate(prompt)
    warm = generate(prompt)
    print(
        f"[kv-offload] {test_name}: cold=({cold.device_tokens},{cold.host_tokens}) "
        f"warm=({warm.device_tokens},{warm.host_tokens}) device/host cached tokens",
        flush=True,
    )

    committed_tails: List[List[int]] = []
    if tail_sizes:
        # Tokenized up front so the requests arrive together and finish on the
        # steps their output lengths set.
        prompts = [
            exact_prompt_ids(base_url=base_url, index=200 + i, num_tokens=n)
            for i, (n, _) in enumerate(tail_sizes)
        ]
        with ThreadPoolExecutor(max_workers=len(tail_sizes)) as pool:
            committed_tails = list(
                pool.map(first_turn, range(len(tail_sizes)), prompts)
            )
    device_replays = [replay(committed) for committed in committed_tails]

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
    tails = tuple(
        TailReplay(
            committed_tokens=len(committed), device=device, host=replay(committed)
        )
        for committed, device in zip(committed_tails, device_replays)
    )
    for i, tail in enumerate(tails):
        print(
            f"[kv-offload] {test_name}: decode tail {i} committed="
            f"{tail.committed_tokens} device=({tail.device.device_tokens},"
            f"{tail.device.host_tokens}) host=({tail.host.device_tokens},"
            f"{tail.host.host_tokens}) logprobs device="
            f"{list(tail.device.output_logprobs)} "
            f"host={list(tail.host.output_logprobs)}",
            flush=True,
        )
    return KvOffloadProbe(cold=cold, warm=warm, reload=reload, tails=tails)


def kv_offload_runner(
    *,
    test_name: str,
    device_pool_tokens: int,
    page_size: int,
    probe_tokens: int,
    hicache_ratio: float,
    spec_decoding: bool = False,
    dp_size: int = 1,
    decode_tails: bool = False,
):
    """Build a runner that validates device hits and complete host reloads.

    ``hicache_ratio`` must leave enough host capacity for the probe and fillers.
    ``spec_decoding`` additionally enables the speculative-decoding checks.
    ``decode_tails`` additionally checks that host copies of page-aligned decode
    KV match the device; it needs overlap scheduling.
    """
    if decode_tails and (spec_decoding or dp_size != 1):
        raise ValueError(
            "decode_tails has not been run with speculative decoding or DP "
            "attention; speculative decoding commits KV in verify steps, so a "
            "page-aligned output need not leave a page-aligned tail"
        )
    shapes = tail_shapes(page_size=page_size) if decode_tails else []
    plan = plan_eviction(
        device_pool_tokens=device_pool_tokens,
        probe_tokens=probe_tokens,
        hicache_ratio=hicache_ratio,
        tail_tokens=sum(map(sum, shapes)),
    )

    if dp_size < 1:
        raise ValueError(f"dp_size must be positive, got {dp_size}")

    def run(base_url: str) -> Dict[str, Any]:
        flush = requests.post(base_url.rstrip("/") + "/flush_cache", timeout=60)
        flush.raise_for_status()
        failures = []
        probes = []
        for dp_rank in range(dp_size):
            routed_dp_rank = dp_rank if dp_size > 1 else None
            probe = run_probe(
                base_url=base_url,
                test_name=f"{test_name}/dp{dp_rank}",
                probe_tokens=probe_tokens,
                plan=plan,
                routed_dp_rank=routed_dp_rank,
                flush_cache=False,
                tail_sizes=shapes,
            )
            probes.append(probe)
            failures += [
                f"dp_rank={dp_rank}: {failure}"
                for failure in validate_probe(
                    probe=probe,
                    page_size=page_size,
                    plan=plan,
                )
            ]
            failures += [
                f"dp_rank={dp_rank}: {failure}"
                for failure in tail_failures(tails=probe.tails)
            ]

            # kind=info: stored and shown in the point's details, charted on no
            # tab -- `probe` is the tag that keeps these filterable.
            labels = {
                "model": test_name,
                "kind": "info",
                "probe": "kv_offload",
                "dp_rank": str(dp_rank),
            }
            dump_metric(
                f"{test_name}.device_cached_tokens",
                probe.warm.device_tokens,
                labels=labels,
            )
            dump_metric(
                f"{test_name}.host_cached_tokens",
                probe.reload.host_tokens,
                labels=labels,
            )

            if spec_decoding:
                accept_length = mean_accept_length(probe=probe)
                verifies = ", ".join(
                    f"{label}={read.spec_verify_ct}"
                    for label, read in _labelled_reads(probe)
                )
                print(
                    f"[kv-offload] {test_name}/dp{dp_rank}: spec verify forwards "
                    f"{verifies}, mean accept length {accept_length:.2f}",
                    flush=True,
                )
                dump_metric(
                    f"{test_name}.spec_accept_length", accept_length, labels=labels
                )
                failures += [
                    f"dp_rank={dp_rank}: {failure}"
                    for failure in spec_failures(probe=probe)
                ]

        return {
            "score": sum(probe.reload.host_tokens for probe in probes),
            "_passed": not failures,
            "_failures": failures,
        }

    return run
