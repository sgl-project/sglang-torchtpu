"""
Minimal HTTP load balancer for prefill and decode servers for testing.
"""

import asyncio
import contextlib
import functools
import ipaddress
import logging
import math
import random
import urllib
import warnings
from http import HTTPStatus
from itertools import chain
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import orjson
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from sglang_router.mini_lb_policy import (
    EngineOverloaded,
    LoadPoller,
    SessionAffinityPolicy,
)
from sglang_router.router_args import MINI_LB_POLICY_CHOICES, RouterArgs

logger = logging.getLogger(__name__)

_CONNECT_TIMEOUT_SECS = 10.0

AIOHTTP_STREAM_READ_CHUNK_SIZE = (
    1024 * 64
)  # 64KB, to prevent aiohttp's "Chunk too big" error

# A server's /health and /health_generate run a one-token generation by default
# and give up after SGLANG_HEALTH_CHECK_TIMEOUT (20 s by default) with a 503;
# wait a little longer so the server's own answer arrives.
HEALTH_CHECK_TIMEOUT_SECS = 30

# The background health probe that decides which backends receive traffic
# always GETs this path with HEALTH_CHECK_TIMEOUT_SECS. It ignores
# --health-check-endpoint and --health-check-timeout-secs: their defaults
# (/health, 5 s) suit the Rust gateway, but on a TPU engine /health can skip
# the generation (SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION=0) and a 5 s timeout
# would mark a server unhealthy before its own 20 s check answers.
HEALTH_PROBE_PATH = "/health_generate"

# How long a completed stream waits, in the background, for the prefill to end
# its own response before the connection to it is closed anyway.
PREFILL_DRAIN_TIMEOUT_SECS = 60

# Strong references to fire-and-forget tasks (asyncio keeps only weak ones).
_background_tasks: set = set()


def _spawn_background(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


async def _read_to_eof(response) -> None:
    async for _ in response.content.iter_chunked(AIOHTTP_STREAM_READ_CHUNK_SIZE):
        pass


def _stream_error_event(message: str) -> bytes:
    """An SSE error event in the shape sglang's /generate stream uses. A
    broken upstream is transient (a retry is routed to a random, possibly
    healthy, decode), so it is marked retryable."""
    error = {
        "message": message,
        "type": "upstream_error",
        "code": HTTPStatus.BAD_GATEWAY.value,
        "retryable": True,
    }
    return b"data: " + orjson.dumps({"error": error}) + b"\n\n"


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


class _UpstreamConnectionError(aiohttp.ClientConnectionError):
    """A connection failure recorded in the affinity load table."""


def _health_poll_interval_secs(router_args: RouterArgs) -> Optional[float]:
    """Seconds between health probes, or None for --disable-health-check."""
    if router_args.disable_health_check:
        return None
    interval = router_args.health_check_interval_secs
    if not (math.isfinite(interval) and interval > 0):
        raise ValueError(
            "--health-check-interval-secs must be a finite number > 0, "
            f"got {interval!r} (use --disable-health-check to turn probes off)"
        )
    return interval


class _BackendHealth:
    """Which backends failed their last health probe.

    A backend that has not been probed since it was last (re)registered counts
    as healthy.
    """

    def __init__(self):
        self.unhealthy: Set[str] = set()
        # Roles routing past unhealthy backends because none is healthy.
        self.roles_failing_open: Set[str] = set()
        # Bumped when a backend is (re)registered or removed, so a probe that
        # started before that change can tell its result is stale.
        self._epochs: Dict[str, int] = {}

    def epoch(self, url: str) -> int:
        return self._epochs.get(url, 0)

    def forget(self, url: str) -> None:
        self.unhealthy.discard(url)
        self._epochs[url] = self.epoch(url) + 1


async def _probe_status(
    session: aiohttp.ClientSession, server: str, path: str
) -> Tuple[Optional[int], Optional[str]]:
    """GET `path` on `server`; return (status, None) or (None, error)."""
    try:
        async with session.get(f"{server}{path}") as response:
            return response.status, None
    except asyncio.TimeoutError:
        return None, f"no answer in {HEALTH_CHECK_TIMEOUT_SECS} s"
    except aiohttp.ClientError as e:
        return None, f"{type(e).__name__}: {e}"


class MiniLoadBalancer:
    """Debug P/D router with opt-in prefill affinity and sampled load placement.

    Routing skips backends whose last /health_generate probe failed. A backend
    that has not been probed yet counts as healthy.
    """

    def __init__(
        self,
        router_args: RouterArgs,
    ):
        self._validate_router_args(router_args)

        self.host = router_args.host
        self.port = router_args.port
        self.timeout = router_args.request_timeout_secs
        self.prefill_urls = [url[0] for url in router_args.prefill_urls]
        self.prefill_bootstrap_ports = [url[1] for url in router_args.prefill_urls]
        self.decode_urls = router_args.decode_urls
        self.test_external_dp_routing = router_args.test_external_dp_routing
        self.prefill_dp_size = None
        self.decode_dp_size = None
        self.affinity_policy = (
            SessionAffinityPolicy()
            if router_args.mini_lb_policy == "session_affinity"
            else None
        )
        self._load_poller = (
            LoadPoller(
                loads=self.affinity_policy.loads,
                metric_names=self.affinity_policy.metric_names,
                interval_secs=1.0,
                timeout_secs=5.0,
            )
            if self.affinity_policy is not None
            else None
        )
        self.health_poll_interval_secs = _health_poll_interval_secs(router_args)

    @functools.cached_property
    def _health(self) -> _BackendHealth:
        # Created on first use, which also covers instances built without
        # __init__.
        return _BackendHealth()

    def _client_timeout(self) -> aiohttp.ClientTimeout:
        affinity = self.affinity_policy is not None
        return aiohttp.ClientTimeout(
            total=self.timeout,
            # The total deadline already bounds affinity reads.
            sock_read=None if affinity else self.timeout,
            connect=_CONNECT_TIMEOUT_SECS if affinity else None,
            sock_connect=_CONNECT_TIMEOUT_SECS if affinity else 300,
        )

    async def _engine_request(self, engine, operation, *args, **kwargs):
        try:
            return await operation(*args, **kwargs)
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as error:
            if self.affinity_policy is None:
                raise
            self.affinity_policy.loads.mark_unreachable(engine)
            raise _UpstreamConnectionError(f"{engine}: {error!r}") from error

    async def _gather_requests(self, *requests):
        """Cancel and close the peer when affinity loses one upstream."""
        if self.affinity_policy is None:
            return await asyncio.gather(*requests)
        tasks = [asyncio.ensure_future(request) for request in requests]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            for response in responses:
                if isinstance(response, aiohttp.ClientResponse):
                    response.close()
            raise

    def _validate_router_args(self, router_args: RouterArgs):
        logger.warning(
            "\x1b[33mMiniLB is for debugging only. "
            "Routing policy comes from --mini-lb-policy.\033[0m"
        )

        # NOTE: too many arguments unsupported, just validate some important ones
        if router_args.policy != "random":
            logger.warning("[MiniLB] Ignoring --policy, see --mini-lb-policy")
            router_args.policy = "random"
        if router_args.mini_lb_policy not in MINI_LB_POLICY_CHOICES:
            raise ValueError(
                f"Unknown MiniLB policy {router_args.mini_lb_policy!r}, "
                f"expected one of {MINI_LB_POLICY_CHOICES}"
            )

        if not router_args.pd_disaggregation:
            raise ValueError("MiniLB only supports PD disaggregation mode")

        if len(router_args.prefill_urls) == 0 or len(router_args.decode_urls) == 0:
            raise ValueError(
                "MiniLB requires at least one prefill and one decode server"
            )

    def start(self):
        global lb
        lb = self
        if self.affinity_policy is not None:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                force=True,
            )
        uvicorn.run(app, host=self.host, port=self.port)

    def start_load_polling(self) -> Optional[asyncio.Task]:
        """Poll engine loads only when session affinity is enabled."""
        if self._load_poller is None:
            return None
        return asyncio.create_task(
            self._load_poller.run(lambda: self.prefill_urls + self.decode_urls)
        )

    def start_health_polling(self) -> Optional[asyncio.Task]:
        """Probe backend health in the background unless it is disabled."""
        if self.health_poll_interval_secs is None:
            return None
        return asyncio.create_task(self._poll_health())

    def _registered_urls(self) -> List[str]:
        return list(dict.fromkeys(chain(self.prefill_urls, self.decode_urls)))

    async def _poll_health(self) -> None:
        """Probe each backend every interval, like LoadPoller: a backend whose
        probe is still in flight is skipped, so one hung backend does not delay
        the others."""
        probes: Dict[str, asyncio.Task] = {}
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_SECS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            try:
                while True:
                    probes = {
                        url: probe for url, probe in probes.items() if not probe.done()
                    }
                    for url in self._registered_urls():
                        if url not in probes:
                            probes[url] = asyncio.create_task(
                                self._probe_health(session, url)
                            )
                    await asyncio.sleep(self.health_poll_interval_secs)
            finally:
                for probe in probes.values():
                    probe.cancel()
                await asyncio.gather(*probes.values(), return_exceptions=True)

    async def probe_backend_health(self) -> None:
        """Probe every registered backend once; each verdict applies as soon as
        that backend answers.

        Run one probe at a time: the background poller is the only caller.
        """
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_SECS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await asyncio.gather(
                *(self._probe_health(session, url) for url in self._registered_urls())
            )

    def request_health_probe(self, url: str) -> None:
        """Probe `url` now instead of at the next poll.

        Called when a backend breaks in the middle of a request, so routing
        stops sending it traffic within one probe rather than one poll
        interval. The probe decides, not the break: a transient error on a
        healthy backend leaves it routable. No-op when health checks are off
        or a requested probe of `url` is still in flight.
        """
        if getattr(self, "health_poll_interval_secs", None) is None:
            return
        # Kept in __dict__ so instances built without __init__ work too.
        requested = self.__dict__.setdefault("_probes_requested", {})
        probe = requested.get(url)
        if probe is not None and not probe.done():
            return
        requested[url] = _spawn_background(self._probe_once(url))

    async def _probe_once(self, url: str) -> None:
        timeout = aiohttp.ClientTimeout(total=HEALTH_CHECK_TIMEOUT_SECS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self._probe_health(session, url)

    async def _probe_health(self, session: aiohttp.ClientSession, url: str) -> None:
        """GET HEALTH_PROBE_PATH on `url` and route to it only if it answers 200."""
        health = self._health
        epoch = health.epoch(url)
        try:
            status, error = await _probe_status(session, url, HEALTH_PROBE_PATH)
        except Exception:
            logger.exception("[MiniLB] health probe of %s failed", url)
            return
        if health.epoch(url) != epoch:
            # Removed or re-registered while the probe was in flight.
            return
        if status == 200:
            if url in health.unhealthy:
                logger.info("[MiniLB] backend %s is healthy again", url)
                health.unhealthy.discard(url)
        elif url not in health.unhealthy:
            logger.warning(
                "[MiniLB] backend %s failed %s (%s), skipping it",
                url,
                HEALTH_PROBE_PATH,
                status if error is None else error,
            )
            health.unhealthy.add(url)

    def _routable(self, role: str, urls: List[str]) -> List[str]:
        """Candidates of `role` for select_pair: healthy and reachable backends,
        else the reachable ones, else all of them.

        Reachability comes from the session affinity load table. The policy
        fails open on it too, so passing it only healthy backends would let
        it fall back to an unreachable subset instead of a reachable one.
        """
        if not urls:
            return []
        loads = self.affinity_policy.loads if self.affinity_policy else None
        reachable = [url for url in urls if loads is None or loads.is_reachable(url)]
        healthy = [url for url in reachable if url not in self._health.unhealthy]
        failing_open = self._health.roles_failing_open
        if healthy:
            if role in failing_open:
                logger.info("[MiniLB] %s has a healthy backend again", role)
                failing_open.discard(role)
            return healthy
        if role not in failing_open:
            logger.warning(
                "[MiniLB] no healthy %s backend, routing to all %d of them",
                role,
                len(reachable or urls),
            )
            failing_open.add(role)
        return reachable or list(urls)

    def _forget_health(self, worker_url) -> None:
        """Treat `worker_url` as not probed yet."""
        self._health.forget(worker_url)

    async def _ensure_dp_sizes(self):
        if self.prefill_dp_size is not None:
            return
        session_args = (
            {"timeout": self._client_timeout()}
            if self.affinity_policy is not None
            else {}
        )
        async with aiohttp.ClientSession(**session_args) as session:
            async with await self._engine_request(
                self.prefill_urls[0], session.get, f"{self.prefill_urls[0]}/server_info"
            ) as resp:
                info = await self._engine_request(self.prefill_urls[0], resp.json)
                prefill_dp_size = len(info.get("internal_states", [1]))
            async with await self._engine_request(
                self.decode_urls[0], session.get, f"{self.decode_urls[0]}/server_info"
            ) as resp:
                info = await self._engine_request(self.decode_urls[0], resp.json)
                decode_dp_size = len(info.get("internal_states", [1]))
        self.prefill_dp_size, self.decode_dp_size = prefill_dp_size, decode_dp_size
        logger.info(
            f"[MiniLB] DP sizes: prefill={self.prefill_dp_size}, decode={self.decode_dp_size}"
        )

    def _fork_dp_requests(self, request):
        p_rank = random.randint(0, self.prefill_dp_size - 1)
        d_rank = random.randint(0, self.decode_dp_size - 1)

        prefill_req = request.copy()
        decode_req = request.copy()
        prefill_req["routed_dp_rank"] = p_rank
        decode_req["routed_dp_rank"] = d_rank
        decode_req["disagg_prefill_dp_rank"] = p_rank

        return prefill_req, decode_req, d_rank

    def select_pair(self, routing_key=None, room=None):
        """Return a random or affinity-selected (prefill, bootstrap port, decode).

        Only healthy backends are candidates (see _routable), so session
        affinity re-pins a routing key whose pinned prefill became unhealthy.
        """
        prefill_urls = self._routable("prefill", self.prefill_urls)
        decode_urls = self._routable("decode", self.decode_urls)
        assert prefill_urls, "No prefill servers available"
        assert decode_urls, "No decode servers available"
        if self.affinity_policy is None:
            prefill = prefill_urls[random.randint(0, len(prefill_urls) - 1)]
            decode = decode_urls[random.randint(0, len(decode_urls) - 1)]
        else:
            decisions = self.affinity_policy.select(
                prefill_urls=prefill_urls,
                decode_urls=decode_urls,
                routing_key=routing_key,
            )
            for decision in decisions:
                logger.info(
                    "[MiniLB] routing decision %s",
                    decision.to_log_json(room=room, routing_key=routing_key),
                )
            prefill, decode = (decision.engine for decision in decisions)
        pidx = self.prefill_urls.index(prefill)
        return prefill, self.prefill_bootstrap_ports[pidx], decode

    def current_role_and_port(self, worker_url):
        """Return (role, bootstrap_port) of a registered server, or (None, None)
        if it is not in either routing list."""
        if worker_url in self.prefill_urls:
            return (
                "prefill",
                self.prefill_bootstrap_ports[self.prefill_urls.index(worker_url)],
            )
        if worker_url in self.decode_urls:
            return "decode", None
        return None, None

    def remove_worker(self, worker_url):
        """Drop a server from both routing lists so no new requests are sent to
        it (used to quiesce it before a role switch)."""
        self._forget_health(worker_url)
        if worker_url in self.decode_urls:
            self.decode_urls.remove(worker_url)
        if worker_url in self.prefill_urls:
            idx = self.prefill_urls.index(worker_url)
            self.prefill_urls.pop(idx)
            self.prefill_bootstrap_ports.pop(idx)

    def add_worker(self, worker_url, role, bootstrap_port=None):
        """Register a server under a role in the routing lists."""
        self._forget_health(worker_url)
        if role == "prefill":
            self.prefill_urls.append(worker_url)
            self.prefill_bootstrap_ports.append(bootstrap_port or 8998)
        elif role == "decode":
            self.decode_urls.append(worker_url)

    def apply_role_switch(self, worker_url, new_role, bootstrap_port=None):
        """Move a server between the prefill and decode routing lists after its
        role has been switched on the backend. Idempotent."""
        self.remove_worker(worker_url)
        self.add_worker(worker_url, new_role, bootstrap_port)

    async def generate(
        self, modified_request, prefill_server, decode_server, endpoint
    ) -> ORJSONResponse:
        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        expected_decode_dp_rank = None
        if self.test_external_dp_routing:
            await self._ensure_dp_sizes()
            prefill_req, decode_req, expected_decode_dp_rank = self._fork_dp_requests(
                modified_request
            )
        else:
            prefill_req = modified_request
            decode_req = modified_request

        async with aiohttp.ClientSession(timeout=self._client_timeout()) as session:
            tasks = [
                self._engine_request(
                    prefill_server,
                    session.post,
                    f"{prefill_server}/{endpoint}",
                    json=prefill_req,
                ),
                self._engine_request(
                    decode_server,
                    session.post,
                    f"{decode_server}/{endpoint}",
                    json=decode_req,
                ),
            ]

            # Wait for both responses to complete. Prefill should end first.
            prefill_response, decode_response = await self._gather_requests(*tasks)

            if "return_logprob" in modified_request:
                prefill_json = await self._engine_request(
                    prefill_server, prefill_response.json
                )
                ret_json = await self._engine_request(
                    decode_server, decode_response.json
                )

                # merge `meta_info.input_token_logprobs` from prefill to decode
                if "meta_info" in ret_json:
                    if "input_token_logprobs" in ret_json["meta_info"]:
                        ret_json["meta_info"]["input_token_logprobs"] = (
                            prefill_json["meta_info"]["input_token_logprobs"]
                            + ret_json["meta_info"]["input_token_logprobs"]
                        )
            else:
                ret_json = await self._engine_request(
                    decode_server, decode_response.json
                )

            if expected_decode_dp_rank is not None:
                actual = ret_json.get("meta_info", {}).get("dp_rank")
                if actual != expected_decode_dp_rank:
                    return ORJSONResponse(
                        content={
                            "error": f"DP rank mismatch: expected {expected_decode_dp_rank}, got {actual}"
                        },
                        status_code=500,
                    )

            return ORJSONResponse(
                content=ret_json,
                status_code=decode_response.status,
            )

    async def generate_stream(
        self, modified_request, prefill_server, decode_server, endpoint="generate"
    ):

        if self.test_external_dp_routing:
            warnings.warn("--test-external-dp-routing is not supported with streaming")

        assert endpoint[0] != "/", f"Endpoint should not start with '/': {endpoint}"

        session = aiohttp.ClientSession(timeout=self._client_timeout())
        responses = []

        async def close_upstreams():
            for response in responses:
                response.close()
            await session.close()

        async def drain_prefill_then_close():
            # The decode has answered in full, but the prefill may not have
            # wrapped up its side yet (it ends its stream once its own request,
            # KV transfer included, is done). Closing its connection now looks
            # like a client disconnect to it: it aborts the request and counts
            # it as aborted, or as a failed transfer. Read its stream to the end
            # first, bounded so a stuck prefill cannot hold the session forever.
            try:
                await asyncio.wait_for(
                    _read_to_eof(prefill_response),
                    timeout=min(PREFILL_DRAIN_TIMEOUT_SECS, self.timeout),
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.debug(
                    f"[MiniLB] prefill {prefill_server} did not finish after a "
                    f"completed stream: {type(e).__name__}: {e}"
                )
            finally:
                await close_upstreams()

        async def stream_results():
            completed = False
            upstream = prefill_server
            try:
                if modified_request.get("return_logprob", False):
                    prefill_chunks = []
                    async for chunk in prefill_response.content:
                        prefill_chunks.append(chunk)
                    first_prefill_chunk = (
                        prefill_chunks[0].decode("utf-8")[5:].strip("\n")
                    )
                    first_prefill_chunk_json = orjson.loads(first_prefill_chunk)

                    upstream = decode_server
                    async for chunk in decode_response.content:
                        # Note: This is inefficient
                        # merge prefill input_token_logprobs, output_token_logprobs to decode
                        decoded_chunk = chunk.decode("utf-8")
                        if (
                            decoded_chunk
                            and decoded_chunk.startswith("data:")
                            and "[DONE]" not in decoded_chunk
                        ):
                            ret_json = orjson.loads(decoded_chunk[5:].strip("\n"))
                            ret_json["meta_info"]["input_token_logprobs"] = (
                                first_prefill_chunk_json["meta_info"][
                                    "input_token_logprobs"
                                ]
                                + ret_json["meta_info"]["input_token_logprobs"]
                            )
                            yield b"data: " + orjson.dumps(ret_json) + b"\n\n"
                        else:
                            yield chunk
                else:
                    upstream = decode_server
                    async for chunk in decode_response.content.iter_chunked(
                        AIOHTTP_STREAM_READ_CHUNK_SIZE
                    ):
                        yield chunk
                completed = True
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                # The upstream broke after the 200 status went out (e.g. the
                # decode died mid-stream). Without this the stream would just
                # stop, and a cut-off answer would look complete. Tell the
                # client explicitly, in the same shape sglang's own /generate
                # stream reports errors.
                message = f"upstream {upstream} stream broke: {type(e).__name__}: {e}"
                logger.error(f"[MiniLB] {message}")
                # Let health routing (#1801) re-check this backend now.
                self.request_health_probe(upstream)
                yield _stream_error_event(message)
                yield b"data: [DONE]\n\n"
            finally:
                if completed and responses:
                    _spawn_background(drain_prefill_then_close())
                else:
                    await close_upstreams()

        # The session and both upstream responses stay open only for an
        # accepted request, whose stream_results() closes them when the
        # stream ends; every other exit closes them here.
        streaming = False
        try:
            # Both servers answer the headers before any token is produced, so
            # a request either server rejects is known here, before the
            # streaming response (and its 200 status) is committed.
            prefill_response, decode_response = await self._gather_requests(
                self._engine_request(
                    prefill_server,
                    session.post,
                    f"{prefill_server}/{endpoint}",
                    json=modified_request,
                ),
                self._engine_request(
                    decode_server,
                    session.post,
                    f"{decode_server}/{endpoint}",
                    json=modified_request,
                ),
            )
            responses.extend((prefill_response, decode_response))
            for response, engine in zip(responses, (prefill_server, decode_server)):
                if response.status != 200:
                    body = await self._engine_request(engine, response.read)
                    return Response(
                        content=body,
                        status_code=response.status,
                        media_type=response.content_type,
                    )
            streaming = True
            return StreamingResponse(
                stream_results(),
                media_type="text/event-stream",
            )
        finally:
            if not streaming:
                await close_upstreams()


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    pollers = (
        [lb.start_load_polling(), lb.start_health_polling()] if lb is not None else []
    )
    pollers = [poller for poller in pollers if poller is not None]
    try:
        yield
    finally:
        for poller in pollers:
            poller.cancel()
        for poller in pollers:
            with contextlib.suppress(asyncio.CancelledError):
                await poller


app = FastAPI(lifespan=_lifespan)
lb: Optional[MiniLoadBalancer] = None


async def _probe_backends(path: str) -> Response:
    """GET `path` on every registered server; 200 only if all answer 200.

    Routing skips unhealthy servers, but only after the next background probe
    and with fallback to all servers of a role when none is healthy, so this
    stays strict: one dead server reports the whole deployment as degraded.
    This is stricter than the Rust gateway's /readiness.
    """
    timeout = HEALTH_CHECK_TIMEOUT_SECS

    roles = {"prefill": list(lb.prefill_urls), "decode": list(lb.decode_urls)}
    backends = [(role, url) for role, urls in roles.items() for url in urls]
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=timeout)
    ) as session:
        results = await asyncio.gather(
            *(_probe_status(session, url, path) for _, url in backends)
        )
    unhealthy = [
        {"role": role, "url": url, "status": status, "error": error}
        for (role, url), (status, error) in zip(backends, results)
        if status != 200
    ]
    # A role switch can briefly leave a role with no registered server.
    unhealthy += [
        {"role": role, "url": None, "status": None, "error": "no server registered"}
        for role, urls in roles.items()
        if not urls
    ]
    if not unhealthy:
        return Response(status_code=200)
    logger.warning(f"[MiniLB] {path}: unhealthy backends {unhealthy}")
    return ORJSONResponse(
        content={"unhealthy": unhealthy},
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
    )


@app.get("/liveness")
async def liveness():
    """200 while the MiniLB process serves requests; ignores the backends."""
    return Response(status_code=200)


@app.get("/health")
async def health_check():
    return await _probe_backends("/health")


@app.get("/health_generate")
async def health_generate():
    return await _probe_backends("/health_generate")


async def _post_role_switch(worker_url, body):
    """POST the role switch to a backend server; return (status, json)."""
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=lb.timeout)
        ) as session:
            async with session.post(f"{worker_url}/pd_role_switch", json=body) as resp:
                return resp.status, await resp.json()
    except Exception as e:  # transport error -> report as a failure
        return 502, {"success": False, "message": str(e)}


@app.post("/pd_role_switch")
async def pd_role_switch(request_data: dict):
    """Switch a running server's PD role (prefill<->decode) at runtime and
    update the LB's routing lists. Body: {"worker_url", "new_role":
    "prefill"|"decode", "bootstrap_port"?, "decode_cuda_graph_bs"?,
    "decode_cuda_graph_memory_gb"?, "drain"?, "drain_timeout_secs"?}.

    The backend rejects a switch unless the instance is idle. To make this
    safe while serving, by default the LB first removes the server from its
    routing lists (so no new requests arrive), then retries the switch while
    the server drains its in-flight requests, and only then registers it
    under the new role. A failed server is restored only when the backend
    confirms that no role state changed."""
    worker_url = request_data.get("worker_url")
    new_role = request_data.get("new_role")
    if worker_url is None:
        raise HTTPException(status_code=400, detail="worker_url is required")
    if new_role not in ("prefill", "decode"):
        raise HTTPException(status_code=400, detail=f"invalid new_role={new_role!r}")

    drain = request_data.get("drain", True)
    drain_timeout = request_data.get("drain_timeout_secs", 300)
    old_role, old_port = lb.current_role_and_port(worker_url)

    body = {"new_role": new_role}
    for field in ("decode_cuda_graph_bs", "decode_cuda_graph_memory_gb"):
        if request_data.get(field) is not None:
            body[field] = request_data[field]

    # Stop routing new requests to this server so it can drain to idle.
    if drain and old_role is not None:
        lb.remove_worker(worker_url)

    deadline = asyncio.get_event_loop().time() + drain_timeout
    while True:
        status, result = await _post_role_switch(worker_url, body)
        if status == 200 and result.get("success", False):
            break
        # The backend rejects while not idle; keep retrying as it drains.
        not_idle = "not idle" in (result.get("message", "") or "").lower()
        if drain and not_idle and asyncio.get_event_loop().time() < deadline:
            await asyncio.sleep(1.0)
            continue
        if drain and old_role is not None and result.get("safe_to_restore", False):
            lb.add_worker(worker_url, old_role, old_port)
        return ORJSONResponse(content=result, status_code=status)

    lb.apply_role_switch(worker_url, new_role, request_data.get("bootstrap_port"))
    return ORJSONResponse(content=result, status_code=200)


@app.post("/flush_cache")
async def flush_cache(timeout: Optional[float] = None):
    # `timeout` must reach the workers. The scheduler treats a missing or
    # non-positive timeout as "flush now, skip the idle check", so dropping it
    # here frees KV buffers while a PD KV transfer is still reading them: the
    # transfer then fails for real and the peer session gets blacklisted.
    # Forwarding it keeps the scheduler on its deferred, drain-first path.
    params = None if timeout is None else {"timeout": timeout}
    async with aiohttp.ClientSession() as session:
        # Create the tasks
        tasks = []
        for server in chain(lb.prefill_urls, lb.decode_urls):
            tasks.append(session.post(f"{server}/flush_cache", params=params))
        for i, response in enumerate(asyncio.as_completed(tasks)):
            await response
    return Response(status_code=200)


# TODO: Remove `/get_server_info` alias after one release-cycle deprecation window.
@app.get("/server_info")
@app.get("/get_server_info")
async def get_server_info():
    prefill_infos = []
    decode_infos = []
    all_internal_states = []

    async with aiohttp.ClientSession() as session:
        for server in lb.prefill_urls:
            server_info = await session.get(f"{server}/server_info")
            prefill_infos.append(await server_info.json())
        for server in lb.decode_urls:
            server_info = await session.get(f"{server}/server_info")
            info_json = await server_info.json()
            decode_infos.append(info_json)
            # Extract internal_states from decode servers
            if "internal_states" in info_json:
                all_internal_states.extend(info_json["internal_states"])

    # Return format expected by bench_one_batch_server.py
    if all_internal_states:
        return {
            "internal_states": all_internal_states,
            "prefill": prefill_infos,
            "decode": decode_infos,
        }
    else:
        # Fallback with dummy data if no internal states found
        return {
            "internal_states": [
                {
                    "last_gen_throughput": 0.0,
                    "avg_spec_accept_length": None,
                }
            ],
            "prefill": prefill_infos,
            "decode": decode_infos,
        }


async def _get_model_info_impl():
    if not lb or not lb.prefill_urls:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="There is no server registered",
        )

    target_server_url = lb.prefill_urls[0]
    endpoint_url = f"{target_server_url}/model_info"

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint_url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=HTTPStatus.BAD_GATEWAY,
                        detail=(
                            f"Failed to get model info from {target_server_url}"
                            f"Status: {response.status}, Response: {error_text}"
                        ),
                    )

                model_info_json = await response.json()
                return ORJSONResponse(content=model_info_json)

        except aiohttp.ClientError:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail=f"Failed to get model info from backend",
            )


@app.get("/model_info")
async def model_info():
    return await _get_model_info_impl()


@app.get("/get_model_info")
async def get_model_info():
    return await _get_model_info_impl()


@app.post("/generate")
async def handle_generate_request(
    request_data: dict, x_routing_key: Optional[str] = Header(default=None)
):
    return await _forward_to_backend(request_data, "generate", x_routing_key)


async def _forward_to_backend(
    request_data: dict, endpoint_name: str, routing_key: Optional[str]
):
    batch_size = _get_request_batch_size(request_data)
    bootstrap_rooms = [
        _generate_bootstrap_room()
        for _ in range(batch_size if batch_size is not None else 1)
    ]
    try:
        prefill_server, bootstrap_port, decode_server = lb.select_pair(
            routing_key=routing_key,
            room=bootstrap_rooms[0] if bootstrap_rooms else None,
        )
    except EngineOverloaded as error:
        logger.warning(f"[MiniLB] POST /{endpoint_name} failed: {error}")
        return ORJSONResponse(
            content={"error": {"message": str(error)}},
            status_code=HTTPStatus.TOO_MANY_REQUESTS,
        )

    parsed_url = urllib.parse.urlparse(prefill_server)
    hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
    modified_request = request_data.copy()
    modified_request.update(
        {
            "bootstrap_host": (
                [hostname] * batch_size if batch_size is not None else hostname
            ),
            "bootstrap_port": (
                [bootstrap_port] * batch_size
                if batch_size is not None
                else bootstrap_port
            ),
            "bootstrap_room": (
                bootstrap_rooms if batch_size is not None else bootstrap_rooms[0]
            ),
        }
    )
    forward = lb.generate_stream if request_data.get("stream", False) else lb.generate
    try:
        return await forward(
            modified_request, prefill_server, decode_server, endpoint=endpoint_name
        )
    except _UpstreamConnectionError as error:
        logger.warning("[MiniLB] upstream unreachable: %s", error)
        return ORJSONResponse(
            content={"error": {"message": f"upstream unreachable: {error}"}},
            status_code=HTTPStatus.BAD_GATEWAY,
        )


@app.post("/v1/chat/completions")
async def handle_chat_completion_request(
    request_data: dict, x_routing_key: Optional[str] = Header(default=None)
):
    return await _forward_to_backend(request_data, "v1/chat/completions", x_routing_key)


@app.post("/v1/completions")
async def handle_completion_request(
    request_data: dict, x_routing_key: Optional[str] = Header(default=None)
):
    return await _forward_to_backend(request_data, "v1/completions", x_routing_key)


def _generate_bootstrap_room():
    return random.randint(0, 2**63 - 1)


# We may utilize `GenerateReqInput`'s logic later
def _get_request_batch_size(request):
    if (text := request.get("text")) is not None:
        return None if isinstance(text, str) else len(text)
    if (input_ids := request.get("input_ids")) is not None:
        return None if isinstance(input_ids[0], int) else len(input_ids)
    return None


@app.get("/v1/models")
async def get_models():
    prefill_server = lb.prefill_urls[0]  # Get the first prefill server
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.get(f"{prefill_server}/v1/models")
            if response.status != 200:
                raise HTTPException(
                    status_code=response.status,
                    detail=f"Prefill server error: Status {response.status}",
                )
            return ORJSONResponse(content=await response.json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
