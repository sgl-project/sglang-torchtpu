"""Preserve the input leader's serving address after TPU rank normalization."""

import logging
import os
import signal
import socket
from concurrent.futures import CancelledError
from contextlib import AsyncExitStack, contextmanager, suppress
from functools import partial

import anyio
from anyio.abc import SocketListener
from anyio.from_thread import start_blocking_portal

from sglang.srt.arg_groups.overrides import resolving_view
from sglang.srt.utils.network import try_bind_socket

logger = logging.getLogger(__name__)


async def _write(write, *args):
    try:
        await write(*args)
    except RuntimeError as error:
        # uvloop raises RuntimeError, not an anyio error, writing to a transport
        # the peer already reset.
        raise anyio.ClosedResourceError from error


async def _copy_stream(source, destination, cancel_scope):
    try:
        async for data in source:
            await _write(destination.send, data)
        # Preserve half-close: the peer can still send its response after EOF.
        await _write(destination.send_eof)
    except (anyio.BrokenResourceError, anyio.ClosedResourceError, OSError):
        cancel_scope.cancel()


async def _forward_connection(target_host, port, client):
    # One connection must never stop the leader forwarder: that shuts the worker down.
    try:
        await _forward_connection_inner(target_host, port, client)
    except Exception:
        logger.warning(
            "TPU leader forwarder dropped a connection on port %s", port, exc_info=True
        )


async def _forward_connection_inner(target_host, port, client):
    async with client:
        try:
            with anyio.fail_after(10):
                leader = await anyio.connect_tcp(target_host, port)
        except (OSError, TimeoutError) as error:
            # Closing the client makes readiness fail until the leader listens.
            logger.debug("TPU leader connection on port %s failed: %s", port, error)
            return

        async with leader, anyio.create_task_group() as tasks:
            tasks.start_soon(_copy_stream, client, leader, tasks.cancel_scope)
            tasks.start_soon(_copy_stream, leader, client, tasks.cancel_scope)


async def _serve_ports(host, target_host, ports, *, task_status):
    async with AsyncExitStack() as stack:
        listeners = []
        for port in ports:
            sock = stack.enter_context(try_bind_socket(host, port))
            sock.listen(socket.SOMAXCONN)
            listener = await stack.enter_async_context(
                await SocketListener.from_socket(sock)
            )
            listeners.append((listener, port))

        async with anyio.create_task_group() as tasks:
            for listener, port in listeners:
                tasks.start_soon(
                    listener.serve, partial(_forward_connection, target_host, port)
                )
            task_status.started()
            await anyio.sleep_forever()


def _on_forwarder_exit(serving):
    if serving.cancelled():
        return
    logger.error(
        "TPU leader forwarder stopped unexpectedly; shutting down worker",
        exc_info=serving.exception(),
    )
    # Use Engine's fatal-error handler: workers defer SIGTERM until leader drain.
    os.kill(os.getpid(), signal.SIGQUIT)


@contextmanager
def _forward_ports(host: str, target_host: str, ports: list[int]):
    """Bind before yielding; supervise serving and close connections on exit."""
    with start_blocking_portal(name="tpu-leader-forwarder") as portal:
        serving, _ = portal.start_task(_serve_ports, host, target_host, ports)
        serving.add_done_callback(_on_forwarder_exit)
        try:
            yield
        finally:
            serving.cancel()
            with suppress(CancelledError):
                serving.result()


@contextmanager
def forward_normalized_leader(server_args):
    """Yield whether this worker owns the input leader's serving listeners."""
    cfg = resolving_view(server_args)
    if (
        not cfg.tpu_node_normalization
        or cfg.nnodes == 1
        or server_args.node_rank != 0
        or cfg.node_rank == 0
    ):
        yield False
        return

    ports = [cfg.port]
    if cfg.disaggregation_mode == "prefill" or (
        cfg.enable_pd_role_switch and cfg.disaggregation_mode != "null"
    ):
        ports.append(cfg.disaggregation_bootstrap_port)
    target_host = cfg.dist_init_addr.rpartition(":")[0]
    with _forward_ports(cfg.host, target_host, ports):
        logger.info(
            "Forwarding input TPU leader ports %s to normalized leader %s "
            "(local node rank %s)",
            ports,
            target_host,
            cfg.node_rank,
        )
        yield True
