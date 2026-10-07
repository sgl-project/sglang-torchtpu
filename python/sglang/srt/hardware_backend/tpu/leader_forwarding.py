"""Preserve the input leader's serving address after TPU rank normalization."""

import asyncio
import logging
import threading
from contextlib import contextmanager

from sglang.srt.arg_groups.overrides import resolving_view

logger = logging.getLogger(__name__)


class LeaderForwarder:
    """Own TCP listeners until the local worker schedulers exit."""

    def __init__(self, host: str, target_host: str, ports: list[int]):
        self.host = host
        self.target_host = target_host
        self.ports = ports
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._connections: set[asyncio.Task] = set()
        self._thread = threading.Thread(
            target=self._run, name="tpu-leader-forwarder", daemon=True
        )

    @staticmethod
    async def _pipe(reader, writer):
        while data := await reader.read(64 * 1024):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()

    async def _handle(self, reader, writer, port):
        upstream_writer = None
        pumps = []
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(self.target_host, port), timeout=10
            )
            pumps = [
                asyncio.create_task(self._pipe(reader, upstream_writer)),
                asyncio.create_task(self._pipe(upstream_reader, writer)),
            ]
            await asyncio.gather(*pumps)
        except (ConnectionError, OSError) as error:
            # During startup the normalized leader may not be listening yet.
            # Close this connection so readiness fails and the client can retry.
            logger.debug("TPU leader connection on port %s closed: %s", port, error)
        finally:
            for pump in pumps:
                pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            for connection in (writer, upstream_writer):
                if connection is not None:
                    connection.close()
                    try:
                        await asyncio.wait_for(connection.wait_closed(), timeout=5)
                    except TimeoutError:
                        connection.transport.abort()
                    except (ConnectionError, OSError):
                        logger.debug("TPU leader connection closed by peer")

    async def _serve(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        servers = []
        try:
            for port in self.ports:
                server = await asyncio.start_server(
                    lambda reader, writer, port=port: self._accept(
                        reader, writer, port
                    ),
                    self.host,
                    port,
                )
                servers.append(server)
            self._ready.set()
            await self._stop.wait()
        finally:
            for server in servers:
                server.close()
            # Stop accepting before cancelling active streams. wait_closed()
            # can otherwise wait forever for an idle keep-alive connection.
            await asyncio.sleep(0)
            connections = tuple(self._connections)
            for connection in connections:
                connection.cancel()
            await asyncio.gather(*connections, return_exceptions=True)
            for server in servers:
                await server.wait_closed()

    def _accept(self, reader, writer, port):
        task = asyncio.create_task(self._handle(reader, writer, port))
        self._connections.add(task)
        task.add_done_callback(self._connections.discard)

    def _run(self):
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._error = error
        finally:
            self._ready.set()

    def __enter__(self):
        self._thread.start()
        self._ready.wait()
        if self._error is not None:
            self._thread.join()
            raise self._error
        return self

    def __exit__(self, *exc):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join()
        if self._error is not None:
            raise self._error


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
    with LeaderForwarder(cfg.host, target_host, ports):
        logger.info(
            "Forwarding input TPU leader ports %s to normalized leader %s "
            "(local node rank %s)",
            ports,
            target_host,
            cfg.node_rank,
        )
        yield True
