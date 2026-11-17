# SPDX-License-Identifier: Apache-2.0
"""TPU keep-alive for a server that does no device work
(SGLANG_TPU_KEEPALIVE_SECONDS).

Shared TPU pools may reclaim a node that shows no TPU utilization for a while
(tpuq: 15 minutes), and a token op does not count: the load has to be a real
share of the time. A cold compile of a large model or a server waiting for
requests runs no device work, and no other process can fill in because the
server holds the chips.

keepalive_tick() runs short matmul bursts for SGLANG_TPU_KEEPALIVE_DUTY of the
time once no forward has started for SGLANG_TPU_KEEPALIVE_SECONDS. It runs on
the caller's thread, between the scheduler's idle iterations and between
precompile steps: torch_tpu's deferred execution is not thread-safe, and a
keep-alive thread failed forwards running at the same time.

A precompile step is minutes of compiling followed by milliseconds of device
work, and a 0.5 s burst per such step still reads as idle. After one, the
caller asks for a catch-up burst that keeps the chips busy for DUTY of the
time since the last burst.

A single graph's compile can itself outlast the reclaim window. Only there
does a helper thread tick (install_compile_keepalive): the calling thread is
inside the compiler then, so no forward runs beside it.
"""

import logging
import threading
import time

import torch

logger = logging.getLogger(__name__)

# Longest a burst holds the caller (an idle scheduler takes no request meanwhile).
_BURST_S = 0.5
# Longest catch-up burst after a precompile step.
_CATCH_UP_MAX_S = 45.0
# Seconds of weight loading between catch-up bursts.
_LOAD_TICK_S = 30.0

# time.monotonic() of the latest forward call (note_forward) and burst end.
_last_forward = 0.0
_last_burst_end = 0.0
_x: torch.Tensor | None = None


def note_forward() -> None:
    """Mark device work in progress; called at the start of every forward."""
    global _last_forward
    _last_forward = time.monotonic()
    install_compile_keepalive()


def keepalive_enabled() -> bool:
    from sglang_tpu.envs import envs as tpu_envs

    return tpu_envs.SGLANG_TPU_KEEPALIVE_SECONDS.get() > 0


def keepalive_tick(catch_up: bool = False) -> None:
    """Run one burst if the server has been without device work long enough.

    With ``catch_up`` (after a precompile step), run regardless of the step's
    own forward, long enough to cover DUTY of the time since the last burst.
    """
    global _last_burst_end, _x
    from sglang_tpu.envs import envs as tpu_envs

    period = tpu_envs.SGLANG_TPU_KEEPALIVE_SECONDS.get()
    if period <= 0:
        return
    install_compile_keepalive()
    duty = tpu_envs.SGLANG_TPU_KEEPALIVE_DUTY.get()
    now = time.monotonic()
    if catch_up:
        burst = min(_CATCH_UP_MAX_S, (now - _last_burst_end) * duty / (1.0 - duty))
        if burst < _BURST_S:
            return
    else:
        if now - _last_forward < period:
            return
        if now - _last_burst_end < _BURST_S * (1.0 / duty - 1.0):
            return
        burst = _BURST_S
    if _x is None:
        _x = torch.full((4096, 4096), 1e-3, dtype=torch.bfloat16, device="tpu")
    end = now + burst
    while time.monotonic() < end:
        y = _x
        for _ in range(16):
            y = y @ _x
        y.sum().item()  # wait for the device, so the duty is real time
    _last_burst_end = time.monotonic()


def keepalive_after_idle(result, scheduler):
    """AFTER hook for Scheduler.on_idle."""
    keepalive_tick()
    return result


def _keepalive_weights(weights):
    """Yield ``weights``, with a catch-up burst every _LOAD_TICK_S of loading."""
    last = time.monotonic()
    for item in weights:
        yield item
        if time.monotonic() - last >= _LOAD_TICK_S:
            keepalive_tick(catch_up=True)
            last = time.monotonic()


def keepalive_while_loading(weights, *args, **kwargs):
    """AFTER hook for DefaultModelLoader._get_all_weights: a 93L checkpoint
    loads for 5+ minutes with no device work."""
    from sglang_tpu.envs import envs as tpu_envs

    if tpu_envs.SGLANG_TPU_KEEPALIVE_SECONDS.get() <= 0:
        return None
    return _keepalive_weights(weights)


_compile_patch_checked = False


def start_catch_up() -> None:
    """Measure catch-up bursts from now: a compile while serving must not make
    up for the whole time since the last idle burst (up to _CATCH_UP_MAX_S)."""
    global _last_burst_end
    _last_burst_end = time.monotonic()


def _tick_beside_compile(stop: threading.Event) -> None:
    """Catch-up bursts from a helper thread until stop; the compile keeps the main thread."""
    start_catch_up()
    try:
        with torch.no_grad():
            while not stop.wait(1.0):
                keepalive_tick(catch_up=True)
    except Exception:
        logger.exception("Keep-alive beside a compile failed; waiting without it.")


def install_compile_keepalive() -> None:
    """Keep the chips busy from a helper thread while torch_tpu compiles synchronously.

    Why: one 93-layer Kimi-K3.1 speculative prefill graph compiles for 12-30
    minutes with the chips idle, and tpuq reclaims a slice after ~15 idle
    minutes; keepalive_tick only ran between graphs.
    How: wrap torch_tpu's StaticCompiler._compile. The compile stays on the
    calling thread (its locks and thread-locals unchanged), and a helper thread
    runs catch-up matmul bursts until it returns. torch_tpu's eager mode is
    thread-local, so the helper's ops run eagerly even while the main thread
    compiles under _INTERNAL_COMPILE_FX_GRAPH. The helper is joined before the
    compiled program can run, so it never overlaps a forward. Moving the
    compile itself to a worker thread deadlocked on a dynamic-shape recompile.
    Upstream PR or issue: none filed yet (torch_tpu keep-alive during compile).
    Remove when: torch_tpu can keep the device busy during compiles, or a
    reclaimer that counts compiles as activity.
    """
    global _compile_patch_checked
    if _compile_patch_checked:
        return
    _compile_patch_checked = True
    if not keepalive_enabled():
        return
    try:
        from torch_tpu._internal.compile import compiler

        original = compiler.StaticCompiler._compile
    except (ImportError, AttributeError):
        logger.warning(
            "torch_tpu has no StaticCompiler._compile; long compiles run without "
            "keep-alive."
        )
        return

    def _compile(self, *args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            return original(self, *args, **kwargs)
        stop = threading.Event()
        helper = threading.Thread(
            target=_tick_beside_compile, args=(stop,), name="tpu_keepalive_compile"
        )
        helper.start()
        try:
            return original(self, *args, **kwargs)
        finally:
            stop.set()
            helper.join()

    compiler.StaticCompiler._compile = _compile
    logger.info("Keep-alive installed beside synchronous TPU compiles.")
