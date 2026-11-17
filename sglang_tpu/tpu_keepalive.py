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
"""

import time

import torch

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
