"""tpu_dist collectives across a multi-host slice, without the diffusion stack.

CFG x Ulysses needs all_reduce, all_gather and all_to_all_single on `tpu` tensors.
"""

import os
import queue
import traceback
import unittest

import torch
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_tpu_ci

# Hand-wired in exec mode (diffusion-mh-collectives): every host runs this file,
# which a MULTIHOST_LAUNCH group cannot do.
register_tpu_ci(
    est_time=600,
    suite="tpu-integration",
    nightly=True,
    labels=["multihost", "diffusion", "collectives"],
)

from sglang.test.test_utils import CustomTestCase, is_in_ci

DIST_PORT = 10011
# tpu7x: 4 chips per host, 2 cores per chip.
DEVICES_PER_HOST = int(os.environ.get("TORCH_TPU_DEVICES_PER_HOST", "8"))


def _slice_hosts() -> list[str]:
    return [h for h in os.environ.get("TPU_WORKER_HOSTNAMES", "").split(",") if h]


def _worker(local_rank: int, hosts: list[str], node_rank: int, result_q) -> None:
    from sglang.multimodal_gen.runtime.distributed.parallel_state import (
        init_distributed_environment,
    )
    from sglang.multimodal_gen.runtime.platforms import current_platform

    world_size = len(hosts) * DEVICES_PER_HOST
    rank = node_rank * DEVICES_PER_HOST + local_rank
    os.environ["MASTER_ADDR"] = hosts[0]
    os.environ["MASTER_PORT"] = str(DIST_PORT)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    status = 1
    try:
        current_platform.set_device(current_platform.get_device(local_rank))
        backend = current_platform.get_torch_distributed_backend_str()
        assert backend == "tpu_dist", backend

        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://{hosts[0]}:{DIST_PORT}",
            local_rank=local_rank,
            backend=backend,
        )
        device = current_platform.get_local_torch_device()

        summed = torch.full((8,), float(rank), device=device)
        torch.distributed.all_reduce(summed)
        expected_sum = world_size * (world_size - 1) / 2
        assert torch.allclose(summed.cpu(), torch.full((8,), expected_sum)), (
            f"all_reduce: {summed.cpu()[0].item()} != {expected_sum}"
        )

        gathered = torch.empty(world_size, device=device)
        torch.distributed.all_gather_into_tensor(
            gathered, torch.full((1,), float(rank), device=device)
        )
        assert torch.equal(
            gathered.cpu(), torch.arange(world_size, dtype=gathered.dtype)
        ), f"all_gather: {gathered.cpu().tolist()}"

        # Rank r sends slice j to rank j, so every rank receives [0..world).
        send = torch.full((world_size,), float(rank), device=device)
        recv = torch.empty(world_size, device=device)
        torch.distributed.all_to_all_single(recv, send)
        assert torch.equal(recv.cpu(), torch.arange(world_size, dtype=recv.dtype)), (
            f"all_to_all_single: {recv.cpu().tolist()}"
        )

        result_q.put((rank, "ok"))
        status = 0
    except Exception as e:  # surfaced by the parent as the failure message
        result_q.put((rank, f"{type(e).__name__}: {e}"))
        traceback.print_exc()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        # libtpu's atexit vBAR StopSession can hang on GKE, and a parent SIGTERM mid-teardown
        # wedges the slice; exit without it, as node_normalization does.
        result_q.close()
        result_q.join_thread()  # flush the result before the hard exit drops the feeder thread
        os._exit(status)


class TestTpuDistCollectivesMultihost(CustomTestCase):
    def test_collectives_across_the_slice(self):
        hosts = _slice_hosts()
        if len(hosts) < 2:
            # A skip is exit 0: in CI that is a green job that measured nothing.
            message = f"needs a multi-host slice, TPU_WORKER_HOSTNAMES={hosts}"
            if is_in_ci():
                self.fail(message)
            self.skipTest(message)
        node_rank = int(os.environ.get("TPU_WORKER_ID", "0"))

        ctx = mp.get_context("spawn")
        result_q = ctx.Queue()
        procs = [
            ctx.Process(target=_worker, args=(i, hosts, node_rank, result_q))
            for i in range(DEVICES_PER_HOST)
        ]
        for p in procs:
            p.start()
        # A rank left alive holds its PJRT device lock for the next job.
        try:
            results = {}
            for _ in procs:
                try:
                    rank, status = result_q.get(timeout=900)
                except queue.Empty:
                    break
                results[rank] = status
            for p in procs:
                p.join(timeout=60)
        finally:
            for p in procs:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=10)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=10)
        # Exit codes first: a rank that aborts in libtpu reports nothing.
        exit_codes = [p.exitcode for p in procs]
        self.assertTrue(
            all(c == 0 for c in exit_codes),
            f"exitcodes={exit_codes} (negative = killed by that signal); "
            f"reported={results}",
        )
        failures = {r: s for r, s in results.items() if s != "ok"}
        self.assertEqual(failures, {}, f"rank failures: {failures}")
        self.assertEqual(
            len(results),
            DEVICES_PER_HOST,
            f"only {len(results)} of {DEVICES_PER_HOST} ranks reported",
        )


if __name__ == "__main__":
    unittest.main()
