import asyncio
import unittest
from unittest.mock import patch

from sglang.benchmark import serving
from sglang.benchmark.datasets.common import DatasetRow
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestServingDuration(unittest.TestCase):
    def test_request_generation_stops_at_traffic_deadline(self):
        requests = [
            DatasetRow(prompt=[index], prompt_len=1, output_len=1) for index in range(3)
        ]

        async def collect_requests():
            return [
                request
                async for request in serving.get_request(
                    requests,
                    request_rate=1.0,
                    traffic_duration_seconds=1.0,
                )
            ]

        with (
            patch.object(serving.time, "perf_counter", side_effect=(0.0, 0.0, 0.0)),
            patch.object(serving.np.random, "exponential", return_value=2.0),
            patch.object(serving.asyncio, "sleep") as sleep,
        ):
            admitted = asyncio.run(collect_requests())

        self.assertEqual(admitted, requests[:1])
        sleep.assert_awaited_once_with(1.0)

    def test_unbounded_drain_preserves_existing_behavior(self):
        async def drain_tasks():
            tasks = [
                asyncio.create_task(asyncio.sleep(0, result=index))
                for index in range(2)
            ]
            done, timed_out = await serving._drain_request_tasks(
                tasks, drain_timeout_seconds=None
            )
            return tasks, done, timed_out

        tasks, done, timed_out = asyncio.run(drain_tasks())

        self.assertEqual(done, set(tasks))
        self.assertEqual(timed_out, 0)
        self.assertEqual([task.result() for task in tasks], [0, 1])

    def test_drain_timeout_cancels_only_unfinished_requests(self):
        async def drain_tasks():
            async def finish():
                return "done"

            async def block():
                await asyncio.Event().wait()

            finished = asyncio.create_task(finish())
            blocked = asyncio.create_task(block())
            await asyncio.sleep(0)
            done, timed_out = await serving._drain_request_tasks(
                [finished, blocked], drain_timeout_seconds=0.01
            )
            return finished, blocked, done, timed_out

        finished, blocked, done, timed_out = asyncio.run(drain_tasks())

        self.assertEqual(done, {finished})
        self.assertEqual(timed_out, 1)
        self.assertEqual(finished.result(), "done")
        self.assertTrue(blocked.cancelled())


if __name__ == "__main__":
    unittest.main()
