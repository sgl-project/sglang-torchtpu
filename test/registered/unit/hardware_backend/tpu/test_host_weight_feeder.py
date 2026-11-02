"""Host-shared weight feeder."""

import contextlib
import multiprocessing
import os
import shutil
import tempfile
import threading
import time
import traceback
import unittest
from unittest.mock import MagicMock, mock_open, patch

import safetensors.torch
import torch

from sglang.srt.configs.load_config import LoadFormat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu import weight_feeder as host_weight_feeder
from sglang_tpu.hooks.host_weight_feeder import weight_feeder_load_local_weights
from sglang_tpu.weight_feeder import feeder

register_cpu_ci(est_time=30, suite="base-a-test-cpu")

NUM_RANKS = 3
NUM_SHARDS = 5


@contextlib.contextmanager
def _engine_on_local_path(model_path, ranks):
    """Server args of an engine whose checkpoint is at a local path."""
    with (
        patch.object(
            host_weight_feeder, "_engine_replica_key", lambda replica: "owner"
        ),
        patch("sglang.srt.runtime_context.get_exec") as get_exec,
        patch("sglang.srt.runtime_context.get_model") as get_model,
        patch("sglang.srt.runtime_context.get_parallel") as get_parallel,
        patch("sglang.srt.runtime_context.get_server_args") as get_server_args,
    ):
        get_model.return_value.model_path = model_path
        get_model.return_value.load_format = LoadFormat.AUTO
        get_exec.return_value.moe.ep_join_mode = "off"
        get_parallel.return_value.num_dp_ranks = 1
        get_parallel.return_value.tp_size = ranks
        get_parallel.return_value.pp_size = 1
        get_server_args.return_value.nnodes = 1
        yield


def _prefetch_from_engine(model_path, uris, ranks, results):
    """An engine process: holds the feeder, then its children load as ranks."""
    with _engine_on_local_path(model_path, ranks):
        host_weight_feeder.start_weight_prefetch()
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_load_rank, args=(uris, ranks, results))
        for _ in range(ranks)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)


def _load_rank(uris, ranks, results):
    try:
        with (
            patch.object(host_weight_feeder, "_num_ranks_on_host", lambda: ranks),
            patch.object(host_weight_feeder, "_replica_id", lambda: 0),
            patch.object(
                host_weight_feeder, "_engine_replica_key", lambda replica: "owner"
            ),
        ):
            loaded = {
                name: tensor.tolist()
                for name, tensor in host_weight_feeder.shared_weights_iterator(
                    uris, lambda files: iter(())
                )
            }
        results.put((os.getpid(), loaded))
    except BaseException as e:
        results.put((os.getpid(), f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


class TestHostWeightFeeder(unittest.TestCase):
    MOUNTS = (
        "24 1 0:1 / / rw - ext4 /dev/sda rw\n"
        "25 24 0:2 / /staging rw - tmpfs tmpfs rw\n"
        "26 25 0:3 / /staging/nested rw - ext4 /dev/sdb rw\n"
    )

    def setUp(self):
        environment_patch = patch.dict(
            os.environ, {"SGLANG_TPU_WEIGHT_FEEDER_ENABLED": "1"}
        )
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    def _filesystem_type(self, path: str) -> str:
        with patch("builtins.open", mock_open(read_data=self.MOUNTS)):
            with patch("os.path.realpath", side_effect=lambda p: p):
                return host_weight_feeder._filesystem_type(path)

    def test_longest_mount_decides(self):
        self.assertEqual(self._filesystem_type("/staging/a"), "tmpfs")
        self.assertEqual(self._filesystem_type("/staging/nested/a"), "ext4")
        self.assertEqual(self._filesystem_type("/stagingish/a"), "ext4")

    def test_local_path_reads_through_the_feeder(self):
        for kwargs in ({}, {"local_expert_range": (0, 4)}):
            with self.subTest(kwargs=kwargs):
                original = MagicMock()
                with (
                    patch(
                        "sglang_tpu.weight_feeder.can_use_shared_staging",
                        return_value=True,
                    ),
                    patch("sglang_tpu.weight_feeder.shared_weights_iterator") as shared,
                ):
                    weight_feeder_load_local_weights(
                        original, ["/weights/a.safetensors"], **kwargs
                    )
                shared.assert_called_once()
                original.assert_not_called()

    def test_feeder_is_off_by_default(self):
        """Loaders that keep tensors must stay off the feeder unless it is enabled."""
        del os.environ["SGLANG_TPU_WEIGHT_FEEDER_ENABLED"]
        original = MagicMock(return_value=iter(()))
        weight_feeder_load_local_weights(original, ["/weights/a.safetensors"])
        original.assert_called_once()

    def test_reads_entire_shard(self):
        payload = os.urandom(3 * (1 << 20) + 17)
        with tempfile.NamedTemporaryFile() as f:
            f.write(payload)
            f.flush()
            target = bytearray(len(payload))
            with memoryview(target) as view:
                feeder._read_shard(feeder.Shard(0, f.name, len(payload)), view)
        self.assertEqual(bytes(target), payload)

    def test_short_shard_is_an_error(self):
        with tempfile.NamedTemporaryFile() as f:
            with memoryview(bytearray(8)) as view:
                with self.assertRaises(feeder.WeightFeederError):
                    feeder._read_shard(feeder.Shard(0, f.name, 8), view)

    def test_tmpfs_too_small_for_a_shard_keeps_the_core_iterator(self):
        original = MagicMock(return_value=iter(()))
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"12345")
            f.flush()
            with (
                patch(
                    "sglang_tpu.weight_feeder.shutil.disk_usage",
                    return_value=MagicMock(total=1),
                ),
            ):
                weight_feeder_load_local_weights(original, [f.name])
        original.assert_called_once()

    def test_unwritable_staging_keeps_the_core_iterator(self):
        original = MagicMock(return_value=iter(()))
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"12345")
            f.flush()
            with (
                patch("sglang_tpu.weight_feeder.os.access", return_value=False),
            ):
                weight_feeder_load_local_weights(original, [f.name])
        original.assert_called_once()

    def test_relative_local_paths_become_absolute(self):
        with tempfile.NamedTemporaryFile(dir=".") as f:
            shard = host_weight_feeder._list_shards([os.path.basename(f.name)])[0]
        self.assertTrue(os.path.isabs(shard.uri))

    def test_local_shards_take_their_size_from_the_file(self):
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"12345")
            f.flush()
            self.assertEqual(host_weight_feeder._list_shards([f.name])[0].size, 5)

    def test_window_stays_charged_until_deleted_shards_are_freed(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        staging = feeder.StagingDir(tmp)
        shards = [feeder.Shard(i, f"s{i}", 60) for i in range(2)]
        weight_feeder = feeder.WeightFeeder(staging, shards, 1, 100, 1)
        with patch.object(feeder, "_REAP_DELAY_SECONDS", 0.3):
            self.assertEqual(weight_feeder._claim_next_shard().index, 0)
            with open(staging.shard_path(0), "w"):
                pass
            staging.mark_consumed(0)
            weight_feeder._release_consumed_shards()
            start = time.monotonic()
            self.assertEqual(weight_feeder._claim_next_shard().index, 1)
        self.assertGreaterEqual(time.monotonic() - start, 0.25)

    def test_consumed_shard_is_not_left_mapped(self):
        """A rank waiting for the next shard must not pin the one it consumed."""
        tmp = tempfile.mkdtemp(dir="/dev/shm")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        staging = feeder.StagingDir(tmp)
        for index in range(2):
            tensors = {f"s{index}-{i}": torch.full((4,), float(i)) for i in range(2)}
            safetensors.torch.save_file(tensors, staging.shard_path(index))

        def next_ready_shard(staging, remaining, feeder):
            if remaining == [0, 1]:
                return 0
            with open("/proc/self/maps") as maps:
                self.assertNotIn(staging.shard_path(0), maps.read())
            self.assertEqual(staging.count_consumed_ranks(os.listdir(tmp), 0), 1)
            return 1

        loaded = {}
        with patch.object(
            host_weight_feeder, "_wait_for_ready_shard", next_ready_shard
        ):
            for name, tensor in host_weight_feeder._iter_staged_weights(
                staging, [0, 1], None
            ):
                loaded[name] = tensor[0].item()
        self.assertEqual(loaded, {"s0-0": 0.0, "s0-1": 1.0, "s1-0": 0.0, "s1-1": 1.0})

    def test_timeout_reads_remaining_shards_per_rank(self):
        tmp = tempfile.mkdtemp(dir="/dev/shm")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        staging = feeder.StagingDir(tmp)
        safetensors.torch.save_file({"staged": torch.ones(1)}, staging.shard_path(0))
        with open(staging.ready_path(0), "w"):
            pass
        read_shards = MagicMock(return_value=iter([("fallback", torch.ones(1))]))
        with (
            patch.object(
                host_weight_feeder, "_rotate_shards_for_host", lambda uris: uris
            ),
            patch.object(host_weight_feeder, "_replica_id", return_value=0),
            patch.object(host_weight_feeder, "_open_staging_dir", return_value=staging),
            patch.object(
                host_weight_feeder, "_try_acquire_owner_lock", return_value=None
            ),
            patch.object(
                host_weight_feeder,
                "_wait_for_ready_shard",
                side_effect=[0, feeder.WeightFeederTimeout("stalled")],
            ),
        ):
            names = [
                name
                for name, _ in host_weight_feeder.shared_weights_iterator(
                    ["a", "b"], read_shards
                )
            ]
        self.assertEqual(names, ["staged", "fallback"])
        read_shards.assert_called_once_with(["b"])
        with self.assertRaises(feeder.WeightFeederTimeout):
            staging.raise_if_failed()

    def test_cleaning_up_a_failed_feeder_keeps_its_error(self):
        tmp = tempfile.mkdtemp(dir="/dev/shm")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        staging = feeder.StagingDir(tmp)
        process = host_weight_feeder._WeightFeederProcess(
            staging, [feeder.Shard(0, "/missing.safetensors", 1)], 1
        )
        self.assertNotEqual(process.wait(), 0)
        process.kill()
        process.finish()
        with self.assertRaises(feeder.WeightFeederError):
            staging.raise_if_failed()

    def _staged_shard(self):
        """A staging directory and a one-byte shard to stage into it."""
        tmp = tempfile.mkdtemp(dir="/dev/shm")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        source = os.path.join(tmp, "source")
        with open(source, "wb") as f:
            f.write(b"x")
        return feeder.StagingDir(tmp), feeder.Shard(0, source, 1)

    def test_stalled_ranks_end_the_load_with_a_timeout(self):
        staging, shard = self._staged_shard()
        weight_feeder = feeder.WeightFeeder(staging, [shard], 1, 100, 1)
        with patch.object(feeder, "STALL_TIMEOUT_SECONDS", 0.0):
            weight_feeder.run()
        self.assertTrue(weight_feeder.failed)
        with self.assertRaises(feeder.WeightFeederTimeout):
            staging.raise_if_failed()

    def test_timeout_does_not_wait_for_a_stuck_reader(self):
        staging, shard = self._staged_shard()
        weight_feeder = feeder.WeightFeeder(staging, [shard], 1, 100, 1)
        stuck = threading.Event()
        self.addCleanup(stuck.set)
        with (
            patch.object(feeder, "_read_shard", lambda shard, view: stuck.wait()),
            patch.object(feeder, "STALL_TIMEOUT_SECONDS", 0.0),
        ):
            runner = threading.Thread(target=weight_feeder.run, daemon=True)
            runner.start()
            runner.join(timeout=10)
        self.assertFalse(runner.is_alive())
        self.assertTrue(weight_feeder.failed)

    def test_abort_frees_shards_when_the_report_fails(self):
        staging, shard = self._staged_shard()
        shutil.copy(shard.uri, staging.shard_path(0))
        report = staging.fail
        attempts = []

        def failing_once(message):
            attempts.append(message)
            if len(attempts) == 1:
                raise OSError("No space left on device")
            report(message)

        with patch.object(staging, "fail", failing_once):
            staging.abort("Weight feeder owner exited")
        self.assertFalse(os.path.exists(staging.shard_path(0)))
        with self.assertRaisesRegex(feeder.WeightFeederError, "owner exited"):
            staging.raise_if_failed()

    def test_owner_death_frees_staging_and_reports_it(self):
        staging, shard = self._staged_shard()
        process = host_weight_feeder._WeightFeederProcess(staging, [shard], 1)
        while not os.path.exists(staging.ready_path(0)):
            time.sleep(0.01)
        process._alive_pipe.close()
        self.assertNotEqual(process.wait(), 0)
        self.assertFalse(os.path.exists(staging.shard_path(0)))
        with self.assertRaisesRegex(feeder.WeightFeederError, "owner exited"):
            staging.raise_if_failed()

    def _write_checkpoint(self):
        tmp = tempfile.mkdtemp(dir="/dev/shm")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        expected, paths = {}, []
        for shard in range(NUM_SHARDS):
            tensors = {f"s{shard}": torch.full((64, 64), float(shard))}
            expected.update(tensors)
            paths.append(os.path.join(tmp, f"m-{shard:05d}.safetensors"))
            safetensors.torch.save_file(tensors, paths[-1])
        return tmp, expected, paths

    def test_every_engine_prefetch_targets_the_first_load_of_its_ranks(self):
        tmp, _, paths = self._write_checkpoint()
        staging_root = os.path.join(tmp, "staging")
        with (
            patch.object(host_weight_feeder, "_STAGING_ROOT", staging_root),
            patch.object(
                host_weight_feeder, "_checkpoint_shard_paths", lambda path: paths
            ),
            patch.object(host_weight_feeder, "_WeightFeederProcess") as process,
            _engine_on_local_path(tmp, NUM_RANKS),
        ):
            process.return_value.finish.side_effect = lambda: shutil.rmtree(
                process.call_args.args[0].path
            )
            for _ in range(2):
                host_weight_feeder.start_weight_prefetch()
                while os.listdir(staging_root):
                    time.sleep(0.01)
            rank = host_weight_feeder._open_staging_dir(
                host_weight_feeder._rotate_shards_for_host(paths), os.getpid(), 0
            )
        engine_paths = [call.args[0].path for call in process.call_args_list]
        self.assertEqual(engine_paths, [rank.path, rank.path])

    def test_staging_on_disk_keeps_the_core_iterator(self):
        original = MagicMock(return_value=iter(()))
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"12345")
            f.flush()
            with (
                patch("sglang_tpu.weight_feeder._filesystem_type", return_value="ext4"),
            ):
                weight_feeder_load_local_weights(original, [f.name])
        original.assert_called_once()

    def test_ranks_read_the_feed_their_engine_started(self):
        tmp, expected, paths = self._write_checkpoint()
        staging_root = os.path.join(tmp, "staging")
        context = multiprocessing.get_context("fork")
        results = context.Queue()
        engine = context.Process(
            target=_prefetch_from_engine, args=(tmp, paths, NUM_RANKS, results)
        )
        with patch.object(host_weight_feeder, "_STAGING_ROOT", staging_root):
            engine.start()
            outcomes = [results.get(timeout=60)[1] for _ in range(NUM_RANKS)]
            engine.join(timeout=60)
        for outcome in outcomes:
            self.assertIsInstance(outcome, dict, outcome)
            self.assertEqual(set(outcome), set(expected))

    def test_every_rank_gets_all_tensors_and_staging_is_removed(self):
        tmp, expected, paths = self._write_checkpoint()
        staging_root = os.path.join(tmp, "staging")
        context = multiprocessing.get_context("fork")
        results = context.Queue()
        ranks = [
            context.Process(target=_load_rank, args=(paths, NUM_RANKS, results))
            for _ in range(NUM_RANKS)
        ]
        with patch.object(host_weight_feeder, "_STAGING_ROOT", staging_root):
            for rank in ranks:
                rank.start()
            outcomes = [results.get(timeout=60)[1] for _ in ranks]
            for rank in ranks:
                rank.join(timeout=30)
        for outcome in outcomes:
            self.assertIsInstance(outcome, dict, outcome)
            self.assertEqual(
                outcome, {name: t.tolist() for name, t in expected.items()}
            )
        self.assertEqual(os.listdir(staging_root), [])


if __name__ == "__main__":
    unittest.main()
