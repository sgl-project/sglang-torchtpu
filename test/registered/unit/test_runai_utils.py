import unittest
from pathlib import Path
from unittest import mock

from sglang.srt.utils import runai_utils
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="stage-b-test-cpu-intel")


class TestRunaiUtils(CustomTestCase):
    def test_is_runai_obj_uri_s3(self):
        self.assertTrue(is_runai_obj_uri("s3://bucket/model/"))
        self.assertTrue(is_runai_obj_uri("S3://Bucket/Model/"))

    def test_is_runai_obj_uri_gs(self):
        self.assertTrue(is_runai_obj_uri("gs://bucket/model/"))
        self.assertTrue(is_runai_obj_uri("GS://Bucket/Model/"))

    def test_is_runai_obj_uri_az(self):
        self.assertTrue(is_runai_obj_uri("az://container/model/"))
        self.assertTrue(is_runai_obj_uri("AZ://Container/Model/"))

    def test_is_runai_obj_uri_local_paths(self):
        self.assertFalse(is_runai_obj_uri("/path/to/model"))
        self.assertFalse(is_runai_obj_uri("./relative/path"))
        self.assertFalse(is_runai_obj_uri("meta-llama/Llama-3.2-1B"))

    def test_is_runai_obj_uri_other_schemes(self):
        self.assertFalse(is_runai_obj_uri("http://example.com/model"))
        self.assertFalse(is_runai_obj_uri("https://example.com/model"))
        self.assertFalse(is_runai_obj_uri("ftp://example.com/model"))

    def test_is_runai_obj_uri_pathlib(self):
        self.assertFalse(is_runai_obj_uri(Path("/local/model")))

    def test_get_path_deterministic(self):
        path1 = ObjectStorageModel.get_path("s3://bucket/model/")
        path2 = ObjectStorageModel.get_path("s3://bucket/model/")
        self.assertEqual(path1, path2)

    def test_get_path_different_uris(self):
        path1 = ObjectStorageModel.get_path("s3://bucket/model-a/")
        path2 = ObjectStorageModel.get_path("s3://bucket/model-b/")
        self.assertNotEqual(path1, path2)


class RetryError(Exception):
    """Named like google.api_core's: the client gave up on a throttled read."""


class _FakeRunaiObject:
    def __init__(self, failures):
        self.failures = list(failures)
        self.pulls = 0

    def __exit__(self, *exc):
        return False

    def pull_files(self, allow_pattern, ignore_pattern):
        self.pulls += 1
        if self.failures:
            raise self.failures.pop(0)


class TestMetadataPullRetry(CustomTestCase):
    """A metadata pull that GCS throttles past the client's own ~120 s retry
    deadline used to fail the whole server start; it now retries throttling and
    server errors, and still fails fast on anything else."""

    def _pull(self, failures):
        fake = _FakeRunaiObject(failures)

        def init(obj, url):
            obj.dir = "/tmp/model"
            obj._runai_obj = fake

        with (
            mock.patch.object(ObjectStorageModel, "__init__", init),
            mock.patch.object(runai_utils.time, "sleep") as sleep,
        ):
            try:
                return ObjectStorageModel.download_and_get_path("gs://b/m"), fake, sleep
            except Exception as exc:
                exc.fake = fake
                raise

    def test_retries_throttling_then_succeeds(self):
        throttled = RetryError("Timeout of 120.0s exceeded")
        throttled.__cause__ = Exception("Request failed with status code", 429)
        path, fake, sleep = self._pull([throttled, Exception("status", 503)])
        self.assertEqual(path, "/tmp/model")
        self.assertEqual(fake.pulls, 3)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [5.0, 10.0])

    def test_non_transient_error_raises_at_once(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            self._pull([FileNotFoundError("no such bucket")])
        self.assertEqual(ctx.exception.fake.pulls, 1)

    def test_gives_up_at_the_deadline(self):
        with mock.patch.object(runai_utils, "_METADATA_PULL_DEADLINE_S", 7.0):
            with self.assertRaises(RetryError) as ctx:
                self._pull([RetryError("429")] * 5)
        # 7 s allows one 5 s backoff, not the next 10 s one.
        self.assertEqual(ctx.exception.fake.pulls, 2)


if __name__ == "__main__":
    unittest.main()
