import os
import tempfile
import unittest

from sglang.multimodal_gen.runtime.entrypoints.cli.generate import (
    verify_generation_output,
)


class TestVerifyGenerationOutput(unittest.TestCase):
    def test_missing_result_exits_nonzero(self):
        with self.assertRaises(SystemExit):
            verify_generation_output(
                results=None, output_file_path=None, num_outputs=1
            )
        with self.assertRaises(SystemExit):
            verify_generation_output(
                results=[], output_file_path=None, num_outputs=1
            )

    def test_missing_output_file_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "never_written.mp4")
            with self.assertRaises(SystemExit):
                verify_generation_output(
                    results=object(), output_file_path=missing, num_outputs=1
                )

    def test_existing_output_file_passes(self):
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            verify_generation_output(
                results=object(), output_file_path=f.name, num_outputs=1
            )

    def test_multi_output_skips_exact_path_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "never_written.mp4")
            verify_generation_output(
                results=object(), output_file_path=missing, num_outputs=4
            )


if __name__ == "__main__":
    unittest.main()
