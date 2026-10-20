"""Video frames are pinned only on platforms that support pinned host memory."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from sglang.srt.utils import video_decoder
from sglang.srt.utils.video_decoder import VideoDecoderWrapper
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _decord_wrapper(frames: np.ndarray) -> VideoDecoderWrapper:
    wrapper = VideoDecoderWrapper.__new__(VideoDecoderWrapper)
    wrapper._num_decode_threads = 1
    decoder = MagicMock()
    decoder.get_batch.return_value.asnumpy.return_value = frames
    wrapper._decoder = decoder
    return wrapper


class TestVideoDecoderPinning(CustomTestCase):
    def _frames(self, *, pin_available: bool):
        wrapper = _decord_wrapper(np.zeros((2, 4, 4, 3), dtype=np.uint8))
        with (
            patch.object(video_decoder, "_BACKEND", "decord"),
            patch(
                "sglang.srt.utils.common.is_pin_memory_available",
                return_value=pin_available,
            ),
            patch.object(torch.Tensor, "pin_memory") as pin,
        ):
            wrapper.get_frames_as_tensor([0, 1])
        return pin

    def test_no_pin_without_pinned_host_memory(self):
        # Pinning there asks a device runtime for a host allocator that the
        # tokenizer process never initialized, which failed every video request.
        self._frames(pin_available=False).assert_not_called()

    def test_pin_when_the_platform_supports_it(self):
        self._frames(pin_available=True).assert_called_once()


if __name__ == "__main__":
    unittest.main()
