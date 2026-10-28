"""Tests for host-side placement of multimodal embeddings on TPU.

The vision tower already returns host tensors, so TPU places the rows at their
final positions on the host and lets the device do a plain select. That replaces
masked_scatter_, whose TPU fallback ran a cumsum over num_tokens x hidden just to
rediscover the row mapping. These tests pin the equivalence that swap relies on.

    python -m pytest test/registered/unit/managers/test_mm_host_embed_placement.py -v
"""

import unittest

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.mm_embed_tpu import embed_mm_on_host

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROWS = 12
HIDDEN = 4


def _tpu_masked_scatter(dest, mask, source):
    """The TPU kernel override from hardware_backend/tpu/patches.py.

    Reproduced here because the real one is registered for PrivateUse1, and the
    new host path has to match it, not just the native op.
    """
    src_flat = source.to(dest.dtype).reshape(-1)
    flat_mask = mask.to(torch.bool).broadcast_to(dest.shape).reshape(-1)
    idx = (torch.cumsum(flat_mask.to(torch.int32), 0, dtype=torch.int32) - 1).clamp(
        min=0
    )
    return torch.where(flat_mask, src_flat[idx], dest.reshape(-1)).reshape(dest.shape)


def _random_mask(num_set: int) -> torch.Tensor:
    mask = torch.zeros(ROWS, 1, dtype=torch.bool)
    mask[torch.randperm(ROWS)[:num_set]] = True
    return mask


def _embed_only(**kwargs) -> torch.Tensor:
    embeds, deepstack = embed_mm_on_host(deepstack_shape=None, **kwargs)
    assert deepstack is None
    return embeds


class TestHostEmbedPlacement(CustomTestCase):
    def test_host_placement_matches_masked_scatter(self):
        """The equivalence the rewrite rests on, against both implementations.

        masked_scatter_ consumes the source positionally in mask order; host
        placement instead indexes rows directly. A regression that mixed up that
        ordering -- say by sorting indices differently or filling row-major --
        diverges here even though both still produce a well-formed matrix.
        """
        torch.manual_seed(0)
        for num_mm_tokens in (1, 3, 7, ROWS):
            with self.subTest(num_mm_tokens=num_mm_tokens):
                dest = torch.randn(ROWS, HIDDEN)
                mask = _random_mask(num_mm_tokens)
                src = torch.randn(num_mm_tokens, HIDDEN)

                got = _embed_only(
                    input_embeds=dest,
                    modalities=[Modality.IMAGE],
                    embeddings=[src],
                    masks=[mask],
                    deepstack_embeddings=[None],
                    use_deepstack={},
                )

                native = dest.clone()
                native.masked_scatter_(mask.expand_as(native), src)
                self.assertTrue(torch.equal(got, native))
                self.assertTrue(
                    torch.equal(
                        got, _tpu_masked_scatter(dest, mask.expand_as(dest), src)
                    )
                )

    def test_modalities_merge_into_one_transfer(self):
        """Two modalities collapse to one matrix and one union mask.

        The device op count must not scale with modality count, so the merge
        happens on the host. If the union mask dropped a modality, or a later
        modality overwrote an earlier one's rows, this diverges from applying
        masked_scatter_ once per modality.
        """
        dest = torch.randn(ROWS, HIDDEN)
        first = torch.zeros(ROWS, 1, dtype=torch.bool)
        first[[1, 4, 9]] = True
        second = torch.zeros(ROWS, 1, dtype=torch.bool)
        second[[0, 7]] = True
        src_first = torch.randn(3, HIDDEN)
        src_second = torch.randn(2, HIDDEN)

        got = _embed_only(
            input_embeds=dest,
            modalities=[Modality.IMAGE, Modality.VIDEO],
            embeddings=[src_first, src_second],
            masks=[first, second],
            deepstack_embeddings=[None, None],
            use_deepstack={},
        )

        expected = dest.clone()
        expected.masked_scatter_(first.expand_as(expected), src_first)
        expected.masked_scatter_(second.expand_as(expected), src_second)
        self.assertTrue(torch.equal(got, expected))

    def test_row_count_mismatch_is_rejected(self):
        """masked_scatter_ tolerated an over-long source; placement must not.

        The old path silently ignored source rows past mask.sum(), so a
        miscounted embedding stayed invisible. Host placement indexes by
        position, where the same mistake would misalign every later row.
        """
        mask = _random_mask(3)
        for rows in (2, 4):
            with self.subTest(embedding_rows=rows):
                with self.assertRaises(ValueError):
                    _embed_only(
                        input_embeds=torch.randn(ROWS, HIDDEN),
                        modalities=[Modality.IMAGE],
                        embeddings=[torch.randn(rows, HIDDEN)],
                        masks=[mask],
                        deepstack_embeddings=[None],
                        use_deepstack={},
                    )

    def test_deepstack_grid_matches_zeros_plus_scatter(self):
        """Binding the host deepstack matrix must equal zeros-then-scatter.

        The deepstack grid skips the device merge entirely -- the transferred
        host matrix is used as-is -- which is only sound because its non-
        multimodal rows are zero. A non-zero fill would silently corrupt them.
        """
        deepstack_hidden = HIDDEN * 3
        mask = _random_mask(5)
        deepstack_src = torch.randn(5, deepstack_hidden)

        _, got_deepstack = embed_mm_on_host(
            input_embeds=torch.randn(ROWS, HIDDEN),
            deepstack_shape=torch.Size((ROWS, deepstack_hidden)),
            modalities=[Modality.IMAGE],
            embeddings=[torch.randn(5, HIDDEN)],
            masks=[mask],
            deepstack_embeddings=[deepstack_src],
            use_deepstack={Modality.IMAGE: True},
        )

        expected = torch.zeros(ROWS, deepstack_hidden)
        expected.masked_scatter_(mask.expand_as(expected), deepstack_src)
        self.assertTrue(torch.equal(got_deepstack, expected))

    def test_deepstack_falls_back_to_zeros_without_multimodal_rows(self):
        """No modality placed rows, so the caller still needs a zeroed grid.

        Placement returns nothing here, and dropping the fallback would hand the
        model None for input_deepstack_embeds instead of zeros.
        """
        dest = torch.randn(ROWS, HIDDEN)
        deepstack_hidden = HIDDEN * 3

        got_embeds, got_deepstack = embed_mm_on_host(
            input_embeds=dest,
            deepstack_shape=torch.Size((ROWS, deepstack_hidden)),
            modalities=[Modality.IMAGE],
            embeddings=[None],
            masks=[None],
            deepstack_embeddings=[None],
            use_deepstack={Modality.IMAGE: True},
        )

        self.assertTrue(torch.equal(got_embeds, dest))
        self.assertTrue(torch.equal(got_deepstack, torch.zeros(ROWS, deepstack_hidden)))

    def test_reused_deepstack_grid_drops_rows_of_the_previous_batch(self):
        """The host grid is reused across prefills; rows written for one batch must not leak into the next."""
        deepstack_hidden = HIDDEN * 3

        def place(mask, deepstack_src):
            _, got = embed_mm_on_host(
                input_embeds=torch.randn(ROWS, HIDDEN),
                deepstack_shape=torch.Size((ROWS, deepstack_hidden)),
                modalities=[Modality.IMAGE],
                embeddings=[torch.randn(deepstack_src.shape[0], HIDDEN)],
                masks=[mask],
                deepstack_embeddings=[deepstack_src],
                use_deepstack={Modality.IMAGE: True},
            )
            return got.clone()

        place(_random_mask(7), torch.randn(7, deepstack_hidden))
        mask = _random_mask(3)
        deepstack_src = torch.randn(3, deepstack_hidden)
        got = place(mask, deepstack_src)

        expected = torch.zeros(ROWS, deepstack_hidden)
        expected.masked_scatter_(mask.expand_as(expected), deepstack_src)
        self.assertTrue(torch.equal(got, expected))


if __name__ == "__main__":
    unittest.main()
