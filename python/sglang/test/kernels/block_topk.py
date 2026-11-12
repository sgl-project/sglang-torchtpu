"""Independent NumPy contract oracle for exact FP32 attention index selection."""

import numpy as np


def ordered_bits(values):
    """Unsigned monotone IEEE keys; +0 ranks above -0, NaNs unsupported."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))


def exact_topk(scores, lengths, k):
    """Score descending / logical ID ascending, then ascending IDs and -1 tail."""
    scores = np.asarray(scores, dtype=np.float32)
    result = np.full((scores.shape[0], k), -1, np.int32)
    for row, length in enumerate(np.clip(lengths, 0, scores.shape[1])):
        ids = np.flatnonzero(scores[row, :length] > -np.inf)
        order = np.lexsort((ids, ~ordered_bits(scores[row, ids])))[:k]
        selected = np.sort(ids[order])
        result[row, : selected.size] = selected
    return result


def published_blocks(scores, lengths, count, block_size):
    """Visible block maxima, newest mandatory even when other blocks have +inf."""
    scores = np.asarray(scores, dtype=np.float32)
    if np.any(np.isnan(scores)):
        raise ValueError("NaN scores are unsupported")
    width = scores.shape[1]
    blocks = (width + block_size - 1) // block_size
    maxima = np.full((scores.shape[0], blocks), -np.inf, np.float32)
    for row, length in enumerate(np.clip(lengths, 0, width)):
        for block in range((int(length) + block_size - 1) // block_size):
            values = scores[
                row, block * block_size : min((block + 1) * block_size, length)
            ]
            # Integer-key maximum deliberately distinguishes signed zeros.
            maxima[row, block] = values[np.argmax(ordered_bits(values))]
    result = np.full((scores.shape[0], count), -1, np.int32)
    for row, length in enumerate(np.clip(lengths, 0, width)):
        if not length or not count:
            continue
        latest = (length - 1) // block_size
        maxima[row, latest] = -np.inf
        others = exact_topk(maxima[row : row + 1], [blocks], count - 1)[0]
        selected = np.sort(np.append(others[others >= 0], latest))
        result[row, : selected.size] = selected
    return result


def adversarial_scores(width, k, block_size):
    """Different failure modes per row, batched to share one kernel compile."""
    rng = np.random.default_rng(20261112)
    scores = rng.normal(size=(16, width)).astype(np.float32)
    lengths = np.array(
        [
            -7,
            0,
            1,
            k - 1,
            k,
            k + 1,
            block_size - 1,
            block_size,
            block_size + 1,
            width - 1,
            width,
            width + 7,
            width,
            width,
            width,
            width,
        ],
        np.int32,
    )
    scores[6] = np.arange(width, dtype=np.float32)
    scores[7] = -np.arange(width, dtype=np.float32)
    scores[8] = 1
    # Many blocks have the same maximum; only the lowest logical IDs win ties.
    scores[9] = (np.arange(width) // block_size % 5).astype(np.float32)
    scores[10] = -np.inf
    # Alternating zero signs also lie within each block, not just across blocks.
    scores[11] = np.float32(-0.0)
    scores[11, 1::2] = np.float32(0.0)
    scores[12, ::3] = -np.inf
    scores[13] = 2
    scores[13, -block_size:] = -100  # forced publication must not change token set
    scores[14] = -1
    scores[14, ::block_size] = 3
    scores[15] = -np.inf
    scores[15, -1] = -10
    scores[np.arange(width)[None, :] >= np.clip(lengths, 0, width)[:, None]] = -np.inf
    return scores, lengths
