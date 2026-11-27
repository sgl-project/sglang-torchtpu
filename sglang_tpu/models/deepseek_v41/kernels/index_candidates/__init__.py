# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 candidate blocks over the packed FP8 index keys.

Candidate-source layers publish each query's best token blocks; later layers
gather and score only those blocks. Dense scoring and top-k are shared from
``sglang_tpu.kernels.index_selection``.
"""
