# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
import enum

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sglang_tpu.attention.pallas.batched_rpa import utils


@dataclasses.dataclass(frozen=True)
class BlockSizes:
    """Tuning parameters for the RPA kernel."""

    bq_sz: int
    bq_c_sz: int
    bkv_sz: int
    batch_size: int
    n_buffer: int


@dataclasses.dataclass(frozen=True)
class ModelConfigs:
    """Model config that will always stay constant."""

    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    mask_value: float
    sm_scale: float = 1.0
    soft_cap: float | None = None
    sliding_window: int | None = None
    has_attention_sink: bool = False

    @property
    def num_q_heads_per_kv_head(self) -> int:
        return self.num_q_heads // self.num_kv_heads


class KVLayout(enum.StrEnum):
    """Represents the different layouts for KV cache.

    - HEAD_ALONG_SUBLANE: Number of heads on sublane, head_dim on lane.
    - SEQ_ALONG_LANE: Sequence is packed along the lane, head_dim on sublane.
    """

    HEAD_ALONG_SUBLANE = enum.auto()
    SEQ_ALONG_LANE = enum.auto()


@dataclasses.dataclass(frozen=True)
class ServingConfigs:
    """Serving config that can change depending on use cases."""

    num_seqs: int
    page_size: int
    total_q_tokens: int
    num_page_indices: int
    dtype_q: jnp.dtype
    dtype_kv: jnp.dtype
    dtype_out: jnp.dtype
    scale_q: int | None = None
    scale_k: int | None = None
    scale_v: int | None = None
    kv_layout: KVLayout = KVLayout.HEAD_ALONG_SUBLANE
    smem_fraction_limit_for_schedule_generation: float = 0.33
    max_schedule_size_multiplier: int = 16
    # Context-parallel modes; see `wrapper.ragged_paged_attention`.
    cache_only: bool = False
    skip_kv_write: bool = False
    return_lse: bool = False
    cp_group_size: int | None = None
    # Prefill tiles whose every key precedes every query (and lies inside the
    # sliding window) run without the mask, and a power-of-two sm_scale is folded
    # into q; both are exact.
    skip_full_tile_mask: bool = False
    # Softmax variants (env-gated in the wrapper): exp2 with log2e folded into the
    # q scale; row sums from ones rows appended to V (l rides in the accumulator).
    exp2_softmax: bool = False
    ones_v: bool = False
    # Softmax without the running row max (env-gated in the wrapper): scores must be
    # bounded so exp cannot overflow, e.g. QK-normed models.
    no_max: bool = False
    # Prefill tiles with transposed scores and fp8 q (see transposed.py);
    # env-gated in the wrapper, applied where `use_transposed` holds.
    transposed: bool = False
    # Transposed tiles skip the compute blocks past a partial q block's last
    # query (a sequence's last block) instead of scoring padded rows; exact.
    skip_pad_q: bool = False
    # Per-head RMS norm of the output rows (weightless, this eps), applied to the
    # output-dtype values as a model's following norm would (Muse Spark).
    out_head_norm_eps: float | None = None

    @property
    def pages_per_seq(self) -> int:
        return self.num_page_indices // self.num_seqs

    @property
    def page_size_log2(self) -> int:
        return (self.page_size - 1).bit_length()

    @property
    def page_size_mask(self) -> int:
        return self.page_size - 1

    @property
    def packing_q(self) -> int:
        return utils.get_dtype_packing(self.dtype_q)

    @property
    def packing_kv(self) -> int:
        return utils.get_dtype_packing(self.dtype_kv)


class RpaCase(enum.StrEnum):
    """Represents the different cases for Ragged Paged Attention.

    - DECODE: Sequences are in decode-only mode (q_len = 1).
    - PREFILL: Sequences are in prefill-only mode (q_len > 1, static).
    - MIXED: Sequences can be a mix of prefill and decode (q_len > 1, dynamic).
    """

    DECODE = enum.auto()
    PREFILL = enum.auto()
    MIXED = enum.auto()

    @property
    def symbol(self):
        return {
            RpaCase.DECODE: "d",
            RpaCase.PREFILL: "p",
            RpaCase.MIXED: "m",
        }[self]

    def get_range(
        self, distribution: jax.Array
    ) -> tuple[jax.typing.ArrayLike, jax.typing.ArrayLike]:
        assert distribution.shape == (3,)
        match self:
            case RpaCase.DECODE:
                return 0, distribution[0]
            case RpaCase.PREFILL:
                return distribution[0], distribution[1]
            case RpaCase.MIXED:
                return distribution[1], distribution[2]


@dataclasses.dataclass(frozen=True, eq=True)
class RpaConfigs:
    block: BlockSizes
    model: ModelConfigs
    serve: ServingConfigs
    mode: RpaCase
    vmem_limit_bytes: int

    # Expose block sizes for ease of use.

    @property
    def bq_sz(self) -> int:
        return self.block.bq_sz

    @property
    def bq_c_sz(self) -> int:
        return self.block.bq_c_sz

    @property
    def bkv_sz(self) -> int:
        return self.block.bkv_sz

    @property
    def batch_size(self) -> int:
        return self.block.batch_size

    @property
    def n_buffer(self) -> int:
        return self.block.n_buffer

    # Define derived values.

    @property
    def max_steps_ub(self) -> int:
        """Get maximum upper bound of kernel steps based on SMEM limit."""

        fixed_bytes = 0
        fixed_bytes += self.serve.num_seqs  # kv_lens
        fixed_bytes += self.serve.num_seqs + 1  # cu_q_lens
        fixed_bytes += self.serve.num_seqs * self.serve.pages_per_seq  # page_indices
        fixed_bytes += 3  # distribution
        fixed_bytes += self.block.batch_size  # lane_lengths
        fixed_bytes += 1  # actual_steps
        if self.serve.cp_group_size is not None:
            fixed_bytes += self.serve.num_seqs + 1  # kv_lens_global, cp_rank

        word_size_bytes = 4
        fixed_bytes *= word_size_bytes

        smem_limit_bytes = (
            pltpu.get_tpu_info().smem_capacity_bytes - 32 * 1024
        ) * self.serve.smem_fraction_limit_for_schedule_generation
        available_bytes = smem_limit_bytes - fixed_bytes

        # Per step per batch item:
        # s_idx, q_idx, k_idx, is_last_k, do_writeback: 5 * 4 = 20
        # dma_q: 2 * 4 = 8
        # dma_kv_cache: bkv_p_cache * 3 * 4 = 12 * bkv_p_cache
        # dma_kv_new: bkv_p_new * self.dma_kv_new_size * 4
        bytes_per_step = (
            28 + 12 * self.bkv_p_cache + 4 * self.dma_kv_new_size * self.bkv_p_new
        )
        bytes_per_step *= self.block.batch_size
        # Add 16 bytes for the 4 total_wait fields (total_wait_kv_in, total_wait_kv_out,
        # total_wait_q_in, total_wait_o_out) which are 1D arrays (not multiplied by batch_size).
        bytes_per_step += 16

        max_steps_ub = available_bytes // bytes_per_step

        num_lanes = pltpu.get_tpu_info().num_lanes
        max_steps_ub = max(1, max_steps_ub // num_lanes) * num_lanes
        return int(max_steps_ub)

    @property
    def bkv_p(self) -> int:
        return pl.cdiv(self.block.bkv_sz, self.serve.page_size)

    @property
    def bkv_p_cache(self) -> int:
        if self.mode == RpaCase.PREFILL:
            return 0
        return self.bkv_p

    @property
    def bkv_p_new(self) -> int:
        if self.mode == RpaCase.DECODE or self.block.bq_sz == 1:
            return 1
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return self.bkv_p + 1
        return self.bkv_p

    @property
    def bkv_stride(self) -> int:
        bkv_stride = pl.cdiv(self.model.num_kv_heads * 2, self.serve.packing_kv)

        if utils.has_bank_conflicts(bkv_stride):
            bkv_stride += 1
        return bkv_stride

    @property
    def aligned_q_head_dim(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        return utils.align_to(self.model.head_dim, num_lanes)

    @property
    def aligned_kv_head_dim(self) -> int:
        num_lanes = pltpu.get_tpu_info().num_lanes
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        kv_packing = utils.get_dtype_packing(self.serve.dtype_kv)
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return utils.align_to(self.model.head_dim, num_sublanes * kv_packing)
        return utils.align_to(self.model.head_dim, num_lanes)

    @property
    def aligned_num_kv_heads_x2(self) -> int:
        packing_kv = self.serve.packing_kv
        return utils.align_to(self.model.num_kv_heads * 2, packing_kv)

    @property
    def aligned_num_q_heads_per_kv_head(self) -> int:
        packing_q = self.serve.packing_q
        return utils.align_to(self.model.num_q_heads_per_kv_head, packing_q)

    @property
    def kv_hbm_stride(self) -> int:
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return self.model.num_kv_heads * 2
        kv_packing = utils.get_dtype_packing(self.serve.dtype_kv)
        return utils.align_to(self.model.num_kv_heads * 2, kv_packing) // kv_packing

    @property
    def fuse_accum(self) -> bool:
        return self.mode == RpaCase.DECODE

    @property
    def kv_bytes_per_token(self) -> int:
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            num_elements = self.model.num_kv_heads * 2 * self.aligned_kv_head_dim
        else:
            num_elements = self.aligned_num_kv_heads_x2 * self.aligned_kv_head_dim
        return num_elements * self.serve.dtype_kv.itemsize

    @property
    def q_bytes_per_token(self) -> int:
        return (
            self.model.num_kv_heads
            * self.aligned_num_q_heads_per_kv_head
            * self.aligned_q_head_dim
            * self.serve.dtype_q.itemsize
        )

    @property
    def o_bytes_per_token(self) -> int:
        return (
            self.model.num_kv_heads
            * self.aligned_num_q_heads_per_kv_head
            * self.aligned_q_head_dim
            * self.serve.dtype_out.itemsize
        )

    @property
    def q_vmem_shape(self):
        q_per_kv_packing = self.aligned_num_q_heads_per_kv_head // self.serve.packing_q
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz,
            q_per_kv_packing,
            self.serve.packing_q,
            self.aligned_q_head_dim,
        )

    @property
    def kv_vmem_shape(self):
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return (
                self.block.batch_size,
                self.model.num_kv_heads * 2,
                self.aligned_kv_head_dim // self.serve.packing_kv,
                self.serve.packing_kv,
                self.block.bkv_sz + 2 * self.serve.page_size,
            )
        return (
            self.block.batch_size,
            self.block.bkv_sz,
            self.bkv_stride,
            self.serve.packing_kv,
            self.aligned_kv_head_dim,
        )

    @property
    def dma_kv_new_size(self) -> int:
        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            return 5
        return 4

    @property
    def lm_scratch_shape(self):
        num_lanes = pltpu.get_tpu_info().num_lanes
        return (
            self.model.num_kv_heads,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
            num_lanes,
        )

    @property
    def lse_vmem_shape(self):
        # One f32 row of lanes per (KV head, token); lane g is aligned q head g.
        num_lanes = pltpu.get_tpu_info().num_lanes
        return (
            self.block.batch_size,
            self.model.num_kv_heads,
            self.block.bq_sz,
            1,
            num_lanes,
        )

    @property
    def lse_stage_shape(self):
        # One KV head's (token, aligned q head) rows of `lm_scratch_shape`.
        num_lanes = pltpu.get_tpu_info().num_lanes
        return (self.block.bq_sz * self.aligned_num_q_heads_per_kv_head, num_lanes)

    @property
    def acc_scratch_shape(self):
        # With ones_v the accumulator is a full vreg wide: the head in the low
        # lanes, the row sum l in the lanes above it.
        width = self.acc_width if self.serve.ones_v else self.aligned_kv_head_dim
        return (
            self.model.num_kv_heads,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
            width,
        )

    @property
    def acc_width(self) -> int:
        return pltpu.get_tpu_info().num_lanes

    @property
    def use_transposed(self) -> bool:
        # One sequence per tile, compute blocks of whole lane vregs, the head
        # inside half a vreg so O^T pads to one lane tile at the store.
        num_lanes = pltpu.get_tpu_info().num_lanes
        return (
            self.serve.transposed
            and self.mode != RpaCase.DECODE
            and self.batch_size == 1
            and self.bq_sz > 1
            and self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE
            and (self.bq_c_sz * self.aligned_num_q_heads_per_kv_head) % num_lanes == 0
            and self.aligned_kv_head_dim * 2 <= num_lanes
        )

    @property
    def mt_scratch_shape(self):
        return (
            self.model.num_kv_heads,
            1,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
        )

    @property
    def acct_scratch_shape(self):
        # O^T rows, then one bf16 sublane tile of row sums (ones rows in V^T).
        num_sublanes = pltpu.get_tpu_info().num_sublanes
        return (
            self.model.num_kv_heads,
            self.aligned_kv_head_dim + 2 * num_sublanes,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
        )

    @property
    def qt_scratch_shape(self):
        return (
            self.model.num_kv_heads,
            self.aligned_kv_head_dim,
            self.block.bq_sz * self.aligned_num_q_heads_per_kv_head,
        )

    @property
    def max_schedule_size_multiplier(self) -> int:
        # By default, setting an upper bound of ~5x the SMEM capacity on schedule.
        # Grown to the static step bound: a long prefill padded to many rows (whose
        # page table shrinks max_steps_ub) needs more steps than 16 SMEM chunks, and
        # the schedule kernel flushes chunks past the HBM buffer end without a check.
        return max(
            self.serve.max_schedule_size_multiplier,
            -(-(self.static_max_steps + 1) // self.max_steps_ub),
        )

    @property
    def static_max_steps(self) -> int:
        """Upper bound on schedule steps for any input of these static shapes."""
        num_k = -(
            -(self.serve.pages_per_seq * self.serve.page_size) // self.block.bkv_sz
        )
        if self.mode == RpaCase.DECODE:
            steps = self.serve.num_seqs * num_k
        else:
            num_q = (
                -(-self.serve.total_q_tokens // self.block.bq_sz) + self.serve.num_seqs
            )
            steps = num_q * num_k
        return -(-steps // self.block.batch_size)

    def validate_inputs(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        kv_cache: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        distribution: jax.Array,
        cp_rank: jax.Array | None = None,
        kv_lens_global: jax.Array | None = None,
    ):
        """Validate inputs to the RPA kernel statically."""

        if not q.ndim == k.ndim == v.ndim == 3:
            raise ValueError(
                f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}"
            )
        if k.shape != v.shape:
            raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
        if not (q.shape[0] == k.shape[0] == v.shape[0]):
            raise ValueError(
                "Expected number of sequences in Q, K, and V to be the same, but got"
                f" {q.shape[0]=}, {k.shape[0]=}, and {v.shape[0]=}"
            )
        if not (q.shape[2] == k.shape[2] == v.shape[2]):
            raise ValueError(
                "Expected number of head dimensions in Q, K, and V to be the same,"
                f" but got {q.shape[2]=}, {k.shape[2]=}, and {v.shape[2]=}"
            )

        if self.serve.kv_layout == KVLayout.SEQ_ALONG_LANE:
            if self.serve.page_size % 128 != 0:
                raise ValueError(
                    "Expected page_size to be a multiple of 128 for SEQ_ALONG_LANE"
                    f" tile alignment, but got {self.serve.page_size=}"
                )
            expected_kv_cache_shape = (
                kv_cache.shape[0],
                self.model.num_kv_heads * 2,
                self.aligned_kv_head_dim // self.serve.packing_kv,
                self.serve.packing_kv,
                self.serve.page_size,
            )
        else:
            expected_kv_cache_shape = (
                kv_cache.shape[0],
                self.serve.page_size,
                self.aligned_num_kv_heads_x2 // self.serve.packing_kv,
                self.serve.packing_kv,
                self.aligned_kv_head_dim,
            )

        if kv_cache.shape != expected_kv_cache_shape:
            raise ValueError(
                f"Expected {kv_cache.shape=} to be equal to {expected_kv_cache_shape=}"
            )

        # Integer kv quantization is currently not supported.
        if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
            raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")
        if not (kv_cache.dtype == k.dtype == v.dtype):
            raise ValueError(
                "Expected KV cache dtype and K/V dtype to be the same, but got"
                f" {kv_cache.dtype=}, {k.dtype=}, and {v.dtype=}"
            )

        if not (
            jnp.int32
            == kv_lens.dtype
            == page_indices.dtype
            == cu_q_lens.dtype
            == distribution.dtype
        ):
            raise ValueError(
                f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
                f" {cu_q_lens.dtype=}, {distribution.dtype=}"
            )

        if not (kv_lens.ndim == page_indices.ndim == cu_q_lens.ndim == 1):
            raise ValueError(
                f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=},"
                f" {cu_q_lens.shape=}"
            )

        max_num_seqs = kv_lens.shape[0]
        num_page_indices = page_indices.shape[0]
        if num_page_indices % max_num_seqs != 0:
            raise ValueError(
                f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}."
            )
        if cu_q_lens.shape != (max_num_seqs + 1,):
            raise ValueError(
                f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},)."
            )
        if distribution.shape != (3,):
            raise ValueError(f"Expected {distribution.shape=} to be (3,).")
        if self.bkv_p_cache and self.block.bkv_sz % self.serve.page_size:
            # Block k's cached pages start at k * cdiv(bkv_sz, page_size).
            raise ValueError(
                f"Expected {self.block.bkv_sz=} to be a multiple of"
                f" {self.serve.page_size=} when the kernel reads cached pages."
            )

        if (
            self.serve.cache_only
            and self.serve.kv_layout != KVLayout.HEAD_ALONG_SUBLANE
        ):
            # SEQ_ALONG_LANE fetches whole pages, so the stale tail of a shard's
            # last page would reach p * v unmasked.
            raise NotImplementedError("cache_only requires HEAD_ALONG_SUBLANE.")
        num_lanes = pltpu.get_tpu_info().num_lanes
        if self.serve.return_lse and self.aligned_num_q_heads_per_kv_head > num_lanes:
            raise ValueError(
                f"return_lse packs {self.aligned_num_q_heads_per_kv_head=} q heads"
                f" into {num_lanes} lanes."
            )
        if self.serve.cp_group_size is not None:
            if self.serve.cp_group_size < 1:
                raise ValueError(f"Expected {self.serve.cp_group_size=} >= 1.")
            if not self.serve.cache_only:
                raise ValueError("cp_group_size requires cache_only.")
            if cp_rank is None or kv_lens_global is None:
                raise ValueError("cp_group_size requires cp_rank and kv_lens_global.")
            if cp_rank.shape != (1,) or cp_rank.dtype != jnp.int32:
                raise ValueError(f"Expected int32 {cp_rank.shape=} to be (1,).")
            if kv_lens_global.shape != kv_lens.shape or (
                kv_lens_global.dtype != jnp.int32
            ):
                raise ValueError(
                    f"Expected int32 {kv_lens_global.shape=} to be {kv_lens.shape}."
                )
        elif cp_rank is not None or kv_lens_global is not None:
            raise ValueError("cp_rank and kv_lens_global require cp_group_size.")
