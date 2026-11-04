# SPDX-License-Identifier: Apache-2.0
"""Static configuration of the K3 megakernel: model dims and rank topology.

Ranks are host-major (``rank = lanes * host + lane``) and ``rank ^ 1`` is the
other TensorCore of the same chip; both hold for torch ranks under
``get_pallas_mesh`` on tpu7x (measured: torch rank -> physical id
``[0, 1, 4, 5, 6, 7, 2, 3]`` on one 2x2x1 host). Every peer set is an XOR
offset, so ``lanes`` and ``hosts`` must be powers of two.
"""

import msgspec

# Pallas collective id of every megakernel call. Ids are one per-process namespace shared by all
# TPU kernels that synchronize across devices. In sglang_tpu: 0 fused-RS MoE
# (moe/pallas/fused_rs), 1 two-level all-reduce (twolevel_allreduce.py), 2 host-split GEMV
# (kernels/host_split_gemv.py), 5 one-shot all-gather and 7 two-shot all-reduce (the defaults
# of kernels/oneshot_all_gather.py and kernels/twoshot_all_reduce.py), 6 Muse megakernel.
K3MK_COLLECTIVE_ID = 4


def _is_pow2(n: int) -> bool:
    return n >= 1 and (n & (n - 1)) == 0


class Topology(msgspec.Struct, frozen=True):
    """``hosts`` x ``lanes`` ranks; TP8 is (1, 8), two hosts (2, 8), TP32 (4, 8).
    Tests split one host's 8 cores into fake hosts, e.g. (2, 4)."""

    hosts: int
    lanes: int = 8

    def __post_init__(self):
        if not (_is_pow2(self.hosts) and _is_pow2(self.lanes)):
            raise ValueError(
                f"hosts={self.hosts}, lanes={self.lanes} must be powers of two"
            )

    @property
    def world(self) -> int:
        return self.hosts * self.lanes

    @property
    def local_offsets(self) -> tuple[int, ...]:
        """XOR offsets of the other lanes of this host."""
        return tuple(range(1, self.lanes))

    @property
    def host_offsets(self) -> tuple[int, ...]:
        """XOR offsets of the same lane on every other host."""
        return tuple(self.lanes * h for h in range(1, self.hosts))

    @property
    def all_offsets(self) -> tuple[int, ...]:
        return tuple(range(1, self.world))


class K3Dims(msgspec.Struct, frozen=True):
    """Kimi-K3 text-model dimensions the kernel is specialized on."""

    hidden: int = 7168
    latent: int = 3584  # routed_expert_hidden_size
    expert_inter: int = 3072  # moe_intermediate_size
    num_experts: int = 896
    top_k: int = 16
    shared_inter: int = 6144  # moe_intermediate_size * num_shared_experts
    rms_eps: float = 1e-5
    situ_beta: float = 4.0
    situ_linear_beta: float = 25.0


class MoeLayout(msgspec.Struct, frozen=True):
    """Per-rank split of one MoE block over a ``Topology``.

    Experts: host ``h`` owns experts ``[h * E/H, (h+1) * E/H)``; inside a host
    every expert is tensor-parallel over the lanes (``expert_inter / lanes``
    channels each). The router is split over the chip pair (``rank & 1``).
    ``latent_down`` columns are split over all ranks in 128-lane chunks (the
    tail ranks hold zero columns); ``latent_up`` output columns over lanes.

    ``replicas`` (``Relax.expert_replicas``, multi-host only): host ``h`` also holds the
    first ``replicas`` experts of host ``h - 1`` (mod hosts) in expert slots
    ``[E/H, E/H + replicas)``, so a route to one of those can be served by either host.
    """

    dims: K3Dims
    topo: Topology
    replicas: int = 0

    def __post_init__(self):
        if not 0 <= self.replicas <= self.experts_per_host:
            raise ValueError(
                f"expert replicas {self.replicas} not in [0, {self.experts_per_host}]"
            )

    @property
    def experts_per_host(self) -> int:
        return self.dims.num_experts // self.topo.hosts

    @property
    def replica_count(self) -> int:
        """Replicated experts per host (none on a single host: there is no other host)."""
        return self.replicas if self.topo.hosts > 1 else 0

    @property
    def expert_slots(self) -> int:
        """Routed-expert slots per rank: own experts, then the replicas."""
        return self.experts_per_host + self.replica_count

    def slot_experts(self, host: int) -> list[int]:
        """Global expert id held in each expert slot of ``host``."""
        eh = self.experts_per_host
        prev = (host - 1) % self.topo.hosts
        return [host * eh + j for j in range(eh)] + [
            prev * eh + j for j in range(self.replica_count)
        ]

    def expert_slot(self, host: int, expert: int) -> int | None:
        """Slot of global ``expert`` on ``host``, or None when the host does not hold it."""
        eh = self.experts_per_host
        owner, local = divmod(expert, eh)
        if owner == host:
            return local
        if (
            self.replica_count
            and owner == (host - 1) % self.topo.hosts
            and local < self.replica_count
        ):
            return eh + local
        return None

    @property
    def lane_inter(self) -> int:
        return self.dims.expert_inter // self.topo.lanes

    @property
    def router_cols(self) -> int:
        return self.dims.num_experts // 2

    @property
    def router_pad(self) -> int:
        """Router columns per core, padded to whole 128-lane vregs."""
        return -(-self.router_cols // 128) * 128

    @property
    def latent_chunk(self) -> int:
        per = -(-self.dims.latent // self.topo.world)
        return -(-per // 128) * 128

    @property
    def latent_up_cols(self) -> int:
        return self.dims.hidden // self.topo.lanes

    @property
    def shared_cols(self) -> int:
        return self.dims.shared_inter // self.topo.world


class Relax(msgspec.Struct, frozen=True):
    """Numeric relaxations (``SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX``); the default is exact.

    ``dense_fp8``: dense projections stored as e4m3 with an f32 scale per output
    column (amax / 448); the dot runs on the e4m3 weight and its f32 result is scaled.
    ``bf16_transport``: reduction partials travel as bf16 (sums stay f32).
    ``host_bf16``: only the cross-host level of the reductions travels as bf16 (the in-host
    partials stay f32; exact on one host).
    ``expert_replicas``: each host also serves the first R experts of host h-1.
    ``pair_oproj``: attention output projection reduced through the chip pair
    (``stages.common.OutProjSpec``); f32 sums in a different order than the default.
    ``oproj_flat``: the attention output all-reduce (with ``pair_oproj``: the same-parity
    level) is one one-shot over all its ranks instead of in-host + cross-host rounds; the
    f32 sums keep their order (bit-identical with an f32 wire).
    """

    dense_fp8: bool = False
    bf16_transport: bool = False
    host_bf16: bool = False
    expert_replicas: int = 0
    pair_oproj: bool = False
    oproj_flat: bool = False

    @classmethod
    def parse(cls, items) -> "Relax":
        """``items``: e.g. ``("dense_fp8", "bf16_transport", "expert_replicas=48")``."""
        kw = {}
        for item in items:
            item = item.strip()
            if not item:
                continue
            name, _, value = item.partition("=")
            if name not in cls.__struct_fields__:
                raise ValueError(f"unknown K3 megakernel relaxation {name!r}")
            kw[name] = int(value) if value else True
        return cls(**kw)

    @classmethod
    def from_env(cls) -> "Relax":
        from sglang_tpu.envs import envs

        return cls.parse(envs.SGLANG_TPU_OPT_K3_MEGAKERNEL_RELAX.get())

    @property
    def transport(self):
        import jax.numpy as jnp

        return jnp.bfloat16 if self.bf16_transport else jnp.float32

    @property
    def host_transport(self):
        """Wire dtype of the cross-host level."""
        import jax.numpy as jnp

        return jnp.bfloat16 if (self.bf16_transport or self.host_bf16) else jnp.float32

    @property
    def dense_dtype(self):
        import jax.numpy as jnp

        return jnp.float8_e4m3fn if self.dense_fp8 else jnp.bfloat16
