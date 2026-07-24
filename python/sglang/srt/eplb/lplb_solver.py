"""
LPLBSolver — Linear-Programming Load Balancer for Expert Parallelism.

Encapsulates LP matrix construction (offline, at init/rebalance) and
per-batch solving (online, per MoE layer forward pass).

Design for DP-attention:
    Each EP rank counts its local tokens, then all ranks participate in an
    all-reduce to obtain identical global counts.  Every rank then solves
    the same LP independently, producing the same log2phy_prob — no
    broadcast is needed.  Empty-token ranks contribute zeros in the
    all-reduce so the collective never deadlocks.

Usage:
    solver = LPLBSolver(phy2log, log2phy, num_gpus, ep_group)
    log2phy_prob = solver.solve(topk_ids)  # per batch
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Global per-layer LPLB solvers


# LP dispatch requires every EP rank to call solver.solve() on every forward
# pass (including empty-topk ranks under DP-attention) — the all-reduce inside
# would otherwise hang. Only the DeepSeek-v2 family and its subclasses route
# empty-rank paths through solver.solve(); other MoE families would deadlock.
_LPLB_SUPPORTED_MODEL_ARCHS: frozenset[str] = frozenset(
    {
        "DeepseekV2ForCausalLM",
        "DeepseekV3ForCausalLM",
        "DeepseekV32ForCausalLM",
        "MistralLarge3ForCausalLM",
        "MistralLarge3ForCausalLMEagle",
        "Glm4MoeLiteForCausalLM",
        "GlmMoeDsaForCausalLM",
    }
)


def assert_lplb_supported_model(architecture: str) -> None:
    if architecture not in _LPLB_SUPPORTED_MODEL_ARCHS:
        supported = ", ".join(sorted(_LPLB_SUPPORTED_MODEL_ARCHS))
        raise NotImplementedError(
            f"{architecture} does not support --ep-dispatch-algorithm lp. "
            f"Validated targets: {supported}. Other MoE families have "
            "empty-token early returns that don't participate in the EP "
            "all-reduce inside LPLBSolver.solve(), which would deadlock "
            "under DP-attention."
        )


def get_global_lplb_solver(layer_id: int) -> Optional[LPLBSolver]:
    from sglang.srt.runtime_context import get_resources

    return get_resources().lplb_solvers.get(layer_id)


def set_global_lplb_solver(layer_id: int, solver: LPLBSolver):
    from sglang.srt.runtime_context import get_resources

    get_resources().lplb_solvers[layer_id] = solver


def clear_global_lplb_solvers():
    from sglang.srt.runtime_context import get_resources

    get_resources().lplb_solvers.clear()


class LPLBSolver:
    """
    Per-layer LPLB solver.

    At init: pre-computes LP constraint matrices from expert-to-GPU mapping.
    At solve: takes topk_ids, counts tokens, all-reduces, runs LP,
              returns log2phy_prob for probability-based token dispatch.
    """

    def __init__(
        self,
        phy2log: torch.Tensor,
        log2phy: torch.Tensor,
        num_gpus: int,
        ep_group=None,
        logical_to_all_physical_map_num_valid=None,
    ):
        """
        Args:
            phy2log: (num_physical_experts,) physical-to-logical expert mapping.
            log2phy: (num_logical_experts, max_copies) logical-to-physical mapping (-1 padded).
            num_gpus: Number of GPUs in the EP group.
            ep_group: GroupCoordinator for EP communication (all-reduce).
            logical_to_all_physical_map_num_valid: (num_logical_experts,) number of valid physical copies.
        """
        device = phy2log.device
        self.num_gpus = num_gpus
        self.ep_group = ep_group
        self._has_redundancy = False
        if logical_to_all_physical_map_num_valid is not None:
            self._has_redundancy = bool(
                (logical_to_all_physical_map_num_valid > 1).any()
            )

        self.num_logical = log2phy.shape[0]
        self.max_copies = log2phy.shape[1]
        self.num_phy = phy2log.shape[0]
        # B1/B2 GPU-assignment matrices below assume each rank owns a
        # contiguous block of num_phy // num_gpus physical experts.
        if self.num_phy % num_gpus != 0:
            raise ValueError(
                f"LPLBSolver requires num_phy ({self.num_phy}) to be divisible "
                f"by num_gpus ({num_gpus}); per-rank-contiguous ownership is "
                "currently the only supported allocation."
            )
        num_phy_per_gpu = self.num_phy // num_gpus

        # Count copies per logical expert
        logcnt = torch.bincount(phy2log, minlength=self.num_logical)

        # Separate single-copy vs replicated experts.
        # Stored as int64 so they can be used directly as index tensors in
        # _solve without per-call .long() casts (Tier 1 optimization).
        self.log_single = torch.nonzero(logcnt == 1).flatten().to(torch.int64)
        self.phy_single = log2phy[self.log_single, 0].to(torch.int64)
        self.log_replicated = torch.nonzero(logcnt > 1).flatten().to(torch.int64)
        self.phy_replicated = (
            torch.nonzero(logcnt[phy2log] > 1).flatten().to(torch.int64)
        )

        self.num_single = len(self.log_single)
        self.num_red_log = len(self.log_replicated)
        self.num_red_phy = len(self.phy_replicated)

        # Build GPU assignment matrices
        B_full = torch.zeros(
            (num_gpus, self.num_phy), dtype=torch.float32, device=device
        )
        for i in range(num_gpus):
            B_full[i, i * num_phy_per_gpu : (i + 1) * num_phy_per_gpu] = 1
        self.B1 = B_full[:, self.phy_single].contiguous()
        B2 = B_full[:, self.phy_replicated]

        # Build C matrix (copy-to-logical mapping)
        C = torch.zeros(
            (self.num_red_log, self.num_red_phy), dtype=torch.float32, device=device
        )
        phy2log_rep = phy2log[self.phy_replicated]
        for i in range(self.num_red_log):
            C[i, phy2log_rep == self.log_replicated[i]] = 1.0

        # Build A_base = [[C, 0, 0], [B2, I, -1]]  (without Big-M column)
        zeros_top_g = torch.zeros(
            (self.num_red_log, num_gpus), dtype=torch.float32, device=device
        )
        zeros_top_1 = torch.zeros(
            (self.num_red_log, 1), dtype=torch.float32, device=device
        )
        I_g = torch.eye(num_gpus, dtype=torch.float32, device=device)
        neg_ones = torch.full((num_gpus, 1), -1.0, dtype=torch.float32, device=device)

        A_top = torch.hstack([C, zeros_top_g, zeros_top_1])
        A_bottom = torch.hstack([B2, I_g, neg_ones])
        self.A_base = torch.vstack([A_top, A_bottom]).contiguous()

        # Objective: minimize M (second-to-last var), penalize Big-M auxiliary
        nv = self.A_base.shape[1] + 1  # +1 for Big-M column
        self.c_vec = torch.zeros(nv, dtype=torch.float32, device=device)
        self.c_vec[-2] = 1.0
        self.c_vec[-1] = 1000.0

        # Store log2phy as int64 so it can be used directly as index tensor
        # without per-call .long() casts (Tier 1 optimization).
        self.log2phy = log2phy.to(torch.int64).contiguous()

        # Compact the padded logical-to-physical table for the decode hot path.
        # A logical expert can have many redundant physical copies, but only
        # one copy per rank can reduce that rank's active-expert load. Store
        # one physical id per eligible rank plus an eligibility bitmask.
        if num_gpus > 32:
            raise ValueError(
                "Integral decode dispatch supports at most 32 EP ranks; "
                f"got {num_gpus}."
            )
        valid = self.log2phy >= 0
        safe_physical = self.log2phy.masked_fill(~valid, 0)
        physical_rank = torch.div(safe_physical, num_phy_per_gpu, rounding_mode="floor")
        physical_by_rank = []
        for rank in range(num_gpus):
            on_rank = valid & (physical_rank == rank)
            first_index = on_rank.to(torch.int32).argmax(dim=1, keepdim=True)
            first_physical = self.log2phy.gather(1, first_index).squeeze(1)
            physical_by_rank.append(first_physical.masked_fill(~on_rank.any(dim=1), -1))
        self.decode_physical_by_rank = (
            torch.stack(physical_by_rank, dim=1).to(torch.int32).contiguous()
        )
        if bool((self.decode_physical_by_rank < 0).all(dim=1).any()):
            raise ValueError(
                "Every logical expert must have at least one physical copy."
            )
        self.decode_rank_mask = torch.zeros(
            self.num_logical, dtype=torch.int32, device=device
        )
        for rank in range(num_gpus):
            self.decode_rank_mask.bitwise_or_(
                (self.decode_physical_by_rank[:, rank] >= 0).to(torch.int32) << rank
            )
        self.decode_rank_mask = self.decode_rank_mask.contiguous()
        self.decode_log_replicated = (
            torch.nonzero(
                self.decode_rank_mask.bitwise_and(self.decode_rank_mask - 1) != 0
            )
            .flatten()
            .to(torch.int32)
            .contiguous()
        )

        # Pre-JIT-compile the fused IPM kernel for this (NC, NV) shape so the
        # 20-40s compile cost happens once at startup rather than on the first
        # real request. No-op when the fused backend is unavailable.
        nc = self.A_base.shape[0]
        nv = self.A_base.shape[1] + 1  # +1 for Big-M column added in solve()
        from sglang.kernels.ops.lplb.torch_solver import warmup as _ipm_warmup

        _ipm_warmup(nc, nv, num_iters=5, device=device)

        # Pre-compute A_base row sum (used in every prep call).
        self._A_base_row_sum = self.A_base.sum(dim=1).contiguous()  # (NC,)

        # Pre-allocate the buffers the JIT CUDA prep / IPM / post kernels write
        # into. All writes are contiguous full-tensor stores (no strided
        # ``out=`` semantics), so the reuse is safe under high concurrency.
        # Constructed lazily on the first solve() call (we don't know the
        # device-side log2phy_prob shape until then) — see _solve.
        self._A_full = torch.empty(nc, nv, dtype=torch.float32, device=device)
        self._A_full[:, : nv - 1].copy_(self.A_base)
        self._b = torch.empty(nc, dtype=torch.float32, device=device)
        self._t1 = torch.empty(self.num_single, dtype=torch.float32, device=device)
        self._x = torch.empty(nv, dtype=torch.float32, device=device)
        self._log2phy_prob = torch.empty(
            log2phy.shape, dtype=torch.float32, device=device
        )
        self._decode_p2p_resources = None
        self._decode_all_active_physical = None

    def initialize_decode_p2p(self) -> None:
        """Create compact symmetric-memory resources for decode active-set union.

        The forward kernel exchanges eight uint32 words for DeepSeek-V3 rather
        than launching an EP all-reduce over 256 float counters. Initialization
        is collective over the EP device group and must run on every rank.
        """
        if self._decode_p2p_resources is not None:
            return
        if self.ep_group is None:
            raise RuntimeError("P2P decode LPLB requires an EP process group.")
        if self.ep_group.world_size != self.num_gpus:
            raise RuntimeError(
                "P2P decode LPLB EP group size mismatch: "
                f"{self.ep_group.world_size} != {self.num_gpus}."
            )

        import torch.distributed._symmetric_memory as symm_mem

        from sglang.kernels.ops.communication.inkling_all_reduce import (
            STATE_SIZE,
            flags_numel,
        )
        from sglang.kernels.ops.lplb.cuda_solver import (
            warmup_dispatch_decode_integral,
        )

        device = self.decode_physical_by_rank.device
        group = self.ep_group.device_group
        active_words = (self.num_logical + 31) // 32
        with torch.inference_mode(False), torch.no_grad():
            local_active = symm_mem.empty(
                active_words, dtype=torch.uint32, device=device
            )
            flags = symm_mem.empty(
                flags_numel(self.num_gpus),
                dtype=torch.uint32,
                device=device,
            )
        local_active.zero_()
        flags.zero_()
        active_handle = symm_mem.rendezvous(local_active, group=group)
        flag_handle = symm_mem.rendezvous(flags, group=group)
        # Ensure no peer can enter the first device-side epoch barrier while
        # another rank is still zeroing its symmetric flag slots.
        flag_handle.barrier()
        barrier_state = torch.zeros(STATE_SIZE, dtype=torch.uint32, device=device)
        warmup_dispatch_decode_integral(
            self.num_logical,
            self.num_gpus,
            self.decode_log_replicated.numel(),
        )
        self._decode_p2p_resources = (
            local_active,
            flags,
            barrier_state,
            active_handle,
            flag_handle,
        )

    def initialize_decode_all_active(self) -> None:
        """Precompute one globally consistent map with every expert active."""
        if self._decode_all_active_physical is not None:
            return
        from sglang.kernels.ops.lplb.cuda_solver import dispatch_decode_integral

        device = self.decode_physical_by_rank.device
        logical_ids = torch.arange(self.num_logical, dtype=torch.int32, device=device)
        all_active = torch.ones(self.num_logical, dtype=torch.float32, device=device)
        self._decode_all_active_physical = dispatch_decode_integral(
            logical_ids,
            all_active,
            self.decode_physical_by_rank,
            self.decode_rank_mask,
            self.decode_log_replicated,
        )

    def solve(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """
        Full LPLB pipeline: count -> all-reduce -> LP solve -> return log2phy_prob.

        All EP ranks must call this method every MoE layer forward pass,
        including empty-token ranks (which pass an empty topk_ids tensor).
        This ensures the all-reduce collective does not deadlock under
        DP-attention where different ranks may have different token counts.

        Args:
            topk_ids: (num_tokens, topk) int32 tensor of logical expert IDs.
                      Can be empty (shape (0, topk)) for idle ranks.
        Returns:
            log2phy_prob: (num_logical, max_copies) float32 probability tensor.
        """
        # Step 1-2: Count local tokens and all-reduce across EP ranks.
        # All EP ranks must participate — empty-token ranks contribute zeros.
        # After all-reduce, every rank has identical global_counts and solves
        # the same LP independently, so no broadcast is needed.
        # GroupCoordinator.all_reduce may be in-place (pynccl) or out-of-place
        # (ca_comm / pymscclpp / ...) depending on tensor size; small tensors
        # like ours (~num_logical * 4 B) typically take the out-of-place path,
        # so we must capture the return value.
        global_counts = self._count_and_all_reduce(topk_ids)

        # Step 3: Run LP solver
        return self._solve(global_counts)

    def solve_decode_active_experts_p2p(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Union active sets with fused GPU P2P and balance replicas globally."""
        if self._decode_p2p_resources is None:
            raise RuntimeError(
                "P2P decode LPLB resources were not initialized at model setup."
            )
        from sglang.kernels.ops.lplb.cuda_solver import (
            dispatch_decode_integral_p2p,
        )

        local_active, _, barrier_state, active_handle, flag_handle = (
            self._decode_p2p_resources
        )
        return dispatch_decode_integral_p2p(
            topk_ids,
            self.decode_physical_by_rank,
            self.decode_rank_mask,
            self.decode_log_replicated,
            local_active=local_active,
            active_ptrs_dev=active_handle.buffer_ptrs_dev,
            flag_ptrs_dev=flag_handle.buffer_ptrs_dev,
            barrier_state=barrier_state,
            rank=self.ep_group.rank_in_group,
        )

    def solve_decode_all_active(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Map decode routes through the precomputed all-active assignment."""
        if self._decode_all_active_physical is None:
            raise RuntimeError(
                "All-active decode LPLB was not initialized at model setup."
            )
        return self._decode_all_active_physical[topk_ids]

    def _count_and_all_reduce(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Return global logical-expert token counts as float32."""
        device = topk_ids.device
        local_counts = torch.zeros(self.num_logical, dtype=torch.int32, device=device)
        flat_ids = topk_ids.flatten()
        local_counts.scatter_add_(
            0,
            flat_ids.long(),
            torch.ones_like(flat_ids, dtype=torch.int32),
        )
        global_counts = local_counts.float()
        if self.ep_group is not None:
            global_counts = self.ep_group.all_reduce(global_counts)
        return global_counts

    def _solve(self, global_counts: torch.Tensor) -> torch.Tensor:
        """Three CUDA kernel launches replace ~14 torch ops.

        Pipeline (all writes go into pre-allocated buffers from __init__):
            prep_lp_inputs → solve_ipm → extract_log2phy_prob
        Raises if the JIT CUDA backend is unavailable.
        """
        from sglang.kernels.ops.lplb import cuda_solver

        cuda_solver.prep_lp_inputs(
            self._A_full,
            self._b,
            self._t1,
            global_counts,
            self.log_single,
            self.log_replicated,
            self.B1,
            self._A_base_row_sum,
        )
        cuda_solver.solve_ipm(self._A_full, self._b, self.c_vec, result=self._x)
        cuda_solver.extract_log2phy_prob(
            self._log2phy_prob,
            self._x,
            self._t1,
            self.phy_single,
            self.phy_replicated,
            self.log2phy,
        )
        return self._log2phy_prob
