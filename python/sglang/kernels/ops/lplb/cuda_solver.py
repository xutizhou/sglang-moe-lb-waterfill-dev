"""JIT-compiled CUDA Interior Point Method LP solver.

Replaces the Numba/nvmath-python implementation in ``cublasdx_solver.py``.
The kernel is a single-block fused IPM defined in
``csrc/lplb/ipm.cuh`` and compiled per ``(NC, NV, BLOCK_DIM, SM_VER,
NUM_ITERS)`` tuple via sglang's ``tvm-ffi`` ``load_jit``.

Per-call CPU overhead is dominated by the pybind11 dispatch + four
``data_ptr()`` calls (~5–10 µs total), versus ~500–700 µs for the prior
Numba path (numba dispatcher chain + ``as_cuda_array`` per tensor).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    get_jit_cuda_arch,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_DIM = 256
# Per-element kernels (post-LP dispatch) saturate easily — also 256.
DISPATCH_BLOCK_DIM = 256
DEFAULT_NUM_ITERS = 5


def _sm_ver() -> int:
    arch = get_jit_cuda_arch()
    return arch.major * 100 + arch.minor * 10


@cache_once
def _ipm_module(
    nc: int, nv: int, block_dim: int, num_iters: int, sm_ver: int
) -> Module:
    """JIT-compile the IPM kernel for one shape. Cached for the process lifetime."""
    args = make_cpp_args(nc, nv, block_dim, sm_ver, num_iters)
    # The kernel uses cuBLASDx (header-only) for the GEMMs and a hand-written
    # block-level Cholesky for the POSV. No -rdc=true / static-lib linkage
    # required, so sglang's tvm-ffi load_jit handles the build with the
    # default flags.
    return load_jit(
        "lplb_ipm",
        *args,
        cuda_files=["lplb/ipm.cuh"],
        cuda_wrappers=[("ipm_solve", f"ipm_solve<{args}>")],
        extra_dependencies=["mathdx"],
    )


def warmup(
    nc: int,
    nv: int,
    num_iters: int = DEFAULT_NUM_ITERS,
    device: str = "cuda",
) -> None:
    """JIT-compile the kernel for ``(nc, nv)`` so the first real solve isn't
    paying the compile cost. Raises on compile or launch failure.
    """
    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
    # Trigger any first-call lazy initialization.
    A = torch.zeros(nc, nv, dtype=torch.float32, device=device)
    b = torch.zeros(nc, dtype=torch.float32, device=device)
    c = torch.zeros(nv, dtype=torch.float32, device=device)
    result = torch.empty(nv, dtype=torch.float32, device=device)
    module.ipm_solve(A, b, c, result)
    logger.info(f"LPLB CUDA IPM solver: warmed up for (NC={nc}, NV={nv})")


def solve_ipm(
    A: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_iters: int = DEFAULT_NUM_ITERS,
    result: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the fused single-SM IPM kernel.

    cuBLASDx GEMMs + hand-written block Cholesky, dispatched per the
    module docstring.

    Args:
        A: Constraint matrix, shape ``(NC, NV)``, float32, on CUDA.
        b: RHS vector, shape ``(NC,)``, float32, on CUDA.
        c: Objective coefficients, shape ``(NV,)``, float32, on CUDA.
        num_iters: Number of barrier iterations (default 5).
        result: Optional pre-allocated ``(NV,)`` float32 CUDA buffer to write
            into. When omitted the kernel allocates a fresh result tensor
            (~20 µs of CPU overhead). Passing in a long-lived buffer skips
            that alloc on every solve.

    Returns:
        x: Solution vector, shape ``(NV,)``, float32. The kernel writes 0.5
        for every entry on non-convergence (matches the prior Numba behavior).
    """
    assert A.is_cuda and b.is_cuda and c.is_cuda
    assert A.dtype == torch.float32
    nc, nv = A.shape
    assert b.shape == (nc,), f"b shape mismatch: {b.shape} vs ({nc},)"
    assert c.shape == (nv,), f"c shape mismatch: {c.shape} vs ({nv},)"

    module = _ipm_module(nc, nv, DEFAULT_BLOCK_DIM, num_iters, _sm_ver())
    if result is None:
        result = torch.empty(nv, dtype=torch.float32, device=A.device)
    module.ipm_solve(A, b, c, result)
    return result


@cache_once
def _prep_module(
    nc: int,
    nv: int,
    num_single: int,
    num_red_log: int,
    num_gpus: int,
    block_dim: int,
) -> Module:
    args = make_cpp_args(nc, nv, num_single, num_red_log, num_gpus, block_dim)
    return load_jit(
        "lplb_lp_prep",
        *args,
        cuda_files=["lplb/lp_prep.cuh"],
        cuda_wrappers=[("lp_prep", f"lp_prep<{args}>")],
    )


def prep_lp_inputs(
    A_full: torch.Tensor,
    b: torch.Tensor,
    t1: torch.Tensor,
    global_counts: torch.Tensor,
    log_single: torch.Tensor,
    log_replicated: torch.Tensor,
    B1: torch.Tensor,
    A_base_row_sum: torch.Tensor,
) -> None:
    """Replace the 8 torch ops that built the IPM inputs with one CUDA kernel.

    Writes into the caller-provided ``A_full`` (last column only), ``b``,
    and ``t1`` buffers. ``A_full``'s first ``NV-1`` columns must already
    hold ``A_base.copy_()`` from solver init — this kernel does not touch
    them.
    """
    nc, nv = A_full.shape
    num_single = log_single.shape[0]
    num_red_log = log_replicated.shape[0]
    num_gpus, _ns = B1.shape
    module = _prep_module(nc, nv, num_single, num_red_log, num_gpus, DEFAULT_BLOCK_DIM)
    module.lp_prep(
        A_full, b, t1, global_counts, log_single, log_replicated, B1, A_base_row_sum
    )


@cache_once
def _post_module(
    num_logical: int,
    max_copies: int,
    num_single: int,
    num_red_phy: int,
    block_dim: int,
) -> Module:
    args = make_cpp_args(num_logical, max_copies, num_single, num_red_phy, block_dim)
    return load_jit(
        "lplb_lp_post",
        *args,
        cuda_files=["lplb/lp_post.cuh"],
        cuda_wrappers=[("lp_post", f"lp_post<{args}>")],
    )


def extract_log2phy_prob(
    log2phy_prob: torch.Tensor,
    x: torch.Tensor,
    t1: torch.Tensor,
    phy_single: torch.Tensor,
    phy_replicated: torch.Tensor,
    log2phy: torch.Tensor,
) -> None:
    """Replace the 5 torch ops that turned the IPM output into log2phy_prob
    with one CUDA kernel. Writes into the caller-provided ``log2phy_prob``
    buffer of shape ``(num_logical, max_copies)``.
    """
    num_logical, max_copies = log2phy_prob.shape
    num_single = phy_single.shape[0]
    num_red_phy = phy_replicated.shape[0]
    module = _post_module(
        num_logical, max_copies, num_single, num_red_phy, DEFAULT_BLOCK_DIM
    )
    module.lp_post(log2phy_prob, x, t1, phy_single, phy_replicated, log2phy)


@cache_once
def _dispatch_module(max_copies: int, block_dim: int) -> Module:
    args = make_cpp_args(max_copies, block_dim)
    return load_jit(
        "lplb_dispatch_probability",
        *args,
        cuda_files=["lplb/dispatch_probability.cuh"],
        cuda_wrappers=[("dispatch_probability", f"dispatch_probability<{args}>")],
    )


def dispatch_probability(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    random_vals: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Replace the 7 torch ops in `_topk_ids_logical_to_physical_probability`
    with a single per-token-per-slot CUDA kernel.

    Samples a physical expert per (token, slot) via inverse-CDF on the
    per-row LP probabilities. Bit-equivalent to
    :func:`dispatch_probability_torch_reference` when given the same
    ``random_vals`` (modulo float rounding in the cumulative sum).

    Args:
        topk_ids: (num_tokens, topk) int32, on CUDA. Logical expert ids from
            the router.
        log2phy_prob: (num_logical, max_copies) float32. LP solver output.
        log2phy_map: (num_logical, max_copies) int32. -1 entries are
            unused replicas; treated as 0-weight in the multinomial.
        random_vals: Optional (N,) float32 CUDA tensor of uniform samples in
            [0, 1). When omitted, the function generates fresh values via
            ``torch.rand``. Pass explicitly when comparing against the torch
            reference for deterministic equivalence.

    Returns:
        Physical topk ids tensor with the same shape as ``topk_ids``.
    """
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    n = flat_ids.shape[0]
    num_logical, max_copies = log2phy_prob.shape
    assert log2phy_map.shape == (num_logical, max_copies)
    map32 = log2phy_map.contiguous().to(torch.int32)

    out = torch.empty(n, dtype=torch.int32, device=topk_ids.device)
    if random_vals is None:
        random_vals = torch.rand(n, dtype=torch.float32, device=topk_ids.device)
    else:
        assert random_vals.shape == (
            n,
        ), f"random_vals must be shape ({n},), got {tuple(random_vals.shape)}"
    module = _dispatch_module(max_copies, DISPATCH_BLOCK_DIM)
    module.dispatch_probability(out, flat_ids, log2phy_prob, map32, random_vals)
    return out.view(original_shape).to(topk_ids.dtype)


@cache_once
def _dispatch_decode_integral_module(
    num_logical: int,
    num_gpus: int,
    num_replicated: int,
    block_dim: int,
) -> Module:
    args = make_cpp_args(
        num_logical,
        num_gpus,
        num_replicated,
        block_dim,
    )
    return load_jit(
        "lplb_dispatch_decode_integral",
        *args,
        cuda_files=["lplb/dispatch_decode_integral.cuh"],
        cuda_wrappers=[
            ("dispatch_decode_integral", f"dispatch_decode_integral<{args}>"),
            ("dispatch_decode_metro", f"dispatch_decode_metro<{args}>"),
            (
                "dispatch_decode_metro_global",
                f"dispatch_decode_metro_global<{args}>",
            ),
            (
                "dispatch_decode_integral_p2p",
                f"dispatch_decode_integral_p2p<{args}>",
            ),
            (
                "dispatch_decode_metro_p2p",
                f"dispatch_decode_metro_p2p<{args}>",
            ),
        ],
    )


def dispatch_decode_integral(
    topk_ids: torch.Tensor,
    global_counts: torch.Tensor,
    physical_by_rank: torch.Tensor,
    rank_mask: torch.Tensor,
    replicated_logical: torch.Tensor,
) -> torch.Tensor:
    """Assign decode experts integrally and map top-k ids in one CUDA launch."""
    if not topk_ids.is_cuda:
        raise RuntimeError(
            "Greedy decode dispatch requires CUDA tensors; got topk_ids on "
            f"{topk_ids.device}."
        )
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    num_logical, num_gpus = physical_by_rank.shape
    assert global_counts.shape == (num_logical,)
    assert global_counts.dtype == torch.float32
    assert physical_by_rank.dtype == torch.int32
    assert rank_mask.shape == (num_logical,)
    assert rank_mask.dtype == torch.int32
    assert replicated_logical.dtype == torch.int32
    num_replicated = replicated_logical.shape[0]

    out = torch.empty(flat_ids.shape[0], dtype=torch.int32, device=topk_ids.device)
    module = _dispatch_decode_integral_module(
        num_logical,
        num_gpus,
        num_replicated,
        DISPATCH_BLOCK_DIM,
    )
    module.dispatch_decode_integral(
        out,
        flat_ids,
        global_counts,
        physical_by_rank,
        rank_mask,
        replicated_logical,
    )
    return out.view(original_shape).to(topk_ids.dtype)


def dispatch_decode_integral_p2p(
    topk_ids: torch.Tensor,
    physical_by_rank: torch.Tensor,
    rank_mask: torch.Tensor,
    replicated_logical: torch.Tensor,
    *,
    local_active: torch.Tensor,
    active_ptrs_dev: int,
    flag_ptrs_dev: int,
    barrier_state: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Gather compact active sets with GPU P2P and assign replicas in one launch."""
    if not topk_ids.is_cuda:
        raise RuntimeError(
            "P2P decode dispatch requires CUDA tensors; got topk_ids on "
            f"{topk_ids.device}."
        )
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    num_logical, num_gpus = physical_by_rank.shape
    active_words = (num_logical + 31) // 32
    assert local_active.shape == (active_words,)
    assert local_active.dtype == torch.uint32
    assert physical_by_rank.dtype == torch.int32
    assert rank_mask.shape == (num_logical,)
    assert rank_mask.dtype == torch.int32
    assert replicated_logical.dtype == torch.int32
    assert barrier_state.dtype == torch.uint32
    num_replicated = replicated_logical.shape[0]

    out = torch.empty(flat_ids.shape[0], dtype=torch.int32, device=topk_ids.device)
    module = _dispatch_decode_integral_module(
        num_logical,
        num_gpus,
        num_replicated,
        DISPATCH_BLOCK_DIM,
    )
    module.dispatch_decode_integral_p2p(
        out,
        flat_ids,
        local_active,
        physical_by_rank,
        rank_mask,
        replicated_logical,
        active_ptrs_dev,
        flag_ptrs_dev,
        barrier_state.data_ptr(),
        rank,
    )
    return out.view(original_shape).to(topk_ids.dtype)


def dispatch_decode_metro(
    topk_ids: torch.Tensor,
    global_counts: torch.Tensor,
    physical_by_rank: torch.Tensor,
    rank_mask: torch.Tensor,
    replicated_logical: torch.Tensor,
) -> torch.Tensor:
    """Apply METRO to an already-global active set in one CUDA launch."""
    if not topk_ids.is_cuda:
        raise RuntimeError(
            "METRO decode dispatch requires CUDA tensors; got topk_ids on "
            f"{topk_ids.device}."
        )
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    num_logical, num_gpus = physical_by_rank.shape
    assert global_counts.shape == (num_logical,)
    assert global_counts.dtype == torch.float32
    assert physical_by_rank.dtype == torch.int32
    assert rank_mask.shape == (num_logical,)
    assert rank_mask.dtype == torch.int32
    assert replicated_logical.dtype == torch.int32

    out = torch.empty(flat_ids.shape[0], dtype=torch.int32, device=topk_ids.device)
    module = _dispatch_decode_integral_module(
        num_logical,
        num_gpus,
        replicated_logical.shape[0],
        DISPATCH_BLOCK_DIM,
    )
    module.dispatch_decode_metro(
        out,
        flat_ids,
        global_counts,
        physical_by_rank,
        rank_mask,
        replicated_logical,
    )
    return out.view(original_shape).to(topk_ids.dtype)


def dispatch_decode_metro_global(
    topk_ids: torch.Tensor,
    physical_by_rank: torch.Tensor,
    rank_mask: torch.Tensor,
    replicated_logical: torch.Tensor,
) -> torch.Tensor:
    """Count and route an already-global TopK in one CUDA launch."""
    if not topk_ids.is_cuda:
        raise RuntimeError(
            "METRO decode dispatch requires CUDA tensors; got topk_ids on "
            f"{topk_ids.device}."
        )
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    num_logical, num_gpus = physical_by_rank.shape
    assert physical_by_rank.dtype == torch.int32
    assert rank_mask.shape == (num_logical,)
    assert rank_mask.dtype == torch.int32
    assert replicated_logical.dtype == torch.int32

    out = torch.empty(flat_ids.shape[0], dtype=torch.int32, device=topk_ids.device)
    module = _dispatch_decode_integral_module(
        num_logical,
        num_gpus,
        replicated_logical.shape[0],
        DISPATCH_BLOCK_DIM,
    )
    module.dispatch_decode_metro_global(
        out,
        flat_ids,
        physical_by_rank,
        rank_mask,
        replicated_logical,
    )
    return out.view(original_shape).to(topk_ids.dtype)


def warmup_dispatch_decode_integral(
    num_logical: int,
    num_gpus: int,
    num_replicated: int,
) -> None:
    """Compile the integral decode module before CUDA-graph capture."""
    _dispatch_decode_integral_module(
        num_logical,
        num_gpus,
        num_replicated,
        DISPATCH_BLOCK_DIM,
    )


def dispatch_decode_metro_p2p(
    topk_ids: torch.Tensor,
    physical_by_rank: torch.Tensor,
    rank_mask: torch.Tensor,
    replicated_logical: torch.Tensor,
    *,
    local_active: torch.Tensor,
    active_ptrs_dev: int,
    flag_ptrs_dev: int,
    barrier_state: torch.Tensor,
    rank: int,
) -> torch.Tensor:
    """Gather the current active set and apply METRO Algorithm 1."""
    if not topk_ids.is_cuda:
        raise RuntimeError(
            "METRO decode dispatch requires CUDA tensors; got topk_ids on "
            f"{topk_ids.device}."
        )
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).contiguous().to(torch.int32)
    num_logical, num_gpus = physical_by_rank.shape
    active_words = (num_logical + 31) // 32
    assert local_active.shape == (active_words,)
    assert local_active.dtype == torch.uint32
    assert physical_by_rank.dtype == torch.int32
    assert rank_mask.shape == (num_logical,)
    assert rank_mask.dtype == torch.int32
    assert replicated_logical.dtype == torch.int32
    assert barrier_state.dtype == torch.uint32

    out = torch.empty(flat_ids.shape[0], dtype=torch.int32, device=topk_ids.device)
    module = _dispatch_decode_integral_module(
        num_logical,
        num_gpus,
        replicated_logical.shape[0],
        DISPATCH_BLOCK_DIM,
    )
    module.dispatch_decode_metro_p2p(
        out,
        flat_ids,
        local_active,
        physical_by_rank,
        rank_mask,
        replicated_logical,
        active_ptrs_dev,
        flag_ptrs_dev,
        barrier_state.data_ptr(),
        rank,
    )
    return out.view(original_shape).to(topk_ids.dtype)


def dispatch_decode_integral_torch_reference(
    topk_ids: torch.Tensor,
    global_counts: torch.Tensor,
    log2phy_map: torch.Tensor,
    *,
    num_physical: int,
    num_gpus: int,
) -> torch.Tensor:
    """CPU/CUDA reference for the integral decode assignment."""
    num_logical, max_copies = log2phy_map.shape
    physical_per_gpu = num_physical // num_gpus
    fixed_active_load = [0] * num_gpus
    fixed_token_load = [0] * num_gpus
    chosen = [int(log2phy_map[logical, 0]) for logical in range(num_logical)]
    replicated = []

    for logical in range(num_logical):
        count = int(global_counts[logical])
        if count <= 0:
            continue
        if max_copies == 1 or int(log2phy_map[logical, 1]) < 0:
            rank = chosen[logical] // physical_per_gpu
            fixed_active_load[rank] += 1
            fixed_token_load[rank] += count
        else:
            replicated.append(logical)

    minimum_capacity = max(fixed_active_load)
    for capacity in range(
        minimum_capacity,
        minimum_capacity + len(replicated) + 1,
    ):
        active_load = fixed_active_load.copy()
        token_load = fixed_token_load.copy()
        replicated_rank = [-1] * len(replicated)
        feasible = True

        for root, root_logical in enumerate(replicated):
            eligible = []
            for physical_tensor in log2phy_map[root_logical]:
                physical = int(physical_tensor)
                if physical < 0:
                    break
                eligible.append(physical // physical_per_gpu)
            eligible = sorted(
                set(eligible),
                key=lambda rank: (token_load[rank], rank),
            )

            queue = list(eligible)
            visited = set(eligible)
            parent_expert = {rank: root for rank in eligible}
            parent_rank = {rank: -1 for rank in eligible}
            free_rank = -1
            queue_head = 0

            while queue_head < len(queue):
                rank = queue[queue_head]
                queue_head += 1
                if active_load[rank] < capacity:
                    free_rank = rank
                    break
                for expert in range(root):
                    if replicated_rank[expert] != rank:
                        continue
                    logical = replicated[expert]
                    for physical_tensor in log2phy_map[logical]:
                        physical = int(physical_tensor)
                        if physical < 0:
                            break
                        alternate = physical // physical_per_gpu
                        if alternate in visited:
                            continue
                        visited.add(alternate)
                        parent_expert[alternate] = expert
                        parent_rank[alternate] = rank
                        queue.append(alternate)

            if free_rank < 0:
                feasible = False
                break

            destination = free_rank
            while destination >= 0:
                expert = parent_expert[destination]
                old_rank = replicated_rank[expert]
                count = int(global_counts[replicated[expert]])
                if old_rank >= 0:
                    active_load[old_rank] -= 1
                    token_load[old_rank] -= count
                replicated_rank[expert] = destination
                active_load[destination] += 1
                token_load[destination] += count
                destination = parent_rank[destination]

        if not feasible:
            continue

        for expert, logical in enumerate(replicated):
            assigned_rank = replicated_rank[expert]
            for physical_tensor in log2phy_map[logical]:
                physical = int(physical_tensor)
                if physical < 0:
                    break
                if physical // physical_per_gpu == assigned_rank:
                    chosen[logical] = physical
                    break
        break

    chosen_tensor = torch.tensor(
        chosen,
        dtype=topk_ids.dtype,
        device=topk_ids.device,
    )
    return chosen_tensor[topk_ids.long()]


def dispatch_probability_torch_reference(
    topk_ids: torch.Tensor,
    log2phy_prob: torch.Tensor,
    log2phy_map: torch.Tensor,
    random_vals: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch reference of :func:`dispatch_probability`.

    Mirrors the CUDA kernel's algorithm exactly (inverse-CDF via cumsum
    + threshold) so the two paths are bit-equivalent for identical
    ``random_vals``, modulo floating-point rounding in the cumsum. Kept
    for numerical comparison and testing — not on the production hot
    path (it allocates and runs ~8 torch ops; the fused kernel collapses
    them into one launch).

    Algorithm (matches ``csrc/lplb/dispatch_probability.cuh``):

      1. Gather the per-row probability vector and physical-id map for
         each logical id in ``topk_ids``.
      2. If the row sum is zero (LP gave no signal), fall back to
         uniform over valid replicas (``log2phy_map != -1``).
      3. Sample: smallest ``c`` such that ``cumsum[0..c] > u * row_sum``,
         where ``u = random_vals[i]``. Ties favor advancing ``c``,
         matching the CUDA kernel.
      4. Return ``log2phy_map[logical_id, c]``.

    Args:
        topk_ids: (num_tokens, topk) int, on CUDA or CPU. Logical expert ids.
        log2phy_prob: (num_logical, max_copies) float32. LP solver output.
        log2phy_map: (num_logical, max_copies) int. -1 = unused replica.
        random_vals: (N,) float32, where N = ``topk_ids.numel()``. Uniform
            samples in [0, 1) — same shape and semantics as the CUDA kernel.

    Returns:
        Physical topk ids tensor with the same shape and dtype as ``topk_ids``.
    """
    original_shape = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()
    n = flat_ids.shape[0]
    num_logical, max_copies = log2phy_prob.shape
    assert log2phy_map.shape == (num_logical, max_copies)
    assert random_vals.shape == (
        n,
    ), f"random_vals must be shape ({n},), got {tuple(random_vals.shape)}"

    # Gather per-row probabilities and physical maps.
    probs = log2phy_prob[flat_ids]  # (N, max_copies), float32
    maps = log2phy_map[flat_ids]  # (N, max_copies), same int dtype as input

    # Fallback when row_sum == 0: uniform over valid replicas.
    row_sum = probs.sum(dim=-1, keepdim=True)  # (N, 1)
    fallback_probs = (maps >= 0).to(probs.dtype)  # (N, max_copies)
    probs = torch.where(row_sum > 0, probs, fallback_probs)
    row_sum = probs.sum(dim=-1)  # (N,)

    # Inverse-CDF sample: smallest c such that cumsum[..c] > u.
    # ``(cum <= u).sum(dim=-1)`` counts how many slots are still below u,
    # which equals the CUDA kernel's ``chosen`` after its for-loop.
    u = (random_vals * row_sum).unsqueeze(-1)  # (N, 1)
    cum = probs.cumsum(dim=-1)  # (N, max_copies)
    chosen = (cum <= u).sum(dim=-1).clamp(max=max_copies - 1)  # (N,)

    out = maps.gather(1, chosen.unsqueeze(-1)).squeeze(-1)
    return out.view(original_shape).to(topk_ids.dtype)
