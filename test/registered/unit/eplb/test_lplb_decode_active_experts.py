from itertools import product
from types import SimpleNamespace

import torch

from sglang.kernels.ops.lplb.cuda_solver import (
    dispatch_decode_integral,
    dispatch_decode_integral_torch_reference,
)
from sglang.srt.eplb.expert_location_dispatch import (
    _topk_ids_logical_to_physical_active_experts,
    _topk_ids_logical_to_physical_active_experts_torch_reference,
)
from sglang.srt.eplb.lplb_solver import LPLBSolver
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _integral_decode_case(device: str):
    # Four ranks, three contiguous physical slots per rank. Logical experts
    # 0-3 are fixed; 4-7 each have two eligible physical replicas.
    log2phy_map = torch.tensor(
        [
            [0, -1],
            [3, -1],
            [6, -1],
            [9, -1],
            [1, 4],
            [7, 10],
            [2, 8],
            [5, 11],
        ],
        dtype=torch.int64,
        device=device,
    )
    global_counts = torch.tensor(
        [1, 3, 2, 1, 5, 4, 3, 2],
        dtype=torch.float32,
        device=device,
    )
    topk_ids = torch.tensor(
        [[4, 4, 6, 0], [5, 7, 1, 2], [6, 5, 4, 3]],
        dtype=torch.int32,
        device=device,
    )
    return topk_ids, global_counts, log2phy_map


def test_decode_solver_uses_binary_expert_activation_load():
    solver = LPLBSolver.__new__(LPLBSolver)
    solver.num_logical = 4
    solver.ep_group = None
    solver._solve = lambda counts: counts

    topk_ids = torch.tensor([[0, 0], [0, 1], [3, 3]], dtype=torch.int32)

    token_load = solver.solve(topk_ids)
    activation_load = solver.solve(topk_ids, minimize_active_experts=True)

    torch.testing.assert_close(token_load, torch.tensor([3.0, 1.0, 0.0, 2.0]))
    torch.testing.assert_close(activation_load, torch.tensor([1.0, 1.0, 0.0, 1.0]))


def test_decode_dispatch_co_locates_each_logical_expert():
    info = SimpleNamespace(
        partial_logical_to_all_physical_map=torch.tensor(
            [
                [0, 4, -1],
                [1, -1, -1],
                [2, 6, 10],
            ],
            dtype=torch.int64,
        )
    )
    log2phy_prob = torch.tensor(
        [
            [0.25, 0.75, 0.0],
            [1.0, 0.0, 0.0],
            [0.20, 0.55, 0.25],
        ],
        dtype=torch.float32,
    )
    topk_ids = torch.tensor(
        [[0, 2], [2, 0], [1, 2]],
        dtype=torch.int32,
    )

    physical_ids = _topk_ids_logical_to_physical_active_experts_torch_reference(
        topk_ids, info, log2phy_prob
    )

    torch.testing.assert_close(
        physical_ids,
        torch.tensor([[4, 6], [6, 4], [1, 6]], dtype=torch.int32),
    )


def test_decode_dispatch_cuda_matches_torch_reference():
    if not torch.cuda.is_available():
        return
    info = SimpleNamespace(
        partial_logical_to_all_physical_map=torch.tensor(
            [
                [0, 4, -1],
                [1, -1, -1],
                [2, 6, 10],
            ],
            dtype=torch.int64,
            device="cuda",
        )
    )
    log2phy_prob = torch.tensor(
        [
            [0.25, 0.75, 0.0],
            [1.0, 0.0, 0.0],
            [0.20, 0.55, 0.25],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    topk_ids = torch.tensor(
        [[0, 2], [2, 0], [1, 2]],
        dtype=torch.int32,
        device="cuda",
    )

    expected = _topk_ids_logical_to_physical_active_experts_torch_reference(
        topk_ids, info, log2phy_prob
    )
    actual = _topk_ids_logical_to_physical_active_experts(topk_ids, info, log2phy_prob)

    torch.testing.assert_close(actual, expected)


def test_decode_integral_dispatch_is_deterministic_and_co_located():
    topk_ids, global_counts, log2phy_map = _integral_decode_case("cpu")
    physical_ids = dispatch_decode_integral_torch_reference(
        topk_ids,
        global_counts,
        log2phy_map,
        num_physical=12,
        num_gpus=4,
    )

    # Repeated logical ids must always select one physical replica.
    for logical in topk_ids.unique():
        routed = physical_ids[topk_ids == logical]
        assert routed.unique().numel() == 1
    torch.testing.assert_close(
        physical_ids,
        torch.tensor(
            [[1, 1, 8, 0], [10, 5, 3, 6], [8, 10, 1, 9]],
            dtype=torch.int32,
        ),
    )

    physical_per_logical = {}
    for logical in topk_ids.unique():
        physical_per_logical[int(logical)] = int(physical_ids[topk_ids == logical][0])
    active_load = [0] * 4
    for logical, physical in physical_per_logical.items():
        if global_counts[logical] > 0:
            active_load[physical // 3] += 1

    # Exhaust all replica choices for this small graph. The integral solver
    # must attain the exact minimum possible maximum rank load.
    fixed = [1, 1, 1, 1]
    replicated_ranks = [(0, 1), (2, 3), (0, 2), (1, 3)]
    optimum = min(
        max(
            fixed[rank] + sum(choice == rank for choice in choices) for rank in range(4)
        )
        for choices in product(*replicated_ranks)
    )
    assert max(active_load) == optimum


def test_decode_integral_cuda_matches_torch_reference():
    if not torch.cuda.is_available():
        return
    topk_ids, global_counts, log2phy_map = _integral_decode_case("cuda")
    expected = dispatch_decode_integral_torch_reference(
        topk_ids,
        global_counts,
        log2phy_map,
        num_physical=12,
        num_gpus=4,
    )
    actual = dispatch_decode_integral(
        topk_ids,
        global_counts,
        log2phy_map,
        num_physical=12,
        num_gpus=4,
        num_replicated=4,
    )

    torch.testing.assert_close(actual, expected)
