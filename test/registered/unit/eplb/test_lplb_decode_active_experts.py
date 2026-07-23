from types import SimpleNamespace

import torch

from sglang.srt.eplb.expert_location_dispatch import (
    _topk_ids_logical_to_physical_active_experts,
    _topk_ids_logical_to_physical_active_experts_torch_reference,
)
from sglang.srt.eplb.lplb_solver import LPLBSolver
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=3, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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
