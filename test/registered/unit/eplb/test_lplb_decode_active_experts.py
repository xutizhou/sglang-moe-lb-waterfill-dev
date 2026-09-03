from contextlib import nullcontext
from itertools import product
from types import SimpleNamespace

import torch

from sglang.kernels.ops.lplb.cuda_solver import (
    dispatch_decode_integral,
    dispatch_decode_integral_torch_reference,
    dispatch_decode_metro,
    dispatch_decode_metro_global,
)
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.eplb.lplb_solver import LPLBSolver
from sglang.srt.layers.moe import topk as topk_module
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


def _compact_decode_map(
    log2phy_map: torch.Tensor,
    *,
    num_physical: int,
    num_gpus: int,
):
    physical_per_gpu = num_physical // num_gpus
    valid = log2phy_map >= 0
    ranks = log2phy_map.masked_fill(~valid, 0) // physical_per_gpu
    physical_by_rank = []
    for rank in range(num_gpus):
        on_rank = valid & (ranks == rank)
        first_index = on_rank.to(torch.int32).argmax(dim=1, keepdim=True)
        first_physical = log2phy_map.gather(1, first_index).squeeze(1)
        physical_by_rank.append(first_physical.masked_fill(~on_rank.any(dim=1), -1))
    physical_by_rank = torch.stack(physical_by_rank, dim=1).to(torch.int32)
    rank_mask = torch.zeros(
        log2phy_map.shape[0], dtype=torch.int32, device=log2phy_map.device
    )
    for rank in range(num_gpus):
        rank_mask.bitwise_or_((physical_by_rank[:, rank] >= 0).to(torch.int32) << rank)
    replicated_logical = (
        torch.nonzero(rank_mask.bitwise_and(rank_mask - 1) != 0)
        .flatten()
        .to(torch.int32)
    )
    return (
        physical_by_rank.contiguous(),
        rank_mask.contiguous(),
        replicated_logical.contiguous(),
    )


def test_decode_compaction_counts_distinct_ranks(monkeypatch):
    from sglang.kernels.ops.lplb import torch_solver

    monkeypatch.setattr(torch_solver, "warmup", lambda *_args, **_kwargs: None)
    # Logical expert 0 has two physical copies on rank 0 and one on rank 1.
    # Only the first rank-0 copy is retained and the expert is movable across
    # two ranks, not counted as three independent choices.
    phy2log = torch.tensor([0, 0, 1, 0], dtype=torch.int64)
    log2phy = torch.tensor(
        [[0, 1, 3], [2, -1, -1]],
        dtype=torch.int64,
    )
    solver = LPLBSolver(phy2log, log2phy, num_gpus=2)

    torch.testing.assert_close(
        solver.decode_physical_by_rank,
        torch.tensor([[0, 3], [-1, 2]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        solver.decode_rank_mask,
        torch.tensor([0b11, 0b10], dtype=torch.int32),
    )
    torch.testing.assert_close(
        solver.decode_log_replicated,
        torch.tensor([0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        solver.decode_static_physical,
        torch.tensor([0, 2], dtype=torch.int32),
    )


def test_static_decode_skips_online_lplb(monkeypatch):
    monkeypatch.setattr(topk_module, "_is_cuda", True)
    topk_ids = torch.tensor([[0, 2], [1, 0]], dtype=torch.int32)
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    info = ExpertLocationDispatchInfo(
        ep_dispatch_algorithm="lp",
        partial_logical_to_rank_dispatch_physical_map=torch.tensor(
            [4, 1, 6], dtype=torch.int64
        ),
        partial_logical_to_all_physical_map=torch.tensor(
            [[0, 4], [1, -1], [2, 6]], dtype=torch.int64
        ),
        partial_logical_to_all_physical_map_num_valid=torch.tensor(
            [2, 1, 2], dtype=torch.int64
        ),
        num_physical_experts=8,
    )
    config = SimpleNamespace(
        allow_routed_experts_capture=False,
        num_fused_shared_experts=0,
        fused_shared_experts_scaling_factor=None,
    )

    expected = torch.tensor([[4, 6], [1, 4]], dtype=torch.int32)
    for policy in ("static",):
        physical_ids, actual_weights, recorder_ids = (
            topk_module._post_process_topk_ids(
                topk_ids.clone(),
                topk_weights.clone(),
                config,
                router_logits=torch.empty((2, 3)),
                layer_id=0,
                expert_location_dispatch_info=info,
                lplb_decode_load_metric=policy,
            )
        )

        torch.testing.assert_close(physical_ids, expected)
        torch.testing.assert_close(recorder_ids, expected)
        torch.testing.assert_close(actual_weights, topk_weights)

    gathered_ids, gathered_weights, recorder_ids = (
        topk_module._post_process_topk_ids(
            topk_ids.clone(),
            topk_weights.clone(),
            config,
            router_logits=torch.empty((2, 3)),
            layer_id=0,
            expert_location_dispatch_info=info,
            lplb_decode_load_metric="static_allgather",
        )
    )
    torch.testing.assert_close(gathered_ids, topk_ids)
    torch.testing.assert_close(recorder_ids, torch.full_like(topk_ids, -1))
    torch.testing.assert_close(gathered_weights, topk_weights)


def test_empty_decode_participation_matches_policy(monkeypatch):
    calls = []
    solver = SimpleNamespace(
        solve=lambda ids: calls.append(("tokens", tuple(ids.shape))),
        solve_decode_active_experts_p2p=lambda ids: calls.append(
            ("active_experts", tuple(ids.shape))
        ),
        solve_decode_metro_p2p=lambda ids: calls.append(
            ("metro", tuple(ids.shape))
        ),
        solve_decode_metro=lambda ids: calls.append(("metro", tuple(ids.shape))),
        mark_metro_allgather_decode=lambda: calls.append(
            ("metro_allgather", (0, 2))
        ),
        solve_decode_all_active=lambda ids: calls.append(
            ("active_experts_prior", tuple(ids.shape))
        ),
    )
    monkeypatch.setattr(
        "sglang.srt.eplb.lplb_solver.get_global_lplb_solver",
        lambda _layer_id: solver,
    )
    monkeypatch.setattr(
        topk_module,
        "use_symmetric_memory",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(topk_module, "get_tp_group", lambda: None)
    topk = topk_module.TopK.__new__(topk_module.TopK)
    topk.topk_config = SimpleNamespace(top_k=2, num_fused_shared_experts=0)
    topk.enable_waterfill = False
    topk.waterfill_balancer = None

    for policy, expected in (
        ("tokens", "tokens"),
        ("active_experts", "active_experts"),
        ("metro", "metro"),
        ("metro_allgather", "metro_allgather"),
        ("static_allgather", "metro_allgather"),
        ("active_experts_prior", "active_experts_prior"),
        ("static", None),
    ):
        calls.clear()
        topk.lplb_decode_load_metric = policy
        topk.empty_topk_output(torch.device("cpu"), layer_id=0, is_decode=True)
        assert calls == ([] if expected is None else [(expected, (0, 2))])


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
    physical_by_rank, rank_mask, replicated_logical = _compact_decode_map(
        log2phy_map,
        num_physical=12,
        num_gpus=4,
    )
    actual = dispatch_decode_integral(
        topk_ids,
        global_counts,
        physical_by_rank,
        rank_mask,
        replicated_logical,
    )

    torch.testing.assert_close(actual, expected)


def test_decode_metro_cuda_routes_each_active_expert_once():
    if not torch.cuda.is_available():
        return
    topk_ids, global_counts, log2phy_map = _integral_decode_case("cuda")
    physical_by_rank, rank_mask, replicated_logical = _compact_decode_map(
        log2phy_map,
        num_physical=12,
        num_gpus=4,
    )

    physical_ids = dispatch_decode_metro(
        topk_ids,
        global_counts,
        physical_by_rank,
        rank_mask,
        replicated_logical,
    )
    physical_ids_fused = dispatch_decode_metro_global(
        topk_ids,
        physical_by_rank,
        rank_mask,
        replicated_logical,
    )

    torch.testing.assert_close(physical_ids_fused, physical_ids)

    torch.testing.assert_close(
        physical_ids,
        torch.tensor(
            [[1, 1, 2, 0], [7, 5, 3, 6], [2, 7, 1, 9]],
            dtype=torch.int32,
            device="cuda",
        ),
    )
    active_load = [0, 0, 0, 0]
    for logical in topk_ids.unique():
        routed = physical_ids[topk_ids == logical]
        assert routed.unique().numel() == 1
        active_load[int(routed[0]) // 3] += 1
    assert active_load == [3, 2, 2, 1]


def test_decode_all_active_prior_is_precomputed_and_consistent(monkeypatch):
    if not torch.cuda.is_available():
        return
    from sglang.kernels.ops.lplb import torch_solver

    monkeypatch.setattr(torch_solver, "warmup", lambda *_args, **_kwargs: None)
    topk_ids, _, log2phy_map = _integral_decode_case("cuda")
    phy2log = torch.tensor(
        [0, 4, 6, 1, 4, 7, 2, 5, 6, 3, 5, 7],
        dtype=torch.int64,
        device="cuda",
    )
    solver = LPLBSolver(phy2log, log2phy_map, num_gpus=4)
    solver.initialize_decode_all_active()

    actual = solver.solve_decode_all_active(topk_ids)
    for logical in topk_ids.unique():
        assert actual[topk_ids == logical].unique().numel() == 1
    logical_ids = torch.arange(8, dtype=torch.int32, device="cuda")
    physical_ids = solver.solve_decode_all_active(logical_ids)
    active_load = [0] * 4
    for logical, physical in zip(
        logical_ids.cpu().tolist(), physical_ids.cpu().tolist(), strict=True
    ):
        assert physical in log2phy_map[logical].cpu().tolist()
        active_load[physical // 3] += 1
    assert active_load == [2, 2, 2, 2]
