"""UltraEP placement state must commit together with the expert weights."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("moe_load_balancer")

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.runtime_context import get_context


def metadata(quota):
    return ExpertLocationMetadata(
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]]),
        physical_to_logical_map_cpu=torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]]),
        logical_to_all_physical_map=torch.tensor(
            [[[0, 2, -1, -1], [1, 3, -1, -1]]] * 2
        ),
        logical_to_all_physical_map_cpu=torch.tensor(
            [[[0, 2, -1, -1], [1, 3, -1, -1]]] * 2
        ),
        logical_to_all_physical_map_num_valid=torch.tensor([[2, 2], [2, 2]]),
        ep_size=2,
        logical_to_rank_dispatch_physical_map=None,
        mlb_routing_metadata={i: {"rank_quota_prefix": q} for i, q in enumerate(quota)},
    )


def test_partial_commit_keeps_other_layer_quota():
    from moe_load_balancer.adapters.sglang import to_placement_snapshot

    old = [torch.tensor([[1, 2], [1, 2]]) for _ in range(2)]
    new = [torch.tensor([[3, 4], [3, 4]]) for _ in range(2)]
    live = metadata(old)
    candidate = metadata(new)
    assert to_placement_snapshot(live, 0).metadata["rank_quota_prefix"] is old[0]
    live.update(candidate, [1])
    assert to_placement_snapshot(live, 0).metadata["rank_quota_prefix"] is old[0]
    assert to_placement_snapshot(live, 1).metadata["rank_quota_prefix"] is new[1]


def test_rank_counts_retain_source_rank_and_sum_recorded_steps():
    from sglang.srt.eplb.expert_distribution import _StatAccumulator

    physical = torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=torch.int32)
    expected = torch.tensor([[16, 20]], dtype=torch.int64)

    class Group:
        world_size = 2

        def all_gather(self, value, dim):
            torch.testing.assert_close(value, expected)
            return torch.cat((value, value * 2), dim=dim)

    accumulator = object.__new__(_StatAccumulator)
    accumulator._global_physical_count_of_buffered_step = SimpleNamespace(
        get_all=lambda: physical
    )
    accumulator._expert_location_metadata = SimpleNamespace(
        num_layers=1,
        num_logical_experts=2,
        physical_to_logical_map=torch.tensor([[0, 1, 0, 1]]),
    )
    accumulator._first_dump = False
    accumulator._rank = 0
    accumulator._get_global_average_utilization_rate = lambda: None
    with (
        get_context().parallel.override(moe_ep_group=Group()),
        patch(
            "sglang.srt.eplb.expert_distribution._placement_requires_rank_counts",
            return_value=True,
        ),
        patch(
            "torch.distributed.all_reduce",
            side_effect=AssertionError("rank counts must not be reduced"),
        ),
    ):
        output = accumulator.dump("object")
    torch.testing.assert_close(
        output["logical_count"], torch.stack((expected, expected * 2))
    )

    assert output["logical_count_layout"] == "rank_layer_expert"


@pytest.mark.parametrize(
    "data, message",
    [
        ('{"physical_to_logical_map": [[0, 1]]}', "physical mapping"),
        ('{"logical_count": [[[1, 2]], [[3, 4]]]}', "per source EP rank"),
    ],
)
def test_offline_placement_rejects_missing_rank_statistics(data, message):
    from moe_load_balancer import MoELoadBalancer

    from sglang.srt.eplb.expert_location import compute_initial_expert_location_metadata

    mlb = MoELoadBalancer.from_algorithm(
        "ultraep", ep_size=2, source_rank=0, experts_per_rank=2
    )
    with get_context().override_server_args(init_expert_location=data):
        with pytest.raises(ValueError, match=message):
            compute_initial_expert_location_metadata(
                object(), moe_ep_rank=0, moe_load_balancer=mlb
            )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"ep_num_redundant_experts": 0}, "redundant"),
        ({"ep_num_redundant_experts": 3}, "every EP rank"),
        ({"pp_size": 2}, "PP=1"),
        ({"enable_eplb": False}, "recorded placement"),
        ({"disable_cuda_graph": False}, "disable-cuda-graph"),
        ({"enable_two_batch_overlap": True}, "microbatches"),
        ({"expert_distribution_recorder_mode": "per_token"}, "stat expert"),
    ],
)
def test_unsupported_coupled_configuration_fails_early(overrides, message):
    from sglang.srt.arg_groups.moe_hook import handle_moe_load_balancer

    values = dict(
        moe_load_balancer_algorithm="ultraep",
        ep_dispatch_algorithm=None,
        enable_waterfill=False,
        ep_num_redundant_experts=2,
        ep_size=2,
        elastic_ep_backend=None,
        pp_size=1,
        enable_eplb=True,
        init_expert_location="trivial",
        disable_cuda_graph=True,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        expert_distribution_recorder_mode=None,
    )
    values.update(overrides)
    with patch(
        "sglang.srt.arg_groups.moe_hook.resolving_view",
        return_value=SimpleNamespace(**values),
    ):
        with pytest.raises(ValueError, match=message):
            handle_moe_load_balancer(object())


def test_coupled_bootstrap_uses_fixed_master_slots():
    from moe_load_balancer import MoELoadBalancer

    from sglang.srt.eplb.expert_location import compute_initial_expert_location_metadata

    mlb = MoELoadBalancer.from_algorithm(
        "ultraep", ep_size=2, source_rank=0, experts_per_rank=3
    )
    common = dict(
        ep_size=2,
        model_config_for_expert_location=SimpleNamespace(
            num_layers=1, num_logical_experts=4
        ),
    )
    with (
        get_context().override_server_args(
            init_expert_location="trivial", ep_num_redundant_experts=2
        ),
        patch.object(ExpertLocationMetadata, "_init_common", return_value=common),
        patch.object(ExpertLocationMetadata, "init_by_mapping") as initialize,
    ):
        compute_initial_expert_location_metadata(object(), 0, mlb)
    assert initialize.call_args.args[1].tolist() == [[0, 1, 0, 2, 3, 2]]


def test_auto_deepep_records_source_counts_before_dispatch():
    from sglang.srt.eplb.expert_distribution import _SinglePassGatherer

    with (
        get_context().override_server_args(
            moe_load_balancer_algorithm="ultraep",
            moe_a2a_backend="deepep",
            deepep_mode="auto",
        ),
        patch(
            "sglang.srt.eplb.expert_distribution._SelectExpertsSinglePassGatherer"
        ) as select,
    ):
        result = _SinglePassGatherer.init_new(object(), rank=0)
    assert result is select.return_value
