"""UltraEP placement state must commit together with the expert weights."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

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


def test_single_layer_commit_moves_weights_before_publishing_metadata():
    from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater

    old = [torch.ones(2, 2) for _ in range(2)]
    new = [torch.full((2, 2), 3) for _ in range(2)]
    live, candidate = metadata(old), metadata(new)
    for name in (
        "physical_to_logical_map",
        "physical_to_logical_map_cpu",
        "logical_to_all_physical_map",
        "logical_to_all_physical_map_cpu",
        "logical_to_all_physical_map_num_valid",
    ):
        setattr(candidate, name, getattr(candidate, name)[:1].clone())
    candidate.physical_to_logical_map[0] = torch.tensor([1, 0, 1, 0])
    candidate.physical_to_logical_map_cpu.copy_(candidate.physical_to_logical_map)
    weights = [torch.zeros(2, 4), torch.ones(2, 1)]
    updater = ExpertLocationUpdater()
    buffers = []

    def move(**kwargs):
        assert live.mlb_routing_metadata[1]["rank_quota_prefix"] is old[1]
        assert kwargs["old_physical_to_logical_map"] == [0, 1, 0, 1]
        assert kwargs["new_physical_to_logical_map"] == [1, 0, 1, 0]
        buffers.append(kwargs["temp_buffers"])

    with (
        get_context().resources.override(expert_location_metadata=live),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.barrier") as initialize_group,
        patch(
            "sglang.srt.eplb.expert_location_updater.update_expert_weights_single_layer",
            side_effect=move,
        ),
    ):
        updater.update_layer(weights, candidate, 1, nnodes=1, rank=0)
    initialize_group.assert_called_once_with()
    assert len(buffers) == 1
    assert live.mlb_routing_metadata[1]["rank_quota_prefix"] is new[0]
    assert live.mlb_routing_metadata[0]["rank_quota_prefix"] is old[0]
    torch.testing.assert_close(
        live.physical_to_logical_map[0], torch.tensor([0, 1, 0, 1])
    )
    torch.testing.assert_close(
        live.physical_to_logical_map[1], torch.tensor([1, 0, 1, 0])
    )


@pytest.mark.parametrize("hashed", [False, True])
def test_topk_refresh_precedes_quota_route_even_for_empty_batch(hashed):
    from sglang.srt.layers.moe.hash_topk import HashTopK
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK

    events = []
    topk = SimpleNamespace(
        moe_load_balancer=object(),
        layer_id=3,
        mlb_placement_refresh=lambda *args: events.append("refresh"),
        topk_config=SimpleNamespace(routed_scaling_factor=1.0),
        routed_scaling_factor=1.0,
    )
    output = StandardTopKOutput(
        topk_ids=torch.empty((0, 2), dtype=torch.int32),
        topk_weights=torch.empty((0, 2)),
        router_logits=torch.empty((0, 4)),
    )
    with patch(
        "sglang.srt.eplb.moe_load_balancer_glue.route_topk_with_mlb",
        side_effect=lambda **kwargs: events.append("route"),
    ):
        cls = HashTopK if hashed else TopK
        cls._apply_moe_load_balancer(topk, output, 0)
    assert events == ["refresh", "route"]


@pytest.mark.parametrize("empty", [False, True])
def test_current_batch_refresh_includes_idle_rank_and_commits_before_routing(empty):
    from moe_load_balancer.policies.l1 import RefreshGate
    from sglang.srt.eplb.eplb_manager import EPLBManager

    context = get_context()
    live = metadata([torch.ones(2, 2), torch.ones(2, 2)])
    candidate = metadata([torch.full((2, 2), 4), torch.ones(2, 2)])
    events = []
    mlb = Mock()
    mlb.routing_capabilities.requires_placement_state = True
    mlb.plan_placement.return_value = SimpleNamespace(
        physical_to_logical_map=candidate.physical_to_logical_map[:1],
        logical_to_all_physical_map=candidate.logical_to_all_physical_map[:1],
        logical_to_physical_count=candidate.logical_to_all_physical_map_num_valid[:1],
        metadata={"rank_quota_prefix": torch.full((1, 2, 2), 4)},
    )

    def solve(request):
        torch.testing.assert_close(
            request.stats.logical_count[:, 0],
            torch.tensor([[0, 0] if empty else [2, 1], [4, 2]], dtype=torch.int32),
        )
        events.append("solve")
        return mlb.plan_placement.return_value

    mlb.plan_placement.side_effect = solve
    mlb.on_placement_committed.side_effect = lambda _: events.append("publish")
    updater = Mock()
    updater.update_layer.side_effect = lambda *args, **kwargs: events.append("move")
    manager = object.__new__(EPLBManager)
    manager._refresh_gate = RefreshGate(2)
    manager._rebalance_disabled_reason = None
    manager._moe_load_balancer = mlb
    manager._get_expert_location_updater = lambda: updater
    manager._get_model = lambda: SimpleNamespace(
        routed_experts_weights_of_layer={1: []}
    )
    manager._ps = SimpleNamespace(tp_rank=0)
    manager._forward_batch = SimpleNamespace(
        global_forward_mode=SimpleNamespace(
            is_decode=lambda: False, is_idle=lambda: False
        ),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        is_extend_in_batch=True,
        original_global_num_tokens_cpu=[0 if empty else 2, 1024],
    )

    def gather(output, local, group):
        output.copy_(torch.stack((local, local.new_tensor([4, 2]))))

    with (
        context.override_server_args(
            device="cpu",
            tp_size=2,
            ep_size=2,
            ep_num_redundant_experts=2,
            moe_load_balancer_algorithm="ultraep",
        ),
        context.resources.override(
            expert_location_metadata=live,
            mlb_model_info={"num_logical_experts": 2, "num_groups": 1},
        ),
        context.parallel.override(
            moe_ep_rank=0, moe_ep_group=SimpleNamespace(device_group=object())
        ),
        patch("torch.cuda.is_current_stream_capturing", return_value=False),
        patch(
            "torch.distributed.all_gather_into_tensor", side_effect=gather
        ) as collective,
        patch.object(ExpertLocationMetadata, "_init_raw", return_value=candidate),
    ):
        ids = torch.tensor([[0, 1], [0, -1]])
        if empty:
            ids = ids[:0]
        manager.refresh_layer(1, ids, None)
        manager.refresh_layer(1, ids, None)
        assert collective.call_count == 1
        manager.refresh_layer(1, ids, None)
        assert collective.call_count == 2
        manager._forward_batch.global_forward_mode = SimpleNamespace(
            is_decode=lambda: True
        )
        manager._forward_batch.is_extend_in_batch = False
        manager.refresh_layer(1, ids, None)
        assert collective.call_count == 2
        manager._forward_batch.global_forward_mode = None
        manager._forward_batch.forward_mode = SimpleNamespace(
            is_decode=lambda: False, is_idle=lambda: True
        )
        before = dict(manager._refresh_gate._batches_since_refresh)
        manager.refresh_layer(1, ids, None)
        assert manager._refresh_gate._batches_since_refresh == before
        assert collective.call_count == 2
    assert events == ["solve", "move", "publish"] * 2


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"ep_num_redundant_experts": 0}, "redundant"),
        ({"ep_num_redundant_experts": 3}, "every EP rank"),
        ({"pp_size": 2}, "PP=1"),
        ({"enable_eplb": False}, "--enable-eplb"),
        ({"init_expert_location": "some-layout.json"}, "init-expert-location trivial"),
        ({"disable_cuda_graph": False}, "disable-cuda-graph"),
        ({"enable_two_batch_overlap": True}, "microbatches"),
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
