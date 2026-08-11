from types import SimpleNamespace

import torch
from moe_load_balancer import PlacementPlan

from sglang.srt.eplb.expert_placement_state import ExpertPlacementState
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")


def test_candidate_is_published_only_after_commit():
    metadata = SimpleNamespace(
        num_layers=2,
        num_logical_experts=4,
        num_physical_experts=6,
        num_local_physical_experts=3,
        ep_size=2,
        physical_to_logical_map=torch.zeros((2, 6), dtype=torch.int32),
        physical_to_logical_map_cpu=torch.zeros((2, 6), dtype=torch.int32),
        logical_to_all_physical_map=torch.full((2, 4, 6), -1, dtype=torch.int32),
        logical_to_all_physical_map_cpu=torch.full((2, 4, 6), -1, dtype=torch.int32),
        logical_to_all_physical_map_num_valid=torch.zeros((2, 4), dtype=torch.int32),
    )
    plan = PlacementPlan(
        physical_to_logical_map=torch.tensor([[0, 1, 0, 2, 3, 2]], dtype=torch.int32),
        logical_to_all_physical_map=torch.tensor(
            [[[0, 2], [1, -1], [3, 5], [4, -1]]], dtype=torch.int32
        ),
        logical_to_physical_count=torch.tensor([[2, 1, 2, 1]], dtype=torch.int32),
        metadata={"rank_quota_prefix": torch.zeros((1, 4, 2), dtype=torch.int32)},
    )

    state = ExpertPlacementState()
    assert not state.has_active(1)
    snapshot = state.stage(1, plan, metadata)
    assert torch.count_nonzero(metadata.physical_to_logical_map[1]) == 0
    assert snapshot.physical_to_logical_map.tolist() == [0, 1, 0, 2, 3, 2]

    state.commit(1, metadata)
    assert state.pending(1) is None
    assert state.has_active(1)
    assert metadata.physical_to_logical_map[1].tolist() == [0, 1, 0, 2, 3, 2]
    assert metadata.physical_to_logical_map_cpu[1].tolist() == [0, 1, 0, 2, 3, 2]
    assert metadata.logical_to_all_physical_map_num_valid[1].tolist() == [2, 1, 2, 1]
    active = state.active_snapshot(1, metadata)
    assert active is not None
    torch.testing.assert_close(
        active.metadata["rank_quota_prefix"],
        plan.metadata["rank_quota_prefix"][0],
    )


def test_refresh_cadence_counts_only_representative_prefill_batches():
    state = ExpertPlacementState()

    assert state.should_refresh(3, "prefill", 4, representative=False)
    assert not state.should_refresh(3, "decode", 4, representative=True)
    assert state.should_refresh(3, "prefill", 4, representative=True)

    state._active[3] = object()
    assert not state.should_refresh(3, "prefill", 4, representative=True)
    assert not state.should_refresh(3, "mixed", 4, representative=True)
    assert not state.should_refresh(3, "prefill", 4, representative=True)
    assert not state.should_refresh(3, "prefill", 4, representative=False)
    assert state.should_refresh(3, "prefill", 4, representative=True)
    assert state.should_refresh(4, "prefill", 4, representative=True)
