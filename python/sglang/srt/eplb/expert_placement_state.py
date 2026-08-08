"""SGLang-owned staged and active expert placement state."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from moe_load_balancer import PlacementPlan, PlacementSnapshot


@dataclass(frozen=True)
class LayerPlacement:
    physical_to_logical: torch.Tensor
    logical_to_physical: torch.Tensor
    replica_counts: torch.Tensor
    rank_quota_prefix: torch.Tensor

    def snapshot(self, layer_id: int, metadata) -> PlacementSnapshot:
        return PlacementSnapshot(
            layer_id=layer_id,
            num_logical_experts=metadata.num_logical_experts,
            num_physical_experts=metadata.num_physical_experts,
            ep_size=metadata.ep_size,
            num_local_physical_experts=metadata.num_local_physical_experts,
            physical_to_logical_map=self.physical_to_logical,
            logical_to_physical_candidates=self.logical_to_physical,
            logical_to_physical_count=self.replica_counts,
            metadata={"rank_quota_prefix": self.rank_quota_prefix},
        )


class ExpertPlacementState:
    """Keep candidates private until their weights have materialized."""

    def __init__(self) -> None:
        self._pending: dict[int, LayerPlacement] = {}
        # SGLang's generic metadata owns the public maps; policy-private
        # routing metadata stays with the committed placement here.
        self._active: dict[int, LayerPlacement] = {}
        self._batches_since_refresh: dict[int, int] = {}

    def should_refresh(
        self,
        layer_id: int,
        stage: str | None,
        interval: int,
        representative: bool,
    ) -> bool:
        if stage not in (None, "prefill", "mixed"):
            return False
        if layer_id not in self._active:
            self._batches_since_refresh[layer_id] = 0 if representative else interval
            return True
        batches = min(self._batches_since_refresh.get(layer_id, 0) + 1, interval)
        if batches >= interval and representative:
            self._batches_since_refresh[layer_id] = 0
            return True
        self._batches_since_refresh[layer_id] = batches
        return False

    def stage(self, layer_id: int, plan: PlacementPlan, metadata) -> PlacementSnapshot:
        if layer_id in self._pending:
            raise RuntimeError(f"layer {layer_id} already has a pending placement")
        rank_quota_prefix = plan.metadata.get("rank_quota_prefix")
        if rank_quota_prefix is None:
            raise ValueError("replica-quota placement requires rank_quota_prefix")
        candidate = LayerPlacement(
            physical_to_logical=plan.physical_to_logical_map[0],
            logical_to_physical=plan.logical_to_all_physical_map[0],
            replica_counts=plan.logical_to_physical_count[0],
            rank_quota_prefix=rank_quota_prefix[0],
        )
        self._pending[layer_id] = candidate
        return candidate.snapshot(layer_id, metadata)

    def pending(self, layer_id: int) -> LayerPlacement | None:
        return self._pending.get(layer_id)

    def has_active(self, layer_id: int) -> bool:
        return layer_id in self._active

    def active_snapshot(self, layer_id: int, metadata) -> PlacementSnapshot | None:
        active = self._active.get(layer_id)
        return None if active is None else active.snapshot(layer_id, metadata)

    def commit(self, layer_id: int, metadata) -> None:
        candidate = self._pending.pop(layer_id)
        metadata.physical_to_logical_map[layer_id].copy_(candidate.physical_to_logical)
        metadata.logical_to_all_physical_map[layer_id].fill_(-1)
        width = candidate.logical_to_physical.shape[-1]
        metadata.logical_to_all_physical_map[layer_id, :, :width].copy_(
            candidate.logical_to_physical
        )
        metadata.logical_to_all_physical_map_num_valid[layer_id].copy_(
            candidate.replica_counts
        )
        self._active[layer_id] = candidate


_global_expert_placement_state = ExpertPlacementState()


def get_global_expert_placement_state() -> ExpertPlacementState:
    return _global_expert_placement_state
