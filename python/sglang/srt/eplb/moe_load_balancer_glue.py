"""SGLang-owned glue for the framework-neutral MoE load balancer."""

from __future__ import annotations

from typing import Optional

import torch
from moe_load_balancer.adapters.sglang import (
    count_logical_experts,
    to_placement_request,
    to_placement_snapshot,
    to_routing_request,
    to_sglang_routing_output,
)

from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.server_args import get_global_server_args

_FORWARD_MODE_TO_STAGE = {
    "EXTEND": "prefill",
    "DECODE": "decode",
    "MIXED": "mixed",
    "IDLE": "idle",
    "TARGET_VERIFY": "speculative",
    "DRAFT_EXTEND": "speculative",
}


class SGLangRoutingCollectives:
    """Expose SGLang's EP process group through MLB's generic transport API."""

    @property
    def rank(self) -> int:
        return get_moe_ep_group().rank_in_group

    @property
    def world_size(self) -> int:
        return get_moe_ep_group().world_size

    def all_reduce_sum(self, payload: torch.Tensor) -> torch.Tensor:
        return get_moe_ep_group().all_reduce(payload)


def route_topk_with_mlb(
    *,
    moe_load_balancer,
    layer_id: int,
    topk_output,
    num_tokens: int,
    num_token_non_padded: Optional[torch.Tensor],
    forward_batch,
    routed_scaling_factor: float,
):
    """Run the configured L2 pipeline and materialize SGLang TopK output."""

    from sglang.srt.layers.moe.topk import StandardTopKOutput

    metadata = get_global_expert_location_metadata()
    if metadata is None:
        raise RuntimeError("MLB L2 routing requires committed expert metadata.")

    stage = _stage_from_forward_batch(forward_batch)
    snapshot = _placement_snapshot_for_routing(
        moe_load_balancer=moe_load_balancer,
        layer_id=layer_id,
        logical_topk_ids=topk_output.topk_ids,
        metadata=metadata,
        stage=stage,
        forward_batch=forward_batch,
        num_tokens=num_tokens,
    )
    request = to_routing_request(
        layer_id=layer_id,
        logical_topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        placement=snapshot,
        stage=stage,
        token_count=(
            num_token_non_padded if num_token_non_padded is not None else num_tokens
        ),
        routed_scaling_factor=routed_scaling_factor,
    )
    decision = moe_load_balancer.route_tokens(request)
    output = to_sglang_routing_output(
        decision,
        router_logits=topk_output.router_logits,
        num_routed_experts=metadata.num_physical_experts,
        world_size=metadata.ep_size,
        routed_scaling_factor=routed_scaling_factor,
    )

    get_global_expert_distribution_recorder().on_select_experts(
        topk_ids=output.recorded_physical_topk_ids
    )
    from sglang.srt.state_capturer.routed_experts import get_global_experts_capturer

    if (capturer := get_global_experts_capturer()) is not None:
        capturer.capture(
            layer_id=layer_id,
            topk_indices=output.recorded_physical_topk_ids,
        )
    return StandardTopKOutput(
        topk_weights=output.topk_weights,
        topk_ids=output.topk_ids,
        router_logits=output.router_logits,
    )


def _placement_snapshot_for_routing(
    *,
    moe_load_balancer,
    layer_id: int,
    logical_topk_ids: torch.Tensor,
    metadata,
    stage: Optional[str],
    forward_batch,
    num_tokens: int,
):
    placement_policy = moe_load_balancer.placement_policy
    if placement_policy is None:
        return to_placement_snapshot(metadata, layer_id)

    from sglang.srt.eplb.expert_placement_state import (
        get_global_expert_placement_state,
    )
    from sglang.srt.expert_transfer import get_expert_transfer

    server_args = get_global_server_args()
    transfer = get_expert_transfer()
    if transfer is None:
        raise RuntimeError("MoE runtime balancing requires an expert transfer.")
    placement_state = get_global_expert_placement_state()
    if placement_state.should_refresh(
        layer_id,
        stage,
        server_args.moe_balance_refresh_interval,
        _representative_token_count(forward_batch, num_tokens)
        >= server_args.moe_balance_refresh_min_tokens,
    ):
        per_rank_count = _per_rank_logical_count(
            logical_topk_ids,
            metadata.num_logical_experts,
            transfer,
        )
        placement_request = to_placement_request(
            logical_count=per_rank_count[:, None, :],
            num_physical_experts=metadata.num_physical_experts,
            num_local_physical_experts=metadata.num_local_physical_experts,
            num_groups=None,
            num_nodes=server_args.nnodes,
            algorithm=placement_policy,
            policy_metadata={
                "rank": get_moe_ep_group().rank_in_group,
                "num_nvl_ranks": transfer.nvl_domain_size,
            },
        )
        plan = moe_load_balancer.plan_placement(placement_request)
        return placement_state.stage(layer_id, plan, metadata)

    active = placement_state.active_snapshot(layer_id, metadata)
    return active if active is not None else to_placement_snapshot(metadata, layer_id)


def _per_rank_logical_count(
    logical_topk_ids: torch.Tensor,
    num_logical_experts: int,
    transfer=None,
) -> torch.Tensor:
    local_count = count_logical_experts(
        logical_topk_ids,
        num_logical_experts,
        dtype=torch.int32,
    )
    gather_loads = getattr(transfer, "all_gather_loads", None)
    if gather_loads is not None:
        gathered = gather_loads(local_count)
        if gathered is not None:
            return gathered
    gathered = get_moe_ep_group().all_gather(local_count, dim=0)
    return gathered.reshape(get_moe_ep_group().world_size, num_logical_experts)


def _stage_from_forward_batch(forward_batch) -> Optional[str]:
    if forward_batch is None:
        return None
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None:
        return None
    name = getattr(forward_mode, "name", None) or str(forward_mode).upper()
    return _FORWARD_MODE_TO_STAGE.get(name)


def _representative_token_count(forward_batch, fallback: int) -> int:
    if forward_batch is None:
        return fallback
    global_counts = getattr(forward_batch, "original_global_num_tokens_cpu", None)
    if global_counts:
        return sum(global_counts)
    local_count = getattr(forward_batch, "num_token_non_padded_cpu", None)
    return fallback if local_count is None else int(local_count)
