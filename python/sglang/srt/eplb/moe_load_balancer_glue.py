"""SGLang-owned glue for the framework-neutral MoE load balancer."""

from __future__ import annotations

from typing import Optional

import torch

from moe_load_balancer import RoutingPolicyConfig
from moe_load_balancer.adapters.sglang import (
    count_logical_experts,
    to_placement_snapshot,
    to_routing_request,
    to_sglang_routing_output,
)
from sglang.srt.distributed import get_moe_ep_group
from sglang.srt.distributed.communication_op import (
    moe_expert_parallel_all_reduce,
)
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import (
    get_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    get_global_expert_location_metadata,
)
from sglang.srt.server_args import get_global_server_args


_FORWARD_MODE_TO_STAGE = {
    "EXTEND": "prefill",
    "DECODE": "decode",
    "MIXED": "mixed",
    "IDLE": "idle",
    "TARGET_VERIFY": "speculative",
    "DRAFT_EXTEND": "speculative",
}


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

    server_args = get_global_server_args()
    metadata = get_global_expert_location_metadata()
    if metadata is None:
        raise RuntimeError("MLB L2 routing requires committed expert metadata.")

    enable_lplb = server_args.ep_dispatch_algorithm == "lp"
    enable_ultraep = server_args.ep_dispatch_algorithm == "ultraep"
    enable_waterfill = server_args.enable_deepep_waterfill
    policies = []
    global_logical_count = None
    routed_rank_load = None
    active_rank_token_count = None

    if enable_lplb:
        global_logical_count = _global_logical_count(
            topk_output.topk_ids,
            metadata.num_logical_experts,
        )
        policies.append(RoutingPolicyConfig(name="lplb"))

    if enable_ultraep:
        policies.append(RoutingPolicyConfig(name="ultraep"))

    if enable_waterfill:
        policies.append(
            RoutingPolicyConfig(
                name="waterfill",
                metadata={
                    "experts_per_rank": metadata.num_local_physical_experts,
                    "world_size": metadata.ep_size,
                },
            )
        )
        if not enable_lplb and envs.SGLANG_DISABLE_STATIC_WATERFILL.get():
            local_rank_load = _count_physical_per_rank(
                topk_output.topk_ids,
                world_size=metadata.ep_size,
                experts_per_rank=metadata.num_local_physical_experts,
            )
            routed_rank_load, active_rank_token_count = _dynamic_waterfill_load(
                local_rank_load,
                (
                    num_token_non_padded
                    if num_token_non_padded is not None
                    else num_tokens
                ),
            )

    snapshot = to_placement_snapshot(metadata, layer_id)
    stage = _stage_from_forward_batch(forward_batch)
    request = to_routing_request(
        layer_id=layer_id,
        logical_topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        policies=policies,
        placement=snapshot,
        routed_physical_topk_ids=(
            None if enable_lplb or enable_ultraep else topk_output.topk_ids
        ),
        stage=stage,
        routed_rank_load=routed_rank_load,
        active_rank_token_count=active_rank_token_count,
        global_logical_count=global_logical_count,
        token_count=num_token_non_padded,
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
    return StandardTopKOutput(
        topk_weights=output.topk_weights,
        topk_ids=output.topk_ids,
        router_logits=output.router_logits,
    )


def _global_logical_count(
    logical_topk_ids: torch.Tensor,
    num_logical_experts: int,
) -> torch.Tensor:
    local_count = count_logical_experts(logical_topk_ids, num_logical_experts)
    return get_moe_ep_group().all_reduce(local_count)


def _count_physical_per_rank(
    physical_topk_ids: torch.Tensor,
    *,
    world_size: int,
    experts_per_rank: int,
) -> torch.Tensor:
    valid = physical_topk_ids >= 0
    ranks = physical_topk_ids.clamp(min=0).to(torch.int64) // experts_per_rank
    return torch.bincount(ranks[valid].reshape(-1), minlength=world_size)


def _dynamic_waterfill_load(
    local_routed_counts: torch.Tensor,
    local_token_count: int | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    group = get_moe_ep_group()
    world_size = group.world_size
    payload = torch.zeros(
        world_size * 2,
        dtype=torch.int64,
        device=local_routed_counts.device,
    )
    payload[:world_size] = local_routed_counts
    local_count_slot = payload[world_size + group.rank_in_group]
    if isinstance(local_token_count, torch.Tensor):
        local_count_slot.copy_(local_token_count.reshape(-1)[0].to(torch.int64))
    else:
        local_count_slot.fill_(local_token_count)
    payload = moe_expert_parallel_all_reduce(payload)
    return payload[:world_size], payload[world_size:]


def _stage_from_forward_batch(forward_batch) -> Optional[str]:
    if forward_batch is None:
        return None
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None:
        return None
    name = getattr(forward_mode, "name", None) or str(forward_mode).upper()
    return _FORWARD_MODE_TO_STAGE.get(name)
