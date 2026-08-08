"""SGLang-owned glue for the framework-neutral MoE load balancer."""

from __future__ import annotations

from typing import Optional

import torch

from moe_load_balancer import RoutingPolicyConfig
from moe_load_balancer.adapters.sglang import (
    count_logical_experts,
    to_placement_request,
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
    enable_runtime_balance = server_args.moe_balance_policy is not None
    enable_waterfill = server_args.enable_deepep_waterfill
    runtime_fast_path = (
        enable_runtime_balance and not enable_lplb and not enable_waterfill
    )
    stage = _stage_from_forward_batch(forward_batch)
    policies = []
    snapshot = None
    global_logical_count = None
    routed_rank_load = None
    active_rank_token_count = None

    if enable_lplb:
        global_logical_count = _global_logical_count(
            topk_output.topk_ids,
            metadata.num_logical_experts,
        )
        policies.append(RoutingPolicyConfig(name="lplb"))

    if enable_runtime_balance:
        from sglang.srt.eplb.expert_placement_state import (
            get_global_expert_placement_state,
        )
        from sglang.srt.expert_transfer import get_expert_transfer

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
                topk_output.topk_ids,
                metadata.num_logical_experts,
                transfer,
            )
            placement_request = to_placement_request(
                logical_count=per_rank_count[:, None, :],
                num_physical_experts=metadata.num_physical_experts,
                num_local_physical_experts=metadata.num_local_physical_experts,
                num_groups=None,
                num_nodes=server_args.nnodes,
                algorithm=server_args.moe_balance_policy,
                policy_metadata={
                    "rank": get_moe_ep_group().rank_in_group,
                    "num_nvl_ranks": transfer.nvl_domain_size,
                },
            )
            plan = moe_load_balancer.plan_placement(placement_request)
            snapshot = placement_state.stage(
                layer_id,
                plan,
                metadata,
            )
            moe_load_balancer.prepare_routing_layer(
                server_args.moe_balance_policy,
                snapshot,
            )
        elif runtime_fast_path and placement_state.has_active(layer_id):
            pass
        else:
            snapshot = placement_state.active_snapshot(layer_id, metadata)
            if snapshot is None:
                snapshot = to_placement_snapshot(metadata, layer_id)
            moe_load_balancer.prepare_routing_layer(
                server_args.moe_balance_policy,
                snapshot,
            )
        if runtime_fast_path:
            physical_ids = moe_load_balancer.route_prepared_tokens(
                server_args.moe_balance_policy,
                layer_id,
                topk_output.topk_ids,
            )
            get_global_expert_distribution_recorder().on_select_experts(
                topk_ids=physical_ids
            )
            return StandardTopKOutput(
                topk_weights=topk_output.topk_weights,
                topk_ids=physical_ids,
                router_logits=topk_output.router_logits,
            )
        policies.append(RoutingPolicyConfig(name=server_args.moe_balance_policy))

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

    if not enable_runtime_balance:
        snapshot = to_placement_snapshot(metadata, layer_id)
    request = to_routing_request(
        layer_id=layer_id,
        logical_topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        policies=policies,
        placement=snapshot,
        routed_physical_topk_ids=(
            None if enable_lplb or enable_runtime_balance else topk_output.topk_ids
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


def _representative_token_count(forward_batch, fallback: int) -> int:
    if forward_batch is None:
        return fallback
    global_counts = getattr(forward_batch, "original_global_num_tokens_cpu", None)
    if global_counts:
        return sum(global_counts)
    local_count = getattr(forward_batch, "num_token_non_padded_cpu", None)
    return fallback if local_count is None else int(local_count)
