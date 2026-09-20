"""SGLang-owned glue for the framework-neutral MoE load balancer."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.runtime_context import get_parallel, get_resources

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
        return get_parallel().moe_ep_group.rank_in_group

    @property
    def world_size(self) -> int:
        return get_parallel().moe_ep_group.world_size

    def all_reduce_sum(self, payload: torch.Tensor) -> torch.Tensor:
        return get_parallel().moe_ep_group.all_reduce(payload)


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

    from moe_load_balancer.adapters.sglang import (
        to_placement_snapshot,
        to_routing_request,
        to_sglang_routing_output,
    )

    from sglang.srt.layers.moe.topk import StandardTopKOutput

    metadata = get_resources().expert_location_metadata
    if metadata is None:
        raise RuntimeError("MLB L2 routing requires committed expert metadata.")

    snapshot = to_placement_snapshot(metadata, layer_id)
    stage = _stage_from_forward_batch(forward_batch)
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

    get_resources().expert_distribution_recorder.on_select_experts(
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


def _stage_from_forward_batch(forward_batch) -> Optional[str]:
    if forward_batch is None:
        return None
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None:
        return None
    name = getattr(forward_mode, "name", None) or str(forward_mode).upper()
    return _FORWARD_MODE_TO_STAGE.get(name)


def get_moe_load_balancer_pipeline(algorithm):
    if algorithm is None:
        return None
    from moe_load_balancer import RoutingPipeline

    return RoutingPipeline.from_value(algorithm)
