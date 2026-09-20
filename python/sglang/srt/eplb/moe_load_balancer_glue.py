"""SGLang-owned glue for the framework-neutral MoE load balancer."""

from __future__ import annotations

from typing import Optional

import torch

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

    output = moe_load_balancer.route_topk(
        layer_id=layer_id,
        logical_topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        router_logits=topk_output.router_logits,
        token_count=(
            num_token_non_padded if num_token_non_padded is not None else num_tokens
        ),
        stage=_stage_from_forward_batch(forward_batch),
        routed_scaling_factor=routed_scaling_factor,
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
