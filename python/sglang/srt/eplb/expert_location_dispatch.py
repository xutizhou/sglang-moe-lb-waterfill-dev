# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.server_args import get_global_server_args


@dataclass
class ExpertLocationDispatchInfo:
    # Opaque MLB-selected inline policy. SGLang does not interpret this value.
    replica_routing_policy: str
    # (num_logical_experts,)
    partial_logical_to_rank_dispatch_physical_map: Optional[torch.Tensor]
    # (num_logical_experts, X)
    partial_logical_to_all_physical_map: torch.Tensor
    # (num_logical_experts,)
    partial_logical_to_all_physical_map_num_valid: torch.Tensor

    @classmethod
    def init_new(cls, layer_id: int):
        pipeline = get_global_server_args().get_moe_load_balancer_pipeline()
        expert_location_metadata = get_global_expert_location_metadata()
        assert expert_location_metadata is not None

        if pipeline is None or pipeline.capabilities.replica_policy is None:
            return None

        return cls(
            replica_routing_policy=pipeline.capabilities.replica_policy,
            partial_logical_to_rank_dispatch_physical_map=(
                expert_location_metadata.logical_to_rank_dispatch_physical_map[
                    layer_id, :
                ]
                if expert_location_metadata.logical_to_rank_dispatch_physical_map
                is not None
                else None
            ),
            partial_logical_to_all_physical_map=expert_location_metadata.logical_to_all_physical_map[
                layer_id, :
            ],
            partial_logical_to_all_physical_map_num_valid=expert_location_metadata.logical_to_all_physical_map_num_valid[
                layer_id, :
            ],
        )


def transform_select_experts_inputs(
    router_logits: torch.Tensor,
    correction_bias: Optional[torch.Tensor],
    info: Optional[ExpertLocationDispatchInfo],
):
    if info is not None:
        from moe_load_balancer.policies.l2.replica import (
            transform_replica_routing_inputs,
        )

        router_logits, correction_bias = transform_replica_routing_inputs(
            info.replica_routing_policy, router_logits, correction_bias
        )
    return router_logits, correction_bias


def topk_ids_logical_to_physical(
    topk_ids: torch.Tensor, info: Optional[ExpertLocationDispatchInfo]
) -> torch.Tensor:
    if info is None:
        return topk_ids

    from moe_load_balancer.policies.l2.replica import route_replicas

    return route_replicas(
        info.replica_routing_policy,
        topk_ids,
        default_physical_for_logical=info.partial_logical_to_rank_dispatch_physical_map,
        logical_to_physical_candidates=info.partial_logical_to_all_physical_map,
        logical_to_physical_count=info.partial_logical_to_all_physical_map_num_valid,
    )
