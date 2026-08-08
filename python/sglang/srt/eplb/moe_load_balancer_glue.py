"""SGLang-owned glue for the framework-neutral MoE load balancer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

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

if TYPE_CHECKING:
    from moe_load_balancer import L3Placement
    from moe_load_balancer.kernels.ultraep.profiling import ExpertLoadProfiler
    from ultra_ep import CollectedExpertLoads

    from sglang.srt.eplb.ultraep_expert_transfer import UltraEPExpertTransfer


_FORWARD_MODE_TO_STAGE = {
    "EXTEND": "prefill",
    "DECODE": "decode",
    "MIXED": "mixed",
    "IDLE": "idle",
    "TARGET_VERIFY": "speculative",
    "DRAFT_EXTEND": "speculative",
    "DRAFT_EXTEND_V2": "speculative",
    "SPLIT_PREFILL": "prefill",
    "DLLM_EXTEND": "prefill",
}


@dataclass(frozen=True)
class UltraEPLoadBuffers:
    """Per-runner load buffers reused by sequential MoE layers on one stream."""

    logical_loads_per_rank: torch.Tensor
    profile_global_logical_loads: Optional[torch.Tensor]
    profile_physical_loads: Optional[torch.Tensor]

    @classmethod
    def allocate(
        cls,
        *,
        ep_size: int,
        num_logical_experts: int,
        device: torch.device,
        logical_loads_per_rank: Optional[torch.Tensor] = None,
        profile_num_physical_experts: Optional[int] = None,
    ) -> UltraEPLoadBuffers:
        expected_shape = (ep_size, num_logical_experts)
        if logical_loads_per_rank is None:
            per_rank = torch.empty(
                expected_shape,
                dtype=torch.int32,
                device=device,
            )
        else:
            per_rank = logical_loads_per_rank
            if tuple(per_rank.shape) != expected_shape:
                raise ValueError(
                    "logical_loads_per_rank must have shape "
                    f"{expected_shape}, got {tuple(per_rank.shape)}"
                )
            if per_rank.dtype != torch.int32:
                raise TypeError("logical_loads_per_rank must use torch.int32")
            if per_rank.device != device:
                raise ValueError(
                    f"logical_loads_per_rank must be on {device}, got {per_rank.device}"
                )
            if not per_rank.is_contiguous():
                raise ValueError("logical_loads_per_rank must be contiguous")
        profile_global_logical_loads = profile_physical_loads = None
        if profile_num_physical_experts is not None:
            profile_global_logical_loads = torch.empty(
                num_logical_experts,
                dtype=torch.int32,
                device=device,
            )
            profile_physical_loads = torch.empty(
                profile_num_physical_experts,
                dtype=torch.int32,
                device=device,
            )
        return cls(
            logical_loads_per_rank=per_rank,
            profile_global_logical_loads=profile_global_logical_loads,
            profile_physical_loads=profile_physical_loads,
        )


@dataclass(frozen=True)
class UltraEPBalanceProfiler:
    """SGLang-side adapter for recording actual UltraEP routing balance."""

    profiler: ExpertLoadProfiler
    ep_group: Any
    buffers: UltraEPLoadBuffers

    def record_refresh(
        self,
        *,
        layer_id: int,
        routed_physical_topk_ids: torch.Tensor,
        placement: L3Placement,
    ) -> None:
        global_pre = self.buffers.profile_global_logical_loads
        physical_loads = self.buffers.profile_physical_loads
        if global_pre is None or physical_loads is None:
            raise RuntimeError("UltraEP profiling buffers were not allocated.")

        count_logical_experts(
            routed_physical_topk_ids,
            physical_loads.numel(),
            dtype=torch.int32,
            out=physical_loads,
        )
        torch.distributed.all_reduce(
            physical_loads,
            group=self.ep_group.device_group,
        )
        if not self.profiler.enabled:
            return

        torch.sum(
            self.buffers.logical_loads_per_rank,
            dim=0,
            dtype=torch.int32,
            out=global_pre,
        )
        self.profiler.stage_pre(layer_id, layer_id, global_pre)
        self.profiler.record_post(
            layer_id,
            physical_loads,
            placement.physical_to_logical_map,
        )


def _create_ultraep_balance_profiler(
    *,
    config,
    ep_group,
    buffers: UltraEPLoadBuffers,
    num_layers: int,
    num_logical_experts: int,
    num_redundant_experts_per_rank: int,
    nvl_domain_size: int,
) -> Optional[UltraEPBalanceProfiler]:
    if not config.enabled:
        return None

    from moe_load_balancer.kernels.ultraep.profiling import ExpertLoadProfiler

    group_ranks = [int(rank) for rank in ep_group.ranks]
    unique_name = str(ep_group.unique_name)
    group_id = (
        f"{unique_name.replace(':', '-')}_r"
        f"{'-'.join(str(rank) for rank in group_ranks)}"
    )
    num_local_master_experts = num_logical_experts // ep_group.world_size
    num_local_physical_experts = (
        num_local_master_experts + num_redundant_experts_per_rank
    )
    profiler = ExpertLoadProfiler(
        config=config,
        group_rank=ep_group.rank_in_group,
        global_rank=ep_group.rank,
        metadata={
            "framework": "sglang",
            "ep_group_id": group_id,
            "ep_group_unique_name": unique_name,
            "ep_group_ranks": group_ranks,
            "global_rank": ep_group.rank,
            "ep_rank": ep_group.rank_in_group,
            "ep_size": ep_group.world_size,
            "num_layers": num_layers,
            "max_microbatches": 1,
            "num_local_master_experts": num_local_master_experts,
            "num_local_redundant_experts": num_redundant_experts_per_rank,
            "num_local_physical_experts": num_local_physical_experts,
            "num_global_logical_experts": num_logical_experts,
            "num_global_physical_experts": (
                num_local_physical_experts * ep_group.world_size
            ),
            "nvl_domain_size": nvl_domain_size,
            "placement_mode": "quota",
        },
    )
    return UltraEPBalanceProfiler(
        profiler=profiler,
        ep_group=ep_group,
        buffers=buffers,
    )


def attach_ultraep(
    *, model, model_config, server_args, moe_load_balancer
) -> tuple[int, UltraEPExpertTransfer]:
    """Attach MLB placement/routing and standalone UltraEP communication."""

    try:
        from moe_load_balancer.policies.l2.ultraep import UltraEPL2Router
        from moe_load_balancer.policies.l3 import UltraEPL3Policy
    except ImportError as exc:
        raise ImportError(
            "--enable-ultraep requires moe-load-balancer with its UltraEP "
            "L2/L3 placement algorithms."
        ) from exc
    from sglang.srt.eplb.ultraep_expert_transfer import UltraEPExpertTransfer

    num_logical_experts = getattr(model_config.hf_config, "n_routed_experts", None)
    if num_logical_experts is None:
        raise ValueError("UltraEP currently supports DeepSeek-style n_routed_experts.")

    layer_storages = {}
    moe_modules = []
    unsupported_moe_types = set()
    for module in model.modules():
        if not (
            hasattr(module, "forward_deepep")
            and hasattr(module, "experts")
            and hasattr(module, "layer_id")
        ):
            continue
        if not getattr(module, "supports_ultraep_external_transfer", False):
            unsupported_moe_types.add(type(module).__name__)
            continue
        if getattr(module, "num_fused_shared_experts", 0) != 0:
            raise ValueError("UltraEP requires shared-expert fusion to be disabled.")

        experts = module.experts
        if experts.quant_method is None:
            raise ValueError("UltraEP requires a configured MoE quantization method.")
        try:
            tensor_view = experts.quant_method.get_expert_replication_tensor_view(
                experts
            )
        except NotImplementedError as exc:
            raise ValueError(
                "UltraEP does not support expert replication for quantization "
                f"method {type(experts.quant_method).__name__}."
            ) from exc

        layer_storages[module.layer_id] = tensor_view
        moe_modules.append(module)

    if unsupported_moe_types:
        names = ", ".join(sorted(unsupported_moe_types))
        raise ValueError(
            "UltraEP found DeepEP MoE modules that do not implement SGLang's "
            f"external-transfer protocol: {names}."
        )
    if not layer_storages:
        raise ValueError("UltraEP did not find any compatible DeepEP MoE layers.")

    from moe_load_balancer.kernels.ultraep.profiling import load_profile_config

    ep_group = get_moe_ep_group()
    profile_config = load_profile_config()
    expert_transfer = UltraEPExpertTransfer(
        group=ep_group.device_group,
        layer_storages=layer_storages,
        num_logical_experts=num_logical_experts,
        num_redundant_experts_per_rank=(
            server_args.ultraep_num_redundant_experts_per_rank
        ),
        overlap_transfer_with_dispatch=(
            getattr(server_args, "deepep_mode", "auto") in ("auto", "normal")
        ),
    )
    placement_policy = UltraEPL3Policy(
        layer_ids=tuple(layer_storages),
        num_logical_experts=num_logical_experts,
        ep_size=ep_group.world_size,
        num_redundant_experts_per_rank=(
            server_args.ultraep_num_redundant_experts_per_rank
        ),
        rank=ep_group.rank_in_group,
        num_nvl_ranks=expert_transfer.nvl_domain_size,
    )
    routing_policy = UltraEPL2Router()
    num_local_physical_experts = (
        num_logical_experts // ep_group.world_size
        + server_args.ultraep_num_redundant_experts_per_rank
    )
    load_buffers = UltraEPLoadBuffers.allocate(
        ep_size=ep_group.world_size,
        num_logical_experts=num_logical_experts,
        device=next(iter(layer_storages.values())).w13_weight.device,
        logical_loads_per_rank=expert_transfer.logical_loads_per_rank,
        profile_num_physical_experts=(
            num_local_physical_experts * ep_group.world_size
            if profile_config.enabled
            else None
        ),
    )
    balance_profiler = _create_ultraep_balance_profiler(
        config=profile_config,
        ep_group=ep_group,
        buffers=load_buffers,
        num_layers=len(layer_storages),
        num_logical_experts=num_logical_experts,
        num_redundant_experts_per_rank=(
            server_args.ultraep_num_redundant_experts_per_rank
        ),
        nvl_domain_size=expert_transfer.nvl_domain_size,
    )
    moe_load_balancer.register_l3_policy(placement_policy)
    moe_load_balancer.register_routing_policy("ultraep", routing_policy)
    for module in moe_modules:
        module.moe_load_balancer = moe_load_balancer
        module.expert_transfer = expert_transfer
        module._ultraep_load_buffers = load_buffers
        module._ultraep_balance_profiler = balance_profiler
        module._ultraep_active_placement = None
        module._ultraep_refresh_interval = (
            server_args.ultraep_placement_refresh_interval
        )
        module._ultraep_refresh_min_tokens = (
            server_args.ultraep_placement_refresh_min_tokens
        )
        module._ultraep_prefill_batches_since_refresh = 0
    return len(moe_modules), expert_transfer


def collect_ultraep_logical_loads(
    *,
    logical_topk_ids: torch.Tensor,
    buffers: UltraEPLoadBuffers,
    expert_transfer: UltraEPExpertTransfer,
) -> CollectedExpertLoads:
    """Enqueue one explicitly requested per-rank logical-load collection."""

    collected = expert_transfer.collect_topk_loads_async(logical_topk_ids)
    if collected.loads_per_rank is not buffers.logical_loads_per_rank:
        raise RuntimeError("UltraEP returned an unexpected load-collection buffer")
    return collected


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
    stage = stage_from_forward_batch(forward_batch)
    request = to_routing_request(
        layer_id=layer_id,
        logical_topk_ids=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        policies=policies,
        placement=snapshot,
        routed_physical_topk_ids=(None if enable_lplb else topk_output.topk_ids),
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


def stage_from_forward_batch(forward_batch) -> Optional[str]:
    if forward_batch is None:
        return None
    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None:
        return None
    name = getattr(forward_mode, "name", None) or str(forward_mode).upper()
    return _FORWARD_MODE_TO_STAGE.get(name)
