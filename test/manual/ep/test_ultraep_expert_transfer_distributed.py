"""Multi-rank correctness smoke for SGLang's UltraEP expert transfer backend.

Run with one process per EP rank, for example::

    torchrun --standalone --nproc-per-node=8 \
      test/manual/ep/test_ultraep_expert_transfer_distributed.py
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from moe_load_balancer import L3Request, MoELoadBalancer, RoutingRequest
from moe_load_balancer.policies.l2.ultraep import UltraEPL2Router
from moe_load_balancer.policies.l3 import UltraEPL3Policy

from sglang.srt.eplb.ultraep_expert_transfer import UltraEPExpertTransfer


def _fill_master_rows(tensor: torch.Tensor, rank: int, masters: int, offset: float):
    for local_id in range(masters):
        logical_id = rank * masters + local_id
        tensor[local_id].fill_(logical_id + offset)


def _assert_replica_rows(
    tensor: torch.Tensor,
    physical_to_logical: torch.Tensor,
    *,
    rank: int,
    masters: int,
    replicas: int,
    offset: float,
) -> int:
    local_physical = masters + replicas
    assigned = 0
    for slot in range(replicas):
        physical_id = rank * local_physical + masters + slot
        logical_id = int(physical_to_logical[physical_id])
        if logical_id < 0:
            continue
        expected = torch.full_like(tensor[masters + slot], logical_id + offset)
        torch.testing.assert_close(tensor[masters + slot], expected, rtol=0, atol=0)
        assigned += 1
    return assigned


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    world_size = dist.get_world_size(group)

    logical_experts = 32
    replicas = 2
    masters = logical_experts // world_size
    physical = masters + replicas
    device = torch.device("cuda", local_rank)
    storage = SimpleNamespace(
        w13_weight=torch.empty((physical, 1024), dtype=torch.uint8, device=device),
        w2_weight=torch.empty((physical, 512), dtype=torch.uint8, device=device),
        w13_weight_scale=torch.empty(
            (physical, 16), dtype=torch.float32, device=device
        ),
        w2_weight_scale=torch.empty((physical, 8), dtype=torch.float32, device=device),
        auxiliary_tensors=(),
    )
    storage.w13_weight.fill_(0xFF)
    storage.w2_weight.fill_(0xFF)
    storage.w13_weight_scale.fill_(-1)
    storage.w2_weight_scale.fill_(-1)
    _fill_master_rows(storage.w13_weight, rank, masters, 1)
    _fill_master_rows(storage.w2_weight, rank, masters, 17)
    _fill_master_rows(storage.w13_weight_scale, rank, masters, 0.25)
    _fill_master_rows(storage.w2_weight_scale, rank, masters, 0.5)

    transfer = UltraEPExpertTransfer(
        group=group,
        layer_storages={0: storage},
        num_logical_experts=logical_experts,
        num_redundant_experts_per_rank=replicas,
        overlap_transfer_with_dispatch=True,
    )
    placement_policy = UltraEPL3Policy(
        layer_ids=(0,),
        num_logical_experts=logical_experts,
        ep_size=world_size,
        num_redundant_experts_per_rank=replicas,
        rank=rank,
        num_nvl_ranks=transfer.nvl_domain_size,
    )
    load_balancer = MoELoadBalancer(
        placement_planners={},
        routing_policies={
            "ultraep": UltraEPL2Router(),
        },
        l3_policy=placement_policy,
    )

    # A fixed hotspot makes placement replicate experts from rank zero onto
    # remote ranks, so this exercises communication rather than a no-op update.
    logical_topk = (
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int64, device=device)
        .expand(1024, -1)
        .contiguous()
    )
    topk_weights = torch.ones_like(logical_topk, dtype=torch.float32)
    collected_loads = transfer.collect_topk_loads_async(logical_topk)
    with torch.cuda.stream(transfer.communication_stream):
        placement = load_balancer.compute_placement(
            L3Request(
                layer_id=0,
                logical_topk_ids=logical_topk,
                topk_weights=topk_weights,
                logical_loads_per_rank=collected_loads.loads_per_rank,
            )
        )
        placement_ready = transfer.record_placement_ready(0)
        weights_ready = transfer.apply_async(placement)
    placement_ready.current_stream_wait()
    decision = load_balancer.route_tokens(
        RoutingRequest(
            layer_id=0,
            logical_topk_ids=logical_topk,
            topk_weights=topk_weights,
            policies=("ultraep",),
            transient_placement=placement,
        )
    )
    # Surrogate the work done by DeepEP normal dispatch while UltraEP's exact
    # incoming-epoch guard and local materialization run on the copy stream.
    dispatch_marker = torch.ones(1, dtype=torch.int32, device=device)
    dist.all_reduce(dispatch_marker, group=group)
    weights_ready.current_stream_wait()
    torch.cuda.synchronize()

    assigned = 0
    assigned += _assert_replica_rows(
        storage.w13_weight,
        placement.physical_to_logical_map,
        rank=rank,
        masters=masters,
        replicas=replicas,
        offset=1,
    )
    assigned += _assert_replica_rows(
        storage.w2_weight,
        placement.physical_to_logical_map,
        rank=rank,
        masters=masters,
        replicas=replicas,
        offset=17,
    )
    assigned += _assert_replica_rows(
        storage.w13_weight_scale,
        placement.physical_to_logical_map,
        rank=rank,
        masters=masters,
        replicas=replicas,
        offset=0.25,
    )
    assigned += _assert_replica_rows(
        storage.w2_weight_scale,
        placement.physical_to_logical_map,
        rank=rank,
        masters=masters,
        replicas=replicas,
        offset=0.5,
    )
    assigned_tensor = torch.tensor(assigned, dtype=torch.int32, device=device)
    dist.all_reduce(assigned_tensor)
    assert int(assigned_tensor) > 0
    assert decision.routed_physical_topk_ids.min() >= 0
    assert decision.routed_physical_topk_ids.max() < physical * world_size

    dist.barrier()
    if rank == 0:
        print(
            "PASS: UltraEP NVSHMEM load collection/transfer, MLB placement/reroute, "
            f"local materialization, and L2 routing ({int(assigned_tensor)} rows checked)"
        )
    load_balancer.close()
    transfer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
