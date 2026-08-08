"""Eight-rank correctness smoke for the minimal UltraEP transfer adapter."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

import sglang.srt.eplb.ultraep_expert_transfer as transfer_module


class _QuantMethod:
    @staticmethod
    def get_triton_quant_info(experts):
        return SimpleNamespace(
            w13_weight=experts.w13,
            w2_weight=experts.w2,
            w13_scale=experts.w13_scale,
            w2_scale=experts.w2_scale,
        )


class _Experts(nn.Module):
    def __init__(self, physical_count: int, device: torch.device):
        super().__init__()
        self.w13 = nn.Parameter(
            torch.empty(physical_count, 1024, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.w2 = nn.Parameter(
            torch.empty(physical_count, 512, dtype=torch.uint8, device=device),
            requires_grad=False,
        )
        self.w13_scale = nn.Parameter(
            torch.empty(physical_count, 16, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.w2_scale = nn.Parameter(
            torch.empty(physical_count, 8, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.quant_method = _QuantMethod()


class _Layer(nn.Module):
    def __init__(self, physical_count: int, device: torch.device):
        super().__init__()
        self.layer_id = 0
        self.experts = _Experts(physical_count, device)


def _placement(world_size: int, masters: int, replicas: int, device):
    local_physical = masters + replicas
    logical_count = world_size * masters
    physical_count = world_size * local_physical
    p2l = torch.empty(physical_count, dtype=torch.int32, device=device)
    l2p = torch.full((logical_count, world_size), -1, dtype=torch.int32, device=device)
    counts = torch.ones(logical_count, dtype=torch.int32, device=device)

    for rank in range(world_size):
        physical_base = rank * local_physical
        logical_base = rank * masters
        for local_id in range(masters):
            logical_id = logical_base + local_id
            physical_id = physical_base + local_id
            p2l[physical_id] = logical_id
            l2p[logical_id, 0] = physical_id
        for slot in range(replicas):
            logical_id = ((rank + 1) % world_size) * masters + slot % masters
            physical_id = physical_base + masters + slot
            p2l[physical_id] = logical_id
            replica_index = counts[logical_id]
            l2p[logical_id, replica_index] = physical_id
            counts[logical_id] += 1

    return SimpleNamespace(
        physical_to_logical_map=p2l.unsqueeze(0),
        logical_to_all_physical_map=l2p.unsqueeze(0),
        logical_to_all_physical_map_num_valid=counts.unsqueeze(0),
    )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    masters, replicas = 4, 2
    local_physical = masters + replicas

    layer = _Layer(local_physical, device)
    model = nn.Sequential(layer)
    for local_id in range(masters):
        logical_id = rank * masters + local_id
        layer.experts.w13[local_id].fill_(logical_id + 1)
        layer.experts.w2[local_id].fill_(logical_id + 17)
        layer.experts.w13_scale[local_id].fill_(logical_id + 0.25)
        layer.experts.w2_scale[local_id].fill_(logical_id + 0.5)

    transfer_module.get_moe_ep_group = lambda: SimpleNamespace(
        device_group=dist.group.WORLD,
        world_size=world_size,
    )
    transfer = transfer_module.UltraEPExpertTransfer(
        model,
        num_logical_experts=world_size * masters,
        num_redundant_per_rank=replicas,
    )
    placement = _placement(world_size, masters, replicas, device)
    transfer.update(placement, [0])
    torch.cuda.synchronize()

    for slot in range(replicas):
        logical_id = int(
            placement.physical_to_logical_map[0, rank * local_physical + masters + slot]
        )
        torch.testing.assert_close(
            layer.experts.w13[masters + slot],
            torch.full_like(layer.experts.w13[masters + slot], logical_id + 1),
        )
        torch.testing.assert_close(
            layer.experts.w2[masters + slot],
            torch.full_like(layer.experts.w2[masters + slot], logical_id + 17),
        )
        torch.testing.assert_close(
            layer.experts.w13_scale[masters + slot],
            torch.full_like(layer.experts.w13_scale[masters + slot], logical_id + 0.25),
        )
        torch.testing.assert_close(
            layer.experts.w2_scale[masters + slot],
            torch.full_like(layer.experts.w2_scale[masters + slot], logical_id + 0.5),
        )

    dist.barrier()
    if rank == 0:
        print("PASS: minimal SGLang adapter called UltraEP external placement API")
    transfer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
