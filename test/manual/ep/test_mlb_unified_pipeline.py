"""Two-rank H20 smoke for the unified MLB EPLB/LPLB/Waterfill pipeline.

Run one process per node with ``RANK``, ``WORLD_SIZE``, ``MASTER_ADDR`` and
``MASTER_PORT`` set. This test uses real NCCL and real MLB Triton policies,
while replacing only the SGLang process-global accessors with small fixtures.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.distributed as dist
from moe_load_balancer import MoELoadBalancer
from moe_load_balancer.adapters.sglang import (
    to_placement_request,
    to_placement_snapshot,
)
from moe_load_balancer.kernels.lplb import CUDALPLBKernels
from moe_load_balancer.policies.l2.lplb import LPLBL2Router
from moe_load_balancer.policies.l2.waterfill import WaterfillL2Router
from sglang.srt.eplb import moe_load_balancer_glue as glue
from sglang.srt.layers.moe.topk import StandardTopKOutput


class _EPGroup:
    @property
    def world_size(self):
        return dist.get_world_size()

    @property
    def rank_in_group(self):
        return dist.get_rank()

    def all_reduce(self, value):
        dist.all_reduce(value)
        return value


class _Recorder:
    def __init__(self):
        self.last_ids = None

    def on_select_experts(self, topk_ids):
        self.last_ids = topk_ids


class _Metadata:
    num_logical_experts = 2
    num_physical_experts = 4
    num_local_physical_experts = 2
    ep_size = 2

    def __init__(self, device):
        self.physical_to_logical_map = torch.tensor(
            [[0, 1, 0, 1]], dtype=torch.int32, device=device
        )
        self.logical_to_all_physical_map = torch.tensor(
            [[[0, 2], [1, 3]]], dtype=torch.int32, device=device
        )
        self.logical_to_all_physical_map_num_valid = torch.tensor(
            [[2, 2]], dtype=torch.int32, device=device
        )
        self.logical_to_rank_dispatch_physical_map = torch.tensor(
            [[0, 1]], dtype=torch.int32, device=device
        )


def _run_l1_eplb(mlb, device):
    request = to_placement_request(
        torch.tensor([[12, 4]], dtype=torch.int32, device=device),
        num_physical_experts=4,
        num_local_physical_experts=2,
        num_groups=1,
        num_nodes=1,
        algorithm="deepseek",
    )
    plan = mlb.plan_placement(request)
    assert plan.physical_to_logical_map.shape == (1, 4)
    assert plan.logical_to_all_physical_map.shape[:2] == (1, 2)


def _run_l2_mode(mlb, recorder, rank, mode, device):
    enable_lplb = mode in ("lplb", "combined", "combined_idle")
    enable_waterfill = mode in (
        "waterfill",
        "waterfill_dynamic",
        "combined",
        "combined_idle",
    )
    dynamic_waterfill = mode == "waterfill_dynamic"
    glue.get_global_server_args = lambda: SimpleNamespace(
        ep_dispatch_algorithm="lp" if enable_lplb else "static",
        enable_deepep_waterfill=enable_waterfill,
    )
    glue.envs.SGLANG_DISABLE_STATIC_WATERFILL = SimpleNamespace(
        get=lambda: dynamic_waterfill
    )

    if mode == "combined_idle" and rank != 0:
        logical_ids = torch.empty((0, 2), dtype=torch.int32, device=device)
    elif rank == 0:
        logical_ids = torch.tensor(
            [[0, 1], [0, 1]], dtype=torch.int32, device=device
        )
    else:
        logical_ids = torch.tensor(
            [[0, 1], [1, 1]], dtype=torch.int32, device=device
        )
    num_tokens = logical_ids.shape[0]
    output = glue.route_topk_with_mlb(
        moe_load_balancer=mlb,
        layer_id=0,
        topk_output=StandardTopKOutput(
            topk_weights=torch.ones(
                (num_tokens, 2), dtype=torch.float32, device=device
            ),
            topk_ids=logical_ids,
            router_logits=torch.zeros(
                (num_tokens, 2), dtype=torch.float32, device=device
            ),
        ),
        num_tokens=num_tokens,
        num_token_non_padded=torch.tensor(
            num_tokens, dtype=torch.int32, device=device
        ),
        forward_batch=None,
        routed_scaling_factor=1.0,
    )

    expected_width = 3 if enable_waterfill else 2
    assert output.topk_ids.shape == (num_tokens, expected_width)
    assert recorder.last_ids.shape == (num_tokens, 2)
    assert torch.all((recorder.last_ids >= 0) & (recorder.last_ids < 4))


def main():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)

    recorder = _Recorder()
    metadata = _Metadata(device)
    ep_group = _EPGroup()
    glue.get_moe_ep_group = lambda: ep_group
    glue.moe_expert_parallel_all_reduce = ep_group.all_reduce
    glue.get_global_expert_location_metadata = lambda: metadata
    glue.get_global_expert_distribution_recorder = lambda: recorder

    mlb = MoELoadBalancer(
        routing_policies={
            "lplb": LPLBL2Router(
                kernels=CUDALPLBKernels(),
                num_gpus=2,
            ),
            "waterfill": WaterfillL2Router(source_rank=rank, world_size=2),
        }
    )
    _run_l1_eplb(mlb, device)
    placement = to_placement_snapshot(metadata, 0)
    mlb.prepare_routing_layer("lplb", placement)
    for mode in (
        "lplb",
        "waterfill",
        "waterfill_dynamic",
        "combined",
        "combined_idle",
    ):
        _run_l2_mode(mlb, recorder, rank, mode, device)
        dist.barrier()

    if rank == 0:
        print("PASS: EPLB, LPLB, Waterfill, and combined L2 pipeline")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
