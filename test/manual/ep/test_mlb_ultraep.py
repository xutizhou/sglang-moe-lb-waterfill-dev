"""Real UltraEP planning, SGLang weight migration, and quota routing on two GPUs.

Run: torchrun --standalone --nproc-per-node=2 test/manual/ep/test_mlb_ultraep.py
Requires MLB built with MLB_BUILD_ULTRAEP_PLACEMENT=1.
"""

import os

import torch
import torch.distributed as dist
from moe_load_balancer import MoELoadBalancer
from moe_load_balancer.adapters.sglang import (
    commit_placement,
    to_placement_request,
    to_placement_snapshot,
    to_routing_request,
    to_sglang_maps,
)

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.runtime_context import get_context


def make_metadata(mapping, candidates=None, routing_metadata=None):
    if candidates is None:
        candidates = torch.full((2, 8, 10), -1, dtype=torch.int32, device="cuda")
        for layer in range(2):
            for expert in range(8):
                slots = (mapping[layer] == expert).nonzero().flatten()
                candidates[layer, expert, : slots.numel()] = slots.int()
    return ExpertLocationMetadata._init_raw(
        ep_size=2,
        physical_to_logical_map=mapping,
        logical_to_all_physical_map=candidates,
        mlb_routing_metadata=routing_metadata,
    )


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    context = get_context()
    mlb = MoELoadBalancer.from_algorithm(
        "ultraep", ep_size=2, source_rank=rank, experts_per_rank=5
    )
    mapping = mlb.build_initial_physical_to_logical_map(
        num_layers=2,
        num_logical_experts=8,
        ep_size=2,
        num_redundant_experts_per_rank=1,
        device="cuda",
    )
    with context.override_server_args(
        device="cuda",
        moe_load_balancer_algorithm="ultraep",
        ep_size=2,
        tp_size=2,
        ep_num_redundant_experts=2,
    ):
        live = make_metadata(mapping)
        weights = {}
        for layer in range(2):
            ids = mapping[layer, rank * 5 : (rank + 1) * 5]
            weights[layer] = [
                ids[:, None, None]
                .expand(5, 32, 16)
                .to(torch.float8_e4m3fn)
                .contiguous(),
                ids[:, None, None]
                .expand(5, 16, 16)
                .to(torch.float8_e4m3fn)
                .contiguous(),
                (ids.float() + 1)[:, None].expand(5, 16).contiguous(),
                (ids.float() + 2)[:, None].expand(5, 8).contiguous(),
            ]
        updater = ExpertLocationUpdater()
        cases = 0
        with context.resources.override(expert_location_metadata=live):
            commit_placement(mlb, context, [0, 1])
            for hot in (0, 7, 3):
                counts = torch.full((2, 2, 8), 32, dtype=torch.int32, device="cuda")
                counts[:, :, hot] = 8192
                request = to_placement_request(
                    counts,
                    num_physical_experts=10,
                    num_local_physical_experts=5,
                    num_groups=1,
                    num_nodes=1,
                    algorithm="ultraep",
                    policy_metadata={"rank": rank, "num_nvl_ranks": 2},
                )
                plan = mlb.plan_placement(request)
                maps = to_sglang_maps(plan)
                target = make_metadata(
                    maps.physical_to_logical_map,
                    maps.logical_to_all_physical_map,
                    maps.routing_metadata,
                )
                assert (target.logical_to_all_physical_map_num_valid > 1).any()
                for layer in range(2):
                    updater.update(weights, target, [layer], nnodes=1, rank=rank)
                    commit_placement(mlb, context, [layer])
                    expected = live.physical_to_logical_map[
                        layer, rank * 5 : (rank + 1) * 5
                    ]
                    for index, tensor in enumerate(weights[layer]):
                        values = expected.float() + (index - 1 if index >= 2 else 0)
                        torch.testing.assert_close(
                            tensor.float(),
                            values.view(5, *([1] * (tensor.ndim - 1))).expand_as(
                                tensor
                            ),
                            rtol=0,
                            atol=0,
                        )
                        cases += 1
                    for size in (0, 1, 17, 1024):
                        ids = torch.full(
                            (size, 2), hot, dtype=torch.int64, device="cuda"
                        )
                        ids[::3, 1] = -1
                        snapshot = to_placement_snapshot(live, layer)
                        result = mlb.route_tokens(
                            to_routing_request(
                                layer_id=layer,
                                logical_topk_ids=ids,
                                topk_weights=torch.ones_like(ids, dtype=torch.float32),
                                placement=snapshot,
                            )
                        )
                        routed = result.routed_physical_topk_ids
                        valid = ids >= 0
                        assert torch.equal(routed[~valid], ids[~valid])
                        assert torch.equal(
                            live.physical_to_logical_map[layer][routed[valid]],
                            ids[valid].int(),
                        )
                        quota = snapshot.metadata["rank_quota_prefix"][hot]
                        replicas = int(snapshot.logical_to_physical_count[hot])
                        quota = quota[:replicas].long()
                        total = int(quota[-1])
                        if total > 0:
                            boundaries = torch.div(
                                quota * valid.sum() + total - 1,
                                total,
                                rounding_mode="floor",
                            )
                            expected_counts = boundaries.diff(
                                prepend=boundaries.new_zeros(1)
                            )
                            candidates = snapshot.logical_to_physical_candidates[
                                hot, :replicas
                            ]
                            actual_counts = (
                                routed[valid, None] == candidates[None, :]
                            ).sum(0)
                            torch.testing.assert_close(actual_counts, expected_counts)
                        cases += 1
                dist.barrier()
        torch.cuda.synchronize()
        print(
            f'KBC_ACCURACY {{"pass": true, "cases": {cases}, "failed_samples": 0, "tolerance": "exact"}}',
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
