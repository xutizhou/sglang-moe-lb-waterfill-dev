"""Real UltraEP planning, SGLang weight migration, and quota routing on two GPUs.

Run: torchrun --standalone --nproc-per-node=2 test/manual/ep/test_mlb_ultraep.py
Requires MLB built with MLB_BUILD_ULTRAEP_PLACEMENT=1.
"""

import os
from types import SimpleNamespace

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
    # Match SGLang's normal trivial bootstrap. The first current-batch solve
    # moves this layout to UltraEP's fixed-master layout before dispatch.
    mapping = (
        torch.arange(10, dtype=torch.int32, device="cuda").remainder(8).repeat(2, 1)
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
                        if int(quota[-1]) > 0:
                            boundaries = quota.clamp_max(valid.sum())
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
            cases += check_current_batch_refresh(mlb, live, weights, updater, rank)
        torch.cuda.synchronize()
        print(
            f'KBC_ACCURACY {{"pass": true, "cases": {cases}, "failed_samples": 0, "tolerance": "exact"}}',
            flush=True,
        )
    dist.destroy_process_group()


def check_current_batch_refresh(mlb, live, weights, updater, rank):
    from moe_load_balancer.policies.l1 import RefreshGate
    from sglang.srt.eplb.eplb_manager import EPLBManager

    manager = object.__new__(EPLBManager)
    manager._moe_load_balancer = mlb
    manager._refresh_gate = RefreshGate(1)
    manager._rebalance_disabled_reason = None
    manager._get_model = lambda: SimpleNamespace(
        routed_experts_weights_of_layer=weights
    )
    manager._get_expert_location_updater = lambda: updater
    manager._ps = SimpleNamespace(tp_rank=rank)
    context = get_context()
    cases = 0
    with (
        context.resources.override(
            mlb_model_info={"num_logical_experts": 8, "num_groups": 1}
        ),
        context.parallel.override(
            moe_ep_rank=rank,
            moe_ep_group=SimpleNamespace(device_group=dist.group.WORLD),
        ),
    ):
        for step, hot in enumerate((0, 7, 3)):
            sizes = [4096, 0 if step == 1 else 2048]
            manager._forward_batch = SimpleNamespace(
                global_forward_mode=SimpleNamespace(
                    is_decode=lambda: False, is_idle=lambda: False
                ),
                forward_mode=SimpleNamespace(is_decode=lambda: False),
                is_extend_in_batch=True,
                original_global_num_tokens_cpu=sizes,
            )
            for layer in (0, 1):
                ids = torch.full(
                    (sizes[rank], 2), hot, dtype=torch.int64, device="cuda"
                )
                ids[::3, 1] = -1
                manager.refresh_layer(layer, ids, None)
                snapshot = to_placement_snapshot(live, layer)
                expected = live.physical_to_logical_map[
                    layer, rank * 5 : (rank + 1) * 5
                ]
                for index, tensor in enumerate(weights[layer]):
                    values = expected.float() + (index - 1 if index >= 2 else 0)
                    torch.testing.assert_close(
                        tensor.float(),
                        values.view(5, *([1] * (tensor.ndim - 1))).expand_as(tensor),
                        rtol=0,
                        atol=0,
                    )
                    cases += 1
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
                    live.physical_to_logical_map[layer][routed[valid]], ids[valid].int()
                )
                cases += 1
            # A decode-only DP batch can leave a peer idle. Neither rank may
            # advance the refresh gate or enter its placement collective.
            before = dict(manager._refresh_gate._batches_since_refresh)
            manager._forward_batch.global_forward_mode = None
            manager._forward_batch.is_extend_in_batch = False
            manager._forward_batch.forward_mode = SimpleNamespace(
                is_decode=lambda: rank == 0, is_idle=lambda: rank != 0
            )
            for layer in (0, 1):
                manager.refresh_layer(layer, ids, None)
            assert manager._refresh_gate._batches_since_refresh == before
            cases += 1
            dist.barrier()
    return cases


if __name__ == "__main__":
    main()
