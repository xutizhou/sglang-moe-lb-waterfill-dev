"""Cold native mover with ranks that have no P2P operations.

Run: torchrun --standalone --nproc-per-node=4 test/manual/ep/test_mlb_layer_commit.py
"""

import os

import torch
import torch.distributed as dist

from sglang.srt.eplb.expert_location import ExpertLocationMetadata
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.runtime_context import get_context


def metadata(mapping):
    candidates = torch.full((1, 8, 12), -1, dtype=torch.int32, device="cuda")
    for expert in range(8):
        slots = (mapping[0] == expert).nonzero().flatten()
        candidates[0, expert, : slots.numel()] = slots.int()
    return ExpertLocationMetadata._init_raw(
        ep_size=4,
        physical_to_logical_map=mapping,
        logical_to_all_physical_map=candidates,
    )


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 4
    rank = dist.get_rank()
    cpu_group = dist.new_group(backend="gloo")
    context = get_context()
    with context.override_server_args(device="cuda", tp_size=4, ep_size=4):
        before = torch.tensor([[0, 1, 0, 2, 3, 2, 4, 5, 4, 6, 7, 6]], device="cuda")
        after = before.clone()
        after[0, 5] = 0  # Only ranks0 and1 need P2P; ranks2 and3 must still initialize.
        live, target = metadata(before), metadata(after)
        weights = (
            before[0, rank * 3 : (rank + 1) * 3, None].expand(3, 16).float().clone()
        )
        with context.resources.override(expert_location_metadata=live):
            updater = ExpertLocationUpdater()
            updater.update_layer([weights], target, 0, nnodes=1, rank=rank)
            expected = after[0, rank * 3 : (rank + 1) * 3, None].expand_as(weights)
            torch.testing.assert_close(weights, expected.float(), rtol=0, atol=0)
            torch.testing.assert_close(live.physical_to_logical_map, after)
        # A CPU barrier cannot accidentally initialize the NCCL communicator
        # that update_layer itself must initialize before subset-only P2P.
        dist.barrier(group=cpu_group)
        print(
            'KBC_ACCURACY {"pass": true, "cases": 2, "failed_samples": 0, "tolerance": "exact"}',
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
