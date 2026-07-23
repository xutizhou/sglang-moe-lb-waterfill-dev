"""Correctness tests for decode-aware DeepEP Waterfill kernels."""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.moe.deepep_waterfill_kernels import (
    LOCAL_SHARED_MARKER,
    count_active_experts_per_rank,
    materialize_waterfill_dispatch_fused,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@torch.inference_mode()
def test_count_active_experts_per_rank_deduplicates_chunk() -> None:
    topk_ids = torch.tensor(
        [
            [0, 0, 1, 4, 8, 8, 12, -1],
            [0, 1, 4, 8, 9, 10, 12, -1],
            [1, 1, 4, 9, 10, 10, 12, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    active_experts = torch.empty(16, dtype=torch.int32, device="cuda")
    counts = torch.empty(4, dtype=torch.int64, device="cuda")

    actual = count_active_experts_per_rank(
        topk_ids,
        active_experts,
        counts,
        num_routed_experts=16,
        world_size=4,
    )

    torch.testing.assert_close(
        actual.cpu(), torch.tensor([2, 1, 3, 1], dtype=torch.int64)
    )


@pytest.mark.parametrize("num_tokens", [1, 63, 64, 257, 513])
@torch.inference_mode()
def test_count_active_experts_per_rank_matches_cpu_reference(
    num_tokens: int,
) -> None:
    generator = torch.Generator().manual_seed(num_tokens)
    topk_ids_cpu = torch.randint(
        -1,
        256,
        (num_tokens, 8),
        dtype=torch.int32,
        generator=generator,
    )
    active_experts = torch.empty(256, dtype=torch.int32, device="cuda")
    counts = torch.empty(8, dtype=torch.int64, device="cuda")

    actual = count_active_experts_per_rank(
        topk_ids_cpu.cuda(),
        active_experts,
        counts,
        num_routed_experts=256,
        world_size=8,
    )
    expected = torch.zeros(8, dtype=torch.int64)
    for expert_id in torch.unique(topk_ids_cpu[topk_ids_cpu >= 0]):
        expected[expert_id.item() // 32] += 1

    torch.testing.assert_close(actual.cpu(), expected)


@torch.inference_mode()
def test_active_expert_count_is_cuda_graph_capturable() -> None:
    topk_ids = torch.tensor(
        [[0, 0, 1, 4, 8, 8, 12, -1], [0, 1, 4, 8, 9, 10, 12, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    active_experts = torch.empty(16, dtype=torch.int32, device="cuda")
    counts = torch.empty(4, dtype=torch.int64, device="cuda")
    count_active_experts_per_rank(
        topk_ids,
        active_experts,
        counts,
        num_routed_experts=16,
        world_size=4,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        count_active_experts_per_rank(
            topk_ids,
            active_experts,
            counts,
            num_routed_experts=16,
            world_size=4,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        counts.cpu(), torch.tensor([2, 1, 3, 1], dtype=torch.int64)
    )


@pytest.mark.parametrize(
    "source_rank,expected_shared_id",
    [
        (2, 9),  # rank 1 is the first least-active rank
        (3, 19),  # local rank wins a tie for minimum load
    ],
)
@torch.inference_mode()
def test_active_expert_waterfill_uses_strict_least_loaded_rank(
    source_rank: int, expected_shared_id: int
) -> None:
    topk_ids = torch.tensor(
        [[0, 1, 2, 4], [8, 9, 12, -1], [-1, -1, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    # The fused path derives [3, 1, 2, 1] from topk_ids and ignores this
    # deliberately contradictory precomputed buffer.
    rank_load = torch.tensor([0, 9, 0, 9], dtype=torch.int64, device="cuda")

    expanded_ids, expanded_weights = materialize_waterfill_dispatch_fused(
        topk_ids,
        topk_weights,
        rank_load,
        num_routed_experts=16,
        world_size=4,
        source_rank=source_rank,
        shared_weight=0.5,
        allow_all_ranks=True,
        minimize_active_experts=True,
        fuse_active_expert_load=True,
    )

    torch.testing.assert_close(
        expanded_ids[:2, -1].cpu(),
        torch.full((2,), expected_shared_id, dtype=torch.int32),
    )
    assert expanded_ids[2, -1].item() == LOCAL_SHARED_MARKER
    torch.testing.assert_close(
        expanded_weights[:, -1].cpu(), torch.tensor([0.5, 0.5, 0.0])
    )


@pytest.mark.parametrize("num_tokens", [1, 63, 64, 256])
@torch.inference_mode()
def test_fused_active_expert_waterfill_matches_cpu_reference(
    num_tokens: int,
) -> None:
    generator = torch.Generator().manual_seed(17 + num_tokens)
    scores = torch.rand((num_tokens, 256), generator=generator)
    topk_ids_cpu = torch.topk(scores, 8, dim=1, sorted=False).indices.to(torch.int32)
    topk_ids = topk_ids_cpu.cuda()
    topk_weights = torch.ones_like(topk_ids, dtype=torch.float32)
    source_rank = 3

    counts = []
    for rank in range(8):
        lower = rank * 32
        upper = lower + 32
        counts.append(
            int(
                torch.unique(
                    topk_ids_cpu[(topk_ids_cpu >= lower) & (topk_ids_cpu < upper)]
                ).numel()
            )
        )
    min_count = min(counts)
    expected_rank = (
        source_rank if counts[source_rank] == min_count else counts.index(min_count)
    )
    expected_shared_id = expected_rank * 33 + 32

    expanded_ids, _ = materialize_waterfill_dispatch_fused(
        topk_ids,
        topk_weights,
        torch.full((8,), -1, dtype=torch.int64, device="cuda"),
        num_routed_experts=256,
        world_size=8,
        source_rank=source_rank,
        shared_weight=0.5,
        allow_all_ranks=True,
        minimize_active_experts=True,
        fuse_active_expert_load=True,
    )

    torch.testing.assert_close(
        expanded_ids[:, -1].cpu(),
        torch.full((num_tokens,), expected_shared_id, dtype=torch.int32),
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
