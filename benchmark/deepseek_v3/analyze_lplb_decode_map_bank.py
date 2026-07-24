#!/usr/bin/env python3

import argparse
import json
from dataclasses import asdict, dataclass

import torch

from sglang.srt.eplb import eplb_algorithms


@dataclass
class LayerResult:
    layer: int
    num_replicated_experts: int
    fixed_mean_max_active: float
    fixed_p95_max_active: float
    oracle_mean_max_active: float
    oracle_p95_max_active: float
    bank_mean_max_active: dict[int, float]
    bank_p95_max_active: dict[int, float]


def _score(active_load: torch.Tensor, token_load: torch.Tensor) -> tuple:
    max_active = active_load.max(dim=1).values
    max_tokens = token_load.max(dim=1).values
    return (
        float(max_active.float().mean()),
        float(torch.quantile(max_active.float(), 0.95)),
        float(max_tokens.float().mean()),
        float(torch.quantile(max_tokens.float(), 0.95)),
    )


def _loads(
    counts: torch.Tensor,
    fixed_rank: torch.Tensor,
    movable_experts: list[int],
    movable_rank: torch.Tensor,
    phase: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    active = counts > 0
    num_chunks = counts.shape[0]
    num_ranks = int(fixed_rank.max()) + 1
    active_load = torch.zeros((num_chunks, num_ranks), dtype=torch.int32)
    token_load = torch.zeros((num_chunks, num_ranks), dtype=torch.int32)
    rank = fixed_rank.expand(num_chunks, -1).clone()
    if movable_experts:
        movable = torch.tensor(movable_experts, dtype=torch.int64)
        rank[:, movable] = movable_rank[phase][:, movable]
    active_load.scatter_add_(1, rank, active.to(torch.int32))
    token_load.scatter_add_(1, rank, counts.to(torch.int32))
    return active_load, token_load


def _optimize_bank(
    counts: torch.Tensor,
    fixed_rank: torch.Tensor,
    movable_experts: list[int],
    eligible_ranks: list[list[int]],
    bank_size: int,
    *,
    passes: int = 8,
) -> tuple[torch.Tensor, tuple]:
    num_chunks = counts.shape[0]
    phase = torch.arange(num_chunks) % bank_size
    movable_rank = fixed_rank.repeat(bank_size, 1)

    # Initialize each phase independently with a frequency-aware greedy packing.
    for bank_id in range(bank_size):
        subset = counts[phase == bank_id]
        active_weight = (subset > 0).float().mean(dim=0)
        token_weight = subset.float().mean(dim=0)
        rank_active = torch.zeros(int(fixed_rank.max()) + 1)
        rank_tokens = torch.zeros_like(rank_active)
        movable_set = set(movable_experts)
        for expert in range(counts.shape[1]):
            if expert in movable_set:
                continue
            rank = int(fixed_rank[expert])
            rank_active[rank] += active_weight[expert]
            rank_tokens[rank] += token_weight[expert]
        for expert in sorted(
            movable_experts,
            key=lambda e: (-float(active_weight[e]), -float(token_weight[e]), e),
        ):
            chosen = min(
                eligible_ranks[expert],
                key=lambda rank: (
                    max(
                        float(rank_active[other])
                        + (float(active_weight[expert]) if other == rank else 0.0)
                        for other in range(rank_active.numel())
                    ),
                    max(
                        float(rank_tokens[other])
                        + (float(token_weight[expert]) if other == rank else 0.0)
                        for other in range(rank_tokens.numel())
                    ),
                    rank,
                ),
            )
            movable_rank[bank_id, expert] = chosen
            rank_active[chosen] += active_weight[expert]
            rank_tokens[chosen] += token_weight[expert]

    # Coordinate descent directly on the observed per-chunk critical rank.
    for _ in range(passes):
        changed = False
        for bank_id in range(bank_size):
            mask = phase == bank_id
            for expert in movable_experts:
                old_rank = int(movable_rank[bank_id, expert])
                best_rank = old_rank
                active_load, token_load = _loads(
                    counts[mask],
                    fixed_rank,
                    movable_experts,
                    movable_rank[bank_id : bank_id + 1],
                    torch.zeros(mask.sum(), dtype=torch.int64),
                )
                best_score = _score(active_load, token_load)
                for rank in eligible_ranks[expert]:
                    if rank == old_rank:
                        continue
                    movable_rank[bank_id, expert] = rank
                    active_load, token_load = _loads(
                        counts[mask],
                        fixed_rank,
                        movable_experts,
                        movable_rank[bank_id : bank_id + 1],
                        torch.zeros(mask.sum(), dtype=torch.int64),
                    )
                    score = _score(active_load, token_load)
                    if score < best_score:
                        best_score = score
                        best_rank = rank
                movable_rank[bank_id, expert] = best_rank
                changed |= best_rank != old_rank
        if not changed:
            break

    return movable_rank, _score(
        *_loads(counts, fixed_rank, movable_experts, movable_rank, phase)
    )


def _oracle_score(
    counts: torch.Tensor,
    fixed_rank: torch.Tensor,
    movable_experts: list[int],
    eligible_ranks: list[list[int]],
) -> tuple:
    active = counts > 0
    num_ranks = int(fixed_rank.max()) + 1
    active_load = torch.zeros((counts.shape[0], num_ranks), dtype=torch.int32)
    token_load = torch.zeros_like(active_load)
    movable_set = set(movable_experts)
    for expert in range(counts.shape[1]):
        if expert in movable_set:
            continue
        rank = int(fixed_rank[expert])
        active_load[:, rank] += active[:, expert]
        token_load[:, rank] += counts[:, expert]
    for chunk in range(counts.shape[0]):
        for expert in sorted(
            movable_experts,
            key=lambda e: (
                -int(active[chunk, e]),
                -int(counts[chunk, e]),
                e,
            ),
        ):
            if not active[chunk, expert]:
                continue
            rank = min(
                eligible_ranks[expert],
                key=lambda r: (
                    max(
                        int(active_load[chunk, q]) + (q == r) for q in range(num_ranks)
                    ),
                    max(
                        int(token_load[chunk, q])
                        + (int(counts[chunk, expert]) if q == r else 0)
                        for q in range(num_ranks)
                    ),
                    r,
                ),
            )
            active_load[chunk, rank] += 1
            token_load[chunk, rank] += counts[chunk, expert]
    return _score(active_load, token_load)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--placement-profile", required=True)
    parser.add_argument("--decode-profile", required=True)
    parser.add_argument("--bank-sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    args = parser.parse_args()

    placement = torch.load(
        args.placement_profile, map_location="cpu", weights_only=True
    )["logical_count"]
    decode = torch.load(args.decode_profile, map_location="cpu", weights_only=True)[
        "logical_count"
    ]
    _, logical_to_all, _ = eplb_algorithms.rebalance_experts(
        tokens_per_expert=placement,
        num_physical_experts=288,
        num_local_physical_experts=72,
        num_groups=8,
        num_nodes=1,
        algorithm=eplb_algorithms.EplbAlgorithm.deepseek_hierarchical,
    )
    physical_per_rank = 72
    results = []
    for layer in range(decode.shape[1]):
        counts = decode[:, layer]
        if not torch.any(counts):
            continue
        eligible_ranks = []
        fixed_rank = torch.empty(counts.shape[1], dtype=torch.int64)
        movable_experts = []
        for expert in range(counts.shape[1]):
            physical = logical_to_all[layer, expert]
            physical = physical[physical >= 0]
            ranks = torch.unique(
                torch.div(physical, physical_per_rank, rounding_mode="floor"),
                sorted=True,
            )
            eligible_ranks.append(ranks.tolist())
            fixed_rank[expert] = ranks[0]
            if len(ranks) > 1:
                movable_experts.append(expert)

        bank_scores = {}
        for bank_size in args.bank_sizes:
            _, score = _optimize_bank(
                counts,
                fixed_rank,
                movable_experts,
                eligible_ranks,
                bank_size,
            )
            bank_scores[bank_size] = score
        oracle = _oracle_score(counts, fixed_rank, movable_experts, eligible_ranks)
        fixed = bank_scores[1]
        results.append(
            LayerResult(
                layer=layer,
                num_replicated_experts=len(movable_experts),
                fixed_mean_max_active=fixed[0],
                fixed_p95_max_active=fixed[1],
                oracle_mean_max_active=oracle[0],
                oracle_p95_max_active=oracle[1],
                bank_mean_max_active={k: v[0] for k, v in bank_scores.items()},
                bank_p95_max_active={k: v[1] for k, v in bank_scores.items()},
            )
        )
    print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
