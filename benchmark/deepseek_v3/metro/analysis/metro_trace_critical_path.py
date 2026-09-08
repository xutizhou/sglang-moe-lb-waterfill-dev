#!/usr/bin/env python3
"""Per-layer critical-path analysis of SGLang decode Torch-Profiler traces.

Why this exists
---------------
``artifacts/metro_decode_full61_20260904/analyze_traces.py`` reports the MoE
grouped-GEMM "critical" time as ``max_rank( sum_layers(t) )``: it first sums
every MoE layer's grouped GEMM inside one decode step per rank, then takes the
maximum over EP ranks.  That is the wrong statistic for a pipeline in which
every MoE layer ends in a collective (reduce-scatter / combine): the layer
cannot finish before the slowest rank finishes, so the critical path is
``sum_layers( max_rank(t) )``.  The two agree only when the same rank is the
slowest in every layer.  With random expert activation the slow rank rotates,
so summing first averages away exactly the imbalance a policy such as METRO
removes.  On the 61-layer run this made a ~1% "imbalance" out of what is very
likely 5-8%.

This script recomputes both statistics from the same traces, so the two can be
compared directly, and additionally reports GPU-utilisation figures that show
whether the run was GPU-bound at all (it was not: CUDA graphs were disabled).

Input
-----
One directory per mode containing ``*.trace.json.gz`` Chrome traces whose file
names carry ``TP-x-DP-y-PP-z-EP-w``; the same layout Codex's analyzer expects.
Decode steps are ``user_annotation`` events whose name matches
``--step-name`` (default ``step[DECODE bs=8]``).  Grouped-GEMM kernels are
matched by regex (defaults follow the DeepGEMM names in the 61-layer traces).

No torch or numpy is required.  Run ``--self-test`` to validate the parsing and
the statistics on a synthetic trace.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import random
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

RANK_RE = re.compile(r"TP-(\d+)-DP-(\d+)-PP-(\d+)-EP-(\d+)")


# --------------------------------------------------------------------------- #
# Kernel classification
# --------------------------------------------------------------------------- #
@dataclass
class KernelMatcher:
    grouped_gemm: re.Pattern
    gate_up: re.Pattern
    down: re.Pattern
    nccl: re.Pattern

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "KernelMatcher":
        return cls(
            grouped_gemm=re.compile(args.grouped_gemm_regex, re.IGNORECASE),
            gate_up=re.compile(args.gate_up_regex),
            down=re.compile(args.down_regex),
            nccl=re.compile(args.nccl_regex, re.IGNORECASE),
        )

    def is_grouped_gemm(self, name: str) -> bool:
        return bool(self.grouped_gemm.search(name)) and (
            bool(self.gate_up.search(name)) or bool(self.down.search(name))
        )

    def is_gate_up(self, name: str) -> bool:
        return bool(self.gate_up.search(name))

    def is_nccl(self, name: str) -> bool:
        return bool(self.nccl.search(name))


# --------------------------------------------------------------------------- #
# Trace parsing
# --------------------------------------------------------------------------- #
def union_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    start, end = intervals[0]
    for s, e in intervals[1:]:
        if s <= end:
            end = max(end, e)
        else:
            total += end - start
            start, end = s, e
    return total + end - start


@dataclass
class RankTrace:
    tp: int
    dp: int
    pp: int
    ep: int
    file: str
    # steps[i] -> list of per-layer grouped-GEMM union durations (ms)
    layer_ms: list[list[float]]
    # steps[i] -> dict of utilisation figures
    util: list[dict[str, float]]
    warnings: list[str] = field(default_factory=list)


def _load_trace(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
        return json.load(handle)


def parse_rank(
    path: Path,
    matcher: KernelMatcher,
    step_name: str,
    skip_first: int,
    skip_last: int,
) -> RankTrace:
    match = RANK_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse rank ids from {path.name}")
    tp, dp, pp, ep = map(int, match.groups())
    trace = _load_trace(path)
    events = trace["traceEvents"]

    steps = sorted(
        (
            e
            for e in events
            if e.get("cat") == "user_annotation"
            and e.get("ph") == "X"
            and e.get("name") == step_name
        ),
        key=lambda e: e["ts"],
    )
    if len(steps) <= skip_first + skip_last:
        raise ValueError(f"{path.name}: only {len(steps)} steps named {step_name!r}")
    starts = [e["ts"] for e in steps]
    # Step i spans [starts[i], starts[i+1]); the last step has no end boundary.
    selected = list(range(skip_first, len(steps) - max(skip_last, 1)))
    selected_set = set(selected)

    per_step_gemm: dict[int, list[tuple[float, float, bool]]] = defaultdict(list)
    per_step_all: dict[int, list[tuple[float, float]]] = defaultdict(list)
    per_step_nccl: dict[int, list[tuple[float, float]]] = defaultdict(list)

    for e in events:
        if e.get("cat") != "kernel" or e.get("ph") != "X":
            continue
        ts = e["ts"]
        idx = bisect.bisect_right(starts, ts) - 1
        if idx not in selected_set:
            continue
        end_bound = starts[idx + 1]
        if ts >= end_bound:
            continue
        end = min(ts + e.get("dur", 0.0), end_bound)
        if end <= ts:
            continue
        name = e.get("name", "")
        per_step_all[idx].append((ts, end))
        if matcher.is_nccl(name):
            per_step_nccl[idx].append((ts, end))
        if matcher.is_grouped_gemm(name):
            per_step_gemm[idx].append((ts, end, matcher.is_gate_up(name)))

    warnings: list[str] = []
    layer_ms: list[list[float]] = []
    util: list[dict[str, float]] = []
    for idx in selected:
        kernels = sorted(per_step_gemm.get(idx, []))
        layers: list[list[tuple[float, float]]] = []
        for ts, end, is_gate_up in kernels:
            if is_gate_up or not layers:
                if not is_gate_up:
                    warnings.append(
                        f"step {idx}: down-proj grouped GEMM before any gate/up kernel"
                    )
                layers.append([])
            layers[-1].append((ts, end))
        layer_ms.append([union_us(layer) / 1000.0 for layer in layers])
        span = (starts[idx + 1] - starts[idx]) / 1000.0
        all_busy = union_us(per_step_all.get(idx, [])) / 1000.0
        nccl_busy = union_us(per_step_nccl.get(idx, [])) / 1000.0
        nccl_set = set(per_step_nccl.get(idx, []))
        non_nccl = [iv for iv in per_step_all.get(idx, []) if iv not in nccl_set]
        util.append(
            {
                "step_span_ms": span,
                "gpu_busy_ms": all_busy,
                "nccl_busy_ms": nccl_busy,
                "non_nccl_busy_ms": union_us(non_nccl) / 1000.0,
                "grouped_gemm_ms": sum(layer_ms[-1]),
            }
        )
    return RankTrace(tp, dp, pp, ep, path.name, layer_ms, util, warnings)


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def summarize_stage(ranks: list[RankTrace]) -> dict:
    """Per PP stage: compare sum-of-max vs max-of-sum grouped-GEMM statistics."""

    ranks = sorted(ranks, key=lambda r: r.ep)
    num_steps = min(len(r.layer_ms) for r in ranks)
    per_step: list[dict] = []
    slowest_hist: Counter = Counter()
    layer_excess: list[float] = []
    skipped = 0

    for step in range(num_steps):
        counts = {len(r.layer_ms[step]) for r in ranks}
        if len(counts) != 1:
            skipped += 1
            continue
        num_layers = counts.pop()
        if num_layers == 0:
            skipped += 1
            continue
        sum_max = 0.0
        sum_mean = 0.0
        sum_min = 0.0
        for layer in range(num_layers):
            values = [r.layer_ms[step][layer] for r in ranks]
            layer_max = max(values)
            layer_mean = _mean(values)
            sum_max += layer_max
            sum_mean += layer_mean
            sum_min += min(values)
            slowest_hist[values.index(layer_max)] += 1
            if layer_mean > 0:
                layer_excess.append(layer_max / layer_mean - 1.0)
        rank_sums = [sum(r.layer_ms[step]) for r in ranks]
        per_step.append(
            {
                "num_layers": num_layers,
                "sum_of_rank_max_ms": sum_max,
                "sum_of_rank_mean_ms": sum_mean,
                "sum_of_rank_min_ms": sum_min,
                "max_of_rank_sum_ms": max(rank_sums),
                "rank_sums_ms": rank_sums,
            }
        )

    if not per_step:
        raise ValueError("No step with consistent layer counts across ranks")

    def col(key: str) -> list[float]:
        return [s[key] for s in per_step]

    sum_max = _mean(col("sum_of_rank_max_ms"))
    sum_mean = _mean(col("sum_of_rank_mean_ms"))
    max_sum = _mean(col("max_of_rank_sum_ms"))
    rank_sum_means = [
        _mean(s["rank_sums_ms"][i] for s in per_step) for i in range(len(ranks))
    ]

    util_keys = ["step_span_ms", "gpu_busy_ms", "nccl_busy_ms", "non_nccl_busy_ms"]
    util = {
        key: [_mean(r.util[s][key] for s in range(num_steps)) for r in ranks]
        for key in util_keys
    }
    util["gpu_busy_fraction"] = [
        b / s if s else float("nan")
        for b, s in zip(util["gpu_busy_ms"], util["step_span_ms"])
    ]
    util["non_nccl_busy_fraction"] = [
        b / s if s else float("nan")
        for b, s in zip(util["non_nccl_busy_ms"], util["step_span_ms"])
    ]

    return {
        "ranks": [r.file for r in ranks],
        "num_steps_used": len(per_step),
        "num_steps_skipped_inconsistent_layers": skipped,
        "layers_per_step": per_step[0]["num_layers"],
        # --- the statistic that matters for the critical path ---
        "grouped_gemm_sum_of_rank_max_ms": sum_max,
        # --- what a perfectly balanced policy could reach ---
        "grouped_gemm_sum_of_rank_mean_ms": sum_mean,
        "grouped_gemm_sum_of_rank_min_ms": _mean(col("sum_of_rank_min_ms")),
        # --- Codex's original statistic, for comparison ---
        "grouped_gemm_max_of_rank_sum_ms": max_sum,
        "per_rank_mean_sum_ms": rank_sum_means,
        "critical_path_excess_over_mean_pct": (sum_max / sum_mean - 1.0) * 100.0,
        "codex_metric_excess_over_mean_pct": (max_sum / sum_mean - 1.0) * 100.0,
        "hidden_imbalance_pct": (sum_max / max_sum - 1.0) * 100.0,
        "per_layer_max_over_mean_minus_1": {
            "mean_pct": _mean(layer_excess) * 100.0,
            "p50_pct": statistics.median(layer_excess) * 100.0,
            "p90_pct": sorted(layer_excess)[int(0.9 * (len(layer_excess) - 1))] * 100.0,
        },
        "slowest_rank_histogram": {str(k): v for k, v in sorted(slowest_hist.items())},
        "utilisation": util,
        "warnings": sorted({w for r in ranks for w in r.warnings})[:20],
    }


def analyze_dir(
    directory: Path,
    matcher: KernelMatcher,
    step_name: str,
    skip_first: int,
    skip_last: int,
) -> dict:
    paths = sorted(directory.glob("*.trace.json*"))
    if not paths:
        raise FileNotFoundError(f"No *.trace.json[.gz] under {directory}")
    ranks = [parse_rank(p, matcher, step_name, skip_first, skip_last) for p in paths]
    stages: dict[str, dict] = {}
    for pp in sorted({r.pp for r in ranks}):
        stages[str(pp)] = summarize_stage([r for r in ranks if r.pp == pp])
    total = {
        key: sum(stage[key] for stage in stages.values())
        for key in (
            "grouped_gemm_sum_of_rank_max_ms",
            "grouped_gemm_sum_of_rank_mean_ms",
            "grouped_gemm_max_of_rank_sum_ms",
        )
    }
    total["critical_path_excess_over_mean_pct"] = (
        total["grouped_gemm_sum_of_rank_max_ms"]
        / total["grouped_gemm_sum_of_rank_mean_ms"]
        - 1.0
    ) * 100.0
    return {"stages": stages, "all_stages": total}


def compare(baseline: dict, candidate: dict) -> dict:
    b = baseline["all_stages"]
    c = candidate["all_stages"]
    out: dict[str, dict] = {}
    for key, label in (
        ("grouped_gemm_sum_of_rank_max_ms", "critical_path_sum_of_rank_max"),
        ("grouped_gemm_max_of_rank_sum_ms", "codex_metric_max_of_rank_sum"),
        ("grouped_gemm_sum_of_rank_mean_ms", "total_work_sum_of_rank_mean"),
    ):
        out[label] = {
            "baseline_ms": b[key],
            "candidate_ms": c[key],
            "delta_ms": c[key] - b[key],
            "delta_pct": (c[key] / b[key] - 1.0) * 100.0 if b[key] else float("nan"),
        }
    return out


def print_report(result: dict) -> None:
    for mode in ("baseline", "candidate"):
        if mode not in result:
            continue
        print(f"\n=== {mode}: {result[mode]['dir']} ===")
        for pp, stage in result[mode]["analysis"]["stages"].items():
            print(
                f"  PP{pp}: layers/step={stage['layers_per_step']} "
                f"steps={stage['num_steps_used']} "
                f"(skipped {stage['num_steps_skipped_inconsistent_layers']})"
            )
            print(
                f"    grouped GEMM  sum_l max_r = {stage['grouped_gemm_sum_of_rank_max_ms']:.3f} ms"
                f"   max_r sum_l = {stage['grouped_gemm_max_of_rank_sum_ms']:.3f} ms"
                f"   sum_l mean_r = {stage['grouped_gemm_sum_of_rank_mean_ms']:.3f} ms"
            )
            print(
                f"    excess over perfect balance: critical-path {stage['critical_path_excess_over_mean_pct']:.2f}%"
                f"   codex-metric {stage['codex_metric_excess_over_mean_pct']:.2f}%"
                f"   (hidden by summing first: {stage['hidden_imbalance_pct']:.2f}%)"
            )
            ex = stage["per_layer_max_over_mean_minus_1"]
            print(
                f"    per-layer max/mean-1: mean {ex['mean_pct']:.2f}%  p50 {ex['p50_pct']:.2f}%  p90 {ex['p90_pct']:.2f}%"
                f"   slowest-rank histogram {stage['slowest_rank_histogram']}"
            )
            u = stage["utilisation"]
            busy = ", ".join(f"{x*100:.0f}%" for x in u["gpu_busy_fraction"])
            nonnccl = ", ".join(f"{x*100:.0f}%" for x in u["non_nccl_busy_fraction"])
            print(
                f"    step span {u['step_span_ms'][0]:.1f} ms; GPU busy incl. NCCL [{busy}];"
                f" compute-only (no NCCL) [{nonnccl}]"
            )
            for w in stage["warnings"]:
                print(f"    WARNING: {w}")
    if "comparison" in result:
        print("\n=== baseline -> candidate (all PP stages summed) ===")
        for label, row in result["comparison"].items():
            print(
                f"  {label:32s} {row['baseline_ms']:.3f} -> {row['candidate_ms']:.3f} ms"
                f"  ({row['delta_ms']:+.3f} ms, {row['delta_pct']:+.2f}%)"
            )


# --------------------------------------------------------------------------- #
# Self-test with a synthetic trace
# --------------------------------------------------------------------------- #
def _synthetic_trace(
    directory: Path,
    *,
    ranks: int,
    steps: int,
    layers: int,
    rotate_slow_rank: bool,
    seed: int,
) -> dict[str, float]:
    """Write ranks x traces where each layer has one slow rank (rotating or fixed).

    Returns ground-truth sum_l max_r and max_r sum_l per step.
    """

    rng = random.Random(seed)
    base = 800.0  # us per layer per rank
    slow_extra = 100.0
    per_layer_rank: list[list[list[float]]] = []  # [step][layer][rank]
    for s in range(steps):
        step_layers = []
        for l in range(layers):
            slow = (l + s) % ranks if rotate_slow_rank else 0
            vals = [base + rng.uniform(-5, 5) + (slow_extra if r == slow else 0.0) for r in range(ranks)]
            step_layers.append(vals)
        per_layer_rank.append(step_layers)

    step_span = 20000.0
    gate_name = "deep_gemm::sm90_fp8_gemm_1d1d_impl<(deep_gemm::GemmType)2, 4096u, 7168u, ...>"
    down_name = "deep_gemm::sm90_fp8_gemm_1d1d_impl<(deep_gemm::GemmType)2, 7168u, 2048u, ...>"
    for r in range(ranks):
        events = []
        for s in range(steps + 2):  # +2: first and last are skipped by defaults
            t0 = s * step_span
            events.append(
                {"cat": "user_annotation", "ph": "X", "name": "step[DECODE bs=8]", "ts": t0, "dur": step_span}
            )
            if 1 <= s <= steps:
                cursor = t0 + 100.0
                for l in range(layers):
                    dur = per_layer_rank[s - 1][l][r]
                    events.append({"cat": "kernel", "ph": "X", "name": gate_name, "ts": cursor, "dur": dur * 0.6})
                    cursor += dur * 0.6
                    events.append({"cat": "kernel", "ph": "X", "name": down_name, "ts": cursor, "dur": dur * 0.4})
                    cursor += dur * 0.4 + 50.0
                    events.append({"cat": "kernel", "ph": "X", "name": "ncclDevKernel_ReduceScatter", "ts": cursor, "dur": 30.0})
                    cursor += 40.0
        path = directory / f"synthetic-TP-{r}-DP-{r}-PP-0-EP-{r}.trace.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump({"traceEvents": events}, handle)

    truth_sum_max = _mean(
        sum(max(per_layer_rank[s][l]) for l in range(layers)) for s in range(steps)
    ) / 1000.0
    truth_max_sum = _mean(
        max(sum(per_layer_rank[s][l][r] for l in range(layers)) for r in range(ranks))
        for s in range(steps)
    ) / 1000.0
    return {"sum_of_rank_max_ms": truth_sum_max, "max_of_rank_sum_ms": truth_max_sum}


def self_test(matcher: KernelMatcher) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rotate in (True, False):
            d = root / ("rotate" if rotate else "fixed")
            d.mkdir()
            truth = _synthetic_trace(d, ranks=4, steps=6, layers=10, rotate_slow_rank=rotate, seed=1)
            analysis = analyze_dir(d, matcher, "step[DECODE bs=8]", skip_first=1, skip_last=1)
            stage = analysis["stages"]["0"]
            got_sum_max = stage["grouped_gemm_sum_of_rank_max_ms"]
            got_max_sum = stage["grouped_gemm_max_of_rank_sum_ms"]
            assert stage["layers_per_step"] == 10, stage["layers_per_step"]
            assert math.isclose(got_sum_max, truth["sum_of_rank_max_ms"], rel_tol=1e-6), (got_sum_max, truth)
            assert math.isclose(got_max_sum, truth["max_of_rank_sum_ms"], rel_tol=1e-6), (got_max_sum, truth)
            if rotate:
                # Slow rank rotates: summing first must hide the imbalance.
                assert stage["hidden_imbalance_pct"] > 5.0, stage["hidden_imbalance_pct"]
                assert len(stage["slowest_rank_histogram"]) == 4
            else:
                # Same rank always slow: both statistics agree.
                assert math.isclose(got_sum_max, got_max_sum, rel_tol=1e-6)
                assert stage["slowest_rank_histogram"] == {"0": 60}
            print(
                f"self-test {'rotating' if rotate else 'fixed'} slow rank: "
                f"sum_l max_r={got_sum_max:.3f} ms, max_r sum_l={got_max_sum:.3f} ms, "
                f"hidden={stage['hidden_imbalance_pct']:.2f}%  OK"
            )
    print("self-test passed")


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline-dir", type=Path, help="trace dir for the baseline (e.g. static_allgather)")
    parser.add_argument("--candidate-dir", type=Path, help="trace dir for the candidate (e.g. metro_allgather)")
    parser.add_argument("--output", type=Path, help="write full JSON here")
    parser.add_argument("--step-name", default="step[DECODE bs=8]")
    parser.add_argument("--skip-first", type=int, default=1, help="drop N leading decode steps (JIT/shape setup)")
    parser.add_argument("--skip-last", type=int, default=1, help="drop N trailing steps (no end boundary)")
    parser.add_argument(
        "--grouped-gemm-regex",
        default=r"deep_gemm.*gemmtype\)2",
        help="kernel-name regex identifying routed grouped GEMMs (case-insensitive)",
    )
    parser.add_argument("--gate-up-regex", default=r"4096u, 7168u", help="regex marking the gate/up grouped GEMM")
    parser.add_argument("--down-regex", default=r"7168u, 2048u", help="regex marking the down grouped GEMM")
    parser.add_argument("--nccl-regex", default=r"nccl")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    matcher = KernelMatcher.from_args(args)
    if args.self_test:
        self_test(matcher)
        return
    if not args.baseline_dir and not args.candidate_dir:
        parser.error("provide --baseline-dir and/or --candidate-dir, or --self-test")

    result: dict = {}
    for mode, directory in (("baseline", args.baseline_dir), ("candidate", args.candidate_dir)):
        if directory is None:
            continue
        print(f"[{mode}] parsing {directory} ...", file=sys.stderr)
        result[mode] = {
            "dir": str(directory),
            "analysis": analyze_dir(directory, matcher, args.step_name, args.skip_first, args.skip_last),
        }
    if "baseline" in result and "candidate" in result:
        result["comparison"] = compare(result["baseline"]["analysis"], result["candidate"]["analysis"])
    print_report(result)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
