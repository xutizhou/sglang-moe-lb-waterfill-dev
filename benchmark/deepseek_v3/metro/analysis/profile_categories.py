#!/usr/bin/env python3
"""Per-decode-step GPU kernel budget from Torch-Profiler traces (CUDA-graph aware).

For each rank trace, decode steps are delimited by the ``step[DECODE ...]``
user annotations (any batch size).  Kernel events inside each step are bucketed
by name into: routed grouped GEMM, the METRO/LPLB routing path (count,
all-reduce of the 256-float active set, assignment kernel), other NCCL, attention,
other GEMM, and the rest.  Times are per-step sums of kernel durations (kernels
inside graph replays are individual CUPTI events).  Output: per-category mean
per step per rank, rank-max, and the routing path's share of the step.

Use ``--dump-names`` once to see the kernel names in a new trace and adjust the
regexes if a different MoE runner / dispatcher is in play.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

RANK_RE = re.compile(r"TP-(\d+)-DP-(\d+)-PP-(\d+)-EP-(\d+)")
STEP_RE = re.compile(r"^step\[DECODE")
# gate/up routed grouped GEMM: one launch per MoE layer per step
PERIOD_RE = re.compile(r"deep_gemm.*, 4096u, 7168u, \d+u,|gemmtype\)[23].*4096u, 7168u", re.I)

CATEGORIES = [
    # DeepGEMM grouped (masked/contiguous) routed-expert GEMMs carry the expert
    # shape and the local-expert count: "..., 4096u, 7168u, 80u, ..." (gate/up)
    # and "..., 7168u, 2048u, 80u, ..." (down).  Dense DeepGEMM calls have zeros
    # there.  Normal-mode traces use "GemmType)2" instead.
    ("routed_gemm", re.compile(r"deep_gemm.*(?:, (?:4096u, 7168u|7168u, 2048u), \d+u,|gemmtype\)[23])|fused_moe_kernel", re.I)),
    ("metro_assign", re.compile(r"dispatch_decode_integral_kernel|metro_route_kernel|metro_route_v2_kernel", re.I)),
    ("metro_allreduce", re.compile(r"ncclDevKernel_AllReduce|cross_device_reduce|one_shot_all_reduce|two_shot_all_reduce|allreduce", re.I)),
    ("metro_count", re.compile(r"scatter_add|index_add|bincount|FillFunctor<int>", re.I)),
    ("deepep", re.compile(r"deep_ep::|low_latency|internode|intranode", re.I)),
    ("nccl_other", re.compile(r"nccl", re.I)),
    ("attention", re.compile(r"flash|fmha|mla_kv|attn|rope|paged", re.I)),
    ("moe_aux", re.compile(r"topk|router_gemm|quant_masked|act_and_mul|mask_topk|gateup|reorder", re.I)),
    ("other_gemm", re.compile(r"deep_gemm|gemm|nvjet|cutlass|cublas|matmul", re.I)),
]


def classify(name: str) -> str:
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other"


def load(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as h:  # type: ignore[arg-type]
        return json.load(h)


def parse(path: Path, skip_first: int, skip_last: int, dump_names: bool, gap_us: float, period_re=None):
    period_re = period_re or PERIOD_RE
    trace = load(path)
    ev = trace["traceEvents"]
    steps = sorted(
        (e for e in ev if e.get("cat") == "user_annotation" and e.get("ph") == "X" and STEP_RE.match(e.get("name", ""))),
        key=lambda e: e["ts"],
    )
    if steps:
        starts = [e["ts"] for e in steps]
    else:
        # GPU-only traces carry no annotations, and under CUDA graphs the GPU is
        # fed back-to-back so there is no idle gap between steps either.  Use the
        # periodic structure instead: every ``period`` occurrences of the gate/up
        # routed grouped GEMM (one per MoE layer) start a new step window.  The
        # window is phase-shifted relative to the scheduler's step but covers
        # exactly one step's worth of kernels.
        kernels = sorted((e for e in ev if e.get("cat") == "kernel" and e.get("ph") == "X"), key=lambda e: e["ts"])
        marks = [e["ts"] for e in kernels if period_re.search(e.get("name", ""))]
        starts = marks[:: gap_us and int(gap_us) or 1]
    if len(starts) <= skip_first + skip_last + 1:
        raise ValueError(f"{path.name}: only {len(starts)} decode steps")
    sel = list(range(skip_first, len(starts) - max(skip_last, 1)))
    per_step = {i: defaultdict(float) for i in sel}
    names = Counter()
    for e in ev:
        if e.get("cat") != "kernel" or e.get("ph") != "X":
            continue
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i not in per_step or e["ts"] >= starts[i + 1]:
            continue
        cat = classify(e.get("name", ""))
        per_step[i][cat] += e.get("dur", 0.0)
        per_step[i]["all"] += e.get("dur", 0.0)
        if dump_names:
            names[(cat, e["name"][:110])] += 1
    spans = [(starts[i + 1] - starts[i]) for i in sel]
    return per_step, spans, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path, help="one profile dir per mode (dir name = label)")
    ap.add_argument("--skip-first", type=int, default=2)
    ap.add_argument("--skip-last", type=int, default=1)
    ap.add_argument("--dump-names", action="store_true")
    ap.add_argument("--period-re", default=None,
                    help="kernel-name regex marking one MoE layer (default: DeepGEMM routed gate/up). "
                         "For the NCCL (--moe-a2a-backend none) path use 'deepseek_v3_topk_kernel'.")
    ap.add_argument("--gap-us", type=float, default=5, dest="gap_us", help="(annotation-less traces) MoE layers per step = gate/up GEMMs per step window")
    ap.add_argument("--min-step-ms", type=float, default=1.0, help="drop pseudo-steps shorter than this (prefill / partial)")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    result = {}
    cats = [c for c, _ in CATEGORIES] + ["other", "all"]
    for d in args.dirs:
        files = sorted(d.glob("*.trace.json*"))
        if not files:
            print(f"{d}: no traces"); continue
        ranks = {}
        all_names = Counter()
        for f in files:
            m = RANK_RE.search(f.name)
            rank = int(m.group(4)) if m else len(ranks)
            per_step, spans, names = parse(
                f, args.skip_first, args.skip_last, args.dump_names, args.gap_us,
                re.compile(args.period_re, re.I) if args.period_re else None,
            )
            all_names.update(names)
            keep = [i for i, sp in zip(sorted(per_step), spans) if sp / 1000.0 >= args.min_step_ms and per_step[i].get("routed_gemm", 0) > 0]
            if not keep:
                raise ValueError(f"{f.name}: no decode-like steps kept")
            means = {c: statistics.fmean(per_step[i].get(c, 0.0) for i in keep) / 1000.0 for c in cats}
            means["step_span_ms"] = statistics.fmean(spans[sorted(per_step).index(i)] for i in keep) / 1000.0
            means["steps"] = len(keep)
            ranks[rank] = means
        label = d.parent.name if d.name == "profile" else d.name
        summary = {c: {"rank_max_ms": max(r[c] for r in ranks.values()), "rank_mean_ms": statistics.fmean(r[c] for r in ranks.values())} for c in cats}
        summary["step_span_ms"] = statistics.fmean(r["step_span_ms"] for r in ranks.values())
        result[label] = {"ranks": ranks, "summary": summary}
        print(f"\n=== {label}: {len(ranks)} ranks, {next(iter(ranks.values()))['steps']} steps, step span {summary['step_span_ms']:.3f} ms ===")
        print(f"{'category':16s} {'rank-max ms':>11s} {'rank-mean ms':>12s} {'% of span':>9s}")
        for c in cats:
            s = summary[c]
            print(f"{c:16s} {s['rank_max_ms']:>11.3f} {s['rank_mean_ms']:>12.3f} {100*s['rank_max_ms']/summary['step_span_ms']:>8.1f}%")
        if args.dump_names:
            print("-- kernel names (count over selected steps, rank-summed):")
            for (cat, name), n in all_names.most_common(40):
                print(f"  {cat:14s} {n:6d}  {name}")
    if len(result) == 2:
        (la, a), (lb, b) = result.items()
        print(f"\n=== {la} -> {lb} (rank-max per step) ===")
        for c in cats:
            da = a["summary"][c]["rank_max_ms"]; db = b["summary"][c]["rank_max_ms"]
            print(f"{c:16s} {da:8.3f} -> {db:8.3f} ms  ({db-da:+.3f})")
        print(f"{'step_span':16s} {a['summary']['step_span_ms']:8.3f} -> {b['summary']['step_span_ms']:8.3f} ms")
    if args.output:
        args.output.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
