#!/usr/bin/env python3
"""Summarise metro_graph_ab_h20.sh runs: per-mode timing, paired deltas, noise.

Reads ``<root>/NN_<mode>/bench_*.log`` (sglang.benchmark.serving output) and
prints, per decode mode, the per-container medians and an overall median, then
compares every other mode against ``--baseline`` (default static_global).
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

DUR = re.compile(r"Benchmark duration \(s\):\s+([0-9.]+)")
TPUT = re.compile(r"Output token throughput \(tok/s\):\s+([0-9.]+)")
OK = re.compile(r"Successful requests:\s+(\d+)")
TOK = re.compile(r"Total generated tokens:\s+(\d+)")


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="replace")
    d, t, ok, tok = DUR.search(text), TPUT.search(text), OK.search(text), TOK.search(text)
    if not (d and t and ok and tok):
        return None
    return {
        "duration_s": float(d.group(1)),
        "tok_per_s": float(t.group(1)),
        "successful": int(ok.group(1)),
        "generated": int(tok.group(1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", default="static_global")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = []
    for run_dir in sorted(p for p in args.root.iterdir() if p.is_dir()):
        mode_file = run_dir / "decode_mode.txt"
        if not mode_file.exists():
            continue
        mode = mode_file.read_text().strip()
        samples = []
        for log in sorted(run_dir.glob("bench_*.log")):
            parsed = parse_log(log)
            if parsed:
                parsed["file"] = log.name
                samples.append(parsed)
        if not samples:
            continue
        runs.append(
            {
                "run": run_dir.name,
                "mode": mode,
                "samples": samples,
                "median_duration_s": statistics.median(s["duration_s"] for s in samples),
                "median_tok_per_s": statistics.median(s["tok_per_s"] for s in samples),
                "all_ok": all(s["successful"] > 0 for s in samples),
                "generated": {s["generated"] for s in samples},
            }
        )

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_mode[r["mode"]].append(r)

    print(f"{'run':22s} {'mode':16s} {'dur(s) per repeat':34s} {'median dur':>10s} {'median tok/s':>12s} tokens")
    for r in runs:
        durs = " ".join(f"{s['duration_s']:.3f}" for s in r["samples"])
        print(
            f"{r['run']:22s} {r['mode']:16s} {durs:34s} {r['median_duration_s']:>10.3f}"
            f" {r['median_tok_per_s']:>12.1f} {sorted(r['generated'])}"
        )

    summary = {}
    print("\nper mode (median over all repeats of all containers; process medians in brackets):")
    for mode, rs in by_mode.items():
        all_dur = [s["duration_s"] for r in rs for s in r["samples"]]
        proc_medians = [r["median_duration_s"] for r in rs]
        summary[mode] = {
            "containers": len(rs),
            "samples": len(all_dur),
            "median_duration_s": statistics.median(all_dur),
            "mean_duration_s": statistics.fmean(all_dur),
            "min_duration_s": min(all_dur),
            "max_duration_s": max(all_dur),
            "process_medians_s": proc_medians,
            "median_tok_per_s": statistics.median(s["tok_per_s"] for r in rs for s in r["samples"]),
        }
        print(
            f"  {mode:16s} n={len(all_dur):2d}  median {summary[mode]['median_duration_s']:.3f} s"
            f"  mean {summary[mode]['mean_duration_s']:.3f}  [{min(all_dur):.3f}, {max(all_dur):.3f}]"
            f"  proc medians {', '.join(f'{m:.3f}' for m in proc_medians)}"
            f"  -> {summary[mode]['median_tok_per_s']:.1f} tok/s"
        )

    comparisons = {}
    base = summary.get(args.baseline)
    if base:
        print(f"\nrelative to {args.baseline}:")
        for mode, s in summary.items():
            if mode == args.baseline:
                continue
            d = (s["median_duration_s"] / base["median_duration_s"] - 1.0) * 100.0
            t = (s["median_tok_per_s"] / base["median_tok_per_s"] - 1.0) * 100.0
            comparisons[mode] = {"latency_delta_pct": d, "throughput_delta_pct": t}
            print(f"  {mode:16s} latency {d:+.2f}%   throughput {t:+.2f}%")
        # Adjacent-pair view: each non-baseline container vs the baseline container
        # that immediately precedes it in run order (drift control).
        print("\nadjacent pairs (candidate container vs preceding baseline container):")
        prev_base = None
        pairs = defaultdict(list)
        for r in runs:
            if r["mode"] == args.baseline:
                prev_base = r
            elif prev_base is not None:
                d = (r["median_duration_s"] / prev_base["median_duration_s"] - 1.0) * 100.0
                pairs[r["mode"]].append(d)
                print(f"  {prev_base['run']} -> {r['run']:20s} latency {d:+.2f}%")
        for mode, ds in pairs.items():
            comparisons.setdefault(mode, {})["adjacent_pair_latency_deltas_pct"] = ds

    if args.output:
        args.output.write_text(
            json.dumps({"runs": runs, "summary": summary, "comparisons": comparisons}, indent=2, default=list) + "\n"
        )
        print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
