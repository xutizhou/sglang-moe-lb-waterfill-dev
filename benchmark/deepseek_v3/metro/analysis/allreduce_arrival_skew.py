"""Per-layer arrival skew at METRO's all-reduce across the 8 ranks of one node,
and per-rank attention time, from GPU-only traces (same node => same clock)."""
import glob, gzip, json, sys, statistics
d = sys.argv[1]
ranks = {}
for f in sorted(glob.glob(d + "/*.trace.json.gz")):
    r = int(f.split("-TP-")[1].split("-")[0])
    t = json.load(gzip.open(f, "rt"))
    ks = [e for e in t["traceEvents"] if e.get("cat") == "kernel"]
    ar = sorted((e for e in ks if "AllReduce_Sum_f32" in e["name"]), key=lambda e: e["ts"])
    attn = [e for e in ks if "FlashAttnFwd" in e["name"] and "Combine" not in e["name"]]
    ranks[r] = (ar, attn)
n = min(len(v[0]) for v in ranks.values())
R = sorted(ranks)
# align by index: all ranks run the same layer sequence
skews, durs_first, durs_last, ends = [], [], [], []
for i in range(58, n - 58):  # skip first/last partial step
    starts = {r: ranks[r][0][i]["ts"] for r in R}
    dur = {r: ranks[r][0][i]["dur"] for r in R}
    first = min(starts, key=starts.get); last = max(starts, key=starts.get)
    skews.append(starts[last] - starts[first])
    durs_first.append(dur[first]); durs_last.append(dur[last])
    end = {r: starts[r] + dur[r] for r in R}
    ends.append(max(end.values()) - min(end.values()))
print(f"{d}: {len(skews)} layers")
print(f"  arrival skew at all-reduce (last rank start - first rank start): median {statistics.median(skews):.1f} us, p90 {sorted(skews)[int(0.9*len(skews))]:.1f}, max {max(skews):.1f}")
print(f"  all-reduce kernel dur: first-arriving rank median {statistics.median(durs_first):.1f} us, last-arriving rank median {statistics.median(durs_last):.1f} us")
print(f"  completion spread (max end - min end): median {statistics.median(ends):.1f} us")
att = {r: sum(e['dur'] for e in ranks[r][1]) / (len(ranks[r][1]) / 58) / 1000 if ranks[r][1] else 0 for r in R}
print("  attention (FlashAttnFwd) ms/step per rank: " + " ".join(f"{att[r]:.2f}" for r in R))
