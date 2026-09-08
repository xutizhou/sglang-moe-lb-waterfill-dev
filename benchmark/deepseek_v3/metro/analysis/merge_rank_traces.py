"""(1) Merge the 8 node-local per-rank torch-profiler traces into one Perfetto
file (pid = rank) so all ranks sit on one timeline.  (2) Print one MoE layer's
phase timeline per rank in absolute microseconds so the spread can be read
directly.  (3) Sanity-check cross-process clock alignment."""
import glob, gzip, json, sys, statistics
d, out = sys.argv[1], sys.argv[2]
per_rank = {}
for f in sorted(glob.glob(d + "/*.trace.json.gz")):
    r = int(f.split("-TP-")[1].split("-")[0])
    t = json.load(gzip.open(f, "rt"))
    per_rank[r] = [e for e in t["traceEvents"] if e.get("cat") == "kernel" and e.get("ph") == "X"]
R = sorted(per_rank)
# pick a window: 3 decode steps in the middle, delimited by dispatch kernels
def layers(evs):
    ks = sorted(evs, key=lambda e: e["ts"])
    disp = [e for e in ks if "internode_ll::dispatch" in e["name"]]
    comb = [e for e in ks if "internode_ll::combine" in e["name"]]
    gu = [e for e in ks if "deep_gemm" in e["name"] and ", 4096u, 7168u," in e["name"]]
    dn = [e for e in ks if "deep_gemm" in e["name"] and ", 7168u, 2048u," in e["name"]]
    # pair GEMMs with the dispatch window they fall into (the trace may hold
    # routed GEMMs from before the first captured dispatch)
    import bisect
    ds_ts = [e["ts"] for e in disp[0::2]]
    def in_window(lst):
        out = [None] * len(ds_ts)
        for e in lst:
            i = bisect.bisect_right(ds_ts, e["ts"]) - 1
            if i >= 0 and out[i] is None: out[i] = e
        return out
    gu_w, dn_w = in_window(gu), in_window(dn)
    n = min(len(disp)//2, len(comb)//2)
    layers = [dict(ds=disp[2*i], dr=disp[2*i+1], gu=gu_w[i], dn=dn_w[i], cs=comb[2*i], cr=comb[2*i+1]) for i in range(n)]
    layers = [l for l in layers if l["gu"] is not None and l["dn"] is not None]
    return layers
L = {r: layers(per_rank[r]) for r in R}
n = min(len(v) for v in L.values())
mid = n // 2
# --- (2) one layer, absolute us relative to earliest dispatch-send start among ranks
lay = mid
t0 = min(L[r][lay]["ds"]["ts"] for r in R)
print(f"one MoE layer (index {lay}), times in us relative to the earliest dispatch-send start; [start, end]")
print(f"{'rank':>4} {'disp send':>16} {'disp recv':>16} {'gate_up GEMM':>16} {'down GEMM':>16} {'comb send':>16} {'comb recv':>16}")
for r in R:
    row = []
    for k in ("ds", "dr", "gu", "dn", "cs", "cr"):
        e = L[r][lay][k]; row.append(f"[{e['ts']-t0:6.0f},{e['ts']+e['dur']-t0:6.0f}]")
    print(f"{r:>4} " + " ".join(f"{x:>16}" for x in row))
def spread(key, edge, i):
    v = [L[r][i][key]["ts"] + (L[r][i][key]["dur"] if edge == "end" else 0) for r in R]; return max(v) - min(v)
print("\ncross-rank spread of END times over all layers (us): median / p90 / max")
for label, key in (("dispatch recv end", "dr"), ("down GEMM end", "dn"), ("combine recv end", "cr")):
    s = sorted(spread(key, "end", i) for i in range(58, n - 58))
    print(f"  {label:18s} {statistics.median(s):6.1f} / {s[int(.9*len(s))]:6.1f} / {s[-1]:6.1f}")
# --- (3) alignment check: a graph-replay step boundary is a host-driven launch; compare instead the
# combine-recv END (a physical rendezvous) vs dispatch-send START (free-running) — if clocks were
# misaligned by X us per rank, BOTH spreads would be >= X.  Report the min over layers of each.
mins = {k: min(spread(k, e, i) for i in range(58, n - 58)) for k, e in (("dr", "end"), ("cr", "end"), ("ds", "start"))}
print(f"\nmin-over-layers spread: dispatch recv end {mins['dr']:.1f} us, combine recv end {mins['cr']:.1f} us, dispatch send start {mins['ds']:.1f} us"
      f"  -> per-process clock offsets on this node are bounded by ~{min(mins.values()):.0f} us")
# --- (1) merged perfetto trace for a 3-step window around `mid`
w0 = min(L[r][mid - 58]["ds"]["ts"] for r in R) - 50; w1 = max(L[r][mid + 2 * 58]["cr"]["ts"] + L[r][mid + 2*58]["cr"]["dur"] for r in R) + 50
ev = []
for r in R:
    ev.append({"ph": "M", "name": "process_name", "pid": r, "tid": 0, "args": {"name": f"rank {r}"}})
    ev.append({"ph": "M", "name": "process_sort_index", "pid": r, "tid": 0, "args": {"sort_index": r}})
    for e in per_rank[r]:
        if w0 <= e["ts"] <= w1:
            ev.append({"ph": "X", "cat": "kernel", "name": e["name"], "pid": r, "tid": e.get("tid", 0), "ts": e["ts"], "dur": e["dur"]})
json.dump({"traceEvents": ev, "displayTimeUnit": "us"}, open(out, "w"))
print(f"\nmerged trace ({len(ev)} events, 3 steps, 8 ranks on one timeline): {out}")
