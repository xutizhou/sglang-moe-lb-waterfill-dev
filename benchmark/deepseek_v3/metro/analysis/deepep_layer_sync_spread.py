"""Where do the 8 node-local ranks drift apart and where do they re-converge
inside one MoE layer?  Cross-rank spread (max-min over ranks, us) of the
start/end of each phase, median over layers.  Shared clock: same node."""
import glob, gzip, json, sys, statistics
d = sys.argv[1]
ranks = {}
for f in sorted(glob.glob(d + "/*.trace.json.gz")):
    r = int(f.split("-TP-")[1].split("-")[0])
    t = json.load(gzip.open(f, "rt"))
    ks = sorted((e for e in t["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
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
    ranks[r] = layers
R = sorted(ranks); n = min(len(v) for v in ranks.values())
def spread(key, edge):
    out = []
    for i in range(58, n - 58):
        vals = [ranks[r][i][key]["ts"] + (ranks[r][i][key]["dur"] if edge == "end" else 0) for r in R]
        out.append(max(vals) - min(vals))
    return statistics.median(out)
def dur_by_arrival(key_wait, key_arrive):
    """recv-kernel duration of the rank that arrived first vs last at key_arrive."""
    first, last = [], []
    for i in range(58, n - 58):
        st = {r: ranks[r][i][key_arrive]["ts"] for r in R}
        f = min(st, key=st.get); l = max(st, key=st.get)
        first.append(ranks[f][i][key_wait]["dur"]); last.append(ranks[l][i][key_wait]["dur"])
    return statistics.median(first), statistics.median(last)
print(f"{d}: {n-116} layers, cross-rank spread (us, median over layers)")
for label, key, edge in [("dispatch send start", "ds", "start"), ("dispatch recv end", "dr", "end"),
                         ("gate_up GEMM start", "gu", "start"), ("down GEMM end", "dn", "end"),
                         ("combine send start", "cs", "start"), ("combine recv end", "cr", "end")]:
    print(f"  {label:20s} {spread(key, edge):6.1f}")
f, l = dur_by_arrival("dr", "ds"); print(f"  dispatch recv dur: first-arriving rank {f:.1f} us, last-arriving {l:.1f} us")
f, l = dur_by_arrival("cr", "cs"); print(f"  combine  recv dur: rank whose GEMM finished first {f:.1f} us, last {l:.1f} us")
