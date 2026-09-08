"""Does DeepEP low-latency dispatch act as an arrival barrier?  From node-local
GPU traces (shared clock): per layer, the spread of dispatch-send start times
across the 8 ranks, and the recv-phase duration of the first- vs last-arriving
rank.  If recv(first) ~= recv(last) + skew, the early rank waited for the late one."""
import glob, gzip, json, sys, statistics
d = sys.argv[1]
ranks = {}
for f in sorted(glob.glob(d + "/*.trace.json.gz")):
    r = int(f.split("-TP-")[1].split("-")[0])
    t = json.load(gzip.open(f, "rt"))
    ks = sorted((e for e in t["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    disp = [e for e in ks if "internode_ll::dispatch" in e["name"]]
    # per layer: launches alternate send, recv (2 per layer); pair them by order
    pairs = [(disp[i], disp[i + 1]) for i in range(0, len(disp) - 1, 2)]
    comb = [e for e in ks if "internode_ll::combine" in e["name"]]
    ranks[r] = (pairs, comb)
R = sorted(ranks); n = min(len(v[0]) for v in ranks.values())
skew, rf, rl, send_d = [], [], [], []
for i in range(58, n - 58):
    st = {r: ranks[r][0][i][0]["ts"] for r in R}
    recv = {r: ranks[r][0][i][1]["dur"] for r in R}
    first = min(st, key=st.get); last = max(st, key=st.get)
    skew.append(st[last] - st[first]); rf.append(recv[first]); rl.append(recv[last])
    send_d.append(statistics.median(ranks[r][0][i][0]["dur"] for r in R))
med = statistics.median
print(f"{d}: {len(skew)} layers, dispatch launches/layer/rank = {len(ranks[R[0]][0]) / (n and 1)} pairs")
print(f"  dispatch arrival skew across 8 node-local ranks: median {med(skew):.1f} us, p90 {sorted(skew)[int(.9*len(skew))]:.1f}, max {max(skew):.1f}")
print(f"  dispatch send kernel: median {med(send_d):.1f} us")
print(f"  dispatch recv kernel: first-arriving rank median {med(rf):.1f} us, last-arriving rank median {med(rl):.1f} us  (diff {med(rf)-med(rl):.1f} vs skew {med(skew):.1f})")
