"""Per-layer phase timestamps aggregated over this node's ranks -> JSON, so two
nodes' series can be compared (node clocks differ by an unknown constant)."""
import glob, gzip, json, sys, statistics, bisect
d, out = sys.argv[1], sys.argv[2]
per_rank = {}
for f in sorted(glob.glob(d + "/*.trace.json.gz")):
    r = int(f.split("-TP-")[1].split("-")[0])
    t = json.load(gzip.open(f, "rt"))
    ks = sorted((e for e in t["traceEvents"] if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    disp = [e for e in ks if "internode_ll::dispatch" in e["name"]]
    comb = [e for e in ks if "internode_ll::combine" in e["name"]]
    n = min(len(disp) // 2, len(comb) // 2)
    per_rank[r] = [dict(ds=disp[2*i]["ts"], dr_end=disp[2*i+1]["ts"] + disp[2*i+1]["dur"],
                        cs=comb[2*i]["ts"], cr_end=comb[2*i+1]["ts"] + comb[2*i+1]["dur"]) for i in range(n)]
R = sorted(per_rank); n = min(len(v) for v in per_rank.values())
series = {k: [statistics.median(per_rank[r][i][k] for r in R) for i in range(n)] for k in ("ds", "dr_end", "cs", "cr_end")}
series["spread_dr_end"] = [max(per_rank[r][i]["dr_end"] for r in R) - min(per_rank[r][i]["dr_end"] for r in R) for i in range(n)]
series["ranks"] = R
json.dump(series, open(out, "w"))
print(f"{d}: ranks {R}, {n} layers -> {out}")
