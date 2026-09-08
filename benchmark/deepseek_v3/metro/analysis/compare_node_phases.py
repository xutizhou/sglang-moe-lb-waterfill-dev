import json, sys, statistics
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
n = min(len(a["ds"]), len(b["ds"]))
print(f"node0 ranks {a['ranks']}  node1 ranks {b['ranks']}  layers {n}")
for k, label in (("ds", "dispatch send START"), ("dr_end", "dispatch recv END"), ("cs", "combine send START"), ("cr_end", "combine recv END")):
    diff = [a[k][i] - b[k][i] for i in range(58, n - 58)]
    med = statistics.median(diff); dev = [x - med for x in diff]
    print(f"  {label:20s}: node0-node1 offset median {med:10.1f} us; after removing the constant clock offset: "
          f"|dev| median {statistics.median(abs(x) for x in dev):6.1f}, p90 {sorted(abs(x) for x in dev)[int(.9*len(dev))]:6.1f}, max {max(abs(x) for x in dev):6.1f} us")
