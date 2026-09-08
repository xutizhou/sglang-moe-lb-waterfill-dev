"""Per-(step, layer) routed-GEMM imbalance across the ranks in a profile dir.

For every rank, take the routed-expert GEMM kernels in time order (2 per MoE
layer: gate_up then down), sum them per layer, then over all ranks compute
  sum_l max_r t[l, r]   (what a per-layer-synchronised step pays)
  sum_l mean_r t[l, r]  (perfectly balanced lower bound)
The ratio is the imbalance tax a decode replica policy can recover.

usage: imbalance_tax.py <profile_dir> [layers_per_step=58] [kernel_regex]
  default regex matches both the NCCL path (fused_moe_kernel) and the DeepEP
  path (DeepGEMM masked grouped GEMM with the expert shape in the name).
"""
import glob, gzip, json, re, statistics, sys
d = sys.argv[1]; period = int(sys.argv[2]) if len(sys.argv) > 2 else 58
rx = re.compile(sys.argv[3] if len(sys.argv) > 3 else r"fused_moe_kernel|deep_gemm.*(?:, 4096u, 7168u, \d+u,|, 7168u, 2048u, \d+u,)")
per_rank = {}
for f in sorted(glob.glob(d + "/*.trace.json.gz")):
    t = json.load(gzip.open(f, "rt"))
    ks = sorted((e for e in t["traceEvents"] if e.get("cat") == "kernel" and rx.search(e["name"])), key=lambda e: e["ts"])
    durs = [e["dur"] for e in ks]
    layers = [durs[i] + durs[i + 1] for i in range(0, len(durs) - 1, 2)]
    per_rank[f.split("-TP-")[1].split("-")[0]] = layers
n = min(len(v) for v in per_rank.values())
ranks = sorted(per_rank)
start, stop = period, (n // period - 1) * period
if stop <= start:
    sys.exit(f"{d}: only {n} routed-GEMM layer pairs matched; check the kernel regex")
mx = mean = 0.0
for l in range(start, stop):
    col = [per_rank[r][l] for r in ranks]
    mx += max(col); mean += statistics.mean(col)
steps = (stop - start) / period
print(f"{d}: ranks={len(ranks)} layers={stop-start} steps={steps:.0f}")
print(f"  routed GEMM per step: sum_l max_r = {mx/steps/1000:.2f} ms, sum_l mean_r = {mean/steps/1000:.2f} ms, tax = {(mx/mean-1)*100:.1f}%")
print(f"  mean per-layer routed time = {mean/(stop-start):.0f} us; mean per-layer max-mean gap = {(mx-mean)/(stop-start):.0f} us")
