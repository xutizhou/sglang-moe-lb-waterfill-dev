"""Micro-benchmark: latency of a 256-float32 sum all-reduce (METRO's per-layer
count exchange) with NCCL, eager and inside a CUDA graph (58 per graph = one
decode step), for several communicator shapes.

  world   : all ranks (16 = 2 nodes x 8)          <- what SGLang's EP group does
  intra   : the 8 ranks of this node
  pair    : rank i <-> rank i+8 (one cross-node hop, 2 ranks)
  2level  : intra all-reduce, then pair all-reduce (hierarchical alternative)
  agather : all_gather of 16 x 256 floats on world (bitset-style alternative)
"""
import os, statistics, torch, torch.distributed as dist

local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
rank, world = dist.get_rank(), dist.get_world_size()
per_node = int(os.environ.get("PER_NODE", "8")); nodes = world // per_node
node = rank // per_node
g_world = dist.group.WORLD
g_intra = [dist.new_group(list(range(n * per_node, (n + 1) * per_node))) for n in range(nodes)][node]
g_pair = [dist.new_group([i + n * per_node for n in range(nodes)]) for i in range(per_node)][rank % per_node] if nodes > 1 else None
N = 256
x = torch.full((N,), float(rank), device="cuda")
xs = [torch.full((N,), float(rank), device="cuda") for _ in range(4)]
gath = torch.empty(world * N, device="cuda")

ops = {
    "world":   lambda: dist.all_reduce(xs[0], group=g_world),
    "intra":   lambda: dist.all_reduce(xs[1], group=g_intra),
    "agather": lambda: dist.all_gather_into_tensor(gath, xs[3], group=g_world),
}
if g_pair is not None:
    ops["pair"] = lambda: dist.all_reduce(xs[2], group=g_pair)
    def two_level():
        dist.all_reduce(xs[1], group=g_intra); dist.all_reduce(xs[1], group=g_pair)
    ops["2level"] = two_level

def sync_all():
    torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()

def time_eager(fn, iters=300):
    for _ in range(30): fn()
    sync_all()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters

def time_graph(fn, per_graph=58, replays=40):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(per_graph): fn()
    torch.cuda.current_stream().wait_stream(s); sync_all()
    for _ in range(3): g.replay()
    sync_all()
    st = torch.cuda.Event(enable_timing=True); en = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(replays): g.replay()
    en.record(); torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000 / (replays * per_graph)

def kernel_us_in_graph(fn, per_graph=58):
    """Sum of NCCL kernel durations per op inside a replayed graph (rank-local)."""
    from torch.profiler import profile, ProfilerActivity
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(per_graph): fn()
    torch.cuda.current_stream().wait_stream(s); sync_all()
    for _ in range(3): g.replay()
    sync_all()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10): g.replay()
        torch.cuda.synchronize()
    ks = [e for e in prof.events() if e.device_type.name == "CUDA" and "nccl" in e.name.lower()]
    names = sorted({e.name.split("(")[0] for e in ks})
    return sum(e.device_time for e in ks) / (10 * per_graph), names

res = {}
for name, fn in ops.items():
    te = time_eager(fn); tg = time_graph(fn)
    tk, knames = kernel_us_in_graph(fn)
    kv = [torch.zeros(1, device="cuda") for _ in range(world)]
    dist.all_gather(kv, torch.tensor([tk], device="cuda"))
    res.setdefault("_kernel", {})[name] = ([v.item() for v in kv], knames)
    # gather all ranks' numbers so we report the max (the one that gates the step) and median
    vals = [torch.zeros(2, device="cuda") for _ in range(world)]
    dist.all_gather(vals, torch.tensor([te, tg], device="cuda"))
    res[name] = ([v[0].item() for v in vals], [v[1].item() for v in vals])
if rank == 0:
    tag = os.environ.get("TAG", "default")
    print(f"=== {tag}: world={world} ({nodes} nodes x {per_node}), 256 x f32 sum all-reduce; us per op")
    print(f"{'op':8s} {'eager med':>10s} {'eager max':>10s} {'graph med':>10s} {'graph max':>10s} {'krnl med':>9s} {'krnl max':>9s}  kernels")
    for name in [k for k in res if k != "_kernel"]:
        te, tg = res[name]
        tk, kn = res["_kernel"][name]
        print(f"{name:8s} {statistics.median(te):10.1f} {max(te):10.1f} {statistics.median(tg):10.1f} {max(tg):10.1f} {statistics.median(tk):9.1f} {max(tk):9.1f}  {','.join(k.replace('ncclDevKernel_','') for k in kn)}", flush=True)
dist.barrier(); dist.destroy_process_group()
