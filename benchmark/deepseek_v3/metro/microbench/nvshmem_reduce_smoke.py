"""Latency of a 256-float sum all-reduce across the EP world:
NCCL (torch.distributed) vs NVSHMEM host on-stream reduce (nvshmemx_float_sum_reduce_on_stream),
eager and inside a CUDA graph.  NVSHMEM is initialised by DeepEP's low-latency Buffer,
so this is exactly the runtime the SGLang decode path has."""
import ctypes, os, time
import torch, torch.distributed as dist
from deep_ep import Buffer

local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
dist.init_process_group("nccl"); group = dist.group.WORLD
rank, world = dist.get_rank(), dist.get_world_size()
num_tokens, hidden, num_experts = 8, 7168, 256
rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, world, num_experts)
buf = Buffer(group, 0, rdma_bytes, low_latency_mode=True, num_qps_per_rank=num_experts // world)
torch.cuda.synchronize(); dist.barrier()

lib = ctypes.CDLL("libnvshmem_host.so.3")
lib.nvshmem_malloc.restype = ctypes.c_void_p; lib.nvshmem_malloc.argtypes = [ctypes.c_size_t]
lib.nvshmem_my_pe.restype = ctypes.c_int; lib.nvshmem_n_pes.restype = ctypes.c_int
lib.nvshmemx_float_sum_reduce_on_stream.restype = ctypes.c_int
lib.nvshmemx_float_sum_reduce_on_stream.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
lib.nvshmemx_barrier_all_on_stream.argtypes = [ctypes.c_void_p]
pe, npes = lib.nvshmem_my_pe(), lib.nvshmem_n_pes()
assert npes == world and pe == rank, (pe, npes, rank, world)
N = 256
src_ptr = lib.nvshmem_malloc(N * 4); dst_ptr = lib.nvshmem_malloc(N * 4)
assert src_ptr and dst_ptr

class CAI:  # expose an nvshmem pointer as a torch tensor
    def __init__(self, ptr, n): self.__cuda_array_interface__ = {"shape": (n,), "typestr": "<f4", "data": (ptr, False), "version": 3}
src = torch.as_tensor(CAI(src_ptr, N), device="cuda"); dst = torch.as_tensor(CAI(dst_ptr, N), device="cuda")
src.fill_(float(rank)); dst.zero_(); torch.cuda.synchronize()
stream = torch.cuda.current_stream()
TEAM_WORLD = 0
rc = lib.nvshmemx_float_sum_reduce_on_stream(TEAM_WORLD, dst_ptr, src_ptr, N, ctypes.c_void_p(stream.cuda_stream))
torch.cuda.synchronize()
expect = sum(range(world))
ok = bool((dst == expect).all())
print(f"NVSHMEM_REDUCE rank={rank} rc={rc} correct={ok} dst[0]={dst[0].item()} expect={expect}", flush=True)
assert rc == 0 and ok

x = torch.full((N,), float(rank), device="cuda")
def nccl(): dist.all_reduce(x, group=group)
def nvs(): lib.nvshmemx_float_sum_reduce_on_stream(TEAM_WORLD, dst_ptr, src_ptr, N, ctypes.c_void_p(torch.cuda.current_stream().cuda_stream))

def timeit(fn, iters=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); dist.barrier(); torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) * 1000 / iters

t_nccl = timeit(nccl); t_nvs = timeit(nvs)
# graph: 58 back-to-back reduces (one decode step's worth)
def capture(fn):
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn()
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            for _ in range(58): fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    return g
res = {}
for name, fn in (("nccl", nccl), ("nvshmem", nvs)):
    try:
        g = capture(fn)
        def replay(): g.replay()
        res[name] = timeit(replay, iters=50) / 58
    except Exception as ex:  # noqa
        res[name] = f"capture failed: {type(ex).__name__}: {str(ex)[:80]}"
fmt = lambda v: v if isinstance(v, str) else f"{v:.1f} us"
dist.barrier()
if rank == 0 or rank == world - 1:
    print(f"LAT rank={rank} eager: nccl {t_nccl:.1f} us  nvshmem {t_nvs:.1f} us | in-graph per reduce: nccl {fmt(res["nccl"])}  nvshmem {fmt(res["nvshmem"])}", flush=True)
dist.barrier(); dist.destroy_process_group()
