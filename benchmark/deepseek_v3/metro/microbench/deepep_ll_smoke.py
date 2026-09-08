"""Two-node DeepEP low-latency smoke: Buffer init + one dispatch/combine round.

Exercises exactly what SGLang's decode path needs (IBGDA LL kernels), without
the model.  Prints DEEPEP_LL_OK per rank on success.
"""
import os, sys, time
import torch, torch.distributed as dist
from deep_ep import Buffer

local_rank = int(os.environ["LOCAL_RANK"]); torch.cuda.set_device(local_rank)
dist.init_process_group("nccl"); group = dist.group.WORLD
rank, world = dist.get_rank(), dist.get_world_size()
num_tokens, hidden, num_topk, num_experts = 8, 7168, 8, 256
assert num_experts % world == 0
t0 = time.time()
rdma_bytes = Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, world, num_experts)
buf = Buffer(group, 0, rdma_bytes, low_latency_mode=True, num_qps_per_rank=num_experts // world)
torch.cuda.synchronize(); dist.barrier()
print(f"BUFFER_OK rank={rank} init={time.time()-t0:.1f}s", flush=True)
x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
topk_idx = torch.stack([torch.randperm(num_experts, device="cuda")[:num_topk] for _ in range(num_tokens)]).to(torch.int64)
topk_w = torch.rand(num_tokens, num_topk, dtype=torch.float32, device="cuda")
for it in range(3):
    t1 = time.time()
    recv_x, recv_count, handle, _, _ = buf.low_latency_dispatch(x, topk_idx, num_tokens, num_experts, use_fp8=False, async_finish=False, return_recv_hook=False)
    torch.cuda.synchronize()
    combined, _, _ = buf.low_latency_combine(recv_x, topk_idx, topk_w, handle, async_finish=False, return_recv_hook=False)
    torch.cuda.synchronize()
    print(f"ROUND{it} rank={rank} dispatch+combine={1e3*(time.time()-t1):.2f}ms recv_count_sum={int(recv_count.sum())}", flush=True)
# correctness: combine of identity expert output = sum_k w_k * x  (each token's own x weighted)
ref = x.float() * topk_w.sum(-1, keepdim=True)
err = (combined.float() - ref).abs().max().item() / ref.abs().max().item()
print(f"DEEPEP_LL_OK rank={rank} rel_err={err:.3e}", flush=True)
dist.barrier(); dist.destroy_process_group()
