"""Parity + timing: metro_route_v2 / metro_route_stale vs dispatch_decode_metro (v1)."""
import time, torch
from sglang.kernels.ops.lplb.cuda_solver import (
    dispatch_decode_metro, metro_route_v2, metro_route_stale, count_logical_f32, MetroStaticTables)

torch.manual_seed(0)
dev = "cuda"
NL, NG = 256, 16

def make_placement(num_redundant):
    # each rank owns NL//NG base experts + num_redundant/NG replicas of random logical experts
    per_rank = NL // NG + num_redundant // NG
    phy2log = torch.empty(NG * per_rank, dtype=torch.int64)
    for r in range(NG):
        base = torch.arange(r * (NL // NG), (r + 1) * (NL // NG))
        extra = torch.randperm(NL)[: num_redundant // NG]
        phy2log[r * per_rank:(r + 1) * per_rank] = torch.cat([base, extra])
    phy2log = phy2log.to(dev)
    # physical_by_rank: first physical copy of logical l on rank r, -1 if none
    pbr = torch.full((NL, NG), -1, dtype=torch.int32, device=dev)
    for p in range(NG * per_rank):
        l = int(phy2log[p]); r = p // per_rank
        if pbr[l, r] < 0: pbr[l, r] = p
    mask = torch.zeros(NL, dtype=torch.int32, device=dev)
    for r in range(NG): mask |= (pbr[:, r] >= 0).to(torch.int32) << r
    rep = torch.nonzero((mask & (mask - 1)) != 0).flatten().to(torch.int32)
    return pbr, mask, rep

for R in (32, 64, 128):
    pbr, mask, rep = make_placement(R)
    tables = MetroStaticTables(pbr, mask, rep)
    assert tables.num_replicated == rep.numel()
    for trial in range(20):
        ntok = [2, 32, 64][trial % 3]
        topk = torch.stack([torch.randperm(NL, device=dev)[:8] for _ in range(ntok)]).to(torch.int32)
        if trial % 5 == 4: topk[-1] = -1  # padded row
        counts = torch.zeros(NL, dtype=torch.float32, device=dev)
        # global counts: this rank + fake peers
        peers = torch.randint(0, NL, (15 * ntok * 8 // 4,), device=dev)
        counts.scatter_add_(0, peers, torch.ones_like(peers, dtype=torch.float32))
        counts.scatter_add_(0, topk[topk >= 0].long().flatten(), torch.ones(int((topk >= 0).sum()), device=dev))
        ref = dispatch_decode_metro(topk, counts, pbr, mask, rep)
        got = metro_route_v2(topk, counts.clone(), tables)
        assert torch.equal(ref, got), f"v2 mismatch R={R} trial={trial}: {(ref != got).sum()} entries"
        buf = counts.clone()
        got2 = metro_route_stale(topk, buf, tables)
        assert torch.equal(ref, got2), f"stale routing mismatch R={R} trial={trial}"
        local = count_logical_f32(topk, NL, NG, rep.numel())
        assert torch.equal(buf, local), f"stale local-count mismatch R={R}"
    # timing
    topk = torch.stack([torch.randperm(NL, device=dev)[:8] for _ in range(2)]).to(torch.int32)
    counts = torch.zeros(NL, dtype=torch.float32, device=dev)
    peers = torch.randint(0, NL, (256,), device=dev); counts.scatter_add_(0, peers, torch.ones_like(peers, dtype=torch.float32))
    def bench(fn, n=200):
        for _ in range(10): fn()
        torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n): fn()
        e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) * 1000 / n
    t1 = bench(lambda: dispatch_decode_metro(topk, counts, pbr, mask, rep))
    t2 = bench(lambda: metro_route_v2(topk, counts, tables))
    t3 = bench(lambda: metro_route_stale(topk, counts.clone(), tables))
    tc = bench(lambda: count_logical_f32(topk, NL, NG, rep.numel()))
    print(f"R={R:3d} replicated={rep.numel():3d}: v1 {t1:6.1f} us | v2 {t2:6.1f} us | stale(+clone) {t3:6.1f} us | count {tc:5.1f} us  [parity OK, 20 trials]")
print("ALL_OK")

# inactive-weight pass: every replicated expert gets a legal replica; active ones unchanged vs weight 0
for R in (64, 128):
    pbr, mask, rep = make_placement(R); tables = MetroStaticTables(pbr, mask, rep)
    topk = torch.stack([torch.randperm(NL, device=dev)[:8] for _ in range(32)]).to(torch.int32)
    counts = torch.zeros(NL, dtype=torch.float32, device=dev)
    peers = torch.randint(0, NL, (200,), device=dev); counts.scatter_add_(0, peers, torch.ones_like(peers, dtype=torch.float32))
    a = metro_route_stale(topk, counts.clone(), tables, 0)
    b = metro_route_stale(topk, counts.clone(), tables, 3)
    active = counts[topk.long()] > 0
    assert torch.equal(a[active], b[active]), "inactive pass changed active experts"
    # legality: physical maps back to the same logical
    per_rank = pbr.shape[1]; phy2log = torch.full((int(pbr.max()) + 1,), -1, dtype=torch.int64, device=dev)
    for l in range(NL):
        for r in range(NG):
            if pbr[l, r] >= 0: phy2log[pbr[l, r]] = l
    assert torch.equal(phy2log[b.long()], topk.long()), "inactive pass routed to a wrong logical expert"
    changed = int((a != b).sum()); print(f"R={R}: inactive_weight=3 moved {changed}/{topk.numel()} entries of inactive experts; legality OK")
print("INACTIVE_OK")
