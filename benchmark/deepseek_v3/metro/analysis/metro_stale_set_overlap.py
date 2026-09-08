"""How predictive is the previous decode step's active-expert set?  From
teacher-forced routes [seq, pos, layer, topk]: for BS=32 consecutive sequences
at position p vs p+1, per MoE layer: |A_t ∩ A_{t+1}| / |A_{t+1}| (recall of the
current set by the stale set), plus the fraction of the current step's active
*replicated-candidate* experts that were inactive last step."""
import sys, numpy as np
routes = np.load(sys.argv[1])  # [seq, pos, layer, topk]
S, P, L, K = routes.shape
moe_layers = range(3, L); pos = range(128, 190); B = 32
rec, jac, size = [], [], []
for l in moe_layers:
    for p in pos:
        for b0 in range(0, S - B + 1, B):
            a_prev = set(routes[b0:b0+B, p, l, :].reshape(-1).tolist()); a_prev.discard(-1)
            a_cur = set(routes[b0:b0+B, p+1, l, :].reshape(-1).tolist()); a_cur.discard(-1)
            if not a_cur: continue
            inter = len(a_prev & a_cur)
            rec.append(inter / len(a_cur)); jac.append(inter / len(a_prev | a_cur)); size.append(len(a_cur))
print(f"{sys.argv[1].split('/')[-2]}: BS={B} active set |A_t|={np.mean(size):.0f}/256; recall(A_t-1 -> A_t)={np.mean(rec):.3f}; Jaccard={np.mean(jac):.3f}; random-set baseline recall={np.mean(size)/256:.3f}")
