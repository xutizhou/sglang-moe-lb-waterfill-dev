# METRO decode replica routing — reproduction package

Everything needed to reproduce the METRO (arXiv:2512.09277) decode-balancing
experiments on this branch: launch flags, the A/B harnesses, the trace-analysis
scripts, the micro-benchmarks, the data files and the measured results.
The full experiment log (Chinese) is in `docs/metro-reproduction-review.md`.

## What the branch adds (relative to upstream `11b0e5c5ad`)

| Flag / env | Meaning |
|---|---|
| `--ep-dispatch-algorithm static --lplb-decode-load-metric metro` | Stock EPLB static dispatch for prefill; **decode** picks one physical replica per active logical expert with METRO's greedy (min active experts per rank). Needs `--init-expert-location <placement>` and `--ep-num-redundant-experts R`. |
| `--lplb-decode-load-metric static_global \| dynamic_random` | Decode-only control policies (first replica everywhere / paper-style random replica). |
| `SGLANG_METRO_KERNEL=v2` (default) / `v1` | Warp-parallel assignment kernel (`metro_route_v2`, bit-exact with v1; R=128: 24.5 -> 10.6 us/layer in graph). |
| `SGLANG_METRO_COUNT_MODE=sync` (default) / `stale` | `stale` = assign from the previous step's reduced counts, all-reduce on a side stream. **Measured to not balance** (see results); kept for reference. |
| `SGLANG_METRO_STALE_INACTIVE_WEIGHT=0..8` | Second greedy pass for experts inactive in the stale set (stale mode only). |

Precision is unchanged: only the physical replica of the router's chosen
logical expert changes.

## Baseline definition

Upstream SGLang at the branch base `11b0e5c5ad`, `--ep-dispatch-algorithm static`,
no lplb flags. Two zero points were measured: `plain_trivial` (no placement
file, no dispatch algorithm) and `plain_static@0` (EPLB placement, R=0).
For `--moe-a2a-backend none` runs the baseline needs the one-hunk fix in
`harness/baseline_11b0e5c5ad_nccl_path_dispatch_info.patch` (upstream
`forward_normal` only builds the logical->physical map under `--enable-eplb`,
so redundant experts were silently mis-routed); the DeepEP path does not.

## Environment used

* 8x H20 per node (`10.6.131.25` node0, `10.6.131.6` node1), RoCE v2 400G, 8 rails, bond0 mgmt.
* Single node: `lmsysorg/sglang` image with DeepEP 1.2.1 / NVSHMEM 3.4.5 (`metro-repro` tag, id `06e0aa8359f5`).
* Two nodes with DeepEP low-latency: image `lmsysorg/sglang:v0.5.18-cu130` + **NVSHMEM 3.5.21 overlay + DeepEP rebuilt against it**
  (`dsv4_h20_cumulative_repro_20260829/site_nvshmem_3_5_21`, `site_deepep_pr483_nvshmem3521_rcfix_v1`), IBGDA gpu handler,
  `NVSHMEM_HCA_PE_MAPPING`, TC 106. The images' NVSHMEM 3.4.5 fails IBGDA init (`ibgda_create_dct: Unable to create ah`)
  even on one node. `microbench/deepep_ll_preflight_3521.sh` + `deepep_ll_smoke.py` verify the stack in 2 minutes.
* Model: DeepSeek-V3 FP8, 61 layers. Placement: `data/ep8_logical_count_sum.pt` (EPLB weights; the same file is used for EP16).
  Prompts: `data/gsm8k_input_ids.npy` (128-token prefixes), 32 concurrent sequences, 1024 output tokens, greedy probe per run.

## Minimal manual launch (EP8, one node)

```bash
# candidate: this branch
python3 -m sglang.launch_server --model-path /models/DeepSeek-V3 --trust-remote-code \
  --tp-size 8 --dp-size 8 --ep-size 8 --enable-dp-attention \
  --moe-a2a-backend deepep --deepep-mode auto --cuda-graph-max-bs 4 --mem-fraction-static 0.88 \
  --attention-backend fa3 --disable-radix-cache --skip-server-warmup \
  --init-expert-location data/ep8_logical_count_sum.pt --ep-num-redundant-experts 64 \
  --ep-dispatch-algorithm static --lplb-decode-load-metric metro
# baseline: upstream 11b0e5c5ad, same command without --lplb-decode-load-metric, --ep-num-redundant-experts 0
```

## Harness

`harness/metro_graph_ab_h20.sh` (one node) and `harness/metro_graph_ab_2node.sh` (two nodes, node0 drives node1 over ssh)
start a fresh server per cell, run a greedy probe, a warm-up and N benches, and store `bench_*.jsonl`, `probe.txt`,
`server.log`, optional torch profiles. Mode syntax: `[base:]<policy>[@R]`, e.g.
`base:plain_static@0 base:plain_static@64 static:metro@64`. `data/run_v3_matrix.sh` / `data/env25.sh` are the exact
EP8 invocations; `harness/run_ep16_deepep.example.sh` the EP16 one (`DEEPEP_STACK=nvshmem3521`); `harness/ep16_watch.sh`
polls for a free node1 and runs cells with a sanity gate; `harness/summarize_graph_ab.py` prints medians.

## Analysis

* `analysis/profile_categories.py <profile dirs> --gap-us 58 [--period-re deepseek_v3_topk_kernel]` — per-step kernel
  time by category (routed GEMM, DeepEP wait, METRO all-reduce/assign, attention, ...). Use `--period-re` for the NCCL path.
* `analysis/routed_gemm_imbalance_tax.py <profile dir>` — sum_l max_r / sum_l mean_r of routed GEMM time = imbalance tax.
* `analysis/deepep_layer_sync_spread.py`, `deepep_dispatch_barrier.py`, `allreduce_arrival_skew.py` — cross-rank spread
  of phase start/end inside a layer (shows DeepEP LL recv phases converge; waiting is spin time inside the recv kernels).
* `analysis/merge_rank_traces.py <profile dir> out.json` — 8 ranks on one Perfetto timeline (pid = rank).
* `analysis/node_phase_times.py` + `compare_node_phases.py` — cross-node comparison after removing the host-clock offset.
* `analysis/metro_stale_set_overlap.py routes.npy` — consecutive-step active-set overlap (why stale counts do not balance).
* `analysis/test_metro_v2_kernels.py` — v1/v2/stale kernel parity + timing (run inside the container, 1 GPU).
* `microbench/allreduce_latency.py` (+ `allreduce_bench.sh`, `run_ar_variants.sh`) — 256-float all-reduce latency by
  communicator shape, eager and in-graph. `microbench/nvshmem_reduce_smoke.py` — NVSHMEM on-stream reduce via ctypes.

## Measured results (61-layer DeepSeek-V3, BS32, OSL 1024, CUDA graph, 32 seqs x 1024 tokens wall time, median of 3)

EP8, one node, DeepEP LL (`results/v3_ep8_matrix_base_vs_metro`, `results/profile_v3_ep8`):

| R | upstream static | METRO (v1 kernel) | vs same R | vs R=0 |
|---:|---:|---:|---:|---:|
| 0 | 45.50 s | — | | |
| 16 | 47.28 | 44.92 | -5.0% | -1.3% |
| 32 | 46.97 | 43.60 | -7.2% | -4.2% |
| 64 | 49.10 | **43.17** | **-12.1%** | **-5.1%** |

EP16, two nodes, DeepEP LL on the NVSHMEM 3.5.21 stack (`results/ep16_deepep_v1`, `results/ep16_deepep_variants`):

| R | upstream static | METRO v1 (sync all-reduce) | METRO v2 kernel (sync) | vs static@0 37.27 / trivial@0 37.65 |
|---:|---:|---:|---:|---|
| 0 trivial | 37.65 | | | |
| 0 | 37.27 | | | |
| 32 | 37.33 | 38.49 | | |
| 64 | 38.98 | 38.38 | 38.36 | -1.6% / -0.8% ... +3% |
| 128 | 40.09 | 37.83 | **36.95** | **-0.8% / -1.9%** |

Profile at EP16 R=128: METRO cuts the routed-GEMM imbalance tax 32% -> 13% and DeepEP wait 9.5 -> 7.7 ms/step, but the
per-layer cross-node NCCL count all-reduce costs 38-44 us/layer (2.3 ms/step; single-node custom all-reduce is 7 us)
plus ~15 us/layer of induced dispatch wait, so the net is small. `stale` counts do not balance (tax 33%): consecutive
decode steps' active sets overlap only 64-65% vs a 58-62% random baseline (`analysis/metro_stale_set_overlap.py`).
EP16, NCCL all-gather path (`--moe-a2a-backend none`, `results/ep16_none_v3`): METRO@128 37.68 vs static@0 38.87 (-3.1%),
because the count all-reduce is not needed there (TopK is already global).

Next engineering step: device-side (IBGDA) one-hop count exchange fused into the assignment kernel, i.e. a
cross-node custom all-reduce, expected 44 -> ~5-10 us/layer.
