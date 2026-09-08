# METRO 复现结果复核：为什么 E2E 看不到收益，以及它实际应该有多少

日期：2026-09-04（分析）/ 2026-09-05（实测验证，§5）
复核对象：`docs/metro-reproduction.md`、`artifacts/metro_decode_full61_20260904/`、
`/scratch/gitsrc/cake-dev/artifacts/metro_decode_breakdown_20260904/REPORT.md`，
以及 Codex session `decode_lplb_active_experts_20260723`（2026-07-23 至 2026-09-04）。

## 0. 一句话结论

Codex 测到的**路由与 kernel 数据是可靠的**：METRO 把最慢 rank 的激活专家数降低约 10%，
grouped GEMM 关键路径随之降低约 7%，达到带宽模型预测的 92%。
但它据此得出的"E2E 上限约 1%、METRO 基本无效"**不成立**：E2E 是在一个 GPU 空转 2/3 时间的
launch-bound 系统上测的，61 层的"不均衡只有 1–2%"用了错误的统计口径，EP=4 是对 METRO
最不利的配置，而 baseline 选的是理想化的最强 static。

**实测验证（§5）**：把 Codex 的 harness 改成 `--deepep-mode auto` + CUDA graph，其它不变，同一个 8 层
DeepSeek-V3 / 4×H20 / BS32 上 METRO 相对 Codex 的最强 static baseline **latency -4.4% / throughput +4.6%**
（预测 4.7%），相对 SGLang 默认 static **-8.8% / +9.6%**（1.25×）、**-11.6% / +13.1%**（1.5×）；6 个 fresh
server 复现到 0.2%。同一份代码关掉 graph（Codex 测法），METRO 变成 **-4%**。在论文的 EP8 / 1.5× 配置上
（`.25` 整机），METRO 相对论文式 token-balanced baseline **-11.1% latency / +12.5% throughput**。
**完整 61 层 DeepSeek-V3、单机 8×H20 EP8、graph 开：METRO 相对最强 static -7.0% / +7.6%，相对 SGLang 默认 static
-11.4% / +12.9%**（§6.4）。以**未改动的上游代码、不加任何副本、不做任何均衡**的 decode 为基线，METRO（R=64）为
**-5.1% / +5.4%**；不开 METRO 时加副本会让 decode 变慢 8%（§6.5）。对 Codex 的 61 层
trace 按每层 rank-max 重算，关键路径 GEMM 改善是 **-6.1%**（Codex 口径 -1.2%）。真实路由 + 真实 EPLB
placement 的扫描显示 1.5× 副本率下 METRO vs 最强 static 为 EP4 9.9% → EP32 16.6%，vs 论文式 baseline 16–23%
——论文的 11–22% 不需要"水分"来解释。

## 1. Codex 已经证实、可以直接引用的事实

| 事实 | 数据 | 出处 |
|---|---|---|
| METRO 达成路由目标 | rank-max 激活专家 39.41 → 35.61（-9.6%），离穷举最优只差 0.14–0.25 | 8 层 DSv3、EP4、BS32、1.25×；在线 recorder 与离线重放 320/320 一致 |
| 真实数据集上普遍成立 | 8 个数据集（DAPO-Math/GSM8K/MMLU/LongBench/GPQA/ShareGPT/HumanEval/mix）BS32 降幅 8.6%–10.1%；BS4–8 时 13%–15% | `metro_workload_sweep_20260904` |
| GEMM 时间随激活专家数线性 | 每层 rank-max grouped GEMM 890 → 825 μs（-7.3%），为 HBM 模型预测的 92%；边际带宽 ≈ 2.46 TB/s（≈17.9 μs/expert） | 8 层 profile，正反序两组一致 |
| 路由开销 | fused 单 CTA 13.6 μs/层 | |
| 7 月同类工作 | 全局一致 active-expert 映射（零通信）vs SGLang 默认 static：MoE critical -2.7%，E2E +0.74%（KBC）；带 all-reduce 的精确 integral 分配 MoE critical -6.8% | 7 月 session |

## 2. E2E 为零的四个原因

### 2.1 测试系统是 CPU launch-bound，GPU 不在关键路径上

61 层 PP2/TP4/DP4/EP4 的 profile（`comparison_summary.json`，static，PP0 四个 rank）：

| | rank0 | rank1 | rank2 | rank3 |
|---|---:|---:|---:|---:|
| GPU kernel 驻留合计 | 38.9 ms | 76.7 ms | 72.8 ms | 69.0 ms |
| 其中 NCCL kernel（≈ 等对端到达） | 10.4 ms | 48.7 ms | 44.3 ms | 40.6 ms |
| **真正计算的 GPU 时间** | **28.5 ms** | 28.0 ms | 28.5 ms | 28.4 ms |
| decode step | 218 ms（profiler 开）/ 183 ms（关） | | | |

PP1 同样约 33 ms/rank。两个 stage 的计算合计约 62 ms，一步却要 183 ms：**GPU 至少 2/3 的时间在等
CPU 发 kernel、等 PP、等 rank 同步**。原因是 `--disable-cuda-graph`（DeepEP normal 模式强制）、
PP 强制 `disable_overlap_schedule`、外加 profiler。8 层实验同理：unprofiled step 22.4 ms，
GPU 实际忙约 7–8 ms。

在这种系统里 GPU 侧省 0.5 ms 还是 5 ms 都不会体现到 step time 上。**-0.06% / +0.05% 对 METRO
的有效性是零信息量的。** 同理，"NCCL 驻留增加 466 μs 吃掉了收益"也是 CPU 抖动：61 层里
METRO 的 NCCL 反而少了 4.9 ms，方向相反，说明就是噪声。

### 2.2 "GEMM 只占 MoE 的 21%" 是 profiler 伪影

Codex 的核心论证是：MoE wall 4246 μs/层，GEMM 只有 890 μs，所以专家少 10% 只能让 MoE 快 2%。
但 BS32 下 all-gather/reduce-scatter 各只搬 32×7168×2 B = 459 KB（几十 μs），gate+METRO 不到
30 μs，shared expert 很小——**剩下的 3000+ μs 是 launch gap 和同步等待**，是测试环境的开销，
不是 MoE 的开销。生产形态下 MoE 层 ≈ AG(~20 μs) + gate(~10) + METRO(14) + GEMM(~830) +
RS(~20)，GEMM 占 85% 以上；"专家少 10% → MoE 快 8–9%" 是成立的。

### 2.3 61 层"static 已经均衡到 1–2.4%"用错了统计口径

`analyze_traces.py::summarize()` 算的是 `max_rank( Σ_layers t )`：每个 rank 先把 29 层的 GEMM 求和，
再对 rank 取 max。但每个 MoE 层结尾都有 reduce-scatter 同步，真正的关键路径是
`Σ_layers( max_rank t )`。慢 rank 每层随机变化，29 层求和后各 rank 自然趋于相等——这正好把
METRO 要消除的东西平均掉了。

- 8 层实验用的是 per-layer max → -7.3%
- 61 层用的是 per-step sum → -1.19%
- Codex 反过来判定 8 层"小样本偏高"

`scripts/metro_trace_critical_path.py --self-test` 用合成 trace 验证了这个现象：4 rank、每层一个
慢 rank 轮换（+12.5%），先求和再取 max 会**隐藏 8.5%** 的不均衡；慢 rank 固定时两种口径才一致。
用该脚本对 61 层原始 trace 重算，预期 static→METRO 的关键路径 GEMM 改善为 5–8%（约 2.5–4 ms/step），
而非 0.575 ms。

### 2.4 EP=4 是对 METRO 最不利的配置，baseline 是理想化的最强 static

**EP 规模。** static 映射下每个 rank 的激活数近似 Binomial(256/EP, p)，rank-max 超出均值的幅度由
order statistics 决定。用 `scripts/metro_ep_sweep.py` 的合成路由（Zipf α=0.6，校准到实测的
"BS32 每层激活 140 个 logical expert"；EP4 上 static rank-max 39.43 vs 实测 39.41，
完美均衡 35.27 vs 实测 35.06）：

| EP | 完美均衡 | static_global rank-max | 超额 | METRO 1.25× | METRO 1.5× | 最优 1.5× |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 35.3 | 39.4 | +12% | 37.1（-6.0%） | 36.7（-7.6%） | 35.7 |
| 8 | 17.6 | 22.1 | +25% | 20.1（-9.0%） | 19.6（-10.5%） | 18.1 |
| 16 | 8.8 | 12.9 | +46% | 11.6（-9.7%） | 11.1（-13.7%） | 9.8 |
| 32 | 4.4 | 7.6 | +73% | 7.2（-5.9%） | 6.7（-20.1%） | 5.9 |

（BS32、`--steps 64`、每层独立 placement。合成模型在 EP4 上对 METRO 偏保守：实测 -9.6%，
合成 -6.0%；greedy 只拿到可消除空间的 54–76%，`metro_fixed_first` 变体和精确最优明显更好。）

两个结论：可消除的不均衡随 EP 快速增长；但 1.25× 副本率下能搬的 expert 太少，
EP≥16 时大部分超额来自单副本 expert，**论文用的 1.5× 副本率在这里是关键变量**——Codex 因
LPLB IPM 的 shared-memory bug 没有测过 1.5×。

**Baseline。** Codex 最终对比的 `static_allgather` 是"全 rank 一致、每个 logical expert 固定一个副本"，
已经天然满足 METRO 的第一条性质（去重）。论文的 baseline（token 在副本间均分）和 SGLang 真实默认的
`static`（按 source rank 选最近副本）都会让**同一个 logical expert 在一步里激活多个物理副本**。
同一套合成路由下，BS32 每层的重复激活数与 METRO 降幅：

| EP | 副本率 | sglang_static 重复激活/层 | dynamic 重复激活/层 | METRO vs static_global | METRO vs sglang_static | METRO vs dynamic |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.25 | 26.5 | 31.8 | 6.0% | **19.5%** | **21.9%** |
| 4 | 1.50 | 35.8 | 45.1 | 7.6% | 24.8% | 29.0% |
| 8 | 1.25 | 23.0 | 31.9 | 9.0% | 19.6% | 22.9% |
| 8 | 1.50 | 33.3 | 45.0 | 10.5% | 25.3% | 29.8% |
| 16 | 1.50 | 23.9 | 45.3 | 13.7% | 23.9% | 29.8% |

METRO 的收益 = **去重（大头）+ 均衡（Codex 测的那一半）**。对论文式 baseline，仅去重就是
20–30% 的 rank-max 降幅，这就是论文 11–22% 的来源；Codex 的 "+17.8% vs token-LPLB" 其实已经
看到了这一效应，但被 IPM 开销的混杂掩盖了。

## 3. 修正后的 E2E 推算

用 Codex 的可靠数字，换到生产形态（CUDA graph）的分母上。生产 step 的估算方法：把 profile 里
非 NCCL 的 GPU kernel 时间加总（这部分不受 launch 开销影响），再加上小 batch 下的真实通信量。

| | Codex 的系统 | 生产形态估算 |
|---|---:|---:|
| **8 层 proxy，EP4，BS32** | | |
| step | 22.4 ms | ≈ 7 ms（GEMM 4.45 + dense/attn/aux ≈ 2 + comm ≈ 0.3） |
| critical routed GEMM 占比 | 20% | **≈ 63%** |
| METRO 实测节省 | 0.326 ms | 0.326 ms |
| E2E | 理论 1.5%，实测 ≈ 0 | **≈ 4.7%** |
| **61 层，EP4，BS32** | | |
| step | 183 ms | ≈ 65–70 ms（GEMM 48 + 其他 GPU 15 + comm ≈ 4） |
| critical routed GEMM 占比 | 26% | **≈ 70%** |
| METRO 节省（3.8 experts/层 × 17.9 μs × 58 层） | 3.9 ms | 3.9 ms |
| E2E | 实测 ≈ 0 | **≈ 5.6–6%** |

（61 层这一行用的是实测的 3.8 experts/层；若用 Codex 的 0.575 ms/step 即 per-step-sum 口径，
E2E 也是 0.8–0.9% 而不是 0.31%。）

再叠加 EP8（可消除空间翻倍）、1.5× 副本、以及对 SGLang 真实 `static` / `dynamic` baseline 的去重收益，
落入 10–20% 区间是合理的。

**关于 "10% × 50% = 5%" 的推算**：唯一要修的是 GEMM 只跟踪了专家数的 92%（10% → 7.3%），以及
routed GEMM 在生产 step 中的占比是 60–75% 而非 50%。两项相乘仍然是 4–5%。推算方向和量级都是对的。

## 4. 对论文 claim 的评价

成立的部分：
- decode 应均衡激活专家数而非 token 数——机制已被实测证实（专家数 -9.6% → GEMM -7.3%）。
- 对 token-spread EPLB baseline 的 11–22% decode 收益，量级与合成扫描一致，不需要"水分"解释。

确实偏乐观的部分：
- DeepSeek-V3 结果来自闭源模拟器（16×B200），实机只有 Qwen3-30B/8×A100。
- 摘要选了区间上沿；正文最低配置约 1.9%。
- 1.5× 副本率是收益的重要前提；1.25× 下 EP≥16 时 greedy 收益明显缩水（见 §2.4）。
- 论文的 greedy（Algorithm 1）在大 EP 下离精确最优有明显差距（EP16/1.5× 合成：greedy 超额 25.8% vs
  最优 10.7%）。7 月做过的 exact integral assignment 正好补这一块。

Codex 说的"论文有一定水分"里，只有以上第 1–2 条是实证支持的；"BS32/EP4/4×H20 上 matched E2E
只有 +0.05%" 不能作为反证，因为那个 E2E 测的是 CPU launch 速度。

## 5. 实测验证（2026-09-05）

§2–§3 的每一条推断都已在 H20 上验证。原始数据在
`/lustre/raplab/client/xutingz/workspace/bench/metro_review_20260904/`，摘要在 `artifacts/metro_review_20260904/`。

### 5.1 61 层 trace 重算：不均衡被口径藏掉了 10 个点

对 Codex 的同一批 61 层 trace（`profile_static_manual_r2` vs `profile_metro_manual`）用
`scripts/metro_trace_critical_path.py` 重算（`full61_critical_path.json`）：

| grouped GEMM 统计（两个 PP stage 求和） | static | METRO | Δ |
|---|---:|---:|---:|
| Codex 口径 `max_r Σ_l` | 48.318 ms | 47.743 ms | -0.575 ms（-1.19%，复现 Codex 数字） |
| **关键路径 `Σ_l max_r`** | **53.065 ms** | **49.834 ms** | **-3.232 ms（-6.09%）** |
| 总工作量 `Σ_l mean_r` | 47.085 ms | 46.979 ms | -0.23%（不变，符合预期） |

- static 每层 rank-max 超出均值 **12.9% / 12.6%**（PP0 / PP1；p90 22%），不是 Codex 报告的 1–2.4%；与 §2.4 的
  order-statistics 预测（+11.7%）和实测 EP4 路由一致。METRO 把它压到 6.6% / 5.7%。
- 每层最慢 rank 的直方图四个 rank 近乎均匀（PP0 static：81/78/96/96），正是"先求和再取 max"抹掉不均衡的机制。
- GPU 利用率：compute-only **13–16%**/rank，含 NCCL 17–41%；launch-bound 确认。

### 5.2 真实路由 × 真实 EPLB placement 的 EP / 副本率 / baseline 扫描

`scripts/metro_ep_sweep.py --routing npy` 直接读取 Codex 采集的 8 个数据集 teacher-forced 路由
（128 序列 × 64 decode 位置 × 5 MoE 层）。EP4/1.25× 使用服务器保存的 placement（`placement_r64.pt`），其它配置用
仓库内 bit-for-bit 的 SGLang EPLB（hierarchical，`n_group=8`）从同一份历史 count 重生成——重生成的 EP4 placement
与服务器保存的 **1280/1280 完全一致**，因此 EP8/16/32 的 placement 就是 SGLang 会部署的。管线对齐检查：GSM8K
BS32 static_global 41.57 / optimal 37.51 / 激活 148.3，与 Codex 报告逐位一致。

8 个数据集平均，每层 rank-max 激活物理专家数（`real_sweep/aggregate.json`）：

| EP | 副本率 | BS | 激活 logical | 完美均衡 | static_global | SGLang static | dynamic | METRO 论文顺序(SDK) | METRO fixed-first(SGLang kernel) | 最优 | M(论文顺序) vs static_global | vs SGLang static | vs dynamic |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.25 | 32 | 155 | 38.8 | 43.41 | 45.53 | 46.33 | 40.82 | 39.44 | 39.23 | 6.0% | 10.3% | 11.9% |
| 4 | 1.50 | 32 | 155 | 38.8 | 44.42 | 47.81 | 49.24 | 40.04 | 39.44 | 39.17 | 9.9% | 16.3% | 18.7% |
| 8 | 1.25 | 32 | 155 | 19.4 | 23.81 | 24.53 | 25.30 | 22.14 | 20.61 | 20.36 | 7.0% | 9.7% | 12.5% |
| 8 | 1.50 | 32 | 155 | 19.4 | 24.17 | 25.52 | 27.02 | 21.25 | 20.25 | 19.84 | 12.1% | 16.7% | 21.3% |
| 16 | 1.25 | 32 | 155 | 9.7 | 13.66 | 13.85 | 14.33 | 12.57 | 11.69 | 11.56 | 7.9% | 9.2% | 12.3% |
| 16 | 1.50 | 32 | 155 | 9.7 | 13.91 | 14.32 | 15.37 | 11.84 | 10.91 | 10.31 | 14.9% | 17.3% | 22.9% |
| 32 | 1.25 | 32 | 155 | 4.8 | 8.00 | 8.05 | 8.32 | 7.40 | 6.90 | 6.86 | 7.5% | 8.1% | 11.1% |
| 32 | 1.50 | 32 | 155 | 4.8 | 8.46 | 8.55 | 9.04 | 7.06 | 6.27 | 5.95 | 16.6% | 17.4% | 21.9% |
| 4 | 1.50 | 128 | 244 | 61.1 | 67.01 | 82.04 | 87.01 | 62.19 | 61.78 | 61.47 | 7.2% | 24.2% | 28.5% |
| 8 | 1.50 | 128 | 244 | 30.5 | 35.19 | 40.63 | 44.92 | 32.60 | 31.28 | 31.00 | 7.4% | 19.8% | 27.4% |

三个结论：

1. **1.5× 副本率是关键变量。** 1.25× 下 METRO vs 最强 static 停在 6–8%；1.5× 下 EP4→EP32 为 9.9% → 12.1% →
   14.9% → 16.6%，对论文式 baseline（SGLang static / dynamic）为 16–23%。论文的 11–22% 就在这个区间。
2. **去重收益是大头。** SGLang 默认 static 在 1.5×/BS32 每层多激活 10–16 个副本，dynamic 多 22 个；BS128 时分别
   68 / 91 个。这部分 Codex 的 `static_allgather` baseline 已经预先拿掉了。
3. **贪心的处理顺序值 3–9 个点，而 SGLang 分支与本仓库 SDK 的实现顺序不同。** SGLang 分支部署的
   `dispatch_decode_metro`（`dispatch_decode_integral.cuh`）先把单副本 expert 计入 `fixed_active_load`，再对
   `replicated_logical` 贪心——即 fixed-first；Codex 离线脚本的 `metro_peak` 与之一致，所以它报的 "9.1%、离最优
   0.2" 描述的确实是部署 kernel，§5.3 的 E2E 也是这个顺序。本仓库 SDK 的 `kernels/metro/csrc/metro.cuh` 和
   `metro_greedy_assignment_reference` 则按论文 Algorithm 1 逐 logical id 贪心（不区分固定/可移动），在同一数据上
   只有 6.0%；大 EP 下差距拉到 7–9 个点（EP32/1.5×：16.6% vs 25.9%），离精确最优更远（29.7%）。表中 "fixed-first"
   一列才是 SGLang 实际部署的行为：1.5× 下 vs 最强 static 为 EP4 11.2% → EP8 16.2% → EP16 21.6% → EP32 25.9%。

### 5.3 E2E：开 CUDA graph 后收益完整显现

`scripts/h20/metro_graph_ab_h20.sh`：与 Codex 相同的 8 层 DeepSeek-V3、4×H20、EP4/DP4、BS32、ISL 8、64 冗余专家、
token-count LPLB prefill、同一 seed 请求流；区别是 `--deepep-mode auto`（decode 走 low_latency）并**开启 CUDA graph**
（bs=[1,2,4,8] 捕获成功，含 `metro` 的 LPLB all-reduce）。在 SGLang 分支上新增了 `static_global` decode 模式
（`log2phy[:,0]` 全 rank 一致，= Codex 的 `static_allgather` baseline，但走常规 DeepEP dispatch；commit `ce0484a452`）。
每种模式 2 个 fresh server × 3 次 BS32×1024 测量：

| decode 模式 | 中位时长（32×1024 tok） | step | tok/s | vs static_global |
|---|---:|---:|---:|---:|
| `static_global`（Codex 的 baseline，去重后的最强 static） | 7.135 s | 6.97 ms | 4592 | — |
| **`metro`**（论文 Algorithm 1 + all-reduce + DeepEP LL dispatch） | **6.820 s** | **6.66 ms** | **4805** | **-4.41% latency / +4.64% throughput** |
| `static`（SGLang 默认，按 source rank 选最近副本） | 7.475 s | 7.30 ms | 4383 | +4.77% / -4.56% |

- METRO vs SGLang 默认 static：**-8.8% latency / +9.6% throughput**。
- 两个 fresh server 的中位数：static_global 7.14 / 7.13，metro 6.82 / 6.82，static 7.47 / 7.48；相邻配对
  -4.48% / -4.35%。噪声 ≈ 0.2%，信号是噪声的 20 倍以上。
- step 从 22.4 ms（Codex 无 graph）降到 6.97 ms；METRO 省下 0.31 ms/step，与 Codex profiler 测到的 GEMM 节省
  0.326 ms/step 一致——**同一份 GEMM 节省，在 launch-bound 系统里是 +0.05%，在 GPU-bound 系统里是 +4.6%**。
  这与 §3 的预测（≈4.7%）一致。

### 5.4 对照组：同一份代码关掉 graph，符号反转

同一个 harness、同一个 commit，只改 `--disable-cuda-graph` + `--deepep-mode normal`（即 Codex 的测法），
2 个 fresh server × 3 次：

| decode 模式 | graph 开（§5.3） | graph 关 / DeepEP normal |
|---|---:|---:|
| `static_global` | 7.135 s（6.97 ms/step） | 23.865 s（23.3 ms/step） |
| `metro` | 6.820 s（**-4.4%**） | 24.810 s（**+4.0%**，相邻配对 +5.7% / +3.6%） |

METRO 的 GPU 侧节省在两种设置下都存在（同一份 kernel），但 launch-bound 时它每层多出的三次 launch（count、
all-reduce、METRO）落在 CPU 关键路径上，每 step 多约 1 ms，而 GPU 省下的 0.3 ms 与关键路径无关。
**同一个策略，测量 regime 不同，结论从 +4.4% 变成 -4%**——Codex 看到的"无效/微负"就是这个。

### 5.5 1.5× 副本率（128 个冗余专家）

prefill 的 fused IPM 在 NC=126/NV=256 下不进 shared memory（Codex 卡在这里）；给 SGLang 分支加了 opt-in 的
`SGLANG_LPLB_IPM_TORCH_FALLBACK=1`（commit `4dfe7f9ee7`，不合适的 shape 退回 torch 参考 IPM，只影响几次 ISL=8 的
prefill，各 arm 一致）。其余与 §5.3 相同：

| decode 模式 | 中位时长（2 server × 3） | tok/s | vs static_global |
|---|---:|---:|---:|
| `static_global` | 7.030 s | 4662 | — |
| **`metro`** | **6.715 s** | **4879** | **-4.5% latency / +4.7% throughput** |
| `static`（SGLang 默认） | 7.600 s | 4314 | +8.1% / -7.5% |

- METRO vs SGLang 默认 static：**-11.6% latency / +13.1% throughput**——进入论文 11–22% 区间，在 4×H20 / EP4 /
  8 层 proxy 上。副本越多，SGLang 默认 static 的重复激活越多（1.25× +4.8% → 1.5× +8.1%），去重收益相应增长。
- METRO vs 最强 static 从 1.25× 的 -4.4% 到 1.5× 的 -4.5%，变化小于 §5.2 在 EP4 上的预测（专家数 6.0% → 9.9%）：
  benchmark 用随机 token id 路由，而 placement 是用真实文本的历史 count 算的，多出的副本与随机路由的热点对不上；
  真实 workload 下 1.5× 对 balance 的额外收益需要用真实 prompt 的 E2E 确认。

### 5.6 由实测引出的 SDK 改动

§5.2 显示论文顺序与 fixed-first 的差距在 EP4 是 6.0% vs 9.1%，EP16/1.5× 是 14.9% vs 21.6%。SGLang 分支的 kernel
已经是 fixed-first；本仓库 SDK 的 reference/kernel 是论文顺序。为让 SDK 能表达 SGLang 实际部署的行为，加了
`metro_fixed_first` 变体（reference + CUDA kernel + pipeline 表达式，默认仍是论文顺序；commit `e549b97`），
CUDA kernel 与 reference 在随机 placement 上逐位一致。`docs/metro-reproduction.md` 里 "The CUDA implementation
… serializes contenders in logical expert order" 只描述 SDK kernel，不描述 SGLang 分支——两边应统一为 fixed-first。
（`e549b97` 的 commit message 把 SGLang kernel 也说成论文顺序，是错的，以本节为准。）

### 5.7 真实 prompt 的 E2E（h20-25，GPU 0–3）

`.25` 没有 /lustre，把模型（64 GB）、分支快照、JIT cache、镜像搬到 `/raid` 后用同一 harness 跑
`BENCH_MODE=real`：GSM8K 真实文本（`metro_workload_sweep_20260904` 采集的 `input_ids.npy`）前 32 条序列、
ISL 128、OSL 1024、BS32、greedy + ignore_eos，32 个并发单请求（与随机 id 实验相同的请求模式）。其余与 §5.3 一致。

| 副本率 | decode 模式 | 中位（2 server × 3） | tok/s | vs static_global | METRO vs SGLang 默认 |
|---:|---|---:|---:|---:|---:|
| 1.5× | `static_global` | 7.312 s | 4481 | — | |
| 1.5× | **`metro`** | **6.949 s** | **4715** | **-5.0% / +5.2%** | **-10.4% / +11.6%** |
| 1.5× | `static`（SGLang 默认） | 7.758 s | 4224 | +6.1% / -5.7% | |
| 1.25× | `static_global` | 7.218 s | 4540 | — | |
| 1.25× | **`metro`** | **6.905 s** | **4745** | **-4.3% / +4.5%** | **-8.5% / +9.3%** |
| 1.25× | `static`（SGLang 默认） | 7.546 s | 4343 | +4.5% / -4.3% | |

（每行 2 个 fresh server × 3 次；1.25× 的第二轮因派生端口撞上 ephemeral 区间重跑了一次，合并见
`real_gsm8k_r64_b32_o1024/summary.json`。）真实 prompt 与随机 id 的结论一致：METRO vs 最强 static 为
4.3%（1.25×）/ 5.0%（1.5×），vs SGLang 默认 static 为 8.5% / 10.4%。在 EP4 上 1.5× 只比 1.25× 多 0.7 个点——
副本率的价值主要在更大 EP（§5.2）和对 SGLang 默认 static 的去重（+4.5% → +6.1%）。

### 5.8 第二个数据集（ShareGPT）与论文式 `dynamic` baseline

- `.25`：ShareGPT 真实 prompt，其余与 §5.7 相同。
- `.24`（停掉 `maas-dsv4-v06-freetoken` 后的 GPU 0–3）：GSM8K，新增 `dynamic_random` decode 模式
  （SGLang `--ep-dispatch-algorithm dynamic` 只作用于 decode：每 token 随机选副本，即论文的 token-balanced
  baseline；SGLang 分支 commit `69b912757e`）。同机配对 `static_global / metro / dynamic_random`。

| 节点 / 数据 | 副本率 | static_global | **metro** | 第三 arm | METRO vs static_global | METRO vs 第三 arm |
|---|---:|---:|---:|---:|---:|---:|
| `.25` ShareGPT | 1.25× | 7.288 s | **6.947 s** | SGLang static 7.556 s | **-4.7% / +4.9%** | **-8.1% / +8.8%** |
| `.25` ShareGPT | 1.5× | 7.325 s | **6.987 s** | SGLang static 7.831 s | **-4.6% / +4.8%** | **-10.8% / +12.1%** |
| `.24` GSM8K | 1.25× | 7.278 s | **6.942 s** | dynamic 7.598 s | **-4.6% / +4.9%** | **-8.6% / +9.4%** |
| `.24` GSM8K | 1.5× | 7.340 s | **7.042 s** | dynamic 7.892 s | **-4.1% / +4.2%** | **-10.8% / +12.1%** |

（每个 arm 2 个 fresh server × 3 次。）三个数据源（随机 id、GSM8K、ShareGPT）、两台机器、两种副本率上，
METRO vs 最强 static 稳定在 **4.1–5.0%**；vs SGLang 默认 static 8.1–10.8%；vs 论文式 dynamic baseline
8.6–10.8%。dynamic 比 SGLang static 再差 0.5–1 个点（随机选副本比按 source rank 选更容易激活重复副本），
与 §5.2 离线扫描的排序一致。

### 5.9 全部 E2E 汇总（EP4，8 层 DeepSeek-V3 proxy，4×H20，BS32，OSL 1024，CUDA graph）

| 场景 | 节点 | METRO vs 最强 static（static_global） | METRO vs SGLang 默认 static | METRO vs dynamic |
|---|---|---:|---:|---:|
| 随机 id，1.25× | `.5` | -4.4% | -8.8% | |
| 随机 id，1.5× | `.5` | -4.5% | -11.6% | |
| GSM8K，1.25× | `.25` | -4.3% | -8.5% | |
| GSM8K，1.5× | `.25` | -5.0% | -10.4% | |
| GSM8K，1.25× | `.24` | -4.6% | | -8.6% |
| GSM8K，1.5× | `.24` | -4.1% | | -10.8% |
| ShareGPT，1.25× | `.25` | -4.7% | -8.1% | |
| ShareGPT，1.5× | `.25` | -4.6% | -10.8% | |
| **EP8** GSM8K，1.25× | `.25` | -4.5% | -7.4% | -7.0% |
| **EP8** GSM8K，1.5× | `.25` | -5.4% | -10.6% | **-11.1%** |
| **61 层 DeepSeek-V3，EP8，1.25×，优化 kernel** | `.25` | **-7.0%** | **-11.4%** | |
| **61 层，EP8，vs 未改动上游 static，同 R=64 / R=0（§6.5）** | `.25` | | **-12.1%** | **-5.1%（vs 不加副本）** |
| **同一代码关 graph（Codex 测法），1.25×** | `.5` | **+4.0%（变慢）** | | |

理论预测（§3）EP4 上 vs 最强 static ≈ 4.7%：8 组 EP4 graph-on 实验落在 4.1–5.0%，EP8 为 4.5–5.4%。对论文 /
SGLang 默认 baseline 在 1.5× 下 10.4–11.6%（EP4）、10.6–11.1%（EP8），进入论文 11–22% 区间的下沿。

### 5.10 EP8（h20-25 整机 8 卡，论文的 EP 规模）

停掉 `.25` 的 `maas-dsv4-sglang-decode` 后 8 卡全空。TP8/DP8/EP8、单机 NVLink、DeepEP auto、CUDA graph、
GSM8K 真实 prompt（ISL 128、OSL 1024）、BS32（每 rank 4 token）。placement 由同一份历史 count 经 SGLang EPLB
（hierarchical，n_group=8）生成，1.25× → 40 物理 expert/rank，1.5× → 48。四 arm × 2 fresh server × 3 次：

| 副本率 | static_global | **metro** | SGLang static | dynamic（论文 baseline） | METRO vs static_global | vs SGLang static | vs dynamic |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.25× | 5.052 s（4.93 ms/step） | **4.825 s** | 5.209 s | 5.186 s | **-4.5% / +4.7%** | -7.4% / +8.0% | -7.0% / +7.5% |
| 1.5× | 5.054 s | **4.780 s** | 5.344 s | 5.378 s | **-5.4% / +5.7%** | -10.6% / +11.8% | **-11.1% / +12.5%** |

- 论文配置（EP8、1.5×、token-balanced baseline）在真实 H20 上复现出 **-11.1% latency / +12.5% throughput**，
  正好是论文 11–22% 区间的下沿（论文的 Qwen3-30B/8×A100 实机与这个 8 层 DSv3 proxy 的 MoE 占比不同）。
- vs 最强 static 只从 EP4 的 4.5% 提高到 5.4%，小于 §5.2 中激活专家数降幅的增长（1.5× 下 EP4 11.2% → EP8
  16.2%，fixed-first）。原因是 EP8 上每 rank 只有 4 token、~20 个激活 expert，routed GEMM 在 step 里的占比
  从 EP4 的 ~63% 降到 ~45%，非 MoE 部分（attention、dense、通信、launch）不随 EP 缩小；16% × 45% ≈ 7% 的
  上限兑现了 5.4%，与 EP4 的 11% × 63% ≈ 7% 兑现 4.5% 同一比例。
- 1.5× 比 1.25× 多 1 个点（EP4 上是 0.7），去重收益随副本率增长更明显（SGLang static 落后 static_global 从
  3.1% 到 5.7%，dynamic 从 2.7% 到 6.4%）。

BS64（每 rank 8 token）两次都在 warmup 阶段撞上 DP admission 竞态（64 并发请求进入 8 个 DP rank 时 DeepEP
normal-mode dispatch 等不到对端，18 分钟后 `CPU recv timeout`；`--enable-prefill-delayer` 无效）。这是
`reproduce/metro-decode-fused` 分支 DP-attention 调度的问题（Codex 在 PP+DP 上记录过同类死锁），32 并发从未
触发。EP8 上每 rank 8 token 的点没有拿到；按 §5.2 的路由扫描，BS64 的激活专家降幅与 BS32 接近（1.5×：
EP8 12.7% vs 16.2% fixed-first），routed GEMM 占比会更高，预期 E2E 收益略高于 BS32 的 5.4%。

`.6` GPU0 有 root 的长期任务且 1–4 被占，`.24` 4–7 是 `zfl-flexkv-tp4`（他人的 DSv4-Flash serve）；EP8 只在 `.25` 可做。

## 6. 把不均衡换成吞吐：杠杆在哪（2026-09-05 晚）

目标改为工程目标——降低 decode 激活专家不均衡以提升 decode 吞吐。先用零 GPU 的分析定位杠杆，再用 profile 量化实现开销。

### 6.1 剩余不均衡的构成（真实路由 × EPLB placement，拟合/评估位置分开）

`headroom/aggregate.txt`（gsm8k / sharegpt / dapo_math 平均，BS32）：

| EP | 副本率 | 完美均衡 | static_global | METRO(fixed-first) | 最优 | METRO 距完美 | 最优距完美 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.25 | 37.9 | 42.6 | 38.6 | 38.4 | **+1.7%** | +1.2% |
| 4 | 2.0 | 37.9 | 44.7 | 38.6 | 38.3 | +1.7% | +1.0% |
| 8 | 1.25 | 19.0 | 23.4 | 20.2 | 19.9 | **+6.5%** | +5.2% |
| 8 | 1.5 | 19.0 | 23.8 | 19.8 | 19.4 | +4.5% | +2.4% |
| 16 | 1.25 | 9.5 | 13.4 | 11.4 | 11.3 | **+20.5%** | +18.9% |
| 16 | 1.25，decode-aware placement | 9.5 | 13.3 | 10.8 | 10.4 | +14.0% | +9.4% |
| 16 | 1.5 | 9.5 | 13.7 | 10.7 | 10.1 | +12.7% | +6.4% |

- **EP≤8：路由算法已到头。** METRO 离完美均衡 1.7%（EP4）/ 4.5–6.5%（EP8）；精确最优只再多 0.5–2 个点；副本率
  1.25×→2.0× 不改变 METRO 的绝对值（EP4 始终 38.6），只让 static 更差。EP≤8 上要涨吞吐，杠杆是**实现开销**。
- **EP16+：算法与 placement 才重要。** 贪心距完美 20%；用 decode 激活频率（而非 prefill token 数）做 EPLB 权重拿回
  6 个点，精确分配再拿 4–5 个点，1.5× 副本再拿 8 个点。这是多机 EP16/32 部署上的产品级机会。
- 每层绝对节省（2.46 TB/s）：EP4 ≈ 72 μs，EP8 ≈ 60 μs，EP16 ≈ 40 μs。

### 6.2 收益去向与 METRO 路径开销（graph 开，EP4，Torch profiler，GPU-only，`profile_ep4_r64/`）

`scripts/h20/profile_categories.py` 按 5 个 gate-up grouped GEMM 为周期切 step（graph 下 GPU 被连续喂满，没有空闲间隙可切），
per-step rank-max：

| 类别 | static_global | metro（原 kernel） | metro（优化后） |
|---|---:|---:|---:|
| routed grouped GEMM | 4.036 ms | 3.998 | 4.002 |
| **DeepEP LL dispatch+combine（含等对端）** | 0.681 | **0.275** | 0.295 |
| METRO assign kernel | — | 0.091（18 μs/层） | **0.052（10.4 μs/层）** |
| METRO all-reduce（256 float） | — | 0.038（7.6 μs/层） | 0.048 |
| METRO count | — | 0.016（3 launch） | **0.003（1 launch）** |
| step | 6.852 | 6.574（-4.1%） | 6.566 |

两个关键读数：

1. **不均衡税表现为 DeepEP kernel 里的等待，而不是每 rank GEMM 总和。** 每 rank 的 grouped GEMM 总和几乎不变
   （慢 rank 轮换，求和抹平），METRO 拿回的 0.41 ms/step 全在 dispatch/combine 的等对端时间里。这与 §5.1 的
   `Σ_l max_r` vs `max_r Σ_l` 是同一件事的两个侧面。
2. **METRO 自身开销 ~29 μs/层（2.1% step），其中 assign kernel 18 μs 偏高**：单 CTA 的 thread 0 对 ~50 个副本 expert
   做链式依赖的 global load。改为全线程把副本元数据预载到 shared memory 后 10.4 μs；count 的 fill+scatter_add+cast
   三次 launch 合成一个 kernel（图内 6.3 → 1.1 μs）。SGLang 分支 commit `ac3ec9a9af`，输出逐位一致。
   剩余 ~10 μs 是串行贪心本身（每副本 expert ~150–200 cycle）+ launch；all-reduce 7–10 μs 接近集合通信下限。

### 6.3 优化后 kernel 的 E2E（`.25`，EP4，1.25×，GSM8K，与 §5.7 同配置同节点）

| | static_global | metro | METRO vs static_global |
|---|---:|---:|---:|
| 原 kernel（§5.7） | 7.218 s | 6.905 s | -4.3% / +4.5% |
| **优化 kernel** | 7.278 s | **6.894 s** | **-5.3% / +5.6%**（相邻配对 -5.1% / -5.7%） |

每步省下的 ~0.05 ms 与开销测量一致；EP4 上 METRO 的净收益从 4.3% 提到 5.3%，逼近 §6.1 给出的均衡上限
（不均衡税 ≈ 6.9% step，减去不可压缩的 all-reduce 与串行贪心 ~1.2%）。

### 6.4 完整 61 层 DeepSeek-V3，EP8 单机，CUDA graph（`.25`，2026-09-06）

模型搬到 `.25:/raid`（642 GB），TP8/DP8/EP8、DeepEP auto、graph bs≤4、`mem-fraction 0.88`（权重 ~114 GB/卡，
graph 9.5 GB，KV 10 万 token/卡，余 6.9 GB）、64 冗余 expert（1.25×；1.5× 在 143.7 GB 上放不下）、token-LPLB
prefill、GSM8K 真实 prompt ISL 128、BS32（每 rank 4 token）、OSL 1024、优化后的 METRO kernel。
每 arm 2 fresh server × 3 次（`v3_ep8_gsm8k_r64_b32_o1024/summary.json`）：

| decode 模式 | 中位（32×1024 tok） | ms/step | tok/s | vs static_global |
|---|---:|---:|---:|---:|
| `static_global`（最强 static） | 46.84 s | 45.7 | 700 | — |
| **`metro`** | **43.55 s** | **42.5** | **752** | **-7.0% latency / +7.6% throughput** |
| `static`（SGLang 默认） | 49.16 s | 48.0 | 667 | +4.9% / -4.7% |

- **METRO vs SGLang 默认 static：-11.4% latency / +12.9% throughput。** 这是在真实 DeepSeek-V3、单机 8×H20、
  论文的 EP 规模上，对 SGLang 现有 EPLB 部署形态（`--ep-dispatch-algorithm static` + 冗余 expert）的直接收益。
- 相邻配对 -6.6% / -7.1%，两个 fresh server 的中位数 43.71 / 43.53 s。
- 比 8 层 proxy（EP8 5.4%）高，与 §5.3 的推断一致：58 个 MoE 层让 routed GEMM 在 step 中的占比更高。
- 每步 3.3 ms 的节省 ≈ 58 层 × 57 μs，与 §6.1 的离线估算（EP8 1.25× ≈ 57–62 μs/层）一致。

### 6.5 干净 baseline 与冗余度扫描（未改动的上游 SGLang vs 分支 METRO，只有 decode 不同）

两个修正（2026-09-06）：
1. **baseline 用未改动的上游代码**：分支的基点 `11b0e5c5ad`（upstream main 2026-07-23，已含 LPLB/Waterfill；所有
   METRO/decode 改动都在其上，24 文件 +3286/-172）。以 `--ep-dispatch-algorithm static`、不带任何 lplb 参数运行，
   prefill/decode 都是原生 EPLB 静态派发。
2. **candidate 的 prefill 也用原生 static**：分支新增 `--ep-dispatch-algorithm static --lplb-decode-load-metric metro`
   （`0cb602a447`），只有 decode 走 METRO。为此修了两个只在这种组合下出现的死锁——混合 prefill/decode step 和
   IDLE rank 的集合通信数量必须与 decode rank 一致（`83200a99f4`、`e5b43a9c35`；`lp` 模式下三种 rank 每层恰好各做一次
   all-reduce，所以从未暴露）。

完整 61 层 DeepSeek-V3、EP8、graph、GSM8K、BS32、OSL 1024，每格 1 fresh server × 3 次
（`v3_ep8_matrix_base_vs_metro/`）：

| 冗余 expert R | 每 rank 物理 expert | 上游原生 static（未改代码） | 分支 `static:metro` | METRO vs 同 R 的上游 | METRO vs R=0 上游 |
|---:|---:|---:|---:|---:|---:|
| 0 | 32 | **45.50 s**（720 tok/s） | — | — | — |
| 16 | 34 | 47.28 s | 44.92 s | -5.0% | -1.3% |
| 32 | 36 | 46.97 s | 43.60 s | -7.2% | -4.2% |
| 64 | 40 | 49.10 s | **43.17 s**（759 tok/s） | **-12.1%** | **-5.1% / +5.4%** |

结论：

- **不开 METRO 时，加副本让 decode 变慢**：R 从 0 到 64，原生 static 从 45.5 → 49.1 s（+7.9%），因为按 source rank
  选副本会把同一个 expert 在多个 rank 上同时激活。EPLB 的副本是为 prefill 均衡加的，在 decode 里是净负担。
- **开 METRO 后副本才变成收益**：R=16 -1.3%、R=32 -4.2%、R=64 -5.1%（相对不加副本的原生 decode）。收益随 R 递增但
  边际递减；61 层 EP8 在 143.7 GB 上放不下 1.5×，R=64（1.25×）是能测到的最优点。R=64 多占 20 GB/卡权重。
- 用户要求的 baseline（**完全不做负载均衡、不加副本的 decode**）下，METRO 的真实收益是 **-5.1% latency /
  +5.4% throughput**；相对 SGLang 用户实际部署的"EPLB + 副本 + static 派发"，是 -12.1% / +13.7%。之前报告的
  -11.4% 属于后者。
- 分支改动没有改变 baseline 行为：上游 `plain_static@64` 49.10 s vs 分支 `plain_static@64` 48.91 / `lp+static` 49.07。
- `static:metro`（原生 static prefill）43.17 s 略快于 `lp:metro`（token-LPLB prefill）43.55–43.71 s。

### 6.6 61 层 EP8 的 GPU-only profile：收益是否"紧致"（2026-09-06 晚）

对 §6.5 的三个 arm 各采 40 步 GPU-only torch profile（`profile_v3_ep8/`，`profile_categories.py --gap-us 58`，
per-step rank-max）：

| | 上游 R=0 | 上游 R=64 | 分支 static:metro@64 |
|---|---:|---:|---:|
| step span（profile 下） | 43.58 ms | 46.56 | **41.95** |
| kernel 时长之和 / span | 100.9% | 101.0% | 102.6% |
| routed grouped GEMM | 23.27 | 26.04（重复激活 +2.8） | 23.97（40 组 kernel 固定开销 +0.7） |
| DeepEP LL dispatch+combine（含等对端） | 7.07 | 7.87 | **4.65（-2.42）** |
| METRO assign / all-reduce | — | — | 0.92 / 0.47（16 + 8 μs/层） |
| dense+attention GEMM | 9.95 | 9.98 | 9.79 |
| attention kernels | 2.34 | 2.35 | 2.34 |

- 三个 arm 的 kernel 之和都 ≈ step（多 stream 略有重叠），**GPU 全程有 kernel、无 bubble、CPU 不在关键路径**；profile 下
  的 43.6 ms 与无 profiler 的 44.4 ms 一致，profiler 未扭曲 step。所以 §6.5 的 5.1% 是 GPU 侧的真实节省，不是 CPU
  开销或未开的特性造成的假象。
- 收益机制：等对端时间 7.07 → 4.65 ms/step；METRO 自身 1.39 ms/step（3.3%）——EP8/R=64 下 assign kernel 涨到
  16 μs/层（64 个副本 expert × 8 rank 的串行贪心），是再挤 1.5–2 个点的地方；40 组 grouped GEMM 比 32 组多 0.7 ms。
- 上游 R=64 比 R=0 慢的 3 ms 全在 routed GEMM 上（重复激活），验证"不开 METRO 的副本是负担"。

### 6.7 双机 EP16 的传输：镜像自带的 NVSHMEM 3.4.5 跑不了 DeepEP LL，先用 NCCL 路径（`.25` + `.6`，RoCE 400G，2026-09-07）

- 机间 RDMA 本身很好：`ib_write_bw` 单 rail 380 Gb/s、`ib_write_lat` 4.5 μs；NCCL 以 `NCCL_IB_GID_INDEX=3`
  （RoCE v2 IPv4）走 8 张 ConnectX-8 rail，EP16 的 AllGather/Reduce 已经是 RDMA。bootstrap 需
  `NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0`（否则选到 172.18.x 的点对点 rail 地址）；`--shm-size 32g`。
- **镜像自带栈上 DeepEP low_latency 起不来，且与配对/fabric 无关**（后来在 §6.9 用 NVSHMEM 3.5.21 跑通了）。LL kernel 依赖 NVSHMEM IBGDA；这套环境
  （容器 `lmsysorg/sglang` 系、NVSHMEM 3.4.5、DeepEP 1.2.1、容器 rdma-core 50 / 宿主 MLNX_OFED 25.07 rdma-core 58、
  内核 5.15.0-1069-nvidia）里 IBGDA 初始化在 `ibgda_create_dct()` 为 DCT 建**指向本地 GID 的 address handle** 时失败：
  `ibgda.cpp:2234 Unable to create ah → create DCT share err → connect EPS failed`。`strace` 显示对应的
  `RDMA_VERBS_IOCTL` 被内核以 **EINVAL** 拒绝，此前的 PD/CQ/SRQ 全部成功；同一容器、同一 rdma-core 跑
  `ibv_ud_pingpong` 到自己的 GID（同样的 self-AH）却成功，差别只剩 NVSHMEM 用 DEVX 打开设备。
  - **单机 2 卡也失败**（`.5` GPU0–1，`NVSHMEM_IB_GID_INDEX=3` 与不设都一样）。单机 EP8 的 LL 之所以能跑，是 NVSHMEM
    在所有 PE 都 NVLink 可达时不初始化 IB 传输，这段代码根本没执行；一跨机就必然暴露。
  - 按用户建议试了同机柜配对 `.5`+`.6`（各 4 卡）：与 `.25`+`.6` 完全相同的错。配对无关。
  - `NVSHMEM_IBGDA_NIC_HANDLER=cpu`：Buffer 初始化成功，但首个 `low_latency_dispatch` 自旋到超时（SGLang 里表现为
    graph 捕获挂死 16 分钟以上）。`NUM_DCT=0`/`NUM_DCI=0`/RC-only、`IB_ADDR_FAMILY/ADDR_RANGE`、ARP 预热都无效。
  - 第二个独立现象：宿主机 `ibv_ud_pingpong` 跨机（`.5`↔`.6`、`.24`↔`.25` 都试了）建完 AH 后数据不通、挂死，
    RC pingpong 正常（7 μs/iter）；MTU 9216/active_mtu 4096 不是原因。即使 AH 修好，DC/UD 类流量能否过交换机仍未知。
  - 团队文档 `REPRODUCE_PR19290.md` 里的双机只跑过 `--deepep-mode normal --disable-cuda-graph`（prefill；normal 模式
    不启用 IBGDA，所以那套 env 在 LL 上不适用）。**真正跑通 LL 跨机的是 `dsv4_h20_cumulative_repro_20260829/`
    那套栈（NVSHMEM 3.5.21 + 针对它重编的 DeepEP），见 §6.9。** 3.4.5 的 AH 失败是 NVSHMEM 版本问题。
- DeepEP normal + 关 graph 不能用来测 decode（§2 的 Codex 结果就是这样测出来平的：launch-bound，GPU 13–16%）。
- 在找到 3.5.21 栈之前，EP16 先用 `--moe-a2a-backend none`（sparse 层 `ScatterMode.FULL`：DP-attention all-gather →
  本地 expert → reduce；NCCL、可 graph、无 NVSHMEM，也是 METRO 论文自己的传输形态），结果在 §6.8。baseline 与 METRO
  同传输，EP16 内部公平；但绝对值和同步结构（NCCL kernel 吸收等待）与 DeepEP LL 不同，DeepEP 路径的结果见 §6.9。
- `scripts/h20/metro_graph_ab_2node.sh`：node0 驱动 node1（ssh），两节点同名路径（`.25` 上用符号链接把 `/lustre/...`
  指到 `/raid`），`MOE_A2A=none|deepep`，每格先发固定 prompt 的 greedy 探针存 `probe.txt`。
  `scripts/h20/ep16_watch.sh`：轮询 node1 候选（`.6`→`.5`）直到 8 卡空闲，逐格跑、逐格过门禁（探针含 "Berlin"、
  3 次 bench 齐、metro 必须快于同 R static），不满足即停。

### 6.8 双机 EP16 结果：METRO 在 NCCL 传输下的收益（61 层 DeepSeek-V3，16 rank，BS32 / OSL 1024）

配置：`--tp-size 16 --dp-size 16 --ep-size 16 --enable-dp-attention --moe-a2a-backend none --cuda-graph-max-bs 4
--mem-fraction-static 0.85 --attention-backend fa3`，`ep8_logical_count.pt` 的 EPLB 放置，gsm8k 128-token prompt，
32 条独立请求（2 token/rank），每格 1 台服务器 × 3 次 bench 取 median；baseline 是上游 `11b0e5c5ad` + 下面说的一行修复。

| R | 上游 `plain_static@R` | 分支 `static:metro@R`（旧：每层 all-reduce） | **`static:metro@R`（新：本地 count，无 all-reduce）** |
|---:|---:|---:|---:|
| 0 | **38.87 s** | — | — |
| 32 | 39.78（+2.3%） | — | **38.41（-1.2%；vs 同 R -3.4%）** |
| 64 | 42.61（+9.6%） | 40.66（+4.6%） | **38.15（-1.9%；vs 同 R -10.5%）** |
| 128 | 45.39（+16.8%） | 40.14（+3.3%） | **37.68（-3.1%；vs 同 R -17.0%）** |

括号内是相对 R=0 上游基线。全部格探针输出正确（"Berlin. The capital of Italy is Rome…"），
原始三次 bench 见 `artifacts/metro_review_20260904/ep16_none_all_cells.txt`（每格第一次带 profile，偏高 2–3 s，median 吸收）。

**读法**：

- 在 NCCL 传输下，不配均衡的冗余是纯负担：static 把激活撒到更多物理副本，`fused_moe_kernel` 每 step 13.2 (R=0) →
  14.4 (R=32) → 17.9 ms (R=128)，所有 rank 均匀涨（每个被激活的 expert 都要整读一遍权重，decode 是权重带宽瓶颈）。
- METRO 把激活收拢，routed GEMM 回到 R=0 的水平（R=64: 13.0；R=128: 12.9），同时降低不均衡税
  （`routed_gemm_imbalance_tax.py`，node0 8 rank 上 Σ_l max_r / Σ_l mean_r）：R=0 **29%** → metro@32 23% →
  metro@64 18% → **metro@128 12%**。R 越大 METRO 越有选择余地，也是 R=128 最快的原因。
- **每层 all-reduce 在跨机 EP16 是致命开销**：256-float NCCL all-reduce 跨机 40 μs/层 = 2.34 ms/step（6%），
  比 METRO 省下的 1.5 ms 不均衡还多，所以旧代码 metro@64 比 R=0 慢 4.6%。`--moe-a2a-backend none` 下 sparse 层是
  `ScatterMode.FULL`，每个 rank 的 topk 本来就是全局 batch，改用 `route_decode_metro_global`（一个 kernel 内 count +
  assign，分支 `62e336d342`）后 all-reduce 归零，metro@64 提速 6.2%。DeepEP 路径没有这个免费午餐，那里的 count 交换
  仍要走 all-reduce（EP8 机内 8 μs/层，可接受）或搭 dispatch 的便车。
- METRO 自身开销：assign kernel 0.63 / 0.75 / 1.39 ms/step（R=32/64/128，11–24 μs/层），随副本数线性涨，
  是 warp 并行化能再拿回的部分。
- 与 EP8/DeepEP LL（§6.5：R=64 -5.1%）相比 EP16/NCCL 的净收益更小（-1.9% ~ -3.1%），因为这条路的每层 AllGather +
  Reduce 固定开销 ~200 μs（step 的 27–30%）吸收了部分等待；但趋势一致：**冗余只有配合 METRO 才是正收益**。

**两次无效实验的教训**（都被"先跑一格就核对"抓住，没有烧掉整轮矩阵）：

1. 分支 `forward_normal`（非 DeepEP 路径）没把 `is_decode` 传给 TopK → METRO 从未启动，`metro@64 == plain@64`
   到小数点后一位。修复 `80f6518357`。
2. 上游 `forward_normal` 只在 `--enable-eplb` 时才构建 `ExpertLocationDispatchInfo`；我们用 `--init-expert-location` +
   `--ep-num-redundant-experts` 但不开 EPLB，于是 **logical id 直接当 physical id** 用——算的是错的 expert、冗余副本一个没
   用上、输出退化成重复 token 所以反而"快"（无效的 plain@0 = 34.86 s）。R=0 的 EPLB 放置也不是恒等排列，同样受影响。
   DeepEP 路径无条件构建映射，EP8 结果不受影响。修复 `a7681f6a36`（上游 baseline 也打了同一行，记录在 `BASE_COMMIT`，
   是 baseline 唯一偏离上游之处）。此后 harness 每格先发 greedy 探针，门禁不过不进下一格。

### 6.9 双机 EP16 · DeepEP low-latency（NVSHMEM 3.5.21 栈）：METRO 均衡有效，输在 count 交换的开销

**怎么跑通的**（`scripts/h20/metro_graph_ab_2node.sh` 的 `DEEPEP_STACK=nvshmem3521`，`run_ep16_deepep.example.sh`）：
复用 `~/workspace/dsv4_h20_cumulative_repro_20260829/` 在 8-29 跑通双机 `--deepep-mode auto` + graph 的栈——

- 镜像 `lmsysorg/sglang:v0.5.18-cu130`（torch 2.13、sgl-kernel 0.4.6.post1；我们的分支/上游 baseline 以 `PYTHONPATH` 覆盖进去，
  分支要求 sgl-kernel 0.4.5，`SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` 下没有问题）；
- **NVSHMEM 3.5.21**（`site_nvshmem_3_5_21`）通过 `LD_LIBRARY_PATH` + `LD_PRELOAD=libnvshmem_host.so.3` + `NVSHMEM_PLUGIN_PATH`
  覆盖镜像自带的 3.4.5；
- 针对 3.5.21 重编的 DeepEP（`site_deepep_pr483_nvshmem3521_rcfix_v1`，PYTHONPATH 前置）；
- `NVSHMEM_IBGDA_NIC_HANDLER=gpu`、`NVSHMEM_IB_GID_INDEX=3`、`NVSHMEM_IB_TRAFFIC_CLASS=106`、`NVSHMEM_QP_DEPTH=1024`、
  `NVSHMEM_ENABLE_NIC_PE_MAPPING=1`、`NVSHMEM_HCA_PE_MAPPING=mlx5_3:1:1,mlx5_2:1:1,mlx5_1:1:1,mlx5_0:1:1,mlx5_5:1:1,mlx5_4:1:1,mlx5_7:1:1,mlx5_6:1:1`
  （GPU→网卡显式映射）、`NCCL_IB_TC=106`、`NCCL_IB_HCA=mlx5_0..7`。
- 先用 `scripts/h20/deepep_ll_preflight_3521.sh` + `deepep_ll_smoke.py`（各 4 卡，一轮 LL dispatch/combine + 数值校验）
  验证：`.5`+`.6` 通过，rel_err 3e-3；随后 EP16 服务 graph 捕获 91 s 完成。8-29 那次的配对是 `.5`+`.25`，配对确实无关。

**结果**（61 层 DeepSeek-V3，16 rank `.25`+`.6`，BS32 / OSL 1024，DeepEP auto = LL decode，CUDA graph；每格 1 服务器 × 3 bench 取 median；
原始值 `artifacts/metro_review_20260904/ep16_deepep_v1/ep16_deepep_all_cells.txt`，探针全部正确）：

| R | 上游 `plain_static@R` | 分支 `static:metro@R`（每层 count all-reduce） | METRO vs 同 R | vs `static@0` | vs 原样 `trivial@0` |
|---:|---:|---:|---:|---:|---:|
| 0（原样：无放置文件、无 dispatch 算法） | **37.65** | — | | | |
| 0（EPLB 放置） | **37.27** | — | | | |
| 32 | 37.33（+0.2%） | 38.49 | +3.1% | +3.3% | +2.2% |
| 64 | 38.98（+4.6%） | 38.38 | -1.5% | +3.0% | +1.9% |
| 128 | 40.09（+7.6%） | **37.83** | **-5.6%** | +1.5% | **+0.5%（打平）** |

两个零点相差 1%（EPLB 放置每层多一个 id 映射 gather，被略好的均衡抵消），METRO 对哪个算都差不多。
对比 NCCL 路径（§6.8）：DeepEP 在 EP16 上整体快 4%（37.3 vs 38.9），上游加副本的代价更小（+4.6%/+7.6% vs +9.6%/+16.8%，
token 级 dispatch 下副本的代价主要是等对端而不是权重重读）。

**Profile**（`summary_*.json`，node0 8 rank，rank-max ms/step）：

| | static@0 | static@32 | metro@32 | static@64 | metro@64 | static@128 | metro@128 |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepEP dispatch+combine（含等对端） | 9.50 | 9.32 | 8.47 | 9.97 | 8.13 | 10.62 | **7.70** |
| routed GEMM | 13.03 | 13.00 | 12.58 | 13.99 | 13.09 | 14.13 | 12.67 |
| METRO count all-reduce（f32×256，跨机） | — | — | 2.16 | — | 2.20 | — | 2.33 |
| METRO assign kernel | — | — | 0.67 | — | 0.77 | — | 1.42 |
| routed GEMM 不均衡税 Σmax/Σmean−1 | 32% | 29% | 23% | 30% | 18% | 30% | **13%** |

- **均衡本身有效且随 R 单调变好**：不均衡税 32% → 23% → 18% → 13%，DeepEP 等对端 9.5 → 7.7 ms/step，routed GEMM 也略降
  （副本让激活更集中）。R=128 时 METRO 每步比 static@0 少花 2.2 ms 在 MoE 上。
- **输在实现开销**：count 交换是每层一次 NCCL all-reduce，跨机 **38–40 μs/层 = 2.2–2.3 ms/step（6%）**；assign kernel 11–24 μs/层。
  两项合计 2.8–3.8 ms/step，大于收回的 1.4–2.2 ms，所以净 +0.5 ~ +3%。EP8 机内这两项分别是 8 μs 和 16 μs/层，所以单机净赢 5.1%。
- 因此 EP16 上 METRO 翻正的工程路径很明确：(1) count 交换换成 NVSHMEM/IBGDA 的 1 KB all-gather（论文的做法，跨机应在 ~10 μs 量级）
  或藏到 shared-expert 流后面；(2) assign kernel warp 并行化（R=128 时 24 μs → 目标 <8 μs）。两项做完，metro@128 预计
  比原样快 4–5%，与 EP8 一致。
- NCCL 路径（§6.8）上这个 all-reduce 因为 topk 天然全局而被免掉，所以那里 metro@128 已经 -3.1%；DeepEP 路径没有这个便利。

**门禁记录**：最后一格 `metro@32` 未过"快于同 R static"门禁（38.49 vs 37.33），这是真实结果不是故障——R=32 时可选副本太少，
METRO 收回的 0.85 ms 不够付 2.8 ms 的固定开销。

### 6.10 三条降开销的路（2026-09-08）：kernel 并行化成功，"用上一步计数"失败，NVSHMEM 待做

§6.9 的账：EP16/DeepEP 上 METRO 收回 2.2 ms/step，自身花 3.75 ms（all-reduce 2.33 + assign 1.42）。三条路都试了：

**① assign kernel warp 并行化（成功）** — `metro_route_v2`（分支 `24c7b9c6d5`）：静态表预计算（默认副本、单副本 rank、
副本 expert 列表/掩码/物理 id，一次合并加载代替链式 gather）、活跃副本 expert 的 ballot 有序压缩、每个 expert 一条
`redux.sync` warp argmin（lane r 持 rank r 的负载，(load,rank) 打包保证平局取小 rank）、软件流水预取下一个 expert。
与 v1 在 60 组随机放置上 bit-exact（`scripts/h20/test_metro_v2_kernels.py`）。GPU 时间 R=32/64/128：8.5/11.9/26.4 →
**4.0/4.9/8.0 μs**；in-graph R=128 24.5 → **10.6 μs/层**（0.62 ms/step）。默认启用，`SGLANG_METRO_KERNEL=v1` 回退。

**② 把 all-reduce 挪出关键路径（失败，原因干净）** — `SGLANG_METRO_COUNT_MODE=stale`：一个 kernel 读上一步的全局计数
→ 贪心 → 路由 → 用本步本地计数覆盖同一 buffer；all-reduce 在 side stream 发出，下一层 join（最后一层当场 join，graph 捕获
无悬挂 stream）。所有 rank 用同一份 t−1 数据，副本一致。`SGLANG_METRO_STALE_INACTIVE_WEIGHT=3` 再对 t−1 未激活的副本
expert 做二次贪心（期望权重 3/8）。结果不均衡税 **32.9%**（stale）/ 29%（stale3）—— 与不做均衡的 static（32%）无异。
用离线路由轨迹（`scripts/metro_stale_set_overlap.py`，gsm8k/mmlu/sharegpt）算相邻两步同一层的激活集合：上一步能命中本步的
比例 **64–65%**，而随机集合的基线就是 **58–62%**（148–159/256 个 expert 本来就活跃）。**相邻 decode step 的 expert 激活在集合
层面近乎独立**，t−1 对 t 没有信息量；这条路对 METRO 走不通，对 L3 也是提醒（预测必须来自 hidden state，不能来自上一步）。
另外 side stream 上的 NCCL 与 DeepEP LL 抢 NIC/SM：DeepEP 等待 7.8 → 11.8 ms/step，"异步 all-reduce 不花钱"也不成立。

**③ NVSHMEM 传输（未完成）** — ctypes 直接调 `nvshmemx_float_sum_reduce_on_stream`（`scripts/h20/nvshmem_reduce_smoke.py`，
用 DeepEP 初始化好的 NVSHMEM）：单机 2 卡正确、可 graph 捕获，in-graph 15.9 μs（NCCL 15.8）；跨机 2+2 卡试跑挂住未深究。
host 端 on-stream 集合通信本身有 ~10 μs 启动开销，即便通了收益也有限；真正的做法是 device 侧 put+signal 融进 assign kernel。

**结果**（EP16 · DeepEP LL · R=128；对照旧 METRO 37.83、`plain_static@0` 37.27、原样 37.65；
`artifacts/metro_review_20260904/ep16_deepep_variants/`）：

| 变体 | 耗时 | vs static@0 | vs 原样 | assign | all-reduce | DeepEP 等待 | 不均衡税 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 旧 METRO（v1，同步） | 37.83 | +1.5% | +0.5% | 1.42 ms | 2.33 | 7.70 | 13% |
| **v2 kernel，同步** | **36.95** | **-0.8%** | **-1.9%** | **0.62** | 2.28 | 7.79 | 13.3% |
| stale | 37.99 | +1.9% | +0.9% | 0.65 | 2.85（并行） | 11.8 | 32.9% |
| stale + 二次贪心 | 37.85 | +1.6% | +0.5% | 0.96 | 2.78（并行） | 9.74 | 29.0% |

R=64：v2sync 38.36（assign 0.44），与旧版 38.38 持平——R=64 时 all-reduce 占主导，kernel 提速看不出来。

**结论**：v2 kernel 让 METRO 在 EP16/DeepEP 上第一次同时快过两个零点（-0.8% / -1.9%），但 count 交换**必须同步**，剩下的
2.3 ms/step 是唯一的大杠杆。可行路径是 device 侧 NVSHMEM/IBGDA：assign kernel 内每个 rank 把 256 个计数 put 给 15 个对端并
signal，等 15 个 signal 后本地求和再贪心——DeepEP LL dispatch 的计数交换就是这么做的，跨机 ~5 μs。需要在 DeepEP 的构建里加一个
kernel（nvshmem 3.5.21 device 库 + `nvshmemx_cumodule_init`），预计 2.3 → ~0.5 ms，metro@128 到 -4% ~ -5%，与 EP8 一致。

### 6.11 单测：跨机 256-float all-reduce 到底要多少 μs（2026-09-08，`.25`+`.6`，16 rank）

`scripts/h20/allreduce_latency.py`（`run_ar_variants.sh` 驱动）：256×f32 sum all-reduce，eager 与 CUDA graph 内（每 graph 58 次
= 一个 decode step），wall time per op；同一容器/NCCL 环境（`NCCL_IB_GID_INDEX=3`、TC 106、8 rail）。

| 通信域 | in-graph μs/op | kernel 时长 | NCCL 选的算法 |
|---|---:|---:|---|
| **world = 16 rank（2 机 × 8）** | **44.1** | 55（含等对端） | Tree LL |
| intra = 本机 8 rank | 27.7 | 17 | Ring LL |
| pair = 1 跨机 hop（2 rank） | 31.8 | 21.5 | Ring LL |
| 2level = intra + pair 两级 | 51.4 | 39 | — |
| all_gather 16×256 | 48.0 | 39 | Ring LL |
| world, `NCCL_ALGO=Ring` | 81.6 | 143 | Ring LL |
| world, `NCCL_PROTO=LL` | 44.6 | — | Tree LL（已是默认） |

- **跨机 16 rank 一次 44 μs**，与 EP16 profile 里 38–40 μs 的 kernel 时长一致（profile 不含 graph 内 ~5 μs 的发射间隙）。
  NCCL 已在用最优组合（Tree + LL）；Ring 翻倍，两级/all-gather 都不更快。
- 根因是 NCCL 跨机小消息的路径：GPU 写 host pinned buffer → CPU proxy 线程 post RDMA → 对端 CPU proxy → 对端 GPU，单 hop
  就要 ~20 μs kernel 时间（"pair"），而裸 RDMA `ib_write_lat` 是 4.5 μs。IBGDA（GPU 直接 post RDMA，DeepEP LL 的做法）没有 proxy，
  单 hop ~5 μs——这就是 §6.10 结论里 device 侧 NVSHMEM 方案的依据。
- **EP8 为什么只要 7 μs**：查 EP8 profile，`.25` 单机 EP group 走的是 SGLang 的 custom all-reduce（`all_reduce_kernel<AllReducePushImpl<float>>`，
  one-shot NVLink），每层 6.8 μs；纯 NCCL 机内 8 rank 在 graph 里其实要 27 μs。EP group 一跨机 custom all-reduce 就禁用，退回 NCCL Tree。
  所以 EP8 → EP16 的 all-reduce 成本是 6.8 → 44 μs/层（×6.5），不是简单的"多了一跳"。
- **profile 里的 all-reduce 时长含不含"等慢 rank"？** 含，但这次很少（`scripts/h20/allreduce_arrival_skew.py`，node0 8 rank 同一时钟）：
  各 rank 到达 all-reduce 的时间差中位 **12.9 μs**（p90 21，最大 30）；最早到的 rank kernel 39.6 μs，**最晚到的 25.4 μs**（纯通信下界）；
  完成时刻分散 5 μs。偏差小是因为 benchmark 的 32 条序列 prompt 等长、同步解码，attention 每 rank 1.00–1.03 ms/step 几乎一样。
  **真实混合负载下 KV 长度不同，attention 偏差（几十到几百 μs）会全部落进 attention 之后的第一个集合通信**——开 METRO 是 all-reduce，
  不开是 DeepEP dispatch（LL 要等所有 sender）——profile 会记在 all-reduce 头上，但那不是它的边际成本。因此 all-reduce 的边际成本按
  同步状态的纯时延算：**25–40 μs/层 = 1.5–2.3 ms/step**（前表的 2.3 是上界）；评估 METRO 在生产负载上的收益要看 step 总时间，
  不能看 all-reduce 那一行。
- **DeepEP LL dispatch 本身就是一个"到达屏障"**（`scripts/h20/deepep_dispatch_barrier.py`）。源码 `internode_ll.cu`：发送阶段每个
  rank 给**每个 rank 的每个 expert**都发一个计数（0 也发，编码为 −1），接收阶段对每个 `(local_expert, src_rank)` 自旋等计数到达——
  所以任何 rank 都要等**全部 16 个 rank 执行完发送阶段**才能开始自己的 expert GEMM（不等别人的 GEMM，那部分在 combine 等）。
  trace 证据（static@0，无 all-reduce，node0 8 rank）：到达 dispatch 的偏差中位 11.6 μs；**最早到的 rank recv kernel 30.2 μs，最晚到的
  15.7 μs，差 14.5 ≈ 偏差**——早到的在 recv 里等晚到的。所以"负载轻的 rank 发完就能进 MoE"不成立；不开 METRO 时 attention 偏差
  落在 dispatch recv，开了落在 all-reduce，都只付一次。
- **NCCL tree 的跨机完成偏差是 all-reduce 的隐性成本**：开 METRO 后 node 内到达 dispatch 的偏差降到 5 μs（all-reduce 已同步），
  但 node1 的 all-reduce 比 node0 晚完成 6–10 μs（树的下游一侧），node0 的 dispatch recv 从 16–21 μs 涨到 **37–43 μs**（node1 26–28）。
  即 all-reduce 每层实际花费 ≈ 自身 36–40 + 诱发的 dispatch 等待 ~15 = **50–55 μs ≈ 3 ms/step**，比前表的 2.3 还多。device 侧
  put+signal 交换（每个 rank 等同样的 15 个 sender，完成时刻天然一致）能把两项一起去掉。
- **"trace 很紧致、各 rank 的 dispatch/MoE/combine 起止时间都不一样、看不到显式同步"与上面并不矛盾**
  （`scripts/h20/deepep_layer_sync_spread.py`，static@0，node0 8 rank 同一时钟，一层之内各阶段起止时刻的跨 rank 分散，中位 μs）：

  | 阶段 | 跨 rank 分散 |
  |---|---:|
  | dispatch send 开始 | 11.6 |
  | **dispatch recv 结束** | **8.7**（收敛） |
  | gate_up GEMM 开始 | 8.9 |
  | **down GEMM 结束** | **131.8**（不均衡） |
  | combine send 开始 | 131.0（各算各的，算完就发） |
  | **combine recv 结束** | **10.4**（收敛） |

  combine recv kernel 时长：GEMM 最早算完的 rank **173 μs**，最晚的 29 μs。等待全部发生在 recv kernel 的自旋里——trace 上是一个
  变长的 kernel，不是空隙，所以看起来"紧致、无同步"；开始时刻确实各不相同（发完就走），但每个 recv 阶段的**结束时刻**都收敛到
  ~10 μs 以内。这 131 μs/层的 GEMM 结束分散就是 METRO 要收的不均衡税，它以 combine recv 自旋的形式记在"最快的 rank"账上。
  一层的实例（static@0，第 812 层，相对最早的 dispatch send 起点，μs）——8 个 rank 的 down GEMM 分别在 173/249/307/308/309/362/363/366
  结束，combine recv 却都在 394–409 结束；rank 1 的 combine recv 从 186 自旋到 404（218 μs），rank 4 的只有 380→409。
  8 个 rank 合并到一条时间轴的 Perfetto 文件：`bench/metro_review_20260904/merged_traces/ep16_static0_node0_8ranks.json`
  （`scripts/h20/merge_rank_traces.py`，pid = rank；3 个 decode step）。跨进程时钟对齐的验证：所有层里 dispatch recv 结束分散的
  最小值 1.2 μs，说明 8 个进程的时间戳偏移不超过 ~1 μs。
- **跨机也是同一个屏障，只是 trace 上看不出来**（`scripts/h20/node_phase_times.py` + `compare_node_phases.py`，static@0，每层取本机 8 rank 的
  中位时刻，node0 − node1）：两台机器的 host 时钟相差一个常数 **227.66 ms**——直接看原始时间戳，node1 的 dispatch/combine 会
  "晚" 0.23 s，这就是跨机看起来"完全不同步"的来源。扣掉这个常数后：dispatch recv 结束 node0/node1 之差中位 **12 μs**（最大 25），
  combine recv 结束 **21 μs**（最大 37），而 combine send 开始 26 μs（p90 66、最大 123，GEMM 不均衡）。即跨机的 recv 结束也收敛，
  窗口比机内（~10 μs）宽一倍，因为差的是 RDMA 传播（5–20 μs）对 NVLink（~2 μs）。机制上没有区别：每个 rank 都要等 16 个 source
  （本机 + 远端）的计数/flag 才能结束 recv。
- **为什么 LL 的结束还剩 10–40 μs 的分散，而 normal 模式几乎同时结束**：normal 模式在发数据前有显式的 `notify_dispatch` +
  `barrier_block`（全员 flag 互换、双向屏障），结束也有 barrier，所以收敛到几 μs。LL 没有任何显式屏障——它只把"计数"随数据一起发，
  接收方等的是**自己入站的最后一条消息**，收齐就走，不等别人收齐。所有 rank 的入站都由同一个"最慢的 sender"封口，所以结束时刻
  仍然耦合，但耦合是单向依赖而非同步点，收敛窗口就是：sender 向 256 个 expert 逐个发计数的顺序差（~10 μs）+ 传播差（NVLink ~2 μs、
  RDMA 5–20 μs，不同目的地走不同 NIC/QP）+ 收到最后一条之后各自剩下的活（dispatch 要把收到的 token 拷进 packed 布局，量与收到的
  token 数成正比——正是不均衡本身）。combine recv 也是对**全部 `num_experts` 个 flag** 自旋（源码 "Wait all ranks to arrive"），
  不只等自己 token 用到的 expert，所以也是全员封口。结论：LL 的"发完收完就走"是对的，但"走"的时刻由最慢的 sender 决定，
  因此各 rank 结束时刻相差的是传播与拷贝的十几到几十 μs，不是等待模型上的自由。


按成本从低到高（§5 已完成的划掉）：

1. ~~重算 61 层 trace~~ → §5.1，-6.09%。
   ```bash
   python3 scripts/metro_trace_critical_path.py \
     --baseline-dir .../metro_decode_full61_20260904/profile_static_manual_r2 \
     --candidate-dir .../metro_decode_full61_20260904/profile_metro_manual \
     --output full61_critical_path.json
   ```
2. ~~真实路由 EP 扫描 + baseline 分解~~ → §5.2。
   ```bash
   PYTHONPATH=<repo>/src python3 scripts/metro_ep_sweep.py --routing npy \
     --routes-npy .../metro_workload_sweep_20260904/raw_routes/<dataset>/logical_routes.npy \
     --placement eplb --eplb-weights .../waterfill/decode_unique_20260723/ep4_8l_logical_count.pt \
     --placement-file .../metro_repro_20260903/placements/placement_r64.pt --placement-file-ep 4 \
     --ep-sizes 4,8,16,32 --redundancy 1.25,1.5 --batch-sizes 8,16,32,64,128 --steps 0 \
     --policies static_global,sglang_static,dynamic_random,metro,metro_fixed_first,optimal
   ```
3. ~~修 E2E 测法~~ → §5.3，+4.6% / +9.6%。
   ```bash
   REPO=<sglang branch> ARTIFACT_ROOT=<dir> MODES='static_global metro static' PORT_BASE=37000 \
     scripts/h20/metro_graph_ab_h20.sh && scripts/h20/summarize_graph_ab.py <dir>
   ```
4. ~~1.5× 副本率 E2E~~ → §5.5，-4.8%（随机 id 路由）。
5. ~~kernel 加 fixed-first pass~~ → §5.6：SGLang 分支已是 fixed-first；本仓库 SDK 加 `metro_fixed_first` 追平。建议把 SDK 默认切到 fixed-first 并同步 `metro-reproduction.md` 的描述。
6. ~~单机 8 卡 EP8~~ → §5.10：1.5× 下 vs 最强 static -5.4%，vs dynamic -11.1%。激活专家数降幅按预测增长，但 EP8 上 routed GEMM 的 step 占比下降抵消了一部分。
7. ~~真实 prompt 的 E2E~~ → §5.7，GSM8K 下 -4.3%（1.25×）/ -5.0%（1.5×），vs SGLang 默认 -8.5% / -10.4%。ShareGPT 数据也已在 `.25:/raid/xutingz/bench/`，可同样跑。
8. ~~论文式 `dynamic` baseline 的 E2E~~ → §5.8，比 SGLang static 再差 0.5–1 个点；METRO vs dynamic -8.6% / -10.8%。

## 8. 文件索引

- 本次新增
  - `scripts/metro_trace_critical_path.py` — 逐层 rank-max 关键路径重算（纯 stdlib，含 self-test）
  - `scripts/metro_ep_sweep.py` — EP/副本率/BS 扫描与 6 种策略（含精确最优）对比；输入支持合成路由、
    `[tokens, layers, topk]` `.pt`、SGLang 采集的 `[seq, pos, layer, topk]` `.npy`；placement 支持服务器保存文件、
    仓库内 SGLang EPLB 重生成、纯 Python 近似（含 self-test，最优解与穷举一致）
  - `scripts/h20/metro_graph_ab_h20.sh` — CUDA-graph / low_latency decode A/B harness（任意 decode 模式列表，
    fresh server × 多次测量）
  - `scripts/h20/summarize_graph_ab.py` — A/B 结果汇总（按模式中位数、相邻配对）
  - `artifacts/metro_review_20260904/ep_sweep_synthetic_alpha0.6.json` — §2.4 合成扫描
  - `artifacts/metro_review_20260904/full61_critical_path.{json,log}` — §5.1
  - `artifacts/metro_review_20260904/real_sweep/*.json`、`aggregate.json` — §5.2（8 数据集逐 cell + 平均）
  - `artifacts/metro_review_20260904/graph_ab_r64_b32_o1024/` — §5.3 summary 与样例 bench log
  - `artifacts/metro_review_20260904/nograph_ab_r64_b32_o1024/summary.json` — §5.4
  - `artifacts/metro_review_20260904/graph_ab_r128_b32_o1024*/summary.json` — §5.5
  - `artifacts/metro_review_20260904/real_gsm8k_r{128,64}_b32_o1024*/summary.json` — §5.7
  - `artifacts/metro_review_20260904/real_sharegpt_r{128,64}_b32_o1024/summary.json`、`dyn_gsm8k_r{64,128}_b32_o1024/summary.json` — §5.8
  - `artifacts/metro_review_20260904/ep8_gsm8k_r{128,64}_b32_o1024/summary.json` — §5.10
  - `artifacts/metro_review_20260904/headroom/` — §6.1（9 个 JSON + `aggregate.txt`）
  - `artifacts/metro_review_20260904/profile_ep4_r64/summary_v{1,2}kernel.json` — §6.2
  - `artifacts/metro_review_20260904/real_gsm8k_r64_b32_o1024_v2kernel/summary.json` — §6.3
  - `artifacts/metro_review_20260904/v3_ep8_gsm8k_r64_b32_o1024/` — §6.4（61 层 DeepSeek-V3 EP8）
  - `artifacts/metro_review_20260904/v3_ep8_matrix_base_vs_metro/` — §6.5（上游基点 baseline × 冗余度扫描）
  - `artifacts/metro_review_20260904/profile_v3_ep8/` — §6.6（61 层 EP8 三 arm 的 GPU-only profile）
  - `artifacts/metro_review_20260904/ep16_none_v2/`, `ep16_none_v3/`, `ep16_none_all_cells.txt` — §6.8（双机 EP16 矩阵的 profile 分类与原始 bench）
  - `artifacts/metro_review_20260904/ep16_deepep_v1/` — §6.9（双机 EP16 DeepEP LL 矩阵：profile 分类、原始 bench）
  - `artifacts/metro_review_20260904/ep16_deepep_variants/` — §6.10（v2 kernel / stale / stale3 变体）
  - `scripts/h20/test_metro_v2_kernels.py`, `scripts/metro_stale_set_overlap.py`, `scripts/h20/nvshmem_reduce_smoke.py` — §6.10 工具
  - `scripts/h20/allreduce_latency.py`, `allreduce_bench.sh`, `run_ar_variants.sh` — §6.11 跨机 all-reduce 单测
  - `scripts/h20/deepep_ll_preflight_3521.sh`, `deepep_ll_smoke.py`, `run_ep16_deepep.example.sh` — §6.9 的 NVSHMEM 3.5.21 栈冒烟与启动
  - `scripts/h20/ep16_watch.sh`, `scripts/h20/routed_gemm_imbalance_tax.py` — §6.7/6.8 工具
  - `scripts/h20/metro_graph_ab_2node.sh` — §6.7 双机 harness
  - `scripts/h20/profile_categories.py` — graph 下 per-step kernel 分类（周期切分）
  - SGLang 分支 `reproduce/metro-decode-fused`：`ce0484a452` 新增 `--lplb-decode-load-metric static_global`；
    `4dfe7f9ee7` 新增 `SGLANG_LPLB_IPM_TORCH_FALLBACK`；`69b912757e` 新增 `dynamic_random`；
    `ac3ec9a9af` METRO kernel 共享内存预载 + fused count；`0cb602a447` `static` 算法下的 decode-only 策略；
    `83200a99f4` / `e5b43a9c35` 混合 step 与 IDLE rank 的集合通信配平。上游基点 `11b0e5c5ad`（`.25:/raid/xutingz/repo_base/`）
  - 节点副本：`.24`/`.25` 的 `/raid/xutingz/{models,repo,cache,bench}`（模型、分支快照、JIT cache、镜像 `lmsysorg/sglang:metro-repro`）
  - 本仓库 `e549b97`：`metro_fixed_first` 变体（reference + CUDA + pipeline）
- Codex 产物
  - `docs/metro-reproduction.md`、`artifacts/metro_decode_full61_20260904/`
  - `/scratch/gitsrc/cake-dev/artifacts/metro_decode_breakdown_20260904/REPORT.md`
  - Lustre：`metro_repro_20260903/`、`metro_workload_sweep_20260904/`、`metro_decode_full61_20260904/`、
    `metro_theory_realdata_v2_20260904/`
  - SGLang 分支：`reproduce/metro-decode-fused` @ `55733a87f4`
