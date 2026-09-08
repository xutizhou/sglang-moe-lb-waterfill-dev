#!/usr/bin/env bash
set -u
cd /raid/xutingz/bench
source ./env25.sh
export GPU_DEVICES=0,1,2,3,4,5,6,7 DP_SIZE=8
export MODEL=/raid/xutingz/models/DeepSeek-V3 EXPERT_LOCATION=/raid/xutingz/bench/ep8_logical_count.pt
export MEM_FRACTION=0.88 CUDA_GRAPH_MAX_BS=4 WATCHDOG_TIMEOUT=3600
export BASE_REPO=/raid/xutingz/repo_base/sglang_11b0e5c5ad
MODES="${MODES:-base:plain_static@0 base:plain_static@64 static:metro@64 base:plain_static@32 static:metro@32 base:plain_static@16 static:metro@16}" \
  RUN_TAG=v3mx REPEATS=3 OUTPUT_LEN=1024 WARMUP_OUTPUT_LEN=64 \
  BENCH_MODE=real REAL_INPUT_IDS_NPY=/raid/xutingz/bench/gsm8k_input_ids.npy REAL_PROMPT_LEN=128 \
  ARTIFACT_ROOT=/raid/xutingz/bench/${ARTIFACT_NAME:-v3_ep8_matrix_base_vs_metro} PORT_BASE=${PORT_BASE:-36000} MAX_CONCURRENCY=32 \
  ./metro_graph_ab_h20_v4.sh
echo "[$(date +%T)] MATRIX_DONE"
