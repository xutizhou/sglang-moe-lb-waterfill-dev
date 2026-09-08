#!/usr/bin/env bash
# EP16 two-node with DeepEP low-latency on the NVSHMEM 3.5.21 stack (see
# metro_graph_ab_2node.sh DEEPEP_STACK).  Same cells/harness as run_ep16.sh.
set -u
cd /raid/xutingz/bench
L=/lustre/raplab/client/xutingz/workspace
export NODE0_IP=10.6.131.25 NODE1_SSH=${NODE1_SSH:-xutingz@10.6.131.6}
export IMAGE=lmsysorg/sglang:v0.5.18-cu130 DEEPEP_STACK=nvshmem3521 MOE_A2A=deepep DEEPEP_MODE=auto
export REPO=$L/tmp/metro_decode_fused_20260903 BASE_REPO=$L/tmp/sglang_base_11b0e5c5ad
export MODEL=$L/model/DeepSeek-V3 EXPERT_LOCATION=$L/bench/waterfill/ep8_logical_count.pt
export REAL_CLIENT=/raid/xutingz/bench/real_prompt_client.py REAL_INPUT_IDS_NPY=/raid/xutingz/bench/gsm8k_input_ids.npy REAL_PROMPT_LEN=128
export CACHE_ROOT_NODE0=/raid/xutingz/cache CACHE_ROOT_NODE1=/tmp
export MEM_FRACTION=${MEM_FRACTION:-0.85} CUDA_GRAPH_MAX_BS=4 WATCHDOG_TIMEOUT=3600
MODES="${MODES:?}" ARTIFACT_ROOT=/raid/xutingz/bench/${ARTIFACT_NAME:?} PORT_BASE=${PORT_BASE:-40000} RUN_TAG=${RUN_TAG:-ep16d} \
  REPEATS=${REPEATS:-3} OUTPUT_LEN=${OUTPUT_LEN:-1024} WARMUP_OUTPUT_LEN=64 MAX_CONCURRENCY=${MAX_CONCURRENCY:-32} \
  ./metro_graph_ab_2node.sh
echo "[$(date +%T)] EP16_DONE"
