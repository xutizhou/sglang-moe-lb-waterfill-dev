#!/usr/bin/env bash
# usage: allreduce_bench.sh <node_rank> <master_ip> <gpus> <port> <tag> [extra -e env...]
set -u
NODE_RANK=$1; MASTER=$2; GPUS=$3; PORT=$4; TAG=$5; shift 5
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
NAME=arbench_${NODE_RANK}
S_DIR=${S_DIR:-/lustre/raplab/client/xutingz/workspace/bench/metro_review_20260904}
docker rm -f $NAME >/dev/null 2>&1
timeout 240 docker run --rm --name $NAME --gpus "\"device=${GPUS}\"" --privileged --network host --ipc host --shm-size 16g \
  --ulimit memlock=-1 -v /dev/infiniband:/dev/infiniband -v /raid:/raid -v /lustre:/lustre \
  -e NCCL_SOCKET_IFNAME=bond0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=106 -e NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 \
  -e GLOO_SOCKET_IFNAME=bond0 -e NCCL_DEBUG=WARN -e TAG=$TAG -e PER_NODE=$NPROC "$@" \
  lmsysorg/sglang:v0.5.18-cu130 \
  torchrun --nnodes 2 --node_rank $NODE_RANK --nproc_per_node $NPROC --master_addr $MASTER --master_port $PORT $S_DIR/allreduce_latency.py
