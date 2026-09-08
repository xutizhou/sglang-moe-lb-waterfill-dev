#!/usr/bin/env bash
# Two-node DeepEP low-latency smoke with the stack that worked on 2026-08-29
# (dsv4_h20_cumulative_repro_20260829): image v0.5.18-cu130 + NVSHMEM 3.5.21
# overlay + DeepEP built against it (pr483 rcfix) + IBGDA gpu handler.
# usage: $0 <node_rank> <master_ip> <gpus> [port] [smoke.py]
set -u
NODE_RANK=$1; MASTER=$2; GPUS=$3; PORT=${4:-29590}
T=/lustre/raplab/client/xutingz/workspace/dsv4_h20_cumulative_repro_20260829
S=${5:-/lustre/raplab/client/xutingz/workspace/bench/metro_review_20260904/deepep_ll_smoke.py}
NPROC=$(echo "$GPUS" | tr ',' '\n' | wc -l)
NAME=deepep_ll3521_${NODE_RANK}
docker rm -f $NAME >/dev/null 2>&1
timeout 400 docker run --rm --name $NAME --gpus "\"device=${GPUS}\"" --privileged --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /dev/infiniband:/dev/infiniband -v /lustre:/lustre \
  -v $T/site_deepep_pr483_nvshmem3521_rcfix_v1:/opt/candidate/deepep:ro \
  -v $T/site_nvshmem_3_5_21:/opt/candidate/nvshmem:ro \
  -e PYTHONPATH=/opt/candidate/deepep \
  -e LD_LIBRARY_PATH=/opt/candidate/nvshmem/nvidia/nvshmem/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib \
  -e LD_PRELOAD=/opt/candidate/nvshmem/nvidia/nvshmem/lib/libnvshmem_host.so.3:/usr/local/cuda/lib64/libcudart.so.13 \
  -e NVSHMEM_PLUGIN_PATH=/opt/candidate/nvshmem/nvidia/nvshmem/lib \
  -e NVSHMEM_IB_GID_INDEX=3 -e NVSHMEM_IB_TRAFFIC_CLASS=106 -e NVSHMEM_IBGDA_NIC_HANDLER=${HANDLER:-gpu} -e NVSHMEM_QP_DEPTH=1024 \
  -e NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
  -e NVSHMEM_HCA_PE_MAPPING=mlx5_3:1:1,mlx5_2:1:1,mlx5_1:1:1,mlx5_0:1:1,mlx5_5:1:1,mlx5_4:1:1,mlx5_7:1:1,mlx5_6:1:1 \
  -e NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0 -e NVSHMEM_DEBUG=${NVSHMEM_DEBUG:-WARN} \
  -e NCCL_SOCKET_IFNAME=bond0 -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=106 \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7 -e GLOO_SOCKET_IFNAME=bond0 -e NCCL_DEBUG=WARN \
  lmsysorg/sglang:v0.5.18-cu130 \
  bash -c "python3 -c 'import deep_ep, importlib.metadata as m; print(\"deep_ep from\", deep_ep.__file__, \"torch\", m.version(\"torch\"))'; torchrun --nnodes 2 --node_rank $NODE_RANK --nproc_per_node $NPROC --master_addr $MASTER --master_port $PORT $S"
echo "exit=$?"
