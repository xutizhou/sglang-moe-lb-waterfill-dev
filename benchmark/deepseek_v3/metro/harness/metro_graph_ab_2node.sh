#!/usr/bin/env bash
# Two-node (2 x 8 GPU) decode A/B: EP16 with DP attention, DeepEP auto
# (low_latency decode over RDMA/NVSHMEM), CUDA graphs.  Same mode syntax as
# metro_graph_ab_h20.sh:  [base:]<policy>[@<redundant_experts>]  where policy is
# plain_static | plain_trivial | <metric> (lp prefill) | <algo>:<metric>.
# plain_trivial = stock SGLang with nothing set: no --init-expert-location, no
# --ep-dispatch-algorithm, no redundancy (the untouched zero point).
#
# Runs from node0.  node1 is driven over ssh; both nodes must expose the model,
# repo and base repo under the SAME paths (h20-25 mirrors the Lustre tree with
# symlinks into /raid).  JIT caches are per node (CACHE_ROOT_NODE0/1).
set -euo pipefail

node0_ip="${NODE0_IP:?}"          # this host, also the rendezvous address
node1_ssh="${NODE1_SSH:?}"        # e.g. xutingz@10.6.131.6
image="${IMAGE:-sha256:06e0aa8359f56ab7a316b60900e6d0dff9bfbb4190d9ce1f5c8caa27e875ae2f}"
repo="${REPO:?}"                  # branch checkout (same path on both nodes)
base_repo="${BASE_REPO:-}"        # untouched upstream checkout (same path on both nodes)
model="${MODEL:?}"
expert_location="${EXPERT_LOCATION:?}"
artifact_root="${ARTIFACT_ROOT:?}"
modes="${MODES:?}"
port_base="${PORT_BASE:?}"
run_tag="${RUN_TAG:-ep16}"
output_len="${OUTPUT_LEN:-1024}"
warmup_output_len="${WARMUP_OUTPUT_LEN:-64}"
repeats="${REPEATS:-3}"
max_concurrency="${MAX_CONCURRENCY:-32}"
redundant_experts="${REDUNDANT_EXPERTS:-64}"
deepep_mode="${DEEPEP_MODE:-auto}"
# MOE_A2A=deepep (default) or none.  "none" is SGLang's NCCL all-gather /
# reduce-scatter EP path: graph-capturable and NVSHMEM-free, the fallback when
# DeepEP low-latency cannot initialise across nodes (IBGDA gpu handler fails to
# create DC address handles on this routed RoCE fabric).
moe_a2a="${MOE_A2A:-deepep}"
cuda_graph_max_bs="${CUDA_GRAPH_MAX_BS:-4}"
mem_fraction="${MEM_FRACTION:-0.85}"
watchdog_timeout="${WATCHDOG_TIMEOUT:-3600}"
extra_server_args="${EXTRA_SERVER_ARGS:-}"
extra_docker_env="${EXTRA_DOCKER_ENV:-}"
real_input_ids_npy="${REAL_INPUT_IDS_NPY:?}"
real_prompt_len="${REAL_PROMPT_LEN:-128}"
real_client="${REAL_CLIENT:?}"   # same path on both nodes not needed; runs in node0 container
profile_steps="${PROFILE_STEPS:-0}"
cache_root_node0="${CACHE_ROOT_NODE0:-/tmp}"
cache_root_node1="${CACHE_ROOT_NODE1:-/tmp}"
data_mounts="${DATA_MOUNTS:--v /lustre:/lustre -v /raid:/raid}"
# Cluster settings (RoCE 400G, 8 rails, bond0 mgmt).  Bootstrap must be pinned to
# bond0 or NVSHMEM picks a point-to-point rail address the peer cannot reach.
nccl_env="${NCCL_ENV:--e NCCL_SOCKET_IFNAME=bond0 -e GLOO_SOCKET_IFNAME=bond0 -e NCCL_IB_GID_INDEX=3 -e NCCL_DEBUG=WARN}"
nvshmem_env="${NVSHMEM_ENV:--e NVSHMEM_IB_GID_INDEX=3 -e NVSHMEM_IBGDA_NIC_HANDLER=cpu -e NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0}"
# DEEPEP_STACK=nvshmem3521: the stack that ran DeepEP low-latency across nodes on
# this cluster on 2026-08-29 (dsv4_h20_cumulative_repro_20260829).  The images'
# NVSHMEM 3.4.5 fails IBGDA init in ibgda_create_dct ("Unable to create ah");
# NVSHMEM 3.5.21 + a DeepEP built against it (pr483 rcfix) with the gpu NIC
# handler works.  Overlay dirs must exist at OVERLAY_ROOT on both nodes.
deepep_stack="${DEEPEP_STACK:-}"
overlay_root="${OVERLAY_ROOT:-/lustre/raplab/client/xutingz/workspace/dsv4_h20_cumulative_repro_20260829}"
stack_mounts=""; stack_env=""; pythonpath_prefix=""; cache_tag="decode_waterfill"
if [[ "${deepep_stack}" == "nvshmem3521" ]]; then
  stack_mounts="-v ${overlay_root}/site_deepep_pr483_nvshmem3521_rcfix_v1:/opt/candidate/deepep:ro -v ${overlay_root}/site_nvshmem_3_5_21:/opt/candidate/nvshmem:ro"
  stack_env="-e LD_LIBRARY_PATH=/opt/candidate/nvshmem/nvidia/nvshmem/lib:/usr/local/cuda/lib64:/usr/local/lib/python3.12/dist-packages/torch/lib \
  -e LD_PRELOAD=/opt/candidate/nvshmem/nvidia/nvshmem/lib/libnvshmem_host.so.3:/usr/local/cuda/lib64/libcudart.so.13 \
  -e NVSHMEM_PLUGIN_PATH=/opt/candidate/nvshmem/nvidia/nvshmem/lib \
  -e NVSHMEM_IB_TRAFFIC_CLASS=106 -e NVSHMEM_QP_DEPTH=1024 -e NVSHMEM_ENABLE_NIC_PE_MAPPING=1 \
  -e NVSHMEM_HCA_PE_MAPPING=mlx5_3:1:1,mlx5_2:1:1,mlx5_1:1:1,mlx5_0:1:1,mlx5_5:1:1,mlx5_4:1:1,mlx5_7:1:1,mlx5_6:1:1 \
  -e NCCL_IB_TC=106 -e NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7"
  nvshmem_env="-e NVSHMEM_IB_GID_INDEX=3 -e NVSHMEM_IBGDA_NIC_HANDLER=gpu -e NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=bond0"
  pythonpath_prefix="/opt/candidate/deepep:"
  cache_tag="nvshmem3521_$(echo "${image}" | tr ':/' '__')"
fi
world_gpus=16
dp_size=16

container=""
log_pid=""
cleanup() {
  if [[ -n "${log_pid}" ]]; then kill "${log_pid}" 2>/dev/null || true; wait "${log_pid}" 2>/dev/null || true; log_pid=""; fi
  if [[ -n "${container}" ]]; then
    docker rm -f "${container}" >/dev/null 2>&1 || true
    ssh -o BatchMode=yes "${node1_ssh}" "docker rm -f ${container}_n1 >/dev/null 2>&1 || true" || true
  fi
  container=""
}
trap cleanup EXIT INT TERM

require_idle() {
  local busy
  busy="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2+0 > 2048 {print $1":"$2}')"
  [[ -z "${busy}" ]] || { echo "node0 GPUs busy: ${busy}" >&2; return 1; }
  busy="$(ssh -o BatchMode=yes "${node1_ssh}" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" | awk -F, '$2+0 > 2048 {print $1":"$2}')"
  [[ -z "${busy}" ]] || { echo "node1 GPUs busy: ${busy}" >&2; return 1; }
}

# docker run command for one node; $1=node-rank $2=container name $3=cache root
launch_cmd() {
  local node_rank="$1" name="$2" cache_root="$3"
  cat <<CMD
docker run -d --name ${name} --gpus all --privileged --cap-add=ALL --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /dev/infiniband:/dev/infiniband -v /sys/class/infiniband:/sys/class/infiniband \
  ${data_mounts} \
  -v ${cache_root}/${cache_tag}_triton_cache:/root/.triton/cache \
  -v ${cache_root}/${cache_tag}_deep_gemm_cache:/root/.cache/deep_gemm ${stack_mounts} \
  -w ${run_repo} -e PYTHONPATH=${pythonpath_prefix}${run_repo}/python -e PYTHONDONTWRITEBYTECODE=1 \
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 -e SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 ${nccl_env} ${nvshmem_env} ${stack_env} ${docker_env_str} \
  ${image} python3 -m sglang.launch_server \
    --model-path ${model} --trust-remote-code --random-seed 1 \
    --host 0.0.0.0 --port ${port} --dist-init-addr ${node0_ip}:${dist_port} --nccl-port ${nccl_port} \
    --nnodes 2 --node-rank ${node_rank} \
    --tp-size ${world_gpus} --dp-size ${dp_size} --ep-size ${world_gpus} --enable-dp-attention \
    ${a2a_args_str} \
    --ep-num-redundant-experts ${run_redundant} ${dispatch_args_str} \
    ${expert_location_args} \
    --attention-backend fa3 --mem-fraction-static ${mem_fraction} \
    --max-running-requests ${max_concurrency} --watchdog-timeout ${watchdog_timeout} \
    --max-prefill-tokens 8192 --chunked-prefill-size -1 --disable-radix-cache --skip-server-warmup \
    --cuda-graph-max-bs ${cuda_graph_max_bs} ${extra_server_args}
CMD
}

bench() {
  local out_len="$1" out_file="$2" log_file="$3"
  docker exec "${container}" python3 "${real_client}" --port "${port}" --input-ids-npy "${real_input_ids_npy}" \
    --num-seqs "${max_concurrency}" --prompt-len "${real_prompt_len}" --output-len "${out_len}" \
    --dp-size "${dp_size}" --label "${run_id}" --result-file "${out_file}" >"${log_file}" 2>&1
}

# Greedy 48-token completion of a fixed prompt, saved per run.  Redundant
# experts on a mis-wired path produce fluent-looking garbage that timing alone
# never reveals; comparing probe.txt across arms catches it before the matrix
# burns an hour.
probe() {
  local out_file="$1"
  curl -sS -m 120 "http://127.0.0.1:${port}/generate" -H 'Content-Type: application/json' \
    -d '{"text": "The capital of France is Paris. The capital of Germany is", "sampling_params": {"temperature": 0, "max_new_tokens": 48}}' \
    | python3 -c 'import json,sys; r=json.load(sys.stdin); print(repr(r["text"])); print("completion_tokens", r["meta_info"]["completion_tokens"])' \
    >"${out_file}" 2>&1 || true
  echo "probe: $(head -1 "${out_file}" | cut -c1-120)"
}

mkdir -p "${artifact_root}"
read -r -a mode_list <<<"${modes}"
docker_env_str=""
for kv in ${extra_docker_env//,/ }; do docker_env_str+=" -e ${kv}"; done   # space- or comma-separated KEY=VAL
if [[ "${moe_a2a}" == "deepep" ]]; then
  a2a_args_str="--moe-a2a-backend deepep --deepep-mode ${deepep_mode}"
else
  a2a_args_str="--moe-a2a-backend none"
fi

for run_index in "${!mode_list[@]}"; do
  mode="${mode_list[$run_index]}"
  run_repo="${repo}"; run_redundant="${redundant_experts}"; policy="${mode}"
  if [[ "${policy}" == *@* ]]; then run_redundant="${policy##*@}"; policy="${policy%@*}"; fi
  if [[ "${policy}" == base:* ]]; then run_repo="${base_repo:?BASE_REPO required}"; policy="${policy#base:}"; fi
  expert_location_args="--init-expert-location ${expert_location}"
  if [[ "${policy}" == "plain_trivial" ]]; then
    dispatch_args_str=""; expert_location_args=""
    [[ "${run_redundant}" == "0" ]] || { echo "plain_trivial needs @0" >&2; exit 1; }
  elif [[ "${policy}" == "plain_static" ]]; then
    dispatch_args_str="--ep-dispatch-algorithm static"
  elif [[ "${policy}" == *:* ]]; then
    dispatch_args_str="--ep-dispatch-algorithm ${policy%%:*} --lplb-decode-load-metric ${policy##*:}"
  else
    dispatch_args_str="--ep-dispatch-algorithm lp --lplb-decode-load-metric ${policy}"
  fi
  run_id="$(printf '%02d_%s' "${run_index}" "$(echo "${mode}" | tr ':@' '__')")"
  run_dir="${artifact_root}/${run_id}"
  port="$((port_base + run_index * 100))"; dist_port="$((port + 20))"; nccl_port="$((port + 40))"
  if [[ -s "${run_dir}/bench_$((repeats - 1)).jsonl" ]]; then echo "Skipping completed ${run_id}"; continue; fi

  require_idle
  container="sglang_${run_tag}_${run_id}"
  mkdir -p "${run_dir}"
  printf '%s\n' "${mode}" >"${run_dir}/decode_mode.txt"
  printf 'model=%s repo=%s redundant=%s policy=%s a2a=%s deepep=%s stack=%s image=%s graph_bs=%s mem=%s nodes=%s,%s\n' \
    "${model}" "${run_repo}" "${run_redundant}" "${policy}" "${moe_a2a}" "${deepep_mode}" "${deepep_stack:-image}" "${image}" "${cuda_graph_max_bs}" "${mem_fraction}" "${node0_ip}" "${node1_ssh}" >"${run_dir}/config.txt"
  (git -C "${run_repo}" rev-parse HEAD 2>/dev/null || cat "${run_repo}/BASE_COMMIT") >"${run_dir}/commit.txt" 2>/dev/null || true

  # node1 first (it only needs to reach the rendezvous), then node0
  ssh -o BatchMode=yes "${node1_ssh}" "$(launch_cmd 1 "${container}_n1" "${cache_root_node1}")" >"${run_dir}/container_id_n1.txt"
  eval "$(launch_cmd 0 "${container}" "${cache_root_node0}")" >"${run_dir}/container_id.txt"
  docker logs -f "${container}" >"${run_dir}/server.log" 2>&1 &
  log_pid="$!"
  ssh -o BatchMode=yes "${node1_ssh}" "docker logs -f ${container}_n1" >"${run_dir}/server_n1.log" 2>&1 &
  log1_pid="$!"

  ready=0
  for _ in $(seq 1 300); do
    if curl --max-time 5 -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then ready=1; break; fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${container}" 2>/dev/null)" != "true" ]]; then break; fi
    sleep 10
  done
  if [[ "${ready}" != "1" ]]; then
    docker logs "${container}" >"${run_dir}/server_failed.log" 2>&1 || true
    ssh -o BatchMode=yes "${node1_ssh}" "docker logs ${container}_n1" >"${run_dir}/server_failed_n1.log" 2>&1 || true
    kill "${log1_pid}" 2>/dev/null || true
    echo "Server failed to become ready for ${run_id}" >&2
    exit 1
  fi

  probe "${run_dir}/probe.txt"
  bench "${warmup_output_len}" "${run_dir}/warmup.jsonl" "${run_dir}/warmup.log"
  bench "${output_len}" "${run_dir}/warmup_full.jsonl" "${run_dir}/warmup_full.log"
  for rep in $(seq 0 $((repeats - 1))); do
    if [[ "${profile_steps}" -gt 0 && "${rep}" -eq 0 ]]; then
      mkdir -p "${run_dir}/profile"
      curl -sS -X POST "http://127.0.0.1:${port}/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\": \"${run_dir}/profile\", \"num_steps\": ${profile_steps}, \"activities\": [\"GPU\"], \"with_stack\": false, \"record_shapes\": false}" \
        >"${run_dir}/profile/start.log" 2>&1 || true
    fi
    bench "${output_len}" "${run_dir}/bench_${rep}.jsonl" "${run_dir}/bench_${rep}.log"
    if [[ "${profile_steps}" -gt 0 && "${rep}" -eq 0 ]]; then sleep 60; fi
  done
  kill "${log1_pid}" 2>/dev/null || true
  cleanup
  sleep 20
done
echo "all runs complete: ${artifact_root}"
