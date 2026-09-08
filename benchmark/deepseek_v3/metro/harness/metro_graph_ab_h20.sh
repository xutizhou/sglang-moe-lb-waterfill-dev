#!/usr/bin/env bash
# CUDA-graph-enabled decode A/B for METRO-style replica routing on 4x H20.
#
# Differences from the September reproduction harness
# (metro_repro_20260903/harness/metro_decode_ab_h20.sh):
#   * --deepep-mode auto      : prefill keeps DeepEP normal, decode uses
#                                low_latency, which is what lets SGLang keep
#                                CUDA graphs on for decode.
#   * CUDA graphs enabled     : the reproduction ran with --disable-cuda-graph
#                                and was CPU-launch-bound (GPU compute 13-16% of
#                                the step), so GPU-side savings never surfaced.
#   * MODES is a free list    : any --lplb-decode-load-metric value, run in the
#                                given order in fresh containers, e.g.
#                                "static_global metro static static_global metro static".
#   * REPEATS measurements per container so process-level and request-level
#     noise can be separated.
#
# All arms share prefill (token-count LPLB), placement (64 redundant experts
# from the same init file), model, seed and request stream; only the decode
# replica policy differs.
set -euo pipefail

image="${IMAGE:-sha256:06e0aa8359f56ab7a316b60900e6d0dff9bfbb4190d9ce1f5c8caa27e875ae2f}"
repo="${REPO:?REPO is required}"
artifact_root="${ARTIFACT_ROOT:?ARTIFACT_ROOT is required}"
modes="${MODES:?MODES is required, e.g. 'static_global metro static'}"
port_base="${PORT_BASE:?PORT_BASE is required}"
run_tag="${RUN_TAG:-metro_graph_ab}"
output_len="${OUTPUT_LEN:-1024}"
warmup_output_len="${WARMUP_OUTPUT_LEN:-64}"
repeats="${REPEATS:-3}"
between_run_sleep="${BETWEEN_RUN_SLEEP:-15}"
redundant_experts="${REDUNDANT_EXPERTS:-64}"
max_concurrency="${MAX_CONCURRENCY:-32}"
gpu_devices="${GPU_DEVICES:-0,1,2,3}"
deepep_mode="${DEEPEP_MODE:-auto}"
cuda_graph_max_bs="${CUDA_GRAPH_MAX_BS:-16}"
mem_fraction="${MEM_FRACTION:-0.60}"
watchdog_timeout="${WATCHDOG_TIMEOUT:-1800}"
disable_cuda_graph="${DISABLE_CUDA_GRAPH:-0}"
extra_server_args="${EXTRA_SERVER_ARGS:-}"
# Space-separated KEY=VALUE pairs forwarded into the container, e.g.
# EXTRA_DOCKER_ENV="SGLANG_LPLB_IPM_TORCH_FALLBACK=1" for 1.5x replication.
extra_docker_env="${EXTRA_DOCKER_ENV:-}"
model="${MODEL:-/lustre/raplab/client/xutingz/workspace/model/DeepSeek-V3-8L-waterfill}"
expert_location="${EXPERT_LOCATION:-/lustre/raplab/client/xutingz/workspace/bench/waterfill/decode_unique_20260723/ep4_8l_logical_count.pt}"
# Host directories bind-mounted into the container: the data root that holds
# model/repo/artifacts (mounted at the same path) and the JIT caches.
data_mount="${DATA_MOUNT:-/lustre}"
triton_cache="${TRITON_CACHE:-/tmp/decode_waterfill_triton_cache}"
deep_gemm_cache="${DEEP_GEMM_CACHE:-/tmp/decode_waterfill_deep_gemm_cache}"
# BENCH_MODE=random  : sglang.benchmark.serving with random token ids (default)
# BENCH_MODE=real    : scripts/h20/real_prompt_client.py with REAL_INPUT_IDS_NPY
#                      ([seq, tokens] int array), REAL_PROMPT_LEN tokens per prompt,
#                      one batched request per DP rank pinned with routed_dp_rank.
bench_mode="${BENCH_MODE:-random}"
real_input_ids_npy="${REAL_INPUT_IDS_NPY:-}"
real_prompt_len="${REAL_PROMPT_LEN:-128}"
real_client="${REAL_CLIENT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/real_prompt_client.py}"
dp_size="${DP_SIZE:-4}"
# PROFILE_STEPS>0: after the warmups, start the Torch profiler (GPU activities,
# auto-stops after that many forward steps) around the first measured run and
# write traces to <run_dir>/profile.  Use with REPEATS=1 and a short OUTPUT_LEN.
profile_steps="${PROFILE_STEPS:-0}"
container=""
log_pid=""

cleanup() {
  if [[ -n "${log_pid}" ]]; then
    kill "${log_pid}" 2>/dev/null || true
    wait "${log_pid}" 2>/dev/null || true
    log_pid=""
  fi
  if [[ -n "${container}" ]] && docker inspect "${container}" >/dev/null 2>&1; then
    docker rm -f "${container}" >/dev/null
  fi
  container=""
}
trap cleanup EXIT INT TERM

require_idle_gpus() {
  local busy
  busy="$(
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits |
      awk -F, -v selected="${gpu_devices}" '
        BEGIN {
          count = split(selected, devices, ",")
          for (i = 1; i <= count; ++i) wanted[devices[i] + 0] = 1
        }
        wanted[$1 + 0] && $2 + 0 > 1024 {print $1 ":" $2}
      '
  )"
  if [[ -n "${busy}" ]]; then
    echo "GPUs ${gpu_devices} are not idle: ${busy}" >&2
    return 1
  fi
}

bench() {
  local out_len="$1" out_file="$2" log_file="$3"
  if [[ "${bench_mode}" == "real" ]]; then
    docker exec "${container}" \
      python3 "${real_client}" \
        --port "${port}" \
        --input-ids-npy "${real_input_ids_npy}" \
        --num-seqs "${max_concurrency}" \
        --prompt-len "${real_prompt_len}" \
        --output-len "${out_len}" \
        --dp-size "${dp_size}" \
        --label "${run_id}" \
        --result-file "${out_file}" \
        >"${log_file}" 2>&1
    return
  fi
  docker exec "${container}" \
    python3 -m sglang.benchmark.serving \
      --backend sglang \
      --host 127.0.0.1 \
      --port "${port}" \
      --model "${model}" \
      --tokenizer "${model}" \
      --dataset-name random-ids \
      --random-input-len 8 \
      --random-output-len "${out_len}" \
      --random-range-ratio 1.0 \
      --num-prompts "${max_concurrency}" \
      --request-rate inf \
      --max-concurrency "${max_concurrency}" \
      --warmup-requests 0 \
      --seed 1 \
      --disable-tqdm \
      --disable-stream \
      --tokenize-prompt \
      --output-details \
      --output-file "${out_file}" \
      >"${log_file}" 2>&1
}

mkdir -p "${artifact_root}"
read -r -a mode_list <<<"${modes}"
graph_args=()
if [[ "${disable_cuda_graph}" == "1" ]]; then
  graph_args=(--disable-cuda-graph)
else
  graph_args=(--cuda-graph-max-bs "${cuda_graph_max_bs}")
fi
read -r -a extra_args <<<"${extra_server_args}"
docker_env_args=()
for kv in ${extra_docker_env}; do
  docker_env_args+=(-e "${kv}")
done

for run_index in "${!mode_list[@]}"; do
  mode="${mode_list[$run_index]}"
  # plain_static: stock SGLang EPLB dispatch (--ep-dispatch-algorithm static) with
  # no LPLB flag at all, i.e. no load-balancing code in prefill or decode.  This
  # is what a production EPLB deployment runs and the reference "no balancing"
  # arm.  Every other mode is prefill token-LPLB (lp) + the named decode metric.
  # Mode syntax: [base:]<policy>[@<redundant_experts>]
  #   policy "<metric>"        = lp prefill + decode metric (historical form)
  #          "plain_static"    = stock static dispatch, no LPLB flag at all
  #          "<algo>:<metric>" = explicit, e.g. "static:metro" = stock static
  #                              prefill + METRO decode (decode-only change)
  #   "base:" prefix runs the untouched upstream checkout in BASE_REPO instead
  #   of REPO (the code before any METRO/decode changes); "@R" overrides the
  #   redundant-expert count for this arm only.
  run_repo="${repo}"
  run_redundant="${redundant_experts}"
  policy="${mode}"
  if [[ "${policy}" == *@* ]]; then
    run_redundant="${policy##*@}"
    policy="${policy%@*}"
  fi
  if [[ "${policy}" == base:* ]]; then
    run_repo="${BASE_REPO:?BASE_REPO is required for base: modes}"
    policy="${policy#base:}"
  fi
  if [[ "${policy}" == "plain_static" ]]; then
    dispatch_args=(--ep-dispatch-algorithm static)
  elif [[ "${policy}" == *:* ]]; then
    dispatch_args=(--ep-dispatch-algorithm "${policy%%:*}" --lplb-decode-load-metric "${policy##*:}")
  else
    dispatch_args=(--ep-dispatch-algorithm lp --lplb-decode-load-metric "${policy}")
  fi
  run_id="$(printf '%02d_%s' "${run_index}" "$(echo "${mode}" | tr ':@' '__')")"
  run_dir="${artifact_root}/${run_id}"
  port="$((port_base + run_index * 100))"
  dist_port="$((port + 20))"
  nccl_port="$((port + 40))"
  if [[ -s "${run_dir}/bench_$((repeats - 1)).jsonl" ]]; then
    echo "Skipping completed ${run_id}"
    continue
  fi

  require_idle_gpus
  container="sglang_${run_tag}_${run_id}"
  mkdir -p "${run_dir}"
  if docker inspect "${container}" >/dev/null 2>&1; then
    echo "Refusing to reuse existing container ${container}" >&2
    exit 1
  fi

  printf '%s\n' "${mode}" >"${run_dir}/decode_mode.txt"
  printf 'model=%s deepep_mode=%s cuda_graph=%s max_bs=%s redundant=%s dp_size=%s mem_fraction=%s bench_mode=%s real_npy=%s prompt_len=%s docker_env=%s\n' \
    "${model}" "${deepep_mode}" "$((1 - disable_cuda_graph))" "${cuda_graph_max_bs}" "${redundant_experts}" "${dp_size}" "${mem_fraction}" \
    "${bench_mode}" "${real_input_ids_npy}" "${real_prompt_len}" "${extra_docker_env}" >"${run_dir}/config.txt"
  nvidia-smi \
    --query-gpu=index,name,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader >"${run_dir}/gpu_before.csv"
  git -C "${run_repo}" status --short >"${run_dir}/git_status.txt" 2>&1 || true
  git -C "${run_repo}" rev-parse HEAD >"${run_dir}/commit.txt" 2>&1 || cat "${run_repo}/BASE_COMMIT" >"${run_dir}/commit.txt" 2>/dev/null || true
  git -C "${run_repo}" diff >"${run_dir}/source.diff" 2>&1 || true
  printf 'repo=%s redundant=%s policy=%s\n' "${run_repo}" "${run_redundant}" "${policy}" >>"${run_dir}/config.txt"

  docker run -d \
    --name "${container}" \
    --gpus "\"device=${gpu_devices}\"" \
    --ipc=host \
    --network=host \
    -v "${data_mount}:${data_mount}" \
    -v "${triton_cache}:/root/.triton/cache" \
    -v "${deep_gemm_cache}:/root/.cache/deep_gemm" \
    -w "${run_repo}" \
    -e "PYTHONPATH=${run_repo}/python" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    -e SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 \
    -e SGLANG_JIT_DEEPGEMM_PRECOMPILE=0 \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    "${docker_env_args[@]}" \
    "${image}" \
    python3 -m sglang.launch_server \
      --model-path "${model}" \
      --trust-remote-code \
      --random-seed 1 \
      --host 127.0.0.1 \
      --port "${port}" \
      --dist-init-addr "127.0.0.1:${dist_port}" \
      --nccl-port "${nccl_port}" \
      --tp-size "${dp_size}" \
      --dp-size "${dp_size}" \
      --ep-size "${dp_size}" \
      --enable-dp-attention \
      --moe-a2a-backend deepep \
      --deepep-mode "${deepep_mode}" \
      --ep-num-redundant-experts "${run_redundant}" \
      "${dispatch_args[@]}" \
      --init-expert-location "${expert_location}" \
      --attention-backend fa3 \
      --mem-fraction-static "${mem_fraction}" \
      --max-running-requests "${max_concurrency}" \
      --watchdog-timeout "${watchdog_timeout}" \
      --max-prefill-tokens 8192 \
      --chunked-prefill-size -1 \
      --disable-radix-cache \
      --skip-server-warmup \
      "${graph_args[@]}" \
      "${extra_args[@]}" \
      >"${run_dir}/container_id.txt"

  docker logs -f "${container}" >"${run_dir}/server.log" 2>&1 &
  log_pid="$!"

  ready=0
  for _ in $(seq 1 240); do
    if curl --max-time 5 -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null; then
      ready=1
      break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "${container}")" != "true" ]]; then
      break
    fi
    sleep 10
  done
  if [[ "${ready}" != "1" ]]; then
    docker logs "${container}" >"${run_dir}/server_failed.log" 2>&1 || true
    echo "Server failed to become ready for ${run_id}" >&2
    exit 1
  fi

  bench "${warmup_output_len}" "${run_dir}/warmup.jsonl" "${run_dir}/warmup.log"
  # Second warmup at the measured length so every grouped-GEMM / graph shape
  # is compiled before timing starts.
  bench "${output_len}" "${run_dir}/warmup_full.jsonl" "${run_dir}/warmup_full.log"
  for rep in $(seq 0 $((repeats - 1))); do
    if [[ "${profile_steps}" -gt 0 && "${rep}" -eq 0 ]]; then
      mkdir -p "${run_dir}/profile"
      curl -sS -X POST "http://127.0.0.1:${port}/start_profile" -H 'Content-Type: application/json' \
        -d "{\"output_dir\": \"${run_dir}/profile\", \"num_steps\": ${profile_steps}, \"activities\": [\"GPU\"], \"with_stack\": false, \"record_shapes\": false}" \
        >"${run_dir}/profile/start.log" 2>&1 || true
    fi
    bench "${output_len}" "${run_dir}/bench_${rep}.jsonl" "${run_dir}/bench_${rep}.log"
    if [[ "${profile_steps}" -gt 0 && "${rep}" -eq 0 ]]; then
      # num_steps auto-stops; wait for every rank to flush its trace.
      for _ in $(seq 1 60); do
        n=$(ls "${run_dir}/profile"/*.trace.json* 2>/dev/null | wc -l)
        [[ "${n}" -ge "${dp_size}" ]] && break
        sleep 5
      done
      sleep 10
    fi
  done

  nvidia-smi \
    --query-gpu=index,name,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory \
    --format=csv,noheader >"${run_dir}/gpu_after.csv"
  cleanup
  sleep "${between_run_sleep}"
done
echo "all runs complete: ${artifact_root}"
