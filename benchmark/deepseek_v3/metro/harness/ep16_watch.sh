#!/usr/bin/env bash
# Poll node1 candidates until one has all 8 GPUs idle, then run the EP16 cells
# one at a time.  Every cell is gated (probe text coherent, 3 bench runs
# present, metro must beat same-R static) so a broken configuration stops the
# sweep instead of burning an hour.  A node that becomes busy between cells
# just sends us back to polling.
set -u
cd /raid/xutingz/bench
CANDS="${CANDS:-xutingz@10.6.131.6 xutingz@10.6.131.5}"
ART="${ART:-ep16_none_v2}"
POLL="${POLL:-20}"
IMAGE_TAG=lmsysorg/sglang:metro-repro   # node1 must have this tag
# cell = mode|port|extra-env ; override with CELLS_STR="a|p|e;b|p|e"
DEFAULT_CELLS=(
  "static:metro@64|24500|PROFILE_STEPS=30"
  "base:plain_static@0|24520|PROFILE_STEPS=30"
  "base:plain_static@128|24540|PROFILE_STEPS=30"
  "static:metro@128|24560|PROFILE_STEPS=30"
  "base:plain_static@32|24580|PROFILE_STEPS=30"
  "static:metro@32|24600|PROFILE_STEPS=30"
)
if [[ -n "${CELLS_STR:-}" ]]; then IFS=';' read -r -a CELLS <<<"${CELLS_STR}"; else CELLS=("${DEFAULT_CELLS[@]}"); fi
log() { echo "[$(date '+%m-%d %T')] $*"; }
stop() { log "STOP: $*"; echo "$*" > "${ART}.STOP"; exit 2; }

dirname_of() { local m="${1//:/_}"; echo "00_${m//@/_}"; }   # harness: 00_<policy with : -> _>_<R>
median_s() {  # median of bench_*.jsonl per-rank-max seconds
  python3 - "$1" <<'PY'
import glob, json, statistics, sys
vals = []
for f in sorted(glob.glob(sys.argv[1] + "/bench_*.jsonl")):
    for line in open(f):
        r = json.loads(line); vals.append(max(p["seconds"] for p in r["per_rank"]))
print(f"{statistics.median(vals):.3f}" if vals else "nan")
PY
}
node_free() {  # all 8 GPUs < 2048 MiB and image present
  local c="$1" out n busy
  out="$(timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=5 "$c" \
    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; docker image inspect ${IMAGE_TAG} >/dev/null 2>&1 && echo IMG=ok || echo IMG=missing" 2>/dev/null)" || return 1
  n="$(grep -vc IMG <<<"$out")"; busy="$(grep -v IMG <<<"$out" | awk '$1+0>2048' | wc -l)"
  [[ "$n" -eq 8 && "$busy" -eq 0 && "$out" == *"IMG=ok"* ]]
}
check_cell() {  # $1=mode $2=run_dir -> 0 ok, 1 busy (retry later), 2 fatal
  local mode="$1" d="$2"
  if grep -q "GPUs busy" "${d}.harness.log" 2>/dev/null; then return 1; fi
  [[ -s "$d/bench_2.jsonl" ]] || { log "cell $mode: bench incomplete (see ${d}.harness.log, $d/server_failed*.log)"; return 2; }
  grep -q "Berlin" "$d/probe.txt" || { log "cell $mode: PROBE TEXT WRONG: $(head -1 "$d/probe.txt" | cut -c1-100)"; return 2; }
  local med; med="$(median_s "$d")"; log "cell $mode: median ${med} s  probe ok"
  echo "$mode $med" >> "${ART}.results.txt"
  if [[ "$mode" == static:metro@* ]]; then
    local r base_d base_med
    r="${mode##*@}"; base_d="${ART}/00_base_plain_static_${r}"
    if [[ -d "$base_d" ]]; then
      base_med="$(median_s "$base_d")"
      python3 -c "import sys; sys.exit(0 if float('$med') < float('$base_med') else 1)" \
        || { log "cell $mode ${med} s is NOT faster than plain_static@${r} ${base_med} s"; return 2; }
      log "gate ok: ${mode} ${med} s < plain_static@${r} ${base_med} s"
    fi
  fi
  return 0
}

log "watcher start: cells=${#CELLS[@]} candidates=[${CANDS}] artifact=${ART}"
i=0; sleep_next=0
while [[ $i -lt ${#CELLS[@]} ]]; do
  IFS='|' read -r mode port extra <<<"${CELLS[$i]}"
  d="${ART}/$(dirname_of "$mode")"
  if [[ -s "$d/bench_2.jsonl" ]]; then log "skip completed $mode"; i=$((i+1)); continue; fi
  node=""; while [[ -z "$node" ]]; do
    for c in $CANDS; do node_free "$c" && { node="$c"; break; }; done
    [[ -n "$node" ]] || sleep "$POLL"
  done
  log "node1=${node} free -> running ${mode} (port ${port})"
  rm -rf "$d"; mkdir -p "${ART}"
  env NODE1_SSH="$node" MOE_A2A=none MODES="$mode" ARTIFACT_NAME="$ART" PORT_BASE="$port" \
      RUN_TAG="${ART##*_}w$((i+1))" REPEATS=3 OUTPUT_LEN=1024 ${extra} ./run_ep16.sh > "${d}.harness.log" 2>&1
  check_cell "$mode" "$d"; rc=$?
  case $rc in
    0) i=$((i+1)) ;;
    1) log "node1 ${node} became busy; back to polling" ;;
    2) stop "cell ${mode} failed sanity gate" ;;
  esac
done
log "ALL CELLS DONE"; python3 summarize_graph_ab.py "$ART" 2>/dev/null | tail -12
echo done > "${ART}.ALLDONE"
