#!/usr/bin/env bash
# Sequential EP16 DeepEP-LL variant cells (one ART per variant so the same
# mode can be measured under different env).  Uses the node watcher.
set -u
cd /raid/xutingz/bench
run() {  # ART CELLS_STR
  rm -f "$1.STOP"
  ART="$1" CELLS_STR="$2" ./ep16_watch_deepep.sh >> "ep16_variants.log" 2>&1
  echo "[$(date '+%m-%d %T')] finished $1 (ALLDONE=$(ls $1.ALLDONE 2>/dev/null | wc -l) STOP=$(ls $1.STOP 2>/dev/null | wc -l))" >> ep16_variants.log
}
S0="EXTRA_DOCKER_ENV=SGLANG_METRO_COUNT_MODE=stale PROFILE_STEPS=30"
S3="EXTRA_DOCKER_ENV=SGLANG_METRO_COUNT_MODE=stale,SGLANG_METRO_STALE_INACTIVE_WEIGHT=3 PROFILE_STEPS=30"
run ep16_deepep_stale   "static:metro@128|24900|${S0}"
run ep16_deepep_stale3  "static:metro@128|24910|${S3}"
run ep16_deepep_v2sync  "static:metro@128|24920|PROFILE_STEPS=30"
run ep16_deepep_stale   "static:metro@128|24900|${S0};static:metro@64|24940|${S0}"
run ep16_deepep_stale3  "static:metro@128|24910|${S3};static:metro@64|24950|${S3}"
run ep16_deepep_v2sync  "static:metro@128|24920|PROFILE_STEPS=30;static:metro@64|24960|PROFILE_STEPS=30"
echo "[$(date '+%m-%d %T')] VARIANTS_DONE" >> ep16_variants.log
