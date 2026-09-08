#!/usr/bin/env bash
# runs on .25 (node0); drives .6 (node1).  usage: run_ar_variants.sh "<tag>|<extra docker -e args>" ...
B=/lustre/raplab/client/xutingz/workspace/bench/metro_review_20260904
port=29630
for spec in "$@"; do
  tag="${spec%%|*}"; extra="${spec#*|}"; [ "$extra" = "$spec" ] && extra=""
  ssh -o BatchMode=yes xutingz@10.6.131.6 "docker rm -f arbench_1 >/dev/null 2>&1; (setsid nohup bash $B/allreduce_bench.sh 1 10.6.131.25 0,1,2,3,4,5,6,7 $port $tag $extra > /tmp/ar_${tag}_n1.log 2>&1 < /dev/null &)"
  sleep 3
  S_DIR=/raid/xutingz/bench bash /raid/xutingz/bench/allreduce_bench.sh 0 10.6.131.25 0,1,2,3,4,5,6,7 $port $tag $extra 2>&1 | grep -v 'Ignore import' | grep -E '^===|^op |^world|^intra|^pair|^2level|^agather|Traceback|Error' | head -10
  port=$((port+1))
done
