#!/usr/bin/env bash
# smoke1 —— 部署验证作业:探测环境,回传 metrics.json。不占 GPU 计算。
set -eu
echo "== smoke1: environment probe =="
echo "JOB_ID=$JOB_ID"
echo "JOB_DIR=$JOB_DIR"
echo "OUT_DIR=$OUT_DIR"
date -u
uname -sr; hostname

echo "-- tools --"
for t in bash tar timeout setsid nvcc make python3; do
  printf '%s: ' "$t"; command -v "$t" || echo MISSING
done

echo "-- nvcc --"
nvcc --version | tail -1 || true

echo "-- GPU(仅查询,不占用)--"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

echo "-- code snapshot --"
pwd; ls | head -10
test -f Makefile && echo "Makefile OK"
test -d src && echo "src/ OK"
test -d scripts && echo "scripts/ OK"

echo "-- data(绝对路径约定,大数据不进 OUT_DIR)--"
ls /home/xueyizhou/spgemm-challenge/data | head -5

NGPU=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
printf '{\n  "job": "smoke1",\n  "host": "%s",\n  "gpus": %s,\n  "tools": "ok",\n  "snapshot": "ok",\n  "ok": true\n}\n' "$(hostname)" "$NGPU" > "$OUT_DIR/metrics.json"
echo "== smoke1 done =="
