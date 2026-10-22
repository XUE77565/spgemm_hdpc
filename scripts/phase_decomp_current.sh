#!/usr/bin/env bash
# tick5:现役配置(生产 A 臂,电池 env)全相位分解 12 代表阵 → 对 ocean 逐相位找今日差距
set -u
cd "$(dirname "$0")/.."
OUT=/tmp/phase_now; mkdir -p "$OUT"
for m in CurlCurl_4 rgg_n_2_22_s0 af_shell10 af_1_k101 StocF-1465 nlpkkt120 Bump_2911 Geo_1438 Serena Hook_1498 cage15 nlpkkt80; do
  CUDA_VISIBLE_DEVICES=0 MP_HOST_MB=8192 USE_MEMPOOL=1 METHOD=adaptive DBG=1 HASH_LHT23=0 ./spgemm_test "data/ocean/square/$m.mtx" > "$OUT/$m.log" 2>&1
  echo "done $m"
done