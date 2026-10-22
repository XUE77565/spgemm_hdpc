#!/usr/bin/env bash
# 刀A探针(study/57 tick3):HASH_LHT23 bins2-3 常量折叠扩展,同会话交错 A/B
# A=HASH_LHT23=0(现行为) B=1(LHT 折叠);nnz 门先于计时;控制阵单轮。
set -u
cd "$(dirname "$0")/.."
BIN=./spgemm_test
OUT=/tmp/knifeA; mkdir -p "$OUT"
run1() {  # $1=阵 $2=arm $3=tag
  CUDA_VISIBLE_DEVICES=0 MP_HOST_MB=8192 USE_MEMPOOL=1 METHOD=adaptive DBG=1 HASH_LHT23=$2 "$BIN" "data/ocean/square/$1.mtx" > "$OUT/$1__$3.log" 2>&1
  acc=$(grep -oE '\[hash-prof\] accumulate\s+[0-9.]+' "$OUT/$1__$3.log" | grep -oE '[0-9.]+' | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:a[NR/2]}')
  nnz=$(grep -oE 'C_nnz=[0-9]+' "$OUT/$1__$3.log" | tail -1 | grep -oE '[0-9]+')
  printf "%-18s %s acc_med=%-9s nnz=%s\n" "$1" "$3" "${acc:-NA}" "${nnz:-NA}"
}
echo "== 靶阵(交错 A,B,A,B ×2)=="
for m in CurlCurl_4 rgg_n_2_22_s0 af_shell10 StocF-1465 af_1_k101; do
  run1 "$m" 0 A0; run1 "$m" 1 B0; run1 "$m" 0 A1; run1 "$m" 1 B1
done
echo "== 控制阵(单轮 A/B;bins2-3 零人口或不走梯)=="
for m in ldoor cage15 c-64 F2; do
  run1 "$m" 0 A0; run1 "$m" 1 B0
done