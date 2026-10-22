#!/usr/bin/env bash
# 刀C1 A/B:COMPACT_COPY_V2(4路独立装载展开)——compact+sort 相位同会话交错
set -u
cd "$(dirname "$0")/.."
OUT=/tmp/knifeC1; mkdir -p "$OUT"
run1() {  # $1=阵 $2=arm(0/1) $3=tag
  CUDA_VISIBLE_DEVICES=0 MP_HOST_MB=8192 USE_MEMPOOL=1 METHOD=adaptive DBG=1 HASH_LHT23=0 COMPACT_COPY_V2=$2 \
    ./spgemm_test "data/ocean/square/$1.mtx" > "$OUT/$1__$3.log" 2>&1
  cs=$(grep -oE '\[hash-prof\] compact\+sort\s+[0-9.]+' "$OUT/$1__$3.log" | grep -oE '[0-9.]+' | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:a[NR/2]}')
  tot=$(grep -oE '\[hash-prof\] TOTAL\(GPU\)\s+[0-9.]+' "$OUT/$1__$3.log" | grep -oE '[0-9.]+' | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:a[NR/2]}')
  nnz=$(grep -oE 'C_nnz=[0-9]+' "$OUT/$1__$3.log" | tail -1 | grep -oE '[0-9]+')
  printf "%-16s %s compact=%-8s TOTAL=%-9s nnz=%s\n" "$1" "$3" "${cs:-NA}" "${tot:-NA}" "${nnz:-NA}"
}
echo "== 靶阵(交错 A,B,A,B)=="
for m in cage15 nlpkkt120; do run1 "$m" 0 A0; run1 "$m" 1 B0; run1 "$m" 0 A1; run1 "$m" 1 B1; done
echo "== 控制阵(A,B 单轮)=="
for m in StocF-1465 rgg_n_2_22_s0 c-64 F2; do run1 "$m" 0 A0; run1 "$m" 1 B0; done