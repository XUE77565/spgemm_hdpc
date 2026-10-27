#!/usr/bin/env bash
# Phase 0(study/57 §五十一):compute_61 广谱影响审计
#   sm90 二进制 vs c61 二进制,同会话交错×2,15 阵全路由人群
#   指标:TOTAL(GPU) 中位 / nnz / binning 相位(垫片税专项)
set -u
cd "$(dirname "$0")/.."
OUT=/tmp/phase0; mkdir -p $OUT
run1() {  # $1=阵路径 $2=tag $3=bin
  CUDA_VISIBLE_DEVICES=0 MP_HOST_MB=8192 USE_MEMPOOL=1 METHOD=adaptive DBG=1 HASH_LHT23=0 "$3" "$1" > "$OUT/$(basename $1 .mtx)__$2.log" 2>&1
  tot=$(grep -oE '\[hash-prof\] TOTAL\(GPU\)\s+[0-9.]+|\[prof\] TOTAL\(GPU\)\s+[0-9.]+' "$OUT/$(basename $1 .mtx)__$2.log" | grep -oE '[0-9.]+' | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:a[NR/2]}')
  bin_=$(grep -oE '\[hash-prof\] binning\s+[0-9.]+' "$OUT/$(basename $1 .mtx)__$2.log" | grep -oE '[0-9.]+' | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:a[NR/2]}')
  nnz=$(grep -oE 'C_nnz=[0-9]+' "$OUT/$(basename $1 .mtx)__$2.log" | tail -1 | grep -oE '[0-9]+')
  printf "%s %s TOTAL=%-9s binning=%-7s nnz=%s\n" "$(basename $1 .mtx)" "$2" "${tot:-NA}" "${bin_:-NA}" "${nnz:-NA}"
}
MATS_SQ="Bump_2911 cage15 bloweybq case9 TSOPF_RS_b678_c2 pre2 c-64 F2 StocF-1465 rgg_n_2_22_s0 web-Google ldoor"
for m in $MATS_SQ; do
  [ -f "data/ocean/square/$m.mtx" ] || { echo "$m 缺文件跳过"; continue; }
  for r in 1 2; do
    run1 "data/ocean/square/$m.mtx" "sm90_r$r" ./spgemm_test_sm90
    run1 "data/ocean/square/$m.mtx" "c61_r$r"  ./spgemm_test_c61
  done
done
for m in 1138_bus bcsstk17 arc130; do
  [ -f "data/first100/$m.mtx" ] || { echo "$m 缺文件跳过"; continue; }
  for r in 1 2; do
    run1 "data/first100/$m.mtx" "sm90_r$r" ./spgemm_test_sm90
    run1 "data/first100/$m.mtx" "c61_r$r"  ./spgemm_test_c61
  done
done
echo "=== 电池完成 ==="