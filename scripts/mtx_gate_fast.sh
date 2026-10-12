#!/bin/bash
# 快速值门(10-12):mtx_gate_compare 的 awk 流式版 —— 同判据(col_mismatch/unordered/
# worst_rel_vs_global),单遍 C 速度(纯 python 版在亿级条目+高负载机上要 ~10min/阵)。
# 用法:scripts/mtx_gate_fast.sh <前缀>   # 比较 <前缀>_off.mtx vs <前缀>_on.mtx
set -u
p=$1
paste <(grep -v '^%' ${p}_off.mtx) <(grep -v '^%' ${p}_on.mtx) | awk '
NR==1 { next }                                  # 尺寸行(两侧同阵同 nnz,电池已核)
{
  if ($1!=$4 || $2!=$5) nbc++
  if ($1==pr && $2<=pc) un++
  pr=$1; pc=$2
  v=$3; if(v<0)v=-v; if(v>gm)gm=v
  v=$6; if(v<0)v=-v; if(v>gm)gm=v
  d=$3-$6; if(d<0)d=-d; if(d>mx)mx=d
  if($3!=$6) td++
}
END {
  wr = (gm>0)? mx/gm : 0
  printf "%s: entries_checked col_mismatch=%d unordered=%d text_diff=%d worst_rel_vs_global=%.3e\n", p, nbc, un, td, wr
  verdict = (nbc==0 && un==0 && wr<1e-9) ? "PASS" : "FAIL"
  print "=> " verdict
}' p="$p"
