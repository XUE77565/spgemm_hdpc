#!/bin/bash
# nd24k dense 位图经济屏(study/26 §5)——三臂 ×3 交错配对
# A=强制 PB2_CURSOR=0(全 dense/search 位图) B=强制 PB2_CURSOR=2(全 cursor) C=默认路由
# 记分=dense_count+dense_direct 相位和(排 h2d/d2h/TOTAL(GPU));nnz 对账=189,190,604
# 口径:末值;禁跨迭代求和。GPU0 only。
set -u
cd /home/xueyizhou/spgemm-challenge
MTX=data/ocean/square/nd24k.mtx
BIN=./spgemm_ndd
EXP_NNZ=189190604
FORCED="DITER_MIN_FLOP=1 DITER_EST_MIN=1 SPANF=100000 OCCGATE_COEFF=1000000 LADDER=0"
STAMP=$(date +%H%M)
OUTDIR=/tmp/nd24k_econ_$STAMP
mkdir -p $OUTDIR
echo "== nd24k econ screen $STAMP (GPU0) ==" | tee $OUTDIR/summary.log

# ── 净窗检查:6×30s GPU0 无他人进程 ──
clean=1
for i in 1 2 3 4 5 6; do
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$procs" -ne 0 ] && { echo "round$i: GPU 有 $procs 个计算进程,非净窗" | tee -a $OUTDIR/summary.log; clean=0; }
  sleep 30
done
if [ $clean -eq 0 ]; then echo "ABORT: 非净窗" | tee -a $OUTDIR/summary.log; exit 1; fi
echo "净窗 ✓ (6×30s)" | tee -a $OUTDIR/summary.log

run_arm() {  # $1=label $2=extra env string
  local label=$1 round=$2 envs=$3
  local base=$OUTDIR/${label}_r${round}
  env CUDA_VISIBLE_DEVICES=0 METHOD=hash $envs $BIN $MTX > ${base}.out 2> ${base}.err
  local rc=$?
  local nnz=$(grep -oE "nnz = [0-9]+" ${base}.out | tail -1 | grep -oE "[0-9]+")
  local nnzflag="OK"; [ "$nnz" != "$EXP_NNZ" ] && nnzflag="MISMATCH($nnz)"
  [ $rc -ne 0 ] && nnzflag="RC=$rc $nnzflag"
  # 相位提取(末值):stderr 行格式 [dbg <ms>][hash-prof] <phase> <val> ms
  local dc=$(grep -oE "\[hash-prof\] dense_count +[0-9.]+" ${base}.err | tail -1 | grep -oE "[0-9.]+$")
  local dd=$(grep -oE "\[hash-prof\] dense_direct +[0-9.]+" ${base}.err | tail -1 | grep -oE "[0-9.]+$")
  local acc=$(grep -oE "\[hash-prof\] accumulate +[0-9.]+" ${base}.err | tail -1 | grep -oE "[0-9.]+$")
  local cnt=$(grep -oE "\[hash-prof\] count_flop +[0-9.]+" ${base}.err | tail -1 | grep -oE "[0-9.]+$")
  local tot=$(grep -oE "\[hash-prof\] TOTAL\(GPU\) +[0-9.]+" ${base}.err | tail -1 | grep -oE "[0-9.]+$")
  local dsum=$(python3 -c "d=[${dc:-0},${dd:-0}];print(f'{sum(d):.3f}')")
  echo "$label $round nnz=$nnzflag dense_count=${dc:-NA} dense_direct=${dd:-NA} dense和=$dsum accumulate=${acc:-NA} count_flop=${cnt:-NA} TOTAL=${tot:-NA}" | tee -a $OUTDIR/summary.log
}

for round in 1 2 3; do
  echo "── ROUND $round ──" | tee -a $OUTDIR/summary.log
  run_arm dense0 r$round "$FORCED PB2_CURSOR=0"
  run_arm cursor2 r$round "$FORCED PB2_CURSOR=2"
  run_arm default r$round ""
  sleep 5
done

# ── ocean 同日配对 ×3 ──
echo "── ocean ×3 ──" | tee -a $OUTDIR/summary.log
for round in 1 2 3; do
  env CUDA_VISIBLE_DEVICES=0 ./ocean/spgemm /tmp/oc_nd24k.csr ocean/config/bench.json > $OUTDIR/ocean_r${round}.out 2>&1
  grep -E "Average SpGEMM time|illegal|error" $OUTDIR/ocean_r${round}.out | tail -2 | tee -a $OUTDIR/summary.log
  sleep 5
done
echo "== DONE $STAMP ==" | tee -a $OUTDIR/summary.log
