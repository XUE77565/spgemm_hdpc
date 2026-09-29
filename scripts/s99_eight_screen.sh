#!/bin/bash
# spec99 八阵屏(翻默认前置):默认路由,SPEC99=1 vs 0 ×3 交错
# 记分=accumulate 相位+净TOTAL(排 h2d/d2h);门=每阵两臂 nnz 互对
set -u
cd /home/xueyizhou/spgemm-challenge
BIN=./spgemm_ndd
STAMP=$(date +%H%M)
OUTDIR=/tmp/s99_screen_$STAMP
mkdir -p $OUTDIR
echo "== spec99 八阵屏 $STAMP(默认路由,GPU0)==" | tee $OUTDIR/summary.log

# 净窗快查(2×15s;完整 A 级以首尾哨兵+ECC 为准)
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | { read n; [ "$n" = "0" ] && echo "GPU 净 ✓" || echo "⚠ 有 $n 进程,继续但注意"; } | tee -a $OUTDIR/summary.log

MATS="F2 crankseg_2 in-2004 pre2 msc23052 c-64 nd24k wb-edu"
for m in $MATS; do
  echo "── $m ──" | tee -a $OUTDIR/summary.log
  for r in 1 2 3; do
    for arm in on off; do
      E=""
      [ "$arm" = "on" ] && E="SPEC99=1"
      env CUDA_VISIBLE_DEVICES=0 METHOD=hash $E $BIN data/ocean/square/${m}.mtx > $OUTDIR/${m}_${arm}_r${r}.out 2> $OUTDIR/${m}_${arm}_r${r}.err
      rc=$?
      [ $rc -ne 0 ] && echo "$m $arm r$r RC=$rc ⚠" | tee -a $OUTDIR/summary.log
    done
  done
  python3 - "$OUTDIR" "$m" <<'EOF'
import sys, re
outdir, m = sys.argv[1], sys.argv[2]
def get(path):
    d = {}
    for l in open(path, errors="replace"):
        mm = re.search(r"\[hash-prof\]\s+(\S+)\s+([0-9.]+)\s+ ms", l)
        if mm: d[mm.group(1)] = float(mm.group(2))
    nnz = re.findall(r"nnz = (\d+)", open(path.replace(".err",".out"), errors="replace").read())
    return d, (nnz[-1] if nnz else "?")
ons, offs, ratios, nnz_ok = [], [], [], True
for r in [1,2,3]:
    don, non = get(f"{outdir}/{m}_on_r{r}.err")
    doff, noff = get(f"{outdir}/{m}_off_r{r}.err")
    if non != noff: nnz_ok = False
    a_on, a_off = don.get("accumulate"), doff.get("accumulate")
    t_on = don.get("TOTAL(GPU)",0)-don.get("h2d",0)-don.get("d2h",0)
    t_off = doff.get("TOTAL(GPU)",0)-doff.get("h2d",0)-doff.get("d2h",0)
    if a_off and a_on: ratios.append(a_on/a_off)
    ons.append((a_on, t_on)); offs.append((a_off, t_off))
import statistics
if ratios:
    med = statistics.median(ratios)
    print(f"{m}: acc比 中位={med:.3f} 同号={all(x<1 for x in ratios) if med<1 else all(x>1 for x in ratios)} nnz一致={'✓' if nnz_ok else '✗ MISMATCH'} on净TOTAL中位={statistics.median([t for _,t in ons]):.2f} off={statistics.median([t for _,t in offs]):.2f}", flush=True)
else:
    print(f"{m}: 无相位数据(nnz一致={'✓' if nnz_ok else '✗'})", flush=True)
EOF
done | tee -a $OUTDIR/summary.log
nvidia-smi --query-gpu=index,ecc.errors.uncorrected.volatile.total --format=csv,noheader | tee -a $OUTDIR/summary.log
echo "== DONE $STAMP ==" | tee -a $OUTDIR/summary.log
