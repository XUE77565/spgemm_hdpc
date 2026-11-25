#!/usr/bin/env bash
# bm2_r4 —— 刀三·想法二第四轮(study/69 §二十八):两遍式改成独立核 + 行表划分;只评测与 env 探索,不转正
#
# 与 bm2_r3 的关系:r3 的值级全绿、训练池逐行标定说位图形态下两遍式该赢(R≈0.76-0.80),但电池上"只开位图形态"
#   只是持平(GM 1.004-1.010),强制发射 BM2 实例而不收行则 acc +5-8%——两遍式内联进 hash_heavy 实例会拖累同实例的
#   哈希行,且内环码生成随宿主实例漂移;行门前置每 run 的设备分配+6 次发射在小阵上是 co 的 2-7%。本轮(v3):
#   ① 两遍式 = 独立核 hash_bm2_kernel,吃划分核分出的"合格行表";其余行由**原样的** hash_heavy 实例处理
#      (hash_heavy_kernel 与其发射宏的源码回到与基线逐字相同);
#   ② 划分核单次发射、静态缓冲(计时轮零分配),顺带出每 bin 合格行的 SMEM 需求上界,两遍式发射按需给 SMEM;
#   ③ Pass1 列来源可选(HH_BM2_P1=1:读 B16 列字段,与 Pass2 同一条流);行计时补收尾栅栏;
#   ④ legacy 形态默认不开(HH_BM2_L1=1 才开)。
# 行门标定只用 data/train_disp;电池(428 主账的阵)只评测。门值不入源码常数,默认仍关。
#
# 卡:钉 GPU 0(CUDA_VISIBLE_DEVICES=0)。仅当开跑时 0 上已有别的计算进程而 1 空闲,才改钉 1;
#     两张都忙则等(最多 30 分钟)。实际卡号写在 OUT_DIR/SUMMARY.md 的 S0 段。计时臂同卡同批、逐阵逐轮交错。
# 阶段(S1-S4 任一红,后面不跑):
#   S0 环境、选卡
#   S1 微复现器      行例程加了 Pass1 来源模板参(头文件 sha256 变了)→ 重跑 hh_repro5,两种子
#   S2 构建          make TARGET=spgemm_bm2(审计双绿)+ WRITE_MTX 件
#   S3 冒烟          小阵 off/m1/m2/m2p/m2L 各 1 发:nnz + 普查
#   S4a 值级·对照组  bcsstk17/af23560/cfd2/msc10848/s3dkq4m2:off/m1/m2/m2p 四臂官方三判 + l3_diag + 交叉标定
#   S4b 值级         TSOPF(差分门 + l3_diag)、ldoor(差分门;legacy 形态 m1L/m2L)、bin9 代表阵(探查后取前两个)
#   S5 行门标定      训练池逐行配对周期(off/m1/m2/m2p,两种形态都收)→ bm2_gate_cal.py ×2(Pass1 两种来源各一次)
#   S6 只评测电池    off / m2(位图形态,不设门)/ m2p(同上+Pass1 读 B16)/ m2L(再加 legacy 形态)/ auto(标定结果,与前面某臂相同则不重复),5 轮
#   S7 无侵入电池    基线源码件 spgemm_base 对 spgemm_bm2(开关关),官方 ab_pair.py 16×3
#   S8 值级·大阵     pre2(m2L)差分门
# 回传:OUT_DIR/SUMMARY.md(先看这个)+ 各阶段 s*_ 文件。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-bm2_r4}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea=675b227 形态)
HDR_SHA=25efd6afa6dbd60879d99d5f8dfb2f38d68f3798f7c04c60fc138e11787d4184                                                         # bm2_r1 S1 验过的 src/hh_bm2.cuh
mkdir -p "$OUT" "$JD/vl" "$JD/keep" "$JD/cal" "$JD/calp"
: > "$SUM"
say() { echo "$*" | tee -a "$SUM"; }
TS0=$(date +%s); TSL=$TS0
stage() { local now; now=$(date +%s); echo; echo "################ $* ################ (+$((now - TSL))s 上一段,累计 $((now - TS0))s)"; TSL=$now; date -u +%FT%TZ; }
clean_dumps() { rm -rf "$ROOT"/results/matrices/*/self_product*.mtx "$JD/vl" "$JD/cal" "$JD/calp" 2>/dev/null; }
trap 'clean_dumps; echo "收到终止信号,已清转储"; exit 143' TERM INT
finish() {   # $1=退出码 $2=结论一句话
  clean_dumps
  say ""; say "## 结束"; say "- 结论: $2"; say "- 磁盘: $(df -h "$JD" | tail -1)"
  say "- 结束时间: $(date -u +%FT%TZ),总用时 $(( ($(date +%s) - TS0) / 60 )) 分钟"
  exit "$1"
}

# ───────────────────────── S0 环境 ─────────────────────────
stage "S0 环境"
say "# bm2_r4 汇总(想法二第四轮:两遍式独立核 v3 + 行门标定 + 只评测电池)"
say ""
say "## S0 环境"
say "- 主机: $(hostname)  开始: $(date -u +%FT%TZ)  快照: $ROOT"
say "- nvcc(PATH): $(command -v nvcc) / $(nvcc --version | tail -1)"
HAVE_SCIPY=0; python3 -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)" > "$OUT/s0_py.log" 2>&1 && HAVE_SCIPY=1
say "- python3: $(python3 --version 2>&1) / $(tail -1 "$OUT/s0_py.log")"
HAVE_PATCH=0; command -v patch >/dev/null && HAVE_PATCH=1
say "- 磁盘: $(df -h "$JD" | tail -1)"

pick_gpu() {
  local g uuid n
  for g in 0 1; do
    uuid=$(nvidia-smi -i $g --query-gpu=uuid --format=csv,noheader 2>/dev/null) || continue
    [ -n "$uuid" ] || continue
    n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$uuid")
    [ "$n" = 0 ] && { echo $g; return 0; }
  done
  return 1
}
GPU=""; SHARED=0
for try in $(seq 1 30); do
  GPU=$(pick_gpu) && break
  echo "两张卡都有计算进程,等 60s($try/30)"; sleep 60
done
if [ -z "$GPU" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
  SHARED=1
fi
export CUDA_VISIBLE_DEVICES=$GPU
say "- **用卡: GPU $GPU**$([ $SHARED = 1 ] && echo '(⚠ 等了 30 分钟仍有别的计算进程,与人共卡,计时读数降级为参考)')"
say "- 开跑时两卡: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"
gpu_note() { say "- 卡状态($1): $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';') 计算进程数: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"; }

ln -sfn "$DATA" "$ROOT/data"
find_mtx() {   # $1=阵名(无 .mtx)→ 相对路径
  local d f
  for d in data/ocean/square data/first100 data/train_disp; do
    [ -f "$d/$1.mtx" ] && { echo "$d/$1.mtx"; return 0; }
  done
  f=$(find -L data -maxdepth 4 -name "$1.mtx" 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; return 0; }
  return 1
}
RUNENV="USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0"

# ───────────────────────── S1 微复现器 ─────────────────────────
stage "S1 微复现器"
say ""; say "## S1 微复现器"
S1=red
GOT_HDR=$(sha256sum src/hh_bm2.cuh | cut -d' ' -f1)
if [ "$GOT_HDR" = "$HDR_SHA" ]; then
  S1=green
  say "- src/hh_bm2.cuh 的 sha256 与 bm2_r1 验过的相同 → 沿用 r1 的 ALL-PASS"
else
  say "- src/hh_bm2.cuh 已变($GOT_HDR),重跑 hh_repro5"
  if nvcc -O3 -arch=sm_90 -std=c++14 -Isrc -Iinclude -I/usr/local/cuda/include tools/hh_repro5.cu -o "$JD/hh_repro5" > "$OUT/s1_repro_build.log" 2>&1; then
    ( "$JD/hh_repro5" 40 42; echo "rc=$?"; "$JD/hh_repro5" 25 2026; echo "rc=$?" ) > "$OUT/s1_repro.log" 2>&1
    if [ "$(grep -c '^ALL-PASS' "$OUT/s1_repro.log")" = 2 ] && [ "$(grep -c '^rc=0' "$OUT/s1_repro.log")" = 2 ]; then S1=green; fi
    say "- $(grep -E '^(ALL-PASS|FAIL)' "$OUT/s1_repro.log" | tr '\n' ';')"
    say "- 行数合计: $(awk '/rows=/{for(i=1;i<=NF;i++){split($i,a,"=");s[a[1]]+=a[2]}}END{printf "rows=%d ok=%d ovf=%d skip=%d bad=%d",s["rows"],s["ok"],s["ovf"],s["skip"],s["bad"]}' "$OUT/s1_repro.log")(9 档几何 × B16/值区模式/Pass1 来源 6 种组合)"
  else
    say "- **复现器构建失败**:"; tail -30 "$OUT/s1_repro_build.log" | tee -a "$SUM"
  fi
fi
say "- S1: $S1"

# ───────────────────────── S2 构建 ─────────────────────────
stage "S2 构建"
say ""; say "## S2 构建"
S2=red
T0=$(date +%s)
make -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_bm2 > "$JD/build_bm2.log" 2>&1; RC=$?
T1=$(date +%s)
tail -c 100000 "$JD/build_bm2.log" > "$OUT/s2_build_bm2_tail.log"
say "- make TARGET=spgemm_bm2: rc=$RC,用时 $((T1 - T0))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_bm2.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_bm2 ] && [ "$(grep -c '✓' "$JD/build_bm2.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_bm2.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_bm2w > "$JD/build_bm2w.log" 2>&1; RCW=$?
  say "- WRITE_MTX 件 spgemm_bm2w: rc=$RCW($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_bm2w ] && S2=green
else
  say "- **构建或审计失败**,日志尾:"; tail -60 "$JD/build_bm2.log" | tee -a "$SUM"
fi
say "- S2: $S2"
[ $S1 = green ] || finish 11 "红:S1 微复现器未全绿(S2=$S2);后续阶段未跑"
[ $S2 = green ] || finish 12 "红:S2 构建/审计失败;后续阶段未跑"

# ───────────────────────── S3 冒烟 ─────────────────────────
stage "S3 冒烟"
say ""; say "## S3 冒烟"
python3 scripts/ab_arms.py --bin ./spgemm_bm2 --out "$OUT/s3_smoke" --rounds 1 --gpu "$GPU" \
  --arm off: --arm m1:HH_BM2=1 --arm m2:HH_BM2=2 --arm m2p:HH_BM2=2,HH_BM2_P1=1 --arm m2L:HH_BM2=2,HH_BM2_L1=1 --arm m2g:HH_BM2=2,HH_BM2_FX=200,HH_BM2_CX=40 --only bcsstk17,F2,case9,c-64,TSOPF,pre2 --stat > "$OUT/s3_smoke_run.log" 2>&1
RC=$?
S3=green
[ $RC = 0 ] || S3=red
grep -qE "NNZ-MISMATCH|FAIL|不可测" "$OUT/s3_smoke.txt" 2>/dev/null && S3=red
[ -s "$OUT/s3_smoke.txt" ] || S3=red
TAKEN=$(awk -F, 'NR>1{s+=$7}END{print s+0}' "$OUT/s3_smoke_stat.csv" 2>/dev/null)
say "- 小阵 off/m1/m2/m2p/m2L/m2g(m2g 带门,只为走一遍门的代码路径)各 1 发:nnz 与失败检查 → $S3;走两遍式的行数合计 = ${TAKEN:-0}"
[ $S3 = green ] || { tail -40 "$OUT/s3_smoke_run.log" | tee -a "$SUM"; tail -30 "$OUT/s3_smoke_fail.log" 2>/dev/null | tee -a "$SUM"; finish 13 "红:S3 冒烟 nnz 不一致或运行失败;后续阶段未跑"; }

# ───────────────────────── S4 值级 ─────────────────────────
arm_env() {   # $1=臂名 → env 赋值串
  case $1 in
    off)  echo "HH_BM2=0";;
    m1)   echo "HH_BM2=1";;
    m2)   echo "HH_BM2=2";;
    m2p)  echo "HH_BM2=2 HH_BM2_P1=1";;
    m1L)  echo "HH_BM2=1 HH_BM2_L1=1";;
    m2L)  echo "HH_BM2=2 HH_BM2_L1=1";;
    *)    echo "HH_BM2=0";;
  esac
}
vl_dump() {   # $1=name $2=path $3=臂名 → $JD/vl/<name>_<臂名>.mtx;回显 "nnz taken skip"
  local name=$1 path=$2 tag=$3 log
  log="$JD/vl/${name}_${tag}.log"
  rm -f "results/matrices/$name/self_product_hash.mtx"
  env $RUNENV METHOD=hash WARMUP_N=0 $(arm_env "$tag") HH_BM2_STAT=1 timeout 1800 ./spgemm_bm2w "$path" > "$log" 2>&1
  local rc=$?
  if [ $rc != 0 ] || [ ! -s "results/matrices/$name/self_product_hash.mtx" ]; then
    echo "RUNFAIL rc=$rc"; tail -15 "$log" >&2; return 1
  fi
  mv "results/matrices/$name/self_product_hash.mtx" "$JD/vl/${name}_${tag}.mtx"
  local nnz taken skip
  nnz=$(grep -oE "Result C:.*nnz = [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+$")
  taken=$(grep -oE "taken_rows=[0-9]+" "$log" | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  skip=$(grep -oE "skip_rows=[0-9]+" "$log" | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  grep "bm2-stat" "$log" | sed "s/^/[$name $tag] /" >> "$OUT/s4_value_stat.log"
  echo "$nnz $taken $skip"
}
gate() {      # $1=name $2=tag(m1/m2)→ 回显 "PASS|FAIL 细节"(对 <name>_off.mtx)
  ln -sfn "$JD/vl/${1}_${2}.mtx" "$JD/vl/${1}_on.mtx"
  local g; g=$(bash scripts/mtx_gate_fast.sh "$JD/vl/$1" 2>&1)
  rm -f "$JD/vl/${1}_on.mtx"
  echo "$(echo "$g" | grep -oE "PASS|FAIL" | tail -1) $(echo "$g" | grep -oE "col_mismatch=.*" | head -1)"
}
VL_TAKEN=0; S4=green

stage "S4a 值级·对照组(四臂各自过官方三判 + 归一误差)"
say ""; say "## S4a 值级·对照组:off / m1 / m2 / m2p 各自对 scipy 真值"
say ""
say "判法(dialog/qa-001 双方对齐的作业内口径):差分门 PASS ∧ 官方 L1=L2=0 ∧ 该臂 nerr 最大值 ≤ 1e-10(nerr=|got−want|/Σ|乘积|);官方 L3 错行数照列,并与 l3_diag 用同一判式重算的行数对账(不等=两套口径脱钩=红)。"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方 L1 / L2 / L3 错行 | l3_diag 重算错行 | nerr 最大 | 本作业判 |"
say "|---|---|---|---|---|---|---|---|---|"
XCAL_BAD=0; XCAL_N=0
if [ $HAVE_SCIPY = 1 ]; then
for name in bcsstk17 af23560 cfd2 msc10848 s3dkq4m2; do
  path=$(find_mtx "$name") || { say "| $name | - | - | - | 缺文件,跳过 | - | - | - | - |"; continue; }
  echo "---- 对照组 $name ($path)"
  ok=1; declare -A R=()
  for tag in off m1 m2 m2p; do
    R[$tag]=$(vl_dump "$name" "$path" $tag) || { say "| $name | $tag | ${R[$tag]} | - | **运行失败** | - | - | - | 红 |"; S4=red; ok=0; }
  done
  [ $ok = 1 ] || { rm -f "$JD"/vl/${name}_*.mtx; continue; }
  timeout 3600 python3 scripts/l3_diag.py "$path" "off=$JD/vl/${name}_off.mtx" "m1=$JD/vl/${name}_m1.mtx" "m2=$JD/vl/${name}_m2.mtx" "m2p=$JD/vl/${name}_m2p.mtx" \
    --list 60 --csv "$OUT/s4_l3diag_${name}.csv" --rows-csv "$OUT/s4_l3rows_${name}.csv.gz" --full-csv "$JD/keep/l3full_${name}.csv.gz" \
    > "$OUT/s4_l3diag_${name}.txt" 2>&1 || echo "l3_diag rc=$? ($name)"
  tail -25 "$OUT/s4_l3diag_${name}.txt"
  if [ -s "$JD/keep/l3full_${name}.csv.gz" ]; then   # 条目级全量:留在 C;够小则同时进 OUT_DIR(qa-003 待答 1 的方案 A,可撤回)
    fsz=$(stat -c %s "$JD/keep/l3full_${name}.csv.gz")
    echo "$name $(( $(zcat "$JD/keep/l3full_${name}.csv.gz" | wc -l) - 1 )) $fsz $(sha256sum "$JD/keep/l3full_${name}.csv.gz" | cut -d' ' -f1) $JD/keep/l3full_${name}.csv.gz" >> "$OUT/s4_l3full_index.txt"
    [ "$fsz" -le 1500000 ] && cp "$JD/keep/l3full_${name}.csv.gz" "$OUT/s4_l3full_${name}.csv.gz"
  fi
  if [ -s "$OUT/s4_l3rows_${name}.csv.gz" ] && [ "$(stat -c %s "$OUT/s4_l3rows_${name}.csv.gz")" -gt 1900000 ]; then
    mv "$OUT/s4_l3rows_${name}.csv.gz" "$JD/keep/"; echo "$name 按行聚合清单超 1.9MB,改留 $JD/keep/s4_l3rows_${name}.csv.gz" >> "$OUT/s4_l3full_index.txt"
  fi
  OFF_L3=""
  for tag in off m1 m2 m2p; do
    set -- ${R[$tag]}; nnz=$1; taken=$2; skip=$3
    [ $tag = off ] || VL_TAKEN=$((VL_TAKEN + taken))
    t=$(timeout 3600 python3 scripts/verify_sorted_truth.py "$path" "$JD/vl/${name}_${tag}.mtx" 2>&1)
    echo "$t" | tail -9 > "$OUT/s4_truth_${name}_${tag}.txt"
    l1=$(echo "$t" | grep -oE "^L1 集合错行: [0-9]+" | grep -oE "[0-9]+$")
    l2=$(echo "$t" | grep -oE "^L2 乱序行: +[0-9]+" | grep -oE "[0-9]+$")
    l3=$(echo "$t" | grep -oE "^L3 数值错行: [0-9]+" | grep -oE "[0-9]+$")
    mr=$(echo "$t" | grep -oE "全局最大相对误差: [0-9.e+-]+" | grep -oE "[0-9.e+-]+$")
    ne=$(grep -E "^NERR_MAX $tag " "$OUT/s4_l3diag_${name}.txt" 2>/dev/null | awk '{print $3}')
    ob=$(grep -E "^OFFICIAL_BAD_ROWS $tag " "$OUT/s4_l3diag_${name}.txt" 2>/dev/null | awk '{print $3}')
    xc="?"
    if [ -n "$ob" ] && [ -n "$l3" ]; then
      XCAL_N=$((XCAL_N + 1))
      if [ "$ob" = "$l3" ]; then xc="$ob(=)"; else xc="$ob(≠)"; XCAL_BAD=$((XCAL_BAD + 1)); fi
    fi
    if [ $tag = off ]; then
      OFF_L3=${l3:-x}; gv="-"; verdict="对照"
    else
      gv=$(gate "$name" $tag)
      verdict=绿
      [ "${gv%% *}" = PASS ] || verdict=红
      [ "${l1:-x}" = 0 ] && [ "${l2:-x}" = 0 ] || verdict=红
      if [ -n "$ne" ]; then
        case "$ne" in
          *inf*|*nan*) verdict="红(nerr=$ne)";;
          *) awk -v v="$ne" 'BEGIN{exit !(v+0 <= 1e-10)}' || verdict="红(nerr 超 1e-10)";;
        esac
      elif [ "${l3:-x}" != 0 ]; then
        # l3_diag 没出数:退回"off 臂同阵官方 L3 也不过"的对照判法
        if [ "$OFF_L3" = 0 ] || [ "$OFF_L3" = x ]; then verdict="红(l3_diag 无输出且 off 臂官方 L3 过)"; else [ "$verdict" = 绿 ] && verdict="绿(l3_diag 无输出;官方 L3 双臂同红,仅凭差分门)"; fi
      fi
      [ "${R[off]%% *}" = "$nnz" ] || verdict="红(nnz≠off)"
      case "$verdict" in 红*) S4=red;; esac
    fi
    say "| $name | $tag | $nnz | $taken / $skip | $gv | ${l1:-?} / ${l2:-?} / ${l3:-?} | $xc | ${ne:-?} | $verdict |"
  done
  rm -f "$JD"/vl/${name}_*.mtx
done
else
  say "| (scipy 不可用,对照组跳过) | | | | | | | | |"
fi
say ""
say "- 交叉标定(官方 L3 错行数 对 l3_diag 重算):$XCAL_N 组里不等的 $XCAL_BAD 组"
[ "$XCAL_BAD" = 0 ] || S4=red
say "- 并排清单(前 60 条):s4_l3diag_<阵>.txt / .csv;按行聚合全量:s4_l3rows_<阵>.csv.gz;官方三判原文尾:s4_truth_<阵>_<臂>.txt"
if [ -s "$OUT/s4_l3full_index.txt" ]; then
  say "- 条目级全量清单(阵 条数 字节 sha256 C上路径;≤1.5MB 的同时在 OUT_DIR 的 s4_l3full_<阵>.csv.gz):"
  sed 's/^/    /' "$OUT/s4_l3full_index.txt" | tee -a "$SUM"
fi

stage "S4b 值级·差分门 + 归一误差"
say ""; say "## S4b 值级:HH_BM2=1/2 对 =0 的差分门 + 三臂对 scipy 真值的归一误差(不跑官方逐行三判:这几阵 4-6 千万 nnz,python 逐行太慢)"
say ""
say "判法:差分门 PASS ∧ nnz 同 off ∧(l3_diag 有输出时)该臂 nerr 最大值 ≤ 1e-10。"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方口径错行(仅参考) | nerr 最大 | 本作业判 |"
say "|---|---|---|---|---|---|---|---|"
gate_diag() {   # $1=name:三臂转储 → 差分门 + l3_diag
  local name=$1 path ok=1 mode tag gv ne ob verdict
  path=$(find_mtx "$name") || { say "| $name | - | - | - | 缺文件,跳过 | - | - | - |"; return 0; }
  local free_gb; free_gb=$(df -BG --output=avail "$JD" | tail -1 | tr -dc 0-9)
  [ "${free_gb:-0}" -ge 60 ] || { say "| $name | - | - | - | 磁盘余量 <60G,跳过 | - | - | - |"; return 0; }
  echo "---- 差分门+归一误差 $name ($path)"
  declare -A Q=()
  for tag in off m1 m2; do
    Q[$tag]=$(vl_dump "$name" "$path" $tag) || { say "| $name | $tag | ${Q[$tag]} | - | **运行失败** | - | - | 红 |"; S4=red; ok=0; }
  done
  [ $ok = 1 ] || { rm -f "$JD"/vl/${name}_*.mtx; return 0; }
  if [ $HAVE_SCIPY = 1 ]; then
    timeout 3600 python3 scripts/l3_diag.py "$path" "off=$JD/vl/${name}_off.mtx" "m1=$JD/vl/${name}_m1.mtx" "m2=$JD/vl/${name}_m2.mtx" \
      --list 30 --csv "$OUT/s4_l3diag_${name}.csv" > "$OUT/s4_l3diag_${name}.txt" 2>&1 || echo "l3_diag rc=$? ($name)"
    tail -12 "$OUT/s4_l3diag_${name}.txt"
  fi
  for tag in off m1 m2; do
    set -- ${Q[$tag]}
    ne=$(grep -E "^NERR_MAX $tag " "$OUT/s4_l3diag_${name}.txt" 2>/dev/null | awk '{print $3}')
    ob=$(grep -E "^OFFICIAL_BAD_ROWS $tag " "$OUT/s4_l3diag_${name}.txt" 2>/dev/null | awk '{print $3}')
    if [ $tag = off ]; then
      say "| $name | off | $1 | $2 / $3 | - | ${ob:-?} | ${ne:-?} | 对照 |"
      continue
    fi
    VL_TAKEN=$((VL_TAKEN + $2))
    gv=$(gate "$name" $tag)
    verdict=绿
    [ "${gv%% *}" = PASS ] || verdict=红
    [ "${Q[off]%% *}" = "$1" ] || verdict="红(nnz≠off)"
    if [ -n "$ne" ]; then
      case "$ne" in
        *inf*|*nan*) verdict="红(nerr=$ne)";;
        *) awk -v v="$ne" 'BEGIN{exit !(v+0 <= 1e-10)}' || verdict="红(nerr 超 1e-10)";;
      esac
    else
      [ "$verdict" = 绿 ] && verdict="绿(l3_diag 无输出,仅凭差分门)"
    fi
    case "$verdict" in 红*) S4=red;; esac
    say "| $name | $tag | $1 | $2 / $3 | $gv | ${ob:-?} | ${ne:-?} | $verdict |"
  done
  rm -f "$JD"/vl/${name}_*.mtx
}
gate_only() {   # $1=name $2..=臂名:只做差分门(特大阵)
  local name=$1; shift
  local path r0 r tag gv
  path=$(find_mtx "$name") || { say "| $name | - | - | - | 缺文件,跳过 |"; return 0; }
  local free_gb; free_gb=$(df -BG --output=avail "$JD" | tail -1 | tr -dc 0-9)
  [ "${free_gb:-0}" -ge 60 ] || { say "| $name | - | - | - | 磁盘余量 <60G,跳过 |"; return 0; }
  echo "---- 差分门 $name ($path)"
  r0=$(vl_dump "$name" "$path" off) || { say "| $name | off | $r0 | - | **基线臂运行失败** |"; S4=red; return 0; }
  for tag in "$@"; do
    r=$(vl_dump "$name" "$path" "$tag") || { say "| $name | $tag | $r | - | **运行失败** |"; S4=red; continue; }
    set -- $r
    VL_TAKEN=$((VL_TAKEN + $2))
    gv=$(gate "$name" $tag)
    [ "${gv%% *}" = PASS ] || S4=red
    [ "${r0%% *}" = "$1" ] || { S4=red; gv="$gv NNZ≠off(${r0%% *})"; }
    say "| $name | $tag | $1 | $2 / $3 | $gv |"
    rm -f "$JD/vl/${name}_${tag}.mtx"
  done
  rm -f "$JD/vl/${name}_off.mtx"
}
gate_diag TSOPF_RS_b678_c2
say ""
say "### ldoor(翻 bin 家族:legacy 形态也交两遍式独立核的 m1L/m2L 臂;1.45 亿 nnz,只跑差分门)"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |"
say "|---|---|---|---|---|"
gate_only ldoor m1L m2L
say ""
say "### bin9 形态的生产值级(dialog/qa-004 指出的缺口):先用不转储的普查跑找出 bin9 确有行走两遍式的阵,再对前两个做差分门 + l3_diag"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方口径错行(仅参考) | nerr 最大 | 本作业判 |"
say "|---|---|---|---|---|---|---|---|"
B9N=0; B9SCOUT=""
for name in hangGlider_3 lowThrust_4 pf2177 TSOPF_FS_b162_c1 TSOPF_FS_b39_c7 bloweybq case9 hangGlider_4 lowThrust_5 crankseg_1 nd3k; do
  [ $B9N -ge 2 ] && break
  path=$(find_mtx "$name") || continue
  t9=$(env $RUNENV METHOD=hash WARMUP_N=0 HH_BM2=2 HH_BM2_STAT=1 timeout 600 ./spgemm_bm2 "$path" 2>&1 < /dev/null | grep "bm2-stat" | grep "bin=9 " | grep -oE "taken_rows=[0-9]+" | cut -d= -f2 | tail -1)
  B9SCOUT="$B9SCOUT $name:${t9:-无bin9}"
  if [ "${t9:-0}" -gt 0 ]; then gate_diag "$name"; B9N=$((B9N + 1)); fi
done
say ""
say "- bin9 探查(阵:bin9 走两遍式的行数):$B9SCOUT;做了值级的 $B9N 个$([ $B9N = 0 ] && echo ' **(=0:bin9 形态的生产值级仍是空白,只有微复现器覆盖)**')"
rm -rf "$ROOT"/results/matrices/*/self_product*.mtx
say ""
say "- 本轮值级里走两遍式的行数合计(m1+m2): $VL_TAKEN"
[ "$VL_TAKEN" -gt 0 ] || S4=red
say "- S4(a+b): $S4"
[ $S4 = green ] || finish 14 "红:S4 值级未过(见上表);计时阶段未跑"

# ───────────────────────── S5 行门标定(训练池,逐行配对周期) ─────────────────────────
stage "S5 行门标定"
say ""; say "## S5 行门标定(只用 data/train_disp;逐行配对周期,两遍式核已补收尾栅栏;两种形态都收行以便分形态看)"
gpu_note "S5 前"
AUTO_OK=0; AUTO_ENV=""
NTR=0; : > "$JD/cal/list.txt"
if [ -d data/train_disp ]; then
  # 1MB-800MB,按大小降序最多 60 个;与任何测试账目录同名的剔除;af23560 是电池里的 F2,也剔除(留作样本外评测)
  while read -r sz f; do
    b=$(basename "$f" .mtx)
    [ "$b" = af23560 ] && continue
    [ -f "data/ocean/square/$b.mtx" ] || [ -f "data/first100/$b.mtx" ] || [ -n "$(find -L data/flex142 data/external_72 -maxdepth 3 -name "$b.mtx" 2>/dev/null | head -1)" ] && continue
    echo "$b $f" >> "$JD/cal/list.txt"; NTR=$((NTR + 1))
    [ $NTR -ge 60 ] && break
  done < <(find -L data/train_disp -maxdepth 2 -name '*.mtx' -size +1M -size -800M -printf '%s %p\n' 2>/dev/null | sort -rn)
fi
say "- 训练池候选 $NTR 阵(1MB-800MB,非测试账同名,不含 af23560)"
NRUN=0; NFAIL=0
while read -r b f; do
  for arm in off m1 m2 m2p; do
    case $arm in off) e="HH_BM2=0";; m1) e="HH_BM2=1 HH_BM2_L1=1";; m2) e="HH_BM2=2 HH_BM2_L1=1";; *) e="HH_BM2=2 HH_BM2_L1=1 HH_BM2_P1=1";; esac
    env $RUNENV METHOD=hash WARMUP_N=1 $e HH_BM2_STAT=1 HASH_ROWTIME="$JD/cal/$b.$arm.rt" timeout 600 ./spgemm_bm2 "$f" < /dev/null > /dev/null 2> "$JD/cal/$b.$arm.log"
    rc=$?; NRUN=$((NRUN + 1))
    if [ $rc != 0 ] || [ ! -s "$JD/cal/$b.$arm.rt" ]; then NFAIL=$((NFAIL + 1)); echo "cal FAIL $b $arm rc=$rc"; rm -f "$JD/cal/$b.off.rt"; break; fi
  done
  # 第二个目录:Pass1 读 B16 的那组,冒名成 m2 给标定脚本(m1 照旧)
  if [ -s "$JD/cal/$b.off.rt" ]; then
    for x in off.rt m1.rt m1.log; do ln -sf "$JD/cal/$b.$x" "$JD/calp/$b.$x"; done
    ln -sf "$JD/cal/$b.m2p.rt" "$JD/calp/$b.m2.rt"; ln -sf "$JD/cal/$b.m2p.log" "$JD/calp/$b.m2.log"
  fi
done < "$JD/cal/list.txt"
say "- 采数:$NRUN 发,失败 $NFAIL 发;逐行周期文件 $(ls "$JD/cal"/*.off.rt 2>/dev/null | wc -l) 组,共 $(du -sh "$JD/cal" | cut -f1)"
R0=""; R1=""
if [ $HAVE_SCIPY = 1 ] && ls "$JD/cal"/*.off.rt >/dev/null 2>&1; then
  timeout 1800 python3 scripts/bm2_gate_cal.py --dir "$JD/cal" --out "$OUT/s5_cal" > "$OUT/s5_cal_run.log" 2>&1
  say "- bm2_gate_cal(Pass1 读 B_ci)rc=$?;见 s5_cal.txt / s5_cal_cells.csv / s5_cal.env"
  grep -E "^(# bm2_gate_cal|=====|  全收|  → |  实例税|# auto|# 位图形态|# legacy)" "$OUT/s5_cal.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
  timeout 1800 python3 scripts/bm2_gate_cal.py --dir "$JD/calp" --out "$OUT/s5_calp" > "$OUT/s5_calp_run.log" 2>&1
  say "- bm2_gate_cal(Pass1 读 B16 列字段;表里的'模式 2'即 m2p)rc=$?;见 s5_calp.txt / s5_calp_cells.csv / s5_calp.env"
  grep -E "^(=====|  全收|  → |# auto|# 位图形态)" "$OUT/s5_calp.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
  grep -q '^AUTO_OK=1' "$OUT/s5_cal.env" 2>/dev/null && R0=$(grep '^AUTO_R=' "$OUT/s5_cal.env" | cut -d= -f2)
  grep -q '^AUTO_OK=1' "$OUT/s5_calp.env" 2>/dev/null && R1=$(grep '^AUTO_R=' "$OUT/s5_calp.env" | cut -d= -f2)
  # 两次标定各自给出位图形态的最优 (模式,FX,CX) 与平均 R;取 R 小的那次。只有选中模式 2 时 Pass1 来源才有区别
  if [ -n "$R1" ] && { [ -z "$R0" ] || awk -v a="$R1" -v b="$R0" 'BEGIN{exit !(a+0 < b+0)}'; } && grep -q '^AUTO_MODE=2' "$OUT/s5_calp.env"; then
    AUTO_ENV="$(grep '^AUTO_ENV=' "$OUT/s5_calp.env" | head -1 | cut -d= -f2-),HH_BM2_P1=1"; AUTO_OK=1
  elif [ -n "$R0" ]; then
    AUTO_ENV=$(grep '^AUTO_ENV=' "$OUT/s5_cal.env" | head -1 | cut -d= -f2-); AUTO_OK=1
  fi
  [ -s "$OUT/s5_cal.txt" ] || { tail -20 "$OUT/s5_cal_run.log" | tee -a "$SUM"; }
fi
say "- 标定选出的臂: $([ $AUTO_OK = 1 ] && echo "$AUTO_ENV(平均 R:读 B_ci=${R0:-无} / 读 B16=${R1:-无})" || echo '无(标定未给出 ≥0.5% 平均收益的门,或标定脚本没出数)')"
rm -rf "$JD/cal" "$JD/calp"

# ───────────────────────── S6 只评测电池 ─────────────────────────
stage "S6 只评测电池"
say ""; say "## S6 只评测电池(同一二进制;off / m2=位图形态不设门 / m2p=同上+Pass1 读 B16 / m2L=m2 再加 legacy 形态 / auto=S5 选出的臂;5 轮交错)"
EXTRA=""
for name in c-big cfd2 s3dkq4m2 nlpkkt80 FEM_3D_thermal2 offshore msc10848 bcsstk30; do
  p=$(find_mtx "$name") && EXTRA="$EXTRA --extra $name=$p:hash"
done
ARMS="--arm off: --arm m2:HH_BM2=2 --arm m2p:HH_BM2=2,HH_BM2_P1=1 --arm m2L:HH_BM2=2,HH_BM2_L1=1"
if [ $AUTO_OK = 1 ]; then
  case "$AUTO_ENV" in
    "HH_BM2=2,HH_BM2_FX=0,HH_BM2_CX=0")              say "- auto 臂与 m2 臂配置相同,不重复跑";;
    "HH_BM2=2,HH_BM2_FX=0,HH_BM2_CX=0,HH_BM2_P1=1")  say "- auto 臂与 m2p 臂配置相同,不重复跑";;
    *) ARMS="$ARMS --arm auto:$AUTO_ENV";;
  esac
fi
python3 scripts/ab_arms.py --bin ./spgemm_bm2 --out "$OUT/s6_battery" --rounds 5 --gpu "$GPU" $ARMS --stat $EXTRA > "$OUT/s6_battery_run.log" 2>&1
say "- ab_arms rc=$?;逐阵表(含两遍式 flop 占比/行占比、accumulate 相位比)见 s6_battery.txt,逐发 s6_battery.csv,逐 bin 普查 s6_battery_stat.csv"
grep -E "^(GM|裁定|=====)" "$OUT/s6_battery.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
gpu_note "S6 后"

# ───────────────────────── S7 无侵入电池 ─────────────────────────
stage "S7 无侵入电池"
say ""; say "## S7 无侵入(A=spgemm_base:本提交反打补丁后的基线源码;B=spgemm_bm2 默认 env=开关关;官方 ab_pair.py 16×3)"
if [ $HAVE_PATCH = 1 ]; then
  rm -rf "$JD/base"; mkdir -p "$JD/base"
  cp -r Makefile src include tools "$JD/base/"
  rm -f "$JD"/base/src/*.o
  ( cd "$JD/base" && patch -p1 -R < "$ROOT/jobs/$ID/bm2_src.patch" ) > "$OUT/s7_patch.log" 2>&1
  GOT=$(sha256sum "$JD/base/src/spgemm_kernel_hash.cu" | cut -d' ' -f1)
  if [ "$GOT" = "$BASE_SHA" ]; then
    say "- 基线源码 sha256 核对通过"
    T0=$(date +%s)
    ( cd "$JD/base" && make -j4 TARGET=spgemm_base ) > "$JD/build_base.log" 2>&1; RC=$?
    say "- make TARGET=spgemm_base: rc=$RC,用时 $(( $(date +%s) - T0 ))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_base.log" | tr '\n' ';')"
    if [ $RC = 0 ] && [ -x "$JD/base/spgemm_base" ]; then
      cp "$JD/base/spgemm_base" "$ROOT/spgemm_base"
      python3 scripts/ab_pair.py spgemm_base spgemm_bm2 "$OUT/s7_ab_base_vs_bm2off.csv" 3 "$GPU" > "$OUT/s7_ab_run.log" 2>&1
      sed -n '/===== 判读/,$p' "$OUT/s7_ab_run.log" > "$OUT/s7_ab_verdict.txt"
      grep -E "^(GM|裁定|微级)" "$OUT/s7_ab_verdict.txt" | sed 's/^/    /' | tee -a "$SUM"
    else
      tail -30 "$JD/build_base.log" | tee -a "$SUM"
    fi
  else
    say "- **基线源码 sha256 不符**(得 $GOT),S7 未跑;见 s7_patch.log"
  fi
else
  say "- 跳过:C 上没有 patch 命令"
fi
gpu_note "S7 后"

# ───────────────────────── S8 值级·大阵差分门 ─────────────────────────
stage "S8 值级·大阵差分门"
say ""; say "## S8 值级·大阵差分门(pre2 的 m2L 臂:legacy 形态 bin4/5/7/8;放在计时之后只为省等待时间,红了同样作数)"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |"
say "|---|---|---|---|---|"
S4=green
gate_only pre2 m2L
say ""
say "- S8: $S4"
[ $S4 = green ] || finish 18 "红:S8 大阵差分门未过——S6/S7 的计时读数作废,先修正确性"

finish 0 "S1-S4、S8 全绿;S5 标定、S6/S7 的读数见各段与对应 txt/csv"
