#!/usr/bin/env bash
# bm2_r1 —— 刀三·想法二(HH_BM2:hash_heavy 位图标记-累积两遍式,study/69 §二十六)第一轮
#
# 卡:钉 GPU 0(CUDA_VISIBLE_DEVICES=0)。仅当开跑时 0 上已有别的计算进程而 1 空闲,才改钉 1;
#     两张都忙则等(最多 30 分钟)。全程只用这一张,实际卡号写在 OUT_DIR/SUMMARY.md 的 S0 段。
#     所有计时臂同卡同批、逐阵逐轮交错。
# 阶段(红即停:S1-S4 任一红,后面的计时阶段不跑,SUMMARY 写明原因):
#   S0 环境探测      工具链版本、python 依赖、磁盘、选卡
#   S1 微复现器      tools/hh_repro5.cu(#include 生产头 src/hh_bm2.cuh)两个种子全绿
#   S2 构建          make TARGET=spgemm_bm2(审计双绿)+ WRITE_MTX 件 spgemm_bm2w
#   S3 冒烟          小阵 HH_BM2=0/1/2 三臂 nnz 逐位 + 普查(确有行走两遍式)
#   S4 值级          WRITE_MTX 转储:HH_BM2=1/2 对 HH_BM2=0 的 mtx_gate_fast(列集合/行内序/值)
#                    + scipy 在则 verify_sorted_truth.py 三判;转储即清
#   S5 三臂电池      off / m1(值区 SMEM)/ m2(值直写全局),16 阵 + 靶阵,3 轮交错,带相位与普查
#   S6 行门判别      只在训练池(data/train_disp)上比 无门 / dup 门(HH_BM2_DUP)/ 密度门(HH_BM2_SPANX)——428 主账不参与标定
#   S7 无侵入电池    基线源码(本提交反打补丁,sha256 核对)构建 spgemm_base,ab_pair.py base vs 新件(开关关)
# 回传:OUT_DIR/SUMMARY.md(先看这个)+ 各阶段 s*_*.log/csv/txt。大文件(转储)留在 JOB_DIR 并已删。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-bm2_r1}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c          # 基线 src/spgemm_kernel_hash.cu(提交 70d1dea=675b227 形态)的 sha256
mkdir -p "$OUT" "$JD/vl"
: > "$SUM"
say() { echo "$*" | tee -a "$SUM"; }
stage() { echo; echo "################ $* ################"; date -u +%FT%TZ; }
finish() {   # $1=退出码 $2=结论一句话
  rm -rf "$ROOT"/results/matrices/*/self_product*.mtx "$JD/vl" 2>/dev/null
  say ""; say "## 结束"; say "- 结论: $2"; say "- 磁盘: $(df -h "$JD" | tail -1)"; say "- 结束时间: $(date -u +%FT%TZ)"
  exit "$1"
}

# ───────────────────────── S0 环境 ─────────────────────────
stage "S0 环境探测"
say "# bm2_r1 汇总(想法二 HH_BM2 第一轮)"
say ""
say "## S0 环境"
say "- 主机: $(hostname)  开始: $(date -u +%FT%TZ)  快照: $ROOT"
{
  echo "== which -a nvcc"; which -a nvcc
  echo "== nvcc --version"; nvcc --version
  echo "== ls -ld /usr/local/cuda*"; ls -ld /usr/local/cuda* 2>&1
  for x in /usr/local/cuda/bin/nvcc /usr/local/cuda-12.8/bin/nvcc; do [ -x "$x" ] && { echo "== $x"; "$x" --version | tail -2; }; done
  echo "== g++"; g++ --version | head -1
  echo "== make"; make --version | head -1
  echo "== cuobjdump"; command -v cuobjdump || echo MISSING
  echo "== patch"; command -v patch || echo MISSING
  echo "== python3"; command -v python3; python3 --version
  python3 -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)" 2>&1
  echo "== nvidia-smi"; nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
  echo "== compute apps"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
  echo "== df"; df -h "$JD"
  echo "== data"; ls "$DATA"
} > "$OUT/s0_env.log" 2>&1
cat "$OUT/s0_env.log"
say "- nvcc(PATH,构建用这个,与裸 make 相同): $(command -v nvcc) / $(nvcc --version | tail -1)"
HAVE_SCIPY=0; python3 -c "import numpy, scipy" 2>/dev/null && HAVE_SCIPY=1
say "- python3 scipy: $([ $HAVE_SCIPY = 1 ] && echo '有' || echo '无——verify_sorted_truth.py 三判跳过,见 NEED_USER.md')"
HAVE_PATCH=0; command -v patch >/dev/null && HAVE_PATCH=1

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
stage "S1 微复现器 hh_repro5"
say ""; say "## S1 微复现器(tools/hh_repro5.cu,编进去的是生产头 src/hh_bm2.cuh)"
S1=red
if nvcc -O3 -arch=sm_90 -std=c++14 -Isrc -Iinclude -I/usr/local/cuda/include tools/hh_repro5.cu -o "$JD/hh_repro5" > "$OUT/s1_repro_build.log" 2>&1 \
   || nvcc -O3 -arch=sm_90 -std=c++14 -Isrc tools/hh_repro5.cu -o "$JD/hh_repro5" >> "$OUT/s1_repro_build.log" 2>&1; then
  ( "$JD/hh_repro5" 40 42; echo "rc=$?"; "$JD/hh_repro5" 25 2026; echo "rc=$?" ) > "$OUT/s1_repro.log" 2>&1
  tail -60 "$OUT/s1_repro.log"
  if [ "$(grep -c '^ALL-PASS' "$OUT/s1_repro.log")" = 2 ] && [ "$(grep -c '^rc=0' "$OUT/s1_repro.log")" = 2 ]; then S1=green; fi
  say "- 种子 42×40 / 种子 2026×25: $(grep -E '^(ALL-PASS|FAIL)' "$OUT/s1_repro.log" | tr '\n' ';')"
  say "- 行数合计: $(awk '/rows=/{for(i=1;i<=NF;i++){split($i,a,"=");s[a[1]]+=a[2]}}END{printf "rows=%d ok=%d ovf=%d skip=%d bad=%d",s["rows"],s["ok"],s["ovf"],s["skip"],s["bad"]}' "$OUT/s1_repro.log")"
else
  say "- **构建失败**,见 s1_repro_build.log:"; tail -40 "$OUT/s1_repro_build.log" | tee -a "$SUM"
fi
say "- S1: $S1"

# ───────────────────────── S2 构建 ─────────────────────────
stage "S2 构建 spgemm_bm2 / spgemm_bm2w"
say ""; say "## S2 构建"
S2=red
T0=$(date +%s)
make -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_bm2 > "$JD/build_bm2.log" 2>&1; RC=$?
T1=$(date +%s)
tail -c 150000 "$JD/build_bm2.log" > "$OUT/s2_build_bm2_tail.log"
grep -nE "error|Error|✗|✓|模板实例|opt-in|MISSING" "$JD/build_bm2.log" | head -80 > "$OUT/s2_build_bm2_keylines.log"
cat "$OUT/s2_build_bm2_keylines.log"
say "- make TARGET=spgemm_bm2: rc=$RC,用时 $((T1 - T0))s"
say "- 审计: $(grep -E '✓|✗' "$JD/build_bm2.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_bm2 ] && [ "$(grep -c '✓' "$JD/build_bm2.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_bm2.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_bm2w > "$JD/build_bm2w.log" 2>&1; RCW=$?
  T2=$(date +%s)
  tail -c 60000 "$JD/build_bm2w.log" > "$OUT/s2_build_bm2w_tail.log"
  say "- WRITE_MTX 件 spgemm_bm2w: rc=$RCW,用时 $((T2 - T1))s($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_bm2w ] && S2=green
else
  say "- **构建或审计失败**,日志尾:"; tail -60 "$JD/build_bm2.log" | tee -a "$SUM"
fi
say "- S2: $S2"
[ $S1 = green ] || finish 11 "红:S1 微复现器未全绿(S2 构建=$S2);后续阶段未跑"
[ $S2 = green ] || finish 12 "红:S2 构建/审计失败;后续阶段未跑"

# ───────────────────────── S3 冒烟 ─────────────────────────
stage "S3 冒烟(小阵三臂 nnz + 普查)"
say ""; say "## S3 冒烟"
python3 scripts/ab_arms.py --bin ./spgemm_bm2 --out "$OUT/s3_smoke" --rounds 1 --gpu "$GPU" \
  --arm off: --arm m1:HH_BM2=1 --arm m2:HH_BM2=2 --only bcsstk17,F2,case9,bloweybq,c-64,TSOPF --stat > "$OUT/s3_smoke_run.log" 2>&1
RC=$?
tail -50 "$OUT/s3_smoke_run.log"
S3=green
[ $RC = 0 ] || S3=red
grep -qE "NNZ-MISMATCH|FAIL|不可测" "$OUT/s3_smoke.txt" 2>/dev/null && S3=red
[ -s "$OUT/s3_smoke.txt" ] || S3=red
TAKEN=$(awk -F, 'NR>1{s+=$7}END{print s+0}' "$OUT/s3_smoke_stat.csv" 2>/dev/null)
say "- 小阵 off/m1/m2 各 1 发:nnz 与失败检查 → $S3;普查走两遍式的行数合计(m1+m2)= ${TAKEN:-0}"
[ "${TAKEN:-0}" -gt 0 ] || { say "- ⚠ 冒烟阵里没有任何行走两遍式(门没开或人口不在),值级将是空验"; }
[ $S3 = green ] || { tail -30 "$OUT/s3_smoke_fail.log" 2>/dev/null | tee -a "$SUM"; finish 13 "红:S3 冒烟 nnz 不一致或运行失败;后续阶段未跑"; }

# ───────────────────────── S4 值级 ─────────────────────────
stage "S4 值级"
say ""; say "## S4 值级(HH_BM2=1/2 对 HH_BM2=0 的 mtx_gate_fast;scipy 在则加 verify_sorted_truth 三判)"
say ""
say "| 阵 | 模式 | nnz | 走两遍式行/未走行(HH bin 合计) | gate(vs off) | 三判(vs scipy 真值) |"
say "|---|---|---|---|---|---|"
S4=green; VL_TAKEN=0
vl_run() {   # $1=name $2=path $3=mode → 转储搬到 $JD/vl/<name>_<off|on>.mtx;回显 "nnz taken skip"
  local name=$1 path=$2 mode=$3 tag=on log
  [ "$mode" = 0 ] && tag=off
  log="$JD/vl/${name}_m${mode}.log"
  rm -f "results/matrices/$name/self_product_hash.mtx"
  env $RUNENV METHOD=hash WARMUP_N=0 HH_BM2=$mode HH_BM2_STAT=1 timeout 1800 ./spgemm_bm2w "$path" > "$log" 2>&1
  local rc=$?
  if [ $rc != 0 ] || [ ! -s "results/matrices/$name/self_product_hash.mtx" ]; then
    echo "RUNFAIL rc=$rc"; tail -15 "$log" >&2; return 1
  fi
  mv "results/matrices/$name/self_product_hash.mtx" "$JD/vl/${name}_${tag}.mtx"
  local nnz taken skip
  nnz=$(grep -oE "Result C:.*nnz = [0-9]+" "$log" | tail -1 | grep -oE "[0-9]+$")
  taken=$(grep -oE "taken_rows=[0-9]+" "$log" | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  skip=$(grep -oE "skip_rows=[0-9]+" "$log" | cut -d= -f2 | awk '{s+=$1}END{print s+0}')
  grep "bm2-stat" "$log" | sed "s/^/[$name m$mode] /" >> "$OUT/s4_value_stat.log"
  echo "$nnz $taken $skip"
}
# 阵名:是否跑 scipy 真值(1/0)。前半=位图 bin 人口(窄 span),后半=翻 bin 家族(fused/legacy 契约)与重行 bin
VL_LIST="bcsstk17:1 af23560:1 cfd2:1 s3dkq4m2:1 msc10848:1 bcsstk30:1 c-64:1 pre2:1 TSOPF_RS_b678_c2:0 web-Google:0 ldoor:0"
for item in $VL_LIST; do
  name=${item%%:*}; truth=${item##*:}
  path=$(find_mtx "$name") || { say "| $name | - | - | - | 缺文件,跳过 | - |"; continue; }
  free_gb=$(df -BG --output=avail "$JD" | tail -1 | tr -dc 0-9)
  [ "${free_gb:-0}" -ge 40 ] || { say "| $name | - | - | - | 磁盘余量 <40G,跳过 | - |"; continue; }
  echo "---- 值级 $name ($path)"
  r0=$(vl_run "$name" "$path" 0) || { say "| $name | off | $r0 | - | **基线臂运行失败** | - |"; S4=red; continue; }
  for mode in 1 2; do
    r=$(vl_run "$name" "$path" $mode) || { say "| $name | m$mode | $r | - | **运行失败** | - |"; S4=red; continue; }
    set -- $r; nnz=$1; taken=$2; skip=$3
    VL_TAKEN=$((VL_TAKEN + taken))
    g=$(bash scripts/mtx_gate_fast.sh "$JD/vl/$name" 2>&1); echo "$g"
    gv=$(echo "$g" | grep -oE "PASS|FAIL" | tail -1)
    gd=$(echo "$g" | grep -oE "col_mismatch=.*" | head -1)
    [ "$gv" = PASS ] || S4=red
    tv="-"
    if [ "$truth" = 1 ] && [ $HAVE_SCIPY = 1 ]; then
      t=$(timeout 3600 python3 scripts/verify_sorted_truth.py "$path" "$JD/vl/${name}_on.mtx" 2>&1); echo "$t" | tail -8
      tv=$(echo "$t" | grep -oE "判定: (PASS|FAIL)" | tail -1)
      tl=$(echo "$t" | grep -E "^L[123]" | tr '\n' ' ')
      [ "$tv" = "判定: PASS" ] || S4=red
      tv="$tv($tl)"
    fi
    [ "${r0%% *}" = "$nnz" ] || { S4=red; gv="$gv/NNZ≠off(${r0%% *})"; }
    say "| $name | m$mode | $nnz | $taken / $skip | ${gv:-无输出}($gd) | $tv |"
    rm -f "$JD/vl/${name}_on.mtx"
  done
  rm -f "$JD/vl/${name}_off.mtx"
done
rm -rf "$ROOT"/results/matrices/*/self_product*.mtx
say ""
say "- 值级阵里走两遍式的行数合计(m1+m2): $VL_TAKEN$([ "$VL_TAKEN" -gt 0 ] || echo ' **(=0:空验,不算过)**')"
[ "$VL_TAKEN" -gt 0 ] || S4=red
if [ $HAVE_SCIPY = 0 ]; then
  cat > "$OUT/NEED_USER.md" <<'EOF'
# 需要用户裁决/知道
- 作业环境的 python3 没有 numpy/scipy,协议里的 `verify_sorted_truth.py` 三判(对 scipy 真值)本轮没法跑;
  本轮值级只做了"HH_BM2=1/2 对 HH_BM2=0 转储"的 mtx_gate_fast 差分(列集合/行内序/值)+ 微复现器真值。
  要补三判,需要二选一:允许作业用工作树的 .venv/bin/python(只读执行),或给作业环境装 numpy+scipy。
EOF
  say "- scipy 不可用:三判未跑,已写 NEED_USER.md"
fi
say "- S4: $S4"
[ $S4 = green ] || finish 14 "红:S4 值级未过(见上表);计时阶段未跑"

# ───────────────────────── S5 三臂电池 ─────────────────────────
stage "S5 三臂电池 off/m1/m2"
say ""; say "## S5 三臂电池(同一二进制 spgemm_bm2;off=HH_BM2 关,m1=值区 SMEM,m2=值直写全局;3 轮交错)"
EXTRA=""
for name in c-big delaunay_n24 dielFilterV3real cfd2 s3dkq4m2 nlpkkt80 FEM_3D_thermal2 offshore msc10848; do
  p=$(find_mtx "$name") && EXTRA="$EXTRA --extra $name=$p:hash"
done
python3 scripts/ab_arms.py --bin ./spgemm_bm2 --out "$OUT/s5_battery" --rounds 3 --gpu "$GPU" \
  --arm off: --arm m1:HH_BM2=1 --arm m2:HH_BM2=2 --stat $EXTRA > "$OUT/s5_battery_run.log" 2>&1
say "- ab_arms rc=$?;逐阵表见 s5_battery.txt,逐发 s5_battery.csv,普查 s5_battery_stat.csv"
grep -E "^(GM|裁定|=====)" "$OUT/s5_battery.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
say "- 卡状态: $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';')"

# ───────────────────────── S6 dup 门判别(训练池) ─────────────────────────
stage "S6 dup 门判别实验(只用训练池)"
say ""; say "## S6 行门判别(dup 轴 HH_BM2_DUP=flop×100/est 下限;密度轴 HH_BM2_SPANX=span/est 上限;只用 data/train_disp,428 主账/flex142/external_72 不参与标定)"
TRAIN=""; NTR=0
if [ -d data/train_disp ]; then
  # 取 8MB-800MB 的阵,按大小降序最多 8 个;同名出现在任何测试账目录的剔除
  while read -r sz f; do
    b=$(basename "$f" .mtx)
    [ -f "data/ocean/square/$b.mtx" ] || [ -f "data/first100/$b.mtx" ] || [ -n "$(find -L data/flex142 data/external_72 -maxdepth 3 -name "$b.mtx" 2>/dev/null | head -1)" ] && continue
    TRAIN="$TRAIN --extra $b=$f:hash"; NTR=$((NTR + 1))
    [ $NTR -ge 8 ] && break
  done < <(find -L data/train_disp -maxdepth 2 -name '*.mtx' -size +8M -size -800M -printf '%s %p\n' 2>/dev/null | sort -rn)
fi
if [ $NTR -ge 3 ]; then
  say "- 训练池取 $NTR 阵: $(echo "$TRAIN" | grep -oE "[^ =]+=" | tr -d '=' | tr '\n' ' ')"
  python3 scripts/ab_arms.py --bin ./spgemm_bm2 --out "$OUT/s6_dupsweep" --rounds 3 --gpu "$GPU" --no-battery \
    --arm off: --arm m1:HH_BM2=1 --arm m1d250:HH_BM2=1,HH_BM2_DUP=250 --arm m1sx32:HH_BM2=1,HH_BM2_SPANX=32 \
    --arm m2:HH_BM2=2 --arm m2d250:HH_BM2=2,HH_BM2_DUP=250 --arm m2sx32:HH_BM2=2,HH_BM2_SPANX=32 \
    --stat $TRAIN > "$OUT/s6_dupsweep_run.log" 2>&1
  say "- ab_arms rc=$?;见 s6_dupsweep.txt / .csv / _stat.csv"
  grep -E "^(GM|=====)" "$OUT/s6_dupsweep.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
else
  say "- 跳过:data/train_disp 里符合大小的非测试账阵不足 3 个(找到 $NTR)。dup 门的标定池在哪,需要用户指明。"
  { echo; echo "- S6(dup 门判别)没跑:data/train_disp 里 8MB-800MB 且不在测试账的阵只有 $NTR 个。请指明可用于门标定的训练/扩建池目录。"; ls data 2>&1 | sed 's/^/    /'; } >> "$OUT/NEED_USER.md"
fi

# ───────────────────────── S7 无侵入电池 ─────────────────────────
stage "S7 无侵入电池(基线源码 vs 新源码关开关)"
say ""; say "## S7 无侵入(A=spgemm_base:本提交反打补丁后的基线源码;B=spgemm_bm2 默认 env=开关关;官方 ab_pair.py 16×3)"
if [ $HAVE_PATCH = 1 ]; then
  rm -rf "$JD/base"; mkdir -p "$JD/base"
  cp -r Makefile src include tools "$JD/base/"
  rm -f "$JD"/base/src/*.o
  ( cd "$JD/base" && patch -p1 -R < "$ROOT/jobs/$ID/bm2_src.patch" ) > "$OUT/s7_patch.log" 2>&1
  GOT=$(sha256sum "$JD/base/src/spgemm_kernel_hash.cu" | cut -d' ' -f1)
  if [ "$GOT" = "$BASE_SHA" ]; then
    say "- 基线源码 sha256 核对通过($GOT)"
    T0=$(date +%s)
    ( cd "$JD/base" && make -j4 TARGET=spgemm_base ) > "$JD/build_base.log" 2>&1; RC=$?
    say "- make TARGET=spgemm_base: rc=$RC,用时 $(( $(date +%s) - T0 ))s;审计: $(grep -E '✓|✗' "$JD/build_base.log" | tr '\n' ';')"
    if [ $RC = 0 ] && [ -x "$JD/base/spgemm_base" ]; then
      cp "$JD/base/spgemm_base" "$ROOT/spgemm_base"
      python3 scripts/ab_pair.py spgemm_base spgemm_bm2 "$OUT/s7_ab_base_vs_bm2off.csv" 3 "$GPU" > "$OUT/s7_ab_run.log" 2>&1
      sed -n '/===== 判读/,$p' "$OUT/s7_ab_run.log" > "$OUT/s7_ab_verdict.txt"
      grep -E "^(GM|裁定|微级)" "$OUT/s7_ab_verdict.txt" | sed 's/^/    /' | tee -a "$SUM"
    else
      tail -30 "$JD/build_base.log" | tee -a "$SUM"
    fi
  else
    say "- **基线源码 sha256 不符**(得 $GOT,应 $BASE_SHA),S7 未跑;见 s7_patch.log"
  fi
else
  say "- 跳过:C 上没有 patch 命令"
fi

finish 0 "S1-S4 全绿;S5/S6/S7 的计时结论见各段 GM/裁定 行与对应 txt/csv"
