#!/usr/bin/env bash
# bm2_t1 —— 想法二 v3.1(按需划分)的值级回归 + 宽跨度/bin9 值级 + ncu(修命令行)+ A/A 认证口径的计时
#
# 背景:prof_k1 的 K2(nsys 设备侧逐 kernel 时长)跑通,m2/off 的 Σkernel GM 0.9697、最差 1.016;K3(ncu)试点失败
#   ("Failed to prepare kernel for profiling");T 段等了 10 小时主机始终不满足安静判据(load1 8–12、3–5 个外来 >50% CPU 进程),没跑。
# 本作业:
#   S1 微复现器     hh_bm2.cuh 未变(sha256 同 bm2_r4/r5 验过的)→ 沿用;变了才重跑
#   S2 构建         make TARGET=spgemm_t1(审计双绿)+ WRITE_MTX 件
#   S3 冒烟         三组各自判:G1 默认(按需划分,HH_BM2_LAZY=1)off/m1/m2/m2p/m2L;G2 同批划分(LAZY=0,即 bm2_r5 的形态)
#                   off/m2e/m2pe;G3 强制宽跨度(m2X:legacy 形态也收 + SMEM 预算放到 57856 字)。G1、G2 都红才停;G3 红只停 G3 自己的值级
#   S4 值级         v3.1 回归:bcsstk17 / cfd2 / msc10848 三臂差分门 + l3_diag;pre2(m2L)差分门;
#                   宽跨度封条:web-Google(m2X,生产默认到不了的形态,只为值级留封条);
#                   bin9 探查:PARSEC 族等候选阵先普查,找到 bin9 有行走两遍式的,取前两个做差分门
#   K3 ncu          试点逐个试 4 种组合(重放方式 × 内存池 × 核名正则),选第一个出报告的;5 阵;75 分钟预算
#   T  计时         先按 B 端判据等安静窗口,最多 QUIET_WAIT_S 秒(默认 30 分钟);等不到也跑,计时有效性由 A/A 对照臂认证
#                   (scripts/aa_check.py:|GM−1| ≤ 1% 且非微级阵在 ±5% 内),段内每分钟采 load 一并回传。
#                   T1 电池 off/aa/m2/m2p,24 阵 × 7 轮;T1b 固定成本探针:小阵 × off/aa/m2/m2e × 11 轮(按需划分对同批划分);
#                   T2 无侵入(基线源码件对本件开关关)。
# 只评测:不标定、不动默认、门值不入源码。回传:OUT_DIR/SUMMARY.md(先看这个)+ s*_ / k3_* / t1_* / t2_* 文件。
set -u
ROOT=$(pwd)
OUT=${OUT_DIR:?}
JD=${JOB_DIR:?}
ID=${JOB_ID:-bm2_t1}
DATA=/home/xueyizhou/spgemm-challenge/data
SUM=$OUT/SUMMARY.md
BASE_SHA=141fbc46a690225559b235d79a7da33a7eb7b7029795078e3e9f927e9875011c   # 基线 src/spgemm_kernel_hash.cu(70d1dea)
HDR_SHA=2fc25346de6deabc2eda6d49739ba0627a976969896b861e3798517bab59339b    # bm2_r4 / bm2_r5 的 S1 验过的 src/hh_bm2.cuh
LOAD_MAX=${LOAD_MAX:-6}
QUIET_WAIT_S=${QUIET_WAIT_S:-1800}
BIN=./spgemm_t1
RUNENV="USE_MEMPOOL=1 MP_HOST_MB=8192 D6EXP=0"
MYUID=$(id -u)
mkdir -p "$OUT" "$JD/vl" "$JD/k3" "$JD/keep"
: > "$SUM"
say() { echo "$*" | tee -a "$SUM"; }
TS0=$(date +%s); TSL=$TS0
stage() { local now; now=$(date +%s); echo; echo "################ $* ################ (+$((now - TSL))s 上一段,累计 $((now - TS0))s)"; TSL=$now; date -u +%FT%TZ; }
SAMPLER_PID=""
sampler_start() { ( while :; do echo "$(date +%s) $(cut -d' ' -f1 /proc/loadavg) $(foreign_hot)"; sleep 60; done ) > "$1" 2>/dev/null & SAMPLER_PID=$!; }
sampler_stop() { if [ -n "$SAMPLER_PID" ]; then kill "$SAMPLER_PID" 2>/dev/null; wait "$SAMPLER_PID" 2>/dev/null; SAMPLER_PID=""; fi; }
clean_dumps() { rm -rf "$ROOT"/results/matrices/*/self_product*.mtx "$JD/vl" 2>/dev/null; }
cleanup() { sampler_stop; clean_dumps; rm -f "$JD/TIMING_ACTIVE" "$JD/WAITING_QUIET"; }
trap 'cleanup; echo "收到终止信号,已清转储"; exit 143' TERM INT
finish() {   # $1=退出码 $2=结论一句话
  cleanup
  say ""; say "## 结束"; say "- 结论: $2"; say "- 主机(结束时): $(host_line)"; say "- 磁盘: $(df -h "$JD" | tail -1)"
  say "- 结束时间: $(date -u +%FT%TZ),总用时 $(( ($(date +%s) - TS0) / 60 )) 分钟"
  exit "$1"
}

# ───────────────────────── 主机负载与负载门 ─────────────────────────
foreign_hot() {   # CPU 占用最高的 5 个进程里,不属于本作业用户且 >50% 的个数
  ps -eo pcpu,uid --sort=-pcpu 2>/dev/null | awk -v me="$MYUID" 'NR>1 && NR<=6 && $2!=me && $1+0>50 {n++} END{print n+0}'
}
apps_on() {       # $1=卡号 → 该卡上的计算进程数
  local uuid
  uuid=$(nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader 2>/dev/null)
  [ -n "$uuid" ] || { echo 0; return 0; }
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null | grep -c "$uuid"
}
gpu_busy() {      # 我钉的卡上有别的计算进程,或另一张卡利用率 ≥10% → 返回 0(忙)
  local g u
  [ "$(apps_on "$GPU")" = 0 ] || return 0
  for g in 0 1; do
    [ "$g" = "$GPU" ] && continue
    u=$(nvidia-smi -i $g --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -dc 0-9)
    [ "${u:-0}" -ge 10 ] && return 0
  done
  return 1
}
host_line() { echo "load=$(cut -d' ' -f1-3 /proc/loadavg) 外来>50%CPU进程数=$(foreign_hot) 卡=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | tr '\n' ';')"; }
quiet_once() {
  awk -v l="$(cut -d' ' -f1 /proc/loadavg)" -v t="$LOAD_MAX" 'BEGIN{exit !(l+0 < t+0)}' || return 1
  [ "$(foreign_hot)" = 0 ] || return 1
  gpu_busy && return 1
  return 0
}
quiet_twice() { quiet_once || return 1; sleep 60; quiet_once; }
wait_quiet() {    # $1=最长等待秒数;成功(连续两次安静)返回 0
  local t0 n=0
  t0=$(date +%s)
  while :; do
    quiet_twice && return 0
    [ $(( $(date +%s) - t0 )) -ge "$1" ] && return 1
    [ $((n % 6)) = 0 ] && echo "等安静窗口:$(host_line)(已等 $(( ($(date +%s) - t0) / 60 )) 分钟)"
    n=$((n + 1))
    sleep 300
  done
}
load_med() {      # $1=采样文件 → load1 中位(无采样回显空)
  local n
  n=$(wc -l < "$1" 2>/dev/null || echo 0)
  [ "${n:-0}" -gt 0 ] || return 0
  awk '{print $2}' "$1" | sort -n | sed -n "$(( (n + 1) / 2 ))p"
}
load_stats() {    # $1=采样文件 → 一行文字
  local n
  n=$(wc -l < "$1" 2>/dev/null || echo 0)
  [ "${n:-0}" -gt 0 ] || { echo "无采样"; return 0; }
  echo "load1 中位 $(load_med "$1") / 最大 $(awk '{print $2}' "$1" | sort -n | tail -1)($n 个采样,每分钟一次;其中 $(awk '$3>0{c++}END{print c+0}' "$1") 次见到外来 >50% CPU 进程)"
}


# ───────────────────────── S0 环境 ─────────────────────────
stage "S0 环境"
say "# bm2_t1 汇总(想法二 v3.1:按需划分的值级回归、宽跨度/bin9 值级、ncu、A/A 认证口径的电池与无侵入)"
say ""
say "## S0 环境"
say "- 主机: $(hostname)  开始: $(date -u +%FT%TZ)  快照: $ROOT  nproc=$(nproc 2>/dev/null)"
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
say "- **用卡: GPU $GPU**$([ $SHARED = 1 ] && echo '(⚠ 等了 30 分钟仍有别的计算进程,与人共卡)')"
say "- 主机(开跑时): $(host_line)"

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
arm_env() {   # $1=臂名 → env 赋值串
  case $1 in
    m1)   echo "HH_BM2=1";;
    m2)   echo "HH_BM2=2";;
    m2e)  echo "HH_BM2=2 HH_BM2_LAZY=0";;
    m2p)  echo "HH_BM2=2 HH_BM2_P1=1";;
    m1L)  echo "HH_BM2=1 HH_BM2_L1=1";;
    m2L)  echo "HH_BM2=2 HH_BM2_L1=1";;
    m2X)  echo "HH_BM2=2 HH_BM2_L1=1 HH_BM2_SMX=57856";;
    *)    echo "HH_BM2=0";;
  esac
}

# ───────────────────────── S1 微复现器 ─────────────────────────
stage "S1 微复现器"
say ""; say "## S1 微复现器"
S1=red
GOT_HDR=$(sha256sum src/hh_bm2.cuh | cut -d' ' -f1)
if [ "$GOT_HDR" = "$HDR_SHA" ]; then
  S1=green
  say "- src/hh_bm2.cuh 的 sha256 与 bm2_r4 / bm2_r5 的 S1 验过的相同(两种子 ALL-PASS,168480 行 bad=0)→ 沿用"
else
  say "- src/hh_bm2.cuh 已变($GOT_HDR),重跑 hh_repro5"
  if nvcc -O3 -arch=sm_90 -std=c++14 -Isrc -Iinclude -I/usr/local/cuda/include tools/hh_repro5.cu -o "$JD/hh_repro5" > "$OUT/s1_repro_build.log" 2>&1; then
    ( "$JD/hh_repro5" 40 42; echo "rc=$?"; "$JD/hh_repro5" 25 2026; echo "rc=$?" ) > "$OUT/s1_repro.log" 2>&1
    if [ "$(grep -c '^ALL-PASS' "$OUT/s1_repro.log")" = 2 ] && [ "$(grep -c '^rc=0' "$OUT/s1_repro.log")" = 2 ]; then S1=green; fi
    say "- $(grep -E '^(ALL-PASS|FAIL)' "$OUT/s1_repro.log" | tr '\n' ';')"
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
make -k -j4 -f Makefile -f "jobs/$ID/keep.mk" TARGET=spgemm_t1 > "$JD/build_t1.log" 2>&1; RC=$?   # -k:一个翻译单元报错后继续编另一个,报错一次出全
T1=$(date +%s)
tail -c 100000 "$JD/build_t1.log" > "$OUT/s2_build_tail.log"
say "- make TARGET=spgemm_t1: rc=$RC,用时 $((T1 - T0))s;审计: $(grep -E '✓|✗|模板实例' "$JD/build_t1.log" | tr '\n' ';')"
if [ $RC = 0 ] && [ -x spgemm_t1 ] && [ "$(grep -c '✓' "$JD/build_t1.log")" -ge 2 ] && ! grep -q '✗' "$JD/build_t1.log"; then
  if [ -f src/spgemm_kernel_hash.o ]; then rm -f src/main.o; MK="-f Makefile -f jobs/$ID/keep.mk"; else MK=""; fi
  make -j4 $MK WRITE_MTX=1 TARGET=spgemm_t1w > "$JD/build_t1w.log" 2>&1; RCW=$?
  say "- WRITE_MTX 件 spgemm_t1w: rc=$RCW($([ -n "$MK" ] && echo '只重编 main.cu' || echo '全量重编'))"
  [ $RCW = 0 ] && [ -x spgemm_t1w ] && S2=green
else
  say "- **构建或审计失败**。报错行(前 80 条,全文尾部在 s2_build_tail.log):"
  grep -nE "error|Error|✗|MISSING|undefined reference" "$JD/build_t1.log" | head -80 | tee -a "$SUM"
  say "- 日志尾:"; tail -25 "$JD/build_t1.log" | tee -a "$SUM"
fi
say "- S2: $S2"
[ $S1 = green ] || finish 11 "红:S1 微复现器未全绿(S2=$S2);后续阶段未跑"
[ $S2 = green ] || finish 12 "红:S2 构建/审计失败;后续阶段未跑"

# ───────────────────────── S3 冒烟(三组各自判) ─────────────────────────
stage "S3 冒烟"
say ""; say "## S3 冒烟(各臂 1 发:nnz 与失败检查;单发的时间比不作读数)"
smoke() {   # $1=组名 $2=--only 列表 $3..=臂参数 → 回显 green / red
  local g=$1 only=$2 rc v=green; shift 2
  python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/s3_$g" --rounds 1 --gpu "$GPU" "$@" --only "$only" --stat > "$OUT/s3_${g}_run.log" 2>&1
  rc=$?
  [ $rc = 0 ] || v=red
  grep -qE "NNZ-MISMATCH|FAIL|不可测" "$OUT/s3_$g.txt" 2>/dev/null && v=red
  [ -s "$OUT/s3_$g.txt" ] || v=red
  echo $v
}
smoke_taken() { awk -F, 'NR>1{s+=$7}END{print s+0}' "$OUT/s3_$1_stat.csv" 2>/dev/null; }
LAZY_OK=$(smoke g1 bcsstk17,F2,case9,c-64,TSOPF,pre2 --arm off: --arm m1:HH_BM2=1 --arm m2:HH_BM2=2 --arm m2p:HH_BM2=2,HH_BM2_P1=1 --arm m2L:HH_BM2=2,HH_BM2_L1=1)
say "- G1 默认(按需划分)off/m1/m2/m2p/m2L,6 阵 → $LAZY_OK;走两遍式的行数合计 = $(smoke_taken g1)"
[ $LAZY_OK = green ] || { tail -25 "$OUT/s3_g1_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g1_fail.log" 2>/dev/null | tee -a "$SUM"; }
EAGER_OK=$(smoke g2 bcsstk17,F2,c-64,TSOPF --arm off: --arm m2e:HH_BM2=2,HH_BM2_LAZY=0 --arm m2pe:HH_BM2=2,HH_BM2_P1=1,HH_BM2_LAZY=0)
say "- G2 同批划分(HH_BM2_LAZY=0,bm2_r5 的形态)off/m2e/m2pe,4 阵 → $EAGER_OK;走两遍式的行数合计 = $(smoke_taken g2)"
[ $EAGER_OK = green ] || { tail -25 "$OUT/s3_g2_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g2_fail.log" 2>/dev/null | tee -a "$SUM"; }
X_OK=$(smoke g3 bcsstk17,web-Google,TSOPF,pre2 --arm off: --arm m2X:HH_BM2=2,HH_BM2_L1=1,HH_BM2_SMX=57856)
say "- G3 强制宽跨度(m2X:legacy 形态也收 + SMEM 预算 57856 字;**生产默认到不了的形态**,只为值级封条)off/m2X,4 阵 → $X_OK;走两遍式的行数合计 = $(smoke_taken g3)"
[ $X_OK = green ] || { tail -25 "$OUT/s3_g3_run.log" | tee -a "$SUM"; tail -20 "$OUT/s3_g3_fail.log" 2>/dev/null | tee -a "$SUM"; }
if [ $LAZY_OK != green ] && [ $EAGER_OK != green ]; then finish 13 "红:S3 冒烟两种划分时机都没过;后续阶段未跑"; fi
if [ $LAZY_OK = green ]; then
  ARM_M2="m2:HH_BM2=2"; ARM_M2P="m2p:HH_BM2=2,HH_BM2_P1=1"; VTAG=m2
  say "- 主线臂 = 按需划分(默认 env)"
else
  ARM_M2="m2:HH_BM2=2,HH_BM2_LAZY=0"; ARM_M2P="m2p:HH_BM2=2,HH_BM2_P1=1,HH_BM2_LAZY=0"; VTAG=m2e
  say "- **按需划分冒烟未过 → 主线臂退回同批划分(HH_BM2_LAZY=0)**;S4 的 m1/m2 臂是默认 env,读数只说明按需划分的问题"
fi

# ───────────────────────── S4 值级 ─────────────────────────
vl_dump() {   # $1=name $2=path $3=臂名 → $JD/vl/<name>_<臂名>.mtx;回显 "nnz taken skip"
  local name=$1 path=$2 tag=$3 log
  log="$JD/vl/${name}_${tag}.log"
  rm -f "results/matrices/$name/self_product_hash.mtx"
  env $RUNENV METHOD=hash WARMUP_N=0 $(arm_env "$tag") HH_BM2_STAT=1 timeout 1800 ./spgemm_t1w "$path" > "$log" 2>&1
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
stage "S4 值级"
say ""; say "## S4 值级(v3.1 回归 + 宽跨度封条 + bin9 探查)"
say ""
say "### S4a v3.1 回归:off / m1 / m2(默认 env=按需划分)差分门 + 对 scipy 真值的归一误差"
say ""
say "判法:差分门 PASS ∧ nnz 同 off ∧(l3_diag 有输出时)该臂 nerr 最大值 ≤ 1e-10。行例程本身的官方三判已在 bm2_r5 的 S4a 做过(20 组全绿)。"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) | 官方口径错行(仅参考) | nerr 最大 | 本作业判 |"
say "|---|---|---|---|---|---|---|---|"
if [ $LAZY_OK = green ]; then
  gate_diag bcsstk17
  gate_diag cfd2
  gate_diag msc10848
else
  say "| (按需划分冒烟未过,S4a 跳过) | | | | | | | |"
fi
say ""
say "### S4b legacy 形态的划分掩码路径:pre2(m2L)"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |"
say "|---|---|---|---|---|"
if [ $LAZY_OK = green ]; then gate_only pre2 m2L; else say "| (跳过) | | | | |"; fi
S4MAIN=$S4
say ""
say "- S4a+S4b(主线): $S4MAIN;走两遍式的行数合计 $VL_TAKEN"
say ""
say "### S4c 宽跨度封条:web-Google(m2X)"
say ""
say "**封条注**:m2X = HH_BM2=2 + HH_BM2_L1=1 + HH_BM2_SMX=57856,是生产默认到不了的形态(legacy 形态默认不开;SMEM 预算默认不放大)。"
say "这里只验它的值对不对,防以后门变化把没验过的形态放出来;不计时,不代表建议启用。"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |"
say "|---|---|---|---|---|"
SX=skip
if [ $X_OK = green ]; then S4=green; gate_only web-Google m2X; SX=$S4; else say "| (G3 冒烟未过,跳过) | | | | |"; fi
say ""
say "### S4d bin9 探查(dialog/qa-004:电池与训练池里 bin9 没有 HH 行;在 PARSEC 族等候选阵里找)"
say ""
say "| 阵 | 臂 | nnz | 走两遍式行 / 未走 | 差分门(对 off) |"
say "|---|---|---|---|---|"
B9N=0; B9SCOUT=""; SB9P=green; SB9X=green
for name in H2O GaAsH6 Si34H36 Ga3As3H12 Ga10As10H30 Ge87H76 nd6k nd12k SiO2 Ga19As19H42 gupta3 exdata_1 TSOPF_FS_b300_c2 nd24k; do
  [ $B9N -ge 2 ] && break
  path=$(find_mtx "$name") || continue
  t9=""; a9=""
  for arm in $VTAG m2X; do
    [ $arm = m2X ] && [ $X_OK != green ] && continue
    t9=$(env $RUNENV METHOD=hash WARMUP_N=0 $(arm_env $arm) HH_BM2_STAT=1 timeout 600 "$BIN" "$path" 2>&1 < /dev/null | grep "bm2-stat" | grep "bin=9 " | grep -oE "taken_rows=[0-9]+" | cut -d= -f2 | tail -1)
    if [ "${t9:-0}" -gt 0 ]; then a9=$arm; break; fi
  done
  B9SCOUT="$B9SCOUT $name:${a9:-无}${a9:+=$t9}"
  if [ -n "$a9" ]; then
    S4=green; gate_only "$name" "$a9"; B9N=$((B9N + 1))
    if [ $S4 = red ]; then if [ "$a9" = m2X ]; then SB9X=red; else SB9P=red; fi; fi
  fi
done
say ""
say "- bin9 探查(阵:走了两遍式的臂=bin9 行数;$VTAG = 生产可达形态,m2X = 强制形态):$B9SCOUT;做了值级的 $B9N 个$([ $B9N = 0 ] && echo ' **(=0:bin9 的生产值级仍只有微复现器覆盖)**')"
rm -rf "$ROOT"/results/matrices/*/self_product*.mtx
say "- S4 主线: $S4MAIN | 宽跨度封条(m2X): $SX | bin9: $([ $B9N = 0 ] && echo 无样本 || echo "生产可达形态 $SB9P / 强制形态 $SB9X")"
if [ "$SX" = red ] || [ $SB9X = red ]; then
  say "- **m2X 形态的值级有红**:这是生产默认到不了的形态,不挡主线计时;但在修好之前 HH_BM2_L1 / HH_BM2_SMX 不得放开"
fi
if [ $LAZY_OK = green ] && [ $S4MAIN != green ]; then finish 14 "红:S4 主线值级未过(见上表);计时阶段未跑"; fi
if [ $SB9P = red ]; then finish 14 "红:bin9 生产可达形态的值级未过(见上表);计时阶段未跑"; fi

# ───────────────────────── K3 ncu 机制计数 ─────────────────────────
do_K3() {
  stage "K3 ncu 机制计数"
  say ""; say "## K3 ncu 硬件计数(hash_heavy 对 hash_bm2:缓存命中、显存吞吐、指令数、占用;尽力而为)"
  if ! command -v ncu >/dev/null 2>&1; then say "- **ncu 不在 PATH**,K3 跳过"; return 0; fi
  say "- ncu: $(command -v ncu) / $(ncu --version 2>&1 | grep -iE 'version' | head -1)"
  local pm combo mode mp kre t0 t1 rc got="" NCU_MODE="" NCU_MP="" NCU_KRE=""
  pm=$(find_mtx bcsstk17) || { say "- 缺试点阵 bcsstk17,K3 跳过"; return 0; }
  : > "$OUT/k3_pilot.log"
  # prof_k1 的试点(核重放 + 内存池 + 含划分核)报 "Failed to prepare kernel for profiling"。逐个试:
  for combo in "application:1:hash_heavy_kernel|hash_bm2_kernel|hh_bm2_part_kernel" "kernel:0:hash_heavy_kernel|hash_bm2_kernel|hh_bm2_part_kernel" \
               "kernel:1:hash_heavy_kernel|hash_bm2_kernel" "application:0:hash_heavy_kernel|hash_bm2_kernel"; do
    mode=${combo%%:*}; kre=${combo##*:}; mp=${combo#*:}; mp=${mp%%:*}
    rm -f "$JD/k3/pilot.ncu-rep"
    t0=$(date +%s)
    env USE_MEMPOOL=$mp MP_HOST_MB=8192 D6EXP=0 METHOD=hash WARMUP_N=0 HH_BM2=2 timeout 600 ncu --target-processes all --replay-mode $mode -k "regex:$kre" --launch-count 3 -f -o "$JD/k3/pilot" "$BIN" "$pm" > "$JD/k3/pilot.log" 2>&1 < /dev/null
    rc=$?; t1=$(date +%s)
    echo "ncu 试点 [重放=$mode 内存池=$mp 核名=$kre] rc=$rc 用时 $((t1 - t0))s" >> "$OUT/k3_pilot.log"
    grep -E "==ERROR==|==WARNING==" "$JD/k3/pilot.log" | head -6 >> "$OUT/k3_pilot.log"
    if grep -q "ERR_NVGPUCTRPERM" "$JD/k3/pilot.log"; then
      say "- **ncu 报 ERR_NVGPUCTRPERM(GPU 性能计数器权限)**:K3 未做,已写 NEED_USER.md;没有绕过"
      { echo "# NEED_USER"; echo; echo "- bm2_t1 的 K3:ncu 报 ERR_NVGPUCTRPERM(GPU 性能计数器需要管理员放开权限)。未绕过,等处理后重跑 K3。"; } > "$OUT/NEED_USER.md"
      return 0
    fi
    if [ -s "$JD/k3/pilot.ncu-rep" ]; then got=1; NCU_MODE=$mode; NCU_MP=$mp; NCU_KRE=$kre; break; fi
  done
  if [ -z "$got" ]; then
    say "- ncu 试点 4 种组合都没产出报告(各组的 rc 与报错行在 k3_pilot.log),K3 跳过"
    return 0
  fi
  timeout 300 ncu --import "$JD/k3/pilot.ncu-rep" --csv --page raw > "$JD/k3/pilot.raw.csv" 2>> "$JD/k3/pilot.log"
  head -c 4000 "$JD/k3/pilot.raw.csv" > "$OUT/k3_pilot_raw_head.csv"
  say "- 试点通过:重放方式 $NCU_MODE,内存池 USE_MEMPOOL=$NCU_MP,核名正则 [$NCU_KRE];用时 $((t1 - t0))s;原始表 $(wc -l < "$JD/k3/pilot.raw.csv") 行(表头样例在 k3_pilot_raw_head.csv)"
  rm -f "$JD/k3/pilot".*
  local spec name path arm e pfx n=0 ok=0 k3t0
  k3t0=$(date +%s)
  for spec in "cfd2:off m2 m2p" "s3dkq4m2:off m2 m2p" "msc10848:off m2" "bcsstk30:off m2" "TSOPF_RS_b678_c2:off m2 m2p"; do
    name=${spec%%:*}
    path=$(find_mtx "$name") || { echo "K3 缺 $name"; continue; }
    for arm in ${spec#*:}; do
      if [ $(( $(date +%s) - k3t0 )) -gt 4500 ]; then echo "K3 已用 75 分钟,余下的组跳过"; break 2; fi
      e=$(arm_env $arm); pfx="$JD/k3/${name}__${arm}"; n=$((n + 1))
      rm -f "$pfx.ncu-rep"
      env USE_MEMPOOL=$NCU_MP MP_HOST_MB=8192 D6EXP=0 METHOD=hash WARMUP_N=0 $e timeout 1200 ncu --target-processes all --replay-mode $NCU_MODE -k "regex:$NCU_KRE" --launch-count 16 --set full -f -o "$pfx" "$BIN" "$path" > "$pfx.log" 2>&1 < /dev/null
      if [ ! -s "$pfx.ncu-rep" ]; then
        echo "K3 $name $arm:--set full 没产出,退回默认指标集"; grep -E "==ERROR==" "$pfx.log" | head -3
        env USE_MEMPOOL=$NCU_MP MP_HOST_MB=8192 D6EXP=0 METHOD=hash WARMUP_N=0 $e timeout 600 ncu --target-processes all --replay-mode $NCU_MODE -k "regex:$NCU_KRE" --launch-count 16 -f -o "$pfx" "$BIN" "$path" > "$pfx.log" 2>&1 < /dev/null
      fi
      if [ -s "$pfx.ncu-rep" ] && timeout 600 ncu --import "$pfx.ncu-rep" --csv --page raw > "$pfx.raw.csv" 2>> "$pfx.log" && grep -q "Kernel Name" "$pfx.raw.csv"; then
        ok=$((ok + 1))
        gzip -c "$pfx.raw.csv" > "$OUT/k3_raw_${name}__${arm}.csv.gz"
        mv "$pfx.ncu-rep" "$JD/keep/k3_${name}__${arm}.ncu-rep"
      else
        echo "K3 FAIL $name $arm"; grep -E "==ERROR==" "$pfx.log" | head -3
      fi
    done
  done
  say "- 采数:$n 组(阵 × 臂;整段预算 75 分钟,超了余下跳过),成功 $ok 组;原始宽表 k3_raw_<阵>__<臂>.csv.gz,ncu 报告留在 $JD/keep/"
  python3 scripts/ncu_sum.py --dir "$JD/k3" --out "$OUT/k3_ncu" > "$OUT/k3_ncu_run.log" 2>&1
  say "- ncu_sum rc=$?;挑选指标的并排表 k3_ncu.txt,长表 k3_ncu_long.csv"
  say "- 主机(K3 后): $(host_line)"
}
do_K3

# ───────────────────────── T 计时(A/A 认证口径) ─────────────────────────
stage "T0 负载门"
say ""; say "## T0 负载门"
QUIET=0
: > "$JD/WAITING_QUIET"
TW0=$(date +%s)
if wait_quiet "$QUIET_WAIT_S"; then
  QUIET=1; say "- 等了 $(( ($(date +%s) - TW0) / 60 )) 分钟,主机满足安静判据($(host_line))"
else
  say "- 等了 $(( ($(date +%s) - TW0) / 60 )) 分钟,主机仍不满足安静判据($(host_line))。**照跑:计时是否有效改由 A/A 对照臂认证**(prof_k1 已等过 10 小时,判据一直没满足)"
fi
rm -f "$JD/WAITING_QUIET"
: > "$JD/TIMING_ACTIVE"

stage "T1 电池(带 A/A 对照臂)"
say ""; say "## T1 只评测电池(同一二进制;off / aa=与 off 完全相同的对照臂 / m2=位图形态不设门 / m2p=同上+Pass1 读 B16;7 轮交错)"
say "- 主机(T1 前): $(host_line)"
EXTRA=""
for name in c-big cfd2 s3dkq4m2 nlpkkt80 FEM_3D_thermal2 offshore msc10848 bcsstk30; do
  p=$(find_mtx "$name") && EXTRA="$EXTRA --extra $name=$p:hash"
done
sampler_start "$JD/load_T1.log"
python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/t1_battery" --rounds 7 --gpu "$GPU" \
  --arm off: --arm aa: --arm "$ARM_M2" --arm "$ARM_M2P" --stat $EXTRA > "$OUT/t1_battery_run.log" 2>&1
say "- ab_arms rc=$?;逐阵表 t1_battery.txt,逐发 t1_battery.csv,逐 bin 普查 t1_battery_stat.csv"
sampler_stop
cp "$JD/load_T1.log" "$OUT/t1_load.log" 2>/dev/null
say "- 段内主机负载: $(load_stats "$JD/load_T1.log")"
python3 scripts/aa_check.py "$OUT/t1_battery.csv" > "$OUT/t1_aa.txt" 2>&1
AA=$(grep -oE "^AA_VERDICT [a-z]+" "$OUT/t1_aa.txt" | cut -d' ' -f2)
say "- A/A 对照臂: $(grep -E '^AA_(N|GM|NONMICRO_MIN|NONMICRO_MAX|VERDICT) ' "$OUT/t1_aa.txt" | tr '\n' ';')(逐阵见 t1_aa.txt)"
if [ "${AA:-nodata}" = clean ]; then
  say "- **T1 计时判定: A/A 认证通过**$([ $QUIET = 1 ] && echo '(且开跑时满足安静判据)' || echo '(开跑时不满足安静判据;是否接受这一口径由 dialog/qa-007 定)')"
else
  say "- **T1 计时判定: 污染(A/A 对照臂未过)** —— 下面的 GM 与裁定只作参考,不作定谳"
fi
grep -E "^(GM|裁定|=====)" "$OUT/t1_battery.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"

stage "T1b 固定成本探针"
say ""; say "## T1b 固定成本探针(小阵;off / aa / m2=按需划分 / m2e=同批划分;11 轮交错)"
say "- 看两件事:① 用不上两遍式的小阵上,m2/off 是否回到 A/A 的噪声带内(按需划分应为零新增开销),m2e/off 是固定成本读数;"
say "  ② 用得上的小阵上,按需划分多出的那次同步吃掉多少收益(m2 对 m2e)。"
PEX=""; PN=0
for name in bcspwr05 494_bus 662_bus 685_bus bcsstk01 bcsstk13 can_292 bcspwr10 bcsstm07 bcsstk17 af23560 msc10848 bcsstk30 cfd2; do
  p=$(find_mtx "$name") && { PEX="$PEX --extra $name=$p:hash"; PN=$((PN + 1)); }
done
if [ $PN -gt 0 ] && [ $LAZY_OK = green ] && [ $EAGER_OK = green ]; then
  sampler_start "$JD/load_T1b.log"
  python3 scripts/ab_arms.py --bin "$BIN" --out "$OUT/t1b_probe" --rounds 11 --gpu "$GPU" --no-battery \
    --arm off: --arm aa: --arm m2:HH_BM2=2 --arm m2e:HH_BM2=2,HH_BM2_LAZY=0 --stat $PEX > "$OUT/t1b_probe_run.log" 2>&1
  say "- ab_arms rc=$?($PN 阵);逐阵表 t1b_probe.txt,逐发 t1b_probe.csv"
  sampler_stop
  say "- 段内主机负载: $(load_stats "$JD/load_T1b.log")"
  python3 scripts/aa_check.py "$OUT/t1b_probe.csv" > "$OUT/t1b_aa.txt" 2>&1
  say "- A/A 对照臂: $(grep -E '^AA_(N|GM|NONMICRO_MIN|NONMICRO_MAX|VERDICT) ' "$OUT/t1b_aa.txt" | tr '\n' ';')(这组多是微级阵,A/A 的逐阵比就是该阵的噪声带,见 t1b_aa.txt)"
  grep -E "^(GM|=====)" "$OUT/t1b_probe.txt" 2>/dev/null | sed 's/^/    /' | tee -a "$SUM"
else
  say "- 跳过(候选阵 $PN 个;按需划分冒烟 $LAZY_OK / 同批划分冒烟 $EAGER_OK)"
fi

stage "T2 无侵入电池"
say ""; say "## T2 无侵入(A=spgemm_base:反打补丁后的基线源码;B=spgemm_t1 默认 env=开关关;官方 ab_pair.py 16×3)"
if [ $HAVE_PATCH = 1 ]; then
  rm -rf "$JD/base"; mkdir -p "$JD/base"
  cp -r Makefile src include tools "$JD/base/"
  rm -f "$JD"/base/src/*.o
  ( cd "$JD/base" && patch -p1 -R < "$ROOT/jobs/$ID/bm2_src.patch" ) > "$OUT/t2_patch.log" 2>&1
  GOT=$(sha256sum "$JD/base/src/spgemm_kernel_hash.cu" | cut -d' ' -f1)
  if [ "$GOT" = "$BASE_SHA" ]; then
    say "- 基线源码 sha256 核对通过"
    ( cd "$JD/base" && make -j4 TARGET=spgemm_base ) > "$JD/build_base.log" 2>&1; RCB=$?
    say "- make TARGET=spgemm_base: rc=$RCB;审计: $(grep -E '✓|✗|模板实例' "$JD/build_base.log" | tr '\n' ';')"
    if [ $RCB = 0 ] && [ -x "$JD/base/spgemm_base" ]; then
      cp "$JD/base/spgemm_base" "$ROOT/spgemm_base"
      say "- 主机(T2 前): $(host_line)"
      sampler_start "$JD/load_T2.log"
      python3 scripts/ab_pair.py spgemm_base spgemm_t1 "$OUT/t2_ab_base_vs_t1off.csv" 3 "$GPU" > "$OUT/t2_ab_run.log" 2>&1
      sampler_stop
      sed -n '/===== 判读/,$p' "$OUT/t2_ab_run.log" > "$OUT/t2_ab_verdict.txt"
      grep -E "^(GM|裁定|微级)" "$OUT/t2_ab_verdict.txt" | sed 's/^/    /' | tee -a "$SUM"
      say "- 段内主机负载: $(load_stats "$JD/load_T2.log")(这一段没有 A/A 臂;是否干净参照 T1 的 A/A 判定与本段负载)"
    else
      tail -30 "$JD/build_base.log" | tee -a "$SUM"
    fi
  else
    say "- **基线源码 sha256 不符**(得 $GOT),T2 未跑;见 t2_patch.log"
  fi
else
  say "- 跳过:C 上没有 patch 命令"
fi
rm -f "$JD/TIMING_ACTIVE"
finish 0 "S3 冒烟:按需划分 $LAZY_OK / 同批划分 $EAGER_OK / 强制宽跨度 $X_OK;S4 主线 $S4MAIN(宽跨度封条=$SX,bin9 样本 $B9N 个);计时判定见 T1(A/A=${AA:-nodata});K3、T1b、T2 见各段"
